use super::*;
use arrow::array::Float64Array;
use async_trait::async_trait;
use calc_flow::{
    BatchMetadata, CheckpointManifest, Cursor, DeliveryGuarantee, ExpressionOperator, JobState,
    ManagedCheckpointRuntime, NativeWatermarkCapability, PipelineBuilder, ReplayPositioning,
    Result, SinkBinding, SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent,
    SourceSchema, StreamRequirements, StreamRuntimeConfig, StreamSource, StreamingRunner,
    TransactionalStreamSink, UdfRegistry, WatermarkPolicy,
};
use std::{path::Path, sync::Mutex, time::Duration};
use tokio::sync::Notify;

struct FactorySink(Box<dyn TransactionalStreamSink>);

#[async_trait]
impl TransactionalStreamSink for FactorySink {
    async fn open(&mut self) -> Result<()> {
        self.0.open().await
    }
    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()> {
        self.0.begin_epoch(epoch).await
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0.write(batch).await
    }
    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap> {
        self.0.pre_commit(epoch).await
    }
    async fn pre_commit_segments(&mut self, epoch: Epoch) -> Result<BTreeMap<String, Vec<u8>>> {
        self.0.pre_commit_segments(epoch).await
    }
    async fn commit(&mut self, epoch: Epoch, evidence: &JsonMap) -> Result<()> {
        self.0.commit(epoch, evidence).await
    }
    async fn abort(&mut self, epoch: Epoch, evidence: Option<&JsonMap>) -> Result<()> {
        self.0.abort(epoch, evidence).await
    }
    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        self.0.recover(recovery).await
    }
    async fn close(&mut self) -> Result<()> {
        self.0.close().await
    }
}

#[derive(Default)]
struct Probe {
    paused: Notify,
    offsets: Mutex<Vec<usize>>,
}

struct ReplaySource {
    values: Vec<f64>,
    offset: usize,
    pause_at: Option<usize>,
    probe: Arc<Probe>,
}

fn value_batch(value: f64) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_from_iter([(
                "value",
                Arc::new(Float64Array::from(vec![value])) as arrow::array::ArrayRef,
            )])
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[async_trait]
impl StreamSource for ReplaySource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 1,
            max_batch_bytes: 1024,
            schema: SourceSchema::Exact(value_batch(0.0).table_payload().unwrap().schema().clone()),
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.offset = cursor.map_or(0, |cursor| {
            usize::try_from(cursor.payload()["offset"].as_u64().unwrap()).unwrap()
        });
        self.probe.offsets.lock().unwrap().push(self.offset);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.pause_at == Some(self.offset) {
            self.probe.paused.notify_one();
            return std::future::pending().await;
        }
        let Some(value) = self.values.get(self.offset).copied() else {
            return Ok(None);
        };
        self.offset += 1;
        Ok(Some(SourceEvent::Data {
            batch: value_batch(value),
            cursor: Cursor::unbound(
                u64::try_from(self.offset).unwrap().to_be_bytes().to_vec(),
                JsonMap::from([("offset".into(), json!(self.offset))]),
            )?,
        }))
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

async fn runner(root: &Path, url: &Url, source: ReplaySource) -> StreamingRunner {
    let plan = PipelineBuilder::new("mysql-invalid-checkpoint")
        .unwrap()
        .add_node(
            "copy",
            Box::new(
                ExpressionOperator::new("copy", "", vec!["value".into()], None, vec![]).unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([("output".into(), DeliveryGuarantee::ExactlyOnce)]),
            },
        )
        .unwrap();
    let sink = MySqlSinkFactory::new()
        .open_transactional(&transactional("mysql_invalid_checkpoint"), url)
        .await
        .unwrap()
        .unwrap();
    StreamingRunner::new(
        plan,
        BTreeMap::from([(
            "input".into(),
            SourceBinding::new(source)
                .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        BTreeMap::from([(
            "output".into(),
            vec![SinkBinding::transactional("values", FactorySink(sink)).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap()
    .with_runtime_config(StreamRuntimeConfig {
        checkpoint_interval: Duration::from_secs(3600),
        ..StreamRuntimeConfig::default()
    })
    .unwrap()
}

async fn manifests(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut entries = tokio::fs::read_dir(root.join("manifests")).await.unwrap();
    let mut manifests = BTreeMap::new();
    while let Some(entry) = entries.next_entry().await.unwrap() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("manifest-")
            && entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "json")
        {
            manifests.insert(name, tokio::fs::read(entry.path()).await.unwrap());
        }
    }
    manifests
}

#[tokio::test]
#[ignore = "requires MySQL 8.4 and CALC_FLOW_CONNECTOR_CONTAINERS=1"]
async fn invalid_values_do_not_publish_a_manifest_and_previous_checkpoint_recovers() {
    let url = test_url();
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut conn = admin(&url).await;
        conn.query_drop("DROP TABLE IF EXISTS mysql_invalid_checkpoint")
            .await
            .unwrap();
        conn.query_drop(
            "CREATE TABLE mysql_invalid_checkpoint (value DOUBLE NOT NULL) ENGINE=InnoDB",
        )
        .await
        .unwrap();
        let root = tempfile::tempdir().unwrap();
        let probe = Arc::new(Probe::default());
        let source = |values, pause_at| ReplaySource {
            values,
            offset: 0,
            pause_at,
            probe: probe.clone(),
        };

        let first = runner(root.path(), &url, source(vec![1.0], Some(1)))
            .await
            .start()
            .await
            .unwrap();
        conn.exec_drop(
            "DELETE FROM calc_flow_mysql_epoch_ledger WHERE identity_hash = UNHEX(SHA2(?, 256))",
            (serde_json::to_string(&(
                "mysql_test",
                "mysql_invalid_checkpoint",
                "mysql_invalid_checkpoint",
            ))
            .unwrap(),),
        )
        .await
        .unwrap();
        probe.paused.notified().await;
        let epoch = first.trigger_checkpoint().await.unwrap();
        assert_eq!(first.cancel().await.state, JobState::Cancelled);
        let saved = manifests(root.path()).await;
        assert_eq!(saved.len(), 1);
        let manifest = CheckpointManifest::from_bytes(saved.values().next().unwrap()).unwrap();
        assert_eq!(manifest.epoch(), epoch);

        let bad = runner(root.path(), &url, source(vec![1.0, f64::NAN], None))
            .await
            .start()
            .await
            .unwrap();
        let outcome = bad.wait().await;
        assert!(!outcome.errors.is_empty());
        assert!(
            manifests(root.path()).await == saved,
            "invalid epoch must not replace the durable recovery cut"
        );
        assert_eq!(outcome.state, JobState::Failed);
        let stored: Vec<f64> = conn
            .query("SELECT value FROM mysql_invalid_checkpoint ORDER BY value")
            .await
            .unwrap();
        assert_eq!(stored, [1.0]);

        let resumed = runner(root.path(), &url, source(vec![1.0, 2.0], None))
            .await
            .start()
            .await
            .unwrap();
        assert_eq!(resumed.wait().await.state, JobState::Completed);
        assert_eq!(*probe.offsets.lock().unwrap(), [0, 1, 1]);
        let stored: Vec<f64> = conn
            .query("SELECT value FROM mysql_invalid_checkpoint ORDER BY value")
            .await
            .unwrap();
        assert_eq!(stored, [1.0, 2.0]);
        assert!(manifests(root.path()).await.len() > saved.len());
        conn.disconnect().await.unwrap();
    })
    .await
    .expect("checkpoint recovery scenario must settle");
}
