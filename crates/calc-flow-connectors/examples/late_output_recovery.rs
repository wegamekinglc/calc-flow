//! Replay a persisted source trace into two independent transactional file sinks.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use arrow::{
    array::{Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, Cursor, DecodeBounds, DeliveryGuarantee, EventTime,
    FormatDecoder, JobState, JsonMap, ManagedCheckpointRuntime, NativeWatermarkCapability,
    PipelineBuilder, ReplayPositioning, Result, RollingOperator, SinkBinding, SourceBinding,
    SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema, StreamRequirements,
    StreamRuntimeConfig, StreamSource, StreamingJob, StreamingRunner, UdfRegistry, WatermarkPolicy,
};
use calc_flow_connectors::{FileSinkConfig, TransactionalParquetSink, parquet::ParquetCodec};
use serde_json::{Value, json};
use tokio::sync::Notify;

type ExampleResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const TRACE: &str = r#"[
  {"watermark":10},
  {"times":[20],"sequence":0},
  {"watermark":30},
  {"times":[6,40],"sequence":1},
  {"watermark":50}
]"#;
const DEADLINE: Duration = Duration::from_secs(30);

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("key", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("x", DataType::Float64, true),
    ]))
}

fn batch(times: &[i32], sequence: u64) -> ExampleResult<Batch> {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from(
                    times.iter().copied().map(i64::from).collect::<Vec<_>>(),
                )
                .with_timezone("UTC"),
            ) as ArrayRef,
            Arc::new(StringArray::from(vec!["a"; times.len()])),
            Arc::new(UInt64Array::from(
                (0..times.len())
                    .map(|row| Ok(sequence * 10 + u64::try_from(row)?))
                    .collect::<ExampleResult<Vec<_>>>()?,
            )),
            Arc::new(Float64Array::from(
                times.iter().copied().map(f64::from).collect::<Vec<_>>(),
            )),
        ],
    )?;
    Ok(Batch::table(
        vec![record],
        BatchMetadata::new("trace-v1", sequence, JsonMap::new())?,
    )?)
}

fn cursor(offset: usize) -> Result<Cursor> {
    Cursor::unbound(
        u64::try_from(offset)
            .expect("bounded trace offset")
            .to_be_bytes()
            .to_vec(),
        BTreeMap::from([("offset".into(), json!(offset))]),
    )
}

async fn load_trace(root: &Path) -> ExampleResult<Vec<SourceEvent>> {
    let bytes = tokio::fs::read(root.join("trace-v1.json")).await?;
    // This example's source identity names one immutable trace and one new lineage.
    if bytes != TRACE.as_bytes() {
        return Err(
            "trace-v1.json changed; use a new root for a different source or policy".into(),
        );
    }
    let records: Vec<Value> = serde_json::from_slice(&bytes)?;
    records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            if let Some(watermark) = record["watermark"].as_i64() {
                return Ok(SourceEvent::Watermark(EventTime::from_micros(watermark)));
            }
            let times: Vec<i32> = serde_json::from_value(record["times"].clone())?;
            Ok(SourceEvent::Data {
                batch: batch(
                    &times,
                    record["sequence"].as_u64().ok_or("missing sequence")?,
                )?,
                cursor: cursor(index + 1)?,
            })
        })
        .collect()
}

#[derive(Default)]
struct Lifecycle {
    paused: Notify,
    opens: Mutex<Vec<usize>>,
    closes: AtomicUsize,
}

struct ReplaySource {
    events: Vec<SourceEvent>,
    offset: usize,
    pause_at: Option<usize>,
    lifecycle: Arc<Lifecycle>,
}

#[async_trait]
impl StreamSource for ReplaySource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 2,
            max_batch_bytes: 1 << 20,
            schema: SourceSchema::Exact(schema()),
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.offset = match cursor {
            None => 0,
            Some(cursor) => cursor
                .payload()
                .get("offset")
                .and_then(Value::as_u64)
                .and_then(|offset| usize::try_from(offset).ok())
                .filter(|offset| *offset <= self.events.len())
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: "invalid trace-v1 source offset".into(),
                })?,
        };
        self.lifecycle
            .opens
            .lock()
            .expect("source lifecycle lock")
            .push(self.offset);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.pause_at == Some(self.offset) {
            self.lifecycle.paused.notify_one();
            return std::future::pending().await;
        }
        let event = self.events.get(self.offset).cloned();
        self.offset += usize::from(event.is_some());
        Ok(event)
    }

    async fn close(&mut self) -> Result<()> {
        self.lifecycle.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

async fn runner(
    root: &Path,
    pause_at: Option<usize>,
    lifecycle: Arc<Lifecycle>,
) -> ExampleResult<StreamingRunner> {
    let operator = RollingOperator::new(
        "roll",
        schema(),
        serde_json::from_value(json!({
            "configuration_version":1, "state_layout_version":1,
            "partition_by":["key"], "event_time":"ts", "sequence_by":["seq"],
            "outputs":[{"kind":"lag", "primitive_version":1, "input":"x", "output":"lag", "periods":1}],
            "allowed_lateness_micros":0,
            "late_policy":{"kind":"side_output", "metrics_version":1, "schema_version":1},
            "value_policy":"stateful_numeric_v1"
        }))?,
    )?;
    let plan = PipelineBuilder::new("late-files-v1")?
        .add_node("roll", Box::new(operator))?
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([
                    ("output".into(), DeliveryGuarantee::ExactlyOnce),
                    ("late".into(), DeliveryGuarantee::ExactlyOnce),
                ]),
            },
        )?;
    assert_eq!(plan.source_binding_ids(), ["input"]);
    assert_eq!(plan.sink_binding_ids(), ["late", "output"]);
    let sinks = [("output", "normal"), ("late", "late")]
        .into_iter()
        .map(|(binding, name)| {
            let sink = TransactionalParquetSink::new(FileSinkConfig {
                root: root.join("outputs"),
                output: name.into(),
            })?;
            Ok((
                binding.into(),
                vec![SinkBinding::transactional(name, sink)?],
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let source = ReplaySource {
        events: load_trace(root).await?,
        offset: 0,
        pause_at,
        lifecycle,
    };
    Ok(StreamingRunner::new(
        plan,
        BTreeMap::from([(
            "input".into(),
            SourceBinding::new(source).with_watermark_policy(WatermarkPolicy::SourceProvided),
        )]),
        sinks,
        ManagedCheckpointRuntime::new(root.join("state"))?,
    )?
    .with_runtime_config(StreamRuntimeConfig {
        checkpoint_interval: Duration::from_secs(3600),
        ..StreamRuntimeConfig::default()
    })?)
}

fn check_settled(job: &StreamingJob, lifecycle: &Lifecycle, opens: &[usize]) {
    let status = job.status();
    assert_eq!(status.task_count, 0);
    assert!(status.edges.values().all(|edge| edge.current_envelopes == 0
        && edge.current_rows == 0
        && edge.current_bytes == 0));
    assert_eq!(
        *lifecycle.opens.lock().expect("source lifecycle lock"),
        opens
    );
    assert_eq!(lifecycle.closes.load(Ordering::SeqCst), opens.len());
    assert_eq!(status.delivery.len(), 2);
    for binding in ["output", "late"] {
        assert_eq!(
            status.delivery[binding].requested,
            DeliveryGuarantee::ExactlyOnce
        );
        assert_eq!(
            status.delivery[binding].effective,
            DeliveryGuarantee::ExactlyOnce
        );
    }
}

async fn cut(root: &Path) -> ExampleResult<()> {
    // Refuse existing roots: do not reuse a Drop/Error lineage or overwrite user data.
    tokio::fs::create_dir(root).await?;
    tokio::fs::write(root.join("trace-v1.json"), TRACE).await?;
    let lifecycle = Arc::new(Lifecycle::default());
    let job = runner(root, Some(2), lifecycle.clone())
        .await?
        .start()
        .await?;
    let checkpoint = tokio::time::timeout(DEADLINE, async {
        lifecycle.paused.notified().await;
        job.trigger_checkpoint().await
    })
    .await;
    let stopped = job.cancel().await;
    check_settled(&job, &lifecycle, &[0]);
    let epoch = checkpoint??;
    assert_eq!(epoch.as_u64(), 1);
    assert_eq!(stopped.state, JobState::Cancelled);
    assert_eq!(stopped.completed_epoch, Some(epoch));
    for name in ["normal", "late"] {
        let path = root
            .join("outputs")
            .join(name)
            .join("epoch=1/manifest.json");
        let manifest: Value = serde_json::from_slice(&tokio::fs::read(path).await?)?;
        assert_eq!(manifest["epoch"], 1);
        assert_eq!(manifest["output"], name);
        assert_eq!(manifest["rows"], 0);
        assert_eq!(manifest["parts"], json!([]));
    }
    println!("cut: epoch=1, source next offset=2, normal/late manifests each contain 0 rows");
    Ok(())
}

fn committed_files(root: &Path, name: &str) -> ExampleResult<BTreeMap<PathBuf, Vec<u8>>> {
    let directory = root.join("outputs").join(name);
    let mut files = BTreeMap::new();
    for epoch in std::fs::read_dir(&directory)? {
        let epoch = epoch?;
        if !epoch.file_name().to_string_lossy().starts_with("epoch=") {
            continue;
        }
        for file in std::fs::read_dir(epoch.path())? {
            let path = file?.path();
            files.insert(
                path.strip_prefix(&directory)?.to_path_buf(),
                std::fs::read(path)?,
            );
        }
    }
    Ok(files)
}

fn output_rows(root: &Path, name: &str) -> ExampleResult<Vec<(i64, u64)>> {
    let mut rows = Vec::new();
    for (path, bytes) in committed_files(root, name)? {
        if path
            .extension()
            .is_none_or(|extension| extension != "parquet")
        {
            continue;
        }
        let batch =
            ParquetCodec::new("1")?.decode(&bytes, &DecodeBounds::new(100, 1 << 20)?, &[])?;
        for record in batch.table_payload().expect("Parquet table").batches() {
            check_values(record, name);
            let times = column::<TimestampMicrosecondArray>(record, "ts");
            let sequences = column::<UInt64Array>(record, "seq");
            rows.extend((0..record.num_rows()).map(|row| (times.value(row), sequences.value(row))));
        }
    }
    rows.sort_unstable();
    Ok(rows)
}

fn column<'a, T: Array + 'static>(record: &'a RecordBatch, name: &str) -> &'a T {
    record
        .column_by_name(name)
        .expect("expected output column")
        .as_any()
        .downcast_ref::<T>()
        .expect("expected output Arrow type")
}

fn check_values(record: &RecordBatch, name: &str) {
    let times = column::<TimestampMicrosecondArray>(record, "ts");
    let values = column::<Float64Array>(record, "x");
    for row in 0..record.num_rows() {
        if name == "normal" {
            let lag = column::<Float64Array>(record, "lag");
            match times.value(row) {
                20 => {
                    check_exact_value(values.value(row), 20.0);
                    assert!(lag.is_null(row));
                }
                40 => {
                    check_exact_value(values.value(row), 40.0);
                    check_exact_value(lag.value(row), 20.0);
                    assert!(!lag.is_null(row));
                }
                other => panic!("unexpected normal timestamp: {other}"),
            }
        } else {
            check_exact_value(values.value(row), 6.0);
            assert_eq!(
                column::<StringArray>(record, "_cf_late_node").value(row),
                "roll"
            );
            assert_eq!(
                column::<StringArray>(record, "_cf_late_source").value(row),
                "input"
            );
            assert_eq!(
                column::<UInt64Array>(record, "_cf_late_sequence").value(row),
                1
            );
            assert_eq!(
                column::<UInt64Array>(record, "_cf_late_row_index").value(row),
                0
            );
        }
    }
}

fn check_exact_value(actual: f64, expected: f64) {
    // The source literals and lag are copied exactly, without floating-point arithmetic.
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{actual} != {expected}"
    );
}

fn check_outputs(root: &Path) -> ExampleResult<()> {
    let normal = output_rows(root, "normal")?;
    let late = output_rows(root, "late")?;
    assert_eq!(normal, [(20, 0), (40, 11)]);
    assert_eq!(late, [(6, 10)]);
    println!("independent readback (ts, seq): normal={normal:?}, late={late:?}");
    Ok(())
}

async fn complete(root: &Path, opens: &[usize]) -> ExampleResult<u64> {
    let lifecycle = Arc::new(Lifecycle::default());
    let job = runner(root, None, lifecycle.clone()).await?.start().await?;
    let result = tokio::time::timeout(DEADLINE, job.wait()).await;
    job.cancel().await;
    check_settled(&job, &lifecycle, opens);
    let outcome = result?;
    assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
    Ok(outcome
        .completed_epoch
        .ok_or("missing terminal epoch")?
        .as_u64())
}

async fn resume(root: &Path) -> ExampleResult<()> {
    let terminal = complete(root, &[2]).await?;
    assert!(terminal > 1);
    check_outputs(root)?;
    tokio::fs::write(root.join("terminal-epoch.txt"), terminal.to_string()).await?;
    println!("resume: opened source at offset=2, terminal epoch={terminal}");
    Ok(())
}

async fn verify(root: &Path) -> ExampleResult<()> {
    let expected: u64 = tokio::fs::read_to_string(root.join("terminal-epoch.txt"))
        .await?
        .parse()?;
    let before = [
        committed_files(root, "normal")?,
        committed_files(root, "late")?,
    ];
    assert_eq!(complete(root, &[]).await?, expected);
    assert_eq!(
        [
            committed_files(root, "normal")?,
            committed_files(root, "late")?
        ],
        before
    );
    check_outputs(root)?;
    println!(
        "terminal restart: epoch={expected}, Source.open=0, both committed directories unchanged"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> ExampleResult<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [] => {
            let temporary = tempfile::tempdir()?;
            let root = temporary.path().join("late-files-v1");
            cut(&root).await?;
            resume(&root).await?;
            verify(&root).await?;
            temporary.close()?;
            println!("removed temporary demo state and outputs");
        }
        [phase, root] => match phase.to_str() {
            Some("cut") => cut(Path::new(root)).await?,
            Some("resume") => resume(Path::new(root)).await?,
            Some("verify") => verify(Path::new(root)).await?,
            _ => return Err("phase must be cut, resume, or verify".into()),
        },
        _ => return Err("usage: late_output_recovery [cut|resume|verify NEW_ROOT]".into()),
    }
    Ok(())
}
