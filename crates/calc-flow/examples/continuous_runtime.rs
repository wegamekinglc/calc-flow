use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, Cursor, ExpressionOperator, ManagedCheckpointRuntime,
    NativeWatermarkCapability, PipelineBuilder, ReplayPositioning, Result, SinkBinding,
    SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema,
    StreamRequirements, StreamSink, StreamSource, StreamingRunner, UdfRegistry, WatermarkPolicy,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
use tokio::sync::Mutex;

struct ReplaySource {
    values: Vec<i64>,
    offset: usize,
}

#[async_trait]
impl StreamSource for ReplaySource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 1,
            max_batch_bytes: 1024,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.offset = cursor
            .as_ref()
            .and_then(|cursor| cursor.payload().get("offset"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|offset| usize::try_from(offset).ok())
            .unwrap_or_default();
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let Some(value) = self.values.get(self.offset).copied() else {
            return Ok(None);
        };
        self.offset += 1;
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![value])) as _,
        )])
        .map_err(|error| calc_flow::CalcFlowError::InvalidArgument {
            field: "example.batch".into(),
            message: error.to_string(),
        })?;
        let batch = Batch::table(vec![record], BatchMetadata::default())?;
        let cursor = Cursor::unbound(
            u64::try_from(self.offset)
                .unwrap_or(u64::MAX)
                .to_be_bytes()
                .to_vec(),
            BTreeMap::from([("offset".into(), serde_json::json!(self.offset))]),
        )?;
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct CollectSink {
    values: Arc<Mutex<Vec<i64>>>,
}

#[async_trait]
impl StreamSink for CollectSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        let values = batch.table_payload().expect("table output").batches()[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("i64 result")
            .values();
        self.values.lock().await.extend(values.iter().copied());
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let plan = PipelineBuilder::new("continuous-example")?
        .add_node(
            "calculate",
            Box::new(ExpressionOperator::new(
                "calculate",
                "result = value + 1",
                Vec::new(),
                None,
                Vec::new(),
            )?),
        )?
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )?;
    let source_id = plan.source_binding_ids()[0].to_owned();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    let values = Arc::new(Mutex::new(Vec::new()));
    let directory = tempfile::tempdir().map_err(|source| calc_flow::CalcFlowError::Io {
        path: "temporary checkpoint directory".into(),
        source,
    })?;
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source_id,
            SourceBinding::new(ReplaySource {
                values: vec![1, 2, 3],
                offset: 0,
            })
            .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        BTreeMap::from([(
            output_id,
            vec![SinkBinding::ordinary(
                "collector",
                CollectSink {
                    values: Arc::clone(&values),
                },
            )?],
        )]),
        ManagedCheckpointRuntime::new(directory.path())?,
    )?;
    let job = runner.start().await?;
    let outcome = job.wait().await;

    println!("terminal state: {:?}", outcome.state);
    println!("completed epoch: {:?}", outcome.completed_epoch);
    println!("results: {:?}", *values.lock().await);
    Ok(())
}
