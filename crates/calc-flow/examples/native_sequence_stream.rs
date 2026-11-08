//! Chunked native sequence calculation with a durable terminal restart.

use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, Cursor, ExternalOperatorSpec, JobState,
    JsonMap, ManagedCheckpointRuntime, NativeWatermarkCapability, PipelineBuilder, Port,
    ReplayPositioning, Result, SinkBinding, SourceBinding, SourceCapabilities,
    SourceDeliveryCapability, SourceEvent, SourceSchema, StreamExecutionPlan,
    StreamOperatorFactory, StreamRequirements, StreamSink, StreamSource, StreamingRunner,
    UdfRegistry, WatermarkPolicy,
    array::{FloatValues, NativeArray, NativeArraySequenceFactory},
};

struct Prices {
    binding: String,
    position: usize,
    opens: Arc<AtomicUsize>,
}
#[async_trait]
impl StreamSource for Prices {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 2,
            max_batch_bytes: 1024,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        self.position = cursor.map_or(0, |cursor| usize::from(cursor.order()[0]));
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let chunks = [[1.0, 3.0], [5.0, 7.0]];
        let Some(values) = chunks.get(self.position) else {
            return Ok(None);
        };
        let array = NativeArray::from_f64(values.to_vec(), vec![2], None)?;
        self.position += 1;
        Ok(Some(SourceEvent::Data {
            batch: Batch::external(Arc::new(array), BatchMetadata::default())?,
            cursor: Cursor::new(
                &self.binding,
                vec![u8::try_from(self.position).expect("two chunks")],
                JsonMap::new(),
            )?,
        }))
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct Results(Arc<Mutex<Vec<f64>>>);
#[async_trait]
impl StreamSink for Results {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        let payload = batch.external_payload()?;
        let array = payload
            .as_any()
            .downcast_ref::<NativeArray>()
            .expect("native factory output");
        let FloatValues::Float64(values) = array.float_view()?.values() else {
            return Err(CalcFlowError::Internal {
                message: "native sequence output must be float64".into(),
            });
        };
        self.0
            .lock()
            .expect("owned results")
            .extend_from_slice(values);
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn plan() -> Result<StreamExecutionPlan> {
    let options: JsonMap = serde_json::from_value(serde_json::json!({
        "ir_version":1, "input_dtype":"float64", "kernel":{"function":"ewma", "span":3},
    }))
    .expect("JSON object");
    let spec = ExternalOperatorSpec::new("native", "array_sequence", "1", options)?;
    let operator = NativeArraySequenceFactory::default().create(
        &spec,
        vec![Port::new("input", BatchKind::Array, true, None)?],
        vec![Port::new("output", BatchKind::Array, true, None)?],
    )?;
    PipelineBuilder::new("native-sequence-example")?
        .add_node("ewma", operator)?
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
}
fn runner(
    root: &Path,
    opens: Arc<AtomicUsize>,
    results: Arc<Mutex<Vec<f64>>>,
) -> Result<StreamingRunner> {
    let plan = plan()?;
    let input = plan.source_binding_ids()[0].to_owned();
    let output = plan.sink_binding_ids()[0].to_owned();
    StreamingRunner::new(
        plan,
        BTreeMap::from([(
            input.clone(),
            SourceBinding::new(Prices {
                binding: input,
                position: 0,
                opens,
            })
            .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        BTreeMap::from([(
            output,
            vec![SinkBinding::ordinary("results", Results(results))?],
        )]),
        ManagedCheckpointRuntime::new(root)?,
    )
}
async fn complete(runner: StreamingRunner) -> Result<()> {
    let job = runner.start().await?;
    let outcome = job.wait().await;
    if outcome.state != JobState::Completed {
        return Err(CalcFlowError::Internal {
            message: "sequence example did not complete".into(),
        });
    }
    Ok(())
}
fn verify(opens: &AtomicUsize, results: &Mutex<Vec<f64>>) -> Result<()> {
    if opens.load(Ordering::SeqCst) != 1 {
        return Err(CalcFlowError::Internal {
            message: "terminal restart reopened the source".into(),
        });
    }
    if results.lock().expect("owned results").as_slice() != [1.0, 2.0, 3.5, 5.25] {
        return Err(CalcFlowError::Internal {
            message: "unexpected sequence results".into(),
        });
    }
    Ok(())
}
#[tokio::main]
async fn main() -> Result<()> {
    let directory = tempfile::tempdir().map_err(|source| CalcFlowError::Io {
        path: "temporary checkpoint directory".into(),
        source,
    })?;
    let opens = Arc::new(AtomicUsize::new(0));
    let results = Arc::new(Mutex::new(Vec::new()));
    complete(runner(
        directory.path(),
        Arc::clone(&opens),
        Arc::clone(&results),
    )?)
    .await?;
    complete(runner(
        directory.path(),
        Arc::clone(&opens),
        Arc::clone(&results),
    )?)
    .await?;
    verify(&opens, &results)?;
    println!(
        "EWMA: {:?}; terminal restart validated without reopening source",
        results.lock().expect("owned results")
    );
    Ok(())
}
