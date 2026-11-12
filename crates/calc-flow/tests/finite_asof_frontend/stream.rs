use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, Cursor, JsonMap, NativeWatermarkCapability, ReplayPositioning, Result,
    SinkBinding, SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent,
    SourceSchema, StreamSink, StreamSource, StreamingRunner,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

struct Source {
    batch: Option<Batch>,
    schema: calc_flow::arrow::datatypes::SchemaRef,
    opened: Arc<AtomicUsize>,
}
#[async_trait]
impl StreamSource for Source {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::Unsupported,
            delivery: SourceDeliveryCapability::Lossy,
            max_batch_rows: 100,
            max_batch_bytes: 1 << 20,
            schema: SourceSchema::Exact(self.schema.clone()),
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        assert!(cursor.is_none());
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.batch
            .take()
            .map(|batch| {
                Ok(SourceEvent::Data {
                    batch: batch.with_metadata(BatchMetadata::new("source", 1, JsonMap::new())?),
                    cursor: Cursor::unbound(vec![1], JsonMap::new())?,
                })
            })
            .transpose()
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct Sink(Arc<Mutex<Vec<Batch>>>);
#[async_trait]
impl StreamSink for Sink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0.lock().unwrap().push(batch.clone());
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
pub(super) async fn verify(
    plan: calc_flow::StreamExecutionPlan,
    inputs: BTreeMap<String, Batch>,
) -> Result<()> {
    let opened = Arc::new(AtomicUsize::new(0));
    let sources = inputs
        .into_iter()
        .map(|(name, batch)| {
            let source = Source {
                schema: batch.table_payload()?.schema().clone(),
                batch: Some(batch),
                opened: opened.clone(),
            };
            Ok((
                name,
                SourceBinding::new(source).with_watermark_policy(
                    calc_flow::WatermarkPolicy::BoundedOutOfOrderness {
                        event_time_column: "time".into(),
                        max_out_of_orderness: std::time::Duration::from_micros(1),
                        emit_interval: std::time::Duration::from_millis(1),
                        idle_timeout: None,
                    },
                ),
            ))
        })
        .collect::<Result<_>>()?;
    let mut sinks = BTreeMap::new();
    let mut captured = Vec::new();
    for name in plan.external_outputs().keys() {
        let batches = Arc::new(Mutex::new(Vec::new()));
        sinks.insert(
            name.clone(),
            vec![SinkBinding::ordinary(name, Sink(batches.clone()))?],
        );
        captured.push(batches);
    }
    let outcome = StreamingRunner::without_checkpoints(plan, sources, sinks)?
        .start()
        .await?
        .wait()
        .await;
    assert_eq!(outcome.state, calc_flow::JobState::Completed, "{outcome:?}");
    assert!(outcome.errors.is_empty());
    assert_eq!(opened.load(Ordering::SeqCst), 2);
    for batches in captured {
        assert_eq!(super::values(&batches.lock().unwrap()), vec![Some(7)]);
    }
    Ok(())
}
