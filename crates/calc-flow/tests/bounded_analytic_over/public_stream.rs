use calc_flow::*;
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
};

struct Source(VecDeque<SourceEvent>);
#[async_trait::async_trait]
impl StreamSource for Source {
    fn boundedness(&self) -> SourceBoundedness {
        SourceBoundedness::Bounded
    }
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::Unsupported,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 20,
            max_batch_bytes: 1 << 20,
            schema: SourceSchema::Exact(super::common::schema()),
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }
    async fn open(&mut self, cut: Option<Cursor>) -> Result<()> {
        assert!(cut.is_none());
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(self.0.pop_front())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct Sink(Arc<Mutex<Vec<Batch>>>);
#[async_trait::async_trait]
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
pub async fn verify(plan: StreamExecutionPlan) {
    let raw = [
        super::common::data(true),
        Batch::table(
            vec![arrow::record_batch::RecordBatch::new_empty(
                super::common::schema(),
            )],
            BatchMetadata::default(),
        )
        .unwrap(),
        super::common::data(false),
    ];
    let events = raw
        .into_iter()
        .enumerate()
        .map(|(i, batch)| SourceEvent::Data {
            batch: batch
                .with_metadata(BatchMetadata::new("source", i as u64 + 1, JsonMap::new()).unwrap()),
            cursor: Cursor::unbound(vec![u8::try_from(i).unwrap()], JsonMap::new()).unwrap(),
        })
        .collect::<Vec<_>>();
    let mut source = VecDeque::from(events);
    source.insert(2, SourceEvent::Watermark(EventTime::from_micros(9)));
    source.push_back(SourceEvent::Watermark(EventTime::from_micros(10)));
    let mut captured = vec![];
    let sinks = ["df", "sql"]
        .into_iter()
        .map(|name| {
            let rows = Arc::new(Mutex::new(Vec::new()));
            captured.push(rows.clone());
            (
                name.into(),
                vec![SinkBinding::ordinary(name, Sink(rows)).unwrap()],
            )
        })
        .collect();
    let outcome = StreamingRunner::without_checkpoints(
        plan,
        BTreeMap::from([(
            "events".into(),
            SourceBinding::new(Source(source))
                .with_watermark_policy(WatermarkPolicy::SourceProvided),
        )]),
        sinks,
    )
    .unwrap()
    .start()
    .await
    .unwrap()
    .wait()
    .await;
    assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
    assert!(outcome.errors.is_empty(), "{outcome:?}");
    for batches in captured {
        let rows = batches.lock().unwrap();
        assert_eq!(rows.len(), 1);
        super::common::assert_oracle(&rows[0]);
    }
}
