use calc_flow::{
    arrow::datatypes::{DataType, Field, Schema, TimeUnit},
    *,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

struct NoProgress(Arc<AtomicUsize>);
struct Sink;
#[async_trait::async_trait]
impl StreamSink for Sink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, _: &Batch) -> Result<()> {
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("q", DataType::Int64, true),
    ]))
}
#[async_trait::async_trait]
impl StreamSource for NoProgress {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::Unsupported,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 2,
            max_batch_bytes: 1024,
            schema: SourceSchema::Exact(schema()),
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, _: Option<Cursor>) -> Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(None)
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
#[tokio::test]
async fn test_window_rejects_unknown_source_without_closure_before_open() {
    let opened = Arc::new(AtomicUsize::new(0));
    let input = table_input("events", schema(), InputOptions::default()).unwrap();
    let output = input
        .distinct(
            vec!["q".into()],
            DistinctSpec::closed_window(
                "ts",
                EventTime::from_micros(0),
                EventTime::from_micros(10),
                StateBudget::new(2, 8 << 20).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let program = Program::new(
        "closure",
        CompileMode::Stream,
        BTreeMap::from([("out".into(), output)]),
    )
    .unwrap();
    let plan = program
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await
        .unwrap();
    let runner = StreamingRunner::without_checkpoints(
        plan,
        BTreeMap::from([(
            "events".into(),
            SourceBinding::new(NoProgress(opened.clone()))
                .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        BTreeMap::from([(
            "out".into(),
            vec![SinkBinding::ordinary("sink", Sink).unwrap()],
        )]),
    )
    .unwrap();
    let Err(error) = runner.start().await else {
        panic!("missing closure accepted")
    };
    assert!(error.to_string().contains("finality_unproven"), "{error}");
    assert_eq!(opened.load(Ordering::SeqCst), 0);
}
