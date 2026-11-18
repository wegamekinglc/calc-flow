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
    time::Duration,
};

struct TestSource {
    opened: Arc<AtomicUsize>,
    boundedness: SourceBoundedness,
    native_watermarks: NativeWatermarkCapability,
}
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
        Field::new(
            "other_ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("q", DataType::Int64, true),
    ]))
}
#[async_trait::async_trait]
impl StreamSource for TestSource {
    fn boundedness(&self) -> SourceBoundedness {
        self.boundedness
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::Unsupported,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 2,
            max_batch_bytes: 1024,
            schema: SourceSchema::Exact(schema()),
            native_watermarks: self.native_watermarks,
        }
    }
    async fn open(&mut self, _: Option<Cursor>) -> Result<()> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(None)
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn distinct_window(input: &TableExpr, time: &str) -> TableExpr {
    input
        .distinct(
            vec!["q".into()],
            DistinctSpec::closed_window(
                time,
                EventTime::from_micros(0),
                EventTime::from_micros(10),
                StateBudget::new(2, 8 << 20).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
}

async fn start_window(
    output: TableExpr,
    source: TestSource,
    policy: WatermarkPolicy,
) -> Result<StreamingJob> {
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
    StreamingRunner::without_checkpoints(
        plan,
        BTreeMap::from([(
            "events".into(),
            SourceBinding::new(source).with_watermark_policy(policy),
        )]),
        BTreeMap::from([(
            "out".into(),
            vec![SinkBinding::ordinary("sink", Sink).unwrap()],
        )]),
    )
    .unwrap()
    .start()
    .await
}

fn assert_finality_unproven(error: CalcFlowError, case: &str) {
    let CalcFlowError::Streaming(error) = error else {
        panic!("{case}: {error}");
    };
    assert_eq!(
        error.category(),
        StreamingErrorCategory::Validation,
        "{case}"
    );
    assert_eq!(
        error.component_kind(),
        Some(ComponentKind::Source),
        "{case}"
    );
    assert_eq!(error.component_id(), Some("events"), "{case}");
    assert_eq!(
        error.message(),
        "stream.inputs.events.watermark_policy: finality_unproven/unknown: selected window requires a proved closure route",
        "{case}",
    );
}

#[tokio::test]
async fn test_window_rejects_unknown_source_without_closure_before_open() {
    let opened = Arc::new(AtomicUsize::new(0));
    let input = table_input("events", schema(), InputOptions::default()).unwrap();
    let error = start_window(
        distinct_window(&input, "ts"),
        TestSource {
            opened: opened.clone(),
            boundedness: SourceBoundedness::Unknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        },
        WatermarkPolicy::Disabled { idle_timeout: None },
    )
    .await
    .expect_err("missing closure accepted");
    assert_finality_unproven(error, "unknown source without watermarks");
    assert_eq!(opened.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn test_distinct_and_order_window_finality_after_time_renames() {
    let input = table_input("events", schema(), InputOptions::default()).unwrap();
    let projected = input
        .select(vec![
            input.column("ts").unwrap().alias("projected_ts").unwrap(),
            input.column("q").unwrap().alias("q").unwrap(),
        ])
        .unwrap();
    let renamed = sql::sql(
        "SELECT projected_ts AS window_ts, q FROM projected",
        BTreeMap::from([("projected".into(), projected)]),
    )
    .unwrap();
    let order = renamed
        .top_k(
            1,
            vec![SortKey::new("q", SortDirection::Ascending, NullPlacement::Last).unwrap()],
            SortSpec::closed_window(
                SortWindow::new(
                    "window_ts",
                    EventTime::from_micros(0),
                    EventTime::from_micros(10),
                )
                .unwrap(),
                StateBudget::new(2, 8 << 20).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let generated = |column: &str| WatermarkPolicy::BoundedOutOfOrderness {
        event_time_column: column.into(),
        max_out_of_orderness: Duration::from_micros(1),
        emit_interval: Duration::from_millis(1),
        idle_timeout: None,
    };
    for (operator, output) in [
        ("DISTINCT", distinct_window(&renamed, "window_ts")),
        ("ORDER", order),
    ] {
        for (case, boundedness, native_watermarks, policy, accepted) in [
            (
                "generated watermark on the original timestamp",
                SourceBoundedness::Unknown,
                NativeWatermarkCapability::NeverEmits,
                generated("ts"),
                true,
            ),
            (
                "generated watermark on a different timestamp",
                SourceBoundedness::Unknown,
                NativeWatermarkCapability::NeverEmits,
                generated("other_ts"),
                false,
            ),
            (
                "source-provided watermarks",
                SourceBoundedness::Unknown,
                NativeWatermarkCapability::EmitsNative,
                WatermarkPolicy::SourceProvided,
                true,
            ),
            (
                "bounded EOF with watermarks disabled",
                SourceBoundedness::Bounded,
                NativeWatermarkCapability::NeverEmits,
                WatermarkPolicy::Disabled { idle_timeout: None },
                true,
            ),
        ] {
            let opened = Arc::new(AtomicUsize::new(0));
            let result = start_window(
                output.clone(),
                TestSource {
                    opened: opened.clone(),
                    boundedness,
                    native_watermarks,
                },
                policy,
            )
            .await;
            let context = format!("{operator} with {case}");
            if accepted {
                let job = result.unwrap_or_else(|error| panic!("{context}: {error}"));
                assert_eq!(job.wait().await.state, JobState::Completed, "{context}");
                assert_eq!(opened.load(Ordering::SeqCst), 1, "{context}");
            } else {
                assert_finality_unproven(result.expect_err(&context), &context);
                assert_eq!(opened.load(Ordering::SeqCst), 0, "{context}");
            }
        }
    }
}
