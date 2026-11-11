use std::{any::Any, collections::BTreeMap, hint::black_box, sync::Arc};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchMetadata, BatchOperator, BatchOperatorContext, CalcFlowError,
    CancellationToken, DataFusionConfig, DataFusionRuntime, EdgeBudget, EdgeCollector,
    ExecutionOptions, ExpressionOperator, ExternalPayload, JsonMap, OperatorMetadata,
    PipelineBuilder, Port, Result, StreamCollector, StreamJobContext, StreamMessage,
    StreamOperator, StreamOperatorContext, UdfReference, UdfRegistry, canonical_json, edge_channel,
};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
use serde_json::json;

fn expression_operator() -> ExpressionOperator {
    ExpressionOperator::new("calculate", "total = a + b", Vec::new(), None, Vec::new()).unwrap()
}

fn expression_plan() -> calc_flow::BatchExecutionPlan {
    PipelineBuilder::new("benchmark")
        .unwrap()
        .add_node("calculate", Box::new(expression_operator()))
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap()
}

fn expression_input() -> Batch {
    let record = RecordBatch::try_from_iter(vec![
        (
            "a",
            Arc::new(Int64Array::from_iter_values(0..1_024))
                as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("b", Arc::new(Int64Array::from_iter_values(1..=1_024)) as _),
    ])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn compile_expression(c: &mut Criterion) {
    c.bench_function("compile/expression", |b| {
        b.iter(|| {
            black_box(
                PipelineBuilder::new("benchmark")
                    .unwrap()
                    .add_node("calculate", Box::new(expression_operator()))
                    .unwrap()
                    .compile_batch(&UdfRegistry::new().snapshot())
                    .unwrap(),
            )
        });
    });
}

fn execute_expression(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let plan = expression_plan();
    let input = expression_input();
    c.bench_function("execute/expression_1024_rows", |b| {
        b.to_async(&runtime).iter(|| {
            let inputs = BTreeMap::from([("input".into(), input.clone())]);
            async {
                black_box(
                    plan.execute(inputs, ExecutionOptions::default())
                        .await
                        .unwrap(),
                )
            }
        });
    });
}

fn create_datafusion_runtime(c: &mut Criterion) {
    let config = DataFusionConfig::default();
    c.bench_function("execute/datafusion_runtime_new", |b| {
        b.iter(|| black_box(DataFusionRuntime::new(black_box(config)).unwrap()));
    });

    let snapshot = UdfRegistry::new().snapshot();
    let references: [UdfReference; 0] = [];
    c.bench_function("execute/datafusion_runtime_new_register_udfs", |b| {
        b.iter(|| {
            let mut runtime = DataFusionRuntime::new(black_box(config)).unwrap();
            runtime
                .register_udfs(black_box(&snapshot), black_box(&references))
                .unwrap();
            black_box(runtime)
        });
    });
}

#[derive(Debug)]
struct BenchmarkExternalPayload {
    rows: usize,
}

impl ExternalPayload for BenchmarkExternalPayload {
    fn backend(&self) -> &'static str {
        "benchmark"
    }

    fn len(&self) -> usize {
        self.rows
    }

    fn estimated_bytes(&self) -> usize {
        self.rows.saturating_mul(8)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct PassthroughOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
}

impl PassthroughOperator {
    fn new() -> Self {
        Self {
            inputs: [Port::new("input", BatchKind::Array, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Array, true, None).unwrap()],
        }
    }
}

impl OperatorMetadata for PassthroughOperator {
    fn name(&self) -> &'static str {
        "passthrough"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }
}

#[async_trait]
impl BatchOperator for PassthroughOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let output = inputs
            .get("input")
            .cloned()
            .ok_or_else(|| CalcFlowError::Internal {
                message: "benchmark passthrough input is missing".into(),
            })?;
        Ok(BTreeMap::from([("output".into(), output)]))
    }
}

fn external_passthrough_plan() -> calc_flow::BatchExecutionPlan {
    PipelineBuilder::new("external passthrough benchmark")
        .unwrap()
        .add_node(
            "passthrough",
            Box::new(PassthroughOperator::new()) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap()
}

fn external_passthrough_input() -> Batch {
    Batch::external(
        Arc::new(BenchmarkExternalPayload { rows: 1_000 }),
        BatchMetadata::default(),
    )
    .unwrap()
}

fn execute_external_passthrough(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let plan = external_passthrough_plan();
    assert!(!plan.requires_datafusion());
    let input = external_passthrough_input();
    c.bench_function("execute/external_passthrough_1000_rows", |b| {
        b.to_async(&runtime).iter(|| {
            let inputs = BTreeMap::from([("input".into(), input.clone())]);
            async {
                black_box(
                    plan.execute(inputs, ExecutionOptions::default())
                        .await
                        .unwrap(),
                )
            }
        });
    });
}

fn encode_canonical_json(c: &mut Criterion) {
    let rows = vec![json!({"b": 2, "a": 1}); 16];
    let value = json!({
        "z": rows,
        "a": {"nested": true, "sequence": 42},
    });
    c.bench_function("json/canonical_nested", |b| {
        b.iter(|| black_box(canonical_json(black_box(&value)).unwrap()));
    });
}

fn stream_message(rows: usize) -> StreamMessage {
    StreamMessage::data(
        Batch::external(
            Arc::new(BenchmarkExternalPayload { rows }),
            BatchMetadata::default(),
        )
        .unwrap(),
    )
}

fn stream_channel_roundtrip(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let budget = EdgeBudget {
        max_rows: 1_024,
        max_bytes: 1 << 20,
    };
    let data_channel = tokio::sync::Mutex::new(edge_channel("bench/data", budget).unwrap());
    let data = stream_message(128);
    c.bench_function("stream/channel_data_roundtrip", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut channel = data_channel.lock().await;
            let (sender, receiver) = &mut *channel;
            sender.send(data.clone()).await.unwrap();
            black_box(receiver.recv().await.unwrap().unwrap())
        });
    });

    // Control constructors intentionally remain crate-private. A zero-row,
    // zero-byte public data envelope exercises the same message-only channel
    // charge, queue lock, wakeup, and reservation-release path without
    // widening the runtime's control-construction API for a benchmark.
    let control_channel =
        tokio::sync::Mutex::new(edge_channel("bench/control-cost", budget).unwrap());
    let control_cost = stream_message(0);
    c.bench_function("stream/channel_control_cost_roundtrip", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut channel = control_channel.lock().await;
            let (sender, receiver) = &mut *channel;
            sender.send(control_cost.clone()).await.unwrap();
            black_box(receiver.recv().await.unwrap().unwrap())
        });
    });
}

struct StreamPassthroughOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
}

impl StreamPassthroughOperator {
    fn new() -> Self {
        Self {
            inputs: [Port::new("input", BatchKind::Array, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Array, true, None).unwrap()],
        }
    }
}

impl OperatorMetadata for StreamPassthroughOperator {
    fn name(&self) -> &'static str {
        "stream-passthrough"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }
}

#[async_trait]
impl StreamOperator for StreamPassthroughOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        batch: Batch,
        _context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        output.emit("output", batch).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: calc_flow::EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

fn stream_unary_operator_overhead(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let operator = StreamPassthroughOperator::new();
    let collector = EdgeCollector::new(operator.output_ports().to_vec());
    let state = tokio::sync::Mutex::new((operator, collector));
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "bench", JsonMap::new(), None, cancellation);
    let input = Batch::external(
        Arc::new(BenchmarkExternalPayload { rows: 128 }),
        BatchMetadata::default(),
    )
    .unwrap();
    c.bench_function("stream/unary_operator_overhead", |b| {
        b.to_async(&runtime).iter(|| async {
            let context = StreamOperatorContext::new(&job, "passthrough", None);
            let mut state = state.lock().await;
            let (operator, collector) = &mut *state;
            operator
                .process_data("input", input.clone(), &context, collector)
                .await
                .unwrap();
            black_box(collector.drain("output"))
        });
    });
}

fn stream_fanout(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let budget = EdgeBudget {
        max_rows: 1_024,
        max_bytes: 1 << 20,
    };
    let channels = (0..4)
        .map(|index| edge_channel(format!("bench/fanout/{index}"), budget).unwrap())
        .collect::<Vec<_>>();
    let channels = tokio::sync::Mutex::new(channels);
    let message = stream_message(128);
    c.bench_function("stream/fanout_four", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut channels = channels.lock().await;
            for (sender, _) in &mut *channels {
                sender.send(message.clone()).await.unwrap();
            }
            for (_, receiver) in &mut *channels {
                black_box(receiver.recv().await.unwrap().unwrap());
            }
        });
    });
}

fn stream_channel_fanin(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let budget = EdgeBudget {
        max_rows: 1_024,
        max_bytes: 1 << 20,
    };
    for (name, width) in [
        ("stream/channel_fanin_2", 2),
        ("stream/channel_fanin_4", 4),
        ("stream/channel_fanin_8", 8),
    ] {
        let channels = (0..width)
            .map(|index| edge_channel(format!("bench/fanin/{width}/{index}"), budget).unwrap())
            .collect::<Vec<_>>();
        let channels = tokio::sync::Mutex::new(channels);
        let message = stream_message(128);
        c.bench_function(name, |b| {
            b.to_async(&runtime).iter(|| async {
                let mut channels = channels.lock().await;
                for (sender, _) in &mut *channels {
                    sender.send(message.clone()).await.unwrap();
                }
                for (_, receiver) in &mut *channels {
                    black_box(receiver.recv().await.unwrap().unwrap());
                }
            });
        });
    }
}

fn stream_saturated_backpressure(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let budget = EdgeBudget {
        max_rows: 128,
        max_bytes: 1 << 20,
    };
    let channel = tokio::sync::Mutex::new(edge_channel("bench/backpressure", budget).unwrap());
    let message = stream_message(128);
    c.bench_function("stream/backpressure_saturated", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut channel = channel.lock().await;
            let (sender, receiver) = &mut *channel;
            sender.send(message.clone()).await.unwrap();
            let blocked_send = sender.send(message.clone());
            let release_capacity = async {
                tokio::task::yield_now().await;
                black_box(receiver.recv().await.unwrap().unwrap());
            };
            let (result, ()) = tokio::join!(blocked_send, release_capacity);
            result.unwrap();
            black_box(receiver.recv().await.unwrap().unwrap());
        });
    });
}

criterion_group!(
    core_benchmarks,
    compile_expression,
    create_datafusion_runtime,
    execute_expression,
    execute_external_passthrough,
    encode_canonical_json,
    stream_channel_roundtrip,
    stream_unary_operator_overhead,
    stream_fanout,
    stream_channel_fanin,
    stream_saturated_backpressure,
    rolling_state_history
);
criterion_main!(core_benchmarks);

mod rolling_history {
    use std::{
        collections::VecDeque,
        time::{Duration, Instant},
    };

    use super::{
        Arc, BTreeMap, Batch, BatchMetadata, Criterion, PipelineBuilder, RecordBatch, Result,
        UdfRegistry, async_trait, json,
    };
    use calc_flow::{
        Cursor, DeliveryGuarantee, JobState, LateErrorScope, LatePolicySpec,
        NativeWatermarkCapability, ROLLING_CONFIGURATION_VERSION, ROLLING_STATE_LAYOUT_VERSION,
        ReplayPositioning, RollingFrameSpec, RollingNumericalProfile, RollingOperator,
        RollingOutputSpec, RollingSpec, RollingValuePolicy, SinkBinding, SourceBinding,
        SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema,
        StreamRequirements, StreamRuntimeConfig, StreamSink, StreamSource, StreamingRunner,
        WatermarkPolicy,
    };
    use criterion::SamplingMode;
    use datafusion::arrow::{
        array::{
            Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
        },
        datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    };
    use sha2::{Digest, Sha256};
    use tokio::sync::Mutex;

    const T0: i64 = 1_767_225_600_000_000;
    const WINDOW: i64 = 60_000_000;
    const BATCH_ROWS: usize = 1_000;

    #[derive(Clone, Copy)]
    enum Case {
        DurationSum,
        MixedExtrema,
        RowsSum,
    }

    impl Case {
        fn name(self) -> &'static str {
            match self {
                Self::DurationSum => "duration_sum_100k_keys1000_batch1000",
                Self::MixedExtrema => "duration_min_rows_max_10k_keys64_batch1000",
                Self::RowsSum => "rows_sum20_10k_keys64_batch1000",
            }
        }

        fn rows(self) -> usize {
            if matches!(self, Self::DurationSum) {
                100_000
            } else {
                10_000
            }
        }

        fn groups(self) -> usize {
            if matches!(self, Self::DurationSum) {
                1_000
            } else {
                64
            }
        }

        fn step(self) -> i64 {
            if matches!(self, Self::DurationSum) {
                1_000
            } else {
                10_000
            }
        }

        fn outputs(self) -> Vec<RollingOutputSpec> {
            let duration = RollingFrameSpec::Duration { micros: 60_000_000 };
            let sum = |frame, output: &str| RollingOutputSpec::Sum {
                primitive_version: 1,
                input: "value".into(),
                output: output.into(),
                frame,
                min_periods: 1,
            };
            match self {
                Self::DurationSum => vec![sum(duration, "sum_60s")],
                Self::RowsSum => vec![sum(RollingFrameSpec::Rows { size: 20 }, "sum_20")],
                Self::MixedExtrema => vec![
                    RollingOutputSpec::Min {
                        primitive_version: 1,
                        input: "value".into(),
                        output: "min_60s".into(),
                        frame: duration,
                        min_periods: 1,
                    },
                    RollingOutputSpec::Max {
                        primitive_version: 1,
                        input: "value".into(),
                        output: "max_37".into(),
                        frame: RollingFrameSpec::Rows { size: 37 },
                        min_periods: 1,
                    },
                ],
            }
        }

        fn names(self) -> &'static [&'static str] {
            match self {
                Self::DurationSum => &["sum_60s"],
                Self::RowsSum => &["sum_20"],
                Self::MixedExtrema => &["min_60s", "max_37"],
            }
        }

        fn time(self, index: usize) -> i64 {
            let index = i64::try_from(index).unwrap();
            T0 + index * self.step() + index * 37 % 97
        }
    }

    fn numerator(index: usize) -> i64 {
        i64::try_from(index).unwrap() * 17 % 8192 - 4096
    }

    fn scaled(value: i64) -> f64 {
        f64::from(i32::try_from(value).unwrap()) / 1024.0
    }

    struct Fixture {
        case: Case,
        input_schema: SchemaRef,
        output_schema: SchemaRef,
        batches: Arc<[Batch]>,
        keys: Vec<String>,
        expected: Vec<Vec<f64>>,
        max_bytes: usize,
        expected_sha256: String,
    }

    impl Fixture {
        fn new(case: Case) -> Self {
            let keys = (0..case.groups())
                .map(|n| format!("target_{n:04}"))
                .collect::<Vec<_>>();
            let input_schema = Arc::new(Schema::new(vec![
                Field::new("target_id", DataType::Utf8, false),
                Field::new(
                    "event_time",
                    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                    false,
                ),
                Field::new("seq", DataType::Int64, false),
                Field::new("value", DataType::Float64, false),
            ]));
            let mut fields = input_schema.fields().to_vec();
            fields.extend(
                case.names()
                    .iter()
                    .map(|name| Arc::new(Field::new(*name, DataType::Float64, true))),
            );
            let output_schema = Arc::new(Schema::new(fields));
            let mut batches = Vec::new();
            let mut max_bytes = 0;
            for start in (0..case.rows()).step_by(BATCH_ROWS) {
                let end = (start + BATCH_ROWS).min(case.rows());
                let arrays: Vec<ArrayRef> = vec![
                    Arc::new(StringArray::from_iter_values(
                        (start..end).map(|i| keys[i % case.groups()].as_str()),
                    )),
                    Arc::new(
                        TimestampMicrosecondArray::from_iter_values(
                            (start..end).map(|i| case.time(i)),
                        )
                        .with_timezone("UTC"),
                    ),
                    Arc::new(Int64Array::from_iter_values(
                        (start..end).map(|i| i64::try_from(i).unwrap()),
                    )),
                    Arc::new(Float64Array::from_iter_values(
                        (start..end).map(|i| scaled(numerator(i))),
                    )),
                ];
                let record = RecordBatch::try_new(Arc::clone(&input_schema), arrays).unwrap();
                max_bytes = max_bytes.max(record.get_array_memory_size());
                batches.push(Batch::table(vec![record], BatchMetadata::default()).unwrap());
            }
            let expected = reference(case);
            let expected_sha256 = signature(&expected);
            Self {
                case,
                input_schema,
                output_schema,
                batches: Arc::from(batches),
                keys,
                expected,
                max_bytes,
                expected_sha256,
            }
        }
    }

    fn reference(case: Case) -> Vec<Vec<f64>> {
        let mut timed = vec![VecDeque::<(i64, i64)>::new(); case.groups()];
        let mut counted = vec![VecDeque::<i64>::new(); case.groups()];
        let mut totals = vec![0_i64; case.groups()];
        let mut expected = vec![Vec::with_capacity(case.rows()); case.names().len()];
        for row in 0..case.rows() {
            let key = row % case.groups();
            let at = case.time(row);
            let value = numerator(row);
            let time_queue = &mut timed[key];
            while time_queue
                .front()
                .is_some_and(|(time, _)| *time <= at - WINDOW)
            {
                totals[key] -= time_queue.pop_front().unwrap().1;
            }
            time_queue.push_back((at, value));
            totals[key] += value;
            let row_queue = &mut counted[key];
            row_queue.push_back(value);
            let width = if matches!(case, Case::MixedExtrema) {
                37
            } else {
                20
            };
            if row_queue.len() > width {
                row_queue.pop_front();
            }
            match case {
                Case::DurationSum => expected[0].push(scaled(totals[key])),
                Case::RowsSum => expected[0].push(scaled(row_queue.iter().sum::<i64>())),
                Case::MixedExtrema => {
                    expected[0].push(scaled(time_queue.iter().map(|(_, n)| *n).min().unwrap()));
                    expected[1].push(scaled(*row_queue.iter().max().unwrap()));
                }
            }
        }
        expected
    }

    fn signature(columns: &[Vec<f64>]) -> String {
        let mut digest = Sha256::new();
        for column in columns {
            for value in column {
                digest.update(canonical_bits(*value).to_le_bytes());
            }
        }
        hex::encode(digest.finalize())
    }

    fn canonical_bits(value: f64) -> u64 {
        (if value == 0.0 { 0.0_f64 } else { value }).to_bits()
    }

    fn assert_value(actual: f64, expected: f64) {
        assert_eq!(canonical_bits(actual), canonical_bits(expected));
    }

    struct Source {
        batches: Arc<[Batch]>,
        schema: SchemaRef,
        max_bytes: usize,
        offset: usize,
    }

    #[async_trait]
    impl StreamSource for Source {
        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replay_positioning: ReplayPositioning::Unsupported,
                delivery: SourceDeliveryCapability::Lossy,
                max_batch_rows: BATCH_ROWS,
                max_batch_bytes: self.max_bytes,
                schema: SourceSchema::Exact(Arc::clone(&self.schema)),
                native_watermarks: NativeWatermarkCapability::NeverEmits,
            }
        }
        async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
            assert!(cursor.is_none());
            self.offset = 0;
            Ok(())
        }
        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            let Some(batch) = self.batches.get(self.offset).cloned() else {
                return Ok(None);
            };
            self.offset += 1;
            Ok(Some(SourceEvent::Data {
                batch,
                cursor: Cursor::unbound(self.offset.to_be_bytes().to_vec(), BTreeMap::new())?,
            }))
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
            self.0.lock().await.push(batch.clone());
            Ok(())
        }
        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    async fn run(fixture: &Fixture) -> Vec<Batch> {
        let state = Arc::new(Mutex::new(Vec::new()));
        let spec = RollingSpec {
            configuration_version: ROLLING_CONFIGURATION_VERSION,
            state_layout_version: ROLLING_STATE_LAYOUT_VERSION,
            numerical_profile: RollingNumericalProfile::StableV1,
            partition_by: vec!["target_id".into()],
            event_time: "event_time".into(),
            sequence_by: vec!["event_time".into(), "seq".into()],
            outputs: fixture.case.outputs(),
            allowed_lateness_micros: 0,
            late_policy: LatePolicySpec::Error {
                scope: LateErrorScope::Envelope,
            },
            value_policy: RollingValuePolicy::StatefulNumericV1,
        };
        let plan = PipelineBuilder::new("rolling-history-benchmark")
            .unwrap()
            .add_node(
                "rolling",
                Box::new(
                    RollingOperator::new("rolling", Arc::clone(&fixture.input_schema), spec)
                        .unwrap(),
                ),
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements {
                    delivery: BTreeMap::from([("output".into(), DeliveryGuarantee::BestEffort)]),
                },
            )
            .unwrap();
        let source = SourceBinding::new(Source {
            batches: Arc::clone(&fixture.batches),
            schema: Arc::clone(&fixture.input_schema),
            max_bytes: fixture.max_bytes,
            offset: 0,
        })
        .with_watermark_policy(WatermarkPolicy::BoundedOutOfOrderness {
            event_time_column: "event_time".into(),
            max_out_of_orderness: Duration::from_micros(1),
            emit_interval: Duration::from_micros(1),
            idle_timeout: None,
        });
        let job = StreamingRunner::without_checkpoints(
            plan,
            BTreeMap::from([("input".into(), source)]),
            BTreeMap::from([(
                "output".into(),
                vec![SinkBinding::ordinary("collector", Sink(Arc::clone(&state))).unwrap()],
            )]),
        )
        .unwrap()
        .with_runtime_config(StreamRuntimeConfig {
            checkpointing: false,
            ..StreamRuntimeConfig::default()
        })
        .unwrap()
        .start()
        .await
        .unwrap();
        let outcome = job.wait().await;
        assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
        state.lock().await.clone()
    }

    fn validate(fixture: &Fixture, outputs: &[Batch]) {
        let mut seen = vec![false; fixture.case.rows()];
        let mut actual = vec![vec![0.0; fixture.case.rows()]; fixture.expected.len()];
        let mut count = 0;
        for batch in outputs {
            for record in batch.table_payload().unwrap().batches() {
                assert_eq!(record.schema(), fixture.output_schema);
                assert!(
                    record
                        .columns()
                        .iter()
                        .all(|column| column.null_count() == 0)
                );
                let keys = record
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let times = record
                    .column(1)
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap();
                let sequences = record
                    .column(2)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let values = record
                    .column(3)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                let derived = record.columns()[4..]
                    .iter()
                    .map(|column| column.as_any().downcast_ref::<Float64Array>().unwrap())
                    .collect::<Vec<_>>();
                for row in 0..record.num_rows() {
                    let index = usize::try_from(sequences.value(row)).unwrap();
                    assert!(index < seen.len() && !seen[index]);
                    assert_eq!(keys.value(row), fixture.keys[index % fixture.case.groups()]);
                    assert_eq!(times.value(row), fixture.case.time(index));
                    assert_value(values.value(row), scaled(numerator(index)));
                    for (ordinal, column) in derived.iter().enumerate() {
                        let value = column.value(row);
                        assert_value(value, fixture.expected[ordinal][index]);
                        actual[ordinal][index] = value;
                    }
                    seen[index] = true;
                    count += 1;
                }
            }
        }
        assert_eq!(count, fixture.case.rows());
        assert!(seen.into_iter().all(|value| value));
        assert_eq!(signature(&actual), fixture.expected_sha256);
        let mut position = 0;
        for batch in fixture.batches.iter() {
            for record in batch.table_payload().unwrap().batches() {
                assert_eq!(record.schema(), fixture.input_schema);
                let keys = record
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let times = record
                    .column(1)
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap();
                let sequences = record
                    .column(2)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let values = record
                    .column(3)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                for row in 0..record.num_rows() {
                    assert_eq!(
                        keys.value(row),
                        fixture.keys[position % fixture.case.groups()]
                    );
                    assert_eq!(times.value(row), fixture.case.time(position));
                    assert_eq!(sequences.value(row), i64::try_from(position).unwrap());
                    assert_value(values.value(row), scaled(numerator(position)));
                    position += 1;
                }
            }
        }
        assert_eq!(position, fixture.case.rows());
    }

    fn proof(fixture: &Fixture, iterations: u64) {
        let Some(home) = std::env::var_os("CRITERION_HOME") else {
            return;
        };
        let destination = std::path::PathBuf::from(home);
        std::fs::create_dir_all(&destination).unwrap();
        let report = json!({"schema":"calc-flow.rolling-state-proof.v1", "case":fixture.case.name(),
            "scope":"whole-runner-plan-startup-to-completion", "input_rows":fixture.case.rows(),
            "output_rows":fixture.case.rows(), "groups":fixture.case.groups(), "batch_rows":BATCH_ROWS,
            "event_step_micros":fixture.case.step(),
            "window_micros":if matches!(fixture.case, Case::RowsSum) {None} else {Some(WINDOW)},
            "rows_window":match fixture.case {Case::DurationSum => None, Case::MixedExtrema => Some(37), Case::RowsSum => Some(20)},
            "output_columns":fixture.case.names(), "tokio_workers":4,
            "checkpointing":false, "validated_all_columns":true, "validated_input_immutable":true,
            "result_sha256":fixture.expected_sha256, "validated_iterations":iterations});
        std::fs::write(
            destination.join("rolling-state-proof.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }

    pub(super) fn register(c: &mut Criterion) {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        let mut group = c.benchmark_group("rolling_state_history");
        group
            .sample_size(10)
            .warm_up_time(Duration::from_millis(10))
            .measurement_time(Duration::from_millis(20))
            .sampling_mode(SamplingMode::Flat);
        for case in [Case::DurationSum, Case::MixedExtrema, Case::RowsSum] {
            group.bench_function(case.name(), |b| {
                let fixture = Fixture::new(case);
                let warm = runtime.block_on(run(&fixture));
                validate(&fixture, &warm);
                let mut validated = 1;
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..iterations {
                        let start = Instant::now();
                        let result = runtime.block_on(run(&fixture));
                        elapsed += start.elapsed();
                        validate(&fixture, &result);
                        validated += 1;
                    }
                    proof(&fixture, validated);
                    elapsed
                });
            });
        }
        group.finish();
    }
}

fn rolling_state_history(c: &mut Criterion) {
    rolling_history::register(c);
}
