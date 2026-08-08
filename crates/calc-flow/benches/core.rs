use std::{
    any::Any,
    collections::BTreeMap,
    hint::black_box,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use calc_flow::{
    AggregateFunction, Batch, BatchKind, BatchMetadata, BatchOperator, BatchOperatorContext,
    CalcFlowError, CancellationToken, DataFusionConfig, DataFusionRuntime, EdgeBudget,
    EdgeCollector, Epoch, ExecutionOptions, ExpressionOperator, ExternalPayload, JsonMap,
    LocalStateBackend, OperatorMetadata, PipelineBuilder, Port, Result, StateBackend, StateHandle,
    StateLineageKey, StreamCollector, StreamJobContext, StreamMessage, StreamOperator,
    StreamOperatorContext, UdfReference, UdfRegistry, WindowAggregateOperator, WindowSpec,
    canonical_json, edge_channel,
};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::arrow::{
    array::{ArrayRef, Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

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
    c.bench_function("execute/external_plan_table_requirement", |b| {
        b.iter(|| black_box(plan.requires_datafusion()));
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

const STATE_PIPELINE_FINGERPRINT: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn sha256(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes.as_ref()))
}

fn benchmark_state_handle(
    key: &StateLineageKey,
    operator_id: &str,
    epoch: Epoch,
    segment_id: &str,
    bytes: &[u8],
) -> StateHandle {
    let lineage_hash = sha256(format!(
        "{}\0{}",
        key.pipeline_name(),
        key.pipeline_fingerprint()
    ));
    StateHandle::new(
        operator_id,
        epoch,
        segment_id,
        &format!(
            "committed/{lineage_hash}/{}/{}-{}.arrow",
            sha256(operator_id),
            epoch.as_u64(),
            sha256(segment_id)
        ),
        u64::try_from(bytes.len()).unwrap(),
        &sha256(bytes),
    )
    .unwrap()
}

fn state_backend_io(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let directory = TempDir::new().unwrap();
    let key = StateLineageKey::new("benchmark", STATE_PIPELINE_FINGERPRINT).unwrap();
    let backend = runtime
        .block_on(LocalStateBackend::new(directory.path()))
        .unwrap();
    let lineage = runtime.block_on(backend.open_lineage(&key)).unwrap();
    let incremental_bytes = Arc::new(vec![0x5a; 64 * 1_024]);
    let next_epoch = AtomicU64::new(100);
    c.bench_function("state/incremental_write_64k", |b| {
        b.to_async(&runtime).iter(|| {
            let lineage = lineage.as_ref();
            let epoch = Epoch::new(next_epoch.fetch_add(1, Ordering::Relaxed)).unwrap();
            let segment_id = format!("delta-{:020}", epoch.as_u64());
            let handle =
                benchmark_state_handle(&key, "incremental", epoch, &segment_id, &incremental_bytes);
            let bytes = Arc::clone(&incremental_bytes);
            async move {
                lineage
                    .stage_segment(&handle, bytes.as_slice())
                    .await
                    .unwrap();
                lineage.validate_segment(&handle).await.unwrap();
                lineage.publish_segment(&handle).await.unwrap();
                black_box(handle)
            }
        });
    });

    let restore_bytes = vec![0xa5; 4 * 1_024 * 1_024];
    let restore_handle = benchmark_state_handle(
        &key,
        "restore",
        Epoch::INITIAL,
        "base-restore",
        &restore_bytes,
    );
    runtime.block_on(async {
        lineage
            .stage_segment(&restore_handle, &restore_bytes)
            .await
            .unwrap();
        lineage.validate_segment(&restore_handle).await.unwrap();
        lineage.publish_segment(&restore_handle).await.unwrap();
    });
    c.bench_function("state/full_restore_4m", |b| {
        b.to_async(&runtime)
            .iter(|| async { black_box(lineage.load_segment(&restore_handle).await.unwrap()) });
    });

    let compaction_handles = (0..8)
        .map(|ordinal| {
            let bytes = vec![u8::try_from(ordinal).unwrap(); 64 * 1_024];
            let segment_id = format!("delta-{ordinal:04}");
            let handle = benchmark_state_handle(
                &key,
                "compact",
                Epoch::new(2).unwrap(),
                &segment_id,
                &bytes,
            );
            runtime.block_on(async {
                lineage.stage_segment(&handle, &bytes).await.unwrap();
                lineage.validate_segment(&handle).await.unwrap();
                lineage.publish_segment(&handle).await.unwrap();
            });
            handle
        })
        .collect::<Vec<_>>();
    c.bench_function("state/compaction_read_8x64k", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut restored = Vec::new();
            for handle in &compaction_handles {
                restored.extend(lineage.load_segment(handle).await.unwrap());
            }
            black_box(restored)
        });
    });
}

fn window_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("group", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn window_input(rows: usize) -> Batch {
    let row_count = rows;
    let rows = i64::try_from(row_count).unwrap();
    let record = RecordBatch::try_new(
        window_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![0; row_count])) as ArrayRef,
            Arc::new(Int64Array::from_iter_values(0..rows)),
            Arc::new(Int64Array::from_iter_values(1..=rows)),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn window_operator(hopping: bool) -> WindowAggregateOperator {
    let spec = if hopping {
        WindowSpec::hopping(
            "event_time",
            Duration::from_micros(100),
            Duration::from_micros(10),
        )
        .unwrap()
    } else {
        WindowSpec::tumbling("event_time", Duration::from_micros(100)).unwrap()
    }
    .group_by(["group"])
    .unwrap()
    .aggregate(AggregateFunction::Sum, "value", "sum_value")
    .unwrap();
    WindowAggregateOperator::new("window", window_schema(), spec).unwrap()
}

fn window_execution_and_restore(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let job = StreamJobContext::new(
        1,
        STATE_PIPELINE_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let input = window_input(1_024);
    for (name, hopping) in [("tumbling", false), ("hopping", true)] {
        let operator = window_operator(hopping);
        let collector = EdgeCollector::new(operator.output_ports().to_vec());
        let state = tokio::sync::Mutex::new((operator, collector));
        c.bench_function(&format!("window/{name}_1024_rows"), |b| {
            b.to_async(&runtime).iter(|| async {
                let context = StreamOperatorContext::new(&job, "window", None);
                let mut state = state.lock().await;
                let (operator, collector) = &mut *state;
                operator.reset().unwrap();
                operator
                    .process_data("input", input.clone(), &context, collector)
                    .await
                    .unwrap();
                black_box(());
            });
        });
    }

    let mut source = window_operator(false);
    let mut collector = EdgeCollector::new(source.output_ports().to_vec());
    runtime.block_on(async {
        let context = StreamOperatorContext::new(&job, "window", None);
        source
            .process_data("input", input, &context, &mut collector)
            .await
            .unwrap();
    });
    let snapshot = source.checkpoint(Epoch::INITIAL).unwrap();
    let mut target = window_operator(false);
    c.bench_function("state/window_arrow_restore_1024_keys", |b| {
        b.iter(|| {
            target.restore(black_box(&snapshot)).unwrap();
            black_box(());
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
    state_backend_io,
    window_execution_and_restore
);
criterion_main!(core_benchmarks);
