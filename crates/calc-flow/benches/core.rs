use std::{any::Any, collections::BTreeMap, hint::black_box, sync::Arc};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, DataFusionConfig, DataFusionRuntime,
    ExecutionOptions, ExpressionOperator, ExternalPayload, JsonMap, Operator, OperatorContext,
    PipelineBuilder, Port, Result, UdfReference, UdfRegistry, canonical_json,
};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
use serde_json::json;

fn expression_operator() -> ExpressionOperator {
    ExpressionOperator::new("calculate", "total = a + b", Vec::new(), None, Vec::new()).unwrap()
}

fn expression_plan() -> calc_flow::ExecutionPlan {
    PipelineBuilder::new("benchmark")
        .unwrap()
        .add_node("calculate", Box::new(expression_operator()))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
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
                    .compile(&UdfRegistry::new().snapshot())
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

#[async_trait]
impl Operator for PassthroughOperator {
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

    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &OperatorContext<'_>,
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

fn external_passthrough_plan() -> calc_flow::ExecutionPlan {
    PipelineBuilder::new("external passthrough benchmark")
        .unwrap()
        .add_node("passthrough", Box::new(PassthroughOperator::new()))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
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

criterion_group!(
    core_benchmarks,
    compile_expression,
    create_datafusion_runtime,
    execute_expression,
    execute_external_passthrough,
    encode_canonical_json
);
criterion_main!(core_benchmarks);
