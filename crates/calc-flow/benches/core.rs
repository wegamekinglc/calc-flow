use std::{collections::BTreeMap, hint::black_box, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, ExpressionOperator, PipelineBuilder, UdfRegistry,
    canonical_json,
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
    execute_expression,
    encode_canonical_json
);
criterion_main!(core_benchmarks);
