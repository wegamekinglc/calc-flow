//! Native two-input Union forwarding measured without changing the core bench target.

use std::{hint::black_box, sync::Arc};

use calc_flow::{
    Batch, BatchKind, BatchMetadata, CancellationToken, EdgeCollector, JsonMap, OperatorMetadata,
    Port, StreamJobContext, StreamOperator, StreamOperatorContext, UnionOperator,
};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

fn input() -> Batch {
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from_iter_values(0..1_024)) as _,
    )])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn stream_union_two_inputs(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let operator = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap();
    let collector = EdgeCollector::new(operator.output_ports().to_vec());
    let state = tokio::sync::Mutex::new((operator, collector));
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(2, "union-bench", JsonMap::new(), None, cancellation);
    let batch = input();
    c.bench_function("stream/union_two_1024_row_inputs", |b| {
        b.to_async(&runtime).iter(|| async {
            let context = StreamOperatorContext::new(&job, "merge", None);
            let mut state = state.lock().await;
            let (operator, collector) = &mut *state;
            operator
                .process_data("left", batch.clone(), &context, collector)
                .await
                .unwrap();
            operator
                .process_data("right", batch.clone(), &context, collector)
                .await
                .unwrap();
            let output = collector.drain("output");
            assert_eq!(output.len(), 2);
            black_box(output)
        });
    });
}

criterion_group!(stream_union_benchmarks, stream_union_two_inputs);
criterion_main!(stream_union_benchmarks);
