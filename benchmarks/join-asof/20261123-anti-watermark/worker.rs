use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use calc_flow::arrow::{
    array::{Int64Array, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, EventTime, IngressProgress,
    IngressProgressSnapshot, IngressState, JoinStateLimits, JoinTimeBounds, JsonMap,
    OperatorMetadata, StreamJobContext, StreamJoinOperator, StreamJoinSpec, StreamJoinType,
    StreamOperator, StreamOperatorContext,
};
use serde_json::{Value, json};

mod support;

fn table(rows: &[(Option<i64>, i64, u64)]) -> RecordBatch {
    let fields = [
        Field::new("key", DataType::Int64, true),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
    ];
    RecordBatch::try_new(
        Arc::new(Schema::new(fields.to_vec())),
        vec![
            Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.0))),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(rows.iter().map(|r| r.1))
                    .with_timezone("UTC"),
            ),
            Arc::new(UInt64Array::from_iter_values(rows.iter().map(|r| r.2))),
        ],
    )
    .unwrap()
}

fn operator(schema: &SchemaRef) -> StreamJoinOperator {
    let bounds = JoinTimeBounds::new(Duration::ZERO, Duration::from_micros(2)).unwrap();
    let limits = JoinStateLimits::new(512, 8 << 20, 32).unwrap();
    let spec = StreamJoinSpec::new(
        StreamJoinType::LeftAnti,
        ["key"],
        ["key"],
        "time",
        "time",
        bounds,
        limits,
    )
    .unwrap();
    StreamJoinOperator::new("match", schema.clone(), schema.clone(), spec).unwrap()
}

fn context(job: &StreamJobContext, right_watermark: Option<i64>) -> StreamOperatorContext<'_> {
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, None),
        ),
        (
            "right".into(),
            IngressProgress::new(
                IngressState::Active,
                right_watermark.map(EventTime::from_micros),
            ),
        ),
    ]));
    StreamOperatorContext::with_ingress_progress(job, "match", None, progress)
        .with_benchmark_output_budget(EdgeBudget::new(32, 4096).unwrap())
}

fn verify(out: &mut EdgeCollector, expected: &RecordBatch, sequence: u64) -> Vec<usize> {
    let batches = out.drain("output");
    let mut offset = 0;
    let mut sizes = Vec::with_capacity(batches.len());
    for (index, message) in batches.iter().enumerate() {
        let batch = message.as_data().unwrap();
        assert_eq!(
            batch.metadata().sequence(),
            sequence + u64::try_from(index).unwrap()
        );
        sizes.push(batch.num_rows());
        for record in batch.table_payload().unwrap().batches() {
            assert_eq!(record, &expected.slice(offset, record.num_rows()));
            offset += record.num_rows();
        }
    }
    assert_eq!(offset, expected.num_rows());
    sizes
}

async fn observe(measure: bool) -> Value {
    let mut rows = vec![(Some(7), 9, 0)];
    rows.extend((1..=257).map(|id| (Some(7), 10, id)));
    rows.extend([(Some(8), 10, 258), (Some(7), 11, 259), (None, 11, 260)]);
    let left = table(&rows);
    let right = table(&[(Some(8), 10, 0)]);
    let mut op = operator(&left.schema());
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    let job = StreamJobContext::new(
        1,
        "anti-release",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let before = context(&job, None);
    let envelope = |record| Batch::table(vec![record], BatchMetadata::default()).unwrap();
    op.process_data("right", envelope(right), &before, &mut out)
        .await
        .unwrap();
    for start in (0..left.num_rows()).step_by(32) {
        let part = left.slice(start, 32.min(left.num_rows() - start));
        op.process_data("left", envelope(part), &before, &mut out)
            .await
            .unwrap();
    }
    assert_eq!(verify(&mut out, &left.slice(260, 1), 0), [1]);
    op.on_ingress_progress_with_output("right", &context(&job, Some(12)), &mut out)
        .await
        .unwrap();
    assert_eq!(verify(&mut out, &left.slice(0, 1), 1), [1]);

    let equality_ack = table(&[(None, 11, 261)]);
    op.process_data(
        "left",
        envelope(equality_ack.clone()),
        &context(&job, Some(12)),
        &mut out,
    )
    .await
    .unwrap();
    assert_eq!(verify(&mut out, &equality_ack, 2), [1]);

    let cutoff = context(&job, Some(13));
    let started = measure.then(Instant::now);
    op.on_ingress_progress_with_output("right", &cutoff, &mut out)
        .await
        .unwrap();
    let elapsed = started.map(|start| start.elapsed().as_nanos());

    let chunks = verify(&mut out, &left.slice(1, 257), 3);
    assert_eq!(chunks, [32, 32, 32, 32, 32, 32, 32, 32, 1]);
    op.on_ingress_progress_with_output("right", &context(&job, Some(14)), &mut out)
        .await
        .unwrap();
    assert_eq!(verify(&mut out, &left.slice(259, 1), 12), [1]);
    op.on_end(&context(&job, Some(14)), &mut out).await.unwrap();
    assert!(out.drain("output").is_empty());
    json!({"setup":[260,0,261], "chunks":chunks, "tail":[259], "rows":257, "elapsed_ns":elapsed})
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> support::ExampleResult {
    let arguments = support::arguments()?;
    let measure = match arguments.as_slice() {
        [] => true,
        [option] if option == "--verify-only" => false,
        _ => return Err("expected no arguments or --verify-only".into()),
    };
    println!("{}", observe(measure).await);
    Ok(())
}
