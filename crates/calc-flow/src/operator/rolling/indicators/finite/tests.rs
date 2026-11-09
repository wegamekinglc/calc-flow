use std::{collections::BTreeMap, sync::Arc, task::Poll};

use datafusion::arrow::{
    array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

use crate::{
    Batch, BatchMetadata, BatchOperator, BatchOperatorContext, CalcFlowError, CancellationToken,
    JsonMap, RollingOperator, RunContext, StateBudget, array::RollingScope,
};

#[tokio::test]
async fn test_finite_indicator_process_waits_for_the_owned_worker_pool() {
    let (mut operator, input) = case();
    let run = RunContext::new(JsonMap::new(), None, CancellationToken::new()).unwrap();
    let workers = operator.native_checkpoint_workers();
    let mut release = Vec::new();
    let mut handles = Vec::new();
    for _ in 0..2 {
        let (unblock, blocked) = std::sync::mpsc::channel::<()>();
        let (entered, started) = tokio::sync::oneshot::channel();
        let workers = Arc::clone(&workers);
        let scope = RollingScope::Batch(run.clone());
        handles.push(tokio::spawn(async move {
            workers
                .run_rolling_scoped(scope, "blocker", move |check| {
                    entered.send(()).unwrap();
                    let _ = blocked.recv();
                    check()
                })
                .await
        }));
        release.push(unblock);
        tokio::time::timeout(std::time::Duration::from_secs(5), started)
            .await
            .unwrap()
            .unwrap();
    }
    let inputs = BTreeMap::from([("input".into(), input)]);
    let context = BatchOperatorContext { run: &run };
    let mut process = Box::pin(operator.process(&inputs, &context));
    let pending =
        std::future::poll_fn(|cx| Poll::Ready(process.as_mut().poll(cx).is_pending())).await;
    drop(process);
    drop(release);
    for handle in handles {
        handle.await.unwrap().unwrap();
    }
    assert!(
        pending,
        "finite indicator processing bypassed its worker pool"
    );
}

fn case() -> (RollingOperator, Batch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("price", DataType::Float64, false),
        Field::new("entity", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
    ]));
    let spec = serde_json::from_value(serde_json::json!({
        "configuration_version":1, "state_layout_version":5,
        "partition_by":["entity"], "event_time":"ts", "sequence_by":["seq"],
        "outputs":[{"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":1}],
        "allowed_lateness_micros":0, "late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    }))
    .unwrap();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from(vec![3, 1, 2]).with_timezone("UTC")),
        Arc::new(Float64Array::from(vec![1.0, 1.0, 2.0])),
        Arc::new(StringArray::from(vec!["a"; 3])),
        Arc::new(UInt64Array::from(vec![3, 1, 2])),
    ];
    let batch = Batch::table(
        vec![RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    (
        RollingOperator::new("features", schema, spec).unwrap(),
        batch,
    )
}

#[tokio::test]
async fn test_finite_worker_sorts_immutable_inputs_and_starts_fresh_on_each_call() {
    let (mut operator, input) = case();
    let original = input.table_payload().unwrap().batches()[0].clone();
    let inputs = BTreeMap::from([("input".into(), input)]);
    let run = RunContext::new(JsonMap::new(), None, CancellationToken::new()).unwrap();
    let context = BatchOperatorContext { run: &run };
    let first = operator.process(&inputs, &context).await.unwrap();
    let second = operator.process(&inputs, &context).await.unwrap();
    let record = &first["output"].table_payload().unwrap().batches()[0];
    let rsi = record
        .column(4)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(
        rsi.iter().collect::<Vec<_>>(),
        [None, Some(100.0), Some(0.0)]
    );
    assert_eq!(
        record,
        &second["output"].table_payload().unwrap().batches()[0]
    );
    assert_eq!(
        &original,
        &inputs["input"].table_payload().unwrap().batches()[0]
    );
    assert!(operator.state.buffer.is_empty());
    assert!(operator.state.histories.by_entity.is_empty());
}

#[tokio::test]
async fn test_finite_worker_preserves_empty_schema_and_metadata() {
    let (mut operator, input) = case();
    let record = input.table_payload().unwrap().batches()[0].slice(0, 0);
    let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let run = RunContext::new(JsonMap::new(), None, CancellationToken::new()).unwrap();
    let result = operator
        .process(
            &BTreeMap::from([("input".into(), input)]),
            &BatchOperatorContext { run: &run },
        )
        .await
        .unwrap();
    let record = &result["output"].table_payload().unwrap().batches()[0];
    assert_eq!(record.num_rows(), 0);
    assert_eq!(
        record.schema(),
        operator.output_ports[0].schema().unwrap().clone()
    );
    assert_eq!(result["output"].metadata().source(), "features");
}

#[tokio::test]
async fn test_finite_materialization_budget_is_checked_inside_the_actual_worker() {
    let (mut operator, input) = text_case();
    operator
        .set_state_budget(StateBudget::new(8, 64 << 10).unwrap())
        .unwrap();
    let run = RunContext::new(JsonMap::new(), None, CancellationToken::new()).unwrap();
    let work = operator
        .finite_work(
            input,
            &BatchOperatorContext { run: &run },
            "physical-features",
        )
        .unwrap();
    let (result, measured) = operator
        .native_checkpoint_workers()
        .run_rolling_scoped(
            RollingScope::Batch(run),
            "physical-features",
            move |check| {
                let mut result = None;
                let measured = allocation_counter::measure(|| {
                    result = Some(work.run(check).map(|_| ()));
                });
                Ok((result.unwrap(), measured))
            },
        )
        .await
        .unwrap();
    assert!(
        matches!(result, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "physical-features")
    );
    assert!(
        measured.bytes_total < 64 << 10,
        "worker copied the rejected text: {measured:?}"
    );
}

fn text_case() -> (RollingOperator, Batch) {
    let (prototype, input) = case();
    let record = &input.table_payload().unwrap().batches()[0];
    let mut fields = record.schema().fields().to_vec();
    fields.push(Arc::new(Field::new("context", DataType::Utf8, false)));
    let schema = Arc::new(Schema::new(fields));
    let mut columns = record.columns().to_vec();
    columns.push(Arc::new(StringArray::from(vec!["x".repeat(4 << 20); 3])));
    let input = Batch::table(
        vec![RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    (
        RollingOperator::new("features", schema, prototype.spec.clone()).unwrap(),
        input,
    )
}
