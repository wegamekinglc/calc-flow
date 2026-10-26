use std::sync::Arc;

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, EdgeCollector, Epoch, OperatorMetadata, Result,
    SnapshotSqlOperator, SqlOperator, StateBudget, StreamCollector, StreamOperator,
    StreamOperatorContext,
};
use datafusion::arrow::record_batch::RecordBatch;

#[path = "snapshot_sql/support.rs"]
mod support;
use support::*;

#[test]
fn test_snapshot_identity_is_distinct_from_append_sql() {
    let operator =
        SnapshotSqlOperator::new("valuation", "SELECT * FROM p", vec!["p".into()], vec![]).unwrap();
    let old = SqlOperator::new("valuation", "SELECT * FROM p", vec!["p".into()], vec![]).unwrap();
    assert_eq!(
        operator.configuration()["snapshot_policy"],
        "replace_snapshot_v1"
    );
    assert!(!old.configuration().contains_key("snapshot_policy"));
    for aliases in [
        vec!["a".into(), "A".into()],
        vec!["bad.name".into()],
        vec!["1x".into()],
        vec![],
    ] {
        assert!(SnapshotSqlOperator::new("x", "SELECT * FROM p", aliases, vec![]).is_err());
    }
}

#[tokio::test]
async fn test_both_input_orders_and_each_replacement_match_batch_sql() {
    for reverse in [false, true] {
        let mut operator = operator(QUERY).await;
        let job = job();
        let context = StreamOperatorContext::new(&job, "valuation", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let p = batch(&[&[1], &[2]]);
        let q = batch(&[&[1, 1], &[2]]);
        let order = if reverse {
            [("q", q.clone()), ("p", p.clone())]
        } else {
            [("p", p.clone()), ("q", q.clone())]
        };
        operator
            .process_data(order[0].0, order[0].1.clone(), &context, &mut collector)
            .await
            .unwrap();
        assert!(collector.drain("output").is_empty());
        operator
            .process_data(order[1].0, order[1].1.clone(), &context, &mut collector)
            .await
            .unwrap();
        assert_output(&mut collector, &oracle(QUERY, p.clone(), q).await);
        let q = batch(&[&[2]]);
        operator
            .process_data("q", q.clone(), &context, &mut collector)
            .await
            .unwrap();
        assert_output(&mut collector, &oracle(QUERY, p.clone(), q).await);
        let empty = batch(&[&[]]);
        operator
            .process_data("q", empty.clone(), &context, &mut collector)
            .await
            .unwrap();
        assert_output(&mut collector, &oracle(QUERY, p, empty.clone()).await);
        operator
            .process_data("p", empty.clone(), &context, &mut collector)
            .await
            .unwrap();
        assert_output(&mut collector, &oracle(QUERY, empty.clone(), empty).await);
    }
}

#[tokio::test]
async fn test_first_empty_initializes_but_missing_input_does_not() {
    let query = "SELECT COUNT(*) AS total FROM p CROSS JOIN q";
    let mut operator = operator(query).await;
    let job = job();
    let context = StreamOperatorContext::new(&job, "valuation", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("p", batch(&[&[]]), &context, &mut collector)
        .await
        .unwrap();
    assert!(collector.drain("output").is_empty());
    operator
        .process_data("q", batch(&[&[1]]), &context, &mut collector)
        .await
        .unwrap();
    assert_output(
        &mut collector,
        &oracle(query, batch(&[&[]]), batch(&[&[1]])).await,
    );
    operator.on_end(&context, &mut collector).await.unwrap();
    assert!(collector.drain("output").is_empty());
}

#[tokio::test]
async fn test_max_rows_counts_latest_relations_and_rejection_keeps_old_state() {
    let mut operator = operator(QUERY).await;
    operator
        .set_state_budget(StateBudget::new(3, 1 << 20).unwrap())
        .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "valuation", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("p", batch(&[&[1, 2]]), &context, &mut collector)
        .await
        .unwrap();
    operator
        .process_data("q", batch(&[&[1]]), &context, &mut collector)
        .await
        .unwrap();
    collector.drain("output");
    let before = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(
        operator
            .process_data("q", batch(&[&[1, 2]]), &context, &mut collector)
            .await
            .is_err()
    );
    assert_eq!(
        operator
            .checkpoint(Epoch::new(2).unwrap())
            .unwrap()
            .segments,
        before.segments
    );
    operator
        .process_data("q", batch(&[&[2]]), &context, &mut collector)
        .await
        .unwrap();
    assert_output(
        &mut collector,
        &oracle(QUERY, batch(&[&[1, 2]]), batch(&[&[2]])).await,
    );
}

struct Reject;
#[async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Err(CalcFlowError::Internal {
            message: "sink refused".into(),
        })
    }
}

#[tokio::test]
async fn test_emit_failure_and_cancel_do_not_commit() {
    let mut operator = operator(QUERY).await;
    let job = job();
    let context = StreamOperatorContext::new(&job, "valuation", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("p", batch(&[&[1]]), &context, &mut collector)
        .await
        .unwrap();
    operator
        .process_data("q", batch(&[&[1]]), &context, &mut collector)
        .await
        .unwrap();
    collector.drain("output");
    let before = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(
        operator
            .process_data("q", batch(&[&[2]]), &context, &mut Reject)
            .await
            .is_err()
    );
    assert_eq!(
        operator
            .checkpoint(Epoch::new(2).unwrap())
            .unwrap()
            .segments,
        before.segments
    );
    job.cancellation().cancel();
    assert!(
        operator
            .process_data("q", batch(&[&[2]]), &context, &mut collector)
            .await
            .is_err()
    );
    assert_eq!(
        operator
            .checkpoint(Epoch::new(3).unwrap())
            .unwrap()
            .segments,
        before.segments
    );
}

#[tokio::test]
async fn test_wrong_schema_empty_and_unknown_ingress_fail_without_initialization() {
    let mut operator = operator(QUERY).await;
    let job = job();
    let context = StreamOperatorContext::new(&job, "valuation", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let wrong = Batch::table(
        vec![RecordBatch::new_empty(Arc::new(
            datafusion::arrow::datatypes::Schema::empty(),
        ))],
        BatchMetadata::default(),
    )
    .unwrap();
    assert!(
        operator
            .process_data("p", wrong, &context, &mut collector)
            .await
            .is_err()
    );
    assert!(
        operator
            .process_data("unknown", batch(&[&[]]), &context, &mut collector)
            .await
            .is_err()
    );
    operator
        .process_data("q", batch(&[&[1]]), &context, &mut collector)
        .await
        .unwrap();
    assert!(collector.drain("output").is_empty());
}
