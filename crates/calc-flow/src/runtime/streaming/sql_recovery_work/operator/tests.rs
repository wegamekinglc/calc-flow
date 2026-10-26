use crate::{
    Batch, BatchKind, BatchMetadata, CancellationToken, EdgeCollector, JsonMap, OperatorMetadata,
    Port, SnapshotSqlOperator, StreamJobContext, StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{Array, Int64Array},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryPool;
use std::sync::Arc;

fn operator() -> SnapshotSqlOperator {
    let schema = batch(&[]).table_payload().unwrap().schema().clone();
    SnapshotSqlOperator::new("snapshot", "SELECT * FROM p", vec!["p".into()], vec![])
        .unwrap()
        .with_ports(
            vec![Port::with_schema_ref("p", BatchKind::Table, true, Some(schema.clone())).unwrap()],
            Port::with_schema_ref("output", BatchKind::Table, true, Some(schema)).unwrap(),
        )
        .unwrap()
}
fn batch(values: &[i64]) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_from_iter(vec![(
                "value",
                Arc::new(Int64Array::from(values.to_vec())) as Arc<dyn Array>,
            )])
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}
fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "snapshot-work",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

#[tokio::test]
async fn test_checkpoint_worker_cancelled_completion_keeps_original_and_refunds_candidate() {
    use crate::{
        pipeline::CompiledStreamOperator,
        runtime::streaming::sql_recovery_work::{
            CheckpointWorkOperator, JobSqlRecoveryOwner, SqlCaptureRequest, SqlRecoveryContext,
            SqlWorkIdentity,
        },
    };
    let mut operator = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "snapshot", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("p", batch(&[1]), &context, &mut collector)
        .await
        .unwrap();
    let pool = operator.state_pool_for_test();
    let before = pool.reserved();
    let mut target = CompiledStreamOperator::SnapshotSql(operator);
    let reservation = CheckpointWorkOperator::reserve_checkpoint(&mut target, 8).unwrap();
    let owner = JobSqlRecoveryOwner::new();
    owner.configure(2).unwrap();
    let ticket = owner
        .submit_capture(SqlCaptureRequest {
            operator: CheckpointWorkOperator::from_compiled(target),
            identity: SqlWorkIdentity {
                node_id: "snapshot".into(),
                node_order: 0,
                task_id: None,
            },
            context: SqlRecoveryContext::from(&job),
            launch_cancel: CancellationToken::new(),
            reservation,
        })
        .result
        .unwrap();
    let completion = ticket.join().await.unwrap();
    assert!(pool.reserved() > before);
    job.cancellation().cancel();
    let mut target = CompiledStreamOperator::CheckpointLoan;
    assert!(completion.install_checkpoint(&mut target).is_err());
    assert_eq!(pool.reserved(), before);
    let CompiledStreamOperator::SnapshotSql(operator) = &target else {
        panic!("lost original operator")
    };
    assert_eq!(operator.input_rows_for_test(), 1);
    drop(target);
    assert!(owner.drain().await.is_empty());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_abandoned_snapshot_worker_is_drained_and_refunds_ownership() {
    use crate::{
        pipeline::CompiledStreamOperator,
        runtime::streaming::sql_recovery_work::{
            CheckpointWorkOperator, JobSqlRecoveryOwner, SqlCaptureRequest, SqlRecoveryContext,
            SqlWorkIdentity,
        },
    };
    for join_first in [false, true] {
        let mut operator = operator();
        let job = job();
        let context = StreamOperatorContext::new(&job, "snapshot", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("p", batch(&[1]), &context, &mut collector)
            .await
            .unwrap();
        let pool = operator.state_pool_for_test();
        let mut target = CompiledStreamOperator::SnapshotSql(operator);
        let reservation = CheckpointWorkOperator::reserve_checkpoint(&mut target, 8).unwrap();
        let owner = JobSqlRecoveryOwner::new();
        owner.configure(2).unwrap();
        let ticket = owner
            .submit_capture(SqlCaptureRequest {
                operator: CheckpointWorkOperator::from_compiled(target),
                identity: SqlWorkIdentity {
                    node_id: "snapshot".into(),
                    node_order: 0,
                    task_id: None,
                },
                context: SqlRecoveryContext::from(&job),
                launch_cancel: CancellationToken::new(),
                reservation,
            })
            .result
            .unwrap();
        if join_first {
            drop(ticket.join().await.unwrap());
        } else {
            drop(ticket);
        }
        assert!(owner.drain().await.is_empty());
        assert_eq!(owner.live_counts(), (0, 0));
        assert_eq!(pool.reserved(), 0);
    }
}
