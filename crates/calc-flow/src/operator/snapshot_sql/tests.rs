use std::{cell::Cell, sync::Arc};

use super::*;
use crate::{
    BatchKind, BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, StreamJobContext,
};
use datafusion::arrow::{
    array::{Array, Int64Array},
    record_batch::RecordBatch,
};

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

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "snapshot-test",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

#[tokio::test]
async fn test_byte_limit_holds_old_candidate_and_cache_and_refunds_failed_work() {
    let mut operator = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "snapshot", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let shared = batch(&[1, 2]);
    operator
        .process_data("p", shared.clone(), &context, &mut collector)
        .await
        .unwrap();
    let pool = operator.pool.as_ref().unwrap().clone();
    let owned = pool.reserved();
    operator
        .set_state_budget(StateBudget::new(10, owned as u64).unwrap())
        .unwrap();
    // The exact same owner is not charged twice.
    operator
        .process_data("p", shared, &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(pool.reserved(), owned);
    assert!(
        operator
            .process_data("p", batch(&[3]), &context, &mut collector)
            .await
            .is_err()
    );
    assert_eq!(pool.reserved(), owned);
    assert!(operator.checkpoint(Epoch::INITIAL).is_err());
    assert_eq!(pool.reserved(), owned);
    operator
        .set_state_budget(StateBudget::new(10, 1 << 20).unwrap())
        .unwrap();
    let saved = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(pool.reserved() > owned);
    operator.reset().unwrap();
    assert!(
        pool.reserved() > 0,
        "carried segments keep their byte lease"
    );
    drop(saved);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_output_collection_limit_rejects_without_commit_or_partial_emit() {
    let mut operator = operator();
    let job = job();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let context = StreamOperatorContext::new(&job, "snapshot", None)
        .with_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    assert!(
        operator
            .process_data("p", batch(&[1, 2]), &context, &mut collector)
            .await
            .is_err()
    );
    assert!(collector.drain("output").is_empty());
    assert!(operator.relations[0].is_none());
    assert_eq!(operator.pool.as_ref().unwrap().reserved(), 0);
    let context = StreamOperatorContext::new(&job, "snapshot", None)
        .with_output_budget(EdgeBudget::new(10, 1).unwrap());
    assert!(
        operator
            .process_data("p", batch(&[1]), &context, &mut collector)
            .await
            .is_err()
    );
    assert!(collector.drain("output").is_empty());
}

#[tokio::test]
async fn test_cancelled_prepared_restore_and_capture_are_atomic_and_refund() {
    let mut original = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "snapshot", None);
    let mut collector = EdgeCollector::new(original.output_ports().to_vec());
    original
        .process_data("p", batch(&[1, 2]), &context, &mut collector)
        .await
        .unwrap();
    let saved = original.checkpoint(Epoch::INITIAL).unwrap();
    let mut target = operator();
    target
        .process_data("p", batch(&[9]), &context, &mut collector)
        .await
        .unwrap();
    let pool = target.pool.as_ref().unwrap().clone();
    let before = pool.reserved();
    for stop in 1..16 {
        let count = Cell::new(0);
        let check = || {
            count.set(count.get() + 1);
            if count.get() == stop {
                Err(CalcFlowError::Cancelled {
                    run_id: "test".into(),
                })
            } else {
                Ok(())
            }
        };
        drop(target.prepare_restore(&saved, &check));
        assert_eq!(pool.reserved(), before);
        assert_eq!(target.relations[0].as_ref().unwrap().batch.num_rows(), 1);
        count.set(0);
        drop(target.prepare_checkpoint_work(&check));
        assert_eq!(pool.reserved(), before);
    }
    target.reset().unwrap();
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_sliced_input_pays_for_reachable_backing() {
    let mut operator = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "snapshot", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let large: Arc<dyn Array> = Arc::new(Int64Array::from(vec![7; 1 << 16]));
    let sliced = Batch::table(
        vec![RecordBatch::try_from_iter(vec![("value", large.slice(0, 1))]).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    operator
        .set_state_budget(StateBudget::new(10, 64 << 10).unwrap())
        .unwrap();
    assert!(
        operator
            .process_data("p", sliced, &context, &mut collector)
            .await
            .is_err()
    );
    assert_eq!(operator.pool.as_ref().unwrap().reserved(), 0);
    operator
        .process_data("p", batch(&[7]), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(collector.drain("output").len(), 1);
}

#[tokio::test]
async fn test_aborted_pending_emit_refunds_candidate_without_commit() {
    struct Slow(Arc<tokio::sync::Notify>);
    #[async_trait]
    impl StreamCollector for Slow {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            self.0.notify_one();
            std::future::pending().await
        }
    }
    let mut operator = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "snapshot", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("p", batch(&[1]), &context, &mut collector)
        .await
        .unwrap();
    collector.drain("output");
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    let pool = operator.pool.as_ref().unwrap().clone();
    let owned = pool.reserved();
    let entered = Arc::new(tokio::sync::Notify::new());
    let mut sink = Slow(entered.clone());
    let mut future = Box::pin(operator.process_data("p", batch(&[2, 3]), &context, &mut sink));
    tokio::select! {
        () = entered.notified() => {},
        result = &mut future => panic!("emit must remain pending: {result:?}"),
    }
    assert!(pool.reserved() > owned);
    job.cancellation().cancel();
    drop(future);
    assert_eq!(pool.reserved(), owned);
    assert_eq!(
        operator.checkpoint(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
}

#[tokio::test]
async fn test_same_node_restore_shares_encoded_segment_leases_under_tight_budget() {
    let mut operator = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "snapshot", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("p", batch(&[1]), &context, &mut collector)
        .await
        .unwrap();
    let pool = operator.pool.as_ref().unwrap().clone();
    let raw = pool.reserved();
    let saved = operator.checkpoint(Epoch::INITIAL).unwrap();
    // Allow a decoded candidate, while keeping the encoded allocations shared.
    operator
        .set_state_budget(StateBudget::new(10, (pool.reserved() + raw + 512) as u64).unwrap())
        .unwrap();
    operator.restore(&saved).unwrap();
    let restored = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(Arc::ptr_eq(
        &saved.segments["relation-0"].bytes_arc(),
        &restored.segments["relation-0"].bytes_arc()
    ));
    assert!(Arc::ptr_eq(
        &saved.segments["metadata-0"].bytes_arc(),
        &restored.segments["metadata-0"].bytes_arc()
    ));
    operator.reset().unwrap();
    drop(saved);
    drop(restored);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_precompile_budget_preserves_configured_runtime_identity() {
    let compile = |preset_budget: bool| {
        let mut snapshot = operator();
        if preset_budget {
            snapshot
                .set_state_budget(StateBudget::new(10, 1 << 20).unwrap())
                .unwrap();
        }
        let plan = crate::PipelineBuilder::new("snapshot-config")
            .unwrap()
            .with_datafusion_config(DataFusionConfig {
                batch_size: 17,
                ..DataFusionConfig::default()
            })
            .add_node("snapshot", snapshot)
            .unwrap()
            .compile_stream(
                &UdfRegistrySnapshot::default(),
                &crate::StreamRequirements::default(),
            )
            .unwrap();
        let fingerprint = plan.fingerprint().to_owned();
        let mut parts = plan.into_runtime_parts(EdgeBudget::default()).unwrap();
        (fingerprint, parts.nodes.remove(0).operator)
    };
    let (before_fingerprint, mut before) = compile(true);
    let saved = before.checkpoint(Epoch::INITIAL).unwrap();
    let control: serde_json::Value =
        serde_json::from_slice(saved.segments["control"].bytes()).unwrap();
    assert_eq!(control["identity"]["runtime_config"]["batch_size"], 17);
    let (after_fingerprint, mut after) = compile(false);
    assert_eq!(before_fingerprint, after_fingerprint);
    let result = after.restore(&saved);
    assert!(
        result.is_ok(),
        "same graph and engine configuration must restore: {result:?}"
    );
}
