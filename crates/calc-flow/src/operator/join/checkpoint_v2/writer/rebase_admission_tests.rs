use super::*;

async fn captured_history(
    context: &StreamOperatorContext<'_>,
    pool: Arc<dyn MemoryPool>,
) -> (StreamJoinOperator, OperatorStateSnapshot) {
    let mut join = operator();
    join.runtime.initialize_benchmark_pool(pool).unwrap();
    add_pending(&mut join, context).await;
    let mut snapshot = join.checkpoint(Epoch::INITIAL).unwrap();
    for epoch in 2..=4 {
        add_pending(&mut join, context).await;
        snapshot = join.checkpoint(Epoch::new(epoch).unwrap()).unwrap();
    }
    assert_eq!(join.v2_writer.dirty_epochs, 4);
    assert!(join.state.deltas.pending.is_empty());
    (join, snapshot)
}

async fn reference() -> (Vec<Admission>, (JsonMap, BTreeMap<String, Vec<u8>>)) {
    let pool = Arc::new(CapturePool::new());
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (mut join, old) = captured_history(&context, pool.clone()).await;
    pool.arm(None);
    join.prepare_checkpoint_async(&context).await.unwrap();
    let admissions = pool.finish();
    assert!(
        admissions.len() >= 12,
        "caller, work, input, submission, output, four scans and indexes"
    );
    let snapshot = join.checkpoint(Epoch::new(5).unwrap()).unwrap();
    let wire = snapshot_wire(&snapshot);
    drop((old, snapshot, join));
    drop(context);
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    drop(enabled);
    assert_eq!(pool.reserved(), 0);
    (admissions, wire)
}

#[tokio::test]
async fn test_rebase_rejects_each_admission_atomically_and_retries_the_same_epoch() {
    let (admissions, expected) = reference().await;
    for reject in 1..=admissions.len() {
        let pool = Arc::new(CapturePool::new());
        let enabled = job();
        let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
        let (mut join, old) = captured_history(&context, pool.clone()).await;
        let before = capture_image(&join);
        let records = capture_records(&join);
        let old_wire = snapshot_wire(&old);
        let reserved = pool.reserved();
        pool.arm(Some(reject));
        let result = join.prepare_checkpoint_async(&context).await;
        let observed = pool.finish();
        assert!(result.is_err(), "admission {reject}");
        join.await_compaction_release(&context).await.unwrap();
        assert_eq!(observed.len(), reject);
        assert_eq!(
            observed.last().unwrap().additional,
            admissions[reject - 1].additional
        );
        assert_eq!(capture_image(&join), before, "admission {reject}");
        assert_eq!(capture_records(&join), records, "admission {reject}");
        assert_eq!(snapshot_wire(&old), old_wire);
        assert!(join.v2_writer.history_trusted);
        assert_eq!(
            pool.reserved(),
            reserved,
            "admission {reject} retained temporary credit"
        );
        join.prepare_checkpoint_async(&context).await.unwrap();
        let snapshot = join.checkpoint(Epoch::new(5).unwrap()).unwrap();
        assert_eq!(snapshot_wire(&snapshot), expected, "retry {reject}");
        drop((snapshot, old, join, records));
        drop(context);
        assert!(enabled.gather_owner().close_and_drain().await.is_empty());
        drop(enabled);
        assert_eq!(pool.reserved(), 0, "admission {reject}");
    }
}
