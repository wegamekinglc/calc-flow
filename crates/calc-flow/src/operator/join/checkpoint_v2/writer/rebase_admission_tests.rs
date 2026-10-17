use super::*;

async fn captured_history(
    context: &StreamOperatorContext<'_>,
    pool: Arc<dyn MemoryPool>,
    with_pending: bool,
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
    assert!(join.state.deltas.is_clean());
    if with_pending {
        add_pending(&mut join, context).await;
        assert_eq!(join.state.deltas.pending_count_for_test(), 3);
    }
    (join, snapshot)
}

async fn reference(
    with_pending: bool,
    service: &crate::runtime::streaming::gather_work::TestService,
) -> (Vec<Admission>, (JsonMap, BTreeMap<String, Vec<u8>>)) {
    let pool = Arc::new(CapturePool::new());
    let enabled = job().with_gather_owner(service.owner("rebase-admission-reference".into()));
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (mut join, old) = captured_history(&context, pool.clone(), with_pending).await;
    let bases = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&bases);
    join.set_checkpoint_writer_base_test_hook(Arc::new(move |_| {
        observed.fetch_add(1, Ordering::SeqCst);
    }));
    pool.arm(None);
    join.prepare_checkpoint_async(&context).await.unwrap();
    let admissions = pool.finish();
    assert_eq!(bases.load(Ordering::SeqCst), 0);
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
    eprintln!(
        "REBASE_ADMISSIONS {}",
        json!({
            "pending": with_pending,
            "admissions": admissions.iter().map(|admission| json!({
                "before": admission.before, "additional": admission.additional,
            })).collect::<Vec<_>>(),
        })
    );
    (admissions, wire)
}

#[test]
fn test_rebase_rejects_each_admission_atomically_and_retries_the_same_epoch() {
    isolated_writer_test(|service, runtime| {
        runtime.block_on(check_admission_failures(false, service));
    });
}

#[test]
fn test_pending_rebase_rejects_each_admission_atomically_and_retries_the_same_epoch() {
    isolated_writer_test(|service, runtime| {
        runtime.block_on(check_admission_failures(true, service));
    });
}

async fn check_admission_failures(
    with_pending: bool,
    service: &crate::runtime::streaming::gather_work::TestService,
) {
    let (admissions, expected) = reference(with_pending, service).await;
    for reject in 1..=admissions.len() {
        let pool = Arc::new(CapturePool::new());
        let enabled = job().with_gather_owner(service.owner("rebase-admission-rejection".into()));
        let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
        let (mut join, old) = captured_history(&context, pool.clone(), with_pending).await;
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
