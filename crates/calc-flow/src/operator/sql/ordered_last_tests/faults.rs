use super::{recovery::*, *};
use std::sync::Mutex;

struct HoldEmit(Option<tokio::sync::oneshot::Sender<()>>);

#[async_trait]
impl StreamCollector for HoldEmit {
    async fn emit(&mut self, _: &str, _: Batch) -> Result<()> {
        self.0.take().unwrap().send(()).unwrap();
        std::future::pending().await
    }
}

#[tokio::test]
async fn test_ordered_last_dropped_emit_future_refunds_candidate_keys_and_metadata() {
    let job = job();
    let ctx = StreamOperatorContext::new(&job, "ordered_last", None);
    let mut op = seeded(&ctx).await;
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let pool = op.retention_runtime().unwrap().incremental_memory_pool();
    let basis = pool.reserved();
    let rows = [
        (Some("A"), Some(99.0), Some(8), Some(0)),
        (Some("new"), None, None, None),
    ];
    let batch =
        input(&rows).with_metadata(BatchMetadata::new("uncommitted", 9, JsonMap::new()).unwrap());
    let (entered, ready) = tokio::sync::oneshot::channel();
    let mut hold = HoldEmit(Some(entered));
    let mut pending = Box::pin(op.process_data("input", batch.clone(), &ctx, &mut hold));
    tokio::select! {
        result = &mut pending => panic!("emit must remain pending: {result:?}"),
        result = ready => result.unwrap(),
    }
    drop(pending);
    assert_eq!(pool.reserved(), basis);
    same_snapshot(&before, &op.checkpoint(Epoch::INITIAL).unwrap());
    assert_eq!(op.incremental.as_ref().unwrap().group_count(), 8);
    let result = process(&mut op, batch, &ctx).await;
    assert_eq!(result.num_rows(), 9);
    assert_eq!(
        op.checkpoint(Epoch::INITIAL).unwrap().inline_metadata["rows"],
        json!(10)
    );
}

#[tokio::test]
async fn test_ordered_last_sync_and_async_encoder_failure_keep_dirty_delta_and_old_capture() {
    const NAME: &str = "ordered-last-encoder-failure";
    let job = job();
    let ctx = StreamOperatorContext::new(&job, "ordered_last", None);
    let mut op = SqlOperator::new(NAME, RECOVERY_QUERY, vec!["input".into()], vec![]).unwrap();
    drop(process(&mut op, input(&SEED), &ctx).await);
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let next = [(Some("A"), Some(99.0), Some(8), Some(0))];
    let expected = process(&mut op, input(&next), &ctx).await;
    let pool = op.retention_runtime().unwrap().incremental_memory_pool();
    let basis = pool.reserved();
    let dirty = op.incremental.as_ref().unwrap().checkpoint_changes();
    {
        let _reject = compact::direct_async_tests::reject_ipc(NAME);
        let error = op.checkpoint(Epoch::INITIAL).unwrap_err();
        assert!(error.to_string().contains("injected native IPC failure"));
        assert!(
            op.prepare_checkpoint_async(&ctx)
                .await
                .unwrap_err()
                .to_string()
                .contains("injected native IPC failure")
        );
    }
    assert_eq!(pool.reserved(), basis);
    assert_eq!(op.incremental.as_ref().unwrap().checkpoint_changes(), dirty);
    same_snapshot(
        &before,
        &op.compact
            .as_ref()
            .unwrap()
            .capture
            .as_ref()
            .unwrap()
            .snapshot,
    );
    let mut old = operator(RECOVERY_QUERY);
    StreamOperator::restore(&mut old, &detached(&before)).unwrap();
    assert_eq!(
        values(&process(&mut old, input(&next), &ctx).await),
        values(&expected)
    );
    op.prepare_checkpoint_async(&ctx).await.unwrap();
    let after = op.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(delta_ids(&after).len(), 1);
    assert_eq!(op.incremental.as_ref().unwrap().checkpoint_changes(), 0);
    StreamOperator::restore(&mut old, &detached(&after)).unwrap();
    assert_eq!(
        values(&process(&mut old, input(&[]), &ctx).await),
        values(&expected)
    );
}

#[tokio::test]
async fn test_ordered_last_restore_prepare_failure_and_cancellation_refund_before_install() {
    let job = job();
    let ctx = StreamOperatorContext::new(&job, "ordered_last", None);
    let mut op = seeded(&ctx).await;
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let mut donor = seeded(&ctx).await;
    let incoming = BatchMetadata::new("restore-candidate", 7, JsonMap::new()).unwrap();
    drop(
        process(
            &mut donor,
            input(&[(Some("new"), Some(99.0), Some(8), Some(0))]).with_metadata(incoming.clone()),
            &ctx,
        )
        .await,
    );
    let snapshot = detached(&donor.checkpoint(Epoch::INITIAL).unwrap());
    let pool = op.retention_runtime().unwrap().incremental_memory_pool();
    let basis = pool.reserved();
    let leases = Arc::new(Mutex::new(Vec::new()));
    let guard = op.on_prepared_restore_for_test(&incoming, {
        let leases = leases.clone();
        move |candidate| {
            assert_eq!(candidate.rows, 9);
            assert!(candidate.reserved > 0);
            assert!(!candidate.backing.is_empty());
            assert_eq!(
                candidate
                    .records
                    .iter()
                    .map(RecordBatch::num_rows)
                    .sum::<usize>(),
                9
            );
            *leases.lock().unwrap() = candidate.backing;
            Err(sql_state_error("injected restore prepare failure"))
        }
    });
    assert!(
        StreamOperator::restore(&mut op, &snapshot)
            .unwrap_err()
            .to_string()
            .contains("injected restore prepare failure")
    );
    assert_eq!(guard.installs(), 0);
    assert!(
        leases
            .lock()
            .unwrap()
            .iter()
            .all(|lease| lease.upgrade().is_none())
    );
    assert_eq!(pool.reserved(), basis);
    same_snapshot(&before, &op.checkpoint(Epoch::INITIAL).unwrap());
    drop(guard);
    // Cancel at every real check boundary, including after full validation.
    let calls = std::cell::Cell::new(0);
    drop(
        op.prepare_restore(&snapshot, &|| {
            calls.set(calls.get() + 1);
            Ok(())
        })
        .unwrap(),
    );
    for stop in 1..=calls.get() {
        let seen = std::cell::Cell::new(0);
        let result = op.prepare_restore(&snapshot, &|| {
            seen.set(seen.get() + 1);
            if seen.get() == stop {
                Err(CalcFlowError::Cancelled {
                    run_id: "restore".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
        assert_eq!(pool.reserved(), basis, "cancel boundary {stop}");
        same_snapshot(&before, &op.checkpoint(Epoch::INITIAL).unwrap());
    }
    StreamOperator::restore(&mut op, &snapshot).unwrap();
    same_snapshot(&snapshot, &op.checkpoint(Epoch::INITIAL).unwrap());
}

#[tokio::test]
async fn test_ordered_last_ignored_null_groups_and_timestamp_state_obey_current_restore_budget() {
    let job = job();
    let ctx = StreamOperatorContext::new(&job, "ordered_last", None);
    let mut donor = seeded(&ctx).await;
    let snapshot = detached(&donor.checkpoint(Epoch::INITIAL).unwrap());
    let mut measured = operator(RECOVERY_QUERY);
    StreamOperator::restore(&mut measured, &snapshot).unwrap();
    let bytes = measured.retention_runtime().unwrap().compact_state_bytes();
    for (groups, limit) in [(7, 16 << 20), (8, bytes as u64 - 1)] {
        let mut limited = operator(RECOVERY_QUERY);
        limited
            .set_compact_state_budget(Some(StateBudget::new(groups, limit).unwrap()))
            .unwrap();
        assert!(StreamOperator::restore(&mut limited, &snapshot).is_err());
        assert!(limited.compact.is_none());
        assert_eq!(
            limited.retention_runtime().unwrap().compact_state_bytes(),
            0
        );
    }
    let before = measured.checkpoint(Epoch::INITIAL).unwrap();
    assert!(
        measured
            .set_compact_state_budget(Some(StateBudget::new(7, 16 << 20).unwrap()))
            .is_err()
    );
    assert!(
        measured
            .set_compact_state_budget(Some(StateBudget::new(8, bytes as u64 - 1).unwrap()))
            .is_err()
    );
    same_snapshot(&before, &measured.checkpoint(Epoch::INITIAL).unwrap());
}

#[tokio::test]
async fn test_ordered_last_nonempty_layout_four_budget_conflict_keeps_live_compact_state() {
    let job = job();
    let ctx = StreamOperatorContext::new(&job, "ordered_last", None);
    let mut legacy = operator(RECOVERY_QUERY);
    legacy.incremental_checked = true;
    drop(process(&mut legacy, input(&SEED), &ctx).await);
    let retained = legacy.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(retained.inline_metadata["state_layout"], json!(4));
    let mut receiver = seeded(&ctx).await;
    receiver
        .set_compact_state_budget(Some(StateBudget::new(16, 16 << 20).unwrap()))
        .unwrap();
    let before = receiver.checkpoint(Epoch::INITIAL).unwrap();
    let pool = receiver
        .retention_runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    assert!(
        StreamOperator::restore(&mut receiver, &retained)
            .unwrap_err()
            .to_string()
            .contains("nonempty retained checkpoint")
    );
    assert_eq!(pool.reserved(), basis);
    same_snapshot(&before, &receiver.checkpoint(Epoch::INITIAL).unwrap());
}
