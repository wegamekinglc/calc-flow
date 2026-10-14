use super::*;

async fn admit_side(
    join: &mut StreamJoinOperator,
    side: &str,
    context: &StreamOperatorContext<'_>,
) -> Vec<(u64, Vec<RecordBatch>)> {
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    join.process_data(
        side,
        Batch::table(vec![record(side)], BatchMetadata::default()).unwrap(),
        context,
        &mut output,
    )
    .await
    .unwrap();
    output
        .drain("output")
        .iter()
        .map(|message| {
            let data = message.as_data().unwrap();
            (
                data.metadata().sequence(),
                data.table_payload().unwrap().batches().to_vec(),
            )
        })
        .collect()
}

#[tokio::test]
async fn test_pending_rebase_preserves_payloads_restores_and_continues() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (mut join, old) = checkpointed(&context).await;
    for side in ["right", "left"] {
        drop(admit_side(&mut join, side, &context).await);
    }
    assert_eq!(join.state.deltas.pending.iter().count(), 3);
    let previous = join.status();
    let rows = [
        Arc::as_ptr(&join.state.left.0),
        Arc::as_ptr(&join.state.right.0),
    ];
    let bases = count_base(&mut join);
    join.prepare_checkpoint_async(&context).await.unwrap();
    assert_eq!(bases.load(Ordering::SeqCst), 0);
    assert_eq!(join.status(), previous);
    assert_eq!(
        rows,
        [
            Arc::as_ptr(&join.state.left.0),
            Arc::as_ptr(&join.state.right.0)
        ]
    );
    assert!(join.state.deltas.pending.is_empty());
    let current = join.checkpoint(Epoch::new(5).unwrap()).unwrap();
    assert_eq!(current.inline_metadata["v2_inventory"]["base_epoch"], 4);
    assert_eq!(
        current.inline_metadata["v2_inventory"]["deltas"],
        serde_json::json!([])
    );
    assert!(
        current
            .segments
            .keys()
            .all(|name| !name.contains("-delta-"))
    );
    for (name, segment) in &old.segments {
        if name.contains("-payload-") {
            assert!(Arc::ptr_eq(
                &segment.bytes_arc(),
                &current.segments[name].bytes_arc()
            ));
        }
    }
    let mut restored = operator();
    restored.restore(&current).unwrap();
    assert_same_live(&join, &restored);
    for side in ["right", "left"] {
        let output = admit_side(&mut join, side, &context).await;
        assert!(!output.is_empty());
        assert_eq!(output, admit_side(&mut restored, side, &context).await);
        assert_same_live(&join, &restored);
    }
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    drop((join, restored, old, current));
    drop(context);
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    drop(enabled);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_pending_rebase_cancellation_and_refusal_refund_both_phases() {
    // Shared-service pressure can retire idle generation credit between assertions.
    let service = crate::runtime::streaming::gather_work::TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(pending_rebase_refund(&service));
    drop(runtime);
    service.shutdown();
}

async fn pending_rebase_refund(service: &crate::runtime::streaming::gather_work::TestService) {
    let enabled = job().with_gather_owner(service.owner("pending-rebase-refund".into()));
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (mut join, old) = checkpointed_from(&context, with_one_gib_pool(operator())).await;
    for side in ["left", "right"] {
        drop(admit_side(&mut join, side, &context).await);
    }
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let before = edge_tests::image(&join);
    let checks = std::cell::Cell::new(0_usize);
    let workspace = join.v2_writer_workspace().unwrap();
    let history = encode::rebase::copy(&join, &workspace, &context)
        .await
        .unwrap();
    let pending = encode::rebase::PendingInput::copy(&join, &workspace, &context)
        .await
        .unwrap();
    let encoded = encode::rebase::encode_pending(
        history,
        pending,
        &[schema(), schema()],
        &workspace,
        &|| {
            checks.set(checks.get() + 1);
            Ok(())
        },
    )
    .unwrap();
    let count = checks.get();
    assert!(count > 10);
    drop((encoded, workspace));
    let reserved = pool.reserved();
    for cancel_at in 1..=count {
        let workspace = join.v2_writer_workspace().unwrap();
        let history = encode::rebase::copy(&join, &workspace, &context)
            .await
            .unwrap();
        let pending = encode::rebase::PendingInput::copy(&join, &workspace, &context)
            .await
            .unwrap();
        checks.set(0);
        let result = encode::rebase::encode_pending(
            history,
            pending,
            &[schema(), schema()],
            &workspace,
            &|| {
                checks.set(checks.get() + 1);
                if checks.get() == cancel_at {
                    Err(crate::CalcFlowError::Cancelled {
                        run_id: "pending-rebase-check".into(),
                    })
                } else {
                    Ok(())
                }
            },
        );
        assert!(
            matches!(result, Err(crate::CalcFlowError::Cancelled { .. })),
            "check {cancel_at}"
        );
        drop(workspace);
        assert_eq!(pool.reserved(), reserved, "check {cancel_at}");
        assert_eq!(edge_tests::image(&join), before, "check {cancel_at}");
    }
    let pressure = join
        .runtime
        .runtime()
        .unwrap()
        .incremental_reservation("pending-rebase-refusal");
    pressure.try_grow((1 << 30) - pool.reserved()).unwrap();
    let failure = join.prepare_checkpoint_async(&context).await.unwrap_err();
    assert!(
        matches!(failure, crate::CalcFlowError::Internal { ref message }
            if message == "V2 IPC constructor credit admission failed"),
        "{failure}"
    );
    assert_eq!(edge_tests::image(&join), before);
    drop(pressure);
    assert_eq!(pool.reserved(), reserved);
    drop((join, old));
    drop(context);
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    drop(enabled);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_pending_rebase_install_preserves_stale_pending_and_history() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let disabled = job().with_checkpointing(false);
    let off = StreamOperatorContext::new(&disabled, "prepared-reuse", None);
    let mut pools = Vec::new();
    for invalid in [
        "generation",
        "captured_epoch",
        "revision",
        "dirty_revision",
        "identity",
        "trust",
        "off",
    ] {
        let (mut join, old) = checkpointed(&context).await;
        admit_left(&mut join, &context).await;
        let workspace = join.v2_writer_workspace().unwrap();
        let history = encode::rebase::copy(&join, &workspace, &context)
            .await
            .unwrap();
        let pending = encode::rebase::PendingInput::copy(&join, &workspace, &context)
            .await
            .unwrap();
        let base = encode::rebase::encode_pending(
            history,
            pending,
            &[schema(), schema()],
            &workspace,
            &|| Ok(()),
        )
        .unwrap();
        let mut submitted = join
            .v2_writer
            .tracker
            .snapshot(join.state.last_checkpoint_epoch);
        invalidate_pending_rebase(&mut join, &mut submitted, invalid);
        let before = edge_tests::image(&join);
        join.install_v2_writer_base(
            base,
            submitted,
            Preparation::Rebase,
            if invalid == "off" { &off } else { &context },
        )
        .unwrap();
        assert_eq!(edge_tests::image(&join), before, "{invalid}");
        pools.push(join.runtime.runtime().unwrap().incremental_memory_pool());
        drop((join, old, workspace));
    }
    drop((context, off));
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    assert!(disabled.gather_owner().close_and_drain().await.is_empty());
    drop((enabled, disabled));
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

fn invalidate_pending_rebase(
    join: &mut StreamJoinOperator,
    submitted: &mut cut::Cut,
    invalid: &str,
) {
    match invalid {
        "generation" => join.v2_writer.tracker = join.v2_writer.tracker.next_owner().unwrap(),
        "captured_epoch" => join.state.last_checkpoint_epoch = Epoch::new(5),
        "revision" => join.v2_writer.changed(false).unwrap(),
        "dirty_revision" => join.v2_writer.tracker.dirty_revision += 1,
        "trust" => join.v2_writer.invalidate_history().unwrap(),
        "identity" => {
            submitted.identity = different_identity(join.checkpoint_identity);
            join.v2_writer.tracker.identity = submitted.identity;
        }
        "off" => {}
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn test_pending_rebase_rejects_unique_ids_before_captured_history() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (original, old) = checkpointed(&context).await;
    let mut snapshot = edge_tests::remap_left_ids(old, |id| id + 1);
    snapshot
        .inline_metadata
        .insert("next_left_row_id".into(), 9.into());
    let mut join = operator();
    join.restore(&snapshot).unwrap();
    admit_left(&mut join, &context).await;
    reject_pending_identity(&join, &context).await;
    let first = join
        .state
        .deltas
        .pending
        .slots
        .iter_mut()
        .flatten()
        .next()
        .unwrap();
    let PendingOp::Upsert { row_id, .. } = &mut first.op else {
        unreachable!()
    };
    *row_id = 0;
    let before = edge_tests::image(&join);
    let bases = count_base(&mut join);
    let failure = join.prepare_checkpoint_async(&context).await.unwrap_err();
    assert!(
        failure
            .to_string()
            .contains("pending rows do not follow captured history")
    );
    join.await_compaction_release(&context).await.unwrap();
    assert_eq!(bases.load(Ordering::SeqCst), 0);
    assert_eq!(edge_tests::image(&join), before);
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    drop((join, original, snapshot));
    drop(context);
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    drop(enabled);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_pending_rebase_drops_decoded_rows_before_merge_and_retires_indexes() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (mut join, old) = checkpointed(&context).await;
    admit_left(&mut join, &context).await;
    let rows = Arc::downgrade(&join.state.left.0);
    let pending_column = Arc::downgrade(join.state.left.last().unwrap().record.column(0));
    let indexes = old
        .segments
        .iter()
        .filter(|(name, _)| !name.contains("-payload-"))
        .map(|(_, segment)| Arc::downgrade(&segment.bytes_arc()))
        .collect::<Vec<_>>();
    let held = old
        .segments
        .iter()
        .find(|(name, _)| name.starts_with("left-payload-"))
        .unwrap()
        .1
        .clone();
    let alias = Arc::downgrade(&held.bytes_arc());
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let workspace = join.v2_writer_workspace().unwrap();
    let history = encode::rebase::copy(&join, &workspace, &context)
        .await
        .unwrap();
    let pending = encode::rebase::PendingInput::copy(&join, &workspace, &context)
        .await
        .unwrap();
    drop((join, old));
    assert!(rows.upgrade().is_none());
    assert!(pending_column.upgrade().is_some());
    assert!(indexes.iter().all(|index| index.upgrade().is_some()));
    let retired_during_work = std::cell::Cell::new(false);
    let base = encode::rebase::encode_pending(
        history,
        pending,
        &[schema(), schema()],
        &workspace,
        &|| {
            retired_during_work
                .set(retired_during_work.get() || pending_column.upgrade().is_none());
            Ok(())
        },
    )
    .unwrap();
    assert!(retired_during_work.get());
    assert!(pending_column.upgrade().is_none());
    assert!(indexes.iter().all(|index| index.upgrade().is_none()));
    assert!(base.segments.keys().all(|name| !name.contains("-delta-")));
    drop((base, workspace));
    drop(context);
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    drop(enabled);
    assert!(alias.upgrade().is_some());
    assert!(pool.reserved() >= held.bytes_arc().capacity());
    drop(held);
    assert!(alias.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_pending_rebase_abandonment_and_cancel_retain_owners_until_drain() {
    use std::sync::{Mutex, mpsc};

    struct Release(Option<mpsc::Sender<()>>);
    impl Drop for Release {
        fn drop(&mut self) {
            if let Some(release) = self.0.take() {
                let _ = release.send(());
            }
        }
    }

    for cancel in [false, true] {
        let enabled = job();
        let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
        let (mut join, old) = checkpointed(&context).await;
        admit_left(&mut join, &context).await;
        let before = edge_tests::image(&join);
        let rows = Arc::downgrade(&join.state.left.0);
        let old_column = Arc::downgrade(join.state.left[0].record.column(0));
        let pending_column = Arc::downgrade(join.state.left.last().unwrap().record.column(0));
        let segments = old
            .segments
            .values()
            .map(|segment| Arc::downgrade(&segment.bytes_arc()))
            .collect::<Vec<_>>();
        let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
        let (entered, started) = tokio::sync::oneshot::channel();
        let entered = Mutex::new(Some(entered));
        let (release, wait) = mpsc::channel();
        let release = Release(Some(release));
        let wait = Mutex::new(wait);
        join.set_checkpoint_writer_test_hook(Arc::new(move |credit| {
            assert!(credit.size() > 0);
            entered.lock().unwrap().take().unwrap().send(()).unwrap();
            wait.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }));
        let mut preparing = Box::pin(join.prepare_checkpoint_async(&context));
        tokio::time::timeout(Duration::from_secs(5), async {
            tokio::select! {
                started = started => started.unwrap(),
                result = preparing.as_mut() => panic!("writer completed before release: {result:?}"),
            }
        }).await.unwrap();
        if cancel {
            enabled.cancellation().cancel();
        }
        drop(preparing);
        assert_eq!(edge_tests::image(&join), before);
        drop((join, old));
        let mut drain = Box::pin(enabled.gather_owner().close_and_drain());
        let blocked = futures::poll!(drain.as_mut()).is_pending();
        let ownership = (
            rows.upgrade().is_none(),
            old_column.upgrade().is_none(),
            pending_column.upgrade().is_some(),
            segments.iter().all(|segment| segment.upgrade().is_some()),
        );
        drop(release);
        let failures = tokio::time::timeout(Duration::from_secs(5), drain)
            .await
            .unwrap();
        assert!(blocked);
        assert_eq!(ownership, (true, true, true, true));
        assert!(failures.is_empty());
        assert!(pending_column.upgrade().is_none());
        assert!(segments.iter().all(|segment| segment.upgrade().is_none()));
        drop(context);
        drop(enabled);
        assert_eq!(pool.reserved(), 0);
    }
}

async fn reject_pending_identity(join: &StreamJoinOperator, context: &StreamOperatorContext<'_>) {
    let runtime = join.runtime.runtime.as_ref().unwrap();
    let pool = runtime.incremental_memory_pool();
    let baseline = pool.reserved();
    let workspace = runtime.incremental_reservation("pending-identity-test");
    let history = encode::rebase::copy(join, &workspace, context)
        .await
        .unwrap();
    let mut pending = encode::rebase::PendingInput::copy(join, &workspace, context)
        .await
        .unwrap();
    pending.identity = different_identity(join.checkpoint_identity);
    let before = edge_tests::image(join);
    let error = encode::rebase::encode_pending(
        history,
        pending,
        &[schema(), schema()],
        &workspace,
        &|| Ok(()),
    )
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("pending layout differs from history")
    );
    assert_eq!(edge_tests::image(join), before);
    drop(workspace);
    assert_eq!(pool.reserved(), baseline);
}
