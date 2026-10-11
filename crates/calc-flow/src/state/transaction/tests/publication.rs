use super::*;
use crate::state::local::{
    LocalStateLineageBackend,
    publication_tests::{Operation, set_hook},
};

fn assert_empty_session(transaction: &ManifestTransaction) {
    let session = transaction.session_segments.lock();
    assert!(session.carried.is_empty());
    assert!(session.verified.is_empty());
    assert!(session.working.is_empty());
}

#[tokio::test]
async fn managed_snapshot_batches_publication_before_pins_and_recovery() {
    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
    let lineage = Arc::new(backend.open_local_lineage(&key).await.unwrap());
    let transaction =
        ManifestTransaction::open(lineage.clone(), &key, directory.path().join("manifests"), 2)
            .await
            .unwrap();
    let syncs = Arc::new(AtomicUsize::new(0));
    let observed = syncs.clone();
    let staging_syncs = Arc::new(AtomicUsize::new(0));
    let staged_observed = staging_syncs.clone();
    set_hook(
        &lineage,
        Some(Arc::new(move |operation, _| {
            if operation == Operation::PublishSync {
                observed.fetch_add(1, Ordering::SeqCst);
            }
            if operation == Operation::StagingSync {
                staged_observed.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })),
    );
    let staged = transaction
        .stage_operator_state(
            "window",
            Epoch::INITIAL,
            snapshot_with_segments(&[("a", b"first"), ("b", b"second")]),
        )
        .await
        .unwrap();
    assert_eq!(staging_syncs.load(Ordering::SeqCst), 1);
    assert_eq!(
        syncs.load(Ordering::SeqCst),
        1,
        "one snapshot must sync its committed directory once"
    );
    assert_eq!(transaction.session_segments.lock().working.len(), 2);
    transaction
        .publish(PreparedEpochManifest {
            manifest: manifest_with_operator_segments(Epoch::INITIAL, staged.segments.clone()),
            staged_segments: BTreeMap::new(),
        })
        .await
        .unwrap();
    let selected = transaction
        .select_latest(&identity_with_window())
        .await
        .unwrap()
        .unwrap();
    let restored = transaction
        .load_operator_state("window", &selected.manifest.operators()["window"])
        .await
        .unwrap();
    assert_eq!(restored.segments["a"].bytes(), b"first");
    assert_eq!(restored.segments["b"].bytes(), b"second");
    let carry = transaction
        .stage_operator_state(
            "window",
            Epoch::INITIAL.next().unwrap(),
            snapshot_with_segments(&[("a", b"first"), ("b", b"second")]),
        )
        .await
        .unwrap();
    assert_eq!(staged.segments, carry.segments);
    assert_eq!(syncs.load(Ordering::SeqCst), 1);
    assert_eq!(staging_syncs.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn managed_visible_retry_cannot_advance_session_before_confirmation() {
    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
    let lineage = Arc::new(backend.open_local_lineage(&key).await.unwrap());
    let transaction =
        ManifestTransaction::open(lineage.clone(), &key, directory.path().join("manifests"), 2)
            .await
            .unwrap();
    let prior = transaction
        .stage_operator_state(
            "window",
            Epoch::INITIAL,
            snapshot_with_segments(&[("prior", b"durable")]),
        )
        .await
        .unwrap();
    transaction
        .publish(PreparedEpochManifest {
            manifest: manifest_with_operator_segments(Epoch::INITIAL, prior.segments.clone()),
            staged_segments: BTreeMap::new(),
        })
        .await
        .unwrap();
    let epoch = Epoch::INITIAL.next().unwrap();
    let snapshot =
        || snapshot_with_segments(&[("prior", b"durable"), ("a", b"first"), ("b", b"second")]);
    assert_staging_failure_preserves_session(&transaction, &lineage, epoch, &snapshot).await;
    let failed_segment = format!("{}-{}.segment", epoch.as_u64(), super::super::digest("b"));
    set_hook(
        &lineage,
        Some(Arc::new(move |operation, path| {
            if operation == Operation::Rename
                && path
                    .file_name()
                    .is_some_and(|name| name == failed_segment.as_str())
            {
                Err(std::io::Error::other("second managed rename failed"))
            } else {
                Ok(())
            }
        })),
    );
    assert!(
        transaction
            .stage_operator_state("window", epoch, snapshot())
            .await
            .is_err()
    );
    {
        let session = transaction.session_segments.lock();
        assert_eq!(session.carried.len(), 1);
        assert_eq!(session.verified.len(), 1);
        assert_eq!(session.working.len(), 1);
    }
    set_hook(
        &lineage,
        Some(Arc::new(|operation, _| {
            if operation == Operation::PublishSync {
                Err(std::io::Error::other("injected persistent sync failure"))
            } else {
                Ok(())
            }
        })),
    );
    for _ in 0..3 {
        assert!(
            transaction
                .stage_operator_state("window", epoch, snapshot())
                .await
                .is_err(),
            "visible retry still needs a successful publication sync"
        );
        let session = transaction.session_segments.lock();
        assert_eq!(session.carried.len(), 1);
        assert_eq!(session.verified.len(), 1);
        assert_eq!(session.working.len(), 1);
    }
    assert_eq!(
        transaction
            .select_latest(&identity_with_window())
            .await
            .unwrap()
            .unwrap()
            .manifest
            .epoch(),
        Epoch::INITIAL
    );
    set_hook(&lineage, None);
    let staged = transaction
        .stage_operator_state("window", epoch, snapshot())
        .await
        .unwrap();
    assert_eq!(staged.segments.len(), 3);
    assert_eq!(transaction.session_segments.lock().carried.len(), 3);
}

async fn assert_staging_failure_preserves_session(
    transaction: &ManifestTransaction,
    lineage: &LocalStateLineageBackend,
    epoch: Epoch,
    snapshot: &impl Fn() -> OperatorStateSnapshot,
) {
    set_hook(
        lineage,
        Some(Arc::new(|operation, _| {
            if operation == Operation::StagingSync {
                Err(std::io::Error::other(
                    "persistent managed staging sync failure",
                ))
            } else {
                Ok(())
            }
        })),
    );
    for _ in 0..2 {
        assert!(
            transaction
                .stage_operator_state("window", epoch, snapshot())
                .await
                .is_err()
        );
        let session = transaction.session_segments.lock();
        assert_eq!(session.carried.len(), 1);
        assert_eq!(session.verified.len(), 1);
        assert_eq!(session.working.len(), 1);
    }
    assert_eq!(
        transaction
            .select_latest(&identity_with_window())
            .await
            .unwrap()
            .unwrap()
            .manifest
            .epoch(),
        Epoch::INITIAL
    );
}

#[tokio::test]
async fn cancellation_settles_complete_batch_and_retains_lease_and_lock() {
    for operation in [Operation::FileSync, Operation::Rename] {
        assert_cancellation_settles_batch(operation).await;
    }
}

async fn assert_cancellation_settles_batch(blocked_operation: Operation) {
    use std::sync::Condvar;
    use tokio::time::{Duration, timeout};

    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
    let lineage = Arc::new(backend.open_local_lineage(&key).await.unwrap());
    let transaction = Arc::new(
        ManifestTransaction::open(lineage.clone(), &key, directory.path().join("manifests"), 2)
            .await
            .unwrap(),
    );
    let token = crate::CancellationToken::new();
    let (started, entered) = tokio::sync::oneshot::channel();
    let started = parking_lot::Mutex::new(Some(started));
    let gate = Arc::new((std::sync::Mutex::new(false), Condvar::new()));
    let blocked = gate.clone();
    let renames = Arc::new(AtomicUsize::new(0));
    let observed = renames.clone();
    set_hook(
        &lineage,
        Some(Arc::new(move |operation, _| {
            if operation == blocked_operation && observed.fetch_add(1, Ordering::SeqCst) == 0 {
                started.lock().take().unwrap().send(()).unwrap();
                let (lock, wake) = &*blocked;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
            }
            Ok(())
        })),
    );
    let task_transaction = transaction.clone();
    let task_token = token.clone();
    let mut task = tokio::spawn(async move {
        task_transaction
            .stage_operator_state_cancellable(
                "window",
                Epoch::INITIAL,
                snapshot_with_segments(&[("a", b"first"), ("b", b"second")]),
                &task_token,
            )
            .await
    });
    timeout(Duration::from_secs(5), entered)
        .await
        .unwrap()
        .unwrap();
    token.cancel();
    assert!(timeout(Duration::from_millis(20), &mut task).await.is_err());
    assert!(matches!(
        backend.open_lineage(&key).await,
        Err(CalcFlowError::Conflict { .. })
    ));
    let first = StateHandle::new(
        "window",
        Epoch::INITIAL,
        "a",
        &format!(
            "committed/{}/{}/1-{}.segment",
            transaction.lineage_hash,
            super::super::digest("window"),
            super::super::digest("a")
        ),
        5,
        &super::super::digest("first"),
    )
    .unwrap();
    let competing_lineage = lineage.clone();
    let mut competing =
        tokio::spawn(async move { competing_lineage.publish_segment(&first).await });
    assert!(
        timeout(Duration::from_millis(20), &mut competing)
            .await
            .is_err()
    );
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    assert!(task.await.unwrap().is_err());
    let competing_result = competing.await.unwrap();
    if blocked_operation == Operation::FileSync {
        assert!(matches!(
            competing_result,
            Err(CalcFlowError::Conflict { .. })
        ));
    } else {
        competing_result.unwrap();
    }
    assert_cancelled_batch_state(
        &transaction,
        directory.path(),
        blocked_operation,
        renames.load(Ordering::SeqCst),
    )
    .await;
}

async fn assert_cancelled_batch_state(
    transaction: &ManifestTransaction,
    directory: &Path,
    blocked_operation: Operation,
    completed_operations: usize,
) {
    assert_eq!(
        completed_operations, 2,
        "the admitted finite batch must settle all segments"
    );
    assert_empty_session(transaction);
    assert_eq!(
        committed_file_count(&directory.join("state/committed")),
        if blocked_operation == Operation::FileSync {
            0
        } else {
            2
        }
    );
    if blocked_operation == Operation::FileSync {
        assert_eq!(committed_file_count(&directory.join("state/staging")), 2);
    }
    assert!(
        transaction
            .select_latest(&identity_with_window())
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn cancellation_before_admission_and_during_backend_lock_wait() {
    use crate::state::local::publication_tests::publication_lock;
    use std::{future::Future, task::Poll};

    assert_staging_cancellation_before_admission_and_lock_wait().await;

    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
    let lineage = Arc::new(backend.open_local_lineage(&key).await.unwrap());
    let transaction =
        ManifestTransaction::open(lineage.clone(), &key, directory.path().join("manifests"), 2)
            .await
            .unwrap();
    let token = crate::CancellationToken::new();
    let prepared = transaction
        .collect_staged_state_segments(
            "window",
            Epoch::INITIAL,
            &snapshot_with_segments(&[("a", b"first"), ("b", b"second")]).segments,
            &token,
        )
        .await
        .unwrap();
    transaction
        .validate_staged_segments(&prepared.validation, &token)
        .await
        .unwrap();
    let renames = Arc::new(AtomicUsize::new(0));
    let observed = renames.clone();
    set_hook(
        &lineage,
        Some(Arc::new(move |operation, _| {
            if operation == Operation::Rename {
                observed.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })),
    );
    let cancelled = crate::CancellationToken::new();
    cancelled.cancel();
    assert!(
        transaction
            .publish_staged_segments(&prepared.publication, &cancelled)
            .await
            .is_err()
    );
    assert_eq!(renames.load(Ordering::SeqCst), 0);
    let guard = publication_lock(&lineage).await;
    let operation = transaction.publish_staged_segments(&prepared.publication, &token);
    tokio::pin!(operation);
    std::future::poll_fn(|context| {
        assert!(operation.as_mut().poll(context).is_pending());
        Poll::Ready(())
    })
    .await;
    token.cancel();
    assert_eq!(renames.load(Ordering::SeqCst), 0);
    assert!(matches!(
        backend.open_lineage(&key).await,
        Err(CalcFlowError::Conflict { .. })
    ));
    drop(guard);
    assert!(operation.await.is_err());
    assert_eq!(
        renames.load(Ordering::SeqCst),
        2,
        "an admitted batch remains owned across backend lock waits"
    );
    assert_empty_session(&transaction);
}

async fn assert_staging_cancellation_before_admission_and_lock_wait() {
    use crate::state::local::publication_tests::publication_lock;
    use std::{future::Future, task::Poll};

    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
    let lineage = Arc::new(backend.open_local_lineage(&key).await.unwrap());
    let transaction =
        ManifestTransaction::open(lineage.clone(), &key, directory.path().join("manifests"), 2)
            .await
            .unwrap();
    let snapshot = snapshot_with_segments(&[("a", b"first"), ("b", b"second")]);
    let token = crate::CancellationToken::new();
    let prepared = transaction
        .collect_staged_state_segments("window", Epoch::INITIAL, &snapshot.segments, &token)
        .await
        .unwrap();
    let entries = prepared
        .validation
        .iter()
        .map(|handle| (handle, snapshot.segments[handle.segment_id()].bytes()))
        .collect::<Vec<_>>();
    let syncs = Arc::new(AtomicUsize::new(0));
    let observed = syncs.clone();
    set_hook(
        &lineage,
        Some(Arc::new(move |operation, _| {
            if operation == Operation::StagingSync {
                observed.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })),
    );
    let cancelled = crate::CancellationToken::new();
    cancelled.cancel();
    assert!(
        transaction
            .stage_new_segments(&entries, &cancelled)
            .await
            .is_err()
    );
    assert_eq!(syncs.load(Ordering::SeqCst), 0);

    let guard = publication_lock(&lineage).await;
    let operation = transaction.stage_new_segments(&entries, &token);
    tokio::pin!(operation);
    std::future::poll_fn(|context| {
        assert!(operation.as_mut().poll(context).is_pending());
        Poll::Ready(())
    })
    .await;
    token.cancel();
    assert_eq!(syncs.load(Ordering::SeqCst), 0);
    assert!(matches!(
        backend.open_lineage(&key).await,
        Err(CalcFlowError::Conflict { .. })
    ));
    drop(guard);
    assert!(operation.await.is_err());
    assert_eq!(syncs.load(Ordering::SeqCst), 1);
    assert_empty_session(&transaction);
    for handle in &prepared.handles {
        lineage.validate_segment(handle).await.unwrap();
        assert!(matches!(
            lineage.load_segment(handle).await,
            Err(CalcFlowError::NotFound { .. })
        ));
    }
}

#[tokio::test]
async fn sink_visible_retry_requires_confirmation_without_new_staging_validation() {
    let directory = TempDir::new().unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new("orders", PIPELINE_FINGERPRINT).unwrap();
    let lineage = Arc::new(backend.open_local_lineage(&key).await.unwrap());
    let transaction =
        ManifestTransaction::open(lineage.clone(), &key, directory.path().join("manifests"), 2)
            .await
            .unwrap();
    let segments = || {
        BTreeMap::from([
            ("a".into(), b"first".to_vec()),
            ("b".into(), b"second".to_vec()),
        ])
    };
    let staging_syncs = Arc::new(AtomicUsize::new(0));
    let observed = staging_syncs.clone();
    set_hook(
        &lineage,
        Some(Arc::new(move |operation, _| {
            if operation == Operation::StagingSync {
                observed.fetch_add(1, Ordering::SeqCst);
            }
            if operation == Operation::PublishSync {
                Err(std::io::Error::other("sink sync failed"))
            } else {
                Ok(())
            }
        })),
    );
    for _ in 0..2 {
        assert!(
            transaction
                .stage_sink_segments_cancellable(
                    "sink",
                    Epoch::INITIAL,
                    segments(),
                    &crate::CancellationToken::new()
                )
                .await
                .is_err()
        );
        let session = transaction.session_segments.lock();
        assert!(session.carried.is_empty());
        assert!(session.verified.is_empty());
        assert!(session.working.is_empty());
    }
    assert_eq!(staging_syncs.load(Ordering::SeqCst), 1);
    set_hook(&lineage, None);
    let committed = transaction
        .stage_sink_segments_cancellable(
            "sink",
            Epoch::INITIAL,
            segments(),
            &crate::CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(committed.len(), 2);
    assert_eq!(transaction.session_segments.lock().carried.len(), 2);
}
