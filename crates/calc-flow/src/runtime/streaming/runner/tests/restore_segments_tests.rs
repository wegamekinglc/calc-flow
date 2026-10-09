use super::*;

struct CursorRowsSource {
    inner: BaseRowsSource,
    reopened: Arc<Mutex<Vec<(i64, usize)>>>,
}

#[async_trait]
impl StreamSource for CursorRowsSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.inner.open(cursor).await?;
        self.reopened
            .lock()
            .push((self.inner.timestamp, self.inner.delivered));
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.inner.next().await
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }

    fn capabilities(&self) -> SourceCapabilities {
        self.inner.capabilities()
    }
}

fn cursor_source(
    permitted: &Arc<AtomicUsize>,
    released: &Arc<AtomicBool>,
    reopened: &Arc<Mutex<Vec<(i64, usize)>>>,
    count: usize,
    timestamp: i64,
) -> SourceBinding {
    SourceBinding::new(
        Box::new(CursorRowsSource {
            inner: BaseRowsSource {
                permitted: permitted.clone(),
                released: released.clone(),
                count,
                delivered: 0,
                timestamp,
                watermark_delivered: false,
            },
            reopened: reopened.clone(),
        }),
        None,
        0,
    )
    .unwrap()
}

#[tokio::test]
async fn test_managed_join_delta_rows_have_independent_credit() {
    let directory = tempfile::tempdir().unwrap();
    let managed_root = directory.path().join("managed");
    let observations = Arc::new(Mutex::new(RestoreObservations::default()));
    let parses = Arc::new(AtomicUsize::new(0));
    let left = Arc::new(AtomicUsize::new(0));
    let right = Arc::new(AtomicUsize::new(0));
    let released = Arc::new(AtomicBool::new(false));
    let reopened = Arc::new(Mutex::new(Vec::new()));
    let rows = Arc::new(Mutex::new(Vec::new()));
    let wire_kinds = Arc::new(Mutex::new((0, 0)));
    let observed_kinds = wire_kinds.clone();
    let wire_owners = observations.clone();
    let wire_hook: super::super::super::super::checkpoint_runtime::CheckpointPrepaidReadHook =
        Arc::new(move |bytes, _, credit, _| {
            assert!(credit.size() >= bytes.len());
            let mut kinds = observed_kinds.lock();
            if bytes.starts_with(b"CFJOIN1\0") {
                kinds.0 += 1;
            } else {
                assert!(bytes.starts_with(b"CFJDLT1\0"));
                kinds.1 += 1;
            }
            wire_owners.lock().wire.push(Arc::as_ptr(credit) as usize);
            Ok(())
        });
    let checkpoint = || {
        CheckpointRuntimeSpec::managed(
            ManagedCheckpointRuntime::new(&managed_root).unwrap(),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
        .with_join_preload_read_hook(wire_hook.clone())
    };
    let spec = || {
        let mut spec = ac5_job_spec(v1_restore_plan(&observations, &parses), &rows);
        spec.sources = vec![
            NamedSourceBinding {
                binding_id: "left".into(),
                binding: cursor_source(&left, &released, &reopened, 4, 95),
            },
            NamedSourceBinding {
                binding_id: "right".into(),
                binding: cursor_source(&right, &released, &reopened, 1, 100),
            },
        ];
        spec
    };
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec(), checkpoint())
        .await
        .unwrap();
    for expected in 1..=3 {
        left.store(expected, Ordering::SeqCst);
        wait_retained(&job, u64::try_from(expected).unwrap(), 0).await;
        job.trigger_checkpoint().await.unwrap();
    }
    right.store(1, Ordering::SeqCst);
    wait_retained(&job, 3, 1).await;
    wait_for_join_emission(&job, 3).await;
    job.trigger_checkpoint().await.unwrap();
    job.trigger_checkpoint().await.unwrap();
    left.store(4, Ordering::SeqCst);
    wait_retained(&job, 4, 1).await;
    wait_for_join_emission(&job, 4).await;
    job.trigger_checkpoint().await.unwrap();
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    reopened.lock().clear();

    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec(), checkpoint())
        .await
        .unwrap();
    wait_retained(&job, 4, 1).await;
    let mut cursors = reopened.lock().clone();
    cursors.sort_unstable();
    assert_eq!(cursors, [(95, 4), (100, 1)]);
    assert!(!released.load(Ordering::SeqCst));
    released.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(rows.lock().as_slice(), [4]);
    assert_eq!(*wire_kinds.lock(), (2, 1));
    assert_eq!(parses.load(Ordering::SeqCst), 1);
    assert_restore_owners(&observations);
}

fn assert_restore_owners(observations: &Mutex<RestoreObservations>) {
    let observed = observations.lock();
    assert_eq!(observed.readers.len(), 5);
    assert!(
        observed.readers.iter().all(|(funding, native)| {
            funding.is_some_and(|(identity, paid)| {
                paid > 0
                    && Some(identity) != observed.descriptor
                    && !observed.wire.contains(&identity)
                    && *native
            })
        }),
        "all five real readers, including left delta at ordinal 3, need independent prepaid workspace on the native worker: {:?}",
        observed.readers
    );
    assert_eq!(observed.payloads.len(), 5);
    assert!(
        observed
            .payloads
            .iter()
            .all(|owner| owner.is_some_and(|(_, paid)| paid > 0)),
        "all restored Arrow payloads must keep independent resident credit"
    );
}

#[tokio::test]
async fn test_managed_join_delta_only_rows_have_independent_credit() {
    let directory = tempfile::tempdir().unwrap();
    let managed_root = directory.path().join("managed");
    let observations = Arc::new(Mutex::new(RestoreObservations::default()));
    let parses = Arc::new(AtomicUsize::new(0));
    let left = Arc::new(AtomicUsize::new(0));
    let right = Arc::new(AtomicUsize::new(0));
    let released = Arc::new(AtomicBool::new(false));
    let reopened = Arc::new(Mutex::new(Vec::new()));
    let rows = Arc::new(Mutex::new(Vec::new()));
    let wire_kinds = Arc::new(Mutex::new((0, 0)));
    let observed_kinds = wire_kinds.clone();
    let wire_owners = observations.clone();
    let wire_hook: super::super::super::super::checkpoint_runtime::CheckpointPrepaidReadHook =
        Arc::new(move |bytes, _, credit, _| {
            assert!(credit.size() >= bytes.len());
            let mut kinds = observed_kinds.lock();
            if bytes.starts_with(b"CFJOIN1\0") {
                kinds.0 += 1;
            } else {
                assert!(bytes.starts_with(b"CFJDLT1\0"));
                kinds.1 += 1;
            }
            wire_owners.lock().wire.push(Arc::as_ptr(credit) as usize);
            Ok(())
        });
    let checkpoint = || {
        CheckpointRuntimeSpec::managed(
            ManagedCheckpointRuntime::new(&managed_root).unwrap(),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
        .with_join_preload_read_hook(wire_hook.clone())
    };
    let spec = || {
        let mut spec = ac5_job_spec(v1_restore_plan(&observations, &parses), &rows);
        spec.sources = vec![
            NamedSourceBinding {
                binding_id: "left".into(),
                binding: cursor_source(&left, &released, &reopened, 1, 95),
            },
            NamedSourceBinding {
                binding_id: "right".into(),
                binding: cursor_source(&right, &released, &reopened, 1, 100),
            },
        ];
        spec
    };
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec(), checkpoint())
        .await
        .unwrap();
    left.store(1, Ordering::SeqCst);
    right.store(1, Ordering::SeqCst);
    wait_retained(&job, 1, 1).await;
    wait_for_join_emission(&job, 1).await;
    job.trigger_checkpoint().await.unwrap();
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    reopened.lock().clear();

    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec(), checkpoint())
        .await
        .unwrap();
    wait_retained(&job, 1, 1).await;
    let mut cursors = reopened.lock().clone();
    cursors.sort_unstable();
    assert_eq!(cursors, [(95, 1), (100, 1)]);
    assert!(!released.load(Ordering::SeqCst));
    released.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(rows.lock().as_slice(), [1]);
    assert_eq!(*wire_kinds.lock(), (0, 2));
    assert_eq!(parses.load(Ordering::SeqCst), 1);
    assert_delta_only_restore_owners(&observations);
}

fn assert_delta_only_restore_owners(observations: &Mutex<RestoreObservations>) {
    let observed = observations.lock();
    assert_eq!(observed.readers.len(), 2);
    eprintln!(
        "delta-only restored readers: {:?}; post-copy resident observations: {:?}",
        observed.readers, observed.payloads,
    );
    assert!(
        observed.readers.iter().all(|(funding, native)| {
            funding.is_some_and(|(identity, paid)| {
                paid > 0
                    && Some(identity) != observed.descriptor
                    && !observed.wire.contains(&identity)
                    && *native
            })
        }),
        "both real delta-only readers need independent prepaid workspace on the native worker: {:?}",
        observed.readers
    );
    assert_eq!(observed.payloads.len(), 2);
    assert!(
        observed
            .payloads
            .iter()
            .all(|owner| owner.is_some_and(|(_, paid)| paid > 0)),
        "all restored Arrow payloads must keep independent resident credit"
    );
}
