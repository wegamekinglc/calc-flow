use super::*;
use crate::runtime::streaming::{
    operator_task::OperatorRestoreState,
    runner::{EntryFailure, FailureOrigin},
    supervisor::TaskId,
};
use crate::{CalcFlowError, Epoch};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::{Condvar, Mutex, Weak};

async fn checked_capture_harness(
    case: &str,
    fused: bool,
) -> (Harness, tempfile::TempDir, BatchMetadata) {
    let directory = tempfile::tempdir().unwrap();
    let plan = plan("trades", true);
    let transaction = lifecycle::transaction(&plan, directory.path()).await;
    let parts = plan.into_runtime_parts(EdgeBudget::default()).unwrap();
    let mut harness = harness_with_state(parts, fused, BTreeMap::new(), Some(transaction)).await;
    harness.runtime.data_gate.send(true).unwrap();
    harness
        .send(StreamMessage::data(lifecycle::tagged(
            &input(0, &[2, 5], true),
            case,
        )))
        .await;
    let first = harness.receive().await;
    let metadata = first.as_data().unwrap().metadata().clone();
    (harness, directory, metadata)
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_capture_error_and_panic_do_not_forward_barrier_or_lose_sql_identity() {
    for panic in [false, true] {
        let (mut harness, _directory, metadata) =
            checked_capture_harness(&format!("p1b-capture-fail-{panic}"), true).await;
        let registration =
            position_sql("trades", false).on_prepared_checkpoint_for_test(&metadata, move |_| {
                assert!(!panic, "p1b capture worker panic");
                Err(CalcFlowError::InvalidArgument {
                    field: "p1b.capture".into(),
                    message: "capture refusal".into(),
                })
            });
        harness.send(StreamMessage::barrier(Epoch::INITIAL)).await;
        let report = harness.runtime.supervisor.join_all().await;
        assert_eq!(report.primary_errors().len(), 1, "{report:?}");
        assert_eq!(report.primary_errors()[0].task_id, TaskId::new(1));
        assert_eq!(report.primary_errors()[0].task_name, "operator:positions");
        assert!(
            harness
                .runtime
                .sink_inputs
                .values_mut()
                .next()
                .unwrap()
                .recv()
                .await
                .unwrap()
                .is_none()
        );
        harness.core.sql_recovery.close_admission();
        let _ = harness.core.sql_recovery.drain().await;
        assert_eq!(harness.core.sql_recovery.live_counts(), (0, 0));
        assert_eq!(
            (
                registration.installs(),
                registration.acks(),
                registration.exits()
            ),
            (0, 0, 1)
        );
        assert_eq!(
            harness.core.metrics.snapshot().nodes[HEAD].fully_fanned_out_batches,
            1
        );
        assert_eq!(
            harness.core.metrics.snapshot().nodes[TAIL].fully_fanned_out_batches,
            1
        );
    }
}

struct Witness {
    operator: Weak<()>,
    capture: Weak<dyn std::any::Any + Send + Sync>,
    fee: Weak<MemoryReservation>,
    off_runtime: bool,
    reserved: usize,
}

#[derive(Default)]
struct Gate {
    entered: tokio::sync::Notify,
    witness: Mutex<Option<Witness>>,
    released: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    fn hold(&self, witness: Witness) -> crate::Result<()> {
        *self.witness.lock().unwrap() = Some(witness);
        self.wait()
    }

    fn wait(&self) -> crate::Result<()> {
        self.entered.notify_one();
        let mut released = self.released.lock().unwrap();
        while !*released {
            let (next, timeout) = self
                .changed
                .wait_timeout(released, Duration::from_secs(5))
                .unwrap();
            released = next;
            if timeout.timed_out() {
                return Err(CalcFlowError::Internal {
                    message: "capture test gate timed out".into(),
                });
            }
        }
        Ok(())
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.changed.notify_all();
    }

    fn paid_alive(&self) -> bool {
        self.witness
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|witness| {
                witness.off_runtime
                    && witness.reserved > 0
                    && witness.operator.upgrade().is_some()
                    && witness.capture.upgrade().is_some()
                    && witness.fee.upgrade().is_some()
            })
    }

    fn released_resources(&self) -> bool {
        self.witness
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|witness| {
                witness.operator.upgrade().is_none()
                    && witness.capture.upgrade().is_none()
                    && witness.fee.upgrade().is_none()
            })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_cancelled_capture_retains_the_paid_sql_loan_until_worker_join() {
    for fused in [false, true] {
        let (mut harness, _directory, metadata) =
            checked_capture_harness(&format!("p1b-cancel-capture-{fused}"), fused).await;
        let gate = Arc::new(Gate::default());
        let runtime_thread = std::thread::current().id();
        let held = gate.clone();
        let registration = position_sql("trades", false).on_prepared_checkpoint_for_test(
            &metadata,
            move |candidate| {
                held.hold(Witness {
                    operator: candidate.operator,
                    capture: candidate.capture,
                    fee: candidate.fee,
                    off_runtime: candidate.thread != runtime_thread,
                    reserved: candidate.reserved,
                })
            },
        );
        harness.send(StreamMessage::barrier(Epoch::INITIAL)).await;
        tokio::time::timeout(Duration::from_secs(5), gate.entered.notified())
            .await
            .unwrap();
        assert!(gate.paid_alive());
        assert_eq!(harness.core.sql_recovery.live_counts(), (1, 1));
        let core = harness.core.clone();
        let mut stopped = Box::pin(async {
            harness.cancellation.cancel();
            let report = harness.runtime.supervisor.join_all().await;
            core.sql_recovery.close_admission();
            let cleanup = core.sql_recovery.drain().await;
            (report, cleanup)
        });
        assert!(futures::poll!(stopped.as_mut()).is_pending());
        assert!(gate.paid_alive());
        gate.release();
        let (report, cleanup) = tokio::time::timeout(Duration::from_secs(5), stopped)
            .await
            .unwrap();
        assert!(report.errors.is_empty(), "{report:?}");
        assert!(cleanup.is_empty());
        assert_eq!(core.sql_recovery.live_counts(), (0, 0));
        assert_eq!(
            (
                registration.installs(),
                registration.acks(),
                registration.exits()
            ),
            (0, 0, 1)
        );
        assert!(gate.released_resources());
    }
}

fn restore_context(
    parts: &StreamRuntimePlanParts,
) -> (StreamJobContext, Arc<JobCore>, CancellationToken) {
    let cancellation = CancellationToken::new();
    let context = StreamJobContext::new(
        32,
        &parts.fingerprint,
        JsonMap::new(),
        None,
        cancellation.clone(),
    );
    let (commands, _) = mpsc::unbounded_channel();
    let core = Arc::new(JobCore::new(
        LaunchId::new(0),
        32,
        commands,
        MetricsRecorder::default(),
        StatusProjection::default(),
        true,
        parts.name.clone(),
    ));
    core.sql_recovery.configure(1).unwrap();
    (context, core, cancellation)
}

async fn rejected_restore(
    parts: StreamRuntimePlanParts,
    restores: BTreeMap<String, OperatorRestoreState>,
) -> Arc<JobCore> {
    let (context, core, cancellation) = restore_context(&parts);
    let result = run_operator_entry(parts, &context, &core, &cancellation, restores, None).await;
    let Err(EntryFailure::Failed(failure)) = result else {
        cancellation.cancel();
        panic!("invalid SQL restore entered the group");
    };
    assert_eq!(
        failure.origin,
        FailureOrigin::OperatorEntry {
            node_id: TAIL.into()
        }
    );
    core.sql_recovery.close_admission();
    let _ = core.sql_recovery.drain().await;
    assert_eq!(core.sql_recovery.live_counts(), (0, 0));
    assert!(core.runtime_status.lock().tasks.snapshot().is_empty());
    core
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_restore_failure_returns_the_original_sql_owner_and_cleans_group_entry() {
    for fallback in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let original = plan("trades", !fallback);
        let transaction = lifecycle::transaction(&original, directory.path()).await;
        let (entries, metadata) = lifecycle::capture(
            &transaction,
            fallback,
            &format!("p1b-restore-fail-{fallback}"),
        )
        .await;
        let parts = original.into_runtime_parts(EdgeBudget::default()).unwrap();
        let restores = lifecycle::restores(&transaction, &entries, &parts).await;
        let registration =
            position_sql("trades", false).on_prepared_restore_for_test(&metadata, |_| {
                Err(CalcFlowError::InvalidArgument {
                    field: "p1b.restore".into(),
                    message: "restore refusal".into(),
                })
            });
        let _ = rejected_restore(parts, restores).await;
        assert_eq!(registration.installs(), 0);
    }
}

#[derive(Default)]
struct RestoreWitness {
    reserved: usize,
    arrays: Vec<Weak<dyn datafusion::arrow::array::Array>>,
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_cancelled_retained_restore_awaits_the_original_sql_loan() {
    let directory = tempfile::tempdir().unwrap();
    let plan = plan("trades", false);
    let transaction = lifecycle::transaction(&plan, directory.path()).await;
    let (entries, metadata) =
        lifecycle::capture(&transaction, true, "p1b-retained-entry-cancel").await;
    let parts = plan.into_runtime_parts(EdgeBudget::default()).unwrap();
    let restores = lifecycle::restores(&transaction, &entries, &parts).await;
    let gate = Arc::new(Gate::default());
    let witness = Arc::new(Mutex::new(RestoreWitness::default()));
    let held_gate = gate.clone();
    let held_witness = witness.clone();
    let registration =
        position_sql("trades", false).on_prepared_restore_for_test(&metadata, move |candidate| {
            let arrays = candidate
                .records
                .iter()
                .flat_map(RecordBatch::columns)
                .map(Arc::downgrade)
                .collect();
            *held_witness.lock().unwrap() = RestoreWitness {
                reserved: candidate.reserved,
                arrays,
            };
            drop(candidate.records);
            held_gate.wait()
        });
    let (context, core, cancellation) = restore_context(&parts);
    let mut start = Box::pin(run_operator_entry(
        parts,
        &context,
        &core,
        &cancellation,
        restores,
        None,
    ));
    tokio::time::timeout(Duration::from_secs(5), async {
        tokio::select! {
            () = gate.entered.notified() => {},
            _ = start.as_mut() => panic!("retained entry did not hold its restore loan"),
        }
    })
    .await
    .unwrap();
    assert_eq!(core.sql_recovery.live_counts(), (1, 1));
    let mut stopped = Box::pin(async {
        cancellation.cancel();
        let result = start.await;
        core.sql_recovery.close_admission();
        let cleanup = core.sql_recovery.drain().await;
        (result, cleanup)
    });
    assert!(futures::poll!(stopped.as_mut()).is_pending());
    {
        let witness = witness.lock().unwrap();
        assert!(witness.reserved > 0);
        assert!(witness.arrays.iter().all(|array| array.upgrade().is_some()));
    }
    gate.release();
    let (result, cleanup) = tokio::time::timeout(Duration::from_secs(5), stopped)
        .await
        .unwrap();
    match result {
        Err(EntryFailure::Cancelled) => {}
        Err(EntryFailure::Failed(failure)) => {
            assert!(matches!(failure.error, CalcFlowError::Cancelled { .. }));
        }
        Ok(_) => panic!("cancelled retained entry succeeded"),
    }
    assert!(cleanup.is_empty());
    assert_eq!(registration.installs(), 0);
    assert_eq!(core.sql_recovery.live_counts(), (0, 0));
    assert!(core.runtime_status.lock().tasks.snapshot().is_empty());
    assert!(
        witness
            .lock()
            .unwrap()
            .arrays
            .iter()
            .all(|array| array.upgrade().is_none())
    );
}
