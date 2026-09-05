//! Test-only fault-injection seams for the streaming runner.
//!
//! The checkpoint fault injector, started gate, terminal-commit seam, launch
//! probe, and the unix manifest fault wrapper moved verbatim from
//! `runner.rs`; `runner` re-exports them so existing `runner::` paths (and
//! `soak.rs`) are unchanged. Everything here compiles only under
//! `#[cfg(test)]`.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use parking_lot::Mutex;
use tokio::sync::Notify;

#[cfg(unix)]
use crate::state::{ManifestParentSyncOsFailureProbe, ManifestTransaction};
use crate::{CalcFlowError, CancellationToken};

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointFaultPoint {
    SourceAdmission,
    SourceCut,
    PartialAlignment,
    StateStage,
    SinkPreCommit,
    ManifestWrite,
    ManifestRename,
    ManifestParentSync,
    PartialSinkCommit,
    CompletedCommit,
    Retention,
    Compaction,
}

#[cfg(test)]
impl CheckpointFaultPoint {
    pub(crate) const ALL: [Self; 12] = [
        Self::SourceAdmission,
        Self::SourceCut,
        Self::PartialAlignment,
        Self::StateStage,
        Self::SinkPreCommit,
        Self::ManifestWrite,
        Self::ManifestRename,
        Self::ManifestParentSync,
        Self::PartialSinkCommit,
        Self::CompletedCommit,
        Self::Retention,
        Self::Compaction,
    ];
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointFaultMode {
    Io,
    Panic,
    Cancel,
    Restart,
}

#[cfg(test)]
impl CheckpointFaultMode {
    pub(crate) const ALL: [Self; 4] = [Self::Io, Self::Panic, Self::Cancel, Self::Restart];
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CheckpointFault {
    pub(crate) point: CheckpointFaultPoint,
    pub(crate) mode: CheckpointFaultMode,
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct CheckpointFaultState {
    pub(crate) armed: Option<CheckpointFault>,
    pub(crate) trigger_count: usize,
    pub(crate) cancellation_trigger_count: usize,
    #[cfg(unix)]
    pub(crate) parent_sync_os_failure_probe: ManifestParentSyncOsFailureProbe,
}

#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct CheckpointFaultInjector(Arc<Mutex<CheckpointFaultState>>);

#[cfg(test)]
impl CheckpointFaultInjector {
    pub(crate) fn armed(point: CheckpointFaultPoint, mode: CheckpointFaultMode) -> Self {
        Self(Arc::new(Mutex::new(CheckpointFaultState {
            armed: Some(CheckpointFault { point, mode }),
            trigger_count: 0,
            cancellation_trigger_count: 0,
            #[cfg(unix)]
            parent_sync_os_failure_probe: ManifestParentSyncOsFailureProbe::default(),
        })))
    }

    pub(crate) fn trigger(
        &self,
        point: CheckpointFaultPoint,
        cancellation: &CancellationToken,
    ) -> crate::Result<()> {
        let fault = {
            let mut state = self.0.lock();
            match state.armed {
                Some(fault) if fault.point == point => {
                    state.trigger_count += 1;
                    if fault.mode == CheckpointFaultMode::Cancel {
                        state.cancellation_trigger_count += 1;
                    }
                    state.armed.take()
                }
                _ => None,
            }
        };
        let Some(fault) = fault else {
            return Ok(());
        };
        match fault.mode {
            CheckpointFaultMode::Io => Err(CalcFlowError::Io {
                path: format!("/fault-injection/{point:?}/credential-canary"),
                source: std::io::Error::other("injected checkpoint I/O fault"),
            }),
            CheckpointFaultMode::Panic => {
                panic!("injected checkpoint panic at {point:?}")
            }
            CheckpointFaultMode::Cancel => {
                cancellation.cancel();
                Ok(())
            }
            CheckpointFaultMode::Restart => Err(CalcFlowError::Internal {
                message: format!("injected checkpoint restart at {point:?}"),
            }),
        }
    }

    #[cfg(unix)]
    pub(crate) fn is_armed(&self, point: CheckpointFaultPoint, mode: CheckpointFaultMode) -> bool {
        self.0
            .lock()
            .armed
            .is_some_and(|fault| fault.point == point && fault.mode == mode)
    }

    #[cfg(unix)]
    pub(crate) fn parent_sync_os_failure_probe(&self) -> ManifestParentSyncOsFailureProbe {
        self.0.lock().parent_sync_os_failure_probe.clone()
    }

    #[cfg(unix)]
    pub(crate) fn parent_sync_os_failure_count(&self) -> usize {
        self.0.lock().parent_sync_os_failure_probe.count()
    }

    pub(crate) fn trigger_count(&self) -> usize {
        self.0.lock().trigger_count
    }

    pub(crate) fn cancellation_trigger_count(&self) -> usize {
        self.0.lock().cancellation_trigger_count
    }
}

#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct CheckpointStartedTestGate {
    pub(crate) entered: Arc<AtomicBool>,
    pub(crate) entered_changed: Arc<Notify>,
    pub(crate) released: Arc<AtomicBool>,
    pub(crate) release_changed: Arc<Notify>,
}

#[cfg(test)]
impl CheckpointStartedTestGate {
    pub(crate) fn has_entered(&self) -> bool {
        self.entered.load(Ordering::Acquire)
    }

    pub(crate) async fn wait_until_entered(&self) {
        while !self.entered.load(Ordering::Acquire) {
            let changed = self.entered_changed.notified();
            if self.entered.load(Ordering::Acquire) {
                break;
            }
            changed.await;
        }
    }

    pub(crate) fn release(&self) {
        self.released.store(true, Ordering::Release);
        self.release_changed.notify_waiters();
    }

    pub(crate) async fn pause(&self) {
        self.entered.store(true, Ordering::Release);
        self.entered_changed.notify_waiters();
        while !self.released.load(Ordering::Acquire) {
            let changed = self.release_changed.notified();
            if self.released.load(Ordering::Acquire) {
                break;
            }
            changed.await;
        }
    }
}

#[cfg(test)]
pub(crate) struct TerminalCommitTestSeam {
    pub(crate) reached: tokio::sync::oneshot::Sender<()>,
    pub(crate) release: tokio::sync::oneshot::Receiver<()>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TestLaunchCheckpoint {
    AfterOperatorEntry,
    LivePublished,
}

#[cfg(test)]
pub(crate) struct TestLaunchProbe {
    pub(crate) checkpoint: TestLaunchCheckpoint,
    pub(crate) reached: AtomicBool,
    pub(crate) released: AtomicBool,
    pub(crate) changed: Notify,
}

#[cfg(test)]
impl TestLaunchProbe {
    pub(crate) fn new(checkpoint: TestLaunchCheckpoint) -> Self {
        Self {
            checkpoint,
            reached: AtomicBool::new(false),
            released: AtomicBool::new(false),
            changed: Notify::new(),
        }
    }

    pub(crate) async fn pause_at(&self, checkpoint: TestLaunchCheckpoint) {
        if self.checkpoint != checkpoint {
            return;
        }
        self.reached.store(true, Ordering::Release);
        self.changed.notify_waiters();
        loop {
            let notified = self.changed.notified();
            if self.released.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    pub(crate) async fn wait_until_reached(&self) {
        loop {
            let notified = self.changed.notified();
            if self.reached.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    pub(crate) fn release(&self) {
        self.released.store(true, Ordering::Release);
        self.changed.notify_waiters();
    }
}

#[cfg(all(test, unix))]
#[cfg(unix)]
pub(crate) fn configure_test_manifest_transaction(
    transaction: ManifestTransaction,
    faults: &CheckpointFaultInjector,
) -> ManifestTransaction {
    if faults.is_armed(
        CheckpointFaultPoint::ManifestParentSync,
        CheckpointFaultMode::Io,
    ) {
        transaction.with_real_parent_sync_failure_for_test(faults.parent_sync_os_failure_probe())
    } else {
        transaction
    }
}
