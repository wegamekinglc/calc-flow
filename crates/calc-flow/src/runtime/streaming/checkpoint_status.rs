//! The streaming runner's checkpoint status accounting.
//!
//! `CheckpointStatus`, its private state, and the `CheckpointStatusHandle`
//! every acknowledgement path updates moved verbatim from `runner.rs`;
//! `runner` re-exports the names so projection and runner call sites keep
//! their paths. `CheckpointFailureCategory` moved along because it is part
//! of the status snapshot contract.

use std::{sync::Arc, time::Duration};

use parking_lot::Mutex;

use crate::CalcFlowError;
use crate::state::{PreparedManifestIdentity, SelectedManifest};
use crate::time::Epoch;

use super::checkpoint::coordinator::CheckpointPhase;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointFailureCategory {
    Timeout,
    Protocol,
    Io,
    Maintenance,
    Runtime,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CheckpointStatus {
    pub(crate) current_epoch: Option<Epoch>,
    pub(crate) phase: Option<CheckpointPhase>,
    pub(crate) terminal: bool,
    pub(crate) source_acknowledgements: usize,
    pub(crate) operator_acknowledgements: usize,
    pub(crate) sink_precommit_acknowledgements: usize,
    pub(crate) sink_commit_acknowledgements: usize,
    pub(crate) expected_sources: usize,
    pub(crate) expected_operators: usize,
    pub(crate) expected_sinks: usize,
    pub(crate) elapsed: Option<Duration>,
    pub(crate) last_completed_epoch: Option<Epoch>,
    pub(crate) installed_unknown_epoch: Option<Epoch>,
    pub(crate) failure_category: Option<CheckpointFailureCategory>,
    pub(crate) runtime_config_changed: bool,
}

pub(super) struct CheckpointStatusState {
    snapshot: CheckpointStatus,
    started: Option<tokio::time::Instant>,
}

#[derive(Clone)]
pub(crate) struct CheckpointStatusHandle(Arc<Mutex<CheckpointStatusState>>);

impl CheckpointStatusHandle {
    pub(super) fn new(
        identity: &PreparedManifestIdentity,
        selected: Option<&SelectedManifest>,
    ) -> Self {
        Self(Arc::new(Mutex::new(CheckpointStatusState {
            snapshot: CheckpointStatus {
                current_epoch: None,
                phase: None,
                terminal: false,
                source_acknowledgements: 0,
                operator_acknowledgements: 0,
                sink_precommit_acknowledgements: 0,
                sink_commit_acknowledgements: 0,
                expected_sources: identity.source_ids.len(),
                expected_operators: identity.operator_ids.len(),
                expected_sinks: identity.sink_ids.len(),
                elapsed: None,
                last_completed_epoch: selected.map(|selected| selected.manifest.epoch()),
                installed_unknown_epoch: None,
                failure_category: None,
                runtime_config_changed: selected
                    .is_some_and(|selected| selected.validation.runtime_config_changed),
            },
            started: None,
        })))
    }

    pub(super) fn snapshot(&self) -> CheckpointStatus {
        let state = self.0.lock();
        let mut snapshot = state.snapshot.clone();
        snapshot.elapsed = state.started.map(|started| started.elapsed());
        snapshot
    }

    pub(super) fn set_expected(&self, sources: usize, operators: usize, sinks: usize) {
        let mut state = self.0.lock();
        state.snapshot.expected_sources = sources;
        state.snapshot.expected_operators = operators;
        state.snapshot.expected_sinks = sinks;
    }

    pub(super) fn start(&self, epoch: Epoch, terminal: bool) {
        let mut state = self.0.lock();
        state.snapshot.current_epoch = Some(epoch);
        state.snapshot.phase = Some(CheckpointPhase::Requested);
        state.snapshot.terminal = terminal;
        state.snapshot.source_acknowledgements = 0;
        state.snapshot.operator_acknowledgements = 0;
        state.snapshot.sink_precommit_acknowledgements = 0;
        state.snapshot.sink_commit_acknowledgements = 0;
        state.snapshot.installed_unknown_epoch = None;
        state.snapshot.failure_category = None;
        state.started = Some(tokio::time::Instant::now());
    }

    pub(super) fn promote_terminal(&self, epoch: Epoch) -> crate::Result<()> {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch != Some(epoch) {
            return Err(checkpoint_protocol_error(
                epoch,
                "terminal promotion does not match the active checkpoint",
            ));
        }
        state.snapshot.terminal = true;
        Ok(())
    }

    pub(super) fn advance(&self, epoch: Epoch, phase: CheckpointPhase) {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch == Some(epoch) {
            state.snapshot.phase = Some(phase);
        }
    }

    pub(super) fn acknowledge_sources(&self, epoch: Epoch, count: usize) {
        self.acknowledge(epoch, |status| {
            status.source_acknowledgements = count;
        });
    }

    pub(super) fn acknowledge_operators(&self, epoch: Epoch, count: usize) {
        self.acknowledge(epoch, |status| {
            status.operator_acknowledgements = count;
        });
    }

    pub(super) fn acknowledge_sink_precommits(&self, epoch: Epoch, count: usize) {
        self.acknowledge(epoch, |status| {
            status.sink_precommit_acknowledgements = count;
        });
    }

    pub(super) fn acknowledge_sink_commits(&self, epoch: Epoch, count: usize) {
        self.acknowledge(epoch, |status| {
            status.sink_commit_acknowledgements = count;
        });
    }

    pub(super) fn acknowledge(&self, epoch: Epoch, update: impl FnOnce(&mut CheckpointStatus)) {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch == Some(epoch) {
            update(&mut state.snapshot);
        }
    }

    pub(super) fn sinks_committed(&self, epoch: Epoch) {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch == Some(epoch) {
            state.snapshot.phase = Some(CheckpointPhase::SinksCommitted);
            state.snapshot.last_completed_epoch = Some(epoch);
            state.snapshot.installed_unknown_epoch = None;
        }
    }

    pub(super) fn installed_unknown(&self, epoch: Epoch) {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch == Some(epoch) {
            state.snapshot.installed_unknown_epoch = Some(epoch);
            state.snapshot.failure_category = Some(CheckpointFailureCategory::Runtime);
        }
    }

    pub(super) fn complete(&self, epoch: Epoch) {
        let mut state = self.0.lock();
        if state.snapshot.current_epoch == Some(epoch) {
            state.snapshot.current_epoch = None;
            state.snapshot.phase = None;
            state.snapshot.terminal = false;
            state.snapshot.source_acknowledgements = 0;
            state.snapshot.operator_acknowledgements = 0;
            state.snapshot.sink_precommit_acknowledgements = 0;
            state.snapshot.sink_commit_acknowledgements = 0;
            state.snapshot.elapsed = None;
            state.started = None;
        }
    }

    pub(super) fn fail(&self, category: CheckpointFailureCategory) {
        let mut state = self.0.lock();
        state.snapshot.failure_category = Some(category);
    }

    pub(super) fn fail_if_unset(&self, category: CheckpointFailureCategory) {
        let mut state = self.0.lock();
        if state.snapshot.failure_category.is_none() {
            state.snapshot.failure_category = Some(category);
        }
    }

    pub(super) fn cancel(&self) {
        let mut state = self.0.lock();
        state.snapshot.current_epoch = None;
        state.snapshot.phase = None;
        state.snapshot.terminal = false;
        state.snapshot.source_acknowledgements = 0;
        state.snapshot.operator_acknowledgements = 0;
        state.snapshot.sink_precommit_acknowledgements = 0;
        state.snapshot.sink_commit_acknowledgements = 0;
        state.snapshot.elapsed = None;
        state.started = None;
    }
}

pub(super) fn checkpoint_protocol_error(epoch: Epoch, message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("checkpoint epoch {}: {message}", epoch.as_u64()),
    }
}
