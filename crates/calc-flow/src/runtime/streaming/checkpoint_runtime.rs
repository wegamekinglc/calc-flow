//! The streaming runner's checkpoint runtime description and opened handle.
//!
//! `CheckpointRuntimeSpec` (with its storage variants and test seams),
//! `ValidatedCheckpointRuntime`, and `OpenedCheckpointRuntime` moved verbatim
//! from `runner.rs`; fields and crate-facing constructors stay reachable to
//! the runner's open/recovery functions through `pub(super)` visibility.
//! `runner` re-exports the names so `soak.rs` keeps its paths.

use std::{path::PathBuf, sync::Arc};

#[cfg(test)]
use crate::CancellationToken;
use crate::state::{ManifestTransaction, PreparedManifestIdentity, SelectedManifest, StateBackend};
use crate::time::Epoch;
use crate::{CalcFlowError, StreamRuntimeConfig};

use super::checkpoint::{ManagedCheckpointRuntime, OpenedManagedCheckpointRuntime};
use super::checkpoint_status::CheckpointStatusHandle;
#[cfg(test)]
use super::runner::{
    CheckpointFaultInjector, CheckpointFaultMode, CheckpointFaultPoint, CheckpointStartedTestGate,
};

pub(crate) struct CheckpointRuntimeSpec {
    pub(super) storage: CheckpointRuntimeStorage,
    pub(super) config: StreamRuntimeConfig,
    #[cfg(test)]
    pub(super) faults: CheckpointFaultInjector,
    #[cfg(test)]
    pub(super) started_gate: Option<CheckpointStartedTestGate>,
}

pub(super) enum CheckpointRuntimeStorage {
    // Constructed only by the test-only `new`; production always opens
    // through the managed storage path, but open_checkpoint_runtime still
    // destructure-matches this variant in every build.
    #[cfg_attr(not(test), allow(dead_code))]
    LegacyParts {
        state_backend: Arc<dyn StateBackend>,
        manifest_root: PathBuf,
    },
    Managed(ManagedCheckpointRuntime),
    #[cfg(test)]
    ManagedTestParts {
        state_backend: Arc<dyn StateBackend>,
        manifest_root: PathBuf,
    },
}

impl CheckpointRuntimeSpec {
    #[cfg(test)]
    pub(crate) fn new(
        state_backend: Arc<dyn StateBackend>,
        manifest_root: impl Into<PathBuf>,
        config: StreamRuntimeConfig,
    ) -> crate::Result<Self> {
        validate_checkpoint_config(&config)?;
        Ok(Self {
            storage: CheckpointRuntimeStorage::LegacyParts {
                state_backend,
                manifest_root: manifest_root.into(),
            },
            config,
            #[cfg(test)]
            faults: CheckpointFaultInjector::default(),
            #[cfg(test)]
            started_gate: None,
        })
    }

    pub(super) fn managed(
        storage: ManagedCheckpointRuntime,
        config: StreamRuntimeConfig,
    ) -> crate::Result<Self> {
        validate_checkpoint_config(&config)?;
        Ok(Self {
            storage: CheckpointRuntimeStorage::Managed(storage),
            config,
            #[cfg(test)]
            faults: CheckpointFaultInjector::default(),
            #[cfg(test)]
            started_gate: None,
        })
    }

    #[cfg(test)]
    pub(super) fn managed_test_parts(
        state_backend: Arc<dyn StateBackend>,
        manifest_root: impl Into<PathBuf>,
        config: StreamRuntimeConfig,
    ) -> crate::Result<Self> {
        validate_checkpoint_config(&config)?;
        Ok(Self {
            storage: CheckpointRuntimeStorage::ManagedTestParts {
                state_backend,
                manifest_root: manifest_root.into(),
            },
            config,
            faults: CheckpointFaultInjector::default(),
            started_gate: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_fault(
        mut self,
        point: CheckpointFaultPoint,
        mode: CheckpointFaultMode,
    ) -> Self {
        self.faults = CheckpointFaultInjector::armed(point, mode);
        self
    }

    #[cfg(test)]
    pub(crate) fn with_fault_probe(
        mut self,
        point: CheckpointFaultPoint,
        mode: CheckpointFaultMode,
    ) -> (Self, CheckpointFaultInjector) {
        let faults = CheckpointFaultInjector::armed(point, mode);
        self.faults = faults.clone();
        (self, faults)
    }

    #[cfg(test)]
    pub(crate) fn with_started_gate(mut self, gate: CheckpointStartedTestGate) -> Self {
        self.started_gate = Some(gate);
        self
    }
}

pub(super) fn validate_checkpoint_config(config: &StreamRuntimeConfig) -> crate::Result<()> {
    config.validate()?;
    if config.retained_epochs == 0 {
        return Err(CalcFlowError::InvalidArgument {
            field: "retained_epochs".into(),
            message: "must be positive".into(),
        });
    }
    Ok(())
}

pub(super) struct ValidatedCheckpointRuntime {
    pub(super) spec: CheckpointRuntimeSpec,
    pub(super) identity: PreparedManifestIdentity,
}

pub(super) struct OpenedCheckpointRuntime {
    pub(super) transaction: Arc<ManifestTransaction>,
    pub(super) _managed_storage: Option<OpenedManagedCheckpointRuntime>,
    pub(super) identity: PreparedManifestIdentity,
    pub(super) config: StreamRuntimeConfig,
    pub(super) selected: Option<SelectedManifest>,
    pub(super) next_epoch: Epoch,
    pub(super) status: CheckpointStatusHandle,
    pub(super) startup_orphans_removed: usize,
    pub(super) managed: bool,
    #[cfg(test)]
    pub(super) faults: CheckpointFaultInjector,
    #[cfg(test)]
    pub(super) started_gate: Option<CheckpointStartedTestGate>,
}

impl OpenedCheckpointRuntime {
    #[cfg(test)]
    pub(super) fn inject_fault(
        &self,
        point: CheckpointFaultPoint,
        cancellation: &CancellationToken,
    ) -> crate::Result<bool> {
        let trigger_count = self.faults.trigger_count();
        self.faults.trigger(point, cancellation)?;
        Ok(self.faults.trigger_count() != trigger_count && cancellation.is_cancelled())
    }

    #[cfg(test)]
    pub(super) async fn pause_after_started(&self) {
        if let Some(gate) = &self.started_gate {
            gate.pause().await;
        }
    }
}
