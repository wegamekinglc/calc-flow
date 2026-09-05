//! The streaming runner's shared core: registry state and command channel.
//!
//! `RunnerRegistryState`, the `RunnerCommand` protocol, and the `RunnerCore`
//! handle every lifecycle actor shares moved verbatim from `runner.rs`;
//! fields stay reachable to the runner's lifecycle functions through
//! `pub(super)` visibility.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64},
    },
};

use parking_lot::Mutex;
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinHandle;

use crate::CancellationToken;

use super::checkpoint_runtime::ValidatedCheckpointRuntime;
use super::failure::LaunchId;
use super::job::ValidatedContinuousJob;
use super::runner::{JobCore, RunnerDiagnostics};
#[cfg(test)]
use super::test_seams::TestLaunchProbe;

#[cfg(test)]
pub(super) const ABANDONED_RUNNER_WARNING: &str =
    "continuous runner dropped before shutdown completed; cancellation requested";

pub(super) struct RunnerRegistryState {
    pub(super) provisional: Option<LaunchId>,
    pub(super) live_jobs: BTreeMap<LaunchId, Arc<JobCore>>,
    pub(super) reaper_jobs: BTreeSet<LaunchId>,
    pub(super) pending_start: Option<LaunchId>,
    pub(super) shutting_down: bool,
}

pub(super) enum RunnerCommand {
    Start {
        launch_id: LaunchId,
        core: Arc<JobCore>,
        job: Box<ValidatedContinuousJob>,
        checkpoint: Option<Box<ValidatedCheckpointRuntime>>,
    },
    Wake(LaunchId),
    Shutdown,
}

pub(super) struct RunnerCore {
    pub(super) commands: mpsc::UnboundedSender<RunnerCommand>,
    pub(super) root_cancel: CancellationToken,
    pub(super) stop_after_first_job: bool,
    pub(super) registry: Mutex<RunnerRegistryState>,
    pub(super) driver: Mutex<Option<JoinHandle<()>>>,
    pub(super) diagnostics: RunnerDiagnostics,
    pub(super) next_launch_id: AtomicU64,
    pub(super) closed: AtomicBool,
    pub(super) changed: Notify,
    #[cfg(test)]
    pub(super) abandonment_warnings: AtomicU64,
    #[cfg(test)]
    pub(super) next_launch_probe: Mutex<Option<Arc<TestLaunchProbe>>>,
    #[cfg(test)]
    pub(super) panic_lifecycle_after_shutdown: AtomicBool,
}
