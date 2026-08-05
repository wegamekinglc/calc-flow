use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::{CalcFlowError, CancellationToken, JsonMap, Result};

/// The immutable, job-scoped context shared by every task of one streaming
/// job (plan task M2.1 derives per-task scopes from this value).
///
/// M1.1 introduces the type so `StreamOperatorContext` can expose the frozen
/// `job()` accessor (API note A2); the supervisor model of M2 owns
/// construction in production paths.
#[derive(Clone, Debug)]
pub struct StreamJobContext {
    job_id: u64,
    fingerprint: String,
    settings: JsonMap,
    deadline: Option<DateTime<Utc>>,
    cancellation: CancellationToken,
}

impl StreamJobContext {
    /// Creates a job context. `deadline` is `DateTime<Utc>`, so UTC is
    /// guaranteed by the type and needs no runtime validation.
    pub fn new(
        job_id: u64,
        fingerprint: impl Into<String>,
        settings: JsonMap,
        deadline: Option<DateTime<Utc>>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            job_id,
            fingerprint: fingerprint.into(),
            settings,
            deadline,
            cancellation,
        }
    }

    pub const fn job_id(&self) -> u64 {
        self.job_id
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub const fn settings(&self) -> &JsonMap {
        &self.settings
    }

    pub const fn deadline(&self) -> Option<&DateTime<Utc>> {
        self.deadline.as_ref()
    }

    pub const fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    #[allow(
        dead_code,
        reason = "the crate-private M2 tasks use scopes before the M2.4 public runner"
    )]
    pub(crate) fn for_source(&self, binding_id: &str) -> Result<StreamTaskContext> {
        self.scoped(StreamTaskKind::Source, binding_id)
    }

    #[allow(dead_code, reason = "node tasks begin in plan task M2.3")]
    pub(crate) fn for_node(&self, node_id: &str) -> Result<StreamTaskContext> {
        self.scoped(StreamTaskKind::Node, node_id)
    }

    #[allow(dead_code, reason = "sink tasks begin in plan task M2.4")]
    pub(crate) fn for_sink(&self, binding_id: &str) -> Result<StreamTaskContext> {
        self.scoped(StreamTaskKind::Sink, binding_id)
    }

    fn scoped(&self, kind: StreamTaskKind, scope_id: &str) -> Result<StreamTaskContext> {
        if scope_id.trim().is_empty() || scope_id.contains('\0') {
            return Err(CalcFlowError::InvalidArgument {
                field: kind.id_field().into(),
                message: "must be non-empty, non-whitespace, and contain no NUL".into(),
            });
        }
        Ok(StreamTaskContext {
            job: self.clone(),
            kind,
            scope_id: scope_id.into(),
        })
    }

    /// Verifies that the job remains active.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Cancelled`] when cancellation was requested or
    /// the deadline has passed.
    pub fn check_cancelled(&self) -> Result<()> {
        if self.cancellation.is_cancelled()
            || self.deadline.is_some_and(|deadline| Utc::now() >= deadline)
        {
            return Err(CalcFlowError::Cancelled {
                run_id: self.job_id.to_string(),
            });
        }
        Ok(())
    }
}

#[allow(
    dead_code,
    reason = "the crate-private M2 scopes are not public until the M2.4 runner"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StreamTaskKind {
    Source,
    Node,
    Sink,
}

impl StreamTaskKind {
    const fn id_field(self) -> &'static str {
        match self {
            Self::Source => "source.binding_id",
            Self::Node => "node_id",
            Self::Sink => "sink.binding_id",
        }
    }
}

/// Immutable task scope derived from one job context.
#[allow(
    dead_code,
    reason = "the crate-private M2 scopes are not public until the M2.4 runner"
)]
#[derive(Clone, Debug)]
pub(crate) struct StreamTaskContext {
    job: StreamJobContext,
    kind: StreamTaskKind,
    scope_id: Arc<str>,
}

#[allow(
    dead_code,
    reason = "the crate-private M2 scopes are not public until the M2.4 runner"
)]
impl StreamTaskContext {
    pub(crate) const fn job(&self) -> &StreamJobContext {
        &self.job
    }

    pub(crate) const fn kind(&self) -> StreamTaskKind {
        self.kind
    }

    pub(crate) fn scope_id(&self) -> &str {
        &self.scope_id
    }
}
