use std::{collections::BTreeMap, sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    checkpoint::coordinator::{
        CheckpointPhase as InternalCheckpointPhase, ManualCheckpointFailureCategory,
    },
    job::{SinkCapability, ValidatedContinuousJob},
    metrics::sink_metric_id,
    progress::ReplayPositioningCapability,
    runner::{
        CheckpointFailureCategory, CheckpointStatus as InternalCheckpointStatus,
        ContinuousJobOutcome, ContinuousJobState, ContinuousJobStatus, FailureOrigin,
        RuntimeFailure, StartFailure, TerminalCause as InternalTerminalCause,
    },
    source_task::SourceDeliveryCapability as InternalSourceDeliveryCapability,
};
use crate::{
    CalcFlowError, DeliveryGuarantee, Epoch, RetentionClass,
    continuous::{ReplayPositioning, SourceDeliveryCapability},
};

/// Stable category for a redacted public streaming failure.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingErrorCategory {
    Validation,
    Compile,
    Conflict,
    Cancelled,
    CheckpointTimeout,
    CheckpointMismatch,
    CheckpointPublicationUnknown,
    Io,
    Operator,
    Connector,
    TaskPanicked,
    Internal,
}

/// Public component kind associated with a streaming failure.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKind {
    Job,
    Edge,
    Source,
    Operator,
    Sink,
    Checkpoint,
}

/// Observable phase of the single active checkpoint.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointPhase {
    Requested,
    SourcesCut,
    OperatorsSnapshotted,
    SinksPrecommitted,
    ManifestInstalled,
    ManifestDurable,
    SinksCommitted,
    Completed,
}

impl From<InternalCheckpointPhase> for CheckpointPhase {
    fn from(phase: InternalCheckpointPhase) -> Self {
        match phase {
            InternalCheckpointPhase::Requested => Self::Requested,
            InternalCheckpointPhase::SourcesCut => Self::SourcesCut,
            InternalCheckpointPhase::OperatorsSnapshotted => Self::OperatorsSnapshotted,
            InternalCheckpointPhase::SinksPrecommitted => Self::SinksPrecommitted,
            InternalCheckpointPhase::ManifestDurable => Self::ManifestDurable,
            InternalCheckpointPhase::SinksCommitted => Self::SinksCommitted,
            InternalCheckpointPhase::Completed => Self::Completed,
        }
    }
}

/// Safe, data-only streaming failure with no raw cause or extension payload.
#[derive(Clone, Debug, Eq, Error, PartialEq, Serialize)]
#[error("{message}")]
pub struct StreamingError {
    category: StreamingErrorCategory,
    message: String,
    job_id: Option<u64>,
    epoch: Option<Epoch>,
    checkpoint_phase: Option<CheckpointPhase>,
    component_kind: Option<ComponentKind>,
    component_id: Option<String>,
    diagnostic_id: Option<u64>,
    position: u32,
}

impl StreamingError {
    /// Returns the stable failure category.
    pub const fn category(&self) -> StreamingErrorCategory {
        self.category
    }

    /// Returns the safe engine-authored message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the owning job ID when one was allocated.
    pub const fn job_id(&self) -> Option<u64> {
        self.job_id
    }

    /// Returns the associated checkpoint epoch, when applicable.
    pub const fn epoch(&self) -> Option<Epoch> {
        self.epoch
    }

    /// Returns the checkpoint phase, when applicable.
    pub const fn checkpoint_phase(&self) -> Option<CheckpointPhase> {
        self.checkpoint_phase
    }

    /// Returns the stable component kind, when applicable.
    pub const fn component_kind(&self) -> Option<ComponentKind> {
        self.component_kind
    }

    /// Returns the stable component ID, when applicable.
    pub fn component_id(&self) -> Option<&str> {
        self.component_id.as_deref()
    }

    /// Returns the safe diagnostic correlation ID, when available.
    pub const fn diagnostic_id(&self) -> Option<u64> {
        self.diagnostic_id
    }

    /// Returns this failure's deterministic position in an outcome.
    pub const fn position(&self) -> u32 {
        self.position
    }
}

pub(crate) fn project_public_error(job_id: Option<u64>, error: &CalcFlowError) -> StreamingError {
    let (category, message, component_kind, component_id) = preflight_failure_fields(error);
    StreamingError {
        category,
        message,
        job_id,
        epoch: None,
        checkpoint_phase: None,
        component_kind,
        component_id,
        diagnostic_id: None,
        position: 0,
    }
}

pub(crate) fn validation_error(
    component_kind: ComponentKind,
    component_id: Option<&str>,
    message: String,
) -> StreamingError {
    StreamingError {
        category: StreamingErrorCategory::Validation,
        message,
        job_id: None,
        epoch: None,
        checkpoint_phase: None,
        component_kind: Some(component_kind),
        component_id: component_id.map(str::to_owned),
        diagnostic_id: None,
        position: 0,
    }
}

pub(crate) fn project_start_failure(job_id: u64, failure: &StartFailure) -> StreamingError {
    project_runtime_failure(job_id, &failure.primary, failure.diagnostic_id, 0, None)
}

pub(crate) fn manual_checkpoint_failure_error(
    category: ManualCheckpointFailureCategory,
    epoch: Option<Epoch>,
    phase: Option<InternalCheckpointPhase>,
) -> CalcFlowError {
    CalcFlowError::Streaming(StreamingError {
        category: match category {
            ManualCheckpointFailureCategory::Io => StreamingErrorCategory::Io,
            ManualCheckpointFailureCategory::Timeout => StreamingErrorCategory::CheckpointTimeout,
            ManualCheckpointFailureCategory::Protocol => StreamingErrorCategory::CheckpointMismatch,
            ManualCheckpointFailureCategory::Internal => StreamingErrorCategory::Internal,
        },
        message: manual_checkpoint_failure_message(category, epoch),
        job_id: None,
        epoch,
        checkpoint_phase: phase.map(CheckpointPhase::from),
        component_kind: Some(ComponentKind::Checkpoint),
        component_id: None,
        diagnostic_id: None,
        position: 0,
    })
}

pub(crate) fn manual_sink_commit_failure_error(sink_id: &str, epoch: Epoch) -> CalcFlowError {
    CalcFlowError::Streaming(sink_commit_failure(None, sink_id, epoch, None, 0))
}

fn sink_commit_failure(
    job_id: Option<u64>,
    sink_id: &str,
    epoch: Epoch,
    diagnostic_id: Option<u64>,
    position: u32,
) -> StreamingError {
    StreamingError {
        category: StreamingErrorCategory::Connector,
        message: format!(
            "sink {sink_id:?} commit failed for checkpoint epoch {}",
            epoch.as_u64()
        ),
        job_id,
        epoch: Some(epoch),
        checkpoint_phase: Some(CheckpointPhase::ManifestDurable),
        component_kind: Some(ComponentKind::Sink),
        component_id: Some(sink_id.into()),
        diagnostic_id,
        position,
    }
}

fn manual_checkpoint_failure_message(
    category: ManualCheckpointFailureCategory,
    epoch: Option<Epoch>,
) -> String {
    match (category, epoch) {
        (ManualCheckpointFailureCategory::Io, Some(epoch)) => format!(
            "checkpoint storage operation failed for epoch {}",
            epoch.as_u64()
        ),
        (ManualCheckpointFailureCategory::Io, None) => "checkpoint storage operation failed".into(),
        (ManualCheckpointFailureCategory::Timeout, Some(epoch)) => format!(
            "checkpoint epoch {} exceeded the configured timeout",
            epoch.as_u64()
        ),
        (ManualCheckpointFailureCategory::Timeout, None) => {
            "checkpoint request exceeded the configured timeout".into()
        }
        (ManualCheckpointFailureCategory::Protocol, Some(epoch)) => format!(
            "checkpoint epoch {} did not match the active protocol",
            epoch.as_u64()
        ),
        (ManualCheckpointFailureCategory::Protocol, None) => {
            "checkpoint request did not match the active protocol".into()
        }
        (ManualCheckpointFailureCategory::Internal, _) => "checkpoint runtime failed".into(),
    }
}

pub(crate) fn project_manual_checkpoint_error(
    job_id: u64,
    error: &CalcFlowError,
    status: Option<&InternalCheckpointStatus>,
) -> StreamingError {
    if let CalcFlowError::Streaming(error) = error {
        return StreamingError {
            category: error.category,
            message: error.message.clone(),
            job_id: Some(job_id),
            epoch: error.epoch,
            checkpoint_phase: error.checkpoint_phase,
            component_kind: error.component_kind,
            component_id: error.component_id.clone(),
            diagnostic_id: error.diagnostic_id,
            position: error.position,
        };
    }
    if let Some(status) = status {
        if let Some(epoch) = status.installed_unknown_epoch {
            return checkpoint_publication_unknown(epoch, job_id, None, 0);
        }
        if let Some(category) = status.failure_category {
            let category = match category {
                CheckpointFailureCategory::Timeout => StreamingErrorCategory::CheckpointTimeout,
                CheckpointFailureCategory::Protocol => StreamingErrorCategory::CheckpointMismatch,
                CheckpointFailureCategory::Io => StreamingErrorCategory::Io,
                CheckpointFailureCategory::Runtime if matches!(error, CalcFlowError::Io { .. }) => {
                    StreamingErrorCategory::Io
                }
                CheckpointFailureCategory::Maintenance | CheckpointFailureCategory::Runtime => {
                    StreamingErrorCategory::Internal
                }
            };
            let message = match (category, status.current_epoch) {
                (StreamingErrorCategory::CheckpointTimeout, Some(epoch)) => format!(
                    "checkpoint epoch {} exceeded the configured timeout",
                    epoch.as_u64()
                ),
                (StreamingErrorCategory::CheckpointTimeout, None) => {
                    "checkpoint request exceeded the configured timeout".into()
                }
                (StreamingErrorCategory::CheckpointMismatch, Some(epoch)) => {
                    format!(
                        "checkpoint epoch {} did not match the active protocol",
                        epoch.as_u64()
                    )
                }
                (StreamingErrorCategory::CheckpointMismatch, None) => {
                    "checkpoint request did not match the active protocol".into()
                }
                (StreamingErrorCategory::Io, Some(epoch)) => {
                    format!(
                        "checkpoint storage operation failed for epoch {}",
                        epoch.as_u64()
                    )
                }
                (StreamingErrorCategory::Io, None) => "checkpoint storage operation failed".into(),
                _ => "checkpoint runtime failed".into(),
            };
            return StreamingError {
                category,
                message,
                job_id: Some(job_id),
                epoch: status.current_epoch,
                checkpoint_phase: status.phase.map(CheckpointPhase::from),
                component_kind: Some(ComponentKind::Checkpoint),
                component_id: None,
                diagnostic_id: None,
                position: 0,
            };
        }
    }
    project_public_error(Some(job_id), error)
}

pub(crate) fn project_runtime_failures(
    job_id: u64,
    failures: Vec<Arc<RuntimeFailure>>,
    diagnostic_id: Option<u64>,
) -> Vec<StreamingError> {
    project_runtime_failures_with_checkpoint(job_id, failures, diagnostic_id, None)
}

fn project_runtime_failures_with_checkpoint(
    job_id: u64,
    failures: Vec<Arc<RuntimeFailure>>,
    diagnostic_id: Option<u64>,
    checkpoint: Option<&InternalCheckpointStatus>,
) -> Vec<StreamingError> {
    failures
        .into_iter()
        .enumerate()
        .map(|(position, failure)| {
            project_runtime_failure(
                job_id,
                &failure,
                diagnostic_id,
                u32::try_from(position).unwrap_or(u32::MAX),
                checkpoint,
            )
        })
        .collect()
}

fn project_runtime_failure(
    job_id: u64,
    failure: &RuntimeFailure,
    diagnostic_id: Option<u64>,
    position: u32,
    checkpoint: Option<&InternalCheckpointStatus>,
) -> StreamingError {
    if let (
        FailureOrigin::SinkCheckpoint { sink_id, .. },
        CalcFlowError::RecoveryRequired { .. },
        Some(status),
    ) = (&failure.origin, &failure.error, checkpoint)
        && let Some(epoch) = status.current_epoch
        && status.phase == Some(InternalCheckpointPhase::ManifestDurable)
    {
        return sink_commit_failure(Some(job_id), sink_id, epoch, diagnostic_id, position);
    }
    if let Some(error) =
        project_checkpoint_failure(job_id, failure, diagnostic_id, position, checkpoint)
    {
        return error;
    }
    let (category, message, component_kind, component_id) = safe_failure_fields(failure);
    StreamingError {
        category,
        message,
        job_id: Some(job_id),
        epoch: None,
        checkpoint_phase: None,
        component_kind,
        component_id,
        diagnostic_id,
        position,
    }
}

pub(crate) fn project_manual_terminal_outcome(
    job_id: u64,
    outcome: &ContinuousJobOutcome,
    checkpoint: Option<&InternalCheckpointStatus>,
) -> Option<StreamingError> {
    let (failure, mut error) = outcome
        .errors
        .iter()
        .enumerate()
        .map(|(position, failure)| {
            (
                failure,
                project_runtime_failure(
                    job_id,
                    failure,
                    None,
                    u32::try_from(position).unwrap_or(u32::MAX),
                    checkpoint,
                ),
            )
        })
        .next()?;
    if error.epoch.is_none()
        && let Some(status) = checkpoint
    {
        error.epoch = status.current_epoch;
        error.checkpoint_phase = status.phase.map(CheckpointPhase::from);
    }
    if let (FailureOrigin::SinkCheckpoint { sink_id, .. }, Some(epoch)) =
        (&failure.origin, error.epoch)
        && !matches!(failure.error, CalcFlowError::RecoveryRequired { .. })
    {
        error.message = format!(
            "sink {sink_id:?} pre_commit failed for checkpoint epoch {}",
            epoch.as_u64()
        );
    }
    Some(error)
}

fn project_checkpoint_failure(
    job_id: u64,
    failure: &RuntimeFailure,
    diagnostic_id: Option<u64>,
    position: u32,
    checkpoint: Option<&InternalCheckpointStatus>,
) -> Option<StreamingError> {
    let status = checkpoint?;
    if !matches!(
        &failure.origin,
        FailureOrigin::Task { task_name, .. } if task_name == "checkpoint"
    ) {
        return None;
    }
    if let Some(epoch) = status.installed_unknown_epoch {
        return Some(checkpoint_publication_unknown(
            epoch,
            job_id,
            diagnostic_id,
            position,
        ));
    }
    let category = match status.failure_category? {
        CheckpointFailureCategory::Timeout => StreamingErrorCategory::CheckpointTimeout,
        CheckpointFailureCategory::Protocol => StreamingErrorCategory::CheckpointMismatch,
        CheckpointFailureCategory::Io => StreamingErrorCategory::Io,
        CheckpointFailureCategory::Runtime if matches!(failure.error, CalcFlowError::Io { .. }) => {
            StreamingErrorCategory::Io
        }
        CheckpointFailureCategory::Maintenance | CheckpointFailureCategory::Runtime => {
            StreamingErrorCategory::Internal
        }
    };
    let message = match (category, status.current_epoch) {
        (StreamingErrorCategory::CheckpointTimeout, Some(epoch)) => format!(
            "checkpoint epoch {} exceeded the configured timeout",
            epoch.as_u64()
        ),
        (StreamingErrorCategory::CheckpointTimeout, None) => {
            "checkpoint request exceeded the configured timeout".into()
        }
        (StreamingErrorCategory::CheckpointMismatch, Some(epoch)) => {
            format!(
                "checkpoint epoch {} did not match the active protocol",
                epoch.as_u64()
            )
        }
        (StreamingErrorCategory::CheckpointMismatch, None) => {
            "checkpoint request did not match the active protocol".into()
        }
        (StreamingErrorCategory::Io, Some(epoch)) => {
            format!(
                "checkpoint storage operation failed for epoch {}",
                epoch.as_u64()
            )
        }
        (StreamingErrorCategory::Io, None) => "checkpoint storage operation failed".into(),
        _ => "checkpoint runtime failed".into(),
    };
    Some(StreamingError {
        category,
        message,
        job_id: Some(job_id),
        epoch: status.current_epoch,
        checkpoint_phase: status.phase.map(CheckpointPhase::from),
        component_kind: Some(ComponentKind::Checkpoint),
        component_id: None,
        diagnostic_id,
        position,
    })
}

fn safe_failure_fields(
    failure: &RuntimeFailure,
) -> (
    StreamingErrorCategory,
    String,
    Option<ComponentKind>,
    Option<String>,
) {
    if matches!(failure.error, CalcFlowError::TaskPanicked { .. }) {
        return (
            StreamingErrorCategory::TaskPanicked,
            "streaming task failed".into(),
            None,
            None,
        );
    }
    match &failure.origin {
        FailureOrigin::Preflight => preflight_failure_fields(&failure.error),
        FailureOrigin::RunnerLifecycle => (
            StreamingErrorCategory::Internal,
            "streaming runner lifecycle failed".into(),
            Some(ComponentKind::Job),
            None,
        ),
        FailureOrigin::OperatorEntry { node_id } => (
            StreamingErrorCategory::Operator,
            format!("operator {node_id:?} entry failed"),
            Some(ComponentKind::Operator),
            Some(node_id.clone()),
        ),
        FailureOrigin::SourceOpen { binding_id } => {
            connector_failure_fields(ComponentKind::Source, binding_id, "source", "open")
        }
        FailureOrigin::SourceClose { binding_id } => {
            connector_failure_fields(ComponentKind::Source, binding_id, "source", "close")
        }
        FailureOrigin::SinkOpen { sink_id, .. } => {
            connector_failure_fields(ComponentKind::Sink, sink_id, "sink", "open")
        }
        FailureOrigin::SinkClose { sink_id, .. } => {
            connector_failure_fields(ComponentKind::Sink, sink_id, "sink", "close")
        }
        FailureOrigin::SinkWrite { sink_id, .. } => {
            connector_failure_fields(ComponentKind::Sink, sink_id, "sink", "write")
        }
        FailureOrigin::SinkCheckpoint { sink_id, .. } => {
            connector_failure_fields(ComponentKind::Sink, sink_id, "sink", "checkpoint")
        }
        FailureOrigin::SinkIngress { edge_id, .. } => (
            StreamingErrorCategory::Internal,
            "sink ingress failed".into(),
            Some(ComponentKind::Edge),
            Some(edge_id.clone()),
        ),
        FailureOrigin::Task { task_name, .. } => task_failure_fields(task_name),
        FailureOrigin::Metrics { .. } => (
            StreamingErrorCategory::Internal,
            "streaming metrics accounting failed".into(),
            Some(ComponentKind::Job),
            None,
        ),
    }
}

fn task_failure_fields(
    task_name: &str,
) -> (
    StreamingErrorCategory,
    String,
    Option<ComponentKind>,
    Option<String>,
) {
    if let Some((source_id, unit)) = task_name
        .strip_prefix("source:")
        .and_then(|name| name.rsplit_once(':'))
        && matches!(unit, "pump" | "task")
    {
        return connector_failure_fields(ComponentKind::Source, source_id, "source", "callback");
    }
    if let Some(node_id) = task_name.strip_prefix("operator:") {
        return (
            StreamingErrorCategory::Operator,
            format!("operator {node_id:?} execution failed"),
            Some(ComponentKind::Operator),
            Some(node_id.into()),
        );
    }
    (
        StreamingErrorCategory::Internal,
        "streaming task failed".into(),
        None,
        None,
    )
}

fn preflight_failure_fields(
    error: &CalcFlowError,
) -> (
    StreamingErrorCategory,
    String,
    Option<ComponentKind>,
    Option<String>,
) {
    if let CalcFlowError::InvalidArgument { field, message } = error {
        return invalid_argument_failure_fields(field, message);
    }
    let category = match error {
        CalcFlowError::Compile { .. } => StreamingErrorCategory::Compile,
        CalcFlowError::Conflict { .. } | CalcFlowError::PlanLeased { .. } => {
            StreamingErrorCategory::Conflict
        }
        CalcFlowError::Cancelled { .. } => StreamingErrorCategory::Cancelled,
        CalcFlowError::CheckpointMismatch { .. } => StreamingErrorCategory::CheckpointMismatch,
        CalcFlowError::Io { .. } => StreamingErrorCategory::Io,
        _ => StreamingErrorCategory::Internal,
    };
    let component_kind = matches!(
        error,
        CalcFlowError::CheckpointMismatch { .. }
            | CalcFlowError::RecoveryRequired { .. }
            | CalcFlowError::Io { .. }
    )
    .then_some(ComponentKind::Checkpoint);
    let message = match category {
        StreamingErrorCategory::Validation => "streaming job validation failed",
        StreamingErrorCategory::Compile => "streaming plan compilation failed",
        StreamingErrorCategory::Conflict => "streaming runtime ownership conflict",
        StreamingErrorCategory::Cancelled => "streaming job was cancelled",
        StreamingErrorCategory::CheckpointMismatch => {
            "checkpoint lineage contains invalid recovery data"
        }
        StreamingErrorCategory::Io => "managed checkpoint storage operation failed",
        _ => "streaming runtime initialization failed",
    };
    (category, message.into(), component_kind, None)
}

fn invalid_argument_failure_fields(
    field: &str,
    message: &str,
) -> (
    StreamingErrorCategory,
    String,
    Option<ComponentKind>,
    Option<String>,
) {
    if let Some(value) = field.strip_prefix("operators.") {
        let operator_id = strip_field_suffix(value, &[".checkpoint_capability"]);
        if crate::json::validate_portable_identifier("operator", operator_id).is_err() {
            return (
                StreamingErrorCategory::Validation,
                "operator ID is not a portable identifier".into(),
                Some(ComponentKind::Operator),
                None,
            );
        }
        return (
            StreamingErrorCategory::Validation,
            format!("operator {operator_id:?} checkpoint capability is invalid"),
            Some(ComponentKind::Operator),
            Some(operator_id.into()),
        );
    }
    if let Some(value) = field.strip_prefix("sources.") {
        let source_id = strip_field_suffix(
            value,
            &[
                ".capabilities.declared_schema",
                ".capabilities.max_batch_rows",
                ".capabilities.max_batch_bytes",
                ".watermark_policy.max_out_of_orderness",
                ".watermark_policy.event_time_column",
                ".capabilities.native_watermarks",
                ".watermark_policy",
                ".capabilities",
                ".outputs",
                ".cursor",
                ".watermark",
            ],
        );
        return (
            StreamingErrorCategory::Validation,
            format!("source {source_id:?} capability validation failed"),
            Some(ComponentKind::Source),
            Some(source_id.into()),
        );
    }
    if let Some(output_id) = field.strip_prefix("requirements.delivery.") {
        return delivery_requirement_failure_fields(output_id, message);
    }
    if let Some(value) = field.strip_prefix("sinks.") {
        return (
            StreamingErrorCategory::Validation,
            format!("sink {value:?} binding validation failed"),
            Some(ComponentKind::Sink),
            Some(value.into()),
        );
    }
    (
        StreamingErrorCategory::Validation,
        "streaming job validation failed".into(),
        None,
        None,
    )
}

fn delivery_requirement_failure_fields(
    output_id: &str,
    message: &str,
) -> (
    StreamingErrorCategory,
    String,
    Option<ComponentKind>,
    Option<String>,
) {
    let participant = [
        (ComponentKind::Source, "source "),
        (ComponentKind::Operator, "operator "),
        (ComponentKind::Sink, "sink "),
    ]
    .into_iter()
    .find_map(|(kind, prefix)| quoted_value_after(message, prefix).map(|id| (kind, id)));
    let safe_message = match participant {
        Some((ComponentKind::Source, source_id)) if message.contains("lossy delivery") => {
            format!("output {output_id:?} requires exactly_once but source {source_id:?} is lossy")
        }
        Some((ComponentKind::Source, source_id))
            if message.contains("cannot replay from an exact cursor") =>
        {
            format!(
                "output {output_id:?} requires exactly_once but source {source_id:?} lacks exact_pause_report_and_seek"
            )
        }
        Some((ComponentKind::Operator, operator_id)) => format!(
            "output {output_id:?} requires exactly_once but operator {operator_id:?} is not deterministic"
        ),
        Some((ComponentKind::Sink, sink_id)) if message.contains("bounded retention") => format!(
            "output {output_id:?} requires exactly_once but sink {sink_id:?} has bounded retention"
        ),
        Some((ComponentKind::Sink, sink_id)) if message.contains("not transactional") => {
            format!("output {output_id:?} requires exactly_once but sink {sink_id:?} is ordinary")
        }
        _ => format!("output {output_id:?} delivery capability proof failed"),
    };
    let (kind, component_id) = participant
        .map(|(kind, id)| (kind, id.to_owned()))
        .unwrap_or((ComponentKind::Job, output_id.to_owned()));
    (
        StreamingErrorCategory::Validation,
        safe_message,
        Some(kind),
        Some(component_id),
    )
}

fn strip_field_suffix<'a>(value: &'a str, suffixes: &[&str]) -> &'a str {
    suffixes
        .iter()
        .find_map(|suffix| value.strip_suffix(suffix))
        .unwrap_or(value)
}

fn quoted_value_after<'a>(message: &'a str, prefix: &str) -> Option<&'a str> {
    let suffix = message.split_once(prefix)?.1.strip_prefix('"')?;
    suffix.split_once('"').map(|(value, _)| value)
}

fn connector_failure_fields(
    component_kind: ComponentKind,
    component_id: &str,
    component_name: &str,
    operation: &str,
) -> (
    StreamingErrorCategory,
    String,
    Option<ComponentKind>,
    Option<String>,
) {
    (
        StreamingErrorCategory::Connector,
        format!("{component_name} {component_id:?} {operation} failed"),
        Some(component_kind),
        Some(component_id.into()),
    )
}

pub(crate) fn checkpoint_publication_unknown(
    epoch: Epoch,
    job_id: u64,
    diagnostic_id: Option<u64>,
    position: u32,
) -> StreamingError {
    StreamingError {
        category: StreamingErrorCategory::CheckpointPublicationUnknown,
        message: format!(
            "checkpoint epoch {} was installed but publication durability is unknown",
            epoch.as_u64()
        ),
        job_id: Some(job_id),
        epoch: Some(epoch),
        checkpoint_phase: Some(CheckpointPhase::ManifestInstalled),
        component_kind: Some(ComponentKind::Checkpoint),
        component_id: None,
        diagnostic_id,
        position,
    }
}

/// Public lifecycle state for a continuous job.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Running,
    Draining,
    Completed,
    Cancelled,
    Failed,
    RecoveryRequired,
}

impl From<ContinuousJobState> for JobState {
    fn from(state: ContinuousJobState) -> Self {
        match state {
            ContinuousJobState::Running => Self::Running,
            ContinuousJobState::Draining => Self::Draining,
            ContinuousJobState::Completed => Self::Completed,
            ContinuousJobState::Cancelled => Self::Cancelled,
            ContinuousJobState::Failed => Self::Failed,
            ContinuousJobState::RecoveryRequired => Self::RecoveryRequired,
        }
    }
}

/// Stable cause of one immutable terminal outcome.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalCause {
    NaturalEnd,
    GracefulShutdown,
    ExplicitCancel,
    DeadlineExceeded,
    Failure,
}

impl From<&InternalTerminalCause> for TerminalCause {
    fn from(cause: &InternalTerminalCause) -> Self {
        match cause {
            InternalTerminalCause::NaturalEnd => Self::NaturalEnd,
            InternalTerminalCause::GracefulShutdown => Self::GracefulShutdown,
            InternalTerminalCause::ExplicitCancel => Self::ExplicitCancel,
            InternalTerminalCause::DeadlineExceeded => Self::DeadlineExceeded,
            InternalTerminalCause::TaskFailure { .. } => Self::Failure,
        }
    }
}

impl From<ReplayPositioningCapability> for ReplayPositioning {
    fn from(capability: ReplayPositioningCapability) -> Self {
        match capability {
            ReplayPositioningCapability::ExactPauseReportAndSeek => Self::ExactPauseReportAndSeek,
            ReplayPositioningCapability::Unsupported => Self::Unsupported,
        }
    }
}

impl From<InternalSourceDeliveryCapability> for SourceDeliveryCapability {
    fn from(capability: InternalSourceDeliveryCapability) -> Self {
        match capability {
            InternalSourceDeliveryCapability::Lossless => Self::Lossless,
            InternalSourceDeliveryCapability::Lossy => Self::Lossy,
        }
    }
}

/// Frozen delivery mechanism for one sink binding.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SinkDelivery {
    Ordinary,
    EpochIdempotent {
        mechanism: String,
        retention: RetentionClass,
    },
    Transactional,
}

impl From<SinkCapability> for SinkDelivery {
    fn from(capability: SinkCapability) -> Self {
        match capability {
            SinkCapability::Ordinary => Self::Ordinary,
            SinkCapability::EpochIdempotent {
                mechanism,
                retention,
            } => Self::EpochIdempotent {
                mechanism,
                retention,
            },
            SinkCapability::Transactional => Self::Transactional,
        }
    }
}

/// Requested and proven delivery guarantee for one plan output.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OutputDeliveryStatus {
    pub requested: DeliveryGuarantee,
    pub effective: DeliveryGuarantee,
}

/// Payload-free queue counters and limits for one stable edge.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EdgeStatus {
    pub current_envelopes: usize,
    pub current_rows: usize,
    pub current_bytes: usize,
    pub high_water_envelopes: usize,
    pub high_water_rows: usize,
    pub high_water_bytes: usize,
    pub blocked_sends: u64,
    pub blocked_duration: Duration,
    pub envelope_limit: usize,
    pub row_limit: usize,
    pub byte_limit: usize,
}

/// Frozen capabilities and payload-free counters for one source.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SourceStatus {
    pub replay_positioning: ReplayPositioning,
    pub delivery: SourceDeliveryCapability,
    pub max_batch_rows: usize,
    pub max_batch_bytes: usize,
    pub next_sequence: Option<u64>,
    pub ended: bool,
    pub polls: u64,
    pub data_batches: u64,
    pub data_rows: u64,
    pub data_bytes: u64,
    pub fanned_out_batches: u64,
    pub fanned_out_rows: u64,
    pub fanned_out_bytes: u64,
    pub errors: u64,
}

/// Payload-free execution counters for one operator.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OperatorStatus {
    pub input_batches: u64,
    pub input_rows: u64,
    pub input_bytes: u64,
    pub fanned_out_batches: u64,
    pub fanned_out_rows: u64,
    pub fanned_out_bytes: u64,
    pub processing_duration: Duration,
    pub errors: u64,
    pub ended: bool,
    pub late_rows: u64,
    pub late_affected_batches: u64,
    pub max_lateness: Option<Duration>,
    pub null_event_time_rows: u64,
    pub null_event_time_batches: u64,
    pub datafusion_runtime_created: bool,
}

/// Frozen delivery evidence and counters for one sink.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SinkStatus {
    pub output_id: String,
    pub effective_delivery: SinkDelivery,
    pub delivered_batches: u64,
    pub delivered_rows: u64,
    pub delivered_bytes: u64,
    pub write_duration: Duration,
    pub errors: u64,
    pub ended: bool,
}

/// Progress and completion state for the single active checkpoint.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct CheckpointStatus {
    pub current_epoch: Option<Epoch>,
    pub phase: Option<CheckpointPhase>,
    pub terminal: bool,
    pub source_acknowledgements: usize,
    pub expected_sources: usize,
    pub operator_acknowledgements: usize,
    pub expected_operators: usize,
    pub sink_precommit_acknowledgements: usize,
    pub expected_sink_precommits: usize,
    pub sink_commit_acknowledgements: usize,
    pub expected_sink_commits: usize,
    pub elapsed: Option<Duration>,
    pub last_completed_epoch: Option<Epoch>,
    pub installed_unknown_epoch: Option<Epoch>,
    pub failure_category: Option<StreamingErrorCategory>,
    pub runtime_config_changed: bool,
}

/// Cloned, data-only observation snapshot for one job.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct JobStatus {
    pub job_id: u64,
    pub state: JobState,
    pub terminal_cause: Option<TerminalCause>,
    pub delivery: BTreeMap<String, OutputDeliveryStatus>,
    pub task_count: usize,
    pub task_errors: u64,
    pub metrics_overflowed: bool,
    pub edges: BTreeMap<String, EdgeStatus>,
    pub sources: BTreeMap<String, SourceStatus>,
    pub operators: BTreeMap<String, OperatorStatus>,
    pub sinks: BTreeMap<String, SinkStatus>,
    pub checkpoint: CheckpointStatus,
}

/// Immutable terminal result returned by every owning lifecycle observer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct JobOutcome {
    pub state: JobState,
    pub cause: TerminalCause,
    pub completed_epoch: Option<Epoch>,
    pub errors: Vec<StreamingError>,
}

pub(crate) fn project_job_outcome(
    job_id: u64,
    outcome: &ContinuousJobOutcome,
    checkpoint: Option<&InternalCheckpointStatus>,
    diagnostic_id: Option<u64>,
) -> JobOutcome {
    JobOutcome {
        state: outcome.state.into(),
        cause: TerminalCause::from(&outcome.cause),
        completed_epoch: checkpoint.and_then(|status| status.last_completed_epoch),
        errors: project_runtime_failures_with_checkpoint(
            job_id,
            outcome.errors.clone(),
            diagnostic_id,
            checkpoint,
        ),
    }
}

#[derive(Clone)]
struct SourceProjection {
    replay_positioning: ReplayPositioning,
    delivery: SourceDeliveryCapability,
    max_batch_rows: usize,
    max_batch_bytes: usize,
}

#[derive(Clone)]
struct SinkProjection {
    metric_id: String,
    output_id: String,
    delivery: SinkDelivery,
}

#[derive(Default)]
pub(crate) struct StatusProjection {
    delivery: BTreeMap<String, OutputDeliveryStatus>,
    edge_limits: BTreeMap<String, (usize, usize)>,
    sources: BTreeMap<String, SourceProjection>,
    sinks: BTreeMap<String, SinkProjection>,
}

impl StatusProjection {
    pub(crate) fn new(job: &ValidatedContinuousJob) -> Self {
        let delivery = job
            .plan
            .sink_routes
            .keys()
            .map(|output_id| {
                let requested = job
                    .plan
                    .requirements
                    .delivery
                    .get(output_id)
                    .copied()
                    .unwrap_or(DeliveryGuarantee::AtLeastOnce);
                (
                    output_id.clone(),
                    OutputDeliveryStatus {
                        requested,
                        effective: requested,
                    },
                )
            })
            .collect();
        let edge_limits = job
            .plan
            .edges
            .iter()
            .map(|(edge_id, edge)| {
                (
                    edge_id.clone(),
                    (edge.budget.max_rows, edge.budget.max_bytes),
                )
            })
            .collect();
        let sources = job
            .sources
            .iter()
            .map(|(source_id, binding)| {
                let capabilities = binding.sampled_capabilities();
                (
                    source_id.clone(),
                    SourceProjection {
                        replay_positioning: binding.sampled_replay_positioning().into(),
                        delivery: binding.sampled_delivery().into(),
                        max_batch_rows: capabilities.max_batch_rows,
                        max_batch_bytes: capabilities.max_batch_bytes,
                    },
                )
            })
            .collect();
        let sinks = job
            .sinks
            .iter()
            .flat_map(|(output_id, sinks)| {
                sinks.iter().map(move |sink| {
                    (
                        sink.sink_id.to_string(),
                        SinkProjection {
                            metric_id: sink_metric_id(output_id, sink.sink_id.as_str()),
                            output_id: output_id.clone(),
                            delivery: sink.binding.capability().into(),
                        },
                    )
                })
            })
            .collect();
        Self {
            delivery,
            edge_limits,
            sources,
            sinks,
        }
    }

    pub(crate) fn sink_outputs(&self) -> BTreeMap<String, String> {
        self.sinks
            .values()
            .map(|sink| (sink.metric_id.clone(), sink.output_id.clone()))
            .collect()
    }

    pub(crate) fn project(&self, status: &ContinuousJobStatus) -> JobStatus {
        let state = JobState::from(status.state);
        let terminal_cause = if matches!(
            state,
            JobState::Completed
                | JobState::Cancelled
                | JobState::Failed
                | JobState::RecoveryRequired
        ) {
            status.terminal_cause.as_ref().map(TerminalCause::from)
        } else {
            None
        };
        JobStatus {
            job_id: status.job_id,
            state,
            terminal_cause,
            delivery: self.delivery.clone(),
            task_count: status.tasks.len(),
            task_errors: status.metrics.job.task_errors,
            metrics_overflowed: status.metrics.job.metrics_overflowed,
            edges: self.project_edges(status),
            sources: self.project_sources(status),
            operators: Self::project_operators(status),
            sinks: self.project_sinks(status),
            checkpoint: project_checkpoint(status.checkpoint.as_ref()),
        }
    }

    fn project_edges(&self, status: &ContinuousJobStatus) -> BTreeMap<String, EdgeStatus> {
        status
            .metrics
            .edges
            .iter()
            .map(|(edge_id, metrics)| {
                let (row_limit, byte_limit) = self.edge_limits[edge_id];
                let channel = &metrics.channel;
                (
                    edge_id.clone(),
                    EdgeStatus {
                        current_envelopes: channel.queue_depth,
                        current_rows: channel.charged_rows,
                        current_bytes: channel.charged_bytes,
                        high_water_envelopes: channel.high_water_depth,
                        high_water_rows: channel.high_water_rows,
                        high_water_bytes: channel.high_water_bytes,
                        blocked_sends: channel.blocked_sends,
                        blocked_duration: channel.blocked_duration,
                        envelope_limit: row_limit,
                        row_limit,
                        byte_limit,
                    },
                )
            })
            .collect()
    }

    fn project_sources(&self, status: &ContinuousJobStatus) -> BTreeMap<String, SourceStatus> {
        self.sources
            .iter()
            .map(|(source_id, projection)| {
                let progress = &status.sources[source_id];
                let metrics = &status.metrics.sources[source_id];
                (
                    source_id.clone(),
                    SourceStatus {
                        replay_positioning: projection.replay_positioning,
                        delivery: projection.delivery,
                        max_batch_rows: projection.max_batch_rows,
                        max_batch_bytes: projection.max_batch_bytes,
                        next_sequence: progress.next_sequence,
                        ended: progress.ended,
                        polls: metrics.poll_count,
                        data_batches: metrics.data_batches,
                        data_rows: metrics.data_rows,
                        data_bytes: metrics.data_bytes,
                        fanned_out_batches: metrics.fully_fanned_out_batches,
                        fanned_out_rows: metrics.fully_fanned_out_rows,
                        fanned_out_bytes: metrics.fully_fanned_out_bytes,
                        errors: metrics.errors,
                    },
                )
            })
            .collect()
    }

    fn project_operators(status: &ContinuousJobStatus) -> BTreeMap<String, OperatorStatus> {
        status
            .metrics
            .nodes
            .iter()
            .map(|(node_id, metrics)| {
                let progress = &status.nodes[node_id];
                (
                    node_id.clone(),
                    OperatorStatus {
                        input_batches: metrics.input_batches,
                        input_rows: metrics.input_rows,
                        input_bytes: metrics.input_bytes,
                        fanned_out_batches: metrics.fully_fanned_out_batches,
                        fanned_out_rows: metrics.fully_fanned_out_rows,
                        fanned_out_bytes: metrics.fully_fanned_out_bytes,
                        processing_duration: metrics.processing_duration,
                        errors: metrics.errors,
                        ended: progress.ended,
                        late_rows: metrics.late_rows,
                        late_affected_batches: metrics.affected_batches,
                        max_lateness: metrics.max_lateness_micros.map(Duration::from_micros),
                        null_event_time_rows: metrics.null_event_time_rows,
                        null_event_time_batches: metrics.null_event_time_batches,
                        datafusion_runtime_created: progress.datafusion_runtime_created,
                    },
                )
            })
            .collect()
    }

    fn project_sinks(&self, status: &ContinuousJobStatus) -> BTreeMap<String, SinkStatus> {
        self.sinks
            .iter()
            .map(|(sink_id, projection)| {
                let metrics = &status.metrics.sinks[&projection.metric_id];
                let progress = &status.sinks[&projection.metric_id];
                (
                    sink_id.clone(),
                    SinkStatus {
                        output_id: projection.output_id.clone(),
                        effective_delivery: projection.delivery.clone(),
                        delivered_batches: metrics.delivered_batches,
                        delivered_rows: metrics.delivered_rows,
                        delivered_bytes: metrics.delivered_bytes,
                        write_duration: metrics.write_duration,
                        errors: metrics.errors,
                        ended: progress.ended,
                    },
                )
            })
            .collect()
    }
}

fn project_checkpoint(status: Option<&super::runner::CheckpointStatus>) -> CheckpointStatus {
    status.map_or_else(CheckpointStatus::default, |status| {
        let installed_unknown = status.installed_unknown_epoch.is_some();
        CheckpointStatus {
            current_epoch: status.current_epoch,
            phase: if installed_unknown {
                Some(CheckpointPhase::ManifestInstalled)
            } else {
                status.phase.map(CheckpointPhase::from)
            },
            terminal: status.terminal,
            source_acknowledgements: status.source_acknowledgements,
            expected_sources: status.expected_sources,
            operator_acknowledgements: status.operator_acknowledgements,
            expected_operators: status.expected_operators,
            sink_precommit_acknowledgements: status.sink_precommit_acknowledgements,
            expected_sink_precommits: status.expected_sinks,
            sink_commit_acknowledgements: status.sink_commit_acknowledgements,
            expected_sink_commits: status.expected_sinks,
            elapsed: status.elapsed,
            last_completed_epoch: status.last_completed_epoch,
            installed_unknown_epoch: status.installed_unknown_epoch,
            failure_category: if installed_unknown {
                Some(StreamingErrorCategory::CheckpointPublicationUnknown)
            } else {
                status.failure_category.map(checkpoint_failure_category)
            },
            runtime_config_changed: status.runtime_config_changed,
        }
    })
}

fn checkpoint_failure_category(category: CheckpointFailureCategory) -> StreamingErrorCategory {
    match category {
        CheckpointFailureCategory::Timeout => StreamingErrorCategory::CheckpointTimeout,
        CheckpointFailureCategory::Protocol => StreamingErrorCategory::CheckpointMismatch,
        CheckpointFailureCategory::Io => StreamingErrorCategory::Io,
        CheckpointFailureCategory::Maintenance | CheckpointFailureCategory::Runtime => {
            StreamingErrorCategory::Internal
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, error::Error as _, sync::Arc};

    use crate::{CalcFlowError, Epoch};

    use super::{ComponentKind, StreamingErrorCategory, project_runtime_failures};
    use crate::runtime::streaming::runner::{FailureOrigin, RuntimeFailure};
    use crate::runtime::streaming::{
        checkpoint::coordinator::CheckpointPhase as InternalCheckpointPhase,
        metrics::M2MetricsSnapshot,
        runner::{
            CheckpointFailureCategory, CheckpointStatus as InternalCheckpointStatus,
            ContinuousJobOutcome, ContinuousJobState, ContinuousJobStatus,
            TerminalCause as InternalTerminalCause,
        },
        supervisor::TaskId,
    };

    const SECRET: &str = "private-connector-payload-redaction-sentinel";
    const PATH: &str = "/srv/private/checkpoints/customer-42";
    const PANIC: &str = "private-panic-payload-redaction-sentinel";

    #[test]
    fn streaming_projection_maps_shared_error_vectors_without_sensitive_text() {
        let failures = vec![
            Arc::new(RuntimeFailure {
                origin: FailureOrigin::Preflight,
                error: CalcFlowError::Io {
                    path: PATH.into(),
                    source: std::io::Error::other(SECRET),
                },
            }),
            Arc::new(RuntimeFailure {
                origin: FailureOrigin::SourceOpen {
                    binding_id: "orders".into(),
                },
                error: CalcFlowError::ExternalProvider {
                    provider: "python".into(),
                    name: "source".into(),
                    version: "1".into(),
                    message: SECRET.into(),
                },
            }),
            Arc::new(RuntimeFailure {
                origin: FailureOrigin::Task {
                    task_id: TaskId::new(91),
                    task_name: format!("source:orders:{SECRET}"),
                },
                error: CalcFlowError::TaskPanicked {
                    task_id: 91,
                    message: PANIC.into(),
                },
            }),
        ];

        let projected = project_runtime_failures(17, failures, Some(23));

        assert_eq!(projected.len(), 3);
        assert_eq!(projected[0].category(), StreamingErrorCategory::Io);
        assert_eq!(
            projected[0].component_kind(),
            Some(ComponentKind::Checkpoint)
        );
        assert_eq!(projected[1].category(), StreamingErrorCategory::Connector);
        assert_eq!(projected[1].component_kind(), Some(ComponentKind::Source));
        assert_eq!(projected[1].component_id(), Some("orders"));
        assert_eq!(
            projected[2].category(),
            StreamingErrorCategory::TaskPanicked
        );
        assert_eq!(projected[2].component_id(), None);
        assert_eq!(
            projected
                .iter()
                .map(super::StreamingError::position)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(projected.iter().all(|error| error.job_id() == Some(17)));
        assert!(
            projected
                .iter()
                .all(|error| error.diagnostic_id() == Some(23))
        );
        assert!(projected.iter().all(|error| error.epoch().is_none()));

        let rendered = projected
            .iter()
            .flat_map(|error| {
                [
                    error.to_string(),
                    format!("{error:?}"),
                    format!("{error:#?}"),
                    serde_json::to_string(error).unwrap(),
                ]
            })
            .collect::<Vec<_>>()
            .join("\n");
        for sentinel in [SECRET, PATH, PANIC, "source:orders:", "91"] {
            assert!(!rendered.contains(sentinel), "leaked sentinel {sentinel:?}");
        }
        assert!(projected.iter().all(|error| error.source().is_none()));
    }

    #[test]
    fn runtime_task_and_metrics_failures_use_safe_public_coordinates() {
        const METRIC_ID: &str = "sink/736563726574/63726564656e7469616c";
        let failures = vec![
            Arc::new(RuntimeFailure {
                origin: FailureOrigin::Task {
                    task_id: TaskId::new(1),
                    task_name: "source:orders:pump".into(),
                },
                error: CalcFlowError::ExternalProvider {
                    provider: "python".into(),
                    name: "source".into(),
                    version: "1".into(),
                    message: SECRET.into(),
                },
            }),
            Arc::new(RuntimeFailure {
                origin: FailureOrigin::Task {
                    task_id: TaskId::new(2),
                    task_name: "operator:price".into(),
                },
                error: CalcFlowError::Operator {
                    node_id: "price".into(),
                    message: SECRET.into(),
                },
            }),
            Arc::new(RuntimeFailure {
                origin: FailureOrigin::Metrics {
                    component_id: METRIC_ID.into(),
                    counter: "errors",
                },
                error: CalcFlowError::Internal {
                    message: SECRET.into(),
                },
            }),
        ];

        let projected = project_runtime_failures(17, failures, Some(23));

        assert_eq!(projected[0].category(), StreamingErrorCategory::Connector);
        assert_eq!(projected[0].component_kind(), Some(ComponentKind::Source));
        assert_eq!(projected[0].component_id(), Some("orders"));
        assert_eq!(projected[1].category(), StreamingErrorCategory::Operator);
        assert_eq!(projected[1].component_kind(), Some(ComponentKind::Operator));
        assert_eq!(projected[1].component_id(), Some("price"));
        assert_eq!(projected[2].category(), StreamingErrorCategory::Internal);
        assert_eq!(projected[2].component_kind(), Some(ComponentKind::Job));
        assert_eq!(projected[2].component_id(), None);
        let rendered = serde_json::to_string(&projected).unwrap();
        for sentinel in [SECRET, METRIC_ID, "736563726574", "63726564656e7469616c"] {
            assert!(!rendered.contains(sentinel), "leaked sentinel {sentinel:?}");
        }
    }

    #[test]
    fn publication_unknown_projection_keeps_only_the_safe_epoch_coordinate() {
        let epoch = Epoch::new(7).unwrap();
        let error = super::checkpoint_publication_unknown(epoch, 17, None, 4);

        assert_eq!(
            error.category(),
            StreamingErrorCategory::CheckpointPublicationUnknown
        );
        assert_eq!(error.epoch(), Some(epoch));
        assert_eq!(
            error.to_string(),
            "checkpoint epoch 7 was installed but publication durability is unknown"
        );
        assert_eq!(error.position(), 4);
    }

    #[test]
    fn manual_checkpoint_projection_keeps_timeout_io_and_unknown_coordinates() {
        let epoch = Epoch::new(7).unwrap();
        let status = |failure_category, installed_unknown_epoch| InternalCheckpointStatus {
            current_epoch: Some(epoch),
            phase: Some(InternalCheckpointPhase::SinksPrecommitted),
            terminal: false,
            source_acknowledgements: 1,
            operator_acknowledgements: 1,
            sink_precommit_acknowledgements: 0,
            sink_commit_acknowledgements: 0,
            expected_sources: 1,
            expected_operators: 1,
            expected_sinks: 1,
            elapsed: None,
            last_completed_epoch: None,
            installed_unknown_epoch,
            failure_category: Some(failure_category),
            runtime_config_changed: false,
        };
        let timeout = super::project_manual_checkpoint_error(
            17,
            &CalcFlowError::Internal {
                message: SECRET.into(),
            },
            Some(&status(CheckpointFailureCategory::Timeout, None)),
        );
        let io = super::project_manual_checkpoint_error(
            17,
            &CalcFlowError::Io {
                path: PATH.into(),
                source: std::io::Error::other(SECRET),
            },
            Some(&status(CheckpointFailureCategory::Runtime, None)),
        );
        let unknown = super::project_manual_checkpoint_error(
            17,
            &CalcFlowError::Io {
                path: PATH.into(),
                source: std::io::Error::other(SECRET),
            },
            Some(&status(CheckpointFailureCategory::Runtime, Some(epoch))),
        );
        let outcome_io = super::project_runtime_failures_with_checkpoint(
            17,
            vec![Arc::new(RuntimeFailure {
                origin: FailureOrigin::Task {
                    task_id: TaskId::new(3),
                    task_name: "checkpoint".into(),
                },
                error: CalcFlowError::Io {
                    path: PATH.into(),
                    source: std::io::Error::other(SECRET),
                },
            })],
            None,
            Some(&status(CheckpointFailureCategory::Runtime, None)),
        )
        .remove(0);

        assert_eq!(
            timeout.category(),
            StreamingErrorCategory::CheckpointTimeout
        );
        assert_eq!(timeout.epoch(), Some(epoch));
        assert_eq!(io.category(), StreamingErrorCategory::Io);
        assert_eq!(io.epoch(), Some(epoch));
        assert_eq!(outcome_io.category(), StreamingErrorCategory::Io);
        assert_eq!(outcome_io.epoch(), Some(epoch));
        assert_eq!(
            unknown.category(),
            StreamingErrorCategory::CheckpointPublicationUnknown
        );
        assert_eq!(unknown.epoch(), Some(epoch));
        assert_eq!(
            unknown.checkpoint_phase(),
            Some(super::CheckpointPhase::ManifestInstalled)
        );
        for rendered in [
            format!("{timeout:?}"),
            format!("{io:#?}"),
            format!("{unknown:?}"),
        ] {
            assert!(!rendered.contains(SECRET));
            assert!(!rendered.contains(PATH));
        }
    }

    #[test]
    fn sink_delivery_uses_the_locked_externally_tagged_json_shape() {
        assert_eq!(
            serde_json::to_value(super::SinkDelivery::Ordinary).unwrap(),
            serde_json::json!("ordinary")
        );
        assert_eq!(
            serde_json::to_value(super::SinkDelivery::Transactional).unwrap(),
            serde_json::json!("transactional")
        );
        assert_eq!(
            serde_json::to_value(super::SinkDelivery::EpochIdempotent {
                mechanism: "ledger".into(),
                retention: crate::RetentionClass::Unbounded,
            })
            .unwrap(),
            serde_json::json!({
                "epoch_idempotent": {
                    "mechanism": "ledger",
                    "retention": "unbounded",
                }
            })
        );
    }

    #[test]
    fn validation_projection_preserves_dotted_participant_ids() {
        for (field, message, kind, id) in [
            (
                "operators.warehouse.primary.checkpoint_capability",
                "operator checkpoint capability is unproven",
                ComponentKind::Operator,
                "warehouse.primary",
            ),
            (
                "sources.warehouse.primary.capabilities.max_batch_rows",
                "must be greater than zero",
                ComponentKind::Source,
                "warehouse.primary",
            ),
            (
                "sinks.warehouse.primary",
                "binding validation failed",
                ComponentKind::Sink,
                "warehouse.primary",
            ),
        ] {
            let projected = super::project_public_error(
                None,
                &CalcFlowError::InvalidArgument {
                    field: field.into(),
                    message: message.into(),
                },
            );

            assert_eq!(projected.category(), StreamingErrorCategory::Validation);
            assert_eq!(projected.component_kind(), Some(kind));
            assert_eq!(projected.component_id(), Some(id));
        }
    }

    #[test]
    fn nonterminal_status_never_exposes_a_provisional_terminal_cause() {
        let internal = ContinuousJobStatus {
            job_id: 17,
            state: ContinuousJobState::Running,
            terminal_cause: Some(InternalTerminalCause::ExplicitCancel),
            tasks: BTreeMap::new(),
            edges: BTreeMap::new(),
            sources: BTreeMap::new(),
            nodes: BTreeMap::new(),
            sinks: BTreeMap::new(),
            progress: None,
            checkpoint: None,
            metrics: M2MetricsSnapshot::default(),
        };

        let status = super::StatusProjection::default().project(&internal);

        assert_eq!(status.state, super::JobState::Running);
        assert_eq!(status.terminal_cause, None);
    }

    #[test]
    fn checkpoint_projection_keeps_completed_and_indeterminate_epochs_distinct() {
        let completed = Epoch::new(7).unwrap();
        let indeterminate = Epoch::new(8).unwrap();
        let checkpoint = InternalCheckpointStatus {
            current_epoch: Some(indeterminate),
            phase: Some(InternalCheckpointPhase::SinksPrecommitted),
            terminal: false,
            source_acknowledgements: 2,
            operator_acknowledgements: 3,
            sink_precommit_acknowledgements: 4,
            sink_commit_acknowledgements: 0,
            expected_sources: 2,
            expected_operators: 3,
            expected_sinks: 4,
            elapsed: Some(std::time::Duration::from_micros(91)),
            last_completed_epoch: Some(completed),
            installed_unknown_epoch: Some(indeterminate),
            failure_category: Some(CheckpointFailureCategory::Timeout),
            runtime_config_changed: true,
        };
        let internal = ContinuousJobStatus {
            job_id: 17,
            state: ContinuousJobState::Running,
            terminal_cause: None,
            tasks: BTreeMap::new(),
            edges: BTreeMap::new(),
            sources: BTreeMap::new(),
            nodes: BTreeMap::new(),
            sinks: BTreeMap::new(),
            progress: None,
            checkpoint: Some(checkpoint),
            metrics: M2MetricsSnapshot::default(),
        };

        let status = super::StatusProjection::default().project(&internal);

        assert_eq!(status.checkpoint.last_completed_epoch, Some(completed));
        assert_eq!(
            status.checkpoint.installed_unknown_epoch,
            Some(indeterminate)
        );
        assert_ne!(
            status.checkpoint.last_completed_epoch,
            status.checkpoint.installed_unknown_epoch
        );
        assert_eq!(
            status.checkpoint.failure_category,
            Some(StreamingErrorCategory::CheckpointPublicationUnknown)
        );
        assert_eq!(
            status.checkpoint.phase,
            Some(super::CheckpointPhase::ManifestInstalled)
        );
        assert_eq!(status.checkpoint.expected_sink_precommits, 4);
        assert_eq!(status.checkpoint.expected_sink_commits, 4);
    }

    #[test]
    fn job_outcome_reuses_the_safe_ordered_error_projection() {
        let raw = ContinuousJobOutcome {
            state: ContinuousJobState::Failed,
            cause: InternalTerminalCause::TaskFailure {
                primary_task_id: TaskId::new(91),
            },
            errors: vec![
                Arc::new(RuntimeFailure {
                    origin: FailureOrigin::Task {
                        task_id: TaskId::new(91),
                        task_name: format!("source:orders:{SECRET}"),
                    },
                    error: CalcFlowError::TaskPanicked {
                        task_id: 91,
                        message: PANIC.into(),
                    },
                }),
                Arc::new(RuntimeFailure {
                    origin: FailureOrigin::SourceClose {
                        binding_id: "orders".into(),
                    },
                    error: CalcFlowError::ExternalProvider {
                        provider: "python".into(),
                        name: "source".into(),
                        version: "1".into(),
                        message: SECRET.into(),
                    },
                }),
            ],
        };

        let outcome = super::project_job_outcome(17, &raw, None, Some(23));

        assert_eq!(outcome.state, super::JobState::Failed);
        assert_eq!(outcome.cause, super::TerminalCause::Failure);
        assert_eq!(outcome.completed_epoch, None);
        assert_eq!(
            outcome
                .errors
                .iter()
                .map(super::StreamingError::category)
                .collect::<Vec<_>>(),
            vec![
                StreamingErrorCategory::TaskPanicked,
                StreamingErrorCategory::Connector,
            ]
        );
        assert_eq!(
            outcome
                .errors
                .iter()
                .map(super::StreamingError::position)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        let rendered = serde_json::to_string(&outcome).unwrap();
        for sentinel in [SECRET, PANIC, "source:orders:", "91"] {
            assert!(!rendered.contains(sentinel), "leaked sentinel {sentinel:?}");
        }
    }

    #[test]
    fn job_outcome_projects_indeterminate_checkpoint_as_the_same_safe_value() {
        let completed = Epoch::new(6).unwrap();
        let indeterminate = Epoch::new(7).unwrap();
        let checkpoint = InternalCheckpointStatus {
            current_epoch: Some(indeterminate),
            phase: Some(InternalCheckpointPhase::SinksPrecommitted),
            terminal: false,
            source_acknowledgements: 1,
            operator_acknowledgements: 1,
            sink_precommit_acknowledgements: 1,
            sink_commit_acknowledgements: 0,
            expected_sources: 1,
            expected_operators: 1,
            expected_sinks: 1,
            elapsed: None,
            last_completed_epoch: Some(completed),
            installed_unknown_epoch: Some(indeterminate),
            failure_category: Some(CheckpointFailureCategory::Runtime),
            runtime_config_changed: false,
        };
        let raw = ContinuousJobOutcome {
            state: ContinuousJobState::RecoveryRequired,
            cause: InternalTerminalCause::TaskFailure {
                primary_task_id: TaskId::new(3),
            },
            errors: vec![Arc::new(RuntimeFailure {
                origin: FailureOrigin::Task {
                    task_id: TaskId::new(3),
                    task_name: "checkpoint".into(),
                },
                error: CalcFlowError::RecoveryRequired {
                    pipeline_name: SECRET.into(),
                    message: PANIC.into(),
                },
            })],
        };

        let outcome = super::project_job_outcome(17, &raw, Some(&checkpoint), Some(23));

        assert_eq!(outcome.state, super::JobState::RecoveryRequired);
        assert_eq!(outcome.completed_epoch, Some(completed));
        assert_eq!(outcome.errors.len(), 1);
        assert_eq!(
            outcome.errors[0].category(),
            StreamingErrorCategory::CheckpointPublicationUnknown
        );
        assert_eq!(outcome.errors[0].epoch(), Some(indeterminate));
        assert_eq!(
            outcome.errors[0].checkpoint_phase(),
            Some(super::CheckpointPhase::ManifestInstalled)
        );
        assert_eq!(outcome.errors[0].diagnostic_id(), Some(23));
        let rendered = serde_json::to_string(&outcome).unwrap();
        for sentinel in [SECRET, PANIC] {
            assert!(!rendered.contains(sentinel), "leaked sentinel {sentinel:?}");
        }
    }

    #[test]
    fn rust_projection_matches_the_shared_python_vectors() {
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/a6/streaming_error_projection.json"
        ));
        let document: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let categories = [
            StreamingErrorCategory::Validation,
            StreamingErrorCategory::Compile,
            StreamingErrorCategory::Conflict,
            StreamingErrorCategory::Cancelled,
            StreamingErrorCategory::CheckpointTimeout,
            StreamingErrorCategory::CheckpointMismatch,
            StreamingErrorCategory::CheckpointPublicationUnknown,
            StreamingErrorCategory::Io,
            StreamingErrorCategory::Operator,
            StreamingErrorCategory::Connector,
            StreamingErrorCategory::TaskPanicked,
            StreamingErrorCategory::Internal,
        ];
        assert_eq!(
            serde_json::to_value(categories).unwrap(),
            document["categories"]
        );

        for vector in document["vectors"].as_array().unwrap() {
            let case = vector["case"].as_str().unwrap();
            let error = match case {
                "checkpoint_io" => project_runtime_failures(
                    17,
                    vec![Arc::new(RuntimeFailure {
                        origin: FailureOrigin::Preflight,
                        error: CalcFlowError::Io {
                            path: PATH.into(),
                            source: std::io::Error::other(SECRET),
                        },
                    })],
                    Some(23),
                )
                .remove(0),
                "checkpoint_publication_unknown" => {
                    super::checkpoint_publication_unknown(Epoch::new(7).unwrap(), 17, None, 0)
                }
                "source_open" => project_runtime_failures(
                    17,
                    vec![Arc::new(RuntimeFailure {
                        origin: FailureOrigin::SourceOpen {
                            binding_id: "orders".into(),
                        },
                        error: CalcFlowError::ExternalProvider {
                            provider: "python".into(),
                            name: "source".into(),
                            version: "1".into(),
                            message: SECRET.into(),
                        },
                    })],
                    Some(23),
                )
                .remove(0),
                "task_panicked" => project_runtime_failures(
                    17,
                    vec![Arc::new(RuntimeFailure {
                        origin: FailureOrigin::Task {
                            task_id: TaskId::new(91),
                            task_name: "source:orders:private-task-name".into(),
                        },
                        error: CalcFlowError::TaskPanicked {
                            task_id: 91,
                            message: PANIC.into(),
                        },
                    })],
                    Some(23),
                )
                .remove(0),
                other => panic!("unknown shared projection vector {other:?}"),
            };
            let actual = serde_json::to_value(error).unwrap();
            assert_eq!(actual, vector["expected"], "shared vector {case:?}");
            let rendered = actual.to_string();
            for sentinel in vector["private_sentinels"].as_array().unwrap() {
                assert!(!rendered.contains(sentinel.as_str().unwrap()));
            }
        }
    }
}
