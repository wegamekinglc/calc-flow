use std::collections::{BTreeMap, BTreeSet};

use crate::{CalcFlowError, Result};

use super::{
    prepare::{BindingIdentity, BindingOrdinal},
    types::{
        AdmissionAttemptOrdinal, AdmissionGateGeneration, DrainEpoch, DriverClockCoordinate,
        DriverPhaseError, FenceSequence, GateCloseOrdinal, InboxSequence, ReadyKey,
        ReceiptSequence, TimerIdentity, TracePosition, TraceRecordOrdinal,
    },
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum RawUpstreamPosition {
    Exact {
        delivery_replay_cursor: Vec<u8>,
        control_frontier: Vec<u8>,
    },
    Unavailable,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct AcceptedEnvelopeIdentity {
    pub(crate) binding: BindingIdentity,
    pub(crate) binding_ordinal: BindingOrdinal,
    pub(crate) admission_attempt: AdmissionAttemptOrdinal,
    pub(crate) receipt_sequence: ReceiptSequence,
    pub(crate) inbox_sequence: InboxSequence,
    pub(crate) upstream_position: RawUpstreamPosition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InboxUpperFence {
    Empty,
    Inclusive(InboxSequence),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InboxFenceCoordinate {
    pub(crate) drain_epoch: DrainEpoch,
    pub(crate) binding_ordinal: BindingOrdinal,
    pub(crate) fence_sequence: FenceSequence,
    pub(crate) upper: InboxUpperFence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RawIngressEventKind {
    Data,
    ConnectorWatermark,
    ConnectorIdle,
    EndOfInput,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReadyItemIdentity {
    Raw {
        accepted: AcceptedEnvelopeIdentity,
        event_kind: RawIngressEventKind,
    },
    Timer(TimerIdentity),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadyKeyRange {
    Empty,
    Inclusive { first: ReadyKey, last: ReadyKey },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionGateState {
    Open,
    ClosedAfterEnd,
    ClosedCancelled,
    ClosedFatal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionGateCloseCoordinate {
    pub(crate) close_ordinal: GateCloseOrdinal,
    pub(crate) cause: TerminalTransitionCause,
    pub(crate) old_generation: AdmissionGateGeneration,
    pub(crate) new_generation: AdmissionGateGeneration,
    pub(crate) closed_state: AdmissionGateState,
    pub(crate) next_inbox_sequence_cut: InboxSequence,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionGateSnapshot {
    pub(crate) state: AdmissionGateState,
    pub(crate) generation: AdmissionGateGeneration,
    pub(crate) close: Option<AdmissionGateCloseCoordinate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionDecisionRecord {
    Accepted { accepted: AcceptedEnvelopeIdentity },
    ImmediateRejected { path: String, reason: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionAttemptRecord {
    pub(crate) trace_record_ordinal: TraceRecordOrdinal,
    pub(crate) trace_position: TracePosition,
    pub(crate) attempt_ordinal: AdmissionAttemptOrdinal,
    pub(crate) binding: BindingIdentity,
    pub(crate) binding_ordinal: BindingOrdinal,
    pub(crate) event_kind: RawIngressEventKind,
    pub(crate) upstream_position: RawUpstreamPosition,
    pub(crate) observed_gate: AdmissionGateSnapshot,
    pub(crate) decision: AdmissionDecisionRecord,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DrainEpochOutcomeRecord {
    Committed,
    SelectedItemFailed {
        first_error_key: ReadyKey,
        path: String,
        reason: String,
    },
    DriverPhaseFailed(DriverPhaseError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DrainEpochRecord {
    pub(crate) trace_record_ordinal: TraceRecordOrdinal,
    pub(crate) trace_position: TracePosition,
    pub(crate) epoch: DrainEpoch,
    pub(crate) driver_clock: DriverClockCoordinate,
    pub(crate) inbox_fences: Vec<InboxFenceCoordinate>,
    pub(crate) selected_items_in_ready_order: Vec<ReadyItemIdentity>,
    pub(crate) selected_key_range: ReadyKeyRange,
    pub(crate) due_timers_in_ready_order: Vec<TimerIdentity>,
    pub(crate) outcome: DrainEpochOutcomeRecord,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum TerminalTransitionCause {
    EndCommit,
    Cancellation,
    Fatal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BindingGateTransitionRecord {
    pub(crate) binding: BindingIdentity,
    pub(crate) binding_ordinal: BindingOrdinal,
    pub(crate) close: AdmissionGateCloseCoordinate,
    pub(crate) extracted_tail: Vec<AcceptedEnvelopeIdentity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TerminalTransitionRecord {
    pub(crate) trace_record_ordinal: TraceRecordOrdinal,
    pub(crate) trace_position: TracePosition,
    pub(crate) cause: TerminalTransitionCause,
    pub(crate) driver_clock: DriverClockCoordinate,
    pub(crate) owning_drain: Option<DrainEpoch>,
    pub(crate) transitions_in_binding_order: Vec<BindingGateTransitionRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SettlementDisposition {
    CommitSuccess,
    TransactionError { path: String, reason: String },
    PostEndTailReject,
    Cancelled,
    Fatal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SettlementOwner {
    Drain(DrainEpoch),
    Terminal {
        cause: TerminalTransitionCause,
        owning_drain: Option<DrainEpoch>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SettlementRecord {
    pub(crate) trace_record_ordinal: TraceRecordOrdinal,
    pub(crate) trace_position: TracePosition,
    pub(crate) accepted: AcceptedEnvelopeIdentity,
    pub(crate) owner: SettlementOwner,
    pub(crate) disposition: SettlementDisposition,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProgressTraceRecord {
    Admission(AdmissionAttemptRecord),
    Drain(DrainEpochRecord),
    Terminal(TerminalTransitionRecord),
    Settlement(SettlementRecord),
    DriverPhaseFailure {
        trace_record_ordinal: TraceRecordOrdinal,
        trace_position: TracePosition,
        error: DriverPhaseError,
    },
}

impl ProgressTraceRecord {
    pub(crate) const fn ordinal(&self) -> TraceRecordOrdinal {
        match self {
            Self::Admission(record) => record.trace_record_ordinal,
            Self::Drain(record) => record.trace_record_ordinal,
            Self::Terminal(record) => record.trace_record_ordinal,
            Self::Settlement(record) => record.trace_record_ordinal,
            Self::DriverPhaseFailure {
                trace_record_ordinal,
                ..
            } => *trace_record_ordinal,
        }
    }

    pub(crate) const fn position(&self) -> TracePosition {
        match self {
            Self::Admission(record) => record.trace_position,
            Self::Drain(record) => record.trace_position,
            Self::Terminal(record) => record.trace_position,
            Self::Settlement(record) => record.trace_position,
            Self::DriverPhaseFailure { trace_position, .. } => *trace_position,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProgressExecutionTrace {
    pub(crate) records: Vec<ProgressTraceRecord>,
}

impl ProgressExecutionTrace {
    pub(crate) fn drain_projection(&self) -> Vec<DrainEpochRecord> {
        self.records
            .iter()
            .filter_map(|record| match record {
                ProgressTraceRecord::Drain(record) => Some(record.clone()),
                _ => None,
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
enum TraceMode {
    Record,
    Replay {
        expected: ProgressExecutionTrace,
        cursor: usize,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct TraceController {
    completed: ProgressExecutionTrace,
    mode: TraceMode,
    next_record: u64,
    next_position: u64,
}

impl TraceController {
    pub(crate) fn record() -> Self {
        Self {
            completed: ProgressExecutionTrace::default(),
            mode: TraceMode::Record,
            next_record: 0,
            next_position: 0,
        }
    }

    pub(crate) fn replay(request: ProgressReplayRequest) -> Self {
        Self {
            completed: ProgressExecutionTrace::default(),
            mode: TraceMode::Replay {
                expected: request.expected,
                cursor: 0,
            },
            next_record: 0,
            next_position: 0,
        }
    }

    pub(crate) fn replay_from_prefix(
        prefix: ProgressExecutionTrace,
        next_record: u64,
        next_position: u64,
        request: ProgressReplayRequest,
    ) -> Result<Self> {
        let prefix_len = prefix.records.len();
        if request.expected.records.get(..prefix_len) != Some(prefix.records.as_slice()) {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.progress.replay.trace_prefix".into(),
                message: "replay plan does not begin with the captured execution trace prefix"
                    .into(),
            });
        }
        let restored = Self::restore_prefix(prefix, next_record, next_position)?;
        Ok(Self {
            completed: restored.completed,
            mode: TraceMode::Replay {
                expected: request.expected,
                cursor: prefix_len,
            },
            next_record,
            next_position,
        })
    }

    pub(crate) fn next_coordinates(&self) -> Result<(TraceRecordOrdinal, TracePosition)> {
        if self.next_record == u64::MAX || self.next_position == u64::MAX {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.progress.counters.trace_record".into(),
                message: "trace record coordinate exhausted".into(),
            });
        }
        Ok((
            TraceRecordOrdinal(self.next_record),
            TracePosition(self.next_position),
        ))
    }

    pub(crate) fn append(&mut self, record: ProgressTraceRecord) -> Result<()> {
        let (ordinal, position) = self.next_coordinates()?;
        if record.ordinal() != ordinal || record.position() != position {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.progress.trace.coordinate".into(),
                message: "record does not use the next exact trace coordinate".into(),
            });
        }
        if let TraceMode::Replay { expected, cursor } = &mut self.mode {
            let expected_record =
                expected
                    .records
                    .get(*cursor)
                    .ok_or_else(|| CalcFlowError::InvalidArgument {
                        field: "runtime.progress.replay.trace".into(),
                        message: "execution produced an extra trace record".into(),
                    })?;
            if expected_record != &record {
                return Err(CalcFlowError::InvalidArgument {
                    field: "runtime.progress.replay.trace".into(),
                    message: format!(
                        "execution trace record at position {} does not match the replay plan",
                        position.0
                    ),
                });
            }
            *cursor += 1;
        }
        self.next_record += 1;
        self.next_position += 1;
        self.completed.records.push(record);
        Ok(())
    }

    pub(crate) fn finish_replay(&self) -> Result<()> {
        if let TraceMode::Replay { expected, cursor } = &self.mode
            && *cursor != expected.records.len()
        {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.progress.replay.trace".into(),
                message: "expected execution trace has an unconsumed suffix".into(),
            });
        }
        Ok(())
    }

    pub(crate) fn completed(&self) -> &ProgressExecutionTrace {
        &self.completed
    }

    pub(crate) const fn next_record(&self) -> u64 {
        self.next_record
    }

    pub(crate) const fn next_position(&self) -> u64 {
        self.next_position
    }

    #[cfg(test)]
    pub(crate) fn set_next_coordinates_for_test(&mut self, next: u64) {
        self.next_record = next;
        self.next_position = next;
    }

    pub(crate) fn restore_prefix(
        trace: ProgressExecutionTrace,
        next_record: u64,
        next_position: u64,
    ) -> Result<Self> {
        if trace.records.len() != usize::try_from(next_position).unwrap_or(usize::MAX)
            || next_record != next_position
        {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.progress.snapshot.coordinate.progress_execution_trace".into(),
                message: "trace prefix and next coordinates are not exact".into(),
            });
        }
        Ok(Self {
            completed: trace,
            mode: TraceMode::Record,
            next_record,
            next_position,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ProgressReplayRequest {
    pub(super) expected: ProgressExecutionTrace,
}

#[derive(Default)]
struct ReplayValidationState {
    accepted: BTreeMap<AcceptedEnvelopeIdentity, usize>,
    rejected: BTreeSet<(BindingIdentity, AdmissionAttemptOrdinal)>,
    settlements: BTreeMap<AcceptedEnvelopeIdentity, usize>,
    settlement_owners: BTreeMap<AcceptedEnvelopeIdentity, SettlementOwner>,
    drains: BTreeSet<DrainEpoch>,
    terminals: BTreeSet<(TerminalTransitionCause, Option<DrainEpoch>)>,
    terminal_tails:
        BTreeMap<AcceptedEnvelopeIdentity, (TerminalTransitionCause, Option<DrainEpoch>)>,
    next_attempt: u64,
    next_drain: u64,
    next_gate_close: u64,
}

impl ReplayValidationState {
    fn validate_record(&mut self, index: usize, record: &ProgressTraceRecord) -> Result<()> {
        validate_trace_coordinate(index, record)?;
        match record {
            ProgressTraceRecord::Admission(record) => self.validate_admission(record),
            ProgressTraceRecord::Drain(record) => self.validate_drain(record),
            ProgressTraceRecord::Terminal(record) => self.validate_terminal(record),
            ProgressTraceRecord::Settlement(record) => self.validate_settlement(record),
            ProgressTraceRecord::DriverPhaseFailure { .. } => Ok(()),
        }
    }

    fn validate_admission(&mut self, record: &AdmissionAttemptRecord) -> Result<()> {
        if record.attempt_ordinal.0 != self.next_attempt {
            return Err(replay_plan_error(
                "runtime.progress.replay.trace.admission",
                "admission attempts are missing, repeated, or reordered",
            ));
        }
        self.next_attempt = self.next_attempt.checked_add(1).ok_or_else(|| {
            replay_plan_error(
                "runtime.progress.replay.trace.admission",
                "admission attempt coordinate overflowed",
            )
        })?;
        match &record.decision {
            AdmissionDecisionRecord::Accepted { accepted } => {
                validate_accepted_identity(record, accepted)?;
                *self.accepted.entry(accepted.clone()).or_default() += 1;
            }
            AdmissionDecisionRecord::ImmediateRejected { .. } => {
                self.record_immediate_rejection(record)?;
            }
        }
        Ok(())
    }

    fn record_immediate_rejection(&mut self, record: &AdmissionAttemptRecord) -> Result<()> {
        if self
            .rejected
            .insert((record.binding.clone(), record.attempt_ordinal))
        {
            return Ok(());
        }
        Err(replay_plan_error(
            "runtime.progress.replay.trace.admission",
            "immediate rejection identity is duplicated",
        ))
    }

    fn validate_drain(&mut self, record: &DrainEpochRecord) -> Result<()> {
        if record.epoch.0 != self.next_drain {
            return Err(replay_plan_error(
                "runtime.progress.replay.trace.drain",
                "drain epochs are missing, repeated, or reordered",
            ));
        }
        if !self.drains.insert(record.epoch) {
            return Err(replay_plan_error(
                "runtime.progress.replay.trace.drain",
                "drain epochs are missing, repeated, or reordered",
            ));
        }
        self.next_drain = self.next_drain.checked_add(1).ok_or_else(|| {
            replay_plan_error(
                "runtime.progress.replay.trace.drain",
                "drain epoch coordinate overflowed",
            )
        })?;
        validate_inbox_fences(record)
    }

    fn validate_terminal(&mut self, record: &TerminalTransitionRecord) -> Result<()> {
        if !self.terminals.insert((record.cause, record.owning_drain)) {
            return Err(replay_plan_error(
                "runtime.progress.replay.trace.terminal",
                "terminal transition owner is duplicated",
            ));
        }
        if record
            .owning_drain
            .is_some_and(|epoch| !self.drains.contains(&epoch))
        {
            return Err(replay_plan_error(
                "runtime.progress.replay.trace.terminal",
                "terminal transition references a missing drain or unordered binding",
            ));
        }
        validate_terminal_binding_order(record)?;
        for transition in &record.transitions_in_binding_order {
            self.validate_gate_transition(record, transition)?;
        }
        Ok(())
    }

    fn validate_gate_transition(
        &mut self,
        record: &TerminalTransitionRecord,
        transition: &BindingGateTransitionRecord,
    ) -> Result<()> {
        validate_gate_close(record.cause, transition, self.next_gate_close)?;
        self.next_gate_close = self.next_gate_close.checked_add(1).ok_or_else(|| {
            replay_plan_error(
                "runtime.progress.replay.trace.gate",
                "gate close coordinate overflowed",
            )
        })?;
        for identity in &transition.extracted_tail {
            self.record_terminal_tail(identity, record)?;
        }
        Ok(())
    }

    fn record_terminal_tail(
        &mut self,
        identity: &AcceptedEnvelopeIdentity,
        record: &TerminalTransitionRecord,
    ) -> Result<()> {
        if self
            .terminal_tails
            .insert(identity.clone(), (record.cause, record.owning_drain))
            .is_none()
        {
            return Ok(());
        }
        Err(replay_plan_error(
            "runtime.progress.replay.trace.terminal_tail",
            "accepted identity appears in more than one terminal tail",
        ))
    }

    fn validate_settlement(&mut self, record: &SettlementRecord) -> Result<()> {
        *self.settlements.entry(record.accepted.clone()).or_default() += 1;
        self.settlement_owners
            .entry(record.accepted.clone())
            .or_insert_with(|| record.owner.clone());
        if self.settlement_owner_is_valid(record) {
            return Ok(());
        }
        Err(replay_plan_error(
            "runtime.progress.replay.trace.settlement_owner",
            "settlement disposition names a missing or incompatible owner",
        ))
    }

    fn settlement_owner_is_valid(&self, record: &SettlementRecord) -> bool {
        match (&record.owner, &record.disposition) {
            (
                SettlementOwner::Drain(epoch),
                SettlementDisposition::CommitSuccess
                | SettlementDisposition::TransactionError { .. },
            ) => self.drains.contains(epoch),
            (
                SettlementOwner::Terminal {
                    cause,
                    owning_drain,
                },
                SettlementDisposition::PostEndTailReject,
            ) => self.terminal_owner_exists(
                *cause,
                *owning_drain,
                TerminalTransitionCause::EndCommit,
            ),
            (
                SettlementOwner::Terminal {
                    cause,
                    owning_drain,
                },
                SettlementDisposition::Cancelled,
            ) => self.terminal_owner_exists(
                *cause,
                *owning_drain,
                TerminalTransitionCause::Cancellation,
            ),
            (
                SettlementOwner::Terminal {
                    cause,
                    owning_drain,
                },
                SettlementDisposition::Fatal,
            ) => self.terminal_owner_exists(*cause, *owning_drain, TerminalTransitionCause::Fatal),
            _ => false,
        }
    }

    fn terminal_owner_exists(
        &self,
        cause: TerminalTransitionCause,
        owning_drain: Option<DrainEpoch>,
        expected_cause: TerminalTransitionCause,
    ) -> bool {
        cause == expected_cause && self.terminals.contains(&(cause, owning_drain))
    }

    fn finish(self) -> Result<()> {
        self.validate_settlement_completeness()?;
        for (identity, owner) in &self.terminal_tails {
            if !settlement_matches_terminal_tail(self.settlement_owners.get(identity), *owner) {
                return Err(replay_plan_error(
                    "runtime.progress.replay.trace.terminal_tail",
                    "terminal tail membership does not exactly match settlement ownership",
                ));
            }
        }
        Ok(())
    }

    fn validate_settlement_completeness(&self) -> Result<()> {
        let invalid = self.accepted.values().any(|count| *count != 1)
            || self.settlements.values().any(|count| *count != 1)
            || self.accepted.len() != self.settlements.len()
            || self
                .accepted
                .keys()
                .any(|identity| !self.settlements.contains_key(identity));
        if !invalid {
            return Ok(());
        }
        Err(replay_plan_error(
            "runtime.progress.replay.trace.settlements",
            "every accepted identity must have exactly one settlement",
        ))
    }
}

impl ProgressReplayRequest {
    pub(crate) fn prevalidate(expected: ProgressExecutionTrace) -> Result<Self> {
        let mut validation = ReplayValidationState::default();
        for (index, record) in expected.records.iter().enumerate() {
            validation.validate_record(index, record)?;
        }
        validation.finish()?;
        Ok(Self { expected })
    }
}

fn validate_trace_coordinate(index: usize, record: &ProgressTraceRecord) -> Result<()> {
    let coordinate = u64::try_from(index).map_err(|_| CalcFlowError::InvalidArgument {
        field: "runtime.progress.replay.trace".into(),
        message: "trace is too large".into(),
    })?;
    if record.ordinal().0 == coordinate && record.position().0 == coordinate {
        return Ok(());
    }
    Err(CalcFlowError::InvalidArgument {
        field: "runtime.progress.replay.trace".into(),
        message: "trace coordinates are missing, repeated, or reordered".into(),
    })
}

fn validate_accepted_identity(
    record: &AdmissionAttemptRecord,
    accepted: &AcceptedEnvelopeIdentity,
) -> Result<()> {
    let invalid = record.observed_gate.state != AdmissionGateState::Open
        || accepted.binding != record.binding
        || accepted.binding_ordinal != record.binding_ordinal
        || accepted.admission_attempt != record.attempt_ordinal
        || accepted.upstream_position != record.upstream_position;
    if !invalid {
        return Ok(());
    }
    Err(replay_plan_error(
        "runtime.progress.replay.trace.admission",
        "accepted identity does not exactly match its admission attempt",
    ))
}

fn validate_inbox_fences(record: &DrainEpochRecord) -> Result<()> {
    let ordered = record
        .inbox_fences
        .windows(2)
        .all(|pair| pair[0].binding_ordinal < pair[1].binding_ordinal);
    let same_epoch = record
        .inbox_fences
        .iter()
        .all(|fence| fence.drain_epoch == record.epoch);
    if ordered && same_epoch {
        return Ok(());
    }
    Err(replay_plan_error(
        "runtime.progress.replay.trace.fences",
        "drain fences are not exact binding-ordered members of their epoch",
    ))
}

fn validate_terminal_binding_order(record: &TerminalTransitionRecord) -> Result<()> {
    if record
        .transitions_in_binding_order
        .windows(2)
        .all(|pair| pair[0].binding_ordinal < pair[1].binding_ordinal)
    {
        return Ok(());
    }
    Err(replay_plan_error(
        "runtime.progress.replay.trace.terminal",
        "terminal transition references a missing drain or unordered binding",
    ))
}

fn validate_gate_close(
    terminal_cause: TerminalTransitionCause,
    transition: &BindingGateTransitionRecord,
    next_gate_close: u64,
) -> Result<()> {
    let expected_generation = transition
        .close
        .old_generation
        .0
        .checked_add(1)
        .ok_or_else(|| {
            replay_plan_error(
                "runtime.progress.replay.trace.gate",
                "gate generation overflowed",
            )
        })?;
    let invalid = transition.close.cause != terminal_cause
        || transition.close.close_ordinal.0 != next_gate_close
        || transition.close.new_generation.0 != expected_generation;
    if !invalid {
        return Ok(());
    }
    Err(replay_plan_error(
        "runtime.progress.replay.trace.gate",
        "gate transition coordinates are not exact",
    ))
}

fn settlement_matches_terminal_tail(
    settlement: Option<&SettlementOwner>,
    expected: (TerminalTransitionCause, Option<DrainEpoch>),
) -> bool {
    matches!(
        settlement,
        Some(SettlementOwner::Terminal { cause, owning_drain })
            if *cause == expected.0 && *owning_drain == expected.1
    )
}

fn replay_plan_error(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{ProgressExecutionTrace, ProgressReplayRequest, TraceController};

    #[test]
    fn reserving_capacity_does_not_change_trace_equality() {
        let left = ProgressExecutionTrace::default();
        let mut right = ProgressExecutionTrace::default();
        assert_eq!(left, right);
        right.records.reserve(1);
        assert_eq!(left, right, "capacity is not semantic trace content");
    }

    #[test]
    fn progress_replay_requires_complete_expected_trace_suffix() {
        let request =
            ProgressReplayRequest::prevalidate(ProgressExecutionTrace::default()).unwrap();
        let controller = TraceController::replay(request);
        controller.finish_replay().unwrap();
    }

    #[test]
    fn trace_coordinates_are_checked() {
        let controller = TraceController::record();
        assert_eq!(controller.next_coordinates().unwrap().0.0, 0);
    }
}
