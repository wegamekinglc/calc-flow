use std::{collections::BTreeMap, sync::Arc, time::Duration};

use crate::EventTime;
use parking_lot::Mutex;

use super::{
    aggregate::IngressActivity,
    driver::DriverPhase,
    prepare::{BindingIdentity, BindingOrdinal},
    trace::{
        AdmissionDecisionRecord, AdmissionGateState, ProgressTraceRecord, SettlementDisposition,
    },
    types::LogicalInstant,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProgressCounters {
    pub(crate) admission_attempts: u64,
    pub(crate) accepted_envelopes: u64,
    pub(crate) immediate_rejections: u64,
    pub(crate) drain_epochs: u64,
    pub(crate) inbox_fences: u64,
    pub(crate) due_timers: u64,
    pub(crate) terminal_transitions: u64,
    pub(crate) gate_transitions: u64,
    pub(crate) settlement_attempts: u64,
    pub(crate) commit_success_settlements: u64,
    pub(crate) transaction_error_settlements: u64,
    pub(crate) post_end_tail_settlements: u64,
    pub(crate) cancelled_settlements: u64,
    pub(crate) fatal_settlements: u64,
    pub(crate) driver_phase_failures: u64,
    pub(crate) progress_emissions: u64,
    pub(crate) timer_entries: u64,
    pub(crate) trace_records: u64,
    pub(crate) maximum_inbox_fences_per_drain: u64,
    pub(crate) maximum_selected_items_per_drain: u64,
    pub(crate) maximum_due_timers_per_drain: u64,
}

impl ProgressCounters {
    pub(super) fn observe_record(&mut self, record: &ProgressTraceRecord) {
        self.trace_records = self.trace_records.saturating_add(1);
        match record {
            ProgressTraceRecord::Admission(record) => {
                self.admission_attempts = self.admission_attempts.saturating_add(1);
                match record.decision {
                    AdmissionDecisionRecord::Accepted { .. } => {
                        self.accepted_envelopes = self.accepted_envelopes.saturating_add(1);
                    }
                    AdmissionDecisionRecord::ImmediateRejected { .. } => {
                        self.immediate_rejections = self.immediate_rejections.saturating_add(1);
                    }
                }
            }
            ProgressTraceRecord::Drain(record) => {
                self.drain_epochs = self.drain_epochs.saturating_add(1);
                let inbox_fences = u64::try_from(record.inbox_fences.len()).unwrap_or(u64::MAX);
                let selected_items =
                    u64::try_from(record.selected_items_in_ready_order.len()).unwrap_or(u64::MAX);
                let due_timers =
                    u64::try_from(record.due_timers_in_ready_order.len()).unwrap_or(u64::MAX);
                self.inbox_fences = self.inbox_fences.saturating_add(inbox_fences);
                self.due_timers = self.due_timers.saturating_add(due_timers);
                self.maximum_inbox_fences_per_drain =
                    self.maximum_inbox_fences_per_drain.max(inbox_fences);
                self.maximum_selected_items_per_drain =
                    self.maximum_selected_items_per_drain.max(selected_items);
                self.maximum_due_timers_per_drain =
                    self.maximum_due_timers_per_drain.max(due_timers);
            }
            ProgressTraceRecord::Terminal(record) => {
                self.terminal_transitions = self.terminal_transitions.saturating_add(1);
                self.gate_transitions = self.gate_transitions.saturating_add(
                    u64::try_from(record.transitions_in_binding_order.len()).unwrap_or(u64::MAX),
                );
            }
            ProgressTraceRecord::Settlement(record) => {
                self.settlement_attempts = self.settlement_attempts.saturating_add(1);
                match record.disposition {
                    SettlementDisposition::CommitSuccess => {
                        self.commit_success_settlements =
                            self.commit_success_settlements.saturating_add(1);
                    }
                    SettlementDisposition::TransactionError { .. } => {
                        self.transaction_error_settlements =
                            self.transaction_error_settlements.saturating_add(1);
                    }
                    SettlementDisposition::PostEndTailReject => {
                        self.post_end_tail_settlements =
                            self.post_end_tail_settlements.saturating_add(1);
                    }
                    SettlementDisposition::Cancelled => {
                        self.cancelled_settlements = self.cancelled_settlements.saturating_add(1);
                    }
                    SettlementDisposition::Fatal => {
                        self.fatal_settlements = self.fatal_settlements.saturating_add(1);
                    }
                }
            }
            ProgressTraceRecord::DriverPhaseFailure { .. } => {
                self.driver_phase_failures = self.driver_phase_failures.saturating_add(1);
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BindingProgressStatus {
    pub(crate) identity: BindingIdentity,
    pub(crate) ordinal: BindingOrdinal,
    pub(crate) activity: IngressActivity,
    pub(crate) last_source_watermark: Option<EventTime>,
    pub(crate) generated_max_nanos: Option<i128>,
    pub(crate) gate_state: AdmissionGateState,
    pub(crate) gate_generation: u64,
    pub(crate) queued_envelopes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StreamProgressStatus {
    pub(crate) phase: DriverPhase,
    pub(crate) logical_instant: LogicalInstant,
    pub(crate) aggregate_watermark: Option<EventTime>,
    pub(crate) idle_latched: bool,
    pub(crate) bindings: BTreeMap<BindingIdentity, BindingProgressStatus>,
    pub(crate) counters: ProgressCounters,
    pub(crate) unsettled_receipts: usize,
    pub(crate) next_central_wake: Option<LogicalInstant>,
    pub(crate) terminal_gate_cuts:
        BTreeMap<BindingIdentity, super::trace::AdmissionGateCloseCoordinate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LiveProgressEvidence {
    pub(crate) current: StreamProgressStatus,
    pub(crate) maximum_unsettled_receipts: usize,
    pub(crate) maximum_timer_entries: u64,
    pub(crate) maximum_trace_records: u64,
    pub(crate) maximum_settlement_latency_micros: u128,
}

#[derive(Clone)]
pub(crate) struct LiveProgressStatusHandle(Arc<Mutex<LiveProgressEvidence>>);

impl LiveProgressStatusHandle {
    pub(crate) fn new(current: StreamProgressStatus) -> Self {
        let maximum_unsettled_receipts = current.unsettled_receipts;
        let maximum_timer_entries = current.counters.timer_entries;
        let maximum_trace_records = current.counters.trace_records;
        Self(Arc::new(Mutex::new(LiveProgressEvidence {
            current,
            maximum_unsettled_receipts,
            maximum_timer_entries,
            maximum_trace_records,
            maximum_settlement_latency_micros: 0,
        })))
    }

    pub(crate) fn publish(&self, current: StreamProgressStatus) {
        let mut evidence = self.0.lock();
        evidence.maximum_unsettled_receipts = evidence
            .maximum_unsettled_receipts
            .max(current.unsettled_receipts);
        evidence.maximum_timer_entries = evidence
            .maximum_timer_entries
            .max(current.counters.timer_entries);
        evidence.maximum_trace_records = evidence
            .maximum_trace_records
            .max(current.counters.trace_records);
        evidence.current = current;
    }

    pub(crate) fn observe_settlement_latency(&self, latency: Duration) {
        let mut evidence = self.0.lock();
        evidence.maximum_settlement_latency_micros = evidence
            .maximum_settlement_latency_micros
            .max(latency.as_micros());
    }

    pub(crate) fn snapshot(&self) -> LiveProgressEvidence {
        self.0.lock().clone()
    }
}
