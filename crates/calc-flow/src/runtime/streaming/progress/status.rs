use std::{collections::BTreeMap, sync::Arc, time::Duration};

use crate::EventTime;
use parking_lot::Mutex;

use super::{
    aggregate::IngressActivity,
    driver::DriverPhase,
    prepare::{BindingIdentity, BindingOrdinal},
    trace::AdmissionGateState,
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
