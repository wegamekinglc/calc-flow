use crate::EventTime;
use serde::{Deserialize, Serialize};

/// Payload-free counters and progress for one ASOF ingress.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamAsofJoinSideStatus {
    /// Successfully admitted on-time identities.
    pub accepted_rows: u64,
    /// Rows classified as late by this operator.
    pub late_rows: u64,
    /// On-time duplicate identities rejected during admission.
    pub duplicate_rows: u64,
    /// Most recent accepted watermark.
    pub watermark_micros: Option<EventTime>,
    /// Whether this input is currently idle.
    pub idle: bool,
    /// Whether this input has permanently ended.
    pub ended: bool,
}

/// Version-one ASOF logical counters and charged state gauges.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamAsofJoinStatus {
    /// Left admission and progress.
    pub left: StreamAsofJoinSideStatus,
    /// Right admission and progress.
    pub right: StreamAsofJoinSideStatus,
    /// Left rows waiting for finality.
    pub pending_left_rows: u64,
    /// Right payloads still relevant to a possible answer.
    pub retained_right_rows: u64,
    /// Live identities without retained payloads.
    pub identity_only_rows: u64,
    /// Total live identities across both inputs.
    pub state_rows: u64,
    /// Accounting-version-one persistent byte charge.
    pub state_bytes: u64,
    /// Accepted final left outputs.
    pub emitted_left_rows: u64,
    /// Final left rows with a candidate.
    pub matched_rows: u64,
    /// Final left rows without a candidate.
    pub unmatched_rows: u64,
    /// Right payloads released after becoming irrelevant.
    pub evicted_right_rows: u64,
    /// Failed state admission attempts.
    pub state_limit_failures: u64,
    /// Failed bounded-workspace attempts.
    pub workspace_limit_failures: u64,
    /// Failed output-edge row attempts.
    pub output_limit_failures: u64,
    /// Conservative output frontier derived from both ingress watermarks.
    pub output_watermark_micros: Option<EventTime>,
}
