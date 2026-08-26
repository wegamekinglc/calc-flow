use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::{
    Batch, CalcFlowError, EdgeBudget, Epoch, EventTime, JsonMap, Port, Result, StreamJobContext,
    StreamMessage,
};

use super::OperatorMetadata;

/// One immutable checkpoint state segment with its content digest.
///
/// Segments are shared by allocation: an operator that carries an unchanged
/// segment across epochs clones the cheap `Arc` instead of copying bytes, and
/// the SHA-256 is computed once at construction so re-staging a carried
/// segment never re-hashes the retained state (spec FR47 capture cost stays
/// proportional to the dirty set).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateSegment {
    bytes: Arc<Vec<u8>>,
    sha256: String,
}

impl StateSegment {
    /// Wraps segment bytes, computing their SHA-256 exactly once.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        let sha256 = hex::encode(Sha256::digest(&bytes));
        Self {
            bytes: Arc::new(bytes),
            sha256,
        }
    }

    /// Wraps bytes whose digest was already validated against a committed
    /// handle, so recovery never re-hashes what it just verified.
    pub(crate) fn from_validated(bytes: Vec<u8>, sha256: String) -> Self {
        Self {
            bytes: Arc::new(bytes),
            sha256,
        }
    }

    /// Returns the immutable segment bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Shares the segment allocation with another snapshot or carried buffer.
    #[must_use]
    pub fn bytes_arc(&self) -> Arc<Vec<u8>> {
        Arc::clone(&self.bytes)
    }

    /// Returns the lowercase hexadecimal SHA-256 of the segment bytes.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// Operator-private state captured at one epoch (API note A2.3).
///
/// The runtime wraps it with input progress and segment handles into the
/// checkpoint manifest; keyed row state never appears inline (spec D4.4), and
/// no segment may carry secrets (I4).
#[derive(Clone, Debug, Default)]
pub struct OperatorStateSnapshot {
    /// Small bounded JSON placed inline in the manifest.
    pub inline_metadata: JsonMap,
    /// `segment_id -> shared segment`; the runtime assigns paths, lengths, and
    /// checksums during staging (D4.1).
    pub segments: BTreeMap<String, StateSegment>,
}

/// One named ingress's current runtime-owned progress state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IngressState {
    Active,
    Idle,
    Ended,
}

/// Immutable event-time progress observed for one named ingress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IngressProgress {
    state: IngressState,
    watermark: Option<EventTime>,
}

impl IngressProgress {
    pub const fn new(state: IngressState, watermark: Option<EventTime>) -> Self {
        Self { state, watermark }
    }

    /// Returns whether the ingress is active, idle, or ended.
    pub const fn state(self) -> IngressState {
        self.state
    }

    /// Returns the last accepted watermark, if one has been established.
    pub const fn watermark(self) -> Option<EventTime> {
        self.watermark
    }
}

/// Shared immutable progress for every named ingress of one operator.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IngressProgressSnapshot {
    by_ingress: Arc<BTreeMap<String, IngressProgress>>,
}

impl IngressProgressSnapshot {
    pub fn new(by_ingress: BTreeMap<String, IngressProgress>) -> Self {
        Self {
            by_ingress: Arc::new(by_ingress),
        }
    }

    /// Returns progress for `ingress`, or `None` when it is unknown.
    pub fn get(&self, ingress: &str) -> Option<IngressProgress> {
        self.by_ingress.get(ingress).copied()
    }

    /// Returns all progress in deterministic ingress-name order.
    pub fn by_ingress(&self) -> &BTreeMap<String, IngressProgress> {
        &self.by_ingress
    }

    /// Returns the number of observed ingresses.
    pub fn len(&self) -> usize {
        self.by_ingress.len()
    }

    /// Returns whether the snapshot contains no ingress.
    pub fn is_empty(&self) -> bool {
        self.by_ingress.is_empty()
    }
}

/// The only way a stream operator emits data (API note A2).
///
/// Control messages can never be emitted through this trait (spec S1.3):
/// watermark, barrier, idle, and end-of-input forwarding is runtime-owned.
#[async_trait]
pub trait StreamCollector: Send {
    /// Validates the port name, batch kind, and optional exact schema, then
    /// enqueues the batch onto the port's edge (S10.1), awaiting capacity.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Compile`] before enqueue when the port is
    /// unknown, the kind mismatches, or the exact schema mismatches.
    async fn emit(&mut self, port: &str, batch: Batch) -> Result<()>;
}

/// The execution context the operator task hands to one stream operator.
pub struct StreamOperatorContext<'a> {
    job: &'a StreamJobContext,
    operator_id: &'a str,
    input_watermark: Option<EventTime>,
    ingress_progress: IngressProgressSnapshot,
    output_budget: EdgeBudget,
    late_metrics: Arc<dyn LateMetricSink>,
}

impl<'a> StreamOperatorContext<'a> {
    /// Creates a context for one operator within a job.
    pub fn new(
        job: &'a StreamJobContext,
        operator_id: &'a str,
        input_watermark: Option<EventTime>,
    ) -> Self {
        Self {
            job,
            operator_id,
            input_watermark,
            ingress_progress: IngressProgressSnapshot::default(),
            output_budget: EdgeBudget::default(),
            late_metrics: Arc::new(LateMetricRecorder::default()),
        }
    }

    /// Creates a context carrying an explicit ingress-progress snapshot.
    pub fn with_ingress_progress(
        job: &'a StreamJobContext,
        operator_id: &'a str,
        input_watermark: Option<EventTime>,
        ingress_progress: IngressProgressSnapshot,
    ) -> Self {
        Self {
            job,
            operator_id,
            input_watermark,
            ingress_progress,
            output_budget: EdgeBudget::default(),
            late_metrics: Arc::new(LateMetricRecorder::default()),
        }
    }

    pub(crate) fn for_task(
        job: &'a StreamJobContext,
        operator_id: &'a str,
        input_watermark: Option<EventTime>,
        ingress_progress: IngressProgressSnapshot,
        output_budget: EdgeBudget,
        late_metrics: Arc<dyn LateMetricSink>,
    ) -> Self {
        Self {
            job,
            operator_id,
            input_watermark,
            ingress_progress,
            output_budget,
            late_metrics,
        }
    }

    /// Returns the owning job's immutable context.
    pub const fn job(&self) -> &StreamJobContext {
        self.job
    }

    /// Returns the operator's node identity.
    pub const fn operator_id(&self) -> &str {
        self.operator_id
    }

    /// Returns the current input watermark `WM_in`; `None` while undefined
    /// (spec S5.2).
    pub const fn input_watermark(&self) -> Option<EventTime> {
        self.input_watermark
    }

    /// Returns the complete immutable per-ingress progress snapshot.
    pub const fn ingress_progress(&self) -> &IngressProgressSnapshot {
        &self.ingress_progress
    }

    pub(crate) const fn output_budget(&self) -> EdgeBudget {
        self.output_budget
    }

    /// Verifies that the owning job remains active.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Cancelled`] when the job was cancelled or its
    /// deadline passed.
    pub fn check_cancelled(&self) -> Result<()> {
        self.job.check_cancelled()
    }

    /// Records dropped row-window assignments for the batch currently being
    /// processed (spec D2.5); row payloads are never accepted.
    ///
    /// The runtime derives `affected_batches` from any call with `dropped > 0`
    /// and keeps a running maximum lateness. Only window operators call this.
    /// A call with `dropped == 0` is a complete telemetry no-op: the supplied
    /// `max_lateness` is neither recorded nor validated.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Internal`] for a nonzero sample when the
    /// supplied maximum lateness cannot be represented in `u64` microseconds,
    /// or when accumulating the dropped-assignment or affected-batch counter
    /// would overflow `u64`.
    pub fn record_late_rows(&self, dropped: u64, max_lateness: Option<Duration>) -> Result<()> {
        if dropped == 0 {
            return Ok(());
        }
        let max_lateness_micros = max_lateness
            .map(|lateness| {
                u64::try_from(lateness.as_micros()).map_err(|_| CalcFlowError::Internal {
                    message: "maximum lateness exceeds the UInt64 microsecond range".into(),
                })
            })
            .transpose()?;
        self.late_metrics.record(LateMetricDelta {
            late_rows: dropped,
            affected_batches: u64::from(dropped > 0),
            max_lateness_micros,
            ..LateMetricDelta::default()
        })
    }

    pub(crate) fn record_window_metrics(
        &self,
        late_rows: u64,
        max_lateness_micros: Option<u64>,
        null_event_time_rows: u64,
    ) -> Result<()> {
        self.late_metrics.record(LateMetricDelta {
            late_rows,
            affected_batches: u64::from(late_rows > 0),
            max_lateness_micros,
            null_event_time_rows,
            null_event_time_batches: u64::from(null_event_time_rows > 0),
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LateMetricDelta {
    pub(crate) late_rows: u64,
    pub(crate) affected_batches: u64,
    pub(crate) max_lateness_micros: Option<u64>,
    pub(crate) null_event_time_rows: u64,
    pub(crate) null_event_time_batches: u64,
}

pub(crate) trait LateMetricSink: Send + Sync {
    fn record(&self, delta: LateMetricDelta) -> Result<()>;
}

#[derive(Default)]
struct LateMetricRecorder(Mutex<LateMetricDelta>);

impl LateMetricSink for LateMetricRecorder {
    fn record(&self, delta: LateMetricDelta) -> Result<()> {
        let mut current = self.0.lock();
        let next = accumulate_late_metrics(*current, delta)?;
        *current = next;
        Ok(())
    }
}

pub(crate) fn accumulate_late_metrics(
    current: LateMetricDelta,
    delta: LateMetricDelta,
) -> Result<LateMetricDelta> {
    Ok(LateMetricDelta {
        late_rows: checked_metric_sum(current.late_rows, delta.late_rows, "late_rows")?,
        affected_batches: checked_metric_sum(
            current.affected_batches,
            delta.affected_batches,
            "affected_batches",
        )?,
        max_lateness_micros: match (current.max_lateness_micros, delta.max_lateness_micros) {
            (Some(current), Some(delta)) => Some(current.max(delta)),
            (current, delta) => current.or(delta),
        },
        null_event_time_rows: checked_metric_sum(
            current.null_event_time_rows,
            delta.null_event_time_rows,
            "null_event_time_rows",
        )?,
        null_event_time_batches: checked_metric_sum(
            current.null_event_time_batches,
            delta.null_event_time_batches,
            "null_event_time_batches",
        )?,
    })
}

fn checked_metric_sum(left: u64, right: u64, field: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("window metric {field} overflowed UInt64"),
        })
}

/// A continuously running operator: it receives one named-ingress batch per
/// call and emits only through the [`StreamCollector`] (plan section 2.2).
///
/// The five metadata accessors live on the [`OperatorMetadata`] supertrait so
/// the batch and stream compilers share port, schema, and UDF validation
/// (API note A1). Control forwarding is runtime-owned (S1.3); handlers never
/// see barriers, and their watermarks arrive as typed `EventTime` values.
#[async_trait]
pub trait StreamOperator: OperatorMetadata {
    /// Observes one accepted ingress progress transition before runtime-owned
    /// control forwarding. The default keeps existing operators unchanged.
    ///
    /// # Errors
    ///
    /// Stateful operators may reject inconsistent progress before the
    /// corresponding control is forwarded.
    async fn on_ingress_progress(
        &mut self,
        ingress: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let _ = (ingress, context);
        Ok(())
    }

    /// Processes one batch from the named ingress.
    ///
    /// # Errors
    ///
    /// Returns an error when input validation, cancellation, or calculation
    /// fails; a failed handler never forwards a partial control event (S1.3).
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()>;

    /// Reacts to an input-watermark advance (S5.2); window operators emit
    /// newly closed windows before the runtime forwards the watermark.
    ///
    /// # Errors
    ///
    /// Returns an error when emitting closed results fails.
    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()>;

    /// Flushes once after every ingress has ended (S1.6, S5.5).
    ///
    /// # Errors
    ///
    /// Returns an error when the final flush fails.
    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()>;

    /// Synchronously captures dirty state for `epoch` (API note A2.2).
    ///
    /// The capture is O(dirty-key metadata), never a bulk encode on the
    /// executor thread; durable staging is runtime-owned (D4.1).
    ///
    /// # Errors
    ///
    /// Stateful implementations may reject state that cannot be captured.
    fn checkpoint(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        let _ = epoch;
        Ok(OperatorStateSnapshot::default())
    }

    /// Restores state captured by [`StreamOperator::checkpoint`].
    ///
    /// # Errors
    ///
    /// The default stateless lifecycle rejects a non-empty snapshot.
    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        if snapshot.inline_metadata.is_empty() && snapshot.segments.is_empty() {
            Ok(())
        } else {
            Err(CalcFlowError::Format {
                message: "stateless operator state must be empty".into(),
            })
        }
    }

    /// Resets the operator to its freshly constructed state (API note A3).
    ///
    /// # Errors
    ///
    /// Stateful implementations may fail while releasing owned state.
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A public in-memory validating [`StreamCollector`] helper (API note A2.1).
///
/// `emit` validates the port name, kind, and exact schema before storing the
/// batch in a per-port FIFO outbox. Tests, benchmarks, and direct operator
/// callers can inspect those messages with [`Self::drain`]. The crate-private
/// M2 operator task instead uses `ChannelStreamCollector` to deliver emissions
/// directly through bounded runtime edges.
pub struct EdgeCollector {
    outboxes: BTreeMap<String, (Port, Vec<StreamMessage>)>,
}

impl EdgeCollector {
    /// Creates a collector over the compiled output ports of one operator.
    pub fn new(output_ports: Vec<Port>) -> Self {
        Self {
            outboxes: output_ports
                .into_iter()
                .map(|port| (port.name().to_owned(), (port, Vec::new())))
                .collect(),
        }
    }

    /// Drains one port's outbox in FIFO order; an unknown port drains to an
    /// empty vector.
    pub fn drain(&mut self, port: &str) -> Vec<StreamMessage> {
        self.outboxes
            .get_mut(port)
            .map(|(_, messages)| std::mem::take(messages))
            .unwrap_or_default()
    }
}

#[async_trait]
impl StreamCollector for EdgeCollector {
    async fn emit(&mut self, port: &str, batch: Batch) -> Result<()> {
        let (declared, messages) =
            self.outboxes
                .get_mut(port)
                .ok_or_else(|| CalcFlowError::Compile {
                    message: format!("unknown output port {port:?}"),
                })?;
        declared.validate(&batch, &format!("output {port}"))?;
        messages.push(StreamMessage::data(batch));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CancellationToken;

    #[test]
    fn late_row_recorder_accumulates_drops_batches_and_max_lateness() {
        let recorder = LateMetricRecorder::default();
        recorder.record(LateMetricDelta::default()).unwrap();
        recorder
            .record(LateMetricDelta {
                late_rows: 2,
                affected_batches: 1,
                max_lateness_micros: Some(5),
                ..LateMetricDelta::default()
            })
            .unwrap();
        recorder
            .record(LateMetricDelta {
                late_rows: 3,
                affected_batches: 1,
                max_lateness_micros: Some(7),
                null_event_time_rows: 4,
                null_event_time_batches: 1,
            })
            .unwrap();
        assert_eq!(
            *recorder.0.lock(),
            LateMetricDelta {
                late_rows: 5,
                affected_batches: 2,
                max_lateness_micros: Some(7),
                null_event_time_rows: 4,
                null_event_time_batches: 1,
            }
        );
    }

    #[test]
    fn context_records_late_rows_through_the_shared_recorder() {
        let job = StreamJobContext::new(
            7,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let recorder = Arc::new(LateMetricRecorder::default());
        let context = StreamOperatorContext::for_task(
            &job,
            "window",
            None,
            IngressProgressSnapshot::default(),
            EdgeBudget::default(),
            recorder.clone(),
        );
        context
            .record_late_rows(4, Some(Duration::from_micros(11)))
            .unwrap();
        assert_eq!(
            *recorder.0.lock(),
            LateMetricDelta {
                late_rows: 4,
                affected_batches: 1,
                max_lateness_micros: Some(11),
                ..LateMetricDelta::default()
            }
        );
    }

    #[test]
    fn zero_dropped_rows_do_not_change_late_metrics() {
        let job = StreamJobContext::new(
            7,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let recorder = Arc::new(LateMetricRecorder::default());
        let context = StreamOperatorContext::for_task(
            &job,
            "window",
            None,
            IngressProgressSnapshot::default(),
            EdgeBudget::default(),
            recorder.clone(),
        );
        context
            .record_late_rows(2, Some(Duration::from_micros(5)))
            .unwrap();

        assert!(
            context
                .record_late_rows(0, Some(Duration::from_secs(u64::MAX)))
                .is_ok()
        );
        assert_eq!(
            *recorder.0.lock(),
            LateMetricDelta {
                late_rows: 2,
                affected_batches: 1,
                max_lateness_micros: Some(5),
                ..LateMetricDelta::default()
            }
        );

        context
            .record_late_rows(0, Some(Duration::from_micros(99)))
            .unwrap();

        assert_eq!(
            *recorder.0.lock(),
            LateMetricDelta {
                late_rows: 2,
                affected_batches: 1,
                max_lateness_micros: Some(5),
                ..LateMetricDelta::default()
            }
        );
    }

    #[test]
    fn metric_overflow_is_transactional() {
        let current = LateMetricDelta {
            late_rows: u64::MAX,
            affected_batches: 2,
            max_lateness_micros: Some(4),
            null_event_time_rows: 3,
            null_event_time_batches: 1,
        };
        assert!(
            accumulate_late_metrics(
                current,
                LateMetricDelta {
                    late_rows: 1,
                    ..LateMetricDelta::default()
                }
            )
            .is_err()
        );
        assert_eq!(current.late_rows, u64::MAX);
    }
}
