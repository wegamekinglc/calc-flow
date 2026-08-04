use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;

use crate::{
    Batch, CalcFlowError, Epoch, EventTime, JsonMap, Port, Result, StreamJobContext, StreamMessage,
};

use super::OperatorMetadata;

/// Operator-private state captured at one epoch (API note A2.3).
///
/// The runtime wraps it with input progress and segment handles into the
/// checkpoint manifest; keyed row state never appears inline (spec D4.4), and
/// no segment may carry secrets (I4).
#[derive(Clone, Debug, Default)]
pub struct OperatorStateSnapshot {
    /// Small bounded JSON placed inline in the manifest.
    pub inline_metadata: JsonMap,
    /// `segment_id -> bytes`; the runtime assigns paths, lengths, and
    /// checksums during staging (D4.1).
    pub segments: BTreeMap<String, Vec<u8>>,
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
    late_rows: Arc<LateRowRecorder>,
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
            late_rows: Arc::new(LateRowRecorder::default()),
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
    ///
    /// # Errors
    ///
    /// This first version never fails; the `Result` keeps the frozen
    /// signature stable for the M3/M4 validation rules.
    pub fn record_late_rows(&self, dropped: u64, max_lateness: Option<Duration>) -> Result<()> {
        self.late_rows.record(dropped, max_lateness);
        Ok(())
    }
}

/// Cumulative late-row counters owned by one operator task (spec D2.5).
#[derive(Default)]
pub(crate) struct LateRowRecorder {
    dropped: AtomicU64,
    affected_batches: AtomicU64,
    max_lateness_micros: AtomicU64,
}

impl LateRowRecorder {
    fn record(&self, dropped: u64, max_lateness: Option<Duration>) {
        if dropped == 0 {
            return;
        }
        self.dropped.fetch_add(dropped, Ordering::Relaxed);
        self.affected_batches.fetch_add(1, Ordering::Relaxed);
        if let Some(lateness) = max_lateness {
            let micros = u64::try_from(lateness.as_micros()).unwrap_or(u64::MAX);
            self.max_lateness_micros
                .fetch_max(micros, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.dropped.load(Ordering::Relaxed),
            self.affected_batches.load(Ordering::Relaxed),
            self.max_lateness_micros.load(Ordering::Relaxed),
        )
    }
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

/// The runtime-owned validating [`StreamCollector`] (API note A2.1).
///
/// The operator task constructs one collector per operator from the compiled
/// output ports; `emit` validates the port name, kind, and exact schema
/// before enqueueing, so an invalid batch never reaches an edge (S5.4's
/// fail-closed-before-side-effect rule). This in-memory outbox is the M1.1
/// backing; M1.4's bounded channels replace the storage without changing the
/// validation contract.
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
        let recorder = LateRowRecorder::default();
        recorder.record(0, Some(Duration::from_micros(9)));
        assert_eq!(recorder.snapshot(), (0, 0, 0));

        recorder.record(2, Some(Duration::from_micros(5)));
        recorder.record(3, Some(Duration::from_micros(7)));
        recorder.record(1, None);
        assert_eq!(recorder.snapshot(), (6, 3, 7));
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
        let context = StreamOperatorContext::new(&job, "window", None);
        context
            .record_late_rows(4, Some(Duration::from_micros(11)))
            .unwrap();
        assert_eq!(context.late_rows.snapshot(), (4, 1, 11));
    }
}
