use std::{panic::AssertUnwindSafe, sync::Arc};

use futures::FutureExt;
use parking_lot::Mutex;
use tokio::sync::watch;

use super::{
    EdgeReceiver, EnvelopeCost, StreamMessage, StreamMessageKind,
    context::{StreamTaskContext, wait_for_task_gate},
    job::ValidatedOrdinarySink,
    metrics::{MetricsRecorder, sink_metric_id},
    supervisor::{TaskFailureSignal, TaskId, TaskSupervisor, panic_message},
};
use crate::{CalcFlowError, CancellationToken, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SinkFailurePhase {
    Write,
    Close,
}

#[derive(Debug)]
pub(crate) struct SinkTaskFailure {
    pub(crate) output_id: String,
    pub(crate) sink_id: String,
    pub(crate) phase: SinkFailurePhase,
    pub(crate) error: CalcFlowError,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SinkProgressSnapshot {
    pub(crate) delivered_batches: u64,
    pub(crate) delivered_rows: u64,
    pub(crate) delivered_bytes: u64,
    pub(crate) ended: bool,
}

#[derive(Default)]
struct SinkProgressState {
    snapshot: SinkProgressSnapshot,
    failures: Vec<SinkTaskFailure>,
}

#[derive(Clone, Default)]
pub(crate) struct SinkProgress(Arc<Mutex<SinkProgressState>>);

impl SinkProgress {
    pub(crate) fn snapshot(&self) -> SinkProgressSnapshot {
        self.0.lock().snapshot.clone()
    }

    pub(crate) fn take_failures(&self) -> Vec<SinkTaskFailure> {
        std::mem::take(&mut self.0.lock().failures)
    }

    fn record_delivery(&self, rows: usize, bytes: usize) -> Result<()> {
        let mut state = self.0.lock();
        let (batches, rows, bytes) = next_delivery_totals(&state.snapshot, rows, bytes)?;
        state.snapshot.delivered_batches = batches;
        state.snapshot.delivered_rows = rows;
        state.snapshot.delivered_bytes = bytes;
        Ok(())
    }

    fn record_failure(&self, failure: SinkTaskFailure) {
        self.0.lock().failures.push(failure);
    }

    fn mark_ended(&self) {
        self.0.lock().snapshot.ended = true;
    }
}

fn next_delivery_totals(
    snapshot: &SinkProgressSnapshot,
    rows: usize,
    bytes: usize,
) -> Result<(u64, u64, u64)> {
    let rows = delivery_increment(rows, "delivered_rows")?;
    let bytes = delivery_increment(bytes, "delivered_bytes")?;
    Ok((
        next_delivery_counter(snapshot.delivered_batches, 1, "delivered_batches")?,
        next_delivery_counter(snapshot.delivered_rows, rows, "delivered_rows")?,
        next_delivery_counter(snapshot.delivered_bytes, bytes, "delivered_bytes")?,
    ))
}

fn delivery_increment(value: usize, counter: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| counter_overflow(counter))
}

fn next_delivery_counter(current: u64, increment: u64, counter: &str) -> Result<u64> {
    current
        .checked_add(increment)
        .ok_or_else(|| counter_overflow(counter))
}

fn counter_overflow(counter: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("sink {counter} counter overflowed"),
    }
}

pub(crate) struct SinkTaskInputs {
    pub(crate) output_id: String,
    pub(crate) sinks: Vec<ValidatedOrdinarySink>,
    pub(crate) input: EdgeReceiver,
    pub(crate) context: StreamTaskContext,
    pub(crate) progress: SinkProgress,
    pub(crate) metrics: MetricsRecorder,
    pub(crate) data_gate: watch::Receiver<bool>,
    pub(crate) launch_cancel: CancellationToken,
}

pub(crate) fn spawn_sink_task(supervisor: &mut TaskSupervisor, inputs: SinkTaskInputs) -> TaskId {
    let task_name = format!("sink:{}", inputs.output_id);
    supervisor.spawn_with_failure_signal(task_name, move |failure_signal| async move {
        run_sink_task(inputs, failure_signal).await
    })
}

async fn run_sink_task(
    mut inputs: SinkTaskInputs,
    failure_signal: TaskFailureSignal,
) -> Result<()> {
    let closed_message = format!(
        "sink {:?} data gate closed before release",
        inputs.output_id
    );
    if !wait_for_task_gate(
        &mut inputs.data_gate,
        &inputs.launch_cancel,
        inputs.context.job().cancellation(),
        &closed_message,
    )
    .await?
    {
        let close_failed = close_all(&mut inputs, &failure_signal).await;
        return if close_failed {
            failure_signal.cancel_siblings();
            Err(task_failed(&inputs.output_id))
        } else {
            Ok(())
        };
    }
    let failed = run_sink_loop(&mut inputs, &failure_signal).await;
    let close_failed = close_all(&mut inputs, &failure_signal).await;
    if failed || close_failed {
        failure_signal.cancel_siblings();
        Err(task_failed(&inputs.output_id))
    } else {
        Ok(())
    }
}

fn task_failed(output_id: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!(
            "ordinary sink task {output_id:?} failed; inspect its private failure records"
        ),
    }
}

async fn run_sink_loop(inputs: &mut SinkTaskInputs, failure_signal: &TaskFailureSignal) -> bool {
    loop {
        match next_sink_step(inputs, failure_signal.task_id()).await {
            SinkLoopStep::Continue => {}
            SinkLoopStep::Complete | SinkLoopStep::Cancelled => return false,
            SinkLoopStep::Failed { sink_id, error } => {
                record_sink_failure(inputs, sink_id, error);
                failure_signal.cancel_siblings();
                return true;
            }
        }
    }
}

enum SinkLoopStep {
    Continue,
    Complete,
    Cancelled,
    Failed {
        sink_id: String,
        error: CalcFlowError,
    },
}

async fn next_sink_step(inputs: &mut SinkTaskInputs, task_id: TaskId) -> SinkLoopStep {
    match receive_sink_message(inputs).await {
        Ok(Some(message)) => process_sink_message(inputs, message, task_id).await,
        Ok(None) => SinkLoopStep::Cancelled,
        Err(error) => SinkLoopStep::Failed {
            sink_id: inputs.output_id.clone(),
            error,
        },
    }
}

async fn receive_sink_message(inputs: &mut SinkTaskInputs) -> Result<Option<StreamMessage>> {
    let received = tokio::select! {
        biased;
        () = inputs.context.job().cancellation().cancelled() => return Ok(None),
        result = inputs.input.recv() => result,
    }?;
    match received {
        Some(message) => Ok(Some(message)),
        None if inputs.context.job().cancellation().is_cancelled() => Ok(None),
        None => Err(CalcFlowError::EdgeClosed {
            edge: inputs.input.edge().into(),
        }),
    }
}

async fn process_sink_message(
    inputs: &mut SinkTaskInputs,
    message: StreamMessage,
    task_id: TaskId,
) -> SinkLoopStep {
    match message.kind() {
        StreamMessageKind::Data => deliver_data(inputs, &message, task_id).await,
        StreamMessageKind::Watermark | StreamMessageKind::Idle => SinkLoopStep::Continue,
        StreamMessageKind::EndOfInput => {
            inputs.progress.mark_ended();
            SinkLoopStep::Complete
        }
        StreamMessageKind::Barrier => SinkLoopStep::Failed {
            sink_id: inputs.output_id.clone(),
            error: CalcFlowError::InvalidArgument {
                field: format!("sinks.{}", inputs.output_id),
                message: "barrier delivery is unavailable before M5".into(),
            },
        },
    }
}

async fn deliver_data(
    inputs: &mut SinkTaskInputs,
    message: &StreamMessage,
    task_id: TaskId,
) -> SinkLoopStep {
    let batch = message.as_data().expect("data kind always has a batch");
    let cost = match EnvelopeCost::of_message(message) {
        Ok(cost) => cost,
        Err(error) => {
            return SinkLoopStep::Failed {
                sink_id: inputs.output_id.clone(),
                error,
            };
        }
    };
    for sink in &mut inputs.sinks {
        match write_sink(
            sink,
            batch,
            cost,
            &inputs.output_id,
            &inputs.metrics,
            inputs.context.job().cancellation(),
            task_id,
        )
        .await
        {
            Ok(true) => {}
            Ok(false) => return SinkLoopStep::Cancelled,
            Err(error) => {
                return SinkLoopStep::Failed {
                    sink_id: sink.sink_id.clone(),
                    error,
                };
            }
        }
    }
    match inputs.progress.record_delivery(cost.rows(), cost.bytes()) {
        Ok(()) => SinkLoopStep::Continue,
        Err(error) => SinkLoopStep::Failed {
            sink_id: inputs.output_id.clone(),
            error,
        },
    }
}

async fn write_sink(
    sink: &mut ValidatedOrdinarySink,
    batch: &crate::Batch,
    cost: EnvelopeCost,
    output_id: &str,
    metrics: &MetricsRecorder,
    cancellation: &CancellationToken,
    task_id: TaskId,
) -> Result<bool> {
    let timer = metrics.timer();
    let result = tokio::select! {
        biased;
        () = cancellation.cancelled() => return Ok(false),
        result = AssertUnwindSafe(sink.binding.sink.write(batch)).catch_unwind() => result,
    };
    match result {
        Ok(result) => result?,
        Err(payload) => {
            return Err(CalcFlowError::TaskPanicked {
                task_id: task_id.as_u64(),
                message: panic_message(payload.as_ref()),
            });
        }
    }
    metrics.record_sink_delivery(&sink_metric_id(output_id, &sink.sink_id), cost, &timer)?;
    Ok(true)
}

fn record_sink_failure(inputs: &SinkTaskInputs, sink_id: String, error: CalcFlowError) {
    inputs.progress.record_failure(SinkTaskFailure {
        output_id: inputs.output_id.clone(),
        sink_id,
        phase: SinkFailurePhase::Write,
        error,
    });
}

async fn close_all(inputs: &mut SinkTaskInputs, failure_signal: &TaskFailureSignal) -> bool {
    let mut indexes = (0..inputs.sinks.len()).collect::<Vec<_>>();
    indexes.sort_by(|left, right| {
        inputs.sinks[*left]
            .sink_id
            .cmp(&inputs.sinks[*right].sink_id)
    });
    let mut failed = false;
    for index in indexes {
        let sink = &mut inputs.sinks[index];
        let result = AssertUnwindSafe(sink.binding.sink.close())
            .catch_unwind()
            .await;
        let error = match result {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error),
            Err(payload) => Some(CalcFlowError::TaskPanicked {
                task_id: failure_signal.task_id().as_u64(),
                message: panic_message(payload.as_ref()),
            }),
        };
        if let Some(error) = error {
            failed = true;
            inputs.progress.record_failure(SinkTaskFailure {
                output_id: inputs.output_id.clone(),
                sink_id: sink.sink_id.clone(),
                phase: SinkFailurePhase::Close,
                error,
            });
        }
    }
    failed
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use async_trait::async_trait;
    use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
    use parking_lot::Mutex;
    use tokio::sync::{Notify, watch};

    use super::{SinkFailurePhase, SinkProgress, SinkTaskInputs, spawn_sink_task};
    use crate::{
        Batch, BatchMetadata, CalcFlowError, CancellationToken, EdgeBudget, JsonMap, Result,
        StreamJobContext, StreamMessage, edge_channel,
        runtime::streaming::{
            job::{M2SinkDelivery, OrdinarySinkBinding, OrdinaryStreamSink, ValidatedOrdinarySink},
            supervisor::TaskSupervisor,
        },
    };

    fn batch(sequence: u64) -> Batch {
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![i64::try_from(sequence).unwrap()])) as _,
        )])
        .unwrap();
        Batch::table(
            vec![record],
            BatchMetadata::new("source", sequence, BTreeMap::new()).unwrap(),
        )
        .unwrap()
    }

    struct RecordingSink {
        id: String,
        log: Arc<Mutex<Vec<String>>>,
        fail_write: Option<u64>,
        fail_close: bool,
        write_gate: Option<Arc<Notify>>,
        closes: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl OrdinaryStreamSink for RecordingSink {
        fn delivery_capability(&self) -> M2SinkDelivery {
            M2SinkDelivery::ProcessLocalOrdered
        }

        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, batch: &Batch) -> Result<()> {
            let sequence = batch.metadata().sequence();
            self.log.lock().push(format!("{}:{sequence}", self.id));
            if let Some(gate) = &self.write_gate {
                gate.notified().await;
            }
            if self.fail_write == Some(sequence) {
                Err(CalcFlowError::Internal {
                    message: format!("{} write", self.id),
                })
            } else {
                Ok(())
            }
        }

        async fn close(&mut self) -> Result<()> {
            self.closes.fetch_add(1, Ordering::SeqCst);
            self.log.lock().push(format!("{}:close", self.id));
            if self.fail_close {
                Err(CalcFlowError::Internal {
                    message: format!("{} close", self.id),
                })
            } else {
                Ok(())
            }
        }
    }

    struct Harness {
        supervisor: TaskSupervisor,
        cancellation: CancellationToken,
        sender: crate::EdgeSender,
        data: watch::Sender<bool>,
        progress: SinkProgress,
    }

    fn validated_sink(
        id: &str,
        log: &Arc<Mutex<Vec<String>>>,
        fail_write: Option<u64>,
        fail_close: bool,
        write_gate: Option<Arc<Notify>>,
        closes: &Arc<AtomicUsize>,
    ) -> ValidatedOrdinarySink {
        ValidatedOrdinarySink {
            sink_id: id.into(),
            binding: OrdinarySinkBinding::new(Box::new(RecordingSink {
                id: id.into(),
                log: Arc::clone(log),
                fail_write,
                fail_close,
                write_gate,
                closes: Arc::clone(closes),
            })),
        }
    }

    fn harness(sinks: Vec<ValidatedOrdinarySink>) -> Harness {
        let cancellation = CancellationToken::new();
        let context =
            StreamJobContext::new(9, "fingerprint", JsonMap::new(), None, cancellation.clone());
        let (sender, receiver) = edge_channel(
            "node.output->sink",
            EdgeBudget {
                max_rows: 64,
                max_bytes: 1 << 20,
            },
        )
        .unwrap();
        let (data, data_rx) = watch::channel(false);
        let progress = SinkProgress::default();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        spawn_sink_task(
            &mut supervisor,
            SinkTaskInputs {
                output_id: "output".into(),
                sinks,
                input: receiver,
                context: context.for_sink("output").unwrap(),
                progress: progress.clone(),
                metrics: super::MetricsRecorder::default(),
                data_gate: data_rx,
                launch_cancel: CancellationToken::new(),
            },
        );
        Harness {
            supervisor,
            cancellation,
            sender,
            data,
            progress,
        }
    }

    #[tokio::test]
    async fn configured_sink_order_finishes_batch_before_next_batch() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let sinks = ["one", "two", "three"]
            .map(|id| validated_sink(id, &log, None, false, None, &closes))
            .into();
        let mut harness = harness(sinks);
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::data(batch(0)))
            .await
            .unwrap();
        harness
            .sender
            .send(StreamMessage::data(batch(1)))
            .await
            .unwrap();
        harness
            .sender
            .send(StreamMessage::end_of_input())
            .await
            .unwrap();
        let report = harness.supervisor.join_all().await;
        assert!(report.errors.is_empty());
        assert_eq!(
            &log.lock()[..6],
            ["one:0", "two:0", "three:0", "one:1", "two:1", "three:1"]
        );
        assert_eq!(closes.load(Ordering::SeqCst), 3);
        assert_eq!(harness.progress.snapshot().delivered_batches, 2);
        assert!(harness.progress.snapshot().ended);
    }

    #[tokio::test]
    async fn sink_two_write_failure_stops_sink_three_and_close_failures_are_stable_secondaries() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let sinks = vec![
            validated_sink("one", &log, None, true, None, &closes),
            validated_sink("two", &log, Some(0), false, None, &closes),
            validated_sink("three", &log, None, true, None, &closes),
        ];
        let mut harness = harness(sinks);
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::data(batch(0)))
            .await
            .unwrap();
        let report = harness.supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
        assert_eq!(&log.lock()[..2], ["one:0", "two:0"]);
        assert!(!log.lock().contains(&"three:0".to_owned()));
        assert_eq!(closes.load(Ordering::SeqCst), 3);
        let failures = harness.progress.take_failures();
        assert_eq!(failures.len(), 3);
        assert_eq!(failures[0].sink_id, "two");
        assert_eq!(failures[0].phase, SinkFailurePhase::Write);
        assert_eq!(failures[1].sink_id, "one");
        assert_eq!(failures[2].sink_id, "three");
        assert!(
            failures[1..]
                .iter()
                .all(|failure| failure.phase == SinkFailurePhase::Close)
        );
    }

    #[tokio::test]
    async fn cancel_drops_inflight_write_without_drain_then_closes_and_joins() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Notify::new());
        let sinks = vec![validated_sink(
            "slow",
            &log,
            None,
            false,
            Some(gate),
            &closes,
        )];
        let mut harness = harness(sinks);
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::data(batch(0)))
            .await
            .unwrap();
        while log.lock().is_empty() {
            tokio::task::yield_now().await;
        }
        harness.cancellation.cancel();
        let report = harness.supervisor.join_all().await;
        assert!(report.errors.is_empty());
        assert_eq!(closes.load(Ordering::SeqCst), 1);
        assert_eq!(harness.progress.snapshot().delivered_batches, 0);
        assert!(!harness.progress.snapshot().ended);
    }

    #[tokio::test]
    async fn premature_input_close_is_edge_closed_and_not_natural_end() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let sinks = vec![validated_sink("one", &log, None, false, None, &closes)];
        let mut harness = harness(sinks);
        harness.data.send(true).unwrap();
        drop(harness.sender);
        let report = harness.supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
        let failures = harness.progress.take_failures();
        assert_eq!(failures.len(), 1);
        assert!(
            matches!(failures[0].error, CalcFlowError::EdgeClosed { ref edge } if edge == "node.output->sink")
        );
        assert!(!harness.progress.snapshot().ended);
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn convergence_marking_turns_closed_sink_ingress_into_a_wakeup() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let sinks = vec![validated_sink("one", &log, None, false, None, &closes)];
        let mut harness = harness(sinks);
        harness.data.send(true).unwrap();

        harness.cancellation.cancel();
        drop(harness.sender);

        let report = harness.supervisor.join_all().await;
        assert!(report.errors.is_empty(), "{report:?}");
        assert!(harness.progress.take_failures().is_empty());
        assert!(!harness.progress.snapshot().ended);
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }
}
