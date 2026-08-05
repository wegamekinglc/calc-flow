use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::mpsc;

use super::{
    EdgeSender, StreamJobContext, StreamMessage,
    supervisor::{TaskFailureSignal, TaskSupervisor},
};
use crate::{Batch, BatchMetadata, CalcFlowError, EventTime, JsonMap, Result, canonical_json};

const MAX_CURSOR_ORDER_BYTES: usize = 16 * 1024;

/// Source-defined position with a bytewise order key and opaque JSON payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Cursor {
    order: Vec<u8>,
    payload: JsonMap,
}

impl Cursor {
    pub(crate) fn new(order: Vec<u8>, payload: JsonMap) -> Result<Self> {
        if order.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "cursor.order".into(),
                message: "must not be empty".into(),
            });
        }
        if order.len() > MAX_CURSOR_ORDER_BYTES {
            return Err(CalcFlowError::InvalidArgument {
                field: "cursor.order".into(),
                message: format!("must not exceed {MAX_CURSOR_ORDER_BYTES} bytes"),
            });
        }
        let payload_value = Value::Object(payload.clone().into_iter().collect());
        canonical_json(&payload_value).map_err(|error| CalcFlowError::InvalidArgument {
            field: "cursor.payload".into(),
            message: error.to_string(),
        })?;
        Ok(Self { order, payload })
    }

    fn is_after(&self, previous: &Self) -> bool {
        self.order > previous.order
    }
}

/// Events a connector source can return; barriers remain runtime-only.
#[derive(Clone, Debug)]
pub(crate) enum SourceEvent {
    Data { batch: Batch, cursor: Cursor },
    Watermark(EventTime),
    Idle,
}

/// Capabilities sampled before a source is opened.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SourceCapabilities {
    pub(crate) replayable: bool,
    pub(crate) max_batch_rows: usize,
    pub(crate) max_batch_bytes: usize,
}

/// Internal source lifecycle contract for the M2 vertical slice.
#[async_trait]
pub(crate) trait StreamSource: Send {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()>;

    /// May block indefinitely and need not be cancellation-safe for reuse.
    /// The runtime keeps exactly one call in flight, drops it only at
    /// teardown, never polls this instance again, and then calls `close`.
    async fn next(&mut self) -> Result<Option<SourceEvent>>;

    /// Must tolerate a `next` future dropped during teardown (spec D3.5).
    async fn close(&mut self) -> Result<()>;

    fn capabilities(&self) -> SourceCapabilities;
}

/// One validated source binding. Binding identity is assigned by the job.
pub(crate) struct SourceBinding {
    source: Box<dyn StreamSource>,
    capabilities: SourceCapabilities,
    resume_cursor: Option<Cursor>,
    next_sequence: u64,
}

impl SourceBinding {
    pub(crate) fn new(
        source: Box<dyn StreamSource>,
        resume_cursor: Option<Cursor>,
        next_sequence: u64,
    ) -> Result<Self> {
        let capabilities = source.capabilities();
        if capabilities.max_batch_rows == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "source.capabilities.max_batch_rows".into(),
                message: "must be greater than zero".into(),
            });
        }
        if capabilities.max_batch_bytes == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "source.capabilities.max_batch_bytes".into(),
                message: "must be greater than zero".into(),
            });
        }
        Ok(Self {
            source,
            capabilities,
            resume_cursor,
            next_sequence,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceProgressSnapshot {
    pub(crate) replayable: bool,
    pub(crate) latest_observed_cursor: Option<Cursor>,
    pub(crate) durable_cursor: Option<Cursor>,
    pub(crate) next_sequence: Option<u64>,
    pub(crate) ended: bool,
}

/// Separates volatile observed progress from checkpoint-durable progress.
#[derive(Clone, Debug)]
pub(crate) struct SourceProgress(Arc<Mutex<SourceProgressSnapshot>>);

impl SourceProgress {
    pub(crate) fn snapshot(&self) -> SourceProgressSnapshot {
        self.0.lock().clone()
    }
}

enum PumpEvent {
    Event(SourceEvent),
    End,
}

enum PumpCompletion {
    Cancelled,
    Ended,
}

struct SourceTaskInputs {
    binding_id: String,
    first_sequence: u64,
    resume_cursor: Option<Cursor>,
    outputs: Vec<EdgeSender>,
    slot: mpsc::Receiver<PumpEvent>,
    progress: SourceProgress,
    cancellation: crate::CancellationToken,
}

/// Registers the pump and source task as two owned supervised units.
pub(crate) fn spawn_source_tasks(
    supervisor: &mut TaskSupervisor,
    context: &StreamJobContext,
    binding_id: &str,
    binding: SourceBinding,
    outputs: Vec<EdgeSender>,
) -> Result<SourceProgress> {
    let source_context = context.for_source(binding_id)?;
    let binding_id = source_context.scope_id();
    if outputs.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: format!("sources.{binding_id}.outputs"),
            message: "must contain at least one edge".into(),
        });
    }
    for output in &outputs {
        let budget = output.budget();
        if binding.capabilities.max_batch_rows > budget.max_rows
            || binding.capabilities.max_batch_bytes > budget.max_bytes
        {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sources.{binding_id}.capabilities"),
                message: format!(
                    "maximum batch ({}, {} bytes) exceeds edge {:?} budget ({}, {} bytes)",
                    binding.capabilities.max_batch_rows,
                    binding.capabilities.max_batch_bytes,
                    output.edge(),
                    budget.max_rows,
                    budget.max_bytes
                ),
            });
        }
    }

    let SourceBinding {
        source,
        capabilities,
        resume_cursor,
        next_sequence,
    } = binding;
    let progress = SourceProgress(Arc::new(Mutex::new(SourceProgressSnapshot {
        replayable: capabilities.replayable,
        latest_observed_cursor: None,
        durable_cursor: resume_cursor.clone(),
        next_sequence: Some(next_sequence),
        ended: false,
    })));
    let (slot_tx, slot_rx) = mpsc::channel(1);
    let cancellation = source_context.job().cancellation().clone();
    let pump_cancellation = cancellation.clone();
    let pump_resume_cursor = resume_cursor.clone();
    supervisor.spawn_with_failure_signal(
        format!("source:{binding_id}:pump"),
        move |failure_signal| async move {
            run_source_pump(
                source,
                pump_resume_cursor,
                slot_tx,
                pump_cancellation,
                move || failure_signal.cancel_siblings(),
            )
            .await
        },
    );
    let binding_id = binding_id.to_owned();
    let task_progress = progress.clone();
    supervisor.spawn_with_failure_signal(
        format!("source:{binding_id}:task"),
        move |failure_signal| async move {
            run_source_task(
                SourceTaskInputs {
                    binding_id,
                    first_sequence: next_sequence,
                    resume_cursor,
                    outputs,
                    slot: slot_rx,
                    progress: task_progress,
                    cancellation,
                },
                failure_signal,
            )
            .await
        },
    );
    Ok(progress)
}

async fn run_source_pump(
    mut source: Box<dyn StreamSource>,
    resume_cursor: Option<Cursor>,
    slot: mpsc::Sender<PumpEvent>,
    cancellation: crate::CancellationToken,
    on_error: impl FnOnce(),
) -> Result<()> {
    let operation = async {
        tokio::select! {
            biased;
            () = cancellation.cancelled() => return Ok(PumpCompletion::Cancelled),
            result = source.open(resume_cursor) => result?,
        }
        loop {
            let permit = tokio::select! {
                biased;
                () = cancellation.cancelled() => return Ok(PumpCompletion::Cancelled),
                permit = slot.reserve() => permit.map_err(|_| CalcFlowError::Internal {
                    message: "source prefetch slot closed before pump convergence".into(),
                })?,
            };
            let event = tokio::select! {
                biased;
                () = cancellation.cancelled() => return Ok(PumpCompletion::Cancelled),
                event = source.next() => event?,
            };
            match event {
                Some(event) => permit.send(PumpEvent::Event(event)),
                None => return Ok(PumpCompletion::Ended),
            }
        }
    }
    .await;

    if operation.is_err() {
        on_error();
    }
    let close_result = source.close().await;
    match operation {
        Ok(PumpCompletion::Cancelled) => close_result,
        Ok(PumpCompletion::Ended) => {
            close_result?;
            tokio::select! {
                biased;
                () = cancellation.cancelled() => Ok(()),
                result = slot.send(PumpEvent::End) => result.map_err(|_| CalcFlowError::Internal {
                    message: "source prefetch slot closed before end-of-input delivery".into(),
                }),
            }
        }
        Err(error) => Err(error),
    }
}

async fn run_source_task(
    mut inputs: SourceTaskInputs,
    failure_signal: TaskFailureSignal,
) -> Result<()> {
    let operation = run_source_task_loop(&mut inputs).await;
    if operation.is_err() {
        failure_signal.cancel_siblings();
    }
    operation
}

async fn run_source_task_loop(inputs: &mut SourceTaskInputs) -> Result<()> {
    let SourceTaskInputs {
        binding_id,
        first_sequence,
        resume_cursor,
        outputs,
        slot,
        progress,
        cancellation,
    } = inputs;
    let mut next_sequence = Some(*first_sequence);
    let mut last_cursor = resume_cursor.clone();
    let mut last_watermark: Option<EventTime> = None;
    loop {
        let event = tokio::select! {
            biased;
            () = cancellation.cancelled() => return Ok(()),
            event = slot.recv() => event,
        };
        let Some(event) = event else {
            return if cancellation.is_cancelled() {
                Ok(())
            } else {
                Err(CalcFlowError::Internal {
                    message: format!("source {binding_id:?} pump ended without end-of-input"),
                })
            };
        };
        match event {
            PumpEvent::Event(SourceEvent::Data { batch, cursor }) => {
                if last_cursor
                    .as_ref()
                    .is_some_and(|previous| !cursor.is_after(previous))
                {
                    return Err(CalcFlowError::InvalidArgument {
                        field: "source.cursor".into(),
                        message: format!(
                            "source binding {binding_id:?} emitted a repeated or regressed cursor"
                        ),
                    });
                }
                let sequence = next_sequence.ok_or_else(|| CalcFlowError::Internal {
                    message: format!("source binding {binding_id:?} sequence is exhausted"),
                })?;
                let metadata = BatchMetadata::new(
                    binding_id.clone(),
                    sequence,
                    batch.metadata().attributes().clone(),
                )?;
                let message = StreamMessage::data(batch.with_metadata(metadata));
                if !send_fanout(outputs, message, cancellation).await? {
                    return Ok(());
                }
                next_sequence = sequence.checked_add(1);
                last_cursor = Some(cursor.clone());
                let mut snapshot = progress.0.lock();
                snapshot.latest_observed_cursor = Some(cursor);
                snapshot.next_sequence = next_sequence;
            }
            PumpEvent::Event(SourceEvent::Watermark(watermark)) => {
                if last_watermark.is_some_and(|previous| watermark < previous) {
                    return Err(CalcFlowError::InvalidArgument {
                        field: "source.watermark".into(),
                        message: format!("source binding {binding_id:?} regressed its watermark"),
                    });
                }
                if !send_fanout(outputs, StreamMessage::watermark(watermark), cancellation).await? {
                    return Ok(());
                }
                last_watermark = Some(watermark);
            }
            PumpEvent::Event(SourceEvent::Idle) => {
                if !send_fanout(outputs, StreamMessage::idle(), cancellation).await? {
                    return Ok(());
                }
            }
            PumpEvent::End => {
                if !send_fanout(outputs, StreamMessage::end_of_input(), cancellation).await? {
                    return Ok(());
                }
                progress.0.lock().ended = true;
                return Ok(());
            }
        }
    }
}

async fn send_fanout(
    outputs: &mut [EdgeSender],
    message: StreamMessage,
    cancellation: &crate::CancellationToken,
) -> Result<bool> {
    for output in outputs {
        let sent = tokio::select! {
            biased;
            () = cancellation.cancelled() => return Ok(false),
            result = output.send(message.clone()) => result,
        };
        sent?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::{
        any::Any,
        collections::{BTreeMap, VecDeque},
        future::{Future, pending, poll_fn},
        pin::Pin,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        task::Poll,
    };

    use async_trait::async_trait;
    use parking_lot::Mutex;
    use serde_json::json;
    use tokio::sync::{mpsc, oneshot};

    use super::{
        Cursor, SourceBinding, SourceCapabilities, SourceEvent, StreamSource, run_source_pump,
        spawn_source_tasks,
    };
    use crate::{
        Batch, BatchMetadata, CalcFlowError, CancellationToken, EdgeBudget, ExternalPayload,
        JsonMap, Result, StreamJobContext, StreamMessageKind, edge_channel,
        runtime::streaming::supervisor::TaskSupervisor,
    };

    #[derive(Debug)]
    struct TestPayload;

    impl ExternalPayload for TestPayload {
        fn backend(&self) -> &'static str {
            "m2-source-test"
        }

        fn len(&self) -> usize {
            1
        }

        fn estimated_bytes(&self) -> usize {
            1
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn batch(attribute: u64) -> Batch {
        Batch::external(
            Arc::new(TestPayload),
            BatchMetadata::new(
                "connector-owned",
                99,
                BTreeMap::from([("attribute".into(), json!(attribute))]),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn cursor(position: u8) -> Cursor {
        Cursor::new(
            vec![position],
            BTreeMap::from([("position".into(), json!(position))]),
        )
        .unwrap()
    }

    fn context(cancellation: CancellationToken) -> StreamJobContext {
        StreamJobContext::new(7, "fingerprint", JsonMap::new(), None, cancellation)
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum OpenObservation {
        NotOpened,
        Opened(Option<Cursor>),
    }

    struct StepSource {
        events: VecDeque<Result<Option<SourceEvent>>>,
        capabilities: SourceCapabilities,
        next_calls: Option<mpsc::UnboundedSender<usize>>,
        next_count: usize,
        opened_at: Arc<Mutex<OpenObservation>>,
        closed: Arc<AtomicBool>,
    }

    impl StepSource {
        fn new(events: impl IntoIterator<Item = Result<Option<SourceEvent>>>) -> Self {
            Self {
                events: events.into_iter().collect(),
                capabilities: SourceCapabilities {
                    replayable: true,
                    max_batch_rows: 1,
                    max_batch_bytes: 1,
                },
                next_calls: None,
                next_count: 0,
                opened_at: Arc::new(Mutex::new(OpenObservation::NotOpened)),
                closed: Arc::new(AtomicBool::new(false)),
            }
        }

        fn with_next_calls(mut self, next_calls: mpsc::UnboundedSender<usize>) -> Self {
            self.next_calls = Some(next_calls);
            self
        }

        fn with_capabilities(mut self, capabilities: SourceCapabilities) -> Self {
            self.capabilities = capabilities;
            self
        }
    }

    #[async_trait]
    impl StreamSource for StepSource {
        async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
            *self.opened_at.lock() = OpenObservation::Opened(cursor);
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            self.next_count += 1;
            if let Some(next_calls) = &self.next_calls {
                next_calls.send(self.next_count).unwrap();
            }
            self.events.pop_front().unwrap_or(Ok(None))
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            self.capabilities
        }
    }

    async fn assert_pending<T>(future: &mut (impl Future<Output = T> + Unpin)) {
        poll_fn(|context| match Pin::new(&mut *future).poll(context) {
            Poll::Pending => Poll::Ready(()),
            Poll::Ready(_) => panic!("future unexpectedly completed"),
        })
        .await;
    }

    #[test]
    fn zero_source_capability_is_rejected_before_open() {
        let source = StepSource::new([]).with_capabilities(SourceCapabilities {
            replayable: true,
            max_batch_rows: 0,
            max_batch_bytes: 1,
        });
        let opened_at = Arc::clone(&source.opened_at);

        let result = SourceBinding::new(Box::new(source), None, 0);

        assert!(matches!(
            result,
            Err(CalcFlowError::InvalidArgument { ref field, .. })
                if field == "source.capabilities.max_batch_rows"
        ));
        assert_eq!(*opened_at.lock(), OpenObservation::NotOpened);
    }

    #[test]
    fn downstream_budget_is_rejected_before_open_or_task_registration() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let source = StepSource::new([]).with_capabilities(SourceCapabilities {
            replayable: true,
            max_batch_rows: 2,
            max_batch_bytes: 1,
        });
        let opened_at = Arc::clone(&source.opened_at);
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, _receiver) = edge_channel(
            "source->too-small",
            EdgeBudget {
                max_rows: 1,
                max_bytes: 1,
            },
        )
        .unwrap();

        let result = spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "input",
            binding,
            vec![sender],
        );

        assert!(matches!(
            result,
            Err(CalcFlowError::InvalidArgument { ref field, .. })
                if field == "sources.input.capabilities"
        ));
        assert_eq!(*opened_at.lock(), OpenObservation::NotOpened);
        assert_eq!(supervisor.task_count(), 0);
    }

    #[tokio::test]
    async fn data_metadata_cursor_fanout_and_end_are_ordered() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let source = StepSource::new([
            Ok(Some(SourceEvent::Data {
                batch: batch(10),
                cursor: cursor(1),
            })),
            Ok(Some(SourceEvent::Data {
                batch: batch(20),
                cursor: cursor(2),
            })),
            Ok(None),
        ]);
        let opened_at = Arc::clone(&source.opened_at);
        let closed = Arc::clone(&source.closed);
        let resume = cursor(0);
        let binding = SourceBinding::new(Box::new(source), Some(resume.clone()), 5).unwrap();
        let (left_tx, mut left_rx) = edge_channel("source->left", EdgeBudget::default()).unwrap();
        let (right_tx, mut right_rx) =
            edge_channel("source->right", EdgeBudget::default()).unwrap();

        let progress = spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "orders",
            binding,
            vec![left_tx, right_tx],
        )
        .unwrap();
        let report = supervisor.join_all().await;

        assert!(report.errors.is_empty());
        assert_eq!(*opened_at.lock(), OpenObservation::Opened(Some(resume)));
        assert!(closed.load(Ordering::SeqCst));
        for receiver in [&mut left_rx, &mut right_rx] {
            for (sequence, attribute) in [(5, 10), (6, 20)] {
                let message = receiver.recv().await.unwrap().unwrap();
                let data = message.as_data().unwrap();
                assert_eq!(data.metadata().source(), "orders");
                assert_eq!(data.metadata().sequence(), sequence);
                assert_eq!(data.metadata().attributes()["attribute"], json!(attribute));
            }
            assert_eq!(
                receiver.recv().await.unwrap().unwrap().kind(),
                StreamMessageKind::EndOfInput
            );
            assert!(receiver.recv().await.unwrap().is_none());
        }
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.latest_observed_cursor, Some(cursor(2)));
        assert_eq!(snapshot.durable_cursor, Some(cursor(0)));
        assert_eq!(snapshot.next_sequence, Some(7));
        assert!(snapshot.ended);
    }

    #[tokio::test]
    async fn full_edge_stops_source_polling_after_the_single_prefetch_slot() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let (calls_tx, mut calls_rx) = mpsc::unbounded_channel();
        let source = StepSource::new((1..=4).map(|position| {
            Ok(Some(SourceEvent::Data {
                batch: batch(position.into()),
                cursor: cursor(position),
            }))
        }))
        .with_next_calls(calls_tx);
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, mut receiver) = edge_channel(
            "source->slow",
            EdgeBudget {
                max_rows: 1,
                max_bytes: 1,
            },
        )
        .unwrap();

        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "input",
            binding,
            vec![sender],
        )
        .unwrap();

        assert_eq!(calls_rx.recv().await, Some(1));
        assert_eq!(calls_rx.recv().await, Some(2));
        assert_eq!(calls_rx.recv().await, Some(3));
        let mut fourth_call = Box::pin(calls_rx.recv());
        assert_pending(&mut fourth_call).await;
        drop(fourth_call);

        assert_eq!(
            receiver
                .recv()
                .await
                .unwrap()
                .unwrap()
                .as_data()
                .unwrap()
                .metadata()
                .sequence(),
            0
        );
        assert_eq!(calls_rx.recv().await, Some(4));

        supervisor.cancel();
        assert!(supervisor.join_all().await.errors.is_empty());
    }

    struct PollGuard(Arc<AtomicBool>);

    impl Drop for PollGuard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    struct BlockingOpenSource {
        open_poll_dropped: Arc<AtomicBool>,
        closed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl StreamSource for BlockingOpenSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            let _guard = PollGuard(Arc::clone(&self.open_poll_dropped));
            pending().await
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            panic!("next must not run before open completes");
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replayable: true,
                max_batch_rows: 1,
                max_batch_bytes: 1,
            }
        }
    }

    #[tokio::test]
    async fn cancel_drops_a_blocked_open_then_closes_source() {
        let cancellation = CancellationToken::new();
        let open_poll_dropped = Arc::new(AtomicBool::new(false));
        let closed = Arc::new(AtomicBool::new(false));
        let source = BlockingOpenSource {
            open_poll_dropped: Arc::clone(&open_poll_dropped),
            closed: Arc::clone(&closed),
        };
        let (slot_tx, _slot_rx) = mpsc::channel(1);
        let mut pump = Box::pin(run_source_pump(
            Box::new(source),
            None,
            slot_tx,
            cancellation.clone(),
            || {},
        ));
        assert_pending(&mut pump).await;

        cancellation.cancel();
        let result = poll_fn(|context| match pump.as_mut().poll(context) {
            Poll::Ready(result) => Poll::Ready(result),
            Poll::Pending => panic!("source pump did not observe cancellation during open"),
        })
        .await;

        assert!(result.is_ok());
        assert!(open_poll_dropped.load(Ordering::SeqCst));
        assert!(closed.load(Ordering::SeqCst));
    }

    struct BlockingSource {
        started: Arc<Mutex<Option<oneshot::Sender<()>>>>,
        poll_dropped: Arc<AtomicBool>,
        closed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl StreamSource for BlockingSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            if let Some(started) = self.started.lock().take() {
                let _ = started.send(());
            }
            let _guard = PollGuard(Arc::clone(&self.poll_dropped));
            pending().await
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replayable: true,
                max_batch_rows: 1,
                max_batch_bytes: 1,
            }
        }
    }

    #[tokio::test]
    async fn cancel_drops_a_blocked_next_poll_then_closes_and_joins_source() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let (started_tx, started_rx) = oneshot::channel();
        let poll_dropped = Arc::new(AtomicBool::new(false));
        let closed = Arc::new(AtomicBool::new(false));
        let source = BlockingSource {
            started: Arc::new(Mutex::new(Some(started_tx))),
            poll_dropped: Arc::clone(&poll_dropped),
            closed: Arc::clone(&closed),
        };
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, _receiver) = edge_channel("source->sink", EdgeBudget::default()).unwrap();
        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "quiet",
            binding,
            vec![sender],
        )
        .unwrap();

        started_rx.await.unwrap();
        supervisor.cancel();
        let report = supervisor.join_all().await;

        assert!(report.errors.is_empty());
        assert!(poll_dropped.load(Ordering::SeqCst));
        assert!(closed.load(Ordering::SeqCst));
        assert_eq!(supervisor.task_count(), 0);
    }

    struct ErrorThenBlockingCloseSource {
        close_started: Option<oneshot::Sender<()>>,
        close_release: Option<oneshot::Receiver<()>>,
    }

    #[async_trait]
    impl StreamSource for ErrorThenBlockingCloseSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            Err(CalcFlowError::Internal {
                message: "source operation failed".into(),
            })
        }

        async fn close(&mut self) -> Result<()> {
            self.close_started.take().unwrap().send(()).unwrap();
            self.close_release.take().unwrap().await.unwrap();
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replayable: true,
                max_batch_rows: 1,
                max_batch_bytes: 1,
            }
        }
    }

    #[tokio::test]
    async fn source_error_cancels_siblings_before_waiting_for_close() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let (close_started_tx, close_started_rx) = oneshot::channel();
        let (close_release_tx, close_release_rx) = oneshot::channel();
        let source = ErrorThenBlockingCloseSource {
            close_started: Some(close_started_tx),
            close_release: Some(close_release_rx),
        };
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, _receiver) = edge_channel("source->sink", EdgeBudget::default()).unwrap();
        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation.clone()),
            "broken",
            binding,
            vec![sender],
        )
        .unwrap();
        let (sibling_finished_tx, sibling_finished_rx) = oneshot::channel();
        let sibling_cancellation = cancellation.clone();
        supervisor.spawn("sibling", async move {
            sibling_cancellation.cancelled().await;
            sibling_finished_tx.send(()).unwrap();
            Ok(())
        });

        close_started_rx.await.unwrap();
        assert!(
            cancellation.is_cancelled(),
            "source failure must trigger cancellation before close completes"
        );
        sibling_finished_rx.await.unwrap();
        let mut join = Box::pin(supervisor.join_all());
        assert_pending(&mut join).await;
        close_release_tx.send(()).unwrap();
        let report = join.await;

        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::Internal { message } if message == "source operation failed"
        ));
    }

    #[tokio::test]
    async fn cursor_regression_fails_before_the_offending_batch_is_enqueued() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let source = StepSource::new([
            Ok(Some(SourceEvent::Data {
                batch: batch(1),
                cursor: cursor(2),
            })),
            Ok(Some(SourceEvent::Data {
                batch: batch(2),
                cursor: cursor(1),
            })),
        ]);
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, mut receiver) = edge_channel("source->sink", EdgeBudget::default()).unwrap();
        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "input",
            binding,
            vec![sender],
        )
        .unwrap();

        let report = supervisor.join_all().await;

        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::InvalidArgument { field, .. } if field == "source.cursor"
        ));
        let first = receiver.recv().await.unwrap().unwrap();
        assert_eq!(first.as_data().unwrap().metadata().sequence(), 0);
        assert!(receiver.recv().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn first_cursor_after_recovery_must_advance_past_the_resume_cursor() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let resume = cursor(2);
        let source = StepSource::new([Ok(Some(SourceEvent::Data {
            batch: batch(1),
            cursor: resume.clone(),
        }))]);
        let binding = SourceBinding::new(Box::new(source), Some(resume), 7).unwrap();
        let (sender, mut receiver) = edge_channel("source->sink", EdgeBudget::default()).unwrap();
        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "input",
            binding,
            vec![sender],
        )
        .unwrap();

        let report = supervisor.join_all().await;

        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::InvalidArgument { field, .. } if field == "source.cursor"
        ));
        assert!(receiver.recv().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn source_error_cancels_and_joins_all_supervised_siblings() {
        let cancellation = CancellationToken::new();
        let sibling_finished = Arc::new(AtomicBool::new(false));
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let source = StepSource::new([Err(CalcFlowError::Internal {
            message: "source read failed".into(),
        })]);
        let closed = Arc::clone(&source.closed);
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, _receiver) = edge_channel("source->sink", EdgeBudget::default()).unwrap();
        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation.clone()),
            "broken",
            binding,
            vec![sender],
        )
        .unwrap();
        let sibling_finished_in_task = Arc::clone(&sibling_finished);
        supervisor.spawn("sibling", async move {
            cancellation.cancelled().await;
            sibling_finished_in_task.store(true, Ordering::SeqCst);
            Ok(())
        });

        let report = supervisor.join_all().await;

        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::Internal { message } if message == "source read failed"
        ));
        assert!(closed.load(Ordering::SeqCst));
        assert!(sibling_finished.load(Ordering::SeqCst));
        assert_eq!(supervisor.task_count(), 0);
    }
}
