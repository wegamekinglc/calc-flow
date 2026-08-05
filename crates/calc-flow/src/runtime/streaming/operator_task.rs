use std::{
    collections::BTreeMap,
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::Arc,
};

use async_trait::async_trait;
use futures::{Future, future::select_all};
use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};

use super::{
    EdgeReceiver, EdgeSender, EnvelopeCost, StreamMessage, StreamMessageKind,
    context::StreamTaskContext,
    metrics::MetricsRecorder,
    supervisor::{TaskId, TaskSupervisor, panic_message},
};
use crate::{
    Batch, CalcFlowError, CancellationToken, EventTime, Port, Result, StreamCollector,
    StreamOperatorContext, pipeline::CompiledStreamOperator,
};

pub(crate) struct OperatorEntryAck {
    pub(crate) node_id: String,
    pub(crate) result: Result<()>,
}

pub(crate) struct OperatorIngress {
    pub(crate) edge_id: String,
    pub(crate) receiver: EdgeReceiver,
    saw_explicit_eof: bool,
}

impl OperatorIngress {
    pub(crate) fn new(edge_id: String, receiver: EdgeReceiver) -> Self {
        Self {
            edge_id,
            receiver,
            saw_explicit_eof: false,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct OperatorProgressSnapshot {
    pub(crate) input_batches: u64,
    pub(crate) fully_fanned_out_batches: u64,
    pub(crate) datafusion_runtime_created: bool,
    pub(crate) on_end_calls: u64,
    pub(crate) ended: bool,
}

#[derive(Clone, Default)]
pub(crate) struct OperatorProgress(Arc<Mutex<OperatorProgressSnapshot>>);

impl OperatorProgress {
    pub(crate) fn snapshot(&self) -> OperatorProgressSnapshot {
        self.0.lock().clone()
    }

    fn record_input(&self) -> Result<()> {
        let mut progress = self.0.lock();
        progress.input_batches =
            progress
                .input_batches
                .checked_add(1)
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "operator input batch counter overflowed".into(),
                })?;
        Ok(())
    }

    fn record_output(&self) -> Result<()> {
        let mut progress = self.0.lock();
        progress.fully_fanned_out_batches = progress
            .fully_fanned_out_batches
            .checked_add(1)
            .ok_or_else(|| CalcFlowError::Internal {
                message: "operator output batch counter overflowed".into(),
            })?;
        Ok(())
    }

    fn mark_ended(&self) {
        self.0.lock().ended = true;
    }

    fn record_on_end(&self) -> Result<()> {
        let mut progress = self.0.lock();
        progress.on_end_calls =
            progress
                .on_end_calls
                .checked_add(1)
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "operator on_end counter overflowed".into(),
                })?;
        Ok(())
    }

    fn observe_datafusion_runtime(&self, created: bool) {
        self.0.lock().datafusion_runtime_created |= created;
    }
}

pub(crate) struct OperatorTaskInputs {
    pub(crate) node_id: String,
    pub(crate) operator: CompiledStreamOperator,
    pub(crate) ingresses: BTreeMap<String, OperatorIngress>,
    pub(crate) outputs: BTreeMap<String, Vec<EdgeSender>>,
    pub(crate) output_ports: BTreeMap<String, Port>,
    pub(crate) context: StreamTaskContext,
    pub(crate) progress: OperatorProgress,
    pub(crate) metrics: MetricsRecorder,
    pub(crate) entry_gate: watch::Receiver<bool>,
    pub(crate) entry_ack: mpsc::UnboundedSender<OperatorEntryAck>,
    pub(crate) data_gate: watch::Receiver<bool>,
    pub(crate) launch_cancel: CancellationToken,
}

pub(crate) fn spawn_operator_task(
    supervisor: &mut TaskSupervisor,
    inputs: OperatorTaskInputs,
) -> TaskId {
    let task_name = format!("operator:{}", inputs.node_id);
    supervisor.spawn_with_failure_signal(task_name, move |failure_signal| {
        run_operator_task(inputs, failure_signal.task_id())
    })
}

async fn run_operator_task(mut inputs: OperatorTaskInputs, task_id: TaskId) -> Result<()> {
    if !wait_for_gate(
        &mut inputs.entry_gate,
        &inputs.launch_cancel,
        inputs.context.job().cancellation(),
    )
    .await?
    {
        return Ok(());
    }
    let reset_result = match catch_unwind(AssertUnwindSafe(|| inputs.operator.reset())) {
        Ok(result) => result,
        Err(payload) => Err(CalcFlowError::TaskPanicked {
            task_id: task_id.as_u64(),
            message: panic_message(payload.as_ref()),
        }),
    };
    match reset_result {
        Ok(()) => inputs
            .entry_ack
            .send(OperatorEntryAck {
                node_id: inputs.node_id.clone(),
                result: Ok(()),
            })
            .map_err(|_| CalcFlowError::Internal {
                message: format!(
                    "operator {:?} entry acknowledgement was dropped",
                    inputs.node_id
                ),
            })?,
        Err(error) => {
            inputs
                .entry_ack
                .send(OperatorEntryAck {
                    node_id: inputs.node_id.clone(),
                    result: Err(error),
                })
                .map_err(|_| CalcFlowError::Internal {
                    message: format!(
                        "operator {:?} failed reset after its acknowledgement was dropped",
                        inputs.node_id
                    ),
                })?;
            return Ok(());
        }
    }
    if !wait_for_gate(
        &mut inputs.data_gate,
        &inputs.launch_cancel,
        inputs.context.job().cancellation(),
    )
    .await?
    {
        return Ok(());
    }
    let result = run_operator_loop(&mut inputs).await;
    if inputs.context.job().cancellation().is_cancelled()
        && matches!(result, Err(CalcFlowError::Cancelled { .. }))
    {
        Ok(())
    } else {
        result
    }
}

async fn wait_for_gate(
    gate: &mut watch::Receiver<bool>,
    launch_cancel: &CancellationToken,
    cancellation: &CancellationToken,
) -> Result<bool> {
    loop {
        if *gate.borrow() {
            return Ok(true);
        }
        tokio::select! {
            biased;
            () = launch_cancel.cancelled() => return Ok(false),
            () = cancellation.cancelled() => return Ok(false),
            result = gate.changed() => result.map_err(|_| CalcFlowError::Internal {
                message: "operator launch gate closed before release".into(),
            })?,
        }
    }
}

async fn run_operator_loop(inputs: &mut OperatorTaskInputs) -> Result<()> {
    if inputs.ingresses.is_empty() {
        return Err(CalcFlowError::Internal {
            message: format!("operator {:?} has no runtime ingress", inputs.node_id),
        });
    }
    let mut input_watermark = None;
    loop {
        if inputs
            .ingresses
            .values()
            .all(|ingress| ingress.saw_explicit_eof)
        {
            let cancellation = inputs.context.job().cancellation().clone();
            let context =
                StreamOperatorContext::new(inputs.context.job(), &inputs.node_id, input_watermark);
            let mut collector = ChannelStreamCollector::new(
                &inputs.node_id,
                &inputs.output_ports,
                &mut inputs.outputs,
                inputs.context.job().cancellation(),
                &inputs.progress,
                &inputs.metrics,
            );
            inputs.progress.record_on_end()?;
            tokio::select! {
                biased;
                result = inputs.operator.on_end(&context, &mut collector) => result?,
                () = cancellation.cancelled() => return Ok(()),
            }
            forward_control(
                &mut inputs.outputs,
                StreamMessage::end_of_input(),
                inputs.context.job(),
            )
            .await?;
            inputs.progress.mark_ended();
            return Ok(());
        }

        let received = tokio::select! {
            biased;
            () = inputs.context.job().cancellation().cancelled() => return Ok(()),
            received = receive_ready(&mut inputs.ingresses) => received?,
        };
        let (ingress_name, message) = received;
        let Some(message) = message else {
            if inputs.context.job().cancellation().is_cancelled() {
                return Ok(());
            }
            let edge = inputs.ingresses[&ingress_name].edge_id.clone();
            return Err(CalcFlowError::EdgeClosed { edge });
        };
        let cancellation = inputs.context.job().cancellation().clone();
        tokio::select! {
            biased;
            result = dispatch_message(inputs, &ingress_name, message, &mut input_watermark) => {
                result?;
            }
            () = cancellation.cancelled() => return Ok(()),
        }
    }
}

type ReceiveFuture<'a> =
    Pin<Box<dyn Future<Output = (String, Result<Option<StreamMessage>>)> + Send + 'a>>;

async fn receive_ready(
    ingresses: &mut BTreeMap<String, OperatorIngress>,
) -> Result<(String, Option<StreamMessage>)> {
    let receives = ingresses
        .iter_mut()
        .filter(|(_, ingress)| !ingress.saw_explicit_eof)
        .map(|(name, ingress)| {
            let name = name.clone();
            Box::pin(async move { (name, ingress.receiver.recv().await) }) as ReceiveFuture<'_>
        })
        .collect::<Vec<_>>();
    let ((name, result), _, _) = select_all(receives).await;
    Ok((name, result?))
}

async fn dispatch_message(
    inputs: &mut OperatorTaskInputs,
    ingress_name: &str,
    message: StreamMessage,
    input_watermark: &mut Option<EventTime>,
) -> Result<()> {
    match message.kind() {
        StreamMessageKind::Data => {
            let batch = message
                .as_data()
                .expect("data kind always carries a batch")
                .clone();
            let cost = EnvelopeCost::of_message(&StreamMessage::data(batch.clone()))?;
            inputs
                .metrics
                .record_operator_input(&inputs.node_id, cost)?;
            inputs.progress.record_input()?;
            let processing_timer = inputs.metrics.timer();
            let context =
                StreamOperatorContext::new(inputs.context.job(), &inputs.node_id, *input_watermark);
            let mut collector = ChannelStreamCollector::new(
                &inputs.node_id,
                &inputs.output_ports,
                &mut inputs.outputs,
                inputs.context.job().cancellation(),
                &inputs.progress,
                &inputs.metrics,
            );
            let result = inputs
                .operator
                .process_data(ingress_name, batch, &context, &mut collector)
                .await;
            inputs
                .progress
                .observe_datafusion_runtime(inputs.operator.datafusion_runtime_initialized());
            result?;
            inputs
                .metrics
                .record_operator_processing(&inputs.node_id, &processing_timer)
        }
        StreamMessageKind::Watermark if inputs.ingresses.len() == 1 => {
            let watermark = message
                .as_watermark()
                .expect("watermark kind always carries event time");
            let context =
                StreamOperatorContext::new(inputs.context.job(), &inputs.node_id, *input_watermark);
            let mut collector = ChannelStreamCollector::new(
                &inputs.node_id,
                &inputs.output_ports,
                &mut inputs.outputs,
                inputs.context.job().cancellation(),
                &inputs.progress,
                &inputs.metrics,
            );
            inputs
                .operator
                .on_watermark(watermark, &context, &mut collector)
                .await?;
            forward_control(&mut inputs.outputs, message, inputs.context.job()).await?;
            *input_watermark = Some(watermark);
            Ok(())
        }
        StreamMessageKind::Idle if inputs.ingresses.len() == 1 => {
            forward_control(&mut inputs.outputs, message, inputs.context.job()).await
        }
        StreamMessageKind::Watermark => Err(CalcFlowError::InvalidArgument {
            field: format!(
                "runtime.nodes.{}.ingress.{ingress_name}.watermark",
                inputs.node_id
            ),
            message: "multi-ingress watermark control is unavailable before M3; no downstream control was emitted"
                .into(),
        }),
        StreamMessageKind::Idle => Err(CalcFlowError::InvalidArgument {
            field: format!(
                "runtime.nodes.{}.ingress.{ingress_name}.idle",
                inputs.node_id
            ),
            message: "multi-ingress idle control is unavailable before M3; no downstream control was emitted"
                .into(),
        }),
        StreamMessageKind::Barrier => Err(CalcFlowError::InvalidArgument {
            field: format!(
                "runtime.nodes.{}.ingress.{ingress_name}.barrier",
                inputs.node_id
            ),
            message: "barrier control is unavailable before M5; no downstream control was emitted"
                .into(),
        }),
        StreamMessageKind::EndOfInput => {
            inputs
                .ingresses
                .get_mut(ingress_name)
                .expect("selected ingress remains registered")
                .saw_explicit_eof = true;
            Ok(())
        }
    }
}

async fn forward_control(
    outputs: &mut BTreeMap<String, Vec<EdgeSender>>,
    message: StreamMessage,
    context: &super::StreamJobContext,
) -> Result<()> {
    for senders in outputs.values_mut() {
        for sender in senders {
            tokio::select! {
                biased;
                () = context.cancellation().cancelled() => {
                    return Err(CalcFlowError::Cancelled {
                        run_id: context.job_id().to_string(),
                    });
                }
                result = sender.send(message.clone()) => result?,
            }
        }
    }
    Ok(())
}

pub(crate) struct ChannelStreamCollector<'a> {
    node_id: &'a str,
    output_ports: &'a BTreeMap<String, Port>,
    outputs: &'a mut BTreeMap<String, Vec<EdgeSender>>,
    cancellation: &'a CancellationToken,
    progress: &'a OperatorProgress,
    metrics: &'a MetricsRecorder,
}

impl<'a> ChannelStreamCollector<'a> {
    fn new(
        node_id: &'a str,
        output_ports: &'a BTreeMap<String, Port>,
        outputs: &'a mut BTreeMap<String, Vec<EdgeSender>>,
        cancellation: &'a CancellationToken,
        progress: &'a OperatorProgress,
        metrics: &'a MetricsRecorder,
    ) -> Self {
        Self {
            node_id,
            output_ports,
            outputs,
            cancellation,
            progress,
            metrics,
        }
    }
}

#[async_trait]
impl StreamCollector for ChannelStreamCollector<'_> {
    async fn emit(&mut self, port: &str, batch: Batch) -> Result<()> {
        let declared = self
            .output_ports
            .get(port)
            .ok_or_else(|| CalcFlowError::Compile {
                message: format!("node.{}.outputs.{port} is unknown", self.node_id),
            })?;
        declared.validate(&batch, &format!("node.{}.outputs.{port}", self.node_id))?;
        let message = StreamMessage::data(batch);
        let cost = EnvelopeCost::of_message(&message)?;
        let senders = self
            .outputs
            .get_mut(port)
            .ok_or_else(|| CalcFlowError::Internal {
                message: format!(
                    "node {:?} output {port:?} has no runtime routes",
                    self.node_id
                ),
            })?;
        if senders.is_empty() {
            return Err(CalcFlowError::Internal {
                message: format!(
                    "node {:?} output {port:?} has no runtime routes",
                    self.node_id
                ),
            });
        }
        for sender in senders.iter() {
            sender.validate_message(&message)?;
        }
        for sender in senders {
            tokio::select! {
                biased;
                () = self.cancellation.cancelled() => {
                    return Err(CalcFlowError::Cancelled {
                        run_id: "operator-task".into(),
                    });
                }
                result = sender.send(message.clone()) => result?,
            }
        }
        self.metrics.record_operator_output(self.node_id, cost)?;
        self.progress.record_output()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        any::Any,
        collections::BTreeMap,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        task::Poll,
    };

    use async_trait::async_trait;
    use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
    use parking_lot::Mutex;
    use tokio::sync::{mpsc, watch};

    use super::{
        OperatorEntryAck, OperatorIngress, OperatorProgress, OperatorTaskInputs, run_operator_task,
        spawn_operator_task,
    };
    use crate::{
        Batch, BatchKind, BatchMetadata, CalcFlowError, CancellationToken, EdgeBudget, EventTime,
        JsonMap, OperatorMetadata, Port, Result, StreamCollector, StreamJobContext, StreamMessage,
        StreamMessageKind, StreamOperator, StreamOperatorContext,
        pipeline::CompiledStreamOperator,
        runtime::streaming::supervisor::{TaskId, TaskSupervisor},
    };

    fn batch(source: &str, sequence: u64) -> Batch {
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![i64::try_from(sequence).unwrap()])) as _,
        )])
        .unwrap();
        Batch::table(
            vec![record],
            BatchMetadata::new(source, sequence, BTreeMap::new()).unwrap(),
        )
        .unwrap()
    }

    #[derive(Clone, Copy)]
    enum Behavior {
        Forward,
        BadPort,
        BadKind,
        EmitThenError,
        WatermarkError,
    }

    struct ProbeOperator {
        input_ports: Vec<Port>,
        output_ports: Vec<Port>,
        behavior: Behavior,
        watermarks: Arc<AtomicUsize>,
        ends: Arc<AtomicUsize>,
        observed: Arc<Mutex<Vec<(String, String, u64)>>>,
    }

    impl OperatorMetadata for ProbeOperator {
        fn name(&self) -> &'static str {
            "probe"
        }
        fn input_ports(&self) -> &[Port] {
            &self.input_ports
        }
        fn output_ports(&self) -> &[Port] {
            &self.output_ports
        }
        fn configuration(&self) -> JsonMap {
            JsonMap::new()
        }
    }

    #[async_trait]
    impl StreamOperator for ProbeOperator {
        async fn process_data(
            &mut self,
            ingress: &str,
            batch: Batch,
            _context: &StreamOperatorContext<'_>,
            output: &mut dyn StreamCollector,
        ) -> Result<()> {
            self.observed.lock().push((
                ingress.into(),
                batch.metadata().source().into(),
                batch.metadata().sequence(),
            ));
            let port = if matches!(self.behavior, Behavior::BadPort) {
                "missing"
            } else {
                "output"
            };
            let emitted = if matches!(self.behavior, Behavior::BadKind) {
                Batch::external(Arc::new(TestPayload), batch.metadata().clone()).unwrap()
            } else {
                batch
            };
            output.emit(port, emitted).await?;
            if matches!(self.behavior, Behavior::EmitThenError) {
                return Err(CalcFlowError::Operator {
                    node_id: "probe".into(),
                    message: "after emit".into(),
                });
            }
            Ok(())
        }

        async fn on_watermark(
            &mut self,
            _watermark: EventTime,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            self.watermarks.fetch_add(1, Ordering::SeqCst);
            if matches!(self.behavior, Behavior::WatermarkError) {
                Err(CalcFlowError::Operator {
                    node_id: "probe".into(),
                    message: "watermark".into(),
                })
            } else {
                Ok(())
            }
        }

        async fn on_end(
            &mut self,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            self.ends.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[derive(Clone, Copy)]
    enum PendingHandler {
        Data,
        Watermark,
        End,
    }

    #[derive(Clone)]
    enum HandlerResolution {
        Pending,
        ErrorWhen(Arc<AtomicBool>),
    }

    struct PendingHandlerOperator {
        input_ports: Vec<Port>,
        output_ports: Vec<Port>,
        handler: PendingHandler,
        resolution: HandlerResolution,
        entered: Arc<AtomicBool>,
        poll_dropped: Arc<AtomicBool>,
    }

    impl OperatorMetadata for PendingHandlerOperator {
        fn name(&self) -> &'static str {
            "pending-handler"
        }

        fn input_ports(&self) -> &[Port] {
            &self.input_ports
        }

        fn output_ports(&self) -> &[Port] {
            &self.output_ports
        }

        fn configuration(&self) -> JsonMap {
            JsonMap::new()
        }
    }

    struct PollDrop(Arc<AtomicBool>);

    impl Drop for PollDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    impl PendingHandlerOperator {
        async fn remain_pending(&self) -> Result<()> {
            self.entered.store(true, Ordering::SeqCst);
            let _guard = PollDrop(Arc::clone(&self.poll_dropped));
            std::future::poll_fn(|_| match &self.resolution {
                HandlerResolution::Pending => Poll::Pending,
                HandlerResolution::ErrorWhen(ready) if !ready.load(Ordering::SeqCst) => {
                    Poll::Pending
                }
                HandlerResolution::ErrorWhen(_) => Poll::Ready(Err(CalcFlowError::Operator {
                    node_id: "node".into(),
                    message: "handler-ready-error".into(),
                })),
            })
            .await
        }
    }

    #[async_trait]
    impl StreamOperator for PendingHandlerOperator {
        async fn process_data(
            &mut self,
            _ingress: &str,
            _batch: Batch,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            if matches!(self.handler, PendingHandler::Data) {
                self.remain_pending().await
            } else {
                Ok(())
            }
        }

        async fn on_watermark(
            &mut self,
            _watermark: EventTime,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            if matches!(self.handler, PendingHandler::Watermark) {
                self.remain_pending().await
            } else {
                Ok(())
            }
        }

        async fn on_end(
            &mut self,
            _context: &StreamOperatorContext<'_>,
            _output: &mut dyn StreamCollector,
        ) -> Result<()> {
            if matches!(self.handler, PendingHandler::End) {
                self.remain_pending().await
            } else {
                Ok(())
            }
        }
    }

    type PendingTask = std::pin::Pin<Box<dyn Future<Output = Result<()>> + Send + 'static>>;

    async fn handler_task(
        handler: PendingHandler,
        resolution: HandlerResolution,
        message: StreamMessage,
    ) -> (
        CancellationToken,
        Arc<AtomicBool>,
        Arc<AtomicBool>,
        mpsc::UnboundedReceiver<OperatorEntryAck>,
        PendingTask,
    ) {
        let cancellation = CancellationToken::new();
        let context =
            StreamJobContext::new(7, "fingerprint", JsonMap::new(), None, cancellation.clone());
        let metrics = super::MetricsRecorder::new(
            ["source->input".into(), "output->sink".into()],
            [],
            ["node".into()],
            [],
        );
        let (mut input_sender, input_receiver) =
            crate::runtime::streaming::channel::edge_channel_with_metrics(
                "source->input",
                EdgeBudget {
                    max_rows: 64,
                    max_bytes: 1 << 20,
                },
                metrics.clone(),
            )
            .unwrap();
        input_sender.send(message).await.unwrap();
        let (output_sender, _output_receiver) =
            crate::runtime::streaming::channel::edge_channel_with_metrics(
                "output->sink",
                EdgeBudget {
                    max_rows: 64,
                    max_bytes: 1 << 20,
                },
                metrics.clone(),
            )
            .unwrap();
        let output_port = Port::new("output", BatchKind::Table, false, None).unwrap();
        let entered = Arc::new(AtomicBool::new(false));
        let poll_dropped = Arc::new(AtomicBool::new(false));
        let operator = PendingHandlerOperator {
            input_ports: vec![Port::new("input", BatchKind::Table, true, None).unwrap()],
            output_ports: vec![output_port.clone()],
            handler,
            resolution,
            entered: Arc::clone(&entered),
            poll_dropped: Arc::clone(&poll_dropped),
        };
        let (_entry, entry_gate) = watch::channel(true);
        let (_data, data_gate) = watch::channel(true);
        let (entry_ack, ack) = mpsc::unbounded_channel();
        let task = Box::pin(run_operator_task(
            OperatorTaskInputs {
                node_id: "node".into(),
                operator: CompiledStreamOperator::External(Box::new(operator)),
                ingresses: BTreeMap::from([(
                    "input".into(),
                    OperatorIngress::new("source->input".into(), input_receiver),
                )]),
                outputs: BTreeMap::from([("output".into(), vec![output_sender])]),
                output_ports: BTreeMap::from([("output".into(), output_port)]),
                context: context.for_node("node").unwrap(),
                progress: OperatorProgress::default(),
                metrics,
                entry_gate,
                entry_ack,
                data_gate,
                launch_cancel: CancellationToken::new(),
            },
            TaskId::new(0),
        ));
        (cancellation, entered, poll_dropped, ack, task)
    }

    async fn pending_handler_task(
        handler: PendingHandler,
        message: StreamMessage,
    ) -> (
        CancellationToken,
        Arc<AtomicBool>,
        Arc<AtomicBool>,
        mpsc::UnboundedReceiver<OperatorEntryAck>,
        PendingTask,
    ) {
        handler_task(handler, HandlerResolution::Pending, message).await
    }

    struct Harness {
        supervisor: TaskSupervisor,
        cancellation: CancellationToken,
        inputs: BTreeMap<String, crate::EdgeSender>,
        outputs: Vec<crate::EdgeReceiver>,
        entry: watch::Sender<bool>,
        data: watch::Sender<bool>,
        ack: mpsc::UnboundedReceiver<OperatorEntryAck>,
        progress: OperatorProgress,
        metrics: super::MetricsRecorder,
    }

    #[derive(Debug)]
    struct TestPayload;

    impl crate::ExternalPayload for TestPayload {
        fn backend(&self) -> &'static str {
            "operator-task-test"
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

    fn harness(input_names: &[&str], branch_count: usize, behavior: Behavior) -> Harness {
        let output_port = Port::new("output", BatchKind::Table, false, None).unwrap();
        let operator = ProbeOperator {
            input_ports: input_names
                .iter()
                .map(|name| Port::new(name, BatchKind::Table, true, None).unwrap())
                .collect(),
            output_ports: vec![output_port.clone()],
            behavior,
            watermarks: Arc::new(AtomicUsize::new(0)),
            ends: Arc::new(AtomicUsize::new(0)),
            observed: Arc::new(Mutex::new(Vec::new())),
        };
        harness_with_operator(
            input_names,
            branch_count,
            CompiledStreamOperator::External(Box::new(operator)),
            output_port,
        )
    }

    fn harness_with_operator(
        input_names: &[&str],
        branch_count: usize,
        operator: CompiledStreamOperator,
        output_port: Port,
    ) -> Harness {
        let cancellation = CancellationToken::new();
        let context =
            StreamJobContext::new(7, "fingerprint", JsonMap::new(), None, cancellation.clone());
        let edge_ids = input_names
            .iter()
            .map(|name| format!("source->{name}"))
            .chain((0..branch_count).map(|branch| format!("output->{branch}")))
            .collect::<Vec<_>>();
        let metrics = super::MetricsRecorder::new(edge_ids, [], ["node".into()], []);
        let mut inputs = BTreeMap::new();
        let mut ingresses = BTreeMap::new();
        for name in input_names {
            let edge_id = format!("source->{name}");
            let (sender, receiver) = crate::runtime::streaming::channel::edge_channel_with_metrics(
                edge_id.clone(),
                EdgeBudget {
                    max_rows: 64,
                    max_bytes: 1 << 20,
                },
                metrics.clone(),
            )
            .unwrap();
            inputs.insert((*name).into(), sender);
            ingresses.insert((*name).into(), OperatorIngress::new(edge_id, receiver));
        }
        let mut output_senders = Vec::new();
        let mut outputs = Vec::new();
        for branch in 0..branch_count {
            let (sender, receiver) = crate::runtime::streaming::channel::edge_channel_with_metrics(
                format!("output->{branch}"),
                EdgeBudget {
                    max_rows: 64,
                    max_bytes: 1 << 20,
                },
                metrics.clone(),
            )
            .unwrap();
            output_senders.push(sender);
            outputs.push(receiver);
        }
        let (entry, entry_rx) = watch::channel(false);
        let (data, data_rx) = watch::channel(false);
        let (ack_tx, ack) = mpsc::unbounded_channel();
        let progress = OperatorProgress::default();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let task_context = context.for_node("node").unwrap();
        spawn_operator_task(
            &mut supervisor,
            OperatorTaskInputs {
                node_id: "node".into(),
                operator,
                ingresses,
                outputs: BTreeMap::from([("output".into(), output_senders)]),
                output_ports: BTreeMap::from([("output".into(), output_port)]),
                context: task_context,
                progress: progress.clone(),
                metrics: metrics.clone(),
                entry_gate: entry_rx,
                entry_ack: ack_tx,
                data_gate: data_rx,
                launch_cancel: CancellationToken::new(),
            },
        );
        Harness {
            supervisor,
            cancellation,
            inputs,
            outputs,
            entry,
            data,
            ack,
            progress,
            metrics,
        }
    }

    async fn start(harness: &mut Harness) {
        harness.entry.send(true).unwrap();
        let acknowledgement = harness.ack.recv().await.unwrap();
        acknowledgement.result.unwrap();
        harness.data.send(true).unwrap();
    }

    #[tokio::test]
    async fn cancellation_interrupts_a_pending_process_data_handler() {
        let (cancellation, entered, poll_dropped, _ack, mut task) =
            pending_handler_task(PendingHandler::Data, StreamMessage::data(batch("S", 0))).await;

        assert!(matches!(futures::poll!(task.as_mut()), Poll::Pending));
        assert!(entered.load(Ordering::SeqCst));

        cancellation.cancel();

        assert!(
            matches!(futures::poll!(task.as_mut()), Poll::Ready(Ok(()))),
            "operator task must stop when cancellation interrupts a pending handler"
        );
        assert!(poll_dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn cancellation_interrupts_a_pending_on_watermark_handler() {
        let (cancellation, entered, poll_dropped, _ack, mut task) = pending_handler_task(
            PendingHandler::Watermark,
            StreamMessage::watermark(EventTime::from_micros(9)),
        )
        .await;

        assert!(matches!(futures::poll!(task.as_mut()), Poll::Pending));
        assert!(entered.load(Ordering::SeqCst));

        cancellation.cancel();

        assert!(matches!(futures::poll!(task.as_mut()), Poll::Ready(Ok(()))));
        assert!(poll_dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn cancellation_interrupts_a_pending_on_end_handler() {
        let (cancellation, entered, poll_dropped, _ack, mut task) =
            pending_handler_task(PendingHandler::End, StreamMessage::end_of_input()).await;

        assert!(matches!(futures::poll!(task.as_mut()), Poll::Pending));
        assert!(entered.load(Ordering::SeqCst));

        cancellation.cancel();

        assert!(matches!(futures::poll!(task.as_mut()), Poll::Ready(Ok(()))));
        assert!(poll_dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn ready_handler_error_wins_over_same_poll_cancellation() {
        let error_ready = Arc::new(AtomicBool::new(false));
        let (cancellation, entered, _poll_dropped, _ack, mut task) = handler_task(
            PendingHandler::Data,
            HandlerResolution::ErrorWhen(Arc::clone(&error_ready)),
            StreamMessage::data(batch("S", 0)),
        )
        .await;

        assert!(matches!(futures::poll!(task.as_mut()), Poll::Pending));
        assert!(entered.load(Ordering::SeqCst));

        error_ready.store(true, Ordering::SeqCst);
        cancellation.cancel();

        let Poll::Ready(Err(CalcFlowError::Operator { message, .. })) =
            futures::poll!(task.as_mut())
        else {
            panic!("ready handler failure must win over same-poll cancellation");
        };
        assert_eq!(message, "handler-ready-error");
    }

    #[tokio::test]
    async fn ready_selection_preserves_each_ingress_fifo_and_ends_once() {
        let mut harness = harness(&["left", "right"], 1, Behavior::Forward);
        start(&mut harness).await;
        for (name, source) in [("left", "L"), ("right", "R")] {
            let sender = harness.inputs.get_mut(name).unwrap();
            sender
                .send(StreamMessage::data(batch(source, 0)))
                .await
                .unwrap();
            sender
                .send(StreamMessage::data(batch(source, 1)))
                .await
                .unwrap();
            sender.send(StreamMessage::end_of_input()).await.unwrap();
        }
        let report = harness.supervisor.join_all().await;
        assert!(report.errors.is_empty());
        let receiver = &mut harness.outputs[0];
        let mut per_source = BTreeMap::<String, Vec<u64>>::new();
        loop {
            let message = receiver.recv().await.unwrap().unwrap();
            if message.is_end_of_input() {
                break;
            }
            let batch = message.as_data().unwrap();
            per_source
                .entry(batch.metadata().source().into())
                .or_default()
                .push(batch.metadata().sequence());
        }
        assert_eq!(per_source["L"], [0, 1]);
        assert_eq!(per_source["R"], [0, 1]);
        assert_eq!(harness.progress.snapshot().input_batches, 4);
        assert_eq!(harness.progress.snapshot().on_end_calls, 1);
        assert!(harness.progress.snapshot().ended);
    }

    #[tokio::test]
    async fn unary_runtime_control_is_fifo_and_handler_precedes_watermark_forwarding() {
        let mut harness = harness(&["input"], 1, Behavior::Forward);
        start(&mut harness).await;
        let sender = harness.inputs.get_mut("input").unwrap();
        sender
            .send(StreamMessage::data(batch("S", 0)))
            .await
            .unwrap();
        sender
            .send(StreamMessage::watermark(EventTime::from_micros(9)))
            .await
            .unwrap();
        sender.send(StreamMessage::idle()).await.unwrap();
        sender.send(StreamMessage::end_of_input()).await.unwrap();
        let report = harness.supervisor.join_all().await;
        assert!(report.errors.is_empty());
        let receiver = &mut harness.outputs[0];
        let mut kinds = Vec::new();
        while let Some(message) = receiver.recv().await.unwrap() {
            kinds.push(message.kind());
            if message.is_end_of_input() {
                break;
            }
        }
        assert_eq!(
            kinds,
            [
                StreamMessageKind::Data,
                StreamMessageKind::Watermark,
                StreamMessageKind::Idle,
                StreamMessageKind::EndOfInput
            ]
        );
    }

    #[tokio::test]
    async fn watermark_error_and_multi_ingress_control_and_barrier_fail_closed() {
        for (names, message, behavior) in [
            (
                &["input"][..],
                StreamMessage::watermark(EventTime::from_micros(1)),
                Behavior::WatermarkError,
            ),
            (
                &["left", "right"][..],
                StreamMessage::idle(),
                Behavior::Forward,
            ),
            (
                &["left", "right"][..],
                StreamMessage::watermark(EventTime::from_micros(1)),
                Behavior::Forward,
            ),
            (
                &["input"][..],
                StreamMessage::barrier(crate::Epoch::INITIAL),
                Behavior::Forward,
            ),
        ] {
            let mut harness = harness(names, 1, behavior);
            start(&mut harness).await;
            harness
                .inputs
                .values_mut()
                .next()
                .unwrap()
                .send(message)
                .await
                .unwrap();
            let report = harness.supervisor.join_all().await;
            assert_eq!(report.errors.len(), 1);
            assert!(harness.outputs[0].recv().await.unwrap().is_none());
        }
    }

    #[tokio::test]
    async fn multi_ingress_control_and_barrier_use_exact_fail_closed_diagnostics() {
        for (names, ingress, message, expected_field, expected_message) in [
            (
                &["left", "right"][..],
                "left",
                StreamMessage::watermark(EventTime::from_micros(1)),
                "runtime.nodes.node.ingress.left.watermark",
                "multi-ingress watermark control is unavailable before M3; no downstream control was emitted",
            ),
            (
                &["left", "right"][..],
                "left",
                StreamMessage::idle(),
                "runtime.nodes.node.ingress.left.idle",
                "multi-ingress idle control is unavailable before M3; no downstream control was emitted",
            ),
            (
                &["input"][..],
                "input",
                StreamMessage::barrier(crate::Epoch::INITIAL),
                "runtime.nodes.node.ingress.input.barrier",
                "barrier control is unavailable before M5; no downstream control was emitted",
            ),
        ] {
            let watermarks = Arc::new(AtomicUsize::new(0));
            let ends = Arc::new(AtomicUsize::new(0));
            let observed = Arc::new(Mutex::new(Vec::new()));
            let output_port = Port::new("output", BatchKind::Table, false, None).unwrap();
            let operator = ProbeOperator {
                input_ports: names
                    .iter()
                    .map(|name| Port::new(name, BatchKind::Table, true, None).unwrap())
                    .collect(),
                output_ports: vec![output_port.clone()],
                behavior: Behavior::Forward,
                watermarks: Arc::clone(&watermarks),
                ends: Arc::clone(&ends),
                observed: Arc::clone(&observed),
            };
            let mut harness = harness_with_operator(
                names,
                1,
                CompiledStreamOperator::External(Box::new(operator)),
                output_port,
            );
            start(&mut harness).await;
            harness
                .inputs
                .get_mut(ingress)
                .unwrap()
                .send(message)
                .await
                .unwrap();

            let report = harness.supervisor.join_all().await;

            assert_eq!(report.errors.len(), 1);
            assert!(matches!(
                &report.errors[0].error,
                CalcFlowError::InvalidArgument { field, message }
                    if field == expected_field && message == expected_message
            ));
            assert_eq!(watermarks.load(Ordering::SeqCst), 0);
            assert_eq!(ends.load(Ordering::SeqCst), 0);
            assert!(observed.lock().is_empty());
            assert!(harness.outputs[0].recv().await.unwrap().is_none());
        }
    }

    #[tokio::test]
    async fn validation_is_zero_output_but_late_branch_close_keeps_earlier_prefix() {
        for behavior in [Behavior::BadPort, Behavior::BadKind] {
            let mut invalid = harness(&["input"], 2, behavior);
            start(&mut invalid).await;
            invalid
                .inputs
                .get_mut("input")
                .unwrap()
                .send(StreamMessage::data(batch("S", 0)))
                .await
                .unwrap();
            let report = invalid.supervisor.join_all().await;
            assert_eq!(report.errors.len(), 1);
            for receiver in &mut invalid.outputs {
                assert!(receiver.recv().await.unwrap().is_none());
            }
        }

        let schema_port = Port::new(
            "output",
            BatchKind::Table,
            false,
            Some(vec![datafusion::arrow::datatypes::Field::new(
                "different",
                datafusion::arrow::datatypes::DataType::Int64,
                true,
            )]),
        )
        .unwrap();
        let schema_probe = ProbeOperator {
            input_ports: vec![Port::new("input", BatchKind::Table, true, None).unwrap()],
            output_ports: vec![schema_port.clone()],
            behavior: Behavior::Forward,
            watermarks: Arc::new(AtomicUsize::new(0)),
            ends: Arc::new(AtomicUsize::new(0)),
            observed: Arc::new(Mutex::new(Vec::new())),
        };
        let mut schema = harness_with_operator(
            &["input"],
            1,
            CompiledStreamOperator::External(Box::new(schema_probe)),
            schema_port,
        );
        start(&mut schema).await;
        schema
            .inputs
            .get_mut("input")
            .unwrap()
            .send(StreamMessage::data(batch("S", 0)))
            .await
            .unwrap();
        assert_eq!(schema.supervisor.join_all().await.errors.len(), 1);
        assert!(schema.outputs[0].recv().await.unwrap().is_none());

        let mut partial = harness(&["input"], 2, Behavior::EmitThenError);
        start(&mut partial).await;
        partial.outputs[1].close();
        partial
            .inputs
            .get_mut("input")
            .unwrap()
            .send(StreamMessage::data(batch("S", 0)))
            .await
            .unwrap();
        let report = partial.supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
        let metrics = partial.metrics.snapshot();
        assert_eq!(metrics.edges["output->0"].input_batches, 1);
        assert_eq!(metrics.edges["output->1"].input_batches, 0);
        assert_eq!(metrics.nodes["node"].fully_fanned_out_batches, 0);
        let prefix = partial.outputs[0].recv().await.unwrap().unwrap();
        assert_eq!(prefix.kind(), StreamMessageKind::Data);
        assert_eq!(partial.progress.snapshot().fully_fanned_out_batches, 0);

        let mut emitted_then_failed = harness(&["input"], 2, Behavior::EmitThenError);
        start(&mut emitted_then_failed).await;
        emitted_then_failed
            .inputs
            .get_mut("input")
            .unwrap()
            .send(StreamMessage::data(batch("S", 0)))
            .await
            .unwrap();
        let report = emitted_then_failed.supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
        assert!(
            matches!(report.errors[0].error, CalcFlowError::Operator { ref message, .. } if message == "after emit")
        );
        for receiver in &mut emitted_then_failed.outputs {
            assert_eq!(
                receiver.recv().await.unwrap().unwrap().kind(),
                StreamMessageKind::Data
            );
        }
        assert_eq!(
            emitted_then_failed
                .progress
                .snapshot()
                .fully_fanned_out_batches,
            1
        );
    }

    #[tokio::test]
    async fn premature_sender_drop_is_edge_closed_not_end_of_input() {
        let mut harness = harness(&["input"], 1, Behavior::Forward);
        start(&mut harness).await;
        drop(harness.inputs.remove("input"));
        let report = harness.supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
        assert!(
            matches!(report.errors[0].error, CalcFlowError::EdgeClosed { ref edge } if edge == "source->input")
        );
        assert!(!harness.progress.snapshot().ended);
        assert!(harness.cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn premature_union_ingress_close_fails_the_whole_task() {
        let mut harness = harness(&["left", "right"], 1, Behavior::Forward);
        start(&mut harness).await;
        drop(harness.inputs.remove("right"));
        let report = harness.supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
        assert!(
            matches!(report.errors[0].error, CalcFlowError::EdgeClosed { ref edge } if edge == "source->right")
        );
        assert!(!harness.progress.snapshot().ended);
    }

    #[tokio::test]
    async fn convergence_marking_turns_closed_operator_ingress_into_a_wakeup() {
        let mut harness = harness(&["input"], 1, Behavior::Forward);
        start(&mut harness).await;

        harness.cancellation.cancel();
        drop(harness.inputs.remove("input"));

        let report = harness.supervisor.join_all().await;
        assert!(report.errors.is_empty(), "{report:?}");
        assert!(!harness.progress.snapshot().ended);
        assert!(harness.outputs[0].recv().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn expression_and_sql_reuse_one_lazy_task_owned_datafusion_runtime() {
        let operators = [
            (
                "input",
                CompiledStreamOperator::Expression(
                    crate::ExpressionOperator::new(
                        "calc",
                        "plus_one = value + 1",
                        Vec::new(),
                        None,
                        Vec::new(),
                    )
                    .unwrap(),
                ),
            ),
            (
                "events",
                CompiledStreamOperator::Sql(
                    crate::SqlOperator::new(
                        "project",
                        "SELECT value * 2 AS doubled FROM events",
                        vec!["events".into()],
                        Vec::new(),
                    )
                    .unwrap(),
                ),
            ),
        ];
        for (ingress, operator) in operators {
            assert!(!operator.datafusion_runtime_initialized());
            let output_port = Port::new("output", BatchKind::Table, false, None).unwrap();
            let mut harness = harness_with_operator(&[ingress], 1, operator, output_port);
            start(&mut harness).await;
            let sender = harness.inputs.get_mut(ingress).unwrap();
            sender
                .send(StreamMessage::data(batch("S", 0)))
                .await
                .unwrap();
            sender
                .send(StreamMessage::data(batch("S", 1)))
                .await
                .unwrap();
            sender.send(StreamMessage::end_of_input()).await.unwrap();
            let report = harness.supervisor.join_all().await;
            assert!(report.errors.is_empty(), "{report:?}");
            let mut data_count = 0;
            loop {
                let message = harness.outputs[0].recv().await.unwrap().unwrap();
                if message.is_end_of_input() {
                    break;
                }
                data_count += 1;
            }
            assert_eq!(data_count, 2);
            let progress = harness.progress.snapshot();
            assert_eq!(progress.input_batches, 2);
            assert_eq!(progress.fully_fanned_out_batches, 2);
            assert!(progress.datafusion_runtime_created);
        }

        let union = crate::UnionOperator::new(
            "union",
            vec![
                Port::new("left", BatchKind::Table, true, None).unwrap(),
                Port::new("right", BatchKind::Table, true, None).unwrap(),
            ],
        )
        .unwrap();
        assert!(!CompiledStreamOperator::Union(union).datafusion_runtime_initialized());
    }

    #[tokio::test]
    async fn array_only_external_graph_creates_zero_datafusion_runtimes() {
        let input_port = Port::new("input", BatchKind::Array, true, None).unwrap();
        let output_port = Port::new("output", BatchKind::Array, false, None).unwrap();
        let operator = CompiledStreamOperator::External(Box::new(ProbeOperator {
            input_ports: vec![input_port],
            output_ports: vec![output_port.clone()],
            behavior: Behavior::Forward,
            watermarks: Arc::new(AtomicUsize::new(0)),
            ends: Arc::new(AtomicUsize::new(0)),
            observed: Arc::new(Mutex::new(Vec::new())),
        }));
        assert!(!operator.datafusion_runtime_initialized());
        let mut harness = harness_with_operator(&["input"], 1, operator, output_port);
        start(&mut harness).await;
        let external = Batch::external(
            Arc::new(TestPayload),
            BatchMetadata::new("array", 0, BTreeMap::new()).unwrap(),
        )
        .unwrap();
        let sender = harness.inputs.get_mut("input").unwrap();
        sender.send(StreamMessage::data(external)).await.unwrap();
        sender.send(StreamMessage::end_of_input()).await.unwrap();

        let report = harness.supervisor.join_all().await;

        assert!(report.errors.is_empty(), "{report:?}");
        assert_eq!(
            harness.outputs[0].recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::Data
        );
        assert_eq!(
            harness.outputs[0].recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::EndOfInput
        );
        assert!(!harness.progress.snapshot().datafusion_runtime_created);
    }
}
