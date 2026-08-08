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
    context::{StreamTaskContext, wait_for_task_gate},
    metrics::MetricsRecorder,
    progress::{
        aggregate::{AggregateInput, MultiInputProgress, ProgressEmissionKind},
        prepare::BindingOrdinal,
    },
    supervisor::{TaskId, TaskSupervisor, panic_message},
};
use crate::{
    Batch, CalcFlowError, CancellationToken, EdgeBudget, EventTime, Port, Result, StreamCollector,
    StreamOperatorContext,
    operator::{LateMetricDelta, LateMetricSink, accumulate_late_metrics},
    pipeline::CompiledStreamOperator,
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
    pub(crate) late_rows: u64,
    pub(crate) affected_batches: u64,
    pub(crate) max_lateness_micros: Option<u64>,
    pub(crate) null_event_time_rows: u64,
    pub(crate) null_event_time_batches: u64,
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

impl LateMetricSink for OperatorProgress {
    fn record(&self, delta: LateMetricDelta) -> Result<()> {
        let mut progress = self.0.lock();
        let current = LateMetricDelta {
            late_rows: progress.late_rows,
            affected_batches: progress.affected_batches,
            max_lateness_micros: progress.max_lateness_micros,
            null_event_time_rows: progress.null_event_time_rows,
            null_event_time_batches: progress.null_event_time_batches,
        };
        let next = accumulate_late_metrics(current, delta)?;
        progress.late_rows = next.late_rows;
        progress.affected_batches = next.affected_batches;
        progress.max_lateness_micros = next.max_lateness_micros;
        progress.null_event_time_rows = next.null_event_time_rows;
        progress.null_event_time_batches = next.null_event_time_batches;
        Ok(())
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
    if !wait_for_task_gate(
        &mut inputs.entry_gate,
        &inputs.launch_cancel,
        inputs.context.job().cancellation(),
        "operator launch gate closed before release",
    )
    .await?
    {
        return Ok(());
    }
    if !reset_and_acknowledge(&mut inputs, task_id)? {
        return Ok(());
    }
    if !wait_for_task_gate(
        &mut inputs.data_gate,
        &inputs.launch_cancel,
        inputs.context.job().cancellation(),
        "operator launch gate closed before release",
    )
    .await?
    {
        return Ok(());
    }
    let result = run_operator_loop(&mut inputs).await;
    normalize_cancelled_result(&inputs, result)
}

fn reset_and_acknowledge(inputs: &mut OperatorTaskInputs, task_id: TaskId) -> Result<bool> {
    let reset_result = match catch_unwind(AssertUnwindSafe(|| inputs.operator.reset())) {
        Ok(result) => result,
        Err(payload) => Err(CalcFlowError::TaskPanicked {
            task_id: task_id.as_u64(),
            message: panic_message(payload.as_ref()),
        }),
    };
    let succeeded = reset_result.is_ok();
    let dropped_message = if succeeded {
        format!(
            "operator {:?} entry acknowledgement was dropped",
            inputs.node_id
        )
    } else {
        format!(
            "operator {:?} failed reset after its acknowledgement was dropped",
            inputs.node_id
        )
    };
    inputs
        .entry_ack
        .send(OperatorEntryAck {
            node_id: inputs.node_id.clone(),
            result: reset_result,
        })
        .map_err(|_| CalcFlowError::Internal {
            message: dropped_message,
        })?;
    Ok(succeeded)
}

fn normalize_cancelled_result(inputs: &OperatorTaskInputs, result: Result<()>) -> Result<()> {
    if inputs.context.job().cancellation().is_cancelled()
        && matches!(result, Err(CalcFlowError::Cancelled { .. }))
    {
        Ok(())
    } else {
        result
    }
}

async fn run_operator_loop(inputs: &mut OperatorTaskInputs) -> Result<()> {
    if inputs.ingresses.is_empty() {
        return Err(CalcFlowError::Internal {
            message: format!("operator {:?} has no runtime ingress", inputs.node_id),
        });
    }
    let mut input_progress = OperatorInputProgress::new(inputs.ingresses.keys());
    loop {
        if inputs
            .ingresses
            .values()
            .all(|ingress| ingress.saw_explicit_eof)
        {
            return finish_operator(inputs, input_progress.input_watermark()).await;
        }
        let Some((ingress_name, message)) = receive_operator_message(inputs).await? else {
            return Ok(());
        };
        dispatch_or_cancel(inputs, &ingress_name, message, &mut input_progress).await?;
    }
}

async fn finish_operator(
    inputs: &mut OperatorTaskInputs,
    input_watermark: Option<EventTime>,
) -> Result<()> {
    let cancellation = inputs.context.job().cancellation().clone();
    let output_budget = effective_output_budget(&inputs.outputs);
    let late_metrics: Arc<dyn LateMetricSink> = Arc::new(inputs.progress.clone());
    let context = StreamOperatorContext::for_task(
        inputs.context.job(),
        &inputs.node_id,
        input_watermark,
        output_budget,
        late_metrics,
    );
    let mut collector = ChannelStreamCollector::new(
        &inputs.node_id,
        inputs.context.job().job_id(),
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
    Ok(())
}

async fn receive_operator_message(
    inputs: &mut OperatorTaskInputs,
) -> Result<Option<(String, StreamMessage)>> {
    let received = tokio::select! {
        biased;
        () = inputs.context.job().cancellation().cancelled() => return Ok(None),
        received = receive_ready(&mut inputs.ingresses) => received?,
    };
    let (ingress_name, message) = received;
    match message {
        Some(message) => Ok(Some((ingress_name, message))),
        None if inputs.context.job().cancellation().is_cancelled() => Ok(None),
        None => Err(CalcFlowError::EdgeClosed {
            edge: inputs.ingresses[&ingress_name].edge_id.clone(),
        }),
    }
}

async fn dispatch_or_cancel(
    inputs: &mut OperatorTaskInputs,
    ingress_name: &str,
    message: StreamMessage,
    input_progress: &mut OperatorInputProgress,
) -> Result<()> {
    let cancellation = inputs.context.job().cancellation().clone();
    tokio::select! {
        biased;
        result = dispatch_message(inputs, ingress_name, message, input_progress) => result,
        () = cancellation.cancelled() => Ok(()),
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
    input_progress: &mut OperatorInputProgress,
) -> Result<()> {
    match message.kind() {
        StreamMessageKind::Data => {
            let watermark = input_progress.input_watermark();
            input_progress.evaluate(ingress_name, AggregateInput::Data)?;
            dispatch_data(inputs, ingress_name, message, watermark).await
        }
        StreamMessageKind::Watermark => {
            let previous = input_progress.input_watermark();
            let watermark = message
                .as_watermark()
                .expect("watermark kind always carries event time");
            let emissions =
                input_progress.evaluate(ingress_name, AggregateInput::Watermark(watermark))?;
            dispatch_progress_emissions(inputs, ingress_name, emissions, previous).await
        }
        StreamMessageKind::Idle => {
            let previous = input_progress.input_watermark();
            let emissions = input_progress.evaluate(ingress_name, AggregateInput::Idle)?;
            dispatch_progress_emissions(inputs, ingress_name, emissions, previous).await
        }
        StreamMessageKind::Barrier => Err(unsupported_control(
            inputs,
            ingress_name,
            "barrier",
            "barrier control is unavailable before M5; no downstream control was emitted",
        )),
        StreamMessageKind::EndOfInput => {
            let previous = input_progress.input_watermark();
            let emissions = input_progress.evaluate(ingress_name, AggregateInput::End)?;
            dispatch_progress_emissions(inputs, ingress_name, emissions, previous).await?;
            inputs
                .ingresses
                .get_mut(ingress_name)
                .expect("selected ingress remains registered")
                .saw_explicit_eof = true;
            Ok(())
        }
    }
}

async fn dispatch_data(
    inputs: &mut OperatorTaskInputs,
    ingress_name: &str,
    message: StreamMessage,
    input_watermark: Option<EventTime>,
) -> Result<()> {
    let batch = message
        .as_data()
        .expect("data kind always carries a batch")
        .clone();
    let cost = EnvelopeCost::of_message(&message)?;
    inputs
        .metrics
        .record_operator_input(&inputs.node_id, cost)?;
    inputs.progress.record_input()?;
    let processing_timer = inputs.metrics.timer();
    let output_budget = effective_output_budget(&inputs.outputs);
    let late_metrics: Arc<dyn LateMetricSink> = Arc::new(inputs.progress.clone());
    let context = StreamOperatorContext::for_task(
        inputs.context.job(),
        &inputs.node_id,
        input_watermark,
        output_budget,
        late_metrics,
    );
    let mut collector = ChannelStreamCollector::new(
        &inputs.node_id,
        inputs.context.job().job_id(),
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
        .observe_operator_window_metrics(&inputs.node_id, &inputs.progress.snapshot())?;
    inputs
        .metrics
        .record_operator_processing(&inputs.node_id, &processing_timer)
}

async fn dispatch_watermark(
    inputs: &mut OperatorTaskInputs,
    _ingress_name: &str,
    message: StreamMessage,
    input_watermark: &mut Option<EventTime>,
) -> Result<()> {
    let watermark = message
        .as_watermark()
        .expect("watermark kind always carries event time");
    let output_budget = effective_output_budget(&inputs.outputs);
    let late_metrics: Arc<dyn LateMetricSink> = Arc::new(inputs.progress.clone());
    let context = StreamOperatorContext::for_task(
        inputs.context.job(),
        &inputs.node_id,
        *input_watermark,
        output_budget,
        late_metrics,
    );
    let mut collector = ChannelStreamCollector::new(
        &inputs.node_id,
        inputs.context.job().job_id(),
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

fn effective_output_budget(outputs: &BTreeMap<String, Vec<EdgeSender>>) -> EdgeBudget {
    outputs
        .values()
        .flatten()
        .map(EdgeSender::budget)
        .reduce(|left, right| EdgeBudget {
            max_rows: left.max_rows.min(right.max_rows),
            max_bytes: left.max_bytes.min(right.max_bytes),
        })
        .unwrap_or_default()
}

async fn dispatch_idle(
    inputs: &mut OperatorTaskInputs,
    _ingress_name: &str,
    message: StreamMessage,
) -> Result<()> {
    forward_control(&mut inputs.outputs, message, inputs.context.job()).await
}

async fn dispatch_progress_emissions(
    inputs: &mut OperatorTaskInputs,
    ingress_name: &str,
    emissions: Vec<ProgressEmissionKind>,
    mut previous_watermark: Option<EventTime>,
) -> Result<()> {
    for emission in emissions {
        match emission {
            ProgressEmissionKind::Watermark(watermark) => {
                dispatch_watermark(
                    inputs,
                    ingress_name,
                    StreamMessage::watermark(watermark),
                    &mut previous_watermark,
                )
                .await?;
            }
            ProgressEmissionKind::Idle => {
                dispatch_idle(inputs, ingress_name, StreamMessage::idle()).await?;
            }
            ProgressEmissionKind::EndOfInput => {}
        }
    }
    Ok(())
}

struct OperatorInputProgress {
    ordinal_by_ingress: BTreeMap<String, BindingOrdinal>,
    aggregate: MultiInputProgress,
}

impl OperatorInputProgress {
    fn new<'a>(ingresses: impl IntoIterator<Item = &'a String>) -> Self {
        let ordinal_by_ingress = ingresses
            .into_iter()
            .enumerate()
            .map(|(index, ingress)| {
                (
                    ingress.clone(),
                    BindingOrdinal::new(
                        u64::try_from(index).expect("ingress count fits the binding ordinal"),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        Self {
            aggregate: MultiInputProgress::new(ordinal_by_ingress.values().copied()),
            ordinal_by_ingress,
        }
    }

    fn evaluate(
        &mut self,
        ingress: &str,
        input: AggregateInput,
    ) -> Result<Vec<ProgressEmissionKind>> {
        let ordinal =
            *self
                .ordinal_by_ingress
                .get(ingress)
                .ok_or_else(|| CalcFlowError::Internal {
                    message: format!("unknown operator ingress {ingress:?}"),
                })?;
        self.aggregate
            .evaluate(ordinal, input)
            .map(|emissions| {
                emissions
                    .into_iter()
                    .map(|emission| emission.kind)
                    .collect()
            })
            .map_err(super::progress::types::ProgressFailure::into_existing_error)
    }

    const fn input_watermark(&self) -> Option<EventTime> {
        self.aggregate.last_emitted_watermark()
    }
}

fn unsupported_control(
    inputs: &OperatorTaskInputs,
    ingress_name: &str,
    kind: &str,
    message: &str,
) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!(
            "runtime.nodes.{}.ingress.{ingress_name}.{kind}",
            inputs.node_id
        ),
        message: message.into(),
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
    job_id: u64,
    output_ports: &'a BTreeMap<String, Port>,
    outputs: &'a mut BTreeMap<String, Vec<EdgeSender>>,
    cancellation: &'a CancellationToken,
    progress: &'a OperatorProgress,
    metrics: &'a MetricsRecorder,
}

impl<'a> ChannelStreamCollector<'a> {
    fn new(
        node_id: &'a str,
        job_id: u64,
        output_ports: &'a BTreeMap<String, Port>,
        outputs: &'a mut BTreeMap<String, Vec<EdgeSender>>,
        cancellation: &'a CancellationToken,
        progress: &'a OperatorProgress,
        metrics: &'a MetricsRecorder,
    ) -> Self {
        Self {
            node_id,
            job_id,
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
        let message = validate_emission(self.node_id, self.output_ports, port, batch)?;
        let cost = EnvelopeCost::of_message(&message)?;
        let senders = self
            .outputs
            .get_mut(port)
            .ok_or_else(|| runtime_routes_error(self.node_id, port))?;
        validate_senders(senders, &message, self.node_id, port)?;
        send_emission(senders, message, self.cancellation, self.job_id).await?;
        self.metrics.record_operator_output(self.node_id, cost)?;
        self.progress.record_output()
    }
}

fn validate_emission(
    node_id: &str,
    output_ports: &BTreeMap<String, Port>,
    port: &str,
    batch: Batch,
) -> Result<StreamMessage> {
    let declared = output_ports
        .get(port)
        .ok_or_else(|| CalcFlowError::Compile {
            message: format!("node.{node_id}.outputs.{port} is unknown"),
        })?;
    declared.validate(&batch, &format!("node.{node_id}.outputs.{port}"))?;
    Ok(StreamMessage::data(batch))
}

fn validate_senders(
    senders: &[EdgeSender],
    message: &StreamMessage,
    node_id: &str,
    port: &str,
) -> Result<()> {
    if senders.is_empty() {
        return Err(runtime_routes_error(node_id, port));
    }
    for sender in senders {
        sender.validate_message(message)?;
    }
    Ok(())
}

async fn send_emission(
    senders: &mut [EdgeSender],
    message: StreamMessage,
    cancellation: &CancellationToken,
    job_id: u64,
) -> Result<()> {
    for sender in senders {
        tokio::select! {
            biased;
            () = cancellation.cancelled() => {
                return Err(CalcFlowError::Cancelled {
                    run_id: job_id.to_string(),
                });
            }
            result = sender.send(message.clone()) => result?,
        }
    }
    Ok(())
}

fn runtime_routes_error(node_id: &str, port: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("node {node_id:?} output {port:?} has no runtime routes"),
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
        time::Duration,
    };

    use async_trait::async_trait;
    use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
    use parking_lot::Mutex;
    use tokio::sync::{mpsc, watch};

    use super::{
        OperatorEntryAck, OperatorIngress, OperatorProgress, OperatorTaskInputs, run_operator_task,
        send_emission, spawn_operator_task,
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

    #[test]
    fn window_metrics_accumulate_across_contexts_and_mirror_to_runtime_metrics() {
        let job = StreamJobContext::new(
            7,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let progress = OperatorProgress::default();
        let first = StreamOperatorContext::for_task(
            &job,
            "window",
            None,
            EdgeBudget::default(),
            Arc::new(progress.clone()),
        );
        first
            .record_late_rows(2, Some(Duration::from_micros(5)))
            .unwrap();
        let second = StreamOperatorContext::for_task(
            &job,
            "window",
            Some(EventTime::from_micros(10)),
            EdgeBudget::default(),
            Arc::new(progress.clone()),
        );
        second.record_window_metrics(3, Some(7), 4).unwrap();

        let snapshot = progress.snapshot();
        assert_eq!(snapshot.late_rows, 5);
        assert_eq!(snapshot.affected_batches, 2);
        assert_eq!(snapshot.max_lateness_micros, Some(7));
        assert_eq!(snapshot.null_event_time_rows, 4);
        assert_eq!(snapshot.null_event_time_batches, 1);

        let metrics =
            crate::runtime::streaming::metrics::MetricsRecorder::new([], [], ["window".into()], []);
        metrics
            .observe_operator_window_metrics("window", &snapshot)
            .unwrap();
        let node = &metrics.snapshot().nodes["window"];
        assert_eq!(node.late_rows, 5);
        assert_eq!(node.affected_batches, 2);
        assert_eq!(node.max_lateness_micros, Some(7));
        assert_eq!(node.null_event_time_rows, 4);
        assert_eq!(node.null_event_time_batches, 1);
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
            [
                (
                    "source->input".into(),
                    EdgeBudget {
                        max_rows: 64,
                        max_bytes: 1 << 20,
                    },
                ),
                (
                    "output->sink".into(),
                    EdgeBudget {
                        max_rows: 64,
                        max_bytes: 1 << 20,
                    },
                ),
            ],
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
        let metrics = super::MetricsRecorder::new(
            edge_ids.into_iter().map(|edge_id| {
                (
                    edge_id,
                    EdgeBudget {
                        max_rows: 64,
                        max_bytes: 1 << 20,
                    },
                )
            }),
            [],
            ["node".into()],
            [],
        );
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
    async fn cancelled_emission_uses_job_id_before_outer_normalization() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let (sender, _receiver) = crate::edge_channel(
            "node.output->sink.output",
            EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
        )
        .unwrap();

        let error = send_emission(
            &mut [sender],
            StreamMessage::data(batch("S", 0)),
            &cancellation,
            47,
        )
        .await
        .unwrap_err();

        assert!(matches!(
            error,
            CalcFlowError::Cancelled { ref run_id } if run_id == "47"
        ));
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
    async fn watermark_error_and_barrier_fail_closed() {
        for (names, message, behavior) in [
            (
                &["input"][..],
                StreamMessage::watermark(EventTime::from_micros(1)),
                Behavior::WatermarkError,
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
    async fn barrier_uses_exact_fail_closed_diagnostics() {
        for (names, ingress, message, expected_field, expected_message) in [(
            &["input"][..],
            "input",
            StreamMessage::barrier(crate::Epoch::INITIAL),
            "runtime.nodes.node.ingress.input.barrier",
            "barrier control is unavailable before M5; no downstream control was emitted",
        )] {
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
    async fn multi_ingress_watermark_and_idle_are_aggregated_once() {
        let mut harness = harness(&["left", "right"], 2, Behavior::Forward);
        start(&mut harness).await;
        harness
            .inputs
            .get_mut("left")
            .unwrap()
            .send(StreamMessage::watermark(EventTime::from_micros(9)))
            .await
            .unwrap();
        harness
            .inputs
            .get_mut("right")
            .unwrap()
            .send(StreamMessage::watermark(EventTime::from_micros(7)))
            .await
            .unwrap();
        let watermark = harness.outputs[0].recv().await.unwrap().unwrap();
        assert_eq!(watermark.as_watermark(), Some(EventTime::from_micros(7)));

        for ingress in ["left", "right"] {
            harness
                .inputs
                .get_mut(ingress)
                .unwrap()
                .send(StreamMessage::idle())
                .await
                .unwrap();
        }
        assert!(harness.outputs[0].recv().await.unwrap().unwrap().is_idle());
        for ingress in ["left", "right"] {
            harness
                .inputs
                .get_mut(ingress)
                .unwrap()
                .send(StreamMessage::end_of_input())
                .await
                .unwrap();
        }
        let report = harness.supervisor.join_all().await;
        assert!(report.errors.is_empty());
        assert!(
            harness.outputs[0]
                .recv()
                .await
                .unwrap()
                .unwrap()
                .is_end_of_input()
        );
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
