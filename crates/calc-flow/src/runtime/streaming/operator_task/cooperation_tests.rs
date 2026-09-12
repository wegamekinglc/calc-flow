use std::{
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Wake, Waker},
};

use datafusion::arrow::{
    array::{Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};

use super::*;
use crate::{
    BatchKind, BatchMetadata, ExpressionOperator, JsonMap, OperatorMetadata, RollingOperator,
    StreamJobContext, StreamOperator, runtime::streaming::channel::edge_channel_with_metrics,
};

struct TracedNative {
    inner: Box<dyn StreamOperator>,
    trace: Arc<Mutex<Vec<String>>>,
    metrics: MetricsRecorder,
}

impl OperatorMetadata for TracedNative {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn input_ports(&self) -> &[Port] {
        self.inner.input_ports()
    }

    fn output_ports(&self) -> &[Port] {
        self.inner.output_ports()
    }

    fn configuration(&self) -> JsonMap {
        self.inner.configuration()
    }
}

#[async_trait]
impl StreamOperator for TracedNative {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let sequence = batch.metadata().sequence();
        let queued = self.metrics.snapshot().edges["sink"].channel.queue_depth;
        self.trace
            .lock()
            .push(format!("{}:data:{sequence}:sink:{queued}", self.name()));
        self.inner
            .process_data(ingress, batch, context, output)
            .await?;
        self.trace
            .lock()
            .push(format!("{}:data-committed:{sequence}", self.name()));
        Ok(())
    }

    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.inner.on_watermark(watermark, context, output).await?;
        self.trace
            .lock()
            .push(format!("{}:watermark-committed", self.name()));
        Ok(())
    }

    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.inner.on_end(context, output).await?;
        self.trace.lock().push(format!("{}:ended", self.name()));
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
    ]))
}

fn batch(start: usize, rows: usize, sequence: u64) -> Batch {
    let times = (start..start + rows)
        .map(|value| i64::try_from(value).unwrap())
        .collect::<Vec<_>>();
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
            Arc::new(StringArray::from(vec!["S"; rows])),
            Arc::new(UInt64Array::from_iter_values(
                (start..start + rows).map(|value| u64::try_from(value).unwrap()),
            )),
            Arc::new(Float64Array::from_iter_values(
                (start..start + rows).map(|value| f64::from(u32::try_from(value).unwrap())),
            )),
        ],
    )
    .unwrap();
    Batch::table(
        vec![record],
        BatchMetadata::new("source", sequence, BTreeMap::new()).unwrap(),
    )
    .unwrap()
}

fn native_operators() -> [CompiledStreamOperator; 2] {
    let spec = serde_json::from_value(serde_json::json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["sequence"],
        "outputs": [{"kind": "mean", "primitive_version": 1, "input": "price",
            "output": "mean", "min_periods": 1, "frame": {"kind": "rows", "size": 20}}],
        "allowed_lateness_micros": 0, "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap();
    let rolling = RollingOperator::new("rolling", schema(), spec).unwrap();
    let output = rolling.output_ports()[0].schema().unwrap().clone();
    let project = ExpressionOperator::new(
        "project",
        "",
        vec![
            "ts".into(),
            "symbol".into(),
            "sequence".into(),
            "price".into(),
            "mean".into(),
        ],
        None,
        vec![],
    )
    .unwrap()
    .with_ports(
        Port::with_schema_ref("input", BatchKind::Table, true, Some(output.clone())).unwrap(),
        Port::with_schema_ref("output", BatchKind::Table, false, Some(output)).unwrap(),
    )
    .unwrap();
    [
        CompiledStreamOperator::Rolling(rolling),
        CompiledStreamOperator::Expression(project),
    ]
}

fn traced_inputs(
    operator: CompiledStreamOperator,
    receiver: EdgeReceiver,
    sender: EdgeSender,
    context: &StreamJobContext,
    trace: &Arc<Mutex<Vec<String>>>,
    metrics: &MetricsRecorder,
    entry_ack: &mpsc::UnboundedSender<OperatorEntryAck>,
) -> OperatorTaskInputs {
    let inner: Box<dyn StreamOperator> = match operator {
        CompiledStreamOperator::Rolling(operator) => Box::new(operator),
        CompiledStreamOperator::Expression(operator) => Box::new(operator),
        _ => unreachable!("fixture contains two native operators"),
    };
    let node_id = inner.name().to_owned();
    let output_ports = [("output".into(), inner.output_ports()[0].clone())].into();
    OperatorTaskInputs {
        entity_work: None,
        context: context.for_node(&node_id).unwrap(),
        node_id,
        operator: CompiledStreamOperator::External(Box::new(TracedNative {
            inner,
            trace: trace.clone(),
            metrics: metrics.clone(),
        })),
        checkpoint_capability: OperatorCheckpointCapability::Stateless,
        ingresses: [(
            "input".into(),
            OperatorIngress::new(receiver.edge().into(), receiver),
        )]
        .into(),
        outputs: [("output".into(), vec![sender])].into(),
        output_ports,
        progress: OperatorProgress::default(),
        metrics: metrics.clone(),
        entry_gate: watch::channel(true).1,
        entry_ack: entry_ack.clone(),
        data_gate: watch::channel(true).1,
        launch_cancel: CancellationToken::new(),
        checkpoint: None,
        restore: None,
    }
}

struct Fixture {
    first: OperatorTaskInputs,
    second: OperatorTaskInputs,
    cooperation: OperatorCooperation,
    sink: EdgeReceiver,
    trace: Arc<Mutex<Vec<String>>>,
    _entry_acks: mpsc::UnboundedReceiver<OperatorEntryAck>,
}

async fn fixture(messages: Vec<StreamMessage>) -> Fixture {
    let budget = EdgeBudget::new(256, 1 << 20).unwrap();
    let metrics = MetricsRecorder::new(
        ["source", "internal", "sink"].map(|name| (name.into(), budget)),
        [],
        ["rolling".into(), "project".into()],
        [],
    );
    let (mut source, first_input) =
        edge_channel_with_metrics("source", budget, metrics.clone()).unwrap();
    let (first_output, second_input) =
        edge_channel_with_metrics("internal", budget, metrics.clone()).unwrap();
    let (second_output, sink) = edge_channel_with_metrics("sink", budget, metrics.clone()).unwrap();
    for message in messages {
        source.send(message).await.unwrap();
    }
    source.send(StreamMessage::end_of_input()).await.unwrap();
    let context = StreamJobContext::new(
        7,
        "cooperation-fixture",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let trace = Arc::new(Mutex::new(Vec::new()));
    let (entry_ack, entry_acks) = mpsc::unbounded_channel();
    let [first, second] = native_operators();
    let cooperation = pair_cooperation(&first, &second);
    Fixture {
        first: traced_inputs(
            first,
            first_input,
            first_output,
            &context,
            &trace,
            &metrics,
            &entry_ack,
        ),
        second: traced_inputs(
            second,
            second_input,
            second_output,
            &context,
            &trace,
            &metrics,
            &entry_ack,
        ),
        cooperation,
        sink,
        trace,
        _entry_acks: entry_acks,
    }
}

#[tokio::test]
async fn native_watermark_flush_projects_and_forwards_before_next_large_data() {
    let mut fixture = fixture(vec![
        StreamMessage::data(batch(0, 1, 0)),
        StreamMessage::watermark(EventTime::from_micros(0)),
        StreamMessage::data(batch(1, 64, 1)),
        StreamMessage::watermark(EventTime::from_micros(64)),
    ])
    .await;
    let (second, first) = tokio::join!(
        biased;
        run_operator_task_with_cooperation(fixture.second, TaskId::new(1), fixture.cooperation),
        run_operator_task_with_cooperation(fixture.first, TaskId::new(0), fixture.cooperation)
    );
    first.unwrap();
    second.unwrap();
    let mut kinds = Vec::new();
    let mut rows = 0;
    while let Some(message) = fixture.sink.recv().await.unwrap() {
        kinds.push(message.kind());
        rows += message.as_data().map_or(0, Batch::num_rows);
    }
    assert_eq!(rows, 65);
    assert_eq!(
        kinds,
        [
            StreamMessageKind::Data,
            StreamMessageKind::Watermark,
            StreamMessageKind::Data,
            StreamMessageKind::Watermark,
            StreamMessageKind::EndOfInput
        ]
    );
    let trace = fixture.trace.lock();
    let rolling_commit = trace
        .iter()
        .position(|event| event == "rolling:watermark-committed")
        .unwrap();
    let projection_start = trace
        .iter()
        .position(|event| event.starts_with("project:data:0:"))
        .unwrap();
    assert!(
        rolling_commit < projection_start,
        "native commit must precede handoff: {trace:?}"
    );
    assert!(
        trace.iter().any(|event| event == "rolling:data:1:sink:2"),
        "small Data and watermark must both be enqueued before the next rolling callback: {trace:?}"
    );
    assert_eq!(
        trace
            .iter()
            .filter(|event| event.ends_with(":ended"))
            .count(),
        2
    );
}

#[tokio::test]
async fn consecutive_native_data_dispatches_cooperate_after_at_most_two_callbacks() {
    let fixture = fixture(
        (0..8)
            .map(|row| StreamMessage::data(batch(row, 1, u64::try_from(row).unwrap())))
            .collect(),
    )
    .await;
    let trace = fixture.trace;
    let mut first = Some(Box::pin(run_operator_task_with_cooperation(
        fixture.first,
        TaskId::new(0),
        fixture.cooperation,
    )));
    let mut second = Some(Box::pin(run_operator_task_with_cooperation(
        fixture.second,
        TaskId::new(1),
        fixture.cooperation,
    )));
    let data_count = || {
        trace
            .lock()
            .iter()
            .filter(|event| event.starts_with("rolling:data:"))
            .count()
    };
    let mut maximum = 0;
    let mut polls = 0;
    while first.is_some() || second.is_some() {
        let before = data_count();
        std::future::poll_fn(|cx| {
            for branch in [&mut second, &mut first] {
                if let Some(future) = branch.as_mut()
                    && let Poll::Ready(result) = future.as_mut().poll(cx)
                {
                    result.unwrap();
                    *branch = None;
                }
            }
            Poll::Ready(())
        })
        .await;
        let completed = data_count() - before;
        assert!(
            completed <= 2,
            "a driver poll consumed {completed} data callbacks"
        );
        maximum = maximum.max(completed);
        polls += 1;
        assert!(polls <= 24, "bounded fixture did not finish");
    }
    assert_eq!(data_count(), 8);
    assert_eq!(
        maximum, 2,
        "ready non-emitting Data should share one bounded turn"
    );
    assert_eq!(
        trace
            .lock()
            .iter()
            .filter(|event| event.ends_with(":ended"))
            .count(),
        2
    );
}

#[derive(Default)]
struct WakeCounter(AtomicUsize);

impl Wake for WakeCounter {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct OpenNativeTask {
    task: OperatorTaskInputs,
    source: EdgeSender,
    sink: EdgeReceiver,
    cooperation: OperatorCooperation,
    cancellation: CancellationToken,
    trace: Arc<Mutex<Vec<String>>>,
    _entry_acks: mpsc::UnboundedReceiver<OperatorEntryAck>,
}

fn open_native_task(projection: bool) -> OpenNativeTask {
    let budget = EdgeBudget::new(256, 1 << 20).unwrap();
    let metrics = MetricsRecorder::new(
        ["source", "sink"].map(|name| (name.into(), budget)),
        [],
        ["rolling".into(), "project".into()],
        [],
    );
    let (source, input) = edge_channel_with_metrics("source", budget, metrics.clone()).unwrap();
    let (output, sink) = edge_channel_with_metrics("sink", budget, metrics.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let context = StreamJobContext::new(
        7,
        "pending-cooperation-fixture",
        JsonMap::new(),
        None,
        cancellation.clone(),
    );
    let [rolling, expression] = native_operators();
    let cooperation = pair_cooperation(&rolling, &expression);
    let trace = Arc::new(Mutex::new(Vec::new()));
    let (entry_ack, entry_acks) = mpsc::unbounded_channel();
    let task = traced_inputs(
        if projection { expression } else { rolling },
        input,
        output,
        &context,
        &trace,
        &metrics,
        &entry_ack,
    );
    OpenNativeTask {
        task,
        source,
        sink,
        cooperation,
        cancellation,
        trace,
        _entry_acks: entry_acks,
    }
}

#[test]
fn exhausted_data_budget_parks_on_empty_ingress_without_self_wake() {
    let mut fixture = open_native_task(false);
    for row in 0..2 {
        futures::executor::block_on(fixture.source.send(StreamMessage::data(batch(
            row,
            1,
            u64::try_from(row).unwrap(),
        ))))
        .unwrap();
    }
    let wake_counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(wake_counter.clone());
    let mut context = Context::from_waker(&waker);
    let mut task = Box::pin(run_operator_task_with_cooperation(
        fixture.task,
        TaskId::new(0),
        fixture.cooperation,
    ));
    assert!(task.as_mut().poll(&mut context).is_pending());
    assert_eq!(
        fixture
            .trace
            .lock()
            .iter()
            .filter(|event| event.starts_with("rolling:data:"))
            .count(),
        2
    );
    assert_eq!(
        wake_counter.0.load(Ordering::SeqCst),
        0,
        "the empty ingress must provide the next wake; a completed budget must not self-wake"
    );
    fixture.cancellation.cancel();
    assert!(matches!(
        task.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
}

#[test]
fn natural_receive_pending_resets_a_partially_used_data_budget() {
    let mut fixture = open_native_task(false);
    futures::executor::block_on(fixture.source.send(StreamMessage::data(batch(0, 1, 0)))).unwrap();
    let wake_counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(wake_counter.clone());
    let mut context = Context::from_waker(&waker);
    let mut task = Box::pin(run_operator_task_with_cooperation(
        fixture.task,
        TaskId::new(0),
        fixture.cooperation,
    ));
    assert!(task.as_mut().poll(&mut context).is_pending());
    assert_eq!(wake_counter.0.load(Ordering::SeqCst), 0);

    for message in [
        StreamMessage::data(batch(1, 1, 1)),
        StreamMessage::watermark(EventTime::from_micros(1)),
    ] {
        futures::executor::block_on(fixture.source.send(message)).unwrap();
    }
    assert!(
        wake_counter.0.load(Ordering::SeqCst) > 0,
        "new input must wake a naturally parked task"
    );
    wake_counter.0.store(0, Ordering::SeqCst);
    assert!(task.as_mut().poll(&mut context).is_pending());
    assert!(
        fixture
            .trace
            .lock()
            .iter()
            .any(|event| event == "rolling:watermark-committed"),
        "a new Data+watermark request must not spend the previous poll's remaining budget"
    );
    assert_eq!(wake_counter.0.load(Ordering::SeqCst), 0);
    let output = futures::executor::block_on(fixture.sink.recv())
        .unwrap()
        .unwrap();
    let record = &output.as_data().unwrap().table_payload().unwrap().batches()[0];
    let means = record
        .column(4)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(means, &Float64Array::from(vec![0.0, 0.5]));
    let watermark = futures::executor::block_on(fixture.sink.recv())
        .unwrap()
        .unwrap();
    assert_eq!(watermark.as_watermark(), Some(EventTime::from_micros(1)));
    fixture.cancellation.cancel();
    assert!(matches!(
        task.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
}

#[test]
fn completed_projection_data_and_watermark_park_without_self_wake() {
    let mut fixture = open_native_task(true);
    let original = batch(0, 2, 0);
    let mut columns = original.table_payload().unwrap().batches()[0]
        .columns()
        .to_vec();
    columns.push(Arc::new(Float64Array::from(vec![0.0, 0.5])));
    let expected = RecordBatch::try_new(
        fixture.task.output_ports["output"]
            .schema()
            .unwrap()
            .clone(),
        columns,
    )
    .unwrap();
    for message in [
        StreamMessage::data(
            Batch::table(vec![expected.clone()], original.metadata().clone()).unwrap(),
        ),
        StreamMessage::watermark(EventTime::from_micros(1)),
    ] {
        futures::executor::block_on(fixture.source.send(message)).unwrap();
    }
    let wake_counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(wake_counter.clone());
    let mut context = Context::from_waker(&waker);
    let mut task = Box::pin(run_operator_task_with_cooperation(
        fixture.task,
        TaskId::new(1),
        fixture.cooperation,
    ));
    assert!(task.as_mut().poll(&mut context).is_pending());
    assert_eq!(fixture.sink.metrics().queue_depth, 2);
    assert_eq!(
        wake_counter.0.load(Ordering::SeqCst),
        0,
        "after forwarding finality, an empty internal ingress should park naturally"
    );
    let output = futures::executor::block_on(fixture.sink.recv())
        .unwrap()
        .unwrap();
    let data = output.as_data().unwrap();
    assert_eq!(data.table_payload().unwrap().batches(), &[expected]);
    assert_eq!(data.metadata(), original.metadata());
    let watermark = futures::executor::block_on(fixture.sink.recv())
        .unwrap()
        .unwrap();
    assert_eq!(watermark.as_watermark(), Some(EventTime::from_micros(1)));
    fixture.cancellation.cancel();
    assert!(matches!(
        task.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
}

#[test]
fn cancellation_during_cooperation_prevents_the_next_ready_dispatch() {
    let mut fixture = open_native_task(false);
    for row in 0..3 {
        futures::executor::block_on(fixture.source.send(StreamMessage::data(batch(
            row,
            1,
            u64::try_from(row).unwrap(),
        ))))
        .unwrap();
    }
    let wake_counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(wake_counter);
    let mut context = Context::from_waker(&waker);
    let mut task = Box::pin(run_operator_task_with_cooperation(
        fixture.task,
        TaskId::new(0),
        fixture.cooperation,
    ));
    assert!(task.as_mut().poll(&mut context).is_pending());
    fixture.cancellation.cancel();
    assert!(matches!(
        task.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
    assert_eq!(
        fixture
            .trace
            .lock()
            .iter()
            .filter(|event| event.starts_with("rolling:data:"))
            .count(),
        2,
        "a ready message must not dispatch after cancellation during handoff"
    );
}

#[test]
fn receive_error_and_preexisting_cancellation_keep_the_selected_result() {
    for cancelled in [true, false] {
        let mut fixture = open_native_task(false);
        for row in 0..2 {
            futures::executor::block_on(fixture.source.send(StreamMessage::data(batch(
                row,
                1,
                u64::try_from(row).unwrap(),
            ))))
            .unwrap();
        }
        drop(fixture.source);
        let wake_counter = Arc::new(WakeCounter::default());
        let waker = Waker::from(wake_counter);
        let mut context = Context::from_waker(&waker);
        let mut task = Box::pin(run_operator_task_with_cooperation(
            fixture.task,
            TaskId::new(0),
            fixture.cooperation,
        ));
        if cancelled {
            fixture.cancellation.cancel();
            assert!(matches!(
                task.as_mut().poll(&mut context),
                Poll::Ready(Ok(()))
            ));
        } else {
            assert!(matches!(
                task.as_mut().poll(&mut context),
                Poll::Ready(Err(CalcFlowError::EdgeClosed { edge })) if edge == "source"
            ));
        }
    }
}
