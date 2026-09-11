use std::{
    collections::BTreeMap,
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};

use datafusion::arrow::{
    array::{Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use tokio::sync::{mpsc, watch};

use super::{Observer, Phase};

use crate::{
    Batch, BatchKind, BatchMetadata, CancellationToken, Edge, EdgeBudget, EventTime,
    ExpressionOperator, JsonMap, OperatorMetadata, PipelineBuilder, Port, PortEndpoint,
    RollingOperator, StreamJobContext, StreamMessage, StreamRequirements, UdfRegistry,
    pipeline::RuntimeStreamNode,
    runtime::streaming::{
        EdgeReceiver, EdgeSender,
        channel::edge_channel_with_metrics,
        metrics::MetricsRecorder,
        operator_task::{
            OperatorEntryAck, OperatorIngress, OperatorProgress, OperatorTaskInputs,
            prepare_operator_task_pair,
        },
        supervisor::{TaskId, TaskSupervisor},
    },
};

#[derive(Default)]
struct ParentWake(AtomicUsize);

impl Wake for ParentWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn input_schema() -> SchemaRef {
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

fn native_nodes(budget: EdgeBudget) -> [RuntimeStreamNode; 2] {
    let spec = serde_json::from_value(serde_json::json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["sequence"],
        "outputs": [{"kind": "mean", "primitive_version": 1, "input": "price",
            "output": "mean", "min_periods": 1, "frame": {"kind": "rows", "size": 20}}],
        "allowed_lateness_micros": 0, "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap();
    let rolling = RollingOperator::new("rolling", input_schema(), spec).unwrap();
    let output = rolling.output_ports()[0].schema().unwrap().clone();
    let projection = ExpressionOperator::new(
        "project",
        "",
        ["ts", "symbol", "sequence", "price", "mean"]
            .map(str::to_owned)
            .into(),
        None,
        vec![],
    )
    .unwrap()
    .with_ports(
        Port::with_schema_ref("input", BatchKind::Table, true, Some(output.clone())).unwrap(),
        Port::with_schema_ref("output", BatchKind::Table, false, Some(output)).unwrap(),
    )
    .unwrap();
    PipelineBuilder::new("ready-native-fixture")
        .unwrap()
        .add_node("rolling", rolling)
        .unwrap()
        .add_node("project", Box::new(projection))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("rolling", "output").unwrap(),
            PortEndpoint::new("project", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
        .into_runtime_parts(budget)
        .unwrap()
        .nodes
        .try_into()
        .unwrap_or_else(|_| panic!("native graph must contain exactly two operators"))
}

fn input_batch(start: u32, rows: u32, sequence: u64) -> Batch {
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from_iter_values((start..start + rows).map(i64::from))
                    .with_timezone("UTC"),
            ),
            Arc::new(StringArray::from_iter_values((0..rows).map(|_| "S"))),
            Arc::new(UInt64Array::from_iter_values(
                (start..start + rows).map(u64::from),
            )),
            Arc::new(Float64Array::from_iter_values(
                (start..start + rows).map(f64::from),
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

struct NativePair {
    driver: Pin<Box<dyn Future<Output = Vec<TaskId>> + Send>>,
    supervisor: TaskSupervisor,
    source: EdgeSender,
    sink: EdgeReceiver,
    rolling_progress: OperatorProgress,
    output_schema: SchemaRef,
    cancellation: CancellationToken,
    entry_acks: mpsc::UnboundedReceiver<OperatorEntryAck>,
    metrics: MetricsRecorder,
}

fn native_inputs(
    node: RuntimeStreamNode,
    input: EdgeReceiver,
    output: EdgeSender,
    context: &StreamJobContext,
    metrics: &MetricsRecorder,
    entry_ack: &mpsc::UnboundedSender<OperatorEntryAck>,
) -> OperatorTaskInputs {
    OperatorTaskInputs {
        entity_work: None,
        context: context.for_node(&node.node_id).unwrap(),
        node_id: node.node_id,
        operator: node.operator,
        checkpoint_capability: node.checkpoint_capability,
        ingresses: [(
            "input".into(),
            OperatorIngress::new(input.edge().into(), input),
        )]
        .into(),
        outputs: [("output".into(), vec![output])].into(),
        output_ports: node.output_ports,
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

fn native_pair() -> NativePair {
    observed_native_pair(|_, _| None)
}

fn observed_native_pair(
    observer: impl FnOnce(&MetricsRecorder, &OperatorProgress) -> Option<Observer>,
) -> NativePair {
    let budget = EdgeBudget::new(256, 1 << 20).unwrap();
    let metrics = MetricsRecorder::new(
        ["source", "internal", "sink"].map(|name| (name.into(), budget)),
        [],
        ["rolling".into(), "project".into()],
        [],
    );
    let (source, first_input) =
        edge_channel_with_metrics("source", budget, metrics.clone()).unwrap();
    let (first_output, second_input) =
        edge_channel_with_metrics("internal", budget, metrics.clone()).unwrap();
    let (second_output, sink) = edge_channel_with_metrics("sink", budget, metrics.clone()).unwrap();
    let cancellation = CancellationToken::new();
    let context = StreamJobContext::new(
        7,
        "ready-native-fixture",
        JsonMap::new(),
        None,
        cancellation.clone(),
    );
    let [rolling, projection] = native_nodes(budget);
    assert_eq!(
        rolling.output_edges["output"],
        [projection.ingress_edges["input"].clone()]
    );
    let output_schema = projection.output_ports["output"].schema().unwrap().clone();
    let (entry_ack, entry_acks) = mpsc::unbounded_channel();
    let first = native_inputs(
        rolling,
        first_input,
        first_output,
        &context,
        &metrics,
        &entry_ack,
    );
    let rolling_progress = first.progress.clone();
    let second = native_inputs(
        projection,
        second_input,
        second_output,
        &context,
        &metrics,
        &entry_ack,
    );
    let mut supervisor = TaskSupervisor::new(cancellation.clone());
    let pair = prepare_operator_task_pair(&mut supervisor, first, second);
    let pair = match observer(&metrics, &rolling_progress) {
        Some(observer) => pair.observe(observer),
        None => pair,
    };
    assert_eq!(pair.ids(), [TaskId::new(0), TaskId::new(1)]);
    assert_eq!(supervisor.task_count(), 2);
    NativePair {
        driver: Box::pin(pair.run(true)),
        supervisor,
        source,
        sink,
        rolling_progress,
        output_schema,
        cancellation,
        entry_acks,
        metrics,
    }
}

fn send(pair: &mut NativePair, message: StreamMessage) {
    futures::executor::block_on(pair.source.send(message)).unwrap();
}

fn finish(pair: &mut NativePair, context: &mut Context<'_>) {
    send(pair, StreamMessage::end_of_input());
    let mut exited = None;
    for _ in 0..16 {
        if let Poll::Ready(ids) = pair.driver.as_mut().poll(context) {
            exited = Some(ids);
            break;
        }
    }
    let ids = exited.expect("native EOF must complete both full wrappers");
    assert_eq!(ids, [TaskId::new(0), TaskId::new(1)]);
    assert!(
        futures::executor::block_on(pair.sink.recv())
            .unwrap()
            .unwrap()
            .is_end_of_input()
    );
    for id in ids {
        let exit = pair.supervisor.settled.lock().remove(&id).unwrap();
        assert!(exit.result.is_ok());
        pair.supervisor.record_settled(exit);
    }
    assert_eq!(pair.supervisor.task_count(), 0);
    assert_eq!(pair.sink.metrics().queue_depth, 0);
    assert!(pair.metrics.snapshot().edges.values().all(|edge| {
        edge.channel.queue_depth == 0
            && edge.channel.charged_rows == 0
            && edge.channel.charged_bytes == 0
    }));
}

#[test]
fn production_native_pair_completes_idle_request_in_one_poll_without_internal_parent_wake() {
    let mut pair = native_pair();
    let wake_counter = Arc::new(ParentWake::default());
    let waker = Waker::from(wake_counter.clone());
    let mut context = Context::from_waker(&waker);
    assert!(pair.driver.as_mut().poll(&mut context).is_pending());
    for _ in 0..2 {
        pair.entry_acks.try_recv().unwrap().result.unwrap();
    }
    assert_eq!(wake_counter.0.load(Ordering::SeqCst), 0);
    send(&mut pair, StreamMessage::data(input_batch(0, 2, 0)));
    send(
        &mut pair,
        StreamMessage::watermark(EventTime::from_micros(1)),
    );
    wake_counter.0.store(0, Ordering::SeqCst);
    assert!(pair.driver.as_mut().poll(&mut context).is_pending());
    assert_eq!(
        (
            pair.sink.metrics().queue_depth,
            wake_counter.0.load(Ordering::SeqCst)
        ),
        (2, 0),
        "the production native pair must consume U→D readiness in this parent poll"
    );
    assert_eq!(pair.rolling_progress.snapshot().input_batches, 1);
    let message = futures::executor::block_on(pair.sink.recv())
        .unwrap()
        .unwrap();
    let batch = message.as_data().unwrap();
    let record = &batch.table_payload().unwrap().batches()[0];
    assert_eq!(record.schema(), pair.output_schema);
    assert_eq!(
        record
            .column(4)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap(),
        &Float64Array::from(vec![0.0, 0.5])
    );
    assert_eq!(batch.metadata().sequence(), 0);
    assert_eq!(
        futures::executor::block_on(pair.sink.recv())
            .unwrap()
            .unwrap()
            .as_watermark(),
        Some(EventTime::from_micros(1))
    );
    assert!(!pair.cancellation.is_cancelled());
    finish(&mut pair, &mut context);
}

fn assert_slow_sink_drain_and_finality(pair: &mut NativePair, context: &mut Context<'_>) {
    for at in -254..0 {
        assert_eq!(
            futures::executor::block_on(pair.sink.recv())
                .unwrap()
                .unwrap()
                .as_watermark(),
            Some(EventTime::from_micros(at))
        );
    }
    let small = futures::executor::block_on(pair.sink.recv())
        .unwrap()
        .unwrap();
    assert_eq!(small.as_data().unwrap().num_rows(), 1);
    assert_eq!(
        futures::executor::block_on(pair.sink.recv())
            .unwrap()
            .unwrap()
            .as_watermark(),
        Some(EventTime::from_micros(0))
    );
    for _ in 0..8 {
        assert!(pair.driver.as_mut().poll(context).is_pending());
        if pair.sink.metrics().queue_depth == 2 {
            break;
        }
    }
    assert_eq!(pair.sink.metrics().queue_depth, 2);
    let large = futures::executor::block_on(pair.sink.recv())
        .unwrap()
        .unwrap();
    let batch = large.as_data().unwrap();
    assert_eq!(batch.num_rows(), 64);
    assert_eq!(batch.metadata().sequence(), 1);
    let record = &batch.table_payload().unwrap().batches()[0];
    assert_eq!(record.schema(), pair.output_schema);
    let expected = Float64Array::from_iter_values((1..=64_u32).map(|row| {
        if row < 20 {
            f64::from(row) / 2.0
        } else {
            f64::from(row) - 9.5
        }
    }));
    assert_eq!(
        record
            .column(4)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap(),
        &expected
    );
    assert_eq!(
        futures::executor::block_on(pair.sink.recv())
            .unwrap()
            .unwrap()
            .as_watermark(),
        Some(EventTime::from_micros(64))
    );
    finish(pair, context);
}

#[test]
fn production_native_ready_old_watermark_precedes_the_next_large_callback() {
    let trace = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut pair = observed_native_pair(|metrics, progress| {
        let metrics = metrics.clone();
        let progress = progress.clone();
        let trace = trace.clone();
        Some(Arc::new(move |phase, _, _| {
            if let Phase::BeforeChild(child) = phase {
                trace.lock().push((
                    child,
                    metrics.snapshot().edges["sink"].channel.queue_depth,
                    progress.snapshot().input_batches,
                ));
            }
        }))
    });
    let waker = Waker::from(Arc::new(ParentWake::default()));
    let mut context = Context::from_waker(&waker);
    assert!(pair.driver.as_mut().poll(&mut context).is_pending());
    for at in -255..0 {
        send(
            &mut pair,
            StreamMessage::watermark(EventTime::from_micros(at)),
        );
    }
    for _ in 0..512 {
        assert!(pair.driver.as_mut().poll(&mut context).is_pending());
        if pair.sink.metrics().queue_depth == 255 {
            break;
        }
    }
    assert_eq!(pair.sink.metrics().queue_depth, 255);
    assert!(pair.driver.as_mut().poll(&mut context).is_pending());
    send(&mut pair, StreamMessage::data(input_batch(0, 1, 0)));
    send(
        &mut pair,
        StreamMessage::watermark(EventTime::from_micros(0)),
    );
    send(&mut pair, StreamMessage::data(input_batch(1, 64, 1)));
    send(
        &mut pair,
        StreamMessage::watermark(EventTime::from_micros(64)),
    );
    assert!(pair.driver.as_mut().poll(&mut context).is_pending());
    assert_eq!(pair.rolling_progress.snapshot().input_batches, 1);
    assert_eq!(pair.sink.metrics().queue_depth, 256);
    assert!(pair.metrics.snapshot().edges["sink"].channel.blocked_sends > 0);
    let first = futures::executor::block_on(pair.sink.recv())
        .unwrap()
        .unwrap();
    assert_eq!(first.as_watermark(), Some(EventTime::from_micros(-255)));
    trace.lock().clear();
    assert!(pair.driver.as_mut().poll(&mut context).is_pending());
    assert_eq!(pair.rolling_progress.snapshot().input_batches, 2);
    let observed = trace.lock();
    assert_eq!(
        observed.first().unwrap().0,
        1,
        "ready downstream must be polled first"
    );
    assert!(
        observed.contains(&(0, 256, 1)),
        "old WM must fill the released sink slot before the large U callback: {observed:?}"
    );
    drop(observed);
    assert_slow_sink_drain_and_finality(&mut pair, &mut context);
}

#[test]
fn production_native_driver_preserves_two_data_budget_and_small_finality_before_large() {
    let trace = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut pair = observed_native_pair(|metrics, progress| {
        let metrics = metrics.clone();
        let progress = progress.clone();
        let trace = trace.clone();
        Some(Arc::new(move |phase, _, _| {
            if let Phase::BeforeChild(child) = phase {
                trace.lock().push((
                    child,
                    metrics.snapshot().edges["sink"].channel.queue_depth,
                    progress.snapshot().input_batches,
                ));
            }
        }))
    });
    let waker = Waker::from(Arc::new(ParentWake::default()));
    let mut context = Context::from_waker(&waker);
    assert!(pair.driver.as_mut().poll(&mut context).is_pending());
    for row in 0..8 {
        send(
            &mut pair,
            StreamMessage::data(input_batch(row, 1, u64::from(row))),
        );
    }
    send(
        &mut pair,
        StreamMessage::watermark(EventTime::from_micros(7)),
    );
    send(&mut pair, StreamMessage::data(input_batch(8, 64, 8)));
    send(
        &mut pair,
        StreamMessage::watermark(EventTime::from_micros(71)),
    );
    let mut maximum = 0;
    let mut saw_ready_prefix = false;
    for _ in 0..16 {
        trace.lock().clear();
        let before = pair.rolling_progress.snapshot().input_batches;
        assert!(pair.driver.as_mut().poll(&mut context).is_pending());
        let callbacks = pair.rolling_progress.snapshot().input_batches - before;
        assert!(callbacks <= 2);
        maximum = maximum.max(callbacks);
        let events = trace.lock();
        for child in [0, 1] {
            assert!(events.iter().filter(|event| event.0 == child).count() <= 1);
        }
        saw_ready_prefix |= events.contains(&(0, 2, 8));
        if pair.sink.metrics().queue_depth == 4 {
            break;
        }
    }
    assert_eq!(maximum, 2);
    assert!(saw_ready_prefix);
    assert_eq!(pair.rolling_progress.snapshot().input_batches, 9);
    for (rows, at) in [(8, 7), (64, 71)] {
        let data = futures::executor::block_on(pair.sink.recv())
            .unwrap()
            .unwrap();
        assert_eq!(data.as_data().unwrap().num_rows(), rows);
        assert_eq!(
            futures::executor::block_on(pair.sink.recv())
                .unwrap()
                .unwrap()
                .as_watermark(),
            Some(EventTime::from_micros(at))
        );
    }
    finish(&mut pair, &mut context);
}
