use super::*;
use crate::{EdgeReceiver, EdgeSender, StreamingFailureReason};

struct RoutedOperator {
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    reason: Option<StreamingFailureReason>,
    _lifetime: Arc<()>,
    panic: bool,
    drop_observation: Option<(CancellationToken, Arc<AtomicBool>)>,
}

impl Drop for RoutedOperator {
    fn drop(&mut self) {
        if let Some((cancellation, observed)) = &self.drop_observation {
            observed.store(cancellation.is_cancelled(), Ordering::SeqCst);
        }
    }
}

impl OperatorMetadata for RoutedOperator {
    fn name(&self) -> &'static str {
        "error-propagation-probe"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl StreamOperator for RoutedOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        batch: Batch,
        _context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        assert!(!self.panic, "structured downstream panic");
        if let Some(reason_code) = self.reason {
            return Err(CalcFlowError::OperatorReason {
                node_id: "asof".into(),
                reason_code,
                message: "structured downstream failure".into(),
            });
        }
        output.emit("output", batch).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

struct TaskEnvironment {
    job: StreamJobContext,
    ack: mpsc::UnboundedSender<OperatorEntryAck>,
}

impl TaskEnvironment {
    fn inputs(
        &self,
        node_id: &str,
        receiver: EdgeReceiver,
        sender: EdgeSender,
        operator: RoutedOperator,
    ) -> OperatorTaskInputs {
        let (_, entry_gate) = watch::channel(true);
        let (_, data_gate) = watch::channel(true);
        OperatorTaskInputs {
            late_output_ports: std::collections::BTreeSet::new(),
            entity_work: None,
            sql_recovery: None,
            node_id: node_id.into(),
            output_ports: BTreeMap::from([("output".into(), operator.outputs[0].clone())]),
            operator: CompiledStreamOperator::External(Box::new(operator)),
            checkpoint_capability: OperatorCheckpointCapability::Stateless,
            ingresses: BTreeMap::from([(
                "input".into(),
                OperatorIngress::new(receiver.edge().into(), receiver),
            )]),
            outputs: BTreeMap::from([("output".into(), vec![sender.into()])]),
            context: self.job.for_node(node_id).unwrap(),
            progress: OperatorProgress::default(),
            metrics: super::super::MetricsRecorder::default(),
            entry_gate,
            entry_ack: self.ack.clone(),
            data_gate,
            launch_cancel: CancellationToken::new(),
            checkpoint: None,
            restore: None,
        }
    }
}

fn routed_operator(
    reason: Option<StreamingFailureReason>,
) -> (RoutedOperator, std::sync::Weak<()>) {
    let lifetime = Arc::new(());
    let weak = Arc::downgrade(&lifetime);
    (
        RoutedOperator {
            inputs: vec![Port::new("input", BatchKind::Table, true, None).unwrap()],
            outputs: vec![Port::new("output", BatchKind::Table, false, None).unwrap()],
            reason,
            _lifetime: lifetime,
            panic: false,
            drop_observation: None,
        },
        weak,
    )
}

async fn ready_operator_pair(
    upstream_reason: Option<StreamingFailureReason>,
    reason: StreamingFailureReason,
) -> crate::runtime::streaming::supervisor::SupervisionReport {
    let cancellation = CancellationToken::new();
    let mut supervisor = TaskSupervisor::new(cancellation.clone());
    let (ack, _acks) = mpsc::unbounded_channel();
    let environment = TaskEnvironment {
        job: StreamJobContext::new(7, "fingerprint", JsonMap::new(), None, cancellation.clone()),
        ack,
    };
    let budget = EdgeBudget {
        max_rows: 1,
        max_bytes: 1024,
    };
    let (mut source, upstream_input) = crate::edge_channel("source->trades", budget).unwrap();
    let (mut edge, downstream_input) = crate::edge_channel("trades->asof", budget).unwrap();
    let (sink_output, _sink) = crate::edge_channel("asof->sink", budget).unwrap();
    source
        .send(StreamMessage::data(batch("source", 1)))
        .await
        .unwrap();
    edge.send(StreamMessage::data(batch("source", 0)))
        .await
        .unwrap();
    let (upstream, upstream_lifetime) = routed_operator(upstream_reason);
    let (downstream, downstream_lifetime) = routed_operator(Some(reason));
    let pair = super::super::prepare_operator_task_pair(
        &mut supervisor,
        environment.inputs("trades", upstream_input, edge, upstream),
        environment.inputs("asof", downstream_input, sink_output, downstream),
    );
    // This driver polls the downstream member first. Its ready error precedes
    // the upstream's ready send in the same poll, without a scheduling race.
    assert_eq!(pair.ids(), [TaskId::new(0), TaskId::new(1)]);
    supervisor.spawn_prepared_pair(pair);
    let report = tokio::time::timeout(Duration::from_secs(1), supervisor.join_all())
        .await
        .unwrap();
    assert_eq!(supervisor.task_count(), 0);
    assert!(cancellation.is_cancelled());
    assert!(upstream_lifetime.upgrade().is_none());
    assert!(downstream_lifetime.upgrade().is_none());
    report
}

#[tokio::test]
async fn downstream_reason_precedes_upstream_edge_close() {
    for reason in [
        StreamingFailureReason::AsofDuplicateIdentity,
        StreamingFailureReason::AsofLateRow,
    ] {
        let report = Box::pin(ready_operator_pair(None, reason)).await;
        assert_eq!(report.primary_errors().len(), 1, "{report:?}");
        assert_eq!(report.errors[0].task_id, TaskId::new(1), "{report:?}");
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::OperatorReason { reason_code, .. } if *reason_code == reason
        ));
    }
}

#[tokio::test]
async fn independent_ready_operator_failures_keep_both_primary_identities() {
    let report = Box::pin(ready_operator_pair(
        Some(StreamingFailureReason::AsofDuplicateIdentity),
        StreamingFailureReason::AsofLateRow,
    ))
    .await;
    assert_eq!(report.errors.len(), 2, "{report:?}");
    assert_eq!(report.primary_errors().len(), 2, "{report:?}");
    assert_eq!(report.errors[0].task_id, TaskId::new(0));
    assert_eq!(report.errors[1].task_id, TaskId::new(1));
    assert!(matches!(
        report.errors[0].error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        }
    ));
    assert!(matches!(
        report.errors[1].error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofLateRow,
            ..
        }
    ));
}

async fn assert_single_operator_retains_endpoints(panic: bool) {
    let cancellation = CancellationToken::new();
    let mut supervisor = TaskSupervisor::new(cancellation.clone());
    let (ack, _acks) = mpsc::unbounded_channel();
    let environment = TaskEnvironment {
        job: StreamJobContext::new(7, "fingerprint", JsonMap::new(), None, cancellation.clone()),
        ack,
    };
    let budget = EdgeBudget {
        max_rows: 1,
        max_bytes: 1024,
    };
    let (mut edge, receiver) = crate::edge_channel("trades->asof", budget).unwrap();
    let (sink_output, _sink) = crate::edge_channel("asof->sink", budget).unwrap();
    edge.send(StreamMessage::data(batch("source", 0)))
        .await
        .unwrap();
    supervisor.spawn("operator:trades", async move {
        edge.send(StreamMessage::data(batch("source", 1))).await?;
        edge.send(StreamMessage::data(batch("source", 2))).await
    });
    let dropped_after_cancel = Arc::new(AtomicBool::new(false));
    let (mut operator, lifetime) = routed_operator(Some(StreamingFailureReason::AsofLateRow));
    operator.panic = panic;
    operator.drop_observation = Some((cancellation.clone(), dropped_after_cancel.clone()));
    spawn_operator_task(
        &mut supervisor,
        environment.inputs("asof", receiver, sink_output, operator),
    );
    let report = tokio::time::timeout(Duration::from_secs(1), supervisor.join_all())
        .await
        .unwrap();
    assert_eq!(supervisor.task_count(), 0);
    assert!(lifetime.upgrade().is_none());
    assert!(dropped_after_cancel.load(Ordering::SeqCst));
    assert_eq!(report.primary_errors().len(), 1, "{report:?}");
    assert_eq!(report.errors[0].task_id, TaskId::new(1), "{report:?}");
    assert_eq!(report.errors[1].task_id, TaskId::new(0), "{report:?}");
    assert!(matches!(
        report.errors[1].error,
        CalcFlowError::EdgeClosed { .. }
    ));
    if panic {
        assert!(
            matches!(&report.errors[0].error, CalcFlowError::TaskPanicked { task_id: 1, message } if message == "structured downstream panic")
        );
    } else {
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofLateRow,
                ..
            }
        ));
    }
}

#[tokio::test]
async fn failed_single_operator_retains_endpoints_until_convergence() {
    assert_single_operator_retains_endpoints(false).await;
}

#[tokio::test]
async fn panicking_single_operator_retains_endpoints_until_convergence() {
    assert_single_operator_retains_endpoints(true).await;
}

fn limited_asof() -> (CompiledStreamOperator, Batch, Port) {
    use crate::{AsofJoinSide, AsofStateLimits, StreamAsofJoinOperator, StreamAsofJoinSpec};
    use datafusion::arrow::{
        array::{Int64Array, TimestampMicrosecondArray, UInt64Array},
        record_batch::RecordBatch,
    };
    let record = RecordBatch::try_from_iter([
        (
            "key",
            Arc::new(Int64Array::from(vec![1, 1])) as datafusion::arrow::array::ArrayRef,
        ),
        (
            "ts",
            Arc::new(TimestampMicrosecondArray::from(vec![105, 106]).with_timezone("UTC")),
        ),
        ("sequence", Arc::new(UInt64Array::from(vec![0, 1]))),
    ])
    .unwrap();
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "ts".into(),
            vec!["sequence".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let operator = StreamAsofJoinOperator::new(
        "asof",
        record.schema(),
        record.schema(),
        StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::from_micros(10),
            AsofStateLimits::new(1, 8_000_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let output = operator.output_ports()[0].clone();
    (
        CompiledStreamOperator::StreamAsofJoin(Box::new(operator)),
        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
        output,
    )
}

#[tokio::test]
async fn test_asof_admission_cleanup_preserves_reason_before_upstream_edge_close() {
    let cancellation = CancellationToken::new();
    let mut supervisor = TaskSupervisor::new(cancellation.clone());
    let (ack, _acks) = mpsc::unbounded_channel();
    let environment = TaskEnvironment {
        job: StreamJobContext::new(7, "fingerprint", JsonMap::new(), None, cancellation.clone()),
        ack,
    };
    let (operator, data, port) = limited_asof();
    let budget = EdgeBudget::new(2, 1024).unwrap();
    let (mut left, receiver) = crate::edge_channel("trades->asof", budget).unwrap();
    let (_right, right_receiver) = crate::edge_channel("quotes->asof", budget).unwrap();
    let (sink_output, mut sink) = crate::edge_channel("asof->sink", budget).unwrap();
    left.send(StreamMessage::data(data.clone())).await.unwrap();
    supervisor.spawn("operator:trades", async move {
        left.send(StreamMessage::data(data.clone())).await?;
        left.send(StreamMessage::data(data)).await
    });
    let (placeholder, _) = routed_operator(None);
    let mut inputs = environment.inputs("asof", receiver, sink_output, placeholder);
    inputs.operator = operator;
    inputs.output_ports = BTreeMap::from([("output".into(), port)]);
    let left = inputs.ingresses.remove("input").unwrap();
    inputs.ingresses = BTreeMap::from([
        ("left".into(), left),
        (
            "right".into(),
            OperatorIngress::new("quotes->asof".into(), right_receiver),
        ),
    ]);
    spawn_operator_task(&mut supervisor, inputs);
    let report = tokio::time::timeout(Duration::from_secs(1), supervisor.join_all())
        .await
        .unwrap();
    assert_eq!(report.primary_errors().len(), 1, "{report:?}");
    assert_eq!(report.errors[0].task_id, TaskId::new(1), "{report:?}");
    assert!(
        matches!(&report.errors[0].error, CalcFlowError::OperatorReason {
        reason_code: StreamingFailureReason::AsofStateLimitExceeded, message, ..
        } if message == "stream_asof_join.limits.max_state_rows exceeded"),
        "{report:?}"
    );
    assert!(matches!(
        &report.errors[1].error,
        CalcFlowError::EdgeClosed { .. }
    ));
    assert_eq!(supervisor.task_count(), 0);
    assert!(
        sink.recv().await.unwrap().is_none(),
        "failed admission emits no match"
    );
}
