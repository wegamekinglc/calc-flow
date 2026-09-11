#[cfg(test)]
mod allocation_evidence;

use crate::{
    OperatorMetadata,
    pipeline::{CompiledStreamOperator, RuntimeStreamNode},
};

pub(super) fn eligible_pair(first: &RuntimeStreamNode, second: &RuntimeStreamNode) -> bool {
    let (CompiledStreamOperator::Rolling(rolling), CompiledStreamOperator::Expression(expression)) =
        (&first.operator, &second.operator)
    else {
        return false;
    };
    if !directly_connected(first, second) {
        return false;
    }
    let Some(input) = first
        .output_ports
        .get("output")
        .and_then(|port| port.schema())
    else {
        return false;
    };
    let Some(output) = second.output_ports.get("output") else {
        return false;
    };
    rolling.output_ports()[0].schema() == Some(input)
        && second
            .input_ports
            .get("input")
            .and_then(|port| port.schema())
            == Some(input)
        && expression.is_exact_column_projection(input, output.schema())
}

fn has_single_ports(node: &RuntimeStreamNode) -> bool {
    node.input_ports.len() == 1 && node.ingress_edges.len() == 1 && node.output_ports.len() == 1
}

fn directly_connected(first: &RuntimeStreamNode, second: &RuntimeStreamNode) -> bool {
    if !has_single_ports(first) || first.output_edges.len() != 1 || !has_single_ports(second) {
        return false;
    }
    let Some([edge]) = first.output_edges.get("output").map(Vec::as_slice) else {
        return false;
    };
    second.ingress_edges.get("input") == Some(edge)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
    use tokio::sync::mpsc;

    use super::eligible_pair;
    use crate::{
        BatchKind, CancellationToken, Edge, EdgeBudget, ExpressionOperator, JsonMap,
        OperatorMetadata, PipelineBuilder, Port, PortEndpoint, RollingOperator,
        StreamExecutionPlan, StreamJobContext, StreamRequirements, UdfRegistry,
        runtime::streaming::{
            metrics::MetricsRecorder,
            projection::StatusProjection,
            runner::{JobCore, LaunchId, run_operator_entry},
            supervisor::TaskId,
        },
    };

    fn input_schema() -> SchemaRef {
        Arc::new(Schema::new_with_metadata(
            vec![
                Field::new(
                    "ts",
                    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                    false,
                ),
                Field::new("symbol", DataType::Utf8, false),
                Field::new("sequence", DataType::UInt64, false),
                Field::new("price", DataType::Float64, true)
                    .with_metadata([("units".into(), "USD".into())].into_iter().collect()),
            ],
            [("origin".into(), "fusion-fixture".into())]
                .into_iter()
                .collect(),
        ))
    }

    fn rolling() -> RollingOperator {
        let spec = serde_json::from_value(serde_json::json!({
            "configuration_version": 1,
            "state_layout_version": 1,
            "partition_by": ["symbol"],
            "event_time": "ts",
            "sequence_by": ["sequence"],
            "outputs": [{"kind": "lag", "primitive_version": 1, "input": "price", "output": "previous", "periods": 1}],
            "allowed_lateness_micros": 0,
            "late_policy": {"kind": "error", "scope": "envelope"},
            "value_policy": "stateful_numeric_v1"
        })).unwrap();
        RollingOperator::new("rolling", input_schema(), spec).unwrap()
    }

    fn projection(rolling: &RollingOperator) -> ExpressionOperator {
        let input = rolling.output_ports()[0].schema().unwrap().clone();
        let output = Arc::new(Schema::new_with_metadata(
            vec![
                input
                    .field_with_name("price")
                    .unwrap()
                    .clone()
                    .with_name("current"),
                input.field_with_name("previous").unwrap().clone(),
            ],
            input.metadata().clone(),
        ));
        ExpressionOperator::new(
            "project",
            "",
            vec!["price AS current".into(), "previous".into()],
            None,
            vec![],
        )
        .unwrap()
        .with_ports(
            Port::with_schema_ref("input", BatchKind::Table, true, Some(input)).unwrap(),
            Port::with_schema_ref("output", BatchKind::Table, false, Some(output)).unwrap(),
        )
        .unwrap()
    }

    fn plan() -> StreamExecutionPlan {
        let rolling = rolling();
        let project = projection(&rolling);
        PipelineBuilder::new("fusion-fixture")
            .unwrap()
            .add_node("rolling", rolling)
            .unwrap()
            .add_node("project", Box::new(project))
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
    }

    #[test]
    fn exact_rolling_projection_pair_is_eligible() {
        let parts = plan().into_runtime_parts(EdgeBudget::default()).unwrap();
        assert_eq!(
            parts
                .nodes
                .iter()
                .map(|node| node.node_id.as_str())
                .collect::<Vec<_>>(),
            ["rolling", "project"]
        );
        assert!(eligible_pair(&parts.nodes[0], &parts.nodes[1]));
    }

    #[tokio::test]
    async fn real_runner_registers_two_logical_nodes_in_one_physical_driver() {
        let plan = plan();
        let cancellation = CancellationToken::new();
        let context = StreamJobContext::new(
            7,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            cancellation.clone(),
        );
        let (commands, _commands_rx) = mpsc::unbounded_channel();
        let core = Arc::new(JobCore::new(
            LaunchId::new(0),
            7,
            commands,
            MetricsRecorder::default(),
            StatusProjection::default(),
            false,
            plan.name().into(),
        ));
        let parts = plan.into_runtime_parts(EdgeBudget::default()).unwrap();
        let Ok(mut runtime) =
            run_operator_entry(parts, &context, &core, &cancellation, BTreeMap::new(), None).await
        else {
            panic!("valid pair entry failed");
        };
        assert_eq!(runtime.supervisor.task_count(), 2);
        assert_eq!(runtime.supervisor.ready_pair_spawn_count(), 1);
        let tasks = runtime.supervisor.registry().snapshot();
        assert_eq!(tasks[&TaskId::new(0)].task_name, "operator:rolling");
        assert_eq!(tasks[&TaskId::new(1)].task_name, "operator:project");
        let drivers = runtime.supervisor.physical_driver_count();
        cancellation.cancel();
        assert!(runtime.supervisor.join_all().await.errors.is_empty());
        assert_eq!(runtime.supervisor.task_count(), 0);
        assert_eq!(drivers, 1);
    }

    fn rejected_expression(variant: usize, input: Port, output: Port) -> ExpressionOperator {
        use crate::{DataFusionConfig, UdfKind, UdfReference};
        let selected =
            UdfReference::new("fixture", "selected", "1", UdfKind::DataFusionScalar).unwrap();
        let mut expression = ExpressionOperator::new(
            "project",
            "",
            if variant == 8 {
                vec!["price + 1 AS current".into(), "previous".into()]
            } else {
                vec!["price AS current".into(), "previous".into()]
            },
            (variant == 9).then(|| "price IS NOT NULL".into()),
            if variant == 10 {
                vec![selected.clone()]
            } else {
                vec![]
            },
        )
        .unwrap()
        .with_ports(input, output)
        .unwrap();
        if variant == 11 {
            expression.set_stream_resources(
                DataFusionConfig::default(),
                UdfRegistry::new().snapshot(),
                vec![selected],
            );
        }
        expression
    }

    #[test]
    fn fusion_rejects_unproved_schemas_computation_udfs_and_topology() {
        use crate::pipeline::CompiledStreamOperator;

        for variant in 0..13 {
            let mut parts = plan().into_runtime_parts(EdgeBudget::default()).unwrap();
            match variant {
                0 => {
                    parts.nodes[0].output_ports.insert(
                        "output".into(),
                        Port::new("output", BatchKind::Table, false, None).unwrap(),
                    );
                }
                1 => {
                    parts.nodes[1].input_ports.insert(
                        "input".into(),
                        Port::new("input", BatchKind::Table, true, None).unwrap(),
                    );
                }
                2 => {
                    parts.nodes[1].output_ports.insert(
                        "output".into(),
                        Port::new("output", BatchKind::Table, false, None).unwrap(),
                    );
                }
                3 | 4 => {
                    let schema = parts.nodes[1].output_ports["output"].schema().unwrap();
                    let fields = schema
                        .fields()
                        .iter()
                        .map(|field| field.as_ref().clone())
                        .collect::<Vec<_>>();
                    let mut changed = Schema::new_with_metadata(fields, schema.metadata().clone());
                    if variant == 3 {
                        changed = changed.with_metadata(
                            [("different".into(), "metadata".into())]
                                .into_iter()
                                .collect(),
                        );
                    } else {
                        changed = Schema::new_with_metadata(
                            vec![
                                changed.field(0).clone().with_nullable(false),
                                changed.field(1).clone(),
                            ],
                            changed.metadata().clone(),
                        );
                    }
                    parts.nodes[1].output_ports.insert(
                        "output".into(),
                        Port::with_schema_ref(
                            "output",
                            BatchKind::Table,
                            false,
                            Some(Arc::new(changed)),
                        )
                        .unwrap(),
                    );
                }
                5 => {
                    parts.nodes[0]
                        .output_edges
                        .get_mut("output")
                        .unwrap()
                        .push("another-sink".into());
                }
                6 => {
                    parts.nodes[1]
                        .ingress_edges
                        .insert("another".into(), "another-source".into());
                }
                7 => {
                    parts.nodes[1]
                        .ingress_edges
                        .insert("input".into(), "different-edge".into());
                }
                8..=11 => {
                    parts.nodes[1].operator =
                        CompiledStreamOperator::Expression(rejected_expression(
                            variant,
                            parts.nodes[1].input_ports["input"].clone(),
                            parts.nodes[1].output_ports["output"].clone(),
                        ));
                }
                12 => parts.nodes.swap(0, 1),
                _ => unreachable!(),
            }
            assert!(
                !eligible_pair(&parts.nodes[0], &parts.nodes[1]),
                "variant {variant}"
            );
        }
    }

    struct CallbackProbe {
        name: &'static str,
        input: [Port; 1],
        output: [Port; 1],
        trace: Arc<parking_lot::Mutex<Vec<String>>>,
    }

    impl OperatorMetadata for CallbackProbe {
        fn name(&self) -> &str {
            self.name
        }
        fn input_ports(&self) -> &[Port] {
            &self.input
        }
        fn output_ports(&self) -> &[Port] {
            &self.output
        }
        fn configuration(&self) -> JsonMap {
            JsonMap::new()
        }
    }

    #[async_trait::async_trait]
    impl crate::StreamOperator for CallbackProbe {
        async fn process_data(
            &mut self,
            _: &str,
            batch: crate::Batch,
            _: &crate::StreamOperatorContext<'_>,
            output: &mut dyn crate::StreamCollector,
        ) -> crate::Result<()> {
            let sequence = batch.metadata().sequence();
            self.trace
                .lock()
                .push(format!("{}:data:{sequence}", self.name));
            output.emit("output", batch).await?;
            self.trace
                .lock()
                .push(format!("{}:commit:{sequence}", self.name));
            Ok(())
        }
        async fn on_watermark(
            &mut self,
            _: crate::EventTime,
            _: &crate::StreamOperatorContext<'_>,
            _: &mut dyn crate::StreamCollector,
        ) -> crate::Result<()> {
            self.trace.lock().push(format!("{}:watermark", self.name));
            Ok(())
        }
        async fn on_end(
            &mut self,
            _: &crate::StreamOperatorContext<'_>,
            _: &mut dyn crate::StreamCollector,
        ) -> crate::Result<()> {
            self.trace.lock().push(format!("{}:end", self.name));
            Ok(())
        }
    }

    fn probe_inputs(
        name: &'static str,
        trace: &Arc<parking_lot::Mutex<Vec<String>>>,
        receiver: crate::EdgeReceiver,
        sender: crate::EdgeSender,
        context: &StreamJobContext,
        ack: &mpsc::UnboundedSender<crate::runtime::streaming::operator_task::OperatorEntryAck>,
    ) -> crate::runtime::streaming::operator_task::OperatorTaskInputs {
        use crate::runtime::streaming::operator_task::{
            OperatorIngress, OperatorProgress, OperatorTaskInputs,
        };
        let input = Port::new("input", BatchKind::Table, true, None).unwrap();
        let output = Port::new("output", BatchKind::Table, false, None).unwrap();
        OperatorTaskInputs {
            entity_work: None,
            node_id: name.into(),
            operator: crate::pipeline::CompiledStreamOperator::External(Box::new(CallbackProbe {
                name,
                input: [input],
                output: [output.clone()],
                trace: trace.clone(),
            })),
            checkpoint_capability: crate::pipeline::OperatorCheckpointCapability::Stateless,
            ingresses: BTreeMap::from([(
                "input".into(),
                OperatorIngress::new(format!("to-{name}"), receiver),
            )]),
            outputs: BTreeMap::from([("output".into(), vec![sender])]),
            output_ports: BTreeMap::from([("output".into(), output)]),
            context: context.for_node(name).unwrap(),
            progress: OperatorProgress::default(),
            metrics: MetricsRecorder::default(),
            entry_gate: tokio::sync::watch::channel(true).1,
            entry_ack: ack.clone(),
            data_gate: tokio::sync::watch::channel(true).1,
            launch_cancel: CancellationToken::new(),
            checkpoint: None,
            restore: None,
        }
    }

    #[tokio::test]
    async fn fused_driver_projects_ready_small_data_and_watermark_before_next_callback() {
        use crate::runtime::streaming::{
            channel::edge_channel, operator_task::spawn_operator_task_pair,
            supervisor::TaskSupervisor,
        };
        use crate::{Batch, BatchMetadata, EventTime, StreamMessage};
        use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

        let cancellation = CancellationToken::new();
        let context =
            StreamJobContext::new(7, "fusion", JsonMap::new(), None, cancellation.clone());
        let budget = EdgeBudget::new(100, 1 << 20).unwrap();
        let (mut source, first_input) = edge_channel("source", budget).unwrap();
        let (first_output, second_input) = edge_channel("internal", budget).unwrap();
        let (second_output, mut sink) = edge_channel("sink", budget).unwrap();
        for message in [
            StreamMessage::data(
                Batch::table(
                    vec![
                        RecordBatch::try_from_iter(vec![(
                            "value",
                            Arc::new(Int64Array::from(vec![1])) as _,
                        )])
                        .unwrap(),
                    ],
                    BatchMetadata::new("source", 0, BTreeMap::new()).unwrap(),
                )
                .unwrap(),
            ),
            StreamMessage::watermark(EventTime::from_micros(1)),
            StreamMessage::data(
                Batch::table(
                    vec![
                        RecordBatch::try_from_iter(vec![(
                            "value",
                            Arc::new(Int64Array::from(vec![2; 64])) as _,
                        )])
                        .unwrap(),
                    ],
                    BatchMetadata::new("source", 1, BTreeMap::new()).unwrap(),
                )
                .unwrap(),
            ),
            StreamMessage::end_of_input(),
        ] {
            source.send(message).await.unwrap();
        }
        let trace = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let (ack, mut acks) = mpsc::unbounded_channel();
        let first = probe_inputs("first", &trace, first_input, first_output, &context, &ack);
        let second = probe_inputs(
            "second",
            &trace,
            second_input,
            second_output,
            &context,
            &ack,
        );
        let mut supervisor = TaskSupervisor::new(cancellation);
        spawn_operator_task_pair(&mut supervisor, first, second);
        assert_eq!(supervisor.ready_pair_spawn_count(), 0);
        for _ in 0..2 {
            acks.recv().await.unwrap().result.unwrap();
        }
        while let Some(message) = sink.recv().await.unwrap() {
            if message.is_end_of_input() {
                break;
            }
        }
        assert!(supervisor.join_all().await.errors.is_empty());
        let trace = trace.lock();
        let position = |event: &str| trace.iter().position(|value| value == event).unwrap();
        assert!(
            position("first:commit:0") < position("second:data:0"),
            "{trace:?}"
        );
        assert!(
            position("second:data:0") < position("first:data:1"),
            "{trace:?}"
        );
        assert!(
            position("second:watermark") < position("first:data:1"),
            "{trace:?}"
        );
        assert_eq!(
            trace.iter().filter(|event| event.ends_with(":end")).count(),
            2
        );
    }

    struct GraphHarness {
        runtime: super::super::RegisteredRuntime,
        core: Arc<JobCore>,
        cancellation: CancellationToken,
        acks: mpsc::Receiver<crate::runtime::streaming::operator_task::OperatorCheckpointAck>,
        terminal_ready: mpsc::Receiver<String>,
        terminal_commands: BTreeMap<
            String,
            mpsc::Sender<crate::runtime::streaming::operator_task::OperatorCheckpointCommand>,
        >,
    }

    async fn graph_harness(
        fused: bool,
        budget: EdgeBudget,
        restores: BTreeMap<String, crate::runtime::streaming::operator_task::OperatorRestoreState>,
        transaction: Option<Arc<crate::state::ManifestTransaction>>,
    ) -> GraphHarness {
        let mut parts = plan().into_runtime_parts(budget).unwrap();
        // A private unknown-schema proof selects the unchanged unfused route.
        if !fused {
            parts.nodes[1].input_ports.insert(
                "input".into(),
                Port::new("input", BatchKind::Table, true, None).unwrap(),
            );
        }
        let cancellation = CancellationToken::new();
        let context = StreamJobContext::new(
            7,
            &parts.fingerprint,
            JsonMap::new(),
            None,
            cancellation.clone(),
        );
        let metrics = MetricsRecorder::new(
            parts
                .edges
                .iter()
                .map(|(id, edge)| (id.clone(), edge.budget)),
            [],
            parts.nodes.iter().map(|node| node.node_id.clone()),
            [],
        );
        let (commands, _commands_rx) = mpsc::unbounded_channel();
        let core = Arc::new(JobCore::new(
            LaunchId::new(0),
            7,
            commands,
            metrics,
            StatusProjection::default(),
            transaction.is_some(),
            parts.name.clone(),
        ));
        let (acks_tx, acks) = mpsc::channel(4);
        let (ready_tx, terminal_ready) = mpsc::channel(4);
        let (terminal_commands, receivers): (BTreeMap<_, _>, BTreeMap<_, _>) = parts
            .nodes
            .iter()
            .map(|node| {
                let (tx, rx) = mpsc::channel(1);
                ((node.node_id.clone(), tx), (node.node_id.clone(), rx))
            })
            .unzip();
        let checkpoint =
            transaction.map(|transaction| super::super::OperatorCheckpointRegistration {
                acks: acks_tx,
                transaction,
                terminal_ready: ready_tx,
                terminal_commands: Arc::new(parking_lot::Mutex::new(receivers)),
                faults: super::super::CheckpointFaultInjector::default(),
                fault_cancellation: cancellation.clone(),
            });
        let Ok(runtime) =
            run_operator_entry(parts, &context, &core, &cancellation, restores, checkpoint).await
        else {
            panic!("graph entry failed");
        };
        assert_eq!(
            runtime.supervisor.physical_driver_count(),
            if fused { 1 } else { 2 }
        );
        assert_eq!(
            runtime.supervisor.ready_pair_spawn_count(),
            usize::from(fused)
        );
        GraphHarness {
            runtime,
            core,
            cancellation,
            acks,
            terminal_ready,
            terminal_commands,
        }
    }

    fn input_batch(start: i64, values: &[Option<f64>], batch_sequence: u64) -> crate::Batch {
        use datafusion::arrow::{
            array::{Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
            record_batch::RecordBatch,
        };
        let timestamps = (0..values.len())
            .map(|offset| start + i64::try_from(offset).unwrap())
            .collect::<Vec<_>>();
        let record = RecordBatch::try_new(
            input_schema(),
            vec![
                Arc::new(TimestampMicrosecondArray::from(timestamps.clone()).with_timezone("UTC")),
                Arc::new(StringArray::from(vec!["S"; values.len()])),
                Arc::new(UInt64Array::from(
                    timestamps
                        .iter()
                        .map(|value| u64::try_from(*value).unwrap())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .unwrap();
        crate::Batch::table(
            vec![record],
            crate::BatchMetadata::new("source", batch_sequence, BTreeMap::new()).unwrap(),
        )
        .unwrap()
    }

    impl GraphHarness {
        async fn send(&mut self, message: crate::StreamMessage) {
            self.runtime.source_outputs.values_mut().next().unwrap()[0]
                .send(message)
                .await
                .unwrap();
        }
        async fn receive(&mut self) -> crate::StreamMessage {
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                self.runtime.sink_inputs.values_mut().next().unwrap().recv(),
            )
            .await
            .unwrap()
            .unwrap()
            .unwrap()
        }
        async fn cancel(&mut self) {
            self.cancellation.cancel();
            assert!(self.runtime.supervisor.join_all().await.errors.is_empty());
            assert_eq!(self.runtime.supervisor.task_count(), 0);
        }
    }

    #[tokio::test]
    async fn fused_graph_preserves_empty_nullable_sliced_batches_and_control_fifo() {
        use crate::{Batch, EventTime, StreamMessage, StreamMessageKind};
        use datafusion::arrow::{array::Float64Array, compute::concat_batches};
        for fused in [false, true] {
            let mut harness =
                graph_harness(fused, EdgeBudget::default(), BTreeMap::new(), None).await;
            let input = input_batch(1, &[Some(9.0), Some(1.0), None, Some(3.0)], 0);
            let original = input.table_payload().unwrap().batches()[0].clone();
            let sliced = Batch::table(
                vec![original.slice(1, 1), original.slice(2, 2)],
                input.metadata().clone(),
            )
            .unwrap();
            let empty = Batch::table(vec![original.slice(0, 0)], input.metadata().clone()).unwrap();
            harness.send(StreamMessage::data(empty)).await;
            assert!(
                harness
                    .core
                    .runtime_status
                    .lock()
                    .nodes
                    .values()
                    .all(|node| node.snapshot().input_batches == 0)
            );
            harness.runtime.data_gate.send(true).unwrap();
            harness.send(StreamMessage::data(sliced)).await;
            harness
                .send(StreamMessage::watermark(EventTime::from_micros(4)))
                .await;
            harness.send(StreamMessage::idle()).await;
            harness.send(StreamMessage::end_of_input()).await;
            let mut records = Vec::new();
            let mut controls = Vec::new();
            loop {
                let message = harness.receive().await;
                if let Some(batch) = message.as_data() {
                    records.extend_from_slice(batch.table_payload().unwrap().batches());
                } else {
                    controls.push(message.kind());
                }
                if message.is_end_of_input() {
                    break;
                }
            }
            assert_eq!(
                controls,
                [
                    StreamMessageKind::Watermark,
                    StreamMessageKind::Idle,
                    StreamMessageKind::EndOfInput
                ]
            );
            let result = concat_batches(&records[0].schema(), &records).unwrap();
            assert_eq!(
                result
                    .column(0)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap(),
                &Float64Array::from(vec![Some(1.0), None, Some(3.0)])
            );
            assert_eq!(
                result
                    .column(1)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap(),
                &Float64Array::from(vec![None, Some(1.0), None])
            );
            assert_eq!(result.schema().field(0).metadata()["units"], "USD");
            assert_eq!(input.table_payload().unwrap().batches()[0], original);
            assert!(
                harness
                    .runtime
                    .supervisor
                    .join_all()
                    .await
                    .errors
                    .is_empty()
            );
            assert!(
                harness
                    .core
                    .runtime_status
                    .lock()
                    .nodes
                    .values()
                    .all(|node| node.snapshot().ended && node.snapshot().on_end_calls == 1)
            );
        }
    }

    async fn saturate_graph(
        harness: &mut GraphHarness,
    ) -> tokio::task::JoinHandle<crate::Result<()>> {
        use crate::{EventTime, StreamMessage};
        harness.runtime.data_gate.send(true).unwrap();
        let mut source = harness
            .runtime
            .source_outputs
            .pop_first()
            .unwrap()
            .1
            .pop()
            .unwrap();
        let producer = tokio::spawn(async move {
            for sequence in 0..8 {
                let start = i64::try_from(sequence * 2 + 1).unwrap();
                source
                    .send(StreamMessage::data(input_batch(
                        start,
                        &[Some(1.0), Some(2.0)],
                        sequence,
                    )))
                    .await?;
                source
                    .send(StreamMessage::watermark(EventTime::from_micros(start + 1)))
                    .await?;
            }
            source.send(StreamMessage::end_of_input()).await?;
            crate::Result::Ok(())
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                let metrics = harness.core.metrics.snapshot();
                if metrics
                    .edges
                    .values()
                    .filter(|edge| edge.channel.blocked_sends > 0)
                    .count()
                    >= 2
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        producer
    }

    #[tokio::test]
    async fn fused_full_queues_cancel_without_retaining_budget_or_tasks() {
        let mut harness = graph_harness(
            true,
            EdgeBudget::new(2, 1 << 20).unwrap(),
            BTreeMap::new(),
            None,
        )
        .await;
        let producer = saturate_graph(&mut harness).await;
        harness.cancel().await;
        assert!(producer.await.unwrap().is_err());
        harness.runtime.sink_inputs.clear();
        assert!(harness.core.metrics.snapshot().edges.values().all(
            |edge| edge.channel.charged_rows == 0
                && edge.channel.charged_bytes == 0
                && edge.channel.queue_depth == 0
        ));
    }

    #[tokio::test]
    async fn fused_full_queues_drain_after_slow_sink_resumes() {
        let mut harness = graph_harness(
            true,
            EdgeBudget::new(2, 1 << 20).unwrap(),
            BTreeMap::new(),
            None,
        )
        .await;
        let producer = saturate_graph(&mut harness).await;
        let mut rows = 0;
        let mut watermarks = Vec::new();
        loop {
            let message = harness.receive().await;
            if let Some(batch) = message.as_data() {
                rows += batch.num_rows();
            }
            if let Some(at) = message.as_watermark() {
                watermarks.push(at.as_micros());
            }
            if message.is_end_of_input() {
                break;
            }
        }
        producer.await.unwrap().unwrap();
        assert!(
            harness
                .runtime
                .supervisor
                .join_all()
                .await
                .errors
                .is_empty()
        );
        assert_eq!(rows, 16);
        assert_eq!(watermarks, [2, 4, 6, 8, 10, 12, 14, 16]);
        assert_eq!(harness.runtime.supervisor.task_count(), 0);
    }

    #[tokio::test]
    async fn fused_closed_receiver_preserves_the_downstream_failure_cause() {
        let mut reports = Vec::new();
        for fused in [false, true] {
            let mut harness = graph_harness(
                fused,
                EdgeBudget::new(2, 1 << 20).unwrap(),
                BTreeMap::new(),
                None,
            )
            .await;
            let producer = saturate_graph(&mut harness).await;
            let closed_edge = harness
                .runtime
                .sink_inputs
                .values()
                .next()
                .unwrap()
                .edge()
                .to_owned();
            harness.runtime.sink_inputs.clear();
            let report = harness.runtime.supervisor.join_all().await;
            assert!(producer.await.unwrap().is_err());
            assert_eq!(report.primary_errors().len(), 1, "{report:?}");
            assert_eq!(report.errors.len(), 1, "{report:?}");
            let failure = &report.errors[0];
            assert_eq!(failure.task_id, TaskId::new(1));
            assert_eq!(failure.task_name, "operator:project");
            assert!(
                matches!(&failure.error, crate::CalcFlowError::EdgeClosed { edge } if edge == &closed_edge)
            );
            reports.push((
                failure.task_id,
                failure.task_name.clone(),
                failure.error.to_string(),
            ));
            let public = crate::runtime::streaming::projection::project_runtime_failures(
                7,
                report
                    .errors
                    .into_iter()
                    .map(super::super::task_runtime_failure)
                    .collect(),
                None,
            );
            assert_eq!(public.len(), 1);
            assert_eq!(public[0].component_id(), Some("project"));
            assert_eq!(public[0].message(), "operator \"project\" execution failed");
            assert_eq!(public[0].reason_code(), None);
            assert!(harness.cancellation.is_cancelled());
            assert_eq!(harness.runtime.supervisor.task_count(), 0);
            assert!(harness.core.metrics.snapshot().edges.values().all(|edge| {
                edge.channel.charged_rows == 0
                    && edge.channel.charged_bytes == 0
                    && edge.channel.queue_depth == 0
            }));
        }
        assert_eq!(reports[0], reports[1]);
    }

    async fn checkpoint_pending_rows(
        transaction: &Arc<crate::state::ManifestTransaction>,
    ) -> BTreeMap<String, crate::OperatorManifestEntry> {
        use crate::{Epoch, EventTime, StreamMessage};
        let mut before = graph_harness(
            false,
            EdgeBudget::default(),
            BTreeMap::new(),
            Some(transaction.clone()),
        )
        .await;
        before.runtime.data_gate.send(true).unwrap();
        before
            .send(StreamMessage::data(input_batch(
                1,
                &[Some(1.0), Some(3.0), Some(5.0)],
                0,
            )))
            .await;
        before
            .send(StreamMessage::watermark(EventTime::from_micros(2)))
            .await;
        before.send(StreamMessage::barrier(Epoch::INITIAL)).await;
        assert_eq!(before.receive().await.as_data().unwrap().num_rows(), 2);
        assert_eq!(
            before.receive().await.as_watermark(),
            Some(EventTime::from_micros(2))
        );
        assert_eq!(before.receive().await.as_barrier(), Some(Epoch::INITIAL));
        let mut entries = BTreeMap::new();
        for _ in 0..2 {
            let ack = before.acks.recv().await.unwrap();
            assert_eq!(ack.epoch, Epoch::INITIAL);
            entries.insert(ack.node_id, ack.state);
        }
        assert_eq!(
            entries.keys().map(String::as_str).collect::<Vec<_>>(),
            ["project", "rolling"]
        );
        before.cancel().await;
        entries
    }

    async fn terminal_checkpoint_pair(harness: &mut GraphHarness) {
        use crate::{Epoch, runtime::streaming::operator_task::OperatorCheckpointCommand};
        let terminal_epoch = Epoch::INITIAL.next().unwrap();
        for _ in 0..2 {
            let node = harness.terminal_ready.recv().await.unwrap();
            harness.terminal_commands[&node]
                .send(OperatorCheckpointCommand::Terminal(terminal_epoch))
                .await
                .unwrap();
        }
        let mut terminal_nodes = Vec::new();
        for _ in 0..2 {
            let ack = harness.acks.recv().await.unwrap();
            assert_eq!(ack.epoch, terminal_epoch);
            assert_eq!(
                ack.state.progress["input"].state,
                crate::ManifestIngressState::Ended
            );
            terminal_nodes.push(ack.node_id);
        }
        terminal_nodes.sort();
        assert_eq!(terminal_nodes, ["project", "rolling"]);
    }

    #[tokio::test]
    async fn fused_graph_restores_unfused_pending_state_and_checkpoints_both_nodes_at_eof() {
        use crate::runtime::streaming::operator_task::OperatorRestoreState;
        use crate::{
            Epoch, EventTime, LocalStateBackend, StateBackend, StateLineageKey, StreamMessage,
        };
        use datafusion::arrow::{array::Float64Array, compute::concat_batches};
        let directory = tempfile::tempdir().unwrap();
        let baseline = plan();
        let fingerprint = baseline.fingerprint().to_owned();
        let key = StateLineageKey::new(baseline.name(), &fingerprint).unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let lineage = Arc::from(backend.open_lineage(&key).await.unwrap());
        let transaction = Arc::new(
            crate::state::ManifestTransaction::open(
                lineage,
                &key,
                directory.path().join("manifests"),
                2,
            )
            .await
            .unwrap(),
        );
        let entries = checkpoint_pending_rows(&transaction).await;
        let mut restores = BTreeMap::new();
        let parts = plan().into_runtime_parts(EdgeBudget::default()).unwrap();
        assert_eq!(parts.fingerprint, fingerprint);
        for node in parts.nodes {
            let entry = &entries[&node.node_id];
            let snapshot = transaction
                .load_operator_state(&node.node_id, entry)
                .await
                .unwrap();
            restores.insert(
                node.node_id.clone(),
                OperatorRestoreState {
                    snapshot: node
                        .checkpoint_capability
                        .decode_snapshot(&node.node_id, snapshot)
                        .unwrap(),
                    progress: entry.progress.clone(),
                    output_frontier: None,
                    next_epoch: Epoch::INITIAL.next().unwrap(),
                },
            );
        }
        let mut after =
            graph_harness(true, EdgeBudget::default(), restores, Some(transaction)).await;
        assert!(
            after
                .core
                .runtime_status
                .lock()
                .nodes
                .values()
                .all(|node| node.snapshot().input_batches == 0)
        );
        after.runtime.data_gate.send(true).unwrap();
        after
            .send(StreamMessage::data(input_batch(4, &[Some(7.0)], 1)))
            .await;
        after
            .send(StreamMessage::watermark(EventTime::from_micros(4)))
            .await;
        after.send(StreamMessage::end_of_input()).await;
        let data = after.receive().await;
        let payload = data.as_data().unwrap().table_payload().unwrap();
        let result = concat_batches(payload.schema(), payload.batches()).unwrap();
        assert_eq!(
            result
                .column(0)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap(),
            &Float64Array::from(vec![5.0, 7.0])
        );
        assert_eq!(
            result
                .column(1)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap(),
            &Float64Array::from(vec![3.0, 5.0])
        );
        assert_eq!(
            after.receive().await.as_watermark(),
            Some(EventTime::from_micros(4))
        );
        assert!(after.receive().await.is_end_of_input());
        terminal_checkpoint_pair(&mut after).await;
        assert!(after.runtime.supervisor.join_all().await.errors.is_empty());
        assert_eq!(after.runtime.supervisor.task_count(), 0);
    }

    #[tokio::test]
    async fn paired_entry_restore_failure_keeps_node_identity_and_drains_both_tasks() {
        use crate::runtime::streaming::{
            operator_task::OperatorRestoreState,
            runner::{EntryFailure, FailureOrigin},
        };
        for failing_node in ["rolling", "project"] {
            let plan = plan();
            let cancellation = CancellationToken::new();
            let context = StreamJobContext::new(
                7,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                cancellation.clone(),
            );
            let (commands, _commands_rx) = mpsc::unbounded_channel();
            let core = Arc::new(JobCore::new(
                LaunchId::new(0),
                7,
                commands,
                MetricsRecorder::default(),
                StatusProjection::default(),
                false,
                plan.name().into(),
            ));
            let restore = OperatorRestoreState {
                snapshot: crate::OperatorStateSnapshot::default(),
                progress: BTreeMap::new(),
                output_frontier: None,
                next_epoch: crate::Epoch::INITIAL,
            };
            let result = run_operator_entry(
                plan.into_runtime_parts(EdgeBudget::default()).unwrap(),
                &context,
                &core,
                &cancellation,
                BTreeMap::from([(failing_node.into(), restore)]),
                None,
            )
            .await;
            let Err(EntryFailure::Failed(failure)) = result else {
                panic!("invalid restore must fail during entry");
            };
            assert!(
                matches!(&failure.origin, FailureOrigin::OperatorEntry { node_id } if node_id == failing_node)
            );
            assert!(core.runtime_status.lock().tasks.snapshot().is_empty());
            assert!(
                core.runtime_status
                    .lock()
                    .nodes
                    .values()
                    .all(|node| node.snapshot().input_batches == 0)
            );
        }
    }

    // Baseline wheel export of the unchanged warm SMA20 application graph.
    const ORIGINAL_WARM_PROJECT: &str = r#"{
  "data_sources": [],
  "description": "",
  "format_version": 3,
  "graph": {
    "datafusion": {
      "batch_size": 8192,
      "collect_diagnostics": true,
      "enable_rolling_rewrite": true,
      "max_partitions": 32,
      "min_rows_per_partition": 65536,
      "parallelism_mode": "fixed",
      "small_rows_threshold": 10001,
      "target_partitions": 1
    },
    "edges": [
      {
        "source_node": "indicators__cf_rolling",
        "source_port": "output",
        "target_node": "indicators",
        "target_port": "input"
      }
    ],
    "name": "incremental-rolling-mean",
    "nodes": [
      {
        "id": "indicators__cf_rolling",
        "input_ports": [
          {
            "kind": "table",
            "name": "input",
            "required": true,
            "schema": [
              {
                "data_type": "timestamp[us, UTC]",
                "name": "event_time",
                "nullable": false
              },
              {
                "data_type": "uint64",
                "name": "sequence",
                "nullable": false
              },
              {
                "data_type": "string",
                "name": "symbol",
                "nullable": false
              },
              {
                "data_type": "float64",
                "name": "price",
                "nullable": false
              }
            ]
          }
        ],
        "operator": {
          "kind": "rolling",
          "spec": {
            "allowed_lateness_micros": 0,
            "configuration_version": 1,
            "event_time": "event_time",
            "late_policy": {
              "kind": "error",
              "scope": "envelope"
            },
            "outputs": [
              {
                "frame": {
                  "kind": "rows",
                  "size": 20
                },
                "input": "price",
                "kind": "mean",
                "min_periods": 1,
                "output": "moving_average",
                "primitive_version": 1
              }
            ],
            "partition_by": [
              "symbol"
            ],
            "sequence_by": [
              "sequence"
            ],
            "state_layout_version": 1,
            "value_policy": "stateful_numeric_v1"
          }
        },
        "output_ports": [
          {
            "kind": "table",
            "name": "output",
            "required": true,
            "schema": [
              {
                "data_type": "timestamp[us, UTC]",
                "name": "event_time",
                "nullable": false
              },
              {
                "data_type": "uint64",
                "name": "sequence",
                "nullable": false
              },
              {
                "data_type": "string",
                "name": "symbol",
                "nullable": false
              },
              {
                "data_type": "float64",
                "name": "price",
                "nullable": false
              },
              {
                "data_type": "float64",
                "name": "moving_average",
                "nullable": true
              }
            ]
          }
        ],
        "position": null
      },
      {
        "id": "indicators",
        "input_ports": [
          {
            "kind": "table",
            "name": "input",
            "required": true,
            "schema": [
              {
                "data_type": "timestamp[us, UTC]",
                "name": "event_time",
                "nullable": false
              },
              {
                "data_type": "uint64",
                "name": "sequence",
                "nullable": false
              },
              {
                "data_type": "string",
                "name": "symbol",
                "nullable": false
              },
              {
                "data_type": "float64",
                "name": "price",
                "nullable": false
              },
              {
                "data_type": "float64",
                "name": "moving_average",
                "nullable": true
              }
            ]
          }
        ],
        "operator": {
          "expression": "",
          "filter": null,
          "kind": "expression",
          "select": [
            "\"event_time\"",
            "\"sequence\"",
            "\"symbol\"",
            "\"price\"",
            "\"moving_average\""
          ],
          "udfs": []
        },
        "output_ports": [],
        "position": null
      }
    ]
  },
  "id": "incremental-rolling-mean",
  "name": "incremental-rolling-mean",
  "runtime": {
    "mode": "stream",
    "options": {
      "checkpoint_interval_ms": 30000,
      "max_batch_bytes": 67108864,
      "max_batch_rows": 10000
    }
  },
  "sinks": [],
  "sources": [],
  "state": {
    "retention": 3,
    "root": ".calc-flow-state"
  }
}"#;

    fn original_native_program() -> StreamExecutionPlan {
        let project = crate::import_project_json(ORIGINAL_WARM_PROJECT.as_bytes()).unwrap();
        crate::compile_stream_project_graph(
            &project,
            &crate::ProviderRegistry::default(),
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn native_projection_without_declared_output_schema_uses_fused_runner() {
        let plan = original_native_program();
        let parts = plan.into_runtime_parts(EdgeBudget::default()).unwrap();
        assert!(parts.nodes[1].output_ports["output"].schema().is_none());
        let cancellation = CancellationToken::new();
        let context = StreamJobContext::new(
            7,
            &parts.fingerprint,
            JsonMap::new(),
            None,
            cancellation.clone(),
        );
        let (commands, _commands_rx) = mpsc::unbounded_channel();
        let core = Arc::new(JobCore::new(
            LaunchId::new(0),
            7,
            commands,
            MetricsRecorder::default(),
            StatusProjection::default(),
            false,
            parts.name.clone(),
        ));
        let Ok(mut runtime) =
            run_operator_entry(parts, &context, &core, &cancellation, BTreeMap::new(), None).await
        else {
            panic!("native projection entry failed");
        };
        let drivers = runtime.supervisor.physical_driver_count();
        assert_eq!(runtime.supervisor.ready_pair_spawn_count(), 1);
        cancellation.cancel();
        assert!(runtime.supervisor.join_all().await.errors.is_empty());
        assert_eq!(drivers, 1);
    }
}
