use super::*;
use crate::{
    BatchMetadata, CrossSectionOperator, ExpressionOperator, JsonMap, LatePolicySpec, NodeOperator,
    PipelineBuilder, PortEndpoint, RollingOperator, SqlOperator, StreamRequirements, UdfRegistry,
};
use datafusion::arrow::{
    array::{
        ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array,
    },
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::json;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("key", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("x", DataType::Float64, true),
    ]))
}

fn late_batch(sequence: u64) -> Batch {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![5]).with_timezone("UTC")) as ArrayRef,
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(UInt64Array::from(vec![sequence])),
            Arc::new(Float64Array::from(vec![Some(5.0)])),
        ],
    )
    .unwrap();
    Batch::table(
        vec![record],
        BatchMetadata::new("source", sequence, JsonMap::new()).unwrap(),
    )
    .unwrap()
}

fn plan(kind: &str, path: &str) -> crate::StreamExecutionPlan {
    let policy = LatePolicySpec::SideOutput {
        metrics_version: 1,
        schema_version: 1,
    };
    let operator: NodeOperator = if kind == "rolling" {
        RollingOperator::new("roll", schema(), serde_json::from_value(json!({
            "configuration_version": 1, "state_layout_version": 1,
            "partition_by": ["key"], "event_time": "ts", "sequence_by": ["seq"],
            "outputs": [{"kind":"lag", "primitive_version":1, "input":"x", "output":"lag", "periods":1}],
            "allowed_lateness_micros":0, "late_policy":policy, "value_policy":"stateful_numeric_v1"
        })).unwrap()).unwrap().into()
    } else {
        CrossSectionOperator::new("roll", schema(), serde_json::from_value(json!({
            "configuration_version": 1, "state_layout_version": 1,
            "entity_by": ["key"], "partition_by": [], "event_time": "ts", "sequence_by": ["seq"],
            "grouping": {"kind":"exact_time"},
            "outputs": [{"kind":"rank", "primitive_version":1, "input":"x", "output":"rank", "direction":"ascending", "tie_method":"average", "null_placement":"exclude", "min_samples":1}],
            "allowed_lateness_micros":0, "late_policy":policy, "value_policy":"nan_exclude_preserve_v1"
        })).unwrap()).unwrap().into()
    };
    let late_schema = operator.output_ports()[1].schema().unwrap().clone();
    let mut builder = PipelineBuilder::new("late-control")
        .unwrap()
        .add_node("roll", operator)
        .unwrap();
    if path != "direct" {
        let expression = ExpressionOperator::new(
            "project",
            "",
            vec!["ts".into(), "key".into(), "seq".into(), "x".into()],
            None,
            vec![],
        )
        .unwrap()
        .with_ports(
            Port::with_schema_ref("input", crate::BatchKind::Table, true, Some(late_schema))
                .unwrap(),
            Port::with_schema_ref("output", crate::BatchKind::Table, true, Some(schema())).unwrap(),
        )
        .unwrap();
        builder = builder
            .add_node("project", Box::new(expression))
            .unwrap()
            .connect(crate::Edge::new(
                PortEndpoint::new("roll", "late").unwrap(),
                PortEndpoint::new("project", "input").unwrap(),
            ))
            .unwrap();
    }
    if path == "sql" {
        let sql = SqlOperator::new(
            "count",
            "SELECT COUNT(*) AS n FROM events",
            vec!["events".into()],
            vec![],
        )
        .unwrap()
        .with_ports(
            vec![
                Port::with_schema_ref("events", crate::BatchKind::Table, true, Some(schema()))
                    .unwrap(),
            ],
            Port::with_schema_ref(
                "output",
                crate::BatchKind::Table,
                true,
                Some(Arc::new(Schema::new(vec![Field::new(
                    "n",
                    DataType::Int64,
                    false,
                )]))),
            )
            .unwrap(),
        )
        .unwrap();
        builder = builder
            .add_node("count", Box::new(sql))
            .unwrap()
            .connect(crate::Edge::new(
                PortEndpoint::new("project", "output").unwrap(),
                PortEndpoint::new("count", "events").unwrap(),
            ))
            .unwrap();
    }
    builder
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

struct Observed {
    normal: Vec<StreamMessage>,
    late: Vec<StreamMessage>,
    restores: BTreeMap<String, OperatorRestoreState>,
}

fn source_trace(epoch: Epoch) -> [StreamMessage; 5] {
    [
        StreamMessage::watermark(EventTime::from_micros(10)),
        StreamMessage::idle(),
        StreamMessage::data(late_batch(epoch.as_u64())),
        StreamMessage::barrier(epoch),
        StreamMessage::end_of_input(),
    ]
}

async fn run_graph(
    kind: &str,
    path: &str,
    mut restores: BTreeMap<String, OperatorRestoreState>,
) -> Observed {
    let plan = plan(kind, path)
        .into_runtime_parts(EdgeBudget::default())
        .unwrap();
    let epoch = restores
        .values()
        .next()
        .map_or(Epoch::INITIAL, |restore| restore.next_epoch);
    let cancellation = CancellationToken::new();
    let context = crate::StreamJobContext::new(
        1,
        &plan.fingerprint,
        JsonMap::new(),
        None,
        cancellation.clone(),
    );
    let mut supervisor = TaskSupervisor::new(cancellation);
    let mut senders = BTreeMap::new();
    let mut receivers = BTreeMap::new();
    for (id, edge) in &plan.edges {
        let (sender, receiver) = crate::edge_channel(id, edge.budget).unwrap();
        senders.insert(id.clone(), sender);
        receivers.insert(id.clone(), receiver);
    }
    let (entry, _entries) = mpsc::unbounded_channel();
    let (checkpoint, checkpoints) = mpsc::channel(32);
    let capabilities = plan
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.checkpoint_capability))
        .collect::<BTreeMap<_, _>>();
    for node in plan.nodes {
        let (_, entry_gate) = watch::channel(true);
        let (_, data_gate) = watch::channel(true);
        let inputs = OperatorTaskInputs {
            late_output_ports: node.late_output_ports,
            entity_work: None,
            ingresses: node
                .ingress_edges
                .iter()
                .map(|(name, edge)| {
                    (
                        name.clone(),
                        OperatorIngress::new(edge.clone(), receivers.remove(edge).unwrap()),
                    )
                })
                .collect(),
            outputs: node
                .output_edges
                .iter()
                .map(|(name, edges)| {
                    (
                        name.clone(),
                        edges
                            .iter()
                            .map(|edge| senders.remove(edge).unwrap())
                            .collect(),
                    )
                })
                .collect(),
            output_ports: node.output_ports,
            operator: node.operator,
            checkpoint_capability: node.checkpoint_capability,
            context: context.for_node(&node.node_id).unwrap(),
            progress: OperatorProgress::default(),
            metrics: MetricsRecorder::default(),
            entry_gate,
            entry_ack: entry.clone(),
            data_gate,
            launch_cancel: CancellationToken::new(),
            checkpoint: Some(OperatorCheckpointPort {
                acks: checkpoint.clone(),
                transaction: None,
                terminal: None,
                alignment_fault: None,
            }),
            restore: restores.remove(&node.node_id),
            node_id: node.node_id,
        };
        spawn_operator_task(&mut supervisor, inputs);
    }
    drop(checkpoint);
    let source_route = plan.source_routes.values().next().unwrap();
    let source = senders.get_mut(&source_route.edge_id).unwrap();
    for message in source_trace(epoch) {
        source.send(message).await.unwrap();
    }
    let report = supervisor.join_all().await;
    assert!(report.errors.is_empty(), "{report:?}");
    collect_outputs(
        plan.sink_routes,
        receivers,
        checkpoints,
        capabilities,
        epoch,
    )
    .await
}

async fn collect_outputs(
    sink_routes: BTreeMap<String, crate::pipeline::RuntimeSinkRoute>,
    mut receivers: BTreeMap<String, EdgeReceiver>,
    mut checkpoints: mpsc::Receiver<OperatorCheckpointAck>,
    capabilities: BTreeMap<String, OperatorCheckpointCapability>,
    epoch: Epoch,
) -> Observed {
    let mut observed = Observed {
        normal: vec![],
        late: vec![],
        restores: BTreeMap::new(),
    };
    for route in sink_routes.values() {
        let target = if route.source.node_id == "roll" && route.source.port == "output" {
            &mut observed.normal
        } else {
            &mut observed.late
        };
        let receiver = receivers.get_mut(&route.edge_id).unwrap();
        while let Some(message) = receiver.recv().await.unwrap() {
            target.push(message);
        }
    }
    while let Some(ack) = checkpoints.recv().await {
        let snapshot = capabilities[&ack.node_id]
            .decode_snapshot(
                &ack.node_id,
                crate::OperatorStateSnapshot {
                    inline_metadata: ack.state.inline_metadata,
                    segments: BTreeMap::new(),
                },
            )
            .unwrap();
        observed.restores.insert(
            ack.node_id,
            OperatorRestoreState {
                snapshot,
                progress: ack.state.progress,
                output_frontier: None,
                next_epoch: epoch.next().unwrap(),
            },
        );
    }
    observed
}

#[tokio::test]
async fn test_late_controls_and_restored_derivatives_keep_fifo_without_frontiers() {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        for kind in ["rolling", "cross_section"] {
            for path in ["direct", "expression", "sql"] {
                let mut restores = BTreeMap::new();
                for sequence in 0..2 {
                    let observed = run_graph(kind, path, restores).await;
                    assert_eq!(
                        observed
                            .late
                            .iter()
                            .map(StreamMessage::kind)
                            .collect::<Vec<_>>(),
                        [
                            StreamMessageKind::Data,
                            StreamMessageKind::Barrier,
                            StreamMessageKind::EndOfInput
                        ],
                        "{kind}/{path}"
                    );
                    assert!(
                        observed
                            .normal
                            .iter()
                            .any(|message| message.as_watermark()
                                == Some(EventTime::from_micros(10)))
                            || sequence == 1
                    );
                    assert!(observed.normal.iter().any(StreamMessage::is_idle));
                    assert_eq!(
                        observed.normal[observed.normal.len() - 2].kind(),
                        StreamMessageKind::Barrier
                    );
                    assert!(observed.normal.last().unwrap().is_end_of_input());
                    if path == "direct" {
                        assert_eq!(
                            observed.late[0].as_data().unwrap().metadata().sequence(),
                            sequence
                        );
                    }
                    if path == "sql" {
                        let batch = &observed.late[0]
                            .as_data()
                            .unwrap()
                            .table_payload()
                            .unwrap()
                            .batches()[0];
                        let counts = batch
                            .column_by_name("n")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();
                        assert_eq!(counts.values(), &[1]);
                    }
                    for (id, restore) in &observed.restores {
                        if id != "roll" {
                            assert!(
                                restore
                                    .progress
                                    .values()
                                    .all(|progress| progress.watermark.is_none())
                            );
                            assert!(restore.output_frontier.is_none());
                        }
                    }
                    restores = observed.restores;
                }
            }
        }
    })
    .await
    .unwrap();
}

struct Collectors {
    ports: BTreeMap<String, Port>,
    outputs: BTreeMap<String, Vec<EdgeSender>>,
    receivers: Vec<EdgeReceiver>,
}

fn collectors(ports: &[Port], budget: EdgeBudget, late_fanout: usize) -> Collectors {
    let mut receivers = Vec::new();
    let outputs = ports
        .iter()
        .map(|port| {
            let branches = if port.name() == "late" {
                late_fanout
            } else {
                1
            };
            let senders = (0..branches)
                .map(|index| {
                    let (sender, receiver) =
                        crate::edge_channel(format!("{}-{index}", port.name()), budget).unwrap();
                    receivers.push(receiver);
                    sender
                })
                .collect();
            (port.name().to_owned(), senders)
        })
        .collect();
    Collectors {
        ports: ports
            .iter()
            .map(|port| (port.name().to_owned(), port.clone()))
            .collect(),
        outputs,
        receivers,
    }
}

#[tokio::test]
async fn test_late_runtime_collector_preflights_later_wide_chunk_without_state_or_emission() {
    for kind in ["rolling", "cross_section"] {
        let plan = plan(kind, "direct")
            .into_runtime_parts(EdgeBudget::default())
            .unwrap();
        let mut node = plan.nodes.into_iter().next().unwrap();
        let initial = node.operator.checkpoint(Epoch::INITIAL).unwrap();
        let record = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(TimestampMicrosecondArray::from(vec![5, 6, 20]).with_timezone("UTC"))
                    as ArrayRef,
                Arc::new(StringArray::from(vec!["a", &"x".repeat(160), "a"])),
                Arc::new(UInt64Array::from(vec![0, 1, 2])),
                Arc::new(Float64Array::from(vec![Some(5.0), Some(6.0), Some(20.0)])),
            ],
        )
        .unwrap();
        let batch = Batch::table(
            vec![record],
            BatchMetadata::new("source", 0, JsonMap::new()).unwrap(),
        )
        .unwrap();
        let budget = EdgeBudget::new(3, batch.estimated_bytes().unwrap()).unwrap();
        let Collectors {
            ports,
            mut outputs,
            receivers,
        } = collectors(
            &node.output_ports.into_values().collect::<Vec<_>>(),
            budget,
            2,
        );
        let job = crate::StreamJobContext::new(
            1,
            &plan.fingerprint,
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "roll", Some(EventTime::from_micros(10)))
            .with_test_output_budget(budget);
        let progress = OperatorProgress::default();
        let metrics = MetricsRecorder::default();
        let mut output = ChannelStreamCollector::new(
            "roll",
            1,
            &ports,
            &mut outputs,
            job.cancellation(),
            &progress,
            &metrics,
        );
        let error = node
            .operator
            .process_data("input", batch, &context, &mut output)
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("output late row_index=1"),
            "{error}"
        );
        let after = node.operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
        for field in [
            "late_output",
            "metrics",
            "last_input_watermark",
            "next_output_sequence",
            "segment_inventory",
        ] {
            assert_eq!(after.inline_metadata[field], initial.inline_metadata[field]);
        }
        assert!(after.segments.is_empty());
        assert_eq!(progress.snapshot(), OperatorProgressSnapshot::default());
        assert!(
            receivers
                .iter()
                .all(|receiver| receiver.metrics().queue_depth == 0)
        );
        node.operator
            .process_data("input", late_batch(0), &context, &mut output)
            .await
            .unwrap();
        assert_eq!(
            receivers
                .iter()
                .map(|receiver| receiver.metrics().queue_depth)
                .sum::<usize>(),
            2
        );
        assert!(
            receivers
                .iter()
                .all(|receiver| receiver.metrics().charged_bytes <= budget.max_bytes)
        );
    }
}

#[tokio::test]
async fn test_late_runtime_post_enqueue_counter_failure_preserves_error_and_forbids_retry() {
    for kind in ["rolling", "cross_section"] {
        let plan = plan(kind, "direct")
            .into_runtime_parts(EdgeBudget::default())
            .unwrap();
        let mut node = plan.nodes.into_iter().next().unwrap();
        let initial = node.operator.checkpoint(Epoch::INITIAL).unwrap();
        let Collectors {
            ports,
            mut outputs,
            receivers,
        } = collectors(
            &node.output_ports.into_values().collect::<Vec<_>>(),
            EdgeBudget::default(),
            2,
        );
        let job = crate::StreamJobContext::new(
            1,
            &plan.fingerprint,
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "roll", Some(EventTime::from_micros(10)));
        let progress = OperatorProgress::default();
        progress.0.lock().fully_fanned_out_batches = u64::MAX;
        let metrics = MetricsRecorder::default();
        let mut output = ChannelStreamCollector::new(
            "roll",
            1,
            &ports,
            &mut outputs,
            job.cancellation(),
            &progress,
            &metrics,
        );
        let error = node
            .operator
            .process_data("input", late_batch(0), &context, &mut output)
            .await
            .unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Internal { ref message } if message == "operator output batch counter overflowed")
        );
        assert_eq!(
            receivers
                .iter()
                .map(|receiver| receiver.metrics().queue_depth)
                .sum::<usize>(),
            2
        );
        let retry = node
            .operator
            .process_data("input", late_batch(0), &context, &mut output)
            .await
            .unwrap_err();
        assert!(retry.to_string().contains("retry"), "{retry}");
        let after = node.operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
        for field in ["late_output", "metrics", "segment_inventory"] {
            assert_eq!(after.inline_metadata[field], initial.inline_metadata[field]);
        }
    }
}
