use super::*;
use crate::{Batch, EventTime, StreamMessage, StreamMessageKind};

async fn expression(
    name: &str,
    input: SchemaRef,
    select: &[&str],
    filter: Option<&str>,
) -> ExpressionOperator {
    let expression = ExpressionOperator::new(
        name,
        "",
        select.iter().map(|value| (*value).into()).collect(),
        filter.map(str::to_owned),
        vec![],
    )
    .unwrap();
    let output = expression.infer_stream_schema(input.clone()).await.unwrap();
    expression
        .with_ports(
            Port::with_schema_ref("input", BatchKind::Table, true, Some(input)).unwrap(),
            Port::with_schema_ref("output", BatchKind::Table, false, Some(output)).unwrap(),
        )
        .unwrap()
}

async fn calculation_plan(select: &[&str], filter: Option<&str>) -> StreamExecutionPlan {
    let head = rolling();
    let calculated = expression(
        "project",
        head.output_ports()[0].schema().unwrap().clone(),
        select,
        filter,
    )
    .await;
    let final_expression = expression(
        "tail",
        calculated.output_ports()[0].schema().unwrap().clone(),
        &["current", "previous", "scaled * 2 AS final_value"],
        None,
    )
    .await;
    PipelineBuilder::new("row-local-fusion")
        .unwrap()
        .add_node("rolling", head)
        .unwrap()
        .add_node("project", Box::new(calculated))
        .unwrap()
        .add_node("tail", Box::new(final_expression))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("rolling", "output").unwrap(),
            PortEndpoint::new("project", "input").unwrap(),
        ))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("project", "output").unwrap(),
            PortEndpoint::new("tail", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

#[tokio::test]
async fn p1a_rolling_arithmetic_filter_and_expression_tail_are_eligible() {
    let parts = calculation_plan(
        &["price AS current", "previous", "price + 1 AS scaled"],
        Some("price > 0"),
    )
    .await
    .into_runtime_parts(EdgeBudget::default())
    .unwrap();
    assert!(eligible_pair(&parts.nodes[0], &parts.nodes[1]));
    assert!(eligible_pair(&parts.nodes[1], &parts.nodes[2]));
}

#[tokio::test]
async fn p1a_computed_fusion_rejects_unknown_schema_volatile_and_nonlocal_queries() {
    let head = rolling();
    let input = head.output_ports()[0].schema().unwrap().clone();
    for select in [
        &["random() AS value"][..],
        &["now() AS value"][..],
        &["SUM(price) AS value"][..],
        &["ROW_NUMBER() OVER () AS value"][..],
        &["(SELECT MAX(price) FROM input) AS value"][..],
    ] {
        let mut parts = plan().into_runtime_parts(EdgeBudget::default()).unwrap();
        let candidate = expression("project", input.clone(), select, None).await;
        parts.nodes[1]
            .output_ports
            .insert("output".into(), candidate.output_ports()[0].clone());
        parts.nodes[1].operator = crate::pipeline::CompiledStreamOperator::Expression(candidate);
        assert!(
            !eligible_pair(&parts.nodes[0], &parts.nodes[1]),
            "{select:?}"
        );
    }
    let mut parts = calculation_plan(
        &["price AS current", "previous", "price + 1 AS scaled"],
        None,
    )
    .await
    .into_runtime_parts(EdgeBudget::default())
    .unwrap();
    parts.nodes[1].output_ports.insert(
        "output".into(),
        Port::new("output", BatchKind::Table, false, None).unwrap(),
    );
    assert!(!eligible_pair(&parts.nodes[0], &parts.nodes[1]));
}

#[tokio::test]
async fn p1a_unknown_expression_output_cannot_prove_the_next_internal_edge() {
    let mut parts = calculation_plan(
        &["price AS current", "previous", "price + 1 AS scaled"],
        None,
    )
    .await
    .into_runtime_parts(EdgeBudget::default())
    .unwrap();
    let input = parts.nodes[1].input_ports["input"].clone();
    let output = Port::new("output", BatchKind::Table, false, None).unwrap();
    let crate::pipeline::CompiledStreamOperator::Expression(operator) = &parts.nodes[1].operator
    else {
        panic!("expected expression");
    };
    parts.nodes[1].operator = crate::pipeline::CompiledStreamOperator::Expression(
        operator.clone().with_ports(input, output.clone()).unwrap(),
    );
    parts.nodes[1].output_ports.insert("output".into(), output);
    assert!(!eligible_pair(&parts.nodes[1], &parts.nodes[2]));
}

#[derive(Debug)]
struct Observed {
    output: Vec<Batch>,
    controls: Vec<StreamMessageKind>,
}

async fn observe_calculation(fused: bool) -> Observed {
    let plan = calculation_plan(
        &["price AS current", "previous", "price + 1 AS scaled"],
        Some("price > 0"),
    )
    .await;
    let mut harness =
        graph_harness_for_plan(plan, fused, EdgeBudget::default(), BTreeMap::new(), None).await;
    harness.runtime.data_gate.send(true).unwrap();
    for (sequence, values) in [
        vec![Some(-1.0), None],
        vec![Some(1.0), None, Some(3.0)],
        vec![Some(5.0)],
    ]
    .into_iter()
    .enumerate()
    {
        let sequence = u64::try_from(sequence).unwrap();
        let input = input_batch(i64::try_from(sequence * 10 + 1).unwrap(), &values, sequence);
        let original = input.table_payload().unwrap().batches()[0].clone();
        harness.send(StreamMessage::data(input.clone())).await;
        assert_eq!(input.table_payload().unwrap().batches()[0], original);
    }
    harness
        .send(StreamMessage::watermark(EventTime::from_micros(30)))
        .await;
    harness.send(StreamMessage::idle()).await;
    harness.send(StreamMessage::end_of_input()).await;
    let mut output = Vec::new();
    let mut controls = Vec::new();
    loop {
        let message = harness.receive().await;
        if let Some(batch) = message.as_data() {
            output.push(batch.clone());
        } else {
            controls.push(message.kind());
        }
        if message.is_end_of_input() {
            break;
        }
    }
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
            .all(|node| node.snapshot().ended)
    );
    assert!(
        harness
            .core
            .metrics
            .snapshot()
            .edges
            .values()
            .all(|edge| { edge.channel.charged_rows == 0 && edge.channel.charged_bytes == 0 })
    );
    Observed { output, controls }
}

#[tokio::test]
async fn p1a_computed_chain_matches_unfused_values_metadata_and_control_order() {
    use datafusion::arrow::{array::Float64Array, compute::concat_batches};
    let expected = observe_calculation(false).await;
    let actual = observe_calculation(true).await;
    assert_eq!(actual.controls, expected.controls);
    assert_eq!(actual.output.len(), expected.output.len());
    for (actual, expected) in actual.output.iter().zip(&expected.output) {
        assert_eq!(actual.metadata(), expected.metadata());
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(
            actual.table_payload().unwrap().batches(),
            expected.table_payload().unwrap().batches()
        );
    }
    assert_eq!(
        actual.controls,
        [
            StreamMessageKind::Watermark,
            StreamMessageKind::Idle,
            StreamMessageKind::EndOfInput
        ]
    );
    let records = actual
        .output
        .iter()
        .flat_map(|batch| batch.table_payload().unwrap().batches());
    let result =
        concat_batches(actual.output[0].table_payload().unwrap().schema(), records).unwrap();
    assert_eq!(
        result
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap(),
        &Float64Array::from(vec![1.0, 3.0, 5.0])
    );
    assert_eq!(
        result
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap(),
        &Float64Array::from(vec![4.0, 8.0, 12.0])
    );
    assert_eq!(result.schema().field(0).metadata()["units"], "USD");
}

fn native_computed_plan() -> StreamExecutionPlan {
    native_computed_plan_with_select("price + moving_average AS signal")
}

fn native_computed_plan_with_select(select: &str) -> StreamExecutionPlan {
    let mut project: serde_json::Value = serde_json::from_str(ORIGINAL_WARM_PROJECT).unwrap();
    let operator = &mut project["graph"]["nodes"][1]["operator"];
    operator["select"] = serde_json::json!([select]);
    operator["filter"] = serde_json::json!("price > 1");
    let project = crate::import_project_json(&serde_json::to_vec(&project).unwrap()).unwrap();
    crate::compile_stream_project_graph(
        &project,
        &crate::ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &StreamRequirements::default(),
    )
    .unwrap()
}

fn native_input(schema: SchemaRef, prices: &[f64]) -> Batch {
    use datafusion::arrow::{
        array::{Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
        record_batch::RecordBatch,
    };
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(
                        TimestampMicrosecondArray::from(
                            (1..=i64::try_from(prices.len()).unwrap()).collect::<Vec<_>>(),
                        )
                        .with_timezone("UTC"),
                    ),
                    Arc::new(UInt64Array::from(
                        (1..=u64::try_from(prices.len()).unwrap()).collect::<Vec<_>>(),
                    )),
                    Arc::new(StringArray::from(vec!["S"; prices.len()])),
                    Arc::new(Float64Array::from(prices.to_vec())),
                ],
            )
            .unwrap(),
        ],
        crate::BatchMetadata::new("native", 0, BTreeMap::new()).unwrap(),
    )
    .unwrap()
}

async fn observe_native_computation(fused: bool) -> Observed {
    let plan = native_computed_plan();
    let parts = native_computed_plan()
        .into_runtime_parts(EdgeBudget::default())
        .unwrap();
    assert!(parts.nodes[1].output_ports["output"].schema().is_none());
    if fused {
        assert!(eligible_pair(&parts.nodes[0], &parts.nodes[1]));
    }
    let schema = parts.nodes[0].input_ports["input"]
        .schema()
        .unwrap()
        .clone();
    let input = native_input(schema, &[1.0, 3.0, 5.0]);
    let mut harness =
        graph_harness_for_plan(plan, fused, EdgeBudget::default(), BTreeMap::new(), None).await;
    harness.runtime.data_gate.send(true).unwrap();
    harness.send(StreamMessage::data(input)).await;
    harness
        .send(StreamMessage::watermark(EventTime::from_micros(3)))
        .await;
    harness.send(StreamMessage::end_of_input()).await;
    let mut output = Vec::new();
    let mut controls = Vec::new();
    loop {
        let message = harness.receive().await;
        if let Some(batch) = message.as_data() {
            output.push(batch.clone());
        } else {
            controls.push(message.kind());
        }
        if message.is_end_of_input() {
            break;
        }
    }
    assert!(
        harness
            .runtime
            .supervisor
            .join_all()
            .await
            .errors
            .is_empty()
    );
    Observed { output, controls }
}

#[tokio::test]
async fn p1a_native_computation_without_declared_output_preserves_execution() {
    let expected = observe_native_computation(false).await;
    let actual = observe_native_computation(true).await;
    assert_eq!(actual.controls, expected.controls);
    assert_eq!(actual.output.len(), expected.output.len());
    for (actual, expected) in actual.output.iter().zip(&expected.output) {
        assert_eq!(actual.metadata(), expected.metadata());
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(
            actual.table_payload().unwrap().batches(),
            expected.table_payload().unwrap().batches()
        );
    }
    assert_eq!(actual.output.iter().map(Batch::num_rows).sum::<usize>(), 2);
}

#[tokio::test]
async fn p1a_unknown_tail_preserves_calculation_and_sink_budget_failures() {
    for budget_failure in [false, true] {
        let mut failures = Vec::new();
        for fused in [false, true] {
            let select = if budget_failure {
                "price + moving_average AS signal"
            } else {
                "CAST(price AS BIGINT) AS signal"
            };
            let mut parts = native_computed_plan_with_select(select)
                .into_runtime_parts(EdgeBudget::default())
                .unwrap();
            let producer = parts.nodes[0].node_id.clone();
            let consumer = parts.nodes[1].node_id.clone();
            let schema = parts.nodes[0].input_ports["input"]
                .schema()
                .unwrap()
                .clone();
            if budget_failure {
                let edge = parts.sink_routes.values().next().unwrap().edge_id.clone();
                parts.edges.get_mut(&edge).unwrap().budget = EdgeBudget::new(1, 1 << 20).unwrap();
            }
            let mut harness = graph_harness_for_parts(parts, fused, BTreeMap::new(), None).await;
            harness.runtime.data_gate.send(true).unwrap();
            let prices = if budget_failure {
                [3.0, 5.0]
            } else {
                [1e100, 3e100]
            };
            harness
                .send(StreamMessage::data(native_input(schema, &prices)))
                .await;
            harness
                .send(StreamMessage::watermark(EventTime::from_micros(2)))
                .await;
            let report = harness.runtime.supervisor.join_all().await;
            assert_eq!(report.primary_errors().len(), 1, "{report:?}");
            let error = &report.primary_errors()[0];
            assert_eq!(error.task_id, TaskId::new(1));
            assert_eq!(error.task_name, format!("operator:{consumer}"));
            if budget_failure {
                assert!(matches!(
                    error.error,
                    crate::CalcFlowError::EdgeBudgetExceeded { .. }
                ));
            }
            failures.push(error.error.to_string());
            let status = &harness.core.runtime_status.lock().nodes;
            assert_eq!(status[&producer].snapshot().fully_fanned_out_batches, 1);
            assert_eq!(status[&consumer].snapshot().fully_fanned_out_batches, 0);
            harness.runtime.sink_inputs.clear();
            assert!(
                harness.core.metrics.snapshot().edges.values().all(|edge| {
                    edge.channel.charged_rows == 0 && edge.channel.charged_bytes == 0
                })
            );
        }
        assert_eq!(failures[0], failures[1]);
    }
}

#[tokio::test]
async fn p1a_computed_chain_cancellation_releases_full_edge_budgets() {
    let plan = calculation_plan(
        &["price AS current", "previous", "price + 1 AS scaled"],
        Some("price > 0"),
    )
    .await;
    let mut harness = graph_harness_for_plan(
        plan,
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
    assert!(harness.core.metrics.snapshot().edges.values().all(|edge| {
        edge.channel.charged_rows == 0
            && edge.channel.charged_bytes == 0
            && edge.channel.queue_depth == 0
    }));
}

#[tokio::test]
async fn p1a_computed_failure_preserves_upstream_acceptance_and_error_owner() {
    for fused in [false, true] {
        let plan = calculation_plan(
            &[
                "price AS current",
                "previous",
                "CAST(price AS BIGINT) AS scaled",
            ],
            None,
        )
        .await;
        let mut harness =
            graph_harness_for_plan(plan, fused, EdgeBudget::default(), BTreeMap::new(), None).await;
        harness.runtime.data_gate.send(true).unwrap();
        harness
            .send(StreamMessage::data(input_batch(1, &[Some(1e100)], 0)))
            .await;
        harness
            .send(StreamMessage::watermark(EventTime::from_micros(1)))
            .await;
        let report = harness.runtime.supervisor.join_all().await;
        assert_eq!(report.primary_errors().len(), 1, "{report:?}");
        assert_eq!(report.primary_errors()[0].task_id, TaskId::new(1));
        assert_eq!(report.primary_errors()[0].task_name, "operator:project");
        assert_eq!(
            harness.core.runtime_status.lock().nodes["rolling"]
                .snapshot()
                .fully_fanned_out_batches,
            1
        );
        assert_eq!(
            harness.core.runtime_status.lock().nodes["project"]
                .snapshot()
                .fully_fanned_out_batches,
            0
        );
        harness.runtime.sink_inputs.clear();
        assert!(
            harness
                .core
                .metrics
                .snapshot()
                .edges
                .values()
                .all(|edge| edge.channel.charged_rows == 0 && edge.channel.charged_bytes == 0)
        );
    }
}

async fn asof_calculation_plan() -> StreamExecutionPlan {
    let head = super::super::asof_chain_tests::asof();
    let calculated = expression(
        "project",
        head.output_ports()[0].schema().unwrap().clone(),
        &["right__price + 1 AS quote", "left__sequence AS id"],
        Some("right__price > 0"),
    )
    .await;
    PipelineBuilder::new("asof-row-local-fusion")
        .unwrap()
        .add_node("a_asof", Box::new(head))
        .unwrap()
        .add_node("project", Box::new(calculated))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("a_asof", "output").unwrap(),
            PortEndpoint::new("project", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

#[tokio::test]
async fn p1a_asof_arithmetic_filter_preserves_dual_ingress_and_results() {
    let mut outputs = Vec::new();
    for fused in [false, true] {
        let plan = asof_calculation_plan().await;
        let parts = asof_calculation_plan()
            .await
            .into_runtime_parts(EdgeBudget::default())
            .unwrap();
        let routes = parts
            .source_routes
            .into_iter()
            .map(|(binding, route)| (binding, route.target.port))
            .collect::<Vec<_>>();
        let mut harness =
            graph_harness_for_plan(plan, fused, EdgeBudget::default(), BTreeMap::new(), None).await;
        assert_eq!(harness.runtime.source_outputs.len(), 2);
        harness.runtime.data_gate.send(true).unwrap();
        for (binding, port) in routes {
            let sender = &mut harness.runtime.source_outputs.get_mut(&binding).unwrap()[0];
            sender
                .send(StreamMessage::data(
                    super::super::asof_chain_tests::input_batch(port == "left"),
                ))
                .await
                .unwrap();
            sender.send(StreamMessage::end_of_input()).await.unwrap();
        }
        let mut output = Vec::new();
        loop {
            let message = harness.receive().await;
            if let Some(batch) = message.as_data() {
                output.push(batch.clone());
            }
            if message.is_end_of_input() {
                break;
            }
        }
        assert!(
            harness
                .runtime
                .supervisor
                .join_all()
                .await
                .errors
                .is_empty()
        );
        let status = harness.core.runtime_status.lock().nodes["a_asof"]
            .snapshot()
            .stream_asof_join
            .clone()
            .unwrap();
        assert_eq!((status.emitted_left_rows, status.pending_left_rows), (4, 0));
        outputs.push(output);
    }
    assert_eq!(outputs[0].len(), outputs[1].len());
    for (expected, actual) in outputs[0].iter().zip(&outputs[1]) {
        assert_eq!(actual.metadata(), expected.metadata());
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(
            actual.table_payload().unwrap().batches(),
            expected.table_payload().unwrap().batches()
        );
    }
    assert_eq!(outputs[1].iter().map(Batch::num_rows).sum::<usize>(), 2);
}

async fn checkpoint_computation(
    transaction: &Arc<crate::state::ManifestTransaction>,
) -> BTreeMap<String, crate::OperatorManifestEntry> {
    let plan = calculation_plan(
        &["price AS current", "previous", "price + 1 AS scaled"],
        Some("price > 0"),
    )
    .await;
    let mut before = graph_harness_for_plan(
        plan,
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
    before
        .send(StreamMessage::barrier(crate::Epoch::INITIAL))
        .await;
    assert_eq!(before.receive().await.as_data().unwrap().num_rows(), 2);
    assert_eq!(
        before.receive().await.as_watermark(),
        Some(EventTime::from_micros(2))
    );
    assert_eq!(
        before.receive().await.as_barrier(),
        Some(crate::Epoch::INITIAL)
    );
    let mut entries = BTreeMap::new();
    for _ in 0..3 {
        let ack = before.acks.recv().await.unwrap();
        assert_eq!(ack.epoch, crate::Epoch::INITIAL);
        entries.insert(ack.node_id, ack.state);
    }
    assert_eq!(
        entries.keys().map(String::as_str).collect::<Vec<_>>(),
        ["project", "rolling", "tail"]
    );
    before.cancel().await;
    entries
}

async fn terminal_checkpoints(harness: &mut GraphHarness, terminal_epoch: crate::Epoch) {
    use crate::runtime::streaming::operator_task::OperatorCheckpointCommand;
    for _ in 0..harness.terminal_commands.len() {
        let node = harness.terminal_ready.recv().await.unwrap();
        harness.terminal_commands[&node]
            .send(OperatorCheckpointCommand::Terminal(terminal_epoch))
            .await
            .unwrap();
    }
    for _ in 0..harness.terminal_commands.len() {
        let ack = harness.acks.recv().await.unwrap();
        assert_eq!(ack.epoch, terminal_epoch);
        assert_eq!(
            ack.state.progress["input"].state,
            crate::ManifestIngressState::Ended
        );
    }
}

#[tokio::test]
async fn p1a_computed_chain_restores_unfused_pending_state_and_terminal_checkpoints() {
    use crate::runtime::streaming::operator_task::OperatorRestoreState;
    use crate::{Epoch, LocalStateBackend, StateBackend, StateLineageKey};
    use datafusion::arrow::array::Float64Array;
    let plan = calculation_plan(
        &["price AS current", "previous", "price + 1 AS scaled"],
        Some("price > 0"),
    )
    .await;
    let directory = tempfile::tempdir().unwrap();
    let key = StateLineageKey::new(plan.name(), plan.fingerprint()).unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let transaction = Arc::new(
        crate::state::ManifestTransaction::open(
            Arc::from(backend.open_lineage(&key).await.unwrap()),
            &key,
            directory.path().join("manifests"),
            2,
        )
        .await
        .unwrap(),
    );
    let entries = checkpoint_computation(&transaction).await;
    let nodes = calculation_plan(
        &["price AS current", "previous", "price + 1 AS scaled"],
        Some("price > 0"),
    )
    .await
    .into_runtime_parts(EdgeBudget::default())
    .unwrap()
    .nodes;
    let mut restores = BTreeMap::new();
    for node in nodes {
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
    let mut after = graph_harness_for_plan(
        plan,
        true,
        EdgeBudget::default(),
        restores,
        Some(transaction),
    )
    .await;
    after.runtime.data_gate.send(true).unwrap();
    after
        .send(StreamMessage::data(input_batch(4, &[Some(7.0)], 1)))
        .await;
    after
        .send(StreamMessage::watermark(EventTime::from_micros(4)))
        .await;
    after.send(StreamMessage::end_of_input()).await;
    let data = after.receive().await;
    let result = &data.as_data().unwrap().table_payload().unwrap().batches()[0];
    assert_eq!(
        result
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap(),
        &Float64Array::from(vec![3.0, 5.0])
    );
    assert_eq!(
        result
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap(),
        &Float64Array::from(vec![12.0, 16.0])
    );
    assert_eq!(
        after.receive().await.as_watermark(),
        Some(EventTime::from_micros(4))
    );
    assert!(after.receive().await.is_end_of_input());
    terminal_checkpoints(&mut after, Epoch::INITIAL.next().unwrap()).await;
    assert!(after.runtime.supervisor.join_all().await.errors.is_empty());
}
