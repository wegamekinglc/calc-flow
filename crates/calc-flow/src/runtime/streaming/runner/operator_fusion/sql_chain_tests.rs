mod controls;
mod faults;
mod lifecycle;

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use datafusion::arrow::{
    array::{Int64Array, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use tokio::sync::mpsc;

use crate::{
    Batch, BatchMetadata, CancellationToken, Edge, EdgeBudget, ExpressionOperator, JsonMap,
    PipelineBuilder, PortEndpoint, SqlOperator, StateBudget, StreamExecutionPlan, StreamJobContext,
    StreamMessage, StreamRequirements, UdfRegistry,
    pipeline::{RuntimeEdgeKind, StreamRuntimePlanParts},
    runtime::streaming::{
        metrics::MetricsRecorder,
        projection::StatusProjection,
        runner::{JobCore, LaunchId, RegisteredRuntime, run_operator_entry},
    },
};

const HEAD: &str = "deltas";
const TAIL: &str = "positions";
const SELECT: &[&str] = &[
    "account",
    "stock",
    "CASE WHEN side = 'buy' THEN quantity ELSE -quantity END AS delta",
];

fn position_sql(alias: &str, compact: bool) -> SqlOperator {
    let mut sql = SqlOperator::new(
        TAIL,
        &format!(
            "SELECT account, stock, SUM(delta) AS position FROM \"{alias}\" GROUP BY account, stock"
        ),
        vec![alias.into()],
        vec![],
    )
    .unwrap();
    if compact {
        sql.set_compact_state_budget(Some(StateBudget::new(100, 1 << 24).unwrap()))
            .unwrap();
    }
    sql
}

fn plan(alias: &str, compact: bool) -> StreamExecutionPlan {
    let expression = ExpressionOperator::new(
        HEAD,
        "",
        SELECT.iter().map(|value| (*value).into()).collect(),
        Some("flow_type = 'trade'".into()),
        vec![],
    )
    .unwrap();
    build_plan(expression, position_sql(alias, compact), alias)
}

fn build_plan(
    expression: ExpressionOperator,
    sql: SqlOperator,
    alias: &str,
) -> StreamExecutionPlan {
    PipelineBuilder::new("stock-positions")
        .unwrap()
        .add_node(HEAD, Box::new(expression))
        .unwrap()
        .add_node(TAIL, Box::new(sql))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new(HEAD, "output").unwrap(),
            PortEndpoint::new(TAIL, alias).unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("account", DataType::Utf8, false),
        Field::new("stock", DataType::Utf8, false),
        Field::new("side", DataType::Utf8, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("flow_type", DataType::Utf8, false),
    ]))
}

fn input(sequence: u64, quantity: &[i64], trades: bool) -> Batch {
    let len = quantity.len();
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(StringArray::from(vec!["a"; len])),
                    Arc::new(StringArray::from(vec!["s"; len])),
                    Arc::new(StringArray::from(
                        (0..len)
                            .map(|index| if index % 2 == 0 { "buy" } else { "sell" })
                            .collect::<Vec<_>>(),
                    )),
                    Arc::new(Int64Array::from(quantity.to_vec())),
                    Arc::new(StringArray::from(vec![
                        if trades {
                            "trade"
                        } else {
                            "deposit"
                        };
                        len
                    ])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::new("source", sequence, JsonMap::new()).unwrap(),
    )
    .unwrap()
}

struct Harness {
    runtime: RegisteredRuntime,
    core: Arc<JobCore>,
    cancellation: CancellationToken,
    acks: mpsc::Receiver<crate::runtime::streaming::operator_task::OperatorCheckpointAck>,
    terminal_ready: mpsc::Receiver<String>,
    terminal_commands: BTreeMap<
        String,
        mpsc::Sender<crate::runtime::streaming::operator_task::OperatorCheckpointCommand>,
    >,
}

async fn harness(parts: StreamRuntimePlanParts, fused: bool) -> Harness {
    harness_with_state(parts, fused, BTreeMap::new(), None).await
}

async fn harness_with_state(
    mut parts: StreamRuntimePlanParts,
    fused: bool,
    restores: BTreeMap<String, crate::runtime::streaming::operator_task::OperatorRestoreState>,
    transaction: Option<Arc<crate::state::ManifestTransaction>>,
) -> Harness {
    let logical_nodes = parts.nodes.len();
    if !fused {
        // Change only the private layout proof; traffic and endpoints stay identical.
        for edge in parts
            .edges
            .values_mut()
            .filter(|edge| edge.kind == RuntimeEdgeKind::Internal)
        {
            edge.kind = RuntimeEdgeKind::SourceBoundary;
        }
    }
    let cancellation = CancellationToken::new();
    let context = StreamJobContext::new(
        31,
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
        parts.source_routes.keys().cloned(),
        parts.nodes.iter().map(|node| node.node_id.clone()),
        parts.sink_routes.keys().cloned(),
    );
    let (commands, _) = mpsc::unbounded_channel();
    let core = Arc::new(JobCore::new(
        LaunchId::new(0),
        31,
        commands,
        metrics,
        StatusProjection::default(),
        transaction.is_some(),
        parts.name.clone(),
    ));
    core.sql_recovery.configure(1).unwrap();
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
    let checkpoint = transaction.map(|transaction| super::super::OperatorCheckpointRegistration {
        acks: acks_tx,
        transaction,
        terminal_ready: ready_tx,
        terminal_commands: Arc::new(parking_lot::Mutex::new(receivers)),
        faults: super::super::CheckpointFaultInjector::default(),
        fault_cancellation: cancellation.clone(),
    });
    let runtime = run_operator_entry(parts, &context, &core, &cancellation, restores, checkpoint)
        .await
        .unwrap_or_else(|_| panic!("SQL chain entry failed"));
    assert_eq!(
        runtime.supervisor.physical_driver_count(),
        if fused {
            logical_nodes - 1
        } else {
            logical_nodes
        }
    );
    Harness {
        runtime,
        core,
        cancellation,
        acks,
        terminal_ready,
        terminal_commands,
    }
}

impl Harness {
    async fn send(&mut self, message: StreamMessage) {
        self.runtime.source_outputs.values_mut().next().unwrap()[0]
            .send(message)
            .await
            .unwrap();
    }

    async fn receive(&mut self) -> StreamMessage {
        let message = tokio::time::timeout(
            Duration::from_secs(5),
            self.runtime.sink_inputs.values_mut().next().unwrap().recv(),
        )
        .await
        .unwrap()
        .unwrap();
        match message {
            Some(message) => message,
            None => panic!(
                "SQL chain closed early: {:?}",
                self.runtime.supervisor.join_all().await
            ),
        }
    }

    async fn finish(&mut self) {
        assert!(self.runtime.supervisor.join_all().await.errors.is_empty());
        self.core.sql_recovery.close_admission();
        assert!(self.core.sql_recovery.drain().await.is_empty());
        assert_eq!(self.core.sql_recovery.live_counts(), (0, 0));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_example30_unknown_schemas_and_noninput_sql_alias_are_grouped() {
    for alias in ["input", "trades"] {
        let parts = plan(alias, false)
            .into_runtime_parts(EdgeBudget::default())
            .unwrap();
        assert!(parts.nodes.iter().all(|node| {
            node.input_ports
                .values()
                .chain(node.output_ports.values())
                .all(|port| port.schema().is_none())
        }));
        assert!(super::eligible_pair(&parts.nodes[0], &parts.nodes[1]));
        assert_eq!(super::plan_fusion(&parts).len(), 1);
        assert!(!parts.nodes[0].operator.datafusion_runtime_initialized());
        assert!(!parts.nodes[1].operator.datafusion_runtime_initialized());
        let mut harness = harness(parts, true).await;
        assert_eq!(harness.runtime.supervisor.registry().snapshot().len(), 2);
        harness.cancellation.cancel();
        harness.finish().await;
    }
}

fn scenario_batch(sequence: u64, values: &[i64], trades: bool) -> Batch {
    let batch = input(sequence, values, trades);
    let record = &batch.table_payload().unwrap().batches()[0];
    let records = match sequence {
        1 => vec![record.slice(0, 1), record.slice(1, 1)],
        3 => {
            let mut columns = record.columns().to_vec();
            columns[0] = Arc::new(StringArray::from(vec!["b"]));
            columns[1] = Arc::new(StringArray::from(vec!["t"]));
            vec![RecordBatch::try_new(record.schema(), columns).unwrap()]
        }
        _ => vec![record.clone()],
    };
    Batch::table(records, batch.metadata().clone()).unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_grouped_sql_keeps_every_full_snapshot_and_original_metadata() {
    let mut observations = Vec::new();
    for fused in [false, true] {
        let parts = plan("trades", true)
            .into_runtime_parts(EdgeBudget::default())
            .unwrap();
        let mut harness = harness(parts, fused).await;
        harness.runtime.data_gate.send(true).unwrap();
        let mut snapshots = Vec::new();
        for (sequence, (values, trades)) in [
            (vec![], true),
            (vec![2, 5], true),
            (vec![999], false),
            (vec![3], true),
        ]
        .into_iter()
        .enumerate()
        {
            let input = scenario_batch(u64::try_from(sequence).unwrap(), &values, trades);
            let before = input.table_payload().unwrap().batches().to_vec();
            harness.send(StreamMessage::data(input.clone())).await;
            let message = harness.receive().await;
            snapshots.push(message.as_data().unwrap().clone());
            assert_eq!(input.table_payload().unwrap().batches(), before);
        }
        harness.send(StreamMessage::end_of_input()).await;
        assert!(harness.receive().await.is_end_of_input());
        harness.finish().await;
        assert_eq!(harness.core.metrics.snapshot().nodes[HEAD].input_batches, 4);
        assert_eq!(harness.core.metrics.snapshot().nodes[TAIL].input_batches, 4);
        observations.push(snapshots);
    }
    assert_eq!(observations[0].len(), 4);
    assert_eq!(observations[0][3].num_rows(), 2);
    for (expected, actual) in observations[0].iter().zip(&observations[1]) {
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
}

#[test]
fn p1b_rejects_nonlocal_or_nonimmutable_heads_without_initializing_runtime() {
    for select in [
        "random() AS delta",
        "now() AS delta",
        "SUM(quantity) AS delta",
        "ROW_NUMBER() OVER () AS delta",
        "(SELECT MAX(quantity) FROM input) AS delta",
        "other.quantity AS delta",
    ] {
        let head = ExpressionOperator::new(HEAD, "", vec![select.into()], None, vec![]).unwrap();
        let mut parts = plan("input", false)
            .into_runtime_parts(EdgeBudget::default())
            .unwrap();
        parts.nodes[0].operator = crate::pipeline::CompiledStreamOperator::Expression(head);
        assert!(
            !super::eligible_pair(&parts.nodes[0], &parts.nodes[1]),
            "{select}"
        );
        assert!(super::plan_fusion(&parts).is_empty());
        assert!(!parts.nodes[0].operator.datafusion_runtime_initialized());
        assert!(!parts.nodes[1].operator.datafusion_runtime_initialized());
    }
}

#[test]
fn p1b_rejects_sql_outside_the_grouped_builtin_sum_candidate() {
    for query in [
        "SELECT account, COUNT(delta) FROM input GROUP BY account",
        "SELECT account, AVG(delta) FROM input GROUP BY account",
        "SELECT account, SUM(DISTINCT delta) FROM input GROUP BY account",
        "SELECT account, SUM(delta) FROM input WHERE delta > 0 GROUP BY account",
        "SELECT account, SUM(delta) FROM input GROUP BY account HAVING SUM(delta) > 0",
        "SELECT account, SUM(delta) FROM input GROUP BY account ORDER BY account",
        "SELECT account, SUM(delta) OVER () FROM input",
        "SELECT account, SUM(random()) FROM input GROUP BY account",
        "SELECT account, SUM(delta) FROM other GROUP BY account",
    ] {
        let head = ExpressionOperator::new(
            HEAD,
            "",
            SELECT.iter().map(|value| (*value).into()).collect(),
            Some("flow_type = 'trade'".into()),
            vec![],
        )
        .unwrap();
        let sql = SqlOperator::new(TAIL, query, vec!["input".into()], vec![]).unwrap();
        let mut parts = build_plan(head, position_sql("input", false), "input")
            .into_runtime_parts(EdgeBudget::default())
            .unwrap();
        parts.nodes[1].operator = crate::pipeline::CompiledStreamOperator::Sql(sql);
        assert!(
            !super::eligible_pair(&parts.nodes[0], &parts.nodes[1]),
            "{query}"
        );
    }
}

#[test]
fn p1b_rejects_udf_resources_on_either_logical_member() {
    use crate::{DataFusionConfig, UdfKind, UdfReference, pipeline::CompiledStreamOperator};
    let selected =
        UdfReference::new("fixture", "selected", "1", UdfKind::DataFusionScalar).unwrap();
    for ordinal in [0, 1] {
        let mut parts = plan("input", false)
            .into_runtime_parts(EdgeBudget::default())
            .unwrap();
        match &mut parts.nodes[ordinal].operator {
            CompiledStreamOperator::Expression(operator) => operator.set_stream_resources(
                DataFusionConfig::default(),
                UdfRegistry::new().snapshot(),
                vec![selected.clone()],
            ),
            CompiledStreamOperator::Sql(operator) => operator.set_stream_resources(
                DataFusionConfig::default(),
                UdfRegistry::new().snapshot(),
                vec![selected.clone()],
            ),
            _ => unreachable!(),
        }
        assert!(!super::eligible_pair(&parts.nodes[0], &parts.nodes[1]));
    }
}

#[test]
fn p1b_actual_edge_alias_must_match_the_sql_ingress() {
    use crate::pipeline::RuntimeConsumer;
    let mut parts = plan("trades", false)
        .into_runtime_parts(EdgeBudget::default())
        .unwrap();
    let edge = parts
        .edges
        .values_mut()
        .find(|edge| edge.kind == RuntimeEdgeKind::Internal)
        .unwrap();
    edge.consumer = RuntimeConsumer::Node {
        node_id: TAIL.into(),
        ingress: "input".into(),
    };
    assert!(super::plan_fusion(&parts).is_empty());
}

#[test]
fn p1b_unproved_ports_fanout_and_late_paths_keep_the_original_tasks() {
    use crate::{BatchKind, Port};
    let mutations: &[fn(&mut StreamRuntimePlanParts)] = &[
        |parts| {
            parts.nodes[0].input_ports.insert(
                "input".into(),
                Port::with_schema_ref("input", BatchKind::Table, true, Some(schema())).unwrap(),
            );
        },
        |parts| {
            parts.nodes[0].output_ports.insert(
                "output".into(),
                Port::with_schema_ref("output", BatchKind::Table, true, Some(schema())).unwrap(),
            );
        },
        |parts| {
            parts.nodes[1].input_ports.insert(
                "trades".into(),
                Port::with_schema_ref("trades", BatchKind::Table, true, Some(schema())).unwrap(),
            );
        },
        |parts| {
            parts.nodes[1].output_ports.insert(
                "output".into(),
                Port::with_schema_ref("output", BatchKind::Table, true, Some(schema())).unwrap(),
            );
        },
        |parts| {
            parts.nodes[0]
                .output_edges
                .get_mut("output")
                .unwrap()
                .push("another".into());
        },
        |parts| {
            parts.nodes[1]
                .output_edges
                .get_mut("output")
                .unwrap()
                .push("another".into());
        },
        |parts| {
            parts.nodes[0].late_output_ports.insert("output".into());
        },
        |parts| {
            parts.nodes[1].late_output_ports.insert("output".into());
        },
        |parts| {
            parts.nodes[1]
                .ingress_edges
                .insert("another".into(), "edge".into());
        },
    ];
    for (index, mutate) in mutations.iter().enumerate() {
        let mut parts = plan("trades", false)
            .into_runtime_parts(EdgeBudget::default())
            .unwrap();
        mutate(&mut parts);
        assert!(
            !super::eligible_pair(&parts.nodes[0], &parts.nodes[1]),
            "mutation {index}"
        );
        assert!(super::plan_fusion(&parts).is_empty());
    }
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_actual_python_holdings_export_groups_only_the_case_to_sql_link() {
    use datafusion::arrow::array::Float64Array;
    let project =
        crate::import_project_json(include_bytes!("sql_chain_tests/holdings_project.json"))
            .unwrap();
    let plan = crate::compile_stream_project_graph(
        &project,
        &crate::ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &StreamRequirements::default(),
    )
    .unwrap();
    let mut parts = plan.into_runtime_parts(EdgeBudget::default()).unwrap();
    assert_eq!(parts.nodes.len(), 4);
    assert!(!super::eligible_pair(&parts.nodes[0], &parts.nodes[1]));
    assert!(super::eligible_pair(&parts.nodes[1], &parts.nodes[2]));
    assert!(!super::eligible_pair(&parts.nodes[2], &parts.nodes[3]));
    assert_eq!(super::plan_fusion(&parts).len(), 1);
    let crate::pipeline::CompiledStreamOperator::Sql(sql) = &mut parts.nodes[2].operator else {
        panic!("SQL member missing")
    };
    sql.set_compact_state_budget(Some(StateBudget::new(100, 1 << 24).unwrap()))
        .unwrap();
    let schema = parts.nodes[0].input_ports["input"]
        .schema()
        .unwrap()
        .clone();
    let batch = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["a", "a"])),
                    Arc::new(StringArray::from(vec!["s", "s"])),
                    Arc::new(Int64Array::from(vec![2, 5])),
                    Arc::new(StringArray::from(vec!["buy", "sell"])),
                    Arc::new(Float64Array::from(vec![1.0, 1.0])),
                    Arc::new(StringArray::from(vec!["trade", "trade"])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::new("python", 0, JsonMap::new()).unwrap(),
    )
    .unwrap();
    let mut harness = harness(parts, true).await;
    harness.runtime.data_gate.send(true).unwrap();
    harness.send(StreamMessage::data(batch)).await;
    let message = harness.receive().await;
    let table = message.as_data().unwrap().table_payload().unwrap();
    assert_eq!(
        table.batches()[0]
            .column_by_name("position")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        -3
    );
    harness.send(StreamMessage::end_of_input()).await;
    assert!(harness.receive().await.is_end_of_input());
    harness.finish().await;
}
