mod emission_controls;

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use datafusion::arrow::{
    array::{Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use tokio::sync::mpsc;

use crate::{
    AsofJoinSide, AsofStateLimits, Batch, BatchKind, BatchMetadata, CancellationToken, Edge,
    EdgeBudget, ExpressionOperator, JsonMap, OperatorMetadata, PipelineBuilder, Port, PortEndpoint,
    StreamAsofJoinOperator, StreamAsofJoinSpec, StreamExecutionPlan, StreamJobContext,
    StreamMessage, StreamMessageKind, StreamRequirements, UdfRegistry,
    runtime::streaming::{
        channel::fusion_observer::{Counts, Observer},
        metrics::{M2MetricsSnapshot, MetricsRecorder},
        projection::StatusProjection,
        runner::{JobCore, LaunchId, run_operator_entry},
        supervisor::{TaskId, TaskStatus},
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
        [("origin".into(), "a13-dual-ingress".into())]
            .into_iter()
            .collect(),
    ))
}

fn asof() -> StreamAsofJoinOperator {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["symbol".into()],
            "ts".into(),
            vec!["sequence".into()],
            prefix.into(),
        )
        .unwrap()
    };
    StreamAsofJoinOperator::new(
        "a_asof",
        input_schema(),
        input_schema(),
        StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(1_000, 16 << 20).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

fn select(
    name: &str,
    input: SchemaRef,
    mapping: &[(&str, &str)],
) -> (ExpressionOperator, SchemaRef) {
    let output = Arc::new(Schema::new_with_metadata(
        mapping
            .iter()
            .map(|(old, new)| input.field_with_name(old).unwrap().clone().with_name(*new))
            .collect::<Vec<_>>(),
        input.metadata().clone(),
    ));
    let operator = ExpressionOperator::new(
        name,
        "",
        mapping
            .iter()
            .map(|(old, new)| format!("{old} AS {new}"))
            .collect(),
        None,
        vec![],
    )
    .unwrap()
    .with_ports(
        Port::with_schema_ref("input", BatchKind::Table, true, Some(input)).unwrap(),
        Port::with_schema_ref("output", BatchKind::Table, true, Some(output.clone())).unwrap(),
    )
    .unwrap();
    (operator, output)
}

fn plan(two_selects: bool, interposed: bool) -> StreamExecutionPlan {
    let head = asof();
    let (first, projected) = select(
        "c_select1",
        head.output_ports()[0].schema().unwrap().clone(),
        &[("right__price", "quote"), ("left__sequence", "id")],
    );
    let mut builder = PipelineBuilder::new("a13-asof-chain")
        .unwrap()
        .add_node("a_asof", head)
        .unwrap()
        .add_node("c_select1", Box::new(first))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("a_asof", "output").unwrap(),
            PortEndpoint::new("c_select1", "input").unwrap(),
        ))
        .unwrap();
    if two_selects {
        let (second, _) = select(
            "d_select2",
            projected,
            &[("id", "row_id"), ("quote", "final_price")],
        );
        builder = builder
            .add_node("d_select2", Box::new(second))
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("c_select1", "output").unwrap(),
                PortEndpoint::new("d_select2", "input").unwrap(),
            ))
            .unwrap();
    }
    if interposed {
        let (unrelated, _) = select("b_unrelated", input_schema(), &[("price", "unused")]);
        builder = builder
            .add_node("b_unrelated", Box::new(unrelated))
            .unwrap();
    }
    builder
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

#[test]
fn a13_asof_exact_select_is_eligible_with_both_original_ingresses() {
    let parts = plan(false, false)
        .into_runtime_parts(EdgeBudget::default())
        .unwrap();
    assert_eq!(parts.nodes[0].node_id, "a_asof");
    assert_eq!(
        parts.nodes[0]
            .ingress_edges
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["left", "right"]
    );
    assert_eq!(parts.source_routes.len(), 2);
    assert!(super::eligible_pair(&parts.nodes[0], &parts.nodes[1]));
}

struct EntryObservation {
    drivers: usize,
    readiness_pairs: usize,
    tasks: BTreeMap<TaskId, TaskStatus>,
    source_count: usize,
    registry_empty_after_join: bool,
    clean_join: bool,
}

async fn entry_observation(two_selects: bool, interposed: bool) -> EntryObservation {
    let plan = plan(two_selects, interposed);
    let cancellation = CancellationToken::new();
    let context = StreamJobContext::new(
        13,
        plan.fingerprint(),
        JsonMap::new(),
        None,
        cancellation.clone(),
    );
    let (commands, _commands_rx) = mpsc::unbounded_channel();
    let core = Arc::new(JobCore::new(
        LaunchId::new(0),
        13,
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
        panic!("valid ASOF graph entry failed");
    };
    let drivers = runtime.supervisor.physical_driver_count();
    let readiness_pairs = runtime.supervisor.ready_pair_spawn_count();
    let tasks = runtime.supervisor.registry().snapshot();
    let source_count = runtime.source_outputs.len();
    cancellation.cancel();
    let report = runtime.supervisor.join_all().await;
    let registry_empty_after_join = runtime.supervisor.registry().snapshot().is_empty();
    EntryObservation {
        drivers,
        readiness_pairs,
        tasks,
        source_count,
        registry_empty_after_join,
        clean_join: report.errors.is_empty(),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn a13_asof_pair_real_entry_has_one_ready_driver_and_original_task_ids() {
    let observed = entry_observation(false, false).await;
    assert!(observed.clean_join && observed.registry_empty_after_join);
    assert_eq!(observed.source_count, 2);
    assert_eq!(observed.tasks.len(), 2);
    assert_eq!(observed.tasks[&TaskId::new(0)].task_name, "operator:a_asof");
    assert_eq!(
        observed.tasks[&TaskId::new(1)].task_name,
        "operator:c_select1"
    );
    assert_eq!(observed.drivers, 1);
    assert_eq!(observed.readiness_pairs, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn a13_asof_two_selects_real_entry_uses_one_operator_driver() {
    let observed = entry_observation(true, false).await;
    assert!(observed.clean_join && observed.registry_empty_after_join);
    assert_eq!(observed.source_count, 2);
    assert_eq!(observed.tasks.len(), 3);
    assert_eq!(observed.tasks[&TaskId::new(0)].task_name, "operator:a_asof");
    assert_eq!(
        observed.tasks[&TaskId::new(1)].task_name,
        "operator:c_select1"
    );
    assert_eq!(
        observed.tasks[&TaskId::new(2)].task_name,
        "operator:d_select2"
    );
    assert_eq!(observed.drivers, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn a13_nonadjacent_asof_chain_preserves_topological_logical_task_ids() {
    let parts = plan(true, true)
        .into_runtime_parts(EdgeBudget::default())
        .unwrap();
    assert_eq!(
        parts
            .nodes
            .iter()
            .map(|node| node.node_id.as_str())
            .collect::<Vec<_>>(),
        ["a_asof", "b_unrelated", "c_select1", "d_select2"]
    );
    let observed = entry_observation(true, true).await;
    assert!(observed.clean_join && observed.registry_empty_after_join);
    assert_eq!(observed.source_count, 3);
    assert_eq!(observed.tasks.len(), 4);
    assert_eq!(observed.tasks[&TaskId::new(0)].task_name, "operator:a_asof");
    assert_eq!(
        observed.tasks[&TaskId::new(1)].task_name,
        "operator:b_unrelated"
    );
    assert_eq!(
        observed.tasks[&TaskId::new(2)].task_name,
        "operator:c_select1"
    );
    assert_eq!(
        observed.tasks[&TaskId::new(3)].task_name,
        "operator:d_select2"
    );
    assert_eq!(observed.drivers, 2);
}

fn input_batch(left: bool) -> Batch {
    let prices = if left {
        vec![Some(10.0), Some(20.0), None, Some(40.0)]
    } else {
        vec![Some(1.0), None, Some(3.0)]
    };
    let len = prices.len();
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(
                    (0..len).map(|value| i64::try_from(value).unwrap()),
                )
                .with_timezone("UTC"),
            ),
            Arc::new(StringArray::from(vec!["S"; len])),
            Arc::new(UInt64Array::from_iter_values(
                (0..len).map(|value| u64::try_from(value).unwrap()),
            )),
            Arc::new(Float64Array::from(prices)),
        ],
    )
    .unwrap();
    Batch::table(
        vec![record],
        BatchMetadata::new(
            if left { "left-source" } else { "right-source" },
            0,
            [("immutable".into(), serde_json::json!(true))]
                .into_iter()
                .collect(),
        )
        .unwrap(),
    )
    .unwrap()
}

struct TrafficObservation {
    internal: Vec<String>,
    boundary: Vec<String>,
    physical: BTreeMap<String, Counts>,
    metrics: M2MetricsSnapshot,
    output: Vec<Batch>,
    clean_join: bool,
    registry_empty_after_join: bool,
}

fn traffic_edges(parts: &crate::pipeline::StreamRuntimePlanParts) -> (Vec<String>, Vec<String>) {
    let boundary = parts
        .source_routes
        .values()
        .map(|route| route.edge_id.clone())
        .chain(
            parts
                .sink_routes
                .values()
                .map(|route| route.edge_id.clone()),
        )
        .collect::<std::collections::BTreeSet<_>>();
    let internal = parts
        .edges
        .keys()
        .filter(|edge| !boundary.contains(*edge))
        .cloned()
        .collect::<Vec<_>>();
    let boundary = boundary.into_iter().collect::<Vec<_>>();
    (internal, boundary)
}

fn traffic_metrics(parts: &crate::pipeline::StreamRuntimePlanParts) -> MetricsRecorder {
    MetricsRecorder::new(
        parts
            .edges
            .iter()
            .map(|(id, edge)| (id.clone(), edge.budget)),
        parts.source_routes.keys().cloned(),
        parts.nodes.iter().map(|node| node.node_id.clone()),
        parts.sink_routes.keys().cloned(),
    )
}

fn traffic_routes(
    parts: &crate::pipeline::StreamRuntimePlanParts,
) -> Vec<(String, String, String)> {
    parts
        .source_routes
        .iter()
        .map(|(id, route)| {
            (
                id.clone(),
                route.target.node_id.clone(),
                route.target.port.clone(),
            )
        })
        .collect::<Vec<_>>()
}

async fn traffic_observation(two_selects: bool, interposed: bool) -> TrafficObservation {
    let observer = Observer::start();
    let plan = plan(two_selects, interposed);
    let parts = plan
        .into_runtime_parts(EdgeBudget::new(64, 1 << 20).unwrap())
        .unwrap();
    let (internal, boundary) = traffic_edges(&parts);
    let routes = traffic_routes(&parts);
    let output_id = parts
        .sink_routes
        .iter()
        .find(|(_, route)| route.source.node_id != "b_unrelated")
        .map(|(id, _)| id.clone())
        .unwrap();
    let unrelated_id = parts
        .sink_routes
        .iter()
        .find(|(_, route)| route.source.node_id == "b_unrelated")
        .map(|(id, _)| id.clone());
    let cancellation = CancellationToken::new();
    let context = StreamJobContext::new(
        14,
        &parts.fingerprint,
        JsonMap::new(),
        None,
        cancellation.clone(),
    );
    let (commands, _commands_rx) = mpsc::unbounded_channel();
    let metrics = traffic_metrics(&parts);
    let core = Arc::new(JobCore::new(
        LaunchId::new(0),
        14,
        commands,
        metrics.clone(),
        StatusProjection::default(),
        false,
        parts.name.clone(),
    ));
    let Ok(mut runtime) =
        run_operator_entry(parts, &context, &core, &cancellation, BTreeMap::new(), None).await
    else {
        panic!("valid ASOF traffic graph entry failed");
    };
    runtime.data_gate.send(true).unwrap();
    let traffic = tokio::time::timeout(Duration::from_secs(5), async {
        for (binding, node, port) in &routes {
            let sender = &mut runtime.source_outputs.get_mut(binding).unwrap()[0];
            sender
                .send(StreamMessage::data(input_batch(
                    node == "a_asof" && port == "left",
                )))
                .await?;
            sender.send(StreamMessage::end_of_input()).await?;
        }
        let mut output = Vec::new();
        let receiver = runtime.sink_inputs.get_mut(&output_id).unwrap();
        loop {
            let message = receiver.recv().await?.expect("output closed before EOF");
            if let Some(batch) = message.as_data() {
                output.push(batch.clone());
            }
            if message.kind() == StreamMessageKind::EndOfInput {
                break;
            }
        }
        if let Some(id) = &unrelated_id {
            let receiver = runtime.sink_inputs.get_mut(id).unwrap();
            loop {
                let message = receiver
                    .recv()
                    .await?
                    .expect("unrelated output closed before EOF");
                if message.kind() == StreamMessageKind::EndOfInput {
                    break;
                }
            }
        }
        Ok::<_, crate::CalcFlowError>(output)
    })
    .await;
    cancellation.cancel();
    let report = runtime.supervisor.join_all().await;
    let registry_empty_after_join = runtime.supervisor.registry().snapshot().is_empty();
    let physical = observer.snapshot();
    TrafficObservation {
        internal,
        boundary,
        physical,
        metrics: metrics.snapshot(),
        output: traffic
            .expect("actual graph traffic timed out")
            .expect("actual graph traffic failed"),
        clean_join: report.errors.is_empty(),
        registry_empty_after_join,
    }
}

fn assert_data_and_logical_traffic(observed: &TrafficObservation, two_selects: bool) {
    assert!(observed.clean_join && observed.registry_empty_after_join);
    assert_eq!(observed.output.len(), 1);
    assert_eq!(observed.output[0].metadata().source(), "a_asof");
    assert!(observed.output[0].metadata().attributes().is_empty());
    assert_eq!(observed.output[0].metadata().sequence(), 0);
    let fields = if two_selects {
        vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("final_price", DataType::Float64, true)
                .with_metadata([("units".into(), "USD".into())].into_iter().collect()),
        ]
    } else {
        vec![
            Field::new("quote", DataType::Float64, true)
                .with_metadata([("units".into(), "USD".into())].into_iter().collect()),
            Field::new("id", DataType::UInt64, false),
        ]
    };
    let ids: datafusion::arrow::array::ArrayRef = Arc::new(UInt64Array::from(vec![0, 1, 2, 3]));
    let prices: datafusion::arrow::array::ArrayRef =
        Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0), None]));
    let columns = if two_selects {
        vec![ids, prices]
    } else {
        vec![prices, ids]
    };
    let expected = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    assert_eq!(
        observed.output[0].table_payload().unwrap().batches(),
        &[expected]
    );
    for edge in &observed.internal {
        let metrics = &observed.metrics.edges[edge];
        assert_eq!((metrics.input_batches, metrics.output_batches), (1, 1));
        assert_eq!((metrics.input_rows, metrics.output_rows), (4, 4));
        assert_eq!(metrics.input_bytes, metrics.output_bytes);
        assert!(!metrics.drop_invariant_violated);
    }
    for edge in &observed.boundary {
        assert_eq!(observed.physical[edge].constructions, 1);
        assert!(observed.physical[edge].enqueues > 0 && observed.physical[edge].dequeues > 0);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn a13_asof_exact_select_removes_actual_internal_physical_transport() {
    let observed = traffic_observation(false, false).await;
    assert_data_and_logical_traffic(&observed, false);
    assert_eq!(observed.internal.len(), 1);
    assert_eq!(
        observed
            .physical
            .get(&observed.internal[0])
            .cloned()
            .unwrap_or_default(),
        Counts::default()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a13_asof_two_selects_remove_both_actual_internal_physical_transports() {
    let observed = traffic_observation(true, false).await;
    assert_data_and_logical_traffic(&observed, true);
    assert_eq!(observed.internal.len(), 2);
    for edge in &observed.internal {
        assert_eq!(
            observed.physical.get(edge).cloned().unwrap_or_default(),
            Counts::default()
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn a13_nonadjacent_asof_chain_removes_only_its_actual_internal_physical_transports() {
    let observed = traffic_observation(true, true).await;
    assert_data_and_logical_traffic(&observed, true);
    assert_eq!(observed.internal.len(), 2);
    assert_eq!(observed.boundary.len(), 5);
    for edge in &observed.internal {
        assert_eq!(
            observed.physical.get(edge).cloned().unwrap_or_default(),
            Counts::default()
        );
    }
}
