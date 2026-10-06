use std::{collections::BTreeMap, sync::Arc, time::Duration};

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use tokio::sync::mpsc;

use crate::{
    AsofJoinSide, AsofStateLimits, BatchKind, CancellationToken, Edge, EdgeBudget,
    ExpressionOperator, JsonMap, OperatorMetadata, PipelineBuilder, Port, PortEndpoint,
    StreamAsofJoinOperator, StreamAsofJoinSpec, StreamExecutionPlan, StreamJobContext,
    StreamRequirements, UdfRegistry,
    runtime::streaming::{
        metrics::MetricsRecorder,
        projection::StatusProjection,
        runner::{JobCore, LaunchId, run_operator_entry},
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
fn a16_asof_exact_select_is_eligible() {
    let parts = plan(false, false)
        .into_runtime_parts(EdgeBudget::default())
        .unwrap();
    assert_eq!(parts.nodes[0].ingress_edges.len(), 2);
    assert!(super::eligible_pair(&parts.nodes[0], &parts.nodes[1]));
}

async fn physical_driver_count(two_selects: bool) -> usize {
    let plan = plan(two_selects, false);
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
        panic!("valid ASOF chain entry failed");
    };
    let drivers = runtime.supervisor.physical_driver_count();
    let expected_nodes = if two_selects { 3 } else { 2 };
    assert_eq!(runtime.supervisor.task_count(), expected_nodes);
    cancellation.cancel();
    assert!(runtime.supervisor.join_all().await.errors.is_empty());
    assert_eq!(runtime.supervisor.task_count(), 0);
    drivers
}

#[tokio::test]
async fn a16_asof_select_chain_has_one_physical_driver() {
    assert_eq!(physical_driver_count(false).await, 1);
    assert_eq!(physical_driver_count(true).await, 1);
}
