use std::{sync::Arc, time::Duration};

use crate::{
    BatchKind, Edge, ExpressionOperator, JoinStateLimits, JoinTimeBounds, NodeOperator,
    OperatorMetadata, PipelineBuilder, Port, PortEndpoint, StreamJoinOperator, StreamJoinSpec,
};

mod execution;
pub(super) use execution::execution_and_restore;

fn hot_graph(select: &[&str], band: bool) -> PipelineBuilder {
    hot_graph_with_filter(select, band, None)
}

fn hot_graph_with_filter(select: &[&str], band: bool, filter: Option<&str>) -> PipelineBuilder {
    use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    let quote = || {
        Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ]))
    };
    let left = quote();
    let right = if band {
        quote()
    } else {
        Arc::new(Schema::new(vec![
            Field::new("symbol", DataType::Utf8, false),
            Field::new("factor", DataType::Float64, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
        ]))
    };
    let (left_prefix, right_prefix) = if band {
        ("left", "right")
    } else {
        ("quote", "dimension")
    };
    let spec = StreamJoinSpec::inner(
        ["symbol"],
        ["symbol"],
        "event_time",
        "event_time",
        JoinTimeBounds::new(
            Duration::from_secs(if band { 5 } else { 300 }),
            Duration::from_secs(if band { 5 } else { 0 }),
        )
        .unwrap(),
        JoinStateLimits::new(100, 1_000_000, 1_000).unwrap(),
    )
    .unwrap()
    .with_prefixes(left_prefix, right_prefix)
    .unwrap();
    let join = StreamJoinOperator::new("asof", left, right, spec).unwrap();
    let output = Arc::clone(join.output_ports()[0].schema().unwrap());
    let expression = ExpressionOperator::new(
        "result",
        "",
        select.iter().map(|value| (*value).into()).collect(),
        filter.map(str::to_owned),
        vec![],
    )
    .unwrap()
    .with_ports(
        Port::with_schema_ref("input", BatchKind::Table, true, Some(output)).unwrap(),
        Port::new("output", BatchKind::Table, true, None).unwrap(),
    )
    .unwrap();
    PipelineBuilder::new("hot projection")
        .unwrap()
        .add_node("asof", join)
        .unwrap()
        .add_node("result", Box::new(expression))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("asof", "output").unwrap(),
            PortEndpoint::new("result", "input").unwrap(),
        ))
        .unwrap()
}

#[test]
fn hot_join_fuses_native_final_projection() {
    assert_eq!(
        super::physical_fields(hot_graph(
            &[
                "\"quote__sequence\" AS \"sequence\"",
                "(\"quote__price\" * \"dimension__factor\") AS \"value\"",
            ],
            false,
        )),
        ["sequence", "value"]
    );
    assert_eq!(
        super::physical_fields(hot_graph(
            &[
                "\"left__sequence\" AS \"sequence\"",
                "\"right__sequence\" AS \"right_sequence\"",
                "(\"left__price\" * \"right__price\") AS \"value\"",
            ],
            true,
        )),
        ["sequence", "right_sequence", "value"]
    );
}

#[test]
fn fused_join_keeps_the_full_output_for_named_union_inputs() {
    let graph = hot_graph(
        &[
            "quote__sequence AS sequence",
            "quote__price * dimension__factor AS value",
        ],
        false,
    );
    let output = graph.nodes["asof"].operator.output_ports()[0]
        .schema()
        .unwrap()
        .clone();
    let inputs = ["joined", "other"]
        .map(|name| {
            Port::with_schema_ref(name, BatchKind::Table, true, Some(Arc::clone(&output))).unwrap()
        })
        .to_vec();
    let graph = graph
        .add_node(
            "audit",
            Box::new(crate::UnionOperator::new("audit", inputs).unwrap()),
        )
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("asof", "output").unwrap(),
            PortEndpoint::new("audit", "joined").unwrap(),
        ))
        .unwrap();
    let expected = output
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<Vec<_>>();
    assert_eq!(super::physical_fields(graph), expected);
}

#[test]
fn fused_join_preserves_the_declared_consumer_schema() {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    let mut graph = hot_graph(
        &[
            "quote__sequence AS sequence",
            "quote__price * dimension__factor AS value",
        ],
        false,
    );
    let declared = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("sequence", DataType::UInt64, false),
            Field::new("value", DataType::Float64, false),
        ],
        [("unit".into(), "USD".into())].into(),
    ));
    let NodeOperator::Expression(expression) = &mut graph.nodes.get_mut("result").unwrap().operator
    else {
        unreachable!()
    };
    *expression = expression
        .clone()
        .with_ports(
            expression.input_ports()[0].clone(),
            Port::with_schema_ref(
                "output",
                BatchKind::Table,
                true,
                Some(Arc::clone(&declared)),
            )
            .unwrap(),
        )
        .unwrap();
    super::super::push_output_projections(&mut graph, None).unwrap();
    assert_eq!(
        graph.nodes["result"].operator.output_ports()[0].schema(),
        Some(&declared)
    );
}

fn graph(selects: &[&[&str]]) -> PipelineBuilder {
    let mut builder = super::graph(selects);
    let NodeOperator::StreamAsofJoin(asof) = &builder.nodes["asof"].operator else {
        unreachable!()
    };
    let schema = Arc::clone(asof.input_ports()[0].schema().unwrap());
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 1_000).unwrap(),
    )
    .unwrap();
    let join = StreamJoinOperator::new("asof", Arc::clone(&schema), schema, spec).unwrap();
    let output = Arc::clone(join.output_ports()[0].schema().unwrap());
    for definition in builder.nodes.values_mut() {
        if let NodeOperator::Expression(expression) = &mut definition.operator {
            expression
                .set_projected_stream_input(Arc::clone(&output))
                .unwrap();
        }
    }
    builder.nodes.get_mut("asof").unwrap().operator = NodeOperator::StreamJoin(Box::new(join));
    builder
}

pub(super) fn exact_consumers_prune_join_output() {
    assert_eq!(
        super::physical_fields(graph(&[&["right__value AS price"]])),
        vec!["right__value"]
    );
    assert!(super::physical_fields(graph(&[&["1 AS constant"]])).is_empty());
    assert_eq!(
        super::physical_fields(graph(&[
            &["right__value AS price", "left__seq"],
            &["left__key", "left__seq AS sequence"],
        ])),
        vec!["left__key", "left__seq", "right__value"],
        "Join must materialize the union of its consumers' required columns"
    );
    graph_selected_udf_keeps_full_output();
    assert_eq!(super::physical_fields(graph(&[])).len(), 8);
    for select in [
        "*",
        "sum(right__value) AS total",
        "missing",
        "foreign.left__seq",
    ] {
        assert_eq!(
            super::physical_fields(graph(&[&["left__seq"], &[select]])).len(),
            8,
            "one full-input consumer must prevent Join pruning: {select}"
        );
    }
}

pub(super) fn arithmetic_consumers_prune_join_output() {
    assert_eq!(
        super::physical_fields(graph(&[&["left__seq"], &["right__value + 1 AS price"],])),
        vec!["left__seq", "right__value"],
        "Join dependencies must include the input to arithmetic expressions"
    );
    batch_sensitive_functions_keep_full_output();
}

fn graph_selected_udf_keeps_full_output() {
    use crate::{ExpressionOperator, UdfKind, UdfReference, UdfRegistry};
    use datafusion::{
        arrow::datatypes::DataType,
        logical_expr::{Volatility, create_udf},
    };

    let reference = UdfReference::new("rust", "abs", "1", UdfKind::DataFusionScalar).unwrap();
    let selected = ExpressionOperator::new(
        "selected_elsewhere",
        "",
        vec!["1".into()],
        None,
        vec![reference.clone()],
    )
    .unwrap();
    assert_eq!(
        super::physical_fields(graph(&[&["abs(left__seq) AS value"]])),
        vec!["left__seq"]
    );
    let builder = graph(&[&["abs(left__seq) AS value"]])
        .add_node("selected_elsewhere", Box::new(selected))
        .unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(
            reference,
            Arc::new(create_udf(
                "abs",
                vec![DataType::Int64],
                DataType::Int64,
                Volatility::Immutable,
                Arc::new(|arguments| Ok(arguments[0].clone())),
            )),
            1,
        )
        .unwrap();
    let plan = builder
        .compile_stream(&registry.snapshot(), &crate::StreamRequirements::default())
        .unwrap();
    let join = plan
        .nodes
        .iter()
        .find(|node| node.node_id == "asof")
        .unwrap();
    assert_eq!(
        join.output_ports["output"].schema().unwrap().fields().len(),
        8,
        "a native UDF selected by another node is visible in this consumer's runtime"
    );
}

fn batch_sensitive_functions_keep_full_output() {
    for select in [
        "now() AS value",
        "random() AS value",
        "current_time() AS value",
        "unknown_scalar(left__seq) AS value",
        "sum(left__seq) AS value",
        "row_number() OVER () AS value",
        "*",
        "input.*",
        "(SELECT left__seq) AS value",
    ] {
        let builder = graph(&[&[select]]);
        let NodeOperator::Expression(expression) = &builder.nodes["select0"].operator else {
            unreachable!()
        };
        assert!(
            expression.stream_required_columns().is_none(),
            "batch-sensitive or unknown expression accepted: {select}"
        );
    }
}
