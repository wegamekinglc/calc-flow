use std::{sync::Arc, time::Duration};

use crate::{
    JoinStateLimits, JoinTimeBounds, NodeOperator, OperatorMetadata, PipelineBuilder,
    StreamJoinOperator, StreamJoinSpec,
};

mod execution;
pub(super) use execution::execution_and_restore;

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
