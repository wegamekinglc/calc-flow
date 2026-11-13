//! Structural admission is mandatory even when portability/latest are not requested.
use crate::{
    NodeOperator, OperatorMetadata, PipelineBuilder, PortEndpoint, Result, SortHistory,
    dataframe::invalid,
};
use std::collections::BTreeSet;

fn query(operator: &NodeOperator) -> Option<(&str, &str)> {
    if !operator.udf_references().is_empty() || operator.input_ports().len() != 1 {
        return None;
    }
    match operator {
        NodeOperator::Expression(op) => Some((op.query_text(), op.input_ports()[0].name())),
        NodeOperator::Sql(op) => Some((op.query_text(), op.input_ports()[0].name())),
        _ => None,
    }
}
fn row_local(operator: &NodeOperator) -> Result<bool> {
    match query(operator) {
        Some((query, alias)) => Ok(crate::sql::sql_passthrough_columns(query, alias)?.is_some()),
        None => Ok(false),
    }
}
fn unsupported(builder: &PipelineBuilder, id: &str) -> crate::CalcFlowError {
    let operator = &builder.nodes[id].operator;
    let snapshot = matches!(operator, NodeOperator::SnapshotSql(_))
        || matches!(operator,NodeOperator::Sql(sql) if sql.has_stream_aggregate());
    invalid(
        &format!("graph.nodes.{id}.sort.input"),
        if snapshot && !builder.update_semantics().is_legacy() {
            "update_interpretation_mismatch"
        } else {
            "capability_mismatch"
        },
        "ordering paths require one source and deterministic row-local input/output; snapshots, stateful, multi-input and nested finalizers are unsupported",
    )
}
fn upstream(
    builder: &PipelineBuilder,
    endpoint: &PortEndpoint,
    fields: &[String],
    origins: &mut BTreeSet<PortEndpoint>,
) -> Result<()> {
    let incoming = builder
        .edges
        .iter()
        .filter(|edge| edge.target == *endpoint)
        .collect::<Vec<_>>();
    if incoming.is_empty() {
        origins.insert(endpoint.clone());
        return Ok(());
    }
    for edge in incoming {
        let source = &builder.nodes[&edge.source.node_id].operator;
        if !row_local(source)? {
            return Err(unsupported(builder, &edge.source.node_id));
        }
        let (sql, alias) = query(source).unwrap();
        let original = fields
            .iter()
            .map(|name| {
                crate::sql::window_time_column(sql, alias, name).map_err(|_| {
                    invalid(
                        "sort.spec",
                        "capability_mismatch",
                        "time/sequence must be unchanged input fields or pure renames",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        upstream(
            builder,
            &PortEndpoint {
                node_id: edge.source.node_id.clone(),
                port: alias.into(),
            },
            &original,
            origins,
        )?;
    }
    Ok(())
}
fn downstream(builder: &PipelineBuilder, id: &str, seen: &mut BTreeSet<String>) -> Result<()> {
    if !seen.insert(id.into()) {
        return Ok(());
    }
    for edge in builder
        .edges
        .iter()
        .filter(|edge| edge.source.node_id == id)
    {
        if !row_local(&builder.nodes[&edge.target.node_id].operator)? {
            return Err(unsupported(builder, &edge.target.node_id));
        }
        downstream(builder, &edge.target.node_id, seen)?;
    }
    Ok(())
}
pub(crate) fn validate(builder: &PipelineBuilder) -> Result<()> {
    for (id, node) in &builder.nodes {
        validate_native_identity(&node.operator, id)?;
        let NodeOperator::BoundedOrder(op) = &node.operator else {
            continue;
        };
        validate_order(builder, id, op)?;
    }
    Ok(())
}
fn validate_native_identity(operator: &NodeOperator, id: &str) -> Result<()> {
    if matches!(operator, NodeOperator::Batch(_) | NodeOperator::Stream(_))
        && operator.metadata().type_id() == std::any::TypeId::of::<crate::BoundedOrderOperator>()
    {
        return Err(invalid(
            &format!("graph.nodes.{id}.sort"),
            "capability_mismatch",
            "bounded ordering must retain its native graph identity for mandatory source and structural admission",
        ));
    }
    Ok(())
}
fn validate_order(
    builder: &PipelineBuilder,
    id: &str,
    op: &crate::BoundedOrderOperator,
) -> Result<()> {
    let mut fields = match op.spec().history() {
        SortHistory::BoundedSource => vec![],
        SortHistory::ClosedWindow {
            event_time_column, ..
        } => vec![event_time_column.clone()],
    };
    if let crate::SortTies::Sequence { columns } = op.spec().ties() {
        fields.extend_from_slice(columns);
    }
    let mut origins = BTreeSet::new();
    upstream(
        builder,
        &PortEndpoint {
            node_id: id.into(),
            port: "input".into(),
        },
        &fields,
        &mut origins,
    )?;
    if origins.len() != 1 {
        return Err(unsupported(builder, id));
    }
    downstream(builder, id, &mut BTreeSet::new())
}

// These nodes consume one final ordered relation; a single physical partition
// prevents row-local postprocessing from interleaving output partitions.
pub(crate) fn serial_config(config: crate::DataFusionConfig) -> crate::DataFusionConfig {
    crate::DataFusionConfig {
        parallelism_mode: crate::DataFusionParallelismMode::Fixed,
        target_partitions: 1,
        max_partitions: 1,
        ..config
    }
}
pub(super) fn prepare(builder: &mut PipelineBuilder) {
    let mut pending = builder
        .nodes
        .iter()
        .filter(|(_, node)| matches!(node.operator, NodeOperator::BoundedOrder(_)))
        .map(|(id, _)| id.clone())
        .collect::<Vec<_>>();
    let mut seen = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        let children = builder
            .edges
            .iter()
            .filter(|edge| edge.source.node_id == id)
            .map(|edge| edge.target.node_id.clone())
            .collect::<Vec<_>>();
        for child in children {
            match &mut builder.nodes.get_mut(&child).unwrap().operator {
                NodeOperator::Expression(op) => op.preserve_input_order(),
                NodeOperator::Sql(op) => op.preserve_input_order(),
                _ => {}
            }
            pending.push(child);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_erased_builtin_cannot_bypass_order_admission() {
        use crate::{
            BoundedOrderOperator, NullPlacement, SortDirection, SortKey, SortSpec, StateBudget,
        };
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        let schema = std::sync::Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("seq", DataType::Int64, false),
        ]));
        let op = BoundedOrderOperator::new(
            "order",
            schema,
            vec![SortKey::new("x", SortDirection::Ascending, NullPlacement::Last).unwrap()],
            Some(1),
            SortSpec::bounded_source(StateBudget::new(10, 8 << 20).unwrap())
                .unwrap()
                .with_sequence_by(vec!["seq".into()])
                .unwrap(),
        )
        .unwrap();
        let builder = PipelineBuilder::new("erased")
            .unwrap()
            .add_node("order", NodeOperator::Stream(Box::new(op)))
            .unwrap();
        assert!(
            validate(&builder).is_err(),
            "erasing builtin identity must not skip mandatory source/graph admission"
        );
    }
}
