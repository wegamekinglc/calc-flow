//! Structural admission is mandatory even when portability/latest are not requested.
use crate::{
    DistinctHistory, NodeOperator, OperatorMetadata, PipelineBuilder, PortEndpoint, Result,
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
        &format!("graph.nodes.{id}.distinct.input"),
        if snapshot && !builder.update_semantics().is_legacy() {
            "update_interpretation_mismatch"
        } else {
            "capability_mismatch"
        },
        "distinct paths require one source and deterministic row-local input/output; snapshots, stateful, multi-input and nested DISTINCT are unsupported",
    )
}
fn upstream(
    builder: &PipelineBuilder,
    endpoint: &PortEndpoint,
    time: Option<&str>,
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
        let columns = crate::sql::sql_passthrough_columns(sql, alias)?.unwrap();
        let source_time = match time {
            None => None,
            Some(name) => match columns.get(name) {
                Some(Some(source)) => Some(source.as_str()),
                None if columns.contains_key("*") => Some(name),
                _ => {
                    return Err(invalid(
                        &format!("graph.nodes.{}.distinct.spec.history", edge.source.node_id),
                        "capability_mismatch",
                        "window time must be an unchanged input timestamp or pure rename",
                    ));
                }
            },
        };
        upstream(
            builder,
            &PortEndpoint {
                node_id: edge.source.node_id.clone(),
                port: alias.into(),
            },
            source_time,
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
        let NodeOperator::Distinct(op) = &node.operator else {
            continue;
        };
        let time = match op.spec().history() {
            DistinctHistory::BoundedSource => None,
            DistinctHistory::ClosedWindow {
                event_time_column, ..
            } => Some(event_time_column.as_str()),
        };
        let mut origins = BTreeSet::new();
        upstream(
            builder,
            &PortEndpoint {
                node_id: id.clone(),
                port: "input".into(),
            },
            time,
            &mut origins,
        )?;
        if origins.len() != 1 {
            return Err(unsupported(builder, id));
        }
        downstream(builder, id, &mut BTreeSet::new())?;
    }
    Ok(())
}
