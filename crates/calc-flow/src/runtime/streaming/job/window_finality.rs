//! Source-side finality proof for closed-window ORDER and DISTINCT.
use crate::{
    OperatorMetadata, Result, SourceBoundedness,
    dataframe::invalid,
    pipeline::{CompiledStreamOperator, RuntimeProducer, StreamRuntimePlanParts},
    runtime::streaming::{
        progress::{PreparedStreamJob, prepare::NormalizedWatermarkMode},
        source_task::SourceBinding,
    },
};
use std::collections::BTreeMap;

pub(super) fn validate_progress(
    plan: &StreamRuntimePlanParts,
    bindings: &BTreeMap<String, SourceBinding>,
    progress: &PreparedStreamJob,
    edge: &str,
    event_time_column: &str,
    history_field: &str,
) -> Result<()> {
    let (source, time) = source_time(plan, edge, event_time_column, history_field)?;
    let prepared = progress
        .bindings
        .iter()
        .find(|p| p.identity.as_str() == source)
        .unwrap();
    let valid = progress_matches(&prepared.normalized_watermark, &time);
    if !valid && bindings[source].sampled_boundedness() != SourceBoundedness::Bounded {
        return Err(invalid(
            &format!("stream.inputs.{source}.watermark_policy"),
            "finality_unproven",
            "selected window requires a trustworthy watermark on its original time coordinate or a bounded EOF cut",
        ));
    }
    Ok(())
}

fn progress_matches(mode: &NormalizedWatermarkMode, time: &str) -> bool {
    match mode {
        NormalizedWatermarkMode::SourceProvided { .. } => true,
        NormalizedWatermarkMode::Generated { event_time, .. } => event_time.name.as_ref() == time,
        NormalizedWatermarkMode::Disabled { .. } => false,
    }
}

fn source_time<'a>(
    plan: &'a StreamRuntimePlanParts,
    edge: &str,
    time: &str,
    history_field: &str,
) -> Result<(&'a str, String)> {
    match &plan.edges[edge].producer {
        RuntimeProducer::Source { binding_id } => Ok((binding_id, time.into())),
        RuntimeProducer::Node { node_id, .. } => {
            let node = plan.nodes.iter().find(|n| n.node_id == *node_id).unwrap();
            let (query, alias) = match &node.operator {
                CompiledStreamOperator::Expression(op) => (op.query_text(), "input"),
                CompiledStreamOperator::Sql(op) => (op.query_text(), op.input_ports()[0].name()),
                _ => {
                    return Err(invalid(
                        history_field,
                        "capability_mismatch",
                        "window time lineage is not row-local",
                    ));
                }
            };
            let original = crate::sql::window_time_column(query, alias, time)?;
            source_time(
                plan,
                node.ingress_edges.values().next().unwrap(),
                &original,
                history_field,
            )
        }
    }
}
