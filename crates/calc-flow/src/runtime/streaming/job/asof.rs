//! ASOF's additive source-progress requirement for direct runner bindings.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    CalcFlowError, Result,
    pipeline::{CompiledStreamOperator, RuntimeProducer, StreamRuntimePlanParts},
    runtime::streaming::progress::{PreparedStreamJob, prepare::NormalizedWatermarkMode},
};

pub(super) fn validate_progress(
    plan: &StreamRuntimePlanParts,
    progress: &PreparedStreamJob,
) -> Result<()> {
    let required = reachable_sources(plan);
    for source in progress.bindings.iter() {
        if required.contains(source.identity.as_str())
            && matches!(
                source.normalized_watermark,
                NormalizedWatermarkMode::Disabled { .. }
            )
        {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sources.{}.watermark_policy", source.identity.as_str()),
                message: "a source reaching stream_asof_join must provide watermark progress"
                    .into(),
            });
        }
    }
    Ok(())
}

fn reachable_sources(plan: &StreamRuntimePlanParts) -> BTreeSet<&str> {
    let nodes: BTreeMap<_, _> = plan
        .nodes
        .iter()
        .map(|node| (node.node_id.as_str(), node))
        .collect();
    let mut pending: Vec<_> = plan
        .nodes
        .iter()
        .filter(|node| matches!(node.operator, CompiledStreamOperator::StreamAsofJoin(_)))
        .flat_map(|node| node.ingress_edges.values().map(String::as_str))
        .collect();
    let mut visited = BTreeSet::new();
    let mut sources = BTreeSet::new();
    while let Some(edge_id) = pending.pop() {
        if !visited.insert(edge_id) {
            continue;
        }
        match &plan.edges[edge_id].producer {
            RuntimeProducer::Source { binding_id } => {
                sources.insert(binding_id.as_str());
            }
            RuntimeProducer::Node { node_id, .. } => {
                pending.extend(
                    nodes[node_id.as_str()]
                        .ingress_edges
                        .values()
                        .map(String::as_str),
                );
            }
        }
    }
    sources
}
