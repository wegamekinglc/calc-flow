//! Source obligations are mandatory for every runner, independently of requests.
use super::window_finality;
use crate::runtime::streaming::progress::PreparedStreamJob;
use crate::{
    Result, SourceBoundedness,
    dataframe::invalid,
    pipeline::{CompiledStreamOperator, RuntimeProducer, StreamRuntimePlanParts},
    runtime::streaming::source_task::SourceBinding,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate_progress(
    plan: &StreamRuntimePlanParts,
    bindings: &BTreeMap<String, SourceBinding>,
    progress: &PreparedStreamJob,
) -> Result<()> {
    for node in &plan.nodes {
        let CompiledStreamOperator::BoundedOrder(operator) = &node.operator else {
            continue;
        };
        let crate::SortHistory::ClosedWindow {
            event_time_column, ..
        } = operator.spec().history()
        else {
            continue;
        };
        window_finality::validate_progress(
            plan,
            bindings,
            progress,
            node.ingress_edges.values().next().unwrap(),
            event_time_column,
            "sort.spec.history",
        )?;
    }
    Ok(())
}

fn sources<'a>(
    plan: &'a StreamRuntimePlanParts,
    node: &crate::pipeline::RuntimeStreamNode,
) -> BTreeSet<&'a str> {
    let nodes: BTreeMap<_, _> = plan.nodes.iter().map(|n| (n.node_id.as_str(), n)).collect();
    let mut pending = node
        .ingress_edges
        .values()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let mut seen = BTreeSet::new();
    let mut sources = BTreeSet::new();
    while let Some(edge) = pending.pop() {
        if !seen.insert(edge) {
            continue;
        }
        match &plan.edges[edge].producer {
            RuntimeProducer::Source { binding_id } => {
                sources.insert(binding_id.as_str());
            }
            RuntimeProducer::Node { node_id, .. } => pending.extend(
                nodes[node_id.as_str()]
                    .ingress_edges
                    .values()
                    .map(String::as_str),
            ),
        }
    }
    sources
}

pub(super) fn fingerprint(
    plan: &StreamRuntimePlanParts,
    bindings: &BTreeMap<String, SourceBinding>,
    base: &str,
) -> Result<String> {
    let mut proofs = BTreeMap::new();
    for node in &plan.nodes {
        let CompiledStreamOperator::BoundedOrder(operator) = &node.operator else {
            continue;
        };
        let inputs = sources(plan, node);
        if inputs.len() != 1 {
            return Err(invalid(
                "sort.input",
                "capability_mismatch",
                "one source path required",
            ));
        }
        for source in inputs {
            let boundedness = bindings[source].sampled_boundedness();
            validate_boundedness(operator.spec().history(), source, boundedness)?;
            proofs.insert(source, boundedness);
        }
    }
    if proofs.is_empty() {
        return Ok(base.to_owned());
    }
    let encoded = crate::canonical_json(&serde_json::json!([base, "bounded_order_v1", proofs]))?;
    Ok(hex::encode(Sha256::digest(encoded.as_bytes())))
}
fn validate_boundedness(
    history: &crate::SortHistory,
    source: &str,
    boundedness: SourceBoundedness,
) -> Result<()> {
    if matches!(history, crate::SortHistory::BoundedSource)
        && boundedness != SourceBoundedness::Bounded
    {
        let code = if boundedness == SourceBoundedness::Unknown {
            "source_boundedness_unproved"
        } else {
            "source_unbounded"
        };
        return Err(invalid(
            &format!("stream.inputs.{source}.boundedness"),
            code,
            "final ordering requires a fixed finite cut; a possible future EOF is insufficient",
        ));
    }
    Ok(())
}
