//! Source obligations are mandatory for every runner, independently of requests.
use crate::runtime::streaming::progress::{PreparedStreamJob, prepare::NormalizedWatermarkMode};
use crate::{
    OperatorMetadata, Result, SourceBoundedness,
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
        let (source, time) = source_time(
            plan,
            node.ingress_edges.values().next().unwrap(),
            event_time_column,
        )?;
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
                        "sort.spec.history",
                        "capability_mismatch",
                        "window time lineage is not row-local",
                    ));
                }
            };
            let original = crate::sql::window_time_column(query, alias, time)?;
            source_time(plan, node.ingress_edges.values().next().unwrap(), &original)
        }
    }
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
