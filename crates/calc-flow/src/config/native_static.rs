use std::collections::{BTreeMap, BTreeSet};

use crate::{BatchKind, PipelineBuilder, PortEndpoint, StaticInputSpec};

use super::{ProjectSpec, ValidationIssue, issue, validate_static_input_kind};

pub(super) fn is_native(spec: &StaticInputSpec) -> bool {
    matches!(spec, StaticInputSpec::Array { backend, .. } if backend == "native")
}

pub(super) fn legacy_kind(
    spec: &StaticInputSpec,
    inputs: &BTreeMap<&str, BatchKind>,
) -> Option<BatchKind> {
    if is_native(spec) {
        None
    } else {
        inputs.get(spec.name()).copied()
    }
}

pub(super) fn binding_issues(
    project: &ProjectSpec,
    builder: &PipelineBuilder,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    if !project.static_inputs.iter().any(is_native) {
        return issues;
    }
    let inputs = external_inputs(builder);
    for (index, spec) in project
        .static_inputs
        .iter()
        .enumerate()
        .filter(|(_, spec)| is_native(spec))
    {
        let base = format!("static_inputs[{index}]");
        let Some(kind) = inputs.get(spec.name()) else {
            issues.push(issue(
                format!("{base}.name"),
                "unknown_binding",
                format!(
                    "static input {:?} does not name a graph external input binding",
                    spec.name()
                ),
            ));
            continue;
        };
        validate_static_input_kind(
            spec.name(),
            BatchKind::Array,
            Some(*kind),
            &base,
            &mut issues,
        );
    }
    issues
}

fn external_inputs(builder: &PipelineBuilder) -> BTreeMap<String, BatchKind> {
    let connected = builder
        .edges
        .iter()
        .map(|edge| edge.target.clone())
        .collect::<BTreeSet<_>>();
    let ports = builder
        .nodes
        .iter()
        .flat_map(|(node, definition)| {
            definition.operator.input_ports().iter().map(move |port| {
                (
                    PortEndpoint {
                        node_id: node.clone(),
                        port: port.name().into(),
                    },
                    port.kind(),
                )
            })
        })
        .filter(|(endpoint, _)| !connected.contains(endpoint))
        .collect::<BTreeMap<_, _>>();
    crate::pipeline::external_names(ports.keys().cloned().collect())
        .into_iter()
        .map(|(name, endpoint)| (name, ports[&endpoint]))
        .collect()
}
