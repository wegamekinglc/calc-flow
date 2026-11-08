use super::super::{infer::Plan, ir::Node};
use super::source::Selection;
use crate::{Result, array::invalid};
use std::collections::BTreeSet;

pub(super) fn validate(plan: &Plan, source: &Selection) -> Result<()> {
    if let Some(input) = &source.static_input {
        return required(plan, &input.name);
    }
    if right(plan).is_some() {
        return Err(invalid(
            "options.nodes.right",
            "table matmul requires a declared static RHS",
        ));
    }
    Ok(())
}

fn right(plan: &Plan) -> Option<&str> {
    plan.program.nodes.iter().find_map(|node| match node {
        Node::Matmul { right, .. } => Some(right.as_str()),
        _ => None,
    })
}
fn required(plan: &Plan, name: &str) -> Result<()> {
    let id = right(plan).ok_or_else(|| {
        invalid(
            "options.inputs",
            "a declared static matrix must be consumed by one matmul",
        )
    })?;
    let Node::Input { name: input, .. } = &plan.program.nodes[plan.indices[id]] else {
        return Err(invalid(
            "options.nodes.right",
            "static RHS must be a direct input reference",
        ));
    };
    if input != name {
        return Err(invalid(
            "options.nodes.right",
            "matmul RHS must reference the declared static input",
        ));
    }
    uses(plan, name)
}
fn uses(plan: &Plan, name: &str) -> Result<()> {
    let ids = plan
        .program
        .nodes
        .iter()
        .filter_map(|node| match node {
            Node::Input { id, name: input } if input == name => Some(id.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    for node in &plan.program.nodes {
        if has_reference(node, &ids) && !direct_rhs(node, &ids) {
            return Err(invalid(
                "options.nodes.right",
                "static matrix may only occur as the direct matmul RHS",
            ));
        }
    }
    Ok(())
}
fn has_reference(node: &Node, ids: &BTreeSet<&str>) -> bool {
    node.references()
        .iter()
        .any(|reference| ids.contains(reference))
}
fn direct_rhs(node: &Node, ids: &BTreeSet<&str>) -> bool {
    match node {
        Node::Matmul { left, right, .. } => {
            ids.contains(right.as_str()) && !ids.contains(left.as_str())
        }
        _ => false,
    }
}
