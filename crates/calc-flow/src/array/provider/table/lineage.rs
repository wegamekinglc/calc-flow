use super::super::{
    infer::{self, Descriptor, Plan, Shape},
    ir::Node,
};
use crate::{Result, array::invalid};

#[derive(Clone, Copy, Eq, PartialEq)]
enum Origin {
    Scalar,
    Static,
    Rows,
    Lost,
}

pub(super) fn validate(plan: &Plan, table: &str) -> Result<Descriptor> {
    let descriptors = infer::infer(plan, &infer::abstract_inputs(plan))?;
    let mut origins = vec![Origin::Lost; descriptors.len()];
    for &index in &plan.order {
        let origin = |id: &str| origins[plan.indices[id]];
        let shape = |id: &str| descriptors[plan.indices[id]].shape;
        origins[index] = node(&plan.program.nodes[index], table, &origin, &shape);
    }
    let index = plan.indices[&plan.program.outputs[0].node];
    if origins[index] != Origin::Rows {
        return Err(invalid(
            "options.outputs.node",
            "result must preserve the table row origin",
        ));
    }
    Ok(descriptors[index])
}
fn merge(left: Origin, right: Origin) -> Origin {
    match (left, right) {
        (Origin::Scalar, other) | (other, Origin::Scalar) => other,
        (Origin::Rows, Origin::Rows) => Origin::Rows,
        _ => Origin::Lost,
    }
}
fn node(
    node: &Node,
    table: &str,
    origin: &impl Fn(&str) -> Origin,
    shape: &impl Fn(&str) -> Shape,
) -> Origin {
    match node {
        Node::Input { name, .. } => {
            if name == table {
                Origin::Rows
            } else {
                Origin::Static
            }
        }
        Node::Literal { .. } => Origin::Scalar,
        Node::Unary { input, .. } => origin(input),
        Node::Binary { left, right, .. } => merge(origin(left), origin(right)),
        Node::Where {
            condition,
            if_true,
            if_false,
            ..
        } => merge(origin(condition), merge(origin(if_true), origin(if_false))),
        _ => structural(node, origin, shape),
    }
}
fn structural(
    node: &Node,
    origin: &impl Fn(&str) -> Origin,
    shape: &impl Fn(&str) -> Shape,
) -> Origin {
    match node {
        Node::Reduce {
            input,
            axis: Some(1),
            ..
        } => row_reduction(origin(input), shape(input)),
        Node::Transpose { input, .. } => {
            if shape(input).rank() < 2 {
                origin(input)
            } else {
                Origin::Lost
            }
        }
        Node::Matmul { left, right, .. } => matmul(origin(left), origin(right)),
        Node::Reshape {
            input,
            shape: target,
            ..
        } => reshape(origin(input), shape(input), target),
        _ => Origin::Lost,
    }
}
fn row_reduction(origin: Origin, shape: Shape) -> Origin {
    if shape.rank() == 2 {
        origin
    } else {
        Origin::Lost
    }
}
fn width(shape: Shape) -> Option<usize> {
    match shape {
        Shape::Vector(_) => Some(1),
        Shape::Matrix(_, Some(width)) if width > 0 => Some(width),
        _ => None,
    }
}
fn reshape(origin: Origin, source: Shape, target: &[usize]) -> Origin {
    let target = Shape::known(target).expect("reshape target already validated");
    if width(source).is_some() && width(source) == width(target) {
        origin
    } else {
        Origin::Lost
    }
}

pub(super) fn stream(plan: &Plan) -> Result<()> {
    if plan
        .program
        .nodes
        .iter()
        .any(|node| matches!(node, Node::Reshape { .. }))
    {
        return Err(invalid(
            "options.nodes.shape",
            "concrete reshape is not invariant across arbitrary microbatch lengths",
        ));
    }
    Ok(())
}

fn matmul(left: Origin, right: Origin) -> Origin {
    if left == Origin::Rows && right == Origin::Static {
        Origin::Rows
    } else {
        Origin::Lost
    }
}
