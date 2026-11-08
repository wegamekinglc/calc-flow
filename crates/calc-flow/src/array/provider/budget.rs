use super::{
    infer::{Descriptor, Plan},
    ir::{Node, Reduce},
};
use crate::{
    Result,
    array::{RollingMeanOptions, elementwise, invalid, reduction, rolling, rolling_mean},
};

#[derive(Clone, Copy)]
pub(super) struct Cost {
    pub retained: usize,
    pub peak: usize,
}

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid("outputs.bytes", "allocation byte count overflowed"))
}

pub(super) fn preflight(plan: &Plan, descriptors: &[Descriptor]) -> Result<Vec<Cost>> {
    let mut retained = 0;
    let mut costs = vec![
        Cost {
            retained: 0,
            peak: 0
        };
        descriptors.len()
    ];
    for &index in &plan.order {
        if matches!(plan.program.nodes[index], Node::Input { .. }) {
            continue;
        }
        let cost = admit(
            plan,
            &plan.program.nodes[index],
            descriptors[index],
            retained,
        )?;
        retained = add(retained, cost.retained)?;
        costs[index] = cost;
    }
    Ok(costs)
}

fn admit(plan: &Plan, node: &Node, descriptor: Descriptor, retained: usize) -> Result<Cost> {
    if descriptor.shape.len()? > plan.program.execution.max_elements {
        return Err(invalid(
            "outputs.output.elements",
            "intermediate elements exceed limit",
        ));
    }
    let cost = cost(node, descriptor)?;
    if add(retained, cost.peak)? > plan.program.execution.max_allocation_bytes {
        return Err(invalid(
            "outputs.bytes",
            "program output and scratch bytes exceed limit",
        ));
    }
    Ok(cost)
}
fn cost(node: &Node, descriptor: Descriptor) -> Result<Cost> {
    let len = descriptor.shape.len()?;
    match node {
        Node::Reduce { function, .. } => reduction_cost(*function, descriptor),
        Node::RollingMean { .. } => rolling_cost(node, len),
        Node::Ewma { .. } => from_parts(rolling::ewma::required_bytes(len)?),
        _ => {
            let retained = elementwise::required_bytes(
                descriptor.dtype.native(),
                len,
                descriptor.shape.rank() == 0,
                descriptor.nullable,
            )?;
            Ok(Cost {
                retained,
                peak: retained,
            })
        }
    }
}
fn rolling_cost(node: &Node, len: usize) -> Result<Cost> {
    let Node::RollingMean {
        window,
        min_periods,
        numerical_profile,
        ..
    } = node
    else {
        unreachable!("rolling node was selected")
    };
    let options = RollingMeanOptions::new(*window)?
        .with_min_periods(*min_periods)?
        .with_numerical_profile(numerical_profile.native());
    from_parts(rolling_mean::required_bytes(len, options)?)
}
fn from_parts((retained, scratch): (usize, usize)) -> Result<Cost> {
    Ok(Cost {
        retained,
        peak: add(retained, scratch)?,
    })
}

fn reduction_cost(function: Reduce, descriptor: Descriptor) -> Result<Cost> {
    let retained = descriptor.dtype.width() + usize::from(descriptor.nullable);
    let peak = if matches!(function, Reduce::Sum | Reduce::Mean) {
        add(8, reduction::exact_scratch_bytes())?.max(retained)
    } else {
        retained
    };
    Ok(Cost { retained, peak })
}

pub(super) fn add_retained(current: usize, bytes: usize, maximum: usize) -> Result<usize> {
    let total = add(current, bytes)?;
    if total > maximum {
        return Err(invalid(
            "outputs.bytes",
            "actual retained output bytes exceed program limit",
        ));
    }
    Ok(total)
}
