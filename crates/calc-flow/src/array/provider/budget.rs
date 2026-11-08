use super::{
    infer::{Descriptor, Plan},
    ir::{Node, Reduce},
};
use crate::{
    Result,
    array::{
        ReductionOptions, RollingMeanOptions, elementwise, invalid, reduction, rolling,
        rolling_mean,
    },
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
        let cost = admit(plan, index, descriptors, retained)?;
        retained = add(retained, cost.retained)?;
        costs[index] = cost;
    }
    Ok(costs)
}
fn admit(plan: &Plan, index: usize, descriptors: &[Descriptor], retained: usize) -> Result<Cost> {
    if descriptors[index].shape.len()? > plan.program.execution.max_elements {
        return Err(invalid(
            "outputs.output.elements",
            "intermediate elements exceed limit",
        ));
    }
    let cost = cost(plan, index, descriptors)?;
    if add(retained, cost.peak)? > plan.program.execution.max_allocation_bytes {
        return Err(invalid(
            "outputs.bytes",
            "program output and scratch bytes exceed limit",
        ));
    }
    Ok(cost)
}
fn cost(plan: &Plan, index: usize, descriptors: &[Descriptor]) -> Result<Cost> {
    let node = &plan.program.nodes[index];
    let output = descriptors[index];
    match node {
        Node::Reduce { .. } => reduce_cost(plan, node, output, descriptors),
        Node::RollingMean { .. } => rolling_cost(node, output.shape.len()?),
        Node::Ewma { .. } => ewma_cost(output),
        _ => layout_cost(plan, node, output, descriptors),
    }
}
fn layout_cost(
    plan: &Plan,
    node: &Node,
    output: Descriptor,
    descriptors: &[Descriptor],
) -> Result<Cost> {
    match node {
        Node::Reshape { .. } => Ok(alias_cost(output)),
        Node::Transpose { input, .. } => transpose_cost(descriptors[plan.indices[input]], output),
        _ => pointwise_cost(output),
    }
}
fn alias_cost(output: Descriptor) -> Cost {
    let retained = output.shape.rank() * size_of::<usize>();
    Cost {
        retained,
        peak: retained,
    }
}
fn transpose_cost(input: Descriptor, output: Descriptor) -> Result<Cost> {
    let dimensions = input.shape.concrete()?;
    let trivial = dimensions.len() < 2 || dimensions.contains(&1) || input.shape.len()? == 0;
    if trivial {
        return Ok(alias_cost(output));
    }
    pointwise_cost(Descriptor {
        nullable: input.validity,
        ..output
    })
}
fn pointwise_cost(output: Descriptor) -> Result<Cost> {
    let count = output.shape.len()?;
    let retained = if output.shape.rank() < 2 {
        elementwise::required_bytes(
            output.dtype.native(),
            count,
            output.shape.rank() == 0,
            output.nullable,
        )?
    } else {
        elementwise::required_shape_bytes(
            output.dtype.native(),
            count,
            output.shape.rank(),
            output.nullable,
        )?
    };
    Ok(Cost {
        retained,
        peak: retained,
    })
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
fn ewma_cost(output: Descriptor) -> Result<Cost> {
    from_parts(rolling::ewma::required_bytes(output.shape.len()?)?)
}
fn from_parts((retained, scratch): (usize, usize)) -> Result<Cost> {
    Ok(Cost {
        retained,
        peak: add(retained, scratch)?,
    })
}
fn reduce_cost(
    plan: &Plan,
    node: &Node,
    output: Descriptor,
    descriptors: &[Descriptor],
) -> Result<Cost> {
    let Node::Reduce {
        function,
        input,
        skip_nulls,
        axis,
        keepdims,
        ..
    } = node
    else {
        unreachable!("reduction node was selected")
    };
    if plan.program.ir_version == 1 {
        return legacy_reduction(*function, output);
    }
    let input = descriptors[plan.indices[input]];
    let options = ReductionOptions::default()
        .with_skip_nulls(*skip_nulls)
        .with_axis(*axis)
        .with_keepdims(*keepdims);
    let (_, bytes, scratch) = reduction::reduction_requirements(
        input.dtype.native(),
        &input.shape.concrete()?,
        input.nullable,
        function.native(),
        options,
    )?;
    from_parts((bytes, scratch))
}
fn legacy_reduction(function: Reduce, output: Descriptor) -> Result<Cost> {
    let retained = output.dtype.width() + usize::from(output.nullable);
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
