use std::{collections::BTreeMap, sync::Arc};

use super::{
    budget,
    infer::{self, Descriptor, Plan, Shape},
    ir::{Binary, Node, Reduce, Unary},
};
use crate::{
    Batch, BatchMetadata, Result,
    array::{self, ArrayExecutionOptions, ArrayView, NativeArray, elementwise, invalid},
};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(super) struct Prepared {
    costs: Vec<budget::Cost>,
}

pub(super) fn prepare(plan: &Plan, inputs: &BTreeMap<String, Batch>) -> Result<Prepared> {
    let descriptors = inputs_descriptors(plan, inputs)?;
    let concrete = infer::infer(plan, &descriptors)?;
    final_elements(plan, &concrete)?;
    Ok(Prepared {
        costs: budget::preflight(plan, &concrete)?,
    })
}
fn final_elements(plan: &Plan, descriptors: &[Descriptor]) -> Result<()> {
    let elements = plan
        .program
        .outputs
        .iter()
        .try_fold(0_usize, |total, output| {
            total
                .checked_add(descriptors[plan.indices[&output.node]].shape.len()?)
                .ok_or_else(|| invalid("outputs.elements", "output element count overflowed"))
        })?;
    if elements > plan.program.execution.max_elements {
        return Err(invalid(
            "outputs.elements",
            "named output elements exceed limit",
        ));
    }
    Ok(())
}

struct Run<'a> {
    plan: &'a Plan,
    values: Vec<Option<Batch>>,
    retained: usize,
}
impl Run<'_> {
    fn step(
        &mut self,
        index: usize,
        inputs: &BTreeMap<String, Batch>,
        cost: budget::Cost,
        check: Check<'_>,
    ) -> Result<Batch> {
        let node = &self.plan.program.nodes[index];
        if let Node::Input { name, .. } = node {
            return Ok(inputs[name].clone());
        }
        let remaining = self.plan.program.execution.max_allocation_bytes - self.retained;
        if cost.peak > remaining {
            return Err(invalid(
                "outputs.bytes",
                "program output and scratch bytes exceed limit",
            ));
        }
        let options = self
            .plan
            .program
            .execution
            .options()
            .with_max_allocation_bytes(remaining);
        let array = evaluate(node, self.plan, &self.values, &options, check)?;
        self.retained = budget::add_retained(
            self.retained,
            array.retained_buffer_bytes(),
            self.plan.program.execution.max_allocation_bytes,
        )?;
        Batch::external(Arc::new(array), BatchMetadata::default())
    }
    fn publish(&self) -> Result<BTreeMap<String, Batch>> {
        self.plan
            .program
            .outputs
            .iter()
            .map(|output| {
                let value = self.values[self.plan.indices[&output.node]]
                    .as_ref()
                    .expect("validated output was computed");
                Ok((
                    output.name.clone(),
                    Batch::external(
                        Arc::clone(value.external_payload()?),
                        BatchMetadata::default(),
                    )?,
                ))
            })
            .collect()
    }
}

pub(super) fn run(
    plan: &Plan,
    inputs: &BTreeMap<String, Batch>,
    prepared: &Prepared,
    check: Check<'_>,
) -> Result<BTreeMap<String, Batch>> {
    check()?;
    let mut run = Run {
        plan,
        values: vec![None; plan.program.nodes.len()],
        retained: 0,
    };
    for &index in &plan.order {
        check()?;
        run.values[index] = Some(run.step(index, inputs, prepared.costs[index], check)?);
    }
    check()?;
    run.publish()
}

fn inputs_descriptors(
    plan: &Plan,
    inputs: &BTreeMap<String, Batch>,
) -> Result<BTreeMap<String, Descriptor>> {
    if inputs.len() != plan.program.inputs.len() {
        return Err(invalid("inputs", "input names differ from declarations"));
    }
    let mut descriptors = BTreeMap::new();
    for input in &plan.program.inputs {
        let field = format!("inputs.{}", input.name);
        let batch = inputs
            .get(&input.name)
            .ok_or_else(|| invalid("inputs", "input names differ from declarations"))?;
        let value = native(batch, &format!("{field}.backend"))?;
        if value.dtype() != input.dtype.native() {
            return Err(invalid(
                &format!("{field}.dtype"),
                "dtype differs from declaration",
            ));
        }
        let shape = native_shape(value.shape(), &field)?;
        validate_input_shape(&input.shape, shape, &field)?;
        descriptors.insert(
            input.name.clone(),
            Descriptor {
                dtype: input.dtype,
                shape,
                nullable: value.as_arrow().null_count() != 0,
            },
        );
    }
    Ok(descriptors)
}
pub(super) fn native_shape(shape: &[usize], field: &str) -> Result<Shape> {
    match shape {
        [] => Ok(Shape::Scalar),
        [len] => Ok(Shape::Vector(Some(*len))),
        _ => Err(invalid(
            &format!("{field}.shape"),
            "native IR version 1 requires rank zero or one",
        )),
    }
}

fn validate_input_shape(expected: &[Option<usize>], shape: Shape, field: &str) -> Result<()> {
    let matches = match (Shape::from_declaration(expected), shape) {
        (Shape::Scalar, Shape::Scalar) | (Shape::Vector(None), Shape::Vector(_)) => true,
        (left, right) => left == right,
    };
    if matches {
        Ok(())
    } else {
        Err(invalid(
            &format!("{field}.shape"),
            "shape differs from declaration",
        ))
    }
}
fn native<'a>(batch: &'a Batch, field: &str) -> Result<&'a NativeArray> {
    batch
        .external_payload()
        .ok()
        .and_then(|payload| payload.as_any().downcast_ref::<NativeArray>())
        .ok_or_else(|| invalid(field, "requires native array storage"))
}
fn operand<'a>(id: &str, plan: &Plan, values: &'a [Option<Batch>]) -> Result<&'a NativeArray> {
    native(
        values[plan.indices[id]]
            .as_ref()
            .expect("dependencies were computed"),
        "nodes.backend",
    )
}
fn evaluate(
    node: &Node,
    plan: &Plan,
    values: &[Option<Batch>],
    options: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    let get = |id: &str| operand(id, plan, values);
    match node {
        Node::Literal { dtype, value, .. } => {
            copy_literal(infer::literal(*dtype, value)?, options, check)
        }
        Node::Unary {
            function, input, ..
        } => unary(*function, get(input)?.view(), options, check),
        Node::Binary {
            function,
            left,
            right,
            ..
        } => binary(
            *function,
            get(left)?.view(),
            get(right)?.view(),
            options,
            check,
        ),
        _ => structural(node, &get, options, check),
    }
}
fn copy_literal(
    value: ArrayView<'_>,
    options: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    elementwise::where_checked(
        ArrayView::scalar_boolean(Some(true)),
        value,
        value,
        options,
        check,
    )
}
fn structural<'a>(
    node: &Node,
    get: &impl Fn(&str) -> Result<&'a NativeArray>,
    options: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    if let Node::Where {
        condition,
        if_true,
        if_false,
        ..
    } = node
    {
        return elementwise::where_checked(
            get(condition)?.view(),
            get(if_true)?.view(),
            get(if_false)?.view(),
            options,
            check,
        );
    }
    stateful(node, get, options, check)
}
fn stateful<'a>(
    node: &Node,
    get: &impl Fn(&str) -> Result<&'a NativeArray>,
    options: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    match node {
        Node::Reduce {
            function,
            input,
            skip_nulls,
            ..
        } => reduce(*function, get(input)?, *skip_nulls, options, check),
        Node::RollingMean { input, .. } => rolling_mean(node, get(input)?, options, check),
        Node::Ewma { input, .. } => ewma(node, get(input)?, options, check),
        _ => unreachable!("input and pointwise nodes are handled separately"),
    }
}
fn rolling_mean(
    node: &Node,
    input: &NativeArray,
    execution: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    let Node::RollingMean {
        window,
        min_periods,
        numerical_profile,
        ..
    } = node
    else {
        unreachable!("rolling node was selected")
    };
    let options = array::RollingMeanOptions::new(*window)?
        .with_min_periods(*min_periods)?
        .with_numerical_profile(numerical_profile.native());
    array::rolling_mean::rolling_mean_checked(input.float_view()?, options, execution, check)
}
fn ewma(
    node: &Node,
    input: &NativeArray,
    execution: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    let Node::Ewma {
        span, min_periods, ..
    } = node
    else {
        unreachable!("EWMA node was selected")
    };
    let options = array::EwmaOptions::new(*span)?.with_min_periods(*min_periods)?;
    array::rolling::ewma::ewma_checked(input.float_view()?, options, execution, check)
}
fn unary(
    function: Unary,
    input: ArrayView<'_>,
    options: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    let operation = match function {
        Unary::Neg => elementwise::neg_checked,
        Unary::Abs => elementwise::abs_checked,
        Unary::Sqrt => elementwise::sqrt_checked,
        Unary::Log => elementwise::log_checked,
        Unary::Exp => elementwise::exp_checked,
        Unary::LogicalNot => elementwise::logical_not_checked,
    };
    operation(input, options, check)
}
fn binary(
    function: Binary,
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    options: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    match function {
        Binary::Add | Binary::Subtract | Binary::Multiply | Binary::Divide | Binary::Pow => {
            arithmetic(function, left, right, options, check)
        }
        Binary::LogicalAnd | Binary::LogicalOr | Binary::LogicalXor => {
            logic(function, left, right, options, check)
        }
        _ => comparison(function, left, right, options, check),
    }
}
fn arithmetic(
    function: Binary,
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    options: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    let operation = match function {
        Binary::Add => elementwise::add_checked,
        Binary::Subtract => elementwise::subtract_checked,
        Binary::Multiply => elementwise::multiply_checked,
        Binary::Divide => elementwise::divide_checked,
        Binary::Pow => elementwise::pow_checked,
        _ => unreachable!("arithmetic was selected"),
    };
    operation(left, right, options, check)
}
fn logic(
    function: Binary,
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    options: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    let operation = match function {
        Binary::LogicalAnd => elementwise::logical_and_checked,
        Binary::LogicalOr => elementwise::logical_or_checked,
        Binary::LogicalXor => elementwise::logical_xor_checked,
        _ => unreachable!("logic was selected"),
    };
    operation(left, right, options, check)
}
fn comparison(
    function: Binary,
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    options: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    let operation = match function {
        Binary::Equal => elementwise::equal_checked,
        Binary::NotEqual => elementwise::not_equal_checked,
        Binary::Less => elementwise::less_checked,
        Binary::LessEqual => elementwise::less_equal_checked,
        Binary::Greater => elementwise::greater_checked,
        Binary::GreaterEqual => elementwise::greater_equal_checked,
        _ => unreachable!("comparison was selected"),
    };
    operation(left, right, options, check)
}
fn reduce(
    function: Reduce,
    input: &NativeArray,
    skip_nulls: bool,
    execution: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    let operation = match function {
        Reduce::Sum => array::reduction::sum_checked,
        Reduce::Mean => array::reduction::mean_checked,
        Reduce::Min => array::reduction::min_checked,
        Reduce::Max => array::reduction::max_checked,
    };
    operation(
        input.float_view()?,
        array::ReductionOptions::default().with_skip_nulls(skip_nulls),
        execution,
        check,
    )
}
