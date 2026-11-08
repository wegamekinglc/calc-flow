use std::collections::{BTreeMap, BTreeSet};

use super::{
    super::{ArrayView, EwmaOptions, RollingMeanOptions, invalid},
    ir::{Binary, DType, Node, Program, Reduce, Unary},
};
use crate::Result;

pub(super) use super::shape::Shape;

#[derive(Clone, Copy, Debug)]
pub(super) struct Descriptor {
    pub dtype: DType,
    pub shape: Shape,
    pub nullable: bool,
    pub validity: bool,
}

pub(super) struct Plan {
    pub program: Program,
    pub order: Vec<usize>,
    pub indices: BTreeMap<String, usize>,
}

pub(super) fn plan(program: Program) -> Result<Plan> {
    let indices = program
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id().to_owned(), index))
        .collect();
    let mut plan = Plan {
        program,
        order: Vec::new(),
        indices,
    };
    plan.order = topology(&plan)?;
    validate_outputs(&plan)?;
    Ok(plan)
}

fn topology(plan: &Plan) -> Result<Vec<usize>> {
    let mut state = vec![0; plan.program.nodes.len()];
    let mut depth = vec![0; state.len()];
    for output in &plan.program.outputs {
        visit(
            plan,
            lookup(plan, &output.node)?,
            &mut state,
            &mut depth,
            &mut Vec::new(),
        )?;
    }
    if state.contains(&0) {
        return Err(invalid("options.nodes", "unreachable nodes are rejected"));
    }
    let mut sorted = Vec::new();
    let mut seen = BTreeSet::new();
    for output in &plan.program.outputs {
        order(plan, lookup(plan, &output.node)?, &mut seen, &mut sorted)?;
    }
    Ok(sorted)
}

fn validate_outputs(plan: &Plan) -> Result<()> {
    let inputs = abstract_inputs(plan);
    let descriptors = infer(plan, &inputs)?;
    for output in &plan.program.outputs {
        if let Some(assertion) = &output.assertion {
            let result = descriptors[lookup(plan, &output.node)?];
            if result.dtype != assertion.dtype
                || result.shape != Shape::from_declaration(&assertion.shape)
            {
                return Err(invalid(
                    "options.outputs",
                    "declared descriptor differs from inference",
                ));
            }
        }
    }

    Ok(())
}

pub(super) fn query_outputs(plan: &Plan) -> Result<crate::JsonMap> {
    let inputs = abstract_inputs(plan);
    let descriptors = infer(plan, &inputs)?;
    let outputs=plan.program.outputs.iter().map(|output|{
        let descriptor=descriptors[plan.indices[&output.node]];
        serde_json::json!({"name":output.name,"node":output.node,"dtype":descriptor.dtype.spelling(),"shape":descriptor.shape.dimensions()})
    }).collect::<Vec<_>>();
    Ok(BTreeMap::from([(
        "outputs".into(),
        serde_json::Value::Array(outputs),
    )]))
}
pub(super) fn abstract_inputs(plan: &Plan) -> BTreeMap<String, Descriptor> {
    plan.program
        .inputs
        .iter()
        .map(|input| {
            (
                input.name.clone(),
                Descriptor {
                    dtype: input.dtype,
                    shape: Shape::from_declaration(&input.shape),
                    nullable: true,
                    validity: true,
                },
            )
        })
        .collect()
}

fn lookup(plan: &Plan, id: &str) -> Result<usize> {
    plan.indices
        .get(id)
        .copied()
        .ok_or_else(|| invalid("options.nodes", "node reference is missing"))
}

fn visit(
    plan: &Plan,
    index: usize,
    state: &mut [u8],
    depth: &mut [usize],
    path: &mut Vec<usize>,
) -> Result<usize> {
    if state[index] == 1 {
        return Err(invalid("options.nodes", "node references contain a cycle"));
    }
    if state[index] == 2 {
        return Ok(depth[index]);
    }
    if path.len() >= 24 {
        return Err(invalid("options.nodes.depth", "node depth exceeds 24"));
    }
    state[index] = 1;
    path.push(index);
    let mut maximum = 0;
    for reference in plan.program.nodes[index].references() {
        maximum = maximum.max(visit(plan, lookup(plan, reference)?, state, depth, path)?);
    }
    path.pop();
    depth[index] = maximum + 1;
    if depth[index] > 24 {
        return Err(invalid("options.nodes.depth", "node depth exceeds 24"));
    }
    state[index] = 2;
    Ok(depth[index])
}

fn order(
    plan: &Plan,
    index: usize,
    seen: &mut BTreeSet<usize>,
    output: &mut Vec<usize>,
) -> Result<()> {
    if seen.insert(index) {
        for reference in plan.program.nodes[index].references() {
            order(plan, lookup(plan, reference)?, seen, output)?;
        }
        output.push(index);
    }
    Ok(())
}

pub(super) fn infer(plan: &Plan, inputs: &BTreeMap<String, Descriptor>) -> Result<Vec<Descriptor>> {
    let mut descriptors = vec![
        Descriptor {
            dtype: DType::Bool,
            shape: Shape::Scalar,
            nullable: false,
            validity: false
        };
        plan.program.nodes.len()
    ];
    for &index in &plan.order {
        let operand = |id: &str| lookup(plan, id).map(|index| descriptors[index]);
        let descriptor = node(&plan.program.nodes[index], inputs, &operand, plan)?;
        descriptor.shape.elements()?;
        descriptors[index] = descriptor;
    }
    Ok(descriptors)
}

type Operand<'a> = &'a dyn Fn(&str) -> Result<Descriptor>;

fn node(
    node: &Node,
    inputs: &BTreeMap<String, Descriptor>,
    operand: Operand<'_>,
    plan: &Plan,
) -> Result<Descriptor> {
    match node {
        Node::Input { name, .. } => inputs
            .get(name)
            .copied()
            .ok_or_else(|| invalid("options.nodes.name", "input reference is missing")),
        Node::Literal { dtype, value, .. } => constant(*dtype, value),
        Node::Unary {
            function, input, ..
        } => unary_operand(*function, input, operand),
        Node::Binary {
            function,
            left,
            right,
            ..
        } => binary_operands(*function, left, right, operand, plan),
        _ => nonpointwise(node, operand),
    }
}
fn constant(dtype: DType, value: &serde_json::Value) -> Result<Descriptor> {
    let _ = literal(dtype, value)?;
    Ok(Descriptor {
        dtype,
        shape: Shape::Scalar,
        nullable: false,
        validity: false,
    })
}
fn unary_operand(function: Unary, input: &str, operand: Operand<'_>) -> Result<Descriptor> {
    unary(function, operand(input)?)
}
fn binary_operands(
    function: Binary,
    left: &str,
    right: &str,
    operand: Operand<'_>,
    plan: &Plan,
) -> Result<Descriptor> {
    binary(function, operand(left)?, operand(right)?, plan, right)
}
fn nonpointwise(node: &Node, operand: Operand<'_>) -> Result<Descriptor> {
    match node {
        Node::Where {
            condition,
            if_true,
            if_false,
            ..
        } => select_operands(condition, if_true, if_false, operand),
        Node::Reduce {
            function,
            input,
            axis,
            keepdims,
            ..
        } => reduction(*function, operand(input)?, *axis, *keepdims),
        Node::RollingMean { .. } => rolling_operand(node, operand),
        Node::Ewma { .. } => ewma_operand(node, operand),
        _ => layout(node, operand),
    }
}
fn layout(node: &Node, operand: Operand<'_>) -> Result<Descriptor> {
    match node {
        Node::Matmul { left, right, .. } => matmul_operands(left, right, operand),
        Node::Reshape { input, shape, .. } => reshape_operand(input, shape, operand),
        Node::Transpose { input, .. } => transpose_operand(input, operand),
        _ => unreachable!("pointwise nodes were handled separately"),
    }
}
fn select_operands(
    condition: &str,
    yes: &str,
    no: &str,
    operand: Operand<'_>,
) -> Result<Descriptor> {
    selection(operand(condition)?, operand(yes)?, operand(no)?)
}
fn rolling_operand(node: &Node, operand: Operand<'_>) -> Result<Descriptor> {
    let Node::RollingMean {
        input,
        window,
        min_periods,
        numerical_profile,
        ..
    } = node
    else {
        unreachable!("rolling node was selected")
    };
    let _ = RollingMeanOptions::new(*window)?
        .with_min_periods(*min_periods)?
        .with_numerical_profile(numerical_profile.native());
    rolling(operand(input)?)
}
fn ewma_operand(node: &Node, operand: Operand<'_>) -> Result<Descriptor> {
    let Node::Ewma {
        input,
        span,
        min_periods,
        ..
    } = node
    else {
        unreachable!("EWMA node was selected")
    };
    let _ = EwmaOptions::new(*span)?.with_min_periods(*min_periods)?;
    rolling(operand(input)?)
}

fn matmul_operands(left: &str, right: &str, operand: Operand<'_>) -> Result<Descriptor> {
    let left = operand(left)?;
    let right = operand(right)?;
    require_numeric(left.dtype)?;
    require_numeric(right.dtype)?;
    Ok(Descriptor {
        dtype: promote(left.dtype, right.dtype)?,
        shape: left.shape.matmul(right.shape)?,
        nullable: false,
        validity: false,
    })
}
fn reshape_operand(input: &str, target: &[usize], operand: Operand<'_>) -> Result<Descriptor> {
    let input = operand(input)?;
    Ok(Descriptor {
        shape: input.shape.reshape(target)?,
        ..input
    })
}
fn transpose_operand(input: &str, operand: Operand<'_>) -> Result<Descriptor> {
    let input = operand(input)?;
    Ok(Descriptor {
        shape: input.shape.transpose(),
        ..input
    })
}

fn require_numeric(dtype: DType) -> Result<()> {
    if dtype == DType::Bool {
        Err(invalid(
            "options.nodes.dtype",
            "expected floating-point operands",
        ))
    } else {
        Ok(())
    }
}
fn require_bool(dtype: DType) -> Result<()> {
    if dtype == DType::Bool {
        Ok(())
    } else {
        Err(invalid("options.nodes.dtype", "expected Boolean operands"))
    }
}
fn promote(left: DType, right: DType) -> Result<DType> {
    if left == right {
        return Ok(left);
    }
    require_numeric(left)?;
    require_numeric(right)?;
    Ok(DType::Float64)
}
fn shape(left: Shape, right: Shape) -> Result<Shape> {
    left.broadcast(right)
}

fn unary(function: Unary, input: Descriptor) -> Result<Descriptor> {
    match function {
        Unary::LogicalNot => require_bool(input.dtype)?,
        _ => require_numeric(input.dtype)?,
    }
    Ok(Descriptor {
        validity: input.nullable,
        ..input
    })
}
fn binary(
    function: Binary,
    left: Descriptor,
    right: Descriptor,
    plan: &Plan,
    right_id: &str,
) -> Result<Descriptor> {
    let dtype = binary_dtype(function, left.dtype, right.dtype)?;
    if matches!(function, Binary::Pow) {
        exponent(&plan.program.nodes[lookup(plan, right_id)?])?;
    }
    Ok(computed(
        dtype,
        shape(left.shape, right.shape)?,
        &[left, right],
    ))
}
fn binary_dtype(function: Binary, left: DType, right: DType) -> Result<DType> {
    match function {
        Binary::LogicalAnd | Binary::LogicalOr | Binary::LogicalXor => {
            require_bool(left)?;
            require_bool(right)?;
            Ok(DType::Bool)
        }
        Binary::Equal | Binary::NotEqual => {
            promote(left, right)?;
            Ok(DType::Bool)
        }
        _ => numeric_binary_dtype(function, left, right),
    }
}
fn numeric_binary_dtype(function: Binary, left: DType, right: DType) -> Result<DType> {
    require_numeric(left)?;
    require_numeric(right)?;
    if matches!(
        function,
        Binary::Less | Binary::LessEqual | Binary::Greater | Binary::GreaterEqual
    ) {
        Ok(DType::Bool)
    } else {
        promote(left, right)
    }
}
fn selection(condition: Descriptor, yes: Descriptor, no: Descriptor) -> Result<Descriptor> {
    require_bool(condition.dtype)?;
    let dtype = promote(yes.dtype, no.dtype)?;
    let output_shape = shape(shape(condition.shape, yes.shape)?, no.shape)?;
    Ok(computed(dtype, output_shape, &[condition, yes, no]))
}
fn computed(dtype: DType, shape: Shape, inputs: &[Descriptor]) -> Descriptor {
    let nullable = inputs.iter().any(|input| input.nullable);
    Descriptor {
        dtype,
        shape,
        nullable,
        validity: nullable,
    }
}
fn reduction(
    function: Reduce,
    input: Descriptor,
    axis: Option<usize>,
    keepdims: bool,
) -> Result<Descriptor> {
    require_numeric(input.dtype)?;
    let dtype = if matches!(function, Reduce::Sum | Reduce::Mean) {
        DType::Float64
    } else {
        input.dtype
    };
    let shape = input.shape.reduced(axis, keepdims)?;
    let nullable = input.nullable || input.shape.lane_width(axis)?.is_none_or(|width| width == 0);
    Ok(Descriptor {
        dtype,
        shape,
        nullable,
        validity: nullable,
    })
}

fn rolling(input: Descriptor) -> Result<Descriptor> {
    require_numeric(input.dtype)?;
    if input.shape.rank() != 1 {
        return Err(invalid(
            "options.nodes.shape",
            "rolling input must be a vector",
        ));
    }
    Ok(Descriptor {
        dtype: DType::Float64,
        nullable: true,
        validity: true,
        ..input
    })
}
fn exponent(node: &Node) -> Result<()> {
    let Node::Literal { dtype, value, .. } = node else {
        return Err(invalid(
            "options.nodes.right",
            "pow requires a literal scalar exponent",
        ));
    };
    require_numeric(*dtype)?;
    if value.as_f64().is_none_or(|value| value.abs() > 64.0) {
        return Err(invalid("right.value", "exponent magnitude exceeds 64"));
    }
    Ok(())
}
#[allow(
    clippy::cast_possible_truncation,
    reason = "explicit float32 literals round after finite-range validation"
)]
pub(super) fn literal(dtype: DType, value: &serde_json::Value) -> Result<ArrayView<'static>> {
    if dtype == DType::Bool {
        return value
            .as_bool()
            .map(|value| ArrayView::scalar_boolean(Some(value)))
            .ok_or_else(|| invalid("options.nodes.value", "expected a Boolean literal"));
    }
    let value = numeric_literal(value)?;
    if dtype == DType::Float64 {
        return Ok(ArrayView::scalar_f64(Some(value)));
    }
    let value = value as f32;
    if !value.is_finite() {
        return Err(invalid("options.nodes.value", "float32 literal overflows"));
    }
    Ok(ArrayView::scalar_f32(Some(value)))
}
fn numeric_literal(value: &serde_json::Value) -> Result<f64> {
    let integer = value
        .as_i64()
        .is_some_and(|value| value.unsigned_abs() > 9_007_199_254_740_992)
        || value
            .as_u64()
            .is_some_and(|value| value > 9_007_199_254_740_992);
    if integer {
        return Err(invalid(
            "options.nodes.value",
            "integer-represented literal must be between -9007199254740992 and 9007199254740992",
        ));
    }
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid("options.nodes.value", "expected a finite numeric literal"))
}
