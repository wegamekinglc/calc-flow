use super::{
    DataType, Expr, PhysicalExpr, Result, ScalarValue, Schema, checked_bytes, df_error, unalias,
};
use datafusion::{
    arrow::datatypes::FieldRef,
    common::DFSchema,
    logical_expr::{ExprSchemable, LogicalPlan, Operator, Projection},
    physical_expr::expressions::{
        BinaryExpr, CaseExpr, CastExpr, Column, IsNotNullExpr, IsNullExpr, Literal, NegativeExpr,
        NotExpr, TryCastExpr,
    },
};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::operator::sql) enum UnaryKind {
    Negative,
    Not,
    IsNull,
    IsNotNull,
}

#[derive(Debug, PartialEq)]
pub(in crate::operator::sql) enum NativeAggregateInput {
    Column {
        index: usize,
        field: FieldRef,
    },
    Literal(ScalarValue),
    Cast {
        input: Box<Self>,
        field: FieldRef,
        safe: bool,
    },
    TryCast {
        input: Box<Self>,
        dtype: DataType,
    },
    Binary {
        left: Box<Self>,
        op: Operator,
        right: Box<Self>,
        fail_on_overflow: bool,
    },
    Unary {
        input: Box<Self>,
        op: UnaryKind,
    },
    Case {
        operand: Option<Box<Self>>,
        branches: Vec<(Self, Self)>,
        fallback: Option<Box<Self>>,
    },
}

impl NativeAggregateInput {
    pub(super) fn identity_bytes(&self, name: &str) -> Result<usize> {
        let bytes = match self {
            Self::Column { field, .. } => checked_bytes(2048, [(field.name().len(), 8)], name)?,
            Self::Literal(_) => 4096,
            Self::Cast { input, field, .. } => checked_bytes(
                input.identity_bytes(name)?,
                [(1, 2048), (field.name().len(), 8)],
                name,
            )?,
            Self::TryCast { input, .. } | Self::Unary { input, .. } => {
                checked_bytes(input.identity_bytes(name)?, [(1, 2048)], name)?
            }
            Self::Binary { left, right, .. } => checked_bytes(
                left.identity_bytes(name)?,
                [(right.identity_bytes(name)?, 1), (1, 2048)],
                name,
            )?,
            Self::Case {
                operand,
                branches,
                fallback,
            } => operand
                .iter()
                .map(Box::as_ref)
                .chain(branches.iter().flat_map(|(when, then)| [when, then]))
                .chain(fallback.iter().map(Box::as_ref))
                .try_fold(4096, |bytes, input| {
                    checked_bytes(bytes, [(input.identity_bytes(name)?, 1)], name)
                })?,
        };
        Ok(bytes)
    }
}

pub(super) fn output_work(projection: &Projection) -> Option<usize> {
    let nodes = projection
        .expr
        .iter()
        .try_fold(0_usize, |total, expression| {
            total.checked_add(projection_nodes(expression, projection.input.schema(), 0)?)
        })?;
    match projection.input.as_ref() {
        LogicalPlan::Filter(filter) => nodes.checked_add(projection_nodes(
            &filter.predicate,
            filter.input.schema(),
            0,
        )?),
        LogicalPlan::Aggregate(_) => Some(nodes),
        _ => None,
    }
}

pub(super) fn expression_work(expression: &Expr, schema: &DFSchema) -> Option<usize> {
    projection_nodes(expression, schema, 0)
}

pub(super) fn input_expression_work(expression: &Expr, schema: &DFSchema) -> Option<usize> {
    if let Expr::Column(_) = unalias(expression) {
        return Some(1);
    }
    if !expression
        .get_type(schema)
        .ok()
        .is_some_and(|dtype| fixed(&dtype))
        || expression.column_refs().iter().any(|column| {
            !Expr::Column((*column).clone())
                .get_type(schema)
                .is_ok_and(|dtype| fixed(&dtype))
        })
    {
        return None;
    }
    expression_work(expression, schema)
}

pub(super) fn aggregate_filter_work(expression: &Expr, schema: &DFSchema) -> Option<usize> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    let mut has_case = false;
    expression
        .apply(|expression| {
            has_case |= matches!(expression, Expr::Case(_));
            Ok(TreeNodeRecursion::Continue)
        })
        .ok()?;
    if has_case && !infallible_predicate(expression) {
        return None;
    }
    input_expression_work(expression, schema)
}

fn infallible_predicate(expression: &Expr) -> bool {
    match unalias(expression) {
        Expr::Column(_) | Expr::Literal(_, _) => true,
        Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) => {
            infallible_predicate(inner)
        }
        Expr::BinaryExpr(binary)
            if matches!(
                binary.op,
                Operator::Eq
                    | Operator::NotEq
                    | Operator::Lt
                    | Operator::LtEq
                    | Operator::Gt
                    | Operator::GtEq
                    | Operator::And
                    | Operator::Or
                    | Operator::IsDistinctFrom
                    | Operator::IsNotDistinctFrom
            ) =>
        {
            infallible_predicate(&binary.left) && infallible_predicate(&binary.right)
        }
        Expr::Case(case) => case
            .expr
            .iter()
            .map(Box::as_ref)
            .chain(
                case.when_then_expr
                    .iter()
                    .flat_map(|(when, then)| [when.as_ref(), then.as_ref()]),
            )
            .chain(case.else_expr.iter().map(Box::as_ref))
            .all(infallible_predicate),
        _ => false,
    }
}

pub(super) fn input_work(aggregate: &datafusion::logical_expr::Aggregate) -> Option<usize> {
    let key_nodes = aggregate
        .group_expr
        .iter()
        .try_fold(0_usize, |nodes, expression| {
            nodes.checked_add(
                aggregate_filter_work(expression, aggregate.input.schema())?
                    .saturating_sub(usize::from(matches!(unalias(expression), Expr::Column(_)))),
            )
        })?;
    aggregate
        .aggr_expr
        .iter()
        .try_fold(key_nodes, |nodes, expression| {
            let Expr::AggregateFunction(function) = unalias(expression) else {
                return None;
            };
            let nodes = function
                .params
                .args
                .iter()
                .try_fold(nodes, |nodes, argument| {
                    nodes.checked_add(input_expression_work(argument, aggregate.input.schema())?)
                })?;
            match function.params.filter.as_deref() {
                Some(filter) => nodes.checked_add(
                    aggregate_filter_work(filter, aggregate.input.schema())?
                        .saturating_sub(usize::from(matches!(unalias(filter), Expr::Column(_)))),
                ),
                None => Some(nodes),
            }
        })
}

fn fixed(dtype: &DataType) -> bool {
    dtype.primitive_width().is_some() || matches!(dtype, DataType::Boolean | DataType::Null)
}

fn projection_nodes(expression: &Expr, schema: &DFSchema, depth: usize) -> Option<usize> {
    if depth > 8 {
        return None;
    }
    match unalias(expression) {
        Expr::Column(_) => Some(1),
        Expr::Literal(value, _) if fixed(&value.data_type()) => Some(1),
        Expr::Cast(cast) if fixed(cast.field.data_type()) => {
            projection_nodes(&cast.expr, schema, depth + 1)?.checked_add(1)
        }
        Expr::TryCast(cast) if fixed(cast.field.data_type()) => {
            projection_nodes(&cast.expr, schema, depth + 1)?.checked_add(1)
        }
        Expr::Negative(inner) | Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) => {
            projection_nodes(inner, schema, depth + 1)?.checked_add(1)
        }
        Expr::BinaryExpr(binary) if supported_operator(binary.op) => {
            projection_nodes(&binary.left, schema, depth + 1)?
                .checked_add(projection_nodes(&binary.right, schema, depth + 1)?)?
                .checked_add(1)
        }
        Expr::Case(case)
            if expression
                .get_type(schema)
                .ok()
                .is_some_and(|dtype| fixed(&dtype)) =>
        {
            case.expr
                .iter()
                .map(Box::as_ref)
                .chain(
                    case.when_then_expr
                        .iter()
                        .flat_map(|(when, then)| [when.as_ref(), then.as_ref()]),
                )
                .chain(case.else_expr.iter().map(Box::as_ref))
                .try_fold(1_usize, |nodes, expression| {
                    nodes.checked_add(projection_nodes(expression, schema, depth + 1)?)
                })
        }
        _ => None,
    }
}

fn supported_operator(op: Operator) -> bool {
    matches!(
        op,
        Operator::Plus
            | Operator::Minus
            | Operator::Multiply
            | Operator::Divide
            | Operator::Modulo
            | Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::And
            | Operator::Or
            | Operator::IsDistinctFrom
            | Operator::IsNotDistinctFrom
    )
}

pub(super) fn describe_input(
    expression: &dyn PhysicalExpr,
    schema: &Schema,
    depth: usize,
    name: &str,
) -> Result<NativeAggregateInput> {
    if depth > 8 {
        return Err(df_error(name, "native aggregate input is too deep"));
    }
    if let Some(column) = expression.downcast_ref::<Column>() {
        return Ok(NativeAggregateInput::Column {
            index: column.index(),
            field: schema
                .fields()
                .get(column.index())
                .ok_or_else(|| df_error(name, "native expression column is absent"))?
                .clone(),
        });
    }
    if let Some(literal) = expression.downcast_ref::<Literal>() {
        return Ok(NativeAggregateInput::Literal(literal.value().clone()));
    }
    if let Some(cast) = expression.downcast_ref::<CastExpr>() {
        if cast.cast_options().format_options != datafusion::common::format::DEFAULT_FORMAT_OPTIONS
        {
            return Err(df_error(name, "native cast has unsupported format options"));
        }
        return Ok(NativeAggregateInput::Cast {
            input: Box::new(describe_input(
                cast.expr().as_ref(),
                schema,
                depth + 1,
                name,
            )?),
            field: cast.target_field().clone(),
            safe: cast.cast_options().safe,
        });
    }
    if let Some(cast) = expression.downcast_ref::<TryCastExpr>() {
        return Ok(NativeAggregateInput::TryCast {
            input: Box::new(describe_input(
                cast.expr().as_ref(),
                schema,
                depth + 1,
                name,
            )?),
            dtype: cast.cast_type().clone(),
        });
    }
    if let Some(binary) = expression.downcast_ref::<BinaryExpr>() {
        if !supported_operator(*binary.op()) {
            return Err(df_error(name, "native binary operator is unsupported"));
        }
        let checked = BinaryExpr::new(binary.left().clone(), *binary.op(), binary.right().clone())
            .with_fail_on_overflow(true);
        return Ok(NativeAggregateInput::Binary {
            left: Box::new(describe_input(
                binary.left().as_ref(),
                schema,
                depth + 1,
                name,
            )?),
            op: *binary.op(),
            right: Box::new(describe_input(
                binary.right().as_ref(),
                schema,
                depth + 1,
                name,
            )?),
            fail_on_overflow: binary == &checked,
        });
    }
    if let Some(case) = expression.downcast_ref::<CaseExpr>() {
        return describe_case(case, schema, depth, name);
    }
    if let Some((op, input)) = unary(expression) {
        return Ok(NativeAggregateInput::Unary {
            input: Box::new(describe_input(input, schema, depth + 1, name)?),
            op,
        });
    }
    Err(df_error(
        name,
        "native aggregate input has an unsupported expression",
    ))
}

fn describe_case(
    case: &CaseExpr,
    schema: &Schema,
    depth: usize,
    name: &str,
) -> Result<NativeAggregateInput> {
    let optional = |expression: Option<&std::sync::Arc<dyn PhysicalExpr>>| {
        expression
            .map(|expression| describe_input(expression.as_ref(), schema, depth + 1, name))
            .transpose()
            .map(|input| input.map(Box::new))
    };
    let branches = case
        .when_then_expr()
        .iter()
        .map(|(when, then)| {
            Ok((
                describe_input(when.as_ref(), schema, depth + 1, name)?,
                describe_input(then.as_ref(), schema, depth + 1, name)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(NativeAggregateInput::Case {
        operand: optional(case.expr())?,
        branches,
        fallback: optional(case.else_expr())?,
    })
}

fn unary(expression: &dyn PhysicalExpr) -> Option<(UnaryKind, &dyn PhysicalExpr)> {
    if let Some(expr) = expression.downcast_ref::<NegativeExpr>() {
        Some((UnaryKind::Negative, expr.arg().as_ref()))
    } else if let Some(expr) = expression.downcast_ref::<NotExpr>() {
        Some((UnaryKind::Not, expr.arg().as_ref()))
    } else if let Some(expr) = expression.downcast_ref::<IsNullExpr>() {
        Some((UnaryKind::IsNull, expr.arg().as_ref()))
    } else {
        expression
            .downcast_ref::<IsNotNullExpr>()
            .map(|expr| (UnaryKind::IsNotNull, expr.arg().as_ref()))
    }
}
