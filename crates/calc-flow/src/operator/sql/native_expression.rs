use super::{
    DataType, Expr, PhysicalExpr, Result, ScalarValue, Schema, checked_bytes, df_error, unalias,
};
use datafusion::{
    arrow::datatypes::FieldRef,
    logical_expr::{LogicalPlan, Operator, Projection},
    physical_expr::expressions::{
        BinaryExpr, CastExpr, Column, IsNotNullExpr, IsNullExpr, Literal, NegativeExpr, NotExpr,
        TryCastExpr,
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
        };
        Ok(bytes)
    }
}

pub(super) fn output_work(projection: &Projection) -> Option<usize> {
    let nodes = projection
        .expr
        .iter()
        .try_fold(0_usize, |total, expression| {
            total.checked_add(projection_nodes(expression, 0)?)
        })?;
    match projection.input.as_ref() {
        LogicalPlan::Filter(filter) => nodes.checked_add(projection_nodes(&filter.predicate, 0)?),
        LogicalPlan::Aggregate(_) => Some(nodes),
        _ => None,
    }
}

fn fixed(dtype: &DataType) -> bool {
    dtype.primitive_width().is_some() || matches!(dtype, DataType::Boolean | DataType::Null)
}

fn projection_nodes(expression: &Expr, depth: usize) -> Option<usize> {
    if depth > 8 {
        return None;
    }
    match unalias(expression) {
        Expr::Column(_) => Some(1),
        Expr::Literal(value, _) if fixed(&value.data_type()) => Some(1),
        Expr::Cast(cast) if fixed(cast.field.data_type()) => {
            projection_nodes(&cast.expr, depth + 1)?.checked_add(1)
        }
        Expr::TryCast(cast) if fixed(cast.field.data_type()) => {
            projection_nodes(&cast.expr, depth + 1)?.checked_add(1)
        }
        Expr::Negative(inner) | Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) => {
            projection_nodes(inner, depth + 1)?.checked_add(1)
        }
        Expr::BinaryExpr(binary) if supported_operator(binary.op) => {
            projection_nodes(&binary.left, depth + 1)?
                .checked_add(projection_nodes(&binary.right, depth + 1)?)?
                .checked_add(1)
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
