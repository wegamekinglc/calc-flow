use super::{AnalyticExpr, OverSpec, spec::AnalyticKind};
use crate::{Result, dataframe::invalid};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use std::{collections::BTreeSet, sync::Arc};

pub(crate) fn output_schema(
    schema: &SchemaRef,
    expressions: &[AnalyticExpr],
    spec: &OverSpec,
) -> Result<SchemaRef> {
    if expressions.is_empty() {
        return Err(invalid(
            "over.expressions",
            "invalid_literal",
            "at least one analytic expression required",
        ));
    }
    validate_order_columns(schema, spec)?;
    let columns = schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect::<Vec<_>>();
    let owned = crate::operator::distinct::owned::order_schema(schema, &columns).map_err(remap)?;
    let mut names = columns.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let mut fields = owned
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect::<Vec<_>>();
    for (index, expression) in expressions.iter().enumerate() {
        let path = format!("over.expressions[{index}]");
        if !names.insert(expression.name()) {
            return Err(invalid(
                &format!("{path}.name"),
                "duplicate_name",
                "output names must be unique and must not overwrite input fields",
            ));
        }
        fields.push(analytic_field(schema, expression, &path)?);
    }
    Ok(Arc::new(Schema::new(fields)))
}
fn validate_order_columns(schema: &SchemaRef, spec: &OverSpec) -> Result<()> {
    for (role, names) in [
        ("partition_by", spec.order().partition_by()),
        ("sequence_by", spec.order().sequence_by()),
    ] {
        for (index, name) in names.iter().enumerate() {
            schema.field_with_name(name).map_err(|_| {
                invalid(
                    &format!("over.spec.order.{role}[{index}]"),
                    "unresolved_type",
                    "existing input column required",
                )
            })?;
        }
    }
    Ok(())
}
fn analytic_field(schema: &SchemaRef, expression: &AnalyticExpr, path: &str) -> Result<Field> {
    let (dtype, nullable) = match &expression.0 {
        AnalyticKind::RowNumber { .. } => (DataType::UInt64, false),
        AnalyticKind::Lag { column, .. } => (operand(schema, column, path)?.clone(), true),
        AnalyticKind::Sum { column, .. } => (
            sum_type(operand(schema, column, path)?, &format!("{path}.column"))?,
            true,
        ),
    };
    Ok(Field::new(expression.name(), dtype, nullable))
}
fn operand<'a>(schema: &'a SchemaRef, column: &str, path: &str) -> Result<&'a DataType> {
    schema
        .field_with_name(column)
        .map(Field::data_type)
        .map_err(|_| {
            invalid(
                &format!("{path}.column"),
                "unresolved_type",
                "existing input operand required; aliases cannot reference other analytic outputs",
            )
        })
}
pub(crate) fn sum_type(dtype: &DataType, path: &str) -> Result<DataType> {
    match dtype {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => Ok(DataType::Int64),
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            Ok(DataType::UInt64)
        }
        _ => Err(invalid(
            path,
            "unsupported_type",
            "SUM requires signed or unsigned integer input",
        )),
    }
}
pub(crate) fn remap(error: crate::CalcFlowError) -> crate::CalcFlowError {
    match error {
        crate::CalcFlowError::InvalidArgument { field, message } => {
            crate::CalcFlowError::InvalidArgument {
                field: field
                    .replace("distinct.columns", "over.input.schema")
                    .replace("distinct", "over")
                    .replace("sort.by", "over.spec.order.order_by")
                    .replace("sort.spec.ties", "over.spec.order.sequence_by")
                    .replace("sort.spec", "over.spec")
                    .replace("sort.input", "over.input")
                    .replace(".sort.", ".over."),
                message,
            }
        }
        other => other,
    }
}
