//! Narrow compiled-projection rewrites using existing `DataFusion` expressions.

use std::sync::Arc;

use datafusion::{
    arrow::datatypes::{DataType, Schema},
    common::ScalarValue,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, CaseExpr, Column, Literal, NegativeExpr},
        projection::{ProjectionExpr, ProjectionExprs, Projector},
    },
};

type SignOperands<'a> = (&'a Arc<dyn PhysicalExpr>, &'a Arc<dyn PhysicalExpr>);

pub(super) fn rewrite(original: &Projector, schema: &Schema) -> Option<Projector> {
    let replacements = original
        .projection()
        .iter()
        .map(|item| replace_case(&item.expr, schema))
        .collect::<Vec<_>>();
    let count = replacements.iter().flatten().count();
    if count == 0 {
        return None;
    }
    let expressions = original
        .projection()
        .iter()
        .zip(replacements)
        .map(|(item, replacement)| {
            ProjectionExpr::new(
                replacement.unwrap_or_else(|| Arc::clone(&item.expr)),
                item.alias.clone(),
            )
        });
    let updated = ProjectionExprs::new(expressions)
        .make_projector(schema)
        .ok()?;
    if updated.output_schema() != original.output_schema() {
        return None;
    }
    #[cfg(test)]
    super::sign_case_tests::note_rewrite(count);
    Some(updated)
}

fn sign_operands<'a>(
    expression: &'a Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> Option<SignOperands<'a>> {
    let case = expression.downcast_ref::<CaseExpr>()?;
    if case.expr().is_some() {
        return None;
    }
    let [(predicate, positive)] = case.when_then_expr() else {
        return None;
    };
    let column = positive.downcast_ref::<Column>()?;
    if !valid_types(predicate, column, schema) {
        return None;
    }
    let negative = case.else_expr()?.downcast_ref::<NegativeExpr>()?;
    (negative.arg().downcast_ref::<Column>()? == column).then_some((predicate, positive))
}

fn valid_types(predicate: &Arc<dyn PhysicalExpr>, column: &Column, schema: &Schema) -> bool {
    predicate.data_type(schema).ok() == Some(DataType::Boolean)
        && column.data_type(schema).ok() == Some(DataType::Int64)
}

fn replace_case(
    expression: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> Option<Arc<dyn PhysicalExpr>> {
    let (predicate, column) = sign_operands(expression, schema)?;
    let literal =
        |value| Arc::new(Literal::new(ScalarValue::Int64(Some(value)))) as Arc<dyn PhysicalExpr>;
    let factor = Arc::new(
        CaseExpr::try_new(
            None,
            vec![(Arc::clone(predicate), literal(1))],
            Some(literal(-1)),
        )
        .ok()?,
    );
    let replacement: Arc<dyn PhysicalExpr> = Arc::new(
        BinaryExpr::new(factor, Operator::Multiply, Arc::clone(column))
            .with_fail_on_overflow(false),
    );
    let before = (
        expression.data_type(schema).ok()?,
        expression.nullable(schema).ok()?,
    );
    let after = (
        replacement.data_type(schema).ok()?,
        replacement.nullable(schema).ok()?,
    );
    (before == after).then_some(replacement)
}
