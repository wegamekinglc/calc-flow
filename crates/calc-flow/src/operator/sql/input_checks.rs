use super::{Arc, PhysicalExpr, RecordBatch, Result, SchemaRef, df_error};
use datafusion::{
    logical_expr::execution_props::ExecutionProps,
    physical_expr::{create_physical_expr, utils::reassign_expr_columns},
};

pub(super) fn bind(
    aggregate: &datafusion::logical_expr::Aggregate,
    plans: Option<&super::normalize_groups::Plans>,
    schema: &SchemaRef,
) -> Option<(Vec<Arc<dyn PhysicalExpr>>, usize)> {
    let props = ExecutionProps::new();
    let checks = plans.map_or(&[][..], |plans| plans.checks.as_slice());
    let mut nodes = super::native_expression::input_work(aggregate)?;
    let expressions = checks
        .iter()
        .map(|(expression, logical)| {
            nodes = nodes.checked_add(super::native_expression::input_expression_work(
                expression, logical,
            )?)?;
            let physical = create_physical_expr(expression, logical, &props).ok()?;
            reassign_expr_columns(physical, schema).ok()
        })
        .collect::<Option<Vec<_>>>()?;
    Some((expressions, nodes))
}

pub(super) fn evaluate(
    checks: &[Arc<dyn PhysicalExpr>],
    batch: &RecordBatch,
    name: &str,
) -> Result<()> {
    for expression in checks {
        drop(
            expression
                .evaluate(batch)
                .map_err(|error| df_error(name, error))?,
        );
    }
    Ok(())
}
