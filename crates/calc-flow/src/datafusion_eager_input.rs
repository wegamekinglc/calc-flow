use super::{BTreeMap, Batch, DataFusionRuntime, Result, ValidatedQuery, datafusion_error};
use datafusion::{
    arrow::datatypes::{DataType, Schema},
    physical_expr::{PhysicalExpr, expressions::CastExpr, utils::reassign_expr_columns},
    physical_plan::{ExecutionPlan, aggregates::AggregateExec, projection::ProjectionExec},
};
use std::sync::Arc;

impl DataFusionRuntime {
    pub(crate) async fn prove_eager_sql_input(
        &self,
        query: &ValidatedQuery,
        alias: &str,
        input: &Batch,
        expected: &[Arc<dyn PhysicalExpr>],
        node: &str,
    ) -> Result<bool> {
        if !self.grouped_float_model_supported(node)? || input.num_rows() == 0 {
            return Ok(false);
        }
        let fee = self.incremental_reservation(node);
        fee.try_grow(super::grouped_float::plan_charge(input, query, node)?)
            .map_err(|error| datafusion_error(Some(node), error))?;
        let _guard = self.query_lock.lock().await;
        let context = self.context_for_rows(input.num_rows(), None, "not_evaluated");
        let tables = BTreeMap::from([(alias.to_owned(), input.clone())]);
        let planned = self
            .prepare_query(context, query, &tables, Some(node))
            .await?;
        let mut actual = Vec::new();
        projections(planned.physical_plan.as_ref(), false, &mut actual);
        let schema = input.table_payload()?.schema();
        let mut remaining = actual.iter();
        Ok(expected.iter().all(|expected| {
            remaining.any(|expression| {
                reassign_expr_columns(Arc::clone(expression), schema)
                    .is_ok_and(|actual| evaluates(expected, actual, schema))
            })
        }))
    }
}

fn evaluates(
    expected: &Arc<dyn PhysicalExpr>,
    mut actual: Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> bool {
    for _ in 0..=8 {
        if expected.eq(&actual) {
            return true;
        }
        let Some(cast) = actual.downcast_ref::<CastExpr>() else {
            return false;
        };
        if !matches!(cast.cast_type(), DataType::Float32 | DataType::Float64)
            || !cast.expr().data_type(schema).is_ok_and(|dtype| {
                dtype.is_integer() || matches!(dtype, DataType::Float32 | DataType::Float64)
            })
        {
            return false;
        }
        actual = Arc::clone(cast.expr());
    }
    false
}

fn projections(
    plan: &dyn ExecutionPlan,
    in_input: bool,
    expressions: &mut Vec<Arc<dyn PhysicalExpr>>,
) {
    for child in plan.children() {
        projections(
            child.as_ref(),
            in_input || plan.is::<AggregateExec>(),
            expressions,
        );
    }
    if in_input && let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        expressions.extend(
            projection
                .expr()
                .iter()
                .map(|expression| Arc::clone(&expression.expr)),
        );
    }
}
