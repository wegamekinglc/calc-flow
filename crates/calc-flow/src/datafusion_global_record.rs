use super::{BTreeMap, Batch, DataFusionRuntime, Result, ValidatedQuery, datafusion_error};
use datafusion::{
    arrow::datatypes::DataType,
    datasource::memory::DataSourceExec,
    physical_expr::expressions::Column,
    physical_plan::{
        ExecutionPlan, ExecutionPlanProperties, InputOrderMode,
        aggregates::{AggregateExec, AggregateMode},
        filter::FilterExec,
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::ProjectionExec,
        sorts::sort::SortExec,
    },
};

impl DataFusionRuntime {
    pub(crate) async fn prove_global_record_plan(
        &self,
        query: &ValidatedQuery,
        alias: &str,
        input: &Batch,
        coalesced: bool,
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
        let mut census = [0, 0, 0];
        let valid = inspect(planned.physical_plan.as_ref(), input, &mut census, false)?;
        Ok(valid && census == [1, 1, usize::from(coalesced)])
    }
}

fn inspect(
    plan: &dyn ExecutionPlan,
    input: &Batch,
    census: &mut [usize; 3],
    in_input: bool,
) -> Result<bool> {
    if plan.output_partitioning().partition_count() != 1 {
        return Ok(false);
    }
    let valid = if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        census[0] += 1;
        scalar_aggregate(aggregate)
    } else if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        census[1] += 1;
        super::grouped_float::fifo_source(source, input)?
    } else if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        census[2] += usize::from(in_input);
        filter.batch_size() == 8192 && filter.fetch().is_none()
    } else {
        plan.is::<ProjectionExec>()
            || !in_input
                && (plan.is::<SortExec>()
                    || plan.is::<GlobalLimitExec>()
                    || plan.is::<LocalLimitExec>())
    };
    if !valid {
        return Ok(false);
    }
    for child in plan.children() {
        if !inspect(
            child.as_ref(),
            input,
            census,
            in_input || plan.is::<AggregateExec>(),
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn scalar_aggregate(aggregate: &AggregateExec) -> bool {
    *aggregate.mode() == AggregateMode::Single
        && *aggregate.input_order_mode() == InputOrderMode::Linear
        && scalar_shape(aggregate)
}

fn scalar_shape(aggregate: &AggregateExec) -> bool {
    aggregate.group_expr().expr().is_empty()
        && aggregate.group_expr().groups().is_empty()
        && !aggregate.aggr_expr().is_empty()
        && aggregate.aggr_expr().iter().all(|expression| {
            matches!(
                expression.fun().name(),
                "sum" | "avg" | "count" | "min" | "max"
            )
        })
        && aggregate.limit_options().is_none()
        && aggregate.filter_expr().iter().all(|filter| {
            filter.as_ref().is_none_or(|filter| {
                filter.downcast_ref::<Column>().is_some()
                    && filter
                        .data_type(&aggregate.input().schema())
                        .is_ok_and(|dtype| dtype == DataType::Boolean)
            })
        })
}
