use super::{BTreeMap, Batch, DataFusionRuntime, Result, ValidatedQuery};
use datafusion::{
    arrow::datatypes::DataType,
    datasource::memory::DataSourceExec,
    physical_plan::{
        ExecutionPlan, ExecutionPlanProperties, InputOrderMode,
        aggregates::{AggregateExec, AggregateMode},
        filter::FilterExec,
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
        let Some(_fee) = self.reserve_grouped_proof(input, query, node)? else {
            return Ok(false);
        };
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
    let valid = inspect_node(plan, input, in_input, census)?;
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

fn inspect_node(
    plan: &dyn ExecutionPlan,
    input: &Batch,
    in_input: bool,
    census: &mut [usize; 3],
) -> Result<bool> {
    Ok(
        if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
            census[0] += 1;
            scalar_aggregate(aggregate)
        } else if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
            census[1] += 1;
            super::grouped_float::fifo_source(source, input)?
        } else if let Some(filter) = plan.downcast_ref::<FilterExec>() {
            census[2] += usize::from(in_input);
            filter.batch_size() == 8192 && filter.fetch().is_none()
        } else {
            super::grouped_float::output_node(plan, in_input)
        },
    )
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
                filter
                    .data_type(&aggregate.input().schema())
                    .is_ok_and(|dtype| dtype == DataType::Boolean)
            })
        })
}
