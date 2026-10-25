//! Admission certificate for DF55's single FIFO grouped ordered accumulator.
//!
//! With the default session, a one-partition `MemTable` without advertised
//! ordering has one Single/Linear aggregate. DF55's `CombinePartialFinalAggregate`
//! is independent of row/cardinality statistics for this shape; adaptive partial
//! skipping only applies to Partial mode. Single hash aggregation never skips raw
//! rows. The default unbounded pool cannot spill/reorder groups. Reject every
//! other node/model before dropping history. The native grouped factory and the
//! strict ordered comparison remain the execution implementation after admission.
use super::{BTreeMap, Batch, DataFusionRuntime, Result, ValidatedQuery};
use datafusion::{
    datasource::memory::DataSourceExec,
    execution::context::SessionConfig,
    physical_expr::expressions::Column,
    physical_plan::{
        ExecutionPlan, ExecutionPlanProperties, InputOrderMode,
        aggregates::{AggregateExec, AggregateMode},
        projection::ProjectionExec,
        sorts::sort::SortExec,
    },
};

impl DataFusionRuntime {
    pub(crate) fn ordered_last_model_supported(&self, node: &str) -> Result<bool> {
        if !self.grouped_float_model_supported(node)?
            || std::any::Any::type_id(self.runtime_env.memory_pool.as_ref())
                != std::any::TypeId::of::<datafusion::execution::memory_pool::UnboundedMemoryPool>()
        {
            return Ok(false);
        }
        let state = self.context_for_rows(0, None, "not_evaluated").state();
        let expected = SessionConfig::new()
            .with_batch_size(self.config.batch_size)
            .with_target_partitions(1);
        Ok(state
            .config_options()
            .entries()
            .iter()
            .map(|e| (&e.key, &e.value))
            .eq(expected
                .options()
                .entries()
                .iter()
                .map(|e| (&e.key, &e.value))))
    }

    pub(crate) async fn prove_ordered_last_plan(
        &self,
        query: &ValidatedQuery,
        alias: &str,
        input: &Batch,
        node: &str,
    ) -> Result<bool> {
        if input.num_rows() == 0 || !self.ordered_last_model_supported(node)? {
            return Ok(false);
        }
        let _fee = self.reserve_grouped_proof(input, query, node)?;
        let _guard = self.query_lock.lock().await;
        let planned = self
            .prepare_query(
                self.context_for_rows(input.num_rows(), None, "not_evaluated"),
                query,
                &BTreeMap::from([(alias.to_owned(), input.clone())]),
                Some(node),
            )
            .await?;
        let mut census = (0, 0);
        Ok(inspect(planned.physical_plan.as_ref(), input, false, &mut census)? && census == (1, 1))
    }
}

fn inspect(
    plan: &dyn ExecutionPlan,
    input: &Batch,
    in_input: bool,
    census: &mut (usize, usize),
) -> Result<bool> {
    if plan.output_partitioning().partition_count() != 1 {
        return Ok(false);
    }
    if !inspect_node(plan, input, in_input, census)? {
        return Ok(false);
    }
    for child in plan.children() {
        if !inspect(
            child.as_ref(),
            input,
            in_input || plan.is::<AggregateExec>(),
            census,
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
    census: &mut (usize, usize),
) -> Result<bool> {
    Ok(if let Some(agg) = plan.downcast_ref::<AggregateExec>() {
        census.0 += 1;
        single_aggregate(agg) && ordered_expression(agg)
    } else if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        census.1 += 1;
        super::grouped_float::fifo_source(source, input)?
    } else if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        projection
            .expr()
            .iter()
            .all(|expr| expr.expr.is::<Column>())
    } else {
        !in_input
            && plan
                .downcast_ref::<SortExec>()
                .is_some_and(|sort| sort.fetch().is_none())
    })
}

fn single_aggregate(agg: &AggregateExec) -> bool {
    *agg.mode() == AggregateMode::Single
        && *agg.input_order_mode() == InputOrderMode::Linear
        && agg.limit_options().is_none()
        && agg.group_expr().groups().len() == 1
        && !agg.group_expr().expr().is_empty()
        && agg
            .group_expr()
            .expr()
            .iter()
            .all(|(expr, _)| expr.is::<Column>())
        && agg.filter_expr().iter().all(Option::is_none)
}

fn ordered_expression(agg: &AggregateExec) -> bool {
    matches!(agg.aggr_expr(), [expr]
        if *expr.fun() == *datafusion::functions_aggregate::first_last::last_value_udaf()
            && !expr.order_bys().is_empty() && expr.groups_accumulator_supported()
            && expr.expressions().iter().all(|arg| arg.is::<Column>())
            && expr.order_bys().iter().all(|sort| sort.expr.is::<Column>()))
}
