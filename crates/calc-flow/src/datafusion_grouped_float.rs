use super::{
    Arc, BTreeMap, Batch, DataFusionConfig, DataFusionRuntime, ExecutionPlanProperties, Result,
    ValidatedQuery, datafusion_error,
};
use datafusion::{
    arrow::datatypes::DataType,
    datasource::memory::{DataSourceExec, MemorySourceConfig},
    optimizer::{analyzer::Analyzer, optimizer::Optimizer},
    physical_optimizer::optimizer::PhysicalOptimizer,
    physical_plan::{
        ExecutionPlan, InputOrderMode,
        aggregates::{AggregateExec, AggregateMode},
        filter::FilterExec,
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::ProjectionExec,
        sorts::sort::SortExec,
    },
};

impl DataFusionRuntime {
    pub(crate) fn grouped_float_model_supported(&self, node: &str) -> Result<bool> {
        self.ensure_open()?;
        if self.config != DataFusionConfig::default() || !self.selected_udfs.is_empty() {
            return Ok(false);
        }
        let fee = self.incremental_reservation(node);
        fee.try_grow(131_072)
            .map_err(|error| datafusion_error(Some(node), error))?;
        let context = self.context_for_rows(0, None, "not_evaluated");
        let state = context.state();
        let logical = Optimizer::new();
        let physical = PhysicalOptimizer::new();
        let analyzer = Analyzer::new();
        Ok(state.config().target_partitions() == 1
            && state
                .optimizer()
                .rules
                .iter()
                .map(|rule| rule.name())
                .eq(logical
                    .rules
                    .iter()
                    .map(|rule| rule.name())
                    .chain(std::iter::once("calc_flow_uint64_modulo_predicate")))
            && state
                .physical_optimizers()
                .iter()
                .map(|rule| rule.name())
                .eq(physical.rules.iter().map(|rule| rule.name()))
            && state
                .analyzer()
                .rules
                .iter()
                .map(|rule| rule.name())
                .eq(analyzer.rules.iter().map(|rule| rule.name())))
    }

    pub(crate) async fn prove_grouped_float_plan(
        &self,
        query: &ValidatedQuery,
        alias: &str,
        input: &Batch,
        node: &str,
    ) -> Result<bool> {
        if !self.grouped_float_model_supported(node)? || input.num_rows() == 0 {
            return Ok(false);
        }
        let fee = self.incremental_reservation(node);
        let bound = plan_charge(input, query, node)?;
        fee.try_grow(bound)
            .map_err(|error| datafusion_error(Some(node), error))?;
        let _guard = self.query_lock.lock().await;
        let context = self.context_for_rows(input.num_rows(), None, "not_evaluated");
        let tables = BTreeMap::from([(alias.to_owned(), input.clone())]);
        let planned = self
            .prepare_query(context, query, &tables, Some(node))
            .await?;
        let mut census = Census::default();
        let valid = inspect(planned.physical_plan.as_ref(), input, false, &mut census)?;
        Ok(valid && census.aggregates == 1 && census.sources == 1)
    }
}

#[derive(Default)]
struct Census {
    aggregates: usize,
    sources: usize,
}

fn inspect(
    plan: &dyn ExecutionPlan,
    input: &Batch,
    in_input: bool,
    census: &mut Census,
) -> Result<bool> {
    if plan.output_partitioning().partition_count() != 1 {
        return Ok(false);
    }
    let valid = if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        census.aggregates += 1;
        single_aggregate(aggregate)
    } else if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        census.sources += 1;
        fifo_source(source, input)?
    } else if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        filter.fetch().is_none()
    } else {
        plan.is::<ProjectionExec>()
            || (!in_input
                && (plan.is::<SortExec>()
                    || plan.is::<GlobalLimitExec>()
                    || plan.is::<LocalLimitExec>()))
    };
    if !valid {
        return Ok(false);
    }
    for child in plan.children() {
        let child_in_input = in_input || plan.is::<AggregateExec>();
        if !inspect(child.as_ref(), input, child_in_input, census)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn single_aggregate(aggregate: &AggregateExec) -> bool {
    *aggregate.mode() == AggregateMode::Single
        && *aggregate.input_order_mode() == InputOrderMode::Linear
        && aggregate.limit_options().is_none()
        && aggregate.group_expr().groups().len() == 1
        && !aggregate.group_expr().expr().is_empty()
        && aggregate.filter_expr().iter().all(|filter| {
            filter.as_ref().is_none_or(|filter| {
                filter
                    .data_type(&aggregate.input().schema())
                    .is_ok_and(|dtype| dtype == DataType::Boolean)
            })
        })
}

pub(super) fn fifo_source(source: &DataSourceExec, input: &Batch) -> Result<bool> {
    let Some(scan) = source.data_source().downcast_ref::<MemorySourceConfig>() else {
        return Ok(false);
    };
    let expected = input.table_payload()?;
    if scan.partitions().len() != 1
        || !scan.sort_information().is_empty()
        || source.data_source().fetch().is_some()
        || scan.original_schema() != *expected.schema()
    {
        return Ok(false);
    }
    let records = &scan.partitions()[0];
    Ok(records.len() == expected.batches().len()
        && records
            .iter()
            .zip(expected.batches())
            .all(|(actual, expected)| {
                actual.num_rows() == expected.num_rows()
                    && actual.columns().len() == expected.columns().len()
                    && actual
                        .columns()
                        .iter()
                        .zip(expected.columns())
                        .all(|(left, right)| Arc::ptr_eq(left, right))
            }))
}

pub(super) fn plan_charge(input: &Batch, query: &ValidatedQuery, node: &str) -> Result<usize> {
    let table = input.table_payload()?;
    let schema = table.schema();
    let fields = schema
        .fields()
        .iter()
        .try_fold(0usize, |bytes, field| bytes.checked_add(field.size()))
        .ok_or_else(|| datafusion_error(Some(node), "grouped schema charge overflowed"))?;
    let metadata = schema.metadata().iter().try_fold(
        schema
            .metadata()
            .capacity()
            .checked_mul(128)
            .ok_or_else(|| datafusion_error(Some(node), "grouped metadata charge overflowed"))?,
        |bytes, (key, value)| {
            bytes
                .checked_add(key.capacity())
                .and_then(|bytes| bytes.checked_add(value.capacity()))
                .ok_or_else(|| datafusion_error(Some(node), "grouped metadata strings overflowed"))
        },
    )?;
    fields
        .checked_add(metadata)
        .and_then(|bytes| bytes.checked_add(query.text().len()))
        .and_then(|bytes| bytes.checked_mul(64))
        .and_then(|bytes| {
            table
                .batches()
                .len()
                .checked_mul(schema.fields().len().saturating_add(1))
                .and_then(|shells| shells.checked_mul(512))
                .and_then(|shells| bytes.checked_add(shells))
        })
        .and_then(|bytes| bytes.checked_add(131_072))
        .ok_or_else(|| datafusion_error(Some(node), "grouped proof plan charge overflowed"))
}
