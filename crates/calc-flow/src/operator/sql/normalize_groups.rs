use datafusion::optimizer::{
    eliminate_group_by_constant::EliminateGroupByConstant,
    optimize_projections::OptimizeProjections,
    optimizer::{Optimizer, OptimizerContext},
    push_down_filter::PushDownFilter,
};

use super::{
    Arc, DataFusionRuntime, Expr, LogicalPlan, MemoryReservation, Result, SchemaRef,
    ValidatedQuery, checked_bytes, df_error,
};

pub(super) struct Plans {
    pub raw: LogicalPlan,
    pub analyzed: LogicalPlan,
    _reservation: MemoryReservation,
}

pub(super) fn plans(
    runtime: &DataFusionRuntime,
    query: &ValidatedQuery,
    raw: &LogicalPlan,
    analyzed: &LogicalPlan,
    schema: &SchemaRef,
    name: &str,
) -> Result<Option<Plans>> {
    let Some((_, aggregate)) = super::shape(raw) else {
        return Ok(None);
    };
    if aggregate
        .group_expr
        .iter()
        .all(|expression| matches!(expression, Expr::Column(_)))
    {
        return Ok(None);
    }
    let fields = super::super::ipc::schema_bytes(schema).map_err(|error| df_error(name, error))?;
    let reservation = runtime.incremental_reservation(name);
    reservation
        .try_grow(checked_bytes(
            8192,
            [
                (query.text().len(), 32),
                (fields, 32),
                (schema.fields().len(), 1024),
            ],
            name,
        )?)
        .map_err(|error| df_error(name, error))?;
    let optimizer = Optimizer::with_rules(vec![
        Arc::new(EliminateGroupByConstant::new()),
        Arc::new(PushDownFilter::new()),
        Arc::new(OptimizeProjections::new()),
    ]);
    let config = OptimizerContext::new()
        .without_query_execution_start_time()
        .with_skip_failing_rules(false)
        .with_max_passes(3);
    let normalize = |plan: &LogicalPlan| {
        optimizer
            .optimize(plan.clone(), &config, |_, _| {})
            .map_err(|error| df_error(name, error))
    };
    Ok(Some(Plans {
        raw: normalize(raw)?,
        analyzed: normalize(analyzed)?,
        _reservation: reservation,
    }))
}
