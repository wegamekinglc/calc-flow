use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::{Aggregate, Projection};
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
    let aliases = alias_chain(raw);
    let computed_keys = super::shape(raw).is_some_and(|(_, aggregate)| {
        aggregate
            .group_expr
            .iter()
            .any(|expression| !matches!(expression, Expr::Column(_)))
    });
    if !aliases && !computed_keys {
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
        let plan = if aliases {
            inline_aliases(plan.clone()).map_err(|error| df_error(name, error))?
        } else {
            plan.clone()
        };
        let normalized = optimizer
            .optimize(plan, &config, |_, _| {})
            .map_err(|error| df_error(name, error))?;
        if !aliases {
            return Ok(normalized);
        }
        let normalized = inline_input_columns(normalized).map_err(|error| df_error(name, error))?;
        optimizer
            .optimize(normalized, &config, |_, _| {})
            .map_err(|error| df_error(name, error))
    };
    Ok(Some(Plans {
        raw: normalize(raw)?,
        analyzed: normalize(analyzed)?,
        _reservation: reservation,
    }))
}

fn inline_input_columns(plan: LogicalPlan) -> datafusion::error::Result<LogicalPlan> {
    plan.transform_up(|plan| {
        let LogicalPlan::Aggregate(aggregate) = plan else {
            return Ok(Transformed::no(plan));
        };
        let LogicalPlan::Projection(projection) = aggregate.input.as_ref() else {
            return Ok(Transformed::no(LogicalPlan::Aggregate(aggregate)));
        };
        let columns = projection
            .expr
            .iter()
            .map(|expression| match super::unalias(expression) {
                Expr::Column(column) => Some(column.clone()),
                _ => None,
            })
            .collect::<Option<Vec<_>>>();
        let Some(columns) = columns else {
            return Ok(Transformed::no(LogicalPlan::Aggregate(aggregate)));
        };
        let mapping = projection
            .schema
            .columns()
            .into_iter()
            .zip(columns)
            .collect::<Vec<_>>();
        let replace = |expression: &Expr| {
            expression
                .clone()
                .transform_up(|expression| {
                    if let Expr::Column(column) = &expression
                        && let Some((_, source)) =
                            mapping.iter().find(|(output, _)| output == column)
                    {
                        return Ok(Transformed::yes(Expr::Column(source.clone())));
                    }
                    Ok(Transformed::no(expression))
                })
                .map(|transformed| transformed.data)
        };
        let input = Arc::clone(&projection.input);
        let grouped = Aggregate::try_new(
            input,
            aggregate
                .group_expr
                .iter()
                .map(replace)
                .collect::<datafusion::error::Result<_>>()?,
            aggregate
                .aggr_expr
                .iter()
                .map(replace)
                .collect::<datafusion::error::Result<_>>()?,
        )?;
        let expressions = grouped
            .schema
            .columns()
            .into_iter()
            .enumerate()
            .map(|(index, column)| {
                let (qualifier, field) = aggregate.schema.qualified_field(index);
                Expr::Column(column).alias_qualified(qualifier.cloned(), field.name())
            })
            .collect();
        Projection::try_new_with_schema(
            expressions,
            Arc::new(LogicalPlan::Aggregate(grouped)),
            aggregate.schema,
        )
        .map(LogicalPlan::Projection)
        .map(Transformed::yes)
    })
    .map(|transformed| transformed.data)
}

fn alias_chain(plan: &LogicalPlan) -> bool {
    let (mut aliases, mut aggregates, mut scans, mut nodes) = (0, 0, 0, 0);
    let mut supported = true;
    let visited = plan.apply(|plan| {
        nodes += 1;
        supported &= nodes <= 64;
        match plan {
            LogicalPlan::SubqueryAlias(_) => aliases += 1,
            LogicalPlan::Aggregate(_) => aggregates += 1,
            LogicalPlan::TableScan(_) => scans += 1,
            LogicalPlan::Projection(_)
            | LogicalPlan::Filter(_)
            | LogicalPlan::Sort(_)
            | LogicalPlan::Limit(_) => {}
            _ => supported = false,
        }
        Ok(if supported {
            TreeNodeRecursion::Continue
        } else {
            TreeNodeRecursion::Stop
        })
    });
    visited.is_ok() && supported && aliases != 0 && aggregates == 1 && scans == 1
}

fn inline_aliases(plan: LogicalPlan) -> datafusion::error::Result<LogicalPlan> {
    plan.transform_up(|plan| {
        let LogicalPlan::SubqueryAlias(alias) = plan else {
            return Ok(Transformed::no(plan));
        };
        let expressions = alias
            .input
            .schema()
            .columns()
            .into_iter()
            .zip(alias.schema.fields())
            .map(|(column, field)| {
                Expr::Column(column).alias_qualified(Some(alias.alias.clone()), field.name())
            })
            .collect();
        Projection::try_new_with_schema(expressions, alias.input, alias.schema)
            .map(LogicalPlan::Projection)
            .map(Transformed::yes)
    })
    .map(|transformed| transformed.data)
}
