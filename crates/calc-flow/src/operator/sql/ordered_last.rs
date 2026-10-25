//! DF55 ordered grouped `LAST_VALUE`. State is `[value, ordering..., is_set]`.
//!
//! Raw rows always use the native `GroupsAccumulator`. Its strict comparison
//! preserves the first maximal tuple, including ties across updates/recovery.
use super::{
    AggregateFunctionExpr, Expr, ExprSchemable, LogicalPlan, Result, ScalarValue, df_error, unalias,
};
use datafusion::{arrow::datatypes::DataType, logical_expr::expr::AggregateFunctionParams};

pub(super) fn selected(expr: &AggregateFunctionExpr) -> bool {
    expr.fun().name() == "last_value" && !expr.order_bys().is_empty()
}

pub(super) fn parameters(
    params: &AggregateFunctionParams,
    schema: &datafusion::common::DFSchema,
) -> bool {
    !params.distinct
        && params.filter.is_none()
        && matches!(params.args.as_slice(), [value] if column_type(value, schema, value_type))
        && !params.order_by.is_empty()
        && params
            .order_by
            .iter()
            .all(|sort| column_type(&sort.expr, schema, ordering_type))
}

fn column_type(
    expr: &Expr,
    schema: &datafusion::common::DFSchema,
    allowed: impl FnOnce(&DataType) -> bool,
) -> bool {
    matches!(unalias(expr), Expr::Column(_)) && expr.get_type(schema).is_ok_and(|t| allowed(&t))
}

fn value_type(dtype: &DataType) -> bool {
    dtype.is_integer() || matches!(dtype, DataType::Float32 | DataType::Float64)
}

fn ordering_type(dtype: &DataType) -> bool {
    dtype.is_integer() || matches!(dtype, DataType::Timestamp(..))
}

pub(super) fn shape(plan: &LogicalPlan) -> bool {
    let plan = match plan {
        LogicalPlan::Sort(sort) if sort.fetch.is_none() => sort.input.as_ref(),
        _ => plan,
    };
    let LogicalPlan::Projection(projection) = plan else {
        return false;
    };
    if !projection
        .expr
        .iter()
        .all(|expr| matches!(unalias(expr), Expr::Column(_)))
    {
        return false;
    }
    let LogicalPlan::Aggregate(aggregate) = projection.input.as_ref() else {
        return false;
    };
    aggregate_shape(aggregate)
}

fn aggregate_shape(aggregate: &datafusion::logical_expr::Aggregate) -> bool {
    let LogicalPlan::TableScan(scan) = aggregate.input.as_ref() else {
        return false;
    };
    !aggregate.group_expr.is_empty()
        && aggregate
            .group_expr
            .iter()
            .all(|expr| matches!(unalias(expr), Expr::Column(_)))
        && scan.filters.is_empty()
        && scan.fetch.is_none()
        && matches!(aggregate.aggr_expr.as_slice(), [expr] if matches!(unalias(expr), Expr::AggregateFunction(f) if f.func == datafusion::functions_aggregate::first_last::last_value_udaf() && parameters(&f.params, aggregate.input.schema())))
}

pub(super) fn result(state: &[ScalarValue], ignore_nulls: bool, name: &str) -> Result<ScalarValue> {
    if state.len() < 3 {
        return Err(df_error(name, "ordered LAST_VALUE state is incomplete"));
    }
    let Some(ScalarValue::Boolean(Some(is_set))) = state.last() else {
        return Err(df_error(
            name,
            "ordered LAST_VALUE is_set must be a non-null Boolean",
        ));
    };
    let value = &state[0];
    if (!is_set && state[..state.len() - 1].iter().any(|v| !v.is_null()))
        || (*is_set && ignore_nulls && value.is_null())
    {
        return Err(df_error(
            name,
            "ordered LAST_VALUE candidate state is inconsistent",
        ));
    }
    Ok(value.clone())
}

pub(in crate::operator::sql) fn checkpoint_versions(ordered: bool) -> (u32, u32) {
    if ordered { (5, 2) } else { (3, 1) }
}

pub(super) fn supported_shape(plan: &LogicalPlan, expressions: &[Expr]) -> bool {
    !expressions.iter().any(
        |expr| matches!(unalias(expr), Expr::AggregateFunction(f) if f.func.name() == "last_value"),
    ) || shape(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordered_last_rejects_inconsistent_winner_state() {
        let null = ScalarValue::Int64(None);
        let value = ScalarValue::Int64(Some(1));
        for state in [
            vec![],
            vec![value.clone()],
            vec![null.clone(), null.clone(), ScalarValue::Boolean(None)],
            vec![
                value.clone(),
                null.clone(),
                ScalarValue::Boolean(Some(false)),
            ],
            vec![null.clone(), value, ScalarValue::Boolean(Some(false))],
        ] {
            assert!(result(&state, false, "ordered-last").is_err());
        }
        let state = [null.clone(), null, ScalarValue::Boolean(Some(true))];
        assert!(result(&state, true, "ordered-last").is_err());
        assert_eq!(
            result(&state, false, "ordered-last").unwrap(),
            ScalarValue::Int64(None)
        );
    }
}
