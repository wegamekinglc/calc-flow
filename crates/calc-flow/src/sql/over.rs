//! Explicit analytic dialect lowered to the same data-only native declaration.
use super::order::{
    expression, integer, key_column, normalized, query, sequence_columns, signed, sort_key, string,
    validate_shape,
};
use crate::{
    AnalyticExpr, EventTime, OverOrder, OverSpec, OverWindow, Result, RowsFrame, StateBudget,
    dataframe::invalid,
};
use datafusion::sql::sqlparser::ast::{
    Expr, Function, FunctionArg, FunctionArguments, ObjectName, Query, Select, SelectItem, SetExpr,
    TableFactor, Visit, Visitor, WindowFrame, WindowFrameBound, WindowFrameUnits, WindowSpec,
    WindowType,
};
use serde::Serialize;
use std::ops::ControlFlow;

/// Native AST description consumed by both first-class frontends.
#[doc(hidden)]
#[derive(Clone, Debug, Serialize)]
pub struct OverQuery {
    pub alias: String,
    pub spec: OverSpec,
    pub expressions: Vec<AnalyticExpr>,
}
fn error(path: &str, message: impl std::fmt::Display) -> crate::CalcFlowError {
    invalid(&format!("sql.over{path}"), "capability_mismatch", message)
}
fn is_macro(name: &ObjectName) -> bool {
    name.0
        .last()
        .and_then(|n| n.as_ident())
        .is_some_and(|n| n.value.to_ascii_lowercase().starts_with("cf_over_"))
}
struct Find;
impl Visitor for Find {
    type Break = ();
    fn pre_visit_table_factor(&mut self, table: &TableFactor) -> ControlFlow<()> {
        if matches!(table,TableFactor::Table {name,..}|TableFactor::Function {name,..} if is_macro(name))
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if matches!(expr,Expr::Function(f) if is_macro(&f.name)) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}
/// Recognizes explicit macros without falling back on malformed shapes.
/// # Errors
/// Rejects functions, clauses, frames, literals and mixed orders outside the subset.
#[doc(hidden)]
pub fn over_query(text: &str, aliases: &[String]) -> Result<Option<OverQuery>> {
    let parsed = crate::expression::parse_select_query(text)
        .map_err(|e| invalid("sql.query", "invalid_literal", e))?;
    let statement = parsed.statement();
    let Some(q) = query(&statement) else {
        return Ok(None);
    };
    if !matches!(q.visit(&mut Find), ControlFlow::Break(())) {
        return Ok(None);
    }
    plan(q, aliases).map(Some).map_err(remap)
}
fn remap(error: crate::CalcFlowError) -> crate::CalcFlowError {
    match error {
        crate::CalcFlowError::InvalidArgument { field, message } => {
            crate::CalcFlowError::InvalidArgument {
                field: if field.starts_with("sql.over") {
                    field
                } else if field.starts_with("sql.order") {
                    field.replacen("sql.order", "sql.over", 1)
                } else if field.starts_with("state_budget.") {
                    format!("sql.over.spec.{field}")
                } else {
                    format!("sql.{}", field.replacen("sort", "over", 1))
                },
                message,
            }
        }
        other => other,
    }
}
fn plan(q: &Query, aliases: &[String]) -> Result<OverQuery> {
    let select = analytic_select(q)?;
    let invocation = invocation(select, aliases)?;
    let (order, expressions) = projection(select, &invocation.alias, &invocation.sequence)?;
    validate_shape(q)?;
    let spec = selected_spec(invocation.window, invocation.args, order)?;
    Ok(OverQuery {
        alias: invocation.alias,
        spec,
        expressions,
    })
}
fn analytic_select(q: &Query) -> Result<&Select> {
    let SetExpr::Select(select) = q.body.as_ref() else {
        return Err(error("", "one top-level SELECT required"));
    };
    if select.distinct.is_some()
        || select.projection.len() < 2
        || select.projection[0].to_string() != "*"
        || q.order_by.is_some()
        || q.limit_clause.is_some()
    {
        return Err(error(
            "",
            "requires SELECT *, analytic AS name with no DISTINCT/outer order/limit",
        ));
    }
    Ok(select)
}
fn macro_source(select: &Select) -> Result<(bool, &[FunctionArg])> {
    let [from] = select.from.as_slice() else {
        return Err(error("", "one macro FROM required"));
    };
    if !from.joins.is_empty() {
        return Err(error("", "JOIN unsupported"));
    }
    let (window, args) = macro_args(&from.relation)?;
    if !matches!((window, args.len()), (false, 4) | (true, 7 | 8)) {
        return Err(error(
            ".spec",
            "expected bounded arity 4 or window arity 7/8 with mandatory sequence ARRAY",
        ));
    }
    Ok((window, args))
}
struct Invocation<'a> {
    window: bool,
    args: &'a [FunctionArg],
    alias: String,
    sequence: Vec<String>,
}
fn invocation<'a>(select: &'a Select, aliases: &[String]) -> Result<Invocation<'a>> {
    let (window, args) = macro_source(select)?;
    let alias = string(&args[0])?;
    if aliases != [alias.clone()] {
        return Err(error("", "exactly the referenced alias required"));
    }
    let sequence = sequence_columns(&args[if window { 6 } else { 3 }])?;
    Ok(Invocation {
        window,
        args,
        alias,
        sequence,
    })
}
fn projection(
    select: &Select,
    alias: &str,
    sequence: &[String],
) -> Result<(OverOrder, Vec<AnalyticExpr>)> {
    let mut order = None;
    let mut expressions = Vec::with_capacity(select.projection.len() - 1);
    for (index, item) in select.projection.iter().skip(1).enumerate() {
        let path = format!(".expressions[{index}]");
        let SelectItem::ExprWithAlias {
            expr: Expr::Function(f),
            alias: name,
        } = item
        else {
            return Err(error(
                &path,
                "one analytic function with explicit alias required",
            ));
        };
        let (selected, expr) = analytic(f, &normalized(name), alias, sequence, &path)?;
        if order.as_ref().is_some_and(|o| o != &selected) {
            return Err(error(
                &format!("{path}.order"),
                "all expressions must share partition/order",
            ));
        }
        order = Some(selected);
        expressions.push(expr);
    }
    Ok((order.expect("nonempty projection"), expressions))
}
fn macro_args(table: &TableFactor) -> Result<(bool, &[FunctionArg])> {
    let TableFactor::Table {
        name,
        alias: None,
        args: Some(args),
        with_hints,
        version: None,
        with_ordinality: false,
        partitions,
        json_path: None,
        sample: None,
        index_hints,
    } = table
    else {
        return Err(error("", "unaliased CF_OVER_BOUNDED/WINDOW FROM required"));
    };
    if name.0.len() != 1
        || !with_hints.is_empty()
        || !partitions.is_empty()
        || !index_hints.is_empty()
        || args.settings.is_some()
    {
        return Err(error("", "qualified macros and FROM modifiers unsupported"));
    }
    Ok((macro_window(name)?, &args.args))
}
fn macro_window(name: &ObjectName) -> Result<bool> {
    match name.0[0]
        .as_ident()
        .unwrap()
        .value
        .to_ascii_lowercase()
        .as_str()
    {
        "cf_over_bounded" => Ok(false),
        "cf_over_window" => Ok(true),
        _ => Err(error("", "expected CF_OVER_BOUNDED or CF_OVER_WINDOW")),
    }
}
fn selected_spec(window: bool, args: &[FunctionArg], order: OverOrder) -> Result<OverSpec> {
    let budget = state_budget(window, args)?;
    if !window {
        return OverSpec::bounded_source(order, budget);
    }
    let spec = OverSpec::closed_window(history_window(args)?, order, budget)?;
    spec.with_late_policy(late_policy(args)?)
}
fn state_budget(window: bool, args: &[FunctionArg]) -> Result<StateBudget> {
    let offset = if window { 4 } else { 1 };
    StateBudget::new(integer(&args[offset])?, integer(&args[offset + 1])?)
}
fn history_window(args: &[FunctionArg]) -> Result<OverWindow> {
    OverWindow::new(
        &string(&args[1])?,
        EventTime::from_micros(signed(&args[2])?),
        EventTime::from_micros(signed(&args[3])?),
    )
}
fn late_policy(args: &[FunctionArg]) -> Result<crate::LatePolicySpec> {
    match args.get(7).map(string).transpose()?.as_deref() {
        None | Some("error") => Ok(crate::LatePolicySpec::Error {
            scope: crate::LateErrorScope::Envelope,
        }),
        Some("drop") => Ok(crate::LatePolicySpec::Drop { metrics_version: 1 }),
        _ => Err(error(".spec.late_policy", "requires 'error' or 'drop'")),
    }
}
fn analytic(
    f: &Function,
    name: &str,
    alias: &str,
    sequence: &[String],
    path: &str,
) -> Result<(OverOrder, AnalyticExpr)> {
    validate_function(f, path)?;
    let args = positional_arguments(f, path)?;
    let Some(WindowType::WindowSpec(window)) = &f.over else {
        return Err(error(
            &format!("{path}.order"),
            "explicit OVER partition/order required",
        ));
    };
    let order = window_order(window, alias, sequence, path)?;
    let fname = f.name.0[0].as_ident().unwrap().value.to_ascii_lowercase();
    let expr = analytic_value(&fname, args, window, name, alias, path)?;
    Ok((order, expr))
}
fn validate_function(f: &Function, path: &str) -> Result<()> {
    if f.name.0.len() != 1
        || f.uses_odbc_syntax
        || f.parameters != FunctionArguments::None
        || f.filter.is_some()
        || f.null_treatment.is_some()
        || !f.within_group.is_empty()
    {
        return Err(error(
            path,
            "function modifiers, parameters and NULL-treatment unsupported",
        ));
    }
    Ok(())
}
fn positional_arguments<'a>(f: &'a Function, path: &str) -> Result<&'a [FunctionArg]> {
    let FunctionArguments::List(args) = &f.args else {
        return Err(error(path, "plain positional arguments required"));
    };
    if args.duplicate_treatment.is_some() || !args.clauses.is_empty() {
        return Err(error(path, "DISTINCT/ALL/argument clauses unsupported"));
    }
    Ok(&args.args)
}
fn analytic_value(
    function: &str,
    args: &[FunctionArg],
    window: &WindowSpec,
    name: &str,
    alias: &str,
    path: &str,
) -> Result<AnalyticExpr> {
    match (function, args) {
        ("row_number", []) => {
            without_frame(window, path)?;
            AnalyticExpr::row_number(name)
        }
        ("lag", [column, offset]) => lag_expression(column, offset, window, name, alias, path),
        ("sum", [column]) => sum_expression(column, window, name, alias, path),
        _ => Err(unsupported_function(path)),
    }
}
fn unsupported_function(path: &str) -> crate::CalcFlowError {
    error(
        path,
        "requires ROW_NUMBER(), LAG(column,1) without frame, or SUM(column) with explicit supported ROWS frame",
    )
}
fn without_frame(window: &WindowSpec, path: &str) -> Result<()> {
    if window.window_frame.is_some() {
        Err(unsupported_function(path))
    } else {
        Ok(())
    }
}
fn lag_expression(
    column: &FunctionArg,
    offset: &FunctionArg,
    window: &WindowSpec,
    name: &str,
    alias: &str,
    path: &str,
) -> Result<AnalyticExpr> {
    without_frame(window, path)?;
    if integer(offset)? != 1 {
        return Err(unsupported_function(path));
    }
    AnalyticExpr::lag(&key_column(expression(column)?, alias)?, name)
}
fn sum_expression(
    column: &FunctionArg,
    window: &WindowSpec,
    name: &str,
    alias: &str,
    path: &str,
) -> Result<AnalyticExpr> {
    AnalyticExpr::sum(
        &key_column(expression(column)?, alias)?,
        frame(window.window_frame.as_ref(), path)?,
        name,
    )
}
fn window_order(
    window: &WindowSpec,
    alias: &str,
    sequence: &[String],
    path: &str,
) -> Result<OverOrder> {
    if window.window_name.is_some() {
        return Err(error(&format!("{path}.order"), "named windows unsupported"));
    }
    let partition = window
        .partition_by
        .iter()
        .map(|e| key_column(e, alias))
        .collect::<Result<Vec<_>>>()?;
    let keys = window
        .order_by
        .iter()
        .map(|k| sort_key(k, alias))
        .collect::<Result<Vec<_>>>()?;
    OverOrder::new(
        partition,
        primary_keys(keys, sequence, path)?,
        sequence.to_vec(),
    )
}
fn primary_keys(
    mut keys: Vec<crate::SortKey>,
    sequence: &[String],
    path: &str,
) -> Result<Vec<crate::SortKey>> {
    if sequence.is_empty() || keys.len() <= sequence.len() {
        return Err(error(
            &format!("{path}.order"),
            "primary order and sequence suffix required",
        ));
    }
    let suffix = keys.split_off(keys.len() - sequence.len());
    if suffix.iter().zip(sequence).any(|(k, s)| {
        k.column() != s
            || k.direction() != crate::SortDirection::Ascending
            || k.nulls() != crate::NullPlacement::Last
    }) {
        return Err(error(
            &format!("{path}.order"),
            "ORDER must end with sequence ARRAY in ASC NULLS LAST order",
        ));
    }
    Ok(keys)
}
fn frame(frame: Option<&WindowFrame>, path: &str) -> Result<RowsFrame> {
    let fail = || {
        error(
            &format!("{path}.frame"),
            "SUM requires explicit ROWS BETWEEN UNBOUNDED/N PRECEDING AND CURRENT ROW",
        )
    };
    let frame = frame.ok_or_else(fail)?;
    if frame.units != WindowFrameUnits::Rows
        || frame.end_bound != Some(WindowFrameBound::CurrentRow)
    {
        return Err(fail());
    }
    match &frame.start_bound {
        WindowFrameBound::Preceding(None) => Ok(RowsFrame::cumulative()),
        WindowFrameBound::Preceding(Some(n)) => {
            RowsFrame::preceding(integer(&FunctionArg::Unnamed(
                datafusion::sql::sqlparser::ast::FunctionArgExpr::Expr(*n.clone()),
            ))?)
        }
        _ => Err(fail()),
    }
}
