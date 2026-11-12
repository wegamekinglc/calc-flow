//! AST authority shared by Rust and Python's explicit DISTINCT declarations.
use crate::{DistinctSpec, EventTime, Result, StateBudget, dataframe::invalid};
use datafusion::sql::{
    parser::Statement as DFStatement,
    planner::IdentNormalizer,
    sqlparser::ast::{
        Distinct, Expr, FunctionArg, FunctionArgExpr, Ident, ObjectName, Query, SelectItem,
        SetExpr, Statement, TableFactor, Value, Visit, Visitor,
    },
};
use serde::Serialize;
use std::ops::ControlFlow;

/// A data-only description of the explicit final DISTINCT dialect.
#[doc(hidden)]
#[derive(Clone, Debug, Serialize)]
pub struct DistinctQuery {
    pub alias: String,
    /// `None` expands a single star against the exact input schema.
    pub columns: Option<Vec<String>>,
    pub spec: DistinctSpec,
}

fn macro_name(name: &ObjectName) -> bool {
    name.0
        .last()
        .and_then(|n| n.as_ident())
        .is_some_and(|n| n.value.to_ascii_lowercase().starts_with("cf_distinct_"))
}
struct Find;
impl Visitor for Find {
    type Break = ();
    fn pre_visit_table_factor(&mut self, table: &TableFactor) -> ControlFlow<()> {
        if matches!(table, TableFactor::Table{name,..}|TableFactor::Function{name,..} if macro_name(name))
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if matches!(expr,Expr::Function(f) if macro_name(&f.name)) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}
fn error(message: impl std::fmt::Display) -> crate::CalcFlowError {
    invalid("sql.distinct", "capability_mismatch", message)
}
pub(crate) fn window_time_column(text: &str, alias: &str, time: &str) -> Result<String> {
    let invalid = || {
        invalid(
            "distinct.spec.history",
            "capability_mismatch",
            "window time must be an unchanged input timestamp or pure rename",
        )
    };
    let columns = super::sql_passthrough_columns(text, alias)?.ok_or_else(invalid)?;
    match columns.get(time) {
        Some(Some(original)) => Ok(original.clone()),
        None if columns.contains_key("*") => Ok(time.into()),
        _ => Err(invalid()),
    }
}
fn query(statement: &DFStatement) -> Option<&Query> {
    let DFStatement::Statement(s) = statement else {
        return None;
    };
    let Statement::Query(q) = s.as_ref() else {
        return None;
    };
    Some(q)
}

/// Recognizes explicit DISTINCT macros; malformed macro shapes never fall back.
///
/// # Errors
/// Returns a clause/literal diagnostic outside the admitted top-level subset.
#[doc(hidden)]
pub fn distinct_query(text: &str, aliases: &[String]) -> Result<Option<DistinctQuery>> {
    let parsed = crate::expression::parse_select_query(text)
        .map_err(|error| invalid("sql.query", "invalid_literal", error))?;
    let statement = parsed.statement();
    let Some(q) = query(&statement) else {
        return Ok(None);
    };
    if !matches!(q.visit(&mut Find), ControlFlow::Break(())) {
        return Ok(None);
    }
    let plan = plan_select(q, aliases)?;
    validate_shape(q)?;
    Ok(Some(plan))
}
fn plan_select(q: &Query, aliases: &[String]) -> Result<DistinctQuery> {
    let select = top_select(q)?;
    let (window, args) = macro_arguments(&select.from[0].relation)?;
    validate_arity(window, args.len())?;
    let alias = string(&args[0])?;
    if aliases != [alias.clone()] {
        return Err(error("exactly the referenced input alias must be supplied"));
    }
    let spec = selected_spec(window, args)?;
    let columns = key_columns(select, &alias)?;
    Ok(DistinctQuery {
        alias,
        columns,
        spec,
    })
}
fn top_select(q: &Query) -> Result<&datafusion::sql::sqlparser::ast::Select> {
    let SetExpr::Select(select) = q.body.as_ref() else {
        return Err(error(
            "one top-level SELECT DISTINCT required; no set operation",
        ));
    };
    if !matches!(select.distinct, Some(Distinct::Distinct)) {
        return Err(error(
            "SELECT DISTINCT required; DISTINCT ON is unsupported",
        ));
    }
    let [from] = select.from.as_slice() else {
        return Err(error("exactly one macro FROM required"));
    };
    if !from.joins.is_empty() {
        return Err(error("JOIN is unsupported"));
    }
    Ok(select)
}
fn macro_arguments(table: &TableFactor) -> Result<(bool, &[FunctionArg])> {
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
        return Err(error("unaliased CF_DISTINCT_BOUNDED FROM required"));
    };
    if !with_hints.is_empty()
        || !partitions.is_empty()
        || !index_hints.is_empty()
        || args.settings.is_some()
    {
        return Err(error(
            "qualified macro names and FROM modifiers are unsupported",
        ));
    }
    Ok((window_macro(name)?, &args.args))
}
fn window_macro(name: &ObjectName) -> Result<bool> {
    if name.0.len() != 1 {
        return Err(error("qualified macro names are unsupported"));
    }
    match name.0[0]
        .as_ident()
        .unwrap()
        .value
        .to_ascii_lowercase()
        .as_str()
    {
        "cf_distinct_bounded" => Ok(false),
        "cf_distinct_window" => Ok(true),
        _ => Err(error("expected CF_DISTINCT_BOUNDED or CF_DISTINCT_WINDOW")),
    }
}
fn validate_arity(window: bool, count: usize) -> Result<()> {
    if !matches!((window, count), (true, 6 | 7) | (false, 3)) {
        return Err(error(
            "expected alias, optional window time/start/end, max_rows, max_bytes and optional window late policy",
        ));
    }
    Ok(())
}
fn selected_spec(window: bool, args: &[FunctionArg]) -> Result<DistinctSpec> {
    if window {
        return window_spec(args);
    }
    Ok(DistinctSpec::bounded_source(sql_budget(args, 1)?))
}
fn sql_budget(args: &[FunctionArg], start: usize) -> Result<StateBudget> {
    StateBudget::new(integer(&args[start])?, integer(&args[start + 1])?)
}
fn window_spec(args: &[FunctionArg]) -> Result<DistinctSpec> {
    DistinctSpec::closed_window(
        &string(&args[1])?,
        EventTime::from_micros(signed(&args[2])?),
        EventTime::from_micros(signed(&args[3])?),
        sql_budget(args, 4)?,
    )?
    .with_late_policy(late_policy(args.get(6))?)
}

fn late_policy(arg: Option<&FunctionArg>) -> Result<crate::LatePolicySpec> {
    use crate::{LateErrorScope, LatePolicySpec};
    let invalid = || {
        invalid(
            "sql.distinct.late_policy",
            "invalid_literal",
            "requires the string literal 'error' or 'drop'",
        )
    };
    match arg
        .map(string)
        .transpose()
        .map_err(|_| invalid())?
        .as_deref()
    {
        None | Some("error") => Ok(LatePolicySpec::Error {
            scope: LateErrorScope::Envelope,
        }),
        Some("drop") => Ok(LatePolicySpec::Drop { metrics_version: 1 }),
        _ => Err(invalid()),
    }
}
fn key_columns(
    select: &datafusion::sql::sqlparser::ast::Select,
    alias: &str,
) -> Result<Option<Vec<String>>> {
    let columns = if star_projection(select) {
        None
    } else {
        Some(
            select
                .projection
                .iter()
                .map(|item| {
                    let SelectItem::UnnamedExpr(expr) = item else {
                        return Err(error(
                            "direct key columns only; no AS, expressions or wildcard modifiers",
                        ));
                    };
                    match expr {
                        Expr::Identifier(name) => Ok(normalized(name)),
                        Expr::CompoundIdentifier(names)
                            if names.len() == 2 && normalized(&names[0]) == alias =>
                        {
                            Ok(normalized(&names[1]))
                        }
                        _ => Err(error(
                            "direct key columns only; no expressions, functions or OVER",
                        )),
                    }
                })
                .collect::<Result<Vec<_>>>()?,
        )
    };
    Ok(columns)
}
fn validate_shape(q: &Query) -> Result<()> {
    let seed = crate::expression::parse_select_query("SELECT x FROM input")?;
    let seed_statement = seed.statement();
    let SetExpr::Select(seed) = query(&seed_statement).unwrap().body.as_ref() else {
        unreachable!()
    };
    let mut clean = q.clone();
    let SetExpr::Select(clean_select) = clean.body.as_mut() else {
        unreachable!()
    };
    clean_select.distinct = None;
    clean_select.projection.clone_from(&seed.projection);
    clean_select.from.clone_from(&seed.from);
    if !crate::expression::simple_query(&clean) {
        return Err(error(
            "no WHERE/GROUP/HAVING/QUALIFY, CTE, subquery, ORDER BY, LIMIT/OFFSET/FETCH or other SELECT modifier",
        ));
    }
    Ok(())
}
fn normalized(name: &Ident) -> String {
    IdentNormalizer::new(true).normalize(name.clone())
}
fn expression(arg: &FunctionArg) -> Result<&Expr> {
    match arg {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => Ok(expr),
        _ => Err(error("positional literal macro arguments required")),
    }
}
fn string(arg: &FunctionArg) -> Result<String> {
    match expression(arg)? {
        Expr::Value(v) if matches!(v.value, Value::SingleQuotedString(_)) => {
            let Value::SingleQuotedString(s) = &v.value else {
                unreachable!()
            };
            Ok(s.clone())
        }
        _ => Err(error("string literal required")),
    }
}
fn integer(arg: &FunctionArg) -> Result<u64> {
    let Expr::Value(v) = expression(arg)? else {
        return Err(error("positive decimal integer literal required"));
    };
    let Value::Number(n, false) = &v.value else {
        return Err(error("positive decimal integer literal required"));
    };
    if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return Err(error("decimal integer literal required"));
    }
    n.parse().map_err(error)
}

fn signed(arg: &FunctionArg) -> Result<i64> {
    use datafusion::sql::sqlparser::ast::UnaryOperator;
    let expr = expression(arg)?;
    let (negative, expr) = match expr {
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => (true, expr.as_ref()),
        _ => (false, expr),
    };
    let Expr::Value(v) = expr else {
        return Err(error("signed decimal integer literal required"));
    };
    let Value::Number(n, false) = &v.value else {
        return Err(error("signed decimal integer literal required"));
    };
    if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return Err(error("signed decimal integer literal required"));
    }
    if negative {
        format!("-{n}").parse().map_err(error)
    } else {
        n.parse().map_err(error)
    }
}

fn star_projection(select: &datafusion::sql::sqlparser::ast::Select) -> bool {
    select.projection.len() == 1 && select.projection[0].to_string() == "*"
}
