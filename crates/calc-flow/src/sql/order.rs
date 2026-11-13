//! AST authority shared by Rust and Python's explicit ordering declarations.
use crate::{EventTime, Result, SortSpec, StateBudget, dataframe::invalid};
use datafusion::sql::{
    parser::Statement as DFStatement,
    planner::IdentNormalizer,
    sqlparser::ast::{
        Expr, FunctionArg, FunctionArgExpr, Ident, ObjectName, Query, SetExpr, Statement,
        TableFactor, Value, Visit, Visitor,
    },
};
use serde::Serialize;
use std::ops::ControlFlow;

/// A data-only description of the explicit final ordering dialect.
#[doc(hidden)]
#[derive(Clone, Debug, Serialize)]
pub struct OrderQuery {
    pub alias: String,
    /// Comparator keys, in precedence order.
    pub by: Vec<crate::SortKey>,
    pub fetch: Option<u64>,
    pub spec: SortSpec,
}

fn macro_name(name: &ObjectName) -> bool {
    name.0
        .last()
        .and_then(|n| n.as_ident())
        .is_some_and(|n| n.value.to_ascii_lowercase().starts_with("cf_order_"))
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
    invalid("sql.order", "capability_mismatch", message)
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

/// Recognizes explicit ordering macros; malformed macro shapes never fall back.
///
/// # Errors
/// Returns a clause/literal diagnostic outside the admitted top-level subset.
#[doc(hidden)]
pub fn order_query(text: &str, aliases: &[String]) -> Result<Option<OrderQuery>> {
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
fn plan_select(q: &Query, aliases: &[String]) -> Result<OrderQuery> {
    let select = top_select(q)?;
    let (window, args) = macro_arguments(&select.from[0].relation)?;
    validate_arity(window, args.len())?;
    let alias = string(&args[0])?;
    if aliases != [alias.clone()] {
        return Err(error("exactly the referenced input alias must be supplied"));
    }
    let (offset, fetch) = limits(q)?;
    let spec = selected_spec(window, args)?.with_offset(offset)?;
    let by = key_columns(q, &alias)?;
    Ok(OrderQuery {
        alias,
        by,
        fetch,
        spec,
    })
}
fn top_select(q: &Query) -> Result<&datafusion::sql::sqlparser::ast::Select> {
    let SetExpr::Select(select) = q.body.as_ref() else {
        return Err(error("one top-level SELECT * required; no set operation"));
    };
    if select.distinct.is_some() || !star_projection(select) {
        return Err(error(
            "one SELECT * required; DISTINCT, aliases and projection expressions unsupported",
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
        return Err(error("unaliased CF_ORDER_BOUNDED FROM required"));
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
        "cf_order_bounded" => Ok(false),
        "cf_order_window" => Ok(true),
        _ => Err(error("expected CF_ORDER_BOUNDED or CF_ORDER_WINDOW")),
    }
}
fn validate_arity(window: bool, count: usize) -> Result<()> {
    if !matches!((window, count), (true, 6..=8) | (false, 3 | 4)) {
        return Err(error(
            "expected alias, optional window time/start/end, max_rows, max_bytes and optional window late policy",
        ));
    }
    Ok(())
}
fn selected_spec(window: bool, args: &[FunctionArg]) -> Result<SortSpec> {
    let spec = if window {
        window_spec(args)?
    } else {
        SortSpec::bounded_source(sql_budget(args, 1)?)?
    };
    let sequence = if window { args.get(7) } else { args.get(3) };
    match sequence {
        None => Ok(spec),
        Some(arg) => spec.with_sequence_by(sequence_columns(arg)?),
    }
}
fn sequence_columns(arg: &FunctionArg) -> Result<Vec<String>> {
    let Expr::Array(array) = expression(arg)? else {
        return Err(error(
            "nonempty ARRAY of sequence column string literals required",
        ));
    };
    array
        .elem
        .iter()
        .map(|expr| string(&FunctionArg::Unnamed(FunctionArgExpr::Expr(expr.clone()))))
        .collect()
}
fn limits(q: &Query) -> Result<(u64, Option<u64>)> {
    use datafusion::sql::sqlparser::ast::LimitClause;
    match &q.limit_clause {
        None => Ok((0, None)),
        Some(LimitClause::LimitOffset {
            limit,
            offset,
            limit_by,
        }) if limit_by.is_empty() => {
            let literal =
                |expr: &Expr| integer(&FunctionArg::Unnamed(FunctionArgExpr::Expr(expr.clone())));
            Ok((
                offset
                    .as_ref()
                    .map(|o| literal(&o.value))
                    .transpose()?
                    .unwrap_or(0),
                limit.as_ref().map(literal).transpose()?,
            ))
        }
        _ => Err(error(
            "literal LIMIT/OFFSET only; LIMIT BY and comma form unsupported",
        )),
    }
}
fn sql_budget(args: &[FunctionArg], start: usize) -> Result<StateBudget> {
    StateBudget::new(integer(&args[start])?, integer(&args[start + 1])?)
}
fn window_spec(args: &[FunctionArg]) -> Result<SortSpec> {
    SortSpec::closed_window(
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
            "sql.order.late_policy",
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
fn key_columns(q: &Query, alias: &str) -> Result<Vec<crate::SortKey>> {
    use crate::{NullPlacement, SortDirection, SortKey};
    use datafusion::sql::sqlparser::ast::OrderByKind;
    let order = q
        .order_by
        .as_ref()
        .ok_or_else(|| error("nonempty ORDER BY direct columns required"))?;
    let OrderByKind::Expressions(keys) = &order.kind else {
        return Err(error("ORDER BY ALL unsupported"));
    };
    if keys.is_empty() || order.interpolate.is_some() {
        return Err(error("nonempty ORDER BY without INTERPOLATE required"));
    }
    keys.iter()
        .map(|key| {
            if key.with_fill.is_some() {
                return Err(error("WITH FILL unsupported"));
            }
            let column = match &key.expr {
                Expr::Identifier(name) => normalized(name),
                Expr::CompoundIdentifier(names)
                    if names.len() == 2 && normalized(&names[0]) == alias =>
                {
                    normalized(&names[1])
                }
                _ => {
                    return Err(error(
                        "ORDER BY direct columns only; no ordinal, expression or collation",
                    ));
                }
            };
            let asc = key.options.asc.unwrap_or(true);
            SortKey::new(
                &column,
                if asc {
                    SortDirection::Ascending
                } else {
                    SortDirection::Descending
                },
                if key.options.nulls_first.unwrap_or(!asc) {
                    NullPlacement::First
                } else {
                    NullPlacement::Last
                },
            )
        })
        .collect()
}
fn validate_shape(q: &Query) -> Result<()> {
    let seed = crate::expression::parse_select_query("SELECT x FROM input")?;
    let seed_statement = seed.statement();
    let SetExpr::Select(seed) = query(&seed_statement).unwrap().body.as_ref() else {
        unreachable!()
    };
    let mut clean = q.clone();
    clean.order_by = None;
    clean.limit_clause = None;
    let SetExpr::Select(clean_select) = clean.body.as_mut() else {
        unreachable!()
    };
    clean_select.distinct = None;
    clean_select.projection.clone_from(&seed.projection);
    clean_select.from.clone_from(&seed.from);
    if !crate::expression::simple_query(&clean) {
        return Err(error(
            "no WHERE/GROUP/HAVING/QUALIFY, CTE, subquery, FETCH or other SELECT modifier",
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
