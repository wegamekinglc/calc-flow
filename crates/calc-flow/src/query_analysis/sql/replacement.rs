//! The deliberately finite single-alias replacement aggregate language.
use super::{
    DFStatement, Expr, Function, GroupByExpr, Ident, Select, SelectItem, SetExpr, Statement,
    expression, parse_select_query, simple_aggregate_function, simple_query,
};
use crate::ArrowFieldSpec;
use datafusion::sql::sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};

pub(crate) fn supported(text: &str, fields: &[ArrowFieldSpec]) -> bool {
    inspect(text, fields).unwrap_or(false)
}
fn inspect(text: &str, fields: &[ArrowFieldSpec]) -> crate::Result<bool> {
    let parsed = parse_select_query(text)?;
    let DFStatement::Statement(statement) = parsed.statement() else {
        return Ok(false);
    };
    let Statement::Query(query) = statement.as_ref() else {
        return Ok(false);
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(false);
    };
    let GroupByExpr::Expressions(groups, modifiers) = &select.group_by else {
        return Ok(false);
    };
    if !modifiers.is_empty() || !groups.iter().all(|g| group(g, fields)) {
        return Ok(false);
    }
    if !projection(select, groups, fields) {
        return Ok(false);
    }
    let mut normalized = query.clone();
    let SetExpr::Select(normal) = normalized.body.as_mut() else {
        unreachable!()
    };
    normal.group_by = GroupByExpr::Expressions(vec![], vec![]);
    normal.projection = vec![SelectItem::UnnamedExpr(Expr::Identifier(Ident::new(
        "__aggregate",
    )))];
    Ok(simple_query(&normalized))
}
fn field<'a>(expr: &Expr, fields: &'a [ArrowFieldSpec]) -> Option<&'a ArrowFieldSpec> {
    let name = match expr {
        Expr::Identifier(i) => i,
        Expr::CompoundIdentifier(parts) if parts.len() == 2 => &parts[1],
        _ => return None,
    };
    let normalized = if name.quote_style.is_some() {
        name.value.clone()
    } else {
        name.value.to_ascii_lowercase()
    };
    fields.iter().find(|f| f.name == normalized)
}
fn numeric(dtype: &str) -> bool {
    matches!(
        dtype,
        "int8"
            | "int16"
            | "int32"
            | "int64"
            | "uint8"
            | "uint16"
            | "uint32"
            | "uint64"
            | "float32"
            | "float64"
    )
}
fn group(expr: &Expr, fields: &[ArrowFieldSpec]) -> bool {
    field(expr, fields).is_some_and(|f| {
        matches!(f.data_type.as_str(), "bool" | "string" | "large_string")
            || numeric(&f.data_type) && !f.data_type.starts_with("float")
    })
}
fn projection(select: &Select, groups: &[Expr], fields: &[ArrowFieldSpec]) -> bool {
    select
        .projection
        .iter()
        .any(|i| matches!(expression(i), Some(Expr::Function(_))))
        && select.projection.iter().all(|i| match expression(i) {
            Some(Expr::Function(f)) => function(f, fields),
            Some(expr) => groups.contains(expr) && field(expr, fields).is_some(),
            None => false,
        })
}
fn function(function: &Function, fields: &[ArrowFieldSpec]) -> bool {
    if !simple_aggregate_function(function)
        || function.uses_odbc_syntax
        || function.null_treatment.is_some()
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return false;
    }
    let FunctionArguments::List(args) = &function.args else {
        return false;
    };
    if args.duplicate_treatment.is_some() || !args.clauses.is_empty() || args.args.len() != 1 {
        return false;
    }
    let FunctionArg::Unnamed(operand) = &args.args[0] else {
        return false;
    };
    let count = function.name.to_string().eq_ignore_ascii_case("count");
    match operand {
        FunctionArgExpr::Wildcard => count,
        FunctionArgExpr::Expr(expr) => field(expr, fields)
            .is_some_and(|f| (count && portable(&f.data_type)) || numeric(&f.data_type)),
        _ => false,
    }
}

fn portable(dtype: &str) -> bool {
    numeric(dtype)
        || matches!(
            dtype,
            "bool"
                | "string"
                | "large_string"
                | "date32"
                | "date64"
                | "time32[s]"
                | "time64[us]"
                | "timestamp[ms]"
                | "timestamp[us]"
                | "timestamp[us, UTC]"
        )
}

/// Explains the first structural restriction; schema planning still owns type errors.
pub(crate) fn rejection_reason(text: &str) -> &'static str {
    let Ok(parsed) = parse_select_query(text) else {
        return "SQL semantics are unproved";
    };
    let DFStatement::Statement(statement) = parsed.statement() else {
        return "one SELECT is required";
    };
    let Statement::Query(query) = statement.as_ref() else {
        return "one SELECT is required";
    };
    if query.with.is_some() {
        return "CTE is outside the replacement allowlist";
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return "set operations are outside the replacement allowlist";
    };
    if select.from.len() != 1 || select.from.iter().any(|from| !from.joins.is_empty()) {
        return "JOIN or multiple relations are outside the replacement allowlist";
    }
    if select.distinct.is_some() {
        return "SELECT DISTINCT is outside the replacement allowlist";
    }
    if select.selection.is_some() {
        return "WHERE must be a separate row-local node";
    }
    if select.having.is_some() {
        return "HAVING is outside the replacement allowlist";
    }
    if query.order_by.is_some() {
        return "ORDER BY is outside the replacement allowlist";
    }
    if query.limit_clause.is_some() || query.fetch.is_some() {
        return "LIMIT/OFFSET/FETCH is outside the replacement allowlist";
    }
    for item in &select.projection {
        match expression(item) {
            Some(Expr::Function(function)) => {
                if let Some(reason) = function_rejection(function) {
                    return reason;
                }
            }
            Some(Expr::Identifier(_) | Expr::CompoundIdentifier(_)) => {}
            _ => return "projection requires direct group keys or named reducers",
        }
    }
    "operand/group key type or query modifiers are outside the replacement allowlist"
}
fn function_rejection(function: &Function) -> Option<&'static str> {
    if !simple_aggregate_function(function)
        || function.null_treatment.is_some()
        || function.uses_odbc_syntax
        || !matches!(function.parameters, FunctionArguments::None)
    {
        return Some("only unmodified SUM/COUNT/MIN/MAX/AVG reducers are admitted");
    }
    let FunctionArguments::List(args) = &function.args else {
        return Some("one direct column operand is required");
    };
    if args.duplicate_treatment.is_some() {
        return Some("DISTINCT/ALL is outside the replacement allowlist");
    }
    if args.args.len() != 1 || !args.clauses.is_empty() {
        return Some("one direct column without function clauses is required");
    }
    match &args.args[0] {
        FunctionArg::Unnamed(
            FunctionArgExpr::Expr(Expr::Identifier(_) | Expr::CompoundIdentifier(_))
            | FunctionArgExpr::Wildcard,
        ) => None,
        _ => Some("one direct column operand is required"),
    }
}
