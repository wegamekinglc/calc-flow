//! The deliberately finite single-alias replacement aggregate language.
use super::{
    DFStatement, Expr, Function, GroupByExpr, Ident, Select, SelectItem, SetExpr, Statement,
    expression, parse_select_query, simple_aggregate_function, simple_query,
};
use crate::ArrowFieldSpec;
use datafusion::{
    common::TableReference,
    sql::sqlparser::ast::{
        FunctionArg, FunctionArgExpr, FunctionArguments, Query, TableFactor, Visit, Visitor,
    },
};
use std::ops::ControlFlow;

pub(crate) fn supported(text: &str, fields: &[ArrowFieldSpec], input_alias: &str) -> bool {
    inspect(text, fields, input_alias).unwrap_or(false)
}
fn inspect(text: &str, fields: &[ArrowFieldSpec], input_alias: &str) -> crate::Result<bool> {
    let parsed = parse_select_query(text)?;
    let DFStatement::Statement(statement) = parsed.statement() else {
        return Ok(false);
    };
    Ok(statement_supported(&statement, fields, input_alias))
}
fn statement_supported(
    statement: &Statement,
    fields: &[ArrowFieldSpec],
    input_alias: &str,
) -> bool {
    let Statement::Query(query) = statement else {
        return false;
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return false;
    };
    let Some(groups) = group_expressions(&select.group_by) else {
        return false;
    };
    binding_supported(select, input_alias)
        && groups.iter().all(|g| group(g, fields))
        && projection(select, groups, fields)
        && normalized_query(query)
}

fn normalized_ident(ident: &Ident) -> String {
    if ident.quote_style.is_some() {
        ident.value.clone()
    } else {
        ident.value.to_ascii_lowercase()
    }
}

fn binding_supported(select: &Select, input_alias: &str) -> bool {
    let [from] = select.from.as_slice() else {
        return false;
    };
    let TableFactor::Table { name, alias, .. } = &from.relation else {
        return false;
    };
    let [part] = name.0.as_slice() else {
        return false;
    };
    let Some(name) = part.as_ident() else {
        return false;
    };
    let TableReference::Bare { table } = TableReference::from(input_alias) else {
        return false;
    };
    if table.is_empty() || normalized_ident(name) != table.as_ref() {
        return false;
    }
    let qualifier = match alias {
        Some(alias) if alias.columns.is_empty() => normalized_ident(&alias.name),
        Some(_) => return false,
        None => normalized_ident(name),
    };
    select.visit(&mut BoundQualifier(&qualifier)).is_continue()
}

struct BoundQualifier<'a>(&'a str);

impl Visitor for BoundQualifier<'_> {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<Self::Break> {
        if let Expr::CompoundIdentifier(parts) = expr
            && (parts.len() != 2 || normalized_ident(&parts[0]) != self.0)
        {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}
fn group_expressions(group_by: &GroupByExpr) -> Option<&[Expr]> {
    match group_by {
        GroupByExpr::Expressions(groups, modifiers) if modifiers.is_empty() => Some(groups),
        _ => None,
    }
}
fn normalized_query(query: &Query) -> bool {
    let mut normalized = query.clone();
    let SetExpr::Select(normal) = normalized.body.as_mut() else {
        unreachable!()
    };
    normal.group_by = GroupByExpr::Expressions(vec![], vec![]);
    normal.projection = vec![SelectItem::UnnamedExpr(Expr::Identifier(Ident::new(
        "__aggregate",
    )))];
    simple_query(&normalized)
}
fn field<'a>(expr: &Expr, fields: &'a [ArrowFieldSpec]) -> Option<&'a ArrowFieldSpec> {
    let name = match expr {
        Expr::Identifier(i) => i,
        Expr::CompoundIdentifier(parts) if parts.len() == 2 => &parts[1],
        _ => return None,
    };
    let normalized = normalized_ident(name);
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
    if !unmodified_function(function) {
        return false;
    }
    let Some(operand) = function_operand(function) else {
        return false;
    };
    let count = function.name.to_string().eq_ignore_ascii_case("count");
    operand_supported(operand, count, fields)
}
fn unmodified_function(function: &Function) -> bool {
    simple_aggregate_function(function)
        && !function.uses_odbc_syntax
        && function.null_treatment.is_none()
        && matches!(function.parameters, FunctionArguments::None)
}
fn function_operand(function: &Function) -> Option<&FunctionArgExpr> {
    let FunctionArguments::List(args) = &function.args else {
        return None;
    };
    if args.duplicate_treatment.is_some() || !args.clauses.is_empty() || args.args.len() != 1 {
        return None;
    }
    let FunctionArg::Unnamed(operand) = &args.args[0] else {
        return None;
    };
    Some(operand)
}
fn operand_supported(operand: &FunctionArgExpr, count: bool, fields: &[ArrowFieldSpec]) -> bool {
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
pub(crate) fn rejection_reason(text: &str, input_alias: Option<&str>) -> &'static str {
    let Ok(parsed) = parse_select_query(text) else {
        return "SQL semantics are unproved";
    };
    let DFStatement::Statement(statement) = parsed.statement() else {
        return "one SELECT is required";
    };
    let Statement::Query(query) = statement.as_ref() else {
        return "one SELECT is required";
    };
    query_rejection(query, input_alias)
}
fn query_rejection(query: &Query, input_alias: Option<&str>) -> &'static str {
    if query.with.is_some() {
        return "CTE is outside the replacement allowlist";
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return "set operations are outside the replacement allowlist";
    };
    relation_rejection(select)
        .or_else(|| select_modifier_rejection(select))
        .or_else(|| query_modifier_rejection(query))
        .or_else(|| select.projection.iter().find_map(item_rejection))
        .or_else(|| {
            (!input_alias.is_some_and(|alias| binding_supported(select, alias))).then_some(
                "input relation and column qualifiers must bind the declared bare input alias",
            )
        })
        .unwrap_or(
            "operand/group key type or query modifiers are outside the replacement allowlist",
        )
}
fn relation_rejection(select: &Select) -> Option<&'static str> {
    (select.from.len() != 1 || select.from.iter().any(|from| !from.joins.is_empty()))
        .then_some("JOIN or multiple relations are outside the replacement allowlist")
}
fn select_modifier_rejection(select: &Select) -> Option<&'static str> {
    [
        (
            select.distinct.is_some(),
            "SELECT DISTINCT is outside the replacement allowlist",
        ),
        (
            select.selection.is_some(),
            "WHERE must be a separate row-local node",
        ),
        (
            select.having.is_some(),
            "HAVING is outside the replacement allowlist",
        ),
    ]
    .into_iter()
    .find_map(|(active, reason)| active.then_some(reason))
}
fn query_modifier_rejection(query: &Query) -> Option<&'static str> {
    [
        (
            query.order_by.is_some(),
            "ORDER BY is outside the replacement allowlist",
        ),
        (
            query.limit_clause.is_some() || query.fetch.is_some(),
            "LIMIT/OFFSET/FETCH is outside the replacement allowlist",
        ),
    ]
    .into_iter()
    .find_map(|(active, reason)| active.then_some(reason))
}
fn item_rejection(item: &SelectItem) -> Option<&'static str> {
    match expression(item) {
        Some(Expr::Function(function)) => function_rejection(function),
        Some(Expr::Identifier(_) | Expr::CompoundIdentifier(_)) => None,
        _ => Some("projection requires direct group keys or named reducers"),
    }
}
fn function_rejection(function: &Function) -> Option<&'static str> {
    if !unmodified_function(function) {
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
    (!direct_argument(&args.args[0])).then_some("one direct column operand is required")
}
fn direct_argument(arg: &FunctionArg) -> bool {
    matches!(
        arg,
        FunctionArg::Unnamed(
            FunctionArgExpr::Expr(Expr::Identifier(_) | Expr::CompoundIdentifier(_))
                | FunctionArgExpr::Wildcard
        )
    )
}
