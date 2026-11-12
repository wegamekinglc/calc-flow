//! Shared AST boundary for the explicit fixed UTC event-window dialect.

use crate::{AggregateFunction, Result, WindowSpec, dataframe::invalid};
use datafusion::{
    execution::FunctionRegistry,
    sql::{
        parser::Statement as DFStatement,
        planner::IdentNormalizer,
        sqlparser::ast::{
            Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident,
            ObjectName, Query, Select, SelectItem, SelectItemQualifiedWildcardKind, SetExpr,
            Statement, TableAlias, TableFactor, TableWithJoins, Value, Visit, Visitor,
        },
    },
};
use serde::Serialize;
use std::{collections::BTreeMap, ops::ControlFlow, time::Duration};

#[doc(hidden)]
#[derive(Clone, Debug, Serialize)]
pub struct EventWindowQuery {
    pub alias: String,
    pub spec: WindowSpec,
}

fn macro_name(name: &ObjectName) -> bool {
    name.0.last().and_then(|p| p.as_ident()).is_some_and(|n| {
        matches!(
            n.value.to_ascii_lowercase().as_str(),
            "cf_tumble" | "cf_hop"
        )
    })
}
struct FindWindow;
impl Visitor for FindWindow {
    type Break = ();
    fn pre_visit_table_factor(&mut self, factor: &TableFactor) -> ControlFlow<()> {
        if matches!(factor, TableFactor::Table { name, .. } | TableFactor::Function { name, .. } if macro_name(name))
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if matches!(expr, Expr::Function(f) if macro_name(&f.name)) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}
fn error(message: impl std::fmt::Display) -> crate::CalcFlowError {
    invalid("sql.window", "capability_mismatch", message)
}
fn query(statement: &DFStatement) -> Option<&Query> {
    let DFStatement::Statement(statement) = statement else {
        return None;
    };
    let Statement::Query(query) = statement.as_ref() else {
        return None;
    };
    Some(query)
}

#[doc(hidden)]
pub fn event_window_query(text: &str, aliases: &[String]) -> Result<Option<EventWindowQuery>> {
    let parsed = crate::expression::parse_select_query(text)?;
    let statement = parsed.statement();
    let Some(query) = query(&statement) else {
        return Ok(None);
    };
    if !matches!(query.visit(&mut FindWindow), ControlFlow::Break(())) {
        return Ok(None);
    }
    window_select(query, aliases).map(Some)
}

fn window_select(query: &Query, aliases: &[String]) -> Result<EventWindowQuery> {
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Err(error("one top-level SELECT required"));
    };
    let (hopping, args) = window_from(select)?;
    let result = window_spec(args, hopping, aliases)?;
    let names = group_names(select)?;
    validate_shape(query)?;
    validate_projection(select, &names)?;
    let spec = window_fields(result.spec, &names, &select.projection)?;
    Ok(EventWindowQuery { spec, ..result })
}
fn window_fields(
    spec: WindowSpec,
    names: &[String],
    projection: &[SelectItem],
) -> Result<WindowSpec> {
    let grouped = spec.group_by(names[2..].iter().cloned())?;
    projection[names.len()..]
        .iter()
        .try_fold(grouped, |spec, item| {
            let aggregate = aggregate(item)?;
            spec.aggregate(aggregate.function, &aggregate.column, &aggregate.output)
        })
}

fn window_from(select: &Select) -> Result<(bool, &[FunctionArg])> {
    let [from] = select.from.as_slice() else {
        return Err(error("one CF_TUMBLE/CF_HOP FROM required"));
    };
    if !from.joins.is_empty() {
        return Err(error("JOIN is not supported in event windows"));
    }
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
    } = &from.relation
    else {
        return Err(error(
            "expected unaliased CF_TUMBLE/CF_HOP(input, time, size[, slide])",
        ));
    };
    let permitted = [
        name.0.len() == 1,
        macro_name(name),
        with_hints.is_empty(),
        partitions.is_empty(),
        index_hints.is_empty(),
        args.settings.is_none(),
    ];
    if !permitted.into_iter().all(std::convert::identity) {
        return Err(error(
            "window FROM modifiers and qualified names are unsupported",
        ));
    }
    let hopping = name.0[0]
        .as_ident()
        .expect("one name")
        .value
        .eq_ignore_ascii_case("CF_HOP");
    Ok((hopping, &args.args))
}
fn window_spec(
    args: &[FunctionArg],
    hopping: bool,
    aliases: &[String],
) -> Result<EventWindowQuery> {
    window_arity(args.len(), hopping)?;
    let alias = string_arg(&args[0])?;
    window_alias(&alias, aliases)?;
    let spec = window_geometry(&args[1..], hopping)?;
    Ok(EventWindowQuery { alias, spec })
}
fn window_arity(length: usize, hopping: bool) -> Result<()> {
    if length != if hopping { 4 } else { 3 } {
        return Err(invalid(
            "sql.window.geometry",
            "invalid_literal",
            "expected alias, event time, size and (for CF_HOP) slide; epoch origin only, no offset parameter",
        ));
    }
    Ok(())
}
fn window_alias(alias: &str, aliases: &[String]) -> Result<()> {
    if aliases != [alias] {
        return Err(invalid(
            &crate::dataframe::segment("sql.tables", alias),
            "invalid_literal",
            "exactly the referenced input alias must be supplied",
        ));
    }
    Ok(())
}
fn window_geometry(args: &[FunctionArg], hopping: bool) -> Result<WindowSpec> {
    let event_time = string_arg(&args[0])?;
    let size = duration_arg(&args[1], "size_micros")?;
    if hopping {
        WindowSpec::hopping(&event_time, size, duration_arg(&args[2], "slide_micros")?)
    } else {
        WindowSpec::tumbling(&event_time, size)
    }
}

fn group_names(select: &Select) -> Result<Vec<String>> {
    let GroupByExpr::Expressions(groups, modifiers) = &select.group_by else {
        return Err(error("explicit boundary and group columns required"));
    };
    if !modifiers.is_empty() {
        return Err(error("GROUP BY modifiers are unsupported"));
    }
    let names = groups.iter().map(column).collect::<Result<Vec<_>>>()?;
    if names.len() < 2 || names[..2] != ["window_start", "window_end"] {
        return Err(error("GROUP BY must begin with window_start, window_end"));
    }
    Ok(names)
}
fn validate_shape(query: &Query) -> Result<()> {
    let mut shape = query.clone();
    let SetExpr::Select(clean) = shape.body.as_mut() else {
        unreachable!()
    };
    let seed = crate::expression::parse_select_query("SELECT x FROM input")?;
    let seed_statement = seed.statement();
    let SetExpr::Select(seed) = self::query(&seed_statement)
        .expect("SELECT seed")
        .body
        .as_ref()
    else {
        unreachable!()
    };
    clean.from.clone_from(&seed.from);
    clean.projection.clone_from(&seed.projection);
    clean.group_by = seed.group_by.clone();
    if !crate::expression::simple_query(&shape) {
        return Err(error(
            "window SELECT does not allow filtering, CTE, ordering, limits, DISTINCT or other modifiers",
        ));
    }
    Ok(())
}
fn validate_projection(select: &Select, names: &[String]) -> Result<()> {
    if select.projection.len() <= names.len() {
        return Err(error("provide at least one named window aggregate"));
    }
    for (item, name) in select.projection.iter().zip(names) {
        if !matches!(item, SelectItem::UnnamedExpr(expr) if column(expr).is_ok_and(|n| n == *name))
        {
            return Err(error(
                "SELECT must begin with unchanged boundaries and ordered group columns",
            ));
        }
    }
    Ok(())
}
fn aggregate(item: &SelectItem) -> Result<crate::AggregateSpec> {
    let SelectItem::ExprWithAlias {
        expr: Expr::Function(f),
        alias,
    } = item
    else {
        return Err(error("aggregates require a direct column and AS output"));
    };
    let function = aggregate_function(&f.name)?;
    let column = aggregate_column(f)?;
    Ok(crate::AggregateSpec {
        function,
        column,
        output: normalized(alias),
    })
}
fn aggregate_function(name: &ObjectName) -> Result<AggregateFunction> {
    match name.to_string().to_ascii_lowercase().as_str() {
        "count" => Ok(AggregateFunction::Count),
        "sum" => Ok(AggregateFunction::Sum),
        "min" => Ok(AggregateFunction::Min),
        "max" => Ok(AggregateFunction::Max),
        "avg" => Ok(AggregateFunction::Avg),
        _ => Err(error("supported window aggregates: COUNT/SUM/MIN/MAX/AVG")),
    }
}
fn aggregate_column(f: &Function) -> Result<String> {
    let plain = [
        !f.uses_odbc_syntax,
        matches!(f.parameters, FunctionArguments::None),
        f.over.is_none(),
        f.filter.is_none(),
        f.null_treatment.is_none(),
        f.within_group.is_empty(),
    ];
    if !plain.into_iter().all(std::convert::identity) {
        return Err(error("aggregate modifiers and OVER are unsupported"));
    }
    let FunctionArguments::List(arguments) = &f.args else {
        return Err(error("aggregate requires one direct column"));
    };
    let [FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))] = arguments.args.as_slice() else {
        return Err(error("COUNT(*) and aggregate expressions are unsupported"));
    };
    if arguments.duplicate_treatment.is_some() || !arguments.clauses.is_empty() {
        return Err(error(
            "DISTINCT and aggregate argument modifiers are unsupported",
        ));
    }
    column(expr)
}

fn string_arg(arg: &FunctionArg) -> Result<String> {
    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Value(value))) = arg
        && let Value::SingleQuotedString(value) = &value.value
    {
        return Ok(value.clone());
    }
    Err(error("input alias and event time must be string literals"))
}
fn duration_arg(arg: &FunctionArg, field: &str) -> Result<Duration> {
    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Value(value))) = arg
        && let Value::Number(value, false) = &value.value
        && value.bytes().all(|b| b.is_ascii_digit())
        && let Ok(value) = value.parse::<u64>()
        && value > 0
    {
        return Ok(Duration::from_micros(value));
    }
    Err(invalid(
        &format!("sql.window.geometry.{field}"),
        "invalid_literal",
        "expected decimal integer in [1, u64::MAX]",
    ))
}
fn column(expr: &Expr) -> Result<String> {
    match expr {
        Expr::Identifier(name) => Ok(normalized(name)),
        _ => Err(error("expected a direct unqualified column")),
    }
}

#[doc(hidden)]
pub fn sql_passthrough_columns(
    text: &str,
    alias: &str,
) -> Result<Option<BTreeMap<String, Option<String>>>> {
    if !crate::query_analysis::union_arm_proved(text) {
        return Ok(None);
    }
    let parsed = crate::expression::parse_select_query(text)?;
    let statement = parsed.statement();
    let Some(query) = query(&statement) else {
        return Ok(None);
    };
    Ok(columns_for_query(query, text, alias))
}
fn columns_for_query(
    query: &Query,
    text: &str,
    alias: &str,
) -> Option<BTreeMap<String, Option<String>>> {
    if query.with.is_some() {
        return None;
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return None;
    };
    let qualifier = passthrough_relation(select, alias)?;
    let registry = crate::expression::default_function_registry();
    if crate::expression::validate_no_volatile_functions("window", text, registry).is_err()
        || !native_functions(query)
    {
        return None;
    }
    Some(
        select
            .projection
            .iter()
            .filter_map(|item| projected_column(item, qualifier))
            .collect(),
    )
}
fn native_functions(query: &Query) -> bool {
    let registry = crate::expression::default_function_registry();
    let functions = datafusion::sql::sqlparser::ast::visit_expressions(query, |expr| {
        if let Expr::Function(f) = expr
            && (f.name.0.len() != 1
                || registry
                    .udf(&f.name.to_string().to_ascii_lowercase())
                    .is_err())
        {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });
    !matches!(functions, ControlFlow::Break(()))
}

fn normalized(name: &Ident) -> String {
    IdentNormalizer::default().normalize(name.clone())
}
fn identifier_matches(name: &Ident, expected: &Ident) -> bool {
    normalized(name) == normalized(expected)
}
fn passthrough_relation<'a>(select: &'a Select, alias: &str) -> Option<&'a Ident> {
    let [from] = select.from.as_slice() else {
        return None;
    };
    let (name, correlation) = direct_relation(from)?;
    if normalized(name) != alias {
        return None;
    }
    match correlation {
        Some(correlation) if correlation.columns.is_empty() => Some(&correlation.name),
        None => Some(name),
        _ => None,
    }
}
fn direct_relation(from: &TableWithJoins) -> Option<(&Ident, Option<&TableAlias>)> {
    if !from.joins.is_empty() {
        return None;
    }
    let TableFactor::Table {
        name,
        alias,
        args: None,
        ..
    } = &from.relation
    else {
        return None;
    };
    let [part] = name.0.as_slice() else {
        return None;
    };
    Some((part.as_ident()?, alias.as_ref()))
}
fn passthrough_column<'a>(expr: &'a Expr, qualifier: &Ident) -> Option<&'a Ident> {
    match expr {
        Expr::Identifier(name) => Some(name),
        Expr::CompoundIdentifier(parts)
            if parts.len() == 2 && identifier_matches(&parts[0], qualifier) =>
        {
            Some(&parts[1])
        }
        _ => None,
    }
}
fn projected_column(item: &SelectItem, qualifier: &Ident) -> Option<(String, Option<String>)> {
    match item {
        SelectItem::UnnamedExpr(expr) => {
            let name = passthrough_column(expr, qualifier)?;
            Some((normalized(name), Some(normalized(name))))
        }
        SelectItem::ExprWithAlias { expr, alias } => Some((
            normalized(alias),
            passthrough_column(expr, qualifier).map(normalized),
        )),
        _ => plain_wildcard(item, qualifier).then(|| ("*".into(), Some("*".into()))),
    }
}
fn plain_wildcard(item: &SelectItem, qualifier: &Ident) -> bool {
    let plain = datafusion::sql::sqlparser::ast::WildcardAdditionalOptions::default();
    match item {
        SelectItem::Wildcard(options) => options == &plain,
        SelectItem::QualifiedWildcard(
            SelectItemQualifiedWildcardKind::ObjectName(name),
            options,
        ) => qualifier_name(name, qualifier) && options == &plain,
        _ => false,
    }
}
fn qualifier_name(name: &ObjectName, qualifier: &Ident) -> bool {
    name.0.len() == 1
        && name.0[0]
            .as_ident()
            .is_some_and(|n| identifier_matches(n, qualifier))
}
