use datafusion::sql::{
    parser::Statement as DFStatement,
    sqlparser::ast::{
        Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident,
        ObjectName, Select, SelectItem, SetExpr, Statement, TableFactor,
    },
};

use crate::expression::ValidatedQuery;

pub(super) fn grouped_sum_candidate(query: &ValidatedQuery, alias: &str) -> bool {
    candidate(query, alias).unwrap_or(false)
}

fn candidate(query: &ValidatedQuery, alias: &str) -> Option<bool> {
    let DFStatement::Statement(statement) = query.statement() else {
        return None;
    };
    let Statement::Query(query) = statement.as_ref() else {
        return None;
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return None;
    };
    let name = single_table(select, alias)?;
    let groups = grouped_keys(select)?;
    let keys = groups
        .iter()
        .map(|group| column(group, alias))
        .collect::<Option<Vec<_>>>()?;
    let projection = select
        .projection
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let group_text = groups
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    if query.to_string() != format!("SELECT {projection} FROM {name} GROUP BY {group_text}") {
        return None;
    }
    projected_sums(select, &keys, alias)
}

fn projected_sums(select: &Select, keys: &[String], alias: &str) -> Option<bool> {
    let mut has_sum = false;
    for item in &select.projection {
        has_sum |= projected_sum(item, keys, alias)?;
    }
    Some(has_sum)
}

fn single_table<'a>(select: &'a Select, alias: &str) -> Option<&'a ObjectName> {
    let [table] = select.from.as_slice() else {
        return None;
    };
    let TableFactor::Table { name, .. } = &table.relation else {
        return None;
    };
    let [table_name] = name.0.as_slice() else {
        return None;
    };
    (identifier(table_name.as_ident()?) == alias).then_some(name)
}

fn grouped_keys(select: &Select) -> Option<&[Expr]> {
    let GroupByExpr::Expressions(groups, modifiers) = &select.group_by else {
        return None;
    };
    (!groups.is_empty() && modifiers.is_empty()).then_some(groups)
}

fn projected_sum(item: &SelectItem, keys: &[String], alias: &str) -> Option<bool> {
    let (SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. }) = item else {
        return None;
    };
    match expr {
        Expr::Function(function) if ordinary_sum(function, alias) => Some(true),
        _ => keys.contains(&column(expr, alias)?).then_some(false),
    }
}

fn ordinary_sum(function: &Function, alias: &str) -> bool {
    builtin_sum_name(&function.name)
        && ordinary_modifiers(function)
        && ordinary_arguments(&function.args, alias)
}

fn builtin_sum_name(name: &ObjectName) -> bool {
    let [name] = name.0.as_slice() else {
        return false;
    };
    let Some(name) = name.as_ident() else {
        return false;
    };
    name.quote_style.is_none() && name.value.eq_ignore_ascii_case("sum")
}

fn ordinary_modifiers(function: &Function) -> bool {
    [
        !function.uses_odbc_syntax,
        function.over.is_none(),
        function.filter.is_none(),
        function.null_treatment.is_none(),
        function.within_group.is_empty(),
        function.parameters == FunctionArguments::None,
    ]
    .into_iter()
    .all(std::convert::identity)
}

fn ordinary_arguments(arguments: &FunctionArguments, alias: &str) -> bool {
    let FunctionArguments::List(arguments) = arguments else {
        return false;
    };
    let [FunctionArg::Unnamed(FunctionArgExpr::Expr(argument))] = arguments.args.as_slice() else {
        return false;
    };
    arguments.duplicate_treatment.is_none()
        && arguments.clauses.is_empty()
        && column(argument, alias).is_some()
}

fn column(expression: &Expr, alias: &str) -> Option<String> {
    match expression {
        Expr::Identifier(name) => Some(identifier(name)),
        Expr::CompoundIdentifier(parts) => {
            let [qualifier, name] = parts.as_slice() else {
                return None;
            };
            (identifier(qualifier) == alias).then(|| identifier(name))
        }
        _ => None,
    }
}

fn identifier(identifier: &Ident) -> String {
    if identifier.quote_style.is_some() {
        identifier.value.clone()
    } else {
        identifier.value.to_lowercase()
    }
}
