use std::collections::BTreeMap;

use datafusion::{
    arrow::datatypes::{DataType, TimeUnit},
    sql::{
        parser::Statement as DFStatement,
        sqlparser::ast::{
            BinaryOperator, DateTimeField, Expr, FunctionArguments, GroupByExpr, JoinConstraint,
            JoinOperator, Query, Select, SelectItem, SetExpr, Statement, TableFactor, Value,
            WindowType,
        },
    },
};

use crate::{Batch, expression::ValidatedQuery};

use super::{AsofSpec, BandSpec, JoinSpec, NativeInput};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ColumnRef<'a> {
    qualifier: Option<&'a str>,
    name: &'a str,
}

pub(super) fn plan(query: &ValidatedQuery, tables: &BTreeMap<String, Batch>) -> Option<JoinSpec> {
    let statement = query.statement();
    let DFStatement::Statement(statement) = statement else {
        return None;
    };
    let Statement::Query(query) = statement.as_ref() else {
        return None;
    };
    asof(query, tables)
        .map(JoinSpec::Asof)
        .or_else(|| band(query, tables).map(JoinSpec::Band))
}

fn query_select(query: &Query) -> Option<&Select> {
    if query.order_by.is_some() || query.limit_clause.is_some() || query.fetch.is_some() {
        return None;
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return None;
    };
    (select.distinct.is_none()
        && select.top.is_none()
        && select.into.is_none()
        && select.selection.is_none()
        && matches!(&select.group_by, GroupByExpr::Expressions(groups, _) if groups.is_empty())
        && select.having.is_none()
        && select.qualify.is_none()
        && select.lateral_views.is_empty()
        && select.cluster_by.is_empty()
        && select.distribute_by.is_empty()
        && select.sort_by.is_empty())
    .then_some(select)
}

fn named_table(factor: &TableFactor) -> Option<(String, String)> {
    let TableFactor::Table {
        name,
        alias,
        args,
        with_hints,
        version,
        partitions,
        sample,
        index_hints,
        ..
    } = factor
    else {
        return None;
    };
    if args.is_some()
        || !with_hints.is_empty()
        || version.is_some()
        || !partitions.is_empty()
        || sample.is_some()
        || !index_hints.is_empty()
    {
        return None;
    }
    let table = name.0.last()?.as_ident()?.value.clone();
    let qualifier = alias
        .as_ref()
        .map_or_else(|| table.clone(), |alias| alias.name.value.clone());
    Some((table, qualifier))
}

fn derived_table(factor: &TableFactor) -> Option<(&Query, String)> {
    let TableFactor::Derived {
        lateral,
        subquery,
        alias,
        sample,
    } = factor
    else {
        return None;
    };
    if *lateral || sample.is_some() {
        return None;
    }
    Some((subquery, alias.as_ref()?.name.value.clone()))
}

fn column(expr: &Expr) -> Option<ColumnRef<'_>> {
    match expr {
        Expr::Identifier(identifier) => Some(ColumnRef {
            qualifier: None,
            name: &identifier.value,
        }),
        Expr::CompoundIdentifier(identifiers) if identifiers.len() == 2 => Some(ColumnRef {
            qualifier: Some(&identifiers[0].value),
            name: &identifiers[1].value,
        }),
        _ => None,
    }
}

fn expression(item: &SelectItem) -> Option<&Expr> {
    match item {
        SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => Some(expr),
        _ => None,
    }
}

fn item_name(item: &SelectItem) -> Option<&str> {
    match item {
        SelectItem::UnnamedExpr(expr) => Some(column(expr)?.name),
        SelectItem::ExprWithAlias { alias, .. } => Some(&alias.value),
        _ => None,
    }
}

fn named_expr<'a>(projection: &'a [SelectItem], name: &str) -> Option<&'a Expr> {
    projection
        .iter()
        .find(|item| item_name(item) == Some(name))
        .and_then(expression)
}

fn conjunction(expr: &Expr) -> Vec<&Expr> {
    match expr {
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => [conjunction(left), conjunction(right)].concat(),
        Expr::Nested(nested) => conjunction(nested),
        other => vec![other],
    }
}

fn equality(expr: &Expr) -> Option<(&Expr, &Expr)> {
    let Expr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = expr
    else {
        return None;
    };
    Some((left, right))
}

fn multiplication(expr: &Expr) -> Option<(&Expr, &Expr)> {
    let Expr::BinaryOp {
        left,
        op: BinaryOperator::Multiply,
        right,
    } = expr
    else {
        return None;
    };
    Some((left, right))
}

fn addition(expr: &Expr) -> Option<(&Expr, &Expr)> {
    let Expr::BinaryOp {
        left,
        op: BinaryOperator::Plus,
        right,
    } = expr
    else {
        return None;
    };
    Some((left, right))
}

fn qualified(column: ColumnRef<'_>, qualifier: &str, name: &str) -> bool {
    column.name == name && column.qualifier == Some(qualifier)
}

fn column_equality(
    left: &Expr,
    right: &Expr,
    left_qualifier: &str,
    right_qualifier: &str,
    name: &str,
) -> bool {
    let Some((left, right)) = equality(left).or_else(|| equality(right)) else {
        return false;
    };
    let (Some(left), Some(right)) = (column(left), column(right)) else {
        return false;
    };
    (qualified(left, left_qualifier, name) && qualified(right, right_qualifier, name))
        || (qualified(right, left_qualifier, name) && qualified(left, right_qualifier, name))
}

fn column_pair_equality(
    expr: &Expr,
    left_qualifier: &str,
    right_qualifier: &str,
    left_name: &str,
    right_name: &str,
) -> bool {
    let Some((left, right)) = equality(expr) else {
        return false;
    };
    let (Some(left), Some(right)) = (column(left), column(right)) else {
        return false;
    };
    (qualified(left, left_qualifier, left_name) && qualified(right, right_qualifier, right_name))
        || (qualified(right, left_qualifier, left_name)
            && qualified(left, right_qualifier, right_name))
}

fn integer_one(expr: &Expr) -> bool {
    matches!(expr, Expr::Value(value) if matches!(&value.value, Value::Number(number, false) if number == "1"))
}

fn input(tables: &BTreeMap<String, Batch>, name: &str) -> Option<NativeInput> {
    NativeInput::new(tables.get(name)?)
}

fn column_index(input: &NativeInput, name: &str, expected: &DataType) -> Option<usize> {
    let index = input.schema.index_of(name).ok()?;
    (input.schema.field(index).data_type() == expected).then_some(index)
}

fn timestamp_index(input: &NativeInput, name: &str) -> Option<(usize, TimeUnit)> {
    let index = input.schema.index_of(name).ok()?;
    let DataType::Timestamp(unit, _) = input.schema.field(index).data_type() else {
        return None;
    };
    Some((index, *unit))
}

fn row_number_definition(select: &Select) -> Option<(&str, &str, &str, &str)> {
    let (row_item, function) = select.projection.iter().find_map(|item| {
        let SelectItem::ExprWithAlias { expr, .. } = item else {
            return None;
        };
        let Expr::Function(function) = expr else {
            return None;
        };
        function
            .name
            .to_string()
            .eq_ignore_ascii_case("row_number")
            .then_some((item, function))
    })?;
    if !matches!(&function.args, FunctionArguments::List(arguments) if arguments.args.is_empty()) {
        return None;
    }
    let Some(WindowType::WindowSpec(window)) = &function.over else {
        return None;
    };
    if window.window_name.is_some()
        || window.window_frame.is_some()
        || window.partition_by.len() != 2
    {
        return None;
    }
    let key = column(&window.partition_by[0])?.name;
    let event_time = column(&window.partition_by[1])?.name;
    let [order] = window.order_by.as_slice() else {
        return None;
    };
    if order.options.asc != Some(false) {
        return None;
    }
    let sequence = column(&order.expr)?.name;
    let position = item_name(row_item)?;
    Some((key, event_time, sequence, position))
}

fn asof(query: &Query, tables: &BTreeMap<String, Batch>) -> Option<AsofSpec> {
    if query.with.is_some() {
        return None;
    }
    let outer = query_select(query)?;
    let [from] = outer.from.as_slice() else {
        return None;
    };
    let [join] = from.joins.as_slice() else {
        return None;
    };
    let JoinOperator::Left(JoinConstraint::On(predicate)) = &join.join_operator else {
        return None;
    };
    let (left_table, left_alias) = named_table(&from.relation)?;
    let (right_query, right_alias) = derived_table(&join.relation)?;
    let right = query_select(right_query)?;
    let (right_table, _) = named_table(&right.from.first()?.relation)?;
    if !right.from.first()?.joins.is_empty() {
        return None;
    }
    let (key, event_time, sequence, position) = row_number_definition(right)?;
    let terms = conjunction(predicate);
    if terms.len() != 3
        || !terms
            .iter()
            .any(|term| column_equality(term, term, &left_alias, &right_alias, key))
        || !terms
            .iter()
            .any(|term| column_equality(term, term, &left_alias, &right_alias, event_time))
        || !terms.iter().any(|term| {
            equality(term).is_some_and(|(left, right)| {
                (column(left).is_some_and(|column| qualified(column, &right_alias, position))
                    && integer_one(right))
                    || (column(right)
                        .is_some_and(|column| qualified(column, &right_alias, position))
                        && integer_one(left))
            })
        })
    {
        return None;
    }
    let [left_output, right_output] = outer.projection.as_slice() else {
        return None;
    };
    let left_output = column(expression(left_output)?)?;
    let right_output = column(expression(right_output)?)?;
    if left_output.qualifier != Some(left_alias.as_str())
        || right_output.qualifier != Some(right_alias.as_str())
    {
        return None;
    }
    for name in [key, event_time, right_output.name] {
        if column(named_expr(&right.projection, name)?)?.name != name {
            return None;
        }
    }
    let left = input(tables, &left_table)?;
    let right = input(tables, &right_table)?;
    let (left_time, left_unit) = timestamp_index(&left, event_time)?;
    let (right_time, right_unit) = timestamp_index(&right, event_time)?;
    if left_unit != right_unit {
        return None;
    }
    Some(AsofSpec {
        left_key: column_index(&left, key, &DataType::Utf8)?,
        right_key: column_index(&right, key, &DataType::Utf8)?,
        left_time,
        right_time,
        right_sequence: column_index(&right, sequence, &DataType::UInt64)?,
        left_output: left.schema.index_of(left_output.name).ok()?,
        right_output: right.schema.index_of(right_output.name).ok()?,
        left,
        right,
    })
}

fn interval_seconds(expr: &Expr) -> Option<i64> {
    let Expr::Interval(interval) = expr else {
        return None;
    };
    if interval.leading_precision.is_some()
        || interval.last_field.is_some()
        || interval.fractional_seconds_precision.is_some()
        || !matches!(interval.leading_field, None | Some(DateTimeField::Second))
    {
        return None;
    }
    let Expr::Value(value) = interval.value.as_ref() else {
        return None;
    };
    match &value.value {
        Value::SingleQuotedString(text) => {
            let mut parts = text.split_whitespace();
            let amount = parts.next()?.parse::<i64>().ok()?;
            match parts.next() {
                Some(unit)
                    if unit.eq_ignore_ascii_case("second")
                        || unit.eq_ignore_ascii_case("seconds") =>
                {
                    if parts.next().is_some() {
                        return None;
                    }
                }
                Some(_) => return None,
                None => {}
            }
            Some(amount)
        }
        Value::Number(number, false) => number.parse::<i64>().ok(),
        _ => None,
    }
}

fn offset_seconds(query: &Query) -> Option<(String, Vec<i64>)> {
    let with = query.with.as_ref()?;
    if with.recursive || with.cte_tables.len() != 1 {
        return None;
    }
    let cte = with.cte_tables.first()?;
    if cte.from.is_some() || cte.materialized.is_some() || cte.alias.columns.len() != 1 {
        return None;
    }
    let SetExpr::Values(values) = cte.query.body.as_ref() else {
        return None;
    };
    let offsets = values
        .rows
        .iter()
        .map(|row| {
            let [expr] = row.content.as_slice() else {
                return None;
            };
            interval_seconds(expr)
        })
        .collect::<Option<Vec<_>>>()?;
    if offsets.is_empty() {
        return None;
    }
    Some((cte.alias.name.value.clone(), offsets))
}

fn band_projection<'a>(
    outer: &'a Select,
    probe: &'a Select,
    left_alias: &str,
    probe_alias: &str,
    right_alias: &str,
) -> Option<(&'a Expr, &'a Expr, &'a Expr, ColumnRef<'a>, ColumnRef<'a>)> {
    let left_sequence = named_expr(&probe.projection, "left_sequence")?;
    let left_key = named_expr(&probe.projection, "left_symbol")?;
    let left_price = named_expr(&probe.projection, "left_price")?;
    for expr in [left_sequence, left_key, left_price] {
        if column(expr)?.qualifier != Some(left_alias) {
            return None;
        }
    }
    let [outer_left_sequence, outer_right_sequence, outer_value] = outer.projection.as_slice()
    else {
        return None;
    };
    let outer_left_sequence = column(expression(outer_left_sequence)?)?;
    let right_sequence = column(expression(outer_right_sequence)?)?;
    let (left_value, right_value) = multiplication(expression(outer_value)?)?;
    let left_value = column(left_value)?;
    let right_value = column(right_value)?;
    if outer_left_sequence.qualifier != Some(probe_alias)
        || outer_left_sequence.name != "left_sequence"
        || right_sequence.qualifier != Some(right_alias)
        || left_value.qualifier != Some(probe_alias)
        || left_value.name != "left_price"
        || right_value.qualifier != Some(right_alias)
    {
        return None;
    }
    Some((
        left_sequence,
        left_key,
        left_price,
        right_sequence,
        right_value,
    ))
}

fn band(query: &Query, tables: &BTreeMap<String, Batch>) -> Option<BandSpec> {
    let (offsets_name, offsets) = offset_seconds(query)?;
    let outer = query_select(query)?;
    let [from] = outer.from.as_slice() else {
        return None;
    };
    let [join] = from.joins.as_slice() else {
        return None;
    };
    let (JoinOperator::Join(JoinConstraint::On(predicate))
    | JoinOperator::Inner(JoinConstraint::On(predicate))) = &join.join_operator
    else {
        return None;
    };
    let (right_table, right_alias) = named_table(&from.relation)?;
    let (probe_query, probe_alias) = derived_table(&join.relation)?;
    let probe = query_select(probe_query)?;
    let [probe_from] = probe.from.as_slice() else {
        return None;
    };
    let [cross] = probe_from.joins.as_slice() else {
        return None;
    };
    if !matches!(
        cross.join_operator,
        JoinOperator::CrossJoin(JoinConstraint::None)
    ) {
        return None;
    }
    let (left_table, left_alias) = named_table(&probe_from.relation)?;
    let (actual_offsets, _) = named_table(&cross.relation)?;
    if actual_offsets != offsets_name {
        return None;
    }
    let (left_sequence, left_key, left_price, right_sequence, right_value) =
        band_projection(outer, probe, &left_alias, &probe_alias, &right_alias)?;
    let match_time = named_expr(&probe.projection, "match_time")?;
    let (event_time, offset_column) = addition(match_time)?;
    let event_time = column(event_time)?;
    let offset_column = column(offset_column)?;
    if event_time.qualifier != Some(left_alias.as_str())
        || offset_column.qualifier != Some(offsets_name.as_str())
        || offset_column.name != "event_offset"
    {
        return None;
    }
    let terms = conjunction(predicate);
    if terms.len() != 2
        || !terms.iter().any(|term| {
            column_pair_equality(term, &probe_alias, &right_alias, "left_symbol", "symbol")
        })
        || !terms.iter().any(|term| {
            column_pair_equality(term, &probe_alias, &right_alias, "match_time", "event_time")
        })
    {
        return None;
    }
    let left = input(tables, &left_table)?;
    let right = input(tables, &right_table)?;
    let (left_time, unit) = timestamp_index(&left, event_time.name)?;
    let (right_time, right_unit) = timestamp_index(&right, "event_time")?;
    if unit != right_unit {
        return None;
    }
    let scale = match unit {
        TimeUnit::Second => 1_i64,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    };
    let offsets = offsets
        .into_iter()
        .map(|offset| offset.checked_mul(scale))
        .collect::<Option<Vec<_>>>()?;
    Some(BandSpec {
        left_key: column_index(&left, column(left_key)?.name, &DataType::Utf8)?,
        right_key: column_index(&right, "symbol", &DataType::Utf8)?,
        left_time,
        right_time,
        left_sequence: column_index(&left, column(left_sequence)?.name, &DataType::UInt64)?,
        right_sequence: column_index(&right, right_sequence.name, &DataType::UInt64)?,
        left_price: column_index(&left, column(left_price)?.name, &DataType::Float64)?,
        right_price: column_index(&right, right_value.name, &DataType::Float64)?,
        offsets: offsets.into(),
        left,
        right,
    })
}
