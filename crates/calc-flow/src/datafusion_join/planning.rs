use std::{collections::BTreeMap, sync::Arc};

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

use super::{AsofSpec, BandSpec, JoinSpec, NativeInput, execution};

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

fn plain_select_form(select: &Select) -> bool {
    select.distinct.is_none()
        && select.top.is_none()
        && select.into.is_none()
        && select.selection.is_none()
        && matches!(&select.group_by, GroupByExpr::Expressions(groups, _) if groups.is_empty())
        && select.having.is_none()
}

fn plain_select_clauses(select: &Select) -> bool {
    select.qualify.is_none()
        && select.lateral_views.is_empty()
        && select.cluster_by.is_empty()
        && select.distribute_by.is_empty()
        && select.sort_by.is_empty()
}

fn query_select(query: &Query) -> Option<&Select> {
    if query.order_by.is_some() || query.limit_clause.is_some() || query.fetch.is_some() {
        return None;
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return None;
    };
    (plain_select_form(select) && plain_select_clauses(select)).then_some(select)
}

fn plain_table_factor(factor: &TableFactor) -> bool {
    let TableFactor::Table {
        args,
        with_hints,
        version,
        partitions,
        sample,
        index_hints,
        ..
    } = factor
    else {
        return false;
    };
    if args.is_some() || !with_hints.is_empty() || version.is_some() {
        return false;
    }
    partitions.is_empty() && sample.is_none() && index_hints.is_empty()
}

fn named_table(factor: &TableFactor) -> Option<(String, String)> {
    if !plain_table_factor(factor) {
        return None;
    }
    let TableFactor::Table { name, alias, .. } = factor else {
        return None;
    };
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

fn row_number_parts(
    select: &Select,
) -> Option<(&SelectItem, &datafusion::sql::sqlparser::ast::WindowSpec)> {
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
    Some((row_item, window))
}

fn row_number_sequence(window: &datafusion::sql::sqlparser::ast::WindowSpec) -> Option<&str> {
    let [order] = window.order_by.as_slice() else {
        return None;
    };
    if order.options.asc != Some(false) || order.options.nulls_first == Some(false) {
        return None;
    }
    Some(column(&order.expr)?.name)
}

fn row_number_window(
    window: &datafusion::sql::sqlparser::ast::WindowSpec,
) -> Option<(&str, &str, &str)> {
    if window.window_name.is_some()
        || window.window_frame.is_some()
        || window.partition_by.len() != 2
    {
        return None;
    }
    let key = column(&window.partition_by[0])?.name;
    let event_time = column(&window.partition_by[1])?.name;
    Some((key, event_time, row_number_sequence(window)?))
}

fn row_number_definition(select: &Select) -> Option<(&str, &str, &str, &str)> {
    let (row_item, window) = row_number_parts(select)?;
    let (key, event_time, sequence) = row_number_window(window)?;
    Some((key, event_time, sequence, item_name(row_item)?))
}

struct AsofShape<'a> {
    left_table: String,
    right_table: String,
    key: &'a str,
    event_time: &'a str,
    sequence: &'a str,
    left_output: ColumnRef<'a>,
    right_output: ColumnRef<'a>,
}

fn asof_join(outer: &Select) -> Option<(&Expr, String, String, &Query, String)> {
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
    Some((predicate, left_table, left_alias, right_query, right_alias))
}

fn asof_right(query: &Query) -> Option<(&Select, String)> {
    let right = query_select(query)?;
    let [right_from] = right.from.as_slice() else {
        return None;
    };
    let (right_table, _) = named_table(&right_from.relation)?;
    if !right_from.joins.is_empty() {
        return None;
    }
    Some((right, right_table))
}

fn position_one(term: &Expr, right_alias: &str, position: &str) -> bool {
    let Some((left, right)) = equality(term) else {
        return false;
    };
    let left_position = column(left).is_some_and(|column| qualified(column, right_alias, position));
    let right_position =
        column(right).is_some_and(|column| qualified(column, right_alias, position));
    (left_position && integer_one(right)) || (right_position && integer_one(left))
}

fn asof_predicate(
    predicate: &Expr,
    left_alias: &str,
    right_alias: &str,
    key: &str,
    event_time: &str,
    position: &str,
) -> bool {
    let terms = conjunction(predicate);
    terms.len() == 3
        && terms
            .iter()
            .any(|term| column_equality(term, term, left_alias, right_alias, key))
        && terms
            .iter()
            .any(|term| column_equality(term, term, left_alias, right_alias, event_time))
        && terms
            .iter()
            .any(|term| position_one(term, right_alias, position))
}

fn asof_outputs<'a>(
    outer: &'a Select,
    left_alias: &str,
    right_alias: &str,
) -> Option<(ColumnRef<'a>, ColumnRef<'a>)> {
    let [left_output, right_output] = outer.projection.as_slice() else {
        return None;
    };
    let left_output = column(expression(left_output)?)?;
    let right_output = column(expression(right_output)?)?;
    if left_output.qualifier != Some(left_alias) || right_output.qualifier != Some(right_alias) {
        return None;
    }
    Some((left_output, right_output))
}

fn right_projection_plain(right: &Select, names: &[&str]) -> bool {
    for name in names {
        let Some(expr) = named_expr(&right.projection, name) else {
            return false;
        };
        let Some(column) = column(expr) else {
            return false;
        };
        if column.name != *name {
            return false;
        }
    }
    true
}

fn asof_output_shape<'a>(
    outer: &'a Select,
    right: &Select,
    left_alias: &str,
    right_alias: &str,
    key: &str,
    event_time: &str,
) -> Option<(ColumnRef<'a>, ColumnRef<'a>)> {
    let outputs = asof_outputs(outer, left_alias, right_alias)?;
    right_projection_plain(right, &[key, event_time, outputs.1.name]).then_some(outputs)
}

fn asof_shape(query: &Query) -> Option<AsofShape<'_>> {
    if query.with.is_some() {
        return None;
    }
    let outer = query_select(query)?;
    let (predicate, left_table, left_alias, right_query, right_alias) = asof_join(outer)?;
    let (right, right_table) = asof_right(right_query)?;
    let (key, event_time, sequence, position) = row_number_definition(right)?;
    if !asof_predicate(
        predicate,
        &left_alias,
        &right_alias,
        key,
        event_time,
        position,
    ) {
        return None;
    }
    let (left_output, right_output) =
        asof_output_shape(outer, right, &left_alias, &right_alias, key, event_time)?;
    Some(AsofShape {
        left_table,
        right_table,
        key,
        event_time,
        sequence,
        left_output,
        right_output,
    })
}

fn asof_input_times(
    left: &NativeInput,
    right: &NativeInput,
    event_time: &str,
) -> Option<(usize, usize)> {
    let (left_time, left_unit) = timestamp_index(left, event_time)?;
    let (right_time, right_unit) = timestamp_index(right, event_time)?;
    (left_unit == right_unit).then_some((left_time, right_time))
}

fn asof_output_indices(
    left: &NativeInput,
    right: &NativeInput,
    shape: &AsofShape<'_>,
) -> Option<(usize, usize)> {
    Some((
        left.schema.index_of(shape.left_output.name).ok()?,
        right.schema.index_of(shape.right_output.name).ok()?,
    ))
}

fn build_asof_spec(tables: &BTreeMap<String, Batch>, shape: &AsofShape<'_>) -> Option<AsofSpec> {
    let left = input(tables, &shape.left_table)?;
    let right = input(tables, &shape.right_table)?;
    let (left_time, right_time) = asof_input_times(&left, &right, shape.event_time)?;
    let (left_output, right_output) = asof_output_indices(&left, &right, shape)?;
    Some(AsofSpec {
        left_key: column_index(&left, shape.key, &DataType::Utf8)?,
        right_key: column_index(&right, shape.key, &DataType::Utf8)?,
        left_time,
        right_time,
        right_sequence: column_index(&right, shape.sequence, &DataType::UInt64)?,
        left_output,
        right_output,
        left,
        right,
    })
}

fn asof(query: &Query, tables: &BTreeMap<String, Batch>) -> Option<AsofSpec> {
    let shape = asof_shape(query)?;
    build_asof_spec(tables, &shape)
}

fn interval_text_seconds(text: &str) -> Option<i64> {
    let mut parts = text.split_whitespace();
    let amount = parts.next()?.parse::<i64>().ok()?;
    match parts.next() {
        Some(unit)
            if unit.eq_ignore_ascii_case("second") || unit.eq_ignore_ascii_case("seconds") =>
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
        Value::SingleQuotedString(text) => interval_text_seconds(text),
        Value::Number(number, false) => number.parse::<i64>().ok(),
        _ => None,
    }
}

fn plain_offset_cte(cte: &datafusion::sql::sqlparser::ast::Cte) -> bool {
    cte.from.is_none()
        && cte.materialized.is_none()
        && cte.alias.columns.len() == 1
        && cte.query.order_by.is_none()
        && cte.query.limit_clause.is_none()
        && cte.query.fetch.is_none()
}

fn offset_cte(query: &Query) -> Option<&datafusion::sql::sqlparser::ast::Cte> {
    let with = query.with.as_ref()?;
    if with.recursive || with.cte_tables.len() != 1 {
        return None;
    }
    let cte = with.cte_tables.first()?;
    plain_offset_cte(cte).then_some(cte)
}

fn offset_seconds(query: &Query) -> Option<(String, Vec<i64>)> {
    let cte = offset_cte(query)?;
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

fn probe_projection<'a>(probe: &'a Select, left_alias: &str) -> Option<[&'a Expr; 3]> {
    let columns = [
        named_expr(&probe.projection, "left_sequence")?,
        named_expr(&probe.projection, "left_symbol")?,
        named_expr(&probe.projection, "left_price")?,
    ];
    for expr in columns {
        if column(expr)?.qualifier != Some(left_alias) {
            return None;
        }
    }
    Some(columns)
}

fn outer_value_columns(item: &SelectItem) -> Option<(ColumnRef<'_>, ColumnRef<'_>)> {
    let (left_value, right_value) = multiplication(expression(item)?)?;
    Some((column(left_value)?, column(right_value)?))
}

fn outer_projection_values(
    outer: &Select,
) -> Option<(ColumnRef<'_>, ColumnRef<'_>, ColumnRef<'_>, ColumnRef<'_>)> {
    let [outer_left_sequence, outer_right_sequence, outer_value] = outer.projection.as_slice()
    else {
        return None;
    };
    let (left_value, right_value) = outer_value_columns(outer_value)?;
    Some((
        column(expression(outer_left_sequence)?)?,
        column(expression(outer_right_sequence)?)?,
        left_value,
        right_value,
    ))
}

fn outer_projection_matches(
    outer_left_sequence: ColumnRef<'_>,
    right_sequence: ColumnRef<'_>,
    left_value: ColumnRef<'_>,
    right_value: ColumnRef<'_>,
    probe_alias: &str,
    right_alias: &str,
) -> bool {
    outer_left_sequence.qualifier == Some(probe_alias)
        && outer_left_sequence.name == "left_sequence"
        && right_sequence.qualifier == Some(right_alias)
        && left_value.qualifier == Some(probe_alias)
        && left_value.name == "left_price"
        && right_value.qualifier == Some(right_alias)
}

fn band_projection<'a>(
    outer: &'a Select,
    probe: &'a Select,
    left_alias: &str,
    probe_alias: &str,
    right_alias: &str,
) -> Option<(&'a Expr, &'a Expr, &'a Expr, ColumnRef<'a>, ColumnRef<'a>)> {
    let [left_sequence, left_key, left_price] = probe_projection(probe, left_alias)?;
    let (outer_left_sequence, right_sequence, left_value, right_value) =
        outer_projection_values(outer)?;
    if !outer_projection_matches(
        outer_left_sequence,
        right_sequence,
        left_value,
        right_value,
        probe_alias,
        right_alias,
    ) {
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

struct BandShape<'a> {
    offsets: Vec<i64>,
    right_table: String,
    left_table: String,
    left_sequence: &'a Expr,
    left_key: &'a Expr,
    left_price: &'a Expr,
    right_sequence: ColumnRef<'a>,
    right_value: ColumnRef<'a>,
    event_time: ColumnRef<'a>,
}

fn band_join(outer: &Select) -> Option<(&Expr, String, String, &Query, String)> {
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
    Some((
        predicate,
        right_table,
        right_alias,
        probe_query,
        probe_alias,
    ))
}

fn band_probe<'a>(query: &'a Query, offsets_name: &str) -> Option<(&'a Select, String, String)> {
    let probe = query_select(query)?;
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
    Some((probe, left_table, left_alias))
}

fn band_match_time<'a>(
    probe: &'a Select,
    left_alias: &str,
    offsets_name: &str,
) -> Option<ColumnRef<'a>> {
    let match_time = named_expr(&probe.projection, "match_time")?;
    let (event_time, offset_column) = addition(match_time)?;
    let event_time = column(event_time)?;
    let offset_column = column(offset_column)?;
    if event_time.qualifier != Some(left_alias)
        || offset_column.qualifier != Some(offsets_name)
        || offset_column.name != "event_offset"
    {
        return None;
    }
    Some(event_time)
}

fn band_predicate(predicate: &Expr, probe_alias: &str, right_alias: &str) -> bool {
    let terms = conjunction(predicate);
    terms.len() == 2
        && terms.iter().any(|term| {
            column_pair_equality(term, probe_alias, right_alias, "left_symbol", "symbol")
        })
        && terms.iter().any(|term| {
            column_pair_equality(term, probe_alias, right_alias, "match_time", "event_time")
        })
}

fn band_shape(query: &Query) -> Option<BandShape<'_>> {
    let (offsets_name, offsets) = offset_seconds(query)?;
    let outer = query_select(query)?;
    let (predicate, right_table, right_alias, probe_query, probe_alias) = band_join(outer)?;
    let (probe, left_table, left_alias) = band_probe(probe_query, &offsets_name)?;
    let (left_sequence, left_key, left_price, right_sequence, right_value) =
        band_projection(outer, probe, &left_alias, &probe_alias, &right_alias)?;
    let event_time = band_match_time(probe, &left_alias, &offsets_name)?;
    if !band_predicate(predicate, &probe_alias, &right_alias) {
        return None;
    }
    Some(BandShape {
        offsets,
        right_table,
        left_table,
        left_sequence,
        left_key,
        left_price,
        right_sequence,
        right_value,
        event_time,
    })
}

fn band_input_times(
    left: &NativeInput,
    right: &NativeInput,
    event_time: &str,
) -> Option<(usize, usize, TimeUnit)> {
    let (left_time, unit) = timestamp_index(left, event_time)?;
    let (right_time, right_unit) = timestamp_index(right, "event_time")?;
    (unit == right_unit).then_some((left_time, right_time, unit))
}

fn scaled_offsets(offsets: Vec<i64>, unit: TimeUnit) -> Option<Arc<[i64]>> {
    let scale = match unit {
        TimeUnit::Second => 1_i64,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    };
    Some(
        offsets
            .into_iter()
            .map(|offset| offset.checked_mul(scale))
            .collect::<Option<Vec<_>>>()?
            .into(),
    )
}

fn band_left_indices(left: &NativeInput, shape: &BandShape<'_>) -> Option<(usize, usize, usize)> {
    Some((
        column_index(left, column(shape.left_key)?.name, &DataType::Utf8)?,
        column_index(left, column(shape.left_sequence)?.name, &DataType::UInt64)?,
        column_index(left, column(shape.left_price)?.name, &DataType::Float64)?,
    ))
}

fn band_right_indices(right: &NativeInput, shape: &BandShape<'_>) -> Option<(usize, usize, usize)> {
    Some((
        column_index(right, "symbol", &DataType::Utf8)?,
        column_index(right, shape.right_sequence.name, &DataType::UInt64)?,
        column_index(right, shape.right_value.name, &DataType::Float64)?,
    ))
}

fn build_band_spec(tables: &BTreeMap<String, Batch>, shape: BandShape<'_>) -> Option<BandSpec> {
    let left = input(tables, &shape.left_table)?;
    let right = input(tables, &shape.right_table)?;
    let (left_time, right_time, unit) = band_input_times(&left, &right, shape.event_time.name)?;
    let (left_key, left_sequence, left_price) = band_left_indices(&left, &shape)?;
    let (right_key, right_sequence, right_price) = band_right_indices(&right, &shape)?;
    let offsets = scaled_offsets(shape.offsets, unit)?;
    Some(BandSpec {
        left_key,
        right_key,
        left_time,
        right_time,
        left_sequence,
        right_sequence,
        left_price,
        right_price,
        offsets,
        shared: Arc::new(execution::BandShared::default()),
        left,
        right,
    })
}

fn band(query: &Query, tables: &BTreeMap<String, Batch>) -> Option<BandSpec> {
    let shape = band_shape(query)?;
    build_band_spec(tables, shape)
}
