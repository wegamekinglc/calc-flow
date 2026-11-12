//! The deliberately narrow bounded-inner SQL declaration dialect.
use crate::{Result, StreamJoinSpec, dataframe::invalid};
use datafusion::sql::{
    parser::Statement as DFStatement,
    sqlparser::ast::{
        Expr, FunctionArg, FunctionArgExpr, ObjectName, Query, Select, SelectItem, SetExpr,
        Statement, TableFactor, Value, Visit, Visitor,
    },
};
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

/// Explicit temporal identity of a joined pair; this declaration does not sort rows.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JoinOutputOrdering {
    /// Prefixed left equality keys in declaration order.
    pub entity_by: Vec<String>,
    /// One prefixed, non-null UTC microsecond join timestamp.
    pub event_time: String,
    /// Complete prefixed left sequence followed by the right sequence.
    pub sequence_by: Vec<String>,
}

#[doc(hidden)]
#[derive(Clone, Debug, Serialize)]
pub struct InnerJoinQuery {
    pub left_alias: String,
    pub right_alias: String,
    pub spec: StreamJoinSpec,
    pub output_ordering: Option<JoinOutputOrdering>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    spec: StreamJoinSpec,
    #[serde(default, deserialize_with = "ordering")]
    output_ordering: Option<JoinOutputOrdering>,
}
fn ordering<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<JoinOutputOrdering>, D::Error> {
    JoinOutputOrdering::deserialize(d).map(Some)
}
fn named(name: &ObjectName) -> bool {
    name.0
        .last()
        .and_then(|p| p.as_ident())
        .is_some_and(|n| n.value.eq_ignore_ascii_case("CF_INNER_JOIN"))
}
struct Find;
impl Visitor for Find {
    type Break = ();
    fn pre_visit_table_factor(&mut self, factor: &TableFactor) -> ControlFlow<()> {
        if matches!(factor, TableFactor::Table { name, .. } | TableFactor::Function { name, .. } if named(name))
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if matches!(expr, Expr::Function(f) if named(&f.name)) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}
fn error(message: impl std::fmt::Display) -> crate::CalcFlowError {
    invalid("sql.inner_join", "capability_mismatch", message)
}
#[doc(hidden)]
pub fn inner_join_query(text: &str, aliases: &[String]) -> Result<Option<InnerJoinQuery>> {
    let parsed = crate::expression::parse_select_query(text)
        .map_err(|e| invalid("sql.query", "invalid_literal", e))?;
    let DFStatement::Statement(statement) = parsed.statement() else {
        return Ok(None);
    };
    let Statement::Query(query) = statement.as_ref() else {
        return Ok(None);
    };
    if !matches!(query.visit(&mut Find), ControlFlow::Break(())) {
        return Ok(None);
    }
    parse(query, aliases).map(Some)
}
fn parse(query: &Query, aliases: &[String]) -> Result<InnerJoinQuery> {
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Err(error("one top-level SELECT * required"));
    };
    let args = join_arguments(select)?;
    validate_shape(query, select)?;
    configuration(args, aliases)
}

fn join_arguments(select: &Select) -> Result<&[FunctionArg]> {
    let [from] = select.from.as_slice() else {
        return Err(error("one FROM required"));
    };
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
            "unaliased CF_INNER_JOIN(left, right, configuration) required",
        ));
    };
    let allowed = [
        from.joins.is_empty(),
        name.0.len() == 1,
        named(name),
        with_hints.is_empty(),
        partitions.is_empty(),
        index_hints.is_empty(),
        args.settings.is_none(),
    ];
    if !allowed.into_iter().all(std::convert::identity) {
        return Err(error(
            "FROM modifiers and qualified function names are unsupported",
        ));
    }
    Ok(&args.args)
}

fn validate_shape(query: &Query, select: &Select) -> Result<()> {
    if !matches!(select.projection.as_slice(), [SelectItem::Wildcard(_)]) {
        return Err(error(
            "exactly SELECT * required; use a separate row-local relation for projections",
        ));
    }
    let mut shape = query.clone();
    let seed = crate::expression::parse_select_query("SELECT * FROM input")?;
    let DFStatement::Statement(seed) = seed.statement() else {
        unreachable!()
    };
    let Statement::Query(seed) = seed.as_ref() else {
        unreachable!()
    };
    let SetExpr::Select(clean) = shape.body.as_mut() else {
        unreachable!()
    };
    let SetExpr::Select(seed) = seed.body.as_ref() else {
        unreachable!()
    };
    clean.from.clone_from(&seed.from);
    if shape.to_string() != "SELECT * FROM input" {
        return Err(error(
            "only SELECT * FROM CF_INNER_JOIN is supported; no clauses or modifiers",
        ));
    }
    Ok(())
}
fn string(arg: &FunctionArg) -> Result<String> {
    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Value(value))) = arg
        && let Value::SingleQuotedString(value) = &value.value
    {
        return Ok(value.clone());
    }
    Err(error(
        "all three arguments must be ordinary SQL string literals",
    ))
}
fn configuration(args: &[FunctionArg], aliases: &[String]) -> Result<InnerJoinQuery> {
    let (left_alias, right_alias, config) = arguments(args)?;
    validate_aliases(&left_alias, &right_alias, aliases)?;
    let text = string(config)?;
    let config = read_configuration(&text)?;
    Ok(InnerJoinQuery {
        left_alias,
        right_alias,
        spec: config.spec,
        output_ordering: config.output_ordering,
    })
}

fn arguments(args: &[FunctionArg]) -> Result<(String, String, &FunctionArg)> {
    let [left, right, config] = args else {
        return Err(error("three arguments required"));
    };
    Ok((string(left)?, string(right)?, config))
}
fn validate_aliases(left: &str, right: &str, aliases: &[String]) -> Result<()> {
    let used = std::collections::BTreeSet::from([left.to_owned(), right.to_owned()]);
    if aliases.len() != 2 || used.len() != 2 || used != aliases.iter().cloned().collect() {
        return Err(error(
            "exactly the two distinct referenced aliases must be supplied",
        ));
    }
    Ok(())
}
fn read_configuration(text: &str) -> Result<Config> {
    let raw: serde_json::Value =
        serde_json::from_str(text).map_err(|e| invalid("sql.inner_join", "invalid_literal", e))?;
    validate_numbers(&raw)?;
    let config: Config = serde_json::from_value(raw)
        .map_err(|e| invalid("sql.inner_join.configuration", "invalid_literal", e))?;
    validate_ordering(config.output_ordering.as_ref())?;
    Ok(config)
}
fn validate_ordering(order: Option<&JoinOutputOrdering>) -> Result<()> {
    if let Some(order) = order
        && (order.entity_by.is_empty()
            || order.event_time.is_empty()
            || order.sequence_by.is_empty())
    {
        return Err(invalid(
            "sql.inner_join.output_ordering",
            "ordering_required",
            "all three ordering fields must be nonempty",
        ));
    }
    Ok(())
}

fn validate_numbers(raw: &serde_json::Value) -> Result<()> {
    for (group, fields, minimum) in [
        ("bounds", &["before_micros", "after_micros"][..], 0),
        (
            "limits",
            &[
                "max_state_rows_per_side",
                "max_state_bytes_per_side",
                "max_matches_per_input_batch",
            ][..],
            1,
        ),
    ] {
        for field in fields {
            let value = &raw["spec"][group][field];
            if !value
                .as_u64()
                .is_some_and(|n| (minimum..=crate::STREAM_JOIN_MAX_SAFE_JSON_INTEGER).contains(&n))
            {
                return Err(invalid(
                    &format!("sql.inner_join.spec.{group}.{field}"),
                    if group == "bounds" {
                        "invalid_time_bound"
                    } else {
                        "invalid_join_limit"
                    },
                    format!(
                        "expected integer {minimum}..={}, found {value}",
                        crate::STREAM_JOIN_MAX_SAFE_JSON_INTEGER
                    ),
                ));
            }
        }
    }
    Ok(())
}
