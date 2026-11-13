//! Read-only SQL relations composable with typed `DataFrame` declarations.

mod asof_join;
mod distinct;
#[doc(hidden)]
pub use asof_join::{AsofJoinQuery, asof_join_query};
pub(crate) use distinct::window_time_column;
#[doc(hidden)]
pub use distinct::{DistinctQuery, distinct_query};
mod inner_join;
pub use inner_join::JoinOutputOrdering;
#[doc(hidden)]
pub use inner_join::{InnerJoinQuery, inner_join_query};
mod union;
mod window;
#[doc(hidden)]
pub use union::{UnionArm, UnionArmQuery, UnionPlan, plan_union_all, union_all_queries};
#[doc(hidden)]
pub use window::{EventWindowQuery, event_window_query, sql_passthrough_columns};

use crate::{
    Result, SqlOperator, TableExpr,
    dataframe::{Relation, RelationKind, digest, invalid},
};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

fn joined_relation(query: &str, tables: &BTreeMap<String, TableExpr>) -> Result<Option<TableExpr>> {
    let aliases = tables.keys().cloned().collect::<Vec<_>>();
    if let Some(plan) = asof_join_query(query, &aliases)? {
        return Ok(Some(tables[&plan.left_alias].asof_join_relation(
            &tables[&plan.right_alias],
            plan.spec,
            true,
        )));
    }
    if let Some(plan) = inner_join_query(query, &aliases)? {
        return tables[&plan.left_alias]
            .inner_join_relation(
                &tables[&plan.right_alias],
                plan.spec,
                plan.output_ordering,
                true,
            )
            .map(Some);
    }
    Ok(None)
}

/// Declares one read-only SELECT/CTE with explicit, uniquely normalized aliases.
///
/// # Errors
/// Rejects empty tables, malformed aliases, case-fold collisions and invalid SQL.
pub fn sql(query: &str, tables: BTreeMap<String, TableExpr>) -> Result<TableExpr> {
    if tables.is_empty() {
        return Err(invalid(
            "sql.tables",
            "invalid_literal",
            "at least one table alias is required",
        ));
    }
    let mut aliases = BTreeSet::new();
    for name in tables.keys() {
        crate::Port::new(name, crate::BatchKind::Table, true, None)
            .map_err(|error| invalid(&format!("sql.tables.{name}"), "invalid_literal", error))?;
        if !aliases.insert(name.to_ascii_lowercase()) {
            return Err(invalid(
                &format!("sql.tables.{name}"),
                "duplicate_name",
                "normalized SQL aliases must be unique",
            ));
        }
    }
    if let Some(plan) = distinct_query(query, &tables.keys().cloned().collect::<Vec<_>>())? {
        return tables[&plan.alias].distinct_sql_relation(plan.columns, plan.spec);
    }
    if let Some(relation) = joined_relation(query, &tables)? {
        return Ok(relation);
    }
    if let Some(plan) = event_window_query(query, &tables.keys().cloned().collect::<Vec<_>>())? {
        return tables[&plan.alias].window_relation(plan.spec, true);
    }
    SqlOperator::new("sql", query, tables.keys().cloned().collect(), vec![])
        .map_err(|error| invalid("sql.query", "invalid_literal", error))?;
    let id = digest(&json!([
        "sql",
        query,
        tables
            .iter()
            .map(|(alias, table)| (alias, &table.0.id))
            .collect::<Vec<_>>()
    ]));
    Ok(TableExpr(Arc::new(Relation {
        id,
        kind: RelationKind::Sql {
            query: query.into(),
            tables,
        },
    })))
}
