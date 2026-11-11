//! Read-only SQL relations composable with typed `DataFrame` declarations.

use crate::{
    Result, SqlOperator, TableExpr,
    dataframe::{Relation, RelationKind, digest, invalid},
};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

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
