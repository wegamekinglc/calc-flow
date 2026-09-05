//! Precompiled column-only projections; arithmetic and relational work stay in SQL.

use std::{collections::BTreeSet, sync::Arc};

use datafusion::{
    arrow::{datatypes::Schema, record_batch::RecordBatch},
    sql::sqlparser::{
        ast::{Expr, Ident, SelectItem, SetExpr, Statement},
        dialect::GenericDialect,
        parser::Parser,
    },
};

use crate::{Batch, CalcFlowError, Result};

#[derive(Clone, Debug)]
pub(super) struct ColumnProjection {
    columns: Vec<(String, String)>,
}

impl ColumnProjection {
    pub(super) fn parse(select: &[String]) -> Option<Self> {
        let columns = select
            .iter()
            .map(|item| parse_column(item))
            .collect::<Option<Vec<_>>>()?;
        let names = columns
            .iter()
            .map(|(_, name)| name)
            .collect::<BTreeSet<_>>();
        // Let SQL diagnose duplicate output names instead of introducing a
        // different successful schema or error on this path.
        if columns.is_empty() || names.len() != columns.len() {
            return None;
        }
        Some(Self { columns })
    }

    pub(super) fn apply(&self, batch: &Batch, node_id: &str) -> Result<Option<Batch>> {
        let table = batch.table_payload()?;
        let mut indices = Vec::with_capacity(self.columns.len());
        let mut fields = Vec::with_capacity(self.columns.len());
        for (source, target) in &self.columns {
            let mut matches = table
                .schema()
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, field)| field.name() == source);
            let Some((index, field)) = matches.next() else {
                return Ok(None);
            };
            if matches.next().is_some() {
                return Ok(None);
            }
            indices.push(index);
            fields.push(field.as_ref().clone().with_name(target));
        }
        if indices.iter().copied().eq(0..table.schema().fields().len())
            && self.columns.iter().all(|(source, target)| source == target)
        {
            return Ok(Some(batch.clone()));
        }
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            table.schema().metadata().clone(),
        ));
        let records = table
            .batches()
            .iter()
            .map(|record| {
                let columns = indices
                    .iter()
                    .map(|&index| record.column(index).clone())
                    .collect();
                RecordBatch::try_new(Arc::clone(&schema), columns).map_err(|error| {
                    CalcFlowError::Operator {
                        node_id: node_id.into(),
                        message: format!("column projection failed: {error}"),
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Batch::table(records, batch.metadata().clone()).map(Some)
    }
}

fn parse_column(source: &str) -> Option<(String, String)> {
    let statements = Parser::parse_sql(&GenericDialect {}, &format!("SELECT {source}")).ok()?;
    let [Statement::Query(query)] = statements.as_slice() else {
        return None;
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return None;
    };
    let [item] = select.projection.as_slice() else {
        return None;
    };
    // Reject every additional query clause, including DISTINCT, FROM, LIMIT,
    // and ORDER BY embedded in a configured select item.
    if query.to_string() != format!("SELECT {item}") {
        return None;
    }
    let (expression, alias) = match item {
        SelectItem::UnnamedExpr(expression) => (expression, None),
        SelectItem::ExprWithAlias { expr, alias } => (expr, Some(alias)),
        _ => return None,
    };
    let column = match expression {
        Expr::Identifier(column) => column,
        Expr::CompoundIdentifier(parts) => {
            let [qualifier, column] = parts.as_slice() else {
                return None;
            };
            if identifier(qualifier) != "input" {
                return None;
            }
            column
        }
        _ => return None,
    };
    Some((identifier(column), identifier(alias.unwrap_or(column))))
}

fn identifier(value: &Ident) -> String {
    if value.quote_style.is_some() {
        value.value.clone()
    } else {
        value.value.to_lowercase()
    }
}
