//! Precompiled column-only projections over read-only Arrow arrays.

use std::{collections::BTreeSet, sync::Arc};

use datafusion::{
    arrow::{
        datatypes::{Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    sql::sqlparser::{
        ast::{BinaryOperator, Expr, Ident, SelectItem, SetExpr, Statement},
        dialect::GenericDialect,
        parser::Parser,
    },
};

use crate::{Batch, CalcFlowError, Result, batch::TableBatchShell};

mod dependencies;
pub(super) use dependencies::required_columns;

#[derive(Clone, Debug)]
pub(crate) struct ColumnProjection {
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

    pub(super) fn schema(&self, input: &SchemaRef) -> Option<(Vec<usize>, SchemaRef)> {
        let mut indices = Vec::with_capacity(self.columns.len());
        let mut fields = Vec::with_capacity(self.columns.len());
        for (source, target) in &self.columns {
            let mut matches = input
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, field)| field.name() == source);
            let (index, field) = matches.next()?;
            if matches.next().is_some() {
                return None;
            }
            indices.push(index);
            fields.push(field.as_ref().clone().with_name(target));
        }
        let schema = if indices.iter().copied().eq(0..input.fields().len())
            && self.columns.iter().all(|(source, target)| source == target)
        {
            Arc::clone(input)
        } else {
            Arc::new(Schema::new_with_metadata(fields, input.metadata().clone()))
        };
        Some((indices, schema))
    }

    pub(super) fn apply(&self, batch: &Batch, node_id: &str) -> Result<Option<Batch>> {
        let table = batch.table_payload()?;
        let Some((indices, schema)) = self.schema(table.schema()) else {
            return Ok(None);
        };
        if Arc::ptr_eq(&schema, table.schema()) {
            return Ok(Some(batch.clone()));
        }
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
        Batch::table(records, batch.metadata().clone())
            .map(|projected| Some(projected.with_source_cursor(batch.source_cursor())))
    }
}

pub(crate) struct PreparedColumnProjection {
    indices: Vec<usize>,
    schema: SchemaRef,
}

impl PreparedColumnProjection {
    pub(crate) fn new(
        projection: &ColumnProjection,
        input: &SchemaRef,
        output: Option<&SchemaRef>,
    ) -> Option<Self> {
        let (indices, schema) = projection.schema(input)?;
        (output.is_none_or(|output| schema == *output)).then_some(Self { indices, schema })
    }

    pub(crate) fn shares_input_schema(&self, input: &Batch) -> bool {
        input
            .table_payload()
            .is_ok_and(|table| Arc::ptr_eq(&self.schema, table.schema()))
    }

    pub(crate) fn apply(
        &self,
        batch: &Batch,
        node_id: &str,
        shell: Option<TableBatchShell>,
    ) -> Result<Batch> {
        if self.shares_input_schema(batch) {
            return Ok(batch.clone());
        }
        if let Some(shell) = shell.filter(|shell| shell.matches(batch)) {
            return shell
                .project(
                    batch,
                    &self.indices,
                    Arc::clone(&self.schema),
                    batch.metadata().clone(),
                    batch.source_cursor(),
                )
                .map_err(|error| projection_error(node_id, &error));
        }
        let table = batch.table_payload()?;
        let records = table
            .batches()
            .iter()
            .map(|record| {
                let columns = self
                    .indices
                    .iter()
                    .map(|&index| record.column(index).clone())
                    .collect();
                RecordBatch::try_new(Arc::clone(&self.schema), columns)
                    .map_err(|error| projection_error(node_id, &error))
            })
            .collect::<Result<Vec<_>>>()?;
        Batch::table(records, batch.metadata().clone())
            .map(|projected| projected.with_source_cursor(batch.source_cursor()))
    }
}

fn projection_error(node_id: &str, error: &datafusion::arrow::error::ArrowError) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node_id.into(),
        message: format!("column projection failed: {error}"),
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

/// Match a row-preserving projection of columns followed by one Float64 product.
pub(super) fn matches_product(
    query: &crate::expression::ValidatedQuery,
    sequences: &[(&str, &str)],
    left: &str,
    right: &str,
) -> bool {
    let Some(select) = query
        .sql_statement()
        .and_then(dependencies::row_local_select)
    else {
        return false;
    };
    let Some((product, columns)) = select.projection.split_last() else {
        return false;
    };
    if select.selection.is_some() || columns.len() != sequences.len() {
        return false;
    }
    let Some(Expr::BinaryOp {
        left: lhs,
        op: BinaryOperator::Multiply,
        right: rhs,
    }) = aliased_expression(product, "value")
    else {
        return false;
    };
    column_matches(lhs, left)
        && column_matches(rhs, right)
        && columns.iter().zip(sequences).all(|(item, &(name, alias))| {
            aliased_expression(item, alias).is_some_and(|expr| column_matches(expr, name))
        })
}

fn aliased_expression<'a>(item: &'a SelectItem, name: &str) -> Option<&'a Expr> {
    let SelectItem::ExprWithAlias { expr, alias } = item else {
        return None;
    };
    (identifier(alias) == name).then(|| without_parentheses(expr))
}

fn without_parentheses(mut expr: &Expr) -> &Expr {
    while let Expr::Nested(inner) = expr {
        expr = inner;
    }
    expr
}

fn column_matches(expr: &Expr, name: &str) -> bool {
    matches!(without_parentheses(expr), Expr::Identifier(column) if identifier(column) == name)
}

fn identifier(value: &Ident) -> String {
    if value.quote_style.is_some() {
        value.value.clone()
    } else {
        value.value.to_ascii_lowercase()
    }
}
