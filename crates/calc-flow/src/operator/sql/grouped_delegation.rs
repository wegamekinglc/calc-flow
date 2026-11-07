//! Delegated delta aggregation for large compact-SQL batches.
//!
//! When a batch is large and the aggregate shape is the inline integer SUM
//! over plain columns, the per-batch delta aggregation runs as one
//! partitioned `DataFusion` `GROUP BY` on a dedicated delegated session
//! (the plan-template reuse of `PR #557` keeps it to a single plan), and the
//! grouped result - one row per touched group - is rebuilt as a mini-batch
//! with the input
//! schema and fed through the ordinary incremental kernel. The kernel then
//! touches ~group-count rows instead of the full input, while state,
//! checkpoints, snapshots, and refunds keep their exact contracts. Any
//! failure falls back to the serial kernel on the original batch; see
//! `.codex/artifacts/specs/sum-key-sharding-20261107.md` for why hand-built
//! parallel kernels were withdrawn in favor of this route.

use std::collections::BTreeMap;

use super::{
    Arc, Batch, Column, DataFusionRuntime, IncrementalSql, PhysicalExpr, RecordBatch, Result,
    StreamOperatorContext, df_error,
};
use crate::expression::ValidatedQuery;
use datafusion::arrow::{
    array::{Array, new_null_array},
    compute::concat_batches,
};

/// Delegation floor: smaller batches keep the direct kernel path.
const MIN_ROWS: usize = 100_000;
const ALIAS: &str = "delta";

/// Pre-aggregates one large batch when its shape and partition policy
/// admit delegation; `None` keeps the serial kernel on the original batch.
pub(in crate::operator::sql) async fn pre_aggregate(
    plan: &IncrementalSql,
    runtime: &DataFusionRuntime,
    batch: &Batch,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<Option<Batch>> {
    #[cfg(test)]
    if plan
        .delegation_disabled
        .load(std::sync::atomic::Ordering::SeqCst)
    {
        return Ok(None);
    }
    let rows = batch.num_rows();
    if rows < MIN_ROWS {
        return Ok(None);
    }
    context.check_cancelled()?;
    let partitions = runtime.delegation_partition_count(rows);
    if partitions < 2 {
        return Ok(None);
    }
    let Some(query) = plan.delegation_query() else {
        return Ok(None);
    };
    let tables = BTreeMap::from([(ALIAS.to_owned(), batch.clone())]);
    let debug = std::env::var_os("DELEGATION_DEBUG").is_some();
    let started = std::time::Instant::now();
    let grouped = match runtime
        .sql_delegated(query, &tables, partitions, Some(name))
        .await
    {
        Ok(grouped) => {
            if debug {
                eprintln!(
                    "DELEGATION_DEBUG rows={rows} partitions={partitions} query_ms={:.1} grouped={}",
                    started.elapsed().as_secs_f64() * 1000.0,
                    grouped.num_rows(),
                );
            }
            grouped
        }
        Ok(grouped) => grouped,
        Err(error) => {
            // Real cancellation or deadline failures propagate; anything
            // else falls back to the serial kernel.
            context.check_cancelled()?;
            let _ = error;
            return Ok(None);
        }
    };
    Ok(rebuild(plan, &grouped, batch))
}

/// Builds the delegated aggregation query once per plan; `None` marks a
/// shape that never delegates.
pub(super) fn build_query(plan: &IncrementalSql) -> Option<ValidatedQuery> {
    if plan.predicate.is_some()
        || plan.aggregate_filters.iter().any(Option::is_some)
        || !plan.input_checks.is_empty()
        || !plan.variable_extrema.is_empty()
        || plan.sequential.is_some()
        || plan.inline_sum.is_none()
        || plan.keys.is_empty()
    {
        return None;
    }
    let mut names = Vec::with_capacity(plan.keys.len() + 1);
    for key in &plan.keys {
        names.push(column_name(key.expression.as_ref())?);
    }
    let arguments = plan.aggregates.first()?.expressions();
    let argument = column_name(arguments.first()?.as_ref())?;
    let selection = names
        .iter()
        .map(|name| format!("\"{}\"", name.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!(
        "SELECT {selection}, SUM(\"{argument_quoted}\") AS \"{argument_quoted}\" \
         FROM \"{ALIAS}\" GROUP BY {selection}",
        argument_quoted = argument.replace('"', "\"\"")
    );
    crate::expression::parse_select_query(&text).ok()
}

fn column_name(expression: &dyn PhysicalExpr) -> Option<&str> {
    expression.downcast_ref::<Column>().map(Column::name)
}

/// Rebuilds the grouped result as a mini-batch carrying the exact input
/// schema: key and argument columns come from the aggregation, unused
/// columns are null-filled and never read.
fn rebuild(plan: &IncrementalSql, grouped: &Batch, original: &Batch) -> Option<Batch> {
    let payload = grouped.table_payload().ok()?;
    let batches = payload.batches();
    let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    if rows == 0 {
        return None;
    }
    let grouped = if batches.len() == 1 {
        batches[0].clone()
    } else {
        concat_batches(payload.schema(), batches.iter()).ok()?
    };
    let schema = plan.schema.clone();
    let mut columns = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let column = grouped
            .column_by_name(field.name())
            .filter(|column| column.data_type() == field.data_type())
            .cloned()
            .unwrap_or_else(|| new_null_array(field.data_type(), rows));
        if column.len() != rows {
            return None;
        }
        columns.push(column);
    }
    let record = RecordBatch::try_new(Arc::clone(&schema), columns)
        .map_err(|error| df_error("delta", error))
        .ok()?;
    Batch::table(vec![record], original.metadata().clone()).ok()
}
