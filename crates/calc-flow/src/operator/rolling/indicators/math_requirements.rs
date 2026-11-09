//! Additional heap for the existing cell evaluator. The caller prepays its
//! whole-state candidate clone separately; this covers the evaluator's further
//! per-entity clones, cell/history owners and scratch, excluding borrowed Arrow
//! backing. Reserve before `compute_cells` and retain until actual owner transfer.

use datafusion::{arrow::datatypes::DataType, common::ScalarValue};

use super::super::{
    CompiledEvaluation, CompiledFrame, CompiledRollingSpec, CompiledWindowGroup,
    EntityRollingState, ExtremaKey, IncrementalScanState, KeyValue, RollingStreamState, ScanKind,
    WindowState,
};
use crate::Result;

#[path = "math_requirements/bytes.rs"]
pub(super) mod bytes;
#[path = "math_requirements/clones.rs"]
mod clones;
#[path = "math_requirements/source.rs"]
mod source;

use bytes::{add, growing, invalid, repeat, slots, total};
use source::{Source, Summary};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(super) fn plan(
    state: &RollingStreamState,
    compiled: &CompiledRollingSpec,
    closing_rows: usize,
    check: Check<'_>,
) -> Result<u64> {
    check()?;
    if closing_rows == 0 {
        return slots::<Vec<Option<ScalarValue>>>(compiled.outputs.len());
    }
    plan_source(&Source::stream(state, compiled), closing_rows, check)
}
pub(super) fn plan_finite(
    rows: &[super::super::BufferedRow],
    compiled: &CompiledRollingSpec,
    check: Check<'_>,
) -> Result<u64> {
    check()?;
    if rows.is_empty() {
        return slots::<Vec<Option<ScalarValue>>>(compiled.outputs.len());
    }
    plan_source(&Source::finite(rows, compiled), rows.len(), check)
}
fn plan_source(source: &Source<'_>, closing_rows: usize, check: Check<'_>) -> Result<u64> {
    let compiled = source.compiled;
    let summary = source.summary(check)?;
    let scope = summary
        .history_rows
        .checked_add(closing_rows)
        .ok_or_else(invalid)?;
    let bytes = total([
        containers(compiled, closing_rows),
        histories(compiled, &summary, closing_rows, scope),
        output_cells(source, closing_rows, check),
        windows(source, &summary, closing_rows, scope, check),
        scans(compiled, scope, check),
    ])?;
    check()?;
    Ok(bytes)
}
fn containers(compiled: &CompiledRollingSpec, rows: usize) -> Result<u64> {
    total([
        repeat(slots::<Option<ScalarValue>>(rows), compiled.outputs.len()),
        slots::<Vec<Option<ScalarValue>>>(compiled.outputs.len()),
        slots::<(Vec<Option<KeyValue>>, EntityRollingState)>(rows),
        bytes::tree_nodes::<&Vec<Option<KeyValue>>, Vec<usize>>(rows),
        growing::<usize>(rows),
        slots::<[usize; 4]>(rows),
    ])
}
fn histories(
    compiled: &CompiledRollingSpec,
    source: &Summary,
    rows: usize,
    scope: usize,
) -> Result<u64> {
    total([
        Ok(source.prior_copies),
        repeat(
            total([slots::<ScalarValue>(source.row_width), Ok(source.row_heap)]),
            rows,
        ),
        growing::<Vec<ScalarValue>>(scope),
        slots::<[Vec<ScalarValue>; 4]>(rows),
        repeat(entity_vectors(compiled, source), rows),
    ])
}
fn entity_vectors(compiled: &CompiledRollingSpec, source: &Summary) -> Result<u64> {
    total([
        slots::<Option<KeyValue>>(compiled.partition_columns.len()),
        Ok(source.entity_heap),
        slots::<WindowState>(compiled.window_groups.len()),
        slots::<super::State>(compiled.indicator_groups.len()),
    ])
}

fn output_cells(source: &Source<'_>, rows: usize, check: Check<'_>) -> Result<u64> {
    let mut bytes = 0;
    for output in &source.compiled.outputs {
        check()?;
        let heap = source.column_heap(output.input_index, check)?;
        bytes = add(
            bytes,
            repeat(Ok(heap.max(null_cell(output, check)?)), rows)?,
        )?;
    }
    Ok(bytes)
}
fn null_cell(output: &super::super::CompiledRollingOutput, check: Check<'_>) -> Result<u64> {
    match output.evaluation {
        CompiledEvaluation::Lag { .. } => {
            super::scalar_materialization_plan::null_requirement(&output.input_type, check)
        }
        _ => Ok(0),
    }
}

fn windows(
    source: &Source<'_>,
    summary: &Summary,
    rows: usize,
    scope: usize,
    check: Check<'_>,
) -> Result<u64> {
    let mut bytes =
        growing::<Option<f64>>(super::readout_width(&source.compiled.indicator_groups)?)?;
    for group in &source.compiled.window_groups {
        check()?;
        let CompiledWindowGroup::Extrema { input_index, .. } = group else {
            continue;
        };
        bytes = add(
            bytes,
            extrema_growth(source, summary, (*input_index, rows, scope), check)?,
        )?;
    }
    Ok(bytes)
}
fn extrema_growth(
    source: &Source<'_>,
    summary: &Summary,
    (input, rows, scope): (usize, usize, usize),
    check: Check<'_>,
) -> Result<u64> {
    let keys = total([
        slots::<KeyValue>(source.compiled.sequence_columns.len()),
        Ok(summary.sequence_heap),
        source.column_heap(input, check),
    ]);
    total([
        growing::<(ExtremaKey, ScalarValue)>(scope),
        slots::<[(ExtremaKey, ScalarValue); 4]>(rows),
        repeat(keys, rows),
    ])
}

fn scans(compiled: &CompiledRollingSpec, scope: usize, check: Check<'_>) -> Result<u64> {
    let mut bytes = slots::<Option<IncrementalScanState>>(compiled.outputs.len())?;
    let mut slow_peak = 0;
    for output in &compiled.outputs {
        check()?;
        let (persistent, temporary) = scan_cost(output, scope)?;
        bytes = add(bytes, persistent)?;
        slow_peak = slow_peak.max(temporary);
    }
    add(bytes, slow_peak)
}
fn scan_cost(output: &super::super::CompiledRollingOutput, scope: usize) -> Result<(u64, u64)> {
    let CompiledEvaluation::Scan(scan) = output.evaluation else {
        return Ok((0, 0));
    };
    if output.input_type == DataType::Float64
        && let CompiledFrame::Rows(window) = scan.frame
        && let Some(cost) = incremental_scan(scan.kind, window, scope)?
    {
        return Ok((cost, 0));
    }
    Ok((0, scan_workspace(scan.kind, scope)?))
}
fn incremental_scan(kind: ScanKind, window: u64, scope: usize) -> Result<Option<u64>> {
    let window = usize::try_from(window).unwrap_or(usize::MAX);
    let count = scope.min(window.saturating_add(1));
    match kind {
        ScanKind::Argmax | ScanKind::Argmin => {
            total([growing::<bool>(count), growing::<(usize, f64)>(count)]).map(Some)
        }
        ScanKind::UniqueCount => {
            total([growing::<Option<u64>>(count), bytes::hash_table(scope)]).map(Some)
        }
        _ => Ok(None),
    }
}

fn scan_workspace(kind: ScanKind, scope: usize) -> Result<u64> {
    let samples = growing::<(usize, &ScalarValue)>(scope);
    if kind != ScanKind::UniqueCount {
        return samples;
    }
    // Stable sort scratch is bounded by the full input reference count.
    total([
        samples,
        slots::<&ScalarValue>(scope),
        slots::<&ScalarValue>(scope),
    ])
}

#[cfg(test)]
#[path = "math_requirements/tests.rs"]
mod tests;
