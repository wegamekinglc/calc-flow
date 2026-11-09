use super::super::{State, math_requirements::bytes};
use super::{
    Check, Plan, Result, Source, add,
    alloc::{exact, growing},
    invalid,
};
use crate::operator::rolling::{
    BufferedRow, EntityRollingState, KeyValue, RowIdentity, WindowState,
};
use datafusion::common::ScalarValue;

type Entity = Vec<Option<KeyValue>>;
pub(super) fn plan(source: &Source<'_>, check: Check<'_>) -> Result<Plan> {
    check()?;
    let candidate_bytes = bytes::total([entities(source), rows(source), pending(source)])?;
    let scratch_bytes =
        bytes::total([dictionary(source), scratch_rows(source), validation(source)])?;
    Ok(Plan {
        candidate_bytes,
        scratch_bytes,
    })
}
fn entities(source: &Source<'_>) -> Result<u64> {
    let compiled = source.compiled;
    let each = bytes::total([
        exact::<Option<KeyValue>>(compiled.partition_columns.len()),
        exact::<WindowState>(compiled.window_groups.len()),
        exact::<State>(compiled.indicator_groups.len()),
    ])?;
    let owners = bytes::repeat(Ok(each), source.counts.initialized)?;
    add(
        owners,
        bytes::tree_nodes::<Entity, EntityRollingState>(source.counts.initialized)?,
    )
}
fn rows(source: &Source<'_>) -> Result<u64> {
    let total = source
        .counts
        .histories
        .checked_add(source.counts.pending)
        .ok_or_else(invalid)?;
    let values = growing::<ScalarValue>(source.input.fields().len())?.retained;
    let retained = bytes::repeat(Ok(values), total)?;
    add(retained, history_slots::<Vec<ScalarValue>>(source)?)
}
fn pending(source: &Source<'_>) -> Result<u64> {
    let entity = source.compiled.partition_columns.len();
    let sequence = source.compiled.sequence_columns.len();
    let identities = bytes::total([
        growing::<Option<KeyValue>>(entity).map(|bytes| bytes.retained),
        growing::<KeyValue>(sequence).map(|bytes| bytes.retained),
        exact::<Option<KeyValue>>(entity),
        exact::<KeyValue>(sequence),
    ])?;
    let rows = bytes::repeat(Ok(identities), source.counts.pending)?;
    add(
        rows,
        bytes::tree_nodes::<RowIdentity, BufferedRow>(source.counts.pending)?,
    )
}
fn dictionary(source: &Source<'_>) -> Result<u64> {
    let count = source.counts.entities;
    let values = growing::<ScalarValue>(source.input.fields().len())?.retained;
    let keys = growing::<Option<KeyValue>>(source.compiled.partition_columns.len())?.retained;
    let rows = bytes::repeat(add(values, keys), count)?;
    let controls = bytes::total([
        peak::<(Entity, Vec<ScalarValue>)>(count),
        peak::<u64>(count),
        bytes::repeat(peak::<bool>(count), 2),
    ])?;
    add(rows, controls)
}
fn scratch_rows(source: &Source<'_>) -> Result<u64> {
    if source.record.num_rows() == 0 {
        return Ok(0);
    }
    let read = growing::<ScalarValue>(source.input.fields().len())?;
    let controls = if has_controls(source) {
        read.retained
    } else {
        0
    };
    let history = history_work(source)?;
    let last = last_buffer(source)?;
    bytes::total([
        Ok(read.overlap),
        Ok(controls),
        Ok(history),
        Ok(last),
        key_growth(source),
        history_overlap(source),
    ])
}
fn history_work(source: &Source<'_>) -> Result<u64> {
    if source.counts.histories == 0 {
        return Ok(0);
    }
    let compiled = source.compiled;
    bytes::total([
        exact::<ScalarValue>(source.input.fields().len()),
        exact::<Option<KeyValue>>(compiled.partition_columns.len()),
        peak::<Option<KeyValue>>(compiled.partition_columns.len()),
        peak::<KeyValue>(compiled.sequence_columns.len()),
    ])
}
fn last_buffer(source: &Source<'_>) -> Result<u64> {
    if source.counts.pending == 0 {
        return Ok(0);
    }
    let identity = add(
        exact::<Option<KeyValue>>(source.compiled.partition_columns.len())?,
        exact::<KeyValue>(source.compiled.sequence_columns.len())?,
    )?;
    bytes::repeat(Ok(identity), 2)
}
fn key_growth(source: &Source<'_>) -> Result<u64> {
    let compiled = source.compiled;
    let entity = growing::<Option<KeyValue>>(compiled.partition_columns.len())?;
    let sequence = growing::<KeyValue>(compiled.sequence_columns.len())?;
    add(entity.overlap, sequence.overlap)
}
fn validation(source: &Source<'_>) -> Result<u64> {
    bytes::total([
        bytes::tree_nodes::<usize, ()>(source.input.fields().len()),
        bytes::tree_nodes::<(u64, u64), ()>(source.counts.cumulative),
        bytes::tree_nodes::<(u64, u64), ()>(source.counts.indicators),
    ])
}
pub(super) fn history_slots<T>(source: &Source<'_>) -> Result<u64> {
    let rows = source.counts.histories;
    if rows == 0 {
        return Ok(0);
    }
    growing::<T>(source.counts.max_history)?;
    let groups = source.counts.initialized.min(rows);
    let capacity = rows
        .checked_mul(2)
        .and_then(|rows| {
            groups
                .checked_mul(4)
                .and_then(|groups| rows.checked_add(groups))
        })
        .ok_or_else(invalid)?;
    bytes::slots::<T>(capacity)
}
fn peak<T>(count: usize) -> Result<u64> {
    let bytes = growing::<T>(count)?;
    add(bytes.retained, bytes.overlap)
}

fn history_overlap(source: &Source<'_>) -> Result<u64> {
    Ok(growing::<Vec<ScalarValue>>(source.counts.max_history)?.overlap)
}

fn has_controls(source: &Source<'_>) -> bool {
    [
        source.counts.ewma,
        source.counts.cumulative,
        source.counts.indicators,
    ]
    .into_iter()
    .any(|count| count > 0)
}
