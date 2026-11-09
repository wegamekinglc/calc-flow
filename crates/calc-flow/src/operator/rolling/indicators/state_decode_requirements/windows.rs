use super::{
    Check, Plan, Result, Source, add,
    alloc::{exact, growing},
    containers::history_slots,
};
use crate::operator::rolling::{CompiledWindowGroup, ExtremaKey, KeyValue, WindowState};
use datafusion::common::ScalarValue;

pub(super) fn plan(source: &Source<'_>, check: Check<'_>) -> Result<Plan> {
    let mut plan = Plan::default();
    for group in &source.compiled.window_groups {
        check()?;
        if let CompiledWindowGroup::Extrema { input_index, .. } = group {
            plan = merge_extrema(plan, source, *input_index, check)?;
        }
    }
    plan.scratch_bytes = add(plan.scratch_bytes, window_overlap(source)?)?;
    Ok(plan)
}
fn merge_extrema(plan: Plan, source: &Source<'_>, index: usize, check: Check<'_>) -> Result<Plan> {
    plan.merge(extrema(source, index, check)?)
}
fn window_overlap(source: &Source<'_>) -> Result<u64> {
    if source.counts.initialized == 0 {
        return Ok(0);
    }
    exact::<WindowState>(source.compiled.window_groups.len())
}

fn extrema(source: &Source<'_>, index: usize, check: Check<'_>) -> Result<Plan> {
    let mut heap = 0;
    let mut key_heap = 0;
    for row in 0..source.record.num_rows() {
        check()?;
        if source.kind(row) == 1 {
            let sequence = source.sequence_heap(row, check)?;
            let value = source.effective(row, index, check)?;
            heap = accumulated(heap, sequence, value)?;
            key_heap = key_heap.max(sequence);
        }
    }
    finish(source, heap, key_heap)
}
fn finish(source: &Source<'_>, heap: u64, key_heap: u64) -> Result<Plan> {
    let entries = history_slots::<(ExtremaKey, ScalarValue)>(source)?;
    let sequence = exact::<KeyValue>(source.compiled.sequence_columns.len())?;
    let sequences =
        super::super::math_requirements::bytes::repeat(Ok(sequence), source.counts.histories)?;
    let candidate_bytes = add(add(entries, sequences)?, heap)?;
    let scratch_bytes = scratch(source, key_heap)?;
    Ok(Plan {
        candidate_bytes,
        scratch_bytes,
    })
}
fn scratch(source: &Source<'_>, key_heap: u64) -> Result<u64> {
    if source.counts.histories == 0 {
        return Ok(0);
    }
    let keys = key_workspace(source, key_heap)?;
    let queue = growing::<(ExtremaKey, ScalarValue)>(source.counts.max_history)?;
    add(keys, queue.overlap)
}
fn key_workspace(source: &Source<'_>, key_heap: u64) -> Result<u64> {
    let key = growing::<KeyValue>(source.compiled.sequence_columns.len())?;
    let temporary = add(add(key.retained, key.overlap)?, key_heap)?;
    add(temporary, temporary)
}

fn accumulated(heap: u64, sequence: u64, value: u64) -> Result<u64> {
    add(heap, add(sequence, value)?)
}
