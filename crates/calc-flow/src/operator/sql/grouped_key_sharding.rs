//! Key-sharded per-batch grouped integer SUM updates.
//!
//! One input batch updates one `IncrementalSql` state on the operator task.
//! When the batch is large, its aggregate shape is the inline integer SUM,
//! and the incremental pool is unbounded, the batch's rows are routed by the
//! hash of their complete group key into shards; each shard runs the chunk
//! update on the job-scoped native CPU service as one
//! [`ParallelCpuWork`] unit, and the returned fragments are folded back on
//! the operator task. Because a shard owns every row of its groups, no group
//! is visited by two units and the coordinator never merges per-group
//! fragments: it only resolves global slots - in the original first-appearance
//! order of the serial path, so emitted snapshots stay byte-identical - and
//! merges each fragment's partial states once. See
//! `.codex/artifacts/specs/sum-key-sharding-20261107.md` and the rejected
//! row-range lane study `benchmarks/sum-state/sum-batch-parallel-20261104.md`
//! for the measured contrast.

use std::{mem::size_of, sync::Mutex};

use super::{
    Arc, ArrayRef, CHUNK_ROWS, Column, DataFusionRuntime, HashMap, IncrementalSql, InputCandidates,
    MemoryReservation, PartialGroups, PhysicalExpr, RandomState, RecordBatch, Result, RowConverter,
    SortField, StreamOperatorContext, checked_bytes, df_error, native_key_width,
};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, ParallelCpuWork,
};
use datafusion::arrow::{
    array::{Array, UInt32Builder},
    compute::{concat_batches, take},
};

/// Shard admission floor: smaller batches keep the serial path.
const MIN_ROWS: usize = 64_000;
/// Rows per shard unit, matching the maintained join-key admission shape.
const ROWS_PER_UNIT: usize = 16_000;
/// Bounded so one batch cannot occupy the whole native service alone.
const MAX_UNITS: usize = 8;
/// Fixed routing seed: shard assignment is deterministic across runs.
fn routing_seed() -> datafusion::common::hash_utils::RandomState {
    datafusion::common::hash_utils::RandomState::default()
}

#[cfg(test)]
use super::grouped_key_sharding_tests;

pub(super) fn unbounded(runtime: &DataFusionRuntime) -> bool {
    runtime
        .incremental_memory_pool()
        .is::<datafusion::execution::memory_pool::UnboundedMemoryPool>()
}

fn column(expression: &dyn PhysicalExpr) -> Option<usize> {
    expression.downcast_ref::<Column>().map(Column::index)
}

/// The v1 admission shape: one grouped inline integer SUM over a plain
/// column, grouped by plain columns, with no predicates or aggregate
/// filters, and no single-primitive-key native shape.
fn inputs_eligible(plan: &IncrementalSql) -> bool {
    plan.inline_sum.is_some()
        && !plan.keys.is_empty()
        && plan.input_checks.is_empty()
        && plan.predicate.is_none()
        && plan.aggregate_filters.iter().all(Option::is_none)
        && plan.variable_extrema.is_empty()
        && plan.sequential.is_none()
        && !native_shape(plan)
        && plan
            .keys
            .iter()
            .all(|key| column(key.expression.as_ref()).is_some())
        && plan.aggregates.iter().all(|expression| {
            matches!(expression.expressions().as_slice(), [argument] if column(argument.as_ref()).is_some())
        })
}

fn native_shape(plan: &IncrementalSql) -> bool {
    matches!(plan.keys.as_slice(), [key] if native_key_width(key.field.data_type()).is_some())
        && !plan.requires_explicit_grouping()
}

struct Dimensions {
    rows: usize,
    units: usize,
}

fn units(rows: usize, cpus: usize) -> usize {
    cpus.min(MAX_UNITS).min(rows / ROWS_PER_UNIT).max(1)
}

fn dimensions(plan: &IncrementalSql, records: &[RecordBatch], cpus: usize) -> Option<Dimensions> {
    if !plan.parallel_scratch || cpus < 2 || !inputs_eligible(plan) {
        return None;
    }
    let rows = records
        .iter()
        .try_fold(0usize, |rows, record| rows.checked_add(record.num_rows()))?;
    let units = units(rows, cpus);
    if rows < MIN_ROWS || units < 2 {
        return None;
    }
    Some(Dimensions { rows, units })
}

/// One shard's gathered rows: the concatenation of every input record's
/// routed rows, with the original row index of each gathered row in
/// ascending order.
struct Routed {
    record: RecordBatch,
    origin: Vec<u32>,
}

struct Shard {
    partial: PartialGroups,
}

struct LocalKey {
    encoded: Arc<[u8]>,
    shard: usize,
    row: usize,
}

struct Work {
    records: Vec<RecordBatch>,
    origins: Vec<Vec<u32>>,
    keys: Vec<usize>,
    arguments: Vec<usize>,
    shards: Vec<Mutex<Option<Shard>>>,
    name: String,
    scratch: Arc<MemoryReservation>,
}

struct Fragment {
    keys: Vec<LocalKey>,
    states: Vec<Vec<ArrayRef>>,
    _partial: MemoryReservation,
    _scratch: Arc<MemoryReservation>,
}

struct LocalGroups<'a> {
    keys: Vec<LocalKey>,
    ranks: HashMap<Arc<[u8]>, usize, RandomState>,
    converter: RowConverter,
    shard: usize,
    name: &'a str,
}

struct LocalInput {
    keys: Vec<ArrayRef>,
    arguments: Vec<Vec<ArrayRef>>,
}

fn scratch_bytes(
    plan: &IncrementalSql,
    routed: &[Routed],
    rows: usize,
    name: &str,
) -> Result<usize> {
    let width = checked_bytes(256, [(plan.keys.len(), 64)], name)?;
    let mut bytes = checked_bytes(
        16_384,
        [
            (rows, width),
            (routed.len(), 4_096),
            (plan.keys.len() + plan.aggregates.len(), 128),
            (plan.plan_bytes, 1),
            (name.len(), 2),
        ],
        name,
    )?;
    for shard in routed {
        bytes = checked_bytes(bytes, [(shard.record.get_array_memory_size(), 1)], name)?;
        bytes = checked_bytes(bytes, [(shard.origin.capacity(), size_of::<u32>())], name)?;
    }
    Ok(bytes)
}

/// Routes every input row to `hash(key) % units` and gathers one record per
/// shard, preserving original row order inside each shard.
fn route(records: &[RecordBatch], keys: &[usize], units: usize, name: &str) -> Result<Vec<Routed>> {
    let schema = records
        .first()
        .map(RecordBatch::schema)
        .ok_or_else(|| df_error(name, "parallel SUM received no records"))?;
    let mut indices = (0..units)
        .map(|_| UInt32Builder::with_capacity(0))
        .collect::<Vec<_>>();
    let mut origin = (0..units).map(|_| Vec::<u32>::new()).collect::<Vec<_>>();
    let mut pieces = vec![Vec::new(); units];
    let mut hashes = Vec::new();
    let mut total = 0u64;
    for record in records {
        let key_arrays = keys
            .iter()
            .map(|&index| record.column(index).clone())
            .collect::<Vec<_>>();
        hashes.clear();
        hashes.resize(record.num_rows(), 0);
        datafusion::common::hash_utils::create_hashes(&key_arrays, &routing_seed(), &mut hashes)
            .map_err(|error| df_error(name, error))?;
        for (row, hash) in hashes.iter().enumerate() {
            let unit =
                usize::try_from(*hash % units as u64).map_err(|error| df_error(name, error))?;
            let original = total
                .checked_add(row as u64)
                .ok_or_else(|| df_error(name, "parallel SUM input row index overflowed u64"))?;
            indices[unit].append_value(u32::try_from(row).map_err(|error| df_error(name, error))?);
            origin[unit].push(u32::try_from(original).map_err(|error| df_error(name, error))?);
        }
        total = total
            .checked_add(record.num_rows() as u64)
            .ok_or_else(|| df_error(name, "parallel SUM input row count overflow"))?;
        let row_index = indices
            .iter_mut()
            .map(|builder| builder.finish_cloned())
            .collect::<Vec<_>>();
        for (unit, selection) in row_index.iter().enumerate() {
            if selection.is_empty() {
                continue;
            }
            let columns = (0..schema.fields().len())
                .map(|index| {
                    take(record.column(index), selection, None)
                        .map_err(|error| df_error(name, error))
                })
                .collect::<Result<Vec<_>>>()?;
            let piece = RecordBatch::try_new(Arc::clone(&schema), columns)
                .map_err(|error| df_error(name, error))?;
            pieces[unit].push(piece);
        }
        for builder in &mut indices {
            builder.finish();
        }
    }
    let mut routed = Vec::new();
    for ((pieces, origin), unit) in pieces.into_iter().zip(origin).zip(0..) {
        if origin.is_empty() {
            continue;
        }
        let record =
            concat_batches(&schema, pieces.iter()).map_err(|error| df_error(name, error))?;
        if record.num_rows() != origin.len() {
            return Err(df_error(name, "routed shard row count mismatch"));
        }
        routed.push((record, origin, unit));
    }
    Ok(routed
        .into_iter()
        .map(|(record, origin, _)| Routed { record, origin })
        .collect())
}

fn prepare(
    plan: &IncrementalSql,
    records: &[RecordBatch],
    cpus: usize,
    name: &str,
) -> Result<Option<Arc<Work>>> {
    let Some(dimensions) = dimensions(plan, records, cpus) else {
        return Ok(None);
    };
    let keys = plan
        .keys
        .iter()
        .map(|key| column(key.expression.as_ref()).expect("admitted key"))
        .collect::<Vec<_>>();
    let routed = route(records, &keys, dimensions.units, name)?;
    if routed.len() < 2 {
        return Ok(None);
    }
    let bytes = scratch_bytes(plan, &routed, dimensions.rows, name)?;
    let scratch = Arc::new(plan.scratch.new_empty());
    scratch
        .try_grow(bytes)
        .map_err(|error| df_error(name, error))?;
    let mut shards = Vec::with_capacity(routed.len());
    for shard in &routed {
        let mut partial = PartialGroups::new(&plan.aggregates, plan.scratch.new_empty(), name)?;
        partial.reserve(shard.record.num_rows(), name)?;
        shards.push(Mutex::new(Some(Shard { partial })));
    }
    Ok(Some(Arc::new(Work {
        records: routed.iter().map(|shard| shard.record.clone()).collect(),
        origins: routed.iter().map(|shard| shard.origin.clone()).collect(),
        keys,
        arguments: plan
            .aggregates
            .iter()
            .map(|expression| {
                column(expression.expressions()[0].as_ref()).expect("admitted SUM argument")
            })
            .collect(),
        shards,
        name: name.into(),
        scratch,
    })))
}

impl Work {
    fn key_arrays(&self, shard: usize) -> Vec<ArrayRef> {
        let record = &self.records[shard];
        self.keys
            .iter()
            .map(|&index| record.column(index).clone())
            .collect()
    }
}

impl ParallelCpuWork for Work {
    type Output = Fragment;

    fn unit_count(&self) -> usize {
        self.shards.len()
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Fragment> {
        #[cfg(test)]
        grouped_key_sharding_tests::before_unit(&self.name, ordinal, stop)?;
        let shard = self.shards[ordinal]
            .lock()
            .map_err(|error| df_error(&self.name, error))?
            .take()
            .expect("owned SUM shard");
        let fragment = self.aggregate_shard(ordinal, shard, stop)?;
        #[cfg(test)]
        grouped_key_sharding_tests::after_unit(&self.name);
        Ok(fragment)
    }
}

impl Work {
    fn aggregate_shard(
        &self,
        ordinal: usize,
        mut shard: Shard,
        stop: &GatherStop,
    ) -> Result<Fragment> {
        let record = &self.records[ordinal];
        let rows = record.num_rows();
        let mut groups = self.local_groups(ordinal)?;
        for start in (0..rows).step_by(CHUNK_ROWS) {
            stop.check()?;
            let chunk = record.slice(start, CHUNK_ROWS.min(rows - start));
            self.aggregate_chunk(&chunk, &mut shard.partial, &mut groups, stop)?;
        }
        stop.check()?;
        let states = shard.partial.emit_states(&self.name)?;
        let PartialGroups { reservation, .. } = shard.partial;
        Ok(Fragment {
            keys: groups.keys,
            states,
            _partial: reservation,
            _scratch: Arc::clone(&self.scratch),
        })
    }

    fn aggregate_chunk(
        &self,
        chunk: &RecordBatch,
        partial: &mut PartialGroups,
        groups: &mut LocalGroups<'_>,
        stop: &GatherStop,
    ) -> Result<()> {
        let keys = self
            .keys
            .iter()
            .map(|&index| chunk.column(index).clone())
            .collect::<Vec<_>>();
        let arguments = self
            .arguments
            .iter()
            .map(|&index| vec![chunk.column(index).clone()])
            .collect::<Vec<_>>();
        let input = LocalInput { keys, arguments };
        let indices = groups.indices(&input, partial, stop)?;
        let filters = Vec::new();
        partial.update(
            &input.arguments,
            &filters,
            &indices,
            groups.keys.len(),
            &self.name,
        )
    }

    fn local_groups(&self, ordinal: usize) -> Result<LocalGroups<'_>> {
        let record = &self.records[ordinal];
        let fields = self
            .keys
            .iter()
            .map(|&index| SortField::new(record.column(index).data_type().clone()))
            .collect();
        let converter = RowConverter::new(fields).map_err(|error| df_error(&self.name, error))?;
        Ok(LocalGroups {
            keys: Vec::with_capacity(record.num_rows()),
            ranks: HashMap::with_capacity_and_hasher(record.num_rows(), RandomState::new()),
            converter,
            shard: ordinal,
            name: &self.name,
        })
    }
}

impl LocalGroups<'_> {
    fn indices(
        &mut self,
        input: &LocalInput,
        partial: &mut PartialGroups,
        stop: &GatherStop,
    ) -> Result<Vec<usize>> {
        let encoded = self
            .converter
            .convert_columns(&input.keys)
            .map_err(|error| df_error(self.name, error))?;
        let mut indices = Vec::with_capacity(encoded.num_rows());
        for row in 0..encoded.num_rows() {
            if row % 1024 == 0 {
                stop.check()?;
            }
            indices.push(self.rank(encoded.row(row).as_ref(), row, partial)?);
        }
        Ok(indices)
    }

    fn rank(&mut self, encoded: &[u8], row: usize, partial: &mut PartialGroups) -> Result<usize> {
        if let Some(&rank) = self.ranks.get(encoded) {
            return Ok(rank);
        }
        let rank = self.keys.len();
        let encoded: Arc<[u8]> = Arc::from(encoded);
        self.ranks.insert(encoded.clone(), rank);
        self.keys.push(LocalKey {
            encoded,
            shard: self.shard,
            row,
        });
        partial.add_slot(rank, self.name)?;
        Ok(rank)
    }
}

pub(super) async fn update(
    plan: &IncrementalSql,
    records: &[RecordBatch],
    candidates: &mut InputCandidates,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<bool> {
    let Some(work) = prepare(
        plan,
        records,
        context.job().gather_owner().parallelism(),
        name,
    )?
    else {
        return Ok(false);
    };
    #[cfg(test)]
    grouped_key_sharding_tests::dispatched(name, work.unit_count());
    let output = submit(&work, plan.scratch.new_empty(), context, name).await?;
    drop(candidates.native.take());
    merge_fragments(plan, &work, &output.value, candidates, context, name).await?;
    context.check_cancelled()?;
    Ok(true)
}

async fn submit(
    work: &Arc<Work>,
    credit: MemoryReservation,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<crate::runtime::streaming::gather_work::WorkOutput<Vec<Fragment>>> {
    let scope = context
        .gather_client(GatherOperatorId::new(name.into()))
        .scope()?;
    let ticket = scope
        .submit_parallel_work(
            Arc::clone(work),
            credit,
            GatherStop::from_job(context.job()),
        )
        .await
        .map_err(|failure| match failure {
            AdmissionFailure::Budget { source, .. } => df_error(name, source),
            AdmissionFailure::Runtime(error) => error,
        })?;
    ticket.finish().await
}

/// Resolves every fragment key to a global candidate in the original
/// first-appearance order of the serial path, then merges each fragment's
/// partial states into the global partial exactly once. Shards own disjoint
/// groups, so no rank is merged twice.
async fn merge_fragments(
    plan: &IncrementalSql,
    work: &Work,
    fragments: &[Fragment],
    candidates: &mut InputCandidates,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<()> {
    let mut order = Vec::new();
    for (fragment, shard) in fragments.iter().zip(0..) {
        for (rank, key) in fragment.keys.iter().enumerate() {
            order.push((u64::from(work.origins[shard][key.row]), shard, rank));
        }
    }
    order.sort_unstable_by_key(|(origin, _, _)| *origin);
    let mut global_ranks = fragments
        .iter()
        .map(|fragment| vec![0_usize; fragment.keys.len()])
        .collect::<Vec<_>>();
    let mut arrays = Vec::new();
    let mut shard = usize::MAX;
    for (index, (origin, shard_index, rank)) in order.iter().enumerate() {
        if index % 1024 == 0 {
            context.check_cancelled()?;
            tokio::task::yield_now().await;
        }
        let fragment = &fragments[*shard_index];
        let key = &fragment.keys[*rank];
        debug_assert_eq!(u64::from(work.origins[*shard_index][key.row]), *origin);
        if shard != key.shard {
            shard = key.shard;
            arrays = work.key_arrays(shard);
        }
        let global = plan.group_rank(&key.encoded, &arrays, key.row, candidates, name)?;
        global_ranks[*shard_index][*rank] = global;
    }
    let partial = candidates.partial.as_mut().expect("grouped SUM partial");
    for (fragment, ranks) in fragments.iter().zip(global_ranks) {
        merge_states(partial, &fragment.states, &ranks, context, name).await?;
    }
    Ok(())
}

async fn merge_states(
    partial: &mut PartialGroups,
    states: &[Vec<ArrayRef>],
    ranks: &[usize],
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<()> {
    for (index, states) in states.iter().enumerate() {
        for (chunk, indices) in ranks.chunks(CHUNK_ROWS).enumerate() {
            context.check_cancelled()?;
            let arrays = states
                .iter()
                .map(|array| array.slice(chunk * CHUNK_ROWS, indices.len()))
                .collect::<Vec<_>>();
            partial.accumulators[index]
                .merge_batch(&arrays, indices, partial.slots.len())
                .map_err(|error| df_error(name, error))?;
            tokio::task::yield_now().await;
        }
    }
    partial.check_size(name)
}
