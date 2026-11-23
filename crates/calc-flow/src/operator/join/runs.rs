//! Immutable columnar Join runs and selection-driven probe/materialization.

use super::{
    JoinSide, JoinStateLimits, JoinTimeBounds, PayloadRef,
    native_dictionary::{TimedEntry, WindowBounds, WindowCursor, ordered_window_bounds},
    operator_error, output_gather,
};
use crate::operator::keyed::{GroupId, KeyStore, MORSEL_ROWS};
use crate::operator::output_chunk::OutputChunkErrors;
use crate::{
    Batch, BatchMetadata, CalcFlowError, EdgeBudget, EventTime, Result, StreamingFailureReason,
};
use datafusion::arrow::{
    array::{Array, BooleanArray, StringArray, UInt64Array},
    compute::take,
    datatypes::SchemaRef,
    record_batch::{RecordBatch, RecordBatchOptions},
};
#[cfg(test)]
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem::size_of,
    sync::Arc,
};

const OUTPUT_CHUNK_ROWS: usize = 64_000;
const RUN_OWNER_BYTES: usize = size_of::<Run>() + size_of::<LiveRun>() + 64;
const BUCKET_OWNER_BYTES: usize = size_of::<Bucket>() + 64;
const ROW_COLUMN_BYTES: usize =
    size_of::<GroupId>() + size_of::<i64>() + size_of::<u64>() + size_of::<RowLocator>();

#[cfg(test)]
thread_local! {
    pub(super) static IDENTITY_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static MERGE_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn reset_merge_visits() {
    MERGE_VISITS.set(0);
}

#[cfg(test)]
pub(super) fn merge_visits() -> usize {
    MERGE_VISITS.get()
}

#[cfg(test)]
thread_local! {
    pub(super) static GATHER_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static RUN_LOOKUPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn reset_run_lookups() {
    RUN_LOOKUPS.set(0);
}

#[cfg(test)]
pub(super) fn run_lookups() -> usize {
    RUN_LOOKUPS.get()
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct RunId(u64);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct RowLocator {
    run: RunId,
    row: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LocatorEntry {
    time: EventTime,
    row_id: u64,
    locator: RowLocator,
    group: GroupId,
}

impl TimedEntry for LocatorEntry {
    fn event_time(&self) -> EventTime {
        self.time
    }
}

impl LocatorEntry {
    fn identity(self) -> (EventTime, u64) {
        #[cfg(test)]
        IDENTITY_READS.set(IDENTITY_READS.get() + 1);
        (self.time, self.row_id)
    }
}

pub(super) struct RouteInput<'a> {
    payload: &'a RecordBatch,
    key_indices: &'a [usize],
    physical: Vec<usize>,
    times: Vec<EventTime>,
    row_ids: Vec<u64>,
    retain: BooleanArray,
    next_row_id: u64,
}

impl<'a> RouteInput<'a> {
    pub(super) fn new(
        payload: &'a RecordBatch,
        key_indices: &'a [usize],
        physical: Vec<usize>,
        times: Vec<EventTime>,
        row_ids: Vec<u64>,
        retain: BooleanArray,
        next_row_id: u64,
    ) -> Self {
        Self {
            payload,
            key_indices,
            physical,
            times,
            row_ids,
            retain,
            next_row_id,
        }
    }
}

/// One admitted input morsel. Physical positions preserve the source payload;
/// group, time, ID and retain columns use admitted positions.
pub(super) struct Incoming {
    payload: RecordBatch,
    physical: Vec<usize>,
    groups: Vec<GroupId>,
    times: Vec<EventTime>,
    row_ids: Vec<u64>,
    retain: BooleanArray,
    next_row_id: u64,
}

impl Incoming {
    pub(super) fn new(
        payload: RecordBatch,
        physical: Vec<usize>,
        groups: Vec<GroupId>,
        times: Vec<EventTime>,
        row_ids: Vec<u64>,
        retain: BooleanArray,
        next_row_id: u64,
    ) -> Result<Self> {
        let rows = physical.len();
        if rows > MORSEL_ROWS
            || groups.len() != rows
            || times.len() != rows
            || row_ids.len() != rows
            || retain.len() != rows
            || retain.null_count() != 0
            || physical.iter().any(|&row| row >= payload.num_rows())
            || row_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || row_ids.last().is_some_and(|last| next_row_id <= *last)
        {
            return Err(run_error("invalid admitted run columns or row-ID frontier"));
        }
        Ok(Self {
            payload,
            physical,
            groups,
            times,
            row_ids,
            retain,
            next_row_id,
        })
    }

    pub(super) fn route(
        input: RouteInput<'_>,
        keys: &mut KeyStore,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Self> {
        let RouteInput {
            payload,
            key_indices,
            physical,
            times,
            row_ids,
            retain,
            next_row_id,
        } = input;
        let rows = physical.len();
        if rows > MORSEL_ROWS
            || times.len() != rows
            || row_ids.len() != rows
            || retain.len() != rows
            || retain.null_count() != 0
            || physical.iter().any(|&row| row >= payload.num_rows())
            || row_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || row_ids.last().is_some_and(|last| next_row_id <= *last)
        {
            return Err(run_error("invalid admitted run columns or row-ID frontier"));
        }
        let dense = physical.len() == payload.num_rows()
            && physical
                .iter()
                .enumerate()
                .all(|(index, &row)| index == row);
        let arrays = if dense {
            key_indices
                .iter()
                .map(|&index| payload.column(index).clone())
                .collect()
        } else {
            let source_rows = physical
                .iter()
                .map(|&row| u64::try_from(row).map_err(|_| run_error("source row exceeds u64")))
                .collect::<Result<Vec<_>>>()?;
            let source_rows = UInt64Array::from(source_rows);
            key_indices
                .iter()
                .map(|&index| {
                    take(payload.column(index).as_ref(), &source_rows, None)
                        .map_err(|error| run_error(format!("key take failed: {error}")))
                })
                .collect::<Result<Vec<_>>>()?
        };
        if arrays.iter().any(|array| array.null_count() != 0) {
            return Err(run_error("Join run routing received a null key"));
        }
        let cut = keys.cut();
        match keys
            .route_with_limit_cached(&arrays, usize::MAX, check)
            .and_then(|groups| {
                Self::new(
                    payload.clone(),
                    physical,
                    groups,
                    times,
                    row_ids,
                    retain,
                    next_row_id,
                )
            }) {
            Ok(incoming) => Ok(incoming),
            Err(error) => {
                keys.rollback(cut);
                Err(error)
            }
        }
    }

    fn len(&self) -> usize {
        self.physical.len()
    }

    fn retained(&self, row: usize) -> bool {
        self.retain.value(row)
    }

    fn payload_at(&self, row: usize) -> PayloadRef<'_> {
        PayloadRef::Record(&self.payload, self.physical[row])
    }
}

#[derive(Clone)]
enum RunCharge {
    Fixed { row: usize },
    Variable { rows: Vec<usize> },
}

impl RunCharge {
    fn read(record: &RecordBatch) -> Result<Self> {
        let fixed = record.columns().iter().try_fold(0usize, |total, column| {
            if column.null_count() != 0 {
                return None;
            }
            let charge = usize::try_from(super::fixed_cell_charge(column.data_type())?).ok()?;
            total.checked_add(charge)
        });
        if let Some(row) = fixed {
            return Ok(Self::Fixed { row });
        }
        let rows = (0..record.num_rows())
            .map(|row| {
                record.columns().iter().try_fold(0usize, |total, column| {
                    let charge = usize::try_from(super::logical_cell_charge(column.as_ref(), row)?)
                        .map_err(|_| run_error("run charge exceeds the address space"))?;
                    total
                        .checked_add(charge)
                        .ok_or_else(|| run_error("run charge overflowed"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::Variable { rows })
    }

    fn row(&self, row: usize) -> usize {
        match self {
            Self::Fixed { row: charge } => *charge,
            Self::Variable { rows } => rows[row],
        }
    }

    fn total(&self, rows: usize) -> Result<usize> {
        let payload = match self {
            Self::Fixed { row } => row
                .checked_mul(rows)
                .ok_or_else(|| run_error("fixed run charge overflowed"))?,
            Self::Variable { rows: charges } => {
                charges.iter().try_fold(0usize, |total, charge| {
                    total
                        .checked_add(*charge)
                        .ok_or_else(|| run_error("variable run charge overflowed"))
                })?
            }
        };
        let identity = rows
            .checked_mul(ROW_COLUMN_BYTES)
            .ok_or_else(|| run_error("run identity charge overflowed"))?;
        payload
            .checked_add(identity)
            .and_then(|bytes| bytes.checked_add(RUN_OWNER_BYTES))
            .ok_or_else(|| run_error("run owner charge overflowed"))
    }

    fn rows_charge(&self, mut rows: impl Iterator<Item = usize>) -> Result<usize> {
        rows.try_fold(0usize, |total, row| {
            total
                .checked_add(self.row(row))
                .and_then(|total| total.checked_add(ROW_COLUMN_BYTES))
                .ok_or_else(|| run_error("run prefix charge overflowed"))
        })
    }
}

#[derive(Clone)]
struct Run {
    id: RunId,
    payload: RecordBatch,
    groups: Arc<[GroupId]>,
    times: Arc<[i64]>,
    row_ids: Arc<[u64]>,
    order: Arc<[usize]>,
    charge: RunCharge,
    bytes: usize,
}

impl Run {
    fn new(id: RunId, incoming: &Incoming) -> Result<Option<Arc<Self>>> {
        let selected = (0..incoming.len())
            .filter(|&row| incoming.retained(row))
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return Ok(None);
        }
        let dense = selected.len() == incoming.payload.num_rows()
            && selected
                .iter()
                .enumerate()
                .all(|(index, &row)| incoming.physical[row] == index);
        let columns = if dense {
            incoming.payload.columns().to_vec()
        } else {
            let source_rows = UInt64Array::from_iter_values(selected.iter().map(|&row| {
                u64::try_from(incoming.physical[row]).expect("validated usize fits u64")
            }));
            incoming
                .payload
                .columns()
                .iter()
                .map(|column| {
                    take(column.as_ref(), &source_rows, None).map_err(|error| {
                        operator_error("Join runs", &format!("retained take failed: {error}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?
        };
        let payload = RecordBatch::try_new_with_options(
            incoming.payload.schema(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(selected.len())),
        )
        .map_err(|error| operator_error("Join runs", &format!("retained batch failed: {error}")))?;
        let groups = selected
            .iter()
            .map(|&row| incoming.groups[row])
            .collect::<Vec<_>>();
        let times = selected
            .iter()
            .map(|&row| incoming.times[row].as_micros())
            .collect::<Vec<_>>();
        let row_ids = selected
            .iter()
            .map(|&row| incoming.row_ids[row])
            .collect::<Vec<_>>();
        let mut order = (0..selected.len()).collect::<Vec<_>>();
        order.sort_by_key(|&row| (times[row], row_ids[row]));
        let charge = RunCharge::read(&payload)?;
        let bytes = charge.total(selected.len())?;
        Ok(Some(Arc::new(Self {
            id,
            payload,
            groups: groups.into(),
            times: times.into(),
            row_ids: row_ids.into(),
            order: order.into(),
            charge,
            bytes,
        })))
    }

    fn len(&self) -> usize {
        self.row_ids.len()
    }

    fn entries(&self) -> impl Iterator<Item = LocatorEntry> + '_ {
        self.order.iter().map(|&row| LocatorEntry {
            time: EventTime::from_micros(self.times[row]),
            row_id: self.row_ids[row],
            locator: RowLocator {
                run: self.id,
                row: u32::try_from(row).expect("run morsel fits u32"),
            },
            group: self.groups[row],
        })
    }
}

#[derive(Clone)]
struct LiveRun {
    run: Arc<Run>,
    head: usize,
    released_bytes: usize,
}

pub(super) fn route_hot_keys(
    record: &RecordBatch,
    key_indices: &[usize],
    keys: &mut KeyStore,
) -> Result<Vec<GroupId>> {
    let arrays = key_indices
        .iter()
        .map(|&index| record.column(index).clone())
        .collect::<Vec<_>>();
    if let [index] = key_indices
        && let Some(values) = record
            .column(*index)
            .as_any()
            .downcast_ref::<StringArray>()
            .filter(|values| values.null_count() == 0)
    {
        return dictionary_route(values, keys);
    }
    keys.route_with_limit(&arrays, record.num_rows(), &mut || Ok(()))
}

fn dictionary_route(values: &StringArray, keys: &mut KeyStore) -> Result<Vec<GroupId>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let mut unique = hashbrown::HashMap::<&str, usize, ahash::RandomState>::default();
    let mut dictionary = Vec::new();
    let mut indices = Vec::with_capacity(values.len());
    for row in 0..values.len() {
        let value = values.value(row);
        let index = if let Some(&index) = unique.get(value) {
            index
        } else {
            let index = dictionary.len();
            dictionary.push(value);
            unique.insert(value, index);
            index
        };
        indices.push(index);
    }
    let groups = keys.route_with_limit(
        &[Arc::new(StringArray::from(dictionary))],
        values.len(),
        &mut || Ok(()),
    )?;
    Ok(indices.into_iter().map(|index| groups[index]).collect())
}

pub(super) fn hot_batch_charge(
    batch: &Batch,
    key_indices: &[usize],
    keys: &mut KeyStore,
) -> Result<(usize, usize)> {
    let mut run_bytes = 0usize;
    let mut groups = BTreeMap::<GroupId, Vec<usize>>::new();
    let mut expiration = Vec::new();
    for record in batch.table_payload()?.batches() {
        for start in (0..record.num_rows()).step_by(MORSEL_ROWS) {
            let rows = (record.num_rows() - start).min(MORSEL_ROWS);
            let record = record.slice(start, rows);
            run_bytes = run_bytes
                .checked_add(RunCharge::read(&record)?.total(rows)?)
                .ok_or_else(|| run_error("hot state run charge overflowed"))?;
            let routed = route_hot_keys(&record, key_indices, keys)?;
            for group in routed {
                groups.entry(group).or_default().push(0);
                expiration.push(0);
            }
        }
    }
    let bucket_bytes = groups.into_values().try_fold(0usize, |total, entries| {
        let charge = BUCKET_OWNER_BYTES + entries.capacity() * size_of::<LocatorEntry>();
        total.checked_add(charge)
    });
    let bytes = bucket_bytes
        .and_then(|bytes| bytes.checked_add(expiration.capacity() * size_of::<LocatorEntry>()))
        .and_then(|bytes| bytes.checked_add(run_bytes))
        .ok_or_else(|| run_error("hot state owner charge overflowed"))?;
    Ok((batch.num_rows(), bytes))
}

#[derive(Clone)]
struct Bucket {
    entries: Vec<LocatorEntry>,
    head: usize,
}

impl Bucket {
    fn active(&self) -> &[LocatorEntry] {
        &self.entries[self.head..]
    }

    fn charge(&self) -> usize {
        BUCKET_OWNER_BYTES + self.entries.capacity() * size_of::<LocatorEntry>()
    }
}

#[derive(Clone)]
struct Side {
    runs: BTreeMap<RunId, LiveRun>,
    buckets: BTreeMap<GroupId, Bucket>,
    expiration: Vec<LocatorEntry>,
    expiration_head: usize,
    rows: usize,
    bytes: usize,
    next_row_id: u64,
}

impl Side {
    fn empty() -> Self {
        Self {
            runs: BTreeMap::new(),
            buckets: BTreeMap::new(),
            expiration: Vec::new(),
            expiration_head: 0,
            rows: 0,
            bytes: 0,
            next_row_id: 0,
        }
    }

    fn expiration_charge(&self) -> usize {
        self.expiration.capacity() * size_of::<LocatorEntry>()
    }
}

struct SideUpdate {
    runs: Vec<Arc<Run>>,
    buckets: Vec<(GroupId, Bucket)>,
    expiration: Vec<LocatorEntry>,
    rows: usize,
    bytes: usize,
}

impl SideUpdate {
    fn prepare(side: &Side, runs: Vec<Arc<Run>>, limits: JoinStateLimits) -> Result<Self> {
        let mut grouped = BTreeMap::<GroupId, Vec<LocatorEntry>>::new();
        let mut expiration_entries = Vec::new();
        let mut rows = side.rows;
        let mut run_bytes = 0usize;
        for run in &runs {
            rows = rows
                .checked_add(run.len())
                .ok_or_else(|| run_error("state row count overflowed"))?;
            run_bytes = run_bytes
                .checked_add(run.bytes)
                .ok_or_else(|| run_error("state run charge overflowed"))?;
            for entry in run.entries() {
                grouped.entry(entry.group).or_default().push(entry);
                expiration_entries.push(entry);
            }
        }
        if expiration_entries
            .windows(2)
            .any(|pair| pair[0].identity() >= pair[1].identity())
        {
            expiration_entries.sort_unstable_by_key(|entry| entry.identity());
        }
        for entries in grouped.values_mut() {
            if entries
                .windows(2)
                .any(|pair| pair[0].identity() >= pair[1].identity())
            {
                entries.sort_unstable_by_key(|entry| entry.identity());
            }
        }
        let buckets = grouped
            .into_iter()
            .map(|(group, entries)| {
                let merged = side.buckets.get(&group).map_or_else(
                    || merge(&[], &entries),
                    |bucket| merge(bucket.active(), &entries),
                );
                (
                    group,
                    Bucket {
                        entries: merged,
                        head: 0,
                    },
                )
            })
            .collect::<Vec<_>>();
        let expiration = merge(
            &side.expiration[side.expiration_head..],
            &expiration_entries,
        );
        let new_owner_bytes = buckets.iter().try_fold(run_bytes, |total, (_, bucket)| {
            total.checked_add(bucket.charge())
        });
        let new_owner_bytes = new_owner_bytes
            .and_then(|bytes| bytes.checked_add(expiration.capacity() * size_of::<LocatorEntry>()))
            .ok_or_else(|| run_error("state update charge overflowed"))?;
        let replaced_bytes =
            buckets
                .iter()
                .try_fold(side.expiration_charge(), |total, (group, _)| {
                    total.checked_add(side.buckets.get(group).map_or(0, Bucket::charge))
                });
        let replaced_bytes =
            replaced_bytes.ok_or_else(|| run_error("replaced state charge overflowed"))?;
        let bytes = side
            .bytes
            .checked_add(new_owner_bytes)
            .and_then(|bytes| bytes.checked_sub(replaced_bytes))
            .ok_or_else(|| run_error("state byte count overflowed"))?;
        let prospective = side
            .bytes
            .checked_add(new_owner_bytes)
            .ok_or_else(|| run_error("prospective state charge overflowed"))?;
        if u64::try_from(rows).unwrap_or(u64::MAX) > limits.max_state_rows_per_side()
            || u64::try_from(bytes).unwrap_or(u64::MAX) > limits.max_state_bytes_per_side()
            || u64::try_from(prospective).unwrap_or(u64::MAX) > limits.max_state_bytes_per_side()
        {
            return Err(super::operator_reason(
                "Join runs",
                StreamingFailureReason::JoinStateLimitExceeded,
                "input batch state limit exceeded",
            ));
        }
        Ok(Self {
            runs,
            buckets,
            expiration,
            rows,
            bytes,
        })
    }

    fn install(self, side: &mut Side) {
        for run in self.runs {
            side.runs.insert(
                run.id,
                LiveRun {
                    run,
                    head: 0,
                    released_bytes: 0,
                },
            );
        }
        for (group, bucket) in self.buckets {
            side.buckets.insert(group, bucket);
        }
        side.expiration = self.expiration;
        side.expiration_head = 0;
        side.rows = self.rows;
        side.bytes = self.bytes;
    }
}

struct ProbeSpan {
    first: usize,
    position: usize,
    window: std::ops::Range<usize>,
}

struct Probe {
    spans: Vec<ProbeSpan>,
    count: usize,
}

impl Probe {
    fn len(&self) -> usize {
        self.count
    }

    fn windows(
        &self,
        range: std::ops::Range<usize>,
    ) -> impl Iterator<Item = (usize, std::ops::Range<usize>)> + Clone {
        let spans = if range == (0..self.count) {
            &self.spans[..]
        } else {
            let start = self
                .spans
                .partition_point(|span| span.first + span.window.len() <= range.start);
            let end = self.spans.partition_point(|span| span.first < range.end);
            &self.spans[start..end]
        };
        spans.iter().map(move |span| {
            let start = range.start.saturating_sub(span.first);
            let end = (range.end - span.first).min(span.window.len());
            (
                span.position,
                span.window.start + start..span.window.start + end,
            )
        })
    }
}

struct RunPayloads<'a> {
    side: &'a Side,
    by_id: BTreeMap<RunId, &'a RecordBatch>,
    last: Option<(RunId, &'a RecordBatch)>,
}

impl<'a> RunPayloads<'a> {
    fn payload(&mut self, locator: RowLocator) -> PayloadRef<'a> {
        let record = if let Some((id, record)) = self.last
            && id == locator.run
        {
            record
        } else {
            let record = *self.by_id.entry(locator.run).or_insert_with(|| {
                #[cfg(test)]
                RUN_LOOKUPS.with(|count| count.set(count.get() + 1));
                &self.side.runs[&locator.run].run.payload
            });
            self.last = Some((locator.run, record));
            record
        };
        PayloadRef::Record(
            record,
            usize::try_from(locator.row).expect("u32 locator fits usize"),
        )
    }
}

struct Prepared {
    incoming_is_left: bool,
    incoming: Incoming,
    probe: Probe,
}

#[derive(Clone, Copy)]
pub(super) struct RunOutputContext<'a> {
    schema: &'a SchemaRef,
    projection: Option<&'a [usize]>,
    left_width: usize,
    budget: EdgeBudget,
    #[cfg(feature = "benchmark-internals")]
    trace: Option<&'a crate::benchmark_internals::JoinCpuTraceOwner>,
}

impl<'a> RunOutputContext<'a> {
    pub(super) fn new(
        schema: &'a SchemaRef,
        projection: Option<&'a [usize]>,
        left_width: usize,
        budget: EdgeBudget,
    ) -> RunOutputContext<'a> {
        RunOutputContext {
            schema,
            projection,
            left_width,
            budget,
            #[cfg(feature = "benchmark-internals")]
            trace: None,
        }
    }

    #[cfg(feature = "benchmark-internals")]
    pub(super) fn with_trace(
        mut self,
        trace: &'a crate::benchmark_internals::JoinCpuTraceOwner,
    ) -> Self {
        self.trace = Some(trace);
        self
    }
}

/// Complete immutable output for one prepared input morsel.
#[derive(Debug)]
pub(super) struct RunOutput {
    pub(super) records: Vec<RecordBatch>,
    pub(super) sequences: Vec<u64>,
}

pub(super) struct PreparedRuns {
    incoming_side: JoinSide,
    matched_rows: usize,
    row_ids: std::ops::Range<u64>,
    output_sequences: std::ops::Range<u64>,
    outputs: Vec<RunOutput>,
    update: Option<SideUpdate>,
    next_run: u64,
}

impl PreparedRuns {
    pub(super) fn outputs(&self) -> &[RunOutput] {
        &self.outputs
    }

    pub(super) fn matched_rows(&self) -> usize {
        self.matched_rows
    }

    pub(super) fn commit(self, state: &mut RunState) -> Result<Vec<RunOutput>> {
        let Self {
            incoming_side,
            outputs,
            row_ids,
            output_sequences,
            update,
            next_run,
            ..
        } = self;
        if outputs.is_empty() {
            return Ok(outputs);
        }
        if state.next_output != output_sequences.start {
            return Err(run_error("prepared batch output sequence is stale"));
        }
        let side = incoming_side.index();
        if state.sides[side].next_row_id != row_ids.start {
            return Err(run_error("prepared batch row-ID frontier is stale"));
        }
        if let Some(update) = update {
            let expected = u64::try_from(update.runs.len())
                .ok()
                .and_then(|count| state.next_run.checked_add(count));
            if update.runs.first().map(|run| run.id.0) != Some(state.next_run)
                || expected != Some(next_run)
            {
                return Err(run_error("prepared run identity is stale"));
            }
            let ids = update.runs.iter().map(|run| run.id).collect::<Vec<_>>();
            let groups = update
                .buckets
                .iter()
                .map(|(group, _)| *group)
                .collect::<Vec<_>>();
            update.install(&mut state.sides[side]);
            state.dirty.runs[side].extend(ids);
            state.dirty.groups[side].extend(groups);
            state.dirty.expiration[side] = true;
            state.next_run = next_run;
        }
        state.sides[side].next_row_id = row_ids.end;
        state.next_output = output_sequences.end;
        Ok(outputs)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Eviction {
    pub(super) expired_rows: usize,
    pub(super) retired_runs: usize,
}

struct EvictionPlan {
    cut: usize,
    run_cuts: BTreeMap<RunId, usize>,
    group_cuts: BTreeMap<GroupId, usize>,
    bytes: usize,
}

impl EvictionPlan {
    fn new(side: &Side, cut: usize) -> Result<Self> {
        let expired = &side.expiration[side.expiration_head..side.expiration_head + cut];
        let mut run_cuts = BTreeMap::<RunId, usize>::new();
        let mut group_cuts = BTreeMap::<GroupId, usize>::new();
        for entry in expired {
            *run_cuts.entry(entry.locator.run).or_default() += 1;
            *group_cuts.entry(entry.group).or_default() += 1;
        }
        let mut bytes = 0usize;
        for (&id, &count) in &run_cuts {
            let live = side
                .runs
                .get(&id)
                .ok_or_else(|| run_error("eviction references a missing run"))?;
            let new_head = live
                .head
                .checked_add(count)
                .filter(|head| *head <= live.run.len())
                .ok_or_else(|| run_error("eviction run prefix exceeds live rows"))?;
            let charge = if new_head == live.run.len() {
                live.run
                    .bytes
                    .checked_sub(live.released_bytes)
                    .ok_or_else(|| run_error("released run charge exceeds its owner"))?
            } else {
                live.run
                    .charge
                    .rows_charge(live.run.order[live.head..new_head].iter().copied())?
            };
            bytes = bytes
                .checked_add(charge)
                .ok_or_else(|| run_error("eviction byte charge overflowed"))?;
        }
        for (&group, &count) in &group_cuts {
            let bucket = side
                .buckets
                .get(&group)
                .ok_or_else(|| run_error("eviction references a missing key"))?;
            if count > bucket.active().len() {
                return Err(run_error("eviction key prefix exceeds live locators"));
            }
            if count == bucket.active().len() {
                bytes = bytes
                    .checked_add(bucket.charge())
                    .ok_or_else(|| run_error("evicted bucket charge overflowed"))?;
            }
        }
        if cut == side.rows {
            bytes = bytes
                .checked_add(side.expiration_charge())
                .ok_or_else(|| run_error("evicted expiration charge overflowed"))?;
        }
        Ok(Self {
            cut,
            run_cuts,
            group_cuts,
            bytes,
        })
    }
}

#[derive(Default)]
struct DirtyLineage {
    runs: [BTreeSet<RunId>; 2],
    groups: [BTreeSet<GroupId>; 2],
    expiration: [bool; 2],
}

/// Live run state for both Join sides.
pub(super) struct RunState {
    name: String,
    limits: JoinStateLimits,
    bounds: JoinTimeBounds,
    sides: [Side; 2],
    next_run: u64,
    next_output: u64,
    match_limit_failures: u64,
    dirty: DirtyLineage,
}

impl RunState {
    pub(super) fn new(
        name: impl Into<String>,
        limits: JoinStateLimits,
        bounds: JoinTimeBounds,
    ) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(run_error("Join run operator name is empty"));
        }
        Ok(Self {
            name,
            limits,
            bounds,
            sides: [Side::empty(), Side::empty()],
            next_run: 0,
            next_output: 0,
            match_limit_failures: 0,
            dirty: DirtyLineage::default(),
        })
    }

    pub(super) fn rows(&self, side: JoinSide) -> usize {
        self.sides[side.index()].rows
    }

    pub(super) fn expiring_rows(
        &self,
        side: JoinSide,
        extension_micros: u64,
        opposite_watermark: EventTime,
    ) -> usize {
        let state = &self.sides[side.index()];
        state.expiration[state.expiration_head..].partition_point(|entry| {
            i128::from(entry.time.as_micros()) + i128::from(extension_micros)
                < i128::from(opposite_watermark.as_micros())
        })
    }

    pub(super) fn bytes(&self, side: JoinSide) -> usize {
        self.sides[side.index()].bytes
    }

    pub(super) fn finish_native_capture(&mut self) {
        self.dirty = DirtyLineage::default();
    }

    #[cfg(test)]
    pub(super) fn retained(&self, side: JoinSide) -> Vec<(EventTime, u64, PayloadRef<'_>)> {
        self.sides[side.index()]
            .runs
            .values()
            .flat_map(|live| {
                live.run.order[live.head..].iter().map(|&row| {
                    (
                        EventTime::from_micros(live.run.times[row]),
                        live.run.row_ids[row],
                        PayloadRef::Record(&live.run.payload, row),
                    )
                })
            })
            .collect()
    }

    #[cfg(test)]
    pub(super) fn locators(&self, side: JoinSide) -> Vec<(usize, i64, u64, u64, u32)> {
        self.sides[side.index()]
            .buckets
            .iter()
            .flat_map(|(group, bucket)| {
                bucket.active().iter().map(|entry| {
                    (
                        group.index(),
                        entry.time.as_micros(),
                        entry.row_id,
                        entry.locator.run.0,
                        entry.locator.row,
                    )
                })
            })
            .collect()
    }

    pub(super) fn restore_counters(
        &mut self,
        next_row_ids: [u64; 2],
        next_output: u64,
        match_limit_failures: u64,
    ) {
        self.sides[0].next_row_id = next_row_ids[0];
        self.sides[1].next_row_id = next_row_ids[1];
        self.next_output = next_output;
        self.match_limit_failures = match_limit_failures;
    }

    pub(super) fn next_row_id(&self, side: JoinSide) -> u64 {
        self.sides[side.index()].next_row_id
    }

    pub(super) fn next_output_sequence(&self) -> u64 {
        self.next_output
    }

    pub(super) fn match_limit_failures(&self) -> u64 {
        self.match_limit_failures
    }

    pub(super) fn evict(
        &mut self,
        side: JoinSide,
        extension_micros: u64,
        opposite_watermark: EventTime,
    ) -> Result<Eviction> {
        let cut = self.expiring_rows(side, extension_micros, opposite_watermark);
        let state = &self.sides[side.index()];
        let plan = EvictionPlan::new(state, cut)?;
        Ok(self.apply_eviction(side, plan))
    }

    pub(super) fn evict_end(&mut self, side: JoinSide) -> Result<Eviction> {
        let state = &self.sides[side.index()];
        let cut = state.expiration.len() - state.expiration_head;
        let plan = EvictionPlan::new(state, cut)?;
        Ok(self.apply_eviction(side, plan))
    }

    fn apply_eviction(&mut self, side: JoinSide, plan: EvictionPlan) -> Eviction {
        let cut = plan.cut;
        let affected = plan.run_cuts.keys().copied().collect::<Vec<_>>();
        let groups = plan.group_cuts.keys().copied().collect::<Vec<_>>();
        let index = side.index();
        let side = &mut self.sides[index];
        let retired_runs = plan
            .run_cuts
            .iter()
            .filter(|entry| {
                let (id, cut) = *entry;
                let live = &side.runs[id];
                live.head + *cut == live.run.len()
            })
            .count();
        for (id, cut) in plan.run_cuts {
            let emptied = {
                let live = side.runs.get_mut(&id).expect("planned live run");
                let new_head = live.head + cut;
                if new_head == live.run.len() {
                    true
                } else {
                    let released = live
                        .run
                        .charge
                        .rows_charge(live.run.order[live.head..new_head].iter().copied())
                        .expect("planned run charge");
                    live.head = new_head;
                    live.released_bytes += released;
                    false
                }
            };
            if emptied {
                side.runs.remove(&id);
            }
        }
        for (group, cut) in plan.group_cuts {
            let bucket = side.buckets.get_mut(&group).expect("planned key bucket");
            bucket.head += cut;
            if bucket.head == bucket.entries.len() {
                side.buckets.remove(&group);
            }
        }
        side.expiration_head += plan.cut;
        side.rows -= plan.cut;
        side.bytes -= plan.bytes;
        if side.rows == 0 {
            side.runs.clear();
            side.buckets.clear();
            side.expiration = Vec::new();
            side.expiration_head = 0;
            side.bytes = 0;
        }
        if cut > 0 {
            self.dirty.runs[index].extend(affected);
            self.dirty.groups[index].extend(groups);
            self.dirty.expiration[index] = true;
        }
        Eviction {
            expired_rows: cut,
            retired_runs,
        }
    }

    #[cfg(test)]
    pub(super) fn apply(
        &mut self,
        incoming_side: JoinSide,
        incoming: Incoming,
        schema: &SchemaRef,
        projection: Option<&[usize]>,
        left_width: usize,
        budget: EdgeBudget,
    ) -> Result<RunOutput> {
        self.apply_checked(
            incoming_side,
            incoming,
            RunOutputContext::new(schema, projection, left_width, budget),
            &mut || Ok(()),
        )
    }

    #[cfg(test)]
    pub(super) fn apply_checked(
        &mut self,
        incoming_side: JoinSide,
        incoming: Incoming,
        output: RunOutputContext<'_>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<RunOutput> {
        let prepared = self.prepare_batch(incoming_side, vec![incoming], output, check)?;
        let mut outputs = prepared.commit(self)?;
        Ok(outputs.remove(0))
    }

    pub(super) fn prepare_batch(
        &mut self,
        incoming_side: JoinSide,
        incoming: Vec<Incoming>,
        output: RunOutputContext<'_>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<PreparedRuns> {
        check()?;
        let mut matches = 0usize;
        let mut next_run = self.next_run;
        let mut next_row_id = self.next_row_id(incoming_side);
        let mut prepared = Vec::<Prepared>::new();
        let mut runs = Vec::new();
        for incoming in incoming {
            next_row_id = incoming.next_row_id;
            let (next, run) = self.prepare(
                incoming_side,
                incoming,
                RunId(next_run),
                &output,
                check,
                &mut matches,
            )?;
            if let Some(run) = run {
                next_run = next_run
                    .checked_add(1)
                    .ok_or_else(|| run_error("run-ID frontier overflowed"))?;
                runs.push(run);
            }
            prepared.push(next);
        }
        #[cfg(feature = "benchmark-internals")]
        let merge_span = output.trace.map(|trace| {
            trace.stage(
                &self.name,
                "run_merge",
                runs.iter().map(|run| run.len()).sum(),
            )
        });
        let update = if runs.is_empty() {
            None
        } else {
            Some(SideUpdate::prepare(
                &self.sides[incoming_side.index()],
                runs,
                self.limits,
            )?)
        };
        #[cfg(feature = "benchmark-internals")]
        drop(merge_span);
        let mut next_output = self.next_output;
        let mut outputs = Vec::with_capacity(prepared.len());
        for prepared in prepared {
            #[cfg(feature = "benchmark-internals")]
            let materialize_span = output
                .trace
                .map(|trace| trace.stage(&self.name, "run_materialize", prepared.probe.len()));
            let records = prepared.materialize(self, &output, next_output, check)?;
            #[cfg(feature = "benchmark-internals")]
            drop(materialize_span);
            let sequences = output_sequence_range(next_output, records.len())?;
            next_output = sequences.end;
            let sequences = sequences.collect();
            outputs.push(RunOutput { records, sequences });
        }
        check()?;
        Ok(PreparedRuns {
            incoming_side,
            matched_rows: matches,
            row_ids: self.next_row_id(incoming_side)..next_row_id,
            output_sequences: self.next_output..next_output,
            outputs,
            update,
            next_run,
        })
    }

    fn prepare(
        &mut self,
        incoming_side: JoinSide,
        incoming: Incoming,
        id: RunId,
        output: &RunOutputContext<'_>,
        check: &mut dyn FnMut() -> Result<()>,
        matches: &mut usize,
    ) -> Result<(Prepared, Option<Arc<Run>>)> {
        check()?;
        #[cfg(not(feature = "benchmark-internals"))]
        let _ = output;
        if incoming.next_row_id < self.next_row_id(incoming_side) {
            return Err(run_error("Join run row-ID frontier regressed"));
        }
        #[cfg(feature = "benchmark-internals")]
        let probe_span = output
            .trace
            .map(|trace| trace.stage(&self.name, "run_probe", incoming.len()));
        let probe = self.probe(incoming_side, &incoming, check, matches)?;
        #[cfg(feature = "benchmark-internals")]
        drop(probe_span);
        #[cfg(feature = "benchmark-internals")]
        let build_span = output
            .trace
            .map(|trace| trace.stage(&self.name, "run_build", incoming.len()));
        let run = Run::new(id, &incoming)?;
        #[cfg(feature = "benchmark-internals")]
        drop(build_span);
        Ok((
            Prepared {
                incoming_is_left: incoming_side == JoinSide::Left,
                incoming,
                probe,
            },
            run,
        ))
    }

    fn probe(
        &mut self,
        incoming_side: JoinSide,
        incoming: &Incoming,
        check: &mut dyn FnMut() -> Result<()>,
        matches: &mut usize,
    ) -> Result<Probe> {
        check()?;
        let opposite_index = opposite(incoming_side).index();
        let opposite = &self.sides[opposite_index];
        let mut cursors = Vec::<Option<WindowCursor>>::new();
        let mut sparse_cursors = None;
        let mut spans = Vec::new();
        let mut count = 0usize;
        for row in 0..incoming.len() {
            let group = incoming.groups[row];
            let range = super::native_lookup::time_range(
                self.bounds,
                incoming_side == JoinSide::Left,
                incoming.times[row],
            );
            let found = if let Some(bucket) = opposite.buckets.get(&group) {
                let cursor = if group.index() < incoming.len() {
                    if group.index() >= cursors.len() {
                        cursors.resize(group.index() + 1, None);
                    }
                    &mut cursors[group.index()]
                } else {
                    sparse_cursors
                        .get_or_insert_with(hashbrown::HashMap::<_, _, ahash::RandomState>::default)
                        .entry(group)
                        .or_default()
                };
                let found = ordered_window_bounds(bucket.active(), range, *cursor);
                *cursor = Some(WindowCursor::new(range, found));
                found
            } else {
                WindowBounds::EMPTY
            };
            if found.matches() != 0 {
                spans.push(ProbeSpan {
                    first: count,
                    position: row,
                    window: found.range(),
                });
                count = count
                    .checked_add(found.matches())
                    .ok_or_else(|| run_error("match count overflowed"))?;
            }
        }
        *matches = matches
            .checked_add(count)
            .ok_or_else(|| run_error("input batch match count overflowed"))?;
        super::enforce_match_limit(
            *matches,
            &mut self.match_limit_failures,
            self.limits.max_matches_per_input_batch(),
            &self.name,
        )?;
        Ok(Probe { spans, count })
    }
}

fn output_sequence_range(start: u64, count: usize) -> Result<std::ops::Range<u64>> {
    let count = u64::try_from(count).map_err(|_| run_error("output sequence overflowed"))?;
    let end = start
        .checked_add(count)
        .ok_or_else(|| run_error("output sequence overflowed"))?;
    Ok(start..end)
}

impl Prepared {
    fn materialize(
        &self,
        state: &RunState,
        output: &RunOutputContext<'_>,
        first_sequence: u64,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<RecordBatch>> {
        check()?;
        if self.probe.len() == 0 {
            return Ok(Vec::new());
        }
        let opposite_side = if self.incoming_is_left {
            JoinSide::Right
        } else {
            JoinSide::Left
        };
        let mut payloads = RunPayloads {
            side: &state.sides[opposite_side.index()],
            by_id: BTreeMap::new(),
            last: None,
        };
        let ranges = self.ranges(output, &mut payloads, check)?;
        // Check planned ranges here; byte-budget splits are counted after materialization.
        let sequences = output_sequence_range(first_sequence, ranges.len())?;
        let mut records = Vec::with_capacity(ranges.len());
        for (sequence, range) in sequences.zip(ranges) {
            check()?;
            self.materialize_range(
                &state.name,
                output,
                &mut payloads,
                range,
                sequence,
                &mut records,
            )?;
        }
        Ok(records)
    }

    fn ranges(
        &self,
        output: &RunOutputContext<'_>,
        payloads: &mut RunPayloads<'_>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<std::ops::Range<usize>>> {
        let schema = output.schema;
        let budget = output.budget;
        if let Some(row_bytes) = fixed_output_row_bytes(schema) {
            let per_chunk = if row_bytes == 0 {
                budget.max_rows
            } else {
                (budget.max_bytes / row_bytes).min(budget.max_rows)
            }
            .min(OUTPUT_CHUNK_ROWS);
            if per_chunk == 0 {
                return Err(
                    OutputChunkErrors::STREAM_JOIN.over_budget_row(row_bytes, budget.max_bytes)
                );
            }
            return Ok((0..self.probe.len())
                .step_by(per_chunk)
                .map(|start| start..(start + per_chunk).min(self.probe.len()))
                .collect());
        }
        let terminal = schema.fields().iter().try_fold(0usize, |total, field| {
            total.checked_add(crate::operator::row_cost::terminal_offset_width(
                field.data_type(),
            ))
        });
        let terminal = terminal.ok_or_else(|| run_error("output terminal charge overflowed"))?;
        let mut ranges = Vec::new();
        let mut start = 0usize;
        let mut bytes = terminal;
        let side = payloads.side;
        let pairs = self
            .probe
            .windows(0..self.probe.len())
            .flat_map(|(position, window)| {
                let entries = &side.buckets[&self.incoming.groups[position]].active()[window];
                entries.iter().map(move |entry| (position, entry.locator))
            });
        for (index, (position, locator)) in pairs.enumerate() {
            if index % MORSEL_ROWS == 0 {
                check()?;
            }
            let row = self.pair_charge(position, payloads.payload(locator), output)?;
            let rows = index - start;
            let full = rows == budget.max_rows.min(OUTPUT_CHUNK_ROWS)
                || bytes
                    .checked_add(row)
                    .is_none_or(|candidate| candidate > budget.max_bytes);
            if rows != 0 && full {
                ranges.push(start..index);
                start = index;
                bytes = terminal;
            }
            bytes = bytes
                .checked_add(row)
                .ok_or_else(|| run_error("output chunk charge overflowed"))?;
        }
        if start < self.probe.len() {
            ranges.push(start..self.probe.len());
        }
        Ok(ranges)
    }

    fn pair_charge(
        &self,
        position: usize,
        retained: PayloadRef<'_>,
        output: &RunOutputContext<'_>,
    ) -> Result<usize> {
        let schema = output.schema;
        let projection = output.projection;
        let left_width = output.left_width;
        let incoming = self.incoming.payload_at(position);
        let (left, right) = if self.incoming_is_left {
            (incoming, retained)
        } else {
            (retained, incoming)
        };
        schema
            .fields()
            .iter()
            .enumerate()
            .try_fold(0usize, |total, (ordinal, _field)| {
                let logical = projection.map_or(ordinal, |columns| columns[ordinal]);
                let (payload, column) = if logical < left_width {
                    (left, logical)
                } else {
                    (right, logical - left_width)
                };
                let bytes = usize::try_from(super::logical_cell_charge(
                    payload.column(column).as_ref(),
                    payload.offset(),
                )?)
                .map_err(|_| run_error("output row charge exceeds the address space"))?;
                total
                    .checked_add(bytes)
                    .ok_or_else(|| run_error("output row charge overflowed"))
            })
    }

    fn materialize_range(
        &self,
        name: &str,
        output: &RunOutputContext<'_>,
        payloads: &mut RunPayloads<'_>,
        range: std::ops::Range<usize>,
        sequence: u64,
        records: &mut Vec<RecordBatch>,
    ) -> Result<()> {
        let record = self.record_range(name, output, payloads, range.clone())?;
        let metadata = BatchMetadata::new(name, sequence, BTreeMap::new())?;
        let bytes = Batch::table(vec![record.clone()], metadata)?.estimated_bytes()?;
        if range.len() <= output.budget.max_rows && bytes <= output.budget.max_bytes {
            records.push(record);
            return Ok(());
        }
        if range.len() == 1 {
            return Err(
                OutputChunkErrors::STREAM_JOIN.over_budget_row(bytes, output.budget.max_bytes)
            );
        }
        let middle = range.start + range.len() / 2;
        self.materialize_range(
            name,
            output,
            payloads,
            range.start..middle,
            sequence,
            records,
        )?;
        self.materialize_range(name, output, payloads, middle..range.end, sequence, records)
    }

    fn record_range(
        &self,
        name: &str,
        output: &RunOutputContext<'_>,
        payloads: &mut RunPayloads<'_>,
        range: std::ops::Range<usize>,
    ) -> Result<RecordBatch> {
        let schema = output.schema;
        let projection = output.projection;
        let left_width = output.left_width;
        let count = range.len();
        let windows = self.probe.windows(range);
        let incoming_rows = windows.clone().flat_map(|(position, window)| {
            std::iter::repeat_n(self.incoming.payload_at(position), window.len())
        });
        #[cfg(test)]
        let incoming_rows = incoming_rows.inspect(|_| GATHER_VISITS.set(GATHER_VISITS.get() + 1));
        let incoming = output_gather::SideGather::with_capacity(incoming_rows, count);
        let side = payloads.side;
        let opposite = output_gather::SideGather::with_capacity(
            windows
                .flat_map(|(position, window)| {
                    side.buckets[&self.incoming.groups[position]].active()[window].iter()
                })
                .map(|entry| {
                    #[cfg(test)]
                    GATHER_VISITS.set(GATHER_VISITS.get() + 1);
                    payloads.payload(entry.locator)
                }),
            count,
        );
        let (left, right) = if self.incoming_is_left {
            (incoming, opposite)
        } else {
            (opposite, incoming)
        };
        let columns = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(ordinal, field)| {
                let logical = projection.map_or(ordinal, |columns| columns[ordinal]);
                let (gather, column) = if logical < left_width {
                    (left.as_ref(), logical)
                } else {
                    (right.as_ref(), logical - left_width)
                };
                output_gather::column(gather, column, field.data_type())
            })
            .collect::<Result<Vec<_>>>()?;
        RecordBatch::try_new_with_options(
            Arc::clone(schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(count)),
        )
        .map_err(|error| operator_error(name, &format!("run output failed: {error}")))
    }
}

#[derive(Clone)]
pub(super) struct RunStateSnapshot {
    name: String,
    limits: JoinStateLimits,
    bounds: JoinTimeBounds,
    sides: [Side; 2],
    next_run: u64,
    next_output: u64,
    match_limit_failures: u64,
}

impl RunStateSnapshot {
    #[cfg(test)]
    pub(super) fn fingerprint(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"join-columnar-runs-v1");
        digest.update(self.next_run.to_le_bytes());
        digest.update(self.next_output.to_le_bytes());
        for side in &self.sides {
            digest.update(side.rows.to_le_bytes());
            digest.update(side.bytes.to_le_bytes());
            digest.update(side.next_row_id.to_le_bytes());
            for (group, bucket) in &side.buckets {
                digest.update(
                    u64::try_from(group.index())
                        .expect("group index fits u64")
                        .to_le_bytes(),
                );
                for entry in bucket.active() {
                    digest.update(entry.time.as_micros().to_le_bytes());
                    digest.update(entry.row_id.to_le_bytes());
                    digest.update(entry.locator.run.0.to_le_bytes());
                    digest.update(entry.locator.row.to_le_bytes());
                }
            }
        }
        digest.finalize().into()
    }
}

impl RunState {
    #[cfg(test)]
    pub(super) fn snapshot(&self) -> RunStateSnapshot {
        RunStateSnapshot {
            name: self.name.clone(),
            limits: self.limits,
            bounds: self.bounds,
            sides: self.sides.clone(),
            next_run: self.next_run,
            next_output: self.next_output,
            match_limit_failures: self.match_limit_failures,
        }
    }

    pub(super) fn restore(snapshot: RunStateSnapshot) -> Result<Self> {
        validate_snapshot(&snapshot)?;
        Ok(Self {
            name: snapshot.name,
            limits: snapshot.limits,
            bounds: snapshot.bounds,
            sides: snapshot.sides,
            next_run: snapshot.next_run,
            next_output: snapshot.next_output,
            match_limit_failures: snapshot.match_limit_failures,
            dirty: DirtyLineage::default(),
        })
    }
}

#[path = "run_pages.rs"]
mod pages;
pub(super) use pages::NativePageCache;

impl RunState {
    pub(super) fn native_pages(
        &mut self,
        epoch: crate::Epoch,
        cache: &mut NativePageCache,
    ) -> Result<crate::state::v4::native::NativePageSet> {
        pages::capture(self, epoch, cache)
    }

    pub(super) fn restore_native(
        name: &str,
        limits: JoinStateLimits,
        bounds: JoinTimeBounds,
        segments: &BTreeMap<String, crate::StateSegment>,
        keys: &mut KeyStore,
    ) -> Result<Self> {
        pages::restore(name, limits, bounds, segments, keys)
    }
}

fn validate_snapshot(snapshot: &RunStateSnapshot) -> Result<()> {
    for side in &snapshot.sides {
        let mut identities = BTreeSet::new();
        let mut live_rows = 0usize;
        let mut bytes = side.expiration_charge();
        for (&id, live) in &side.runs {
            let released = live
                .run
                .charge
                .rows_charge(live.run.order[..live.head].iter().copied())?;
            if id != live.run.id
                || live.head >= live.run.len()
                || live.released_bytes != released
                || live
                    .run
                    .row_ids
                    .iter()
                    .any(|row_id| *row_id >= side.next_row_id)
            {
                return Err(run_error(
                    "restored run identity, prefix, or frontier is invalid",
                ));
            }
            live_rows += live.run.len() - live.head;
            bytes = bytes
                .checked_add(live.run.bytes - live.released_bytes)
                .ok_or_else(|| run_error("restored run charge overflowed"))?;
            for &row in &live.run.order[live.head..] {
                if !identities.insert(live.run.row_ids[row]) {
                    return Err(run_error("restored row IDs are not unique"));
                }
            }
        }
        let mut indexed = BTreeSet::new();
        for (&group, bucket) in &side.buckets {
            if bucket.head > bucket.entries.len()
                || bucket
                    .active()
                    .windows(2)
                    .any(|pair| pair[0].identity() >= pair[1].identity())
            {
                return Err(run_error("restored key locators are not strictly ordered"));
            }
            bytes = bytes
                .checked_add(bucket.charge())
                .ok_or_else(|| run_error("restored bucket charge overflowed"))?;
            for entry in bucket.active() {
                let live = side
                    .runs
                    .get(&entry.locator.run)
                    .ok_or_else(|| run_error("restored locator references a missing run"))?;
                let row = usize::try_from(entry.locator.row).expect("u32 locator fits usize");
                if row >= live.run.len()
                    || live.run.groups[row] != group
                    || live.run.row_ids[row] != entry.row_id
                    || live.run.times[row] != entry.time.as_micros()
                    || !live.run.order[live.head..].contains(&row)
                    || !indexed.insert((entry.time, entry.row_id, entry.locator))
                {
                    return Err(run_error("restored key locator differs from its run"));
                }
            }
        }
        if side.expiration_head > side.expiration.len()
            || side.expiration[side.expiration_head..]
                .windows(2)
                .any(|pair| pair[0].identity() >= pair[1].identity())
        {
            return Err(run_error(
                "restored expiration locators are not strictly ordered",
            ));
        }
        let expiration = side.expiration[side.expiration_head..]
            .iter()
            .map(|entry| (entry.time, entry.row_id, entry.locator))
            .collect::<BTreeSet<_>>();
        if live_rows != side.rows
            || indexed.len() != side.rows
            || expiration != indexed
            || bytes != side.bytes
        {
            return Err(run_error("restored run accounting or coverage differs"));
        }
    }
    if snapshot
        .sides
        .iter()
        .flat_map(|side| side.runs.keys())
        .max()
        .is_some_and(|id| id.0 >= snapshot.next_run)
    {
        return Err(run_error("restored run-ID frontier regressed"));
    }
    Ok(())
}

fn merge(left: &[LocatorEntry], right: &[LocatorEntry]) -> Vec<LocatorEntry> {
    #[cfg(test)]
    MERGE_VISITS.with(|count| count.set(count.get() + left.len() + right.len()));
    let Some((first, mut newer)) = right.split_first() else {
        return left.to_vec();
    };
    let first_key = first.identity();
    let split = if left.last().is_some_and(|last| last.identity() <= first_key) {
        left.len()
    } else {
        left.partition_point(|old| old.identity() <= first_key)
    };
    let (prefix, mut older) = left.split_at(split);
    let mut merged = Vec::with_capacity(left.len() + right.len());
    merged.extend_from_slice(prefix);
    merged.push(*first);

    while let (Some((old, old_tail)), Some((new, new_tail))) =
        (older.split_first(), newer.split_first())
    {
        if old.identity() <= new.identity() {
            merged.push(*old);
            older = old_tail;
        } else {
            merged.push(*new);
            newer = new_tail;
        }
    }
    merged.extend_from_slice(older);
    merged.extend_from_slice(newer);
    merged
}

fn fixed_output_row_bytes(schema: &SchemaRef) -> Option<usize> {
    schema.fields().iter().try_fold(0usize, |total, field| {
        if field.is_nullable()
            || crate::operator::row_cost::terminal_offset_width(field.data_type()) != 0
        {
            return None;
        }
        total.checked_add(crate::operator::row_cost::fixed_width(field.data_type())?)
    })
}

fn opposite(side: JoinSide) -> JoinSide {
    match side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    }
}

fn run_error(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "join.runs".into(),
        message: message.into(),
    }
}

#[cfg(test)]
#[test]
fn test_fanout_probe_stores_ranges_per_input_row() {
    use super::run_tests::fanout_input;
    let mut state = RunState::new(
        "fanout-probe",
        JoinStateLimits::new(256, 4_000_000, 20_000).unwrap(),
        JoinTimeBounds::from_micros(0, 0).unwrap(),
    )
    .unwrap();
    let retained = fanout_input(128, 0, true);
    let schema = retained.payload.schema();
    state
        .apply(
            JoinSide::Left,
            retained,
            &schema,
            Some(&[0, 1]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    let input = fanout_input(128, 500, false);
    let mut matches = 0;
    let mut result = None;
    let cost = allocation_counter::measure(|| {
        result = Some(
            state
                .probe(JoinSide::Right, &input, &mut || Ok(()), &mut matches)
                .unwrap(),
        );
    });
    assert_eq!(matches, 16_384);
    assert_eq!(result.unwrap().len(), matches);
    println!("128 by 128 probe peak allocation {} bytes", cost.bytes_max);
    assert!(cost.bytes_max < 32 * 1024, "{cost:?}");
}
