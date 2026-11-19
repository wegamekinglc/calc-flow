//! Immutable columnar Join runs and selection-driven probe/materialization.
//!
//! The v4 runner does not yet expose Join state pages. This module is the
//! synchronous compute core; production admission and durable ownership remain
//! at the operator boundary.
#![allow(
    dead_code,
    reason = "P5 run kernels are tested directly until v4 Join state-page wiring lands"
)]

use super::{
    JoinSide, JoinStateLimits, JoinTimeBounds, MatchedPair, PayloadRef,
    matches::Selection,
    native_dictionary::{TimedEntry, WindowBounds, WindowCursor, ordered_window_bounds},
    operator_error, output_gather,
};
use crate::operator::keyed::{GroupId, MORSEL_ROWS};
use crate::operator::output_chunk::OutputChunkErrors;
use crate::{
    Batch, BatchMetadata, CalcFlowError, EdgeBudget, EventTime, Result, StreamingFailureReason,
};
use datafusion::arrow::{
    array::{Array, BooleanArray, UInt64Array},
    compute::take,
    datatypes::SchemaRef,
    record_batch::{RecordBatch, RecordBatchOptions},
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem::size_of,
    sync::Arc,
};

const RUN_OWNER_BYTES: usize = size_of::<Run>() + size_of::<LiveRun>() + 64;
const BUCKET_OWNER_BYTES: usize = size_of::<Bucket>() + 64;
const ROW_COLUMN_BYTES: usize =
    size_of::<GroupId>() + size_of::<i64>() + size_of::<u64>() + size_of::<RowLocator>();

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
        (self.time, self.row_id)
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

    #[allow(
        clippy::too_many_arguments,
        reason = "one admitted input carries payload, routing, retention, cancellation and limits"
    )]
    pub(super) fn route(
        payload: &RecordBatch,
        key_indices: &[usize],
        physical: Vec<usize>,
        times: Vec<EventTime>,
        row_ids: Vec<u64>,
        retain: BooleanArray,
        next_row_id: u64,
        keys: &mut crate::operator::keyed::KeyStore,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Self> {
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
        let source_rows = physical
            .iter()
            .map(|&row| u64::try_from(row).map_err(|_| run_error("source row exceeds u64")))
            .collect::<Result<Vec<_>>>()?;
        let source_rows = UInt64Array::from(source_rows);
        let arrays = key_indices
            .iter()
            .map(|&index| {
                take(payload.column(index).as_ref(), &source_rows, None)
                    .map_err(|error| run_error(format!("key take failed: {error}")))
            })
            .collect::<Result<Vec<_>>>()?;
        if arrays.iter().any(|array| array.null_count() != 0) {
            return Err(run_error("Join run routing received a null key"));
        }
        let cut = keys.cut();
        match keys
            .route_with_limit(&arrays, usize::MAX, check)
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
        let source_rows =
            UInt64Array::from_iter_values(selected.iter().map(|&row| {
                u64::try_from(incoming.physical[row]).expect("validated usize fits u64")
            }));
        let columns = incoming
            .payload
            .columns()
            .iter()
            .map(|column| {
                take(column.as_ref(), &source_rows, None).map_err(|error| {
                    operator_error("Join runs", &format!("retained take failed: {error}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
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
    run: Arc<Run>,
    buckets: Vec<(GroupId, Bucket)>,
    expiration: Vec<LocatorEntry>,
    rows: usize,
    bytes: usize,
}

impl SideUpdate {
    fn prepare(side: &Side, run: Arc<Run>, limits: JoinStateLimits) -> Result<Self> {
        let mut grouped = BTreeMap::<GroupId, Vec<LocatorEntry>>::new();
        for entry in run.entries() {
            grouped.entry(entry.group).or_default().push(entry);
        }
        let buckets = grouped
            .into_iter()
            .map(|(group, entries)| {
                let merged = side
                    .buckets
                    .get(&group)
                    .map_or(entries.clone(), |bucket| merge(bucket.active(), &entries));
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
            &run.entries().collect::<Vec<_>>(),
        );
        let new_owner_bytes = buckets.iter().try_fold(run.bytes, |total, (_, bucket)| {
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
        let rows = side
            .rows
            .checked_add(run.len())
            .ok_or_else(|| run_error("state row count overflowed"))?;
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
            run,
            buckets,
            expiration,
            rows,
            bytes,
        })
    }

    fn install(self, side: &mut Side) {
        side.runs.insert(
            self.run.id,
            LiveRun {
                run: self.run,
                head: 0,
                released_bytes: 0,
            },
        );
        for (group, bucket) in self.buckets {
            side.buckets.insert(group, bucket);
        }
        side.expiration = self.expiration;
        side.expiration_head = 0;
        side.rows = self.rows;
        side.bytes = self.bytes;
    }
}

struct Probe {
    selection: Selection,
    opposite: Vec<RowLocator>,
}

impl Probe {
    fn len(&self) -> usize {
        self.selection.len()
    }
}

struct Prepared {
    incoming_is_left: bool,
    incoming: Incoming,
    probe: Probe,
    update: Option<SideUpdate>,
    next_row_id: u64,
    output_base: u64,
}

/// Complete immutable output for one prepared input morsel.
#[derive(Debug)]
pub(super) struct RunOutput {
    pub(super) records: Vec<RecordBatch>,
    pub(super) sequences: Vec<u64>,
}

pub(super) struct PreparedRuns {
    incoming_side: JoinSide,
    prepared: Vec<Prepared>,
    outputs: Vec<RunOutput>,
}

impl PreparedRuns {
    pub(super) fn outputs(&self) -> &[RunOutput] {
        &self.outputs
    }

    pub(super) fn matched_rows(&self) -> usize {
        self.prepared
            .iter()
            .map(|prepared| prepared.probe.len())
            .sum()
    }

    pub(super) fn commit(self, state: &mut RunState) -> Result<Vec<RunOutput>> {
        if self
            .prepared
            .first()
            .is_some_and(|prepared| prepared.output_base != state.next_output)
        {
            return Err(run_error("prepared batch output sequence is stale"));
        }
        let counts = self
            .outputs
            .iter()
            .map(|output| output.records.len())
            .collect::<Vec<_>>();
        for (prepared, count) in self.prepared.into_iter().zip(counts) {
            if prepared.incoming_is_left != (self.incoming_side == JoinSide::Left) {
                return Err(run_error("prepared batch changed its input side"));
            }
            prepared.commit(state, count)?;
        }
        Ok(self.outputs)
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

/// Live run state for both Join sides.
pub(super) struct RunState {
    name: String,
    limits: JoinStateLimits,
    bounds: JoinTimeBounds,
    sides: [Side; 2],
    next_run: u64,
    next_output: u64,
    match_limit_failures: u64,
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
        })
    }

    pub(super) fn rows(&self, side: JoinSide) -> usize {
        self.sides[side.index()].rows
    }

    pub(super) fn bytes(&self, side: JoinSide) -> usize {
        self.sides[side.index()].bytes
    }

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
        let state = &self.sides[side.index()];
        let cut = state.expiration[state.expiration_head..].partition_point(|entry| {
            i128::from(entry.time.as_micros()) + i128::from(extension_micros)
                < i128::from(opposite_watermark.as_micros())
        });
        let plan = EvictionPlan::new(state, cut)?;
        Ok(Self::apply_eviction(&mut self.sides[side.index()], plan))
    }

    pub(super) fn evict_end(&mut self, side: JoinSide) -> Result<Eviction> {
        let state = &self.sides[side.index()];
        let cut = state.expiration.len() - state.expiration_head;
        let plan = EvictionPlan::new(state, cut)?;
        Ok(Self::apply_eviction(&mut self.sides[side.index()], plan))
    }

    fn apply_eviction(side: &mut Side, plan: EvictionPlan) -> Eviction {
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
            side.expiration.clear();
            side.expiration_head = 0;
            side.bytes = 0;
        }
        Eviction {
            expired_rows: plan.cut,
            retired_runs,
        }
    }

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
            schema,
            projection,
            left_width,
            budget,
            &mut || Ok(()),
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one admitted morsel needs its output context"
    )]
    pub(super) fn apply_checked(
        &mut self,
        incoming_side: JoinSide,
        incoming: Incoming,
        schema: &SchemaRef,
        projection: Option<&[usize]>,
        left_width: usize,
        budget: EdgeBudget,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<RunOutput> {
        let prepared = self.prepare_batch(
            incoming_side,
            vec![incoming],
            schema,
            projection,
            left_width,
            budget,
            check,
        )?;
        let mut outputs = prepared.commit(self)?;
        Ok(outputs.remove(0))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one input batch carries all its morsels and the output context"
    )]
    pub(super) fn prepare_batch(
        &mut self,
        incoming_side: JoinSide,
        incoming: Vec<Incoming>,
        schema: &SchemaRef,
        projection: Option<&[usize]>,
        left_width: usize,
        budget: EdgeBudget,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<PreparedRuns> {
        check()?;
        let mut matches = 0usize;
        let mut prepared = incoming
            .into_iter()
            .map(|incoming| {
                let prepared = self.prepare(incoming_side, incoming, check)?;
                matches = matches
                    .checked_add(prepared.probe.len())
                    .ok_or_else(|| run_error("input batch match count overflowed"))?;
                Ok(prepared)
            })
            .collect::<Result<Vec<_>>>()?;
        super::enforce_match_limit(
            matches,
            &mut self.match_limit_failures,
            self.limits.max_matches_per_input_batch(),
            &self.name,
        )?;
        let mut next_output = self.next_output;
        let mut outputs = Vec::with_capacity(prepared.len());
        for prepared in &mut prepared {
            prepared.output_base = next_output;
            let records =
                prepared.materialize(self, schema, projection, left_width, budget, check)?;
            let count = u64::try_from(records.len())
                .map_err(|_| run_error("output sequence overflowed"))?;
            next_output = next_output
                .checked_add(count)
                .ok_or_else(|| run_error("output sequence overflowed"))?;
            let sequences = (0..count)
                .map(|offset| prepared.output_base + offset)
                .collect();
            outputs.push(RunOutput { records, sequences });
        }
        check()?;
        Ok(PreparedRuns {
            incoming_side,
            prepared,
            outputs,
        })
    }

    fn prepare(
        &mut self,
        incoming_side: JoinSide,
        incoming: Incoming,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Prepared> {
        check()?;
        if incoming.next_row_id < self.next_row_id(incoming_side) {
            return Err(run_error("Join run row-ID frontier regressed"));
        }
        let probe = self.probe(incoming_side, &incoming, check)?;
        let id = RunId(self.next_run);
        let run = Run::new(id, &incoming)?;
        let update = run
            .as_ref()
            .map(|run| {
                SideUpdate::prepare(
                    &self.sides[incoming_side.index()],
                    Arc::clone(run),
                    self.limits,
                )
            })
            .transpose()?;
        if run.is_some() {
            self.next_run
                .checked_add(1)
                .ok_or_else(|| run_error("run-ID frontier overflowed"))?;
        }
        Ok(Prepared {
            incoming_is_left: incoming_side == JoinSide::Left,
            next_row_id: incoming.next_row_id,
            incoming,
            probe,
            update,
            output_base: self.next_output,
        })
    }

    fn probe(
        &mut self,
        incoming_side: JoinSide,
        incoming: &Incoming,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Probe> {
        check()?;
        let opposite = &self.sides[opposite(incoming_side).index()];
        let mut cursors = Vec::<Option<WindowCursor>>::new();
        let mut bounds = Vec::with_capacity(incoming.len());
        let mut count = 0usize;
        for row in 0..incoming.len() {
            let group = incoming.groups[row];
            if cursors.len() <= group.index() {
                cursors.resize(group.index() + 1, None);
            }
            let range = super::native_lookup::time_range(
                self.bounds,
                incoming_side == JoinSide::Left,
                incoming.times[row],
            );
            let found = opposite
                .buckets
                .get(&group)
                .map_or(WindowBounds::EMPTY, |bucket| {
                    ordered_window_bounds(bucket.active(), range, cursors[group.index()])
                });
            if opposite.buckets.contains_key(&group) {
                cursors[group.index()] = Some(WindowCursor::new(range, found));
            }
            count = count
                .checked_add(found.matches())
                .ok_or_else(|| run_error("match count overflowed"))?;
            bounds.push(found);
        }
        let mut selection = Selection::with_capacity(count, incoming.len());
        let mut selected = Vec::with_capacity(count);
        for (row, found) in bounds.into_iter().enumerate() {
            if found.matches() == 0 {
                continue;
            }
            let group = incoming.groups[row];
            let bucket = opposite.buckets.get(&group).expect("probed bucket");
            let start = selected.len();
            selected.extend(
                bucket.active()[found.range()]
                    .iter()
                    .map(|entry| entry.locator),
            );
            selection.append_run(row, start..selected.len());
        }
        debug_assert_eq!(selected.len(), count);
        Ok(Probe {
            selection,
            opposite: selected,
        })
    }

    fn payload(&self, side: JoinSide, locator: RowLocator) -> PayloadRef<'_> {
        let run = &self.sides[side.index()].runs[&locator.run].run;
        PayloadRef::Record(
            &run.payload,
            usize::try_from(locator.row).expect("u32 locator fits usize"),
        )
    }

    fn output_sequences(&self, count: usize) -> Result<Vec<u64>> {
        let count = u64::try_from(count).map_err(|_| run_error("output sequence overflowed"))?;
        self.next_output
            .checked_add(count)
            .ok_or_else(|| run_error("output sequence overflowed"))?;
        Ok((0..count).map(|offset| self.next_output + offset).collect())
    }
}

impl Prepared {
    fn materialize(
        &self,
        state: &RunState,
        schema: &SchemaRef,
        projection: Option<&[usize]>,
        left_width: usize,
        budget: EdgeBudget,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<RecordBatch>> {
        check()?;
        if self.probe.len() == 0 {
            return Ok(Vec::new());
        }
        let ranges = self.ranges(state, schema, projection, left_width, budget, check)?;
        // Validate the complete sequence range before gathering any output.
        drop(state.output_sequences(ranges.len())?);
        let mut records = Vec::with_capacity(ranges.len());
        for (ordinal, range) in ranges.into_iter().enumerate() {
            check()?;
            self.materialize_range(
                state,
                schema,
                projection,
                left_width,
                budget,
                range,
                self.output_base + ordinal as u64,
                &mut records,
            )?;
        }
        Ok(records)
    }

    fn ranges(
        &self,
        state: &RunState,
        schema: &SchemaRef,
        projection: Option<&[usize]>,
        left_width: usize,
        budget: EdgeBudget,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<std::ops::Range<usize>>> {
        if let Some(row_bytes) = fixed_output_row_bytes(schema) {
            let per_chunk = if row_bytes == 0 {
                budget.max_rows
            } else {
                (budget.max_bytes / row_bytes).min(budget.max_rows)
            }
            .min(MORSEL_ROWS);
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
        for (index, pair) in self.probe.selection.iter().enumerate() {
            if index % MORSEL_ROWS == 0 {
                check()?;
            }
            let row = self.pair_charge(state, pair, schema, projection, left_width)?;
            let rows = index - start;
            let full = rows == budget.max_rows.min(MORSEL_ROWS)
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
        state: &RunState,
        pair: MatchedPair,
        schema: &SchemaRef,
        projection: Option<&[usize]>,
        left_width: usize,
    ) -> Result<usize> {
        let incoming_side = if self.incoming_is_left {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        let incoming = self.incoming.payload_at(pair.pos);
        let retained = state.payload(
            opposite(incoming_side),
            self.probe.opposite[pair.opposite_index],
        );
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

    #[allow(
        clippy::too_many_arguments,
        reason = "one output range needs its full source context"
    )]
    fn materialize_range(
        &self,
        state: &RunState,
        schema: &SchemaRef,
        projection: Option<&[usize]>,
        left_width: usize,
        budget: EdgeBudget,
        range: std::ops::Range<usize>,
        sequence: u64,
        records: &mut Vec<RecordBatch>,
    ) -> Result<()> {
        let record = self.record_range(state, schema, projection, left_width, range.clone())?;
        let metadata = BatchMetadata::new(&state.name, sequence, BTreeMap::new())?;
        let bytes = Batch::table(vec![record.clone()], metadata)?.estimated_bytes()?;
        if range.len() <= budget.max_rows && bytes <= budget.max_bytes {
            records.push(record);
            return Ok(());
        }
        if range.len() == 1 {
            return Err(OutputChunkErrors::STREAM_JOIN.over_budget_row(bytes, budget.max_bytes));
        }
        let middle = range.start + range.len() / 2;
        self.materialize_range(
            state,
            schema,
            projection,
            left_width,
            budget,
            range.start..middle,
            sequence,
            records,
        )?;
        self.materialize_range(
            state,
            schema,
            projection,
            left_width,
            budget,
            middle..range.end,
            sequence,
            records,
        )
    }

    fn record_range(
        &self,
        state: &RunState,
        schema: &SchemaRef,
        projection: Option<&[usize]>,
        left_width: usize,
        range: std::ops::Range<usize>,
    ) -> Result<RecordBatch> {
        let incoming_side = if self.incoming_is_left {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        let opposite_side = opposite(incoming_side);
        let incoming = output_gather::SideGather::new(
            self.probe
                .selection
                .iter()
                .skip(range.start)
                .take(range.len())
                .map(|pair| self.incoming.payload_at(pair.pos)),
        );
        let opposite = output_gather::SideGather::new(
            self.probe
                .selection
                .iter()
                .skip(range.start)
                .take(range.len())
                .map(|pair| {
                    let locator = self.probe.opposite[pair.opposite_index];
                    state.payload(opposite_side, locator)
                }),
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
            &RecordBatchOptions::new().with_row_count(Some(range.len())),
        )
        .map_err(|error| operator_error(&state.name, &format!("run output failed: {error}")))
    }

    fn commit(self, state: &mut RunState, outputs: usize) -> Result<()> {
        if self.output_base != state.next_output {
            return Err(run_error("prepared output sequence is stale"));
        }
        let incoming_side = if self.incoming_is_left {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        if let Some(update) = self.update {
            state.next_run = state
                .next_run
                .checked_add(1)
                .ok_or_else(|| run_error("run-ID frontier overflowed"))?;
            update.install(&mut state.sides[incoming_side.index()]);
        }
        state.sides[incoming_side.index()].next_row_id = self.next_row_id;
        let outputs = u64::try_from(outputs).map_err(|_| run_error("output count overflowed"))?;
        state.next_output = state
            .next_output
            .checked_add(outputs)
            .ok_or_else(|| run_error("output sequence overflowed"))?;
        Ok(())
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
        })
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
    let mut merged = Vec::with_capacity(left.len() + right.len());
    let (mut left, mut right) = (left.iter().peekable(), right.iter().peekable());
    loop {
        match (left.peek(), right.peek()) {
            (Some(a), Some(b)) if a.identity() <= b.identity() => {
                merged.push(**a);
                left.next();
            }
            (Some(_), Some(_)) => {
                merged.push(**right.peek().expect("right entry"));
                right.next();
            }
            (Some(_), None) => {
                merged.extend(left.copied());
                break;
            }
            (None, Some(_)) => {
                merged.extend(right.copied());
                break;
            }
            (None, None) => break,
        }
    }
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
