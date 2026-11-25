use super::{RowLocator, Selection, compact::compact_accepted, run::Run, run::SequenceColumn};
use crate::operator::asof::{AsofJoinSide, StreamAsofJoinSpec, admission, arrow_error, reason};
use crate::operator::keyed::{GroupId, KeySchema, KeyStore, MORSEL_ROWS, VARIABLE_STEP_BYTES};
use crate::{Result, StreamingFailureReason};
use datafusion::arrow::{
    array::{Array, ArrayRef, StringArray},
    buffer::BooleanBuffer,
    compute::{SortColumn, lexsort_to_indices},
    datatypes::SchemaRef,
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[cfg(test)]
thread_local! {
    pub(super) static PENDING_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static INDEX_IDENTITY_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static ADMISSION_IDENTITY_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static PROJECTED_GROUP_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static PROJECTED_COUNT_UPDATES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static COMMITTED_ROW_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static RETIRED_IDENTITY_LOCATORS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static BULK_IDENTITY_RUN_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FINALITY_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SELECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static LATEST_BINARY_SEARCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static CANDIDATE_TIME_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(in crate::operator::asof) fn take_finality_probes() -> usize {
    FINALITY_PROBES.with(|count| count.replace(0))
}

#[cfg(test)]
pub(in crate::operator::asof) fn take_selections() -> usize {
    SELECTIONS.with(|count| count.replace(0))
}

#[cfg(test)]
pub(in crate::operator::asof) fn take_latest_binary_searches() -> usize {
    LATEST_BINARY_SEARCHES.with(|count| count.replace(0))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::operator::asof) struct EvictionSummary {
    pub(in crate::operator::asof) payloads: u64,
    pub(in crate::operator::asof) identities: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::operator::asof) struct Progress {
    pub(in crate::operator::asof) left_watermark: Option<i64>,
    pub(in crate::operator::asof) right_watermark: Option<i64>,
    pub(in crate::operator::asof) left_ended: bool,
    pub(in crate::operator::asof) right_ended: bool,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct IndexCapacities {
    pub(super) pending: u64,
    pub(super) left: Vec<u64>,
    pub(super) right_payloads: Vec<u64>,
    pub(super) right_identities: Vec<u64>,
}

#[derive(Default)]
struct LocatorColumn {
    values: Vec<RowLocator>,
    head: usize,
}

impl LocatorColumn {
    fn len(&self) -> usize {
        self.values.len() - self.head
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn rows(&self) -> &[RowLocator] {
        &self.values[self.head..]
    }

    fn first(&self) -> Option<RowLocator> {
        self.rows().first().copied()
    }

    fn capacity_bytes(&self) -> usize {
        self.values.capacity() * size_of::<RowLocator>()
    }

    fn spare_bytes(&self) -> usize {
        (self.values.capacity() - self.len()) * size_of::<RowLocator>()
    }

    fn drop_prefix(&mut self, count: usize) -> (usize, usize) {
        let before = self.capacity_bytes();
        assert!(count <= self.len());
        self.head += count;
        if self.is_empty() {
            self.values = Vec::new();
            self.head = 0;
        }
        (before, self.capacity_bytes())
    }

    fn retain(&mut self, mut keep: impl FnMut(RowLocator) -> bool) -> (usize, usize) {
        let before = self.capacity_bytes();
        let mut values = std::mem::take(&mut self.values);
        values.drain(..self.head);
        values.retain(|&locator| keep(locator));
        self.values = values.into_boxed_slice().into_vec();
        self.head = 0;
        (before, self.capacity_bytes())
    }

    fn merge(
        &mut self,
        mut incoming: Vec<RowLocator>,
        incoming_sorted: bool,
        mut compare: impl FnMut(RowLocator, RowLocator) -> Ordering,
    ) -> (usize, usize) {
        let before = self.capacity_bytes();
        if incoming.is_empty() {
            return (before, before);
        }
        if !incoming_sorted {
            incoming.sort_unstable_by(|left, right| compare(*left, *right));
        }
        if self.is_empty() {
            // Projection charges the live length, excluding staging vector spare capacity.
            self.values = incoming.into_boxed_slice().into_vec();
            self.head = 0;
            return (before, self.capacity_bytes());
        }
        let existing = self.rows();
        let mut merged = Vec::with_capacity(existing.len() + incoming.len());
        let mut left = 0;
        let mut right = 0;
        if compare(existing[existing.len() - 1], incoming[0]) != Ordering::Greater {
            merged.extend_from_slice(existing);
            left = existing.len();
        }
        while left < existing.len() && right < incoming.len() {
            if compare(existing[left], incoming[right]) == Ordering::Greater {
                merged.push(incoming[right]);
                right += 1;
            } else {
                merged.push(existing[left]);
                left += 1;
            }
        }
        merged.extend_from_slice(&existing[left..]);
        merged.extend_from_slice(&incoming[right..]);
        self.values = merged;
        self.head = 0;
        (before, self.capacity_bytes())
    }
}

enum IncomingIndexRows<'a> {
    Fresh(&'a [RowLocator]),
    Restored {
        payloads: &'a [RowLocator],
        identities: &'a [RowLocator],
    },
}

impl IncomingIndexRows<'_> {
    fn group_counts(&self, run: &Run) -> BTreeMap<usize, [usize; 2]> {
        let mut counts = BTreeMap::<usize, [usize; 2]>::new();
        let mut count = |rows: &[RowLocator], column: usize| {
            for locator in rows {
                #[cfg(test)]
                PROJECTED_GROUP_VISITS.set(PROJECTED_GROUP_VISITS.get() + 1);
                let entry = counts.entry(run.group(*locator).index()).or_default();
                #[cfg(test)]
                PROJECTED_COUNT_UPDATES.set(PROJECTED_COUNT_UPDATES.get() + 1);
                entry[column] += 1;
            }
        };
        match self {
            IncomingIndexRows::Fresh(locators) => count(locators, 0),
            IncomingIndexRows::Restored {
                payloads,
                identities,
            } => {
                count(payloads, 0);
                count(identities, 1);
            }
        }
        counts
    }
}

enum LeftKeyOrder {
    TypedRank,
    GroupId,
}

impl LeftKeyOrder {
    fn matches_ranks(self, ranks: &[u32]) -> bool {
        matches!(self, Self::TypedRank) || ranks.is_sorted()
    }
}

struct PreparedRun {
    run: Run,
    workspace: MemoryReservation,
    run_id: u64,
    next_run: u64,
}

#[derive(Default)]
struct RightBucket {
    payloads: LocatorColumn,
    identities: LocatorColumn,
}

const SELECTION_ROW_BYTES: usize = 72;
const MIN_PROBES_PER_GROUP: usize = 4;
// While probing, only the two locator columns, validity bytes, and group bounds are live.
const _: () = assert!(
    2 * size_of::<RowLocator>()
        + size_of::<bool>()
        + size_of::<Option<ProbeBounds>>().div_ceil(MIN_PROBES_PER_GROUP)
        <= SELECTION_ROW_BYTES
);

#[derive(Clone, Copy)]
struct ProbeBounds {
    end: usize,
    winner_time: Option<i64>,
    next_time: Option<i64>,
}

impl ProbeBounds {
    fn winner(self, rows: &[RowLocator]) -> Option<(RowLocator, i64)> {
        self.winner_time.map(|time| (rows[self.end - 1], time))
    }

    fn advance(&mut self, rows: &[RowLocator], time: i64, right_time: &impl Fn(RowLocator) -> i64) {
        let Some(crossed) = self.next_time.filter(|next| *next <= time) else {
            return;
        };
        self.end += 1;
        self.winner_time = Some(crossed);
        self.next_time = rows.get(self.end).copied().map(right_time);
        if self.next_time.is_some_and(|next| next <= time) {
            // Bound the search when a time jump or a long tie crosses several rows.
            self.end += 1 + visible_right_rows(&rows[self.end + 1..], time, right_time);
            self.winner_time = Some(right_time(rows[self.end - 1]));
            self.next_time = rows.get(self.end).copied().map(right_time);
        }
    }
}

fn visible_right_rows(
    rows: &[RowLocator],
    time: i64,
    right_time: &impl Fn(RowLocator) -> i64,
) -> usize {
    #[cfg(test)]
    LATEST_BINARY_SEARCHES.with(|count| count.set(count.get() + 1));
    rows.partition_point(|row| right_time(*row) <= time)
}

struct ProbeCursors {
    groups: usize,
    bounds: Vec<Option<ProbeBounds>>,
}

impl ProbeCursors {
    fn predecessor(
        &mut self,
        group: usize,
        rows: &[RowLocator],
        time: i64,
        check_tail: bool,
        right_time: impl Fn(RowLocator) -> i64,
    ) -> Option<(RowLocator, i64)> {
        if let Some(bounds) = self.bounds.get_mut(group).and_then(Option::as_mut) {
            // Within this selection each key sees nondecreasing left times.
            bounds.advance(rows, time, &right_time);
            return bounds.winner(rows);
        }
        let &last = rows.last()?;
        if check_tail {
            let last_time = right_time(last);
            if last_time <= time {
                return Some((last, last_time));
            }
        }
        if self.groups == 0 || rows.len() < MIN_PROBES_PER_GROUP {
            let end = visible_right_rows(rows, time, &right_time);
            return end.checked_sub(1).map(|i| (rows[i], right_time(rows[i])));
        }
        if self.bounds.is_empty() {
            self.bounds.resize(self.groups, None);
        }
        let mut bounds = ProbeBounds {
            end: 0,
            winner_time: None,
            next_time: Some(right_time(rows[0])),
        };
        bounds.advance(rows, time, &right_time);
        self.bounds[group] = Some(bounds);
        bounds.winner(rows)
    }
}

pub(in crate::operator::asof) struct ColumnarState {
    pub(super) name: String,
    pub(super) spec: StreamAsofJoinSpec,
    pub(super) schemas: [SchemaRef; 2],
    pub(super) keys: KeyStore,
    reservation: MemoryReservation,
    run_bytes: usize,
    index_bytes: usize,
    pub(super) left_runs: BTreeMap<u64, Run>,
    pub(super) right_runs: BTreeMap<u64, Run>,
    pending: LocatorColumn,
    left_by_key: Vec<LocatorColumn>,
    right_by_key: Vec<RightBucket>,
    #[cfg(feature = "benchmark-internals")]
    legacy_latest_lookup: bool,
    key_ranks: Vec<u32>,
    right_rows: usize,
    right_payloads: usize,
    right_identity_only: usize,
    pub(super) next_run: u64,
    // Older runs may share a single IPC allocation across all payload columns.
    pub(super) first_fresh_run: u64,
}

fn repeats_string_morsel(keys: &StringArray, start: usize, len: usize) -> bool {
    let previous = start - MORSEL_ROWS;
    let offsets = keys.value_offsets();
    let bytes = offsets[start + len] - offsets[start];
    // Large variable keys keep the router's chunked cancellation checks.
    if bytes != offsets[previous + len] - offsets[previous]
        || usize::try_from(bytes).expect("valid UTF-8 offsets") > VARIABLE_STEP_BYTES
    {
        return false;
    }
    keys.slice(previous, len).to_data() == keys.slice(start, len).to_data()
}

fn group_locators(run: &Run, key_count: usize, locators: &[RowLocator]) -> Vec<Vec<RowLocator>> {
    let mut by_key = vec![Vec::new(); key_count];
    for &locator in locators {
        by_key[run.group(locator).index()].push(locator);
    }
    by_key
}

fn compare_identity(runs: &BTreeMap<u64, Run>, left: RowLocator, right: RowLocator) -> Ordering {
    #[cfg(test)]
    INDEX_IDENTITY_COMPARISONS.set(INDEX_IDENTITY_COMPARISONS.get() + 1);
    compare_time_sequence(&runs[&left.run], left, &runs[&right.run], right)
}

fn admission_identity_order(
    left_run: &Run,
    left: RowLocator,
    right_run: &Run,
    right: RowLocator,
) -> Ordering {
    #[cfg(test)]
    ADMISSION_IDENTITY_COMPARISONS.set(ADMISSION_IDENTITY_COMPARISONS.get() + 1);
    compare_time_sequence(left_run, left, right_run, right)
}

fn compare_time_sequence(
    left_run: &Run,
    left: RowLocator,
    right_run: &Run,
    right: RowLocator,
) -> Ordering {
    left_run
        .time(left)
        .cmp(&right_run.time(right))
        .then_with(|| left_run.sequence(left).cmp(right_run.sequence(right)))
}

fn compare_left(
    runs: &BTreeMap<u64, Run>,
    key_ranks: &[u32],
    left: RowLocator,
    right: RowLocator,
) -> Ordering {
    #[cfg(test)]
    PENDING_COMPARISONS.set(PENDING_COMPARISONS.get() + 1);
    runs[&left.run].compare_left_rows(
        left.row as usize,
        &runs[&right.run],
        right.row as usize,
        |group| key_ranks[group.index()],
    )
}

impl ColumnarState {
    pub(in crate::operator::asof) fn new(
        name: &str,
        spec: &StreamAsofJoinSpec,
        schemas: [SchemaRef; 2],
        pool: &Arc<dyn MemoryPool>,
    ) -> Result<Self> {
        let fields = spec
            .left()
            .keys()
            .iter()
            .map(|name| {
                schemas[0]
                    .field_with_name(name)
                    .cloned()
                    .map(Arc::new)
                    .map_err(|error| arrow_error(&error))
            })
            .collect::<Result<Vec<_>>>()?;
        let schema = KeySchema::groupable(fields)?;
        let keys = KeyStore::new(
            schema,
            MemoryConsumer::new(format!("{name}-asof-keys")).register(pool),
        );
        let reservation = MemoryConsumer::new(format!("{name}-asof-runs")).register(pool);
        Ok(Self {
            name: name.into(),
            spec: spec.clone(),
            schemas,
            keys,
            reservation,
            run_bytes: 0,
            index_bytes: 0,
            left_runs: BTreeMap::new(),
            right_runs: BTreeMap::new(),
            pending: LocatorColumn::default(),
            left_by_key: Vec::new(),
            right_by_key: Vec::new(),
            #[cfg(feature = "benchmark-internals")]
            legacy_latest_lookup: false,
            key_ranks: Vec::new(),
            right_rows: 0,
            right_payloads: 0,
            right_identity_only: 0,
            next_run: 0,
            first_fresh_run: 0,
        })
    }

    #[cfg(feature = "benchmark-internals")]
    pub(in crate::operator::asof) fn set_legacy_latest_lookup_for_benchmark(
        &mut self,
        enabled: bool,
    ) {
        self.legacy_latest_lookup = enabled;
    }

    fn adjust_index(&mut self, before: usize, after: usize) {
        self.index_bytes = self.index_bytes - before + after;
    }

    fn sync_reservation(&self) {
        self.reservation.resize(self.run_bytes + self.index_bytes);
    }

    fn state_limit(&self) -> crate::CalcFlowError {
        reason(
            &self.name,
            StreamingFailureReason::AsofStateLimitExceeded,
            "stream_asof_join state limits exceeded",
        )
    }

    fn keyed_admission_error(&self, error: crate::CalcFlowError) -> crate::CalcFlowError {
        match error {
            crate::CalcFlowError::InvalidArgument { field, .. } if field == "keyed_state" => {
                self.state_limit()
            }
            error => error,
        }
    }

    fn projected_charge(
        &self,
        side: usize,
        run: &Run,
        rows: &IncomingIndexRows<'_>,
    ) -> Result<usize> {
        let (payload_rows, identity_rows) = match rows {
            IncomingIndexRows::Fresh(locators) => (locators.len(), locators.len()),
            IncomingIndexRows::Restored {
                payloads,
                identities,
            } => (payloads.len(), identities.len()),
        };
        let mut projected = self.index_bytes;
        if self.key_ranks.len() != self.keys.len() {
            projected = projected
                .checked_sub(self.key_ranks.capacity() * size_of::<u32>())
                .and_then(|bytes| bytes.checked_add(self.keys.len() * size_of::<u32>()))
                .ok_or_else(|| self.state_limit())?;
        }
        if side == 0 && identity_rows != 0 {
            projected = projected
                .checked_sub(self.pending.capacity_bytes())
                .and_then(|bytes| {
                    bytes
                        .checked_add((self.pending.len() + identity_rows) * size_of::<RowLocator>())
                })
                .ok_or_else(|| self.state_limit())?;
        }
        let resident_groups = if side == 0 {
            self.left_by_key.len()
        } else {
            self.right_by_key.len()
        };
        projected = match rows {
            IncomingIndexRows::Fresh(locators) if resident_groups <= locators.len() => {
                self.project_fresh_group_charge(side, run, locators, resident_groups, projected)?
            }
            _ => self.project_sparse_or_restored_charge(side, run, rows, projected)?,
        };
        let run_owners = self.left_runs.len() + self.right_runs.len() + 1;
        let run_owner_bytes = run_owners
            .checked_mul(size_of::<Run>())
            .ok_or_else(|| self.state_limit())?;
        let transient_locator_bytes = payload_rows
            .checked_add(identity_rows)
            .and_then(|rows| rows.checked_mul(size_of::<RowLocator>()))
            .ok_or_else(|| self.state_limit())?;
        self.run_bytes
            .checked_add(run.charge())
            .and_then(|bytes| bytes.checked_add(projected))
            .and_then(|bytes| bytes.checked_add(self.keys.logical_bytes()))
            .and_then(|bytes| bytes.checked_add(run_owner_bytes))
            .and_then(|bytes| bytes.checked_add(transient_locator_bytes))
            .ok_or_else(|| self.state_limit())
    }

    fn project_fresh_group_charge(
        &self,
        side: usize,
        run: &Run,
        locators: &[RowLocator],
        resident_groups: usize,
        mut projected: usize,
    ) -> Result<usize> {
        // A merge with a nonempty incoming column replaces the old capacity with
        // its live length plus the incoming rows. Each touched group reclaims its
        // unused capacity once, regardless of how many new rows it receives.
        let mut remaining = (0..resident_groups)
            .filter(|&group| self.group_spare_bytes(side, group) != 0)
            .count();
        let mut reclaimed = BTreeSet::new();
        for &locator in locators {
            if remaining == 0 {
                break;
            }
            #[cfg(test)]
            PROJECTED_GROUP_VISITS.set(PROJECTED_GROUP_VISITS.get() + 1);
            let group = run.group(locator).index();
            let spare = self.group_spare_bytes(side, group);
            if spare != 0 && reclaimed.insert(group) {
                projected = projected
                    .checked_sub(spare)
                    .ok_or_else(|| self.state_limit())?;
                remaining -= 1;
            }
        }
        // Right rows are indexed once for payloads and once for identities.
        let columns = 1 + usize::from(side == 1);
        locators
            .len()
            .checked_mul(columns * size_of::<RowLocator>())
            .and_then(|bytes| projected.checked_add(bytes))
            .ok_or_else(|| self.state_limit())
    }

    fn group_spare_bytes(&self, side: usize, group: usize) -> usize {
        if side == 0 {
            self.left_by_key
                .get(group)
                .map_or(0, LocatorColumn::spare_bytes)
        } else {
            self.right_by_key.get(group).map_or(0, |bucket| {
                bucket.payloads.spare_bytes() + bucket.identities.spare_bytes()
            })
        }
    }

    fn project_sparse_or_restored_charge(
        &self,
        side: usize,
        run: &Run,
        rows: &IncomingIndexRows<'_>,
        mut projected: usize,
    ) -> Result<usize> {
        let empty = RightBucket::default();
        for (group, [payloads, restored_identities]) in rows.group_counts(run) {
            let identities = match rows {
                IncomingIndexRows::Fresh(_) => payloads,
                IncomingIndexRows::Restored { .. } => restored_identities,
            };
            let next = if side == 0 {
                let column = self.left_by_key.get(group).unwrap_or(&empty.identities);
                replace_projected_column(projected, column, identities)
            } else {
                let bucket = self.right_by_key.get(group).unwrap_or(&empty);
                replace_projected_column(projected, &bucket.payloads, payloads).and_then(|bytes| {
                    replace_projected_column(bytes, &bucket.identities, identities)
                })
            };
            projected = next.ok_or_else(|| self.state_limit())?;
        }
        Ok(projected)
    }

    fn reserve_charge(&self, projected: usize) -> Result<()> {
        let limit = usize::try_from(self.spec.limits().max_state_bytes())
            .map_err(|_| self.state_limit())?;
        if projected > limit {
            return Err(self.state_limit());
        }
        let run_owners = self.left_runs.len() + self.right_runs.len();
        let run_owner_bytes = run_owners
            .checked_mul(size_of::<Run>())
            .ok_or_else(|| self.state_limit())?;
        let current = self
            .run_bytes
            .checked_add(self.index_bytes)
            .and_then(|bytes| bytes.checked_add(self.keys.logical_bytes()))
            .and_then(|bytes| bytes.checked_add(run_owner_bytes))
            .ok_or_else(|| self.state_limit())?;
        let delta = projected.saturating_sub(current);
        self.reservation
            .try_grow(delta)
            .map_err(|_| self.state_limit())
    }

    pub(in crate::operator::asof) fn column_count(&self, side: usize) -> usize {
        self.schemas[side].fields().len()
    }

    pub(super) fn run(&self, side: usize, id: u64) -> &Run {
        let runs = if side == 0 {
            &self.left_runs
        } else {
            &self.right_runs
        };
        &runs[&id]
    }

    pub(in crate::operator::asof) fn admit(
        &mut self,
        record: &RecordBatch,
        input: admission::ValidatedInput,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        check()?;
        if input.accepted == 0 {
            return Ok(());
        }
        let cut = self.keys.cut();
        let reserved = self.reservation.size();
        let declaration = input.side(&self.spec).clone();
        let selected = usize::try_from(input.accepted).expect("validated input row count");
        let accepted =
            admission::AcceptedRows::for_record(record, &declaration, input, Some(selected));
        let installed = self
            .prepare_run(record, &declaration, &accepted, input, check)
            .and_then(|run| self.install_prepared_run(input.index, run));
        match installed {
            Ok(()) => Ok(()),
            Err(error) => {
                self.keys.rollback(cut);
                self.key_ranks.clear();
                self.rebuild_key_order()?;
                self.reservation.resize(reserved);
                Err(error)
            }
        }
    }

    fn prepare_run(
        &mut self,
        record: &RecordBatch,
        declaration: &AsofJoinSide,
        accepted: &admission::AcceptedRows,
        input: admission::ValidatedInput,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<PreparedRun> {
        let selected_rows = accepted.count;
        let (sequence_charge, encoded_extent) =
            declaration
                .sequence_by()
                .iter()
                .fold((0usize, 0usize), |(charge, extent), name| {
                    let index = record.schema().index_of(name).expect("validated sequence");
                    let selected =
                        super::run::logical_array_selection_charge(record.column(index), accepted);
                    (
                        charge.saturating_add(selected),
                        extent.saturating_add(selected.saturating_add(selected_rows * 64)),
                    )
                });
        let preflight = super::run::logical_record_selection_charge(record, accepted)
            .saturating_add(
                sequence_charge
                    .saturating_mul(super::run::sequence_tuple_copy_count([encoded_extent])),
            );
        let workspace = self.preflight_admission(record, declaration, input, preflight, check)?;
        let compacted = compact_accepted(record, accepted)?;
        let record = &compacted;
        let sequences = SequenceColumn::from_names(record, declaration.sequence_by())?;
        let key_arrays = declaration
            .keys()
            .iter()
            .map(|name| {
                record
                    .schema()
                    .index_of(name)
                    .map(|index| record.column(index).clone())
                    .map_err(|error| arrow_error(&error))
            })
            .collect::<Result<Vec<ArrayRef>>>()?;
        let groups = self.route_run_keys(&key_arrays, record.num_rows(), check)?;
        let times = admission::times(record, declaration).values().to_vec();
        let run_id = self.next_run;
        let next_run = self.next_run.checked_add(1).ok_or_else(|| {
            reason(
                &self.name,
                StreamingFailureReason::AsofCounterOverflow,
                "ASOF run identifier space is exhausted",
            )
        })?;
        let run = Run::new(record.clone(), groups, times, sequences);
        Ok(PreparedRun {
            run,
            workspace,
            run_id,
            next_run,
        })
    }

    fn route_run_keys(
        &mut self,
        keys: &[ArrayRef],
        rows: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<GroupId>> {
        let strings = match keys {
            [key] => key
                .as_any()
                .downcast_ref::<StringArray>()
                .filter(|keys| keys.null_count() == 0),
            _ => None,
        };
        let mut groups = Vec::with_capacity(rows);
        for start in (0..rows).step_by(MORSEL_ROWS) {
            let len = MORSEL_ROWS.min(rows - start);
            if start >= MORSEL_ROWS
                && strings.is_some_and(|keys| repeats_string_morsel(keys, start, len))
            {
                let previous = start - MORSEL_ROWS;
                groups.extend_from_within(previous..previous + len);
            } else {
                let arrays = keys
                    .iter()
                    .map(|array| array.slice(start, len))
                    .collect::<Vec<_>>();
                groups.extend(
                    self.keys
                        .route_with_limit(&arrays, usize::MAX, check)
                        .map_err(|error| self.keyed_admission_error(error))?,
                );
            }
            check()?;
        }
        Ok(groups)
    }

    fn install_prepared_run(&mut self, side: usize, prepared: PreparedRun) -> Result<()> {
        let PreparedRun {
            run,
            workspace,
            run_id,
            next_run,
        } = prepared;
        let locators = Run::fresh_locators(run_id, run.rows());
        let left_order = self.left_append_order(side, &run);
        let ordered_by_key =
            left_order.is_some() || self.validate_new_identities(side, &run, &locators)?;
        let _workspace = self.admission_workspace(
            run.payload_record().expect("new run payload"),
            run.sequences(),
            workspace,
        )?;
        let limit = usize::try_from(self.spec.limits().max_state_rows()).unwrap_or(usize::MAX);
        if self
            .state_rows()
            .checked_add(locators.len())
            .is_none_or(|rows| rows > limit)
        {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofStateLimitExceeded,
                "stream_asof_join.limits.max_state_rows exceeded",
            ));
        }
        let projected = self.projected_charge(side, &run, &IncomingIndexRows::Fresh(&locators))?;
        self.reserve_charge(projected)?;
        self.rebuild_key_order()?;
        self.left_by_key
            .resize_with(self.keys.len(), LocatorColumn::default);
        self.right_by_key
            .resize_with(self.keys.len(), RightBucket::default);
        if side == 0 {
            self.run_bytes += run.charge();
            self.left_runs.insert(run_id, run);
            let pending_sorted =
                left_order.is_some_and(|order| order.matches_ranks(&self.key_ranks));
            self.install_left(run_id, locators, ordered_by_key, pending_sorted);
        } else {
            self.right_rows += locators.len();
            self.right_payloads += locators.len();
            self.run_bytes += run.charge();
            self.right_runs.insert(run_id, run);
            self.install_right(run_id, &locators, ordered_by_key);
        }
        self.next_run = next_run;
        self.sync_reservation();
        Ok(())
    }

    fn contains_duplicate_for_budget(
        &self,
        record: &RecordBatch,
        declaration: &AsofJoinSide,
        input: admission::ValidatedInput,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<bool> {
        if admission::contains_duplicate_identity(record, declaration, input, check)? {
            return Ok(true);
        }
        self.contains_resident_duplicate(record, declaration, input, check)
    }

    fn contains_resident_duplicate(
        &self,
        record: &RecordBatch,
        declaration: &AsofJoinSide,
        input: admission::ValidatedInput,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<bool> {
        let key_arrays = declaration
            .keys()
            .iter()
            .map(|name| {
                record
                    .schema()
                    .index_of(name)
                    .map(|index| record.column(index).clone())
                    .map_err(|error| arrow_error(&error))
            })
            .collect::<Result<Vec<_>>>()?;
        let groups = self.keys.probe(&key_arrays, check)?;
        let times = admission::times(record, declaration);
        for (row, group) in groups.into_iter().enumerate() {
            if input.is_late(times.value(row)) {
                continue;
            }
            let Some(group) = group else {
                continue;
            };
            let (runs, resident) = if input.index == 0 {
                (&self.left_runs, self.left_by_key.get(group.index()))
            } else {
                (
                    &self.right_runs,
                    self.right_by_key
                        .get(group.index())
                        .map(|bucket| &bucket.identities),
                )
            };
            let Some(resident) = resident else {
                continue;
            };
            let time = times.value(row);
            let start = resident
                .rows()
                .partition_point(|locator| runs[&locator.run].time(*locator) < time);
            let end = resident
                .rows()
                .partition_point(|locator| runs[&locator.run].time(*locator) <= time);
            if start == end {
                continue;
            }
            let current =
                SequenceColumn::from_names(&record.slice(row, 1), declaration.sequence_by())?;
            check()?;
            if resident.rows()[start..end]
                .iter()
                .any(|locator| runs[&locator.run].sequence(*locator) == current.value(0))
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn preflight_admission(
        &self,
        record: &RecordBatch,
        declaration: &AsofJoinSide,
        input: admission::ValidatedInput,
        bytes: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<MemoryReservation> {
        let limit = usize::try_from(self.spec.limits().max_state_bytes()).unwrap_or(usize::MAX);
        if bytes > limit {
            if self.contains_duplicate_for_budget(record, declaration, input, check)? {
                return Err(self.duplicate_identity());
            }
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofWorkspaceLimitExceeded,
                "ASOF admission workspace exceeds max_state_bytes",
            ));
        }
        let reservation = self.reservation.new_empty();
        if reservation.try_grow(bytes).is_err() {
            if self.contains_duplicate_for_budget(record, declaration, input, check)? {
                return Err(self.duplicate_identity());
            }
            return Err(self.state_limit());
        }
        Ok(reservation)
    }

    fn admission_workspace(
        &self,
        record: &RecordBatch,
        sequences: &SequenceColumn,
        reservation: MemoryReservation,
    ) -> Result<MemoryReservation> {
        let schema = record.schema();
        let schema_bytes = schema.fields().iter().fold(256usize, |total, field| {
            total.saturating_add(field.size()).saturating_add(192)
        });
        let metadata_bytes = schema
            .metadata()
            .iter()
            .fold(0usize, |total, (key, value)| {
                total
                    .saturating_add(key.len())
                    .saturating_add(value.len())
                    .saturating_add(64)
            });
        let bytes = record
            .get_array_memory_size()
            .saturating_add(sequences.memory_size())
            .saturating_add(schema_bytes)
            .saturating_add(metadata_bytes);
        if bytes > usize::try_from(self.spec.limits().max_state_bytes()).unwrap_or(usize::MAX) {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofWorkspaceLimitExceeded,
                "ASOF admission workspace exceeds max_state_bytes",
            ));
        }
        let current = reservation.size();
        if bytes > current {
            reservation
                .try_grow(bytes - current)
                .map_err(|_| self.state_limit())?;
        } else {
            reservation.shrink(current - bytes);
        }
        Ok(reservation)
    }

    fn resident_identities(&self, side: usize, group: usize) -> Option<&LocatorColumn> {
        if side == 0 {
            self.left_by_key.get(group)
        } else {
            self.right_by_key
                .get(group)
                .map(|bucket| &bucket.identities)
        }
    }

    fn left_append_order(&self, side: usize, run: &Run) -> Option<LeftKeyOrder> {
        if side != 0
            || self
                .pending
                .rows()
                .last()
                .is_some_and(|&last| self.left_runs[&last.run].time(last) >= run.times()[0])
        {
            return None;
        }
        if self.key_ranks.len() == self.keys.len() {
            run.strictly_ordered_left(|group| self.key_ranks[group.index()])
                .then_some(LeftKeyOrder::TypedRank)
        } else {
            // Birth order proves uniqueness; only the rebuilt Arrow ranks can certify output order.
            run.strictly_ordered_left(GroupId::index)
                .then_some(LeftKeyOrder::GroupId)
        }
    }

    fn strictly_appends_identities(&self, side: usize, run: &Run, locators: &[RowLocator]) -> bool {
        if self.keys.len() > locators.len() {
            return false;
        }
        let mut last_rows = vec![None; self.keys.len()];
        for &locator in locators {
            let group = run.group(locator).index();
            // The first row for this key must also follow its last resident identity.
            let previous = match last_rows[group].replace(locator.row) {
                Some(row) => Some((run, RowLocator { row, ..locator })),
                None => self
                    .resident_identities(side, group)
                    .and_then(|column| column.rows().last())
                    .map(|&last| (self.run(side, last.run), last)),
            };
            if let Some((previous_run, previous)) = previous
                && admission_identity_order(previous_run, previous, run, locator) != Ordering::Less
            {
                return false;
            }
        }
        true
    }

    /// Returns whether each key's input is already in strictly increasing identity order.
    fn validate_new_identities(
        &self,
        side: usize,
        run: &Run,
        locators: &[RowLocator],
    ) -> Result<bool> {
        if self.strictly_appends_identities(side, run, locators) {
            return Ok(true);
        }
        let mut sorted = locators.to_vec();
        sorted.sort_unstable_by(|left, right| {
            #[cfg(test)]
            ADMISSION_IDENTITY_COMPARISONS.set(ADMISSION_IDENTITY_COMPARISONS.get() + 1);
            run.group(*left)
                .index()
                .cmp(&run.group(*right).index())
                .then_with(|| compare_time_sequence(run, *left, run, *right))
        });
        if sorted
            .windows(2)
            .any(|pair| run.same_identity(pair[0], pair[1]))
        {
            return Err(self.duplicate_identity());
        }
        for locator in locators.iter().copied() {
            let group = run.group(locator).index();
            let Some(resident) = self.resident_identities(side, group) else {
                continue;
            };
            let duplicate = resident
                .rows()
                .binary_search_by(|existing| {
                    admission_identity_order(self.run(side, existing.run), *existing, run, locator)
                })
                .is_ok();
            if duplicate {
                return Err(self.duplicate_identity());
            }
        }
        Ok(false)
    }

    fn duplicate_identity(&self) -> crate::CalcFlowError {
        reason(
            &self.name,
            StreamingFailureReason::AsofDuplicateIdentity,
            "input contains a duplicate key/event-time/sequence identity",
        )
    }

    pub(in crate::operator::asof) fn restore_keys(&mut self, record: &RecordBatch) -> Result<()> {
        for start in (0..record.num_rows()).step_by(MORSEL_ROWS) {
            let len = MORSEL_ROWS.min(record.num_rows() - start);
            let arrays = record
                .columns()
                .iter()
                .map(|array| array.slice(start, len))
                .collect::<Vec<_>>();
            let groups = self
                .keys
                .route_with_limit(&arrays, usize::MAX, &mut || Ok(()))
                .map_err(|error| self.keyed_admission_error(error))?;
            if groups
                .iter()
                .enumerate()
                .any(|(offset, group)| group.index() != start + offset)
            {
                return Err(self.duplicate_identity());
            }
        }
        self.rebuild_key_order()?;
        self.left_by_key
            .resize_with(self.keys.len(), LocatorColumn::default);
        self.right_by_key
            .resize_with(self.keys.len(), RightBucket::default);
        Ok(())
    }

    pub(super) fn restore_run(&mut self, side: usize, run_id: u64, run: Run) -> Result<()> {
        if self.left_runs.contains_key(&run_id) || self.right_runs.contains_key(&run_id) {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofProtocolError,
                "ASOF snapshot repeats a run identifier",
            ));
        }
        let payloads = run
            .payload_liveness()
            .iter()
            .enumerate()
            .filter(|(_, live)| **live)
            .map(|(row, _)| Run::locator(run_id, row))
            .collect::<Vec<_>>();
        let identities = run
            .identity_liveness()
            .iter()
            .enumerate()
            .filter(|(_, live)| **live)
            .map(|(row, _)| Run::locator(run_id, row))
            .collect::<Vec<_>>();
        if identities.is_empty() && payloads.is_empty() {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofProtocolError,
                "ASOF snapshot contains an empty run",
            ));
        }
        let new_rows = if side == 0 {
            identities.len()
        } else {
            (0..run.rows())
                .filter(|row| {
                    let locator = Run::locator(run_id, *row);
                    run.has_payload(locator) || run.has_identity(locator)
                })
                .count()
        };
        let limit = usize::try_from(self.spec.limits().max_state_rows()).unwrap_or(usize::MAX);
        if self
            .state_rows()
            .checked_add(new_rows)
            .is_none_or(|rows| rows > limit)
        {
            return Err(self.state_limit());
        }
        let ordered_by_key = self.validate_new_identities(side, &run, &identities)?;
        let projected = self.projected_charge(
            side,
            &run,
            &IncomingIndexRows::Restored {
                payloads: &payloads,
                identities: &identities,
            },
        )?;
        self.reserve_charge(projected)?;
        self.run_bytes += run.charge();
        if side == 0 {
            self.left_runs.insert(run_id, run);
            self.install_left(run_id, identities, ordered_by_key, false);
        } else {
            self.right_rows += new_rows;
            self.right_payloads += payloads.len();
            self.right_identity_only += identities
                .iter()
                .filter(|locator| !run.has_payload(**locator))
                .count();
            self.right_runs.insert(run_id, run);
            self.install_right_restored(run_id, &payloads, &identities);
        }
        self.sync_reservation();
        Ok(())
    }

    pub(in crate::operator::asof) fn finish_restore(&mut self, next_run: u64) {
        self.first_fresh_run = next_run;
        self.next_run = next_run;
        self.sync_reservation();
    }

    fn rebuild_key_order(&mut self) -> Result<()> {
        if self.key_ranks.len() == self.keys.len() {
            return Ok(());
        }
        let groups = (0..self.keys.len())
            .map(GroupId::new)
            .collect::<Result<Vec<_>>>()?;
        let columns = self
            .keys
            .columns(&groups)?
            .into_iter()
            .map(|values| SortColumn {
                values,
                options: None,
            })
            .collect::<Vec<_>>();
        let order = lexsort_to_indices(&columns, None).map_err(|error| arrow_error(&error))?;
        let key_order = order
            .iter()
            .map(|index| groups[index.expect("non-null key ordinal") as usize])
            .collect::<Vec<_>>();
        let mut key_ranks = vec![0; groups.len()];
        for (rank, group) in key_order.iter().enumerate() {
            key_ranks[group.index()] = u32::try_from(rank).expect("shared key count fits u32");
        }
        let before = self.key_ranks.capacity() * size_of::<u32>();
        self.key_ranks = key_ranks;
        let after = self.key_ranks.capacity() * size_of::<u32>();
        self.adjust_index(before, after);
        Ok(())
    }

    fn install_left(
        &mut self,
        run_id: u64,
        locators: Vec<RowLocator>,
        ordered_by_key: bool,
        pending_sorted: bool,
    ) {
        let by_key = group_locators(&self.left_runs[&run_id], self.keys.len(), &locators);
        for (group, rows) in by_key.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let (before, after) = {
                let runs = &self.left_runs;
                self.left_by_key[group].merge(rows, ordered_by_key, |left, right| {
                    compare_identity(runs, left, right)
                })
            };
            self.adjust_index(before, after);
        }
        let (before, after) = {
            let runs = &self.left_runs;
            let key_ranks = &self.key_ranks;
            self.pending.merge(locators, pending_sorted, |left, right| {
                compare_left(runs, key_ranks, left, right)
            })
        };
        self.adjust_index(before, after);
    }

    fn install_right(&mut self, run_id: u64, locators: &[RowLocator], ordered_by_key: bool) {
        let by_key = group_locators(&self.right_runs[&run_id], self.keys.len(), locators);
        for (group, rows) in by_key.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let (before, after) = {
                let runs = &self.right_runs;
                self.right_by_key[group].payloads.merge(
                    rows.clone(),
                    ordered_by_key,
                    |left, right| compare_identity(runs, left, right),
                )
            };
            self.adjust_index(before, after);
            let (before, after) = {
                let runs = &self.right_runs;
                self.right_by_key[group]
                    .identities
                    .merge(rows, ordered_by_key, |left, right| {
                        compare_identity(runs, left, right)
                    })
            };
            self.adjust_index(before, after);
        }
    }
    fn install_right_restored(
        &mut self,
        run_id: u64,
        payloads: &[RowLocator],
        identities: &[RowLocator],
    ) {
        for (locators, payloads_column) in [(payloads, true), (identities, false)] {
            let by_key = group_locators(&self.right_runs[&run_id], self.keys.len(), locators);
            for (group, rows) in by_key.into_iter().enumerate() {
                if rows.is_empty() {
                    continue;
                }
                let bucket = &mut self.right_by_key[group];
                let column = if payloads_column {
                    &mut bucket.payloads
                } else {
                    &mut bucket.identities
                };
                let (before, after) = {
                    let runs = &self.right_runs;
                    column.merge(rows, false, |left, right| {
                        compare_identity(runs, left, right)
                    })
                };
                self.adjust_index(before, after);
            }
        }
    }

    pub(in crate::operator::asof) fn finalizable_len(
        &self,
        frontier: Option<i64>,
        ended: bool,
        limit: usize,
    ) -> usize {
        let pending = self.pending.rows();
        let count = pending.len().min(limit);
        if ended || count == 0 {
            return count;
        }
        let Some(frontier) = frontier else {
            return 0;
        };
        // Pending rows are ordered by event time before key rank and sequence.
        let ready = |locator: &RowLocator| {
            #[cfg(test)]
            FINALITY_PROBES.set(FINALITY_PROBES.get() + 1);
            self.left_runs[&locator.run].time(*locator) < frontier
        };
        if !ready(&pending[0]) {
            0
        } else if count == 1 || ready(&pending[count - 1]) {
            count
        } else {
            1 + pending[1..count - 1].partition_point(ready)
        }
    }

    pub(in crate::operator::asof) fn selection_scratch_charge(
        &self,
        rows: usize,
        columns: usize,
    ) -> usize {
        let runs = self.left_runs.len() + self.right_runs.len();
        rows.saturating_mul(SELECTION_ROW_BYTES + columns.saturating_mul(8))
            .saturating_add(runs.saturating_mul(128 + columns.saturating_mul(64)))
            .saturating_add(4096)
    }

    fn aligned_right_run(&self, pending: &[RowLocator]) -> Option<u64> {
        #[cfg(feature = "benchmark-internals")]
        if self.legacy_latest_lookup {
            return None;
        }
        if pending.len() != self.right_payloads
            || self.left_runs.len() != 1
            || self.right_runs.len() != 1
        {
            return None;
        }
        let (_, left) = self.left_runs.first_key_value()?;
        let (&id, right) = self.right_runs.first_key_value()?;
        left.has_unique_aligned_times(right, pending).then_some(id)
    }

    /// Builds a selection for an already verified pending prefix.
    pub(in crate::operator::asof) fn select_prefix(&self, count: usize) -> Selection {
        #[cfg(test)]
        SELECTIONS.with(|count| count.set(count.get() + 1));
        let left = self.pending.rows()[..count].to_vec();
        if let Some(run) = self.aligned_right_run(&left) {
            let right = left
                .iter()
                .map(|row| RowLocator { run, row: row.row })
                .collect();
            return Selection {
                left,
                right,
                right_validity: BooleanBuffer::new_set(count),
            };
        }
        let mut right = Vec::with_capacity(count);
        let mut validity = Vec::with_capacity(count);
        let groups = self.right_by_key.len();
        let reuse = groups <= count / MIN_PROBES_PER_GROUP;
        #[cfg(feature = "benchmark-internals")]
        let reuse = reuse && !self.legacy_latest_lookup;
        let mut cursors = ProbeCursors {
            groups: if reuse { groups } else { 0 },
            bounds: Vec::new(),
        };
        for locator in left.iter().copied() {
            let run = &self.left_runs[&locator.run];
            let group = run.group(locator).index();
            let time = run.time(locator);
            let candidate = self.candidate(group, time, &mut cursors);
            validity.push(candidate.is_some());
            right.push(candidate.unwrap_or_default());
        }
        drop(cursors);
        Selection::new(left, right, validity)
    }

    pub(in crate::operator::asof) fn evict(&mut self, progress: Progress) -> EvictionSummary {
        let future = if progress.left_ended {
            i128::MAX
        } else {
            progress.left_watermark.map_or(i128::MIN, i128::from)
        };
        let pending = self.pending.first().map_or(i128::MAX, |locator| {
            i128::from(self.left_runs[&locator.run].time(locator))
        });
        let threshold = future.min(pending);
        let tolerance = i128::from(self.spec.tolerance_micros());
        let identity_bound = if progress.right_ended {
            i128::MAX
        } else {
            progress.right_watermark.map_or(i128::MIN, i128::from)
        };
        // With no left frontier every payload can still match a future left row.
        if threshold == i128::MIN
            && identity_bound != i128::MIN
            && let Some(retired) = self.try_retire_right_identities(identity_bound)
        {
            return retired;
        }
        if !self.right_runs.is_empty()
            && self.right_by_key.iter().all(|bucket| {
                self.column_before(&bucket.payloads, threshold.saturating_sub(tolerance))
                    && self.column_before(&bucket.identities, identity_bound)
            })
        {
            return self.retire_right_runs();
        }
        self.retire_right_prefixes(threshold, tolerance, identity_bound)
    }

    fn retire_right_prefixes(
        &mut self,
        threshold: i128,
        tolerance: i128,
        identity_bound: i128,
    ) -> EvictionSummary {
        let mut summary = EvictionSummary::default();
        for bucket in &mut self.right_by_key {
            let expired = bucket.payloads.rows().partition_point(|locator| {
                i128::from(self.right_runs[&locator.run].time(*locator)) + tolerance < threshold
            });
            let dominated = bucket
                .payloads
                .rows()
                .partition_point(|locator| {
                    i128::from(self.right_runs[&locator.run].time(*locator)) <= threshold
                })
                .saturating_sub(1);
            let payload_count = expired.max(dominated);
            let identity_count = bucket.identities.rows().partition_point(|locator| {
                i128::from(self.right_runs[&locator.run].time(*locator)) < identity_bound
            });
            for &locator in &bucket.payloads.rows()[..payload_count] {
                retire_from_run(
                    &mut self.right_runs,
                    &mut self.run_bytes,
                    locator.run,
                    |run| {
                        let has_identity = run.has_identity(locator);
                        if run.drop_payload(locator) {
                            summary.payloads += 1;
                            self.right_payloads -= 1;
                            if has_identity {
                                self.right_identity_only += 1;
                            } else {
                                self.right_rows -= 1;
                            }
                        }
                    },
                );
            }
            #[cfg(test)]
            RETIRED_IDENTITY_LOCATORS.set(RETIRED_IDENTITY_LOCATORS.get() + identity_count);
            for &locator in &bucket.identities.rows()[..identity_count] {
                retire_from_run(
                    &mut self.right_runs,
                    &mut self.run_bytes,
                    locator.run,
                    |run| {
                        let has_payload = run.has_payload(locator);
                        if run.drop_identity(locator) {
                            summary.identities += 1;
                            if !has_payload {
                                self.right_rows -= 1;
                                self.right_identity_only -= 1;
                            }
                        }
                    },
                );
            }
            let (before, after) = bucket.payloads.drop_prefix(payload_count);
            self.index_bytes = self.index_bytes - before + after;
            let (before, after) = bucket.identities.drop_prefix(identity_count);
            self.index_bytes = self.index_bytes - before + after;
        }
        self.finish_retirement();
        summary
    }

    // Per-key columns are in event-time order, including after restore.
    fn column_before(&self, column: &LocatorColumn, bound: i128) -> bool {
        column
            .rows()
            .last()
            .is_none_or(|locator| i128::from(self.right_runs[&locator.run].time(*locator)) < bound)
    }

    fn try_retire_right_identities(&mut self, bound: i128) -> Option<EvictionSummary> {
        let identities = self.right_by_key.iter().try_fold(0usize, |total, bucket| {
            self.column_before(&bucket.identities, bound)
                .then(|| total + bucket.identities.len())
        })?;
        if identities != 0 {
            // Keep work proportional to the live identities on sparse watermark steps.
            if identities < self.right_runs.len()
                || !self
                    .right_runs
                    .values()
                    .all(Run::can_clear_identities_densely)
            {
                return None;
            }
            self.clear_right_run_identities();
        }
        for bucket in &mut self.right_by_key {
            self.index_bytes -= std::mem::take(&mut bucket.identities).capacity_bytes();
            // Recovery can retain spare capacity even in an empty payload column.
            let (before, after) = bucket.payloads.drop_prefix(0);
            self.index_bytes = self.index_bytes - before + after;
        }
        self.right_rows = self.right_payloads;
        self.right_identity_only = 0;
        self.finish_retirement();
        Some(EvictionSummary {
            payloads: 0,
            identities: identities as u64,
        })
    }

    fn clear_right_run_identities(&mut self) {
        self.right_runs.retain(|_, run| {
            #[cfg(test)]
            BULK_IDENTITY_RUN_VISITS.set(BULK_IDENTITY_RUN_VISITS.get() + 1);
            run.clear_identities();
            if run.is_live() {
                true
            } else {
                self.run_bytes -= run.charge();
                false
            }
        });
    }

    fn retire_right_runs(&mut self) -> EvictionSummary {
        let mut summary = EvictionSummary::default();
        for bucket in &mut self.right_by_key {
            let old = std::mem::take(bucket);
            summary.payloads += old.payloads.len() as u64;
            summary.identities += old.identities.len() as u64;
            self.index_bytes -= old.payloads.capacity_bytes() + old.identities.capacity_bytes();
        }
        for run in std::mem::take(&mut self.right_runs).into_values() {
            self.run_bytes -= run.charge();
        }
        self.right_rows = 0;
        self.right_payloads = 0;
        self.right_identity_only = 0;
        self.finish_retirement();
        summary
    }

    pub(in crate::operator::asof) fn commit(&mut self, selection: &Selection) -> Result<()> {
        if !self.pending.rows().starts_with(&selection.left) {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofProtocolError,
                "ASOF output commit is not the selected pending prefix",
            ));
        }
        if selection.len() == self.pending.len() {
            for run in std::mem::take(&mut self.left_runs).into_values() {
                self.run_bytes -= run.charge();
            }
            for column in &mut self.left_by_key {
                self.index_bytes -= column.capacity_bytes();
                *column = LocatorColumn::default();
            }
        } else {
            self.retire_left_rows(&selection.left);
        }
        let (before, after) = self.pending.drop_prefix(selection.len());
        self.adjust_index(before, after);
        self.finish_retirement();
        Ok(())
    }

    fn retire_left_rows(&mut self, left: &[RowLocator]) {
        for locator in left.iter().copied() {
            #[cfg(test)]
            COMMITTED_ROW_VISITS.set(COMMITTED_ROW_VISITS.get() + 1);
            let run = self
                .left_runs
                .get_mut(&locator.run)
                .expect("selected ASOF left run");
            let before = run.charge();
            run.drop_payload(locator);
            run.drop_identity(locator);
            let after = run.charge();
            self.run_bytes = self.run_bytes - before + after;
        }
        let runs = &self.left_runs;
        for column in &mut self.left_by_key {
            let (before, after) = column.retain(|locator| {
                runs.get(&locator.run)
                    .is_some_and(|run| run.has_identity(locator))
            });
            self.index_bytes = self.index_bytes - before + after;
        }
        self.left_runs.retain(|_, run| {
            let live = run.is_live();
            if !live {
                self.run_bytes -= run.charge();
            }
            live
        });
    }

    fn finish_retirement(&mut self) {
        if self.left_runs.is_empty() && self.right_runs.is_empty() && !self.keys.is_empty() {
            self.keys.clear();
            self.pending = LocatorColumn::default();
            self.left_by_key = Vec::new();
            self.right_by_key = Vec::new();
            self.key_ranks = Vec::new();
            self.index_bytes = 0;
        }
        self.sync_reservation();
    }

    fn candidate(&self, group: usize, time: i64, cursors: &mut ProbeCursors) -> Option<RowLocator> {
        let right_time = |locator: RowLocator| {
            #[cfg(test)]
            CANDIDATE_TIME_READS.set(CANDIDATE_TIME_READS.get() + 1);
            self.right_runs[&locator.run].time(locator)
        };
        let rows = self.right_by_key.get(group)?.payloads.rows();
        #[cfg(feature = "benchmark-internals")]
        let check_tail = !self.legacy_latest_lookup;
        #[cfg(not(feature = "benchmark-internals"))]
        let check_tail = true;
        let (locator, timestamp) =
            cursors.predecessor(group, rows, time, check_tail, right_time)?;
        (i128::from(timestamp) >= i128::from(time) - i128::from(self.spec.tolerance_micros()))
            .then_some(locator)
    }

    pub(in crate::operator::asof) fn state_rows(&self) -> usize {
        self.pending.len() + self.right_rows
    }

    pub(in crate::operator::asof) fn pending_left_rows(&self) -> usize {
        self.pending.len()
    }

    pub(in crate::operator::asof) fn right_payload_rows(&self) -> usize {
        self.right_payloads
    }

    pub(in crate::operator::asof) fn identity_only_rows(&self) -> usize {
        self.right_identity_only
    }

    pub(in crate::operator::asof) fn persistent_bytes(&self) -> usize {
        self.run_bytes
            + self.index_bytes
            + self.keys.logical_bytes()
            + (self.left_runs.len() + self.right_runs.len()) * size_of::<Run>()
    }

    pub(in crate::operator::asof) fn pending_times(&self) -> impl Iterator<Item = i64> + '_ {
        self.pending
            .rows()
            .iter()
            .map(|locator| self.left_runs[&locator.run].time(*locator))
    }

    pub(in crate::operator::asof) fn right_identities(
        &self,
        threshold: i128,
    ) -> impl Iterator<Item = (i64, bool, bool)> + '_ {
        self.right_by_key.iter().flat_map(move |bucket| {
            let witness = bucket
                .payloads
                .rows()
                .partition_point(|locator| {
                    i128::from(self.right_runs[&locator.run].time(*locator)) <= threshold
                })
                .checked_sub(1)
                .and_then(|index| bucket.payloads.rows().get(index).copied());
            bucket.identities.rows().iter().map(move |locator| {
                let run = &self.right_runs[&locator.run];
                let dominated = witness.is_some_and(|witness| {
                    compare_identity(&self.right_runs, witness, *locator) == Ordering::Greater
                });
                (run.time(*locator), run.has_payload(*locator), dominated)
            })
        })
    }

    pub(super) fn index_capacities(&self) -> IndexCapacities {
        IndexCapacities {
            pending: self.pending.capacity_bytes() as u64,
            left: self
                .left_by_key
                .iter()
                .map(|column| column.capacity_bytes() as u64)
                .collect(),
            right_payloads: self
                .right_by_key
                .iter()
                .map(|bucket| bucket.payloads.capacity_bytes() as u64)
                .collect(),
            right_identities: self
                .right_by_key
                .iter()
                .map(|bucket| bucket.identities.capacity_bytes() as u64)
                .collect(),
        }
    }

    pub(super) fn restore_index_capacities(&mut self, capacities: &IndexCapacities) -> Result<()> {
        if capacities.left.len() != self.keys.len()
            || capacities.right_payloads.len() != self.keys.len()
            || capacities.right_identities.len() != self.keys.len()
        {
            return Err(self.state_limit());
        }
        let target = std::iter::once(capacities.pending)
            .chain(capacities.left.iter().copied())
            .chain(capacities.right_payloads.iter().copied())
            .chain(capacities.right_identities.iter().copied())
            .try_fold(
                (self.key_ranks.capacity() * size_of::<u32>()) as u64,
                u64::checked_add,
            )
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| self.state_limit())?;
        if target > self.index_bytes {
            self.reservation
                .try_grow(target - self.index_bytes)
                .map_err(|_| self.state_limit())?;
        }
        restore_locator_capacity(&mut self.pending, capacities.pending)
            .ok_or_else(|| self.state_limit())?;
        for (column, bytes) in self.left_by_key.iter_mut().zip(&capacities.left) {
            if restore_locator_capacity(column, *bytes).is_none() {
                return Err(self.state_limit());
            }
        }
        for ((bucket, payloads), identities) in self
            .right_by_key
            .iter_mut()
            .zip(&capacities.right_payloads)
            .zip(&capacities.right_identities)
        {
            if restore_locator_capacity(&mut bucket.payloads, *payloads).is_none() {
                return Err(self.state_limit());
            }
            if restore_locator_capacity(&mut bucket.identities, *identities).is_none() {
                return Err(self.state_limit());
            }
        }
        self.index_bytes = target;
        self.sync_reservation();
        Ok(())
    }

    pub(in crate::operator::asof) fn is_empty(&self) -> bool {
        self.state_rows() == 0
    }
}

fn retire_from_run(
    runs: &mut BTreeMap<u64, Run>,
    run_bytes: &mut usize,
    id: u64,
    retire: impl FnOnce(&mut Run),
) {
    let std::collections::btree_map::Entry::Occupied(mut entry) = runs.entry(id) else {
        unreachable!("retained ASOF right run");
    };
    *run_bytes -= entry.get().charge();
    retire(entry.get_mut());
    if entry.get().is_live() {
        *run_bytes += entry.get().charge();
    } else {
        entry.remove();
    }
}

fn restore_locator_capacity(column: &mut LocatorColumn, bytes: u64) -> Option<()> {
    let capacity = usize::try_from(bytes).ok()?;
    if !capacity.is_multiple_of(size_of::<RowLocator>()) {
        return None;
    }
    let capacity = capacity / size_of::<RowLocator>();
    if capacity < column.len() {
        return None;
    }
    let mut values = Vec::with_capacity(capacity);
    values.extend_from_slice(column.rows());
    column.values = values;
    column.head = 0;
    Some(())
}

fn replace_projected_column(
    total: usize,
    column: &LocatorColumn,
    incoming: usize,
) -> Option<usize> {
    if incoming == 0 {
        return Some(total);
    }
    let projected = (column.len() + incoming).checked_mul(size_of::<RowLocator>())?;
    total
        .checked_sub(column.capacity_bytes())?
        .checked_add(projected)
}
