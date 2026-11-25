use super::{RowLocator, Selection, run::Run, run::SequenceColumn};
use crate::operator::asof::{
    AsofJoinSide, AsofLatePolicy, StreamAsofJoinSpec, admission, arrow_error, reason,
};
use crate::operator::keyed::{GroupId, KeySchema, KeyStore, MORSEL_ROWS};
use crate::{Result, StreamingFailureReason};
use datafusion::arrow::{
    array::{ArrayRef, UInt32Array},
    compute::{SortColumn, lexsort_to_indices, take},
    datatypes::SchemaRef,
    error::ArrowError,
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
    pub(super) static ADMISSION_IDENTITY_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static PROJECTED_GROUP_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static PROJECTED_COUNT_UPDATES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
pub(in crate::operator::asof) struct AdmissionSummary {
    pub(in crate::operator::asof) accepted: u64,
    pub(in crate::operator::asof) late: u64,
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
        mut compare: impl FnMut(RowLocator, RowLocator) -> Ordering,
    ) -> (usize, usize) {
        let before = self.capacity_bytes();
        if incoming.is_empty() {
            return (before, before);
        }
        incoming.sort_unstable_by(|left, right| compare(*left, *right));
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
// While probing, only the two locator columns, validity bytes, and group offsets are live.
const _: () = assert!(
    2 * size_of::<RowLocator>()
        + size_of::<bool>()
        + size_of::<usize>().div_ceil(MIN_PROBES_PER_GROUP)
        <= SELECTION_ROW_BYTES
);

struct ProbeOffsets {
    groups: usize,
    ends: Vec<usize>,
}

impl ProbeOffsets {
    fn visible_end(
        &mut self,
        group: usize,
        rows: &[RowLocator],
        visible: impl Fn(&RowLocator) -> bool,
    ) -> usize {
        let search = |suffix: &[RowLocator]| {
            #[cfg(test)]
            LATEST_BINARY_SEARCHES.with(|count| count.set(count.get() + 1));
            suffix.partition_point(&visible)
        };
        if self.groups == 0 || rows.len() < MIN_PROBES_PER_GROUP {
            return search(rows);
        }
        if self.ends.is_empty() {
            self.ends = vec![0; self.groups];
        }
        let end = &mut self.ends[group];
        // Pending time is nondecreasing per group. Bound the walk when time jumps.
        let suffix = &rows[*end..];
        let adjacent = suffix.len().min(2);
        *end += suffix[..adjacent]
            .iter()
            .position(|row| !visible(row))
            .unwrap_or_else(|| adjacent + search(&suffix[adjacent..]));
        *end
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

fn compare_identity(runs: &BTreeMap<u64, Run>, left: RowLocator, right: RowLocator) -> Ordering {
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
    let left_run = &runs[&left.run];
    let right_run = &runs[&right.run];
    left_run
        .time(left)
        .cmp(&right_run.time(right))
        .then_with(|| {
            key_ranks[left_run.group(left).index()].cmp(&key_ranks[right_run.group(right).index()])
        })
        .then_with(|| left_run.sequence(left).cmp(right_run.sequence(right)))
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
        let (payload_rows, identity_rows) = match rows {
            IncomingIndexRows::Fresh(rows) => {
                count(rows, 0);
                (rows.len(), rows.len())
            }
            IncomingIndexRows::Restored {
                payloads,
                identities,
            } => {
                count(payloads, 0);
                count(identities, 1);
                (payloads.len(), identities.len())
            }
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
        for (group, [payloads, restored_identities]) in counts {
            let identities = match rows {
                IncomingIndexRows::Fresh(_) => payloads,
                IncomingIndexRows::Restored { .. } => restored_identities,
            };
            if side == 0 {
                if let Some(column) = self.left_by_key.get(group) {
                    projected = replace_projected_column(projected, column, identities)
                        .ok_or_else(|| self.state_limit())?;
                } else {
                    projected = projected
                        .checked_add(identities * size_of::<RowLocator>())
                        .ok_or_else(|| self.state_limit())?;
                }
            } else if let Some(bucket) = self.right_by_key.get(group) {
                projected = replace_projected_column(projected, &bucket.payloads, payloads)
                    .and_then(|bytes| {
                        replace_projected_column(bytes, &bucket.identities, identities)
                    })
                    .ok_or_else(|| self.state_limit())?;
            } else {
                projected = projected
                    .checked_add((payloads + identities) * size_of::<RowLocator>())
                    .ok_or_else(|| self.state_limit())?;
            }
        }
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
        side: usize,
        record: &RecordBatch,
        watermark: Option<i64>,
        late_policy: AsofLatePolicy,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<AdmissionSummary> {
        check()?;
        if record.num_rows() == 0 {
            return Ok(AdmissionSummary::default());
        }
        let cut = self.keys.cut();
        let reserved = self.reservation.size();
        match self.admit_attempt(side, record, watermark, late_policy, check) {
            Ok(summary) => Ok(summary),
            Err(error) => {
                self.keys.rollback(cut);
                self.key_ranks.clear();
                self.rebuild_key_order()?;
                self.reservation.resize(reserved);
                Err(error)
            }
        }
    }

    fn admit_attempt(
        &mut self,
        side: usize,
        record: &RecordBatch,
        watermark: Option<i64>,
        late_policy: AsofLatePolicy,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<AdmissionSummary> {
        let declaration = if side == 0 {
            self.spec.left().clone()
        } else {
            self.spec.right().clone()
        };
        let accepted = admission::times(record, &declaration)
            .values()
            .iter()
            .map(|time| watermark.is_none_or(|watermark| *time >= watermark))
            .collect::<Vec<_>>();
        let late = accepted.iter().filter(|accepted| !**accepted).count();
        if late != 0 && late_policy == AsofLatePolicy::Error {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofLateRow,
                "input contains event time below its accepted watermark",
            ));
        }
        let accepted_count = record.num_rows() - late;
        if accepted_count == 0 {
            return Ok(AdmissionSummary {
                accepted: 0,
                late: late as u64,
            });
        }
        let input = admission::ValidatedInput {
            index: side,
            watermark,
            accepted: accepted_count as u64,
        };
        let prepared = self.prepare_run(record, &declaration, &accepted, input, check)?;
        self.install_prepared_run(side, prepared)?;
        Ok(AdmissionSummary {
            accepted: accepted_count as u64,
            late: late as u64,
        })
    }

    fn prepare_run(
        &mut self,
        record: &RecordBatch,
        declaration: &AsofJoinSide,
        accepted: &[bool],
        input: admission::ValidatedInput,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<PreparedRun> {
        let selected_rows = accepted.iter().filter(|accepted| **accepted).count();
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
        let mut groups = Vec::with_capacity(record.num_rows());
        for start in (0..record.num_rows()).step_by(MORSEL_ROWS) {
            let len = MORSEL_ROWS.min(record.num_rows() - start);
            let arrays = key_arrays
                .iter()
                .map(|array| array.slice(start, len))
                .collect::<Vec<_>>();
            groups.extend(
                self.keys
                    .route_with_limit(&arrays, usize::MAX, check)
                    .map_err(|error| self.keyed_admission_error(error))?,
            );
            check()?;
        }
        let times = admission::times(record, declaration).values().to_vec();
        let accepted = vec![true; record.num_rows()];
        let run_id = self.next_run;
        let next_run = self.next_run.checked_add(1).ok_or_else(|| {
            reason(
                &self.name,
                StreamingFailureReason::AsofCounterOverflow,
                "ASOF run identifier space is exhausted",
            )
        })?;
        let run = Run::new(record.clone(), groups, times, sequences, &accepted);
        Ok(PreparedRun {
            run,
            workspace,
            run_id,
            next_run,
        })
    }

    fn install_prepared_run(&mut self, side: usize, prepared: PreparedRun) -> Result<()> {
        let PreparedRun {
            run,
            workspace,
            run_id,
            next_run,
        } = prepared;
        let locators = run.accepted_locators(run_id);
        self.validate_new_identities(side, &run, &locators)?;
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
            self.install_left(locators);
        } else {
            self.right_rows += locators.len();
            self.right_payloads += locators.len();
            self.run_bytes += run.charge();
            self.right_runs.insert(run_id, run);
            self.install_right(&locators);
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

    fn validate_new_identities(
        &self,
        side: usize,
        run: &Run,
        locators: &[RowLocator],
    ) -> Result<()> {
        if self.strictly_appends_identities(side, run, locators) {
            return Ok(());
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
        Ok(())
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
        self.validate_new_identities(side, &run, &identities)?;
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
            self.install_left(identities);
        } else {
            self.right_rows += new_rows;
            self.right_payloads += payloads.len();
            self.right_identity_only += identities
                .iter()
                .filter(|locator| !run.has_payload(**locator))
                .count();
            self.right_runs.insert(run_id, run);
            self.install_right_restored(&payloads, &identities);
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

    fn install_left(&mut self, locators: Vec<RowLocator>) {
        let mut by_key = vec![Vec::new(); self.keys.len()];
        for locator in locators.iter().copied() {
            let group = self.left_runs[&locator.run].group(locator).index();
            by_key[group].push(locator);
        }
        for (group, rows) in by_key.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let (before, after) = {
                let runs = &self.left_runs;
                self.left_by_key[group]
                    .merge(rows, |left, right| compare_identity(runs, left, right))
            };
            self.adjust_index(before, after);
        }
        let (before, after) = {
            let runs = &self.left_runs;
            let key_ranks = &self.key_ranks;
            self.pending.merge(locators, |left, right| {
                compare_left(runs, key_ranks, left, right)
            })
        };
        self.adjust_index(before, after);
    }

    fn install_right(&mut self, locators: &[RowLocator]) {
        let mut by_key = vec![Vec::new(); self.keys.len()];
        for locator in locators.iter().copied() {
            let group = self.right_runs[&locator.run].group(locator).index();
            by_key[group].push(locator);
        }
        for (group, rows) in by_key.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let (before, after) = {
                let runs = &self.right_runs;
                self.right_by_key[group]
                    .payloads
                    .merge(rows.clone(), |left, right| {
                        compare_identity(runs, left, right)
                    })
            };
            self.adjust_index(before, after);
            let (before, after) = {
                let runs = &self.right_runs;
                self.right_by_key[group]
                    .identities
                    .merge(rows, |left, right| compare_identity(runs, left, right))
            };
            self.adjust_index(before, after);
        }
    }
    fn install_right_restored(&mut self, payloads: &[RowLocator], identities: &[RowLocator]) {
        for (locators, payloads_column) in [(payloads, true), (identities, false)] {
            let mut by_key = vec![Vec::new(); self.keys.len()];
            for locator in locators.iter().copied() {
                let group = self.right_runs[&locator.run].group(locator).index();
                by_key[group].push(locator);
            }
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
                    column.merge(rows, |left, right| compare_identity(runs, left, right))
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

    /// Builds a selection for an already verified pending prefix.
    pub(in crate::operator::asof) fn select_prefix(&self, count: usize) -> Selection {
        #[cfg(test)]
        SELECTIONS.with(|count| count.set(count.get() + 1));
        let left = self.pending.rows()[..count].to_vec();
        let mut right = Vec::with_capacity(count);
        let mut validity = Vec::with_capacity(count);
        let groups = self.right_by_key.len();
        let reuse = groups <= count / MIN_PROBES_PER_GROUP;
        #[cfg(feature = "benchmark-internals")]
        let reuse = reuse && !self.legacy_latest_lookup;
        let mut offsets = ProbeOffsets {
            groups: if reuse { groups } else { 0 },
            ends: Vec::new(),
        };
        for locator in left.iter().copied() {
            let run = &self.left_runs[&locator.run];
            let group = run.group(locator).index();
            let time = run.time(locator);
            let candidate = self.candidate(group, time, &mut offsets);
            validity.push(candidate.is_some());
            right.push(candidate.unwrap_or_default());
        }
        drop(offsets);
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
        let mut summary = EvictionSummary::default();
        let mut dead_runs = BTreeSet::new();
        for group in 0..self.right_by_key.len() {
            let (payloads, identities) = {
                let bucket = &self.right_by_key[group];
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
                    let time = self.right_runs[&locator.run].time(*locator);
                    progress.right_ended
                        || progress
                            .right_watermark
                            .is_some_and(|watermark| time < watermark)
                });
                (
                    bucket.payloads.rows()[..payload_count].to_vec(),
                    bucket.identities.rows()[..identity_count].to_vec(),
                )
            };
            for locator in payloads.iter().copied() {
                let run = self
                    .right_runs
                    .get_mut(&locator.run)
                    .expect("retained ASOF payload run");
                let has_identity = run.has_identity(locator);
                let before = run.charge();
                if run.drop_payload(locator) {
                    summary.payloads += 1;
                    self.right_payloads -= 1;
                    if has_identity {
                        self.right_identity_only += 1;
                    } else {
                        self.right_rows -= 1;
                    }
                }
                let after = run.charge();
                self.run_bytes = self.run_bytes - before + after;
                if !run.is_live() {
                    dead_runs.insert(locator.run);
                }
            }
            for locator in identities.iter().copied() {
                let run = self
                    .right_runs
                    .get_mut(&locator.run)
                    .expect("retained ASOF identity run");
                let has_payload = run.has_payload(locator);
                let before = run.charge();
                if run.drop_identity(locator) {
                    summary.identities += 1;
                    if !has_payload {
                        self.right_rows -= 1;
                        self.right_identity_only -= 1;
                    }
                }
                let after = run.charge();
                self.run_bytes = self.run_bytes - before + after;
                if !run.is_live() {
                    dead_runs.insert(locator.run);
                }
            }
            let bucket = &mut self.right_by_key[group];
            let (before, after) = bucket.payloads.drop_prefix(payloads.len());
            self.index_bytes = self.index_bytes - before + after;
            let (before, after) = bucket.identities.drop_prefix(identities.len());
            self.index_bytes = self.index_bytes - before + after;
        }
        for run in dead_runs {
            if self.right_runs.get(&run).is_some_and(|run| !run.is_live()) {
                let run = self.right_runs.remove(&run).expect("dead ASOF right run");
                self.run_bytes -= run.charge();
            }
        }
        self.finish_retirement();
        summary
    }

    pub(in crate::operator::asof) fn commit(&mut self, selection: &Selection) -> Result<()> {
        if selection.left != self.pending.rows()[..selection.len()] {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofProtocolError,
                "ASOF output commit is not the selected pending prefix",
            ));
        }
        for locator in selection.left.iter().copied() {
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
        let (before, after) = self.pending.drop_prefix(selection.len());
        self.adjust_index(before, after);
        let runs = &self.left_runs;
        for column in &mut self.left_by_key {
            let (before, after) = column.retain(|locator| {
                runs.get(&locator.run)
                    .is_some_and(|run| run.has_identity(locator))
            });
            self.index_bytes = self.index_bytes - before + after;
        }
        let dead = self
            .left_runs
            .iter()
            .filter(|(_, run)| !run.is_live())
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        for id in dead {
            let run = self.left_runs.remove(&id).expect("dead ASOF left run");
            self.run_bytes -= run.charge();
        }
        self.finish_retirement();
        Ok(())
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

    fn candidate(&self, group: usize, time: i64, offsets: &mut ProbeOffsets) -> Option<RowLocator> {
        let right_time = |locator: RowLocator| {
            #[cfg(test)]
            CANDIDATE_TIME_READS.set(CANDIDATE_TIME_READS.get() + 1);
            self.right_runs[&locator.run].time(locator)
        };
        let rows = self.right_by_key.get(group)?.payloads.rows();
        let latest = rows.last()?;
        let direct = || {
            let timestamp = right_time(*latest);
            (timestamp <= time).then_some((*latest, timestamp))
        };
        #[cfg(feature = "benchmark-internals")]
        let direct = if self.legacy_latest_lookup {
            None
        } else {
            direct()
        };
        #[cfg(not(feature = "benchmark-internals"))]
        let direct = direct();
        let (locator, timestamp) = if let Some(latest) = direct {
            latest
        } else {
            let end = offsets.visible_end(group, rows, |locator| right_time(*locator) <= time);
            let locator = *rows.get(end.checked_sub(1)?)?;
            (locator, right_time(locator))
        };
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

fn compact_accepted(record: &RecordBatch, accepted: &[bool]) -> Result<RecordBatch> {
    let rows = accepted
        .iter()
        .enumerate()
        .filter(|(_, accepted)| **accepted)
        .map(|(row, _)| {
            u32::try_from(row).map_err(|_| {
                arrow_error(&ArrowError::InvalidArgumentError(
                    "ASOF accepted row exceeds UInt32".into(),
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let indices = UInt32Array::from(rows);
    let columns = record
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None).map_err(|error| arrow_error(&error)))
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(record.schema(), columns).map_err(|error| arrow_error(&error))
}
