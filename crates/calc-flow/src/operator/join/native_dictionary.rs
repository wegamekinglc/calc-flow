use super::retained::Rows;
use super::{
    StoredRow,
    columnar::FramedKey,
    key_dictionary::{Dictionary, Lease, SideBindings},
};
use crate::EventTime;
use datafusion::execution::memory_pool::MemoryReservation;
use hashbrown::HashTable;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub(super) const BASE_BYTES: usize = 1_024;

#[derive(Clone, Copy)]
struct RunEntry {
    time: EventTime,
    row_id: u64,
    dense: usize,
}

impl RunEntry {
    fn identity(self) -> (EventTime, u64) {
        (self.time, self.row_id)
    }
}

struct KeyRun {
    key: Arc<FramedKey>,
    hash: u64,
    _identity: Option<Lease>,
    rows: Vec<RunEntry>,
    start: usize,
}

impl KeyRun {
    fn active(&self) -> &[RunEntry] {
        &self.rows[self.start..]
    }

    fn compact(&mut self) {
        let remaining = self.rows.len() - self.start;
        #[cfg(test)]
        super::note_join_work(|work| work.native_shifted_entries += remaining);
        self.rows.copy_within(self.start.., 0);
        self.rows.truncate(remaining);
        self.start = 0;
    }

    fn prepare_append(&mut self) {
        if self.rows.len() < self.rows.capacity() {
            return;
        }
        if self.start > 0 && self.start >= self.rows.len() / 2 {
            self.compact();
        } else {
            self.rows.reserve_exact(self.rows.capacity().max(1));
        }
    }

    fn insert(&mut self, entry: RunEntry) {
        self.prepare_append();
        let position = self.start + insertion_position(self.active(), entry);
        self.rows.insert(position, entry);
    }

    fn remove(&mut self, identity: (EventTime, u64)) -> bool {
        #[cfg(test)]
        super::note_join_work(|work| work.native_eviction_comparisons += 1);
        let position = if self.rows[self.start].identity() == identity {
            self.start
        } else {
            self.start
                + self
                    .active()
                    .binary_search_by_key(&identity, |entry| {
                        #[cfg(test)]
                        super::note_join_work(|work| work.native_eviction_comparisons += 1);
                        entry.identity()
                    })
                    .expect("retained identity")
        };
        if position == self.start {
            self.start += 1;
        } else {
            #[cfg(test)]
            super::note_join_work(|work| {
                work.native_shifted_entries += self.rows.len() - position - 1;
            });
            self.rows.remove(position);
        }
        self.start == self.rows.len()
    }

    fn move_dense(&mut self, identity: (EventTime, u64), dense: usize) {
        #[cfg(test)]
        super::note_join_work(|work| work.native_dense_repairs += 1);
        let position = self.start
            + self
                .active()
                .binary_search_by_key(&identity, |entry| entry.identity())
                .expect("moved retained identity");
        self.rows[position].dense = dense;
    }
}

struct Funding {
    credit: MemoryReservation,
    resident: AtomicUsize,
}

pub(super) struct NativeIndex {
    entries: HashTable<u32>,
    slots: Vec<Option<KeyRun>>,
    free: Vec<u32>,
    rows: usize,
    run_backing_bytes: usize,
    funding: Arc<Funding>,
}

impl NativeIndex {
    fn owner_bytes() -> Option<usize> {
        std::alloc::Layout::new::<[AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<Self>())
            .ok()
            .map(|(layout, _)| layout.pad_to_align().size())
    }

    fn empty(credit: MemoryReservation) -> Self {
        Self {
            entries: HashTable::new(),
            slots: Vec::new(),
            free: Vec::new(),
            rows: 0,
            run_backing_bytes: 0,
            funding: Arc::new(Funding {
                credit,
                resident: AtomicUsize::new(
                    BASE_BYTES + Self::owner_bytes().expect("index owner layout"),
                ),
            }),
        }
    }

    #[cfg(test)]
    pub(super) fn new(rows: &[StoredRow], credit: MemoryReservation) -> Self {
        assert!(
            credit.size() >= Self::build_charge(rows.len()).expect("funded dictionary capacity")
        );
        let mut index = Self::empty(credit);
        index.append(0, rows);
        index.refund();
        index
    }

    pub(super) fn new_identified(
        rows: &dyn Rows,
        dictionary: &Dictionary,
        bindings: &SideBindings,
        credit: MemoryReservation,
    ) -> Self {
        assert!(
            credit.size()
                >= Self::build_charge(rows.len()).expect("funded identity index capacity")
        );
        let mut index = Self::empty(credit);
        for position in rows.references() {
            let (time, row_id) = rows.identity(position);
            let key = bindings.canonical_key(row_id, rows.key(position));
            index.append_identity(position, time, row_id, key, |key| dictionary.lease(key));
        }
        index.rows = rows.len();
        index
            .funding
            .resident
            .store(index.resident_bytes(), Ordering::Relaxed);
        index.refund();
        index
    }

    #[cfg(test)]
    pub(super) fn new_colliding(rows: &[StoredRow], credit: MemoryReservation) -> Self {
        assert!(
            credit.size() >= Self::build_charge(rows.len()).expect("funded dictionary capacity")
        );
        let mut index = Self::empty(credit);
        index.append(0, rows);
        index.refund();
        index
    }

    pub(super) fn build_charge(rows: usize) -> Option<usize> {
        let slots = vector_capacity(0, checked_slots(rows)?)?;
        charge_sum([
            doubled(table_charge(rows)),
            doubled(capacity_bytes(
                slots,
                size_of::<Option<KeyRun>>() + size_of::<u32>(),
            )),
            capacity_bytes(rows, 2 * size_of::<RunEntry>()),
        ])
    }

    fn projected_slots(&self, rows: usize) -> Option<usize> {
        let required = self
            .slots
            .len()
            .checked_add(rows.saturating_sub(self.free.len()))?;
        vector_capacity(self.slots.capacity(), checked_slots(required)?)
    }

    fn table_peak(&self, rows: usize) -> Option<usize> {
        let required = self.entries.len().checked_add(rows)?;
        let projected = table_charge(required)?.max(self.entries.allocation_size());
        self.entries.allocation_size().checked_add(projected)
    }

    fn append_charge(&self, rows: usize) -> Option<usize> {
        let slots = self.projected_slots(rows)?;
        charge_sum([
            self.table_peak(rows),
            capacity_peak(self.slots.capacity(), slots, size_of::<Option<KeyRun>>()),
            capacity_peak(self.free.capacity(), slots, size_of::<u32>()),
            self.run_backing_bytes.checked_mul(3),
            capacity_bytes(rows, 2 * size_of::<RunEntry>()),
        ])
    }

    #[cfg(test)]
    pub(super) fn reserve(&self, rows: usize) -> Option<AppendCredit> {
        self.reserve_checked(rows, "native-test").unwrap()
    }

    pub(super) fn reserve_checked(
        &self,
        rows: usize,
        name: &str,
    ) -> crate::Result<Option<AppendCredit>> {
        let total = self
            .append_charge(rows)
            .ok_or_else(|| super::counter_overflow(name, "native_capacity"))?;
        let bytes = total.saturating_sub(self.funding.credit.size());
        match self.funding.credit.try_grow(bytes) {
            Ok(()) => Ok(Some(AppendCredit {
                funding: Arc::clone(&self.funding),
                bytes,
            })),
            Err(datafusion::common::DataFusionError::ResourcesExhausted(_)) => Ok(None),
            Err(error) => Err(crate::CalcFlowError::DataFusion {
                node_id: Some(name.to_owned()),
                message: error.to_string(),
            }),
        }
    }

    #[cfg(test)]
    pub(super) fn append(&mut self, offset: usize, rows: &[StoredRow]) {
        self.append_resolved(offset, rows, |row| &row.encoded_key, |_| None);
    }

    #[cfg(test)]
    fn append_resolved<'a>(
        &mut self,
        offset: usize,
        rows: &'a [StoredRow],
        key_for: impl Fn(&'a StoredRow) -> &'a Arc<FramedKey>,
        resolve: impl Fn(&Arc<FramedKey>) -> Option<Lease>,
    ) {
        if rows.is_empty() {
            return;
        }
        assert!(
            self.funding.credit.size()
                >= self
                    .append_charge(rows.len())
                    .expect("funded append capacity")
        );
        for (position, row) in rows.iter().enumerate() {
            self.append_entry(offset + position, row, key_for(row), &resolve);
        }
        self.rows += rows.len();
        self.funding
            .resident
            .store(self.resident_bytes(), Ordering::Relaxed);
    }

    fn append_entry(
        &mut self,
        index: usize,
        row: &StoredRow,
        key: &Arc<FramedKey>,
        resolve: impl Fn(&Arc<FramedKey>) -> Option<Lease>,
    ) {
        self.append_identity(index, row.event_time, row.row_id, key, resolve);
    }

    fn append_identity(
        &mut self,
        index: usize,
        time: EventTime,
        row_id: u64,
        key: &Arc<FramedKey>,
        resolve: impl Fn(&Arc<FramedKey>) -> Option<Lease>,
    ) {
        let hash = Self::hash(key);
        let id = self
            .find(hash, key)
            .unwrap_or_else(|| self.insert_key(hash, key, resolve(key)));
        let run = self.slots[id as usize].as_mut().expect("live key ID");
        let old_capacity = run.rows.capacity();
        run.insert(RunEntry {
            time,
            row_id,
            dense: index,
        });
        self.run_backing_bytes += (run.rows.capacity() - old_capacity) * size_of::<RunEntry>();
    }

    pub(super) fn append_identified_indices(
        &mut self,
        indices: &[usize],
        rows: &[StoredRow],
        dictionary: &Dictionary,
    ) {
        for (&index, row) in indices.iter().zip(rows) {
            self.append_entry(index, row, &row.encoded_key, |key| {
                Some(
                    dictionary
                        .lease(key)
                        .expect("committed live canonical identity"),
                )
            });
        }
        self.rows += rows.len();
        self.funding
            .resident
            .store(self.resident_bytes(), Ordering::Relaxed);
    }

    pub(super) fn remove_stable<'a>(
        &mut self,
        row: &StoredRow,
        key: &Arc<FramedKey>,
        key_at: impl FnOnce(usize) -> &'a Arc<FramedKey>,
    ) {
        self.remove_resolved(row, key, None, key_at);
    }


    /// Truncates one key's time-sorted expired prefix in a single step
    /// (spec FR12): advances `start`, releases the key slot when the run
    /// empties, keeps the backing accounting exact, and refreshes the one
    /// surviving owner whose row was cut. The `moved`/dense repair path is
    /// unreachable from stable removals, so no replicated repair is needed.

    pub(super) fn truncate_key<'a>(
        &mut self,
        key: &Arc<FramedKey>,
        expected: u64,
        expired: &dyn Fn(EventTime) -> bool,
        key_at: impl FnOnce(usize) -> &'a Arc<FramedKey>,
        name: &str,
    ) -> crate::Result<()> {
        let count = usize::try_from(expected)
            .map_err(|_| truncation_invariant(name, "truncation count overflow"))?;
        let hash = Self::hash(key);
        let id = self
            .find(hash, key)
            .ok_or_else(|| truncation_invariant(name, "truncated key is not retained"))?;
        let run = self.slots[id as usize].as_mut().expect("live key ID");
        let cut = run.active().partition_point(|entry| expired(entry.time));
        if cut != count {
            return Err(truncation_invariant(
                name,
                "truncated prefix differs from the planned eviction",
            ));
        }
        run.start += cut;
        let emptied = run.start == run.rows.len();
        if emptied {
            self.remove_key(hash, id);
        } else {
            self.refresh_key_owner(id, key_at);
        }
        self.rows = self
            .rows
            .checked_sub(count)
            .ok_or_else(|| truncation_invariant(name, "truncated row count underflow"))?;
        if emptied && self.rows == 0 {
            self.entries = HashTable::new();
            self.slots = Vec::new();
            self.free = Vec::new();
        }
        self.funding
            .resident
            .store(self.resident_bytes(), Ordering::Relaxed);
        self.refund();
        Ok(())
    }

    fn insert_key(&mut self, hash: u64, key: &Arc<FramedKey>, identity: Option<Lease>) -> u32 {
        let run = Some(KeyRun {
            key: Arc::clone(key),
            hash,
            _identity: identity,
            rows: Vec::new(),
            start: 0,
        });
        let id = if let Some(id) = self.free.pop() {
            self.slots[id as usize] = run;
            id
        } else {
            self.grow_slots();
            let id = u32::try_from(self.slots.len()).expect("funded dictionary ID capacity");
            self.slots.push(run);
            id
        };
        let slots = &self.slots;
        self.entries.insert_unique(hash, id, |id| {
            slots[*id as usize].as_ref().expect("live key ID").hash
        });
        id
    }

    fn grow_slots(&mut self) {
        if self.slots.len() == self.slots.capacity() {
            let capacity = vector_capacity(self.slots.capacity(), self.slots.len() + 1)
                .expect("funded dictionary slots");
            self.slots.reserve_exact(capacity - self.slots.len());
            self.free.reserve_exact(capacity - self.free.len());
        }
    }

    #[cfg(test)]
    pub(super) fn remove<'a>(
        &mut self,
        row: &StoredRow,
        moved: Option<(&StoredRow, usize)>,
        key_at: impl FnOnce(usize) -> &'a Arc<FramedKey>,
    ) {
        self.remove_resolved(
            row,
            &row.encoded_key,
            moved.map(|(row, dense)| (row, &row.encoded_key, dense)),
            key_at,
        );
    }

    pub(super) fn remove_resolved<'a>(
        &mut self,
        row: &StoredRow,
        key: &Arc<FramedKey>,
        moved: Option<(&StoredRow, &Arc<FramedKey>, usize)>,
        key_at: impl FnOnce(usize) -> &'a Arc<FramedKey>,
    ) {
        let hash = Self::hash(key);
        let id = self.find(hash, key).expect("retained key ID");
        let run = self.slots[id as usize].as_mut().expect("live key ID");
        let removed_key = run.remove((row.event_time, row.row_id));
        if removed_key {
            self.remove_key(hash, id);
        }
        if let Some((row, key, dense)) = moved {
            self.move_dense(row, key, dense);
        }
        self.refresh_key_owner(id, key_at);
        self.rows -= 1;
        if !removed_key {
            // A live run retains all index backing capacity.
            return;
        }
        if self.rows == 0 {
            self.entries = HashTable::new();
            self.slots = Vec::new();
            self.free = Vec::new();
        }
        self.funding
            .resident
            .store(self.resident_bytes(), Ordering::Relaxed);
        self.refund();
    }

    fn refresh_key_owner<'a>(&mut self, id: u32, key_at: impl FnOnce(usize) -> &'a Arc<FramedKey>) {
        let Some(run) = self.slots[id as usize].as_mut() else {
            return;
        };
        let key = key_at(run.active()[0].dense);
        debug_assert_eq!(run.key.as_slice(), key.as_slice());
        if !Arc::ptr_eq(&run.key, key) {
            run.key = Arc::clone(key);
        }
    }

    fn remove_key(&mut self, hash: u64, id: u32) {
        self.entries
            .find_entry(hash, |entry| *entry == id)
            .unwrap_or_else(|_| panic!("retained key ID"))
            .remove();
        let run = self.slots[id as usize].take().expect("live key ID");
        self.run_backing_bytes -= run.rows.capacity() * size_of::<RunEntry>();
        drop(run);
        self.free.push(id);
    }

    fn move_dense(&mut self, row: &StoredRow, key: &Arc<FramedKey>, dense: usize) {
        let id = self
            .find(Self::hash(key), key)
            .expect("moved retained key ID");
        self.slots[id as usize]
            .as_mut()
            .expect("live key ID")
            .move_dense((row.event_time, row.row_id), dense);
    }

    fn hash(key: &FramedKey) -> u64 {
        key.address()
            .map_or(0, super::super::key_identity::KeyAddress::hash)
    }

    fn find(&self, hash: u64, key: &FramedKey) -> Option<u32> {
        #[cfg(test)]
        super::note_join_work(|work| work.native_key_lookups += 1);
        self.entries
            .find(hash, |id| {
                same_key(
                    &self.slots[*id as usize]
                        .as_ref()
                        .expect("live identity run")
                        .key,
                    key,
                )
            })
            .copied()
    }

    pub(super) fn key_id(&self, key: &FramedKey) -> Option<u32> {
        self.find(Self::hash(key), key)
    }

    /// Length of one key's active run, as a probe-cost estimate input.
    pub(super) fn run_len_by_id(&self, id: u32) -> usize {
        self.slots[id as usize]
            .as_ref()
            .expect("live key ID")
            .active()
            .len()
    }

    /// Uncached window search, kept as the reference the tests compare every
    /// cursor-advanced bound against.
    #[cfg(test)]
    fn window_by_id(&self, id: u32, range: (EventTime, EventTime)) -> &[RunEntry] {
        let run = self.slots[id as usize]
            .as_ref()
            .expect("live key ID")
            .active();
        let lower = run.partition_point(|entry| boundary(entry.time, range.0, false));
        #[cfg(test)]
        note_window_run_search();
        let upper = run.partition_point(|entry| boundary(entry.time, range.1, true));
        #[cfg(test)]
        note_window_run_search();
        &run[lower..upper]
    }

    /// Exact `[lower, upper)` window bounds for `range`, reusing `cursor`
    /// when the window only moves forward — the common case for
    /// event-time-ordered probes — and falling back to a full search when the
    /// new window starts before the cached one. Bounds always equal the plain
    /// `window_by_id` search.
    pub(super) fn window_bounds_by_id(
        &self,
        id: u32,
        range: (EventTime, EventTime),
        cursor: Option<WindowCursor>,
    ) -> WindowBounds {
        let run = self.slots[id as usize]
            .as_ref()
            .expect("live key ID")
            .active();
        let Some(cursor) =
            cursor.filter(|cursor| range.0 >= cursor.range.0 && range.1 >= cursor.range.1)
        else {
            let lower = run.partition_point(|entry| boundary(entry.time, range.0, false));
            #[cfg(test)]
            note_window_run_search();
            let upper = run.partition_point(|entry| boundary(entry.time, range.1, true));
            #[cfg(test)]
            note_window_run_search();
            return WindowBounds { lower, upper };
        };
        let lower = advance_partition_point(run, cursor.bounds.lower, |entry| {
            boundary(entry.time, range.0, false)
        });
        let upper = advance_partition_point(run, lower.max(cursor.bounds.upper), |entry| {
            boundary(entry.time, range.1, true)
        });
        WindowBounds { lower, upper }
    }

    /// Dense indices of a run slice described by previously computed bounds.
    pub(super) fn range_by_bounds(
        &self,
        id: u32,
        bounds: WindowBounds,
    ) -> impl Iterator<Item = usize> + '_ {
        let run = self.slots[id as usize]
            .as_ref()
            .expect("live key ID")
            .active();
        run[bounds.lower..bounds.upper].iter().map(|entry| {
            #[cfg(test)]
            super::note_join_work(|work| work.native_range_visits += 1);
            entry.dense
        })
    }

    /// Uncached window visit order, kept as the test reference for
    /// cursor-advanced fills.
    #[cfg(test)]
    pub(super) fn range_by_id(
        &self,
        id: u32,
        range: (EventTime, EventTime),
    ) -> impl Iterator<Item = usize> + '_ {
        self.window_by_id(id, range).iter().map(|entry| {
            #[cfg(test)]
            super::note_join_work(|work| work.native_range_visits += 1);
            entry.dense
        })
    }

    #[cfg(test)]
    pub(super) fn range<'a>(
        &'a self,
        key: &FramedKey,
        range: (EventTime, EventTime),
    ) -> impl Iterator<Item = usize> + 'a {
        self.key_id(key)
            .into_iter()
            .flat_map(move |id| self.range_by_id(id, range))
    }

    pub(super) fn resident_bytes(&self) -> usize {
        #[cfg(test)]
        assert_eq!(
            self.run_backing_bytes,
            self.slots
                .iter()
                .flatten()
                .map(|run| run.rows.capacity() * size_of::<RunEntry>())
                .sum::<usize>()
        );
        BASE_BYTES
            + Self::owner_bytes().expect("index owner layout")
            + self.entries.allocation_size()
            + self.slots.capacity() * size_of::<Option<KeyRun>>()
            + self.free.capacity() * size_of::<u32>()
            + self.run_backing_bytes
    }

    fn refund(&self) {
        #[cfg(test)]
        super::note_join_work(|work| work.native_funding_refunds += 1);
        self.funding
            .credit
            .shrink(self.funding.credit.size() - self.resident_bytes());
    }

    #[cfg(test)]
    pub(super) fn funded_bytes(&self) -> usize {
        self.funding.credit.size()
    }
    #[cfg(test)]
    pub(super) fn row_count(&self) -> usize {
        self.rows
    }
}

// Test-only per-thread count of full-run window binary searches, so probe
// work stays observable as O(distinct keys + out-of-order fallbacks) instead
// of per row, without racing parallel tests.
#[cfg(test)]
thread_local! {
    static WINDOW_RUN_SEARCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn note_window_run_search() {
    WINDOW_RUN_SEARCHES.set(WINDOW_RUN_SEARCHES.get() + 1);
}

#[cfg(test)]
pub(super) fn reset_window_run_searches() {
    WINDOW_RUN_SEARCHES.set(0);
}

#[cfg(test)]
pub(super) fn window_run_searches() -> usize {
    WINDOW_RUN_SEARCHES.get()
}

/// Exact `[lower, upper)` bounds of one key's active run inside a time range.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct WindowBounds {
    lower: usize,
    upper: usize,
}

impl WindowBounds {
    pub(super) const EMPTY: Self = Self { lower: 0, upper: 0 };

    pub(super) fn matches(self) -> usize {
        debug_assert!(self.upper >= self.lower);
        self.upper - self.lower
    }
}

/// A key's previous probe window and its run position, so the next forward
/// window can advance from it instead of searching the whole run again.
#[derive(Clone, Copy)]
pub(super) struct WindowCursor {
    range: (EventTime, EventTime),
    bounds: WindowBounds,
}

impl WindowCursor {
    pub(super) fn new(range: (EventTime, EventTime), bounds: WindowBounds) -> Self {
        Self { range, bounds }
    }
}

fn same_key(stored: &FramedKey, key: &FramedKey) -> bool {
    match (stored.address(), key.address()) {
        (Some(left), Some(right)) => left == right && std::ptr::eq(stored, key),
        #[cfg(test)]
        (None, None) => stored == key,
        _ => false,
    }
}

fn insertion_position(run: &[RunEntry], entry: RunEntry) -> usize {
    if run
        .last()
        .is_none_or(|last| last.identity() <= entry.identity())
    {
        return run.len();
    }
    run.partition_point(|previous| previous.identity() <= entry.identity())
}

/// Partition point of `pred` over `run`, given `pred` holds on `run[..from]`.
/// Gallops forward exponentially, then bisects the small remainder.
fn advance_partition_point(
    run: &[RunEntry],
    from: usize,
    pred: impl Fn(&RunEntry) -> bool,
) -> usize {
    debug_assert!(from <= run.len());
    let mut low = from;
    let mut high = run.len();
    let mut step = 1_usize;
    while low.saturating_add(step) < high {
        if pred(&run[low + step - 1]) {
            low += step;
            step = step.saturating_mul(2);
        } else {
            high = low + step - 1;
            break;
        }
    }
    low + run[low..high].partition_point(|entry| pred(entry))
}

fn boundary(time: EventTime, bound: EventTime, inclusive: bool) -> bool {
    #[cfg(test)]
    super::note_join_work(|work| work.native_boundary_visits += 1);
    if inclusive {
        time <= bound
    } else {
        time < bound
    }
}

fn vector_capacity(current: usize, required: usize) -> Option<usize> {
    if required <= current {
        return Some(current);
    }
    required
        .checked_next_power_of_two()
        .map(|capacity| capacity.max(4))
}

fn checked_slots(required: usize) -> Option<usize> {
    (u64::try_from(required).ok()? <= u64::from(u32::MAX) + 1).then_some(required)
}

fn charge_sum<const N: usize>(terms: [Option<usize>; N]) -> Option<usize> {
    terms.into_iter().try_fold(
        BASE_BYTES.checked_add(NativeIndex::owner_bytes()?)?,
        |total, term| total.checked_add(term?),
    )
}

fn doubled(bytes: Option<usize>) -> Option<usize> {
    bytes.and_then(|bytes| bytes.checked_mul(2))
}

fn capacity_bytes(capacity: usize, width: usize) -> Option<usize> {
    capacity.checked_mul(width)
}

fn capacity_peak(old: usize, new: usize, width: usize) -> Option<usize> {
    capacity_bytes(old.checked_add(new)?, width)
}

fn table_charge(entries: usize) -> Option<usize> {
    if entries == 0 {
        return Some(0);
    }
    let buckets = entries
        .checked_mul(8)?
        .div_ceil(7)
        .checked_next_power_of_two()?
        .max(4);
    buckets.checked_mul(size_of::<u32>() + 1)?.checked_add(64)
}

pub(super) struct AppendCredit {
    funding: Arc<Funding>,
    bytes: usize,
}

impl AppendCredit {
    pub(super) fn commit(mut self) {
        let resident = self.funding.resident.load(Ordering::Relaxed);
        self.funding
            .credit
            .shrink(self.funding.credit.size() - resident);
        self.bytes = 0;
    }
}

impl Drop for AppendCredit {
    fn drop(&mut self) {
        self.funding.credit.shrink(self.bytes);
    }
}

/// A violated monotone-with-gaps invariant is an internal error, never a
/// silent mis-tombstone (spec FR15).
fn truncation_invariant(name: &str, message: &str) -> crate::CalcFlowError {
    crate::CalcFlowError::Internal {
        message: format!("stream Join {name:?} native index truncation: {message}"),
    }
}
