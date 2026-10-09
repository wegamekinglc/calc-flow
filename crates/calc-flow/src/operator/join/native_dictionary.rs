use super::{
    StoredRow,
    borrowed_key::{KeyHashState, framed_hash},
    columnar::FramedKey,
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
        let position = self.start
            + self
                .active()
                .binary_search_by_key(&identity, |entry| entry.identity())
                .expect("retained identity");
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
    hasher: KeyHashState,
    rows: usize,
    run_backing_bytes: usize,
    funding: Arc<Funding>,
    #[cfg(test)]
    hash_mask: u64,
}

impl NativeIndex {
    fn empty(credit: MemoryReservation) -> Self {
        Self {
            entries: HashTable::new(),
            slots: Vec::new(),
            free: Vec::new(),
            hasher: KeyHashState::default(),
            rows: 0,
            run_backing_bytes: 0,
            funding: Arc::new(Funding {
                credit,
                resident: AtomicUsize::new(BASE_BYTES),
            }),
            #[cfg(test)]
            hash_mask: u64::MAX,
        }
    }

    pub(super) fn new(rows: &[StoredRow], credit: MemoryReservation) -> Self {
        assert!(
            credit.size() >= Self::build_charge(rows.len()).expect("funded dictionary capacity")
        );
        let mut index = Self::empty(credit);
        index.append(0, rows);
        index.refund();
        index
    }

    #[cfg(test)]
    pub(super) fn new_colliding(rows: &[StoredRow], credit: MemoryReservation) -> Self {
        assert!(
            credit.size() >= Self::build_charge(rows.len()).expect("funded dictionary capacity")
        );
        let mut index = Self::empty(credit);
        index.hash_mask = 0;
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

    pub(super) fn reserve(&self, rows: usize) -> Option<AppendCredit> {
        let total = self.append_charge(rows)?;
        let bytes = total.checked_sub(self.funding.credit.size())?;
        self.funding.credit.try_grow(bytes).ok()?;
        Some(AppendCredit {
            funding: Arc::clone(&self.funding),
            bytes,
        })
    }

    pub(super) fn append(&mut self, offset: usize, rows: &[StoredRow]) {
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
            let hash = self.hash(&row.encoded_key);
            let id = self
                .find(hash, &row.encoded_key)
                .unwrap_or_else(|| self.insert_key(hash, &row.encoded_key));
            let run = self.slots[id as usize].as_mut().expect("live key ID");
            let old_capacity = run.rows.capacity();
            run.insert(RunEntry {
                time: row.event_time,
                row_id: row.row_id,
                dense: offset + position,
            });
            self.run_backing_bytes += (run.rows.capacity() - old_capacity) * size_of::<RunEntry>();
        }
        self.rows += rows.len();
        self.funding
            .resident
            .store(self.resident_bytes(), Ordering::Relaxed);
    }

    fn insert_key(&mut self, hash: u64, key: &Arc<FramedKey>) -> u32 {
        let run = Some(KeyRun {
            key: Arc::clone(key),
            hash,
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

    pub(super) fn remove<'a>(
        &mut self,
        row: &StoredRow,
        moved: Option<(&StoredRow, usize)>,
        key_at: impl FnOnce(usize) -> &'a Arc<FramedKey>,
    ) {
        let hash = self.hash(&row.encoded_key);
        let id = self.find(hash, &row.encoded_key).expect("retained key ID");
        let run = self.slots[id as usize].as_mut().expect("live key ID");
        if run.remove((row.event_time, row.row_id)) {
            self.remove_key(hash, id);
        }
        if let Some((row, dense)) = moved {
            self.move_dense(row, dense);
        }
        self.refresh_key_owner(id, key_at);
        self.rows -= 1;
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

    fn move_dense(&mut self, row: &StoredRow, dense: usize) {
        let id = self
            .find(self.hash(&row.encoded_key), &row.encoded_key)
            .expect("moved retained key ID");
        self.slots[id as usize]
            .as_mut()
            .expect("live key ID")
            .move_dense((row.event_time, row.row_id), dense);
    }

    fn hash(&self, key: &FramedKey) -> u64 {
        let hash = framed_hash(&self.hasher, key);
        #[cfg(test)]
        let hash = hash & self.hash_mask;
        hash
    }

    fn find(&self, hash: u64, key: &FramedKey) -> Option<u32> {
        #[cfg(test)]
        super::note_join_work(|work| work.native_key_lookups += 1);
        self.entries
            .find(hash, |id| {
                self.slots[*id as usize]
                    .as_ref()
                    .expect("live key ID")
                    .key
                    .as_ref()
                    == key
            })
            .copied()
    }

    pub(super) fn key_id(&self, key: &FramedKey) -> Option<u32> {
        self.find(self.hash(key), key)
    }

    fn window_by_id(&self, id: u32, range: (EventTime, EventTime)) -> &[RunEntry] {
        let run = self.slots[id as usize]
            .as_ref()
            .expect("live key ID")
            .active();
        let lower = run.partition_point(|entry| boundary(entry.time, range.0, false));
        let upper = run.partition_point(|entry| boundary(entry.time, range.1, true));
        &run[lower..upper]
    }

    pub(super) fn count_window_by_id(&self, id: u32, range: (EventTime, EventTime)) -> usize {
        self.window_by_id(id, range).len()
    }

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
            + self.entries.allocation_size()
            + self.slots.capacity() * size_of::<Option<KeyRun>>()
            + self.free.capacity() * size_of::<u32>()
            + self.run_backing_bytes
    }

    fn refund(&self) {
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

fn insertion_position(run: &[RunEntry], entry: RunEntry) -> usize {
    if run
        .last()
        .is_none_or(|last| last.identity() <= entry.identity())
    {
        return run.len();
    }
    run.partition_point(|previous| previous.identity() <= entry.identity())
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
    terms
        .into_iter()
        .try_fold(BASE_BYTES, |total, term| total.checked_add(term?))
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
