use super::columnar::FramedKey;
use super::{
    AdmittedRow, CompiledJoin, JoinTimeBounds, MatchedPair, SidePlan, StoredRow,
    StreamJoinOperator, enforce_match_limit,
};
use crate::{CalcFlowError, EventTime, Result};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::common::hash_utils::RandomState;
use datafusion::execution::memory_pool::MemoryReservation;
use hashbrown::HashTable;
use std::{hash::BuildHasher, sync::Arc};

/// Funding per indexed row. The dictionary (hash table, key ids, per-key
/// entry lists) stays under this per row at every fill ratio, leaving
/// headroom for the table's amortized growth and shrink reallocations.
const ENTRY_BYTES: usize = 192;
const BASE_BYTES: usize = 1_024;

#[cfg(test)]
#[path = "tests/native_index_allocation_tests.rs"]
mod allocation_tests;

#[cfg(test)]
#[path = "tests/native_index_dictionary_tests.rs"]
mod dictionary_tests;

/// One retained row inside its key's entry list, which is sorted by
/// `(time, row_id)` so inclusive windows are contiguous ranges.
struct KeyEntry {
    time: EventTime,
    row_id: u64,
    row_index: u32,
}

const _: () = assert!(size_of::<KeyEntry>() == 24);

/// Distinct-key dictionary: hash slots hold `u32` key ids whose canonical V1
/// bytes stay owned by one `Arc<FramedKey>` per distinct key.
struct KeyDictionary {
    table: HashTable<u32>,
    keys: Vec<Arc<FramedKey>>,
    hasher: RandomState,
}

impl Default for KeyDictionary {
    fn default() -> Self {
        Self {
            table: HashTable::new(),
            keys: Vec::new(),
            hasher: RandomState::default(),
        }
    }
}

impl KeyDictionary {
    fn hash(&self, bytes: &[u8]) -> u64 {
        self.hasher.hash_one(bytes)
    }

    fn lookup(&self, bytes: &[u8]) -> Option<u32> {
        let hash = self.hash(bytes);
        self.table
            .find(hash, |&id| self.keys[id as usize].as_slice() == bytes)
            .copied()
    }

    /// Inserts `key`, which must not currently be present in byte equality.
    fn insert(&mut self, key: Arc<FramedKey>) -> u32 {
        let id = u32::try_from(self.keys.len()).expect("distinct Join keys fit u32");
        let hash = self.hash(key.as_slice());
        self.keys.push(key);
        let KeyDictionary {
            table,
            keys,
            hasher,
            ..
        } = self;
        table.insert_unique(hash, id, |&id| {
            hasher.hash_one(keys[id as usize].as_slice())
        });
        id
    }

    fn intern(&mut self, key: Arc<FramedKey>) -> u32 {
        match self.lookup(key.as_slice()) {
            Some(id) => id,
            None => self.insert(key),
        }
    }

    /// Swaps `id` with the last key and pops it, keeping ids dense so the
    /// per-key entry lists stay aligned with the dictionary.
    fn remove(&mut self, id: u32) {
        let last = u32::try_from(self.keys.len() - 1).expect("distinct Join keys fit u32");
        {
            let KeyDictionary {
                table,
                keys,
                hasher,
                ..
            } = self;
            // Remove the deleted key's own slot.
            let hash = hasher.hash_one(keys[id as usize].as_slice());
            table
                .find_entry(hash, |&slot| slot == id)
                .expect("removed key interned")
                .remove();
            if id != last {
                // Relocate the last key into the freed id.
                let last_hash = hasher.hash_one(keys[last as usize].as_slice());
                table
                    .find_entry(last_hash, |&slot| slot == last)
                    .expect("last key interned")
                    .remove();
                table.insert_unique(last_hash, id, |&id| {
                    hasher.hash_one(keys[id as usize].as_slice())
                });
                keys.swap(id as usize, last as usize);
            }
        }
        self.keys.pop();
        if self.keys.is_empty() {
            // Drop the peak-sized table once every key left the index so its
            // capacity is not held against a shrunken funding reservation.
            self.table = HashTable::new();
        } else if self.keys.len() * 2 <= self.keys.capacity() {
            self.keys.shrink_to_fit();
            let KeyDictionary {
                table,
                keys,
                hasher,
                ..
            } = self;
            table.shrink_to_fit(|&id| hasher.hash_one(keys[id as usize].as_slice()));
        }
    }
}

pub(super) struct NativeIndex {
    dictionary: KeyDictionary,
    /// Per-key entries sorted by `(time, row_id)`, aligned with the
    /// dictionary's key ids.
    entries: Vec<Vec<KeyEntry>>,
    entry_count: usize,
    credit: Arc<MemoryReservation>,
}

impl NativeIndex {
    #[cfg(test)]
    pub(super) fn funded_bytes(&self) -> usize {
        self.credit.size()
    }

    #[cfg(test)]
    pub(super) fn entries_len(&self) -> usize {
        self.entry_count
    }

    #[cfg(test)]
    pub(super) fn footprint_debug(&self) -> (usize, usize, usize, usize) {
        (
            self.dictionary.table.capacity(),
            self.dictionary.keys.capacity(),
            self.entries.capacity(),
            self.entries.iter().map(Vec::capacity).sum::<usize>(),
        )
    }

    pub(super) fn new(rows: &[StoredRow], credit: MemoryReservation) -> Self {
        let mut index = Self {
            dictionary: KeyDictionary::default(),
            entries: Vec::new(),
            entry_count: 0,
            credit: Arc::new(credit),
        };
        index.append(0, rows);
        index
    }

    pub(super) fn reserve(&self, rows: usize) -> Option<AppendCredit> {
        let bytes = rows.checked_mul(ENTRY_BYTES)?;
        self.credit.try_grow(bytes).ok()?;
        Some(AppendCredit {
            credit: Arc::clone(&self.credit),
            bytes,
        })
    }

    pub(super) fn append(&mut self, offset: usize, rows: &[StoredRow]) {
        for (index, row) in rows.iter().enumerate() {
            let id = self.dictionary.intern(Arc::clone(&row.encoded_key)) as usize;
            if id == self.entries.len() {
                self.entries.push(Vec::with_capacity(1));
            }
            let list = &mut self.entries[id];
            let entry = KeyEntry {
                time: row.event_time,
                row_id: row.row_id,
                row_index: u32::try_from(offset + index).expect("retained rows fit u32"),
            };
            let position = list.partition_point(|existing| {
                (existing.time, existing.row_id) < (entry.time, entry.row_id)
            });
            list.insert(position, entry);
            self.entry_count += 1;
        }
    }

    pub(super) fn remove(&mut self, row: &StoredRow, moved: Option<(&StoredRow, usize)>) {
        let id = self
            .dictionary
            .lookup(row.encoded_key.as_slice())
            .expect("evicted key interned") as usize;
        let list = &mut self.entries[id];
        let position = list.partition_point(|existing| {
            (existing.time, existing.row_id) < (row.event_time, row.row_id)
        });
        debug_assert_eq!(
            (list[position].time, list[position].row_id),
            (row.event_time, row.row_id),
            "evicted entry found by its identity"
        );
        list.remove(position);
        self.entry_count -= 1;
        self.credit.shrink(ENTRY_BYTES);
        if list.is_empty() {
            // Release the empty list and its dictionary slot in lockstep with
            // the funding shrink so eviction keeps the reservation covering
            // the live allocations.
            std::mem::take(list);
            self.dictionary
                .remove(u32::try_from(id).expect("key ids fit u32"));
            self.entries.swap_remove(id);
            if self.entries.len() * 2 <= self.entries.capacity() {
                self.entries.shrink_to_fit();
            }
        }
        if let Some((row, index)) = moved {
            let id = self
                .dictionary
                .lookup(row.encoded_key.as_slice())
                .expect("moved key interned") as usize;
            let list = &mut self.entries[id];
            let position = list.partition_point(|existing| {
                (existing.time, existing.row_id) < (row.event_time, row.row_id)
            });
            list[position].row_index = u32::try_from(index).expect("retained rows fit u32");
        }
    }

    fn key_id(&self, key: &[u8]) -> Option<u32> {
        self.dictionary.lookup(key)
    }

    /// Entries of one key inside the inclusive time window, already ordered
    /// by `(time, row_id)` — the opposite-side emission order.
    fn window_by_id(&self, id: u32, range: (EventTime, EventTime)) -> &[KeyEntry] {
        let list = &self.entries[id as usize];
        let lo = list.partition_point(|entry| entry.time < range.0);
        let hi = list.partition_point(|entry| entry.time <= range.1);
        &list[lo..hi]
    }
}

pub(super) fn eligible(compiled: &CompiledJoin, left: &Schema) -> bool {
    compiled
        .left_key_indices
        .iter()
        .all(|&index| native_type(left.field(index).data_type()))
}

fn native_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean | DataType::Int16 | DataType::Int32 | DataType::Int64
    ) || matches!(
        data_type,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
    ) || matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Timestamp(..)
    )
}

pub(super) struct NativeMatches {
    pub(super) pairs: Vec<MatchedPair>,
    pub(super) keys: NativeKeys,
    pub(super) credit: MemoryReservation,
}

pub(super) struct NativeKeys {
    pub(super) keys: Vec<Arc<FramedKey>>,
    /// Opposite-index key id per distinct probe key; `None` keys never match.
    slots: Vec<Option<u32>>,
    /// Distinct-key id per admitted row.
    ids: Vec<u32>,
    _credit: Arc<MemoryReservation>,
}

impl NativeKeys {
    fn slot(&self, pos: usize) -> Option<u32> {
        self.slots[self.ids[pos] as usize]
    }
}

pub(super) struct AppendCredit {
    credit: Arc<MemoryReservation>,
    bytes: usize,
}

impl AppendCredit {
    pub(super) fn commit(mut self) {
        self.bytes = 0;
    }
}

impl Drop for AppendCredit {
    fn drop(&mut self) {
        self.credit.shrink(self.bytes);
    }
}

impl StreamJoinOperator {
    pub(super) fn optional_credit(&mut self, bytes: usize) -> Result<Option<MemoryReservation>> {
        let credit = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-native");
        Ok(credit.try_grow(bytes).ok().map(|()| credit))
    }

    pub(super) fn native_matches(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
    ) -> Result<Option<NativeMatches>> {
        if !eligible(&self.compiled, self.input_schema(0)) {
            return Ok(None);
        }
        if !self.ensure_native_index(!plan.incoming_is_left)? {
            return Ok(None);
        }
        let matched = self.probe_native_index(plan, admitted)?;
        if matched.is_none() {
            let opposite = if plan.incoming_is_left {
                &mut self.state.right
            } else {
                &mut self.state.left
            };
            opposite.1 = None;
        }
        Ok(matched)
    }

    fn probe_native_index(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
    ) -> Result<Option<NativeMatches>> {
        let key_bytes = probe_key_charge(admitted, &plan.key_indices)?;
        let Some(credit) = self.optional_credit(key_bytes)? else {
            return Ok(None);
        };
        let shared = Arc::new(credit);
        let mut dictionary = KeyDictionary::default();
        let mut ids = Vec::with_capacity(admitted.len());
        let mut buffer: Vec<u8> = Vec::new();
        for row in admitted {
            #[cfg(test)]
            super::note_join_work(|work| work.key_encodings += 1);
            buffer.clear();
            super::append_join_key_columns(
                &mut buffer,
                row.record.columns(),
                row.record.offset(),
                &plan.key_indices,
            )?;
            if let Some(id) = dictionary.lookup(&buffer) {
                ids.push(id);
            } else {
                #[cfg(test)]
                super::note_join_work(|work| work.probe_key_allocations += 1);
                let key = Arc::new(FramedKey::funded(buffer.clone(), Arc::clone(&shared)));
                ids.push(dictionary.insert(key));
            }
        }
        let opposite = opposite_rows(self, plan);
        let index = opposite
            .1
            .as_ref()
            .expect("native index built before probing");
        let slots: Vec<Option<u32>> = dictionary
            .keys
            .iter()
            .map(|key| index.key_id(key.as_slice()))
            .collect();
        let keys = NativeKeys {
            keys: ids
                .iter()
                .map(|&id| Arc::clone(&dictionary.keys[id as usize]))
                .collect(),
            slots,
            ids,
            _credit: shared,
        };
        self.collect_native_pairs(plan, admitted, keys)
    }

    fn collect_native_pairs(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
        keys: NativeKeys,
    ) -> Result<Option<NativeMatches>> {
        let opposite = opposite_rows(self, plan);
        let index = opposite
            .1
            .as_ref()
            .expect("native index built before probing");
        let count = count_pairs(
            index,
            admitted,
            &keys,
            self.spec.bounds,
            plan,
            self.spec.limits.max_matches_per_input_batch,
        )?;
        enforce_match_limit(
            count,
            &mut self.state.metrics.match_limit_failures,
            self.spec.limits.max_matches_per_input_batch,
            &self.name,
        )?;
        let bytes = count
            .checked_mul(size_of::<MatchedPair>() + 256)
            .ok_or_else(|| scratch_error(&self.name))?;
        let Some(credit) = self.optional_credit(bytes)? else {
            return Ok(None);
        };
        let opposite = opposite_rows(self, plan);
        let index = opposite
            .1
            .as_ref()
            .expect("native index built before probing");
        let mut pairs = Vec::with_capacity(count);
        for (pos, row) in admitted.iter().enumerate() {
            let Some(slot) = keys.slot(pos) else {
                continue;
            };
            pairs.extend(
                index
                    .window_by_id(slot, time_range(self.spec.bounds, plan, row.event_time))
                    .iter()
                    .map(|entry| MatchedPair {
                        pos,
                        opposite_index: entry.row_index as usize,
                    }),
            );
        }
        Ok(Some(NativeMatches {
            pairs,
            keys,
            credit,
        }))
    }

    pub(super) fn reserve_native_append(
        &mut self,
        incoming_is_left: bool,
        rows: usize,
    ) -> Result<Option<AppendCredit>> {
        if rows == 0 || !eligible(&self.compiled, self.input_schema(0)) {
            return Ok(None);
        }
        if !self.ensure_native_index(incoming_is_left)? {
            return Ok(None);
        }
        let retained = if incoming_is_left {
            &mut self.state.left
        } else {
            &mut self.state.right
        };
        let credit = retained
            .1
            .as_ref()
            .expect("native append index initialized")
            .reserve(rows);
        if credit.is_none() {
            retained.1 = None;
        }
        Ok(credit)
    }

    fn ensure_native_index(&mut self, left: bool) -> Result<bool> {
        let rows = if left {
            &self.state.left
        } else {
            &self.state.right
        };
        if rows.1.is_some() {
            return Ok(true);
        }
        let Some(bytes) = rows
            .len()
            .checked_mul(ENTRY_BYTES)
            .and_then(|bytes| bytes.checked_add(BASE_BYTES))
        else {
            return Ok(false);
        };
        let Some(credit) = self.optional_credit(bytes)? else {
            return Ok(false);
        };
        let rows = if left {
            &mut self.state.left
        } else {
            &mut self.state.right
        };
        rows.1 = Some(NativeIndex::new(rows, credit));
        Ok(true)
    }
}

fn time_range(bounds: JoinTimeBounds, plan: &SidePlan, time: EventTime) -> (EventTime, EventTime) {
    let (before, after) = if plan.incoming_is_left {
        (bounds.before_micros, bounds.after_micros)
    } else {
        (bounds.after_micros, bounds.before_micros)
    };
    let minimum = (i128::from(time.as_micros()) - i128::from(before)).max(i128::from(i64::MIN));
    let maximum = (i128::from(time.as_micros()) + i128::from(after)).min(i128::from(i64::MAX));
    (
        EventTime::from_micros(i64::try_from(minimum).expect("minimum clamped to EventTime")),
        EventTime::from_micros(i64::try_from(maximum).expect("maximum clamped to EventTime")),
    )
}

fn probe_key_charge(admitted: &[AdmittedRow], indices: &[usize]) -> Result<usize> {
    admitted.iter().try_fold(0_usize, |total, row| {
        indices.iter().try_fold(total, |bytes, &index| {
            let array = row.record.column(index);
            let value = usize::try_from(super::logical_cell_charge(
                array.as_ref(),
                row.record.offset(),
            )?)
            .map_err(|_| scratch_error("join"))?;
            let timezone = match array.data_type() {
                DataType::Timestamp(_, Some(timezone)) => timezone.len(),
                _ => 0,
            };
            value
                .checked_add(timezone)
                .and_then(|value| value.checked_add(64))
                .and_then(|value| value.checked_mul(4))
                .and_then(|value| bytes.checked_add(value))
                .ok_or_else(|| scratch_error("join"))
        })
    })
}

fn scratch_error(name: &str) -> CalcFlowError {
    CalcFlowError::DataFusion {
        node_id: Some(name.to_owned()),
        message: "native Join scratch size overflow".to_owned(),
    }
}

fn opposite_rows<'a>(operator: &'a StreamJoinOperator, plan: &SidePlan) -> &'a super::RetainedRows {
    if plan.incoming_is_left {
        &operator.state.right
    } else {
        &operator.state.left
    }
}

fn count_pairs(
    index: &NativeIndex,
    admitted: &[AdmittedRow],
    keys: &NativeKeys,
    bounds: JoinTimeBounds,
    plan: &SidePlan,
    limit: u64,
) -> Result<usize> {
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let mut count = 0_usize;
    for (pos, row) in admitted.iter().enumerate() {
        let Some(slot) = keys.slot(pos) else {
            continue;
        };
        let remaining = limit.saturating_sub(count).saturating_add(1);
        let window = index.window_by_id(slot, time_range(bounds, plan, row.event_time));
        count = count
            .checked_add(window.len().min(remaining))
            .ok_or_else(|| scratch_error("join"))?;
        if count > limit {
            break;
        }
    }
    Ok(count)
}
