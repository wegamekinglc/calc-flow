#[cfg(test)]
use super::StoredRow;
use super::borrowed_key::KeyHashState;
use super::borrowed_key::framed_hash;
use super::columnar::FramedKey;
use super::key_arena::{ArenaLayout, visit_key_frames};
use super::native_dictionary::BASE_BYTES;
pub(super) use super::native_dictionary::{AppendCredit, NativeIndex};
use super::{
    AdmittedRow, CompiledJoin, JoinTimeBounds, MatchedPair, SidePlan, StreamJoinOperator,
    enforce_match_limit,
};
use crate::{CalcFlowError, EventTime, Result};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::execution::memory_pool::MemoryReservation;
use hashbrown::HashTable;
use std::sync::Arc;

#[cfg(test)]
#[path = "tests/native_index_allocation_tests.rs"]
mod allocation_tests;

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
    ids: Vec<u32>,
    _credit: Arc<MemoryReservation>,
}

impl NativeKeys {
    pub(super) fn id(&self, position: usize) -> u32 {
        self.ids[position]
    }

    pub(super) fn key(&self, position: usize) -> &Arc<FramedKey> {
        &self.keys[self.ids[position] as usize]
    }

    #[cfg(test)]
    pub(super) fn row_keys(&self) -> impl Iterator<Item = &Arc<FramedKey>> {
        self.ids.iter().map(|&id| &self.keys[id as usize])
    }
}

// Opposite IDs cannot outlive the immutable index used by both window passes.
struct ProbeWindows<'a> {
    index: &'a NativeIndex,
    ids: &'a [u32],
    slots: Vec<Option<u32>>,
}

impl<'a> ProbeWindows<'a> {
    fn new(index: &'a NativeIndex, keys: &'a NativeKeys) -> Self {
        let mut slots = Vec::with_capacity(keys.keys.len());
        slots.extend(keys.keys.iter().map(|key| index.key_id(key)));
        Self {
            index,
            ids: &keys.ids,
            slots,
        }
    }

    fn count(&self, pos: usize, range: (EventTime, EventTime)) -> usize {
        self.slots[self.ids[pos] as usize].map_or(0, |id| self.index.count_window_by_id(id, range))
    }

    fn range(&self, pos: usize, range: (EventTime, EventTime)) -> impl Iterator<Item = usize> + '_ {
        self.slots[self.ids[pos] as usize]
            .into_iter()
            .flat_map(move |id| self.index.range_by_id(id, range))
    }
}

struct ProbeKeyInterner {
    keys: Vec<Arc<FramedKey>>,
    ids: Vec<u32>,
    hashes: Vec<u64>,
    dictionary: HashTable<u32>,
    hasher: KeyHashState,
}

impl ProbeKeyInterner {
    fn new(rows: usize) -> Self {
        Self {
            keys: Vec::new(),
            ids: Vec::with_capacity(rows),
            hashes: Vec::new(),
            dictionary: HashTable::new(),
            hasher: KeyHashState::default(),
        }
    }

    fn push(
        &mut self,
        key: super::borrowed_key::BorrowedKey<'_>,
        credit: &Arc<MemoryReservation>,
        name: &str,
    ) -> Result<()> {
        let hash = key.hash(&self.hasher)?;
        self.push_hashed(key, hash, credit, name)
    }

    fn push_hashed(
        &mut self,
        key: super::borrowed_key::BorrowedKey<'_>,
        hash: u64,
        credit: &Arc<MemoryReservation>,
        name: &str,
    ) -> Result<()> {
        self.push_with(
            hash,
            |stored| key.equals(stored),
            || super::columnar::funded_key(key.columns, key.row, key.indices, Arc::clone(credit)),
            name,
        )
    }

    fn push_bytes(
        &mut self,
        bytes: &[u8],
        credit: &Arc<MemoryReservation>,
        name: &str,
    ) -> Result<()> {
        self.push_bytes_hashed(bytes, framed_hash(&self.hasher, bytes), credit, name)
    }

    fn push_bytes_hashed(
        &mut self,
        bytes: &[u8],
        hash: u64,
        credit: &Arc<MemoryReservation>,
        name: &str,
    ) -> Result<()> {
        self.push_with(
            hash,
            |stored| stored.as_slice() == bytes,
            || {
                #[cfg(test)]
                super::note_join_work(|work| work.key_encodings += 1);
                Ok(super::columnar::funded_encoded_key(
                    bytes.to_vec(),
                    Arc::clone(credit),
                ))
            },
            name,
        )
    }

    fn push_with(
        &mut self,
        hash: u64,
        equals: impl Fn(&FramedKey) -> bool,
        encode: impl FnOnce() -> Result<Arc<FramedKey>>,
        name: &str,
    ) -> Result<()> {
        let previous = self
            .dictionary
            .find(hash, |id| equals(&self.keys[*id as usize]));
        let id = if let Some(id) = previous {
            *id
        } else {
            let id = u32::try_from(self.keys.len()).map_err(|_| scratch_error(name))?;
            let encoded = encode()?;
            self.keys.push(encoded);
            self.hashes.push(hash);
            self.dictionary
                .insert_unique(hash, id, |id| self.hashes[*id as usize]);
            id
        };
        self.ids.push(id);
        Ok(())
    }

    fn finish(self, credit: Arc<MemoryReservation>) -> NativeKeys {
        NativeKeys {
            keys: self.keys,
            ids: self.ids,
            _credit: credit,
        }
    }
}

fn intern_probe_keys(
    admitted: &[AdmittedRow],
    indices: &[usize],
    credit: Arc<MemoryReservation>,
    name: &str,
    layout: Option<ArenaLayout>,
) -> Result<NativeKeys> {
    let mut interner = ProbeKeyInterner::new(admitted.len());
    if let Some(layout) = layout.filter(|layout| layout.fits(admitted.len(), indices.len())) {
        visit_key_frames(admitted, indices, layout, |bytes| {
            interner.push_bytes(bytes, &credit, name)
        })?;
    } else {
        for row in admitted {
            let key = super::borrowed_key::BorrowedKey {
                columns: row.record.columns(),
                row: row.record.offset(),
                indices,
            };
            interner.push(key, &credit, name)?;
        }
    }
    Ok(interner.finish(credit))
}

#[cfg(test)]
pub(super) fn colliding_probe_keys(
    admitted: &[AdmittedRow],
    indices: &[usize],
    credit: MemoryReservation,
) -> Result<NativeKeys> {
    let shared = Arc::new(credit);
    let mut interner = ProbeKeyInterner::new(admitted.len());
    let layout = ArenaLayout::measure(admitted, indices)?.expect("eligible collision fixture");
    assert!(layout.fits(admitted.len(), indices.len()));
    visit_key_frames(admitted, indices, layout, |bytes| {
        interner.push_bytes_hashed(bytes, 0, &shared, "collision-test")
    })?;
    Ok(interner.finish(shared))
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
        let Some(keys) = self.native_probe_keys(plan, admitted)? else {
            return Ok(None);
        };
        self.collect_native_pairs(plan, admitted, keys)
    }

    pub(super) fn native_probe_keys(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
    ) -> Result<Option<NativeKeys>> {
        if !eligible(&self.compiled, self.input_schema(0)) {
            return Ok(None);
        }
        let layout = ArenaLayout::measure(admitted, &plan.key_indices)?;
        let bytes = match layout {
            Some(layout) => layout.charge,
            None => probe_key_charge(admitted, &plan.key_indices)?,
        };
        let Some(credit) = self.optional_credit(bytes)? else {
            return Ok(None);
        };
        let shared = Arc::new(credit);
        intern_probe_keys(admitted, &plan.key_indices, shared, &self.name, layout).map(Some)
    }

    fn collect_native_pairs(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
        keys: NativeKeys,
    ) -> Result<Option<NativeMatches>> {
        let opposite = if plan.incoming_is_left {
            &self.state.right
        } else {
            &self.state.left
        };
        let index = opposite
            .1
            .as_ref()
            .expect("native index built before probing");
        let windows = ProbeWindows::new(index, &keys);
        let count = count_pairs(
            &windows,
            admitted,
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
        let credit = self
            .runtime
            .runtime
            .as_ref()
            .expect("native probe initialized runtime")
            .incremental_reservation("stream-join-native");
        if credit.try_grow(bytes).is_err() {
            return Ok(None);
        }
        let mut pairs = Vec::with_capacity(count);
        for (pos, row) in admitted.iter().enumerate() {
            pairs.extend(
                windows
                    .range(
                        pos,
                        time_range(self.spec.bounds, plan.incoming_is_left, row.event_time),
                    )
                    .map(|opposite_index| MatchedPair {
                        pos,
                        opposite_index,
                    }),
            );
        }
        drop(windows);
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

    pub(super) fn ensure_native_index(&mut self, left: bool) -> Result<bool> {
        let rows = if left {
            &self.state.left
        } else {
            &self.state.right
        };
        if rows.1.is_some() {
            return Ok(true);
        }
        let Some(bytes) = NativeIndex::build_charge(rows.len()) else {
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
        rows.1 = Some(Arc::new(NativeIndex::new(rows, credit)));
        Ok(true)
    }
}

pub(super) fn time_range(
    bounds: JoinTimeBounds,
    incoming_is_left: bool,
    time: EventTime,
) -> (EventTime, EventTime) {
    let (before, after) = if incoming_is_left {
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

pub(super) fn probe_key_charge(admitted: &[AdmittedRow], indices: &[usize]) -> Result<usize> {
    let extra = admitted
        .len()
        .checked_mul(size_of::<u32>() + size_of::<Option<u32>>())
        .and_then(|bytes| bytes.checked_add(BASE_BYTES))
        .ok_or_else(|| scratch_error("join"))?;
    admitted.iter().try_fold(extra, |total, row| {
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

pub(super) fn scratch_error(name: &str) -> CalcFlowError {
    CalcFlowError::DataFusion {
        node_id: Some(name.to_owned()),
        message: "native Join scratch size overflow".to_owned(),
    }
}

fn count_pairs(
    windows: &ProbeWindows<'_>,
    admitted: &[AdmittedRow],
    bounds: JoinTimeBounds,
    plan: &SidePlan,
    limit: u64,
) -> Result<usize> {
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let mut count = 0_usize;
    for (pos, row) in admitted.iter().enumerate() {
        let remaining = limit.saturating_sub(count).saturating_add(1);
        count = count
            .checked_add(
                windows
                    .count(
                        pos,
                        time_range(bounds, plan.incoming_is_left, row.event_time),
                    )
                    .min(remaining),
            )
            .ok_or_else(|| scratch_error("join"))?;
        if count > limit {
            break;
        }
    }
    Ok(count)
}
