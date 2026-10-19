#[cfg(test)]
use super::AdmittedRow;
#[cfg(test)]
use super::StoredRow;
#[cfg(test)]
use super::borrowed_key::KeyHashState;
#[cfg(test)]
use super::borrowed_key::framed_hash;
use super::columnar::FramedKey;
#[cfg(test)]
use super::key_arena::{ArenaLayout, visit_key_frames, visit_string_values};
#[cfg(test)]
use super::native_dictionary::BASE_BYTES;
pub(super) use super::native_dictionary::{AppendCredit, NativeIndex, WindowBounds, WindowCursor};
use super::{
    AdmittedRows, CompiledJoin, JoinTimeBounds, MatchedPair, SidePlan, StreamJoinOperator,
    enforce_match_limit,
};
use crate::{CalcFlowError, EventTime, Result};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::execution::memory_pool::MemoryReservation;
#[cfg(test)]
use hashbrown::HashTable;
#[cfg(test)]
use std::hash::BuildHasher;
use std::sync::Arc;

#[cfg(test)]
#[path = "tests/native_index_allocation_tests.rs"]
mod allocation_tests;

#[cfg(test)]
#[path = "tests/native_window_probe_tests.rs"]
mod window_probe_tests;

#[cfg(test)]
pub(super) use super::native_dictionary::{reset_window_run_searches, window_run_searches};

pub(super) fn eligible(compiled: &CompiledJoin, left: &Schema) -> bool {
    compiled
        .left_key_indices
        .iter()
        .all(|&index| native_type(left.field(index).data_type()))
}

fn native_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean | DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
    ) || matches!(
        data_type,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
    ) || matches!(
        data_type,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(..)
    )
}

#[cfg(test)]
pub(super) struct NativeMatches {
    pub(super) pairs: crate::operator::join::matches::Selection,
    pub(super) keys: NativeKeys,
    pub(super) _credit: MemoryReservation,
}

pub(super) struct NativePairs {
    pub(super) pairs: crate::operator::join::matches::Selection,
    pub(super) credit: MemoryReservation,
}

pub(super) struct NativeKeys {
    pub(super) keys: Vec<Option<Arc<FramedKey>>>,
    pub(super) frequencies: Vec<usize>,
    ids: Vec<u32>,
    _credit: Arc<MemoryReservation>,
    _scratch: Arc<MemoryReservation>,
    pub(super) dictionary: Option<super::key_dictionary::Pending>,
}

#[cfg(test)]
thread_local! {
    static KEY_ID_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn reset_key_id_reads() {
    KEY_ID_READS.set(0);
}

#[cfg(test)]
pub(super) fn key_id_reads() -> usize {
    KEY_ID_READS.get()
}

impl NativeKeys {
    pub(super) fn id(&self, position: usize) -> u32 {
        #[cfg(test)]
        KEY_ID_READS.set(KEY_ID_READS.get() + 1);
        self.ids[position]
    }

    pub(super) fn key(&self, position: usize) -> &Arc<FramedKey> {
        self.keys[self.ids[position] as usize]
            .as_ref()
            .expect("retained key identity")
    }

    #[cfg(test)]
    pub(super) fn row_keys(&self) -> impl Iterator<Item = &Arc<FramedKey>> {
        self.ids
            .iter()
            .map(|&id| self.keys[id as usize].as_ref().expect("known fixture key"))
    }
}

// Opposite IDs cannot outlive the immutable index used by both window passes.
struct ProbeWindows<'a> {
    index: &'a NativeIndex,
    ids: &'a [u32],
    slots: Vec<Option<u32>>,
    cursors: Vec<Option<WindowCursor>>,
}

impl<'a> ProbeWindows<'a> {
    fn new(index: &'a NativeIndex, keys: &'a NativeKeys) -> Self {
        let mut slots = Vec::with_capacity(keys.keys.len());
        slots.extend(
            keys.keys
                .iter()
                .map(|key| key.as_ref().and_then(|key| index.key_id(key))),
        );
        let cursors = vec![None; slots.len()];
        Self {
            index,
            ids: &keys.ids,
            slots,
            cursors,
        }
    }

    /// Window bounds for one probe row, advancing the key's cursor so
    /// consecutive probes reuse the previous position.
    fn count(&mut self, pos: usize, range: (EventTime, EventTime)) -> WindowBounds {
        let slot = self.ids[pos] as usize;
        let Some(id) = self.slots.get(slot).copied().flatten() else {
            return WindowBounds::EMPTY;
        };
        let bounds = self
            .index
            .window_bounds_by_id(id, range, self.cursors[slot]);
        self.cursors[slot] = Some(WindowCursor::new(range, bounds));
        bounds
    }

    /// Dense opposite indices for bounds computed by the counting pass.
    fn range_bounds(&self, pos: usize, bounds: WindowBounds) -> impl Iterator<Item = usize> + '_ {
        self.slots
            .get(self.ids[pos] as usize)
            .copied()
            .flatten()
            .into_iter()
            .flat_map(move |id| self.index.range_by_bounds(id, bounds))
    }
}

#[cfg(test)]
struct ProbeKeyInterner {
    keys: Vec<Arc<FramedKey>>,
    ids: Vec<u32>,
    hashes: Vec<u64>,
    dictionary: HashTable<u32>,
    hasher: KeyHashState,
}

#[cfg(test)]
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

    fn push_string(
        &mut self,
        tag: u8,
        bytes: &[u8],
        credit: &Arc<MemoryReservation>,
        name: &str,
    ) -> Result<()> {
        self.push_string_hashed(tag, bytes, self.hasher.hash_one(bytes), credit, name)
    }

    fn push_string_hashed(
        &mut self,
        tag: u8,
        bytes: &[u8],
        hash: u64,
        credit: &Arc<MemoryReservation>,
        name: &str,
    ) -> Result<()> {
        self.push_with(
            hash,
            |stored| stored.first() == Some(&tag) && stored.get(9..) == Some(bytes),
            || {
                let length = u32::try_from(bytes.len())
                    .map_err(|_| super::charge_overflow("key value length"))?;
                let capacity = bytes
                    .len()
                    .checked_add(9)
                    .ok_or_else(|| scratch_error(name))?;
                let mut frame = Vec::with_capacity(capacity);
                frame.extend_from_slice(&[tag, 0, 0, 0, 0]);
                frame.extend_from_slice(&length.to_le_bytes());
                frame.extend_from_slice(bytes);
                #[cfg(test)]
                super::note_join_work(|work| {
                    work.arena_frames += 1;
                    work.key_encodings += 1;
                });
                Ok(super::columnar::funded_encoded_key(
                    frame,
                    Arc::clone(credit),
                ))
            },
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

    fn finish(self, credit: Arc<MemoryReservation>, name: &str) -> Result<NativeKeys> {
        let owner = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<FramedKey>())
            .map_err(|_| scratch_error(name))?
            .0
            .pad_to_align()
            .size();
        let retained = self.keys.iter().try_fold(BASE_BYTES, |bytes, key| {
            bytes.checked_add(owner)?.checked_add(key.capacity())
        });
        let scratch = self
            .keys
            .capacity()
            .checked_mul(size_of::<Arc<FramedKey>>())
            .and_then(|bytes| bytes.checked_add(self.ids.capacity().checked_mul(size_of::<u32>())?))
            .and_then(|bytes| bytes.checked_add(self.keys.len().checked_mul(size_of::<usize>())?))
            // Serial ProbeWindows allocates these slots while it borrows the IDs.
            .and_then(|bytes| {
                bytes.checked_add(self.keys.len().checked_mul(size_of::<Option<u32>>())?)
            });
        let (retained, scratch) = retained
            .zip(scratch)
            .filter(|(retained, scratch)| {
                retained
                    .checked_add(*scratch)
                    .is_some_and(|total| total <= credit.size())
            })
            .ok_or_else(|| scratch_error(name))?;
        let Self {
            keys,
            ids,
            hashes,
            dictionary,
            ..
        } = self;
        drop((hashes, dictionary));
        let mut frequencies = vec![0_usize; keys.len()];
        for &id in &ids {
            if let Some(count) = frequencies.get_mut(id as usize) {
                *count = count.saturating_add(1);
            }
        }
        // Key aliases may outlive this batch's IDs and lookup vectors.
        let scratch = credit.split(scratch);
        credit.shrink(credit.size() - retained);
        Ok(NativeKeys {
            keys: keys.into_iter().map(Some).collect(),
            frequencies,
            ids,
            _credit: credit,
            _scratch: Arc::new(scratch),
            dictionary: None,
        })
    }
}

#[cfg(test)]
fn intern_probe_keys(
    admitted: &dyn AdmittedRows,
    indices: &[usize],
    credit: Arc<MemoryReservation>,
    name: &str,
    layout: Option<ArenaLayout>,
) -> Result<NativeKeys> {
    let mut interner = ProbeKeyInterner::new(admitted.len());
    if let Some(layout) = layout.filter(|layout| layout.fits(admitted.len(), indices.len())) {
        if layout.single_string {
            visit_string_values(admitted, indices[0], |tag, bytes| {
                interner.push_string(tag, bytes, &credit, name)
            })?;
        } else {
            visit_key_frames(admitted, indices, layout, |bytes| {
                interner.push_bytes(bytes, &credit, name)
            })?;
        }
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
    interner.finish(credit, name)
}

#[cfg(test)]
pub(super) fn colliding_probe_keys(
    admitted: &dyn AdmittedRows,
    indices: &[usize],
    credit: MemoryReservation,
) -> Result<NativeKeys> {
    let shared = Arc::new(credit);
    let mut interner = ProbeKeyInterner::new(admitted.len());
    let layout = ArenaLayout::measure(admitted, indices)?.expect("eligible collision fixture");
    assert!(layout.fits(admitted.len(), indices.len()));
    if layout.single_string {
        visit_string_values(admitted, indices[0], |tag, bytes| {
            interner.push_string_hashed(tag, bytes, 0, &shared, "collision-test")
        })?;
    } else {
        visit_key_frames(admitted, indices, layout, |bytes| {
            interner.push_bytes_hashed(bytes, 0, &shared, "collision-test")
        })?;
    }
    interner.finish(shared, "collision-test")
}

impl StreamJoinOperator {
    pub(super) fn optional_credit(&mut self, bytes: usize) -> Result<Option<MemoryReservation>> {
        let credit = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-native");
        super::key_dictionary::grant(&credit, bytes, &self.name)
    }

    #[cfg(test)]
    pub(super) fn native_matches(
        &mut self,
        plan: &SidePlan,
        admitted: &dyn AdmittedRows,
        keys: Option<NativeKeys>,
    ) -> Result<Option<NativeMatches>> {
        let keys = match keys {
            Some(keys) => Some(keys),
            None => self.native_probe_keys(plan, admitted)?,
        };
        let Some(keys) = keys else {
            return Ok(None);
        };
        Ok(self
            .native_pairs(plan, admitted, &keys)?
            .map(|output| NativeMatches {
                pairs: output.pairs,
                keys,
                _credit: output.credit,
            }))
    }

    pub(super) fn native_pairs(
        &mut self,
        plan: &SidePlan,
        admitted: &dyn AdmittedRows,
        keys: &NativeKeys,
    ) -> Result<Option<NativePairs>> {
        if !eligible(&self.compiled, self.input_schema(0)) {
            return Ok(None);
        }
        if !self.ensure_native_index(!plan.incoming_is_left)? {
            return Ok(None);
        }
        let matched = self.collect_native_pairs(plan, admitted, keys)?;
        if matched.is_none() {
            self.drop_native_index(!plan.incoming_is_left);
        }
        Ok(matched)
    }

    #[cfg(test)]
    pub(super) fn native_probe_keys(
        &mut self,
        plan: &SidePlan,
        admitted: &dyn AdmittedRows,
    ) -> Result<Option<NativeKeys>> {
        futures::executor::block_on(self.prepare_key_identities(plan, admitted, None))
    }

    pub(super) async fn prepare_key_identities(
        &mut self,
        plan: &SidePlan,
        admitted: &dyn AdmittedRows,
        context: Option<&crate::StreamOperatorContext<'_>>,
    ) -> Result<Option<NativeKeys>> {
        self.key_dictionary.reclaim_retired(context).await?;
        if self.key_dictionary_needs_rebuild {
            self.key_work.needs_rebuild_routes = super::checked_metric(
                self.key_work.needs_rebuild_routes,
                1,
                &self.name,
                "needs_rebuild_routes",
            )?;
            return Ok(None);
        }
        if !eligible(&self.compiled, self.input_schema(0)) {
            return Err(super::charge_type_mismatch("Join key column"));
        }
        let master = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-dictionary");
        let Some(prepared) = self
            .key_dictionary
            .prepare(admitted, &plan.key_indices, master, context, &self.name)
            .await?
        else {
            self.key_work.dictionary_grant_refusals = super::checked_metric(
                self.key_work.dictionary_grant_refusals,
                1,
                &self.name,
                "dictionary_grant_refusals",
            )?;
            return Ok(None);
        };
        let super::key_dictionary::PreparedKeys {
            dictionary,
            keys,
            ids,
            frequencies,
            scratch,
        } = prepared;
        let credit = Arc::clone(&scratch);
        Ok(Some(NativeKeys {
            keys,
            ids,
            frequencies,
            _credit: credit,
            _scratch: scratch,
            dictionary: Some(dictionary),
        }))
    }

    fn collect_native_pairs(
        &mut self,
        plan: &SidePlan,
        admitted: &dyn AdmittedRows,
        keys: &NativeKeys,
    ) -> Result<Option<NativePairs>> {
        let scratch = window_scratch_charge(keys.keys.len(), admitted.len())
            .ok_or_else(|| scratch_error(&self.name))?;
        // Funds the per-key window cursors and the per-row window bounds that
        // the fill pass reuses; held until those vectors drop.
        let Some(_scratch_credit) = self.optional_credit(scratch)? else {
            return Ok(None);
        };
        let opposite = if plan.incoming_is_left {
            &self.state.right
        } else {
            &self.state.left
        };
        let index = opposite
            .1
            .as_ref()
            .expect("native index built before probing");
        let mut windows = ProbeWindows::new(index, keys);
        let (count, row_bounds) = count_pairs(
            &mut windows,
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
        let Some(credit) = super::key_dictionary::grant(&credit, bytes, &self.name)? else {
            return Ok(None);
        };
        let mut pairs = super::matches::Selection::with_capacity(count, admitted.len());
        for (pos, bounds) in row_bounds.into_iter().enumerate() {
            pairs.extend(
                windows
                    .range_bounds(pos, bounds)
                    .map(|opposite_index| MatchedPair {
                        pos,
                        opposite_index,
                    }),
            );
        }
        drop(windows);
        Ok(Some(NativePairs { pairs, credit }))
    }

    pub(super) fn drop_native_index(&mut self, left: bool) {
        if left {
            self.state.left.1 = None;
        } else {
            self.state.right.1 = None;
        }
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
            .reserve_checked(rows, &self.name)?;
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
        if self.key_dictionary_needs_rebuild {
            return Ok(false);
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
        rows.1 = Some(Arc::new(NativeIndex::new_identified(
            rows,
            &self.key_dictionary,
            self.key_bindings.side(left),
            credit,
        )));
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

#[cfg(test)]
pub(super) fn probe_key_charge(admitted: &dyn AdmittedRows, indices: &[usize]) -> Result<usize> {
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

// Window cursors (one per distinct probe key) plus the per-row window bounds
// stored by the counting pass and reused by the fill pass.
fn window_scratch_charge(keys: usize, rows: usize) -> Option<usize> {
    keys.checked_mul(size_of::<Option<WindowCursor>>())?
        .checked_add(rows.checked_mul(size_of::<WindowBounds>())?)?
        .checked_add(64)
}

fn count_pairs(
    windows: &mut ProbeWindows<'_>,
    admitted: &dyn AdmittedRows,
    bounds: JoinTimeBounds,
    plan: &SidePlan,
    limit: u64,
) -> Result<(usize, Vec<WindowBounds>)> {
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let mut count = 0_usize;
    let mut row_bounds = Vec::with_capacity(admitted.len());
    for pos in 0..admitted.len() {
        let bounds = windows.count(
            pos,
            time_range(bounds, plan.incoming_is_left, admitted.event_time(pos)),
        );
        row_bounds.push(bounds);
        let remaining = limit.saturating_sub(count).saturating_add(1);
        count = count
            .checked_add(bounds.matches().min(remaining))
            .ok_or_else(|| scratch_error("join"))?;
        if count > limit {
            break;
        }
    }
    Ok((count, row_bounds))
}
