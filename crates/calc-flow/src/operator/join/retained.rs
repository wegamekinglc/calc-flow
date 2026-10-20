//! Stable chunk coordinates for online state; wire rows are temporary views.

use std::{collections::BTreeMap, sync::Arc};

use super::{StoredRow, admitted::PayloadRef, columnar::RowPayload};
use crate::{EventTime, Result, StreamOperatorContext};
use datafusion::execution::memory_pool::MemoryReservation;
use hashbrown::HashTable;

pub(super) const CHUNK_ROWS: usize = 1_024;

pub(super) trait Rows: Send + Sync {
    fn batches(&self) -> Box<dyn Iterator<Item = RowBatch<'_>> + Send + '_>;
    fn len(&self) -> usize;
    fn row(&self, index: usize) -> StoredRow;
    fn charge(&self, index: usize) -> u64 {
        self.row(index).charge
    }
    fn identity(&self, index: usize) -> (EventTime, u64) {
        let row = self.row(index);
        (row.event_time, row.row_id)
    }
    fn payload(&self, index: usize) -> PayloadRef<'_>;
    fn cursor_payload<'a>(
        &'a self,
        index: usize,
        _chunk: &mut Option<(usize, &'a Chunk)>,
    ) -> PayloadRef<'a> {
        self.payload(index)
    }
    fn key(&self, index: usize) -> &Arc<super::columnar::FramedKey>;
    fn indexed(&self) -> Box<dyn Iterator<Item = (usize, StoredRow)> + Send + '_>;
    fn references(&self) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        Box::new(self.indexed().map(|(index, _)| index))
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stable coordinates of the live rows whose `row_id` is at or above the
    /// watermark (spec FR3). Chunks entirely below the watermark are skipped
    /// by their ids, so the derive costs the dirty rows, not the state.
    fn references_from(&self, watermark: u64) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        Box::new(
            self.references()
                .filter(move |&index| self.identity(index).1 >= watermark),
        )
    }
}

impl dyn Rows + '_ {
    pub(super) fn cursor(&self) -> PayloadCursor<'_> {
        PayloadCursor {
            rows: self,
            chunk: None,
        }
    }

    #[cfg(any(test, feature = "benchmark-checkpoint-diagnostic"))]
    pub(super) fn iter(&self) -> Box<dyn Iterator<Item = StoredRow> + Send + '_> {
        Box::new(self.indexed().map(|(_, row)| row))
    }
}

/// A borrowed chunk cannot outlive the immutable state snapshot or overlap a
/// removal/repair. Storage stays constant even when stable coordinates are sparse.
pub(super) struct PayloadCursor<'a> {
    rows: &'a dyn Rows,
    chunk: Option<(usize, &'a Chunk)>,
}

impl<'a> PayloadCursor<'a> {
    pub(super) fn payload(&mut self, index: usize) -> PayloadRef<'a> {
        self.rows.cursor_payload(index, &mut self.chunk)
    }
}

struct Source {
    owner: RowPayload,
    count: usize,
}

struct KeyOwner {
    key: Arc<super::columnar::FramedKey>,
    count: usize,
}

fn key_hash(key: &Arc<super::columnar::FramedKey>) -> u64 {
    Arc::as_ptr(key) as usize as u64
}

#[derive(Default)]
pub(super) struct Chunk {
    sources: Vec<Option<Source>>,
    source_ids: Vec<usize>,
    offsets: Vec<usize>,
    times: Vec<EventTime>,
    ids: Vec<u64>,
    charges: Vec<u64>,
    keys: Vec<Option<u32>>,
    key_owners: Vec<Option<KeyOwner>>,
    key_slots: HashTable<usize>,
    live: usize,
    credit: Option<MemoryReservation>,
}

impl Chunk {
    fn source(&mut self, owner: RowPayload) -> usize {
        let columns = owner.columns().as_ptr();
        if let Some(index) = self.sources.iter().position(|source| {
            source
                .as_ref()
                .is_some_and(|source| source.owner.columns().as_ptr() == columns)
        }) {
            self.sources[index].as_mut().unwrap().count += 1;
            return index;
        }
        let index = self
            .sources
            .iter()
            .position(Option::is_none)
            .unwrap_or(self.sources.len());
        let source = Some(Source { owner, count: 1 });
        if index == self.sources.len() {
            self.sources.push(source);
        } else {
            self.sources[index] = source;
        }
        index
    }

    fn reserved(count: usize, sources: usize, credit: Option<MemoryReservation>) -> Self {
        Self::reserved_keys(count, sources, count, credit)
    }

    fn reserved_keys(
        count: usize,
        sources: usize,
        keys: usize,
        credit: Option<MemoryReservation>,
    ) -> Self {
        Self {
            // Sparse repair replaces every live row of one payload owner together.
            // One spare slot covers overlap until the previous owner retires.
            sources: Vec::with_capacity(sources + 1),
            source_ids: Vec::with_capacity(count),
            offsets: Vec::with_capacity(count),
            times: Vec::with_capacity(count),
            ids: Vec::with_capacity(count),
            charges: Vec::with_capacity(count),
            keys: Vec::with_capacity(count),
            key_owners: Vec::with_capacity(keys),
            key_slots: HashTable::with_capacity(keys),
            live: 0,
            credit,
        }
    }

    fn push(&mut self, row: StoredRow) {
        self.offsets.push(row.record.offset());
        let source = self.source(row.record);
        self.source_ids.push(source);
        self.times.push(row.event_time);
        self.ids.push(row.row_id);
        self.charges.push(row.charge);
        let key = self.intern_key(&row.encoded_key);
        self.keys
            .push(Some(u32::try_from(key).expect("chunk key slot fits u32")));
        self.live += 1;
    }

    fn push_columns(
        &mut self,
        payload: PayloadRef<'_>,
        time: EventTime,
        id: u64,
        charge: u64,
        key: &Arc<super::columnar::FramedKey>,
    ) {
        self.offsets.push(payload.offset());
        let columns = payload.columns().as_ptr();
        let source = match self.sources.iter().position(|source| {
            source
                .as_ref()
                .is_some_and(|source| source.owner.columns().as_ptr() == columns)
        }) {
            Some(index) => {
                self.sources[index].as_mut().unwrap().count += 1;
                index
            }
            None => self.source(payload.owned()),
        };
        self.source_ids.push(source);
        self.times.push(time);
        self.ids.push(id);
        self.charges.push(charge);
        let key = self.intern_key(key);
        self.keys
            .push(Some(u32::try_from(key).expect("chunk key slot fits u32")));
        self.live += 1;
    }

    fn owner(&self, offset: usize) -> &RowPayload {
        &self.sources[self.source_ids[offset]]
            .as_ref()
            .unwrap()
            .owner
    }

    fn row(&self, offset: usize) -> StoredRow {
        #[cfg(test)]
        super::note_join_work(|work| work.retained_descriptor_views += 1);
        StoredRow {
            record: match self.owner(offset) {
                RowPayload::Legacy(record) => RowPayload::Legacy(record.clone()),
                owner => owner.with_offset(self.offsets[offset]),
            },
            event_time: self.times[offset],
            row_id: self.ids[offset],
            charge: self.charges[offset],
            encoded_key: Arc::clone(self.key(offset)),
        }
    }

    fn intern_key(&mut self, key: &Arc<super::columnar::FramedKey>) -> usize {
        let hash = key_hash(key);
        if let Some(&index) = self.key_slots.find(hash, |&index| {
            Arc::ptr_eq(&self.key_owners[index].as_ref().unwrap().key, key)
        }) {
            self.key_owners[index].as_mut().unwrap().count += 1;
            return index;
        }
        let index = self
            .key_owners
            .iter()
            .position(Option::is_none)
            .unwrap_or(self.key_owners.len());
        let owner = Some(KeyOwner {
            key: Arc::clone(key),
            count: 1,
        });
        if index == self.key_owners.len() {
            self.key_owners.push(owner);
        } else {
            self.key_owners[index] = owner;
        }
        let Self {
            key_slots,
            key_owners,
            ..
        } = self;
        key_slots.insert_unique(hash, index, |&index| {
            key_hash(&key_owners[index].as_ref().unwrap().key)
        });
        index
    }

    fn key(&self, offset: usize) -> &Arc<super::columnar::FramedKey> {
        &self.key_owners[self.keys[offset].expect("live chunk row") as usize]
            .as_ref()
            .expect("live chunk key")
            .key
    }

    fn release_key(&mut self, offset: usize) {
        let index = self.keys[offset].take().expect("live chunk row") as usize;
        let owner = self.key_owners[index].as_mut().expect("live chunk key");
        owner.count -= 1;
        if owner.count == 0 {
            let hash = key_hash(&owner.key);
            self.key_slots
                .find_entry(hash, |&candidate| candidate == index)
                .unwrap_or_else(|_| unreachable!())
                .remove();
            self.key_owners[index] = None;
        }
    }

    fn release_source(&mut self, offset: usize) {
        let source = &mut self.sources[self.source_ids[offset]];
        source.as_mut().unwrap().count -= 1;
        if source.as_ref().unwrap().count == 0 {
            *source = None;
        }
    }

    #[cfg(test)]
    fn remove(&mut self, offset: usize) -> StoredRow {
        let row = self.row(offset);
        self.release_key(offset);
        self.release_source(offset);
        self.live -= 1;
        row
    }
}

pub(super) struct Append {
    chunks: Vec<Chunk>,
    pub(super) indices: Vec<usize>,
    count: usize,
    control: Option<MemoryReservation>,
}

impl Append {
    /// New chunks retain payload owners in first-seen order.
    pub(super) fn payload_owners(&self) -> impl Iterator<Item = &RowPayload> {
        self.chunks.iter().flat_map(|chunk| {
            chunk
                .sources
                .iter()
                .filter_map(|source| source.as_ref().map(|source| &source.owner))
        })
    }

    pub(super) fn is_unfunded(&self) -> bool {
        self.count != 0
            && (self.control.is_none() || self.chunks.iter().any(|chunk| chunk.credit.is_none()))
    }
}

#[derive(Default)]
pub(super) struct StateRows {
    chunks: BTreeMap<usize, Chunk>,
    next: usize,
    len: usize,
}

impl StateRows {
    fn payload_chunk(&self, id: usize) -> &Chunk {
        #[cfg(test)]
        tests::PAYLOAD_LOOKUPS.with(|count| count.set(count.get() + 1));
        &self.chunks[&id]
    }

    #[cfg(test)]
    fn reference(&self, mut ordinal: usize) -> usize {
        for (&id, chunk) in &self.chunks {
            if ordinal < chunk.live {
                let offset = chunk
                    .keys
                    .iter()
                    .enumerate()
                    .filter(|(_, key)| key.is_some())
                    .nth(ordinal)
                    .expect("live ordinal")
                    .0;
                return id * CHUNK_ROWS + offset;
            }
            ordinal -= chunk.live;
        }
        panic!("live row ordinal");
    }
    #[cfg(test)]
    pub(super) fn at(&self, ordinal: usize) -> StoredRow {
        self.row(self.reference(ordinal))
    }
    #[cfg(test)]
    pub(super) fn payload_at(&self, ordinal: usize) -> PayloadRef<'_> {
        self.payload(self.reference(ordinal))
    }
    #[cfg(test)]
    pub(super) fn key_at(&self, ordinal: usize) -> &Arc<super::columnar::FramedKey> {
        self.key(self.reference(ordinal))
    }
    #[cfg(all(test, feature = "benchmark-checkpoint-diagnostic"))]
    pub(super) fn replace_at(&mut self, ordinal: usize, row: StoredRow) {
        let reference = self.reference(ordinal);
        self.replace_payload(reference, row.record);
        let chunk = self.chunks.get_mut(&(reference / CHUNK_ROWS)).unwrap();
        let offset = reference % CHUNK_ROWS;
        chunk.times[offset] = row.event_time;
        chunk.ids[offset] = row.row_id;
        chunk.charges[offset] = row.charge;
        chunk.release_key(offset);
        chunk.keys[offset] = Some(
            u32::try_from(chunk.intern_key(&row.encoded_key)).expect("chunk key slot fits u32"),
        );
    }
    #[cfg(test)]
    pub(super) fn last(&self) -> Option<StoredRow> {
        self.iter().last()
    }
    #[cfg(test)]
    pub(super) fn funded_bytes(&self) -> usize {
        self.chunks
            .values()
            .filter_map(|chunk| chunk.credit.as_ref())
            .map(MemoryReservation::size)
            .sum()
    }
    #[cfg(test)]
    pub(super) fn to_vec(&self) -> Vec<StoredRow> {
        self.iter().collect()
    }
    #[cfg(test)]
    pub(super) fn as_ptr(&self) -> *const Self {
        self
    }
    #[cfg(test)]
    pub(super) fn push(&mut self, row: StoredRow) {
        self.append(vec![row]);
    }
    pub(super) fn batch_at(&self, reference: usize) -> RowBatch<'_> {
        let base = reference / CHUNK_ROWS * CHUNK_ROWS;
        RowBatch::Columns {
            chunk: &self.chunks[&(base / CHUNK_ROWS)],
            base,
        }
    }
    pub(super) fn row_id(&self, index: usize) -> u64 {
        self.chunks[&(index / CHUNK_ROWS)].ids[index % CHUNK_ROWS]
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub(super) fn iter(&self) -> Box<dyn Iterator<Item = StoredRow> + Send + '_> {
        Box::new(self.indexed().map(|(_, row)| row))
    }
    pub(super) fn indexed(&self) -> Box<dyn Iterator<Item = (usize, StoredRow)> + Send + '_> {
        Rows::indexed(self)
    }
    pub(super) fn get(&self, index: usize) -> Option<StoredRow> {
        self.chunks
            .get(&(index / CHUNK_ROWS))?
            .keys
            .get(index % CHUNK_ROWS)?
            .as_ref()?;
        Some(self.row(index))
    }

    pub(super) async fn prepare_append(
        &self,
        rows: &[StoredRow],
        master: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
        legacy: bool,
    ) -> Result<Append> {
        self.validate_append(rows.len())?;
        let legacy = legacy
            && rows
                .iter()
                .all(|row| matches!(row.record, RowPayload::Legacy(_) | RowPayload::Rowed { .. }));
        let control = reserve_references(master, rows.len(), context.operator_id(), legacy)?;
        let indices = self.append_indices(rows.len());
        let mut chunks = Vec::with_capacity(rows.len().div_ceil(CHUNK_ROWS));
        let mut quantum = super::columnar::Quantum::default();
        for rows in rows.chunks(CHUNK_ROWS) {
            chunks.push(prepare_chunk(rows, master, context, &mut quantum, legacy).await?);
        }
        Ok(Append {
            chunks,
            indices,
            count: rows.len(),
            control,
        })
    }

    fn validate_append(&self, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        self.next
            .checked_add((count - 1) / CHUNK_ROWS)
            .and_then(|index| index.checked_mul(CHUNK_ROWS))
            .and_then(|index| index.checked_add((count - 1) % CHUNK_ROWS))
            .ok_or_else(|| super::native_lookup::scratch_error("Join chunk coordinates"))?;
        Ok(())
    }

    pub(super) fn commit(&mut self, append: Append) {
        let first = self.next;
        self.next += append.chunks.len();
        self.len += append.count;
        for (ordinal, chunk) in append.chunks.into_iter().enumerate() {
            self.chunks.insert(first + ordinal, chunk);
        }
    }

    pub(super) fn append_indices(&self, count: usize) -> Vec<usize> {
        (0..count)
            .map(|position| {
                (self.next + position / CHUNK_ROWS) * CHUNK_ROWS + position % CHUNK_ROWS
            })
            .collect()
    }

    /// The restore decoder has already funded the maximum live-row containers.
    /// Fill one owned chunk at a time rather than collecting an entire side.
    pub(super) fn push_restored(&mut self, row: StoredRow, expected: usize) -> Result<()> {
        if self.len >= expected {
            return Err(crate::CalcFlowError::CheckpointMismatch {
                message: "V2 history exceeds its funded row count".into(),
            });
        }
        let new_chunk = self
            .chunks
            .last_key_value()
            .is_none_or(|(_, chunk)| chunk.ids.len() == CHUNK_ROWS);
        if new_chunk {
            let capacity = (expected - self.len).min(CHUNK_ROWS);
            self.chunks
                .insert(self.next, Chunk::reserved(capacity, capacity, None));
            self.next += 1;
        }
        self.chunks
            .last_entry()
            .expect("restored chunk")
            .get_mut()
            .push(row);
        self.len += 1;
        Ok(())
    }

    pub(super) fn append(&mut self, rows: Vec<StoredRow>) -> Vec<usize> {
        let mut indices = Vec::with_capacity(rows.len());
        let mut rows = rows.into_iter();
        while !rows.as_slice().is_empty() {
            let id = self.next;
            self.next += 1;
            let capacity = rows.len().min(CHUNK_ROWS);
            let source_count = source_count(&rows.as_slice()[..capacity]);
            let mut chunk = Chunk::reserved(capacity, source_count, None);
            for row in rows.by_ref().take(CHUNK_ROWS) {
                indices.push(id * CHUNK_ROWS + chunk.ids.len());
                chunk.push(row);
            }
            self.len += chunk.live;
            self.chunks.insert(id, chunk);
        }
        indices
    }

    #[cfg(test)]
    pub(super) fn remove(&mut self, index: usize) -> StoredRow {
        #[cfg(test)]
        super::note_join_work(|work| work.retained_chunk_visits += 1);
        let id = index / CHUNK_ROWS;
        let chunk = self.chunks.get_mut(&id).expect("live chunk");
        let row = chunk.remove(index % CHUNK_ROWS);
        if chunk.live == 0 {
            self.chunks.remove(&id);
        }
        self.len -= 1;
        row
    }

    /// Bulk prefix removal (spec FR11): clears one planned offset set per
    /// chunk in a single pass, releasing payload sources and sparse-repair
    /// liveness, dropping emptied chunks, and updating the aggregate length
    /// once. Surviving coordinates stay stable.
    pub(super) fn remove_bulk(
        &mut self,
        groups: &BTreeMap<usize, Vec<usize>>,
        queue: &mut super::columnar::SparseQueue,
    ) {
        let mut removed = 0_usize;
        let mut emptied = Vec::new();
        for (&id, offsets) in groups {
            #[cfg(test)]
            super::note_join_work(|work| work.retained_chunk_visits += 1);
            let Some(chunk) = self.chunks.get_mut(&id) else {
                continue;
            };
            for &offset in offsets {
                // The sparse queue tracks the payload offset, which sparse
                // repairs can move independently of the chunk slot.
                let payload_offset = chunk.offsets[offset];
                let owner = chunk.owner(offset);
                queue.remove(&owner.with_offset(payload_offset));
            }
            for &offset in offsets {
                chunk.release_key(offset);
                chunk.release_source(offset);
                chunk.live -= 1;
            }
            removed += offsets.len();
            if chunk.live == 0 {
                emptied.push(id);
            }
        }
        for id in emptied {
            self.chunks.remove(&id);
        }
        self.len -= removed;
    }

    pub(super) fn replace_payload(&mut self, index: usize, payload: RowPayload) {
        let chunk = self
            .chunks
            .get_mut(&(index / CHUNK_ROWS))
            .expect("live chunk");
        let offset = index % CHUNK_ROWS;
        chunk.release_source(offset);
        chunk.offsets[offset] = payload.offset();
        chunk.source_ids[offset] = chunk.source(payload);
    }

    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }
}

/// Borrowed column block; row descriptors are compatibility views only.
#[derive(Clone, Copy)]
pub(super) enum RowBatch<'a> {
    Columns { chunk: &'a Chunk, base: usize },
    Legacy(&'a [StoredRow]),
}
impl RowBatch<'_> {
    pub(super) fn references(&self) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        match self {
            Self::Columns { chunk, base } => Box::new(
                chunk
                    .keys
                    .iter()
                    .enumerate()
                    .filter_map(move |(offset, key)| key.as_ref().map(|_| base + offset)),
            ),
            Self::Legacy(rows) => Box::new(0..rows.len()),
        }
    }
    pub(super) fn identity(&self, reference: usize) -> (EventTime, u64) {
        match self {
            Self::Columns { chunk, base } => {
                (chunk.times[reference - base], chunk.ids[reference - base])
            }
            Self::Legacy(rows) => (rows[reference].event_time, rows[reference].row_id),
        }
    }
    pub(super) fn key(&self, reference: usize) -> &Arc<super::columnar::FramedKey> {
        match self {
            Self::Columns { chunk, base } => chunk.key(reference - base),
            Self::Legacy(rows) => &rows[reference].encoded_key,
        }
    }
    pub(super) fn payload(&self, reference: usize) -> PayloadRef<'_> {
        match self {
            Self::Columns { chunk, base } => PayloadRef::Payload(
                chunk.owner(reference - base),
                chunk.offsets[reference - base],
            ),
            Self::Legacy(rows) => (&rows[reference].record).into(),
        }
    }
    pub(super) fn distinct_keys(
        &self,
    ) -> Box<dyn Iterator<Item = (usize, &Arc<super::columnar::FramedKey>)> + Send + '_> {
        match self {
            Self::Columns { chunk, .. } => Box::new(
                chunk
                    .key_owners
                    .iter()
                    .enumerate()
                    .filter_map(|(index, owner)| owner.as_ref().map(|owner| (index, &owner.key))),
            ),
            Self::Legacy(rows) => Box::new(
                rows.iter()
                    .enumerate()
                    .map(|(index, row)| (index, &row.encoded_key)),
            ),
        }
    }
    pub(super) fn key_slot(&self, reference: usize) -> usize {
        match self {
            Self::Columns { chunk, base } => {
                chunk.keys[reference - base].expect("live chunk row") as usize
            }
            Self::Legacy(_) => reference,
        }
    }
    pub(super) fn charge(&self, reference: usize) -> u64 {
        match self {
            Self::Columns { chunk, base } => chunk.charges[reference - base],
            Self::Legacy(rows) => rows[reference].charge,
        }
    }
}

impl Rows for Append {
    fn len(&self) -> usize {
        self.count
    }
    fn row(&self, index: usize) -> StoredRow {
        self.chunks[index / CHUNK_ROWS].row(index % CHUNK_ROWS)
    }
    fn identity(&self, index: usize) -> (EventTime, u64) {
        let chunk = &self.chunks[index / CHUNK_ROWS];
        (
            chunk.times[index % CHUNK_ROWS],
            chunk.ids[index % CHUNK_ROWS],
        )
    }
    fn charge(&self, index: usize) -> u64 {
        self.chunks[index / CHUNK_ROWS].charges[index % CHUNK_ROWS]
    }
    fn payload(&self, index: usize) -> PayloadRef<'_> {
        let chunk = &self.chunks[index / CHUNK_ROWS];
        let offset = index % CHUNK_ROWS;
        PayloadRef::Payload(chunk.owner(offset), chunk.offsets[offset])
    }
    fn key(&self, index: usize) -> &Arc<super::columnar::FramedKey> {
        self.chunks[index / CHUNK_ROWS].key(index % CHUNK_ROWS)
    }
    fn indexed(&self) -> Box<dyn Iterator<Item = (usize, StoredRow)> + Send + '_> {
        Box::new((0..self.count).map(|index| (index, self.row(index))))
    }
    fn references(&self) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        Box::new(0..self.count)
    }
    fn batches(&self) -> Box<dyn Iterator<Item = RowBatch<'_>> + Send + '_> {
        Box::new(
            self.chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| RowBatch::Columns {
                    chunk,
                    base: index * CHUNK_ROWS,
                }),
        )
    }
}

impl Rows for StateRows {
    fn batches(&self) -> Box<dyn Iterator<Item = RowBatch<'_>> + Send + '_> {
        Box::new(self.chunks.iter().map(|(&index, chunk)| RowBatch::Columns {
            chunk,
            base: index * CHUNK_ROWS,
        }))
    }

    fn references(&self) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        Box::new(self.chunks.iter().flat_map(|(&id, chunk)| {
            chunk
                .keys
                .iter()
                .enumerate()
                .filter_map(move |(offset, key)| key.as_ref().map(|_| id * CHUNK_ROWS + offset))
        }))
    }

    fn references_from(&self, watermark: u64) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        // Ids ascend in admission order, so a chunk whose last id lies below
        // the watermark cannot hold a dirty row; restored chunks always sit
        // wholly below the restored watermark and skip the same way.
        Box::new(
            self.chunks
                .iter()
                .skip_while(move |(_, chunk)| chunk.ids.last().is_some_and(|id| *id < watermark))
                .flat_map(move |(&id, chunk)| {
                    let start = chunk.ids.partition_point(|id| *id < watermark);
                    chunk.keys[start..]
                        .iter()
                        .enumerate()
                        .filter_map(move |(offset, key)| {
                            key.as_ref().map(|_| id * CHUNK_ROWS + start + offset)
                        })
                }),
        )
    }

    fn len(&self) -> usize {
        self.len
    }

    fn row(&self, index: usize) -> StoredRow {
        self.chunks[&(index / CHUNK_ROWS)].row(index % CHUNK_ROWS)
    }

    fn charge(&self, index: usize) -> u64 {
        self.chunks[&(index / CHUNK_ROWS)].charges[index % CHUNK_ROWS]
    }
    fn identity(&self, index: usize) -> (EventTime, u64) {
        let chunk = &self.chunks[&(index / CHUNK_ROWS)];
        let offset = index % CHUNK_ROWS;
        (chunk.times[offset], chunk.ids[offset])
    }

    fn payload(&self, index: usize) -> PayloadRef<'_> {
        let chunk = self.payload_chunk(index / CHUNK_ROWS);
        let offset = index % CHUNK_ROWS;
        PayloadRef::Payload(chunk.owner(offset), chunk.offsets[offset])
    }

    fn cursor_payload<'a>(
        &'a self,
        index: usize,
        cursor: &mut Option<(usize, &'a Chunk)>,
    ) -> PayloadRef<'a> {
        let id = index / CHUNK_ROWS;
        let chunk = match *cursor {
            Some((previous, chunk)) if previous == id => chunk,
            _ => {
                let chunk = self.payload_chunk(id);
                *cursor = Some((id, chunk));
                chunk
            }
        };
        let offset = index % CHUNK_ROWS;
        PayloadRef::Payload(chunk.owner(offset), chunk.offsets[offset])
    }

    fn key(&self, index: usize) -> &Arc<super::columnar::FramedKey> {
        self.chunks[&(index / CHUNK_ROWS)].key(index % CHUNK_ROWS)
    }

    fn indexed(&self) -> Box<dyn Iterator<Item = (usize, StoredRow)> + Send + '_> {
        Box::new(self.chunks.iter().flat_map(|(&id, chunk)| {
            chunk
                .keys
                .iter()
                .enumerate()
                .filter_map(move |(offset, key)| {
                    key.as_ref()
                        .map(|_| (id * CHUNK_ROWS + offset, chunk.row(offset)))
                })
        }))
    }
}

impl Rows for Vec<StoredRow> {
    fn batches(&self) -> Box<dyn Iterator<Item = RowBatch<'_>> + Send + '_> {
        Box::new(std::iter::once(RowBatch::Legacy(self)))
    }

    fn charge(&self, index: usize) -> u64 {
        self[index].charge
    }
    fn identity(&self, index: usize) -> (EventTime, u64) {
        (self[index].event_time, self[index].row_id)
    }

    fn references(&self) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        Box::new(0..self.len())
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }
    fn row(&self, index: usize) -> StoredRow {
        self[index].clone()
    }
    fn payload(&self, index: usize) -> PayloadRef<'_> {
        (&self[index].record).into()
    }
    fn key(&self, index: usize) -> &Arc<super::columnar::FramedKey> {
        &self[index].encoded_key
    }
    fn indexed(&self) -> Box<dyn Iterator<Item = (usize, StoredRow)> + Send + '_> {
        Box::new(self.as_slice().iter().cloned().enumerate())
    }
}

impl Rows for Arc<StateRows> {
    fn batches(&self) -> Box<dyn Iterator<Item = RowBatch<'_>> + Send + '_> {
        self.as_ref().batches()
    }
    fn references(&self) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        self.as_ref().references()
    }

    fn references_from(&self, watermark: u64) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        self.as_ref().references_from(watermark)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
    fn row(&self, index: usize) -> StoredRow {
        self.as_ref().row(index)
    }
    fn charge(&self, index: usize) -> u64 {
        self.as_ref().charge(index)
    }
    fn identity(&self, index: usize) -> (EventTime, u64) {
        self.as_ref().identity(index)
    }
    fn payload(&self, index: usize) -> PayloadRef<'_> {
        self.as_ref().payload(index)
    }
    fn cursor_payload<'a>(
        &'a self,
        index: usize,
        cursor: &mut Option<(usize, &'a Chunk)>,
    ) -> PayloadRef<'a> {
        self.as_ref().cursor_payload(index, cursor)
    }
    fn key(&self, index: usize) -> &Arc<super::columnar::FramedKey> {
        self.as_ref().key(index)
    }
    fn indexed(&self) -> Box<dyn Iterator<Item = (usize, StoredRow)> + Send + '_> {
        self.as_ref().indexed()
    }
}

impl<const N: usize> Rows for [StoredRow; N] {
    fn batches(&self) -> Box<dyn Iterator<Item = RowBatch<'_>> + Send + '_> {
        Box::new(std::iter::once(RowBatch::Legacy(self.as_slice())))
    }

    fn references(&self) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        Box::new(0..N)
    }

    fn len(&self) -> usize {
        N
    }
    fn row(&self, index: usize) -> StoredRow {
        self[index].clone()
    }
    fn payload(&self, index: usize) -> PayloadRef<'_> {
        (&self[index].record).into()
    }
    fn key(&self, index: usize) -> &Arc<super::columnar::FramedKey> {
        &self[index].encoded_key
    }
    fn indexed(&self) -> Box<dyn Iterator<Item = (usize, StoredRow)> + Send + '_> {
        Box::new(self.as_slice().iter().cloned().enumerate())
    }
}

impl From<Vec<StoredRow>> for StateRows {
    fn from(rows: Vec<StoredRow>) -> Self {
        let mut state = Self::default();
        state.append(rows);
        state
    }
}
impl<'a> IntoIterator for &'a StateRows {
    type Item = StoredRow;
    type IntoIter = Box<dyn Iterator<Item = StoredRow> + Send + 'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

fn reserve_references(
    master: &MemoryReservation,
    count: usize,
    name: &str,
    legacy: bool,
) -> Result<Option<MemoryReservation>> {
    if count != 0 {
        let bytes = count
            .checked_mul(size_of::<usize>())
            .and_then(|bytes| bytes.checked_add(4096))
            .ok_or_else(|| super::native_lookup::scratch_error("Join chunk references"))?;
        return grant(master, bytes, name, legacy);
    }
    Ok(None)
}

fn grant(
    master: &MemoryReservation,
    bytes: usize,
    name: &str,
    legacy: bool,
) -> Result<Option<MemoryReservation>> {
    let credit = master.new_empty();
    match credit.try_grow(bytes) {
        Ok(()) => Ok(Some(credit)),
        Err(datafusion::common::DataFusionError::ResourcesExhausted(_)) if legacy => Ok(None),
        Err(error) => Err(crate::CalcFlowError::DataFusion {
            node_id: Some(name.to_owned()),
            message: format!("Join chunk credit admission failed: {error}"),
        }),
    }
}

async fn prepare_chunk(
    rows: &[StoredRow],
    master: &MemoryReservation,
    context: &StreamOperatorContext<'_>,
    quantum: &mut super::columnar::Quantum,
    legacy: bool,
) -> Result<Chunk> {
    context.check_cancelled()?;
    let sources = source_count(rows);
    let credit = grant(master, chunk_bytes(rows)?, context.operator_id(), legacy)?;
    let mut chunk = Chunk::reserved(rows.len(), sources, credit);
    for row in rows {
        quantum.step(context, 1, 128).await?;
        chunk.push(row.clone());
    }
    Ok(chunk)
}

fn source_count(rows: &[StoredRow]) -> usize {
    1 + rows
        .windows(2)
        .filter(|pair| pair[0].record.columns().as_ptr() != pair[1].record.columns().as_ptr())
        .count()
}

fn chunk_bytes(rows: &[StoredRow]) -> Result<usize> {
    chunk_base_bytes(rows.len(), source_count(rows), rows.len())
        .and_then(|fixed| {
            rows.iter().try_fold(fixed, |bytes, row| {
                let legacy = if matches!(row.record, RowPayload::Legacy(_)) {
                    row.record
                        .num_columns()
                        .checked_mul(size_of::<datafusion::arrow::array::ArrayRef>())?
                } else {
                    0
                };
                bytes.checked_add(legacy)
            })
        })
        .ok_or_else(|| super::native_lookup::scratch_error("Join state chunks"))
}

fn chunk_base_bytes(count: usize, source_count: usize, key_count: usize) -> Option<usize> {
    let width = 2 * size_of::<usize>()
        + size_of::<EventTime>()
        + 2 * size_of::<u64>()
        + size_of::<Option<u32>>();
    let sources = source_count
        .checked_add(1)?
        .checked_mul(size_of::<Option<Source>>())?;
    count
        .checked_mul(width)?
        .checked_add(sources)?
        .checked_add(key_count.checked_mul(size_of::<Option<KeyOwner>>() + 24)?)?
        .checked_add(8192)
}

mod admission;
pub(super) use admission::ColumnarPlan;
#[cfg(test)]
pub(super) use admission::parallel::observe_preparation_for_test;

#[cfg(test)]
mod tests;
