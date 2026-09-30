//! Sorted Arrow chunks, merged through borrowed cursor heads.

use super::{
    BatchKey, Encoding, EncodingOwners, LeftOrder, LegacyLeftState, PayloadBatch, PayloadPool,
    RowPayload, RowRef, SequenceColumn, SequenceKind, SequenceRef,
};
use crate::{
    Result,
    operator::asof::{AsofJoinSide, arrow_error, checked},
};
use datafusion::arrow::{
    array::UInt32Array,
    buffer::ScalarBuffer,
    compute::{SortColumn, lexsort_to_indices, take},
};
use std::{
    borrow::Cow,
    cmp::{Ordering, Reverse},
    collections::{BTreeMap, BinaryHeap, HashMap},
    sync::Arc,
};

pub(in super::super) type LeftView<'a> = (&'a i64, &'a Encoding, SequenceRef<'a>);

fn borrowed(order: &LeftOrder) -> LeftView<'_> {
    (&order.0, &order.1, Cow::Borrowed(&order.2))
}

#[derive(Clone)]
pub(in super::super) struct ChunkData {
    pub times: ScalarBuffer<i64>,
    pub positions: Option<Vec<u32>>,
    pub start: u32,
    pub keys: Vec<Option<Encoding>>,
    pub key_counts: Vec<usize>,
    pub key_ids: Vec<u32>,
    pub sequences: SequenceColumn,
    pub index_bytes: u64,
    pub owners: EncodingOwners,
}

fn chunk_sort_indices(
    rows: &[(LeftOrder, u32)],
    batch: &PayloadBatch,
    side: &AsofJoinSide,
    contiguous: bool,
) -> Result<UInt32Array> {
    let physical = UInt32Array::from(rows.iter().map(|(_, row)| *row).collect::<Vec<_>>());
    let record = &batch.record;
    let names = std::iter::once(side.event_time()).chain(
        side.keys()
            .iter()
            .chain(side.sequence_by())
            .map(String::as_str),
    );
    let columns = names
        .map(|name| {
            let array = record.column(
                record
                    .schema()
                    .index_of(name)
                    .expect("validated ASOF identity column"),
            );
            let values = if contiguous && rows.len() == record.num_rows() {
                array.clone()
            } else {
                take(array, &physical, None).map_err(|error| arrow_error(&error))?
            };
            Ok(SortColumn {
                values,
                options: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    lexsort_to_indices(&columns, None).map_err(|error| arrow_error(&error))
}

fn chunk_positions(
    rows: &[(LeftOrder, u32)],
    sorted: Option<&UInt32Array>,
    contiguous: bool,
) -> Option<Vec<u32>> {
    if sorted.is_none() && contiguous {
        None
    } else {
        Some(
            (0..rows.len())
                .map(|row| {
                    let ordinal = sorted.map_or(row, |indices| indices.value(row) as usize);
                    rows[ordinal].1
                })
                .collect(),
        )
    }
}

fn intern_chunk_key(
    interned: &mut HashMap<Encoding, u32, ahash::RandomState>,
    keys: &mut Vec<Option<Encoding>>,
    key_counts: &mut Vec<usize>,
    owners: &mut EncodingOwners,
    key: &Encoding,
    name: &str,
) -> Result<u32> {
    if let Some(id) = interned.get(key) {
        return Ok(*id);
    }
    super::validate_key_count(keys.len() as u64 + 1, name)?;
    let id = u32::try_from(keys.len()).expect("validated left key count");
    interned.insert(key.clone(), id);
    if keys.len() == keys.capacity() {
        keys.reserve_exact(keys.len().max(1));
        key_counts.reserve_exact(key_counts.len().max(1));
    }
    keys.push(Some(key.clone()));
    key_counts.push(0);
    owners.attach(key);
    Ok(id)
}

impl ChunkData {
    fn prepare(
        rows: &[(LeftOrder, u32)],
        batch: &PayloadBatch,
        side: &AsofJoinSide,
        name: &str,
    ) -> Result<Self> {
        let ordered = rows.windows(2).all(|pair| pair[0].0 < pair[1].0);
        let contiguous = rows
            .iter()
            .enumerate()
            .all(|(row, (_, position))| *position as usize == row);
        let sorted = if ordered {
            None
        } else {
            Some(chunk_sort_indices(rows, batch, side, contiguous)?)
        };
        let positions = chunk_positions(rows, sorted.as_ref(), contiguous);
        let mut keys = Vec::with_capacity(1);
        let mut key_counts = Vec::with_capacity(1);
        let mut key_ids = Vec::with_capacity(rows.len());
        let mut sequences = SequenceColumn::with_capacity(
            rows.len(),
            SequenceKind::for_side(&batch.record.schema(), side),
        );
        let mut time_values = Vec::with_capacity(rows.len());
        let mut interned = HashMap::<Encoding, u32, ahash::RandomState>::default();
        let mut index_bytes = 0;
        let mut owners = EncodingOwners::default();
        for ordinal in 0..rows.len() {
            let row = sorted
                .as_ref()
                .map_or(ordinal, |indices| indices.value(ordinal) as usize);
            let (_, key, sequence) = &rows[row].0;
            time_values.push(rows[row].0.0);
            let id = intern_chunk_key(
                &mut interned,
                &mut keys,
                &mut key_counts,
                &mut owners,
                key,
                name,
            )?;
            key_counts[id as usize] += 1;
            key_ids.push(id);
            sequences.push(sequence.clone());
            owners.attach(sequence);
            index_bytes = checked(
                name,
                index_bytes,
                41 + key.len() as u64 + sequence.len() as u64,
            )?;
        }
        Ok(Self {
            // External Arrow owners can hide a larger backing allocation than
            // Buffer::capacity reports. Keep only validated live time values
            // in an owned column so detached work has a provable input charge.
            times: time_values.into(),
            positions,
            start: 0,
            keys,
            key_counts,
            key_ids,
            sequences,
            index_bytes,
            owners,
        })
    }

    fn retained_input_bytes(&self, name: &str) -> Result<u64> {
        let time_buffer = self.times.inner();
        [
            size_of::<Self>() + 2 * size_of::<usize>(),
            time_buffer.capacity().max(time_buffer.len()),
            self.positions
                .as_ref()
                .map_or(0, |rows| rows.capacity() * size_of::<u32>()),
            self.keys.capacity() * size_of::<Option<Encoding>>(),
            self.key_counts.capacity() * size_of::<usize>(),
            self.key_ids.capacity() * size_of::<u32>(),
            self.sequences.allocation_bytes(),
        ]
        .into_iter()
        .try_fold(self.owners.allocation_bytes(), |bytes, capacity| {
            checked(name, bytes, capacity as u64)
        })
    }

    fn compaction_workspace_bytes(&self, remaining: usize, name: &str) -> Result<u64> {
        // A detached worker can outlive reset and the original state. Fund
        // both its retained buffers and the replacement column allocations.
        let scratch = self.keys.len() * size_of::<u32>()
            + remaining
                * (size_of::<Option<Encoding>>()
                    + size_of::<usize>()
                    + size_of::<u32>()
                    + self.sequences.element_bytes()
                    + size_of::<u32>()
                    + size_of::<i64>())
            + size_of::<Self>()
            + 256;
        let scratch = checked(name, scratch as u64, self.owners.metadata_bytes())?;
        checked(name, self.retained_input_bytes(name)?, scratch)
    }

    pub fn position(&self, ordinal: usize) -> u32 {
        self.positions.as_ref().map_or_else(
            || self.start + u32::try_from(ordinal).expect("compact left row"),
            |rows| rows[ordinal],
        )
    }

    fn view(&self, ordinal: usize) -> LeftView<'_> {
        (
            &self.times[ordinal],
            self.keys[self.key_ids[ordinal] as usize]
                .as_ref()
                .expect("live left key"),
            self.sequences.get(ordinal).expect("live left sequence"),
        )
    }

    fn consume(&mut self, range: std::ops::Range<usize>) {
        for ordinal in range {
            let key = self.key_ids[ordinal] as usize;
            let sequence = self.sequences.take(ordinal);
            self.index_bytes -= 41
                + self.keys[key].as_ref().expect("consumed live key").len() as u64
                + sequence.len() as u64;
            self.key_counts[key] -= 1;
            if self.key_counts[key] == 0 {
                self.owners
                    .detach(self.keys[key].as_ref().expect("consumed live key"));
                self.keys[key] = None;
            }
            self.owners.detach(&sequence);
        }
    }

    fn suffix(&self, head: usize) -> Self {
        let positions = self
            .positions
            .as_ref()
            .map(|positions| positions[head..].to_vec());
        let start = if positions.is_some() {
            0
        } else {
            self.start + u32::try_from(head).expect("compact left head")
        };
        let mut remap = vec![u32::MAX; self.keys.len()];
        let mut unique = 0;
        for old in &self.key_ids[head..] {
            if remap[*old as usize] == u32::MAX {
                remap[*old as usize] = unique;
                unique += 1;
            }
        }
        let mut keys = vec![None; unique as usize];
        let mut key_counts = vec![0; unique as usize];
        let mut key_ids = Vec::with_capacity(self.sequences.len() - head);
        let sequences = self.sequences.suffix(head);
        let mut index_bytes = 0;
        let mut owners = EncodingOwners::default();
        for ordinal in head..self.sequences.len() {
            let old = self.key_ids[ordinal] as usize;
            let id = remap[old];
            let key = self.keys[old].as_ref().expect("live suffix key");
            if key_counts[id as usize] == 0 {
                keys[id as usize] = Some(key.clone());
                owners.attach(key);
            }
            key_counts[id as usize] += 1;
            key_ids.push(id);
            let sequence = sequences.get(ordinal - head).expect("live suffix sequence");
            owners.attach(sequence.as_ref());
            index_bytes += 41 + key.len() as u64 + sequence.len() as u64;
        }
        Self {
            times: self.times[head..].to_vec().into(),
            positions,
            start,
            keys,
            key_counts,
            key_ids,
            sequences,
            index_bytes,
            owners,
        }
    }
}

pub(in super::super) struct PreparedLeftChunk {
    owner: Arc<PayloadBatch>,
    data: ChunkData,
}

impl PreparedLeftChunk {
    pub fn capacity_bytes(&self, name: &str) -> Result<u64> {
        checked(
            name,
            self.data.retained_input_bytes(name)? - self.data.owners.buffers_bytes(),
            128,
        )
    }

    pub fn v3_length(&self, kind: SequenceKind) -> u64 {
        73 + 16 * self.data.keys.len() as u64
            + self.data.sequences.len() as u64 * (16 + kind.reference_bytes())
    }
    pub fn from_index(owner: Arc<PayloadBatch>, data: ChunkData) -> Self {
        Self { owner, data }
    }
    pub fn prepare(
        rows: &[(LeftOrder, RowPayload)],
        side: &AsofJoinSide,
        name: &str,
    ) -> Result<Vec<Self>> {
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < rows.len() {
            let owner = &rows[start].1.batch;
            let end = start
                + rows[start..]
                    .iter()
                    .take_while(|(_, row)| row.batch.key == owner.key)
                    .count();
            u32::try_from(end - start).map_err(|_| {
                super::super::reason(
                    name,
                    crate::StreamingFailureReason::AsofCounterOverflow,
                    "ASOF left chunk exceeds compact row range",
                )
            })?;
            let identities = rows[start..end]
                .iter()
                .map(|(order, row)| {
                    let position = u32::try_from(row.row).map_err(|_| {
                        super::super::reason(
                            name,
                            crate::StreamingFailureReason::AsofCounterOverflow,
                            "ASOF payload row exceeds compact reference range",
                        )
                    })?;
                    Ok((order.clone(), position))
                })
                .collect::<Result<Vec<_>>>()?;
            let data = ChunkData::prepare(&identities, owner, side, name)?;
            chunks.push(Self {
                owner: owner.clone(),
                data,
            });
            start = end;
        }
        Ok(chunks)
    }
}

#[derive(Clone)]
struct LeftChunk {
    data: Arc<ChunkData>,
    reference: RowRef,
    head: usize,
}

impl LeftChunk {
    fn row(&self, ordinal: usize) -> (LeftView<'_>, RowRef) {
        (
            self.data.view(ordinal),
            self.reference.with_row(self.data.position(ordinal)),
        )
    }

    fn len(&self) -> usize {
        self.data.sequences.len() - self.head
    }

    fn needs_compaction(&self, amount: usize, remaining: usize) -> bool {
        amount > 0 && (self.head + amount >= remaining || Arc::strong_count(&self.data) > 1)
    }
}

#[derive(Clone, Default)]
pub(in super::super) struct LeftState {
    // Row fixtures use the small reference representation.
    pub(in super::super) legacy: LegacyLeftState,
    chunks: Vec<LeftChunk>,
    rows: usize,
    chunk_bytes: u64,
    first: Option<usize>,
    last: Option<usize>,
}

impl LeftState {
    pub fn projected_drain(
        &self,
        prefix: &super::LeftPrefix,
        prepared: &PreparedLeftDrain,
        pool: &PayloadPool,
        kind: SequenceKind,
        name: &str,
    ) -> Result<(u64, u64)> {
        let capacity = prepared
            .replacement
            .as_ref()
            .map_or(self.chunks.capacity(), Vec::capacity);
        let mut bytes = (capacity * size_of::<LeftChunk>()) as u64;
        let mut removed_index = 0;
        for (index, chunk) in self.chunks.iter().enumerate() {
            let batch = pool.key(chunk.reference);
            let amount = prefix.batches.get(&batch).copied().unwrap_or(0);
            if amount == 0 {
                bytes = checked(name, bytes, chunk_metadata(&chunk.data))?;
                continue;
            }
            let rows = chunk.len() - amount;
            let removed_rows = amount as u64 * (16 + kind.reference_bytes());
            removed_index += removed_rows;
            if rows == 0 {
                removed_index +=
                    73 + 16 * chunk.data.keys.iter().filter(|key| key.is_some()).count() as u64;
                continue;
            }
            let (owners, removed_keys) = projected_chunk_owner_removals(prefix, batch, &chunk.data);
            removed_index += removed_keys;
            let replacement = prepared
                .compacted
                .iter()
                .find(|(current, _)| *current == index)
                .map(|(_, data)| data);
            let retained =
                projected_chunk_bytes(&chunk.data, replacement.map(AsRef::as_ref), &owners, name)?;
            bytes = checked(name, bytes, retained)?;
        }
        Ok((bytes, removed_index))
    }
    pub fn projected_admission_bytes(
        &self,
        chunks: &[PreparedLeftChunk],
        name: &str,
    ) -> Result<u64> {
        let mut capacity = self.chunks.capacity();
        for len in self.chunks.len()..self.chunks.len() + chunks.len() {
            if len == capacity {
                capacity += len.max(1);
            }
        }
        let mut bytes = checked(
            name,
            self.capacity_bytes(name)?,
            ((capacity - self.chunks.capacity()) * size_of::<LeftChunk>()) as u64,
        )?;
        for chunk in chunks {
            bytes = checked(name, bytes, chunk.capacity_bytes(name)?)?;
        }
        Ok(bytes)
    }
    pub fn chunk_capacity(&self) -> usize {
        self.chunks.capacity()
    }

    pub fn reserve_chunks_exact(&mut self, capacity: usize) {
        self.chunks
            .reserve_exact(capacity.saturating_sub(self.chunks.len()));
    }

    pub fn checkpoint_chunks<'a>(
        &'a self,
        pool: &'a PayloadPool,
    ) -> impl ExactSizeIterator<Item = (BatchKey, &'a ChunkData, usize)> + Clone {
        self.chunks
            .iter()
            .map(|chunk| (pool.key(chunk.reference), chunk.data.as_ref(), chunk.head))
    }
    pub fn checkpoint_owned_chunks<'a>(
        &'a self,
        pool: &'a PayloadPool,
    ) -> impl ExactSizeIterator<Item = (BatchKey, Arc<ChunkData>, usize)> + 'a {
        self.chunks
            .iter()
            .map(|chunk| (pool.key(chunk.reference), chunk.data.clone(), chunk.head))
    }
    pub fn capacity_bytes(&self, name: &str) -> Result<u64> {
        let chunks = (self.chunks.capacity() * size_of::<LeftChunk>()) as u64;
        let chunks = checked(name, chunks, self.chunk_bytes)?;
        let legacy = self.legacy.iter().try_fold(0, |bytes, (order, row)| {
            checked(name, bytes, super::left_row_charge(&order.1, &order.2, row))
        })?;
        checked(name, chunks, legacy)
    }

    pub fn len(&self) -> usize {
        self.legacy.len() + self.rows
    }
    pub fn is_empty(&self) -> bool {
        self.legacy.is_empty() && self.chunks.is_empty()
    }

    pub fn iter_workspace_bytes(&self) -> u64 {
        (self.chunks.len() * size_of::<Cursor<'_>>()) as u64
    }

    pub fn iter(&self) -> impl Iterator<Item = (LeftView<'_>, RowRef)> {
        self.legacy
            .iter()
            .map(|(order, row)| (borrowed(order), *row))
            .chain(ChunkIter::new(&self.chunks))
    }

    pub fn unordered_iter(&self) -> impl Iterator<Item = (LeftView<'_>, RowRef)> {
        self.legacy
            .ordered
            .iter()
            .map(|(order, row)| (borrowed(order), *row))
            .chain(
                self.legacy
                    .general
                    .iter()
                    .map(|(order, row)| (borrowed(order), *row)),
            )
            .chain(self.chunks.iter().flat_map(|chunk| {
                (chunk.head..chunk.data.sequences.len()).map(|row| chunk.row(row))
            }))
    }

    #[cfg(test)]
    pub fn keys(&self) -> impl Iterator<Item = LeftView<'_>> {
        self.iter().map(|(key, _)| key)
    }
    pub fn unordered_keys(&self) -> impl Iterator<Item = LeftView<'_>> {
        self.unordered_iter().map(|(key, _)| key)
    }

    pub fn ready_prefix_len(&self, limit: usize, frontier: Option<i64>, ended: bool) -> usize {
        if ended {
            return self.len().min(limit);
        }
        let Some(frontier) = frontier else {
            return 0;
        };
        let legacy = self.legacy.ready_prefix_len(limit, Some(frontier), false);
        let mut count = legacy;
        for chunk in &self.chunks {
            if count >= limit {
                break;
            }
            let mut lower = chunk.head;
            let mut upper = chunk.data.sequences.len();
            while lower < upper {
                let middle = lower + (upper - lower) / 2;
                if *chunk.data.view(middle).0 < frontier {
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            count = count.saturating_add(lower - chunk.head).min(limit);
        }
        count
    }

    pub fn first_key_value(&self) -> Option<(LeftView<'_>, RowRef)> {
        self.legacy
            .first_key_value()
            .map(|(key, row)| (borrowed(key), *row))
            .or_else(|| {
                self.first
                    .map(|index| self.chunks[index].row(self.chunks[index].head))
            })
    }

    pub fn last_key_value(&self) -> Option<(LeftView<'_>, RowRef)> {
        self.legacy
            .last_key_value()
            .map(|(key, row)| (borrowed(key), *row))
            .or_else(|| {
                self.last.map(|index| {
                    self.chunks[index].row(self.chunks[index].data.sequences.len() - 1)
                })
            })
    }

    pub fn contains_key(&self, key: &LeftOrder) -> bool {
        self.legacy.contains_key(key)
            || self.chunks.iter().any(|chunk| {
                let mut lower = chunk.head;
                let mut upper = chunk.data.sequences.len();
                while lower < upper {
                    let middle = lower + (upper - lower) / 2;
                    match chunk.data.view(middle).cmp(&borrowed(key)) {
                        Ordering::Less => lower = middle + 1,
                        Ordering::Greater => upper = middle,
                        Ordering::Equal => return true,
                    }
                }
                false
            })
    }

    #[cfg(test)]
    pub fn insert(&mut self, key: LeftOrder, row: RowRef) {
        assert!(
            self.chunks.is_empty(),
            "legacy rows cannot coexist with Arrow chunks"
        );
        self.legacy.insert(key, row);
    }

    pub fn install(&mut self, chunks: Vec<PreparedLeftChunk>, pool: &mut PayloadPool) {
        assert!(
            self.legacy.is_empty(),
            "validated legacy state must migrate before admission"
        );
        for chunk in chunks {
            let reference = pool.attach_batch(&chunk.owner, chunk.data.sequences.len());
            self.push(LeftChunk {
                data: Arc::new(chunk.data),
                reference,
                head: 0,
            });
        }
    }

    fn push(&mut self, chunk: LeftChunk) {
        let index = self.chunks.len();
        if self.first.is_none_or(|previous| {
            chunk.data.view(chunk.head)
                < self.chunks[previous].data.view(self.chunks[previous].head)
        }) {
            self.first = Some(index);
        }
        if self.last.is_none_or(|previous| {
            chunk.data.view(chunk.data.sequences.len() - 1)
                > self.chunks[previous]
                    .data
                    .view(self.chunks[previous].data.sequences.len() - 1)
        }) {
            self.last = Some(index);
        }
        self.rows += chunk.len();
        self.chunk_bytes += chunk_metadata(&chunk.data);
        if index == self.chunks.capacity() {
            self.chunks.reserve_exact(index.max(1));
        }
        self.chunks.push(chunk);
    }

    fn refresh_extrema(&mut self) {
        self.first = self
            .chunks
            .iter()
            .enumerate()
            .min_by_key(|(_, chunk)| chunk.data.view(chunk.head))
            .map(|(index, _)| index);
        self.last = self
            .chunks
            .iter()
            .enumerate()
            .max_by_key(|(_, chunk)| chunk.data.view(chunk.data.sequences.len() - 1))
            .map(|(index, _)| index);
    }

    pub fn drain_workspace_bytes(
        &self,
        removed: &BTreeMap<BatchKey, usize>,
        pool: &PayloadPool,
        name: &str,
    ) -> Result<u64> {
        let mut bytes = 0;
        let mut remaining_chunks = 0;
        for chunk in &self.chunks {
            let amount = removed
                .get(&pool.key(chunk.reference))
                .copied()
                .unwrap_or(0);
            let remaining = chunk.len() - amount;
            if remaining == 0 {
                continue;
            }
            remaining_chunks += 1;
            if chunk.needs_compaction(amount, remaining) {
                bytes = checked(
                    name,
                    bytes,
                    chunk.data.compaction_workspace_bytes(remaining, name)?,
                )?;
            }
        }
        if self.chunks.capacity() > remaining_chunks * 2 {
            bytes = checked(
                name,
                bytes,
                (remaining_chunks * size_of::<LeftChunk>()) as u64,
            )?;
        }
        Ok(bytes)
    }

    pub fn drain_input(
        &self,
        removed: &BTreeMap<BatchKey, usize>,
        pool: &PayloadPool,
    ) -> LeftDrainInput {
        let mut selected = Vec::new();
        let mut remaining_chunks = 0;
        for (index, chunk) in self.chunks.iter().enumerate() {
            let amount = removed
                .get(&pool.key(chunk.reference))
                .copied()
                .unwrap_or(0);
            let remaining = chunk.len() - amount;
            if remaining == 0 {
                continue;
            }
            remaining_chunks += 1;
            if chunk.needs_compaction(amount, remaining) {
                selected.push((index, chunk.data.clone(), chunk.head + amount));
            }
        }
        let replacement = (self.chunks.capacity() > remaining_chunks * 2)
            .then(|| Vec::with_capacity(remaining_chunks));
        LeftDrainInput {
            selected,
            replacement,
        }
    }

    pub fn drain_prefix(
        &mut self,
        count: usize,
        removed: &BTreeMap<BatchKey, usize>,
        pool: &PayloadPool,
        mut prepared: PreparedLeftDrain,
    ) {
        if !self.legacy.is_empty() {
            self.legacy.drain_prefix(count);
            return;
        }
        let mut selected = prepared.compacted.into_iter().peekable();
        for (index, chunk) in self.chunks.iter_mut().enumerate() {
            let amount = removed
                .get(&pool.key(chunk.reference))
                .copied()
                .unwrap_or(0);
            if selected
                .peek()
                .is_some_and(|(current, _)| *current == index)
            {
                chunk.data = selected.next().expect("prepared left compaction").1;
                chunk.head = 0;
            } else {
                if amount > 0 && amount < chunk.len() {
                    Arc::get_mut(&mut chunk.data)
                        .expect("exclusive unprepared left metadata")
                        .consume(chunk.head..chunk.head + amount);
                }
                chunk.head += amount;
            }
        }
        if let Some(mut replacement) = prepared.replacement.take() {
            replacement.extend(self.chunks.drain(..).filter(|chunk| chunk.len() > 0));
            self.chunks = replacement;
        } else {
            self.chunks.retain(|chunk| chunk.len() > 0);
        }
        self.rows -= count;
        self.chunk_bytes = self
            .chunks
            .iter()
            .map(|chunk| chunk_metadata(&chunk.data))
            .sum();
        self.refresh_extrema();
    }

    #[cfg(test)]
    pub fn append_admission(&mut self, rows: Vec<(LeftOrder, RowRef)>) {
        self.legacy.append_admission(rows);
    }
    #[cfg(test)]
    pub fn is_ordered(&self) -> bool {
        self.legacy.is_ordered()
    }
}

fn projected_chunk_owner_removals(
    prefix: &super::LeftPrefix,
    batch: BatchKey,
    data: &ChunkData,
) -> (super::OwnerRemovals, u64) {
    let mut owners = prefix
        .sequence_owners
        .get(&batch)
        .cloned()
        .unwrap_or_default();
    let mut removed_index = 0;
    for (id, key) in data
        .keys
        .iter()
        .enumerate()
        .filter_map(|(id, key)| key.as_ref().map(|key| (id, key)))
    {
        let removed = prefix.keys.get(&(batch, key.clone())).copied().unwrap_or(0);
        if removed == data.key_counts[id] {
            removed_index += 16;
            EncodingOwners::record_remove(&mut owners, key, 1);
        }
    }
    (owners, removed_index)
}

fn projected_chunk_bytes(
    data: &ChunkData,
    replacement: Option<&ChunkData>,
    owners: &super::OwnerRemovals,
    name: &str,
) -> Result<u64> {
    Ok(if let Some(replacement) = replacement {
        replacement.retained_input_bytes(name)? - replacement.owners.buffers_bytes() + 128
    } else {
        let after = data.owners.projected_metadata_bytes(owners);
        data.retained_input_bytes(name)? - data.owners.allocation_bytes() + after + 128
    })
}

fn chunk_metadata(data: &ChunkData) -> u64 {
    data.retained_input_bytes("asof")
        .expect("preflighted ASOF chunk metadata")
        - data.owners.buffers_bytes()
        + 128
}

pub(in super::super) struct LeftDrainInput {
    selected: Vec<(usize, Arc<ChunkData>, usize)>,
    replacement: Option<Vec<LeftChunk>>,
}

#[derive(Default)]
pub(in super::super) struct PreparedLeftDrain {
    compacted: Vec<(usize, Arc<ChunkData>)>,
    replacement: Option<Vec<LeftChunk>>,
}

impl LeftDrainInput {
    pub fn prepare(self) -> PreparedLeftDrain {
        let compacted = self
            .selected
            .into_iter()
            .map(|(index, data, head)| (index, Arc::new(data.suffix(head))))
            .collect();
        PreparedLeftDrain {
            compacted,
            replacement: self.replacement,
        }
    }
}

#[derive(Clone, Copy)]
struct Cursor<'a> {
    chunk: &'a LeftChunk,
    ordinal: usize,
}
impl PartialEq for Cursor<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Cursor<'_> {}
impl PartialOrd for Cursor<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cursor<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.chunk
            .data
            .view(self.ordinal)
            .cmp(&other.chunk.data.view(other.ordinal))
    }
}

struct ChunkIter<'a> {
    heap: BinaryHeap<Reverse<Cursor<'a>>>,
}
impl<'a> ChunkIter<'a> {
    fn new(chunks: &'a [LeftChunk]) -> Self {
        let cursors = chunks
            .iter()
            .map(|chunk| {
                Reverse(Cursor {
                    chunk,
                    ordinal: chunk.head,
                })
            })
            .collect::<Vec<_>>();
        Self {
            heap: BinaryHeap::from(cursors),
        }
    }
}
impl<'a> Iterator for ChunkIter<'a> {
    type Item = (LeftView<'a>, RowRef);
    fn next(&mut self) -> Option<Self::Item> {
        let Reverse(mut cursor) = self.heap.pop()?;
        let result = cursor.chunk.row(cursor.ordinal);
        cursor.ordinal += 1;
        if cursor.ordinal < cursor.chunk.data.sequences.len() {
            self.heap.push(Reverse(cursor));
        }
        #[cfg(test)]
        super::LEFT_VISITS.with(|visits| visits.set(visits.get() + 1));
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::asof::admission::times;
    use datafusion::arrow::{
        array::*,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use std::sync::OnceLock;

    fn identity_arrays() -> Vec<ArrayRef> {
        macro_rules! signed {
            ($array:ty) => {
                Arc::new(<$array>::from(vec![0, -1, 1, 0, 1, -1])) as ArrayRef
            };
        }
        macro_rules! unsigned {
            ($array:ty) => {
                Arc::new(<$array>::from(vec![1, 0, 2, 1, 2, 0])) as ArrayRef
            };
        }
        vec![
            Arc::new(BooleanArray::from(vec![
                true, false, true, false, true, false,
            ])),
            signed!(Int8Array),
            signed!(Int16Array),
            signed!(Int32Array),
            signed!(Int64Array),
            unsigned!(UInt8Array),
            unsigned!(UInt16Array),
            unsigned!(UInt32Array),
            unsigned!(UInt64Array),
            Arc::new(StringArray::from(vec!["字符", "", "🦀", "字符", "🦀", ""])),
            Arc::new(LargeStringArray::from(vec![
                "字符", "", "🦀", "字符", "🦀", "",
            ])),
            signed!(Date32Array),
            signed!(Date64Array),
            signed!(TimestampSecondArray),
            signed!(TimestampMillisecondArray),
            signed!(TimestampMicrosecondArray),
            signed!(TimestampNanosecondArray),
        ]
    }

    fn fixture(
        key: ArrayRef,
        sequence: ArrayRef,
        id: u64,
    ) -> (
        Arc<PayloadBatch>,
        AsofJoinSide,
        Vec<(LeftOrder, RowPayload)>,
    ) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", key.data_type().clone(), false),
            Field::new(
                "time",
                DataType::Timestamp(
                    datafusion::arrow::datatypes::TimeUnit::Microsecond,
                    Some("UTC".into()),
                ),
                false,
            ),
            Field::new("seq", sequence.data_type().clone(), false),
        ]));
        let times =
            Arc::new(TimestampMicrosecondArray::from(vec![10; key.len()]).with_timezone("UTC"));
        let record = RecordBatch::try_new(schema, vec![key, times, sequence]).unwrap();
        let side = AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            "left".into(),
        )
        .unwrap();
        let keys = super::super::encode_columns(&record, side.keys()).unwrap();
        let sequences = super::super::encode_columns(&record, side.sequence_by()).unwrap();
        let owner = Arc::new(PayloadBatch {
            key: (0, id),
            record: Arc::new(record),
            encoded: OnceLock::new(),
            encoded_charge_bytes: 0,
            body_bytes: 0,
        });
        let rows = (0..owner.record.num_rows())
            .map(|row| {
                (
                    (10, keys.row(row), sequences.row(row)),
                    RowPayload {
                        batch: owner.clone(),
                        row,
                    },
                )
            })
            .collect();
        (owner, side, rows)
    }

    #[test]
    fn chunk_lexsort_matches_canonical_rows_for_supported_identity_types() {
        let columns = identity_arrays();
        for key in &columns {
            for sequence in &columns[1..11] {
                let (owner, side, rows) = fixture(key.clone(), sequence.clone(), 0);
                let compact = rows
                    .iter()
                    .map(|(order, row)| (order.clone(), u32::try_from(row.row).unwrap()))
                    .collect::<Vec<_>>();
                let data = ChunkData::prepare(&compact, &owner, &side, "asof").unwrap();
                let mut expected = compact
                    .iter()
                    .map(|(identity, _)| identity.clone())
                    .collect::<Vec<_>>();
                expected.sort_unstable();
                let actual = (0..rows.len())
                    .map(|row| {
                        let (time, key, sequence) = data.view(row);
                        (*time, key.clone(), sequence.into_owned())
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual,
                    expected,
                    "key {}, sequence {}",
                    key.data_type(),
                    sequence.data_type()
                );
            }
        }
    }

    #[test]
    fn chunk_lexsort_orders_varied_times_composite_keys_and_sequences() {
        let time_type = DataType::Timestamp(
            datafusion::arrow::datatypes::TimeUnit::Microsecond,
            Some("UTC".into()),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("time", time_type, false),
            Field::new("key1", DataType::Int64, false),
            Field::new("key2", DataType::Utf8, false),
            Field::new("seq1", DataType::Int64, false),
            Field::new("seq2", DataType::Utf8, false),
        ]));
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(
                    TimestampMicrosecondArray::from(vec![2, 1, 1, 1, 2, 1]).with_timezone("UTC"),
                ),
                Arc::new(Int64Array::from(vec![1, 0, 0, 0, -1, 0])),
                Arc::new(StringArray::from(vec!["z", "b", "a", "a", "z", "a"])),
                Arc::new(Int64Array::from(vec![9, 3, 2, 2, 9, 1])),
                Arc::new(StringArray::from(vec!["z", "x", "z", "a", "z", "z"])),
            ],
        )
        .unwrap();
        let side = AsofJoinSide::new(
            vec!["key1".into(), "key2".into()],
            "time".into(),
            vec!["seq1".into(), "seq2".into()],
            "left".into(),
        )
        .unwrap();
        let keys = super::super::encode_columns(&record, side.keys()).unwrap();
        let sequences = super::super::encode_columns(&record, side.sequence_by()).unwrap();
        let owner = PayloadBatch {
            key: (0, 0),
            record: Arc::new(record),
            encoded: OnceLock::new(),
            encoded_charge_bytes: 0,
            body_bytes: 0,
        };
        let rows = (0..6)
            .map(|row| {
                (
                    (
                        times(&owner.record, &side).value(row),
                        keys.row(row),
                        sequences.row(row),
                    ),
                    u32::try_from(row).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let data = ChunkData::prepare(&rows, &owner, &side, "asof").unwrap();
        assert_eq!(
            (0..6).map(|row| data.position(row)).collect::<Vec<_>>(),
            vec![5, 3, 2, 1, 4, 0]
        );
    }

    #[test]
    fn sparse_chunk_metadata_fits_legacy_charge_after_each_prefix() {
        for count in [1_usize, 2, 3, 4, 16, 64, 257] {
            let key = Arc::new(StringArray::from_iter_values(
                (0..count).map(|row| format!("key-{row:08}-{}", "x".repeat(32))),
            )) as ArrayRef;
            let sequence = Arc::new(Int64Array::from_iter_values(
                0..i64::try_from(count).unwrap(),
            )) as ArrayRef;
            let (_, side, rows) = fixture(key, sequence, 0);
            for removed in [0, 1, count / 2, count.saturating_sub(1), count] {
                let mut state = super::super::State::default();
                let measured = allocation_counter::measure(|| {
                    let admitted = rows
                        .iter()
                        .map(|(order, row)| {
                            (
                                (
                                    order.0,
                                    Encoding::from_slice(&order.1),
                                    Encoding::from_slice(&order.2),
                                ),
                                row.clone(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let chunks = PreparedLeftChunk::prepare(&admitted, &side, "asof").unwrap();
                    state.left.install(chunks, &mut state.batches);
                    drop(admitted);
                    state.commit_left_prefix(removed);
                });
                let charge = state.inventory(None, "asof").unwrap().bytes;
                assert!(
                    u64::try_from(measured.bytes_current).unwrap() <= charge,
                    "left storage exceeds legacy charge after {removed}/{count}: {measured:?}, charge={charge}"
                );
                if removed == count {
                    assert!(state.left.is_empty());
                    assert!(state.batches.is_empty());
                    assert_eq!(
                        measured.bytes_current, 0,
                        "empty left state retains metadata"
                    );
                }
            }
        }
    }

    #[test]
    fn detached_compaction_does_not_retain_unknown_external_time_owner() {
        use datafusion::arrow::buffer::Buffer;
        use tokio_util::bytes::Bytes;

        struct ExternalTimes {
            buffer: Buffer,
            _live: Arc<()>,
        }
        impl AsRef<[u8]> for ExternalTimes {
            fn as_ref(&self) -> &[u8] {
                &self.buffer
            }
        }
        let live = Arc::new(());
        let weak = Arc::downgrade(&live);
        let bytes = Bytes::from_owner(ExternalTimes {
            buffer: Buffer::from_vec(vec![10_i64; 1_000_000]),
            _live: live,
        })
        .slice(..64 * size_of::<i64>());
        let values = ScalarBuffer::new(Buffer::from(bytes), 0, 64);
        assert_eq!(values.inner().capacity(), 64 * size_of::<i64>());
        let key = Arc::new(StringArray::from(vec!["key"; 64])) as ArrayRef;
        let sequence = Arc::new(Int64Array::from_iter_values(0..64)) as ArrayRef;
        let (original, side, mut rows) = fixture(key, sequence, 0);
        let mut columns = original.record.columns().to_vec();
        columns[1] = Arc::new(TimestampMicrosecondArray::new(values, None).with_timezone("UTC"));
        let owner = Arc::new(PayloadBatch {
            key: original.key,
            record: Arc::new(RecordBatch::try_new(original.record.schema(), columns).unwrap()),
            encoded: OnceLock::new(),
            encoded_charge_bytes: 0,
            body_bytes: 0,
        });
        for (_, row) in &mut rows {
            row.batch = owner.clone();
        }
        drop((original, owner));
        let mut state = super::super::State::default();
        let chunks = PreparedLeftChunk::prepare(&rows, &side, "asof").unwrap();
        state.left.install(chunks, &mut state.batches);
        drop(rows);
        let mut prefix = super::super::LeftPrefix::default();
        for (order, row) in state.left.iter().take(32) {
            prefix.visit(&order, row, &state.batches, "asof").unwrap();
        }
        let input = state.left.drain_input(&prefix.batches, &state.batches);
        drop(state);
        assert!(
            weak.upgrade().is_none(),
            "detached worker retains an external owner with unknown backing capacity"
        );
        let prepared = input.prepare();
        assert_eq!(prepared.compacted[0].1.times[0], 10);
    }

    #[test]
    fn detached_compaction_input_fits_workspace_after_state_release() {
        let count = 64;
        let key = Arc::new(StringArray::from_iter_values(
            (0..count).map(|row| format!("key-{row:08}-{}", "k".repeat(4_096))),
        )) as ArrayRef;
        let sequence = Arc::new(StringArray::from_iter_values(
            (0..count).map(|row| format!("seq-{row:08}-{}", "s".repeat(4_096))),
        )) as ArrayRef;
        let (_, side, rows) = fixture(key, sequence, 0);
        let mut state = super::super::State::default();
        let chunks = PreparedLeftChunk::prepare(&rows, &side, "asof").unwrap();
        state.left.install(chunks, &mut state.batches);
        drop(rows);
        let mut prefix = super::super::LeftPrefix::default();
        for (order, row) in state.left.iter().take(count / 2) {
            prefix.visit(&order, row, &state.batches, "asof").unwrap();
        }
        let charge = state
            .left
            .drain_workspace_bytes(&prefix.batches, &state.batches, "asof")
            .unwrap();
        let input = state.left.drain_input(&prefix.batches, &state.batches);
        drop(state);
        let released = allocation_counter::measure(|| drop(input));
        let retained = u64::try_from(-released.bytes_current).unwrap();
        assert!(
            retained <= charge,
            "detached worker input exceeds workspace after reset: retained={retained}, charge={charge}"
        );
    }

    #[test]
    fn tiny_prefix_prepares_half_consumed_chunk_compaction_before_commit() {
        let count = 10_000;
        let key = Arc::new(StringArray::from_iter_values(
            (0..count).map(|row| format!("key-{row:08}-{}", "x".repeat(32))),
        )) as ArrayRef;
        let sequence = Arc::new(Int64Array::from_iter_values(0..count)) as ArrayRef;
        let (_, side, rows) = fixture(key, sequence, 0);
        let mut state = super::super::State::default();
        let chunks = PreparedLeftChunk::prepare(&rows, &side, "asof").unwrap();
        state.left.install(chunks, &mut state.batches);
        state.commit_left_prefix(4_999);
        assert_eq!(state.left.chunks[0].head, 4_999);
        let mut prefix = super::super::LeftPrefix::default();
        for (order, row) in state.left.iter().take(1) {
            prefix.visit(&order, row, &state.batches, "asof").unwrap();
        }
        let charge = state
            .left
            .drain_workspace_bytes(&prefix.batches, &state.batches, "asof")
            .unwrap();
        assert!(
            charge >= 5_000 * 48,
            "compaction workspace must scale with the remaining chunk"
        );
        let mut prepared = None;
        let allocation = allocation_counter::measure(|| {
            prepared = Some(
                state
                    .left
                    .drain_input(&prefix.batches, &state.batches)
                    .prepare(),
            );
        });
        assert!(
            allocation.bytes_max <= charge,
            "compaction exceeded reserved scratch: {allocation:?}, charge={charge}"
        );
        let committed = allocation_counter::measure(|| {
            state.commit_matched_left_prefix(&prefix, prepared.unwrap());
        });
        assert_eq!(
            committed.count_total, 0,
            "post-delivery compaction must not allocate"
        );
        assert_eq!(state.left.chunks[0].head, 0);
        assert_eq!(state.left.len(), 5_000);
    }

    #[test]
    fn integer_chunks_allocate_only_the_declared_sequence_width() {
        use datafusion::common::ScalarValue;

        let count = 4_096;
        for (data_type, width) in [
            (DataType::Int8, 1),
            (DataType::Int16, 2),
            (DataType::Int32, 4),
            (DataType::Int64, 8),
            (DataType::UInt8, 1),
            (DataType::UInt16, 2),
            (DataType::UInt32, 4),
            (DataType::UInt64, 8),
        ] {
            let key = Arc::new(StringArray::from(vec!["A"; count])) as ArrayRef;
            let sequence = ScalarValue::new_default(&data_type)
                .unwrap()
                .to_array_of_size(count)
                .unwrap();
            let (original, side, mut rows) = fixture(key, sequence, 0);
            let mut columns = original.record.columns().to_vec();
            columns[1] = Arc::new(
                TimestampMicrosecondArray::from_iter_values(0..i64::try_from(count).unwrap())
                    .with_timezone("UTC"),
            );
            let owner = Arc::new(PayloadBatch {
                record: Arc::new(RecordBatch::try_new(original.record.schema(), columns).unwrap()),
                key: original.key,
                encoded: OnceLock::new(),
                encoded_charge_bytes: 0,
                body_bytes: 0,
            });
            for (ordinal, (identity, row)) in rows.iter_mut().enumerate() {
                identity.0 = i64::try_from(ordinal).unwrap();
                row.batch = owner.clone();
            }
            let mut retained = None;
            let allocation = allocation_counter::measure(|| {
                retained = Some(PreparedLeftChunk::prepare(&rows, &side, "asof").unwrap());
            });
            assert_eq!(retained.as_ref().unwrap()[0].data.positions, None);
            let bound =
                (12 + width) * count as u64 + (4 * size_of::<PreparedLeftChunk>() + 512) as u64;
            assert_eq!(
                retained.as_ref().unwrap()[0]
                    .data
                    .sequences
                    .allocation_bytes(),
                usize::try_from(width).unwrap() * count
            );
            assert!(
                u64::try_from(allocation.bytes_current).unwrap() <= bound,
                "{data_type:?}: sequence storage exceeds its declared width: {allocation:?}, bound={bound}"
            );
        }
    }
}
