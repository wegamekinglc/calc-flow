//! Stable chunk coordinates for online state; wire rows are temporary views.

use std::{collections::BTreeMap, sync::Arc};

use super::{StoredRow, admitted::PayloadRef, columnar::RowPayload};
use crate::{EventTime, Result, StreamOperatorContext};
use datafusion::execution::memory_pool::MemoryReservation;

const CHUNK_ROWS: usize = 1_024;

pub(super) trait Rows: Send + Sync {
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
    fn key(&self, index: usize) -> &Arc<super::columnar::FramedKey>;
    fn indexed(&self) -> Box<dyn Iterator<Item = (usize, StoredRow)> + Send + '_>;
    fn references(&self) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        Box::new(self.indexed().map(|(index, _)| index))
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl dyn Rows + '_ {
    pub(super) fn iter(&self) -> Box<dyn Iterator<Item = StoredRow> + Send + '_> {
        Box::new(self.indexed().map(|(_, row)| row))
    }
}

struct Source {
    owner: RowPayload,
    count: usize,
}

#[derive(Default)]
struct Chunk {
    sources: Vec<Option<Source>>,
    source_ids: Vec<usize>,
    offsets: Vec<usize>,
    times: Vec<EventTime>,
    ids: Vec<u64>,
    charges: Vec<u64>,
    keys: Vec<Option<Arc<super::columnar::FramedKey>>>,
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
        Self {
            sources: Vec::with_capacity(sources),
            source_ids: Vec::with_capacity(count),
            offsets: Vec::with_capacity(count),
            times: Vec::with_capacity(count),
            ids: Vec::with_capacity(count),
            charges: Vec::with_capacity(count),
            keys: Vec::with_capacity(count),
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
        self.keys.push(Some(row.encoded_key));
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
            encoded_key: Arc::clone(self.keys[offset].as_ref().expect("live chunk row")),
        }
    }

    fn release_source(&mut self, offset: usize) {
        let source = &mut self.sources[self.source_ids[offset]];
        source.as_mut().unwrap().count -= 1;
        if source.as_ref().unwrap().count == 0 {
            *source = None;
        }
    }

    fn remove(&mut self, offset: usize) -> StoredRow {
        let row = self.row(offset);
        self.keys[offset] = None;
        self.release_source(offset);
        self.live -= 1;
        row
    }
}

pub(super) struct Append {
    chunks: BTreeMap<usize, Chunk>,
    pub(super) indices: Vec<usize>,
    count: usize,
    control: Option<MemoryReservation>,
}

impl Append {
    pub(super) fn is_unfunded(&self) -> bool {
        self.count != 0
            && (self.control.is_none() || self.chunks.values().any(|chunk| chunk.credit.is_none()))
    }
}

#[derive(Default)]
pub(super) struct StateRows {
    chunks: BTreeMap<usize, Chunk>,
    next: usize,
    len: usize,
}

impl StateRows {
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
    #[cfg(all(test, feature = "benchmark-internals"))]
    pub(super) fn replace_at(&mut self, ordinal: usize, row: StoredRow) {
        let reference = self.reference(ordinal);
        self.replace_payload(reference, row.record);
        let chunk = self.chunks.get_mut(&(reference / CHUNK_ROWS)).unwrap();
        let offset = reference % CHUNK_ROWS;
        chunk.times[offset] = row.event_time;
        chunk.ids[offset] = row.row_id;
        chunk.charges[offset] = row.charge;
        chunk.keys[offset] = Some(row.encoded_key);
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
        let mut chunks = BTreeMap::new();
        let mut quantum = super::columnar::Quantum::default();
        for (rows, indices) in rows.chunks(CHUNK_ROWS).zip(indices.chunks(CHUNK_ROWS)) {
            let chunk = prepare_chunk(rows, master, context, &mut quantum, legacy).await?;
            chunks.insert(indices[0] / CHUNK_ROWS, chunk);
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
        self.next += append.chunks.len();
        self.len += append.count;
        for (id, chunk) in append.chunks {
            self.chunks.insert(id, chunk);
        }
    }

    pub(super) fn append_indices(&self, count: usize) -> Vec<usize> {
        (0..count)
            .map(|position| {
                (self.next + position / CHUNK_ROWS) * CHUNK_ROWS + position % CHUNK_ROWS
            })
            .collect()
    }

    pub(super) fn append(&mut self, rows: Vec<StoredRow>) -> Vec<usize> {
        let mut indices = Vec::with_capacity(rows.len());
        let mut rows = rows.into_iter();
        while !rows.as_slice().is_empty() {
            let id = self.next;
            self.next += 1;
            let capacity = rows.len().min(CHUNK_ROWS);
            let source_count = 1 + rows.as_slice()[..capacity]
                .windows(2)
                .filter(|pair| {
                    pair[0].record.columns().as_ptr() != pair[1].record.columns().as_ptr()
                })
                .count();
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

impl Rows for StateRows {
    fn references(&self) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        Box::new(self.chunks.iter().flat_map(|(&id, chunk)| {
            chunk
                .keys
                .iter()
                .enumerate()
                .filter_map(move |(offset, key)| key.as_ref().map(|_| id * CHUNK_ROWS + offset))
        }))
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
        let chunk = &self.chunks[&(index / CHUNK_ROWS)];
        let offset = index % CHUNK_ROWS;
        PayloadRef::Payload(chunk.owner(offset), chunk.offsets[offset])
    }

    fn key(&self, index: usize) -> &Arc<super::columnar::FramedKey> {
        self.chunks[&(index / CHUNK_ROWS)].keys[index % CHUNK_ROWS]
            .as_ref()
            .unwrap()
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
    fn references(&self) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        self.as_ref().references()
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
    fn key(&self, index: usize) -> &Arc<super::columnar::FramedKey> {
        self.as_ref().key(index)
    }
    fn indexed(&self) -> Box<dyn Iterator<Item = (usize, StoredRow)> + Send + '_> {
        self.as_ref().indexed()
    }
}

impl<const N: usize> Rows for [StoredRow; N] {
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
    let sources = 1 + rows
        .windows(2)
        .filter(|pair| pair[0].record.columns().as_ptr() != pair[1].record.columns().as_ptr())
        .count();
    let credit = grant(master, chunk_bytes(rows)?, context.operator_id(), legacy)?;
    let mut chunk = Chunk::reserved(rows.len(), sources, credit);
    for row in rows {
        quantum.step(context, 1, 128).await?;
        chunk.push(row.clone());
    }
    Ok(chunk)
}

fn chunk_bytes(rows: &[StoredRow]) -> Result<usize> {
    let width = 2 * size_of::<usize>()
        + size_of::<EventTime>()
        + 2 * size_of::<u64>()
        + size_of::<Option<Arc<super::columnar::FramedKey>>>();
    // Source slots may grow during funded sparse payload replacement.
    let sources = rows
        .len()
        .max(4)
        .checked_next_power_of_two()
        .and_then(|count| count.checked_mul(2 * size_of::<Option<Source>>()));
    let fixed = rows
        .len()
        .checked_mul(width)
        .and_then(|bytes| bytes.checked_add(sources?));
    fixed
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
        .and_then(|bytes| bytes.checked_add(8192))
        .ok_or_else(|| super::native_lookup::scratch_error("Join state chunks"))
}

#[cfg(test)]
mod tests;
