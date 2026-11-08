//! Transactional raw aggregate reuse; funding and fallback: see the C1 API note.

use std::{mem::size_of, sync::Arc};

use datafusion::{
    arrow::{array::ArrayRef, datatypes::SchemaRef},
    execution::memory_pool::{MemoryLimit, MemoryReservation},
};

use super::CHUNK_ROWS;
use crate::DataFusionRuntime;

const MAX_BYTES: usize = 8 * 1024 * 1024;

pub(super) struct SnapshotAggregates {
    enabled: bool,
    schema: SchemaRef,
    chunks: Vec<Option<Arc<Chunk>>>,
    metadata: MemoryReservation,
    array_bytes: usize,
    output_credit: usize,
}

struct Chunk {
    rows: usize,
    first_key: Arc<[u8]>,
    columns: Arc<[ArrayRef]>,
    reservation: MemoryReservation,
}

impl SnapshotAggregates {
    pub(super) fn with_scratch(
        runtime: &DataFusionRuntime,
        schema: &SchemaRef,
        name: &str,
    ) -> (MemoryReservation, Self) {
        let scratch = runtime.incremental_reservation(name);
        let cache = Self::new(
            schema.clone(),
            scratch.new_empty(),
            matches!(
                runtime.incremental_memory_pool().memory_limit(),
                MemoryLimit::Infinite
            ),
        );
        (scratch, cache)
    }

    pub(super) fn new(schema: SchemaRef, metadata: MemoryReservation, enabled: bool) -> Self {
        Self {
            enabled,
            schema,
            chunks: Vec::new(),
            metadata,
            array_bytes: 0,
            output_credit: 0,
        }
    }

    pub(super) fn prepare(
        &self,
        layout: (&SchemaRef, usize, usize),
        touched: impl Iterator<Item = usize>,
        first_key: impl Fn(usize) -> Arc<[u8]>,
    ) -> Option<Self> {
        if !self.enabled || layout.1 == 0 {
            return None;
        }
        let (mut prepared, slots) = self.allocate(layout)?;
        let mut dirty = Vec::new();
        dirty.try_reserve_exact(slots).ok()?;
        dirty.resize(slots, false);
        for slot in touched {
            if let Some(chunk) = dirty.get_mut(slot / CHUNK_ROWS) {
                *chunk = true;
            }
        }
        prepared.reuse(self, &dirty, layout.1, first_key);
        Some(prepared)
    }

    fn allocate(&self, layout: (&SchemaRef, usize, usize)) -> Option<(Self, usize)> {
        let per_chunk = 2 * size_of::<Option<Arc<Chunk>>>()
            + size_of::<bool>()
            + size_of::<Chunk>()
            + 64
            + layout.2.checked_mul(4 * size_of::<ArrayRef>())?;
        let slots = layout
            .1
            .div_ceil(CHUNK_ROWS)
            .min((MAX_BYTES - 256) / per_chunk);
        let metadata = self.metadata.new_empty();
        metadata.try_grow(256 + slots * per_chunk).ok()?;
        let mut prepared = Self::new(layout.0.clone(), metadata, true);
        prepared.chunks.try_reserve_exact(slots).ok()?;
        prepared.chunks.resize_with(slots, || None);
        Some((prepared, slots))
    }

    fn reuse(
        &mut self,
        committed: &Self,
        dirty: &[bool],
        count: usize,
        first_key: impl Fn(usize) -> Arc<[u8]>,
    ) {
        if !Arc::ptr_eq(&self.schema, &committed.schema) {
            return;
        }
        for (ordinal, &dirty) in dirty.iter().enumerate() {
            let Some(chunk) = committed.chunks.get(ordinal).and_then(Option::as_ref) else {
                continue;
            };
            let start = ordinal * CHUNK_ROWS;
            let rows = (count - start).min(CHUNK_ROWS);
            if dirty || chunk.rows != rows || !Arc::ptr_eq(&chunk.first_key, &first_key(start)) {
                continue;
            }
            if self.fits(chunk.reservation.size()) {
                self.array_bytes += chunk.reservation.size();
                self.chunks[ordinal] = Some(Arc::clone(chunk));
            }
        }
    }

    pub(super) fn shared_bytes(&self) -> usize {
        self.array_bytes
    }

    pub(super) fn set_output_credit(&mut self, bytes: usize) {
        self.output_credit = bytes;
    }

    pub(super) fn get(&self, ordinal: usize) -> Option<&[ArrayRef]> {
        Some(&self.chunks.get(ordinal)?.as_ref()?.columns)
    }

    pub(super) fn stage(
        &mut self,
        location: (usize, usize, &Arc<[u8]>),
        columns: &[ArrayRef],
        reservation: &MemoryReservation,
    ) {
        if location.0 >= self.chunks.len() {
            return;
        }
        let bytes = columns.iter().try_fold(0usize, |bytes, column| {
            bytes.checked_add(column.get_array_memory_size())
        });
        let Some(bytes) = bytes.filter(|&bytes| self.fits(bytes) && bytes <= self.output_credit)
        else {
            return;
        };
        let reservation = reservation.split(bytes);
        let chunk = Chunk {
            rows: location.1,
            first_key: Arc::clone(location.2),
            columns: Arc::from(columns),
            reservation,
        };
        self.output_credit -= bytes;
        self.array_bytes += bytes;
        self.chunks[location.0] = Some(Arc::new(chunk));
    }

    fn fits(&self, bytes: usize) -> bool {
        bytes <= MAX_BYTES - self.metadata.size() - self.array_bytes
    }

    pub(super) fn clear(&mut self) {
        self.chunks = Vec::new();
        self.metadata.free();
        self.array_bytes = 0;
        self.output_credit = 0;
    }
}
