use crate::runtime::streaming::gather_work::RetirementGuard;
use datafusion::arrow::{array::ArrayRef, datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion::execution::memory_pool::MemoryReservation;
use std::{ops::Deref, sync::Arc};

mod metadata;
mod owned_copy;
pub(super) mod restored;
mod sparse;
pub(super) use metadata::schema_inventory;
pub(super) use owned_copy::{CopySelection, Quantum, SelectedRow};
#[cfg(test)]
pub(super) use owned_copy::{observe_string_allocations, take_string_allocations};
pub(super) use sparse::SparseQueue;

#[derive(Clone)]
pub(super) enum RowPayload {
    Legacy(RecordBatch),
    /// One row of an immutable parent record, kept without slicing the
    /// parent into per-row wrappers. `columns` and `offset` address the row.
    Rowed {
        parent: Arc<RecordBatch>,
        row: usize,
    },
    Shared {
        chunk: Arc<PayloadChunk>,
        row: usize,
    },
}

pub(super) struct PayloadChunk {
    columns: Vec<ArrayRef>,
    schema: SchemaRef,
    inventory: Vec<sparse::ChunkRow>,
    backing_bytes: usize,
    terminal_offsets: usize,
    live_count: std::sync::atomic::AtomicUsize,
    live_bytes: std::sync::atomic::AtomicUsize,
    queued: std::sync::atomic::AtomicBool,
    next: parking_lot::Mutex<Option<Arc<PayloadChunk>>>,
    _funding: Arc<PayloadFunding>,
}

struct PayloadFunding {
    schema: SchemaRef,
    credit: Arc<MemoryReservation>,
    _retirement: RetirementGuard,
}

pub(super) enum RowView<'a> {
    Borrowed(&'a RecordBatch),
    Owned {
        record: RecordBatch,
        _owner: Option<Arc<PayloadChunk>>,
    },
}

impl Deref for RowView<'_> {
    type Target = RecordBatch;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(record) => record,
            Self::Owned { record, .. } => record,
        }
    }
}

impl From<RecordBatch> for RowPayload {
    fn from(record: RecordBatch) -> Self {
        Self::Legacy(record)
    }
}

impl RowPayload {
    #[cfg(test)]
    pub(super) fn funded_owner(&self) -> Option<(usize, usize)> {
        match self {
            Self::Legacy(_) | Self::Rowed { .. } => None,
            Self::Shared { chunk, .. } => {
                let PayloadChunk {
                    _funding: funding, ..
                } = chunk.as_ref();
                Some((Arc::as_ptr(chunk) as usize, funding.credit.size()))
            }
        }
    }

    pub(super) fn schema_ref(&self) -> &SchemaRef {
        match self {
            Self::Legacy(record) => record.schema_ref(),
            Self::Rowed { parent, .. } => parent.schema_ref(),
            Self::Shared { chunk, .. } => &chunk.schema,
        }
    }

    pub(super) fn columns(&self) -> &[ArrayRef] {
        match self {
            Self::Legacy(record) => record.columns(),
            Self::Rowed { parent, .. } => parent.columns(),
            Self::Shared { chunk, .. } => &chunk.columns,
        }
    }

    pub(super) fn offset(&self) -> usize {
        match self {
            Self::Legacy(_) => 0,
            Self::Rowed { row, .. } | Self::Shared { row, .. } => *row,
        }
    }

    pub(super) fn view(&self) -> RowView<'_> {
        match self {
            Self::Legacy(record) => RowView::Borrowed(record),
            Self::Rowed { parent, row } => RowView::Owned {
                record: parent.slice(*row, 1),
                _owner: None,
            },
            Self::Shared { chunk, row } => {
                let columns = chunk
                    .columns
                    .iter()
                    .map(|column| column.slice(*row, 1))
                    .collect();
                RowView::Owned {
                    record: RecordBatch::try_new(Arc::clone(&chunk.schema), columns)
                        .expect("private owned-copy constructors preserve canonical types"),
                    _owner: Some(Arc::clone(chunk)),
                }
            }
        }
    }

    pub(super) fn column(&self, index: usize) -> &ArrayRef {
        &self.columns()[index]
    }

    /// Identity of the backing column set shared by every row of a source
    /// record or chunk, so callers can group rows that can be gathered with
    /// one take; legacy single-row records have no shared identity.
    pub(super) fn shared_chunk_id(&self) -> Option<usize> {
        match self {
            Self::Legacy(_) => None,
            Self::Rowed { parent, .. } => Some(std::ptr::from_ref(parent.columns()).addr()),
            Self::Shared { chunk, .. } => Some(Arc::as_ptr(chunk) as usize),
        }
    }

    pub(super) fn column_view(&self, index: usize) -> ArrayRef {
        match self {
            Self::Legacy(record) => Arc::clone(record.column(index)),
            Self::Rowed { parent, row } => parent.column(index).slice(*row, 1),
            Self::Shared { chunk, row } => chunk.columns[index].slice(*row, 1),
        }
    }

    pub(super) fn num_columns(&self) -> usize {
        self.columns().len()
    }

    #[cfg(test)]
    pub(super) fn get_array_memory_size(&self) -> usize {
        self.columns()
            .iter()
            .map(|column| column.get_array_memory_size())
            .sum()
    }

    pub(super) fn at(record: &RecordBatch, shared: Option<&Arc<PayloadChunk>>, row: usize) -> Self {
        shared.map_or_else(
            || Self::Legacy(record.slice(row, 1)),
            |chunk| Self::Shared {
                chunk: Arc::clone(chunk),
                row,
            },
        )
    }
}

pub(super) struct FramedKey {
    bytes: Vec<u8>,
    _credit: Option<Arc<MemoryReservation>>,
}

#[cfg(test)]
impl FramedKey {
    pub(super) fn funded_owner(&self) -> Option<(usize, usize)> {
        let Self {
            _credit: credit, ..
        } = self;
        credit
            .as_ref()
            .map(|credit| (Arc::as_ptr(credit) as usize, credit.size()))
    }
}

impl FramedKey {
    /// Attaches the batch-level funding reservation that owns these bytes.
    pub(super) fn funded(bytes: Vec<u8>, credit: Arc<MemoryReservation>) -> Self {
        Self {
            bytes,
            _credit: Some(credit),
        }
    }
}

impl From<Vec<u8>> for FramedKey {
    fn from(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            _credit: None,
        }
    }
}

impl Deref for FramedKey {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl PartialEq for FramedKey {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl Eq for FramedKey {}

impl PartialOrd for FramedKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FramedKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.bytes.cmp(&other.bytes)
    }
}
