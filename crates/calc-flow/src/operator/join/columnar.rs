use super::encode_join_key_columns_v1;
use crate::Result;
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
    Rowed {
        parent: Arc<RecordBatch>,
        row: usize,
    },
    Shared {
        chunk: Arc<PayloadChunk>,
        row: usize,
    },
    RestoredV2 {
        chunk: Arc<super::checkpoint_v2::payload::OwnedPayload>,
        row: usize,
    },
}

#[cfg(test)]
const _: () = {
    enum V1RowPayload {
        Legacy(RecordBatch),
        Shared {
            chunk: Arc<PayloadChunk>,
            row: usize,
        },
    }
    let _: fn(RecordBatch) -> V1RowPayload = V1RowPayload::Legacy;
    let _: fn(Arc<PayloadChunk>, usize) -> V1RowPayload =
        |chunk, row| V1RowPayload::Shared { chunk, row };
    let _: fn(V1RowPayload) = |payload| match payload {
        V1RowPayload::Legacy(record) => drop(record),
        V1RowPayload::Shared { chunk, row } => drop((chunk, row)),
    };
    assert!(size_of::<RowPayload>() == size_of::<V1RowPayload>());
    assert!(align_of::<RowPayload>() == align_of::<V1RowPayload>());
};

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
    Parent {
        record: RecordBatch,
        _owner: Arc<RecordBatch>,
    },
    Owned {
        record: RecordBatch,
        _owner: Arc<PayloadChunk>,
    },
    RestoredV2 {
        record: RecordBatch,
        _owner: Arc<super::checkpoint_v2::payload::OwnedPayload>,
    },
}

impl Deref for RowView<'_> {
    type Target = RecordBatch;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(record) => record,
            Self::Parent { record, .. }
            | Self::Owned { record, .. }
            | Self::RestoredV2 { record, .. } => record,
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
    pub(super) fn is_unfunded_legacy(&self) -> bool {
        matches!(self, Self::Legacy(_) | Self::Rowed { .. })
    }

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
            Self::RestoredV2 { chunk, .. } => Some(chunk.funded_owner()),
        }
    }

    pub(super) fn schema_ref(&self) -> &SchemaRef {
        match self {
            Self::Legacy(record) => record.schema_ref(),
            Self::Rowed { parent, .. } => parent.schema_ref(),
            Self::Shared { chunk, .. } => &chunk.schema,
            Self::RestoredV2 { chunk, .. } => chunk.record().schema_ref(),
        }
    }

    pub(super) fn columns(&self) -> &[ArrayRef] {
        match self {
            Self::Legacy(record) => record.columns(),
            Self::Rowed { parent, .. } => parent.columns(),
            Self::Shared { chunk, .. } => &chunk.columns,
            Self::RestoredV2 { chunk, .. } => chunk.record().columns(),
        }
    }

    pub(super) fn shared_columns(&self) -> Option<&[ArrayRef]> {
        match self {
            Self::Legacy(_) => None,
            _ => Some(self.columns()),
        }
    }

    pub(super) fn offset(&self) -> usize {
        match self {
            Self::Legacy(_) => 0,
            Self::Rowed { row, .. } | Self::Shared { row, .. } | Self::RestoredV2 { row, .. } => {
                *row
            }
        }
    }

    pub(super) fn view(&self) -> RowView<'_> {
        match self {
            Self::Legacy(record) => RowView::Borrowed(record),
            Self::Rowed { parent, row } => RowView::Parent {
                record: parent.slice(*row, 1),
                _owner: Arc::clone(parent),
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
                    _owner: Arc::clone(chunk),
                }
            }
            Self::RestoredV2 { chunk, row } => RowView::RestoredV2 {
                record: chunk.record().slice(*row, 1),
                _owner: Arc::clone(chunk),
            },
        }
    }

    pub(super) fn column(&self, index: usize) -> &ArrayRef {
        &self.columns()[index]
    }

    pub(super) fn column_view(&self, index: usize) -> ArrayRef {
        #[cfg(test)]
        super::note_join_work(|work| work.output_column_views += 1);
        match self {
            Self::Legacy(record) => Arc::clone(record.column(index)),
            Self::Rowed { parent, row } => parent.column(index).slice(*row, 1),
            Self::Shared { chunk, row } => chunk.columns[index].slice(*row, 1),
            Self::RestoredV2 { chunk, row } => chunk.record().column(index).slice(*row, 1),
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

pub(super) fn funded_key(
    columns: &[ArrayRef],
    row: usize,
    indices: &[usize],
    credit: Arc<MemoryReservation>,
) -> Result<Arc<FramedKey>> {
    Ok(Arc::new(FramedKey {
        bytes: encode_join_key_columns_v1(columns, row, indices)?,
        _credit: Some(credit),
    }))
}

pub(super) fn funded_encoded_key(bytes: Vec<u8>, credit: Arc<MemoryReservation>) -> Arc<FramedKey> {
    Arc::new(FramedKey {
        bytes,
        _credit: Some(credit),
    })
}
