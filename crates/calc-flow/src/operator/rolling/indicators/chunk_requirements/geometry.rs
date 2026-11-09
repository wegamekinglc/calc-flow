use std::{
    mem::{align_of, size_of},
    sync::OnceLock,
};

use datafusion::arrow::{array::ArrayRef, record_batch::RecordBatch};

use super::{
    Result,
    bytes::{size, slots, total},
};

/// New `BatchMetadata` owners, `Arc<[RecordBatch]>`, estimate cache and the cloned
/// column handle `Vec`. Source arrays and schema Arcs remain shared.
pub(super) fn batch(columns: usize, name_bytes: usize) -> Result<u64> {
    total([
        arc::<crate::JsonMap>(),
        arc::<RecordBatch>(),
        arc::<OnceLock<std::result::Result<usize, String>>>(),
        source(name_bytes),
        slots::<ArrayRef>(columns),
    ])
}
fn source(bytes: usize) -> Result<u64> {
    total([
        size(bytes),
        slots::<usize>(2),
        size(align_of::<usize>() - 1),
    ])
}
fn arc<T>() -> Result<u64> {
    total([
        size(size_of::<T>()),
        slots::<usize>(2),
        size(align_of::<T>().max(align_of::<usize>()) - 1),
    ])
}
/// Transient source `String` during `String` to `Arc<str>`, and input `Vec` during
/// `Vec<RecordBatch>` to `Arc<[RecordBatch]>`; final owners are already in O.
pub(super) fn batch_temporary(name_bytes: usize) -> Result<u64> {
    total([size(name_bytes), slots::<RecordBatch>(1)])
}
