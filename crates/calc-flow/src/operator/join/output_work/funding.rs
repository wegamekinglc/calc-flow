use datafusion::arrow::{
    array::{ArrayRef, make_array},
    buffer::{BooleanBuffer, Buffer, NullBuffer},
    datatypes::DataType,
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;
use tokio_util::bytes::Bytes;

use datafusion::arrow::error::ArrowError;

pub(super) struct OutputFunding {
    pub(super) _credit: MemoryReservation,
}

pub(super) struct OutputBuffer {
    buffer: Buffer,
    _funding: Arc<OutputFunding>,
}

impl AsRef<[u8]> for OutputBuffer {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

fn wrap(buffer: Buffer, funding: &Arc<OutputFunding>) -> Buffer {
    Buffer::from(Bytes::from_owner(OutputBuffer {
        buffer,
        _funding: Arc::clone(funding),
    }))
}

fn wrap_nulls(nulls: &NullBuffer, funding: &Arc<OutputFunding>) -> NullBuffer {
    NullBuffer::new(BooleanBuffer::new(
        wrap(nulls.buffer().clone(), funding),
        nulls.offset(),
        nulls.len(),
    ))
}

pub(super) fn bind(
    array: ArrayRef,
    canonical: &DataType,
    funding: &Arc<OutputFunding>,
) -> Result<ArrayRef, ArrowError> {
    let data = array.to_data();
    drop(array);
    let buffers = data
        .buffers()
        .iter()
        .cloned()
        .map(|buffer| wrap(buffer, funding))
        .collect();
    let nulls = data.nulls().map(|nulls| wrap_nulls(nulls, funding));
    let rebuilt = data
        .into_builder()
        .data_type(canonical.clone())
        .buffers(buffers)
        .nulls(nulls)
        .build()?;
    Ok(make_array(rebuilt))
}
