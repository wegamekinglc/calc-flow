//! Detach accepted payload rows from their source buffers before retaining them.

use crate::{Result, operator::asof::arrow_error};
use datafusion::arrow::{
    array::{Array, ArrayRef, AsArray, GenericByteArray, PrimitiveArray, UInt32Array},
    buffer::{Buffer, OffsetBuffer},
    compute::take,
    datatypes::{ArrowNativeType, ArrowPrimitiveType, ByteArrayType, DataType},
    downcast_primitive_array,
    error::ArrowError,
    record_batch::RecordBatch,
};
use std::sync::Arc;

pub(super) fn compact_accepted(
    record: &RecordBatch,
    accepted: &[bool],
    selected_rows: usize,
) -> Result<RecordBatch> {
    let full = selected_rows == record.num_rows()
        && u32::try_from(selected_rows.saturating_sub(1)).is_ok()
        && record.columns().iter().all(|column| {
            column.null_count() == 0
                && (column.data_type().is_primitive()
                    || matches!(
                        column.data_type(),
                        DataType::Utf8
                            | DataType::LargeUtf8
                            | DataType::Binary
                            | DataType::LargeBinary
                    ))
        });
    let columns = if full {
        record
            .columns()
            .iter()
            .map(|column| copy_full_column(column.as_ref()))
            .collect::<Result<_>>()?
    } else {
        let rows = accepted
            .iter()
            .enumerate()
            .filter(|(_, keep)| **keep)
            .map(|(row, _)| {
                u32::try_from(row).map_err(|_| {
                    arrow_error(&ArrowError::InvalidArgumentError(
                        "ASOF accepted row exceeds UInt32".into(),
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let indices = UInt32Array::from(rows);
        record
            .columns()
            .iter()
            .map(|column| {
                take(column.as_ref(), &indices, None).map_err(|error| arrow_error(&error))
            })
            .collect::<Result<_>>()?
    };
    RecordBatch::try_new(record.schema(), columns).map_err(|error| arrow_error(&error))
}

fn copy_full_column(column: &dyn Array) -> Result<ArrayRef> {
    downcast_primitive_array!(
        column => Ok(copy_primitive(column)),
        DataType::Utf8 => copy_bytes(column.as_string::<i32>()),
        DataType::LargeUtf8 => copy_bytes(column.as_string::<i64>()),
        DataType::Binary => copy_bytes(column.as_binary::<i32>()),
        DataType::LargeBinary => copy_bytes(column.as_binary::<i64>()),
        _ => unreachable!("full column copy requires a primitive or byte array")
    )
}

fn copy_primitive<T: ArrowPrimitiveType>(values: &PrimitiveArray<T>) -> ArrayRef {
    Arc::new(
        PrimitiveArray::<T>::new(values.values().to_vec().into(), None)
            .with_data_type(values.data_type().clone()),
    )
}

fn copy_bytes<T: ByteArrayType>(array: &GenericByteArray<T>) -> Result<ArrayRef> {
    let offsets = array.value_offsets();
    let base = offsets[0];
    let end = offsets[array.len()].as_usize();
    let bytes = Buffer::from_vec(array.value_data()[base.as_usize()..end].to_vec());
    let offsets = offsets
        .iter()
        .map(|&offset| offset - base)
        .collect::<Vec<_>>();
    let copied = GenericByteArray::<T>::try_new(OffsetBuffer::new(offsets.into()), bytes, None)
        .map_err(|error| arrow_error(&error))?;
    Ok(Arc::new(copied))
}

#[cfg(test)]
mod tests;
