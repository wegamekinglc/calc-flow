use super::{PayloadFunding, owned_copy};
use datafusion::arrow::{
    array::ArrayRef,
    buffer::{BooleanBuffer, MutableBuffer, NullBuffer},
    datatypes::{DataType, Field},
};
use std::sync::Arc;

pub(in crate::operator::join) fn nullable_float(field: &Field) -> bool {
    field.is_nullable() && field.data_type() == &DataType::Float64
}

pub(in crate::operator::join) fn backing(field: &Field) -> usize {
    if nullable_float(field) { 64 } else { 0 }
}

pub(in crate::operator::join) fn certified(
    field: &Field,
    rows: usize,
    null_count: i64,
    bitmap: &[u8],
) -> bool {
    if null_count == 0 {
        return true;
    }
    if !nullable_float(field) || rows != 1 || null_count != 1 {
        return false;
    }
    bitmap.len() == 1 && bitmap[0] & 1 == 0
}

pub(super) fn visible_bytes(source: &ArrayRef) -> usize {
    usize::from(source.null_count() != 0)
}

pub(super) fn copy(source: &ArrayRef, funding: &Arc<PayloadFunding>) -> Option<NullBuffer> {
    if source.null_count() == 0 {
        return None;
    }
    let bits = source.nulls().expect("certified Float64 null slot").inner();
    let byte = if bits.offset().is_multiple_of(8) {
        bits.values()[bits.offset() / 8]
    } else {
        u8::from(bits.value(0))
    };
    let mut values = MutableBuffer::new(1);
    values.extend_from_slice(&[byte]);
    Some(NullBuffer::new(BooleanBuffer::new(
        owned_copy::wrap_buffer(values.into(), funding),
        0,
        1,
    )))
}
