use super::{Footprint, overflow};
use crate::Result;
use datafusion::arrow::{
    array::{
        Array, BooleanArray, FixedSizeBinaryArray, GenericByteArray, GenericByteViewArray,
        PrimitiveArray,
    },
    datatypes::{
        ArrowPrimitiveType, BinaryType, BinaryViewType, ByteArrayType, ByteViewType, DataType,
        LargeBinaryType, LargeUtf8Type, StringViewType, Utf8Type,
    },
};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
macro_rules! primitive {
    ($ty:ty, $footprint:ident, $array:ident, $node:ident, $check:ident) => {
        primitive::<$ty>($footprint, $array, $node)
    };
}
pub(super) fn backing(
    footprint: &mut Footprint,
    array: &dyn Array,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    datafusion::arrow::array::downcast_primitive! {
        array.data_type() => (primitive, footprint, array, node, check),
        _ => bytes(footprint, array, node, check)
    }
}
fn primitive<T: ArrowPrimitiveType>(
    footprint: &mut Footprint,
    array: &dyn Array,
    node: &str,
) -> Result<()> {
    footprint.buffer(
        typed::<PrimitiveArray<T>>(array, node)?.values().inner(),
        node,
    )
}
fn bytes(footprint: &mut Footprint, array: &dyn Array, node: &str, check: Check<'_>) -> Result<()> {
    match array.data_type() {
        DataType::Utf8 => byte_value::<Utf8Type>(footprint, array, node),
        DataType::LargeUtf8 => byte_value::<LargeUtf8Type>(footprint, array, node),
        DataType::Binary => byte_value::<BinaryType>(footprint, array, node),
        DataType::LargeBinary => byte_value::<LargeBinaryType>(footprint, array, node),
        _ => other(footprint, array, node, check),
    }
}
fn other(footprint: &mut Footprint, array: &dyn Array, node: &str, check: Check<'_>) -> Result<()> {
    match array.data_type() {
        DataType::Utf8View => view_value::<StringViewType>(footprint, array, node, check),
        DataType::BinaryView => view_value::<BinaryViewType>(footprint, array, node, check),
        DataType::FixedSizeBinary(_) => fixed(footprint, array, node),
        DataType::Boolean => boolean(footprint, array, node),
        DataType::Null => Ok(()),
        _ => Err(overflow(node)),
    }
}
fn fixed(footprint: &mut Footprint, array: &dyn Array, node: &str) -> Result<()> {
    footprint.buffer(typed::<FixedSizeBinaryArray>(array, node)?.values(), node)
}
fn boolean(footprint: &mut Footprint, array: &dyn Array, node: &str) -> Result<()> {
    footprint.buffer(typed::<BooleanArray>(array, node)?.values().inner(), node)
}
fn byte_value<T: ByteArrayType>(
    footprint: &mut Footprint,
    array: &dyn Array,
    node: &str,
) -> Result<()> {
    byte_array(footprint, typed::<GenericByteArray<T>>(array, node)?, node)
}
fn view_value<T: ByteViewType>(
    footprint: &mut Footprint,
    array: &dyn Array,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    view(
        footprint,
        typed::<GenericByteViewArray<T>>(array, node)?,
        node,
        check,
    )
}
fn byte_array<T: ByteArrayType>(
    footprint: &mut Footprint,
    array: &GenericByteArray<T>,
    node: &str,
) -> Result<()> {
    footprint.buffer(array.offsets().inner().inner(), node)?;
    footprint.buffer(array.values(), node)
}
fn view<T: ByteViewType>(
    footprint: &mut Footprint,
    array: &GenericByteViewArray<T>,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    footprint.buffer(array.views().inner(), node)?;
    for buffer in array.data_buffers() {
        check()?;
        footprint.buffer(buffer, node)?;
    }
    Ok(())
}
fn typed<'a, T: Array + 'static>(array: &'a dyn Array, node: &str) -> Result<&'a T> {
    array.as_any().downcast_ref().ok_or_else(|| overflow(node))
}
