use super::super::super::{concatenated::Piece, math::size};
use super::{Check, Result, invalid};
use datafusion::arrow::{
    array::{Array, ArrayData, AsArray, GenericByteArray, PrimitiveArray, downcast_primitive},
    datatypes::{
        ArrowPrimitiveType, BinaryType, ByteArrayType, DataType, LargeBinaryType, LargeUtf8Type,
        Utf8Type,
    },
};

pub(super) fn values(piece: Piece<'_>) -> Result<Option<&dyn Array>> {
    let Some(array) = piece.array else {
        return Ok(None);
    };
    Ok(Some(
        array
            .as_any_dictionary_opt()
            .ok_or_else(|| invalid("normal dictionary array"))?
            .values()
            .as_ref(),
    ))
}
pub(super) fn mergeable(dtype: &DataType) -> bool {
    dtype.is_primitive()
        || matches!(
            dtype,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary
        )
}
pub(super) fn equal(
    dtype: &DataType,
    left: Piece<'_>,
    right: Piece<'_>,
    check: Check<'_>,
) -> Result<bool> {
    match (values(left)?, values(right)?) {
        (Some(left), Some(right)) => pair(dtype, left, right, check),
        _ => missing(dtype, left, right, check),
    }
}
fn missing(dtype: &DataType, left: Piece<'_>, right: Piece<'_>, check: Check<'_>) -> Result<bool> {
    let empty = super::super::super::super::materialize::canonical::null_array(
        dtype,
        0,
        "rolling-output-plan",
        check,
    )?;
    let second = super::super::super::super::materialize::canonical::null_array(
        dtype,
        0,
        "rolling-output-plan",
        check,
    )?;
    pair(
        dtype,
        values(left)?.unwrap_or(empty.as_ref()),
        values(right)?.unwrap_or(second.as_ref()),
        check,
    )
}
fn pair(dtype: &DataType, left: &dyn Array, right: &dyn Array, check: Check<'_>) -> Result<bool> {
    check()?;
    match dtype {
        DataType::Utf8 => bytes_equal::<Utf8Type>(left, right),
        DataType::LargeUtf8 => bytes_equal::<LargeUtf8Type>(left, right),
        DataType::Binary => bytes_equal::<BinaryType>(left, right),
        DataType::LargeBinary => bytes_equal::<LargeBinaryType>(left, right),
        _ => primitive_equal(left, right, check),
    }
}
fn primitive_equal(left: &dyn Array, right: &dyn Array, check: Check<'_>) -> Result<bool> {
    let left: ArrayData = left.to_data();
    let right = right.to_data();
    let same = left.ptr_eq(&right);
    check()?;
    Ok(same)
}
fn bytes_equal<T: ByteArrayType>(left: &dyn Array, right: &dyn Array) -> Result<bool> {
    let left = typed::<GenericByteArray<T>>(left)?;
    let right = typed::<GenericByteArray<T>>(right)?;
    Ok(left.values().ptr_eq(right.values())
        && left.offsets().ptr_eq(right.offsets())
        && nulls_equal(left, right))
}
fn nulls_equal(left: &dyn Array, right: &dyn Array) -> bool {
    match (left.nulls(), right.nulls()) {
        (Some(left), Some(right)) => left.inner().ptr_eq(right.inner()),
        (None, None) => true,
        _ => false,
    }
}
macro_rules! raw_helper {
    ($t:ty, $array:expr, $index:expr) => {
        primitive_raw::<$t>($array, $index)
    };
}
pub(super) fn raw(array: &dyn Array, index: usize) -> Result<Option<&[u8]>> {
    if array.is_null(index) {
        return Ok(None);
    }
    let dtype = array.data_type();
    downcast_primitive!(dtype => (raw_helper, array, index), _ => byte_raw(dtype, array, index))
}
fn primitive_raw<T: ArrowPrimitiveType>(array: &dyn Array, index: usize) -> Result<Option<&[u8]>> {
    let array = typed::<PrimitiveArray<T>>(array)?;
    let width = size_of::<T::Native>();
    Ok(Some(
        &array.values().inner().as_slice()[index * width..(index + 1) * width],
    ))
}
fn byte_raw<'a>(dtype: &DataType, array: &'a dyn Array, index: usize) -> Result<Option<&'a [u8]>> {
    match dtype {
        DataType::Utf8 => byte_value::<Utf8Type>(array, index),
        DataType::LargeUtf8 => byte_value::<LargeUtf8Type>(array, index),
        DataType::Binary => byte_value::<BinaryType>(array, index),
        DataType::LargeBinary => byte_value::<LargeBinaryType>(array, index),
        _ => Err(invalid("normal merged value type")),
    }
}
fn byte_value<T: ByteArrayType>(array: &dyn Array, index: usize) -> Result<Option<&[u8]>> {
    Ok(Some(
        typed::<GenericByteArray<T>>(array)?.value(index).as_ref(),
    ))
}
pub(super) fn raw_length(array: &dyn Array, index: usize) -> Result<u64> {
    let dtype = array.data_type();
    match dtype {
        DataType::Utf8 => byte_length::<Utf8Type>(array, index),
        DataType::LargeUtf8 => byte_length::<LargeUtf8Type>(array, index),
        DataType::Binary => byte_length::<BinaryType>(array, index),
        DataType::LargeBinary => byte_length::<LargeBinaryType>(array, index),
        _ => Ok(0),
    }
}
fn byte_length<T: ByteArrayType>(array: &dyn Array, index: usize) -> Result<u64> {
    use datafusion::arrow::datatypes::ArrowNativeType;
    let array = typed::<GenericByteArray<T>>(array)?;
    let offsets = array.value_offsets();
    size(offsets[index + 1].as_usize() - offsets[index].as_usize())
}
fn typed<T: Array + 'static>(array: &dyn Array) -> Result<&T> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| invalid("normal dictionary value representation"))
}
