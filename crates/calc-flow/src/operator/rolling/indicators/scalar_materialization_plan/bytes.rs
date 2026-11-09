use super::{Result, size, typed};
use datafusion::arrow::{
    array::{
        Array, BinaryArray, BinaryViewArray, FixedSizeBinaryArray, LargeBinaryArray,
        LargeStringArray, StringArray, StringViewArray,
    },
    datatypes::DataType as D,
};

pub(super) fn cell(array: &dyn Array, row: usize) -> Result<Option<u64>> {
    let len = match array.data_type() {
        D::Utf8 => typed::<StringArray>(array)?.value(row).len(),
        D::LargeUtf8 => typed::<LargeStringArray>(array)?.value(row).len(),
        D::Utf8View => typed::<StringViewArray>(array)?.value(row).len(),
        _ => return other(array, row),
    };
    size(len).map(Some)
}
fn other(array: &dyn Array, row: usize) -> Result<Option<u64>> {
    let len = match array.data_type() {
        D::Binary => typed::<BinaryArray>(array)?.value(row).len(),
        D::LargeBinary => typed::<LargeBinaryArray>(array)?.value(row).len(),
        _ => return view_binary(array, row),
    };
    size(len).map(Some)
}
fn view_binary(array: &dyn Array, row: usize) -> Result<Option<u64>> {
    let len = match array.data_type() {
        D::BinaryView => typed::<BinaryViewArray>(array)?.value(row).len(),
        D::FixedSizeBinary(_) => typed::<FixedSizeBinaryArray>(array)?.value(row).len(),
        _ => return Ok(None),
    };
    size(len).map(Some)
}
