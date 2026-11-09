use super::{Check, Result, concatenated::Piece, invalid, single};
use datafusion::{arrow::datatypes::DataType, common::ScalarValue};
#[path = "normal/dictionary.rs"]
pub(super) mod dictionary;
#[path = "normal/offsets.rs"]
mod offsets;
#[path = "normal/view_offsets.rs"]
pub(super) mod view_offsets;
#[path = "normal/views.rs"]
mod views;
pub(super) type At<'a> = dyn FnMut(usize) -> Result<Option<Piece<'a>>> + 'a;

pub(super) fn column(dtype: &DataType, values: &[ScalarValue], check: Check<'_>) -> Result<()> {
    if values.len() < 2 || !array_backed(dtype) {
        return Ok(());
    }
    let mut at = |index: usize| root(dtype, &values[index]).map(Some);
    walk(dtype, values.len(), &mut at, check)
}
fn root<'a>(dtype: &DataType, value: &'a ScalarValue) -> Result<Piece<'a>> {
    let array = single::array(value).ok_or_else(|| invalid("normal scalar array"))?;
    if array.data_type() != dtype || array.len() != 1 {
        return Err(invalid("normal scalar shape"));
    }
    Ok(Piece {
        array: Some(array),
        start: 0,
        len: 1,
    })
}
fn array_backed(dtype: &DataType) -> bool {
    match dtype {
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_) => true,
        _ => matches!(dtype, DataType::Map(..) | DataType::Struct(_)),
    }
}
fn walk(dtype: &DataType, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<()> {
    check()?;
    if constructors(count, at, check)? < 2 {
        return Ok(());
    }
    match dtype {
        DataType::List(field) => list::<i32>(field.data_type(), count, at, check),
        DataType::LargeList(field) => list::<i64>(field.data_type(), count, at, check),
        DataType::Map(field, _) => map(field.data_type(), count, at, check),
        _ => nested(dtype, count, at, check),
    }
}
fn constructors(count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<usize> {
    let mut active = 0;
    for index in 0..count {
        check()?;
        active += usize::from(at(index)?.is_some());
    }
    Ok(active)
}
fn list<O: datafusion::arrow::array::OffsetSizeTrait>(
    dtype: &DataType,
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<()> {
    offsets::list::<O>(count, at, check)?;
    let mut child = |index| {
        at(index)?
            .map(super::concatenated::list_piece::<O>)
            .transpose()
    };
    walk(dtype, count, &mut child, check)
}
fn map(dtype: &DataType, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<()> {
    offsets::map(count, at, check)?;
    let mut child = |index| at(index)?.map(super::concatenated::map_piece).transpose();
    walk(dtype, count, &mut child, check)
}
fn nested(dtype: &DataType, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<()> {
    match dtype {
        DataType::Struct(fields) => structure(fields, count, at, check),
        DataType::RunEndEncoded(_, values) => run(values.data_type(), count, at, check),
        DataType::ListView(field) => view::<i32>(field.data_type(), count, at, check),
        DataType::LargeListView(field) => view::<i64>(field.data_type(), count, at, check),
        _ => leaves(dtype, count, at, check),
    }
}
fn leaves(dtype: &DataType, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<()> {
    use datafusion::arrow::datatypes::{BinaryViewType, StringViewType};
    match dtype {
        DataType::Utf8View => views::validate::<StringViewType>(count, at, check),
        DataType::BinaryView => views::validate::<BinaryViewType>(count, at, check),
        DataType::Dictionary(..) => dictionary::validate(dtype, count, at, check),
        DataType::FixedSizeList(..) | DataType::Union(..) => {
            super::producer::normal_fallback(dtype, count, at, check)
        }
        _ => Ok(()),
    }
}
fn structure(
    fields: &datafusion::arrow::datatypes::Fields,
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<()> {
    for (field, value) in fields.iter().enumerate() {
        let mut child = |index| {
            at(index)?
                .map(|piece| super::concatenated::struct_piece(piece, field))
                .transpose()
        };
        walk(value.data_type(), count, &mut child, check)?;
    }
    Ok(())
}
fn run(dtype: &DataType, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<()> {
    let mut child = |index| {
        let piece = at(index)?.filter(|piece| piece.len != 0);
        piece.map(super::concatenated::run_piece).transpose()
    };
    walk(dtype, count, &mut child, check)
}
fn view<O: datafusion::arrow::array::OffsetSizeTrait>(
    dtype: &DataType,
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<()> {
    view_offsets::validate::<O>(count, at, check)?;
    let mut child = |index| {
        at(index)?
            .map(super::concatenated::view_values::<O>)
            .transpose()
    };
    walk(dtype, count, &mut child, check)
}
pub(super) fn list_length<O: datafusion::arrow::array::OffsetSizeTrait>(
    count: usize,
    at: &mut super::concatenated::At<'_>,
    check: Check<'_>,
) -> Result<()> {
    offsets::list::<O>(count, &mut |index| at(index).map(Some), check)
}
pub(super) fn map_length(
    count: usize,
    at: &mut super::concatenated::At<'_>,
    check: Check<'_>,
) -> Result<()> {
    offsets::map(count, &mut |index| at(index).map(Some), check)
}

pub(super) fn view_length<O: datafusion::arrow::array::OffsetSizeTrait>(
    count: usize,
    at: &mut super::concatenated::At<'_>,
    check: Check<'_>,
) -> Result<()> {
    view_offsets::validate::<O>(count, &mut |index| at(index).map(Some), check)
}

pub(super) fn copied_views<T: datafusion::arrow::datatypes::ByteViewType>(
    count: usize,
    at: &mut super::concatenated::At<'_>,
    check: Check<'_>,
) -> Result<()> {
    views::validate::<T>(count, &mut |index| at(index).map(Some), check)
}
