use super::{Check, Result, invalid};
use datafusion::arrow::{
    array::{ArrayData, ArrayRef},
    datatypes::DataType,
};

pub(super) fn validate(arrays: &[ArrayRef], check: Check<'_>) -> Result<()> {
    if arrays.len() < 2 || !needs_guard(arrays[0].data_type()) {
        return Ok(());
    }
    let mut data = Vec::with_capacity(arrays.len());
    for array in arrays {
        check()?;
        data.push(array.to_data());
    }
    validate_data(&data, check)
}
fn validate_data(data: &[ArrayData], check: Check<'_>) -> Result<()> {
    walk(data.len(), &|index| &data[index], check)?;
    super::views::validate(data, check)
}
pub(super) fn normal(
    dtype: &DataType,
    count: usize,
    at: &mut super::super::normal::At<'_>,
    check: Check<'_>,
) -> Result<()> {
    if !needs_guard(dtype) {
        return Ok(());
    }
    let mut data = Vec::with_capacity(count);
    for index in 0..count {
        check()?;
        if let Some(piece) = at(index)? {
            data.push(normal_data(dtype, piece, check)?);
        }
    }
    validate_normal_data(&data, check)
}
fn validate_normal_data(data: &[ArrayData], check: Check<'_>) -> Result<()> {
    if data.len() < 2 {
        return Ok(());
    }
    validate_data(data, check)
}
fn normal_data(
    dtype: &DataType,
    piece: super::super::concatenated::Piece<'_>,
    check: Check<'_>,
) -> Result<ArrayData> {
    let array = match piece.array {
        Some(array) => array.slice(piece.start, piece.len),
        None => super::null_array(dtype, piece.len, check)?,
    };
    check()?;
    let data = array.to_data();
    check()?;
    Ok(data)
}
fn walk<'a>(count: usize, at: &dyn Fn(usize) -> &'a ArrayData, check: Check<'_>) -> Result<()> {
    check()?;
    let first = at(0);
    if let DataType::Dictionary(key, _) = first.data_type() {
        return dictionary(count, at, key, check);
    }
    for child in 0..first.child_data().len() {
        walk(count, &|index| &at(index).child_data()[child], check)?;
    }
    Ok(())
}
fn dictionary<'a>(
    count: usize,
    at: &dyn Fn(usize) -> &'a ArrayData,
    key: &DataType,
    check: Check<'_>,
) -> Result<()> {
    let shared = same_values(count, at, check)?;
    spans(count, at, key, shared, check)?;
    if !shared {
        walk(count, &|index| &at(index).child_data()[0], check)?;
    }
    Ok(())
}
fn spans<'a>(
    count: usize,
    at: &dyn Fn(usize) -> &'a ArrayData,
    key: &DataType,
    shared: bool,
    check: Check<'_>,
) -> Result<()> {
    let mut offset = 0_u64;
    for index in 0..count {
        check()?;
        let len = u64::try_from(at(index).child_data()[0].len())
            .map_err(|_| invalid("dictionary fallback key span"))?;
        span(key, offset, len)?;
        if !shared {
            offset = offset
                .checked_add(len)
                .ok_or_else(|| invalid("dictionary fallback key span"))?;
        }
    }
    Ok(())
}
pub(super) fn same_values<'a>(
    count: usize,
    at: &dyn Fn(usize) -> &'a ArrayData,
    check: Check<'_>,
) -> Result<bool> {
    for index in 1..count {
        check()?;
        if !at(index - 1).child_data()[0].ptr_eq(&at(index).child_data()[0]) {
            return Ok(false);
        }
    }
    Ok(true)
}
pub(super) fn span(key: &DataType, offset: u64, len: u64) -> Result<()> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| invalid("dictionary fallback key span"))?;
    let limit = key_limit(key)?;
    if offset > limit || end.saturating_sub(1) > limit {
        return Err(invalid("dictionary fallback key span"));
    }
    Ok(())
}
pub(super) fn key_limit(key: &DataType) -> Result<u64> {
    match key {
        DataType::Int8 => Ok(u64::try_from(i8::MAX).expect("positive key limit")),
        DataType::Int16 => Ok(u64::try_from(i16::MAX).expect("positive key limit")),
        DataType::Int32 => Ok(u64::try_from(i32::MAX).expect("positive key limit")),
        DataType::Int64 => Ok(u64::try_from(i64::MAX).expect("positive key limit")),
        _ => unsigned_limit(key),
    }
}
fn unsigned_limit(key: &DataType) -> Result<u64> {
    match key {
        DataType::UInt8 => Ok(u64::from(u8::MAX)),
        DataType::UInt16 => Ok(u64::from(u16::MAX)),
        DataType::UInt32 => Ok(u64::from(u32::MAX)),
        DataType::UInt64 => Ok(u64::MAX),
        _ => Err(invalid("dictionary key type")),
    }
}
fn needs_guard(dtype: &DataType) -> bool {
    match dtype {
        DataType::Dictionary(..)
        | DataType::Utf8View
        | DataType::BinaryView
        | DataType::Union(_, datafusion::arrow::datatypes::UnionMode::Dense) => true,
        DataType::Struct(fields) => fields.iter().any(|field| needs_guard(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| needs_guard(field.data_type())),
        DataType::RunEndEncoded(_, field) => needs_guard(field.data_type()),
        _ => list_needs_guard(dtype),
    }
}
fn list_needs_guard(dtype: &DataType) -> bool {
    match dtype {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::Map(field, _) => needs_guard(field.data_type()),
        _ => false,
    }
}
