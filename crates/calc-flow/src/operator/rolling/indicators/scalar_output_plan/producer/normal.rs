use super::{Check, Result, invalid};
use datafusion::arrow::{
    array::{Array, ArrayRef, BooleanBufferBuilder, OffsetSizeTrait},
    buffer::NullBuffer,
    datatypes::{DataType, FieldRef},
};
use std::sync::Arc;
#[path = "normal/lists.rs"]
mod lists;
#[path = "normal/run.rs"]
mod run;
#[path = "normal/structures.rs"]
mod structures;

pub(super) fn needed(dtype: &DataType, check: Check<'_>) -> Result<bool> {
    check()?;
    match dtype {
        DataType::ListView(_) | DataType::LargeListView(_) => Ok(true),
        DataType::Struct(fields) => fields.iter().try_fold(false, |yes, field| {
            Ok(yes || needed(field.data_type(), check)?)
        }),
        _ => inherited(dtype).map_or(Ok(false), |child| needed(child, check)),
    }
}
fn inherited(dtype: &DataType) -> Option<&DataType> {
    match dtype {
        DataType::List(field) | DataType::LargeList(field) | DataType::Map(field, _) => {
            Some(field.data_type())
        }
        DataType::RunEndEncoded(_, values) => Some(values.data_type()),
        _ => None,
    }
}
pub(super) fn concat(arrays: &[&dyn Array], check: Check<'_>) -> Result<ArrayRef> {
    check()?;
    let first = arrays
        .first()
        .ok_or_else(|| invalid("empty normal concat"))?;
    if arrays.len() == 1 || !needed(first.data_type(), check)? {
        return datafusion::arrow::compute::concat(arrays).map_err(|_| invalid("normal concat"));
    }
    nested(first.data_type(), arrays, check)
}
fn nested(dtype: &DataType, arrays: &[&dyn Array], check: Check<'_>) -> Result<ArrayRef> {
    match dtype {
        DataType::ListView(field) => lists::view::<i32>(arrays, field, check),
        DataType::LargeListView(field) => lists::view::<i64>(arrays, field, check),
        DataType::List(field) => lists::list::<i32>(arrays, field, check),
        DataType::LargeList(field) => lists::list::<i64>(arrays, field, check),
        _ => composite(dtype, arrays, check),
    }
}
fn composite(dtype: &DataType, arrays: &[&dyn Array], check: Check<'_>) -> Result<ArrayRef> {
    match dtype {
        DataType::Map(field, ordered) => structures::map(arrays, field, *ordered, check),
        DataType::Struct(fields) => structures::structure(arrays, fields, check),
        DataType::RunEndEncoded(..) => run::concat(arrays, check),
        _ => Err(invalid("normal concat route")),
    }
}
fn typed<T: Array + 'static>(array: &dyn Array) -> Result<&T> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| invalid("normal array representation"))
}
fn length(arrays: &[&dyn Array], check: Check<'_>) -> Result<usize> {
    arrays.iter().try_fold(0usize, |len, array| {
        check()?;
        len.checked_add(array.len())
            .ok_or_else(|| invalid("normal array length"))
    })
}
fn nulls(arrays: &[&dyn Array], len: usize, check: Check<'_>) -> Result<Option<NullBuffer>> {
    if !arrays.iter().any(|array| array.null_count() != 0) {
        return Ok(None);
    }
    let mut builder = BooleanBufferBuilder::new(len);
    for array in arrays {
        check()?;
        match array.nulls() {
            Some(nulls) => builder.append_buffer(nulls.inner()),
            None => builder.append_n(array.len(), true),
        }
    }
    Ok(Some(NullBuffer::new(builder.finish())))
}
fn field_array<T: Array + 'static>(array: T) -> ArrayRef {
    Arc::new(array)
}
fn offsets<O: OffsetSizeTrait>(
    values: impl IntoIterator<Item = usize>,
) -> datafusion::arrow::buffer::OffsetBuffer<O> {
    datafusion::arrow::buffer::OffsetBuffer::from_lengths(values)
}
fn child_refs(arrays: &[ArrayRef]) -> Vec<&dyn Array> {
    arrays.iter().map(AsRef::as_ref).collect()
}

#[cfg(test)]
#[path = "normal/tests.rs"]
mod tests;
