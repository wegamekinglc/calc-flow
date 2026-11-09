use super::{Check, Result, invalid};
use datafusion::{
    arrow::{
        array::{Array, ArrayData, ArrayRef, PrimitiveArray, RunArray},
        datatypes::{
            ArrowNativeType, DataType, FieldRef, Int16Type, Int32Type, Int64Type, RunEndIndexType,
        },
    },
    common::ScalarValue,
};
use std::sync::Arc;

pub(super) fn column(
    values: Vec<ScalarValue>,
    ends: &FieldRef,
    value: &FieldRef,
    check: Check<'_>,
) -> Result<ArrayRef> {
    match ends.data_type() {
        DataType::Int16 => build_column::<Int16Type>(values, ends, value, check),
        DataType::Int32 => build_column::<Int32Type>(values, ends, value, check),
        DataType::Int64 => build_column::<Int64Type>(values, ends, value, check),
        _ => Err(invalid("run index type")),
    }
}
pub(super) fn single(
    ends: &FieldRef,
    field: &FieldRef,
    value: &ScalarValue,
    check: Check<'_>,
) -> Result<ArrayRef> {
    let values = super::single(value, check)?;
    match ends.data_type() {
        DataType::Int16 => finish::<Int16Type>(vec![1], &values, ends, field, 1),
        DataType::Int32 => finish::<Int32Type>(vec![1], &values, ends, field, 1),
        DataType::Int64 => finish::<Int64Type>(vec![1], &values, ends, field, 1),
        _ => Err(invalid("run index type")),
    }
}
fn unpack(value: ScalarValue, ends: &FieldRef, field: &FieldRef) -> Result<ScalarValue> {
    match value {
        ScalarValue::RunEndEncoded(actual_ends, actual_field, value)
            if &actual_ends == ends && &actual_field == field =>
        {
            Ok(*value)
        }
        _ => Err(invalid("run scalar fields")),
    }
}
fn build_column<R: RunEndIndexType>(
    values: Vec<ScalarValue>,
    ends: &FieldRef,
    field: &FieldRef,
    check: Check<'_>,
) -> Result<ArrayRef> {
    let len = values.len();
    index_value::<R>(len)?;
    let (run_ends, values) = coalesce::<R>(values, ends, field, check)?;
    finish::<R>(run_ends, &super::column(values, check)?, ends, field, len)
}
fn first_value(
    values: &mut impl Iterator<Item = ScalarValue>,
    ends: &FieldRef,
    field: &FieldRef,
) -> Result<ScalarValue> {
    unpack(
        values.next().ok_or_else(|| invalid("empty run column"))?,
        ends,
        field,
    )
}
fn coalesce<R: RunEndIndexType>(
    values: Vec<ScalarValue>,
    ends: &FieldRef,
    field: &FieldRef,
    check: Check<'_>,
) -> Result<(Vec<R::Native>, Vec<ScalarValue>)> {
    let len = values.len();
    let mut values = values.into_iter();
    let mut runs = Runs::<R>::new(first_value(&mut values, ends, field)?);
    for (index, value) in values.enumerate() {
        check()?;
        runs.push(index, unpack(value, ends, field)?)?;
    }
    runs.finish(len)
}
struct Runs<R: RunEndIndexType> {
    current: ScalarValue,
    ends: Vec<R::Native>,
    values: Vec<ScalarValue>,
}
impl<R: RunEndIndexType> Runs<R> {
    fn new(current: ScalarValue) -> Self {
        Self {
            current,
            ends: Vec::new(),
            values: Vec::new(),
        }
    }
    fn push(&mut self, index: usize, next: ScalarValue) -> Result<()> {
        if self.current != next {
            let end = index
                .checked_add(1)
                .ok_or_else(|| invalid("run index overflow"))?;
            self.ends.push(index_value::<R>(end)?);
            self.values.push(std::mem::replace(&mut self.current, next));
        }
        Ok(())
    }
    fn finish(mut self, len: usize) -> Result<(Vec<R::Native>, Vec<ScalarValue>)> {
        self.ends.push(index_value::<R>(len)?);
        self.values.push(self.current);
        Ok((self.ends, self.values))
    }
}

fn index_value<R: RunEndIndexType>(len: usize) -> Result<R::Native> {
    R::Native::from_usize(len).ok_or_else(|| invalid("run index overflow"))
}
fn finish<R: RunEndIndexType>(
    ends: Vec<R::Native>,
    values: &ArrayRef,
    ends_field: &FieldRef,
    value_field: &FieldRef,
    len: usize,
) -> Result<ArrayRef> {
    let ends = PrimitiveArray::<R>::from_iter_values(ends);
    let data = ArrayData::builder(DataType::RunEndEncoded(
        Arc::clone(ends_field),
        Arc::clone(value_field),
    ))
    .len(len)
    .add_child_data(ends.to_data())
    .add_child_data(values.to_data())
    .build()
    .map_err(|_| invalid("run array"))?;
    Ok(Arc::new(RunArray::<R>::from(data)))
}
