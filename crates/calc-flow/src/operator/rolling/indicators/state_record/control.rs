use std::alloc::Layout;

use datafusion::arrow::{
    array::builder::NullBufferBuilder,
    array::{ArrayRef, PrimitiveArray},
    datatypes::ArrowPrimitiveType,
};
use std::sync::Arc;

use super::schema::{add, arc, size, total};
use crate::{CalcFlowError, Result};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
#[derive(Clone, Copy)]
pub(super) struct Cost {
    pub(super) retained: u64,
    pub(super) construction_peak: u64,
}
pub(super) fn requirement<T: ArrowPrimitiveType>(rows: usize, nullable: bool) -> Result<Cost> {
    let data = data_bytes::<T>(rows)?;
    let mask = mask_bytes(rows, nullable)?;
    let retained = total([arc::<PrimitiveArray<T>>(), Ok(data), size(mask)])?;
    let headers = (1 + usize::from(mask > 0)) * 7 * size_of::<usize>();
    Ok(Cost {
        retained,
        construction_peak: add(retained, size(headers)?)?,
    })
}
fn data_bytes<T: ArrowPrimitiveType>(rows: usize) -> Result<u64> {
    let data = Layout::array::<T::Native>(rows).map_err(|_| invalid())?;
    size(data.size())
}
pub(super) fn mask_bytes(rows: usize, nullable: bool) -> Result<usize> {
    if !nullable || rows == 0 {
        return Ok(0);
    }
    // BooleanBufferBuilder uses MutableBuffer::with_capacity: 64-byte rounding.
    let bytes = rows / 8 + usize::from(!rows.is_multiple_of(8));
    Ok(bytes.checked_add(63).ok_or_else(invalid)? & !63)
}
pub(super) fn primitive<T: ArrowPrimitiveType>(
    rows: usize,
    nullable: bool,
    cell: &mut dyn FnMut(usize) -> Option<T::Native>,
    check: Check<'_>,
) -> Result<ArrayRef> {
    check()?;
    let (values, nulls) = build::<T>(rows, nullable, cell, check)?;
    check()?;
    let output = Arc::new(PrimitiveArray::<T>::new(values.into(), nulls.build()));
    check()?;
    Ok(output)
}
fn build<T: ArrowPrimitiveType>(
    rows: usize,
    nullable: bool,
    cell: &mut dyn FnMut(usize) -> Option<T::Native>,
    check: Check<'_>,
) -> Result<(Vec<T::Native>, NullBufferBuilder)> {
    requirement::<T>(rows, nullable)?;
    let mut values = Vec::with_capacity(rows);
    let mut nulls = NullBufferBuilder::new(rows);
    for index in 0..rows {
        checkpoint(index, check)?;
        append(&mut values, &mut nulls, cell(index), nullable)?;
    }
    Ok((values, nulls))
}
fn append<T: Copy + Default>(
    values: &mut Vec<T>,
    nulls: &mut NullBufferBuilder,
    value: Option<T>,
    nullable: bool,
) -> Result<()> {
    if value.is_none() && !nullable {
        return Err(invalid());
    }
    values.push(value.unwrap_or_default());
    nulls.append(value.is_some());
    Ok(())
}
fn checkpoint(index: usize, check: Check<'_>) -> Result<()> {
    if index > 0 && index.is_multiple_of(4_096) {
        check()?;
    }
    Ok(())
}
fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling checkpoint control column size or validity is invalid".into(),
    }
}

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
