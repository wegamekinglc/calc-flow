use std::{alloc::Layout, sync::Arc};

use datafusion::arrow::{
    array::{ArrayRef, BinaryArray, builder::NullBufferBuilder},
    buffer::{Buffer, OffsetBuffer},
};

use super::{
    control::{Cost, mask_bytes},
    schema::{arc, size, total},
};
use crate::{CalcFlowError, Result};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub(super) rows: usize,
    pub(super) values: usize,
    pub(super) nullable: bool,
    pub(super) payload_peak: u64,
}
pub(super) fn requirement(shape: Shape) -> Result<Cost> {
    let retained = total([
        arc::<BinaryArray>(),
        offsets(shape.rows),
        value_bytes(shape.values),
        size(mask_bytes(shape.rows, shape.nullable)?),
    ])?;
    let buffers = 2 + usize::from(shape.nullable && shape.rows > 0);
    let headers = buffers * 7 * size_of::<usize>();
    let construction_peak = total([Ok(retained), size(headers), Ok(shape.payload_peak)])?;
    Ok(Cost {
        retained,
        construction_peak,
    })
}
fn offsets(rows: usize) -> Result<u64> {
    let length = rows.checked_add(1).ok_or_else(invalid)?;
    let layout = Layout::array::<i32>(length).map_err(|_| invalid())?;
    size(layout.size())
}
fn value_bytes(length: usize) -> Result<u64> {
    i32::try_from(length).map_err(|_| invalid())?;
    size(length)
}
pub(super) fn column(
    shape: Shape,
    cell: &mut dyn FnMut(usize) -> Result<Option<Vec<u8>>>,
    check: Check<'_>,
) -> Result<ArrayRef> {
    check()?;
    requirement(shape)?;
    let mut builder = Builder::new(shape);
    builder.fill(cell, check)?;
    check()?;
    let output = builder.finish()?;
    check()?;
    Ok(output)
}
struct Builder {
    shape: Shape,
    offsets: Vec<i32>,
    values: Vec<u8>,
    nulls: NullBufferBuilder,
}
impl Builder {
    fn new(shape: Shape) -> Self {
        let mut offsets = Vec::with_capacity(shape.rows + 1);
        offsets.push(0);
        Self {
            shape,
            offsets,
            values: Vec::with_capacity(shape.values),
            nulls: NullBufferBuilder::new(shape.rows),
        }
    }
    fn fill(
        &mut self,
        cell: &mut dyn FnMut(usize) -> Result<Option<Vec<u8>>>,
        check: Check<'_>,
    ) -> Result<()> {
        for row in 0..self.shape.rows {
            if row.is_multiple_of(4_096) {
                check()?;
            }
            self.append(cell(row)?)?;
        }
        Ok(())
    }
    fn append(&mut self, value: Option<Vec<u8>>) -> Result<()> {
        if let Some(value) = value {
            self.append_value(&value)?;
            self.nulls.append_non_null();
        } else {
            if !self.shape.nullable {
                return Err(invalid());
            }
            self.nulls.append_null();
        }
        self.offsets
            .push(i32::try_from(self.values.len()).map_err(|_| invalid())?);
        Ok(())
    }
    fn append_value(&mut self, value: &Vec<u8>) -> Result<()> {
        let total = self
            .values
            .len()
            .checked_add(value.len())
            .ok_or_else(invalid)?;
        if total > self.shape.values || size(value.capacity())? > self.shape.payload_peak {
            return Err(invalid());
        }
        self.values.extend_from_slice(value);
        Ok(())
    }
    fn finish(self) -> Result<ArrayRef> {
        if self.values.len() != self.shape.values {
            return Err(invalid());
        }
        let array = BinaryArray::try_new(
            OffsetBuffer::new(self.offsets.into()),
            Buffer::from(self.values),
            self.nulls.build(),
        )
        .map_err(|_| invalid())?;
        Ok(Arc::new(array))
    }
}
fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling checkpoint binary column size or validity is invalid".into(),
    }
}

#[cfg(test)]
#[path = "binary_tests.rs"]
mod tests;
