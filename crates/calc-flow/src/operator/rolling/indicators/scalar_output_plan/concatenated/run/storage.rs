use super::super::super::fixed::Growing;
use super::super::super::{
    math::{add, buffer_headers, grow, size, sum, vector},
    single, workspace,
};
use super::{At, Check, Cost, Piece, Result};
use datafusion::arrow::array::ArrayRef;

pub(super) fn plan(width: u64, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<Cost> {
    let mut cursor = Cursor {
        data: Growing::new(0, 0)?,
        metadata: 0,
        descriptors: 0,
        width,
        count,
    };
    for index in 0..count {
        check()?;
        cursor.push(at(index)?, check)?;
    }
    cursor.finish()
}
struct Cursor {
    data: Growing,
    metadata: u64,
    descriptors: u64,
    width: u64,
    count: usize,
}
impl Cursor {
    fn push(&mut self, piece: Piece<'_>, check: Check<'_>) -> Result<()> {
        extend(&mut self.data, self.width, piece.len, check)?;
        self.metadata = add(self.metadata, conversion(piece, check)?)?;
        if self.count == 1 {
            self.descriptors = shared_conversion(piece, check)?;
        }
        Ok(())
    }
    fn finish(self) -> Result<Cost> {
        Ok(Cost {
            owned: add(self.data.capacity, self.descriptors)?,
            temporary: scratch(self.count, self.width, self.data, self.metadata)?,
        })
    }
}

fn extend(data: &mut Growing, width: u64, len: usize, check: Check<'_>) -> Result<()> {
    for _ in 0..len {
        check()?;
        data.append(width, 0)?;
    }
    Ok(())
}
fn scratch(count: usize, width: u64, data: Growing, metadata: u64) -> Result<u64> {
    sum([
        Ok(data.capacity),
        Ok(data.old),
        Ok(metadata),
        vectors(count, width),
        buffer_headers(1),
    ])
}
fn vectors(count: usize, width: u64) -> Result<u64> {
    sum([
        vector::<usize>(count),
        vector::<ArrayRef>(count),
        vector::<&dyn datafusion::arrow::array::Array>(count),
        grow(add(size(count)?, 1)?, width),
    ])
}
fn conversion(piece: Piece<'_>, check: Check<'_>) -> Result<u64> {
    let Some(array) = piece.array else {
        return Ok(0);
    };
    add(
        workspace::array_data_workspace(array, check)?,
        single::sliced(array, check)?,
    )
}
fn shared_conversion(piece: Piece<'_>, check: Check<'_>) -> Result<u64> {
    match piece.array {
        Some(array) => workspace::descriptor_tree(array, check),
        None => Ok(0),
    }
}
