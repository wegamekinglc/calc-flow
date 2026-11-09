use super::{
    Check, Cost, Result, Sources, invalid,
    math::{add, buffer_headers, bytes, grow, mul, size, sum},
    node,
};
use super::{
    buffers::Growing,
    sources::{Child, Kind},
};
use datafusion::arrow::datatypes::DataType;

pub(super) fn plan(
    ends: &DataType,
    dtype: &DataType,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let width = width(ends)?;
    validate(ends, sources, check)?;
    let mut child = Child {
        parent: sources,
        kind: Kind::Run,
    };
    let ends = end_storage(width, capacity, &mut child, check)?;
    let values = node(dtype, capacity, &mut child, check)?;
    values.add(ends)
}
fn width(dtype: &DataType) -> Result<u64> {
    match dtype {
        DataType::Int16 => Ok(2),
        DataType::Int32 => Ok(4),
        DataType::Int64 => Ok(8),
        _ => Err(invalid("mutable run end type")),
    }
}
fn validate(dtype: &DataType, sources: &mut dyn Sources<'_>, check: Check<'_>) -> Result<()> {
    let mut rows = 0;
    for index in 0..sources.count() {
        check()?;
        sources.visit(
            index,
            &mut |piece, check| {
                check()?;
                rows = add(rows, size(piece.len)?)?;
                super::super::super::producer::key_index(dtype, rows)
            },
            check,
        )?;
    }
    Ok(())
}
fn end_storage(
    width: u64,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let mut cursor = Ends {
        data: Growing::new(mul(capacity, width)?, 0)?,
        workspace: 0,
        width,
    };
    for index in 0..sources.count() {
        check()?;
        sources.visit(
            index,
            &mut |piece, check| cursor.push(piece.len, check),
            check,
        )?;
    }
    cursor.finish()
}
struct Ends {
    data: Growing,
    workspace: u64,
    width: u64,
}
impl Ends {
    fn push(&mut self, len: usize, check: Check<'_>) -> Result<()> {
        check()?;
        let bytes = mul(size(len)?, self.width)?;
        self.data.append(bytes, 0)?;
        self.workspace = self.workspace.max(temporary(bytes)?);
        Ok(())
    }
    fn finish(self) -> Result<Cost> {
        Ok(Cost {
            output: self.data.capacity,
            scratch: sum([
                Ok(self.data.capacity),
                Ok(self.data.old),
                Ok(self.workspace),
                buffer_headers(1),
            ])?,
            descriptors: bytes::<datafusion::arrow::array::MutableArrayData<'_>>(2)?,
            builders: 1,
        })
    }
}
fn temporary(bytes: u64) -> Result<u64> {
    if bytes == 0 {
        return Ok(0);
    }
    let capacity = bytes
        .max(8)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("mutable run vector"))?;
    let capacity = usize::try_from(capacity).map_err(|_| invalid("mutable run vector"))?;
    std::alloc::Layout::array::<u8>(capacity).map_err(|_| invalid("mutable run vector layout"))?;
    grow(bytes, 1)
}
