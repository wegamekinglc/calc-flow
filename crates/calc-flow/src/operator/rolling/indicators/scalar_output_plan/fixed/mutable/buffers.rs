use super::{
    Check, Result, Sources, invalid,
    math::{add, align, bitmap, mul, physical_width, scale, size, sum},
};
use datafusion::arrow::datatypes::DataType;

#[derive(Clone, Copy, Default)]
pub(in crate::operator::rolling::indicators::scalar_output_plan) struct Growing {
    pub(in crate::operator::rolling::indicators::scalar_output_plan) capacity: u64,
    pub(in crate::operator::rolling::indicators::scalar_output_plan) len: u64,
    pub(in crate::operator::rolling::indicators::scalar_output_plan) old: u64,
}
impl Growing {
    pub(in crate::operator::rolling::indicators::scalar_output_plan) fn new(
        initial: u64,
        len: u64,
    ) -> Result<Self> {
        let capacity = layout(align(initial)?)?;
        Ok(Self {
            capacity,
            len,
            old: 0,
        })
    }
    pub(super) fn mask(capacity: u64) -> Result<Self> {
        Ok(Self {
            capacity: layout(add(capacity, 7)? / 8)?,
            len: 0,
            old: 0,
        })
    }
    pub(in crate::operator::rolling::indicators::scalar_output_plan) fn append(
        &mut self,
        bytes: u64,
        extra: u64,
    ) -> Result<()> {
        let end = add(self.len, bytes)?;
        let required = add(end, extra)?;
        if required > self.capacity {
            self.old = self.old.max(self.capacity);
            self.capacity = layout(align(required)?.max(mul(self.capacity, 2)?))?;
        }
        self.len = end;
        Ok(())
    }
}
pub(super) fn layout(bytes: u64) -> Result<u64> {
    let size = usize::try_from(bytes).map_err(|_| invalid("mutable buffer layout"))?;
    std::alloc::Layout::from_size_align(size, datafusion::arrow::alloc::ALIGNMENT)
        .map_err(|_| invalid("mutable buffer layout"))?;
    Ok(bytes)
}
#[derive(Clone, Copy, Default)]
pub(super) struct Buffers {
    pub(super) output: u64,
    pub(super) scratch: u64,
    pub(super) count: u64,
}
#[derive(Clone, Copy)]
struct Shape {
    width: u64,
    offset: bool,
    second: bool,
    bits: bool,
}
fn shape(dtype: &DataType) -> Result<Shape> {
    if let DataType::Dictionary(key, _) = dtype {
        return shape(key);
    }
    if let Some(width) = physical_width(dtype)? {
        return Ok(Shape {
            width,
            offset: false,
            second: false,
            bits: false,
        });
    }
    composite_shape(dtype)
}
fn composite_shape(dtype: &DataType) -> Result<Shape> {
    match dtype {
        DataType::Boolean => Ok(Shape {
            width: 0,
            offset: false,
            second: false,
            bits: true,
        }),
        DataType::List(_) | DataType::Map(..) => Ok(Shape {
            width: 4,
            offset: true,
            second: false,
            bits: false,
        }),
        DataType::LargeList(_) => Ok(Shape {
            width: 8,
            offset: true,
            second: false,
            bits: false,
        }),
        _ => other_shape(dtype),
    }
}
fn other_shape(dtype: &DataType) -> Result<Shape> {
    match dtype {
        DataType::ListView(_) => Ok(Shape {
            width: 4,
            offset: false,
            second: true,
            bits: false,
        }),
        DataType::LargeListView(_) => Ok(Shape {
            width: 8,
            offset: false,
            second: true,
            bits: false,
        }),
        DataType::RunEndEncoded(..)
        | DataType::Null
        | DataType::Struct(_)
        | DataType::FixedSizeList(..) => Ok(Shape {
            width: 0,
            offset: false,
            second: false,
            bits: false,
        }),
        _ => Err(invalid("mutable buffer datatype")),
    }
}

pub(super) fn plan(
    dtype: &DataType,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Buffers> {
    if let DataType::Union(_, mode) = dtype {
        return super::union::buffers(*mode, capacity, sources, check);
    }
    if let Some(cost) = super::bytes::plan(dtype, capacity, sources, check)? {
        return Ok(cost);
    }
    if let Some(cost) = super::views::plan(dtype, capacity, sources, check)? {
        return Ok(cost);
    }
    let shape = shape(dtype)?;
    scan(shape, capacity, sources, check)
}
struct Cursor {
    first: Growing,
    mask: Growing,
    rows: u64,
    shape: Shape,
}
impl Cursor {
    fn new(shape: Shape, capacity: u64) -> Result<Self> {
        let initial = initial(shape, capacity)?;
        let len = if shape.offset { shape.width } else { 0 };
        Ok(Self {
            first: Growing::new(initial, len)?,
            mask: Growing::mask(capacity)?,
            rows: 0,
            shape,
        })
    }
    fn push(&mut self, len: usize, check: Check<'_>) -> Result<()> {
        check()?;
        self.rows = add(self.rows, size(len)?)?;
        extend(&mut self.first, self.shape, len, self.rows, check)?;
        let end = add(self.rows, 7)? / 8;
        self.mask.append(end - self.mask.len, 0)
    }
}
fn scan(
    shape: Shape,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Buffers> {
    let mut cursor = Cursor::new(shape, capacity)?;
    for index in 0..sources.count() {
        check()?;
        sources.visit(
            index,
            &mut |piece, check| cursor.push(piece.len, check),
            check,
        )?;
    }
    finish(cursor.first, cursor.mask, cursor.rows, shape)
}

fn initial(shape: Shape, capacity: u64) -> Result<u64> {
    if shape.bits {
        return Ok(add(capacity, 7)? / 8);
    }
    mul(add(capacity, u64::from(shape.offset))?, shape.width)
}
fn extend(
    buffer: &mut Growing,
    shape: Shape,
    len: usize,
    rows: u64,
    check: Check<'_>,
) -> Result<()> {
    if shape.second {
        return repeated(buffer, shape.width, len, check);
    }
    if shape.bits {
        return buffer.append(add(rows, 7)? / 8 - buffer.len, 0);
    }
    let bytes = mul(size(len)?, shape.width)?;
    buffer.append(bytes, if shape.offset { shape.width } else { 0 })
}
fn finish(first: Growing, mask: Growing, rows: u64, shape: Shape) -> Result<Buffers> {
    let copies = if shape.second { 2 } else { 1 };
    let data = mul(first.capacity, copies)?;
    let output = add(data, mask.capacity)?;
    Ok(Buffers {
        output,
        scratch: sum([
            Ok(output),
            mul(first.old, copies),
            Ok(mask.old),
            scale(bitmap(rows), 2),
        ])?,
        count: add(copies, 1)?,
    })
}

fn repeated(buffer: &mut Growing, width: u64, len: usize, check: Check<'_>) -> Result<()> {
    for _ in 0..len {
        check()?;
        buffer.append(width, 0)?;
    }
    Ok(())
}
