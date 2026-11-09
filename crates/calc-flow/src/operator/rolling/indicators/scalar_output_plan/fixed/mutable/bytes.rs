use super::buffers::{Buffers, Growing};
use super::{
    Check, Result, Sources, invalid,
    math::{add, bitmap, mul, scale, size, sum},
};
use datafusion::arrow::{
    array::GenericByteArray,
    datatypes::{
        ArrowNativeType, BinaryType, ByteArrayType, DataType, LargeBinaryType, LargeUtf8Type,
        Utf8Type,
    },
};
pub(super) fn plan(
    dtype: &DataType,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Option<Buffers>> {
    let cost = match dtype {
        DataType::Utf8 => scan::<Utf8Type>(capacity, sources, check),
        DataType::Binary => scan::<BinaryType>(capacity, sources, check),
        DataType::LargeUtf8 => scan::<LargeUtf8Type>(capacity, sources, check),
        DataType::LargeBinary => scan::<LargeBinaryType>(capacity, sources, check),
        _ => return Ok(None),
    };
    cost.map(Some)
}
fn scan<T: ByteArrayType>(
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Buffers> {
    let mut cursor = Cursor::new::<T>(capacity)?;
    for index in 0..sources.count() {
        check()?;
        cursor.input::<T>(index, sources, check)?;
    }
    result(cursor.offsets, cursor.data, cursor.mask, cursor.rows)
}
struct Cursor {
    offsets: Growing,
    data: Growing,
    mask: Growing,
    rows: u64,
    width: u64,
}
impl Cursor {
    fn new<T: ByteArrayType>(capacity: u64) -> Result<Self> {
        let width = size(size_of::<T::Offset>())?;
        let initial = mul(add(capacity, 1)?, width)?;
        Ok(Self {
            offsets: Growing::new(initial, width)?,
            data: Growing::new(capacity, 0)?,
            mask: Growing::mask(capacity)?,
            rows: 0,
            width,
        })
    }
    fn input<T: ByteArrayType>(
        &mut self,
        index: usize,
        sources: &mut dyn Sources<'_>,
        check: Check<'_>,
    ) -> Result<()> {
        if let Some(value) = sources.scalar(index)? {
            return self.scalar::<T>(value);
        }
        sources.visit(
            index,
            &mut |piece, check| self.push::<T>(piece, check),
            check,
        )
    }
    fn scalar<T: ByteArrayType>(&mut self, value: &datafusion::common::ScalarValue) -> Result<()> {
        self.rows = add(self.rows, 1)?;
        self.offsets.append(self.width, self.width)?;
        self.data
            .append(super::super::super::leaf::byte_length(value)?, 0)?;
        representable::<T>(self.data.len)?;
        let end = add(self.rows, 7)? / 8;
        self.mask.append(end - self.mask.len, 0)
    }
    fn push<T: ByteArrayType>(
        &mut self,
        piece: super::super::super::concatenated::Piece<'_>,
        check: Check<'_>,
    ) -> Result<()> {
        check()?;
        self.rows = add(self.rows, size(piece.len)?)?;
        self.offsets
            .append(mul(size(piece.len)?, self.width)?, self.width)?;
        self.extend_data::<T>(piece)
    }
    fn extend_data<T: ByteArrayType>(
        &mut self,
        piece: super::super::super::concatenated::Piece<'_>,
    ) -> Result<()> {
        self.data.append(length::<T>(piece)?, 0)?;
        representable::<T>(self.data.len)?;
        let end = add(self.rows, 7)? / 8;
        self.mask.append(end - self.mask.len, 0)
    }
}

fn length<T: ByteArrayType>(piece: super::super::super::concatenated::Piece<'_>) -> Result<u64> {
    let Some(array) = piece.array else {
        return Ok(0);
    };
    let array = array
        .as_any()
        .downcast_ref::<GenericByteArray<T>>()
        .ok_or_else(|| invalid("mutable byte array"))?;
    let offsets = array.value_offsets();
    let start = offsets[piece.start].as_usize();
    let end = offsets[piece.start + piece.len].as_usize();
    size(
        end.checked_sub(start)
            .ok_or_else(|| invalid("mutable byte range"))?,
    )
}
fn representable<T: ByteArrayType>(bytes: u64) -> Result<()> {
    let bytes = usize::try_from(bytes).map_err(|_| invalid("mutable byte offset"))?;
    T::Offset::from_usize(bytes).ok_or_else(|| invalid("mutable byte offset"))?;
    Ok(())
}
fn result(offsets: Growing, data: Growing, mask: Growing, rows: u64) -> Result<Buffers> {
    let output = sum([Ok(offsets.capacity), Ok(data.capacity), Ok(mask.capacity)])?;
    Ok(Buffers {
        output,
        scratch: sum([
            Ok(output),
            Ok(offsets.old),
            Ok(data.old),
            Ok(mask.old),
            scale(bitmap(rows), 2),
        ])?,
        count: 3,
    })
}
