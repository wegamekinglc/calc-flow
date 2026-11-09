use super::super::math::{
    add, align, arc_slice, bitmap, buffer_headers, grow, mul, scale, size, sum,
};
use super::{At, Check, Cost, Result, invalid, typed};
use datafusion::arrow::{
    array::{GenericByteArray, GenericByteViewArray},
    buffer::Buffer,
    datatypes::{
        BinaryType, BinaryViewType, ByteArrayType, ByteViewType, DataType, LargeBinaryType,
        LargeUtf8Type, StringViewType, Utf8Type,
    },
};

pub(super) fn plan(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Option<Cost>> {
    match dtype {
        DataType::Null => buffer(Ok(0)),
        DataType::Boolean => buffer(scale(bitmap(rows), 2)),
        DataType::Utf8 => buffer(byte_data::<Utf8Type>(count, rows, at, check)),
        DataType::LargeUtf8 => buffer(byte_data::<LargeUtf8Type>(count, rows, at, check)),
        _ => other(dtype, count, rows, at, check),
    }
}
fn buffer(bytes: Result<u64>) -> Result<Option<Cost>> {
    bytes.map(|owned| {
        Some(Cost {
            owned,
            temporary: owned,
        })
    })
}

fn other(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Option<Cost>> {
    let bytes = match dtype {
        DataType::Binary => byte_data::<BinaryType>(count, rows, at, check)?,
        DataType::LargeBinary => byte_data::<LargeBinaryType>(count, rows, at, check)?,
        DataType::Utf8View => return view_data::<StringViewType>(count, rows, at, check).map(Some),
        DataType::BinaryView => {
            return view_data::<BinaryViewType>(count, rows, at, check).map(Some);
        }
        _ => return numeric(dtype, rows),
    };
    Ok(Some(Cost {
        owned: bytes,
        temporary: bytes,
    }))
}
fn numeric(dtype: &DataType, rows: u64) -> Result<Option<Cost>> {
    let width = if let DataType::FixedSizeBinary(width) = dtype {
        usize::try_from(*width).map_err(|_| invalid("fixed binary width"))?
    } else if let Some(width) = dtype.primitive_width() {
        width
    } else {
        return Ok(None);
    };
    let bytes = align(mul(rows, size(width)?)?)?;
    Ok(Some(Cost {
        owned: bytes,
        temporary: bytes,
    }))
}
fn byte_data<T: ByteArrayType>(
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<u64> {
    let mut bytes = 0;
    for index in 0..count {
        check()?;
        bytes = add(bytes, byte_length::<T>(at(index)?)?)?;
    }
    sum([Ok(bytes), offset_bytes::<T>(rows)])
}
fn offset_bytes<T: ByteArrayType>(rows: u64) -> Result<u64> {
    mul(add(rows, 1)?, size(size_of::<T::Offset>())?)
}
fn byte_length<T: ByteArrayType>(piece: super::Piece<'_>) -> Result<u64> {
    let Some(array) = piece.array else {
        return Ok(0);
    };
    let array = typed::<GenericByteArray<T>>(array)?;
    let offsets = array.value_offsets();
    let start = offsets[piece.start].as_usize();
    let end = offsets[piece.start + piece.len].as_usize();
    size(
        end.checked_sub(start)
            .ok_or_else(|| invalid("byte range"))?,
    )
}
fn view_data<T: ByteViewType>(
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    super::super::normal::copied_views::<T>(count, at, check)?;
    let mut blocks = 0_usize;
    for index in 0..count {
        check()?;
        blocks = block_count::<T>(at(index)?, blocks)?;
    }
    view_buffers(rows, blocks)
}
fn block_count<T: ByteViewType>(piece: super::Piece<'_>, count: usize) -> Result<usize> {
    let Some(array) = piece.array else {
        return Ok(count);
    };
    let array = typed::<GenericByteViewArray<T>>(array)?;
    count
        .checked_add(array.data_buffers().len())
        .ok_or_else(|| invalid("view buffers"))
}
fn view_buffers(rows: u64, blocks: usize) -> Result<Cost> {
    let data = sum([mul(rows, 16), arc_slice::<Buffer>(blocks)])?;
    Ok(Cost {
        owned: data,
        temporary: sum([
            Ok(data),
            grow(size(blocks)?, size(size_of::<Buffer>())?),
            buffer_headers(1),
        ])?,
    })
}

use datafusion::arrow::datatypes::ArrowNativeType;

#[cfg(test)]
mod tests {
    #[test]
    fn test_normal_view_construction_prices_completed_vec_minimum_capacity() {
        use datafusion::arrow::{
            array::{StringViewArray, builder::StringViewBuilder},
            buffer::Buffer,
        };
        let source = StringViewArray::new(
            Vec::<u128>::new().into(),
            vec![Buffer::from_vec(vec![1_u8; 16])],
            None,
        );
        let quote = super::view_buffers(0, 2).unwrap();
        let actual = allocation_counter::measure(|| {
            let mut builder = StringViewBuilder::with_capacity(0);
            builder.append_array(&source);
            builder.append_array(&source);
            std::hint::black_box(builder.finish());
        });
        assert!(
            actual.bytes_max <= quote.temporary,
            "{actual:?} > component {}",
            quote.temporary
        );
    }
}
