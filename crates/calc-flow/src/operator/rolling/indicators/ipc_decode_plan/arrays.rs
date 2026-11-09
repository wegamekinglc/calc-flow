use arrow_data::{ArrayData, BufferSpec};
use datafusion::arrow::{
    array::{self, ArrayRef},
    buffer::{Buffer, NullBuffer},
    datatypes::{DataType, Float64Type, Int32Type, UnionMode},
};

use super::{
    Result,
    math::{add, arc, bitmap, bytes, grow, hash, mul, scale, size, slice_arc, sum},
    schema::clone_type,
};

#[derive(Clone, Copy, Default)]
pub(super) struct Graph {
    pub(super) retained: u64,
    pub(super) data: u64,
    pub(super) rebuild: u64,
    pub(super) work: u64,
    pub(super) logical: u64,
    pub(super) layouts: u64,
    pub(super) count: u64,
}
impl Graph {
    pub(super) fn merge(&mut self, other: Self) -> Result<()> {
        self.merge_storage(other)?;
        self.rebuild = add(self.rebuild, other.rebuild)?;
        self.work = add(self.work, other.work)?;
        self.logical = add(self.logical, other.logical)?;
        self.count = add(self.count, 1)?;
        Ok(())
    }
    fn merge_storage(&mut self, other: Self) -> Result<()> {
        self.retained = add(self.retained, other.retained)?;
        self.data = add(self.data, other.data)?;
        self.layouts = add(self.layouts, other.layouts)?;
        Ok(())
    }
}

struct Headers {
    retained: u64,
    data: u64,
    layouts: u64,
    validation: u64,
}
fn headers(dtype: &DataType, len: u64, buffers: u64, child: Graph) -> Result<Headers> {
    let retained = sum([
        shell(dtype),
        slots(dtype, buffers, child.count),
        clone_type(dtype),
    ])?;
    let data = data_storage(dtype, buffers, child)?;
    let layouts = sum([bytes::<BufferSpec>(buffers), Ok(child.layouts)])?;
    let validation = sum([
        scale(Ok(layouts), 2),
        validation_extra(dtype, len, child.logical),
    ])?;
    Ok(Headers {
        retained,
        data,
        layouts,
        validation,
    })
}
fn data_storage(dtype: &DataType, buffers: u64, child: Graph) -> Result<u64> {
    sum([
        clone_type(dtype),
        grow::<Buffer>(buffers),
        grow::<ArrayData>(child.count),
        Ok(child.data),
        slots(dtype, buffers, child.count),
    ])
}

pub(super) fn build(dtype: &DataType, len: u64, buffers: u64, child: Graph) -> Result<Graph> {
    let header = headers(dtype, len, buffers, child)?;
    let rebuild = sum([
        Ok(header.retained),
        Ok(child.rebuild),
        Ok(child.data),
        Ok(header.validation),
    ])?;
    let work = decode_work(dtype, child, &header, buffers)?;
    Ok(Graph {
        retained: add(header.retained, child.retained)?,
        data: header.data,
        rebuild,
        work,
        logical: logical(dtype, len, child)?,
        layouts: header.layouts,
        count: 1,
    })
}

fn view_work(dtype: &DataType, buffers: u64) -> Result<u64> {
    if !matches!(dtype, DataType::BinaryView | DataType::Utf8View) {
        return Ok(0);
    }
    // Eager ok_or(format!) and gathered validity + views + variadic buffer handles.
    sum([grow::<Buffer>(add(buffers, 1)?), grow::<u8>(64)])
}
fn decode_work(dtype: &DataType, child: Graph, header: &Headers, buffers: u64) -> Result<u64> {
    let reconstruct = match dtype {
        DataType::Struct(_) => child.logical,
        _ => add(child.data, child.rebuild)?,
    };
    sum([
        Ok(child.work),
        Ok(header.data),
        Ok(header.retained),
        Ok(header.validation),
        Ok(reconstruct),
        view_work(dtype, buffers),
    ])
}
fn validation_extra(dtype: &DataType, len: u64, logical: u64) -> Result<u64> {
    match dtype {
        DataType::FixedSizeList(_, width) => expanded_bitmap(len, *width),
        DataType::Union(_, _) => bytes::<Option<usize>>(128),
        DataType::Struct(_) => Ok(logical),
        _ => Ok(0),
    }
}
fn expanded_bitmap(len: u64, width: i32) -> Result<u64> {
    let width = u64::try_from(width).map_err(|_| super::invalid("fixed list length"))?;
    bitmap(mul(len, width)?)
}
fn logical(dtype: &DataType, len: u64, child: Graph) -> Result<u64> {
    match dtype {
        DataType::Null => bitmap(len),
        DataType::Dictionary(_, _) | DataType::RunEndEncoded(_, _) => {
            add(child.logical, bitmap(len)?)
        }
        DataType::Union(fields, _) => union_nulls(size(fields.len())?, len, child.logical),
        _ => Ok(0),
    }
}
fn union_nulls(fields: u64, len: u64, children: u64) -> Result<u64> {
    // Union logical_nulls: null tuples, two hash sets, two bit-chunk vectors.
    sum([
        Ok(children),
        grow::<(i8, NullBuffer)>(fields),
        scale(hash::<i8>(fields), 2),
        grow::<i8>(fields),
        scale(grow::<(i8, [usize; 6])>(fields), 2),
        scale(bitmap(len), 3),
        scale(bitmap(1), 2),
    ])
}
fn slots(dtype: &DataType, buffers: u64, children: u64) -> Result<u64> {
    match dtype {
        DataType::Struct(_) | DataType::Map(_, _) => grow::<ArrayData>(children),
        DataType::Union(fields, _) => {
            let max = fields.iter().map(|(id, _)| id).max().unwrap_or_default();
            bytes::<Option<ArrayRef>>(add(
                u64::try_from(max).map_err(|_| super::invalid("union id"))?,
                1,
            )?)
        }
        DataType::BinaryView | DataType::Utf8View => slice_arc::<Buffer>(buffers),
        _ => Ok(0),
    }
}
fn shell(dtype: &DataType) -> Result<u64> {
    use DataType as D;
    match dtype {
        D::Struct(_) => arc::<array::StructArray>(),
        D::Map(_, _) => arc::<array::MapArray>(),
        D::Union(_, _) => arc::<array::UnionArray>(),
        D::Dictionary(_, _) => arc::<array::DictionaryArray<Int32Type>>(),
        D::RunEndEncoded(_, _) => arc::<array::RunArray<Int32Type>>(),
        dtype => list_shell(dtype),
    }
}
fn list_shell(dtype: &DataType) -> Result<u64> {
    use DataType as D;
    match dtype {
        D::List(_) | D::LargeList(_) => arc::<array::LargeListArray>(),
        D::ListView(_) | D::LargeListView(_) => arc::<array::LargeListViewArray>(),
        D::FixedSizeList(_, _) => arc::<array::FixedSizeListArray>(),
        dtype => leaf_shell(dtype),
    }
}
fn leaf_shell(dtype: &DataType) -> Result<u64> {
    use DataType as D;
    match dtype {
        D::Null => arc::<array::NullArray>(),
        D::Boolean => arc::<array::BooleanArray>(),
        D::Binary | D::Utf8 | D::LargeBinary | D::LargeUtf8 => arc::<array::LargeBinaryArray>(),
        D::BinaryView | D::Utf8View => arc::<array::BinaryViewArray>(),
        D::FixedSizeBinary(_) => arc::<array::FixedSizeBinaryArray>(),
        _ => arc::<array::PrimitiveArray<Float64Type>>(),
    }
}
pub(super) fn buffers(dtype: &DataType) -> u64 {
    use DataType as D;
    match dtype {
        D::Null | D::Struct(_) | D::FixedSizeList(_, _) | D::RunEndEncoded(_, _) => 0,
        D::Binary
        | D::Utf8
        | D::LargeBinary
        | D::LargeUtf8
        | D::ListView(_)
        | D::LargeListView(_)
        | D::Union(_, UnionMode::Dense) => 2,
        _ => 1,
    }
}
