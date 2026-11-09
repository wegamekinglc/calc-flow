use super::{
    Check,
    math::{add, elements, invalid, mul, size, sum, vec_slots},
};
use crate::Result;
use datafusion::arrow::{
    array::{
        Array, ArrayData, ArrayRef, BinaryViewArray, DictionaryArray, FixedSizeListArray,
        LargeListArray, LargeListViewArray, ListArray, ListViewArray, MapArray, RunArray,
        StringViewArray, StructArray, UnionArray,
    },
    buffer::Buffer,
    datatypes::{
        ArrowDictionaryKeyType, DataType, Int8Type, Int16Type, Int32Type, Int64Type,
        RunEndIndexType, UInt8Type, UInt16Type, UInt32Type, UInt64Type, UnionFields, UnionMode,
    },
};
use std::mem::size_of_val;

#[derive(Clone, Copy, Default)]
pub(super) struct Counts {
    pub(super) metadata: u64,
    shells: u64,
    pub(super) writer: u64,
}
impl Counts {
    fn merge(self, other: Self) -> Result<Self> {
        Ok(Self {
            metadata: add(self.metadata, other.metadata)?,
            shells: add(self.shells, other.shells)?,
            writer: add(self.writer, other.writer)?,
        })
    }
    fn node(value: &dyn Array, parts: (usize, usize, Self)) -> Result<Self> {
        let (buffers, children, nested) = parts;
        let metadata = sum([
            own_metadata(value.data_type(), buffers, children),
            Ok(nested.metadata),
        ])?;
        let shells = sum([own_shell(value, buffers, children), Ok(nested.shells)])?;
        let copies = extra_copies(value.data_type(), (metadata, shells, nested.metadata))?;
        Ok(Self {
            metadata,
            shells,
            writer: add(nested.writer, copies)?,
        })
    }
}
pub(super) fn array(value: &dyn Array, check: Check<'_>) -> Result<Counts> {
    check()?;
    Counts::node(value, shape(value, check)?)
}
fn own_metadata(dt: &DataType, buffers: usize, children: usize) -> Result<u64> {
    sum([
        vec_slots::<Buffer>(buffers),
        vec_slots::<ArrayData>(children),
        type_boxes(dt),
        to_data_header(dt, children),
    ])
}
fn to_data_header(dt: &DataType, children: usize) -> Result<u64> {
    match dt {
        DataType::Struct(_) => elements::<ArrayRef>(size(children)?),
        DataType::Union(_, _) => elements::<Option<ArrayRef>>(size(child_slots(dt, children)?)?),
        _ => Ok(0),
    }
}
fn own_shell(value: &dyn Array, buffers: usize, children: usize) -> Result<u64> {
    let refs = child_slots(value.data_type(), children)?;
    sum([
        size(size_of_val(value)),
        elements::<usize>(2),
        vec_slots::<ArrayRef>(refs),
        vec_slots::<Buffer>(buffers),
    ])
}
fn child_slots(dt: &DataType, children: usize) -> Result<usize> {
    if let DataType::Union(fields, _) = dt {
        return fields.iter().try_fold(0, |highest, (id, _)| {
            let next = usize::try_from(id)
                .map_err(|_| invalid())?
                .checked_add(1)
                .ok_or_else(invalid)?;
            Ok(highest.max(next))
        });
    }
    Ok(children)
}
fn extra_copies(dt: &DataType, parts: (u64, u64, u64)) -> Result<u64> {
    let (metadata, shells, child_metadata) = parts;
    let whole = add(metadata, shells)?;
    match dt {
        DataType::Dictionary(_, _) => mul(whole, 3),
        DataType::RunEndEncoded(_, _) => mul(whole, 4),
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::FixedSizeList(_, _)
        | DataType::Map(_, _) => Ok(child_metadata),
        _ => Ok(0),
    }
}
fn type_boxes(dt: &DataType) -> Result<u64> {
    match dt {
        DataType::Dictionary(key, value) => {
            sum([elements::<DataType>(2), type_boxes(key), type_boxes(value)])
        }
        _ => Ok(0),
    }
}
fn shape(value: &dyn Array, check: Check<'_>) -> Result<(usize, usize, Counts)> {
    match value.data_type() {
        DataType::Struct(_) => struct_parts(value, check),
        DataType::Union(fields, mode) => union_parts(value, fields, *mode, check),
        DataType::Map(_, _) => {
            typed::<MapArray>(value).and_then(|map| child_parts(map.entries(), 1, check))
        }
        DataType::Dictionary(key, _) => dictionary(value, key, check).map(|nested| (1, 1, nested)),
        DataType::RunEndEncoded(key, _) => run(value, key.data_type(), check),
        _ => lists(value, check),
    }
}
fn struct_parts(value: &dyn Array, check: Check<'_>) -> Result<(usize, usize, Counts)> {
    let columns = typed::<StructArray>(value)?.columns();
    let mut nested = Counts::default();
    for child in columns {
        nested = nested.merge(array(child.as_ref(), check)?)?;
    }
    Ok((0, columns.len(), nested))
}
fn union_parts(
    value: &dyn Array,
    fields: &UnionFields,
    mode: UnionMode,
    check: Check<'_>,
) -> Result<(usize, usize, Counts)> {
    let union = typed::<UnionArray>(value)?;
    let mut nested = Counts::default();
    for (id, _) in fields.iter() {
        nested = nested.merge(array(union.child(id).as_ref(), check)?)?;
    }
    Ok((
        if mode == UnionMode::Dense { 2 } else { 1 },
        fields.len(),
        nested,
    ))
}
fn child_parts(
    value: &dyn Array,
    buffers: usize,
    check: Check<'_>,
) -> Result<(usize, usize, Counts)> {
    Ok((buffers, 1, array(value, check)?))
}
fn lists(value: &dyn Array, check: Check<'_>) -> Result<(usize, usize, Counts)> {
    match value.data_type() {
        DataType::List(_) => {
            typed::<ListArray>(value).and_then(|list| child_parts(list.values().as_ref(), 1, check))
        }
        DataType::LargeList(_) => typed::<LargeListArray>(value)
            .and_then(|list| child_parts(list.values().as_ref(), 1, check)),
        DataType::FixedSizeList(_, _) => typed::<FixedSizeListArray>(value)
            .and_then(|list| child_parts(list.values().as_ref(), 0, check)),
        _ => views(value, check),
    }
}
fn views(value: &dyn Array, check: Check<'_>) -> Result<(usize, usize, Counts)> {
    match value.data_type() {
        DataType::ListView(_) => typed::<ListViewArray>(value)
            .and_then(|list| child_parts(list.values().as_ref(), 2, check)),
        DataType::LargeListView(_) => typed::<LargeListViewArray>(value)
            .and_then(|list| child_parts(list.values().as_ref(), 2, check)),
        _ => leaf(value),
    }
}
fn leaf(value: &dyn Array) -> Result<(usize, usize, Counts)> {
    let buffers = match value.data_type() {
        DataType::Null => 0,
        DataType::Utf8View | DataType::BinaryView => byte_view_buffers(value)?,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => 2,
        _ => 1,
    };
    Ok((buffers, 0, Counts::default()))
}
fn byte_view_buffers(value: &dyn Array) -> Result<usize> {
    let data_buffers = match value.data_type() {
        DataType::Utf8View => typed::<StringViewArray>(value)?.data_buffers().len(),
        _ => typed::<BinaryViewArray>(value)?.data_buffers().len(),
    };
    data_buffers.checked_add(1).ok_or_else(invalid)
}
fn dictionary(value: &dyn Array, key: &DataType, check: Check<'_>) -> Result<Counts> {
    match key {
        DataType::Int8 => dict::<Int8Type>(value, check),
        DataType::Int16 => dict::<Int16Type>(value, check),
        DataType::Int32 => dict::<Int32Type>(value, check),
        DataType::Int64 => dict::<Int64Type>(value, check),
        _ => unsigned(value, key, check),
    }
}
fn unsigned(value: &dyn Array, key: &DataType, check: Check<'_>) -> Result<Counts> {
    match key {
        DataType::UInt8 => dict::<UInt8Type>(value, check),
        DataType::UInt16 => dict::<UInt16Type>(value, check),
        DataType::UInt32 => dict::<UInt32Type>(value, check),
        DataType::UInt64 => dict::<UInt64Type>(value, check),
        _ => Err(invalid()),
    }
}
fn dict<K: ArrowDictionaryKeyType>(value: &dyn Array, check: Check<'_>) -> Result<Counts> {
    array(typed::<DictionaryArray<K>>(value)?.values().as_ref(), check)
}
fn run(value: &dyn Array, key: &DataType, check: Check<'_>) -> Result<(usize, usize, Counts)> {
    let nested = match key {
        DataType::Int16 => run_values::<Int16Type>(value, check),
        DataType::Int32 => run_values::<Int32Type>(value, check),
        DataType::Int64 => run_values::<Int64Type>(value, check),
        _ => Err(invalid()),
    }?;
    run_parts(nested)
}
fn run_parts(nested: Counts) -> Result<(usize, usize, Counts)> {
    Ok((
        0,
        2,
        Counts {
            metadata: sum([Ok(nested.metadata), vec_slots::<Buffer>(1)])?,
            ..nested
        },
    ))
}
fn run_values<R: RunEndIndexType>(value: &dyn Array, check: Check<'_>) -> Result<Counts> {
    array(typed::<RunArray<R>>(value)?.values().as_ref(), check)
}
fn typed<T: Array + 'static>(value: &dyn Array) -> Result<&T> {
    value.as_any().downcast_ref().ok_or_else(invalid)
}
