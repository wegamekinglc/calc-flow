pub(in crate::operator::join::checkpoint_v2) mod bulk;

use arrow_data::ArrayData;
use datafusion::arrow::{array::Array, buffer::Buffer, datatypes::DataType};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

use super::accounting::{self, add, product, sum};
use crate::Result;

mod snapshot;

#[derive(Default)]
struct Requests {
    workspace: usize,
    backing: usize,
}

impl Requests {
    fn include(&mut self, other: &Self) -> Result<()> {
        self.workspace = add(self.workspace, other.workspace)?;
        self.backing = add(self.backing, other.backing)?;
        Ok(())
    }
}

pub(in crate::operator::join::checkpoint_v2) fn admit(
    previous: &dyn Array,
    incoming: &dyn Array,
    workspace: &MemoryReservation,
    resident: &Arc<super::super::payload::Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    accounting::reserve(workspace, snapshot_requests(previous, incoming, check)?)?;
    check()?;
    let sources = [previous.to_data(), incoming.to_data()];
    let rows = add(previous.len(), incoming.len())?;
    let requests = source_requests(&sources, rows, check)?;
    accounting::reserve(workspace, requests.workspace)?;
    resident.grow(requests.backing)?;
    check()
}

fn snapshot_requests(
    previous: &dyn Array,
    incoming: &dyn Array,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let nodes = accounting::shape_nodes(previous.data_type())?;
    sum(&[
        accounting::array_controls(product(nodes, 2)?, product(nodes, 6)?)?,
        snapshot::bytes(previous, check)?,
        snapshot::bytes(incoming, check)?,
    ])
}

fn source_requests(
    sources: &[ArrayData; 2],
    rows: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<Requests> {
    let mut requests = Requests::default();
    for source in sources {
        check()?;
        requests.include(&array_requests(source, 1, rows, check)?)?;
    }
    requests.include(&nested_merges(&sources[0], &sources[1], 1, check)?)?;
    Ok(requests)
}

fn nested_merges(
    previous: &ArrayData,
    incoming: &ArrayData,
    repetitions: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<Requests> {
    check()?;
    let mut requests = dictionary_requests(previous, incoming, repetitions)?;
    let occurrences = merge_occurrences(previous, incoming, repetitions)?;
    for (left, right) in previous.child_data().iter().zip(incoming.child_data()) {
        requests.include(&nested_merges(left, right, occurrences, check)?)?;
    }
    Ok(requests)
}

fn dictionary_requests(
    previous: &ArrayData,
    incoming: &ArrayData,
    repetitions: usize,
) -> Result<Requests> {
    let mut requests = Requests::default();
    if let DataType::Dictionary(key, _) = previous.data_type() {
        let rows = product(add(previous.len(), incoming.len())?, repetitions)?;
        let merge = dictionary_merge([previous, incoming], rows, key)?;
        requests.workspace = product(merge.workspace, repetitions)?;
        requests.backing = product(merge.backing, repetitions)?;
    }
    Ok(requests)
}

fn merge_occurrences(left: &ArrayData, right: &ArrayData, repetitions: usize) -> Result<usize> {
    match left.data_type() {
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::Map(_, _)
        | DataType::Union(_, _) => product(repetitions, add(left.len(), right.len())?.max(1)),
        _ => Ok(repetitions),
    }
}

fn array_requests(
    data: &ArrayData,
    repetitions: usize,
    initial_rows: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<Requests> {
    check()?;
    let rows = product(data.len(), repetitions)?;
    let initial_rows = initial_rows.max(rows);
    let mut requests = initial_requests(data, rows, initial_rows)?;
    let top = top_backing(data, repetitions, rows, initial_rows)?;
    requests.backing = add(requests.backing, top)?;
    // Default child capacities coexist with repeated selections in the fallback kernel.
    let child_repetitions = child_repetitions(data, repetitions)?;
    include_children(&mut requests, data, child_repetitions, initial_rows, check)?;
    // The kernel holds source ArrayData, extension closures and completed sibling arrays.
    finish_requests(requests, data)
}

fn initial_requests(data: &ArrayData, rows: usize, initial_rows: usize) -> Result<Requests> {
    Ok(Requests {
        workspace: accounting::array_controls(1, add(data.buffers().len(), 1)?)?,
        backing: mutable_peak(bits(rows)?, bits(initial_rows)?)?,
    })
}

fn finish_requests(mut requests: Requests, data: &ArrayData) -> Result<Requests> {
    requests.workspace = add(requests.workspace, extension_controls(data)?)?;
    Ok(requests)
}

fn child_repetitions(data: &ArrayData, repetitions: usize) -> Result<usize> {
    match data.data_type() {
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::Map(_, _)
        | DataType::Union(_, _) => product(repetitions, data.len().max(1)),
        _ => Ok(repetitions),
    }
}

fn include_children(
    requests: &mut Requests,
    data: &ArrayData,
    child_repetitions: usize,
    initial_rows: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for child in data.child_data() {
        check()?;
        let child_rows = product(child.len(), child_repetitions)?;
        requests.include(&array_requests(
            child,
            child_repetitions,
            child_initial(data.data_type(), initial_rows)?.max(child_rows),
            check,
        )?)?;
    }
    Ok(())
}

fn top_backing(
    data: &ArrayData,
    repetitions: usize,
    rows: usize,
    initial_rows: usize,
) -> Result<usize> {
    match data.data_type() {
        DataType::Null | DataType::Struct(_) | DataType::FixedSizeList(_, _) => Ok(0),
        DataType::RunEndEncoded(_, _) => run_end_backing(data, repetitions),
        DataType::Boolean => mutable_peak(bits(rows)?, bits(initial_rows)?),
        DataType::Utf8 | DataType::Binary => byte_backing(data, repetitions, rows, initial_rows, 4),
        DataType::LargeUtf8 | DataType::LargeBinary => {
            byte_backing(data, repetitions, rows, initial_rows, 8)
        }
        DataType::List(_) | DataType::Map(_, _) => offsets(rows, initial_rows, 4),
        DataType::LargeList(_) => offsets(rows, initial_rows, 8),
        DataType::ListView(_) => product(2, values(rows, initial_rows, 4)?),
        DataType::LargeListView(_) => product(2, values(rows, initial_rows, 8)?),
        DataType::Union(_, mode) => union_backing(*mode, rows, initial_rows),
        DataType::Utf8View | DataType::BinaryView => {
            view_backing(data, repetitions, rows, initial_rows)
        }
        _ => primitive_backing(data, repetitions, initial_rows),
    }
}

fn run_end_backing(data: &ArrayData, repetitions: usize) -> Result<usize> {
    // The fallback additionally bulk-appends a temporary adjusted run-end Vec.
    data.child_data().first().map_or(Ok(0), |run_ends| {
        let bytes = product(
            run_ends.buffers().first().map_or(0, Buffer::len),
            repetitions,
        )?;
        product(bytes.max(8), 3)
    })
}

fn union_backing(
    mode: datafusion::arrow::datatypes::UnionMode,
    rows: usize,
    initial_rows: usize,
) -> Result<usize> {
    let ids = values(rows, initial_rows, 1)?;
    if mode == datafusion::arrow::datatypes::UnionMode::Dense {
        add(ids, values(rows, initial_rows, 4)?)
    } else {
        Ok(ids)
    }
}

fn view_backing(
    data: &ArrayData,
    repetitions: usize,
    rows: usize,
    initial_rows: usize,
) -> Result<usize> {
    let views = values(rows, initial_rows, 16)?;
    let bytes = data.buffers().iter().skip(1).try_fold(0, |total, buffer| {
        add(total, product(buffer.len(), repetitions)?)
    })?;
    add(views, mutable_peak(bytes, 0)?)
}

fn primitive_backing(data: &ArrayData, repetitions: usize, initial_rows: usize) -> Result<usize> {
    let bytes = data.buffers().iter().try_fold(0, |total, buffer| {
        add(total, product(buffer.len(), repetitions)?)
    })?;
    let width = data.data_type().primitive_width().unwrap_or(0);
    mutable_peak(bytes, product(initial_rows, width)?)
}

fn byte_backing(
    data: &ArrayData,
    repetitions: usize,
    rows: usize,
    initial_rows: usize,
    width: usize,
) -> Result<usize> {
    let bytes = product(data.buffers().get(1).map_or(0, Buffer::len), repetitions)?;
    add(
        offsets(rows, initial_rows, width)?,
        mutable_peak(bytes, initial_rows)?,
    )
}

fn offsets(rows: usize, initial_rows: usize, width: usize) -> Result<usize> {
    values(add(rows, 1)?, add(initial_rows, 1)?, width)
}

fn values(rows: usize, initial_rows: usize, width: usize) -> Result<usize> {
    mutable_peak(product(rows, width)?, product(initial_rows, width)?)
}

fn child_initial(data_type: &DataType, rows: usize) -> Result<usize> {
    if let DataType::FixedSizeList(_, width) = data_type {
        let width = usize::try_from(*width)
            .map_err(|_| super::super::geometry::invalid("V2 fixed-list width is negative"))?;
        product(rows, width)
    } else {
        Ok(rows)
    }
}

fn extension_controls(data: &ArrayData) -> Result<usize> {
    let fields = add(data.buffers().len(), data.child_data().len())?;
    sum(&[
        accounting::vector_peak::<ArrayData>(2)?,
        accounting::vector_peak::<&ArrayData>(2)?,
        extension_closures(fields)?,
        extension_captures(fields)?,
        accounting::data_type_bytes(data.data_type())?,
    ])
}

fn extension_closures(fields: usize) -> Result<usize> {
    accounting::vector_peak::<Box<dyn FnMut()>>(product(fields, 2)?)
}

fn extension_captures(fields: usize) -> Result<usize> {
    product(
        product(fields, 2)?,
        sum(&[size_of::<ArrayData>(), size_of::<usize>()])?,
    )
}

fn dictionary_merge(sources: [&ArrayData; 2], rows: usize, key: &DataType) -> Result<Requests> {
    let value_count = sources.iter().try_fold(0, |total, source| {
        add(total, source.child_data().first().map_or(0, ArrayData::len))
    })?;
    let buckets = add(value_count, 129)?
        .checked_next_power_of_two()
        .ok_or_else(|| super::super::geometry::invalid("V2 dictionary interner overflow"))?;
    let (key_bytes, entry_bytes) = key_layout(key)?;
    Ok(Requests {
        workspace: dictionary_workspace(value_count, buckets, key_bytes, entry_bytes)?,
        backing: dictionary_backing(rows, key_bytes)?,
    })
}

fn dictionary_workspace(
    value_count: usize,
    buckets: usize,
    key_bytes: usize,
    entry_bytes: usize,
) -> Result<usize> {
    sum(&[
        product(buckets, entry_bytes)?,
        accounting::vector_peak::<(usize, Option<&[u8]>)>(value_count)?,
        accounting::vector_peak::<(usize, usize)>(value_count)?,
        dictionary_key_workspace(value_count, key_bytes)?,
        mutable_peak(bits(value_count)?, 0)?,
        accounting::vector_peak::<&dyn Array>(2)?,
    ])
}

fn dictionary_key_workspace(value_count: usize, key_bytes: usize) -> Result<usize> {
    mutable_peak(product(value_count, key_bytes)?, 0)
}

fn dictionary_backing(rows: usize, key_bytes: usize) -> Result<usize> {
    add(
        values(rows, rows, key_bytes)?,
        mutable_peak(bits(rows)?, 0)?,
    )
}

fn key_layout(key: &DataType) -> Result<(usize, usize)> {
    macro_rules! layout {
        ($native:ty) => {
            (
                size_of::<$native>(),
                size_of::<Option<(Option<&[u8]>, $native)>>(),
            )
        };
    }
    Ok(match key {
        DataType::Int8 => layout!(i8),
        DataType::Int16 => layout!(i16),
        DataType::Int32 => layout!(i32),
        DataType::Int64 => layout!(i64),
        DataType::UInt8 => layout!(u8),
        DataType::UInt16 => layout!(u16),
        DataType::UInt32 => layout!(u32),
        DataType::UInt64 => layout!(u64),
        _ => {
            return Err(super::super::geometry::invalid(
                "V2 dictionary key type is invalid",
            ));
        }
    })
}

fn bits(rows: usize) -> Result<usize> {
    Ok(add(rows, 7)? / 8)
}

fn mutable_peak(final_bytes: usize, initial_bytes: usize) -> Result<usize> {
    let final_capacity = product(round64(final_bytes)?, 2)?;
    let capacity = round64(initial_bytes)?.max(final_capacity);
    let peak = add(capacity, capacity / 2)?;
    std::alloc::Layout::from_size_align(peak, 64)
        .map_err(|_| super::super::geometry::invalid("V2 concat allocation layout overflow"))?;
    Ok(peak)
}

fn round64(bytes: usize) -> Result<usize> {
    product(add(bytes, 63)? / 64, 64)
}
