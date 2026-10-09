use std::sync::Arc;

use arrow_data::ArrayData;
use datafusion::{
    arrow::{array::Array, datatypes::DataType},
    execution::memory_pool::MemoryReservation,
};

use super::{
    Requests, accounting, add, array_requests, nested_merges, product, snapshot_requests, sum,
};
use crate::Result;

pub(in crate::operator::join::checkpoint_v2) fn admit(
    arrays: &[&dyn Array],
    workspace: &MemoryReservation,
    resident: &Arc<super::super::super::payload::Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    accounting::reserve(workspace, snapshots(arrays, check)?)?;
    check()?;
    let sources = arrays
        .iter()
        .map(|array| array.to_data())
        .collect::<Vec<_>>();
    let requests = requests(&sources, check)?;
    accounting::reserve(workspace, requests.workspace)?;
    resident.grow(retained_bytes(&requests, &sources, check)?)?;
    check()
}

fn retained_bytes(
    requests: &Requests,
    sources: &[ArrayData],
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    add(requests.backing, output_controls(sources, check)?)
}

fn output_controls(sources: &[ArrayData], check: &dyn Fn() -> Result<()>) -> Result<usize> {
    let Some(first) = sources.first() else {
        return Ok(0);
    };
    let nodes = accounting::shape_nodes(first.data_type())?;
    let buffers = sources.iter().try_fold(nodes, |total, source| {
        add(total, graph_buffers(source, check)?)
    })?;
    add(
        accounting::array_controls(nodes, buffers)?,
        graph_types(first, check)?,
    )
}

fn graph_buffers(data: &ArrayData, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    check()?;
    data.child_data()
        .iter()
        .try_fold(data.buffers().len(), |total, child| {
            add(total, graph_buffers(child, check)?)
        })
}

fn graph_types(data: &ArrayData, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    check()?;
    data.child_data().iter().try_fold(
        accounting::data_type_bytes(data.data_type())?,
        |total, child| add(total, graph_types(child, check)?),
    )
}

fn snapshots(arrays: &[&dyn Array], check: &dyn Fn() -> Result<()>) -> Result<usize> {
    let mut bytes = sum(&[
        accounting::vector_peak::<ArrayData>(arrays.len())?,
        accounting::vector_peak::<&ArrayData>(arrays.len())?,
    ])?;
    for array in arrays {
        check()?;
        bytes = add(bytes, snapshot_requests(*array, *array, check)?)?;
    }
    Ok(bytes)
}

fn requests(sources: &[ArrayData], check: &dyn Fn() -> Result<()>) -> Result<Requests> {
    let mut requests = Requests::default();
    for source in sources {
        check()?;
        requests.include(&single_requests(source, check)?)?;
    }
    finish_requests(&requests, sources)
}

fn finish_requests(requests: &Requests, sources: &[ArrayData]) -> Result<Requests> {
    // Global dictionary buckets and grown Vecs can round above summed local capacities.
    Ok(Requests {
        workspace: add(product(requests.workspace, 2)?, bulk_controls(sources)?)?,
        backing: product(requests.backing, 2)?,
    })
}

fn bulk_controls(sources: &[ArrayData]) -> Result<usize> {
    let Some(first) = sources.first() else {
        return Ok(0);
    };
    let nodes = accounting::shape_nodes(first.data_type())?;
    let references = product(accounting::vector_peak::<&dyn Array>(sources.len())?, 3)?;
    let runs = run_controls(sources.len())?;
    product(
        nodes,
        sum(&[
            references,
            runs,
            size_of::<arrow_data::transform::Capacities>(),
        ])?,
    )
}

fn run_controls(sources: usize) -> Result<usize> {
    sum(&[
        accounting::vector_peak::<Arc<dyn Array>>(sources)?,
        accounting::vector_peak::<u64>(add(sources, 1)?)?,
    ])
}

fn single_requests(source: &ArrayData, check: &dyn Fn() -> Result<()>) -> Result<Requests> {
    if let Some(backing) = flat_backing(source)? {
        let mut requests = super::initial_requests(source, source.len(), source.len())?;
        requests.backing = add(requests.backing, backing)?;
        return super::finish_requests(requests, source);
    }
    let mut requests = array_requests(source, 1, source.len(), check)?;
    requests.include(&nested_merges(source, source, 1, check)?)?;
    Ok(requests)
}

fn flat_backing(source: &ArrayData) -> Result<Option<usize>> {
    let bytes = match source.data_type() {
        DataType::Utf8 | DataType::Binary => byte_backing::<i32>(source)?,
        DataType::LargeUtf8 | DataType::LargeBinary => byte_backing::<i64>(source)?,
        data_type => {
            let Some(width) = data_type.primitive_width() else {
                return Ok(None);
            };
            super::values(source.len(), source.len(), width)?
        }
    };
    Ok(Some(bytes))
}

fn byte_backing<O>(source: &ArrayData) -> Result<usize>
where
    O: datafusion::arrow::datatypes::ArrowNativeType,
    usize: TryFrom<O>,
{
    let bytes = selected_bytes::<O>(source)?;
    add(
        super::offsets(source.len(), source.len(), size_of::<O>())?,
        super::mutable_peak(bytes, source.len())?,
    )
}

fn selected_bytes<O>(source: &ArrayData) -> Result<usize>
where
    O: datafusion::arrow::datatypes::ArrowNativeType,
    usize: TryFrom<O>,
{
    let buffer = source.buffers().first().ok_or_else(invalid_offsets)?;
    let offsets = buffer.typed_data::<O>();
    let start = offset::<O>(offsets, source.offset())?;
    let end = offset::<O>(offsets, add(source.offset(), source.len())?)?;
    end.checked_sub(start).ok_or_else(invalid_offsets)
}

fn offset<O>(offsets: &[O], index: usize) -> Result<usize>
where
    O: Copy,
    usize: TryFrom<O>,
{
    offsets
        .get(index)
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(invalid_offsets)
}

fn invalid_offsets() -> crate::CalcFlowError {
    super::super::super::geometry::invalid("V2 concat selected byte offsets differ")
}
