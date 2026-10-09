use super::*;

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
    resident.grow(requests.backing)?;
    check()
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
    finish_requests(requests, sources)
}

fn finish_requests(requests: Requests, sources: &[ArrayData]) -> Result<Requests> {
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
    let mut requests = array_requests(source, 1, source.len(), check)?;
    requests.include(&nested_merges(source, source, 1, check)?)?;
    Ok(requests)
}
