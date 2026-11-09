use super::{Array, ArrayRef, Check, Result, child_refs, concat as concatenate};
use datafusion::arrow::{
    array::{ArrayData, PrimitiveArray, RunArray, downcast_run_array, make_array},
    buffer::MutableBuffer,
    datatypes::{ArrowNativeType, RunEndIndexType},
};

pub(super) fn concat(arrays: &[&dyn Array], check: Check<'_>) -> Result<ArrayRef> {
    let first = arrays[0];
    downcast_run_array!(first => typed(first, arrays, check), _ => Err(super::invalid("normal run type")))
}
fn typed<R: RunEndIndexType>(
    _: &RunArray<R>,
    arrays: &[&dyn Array],
    check: Check<'_>,
) -> Result<ArrayRef> {
    let runs = arrays
        .iter()
        .map(|array| super::typed::<RunArray<R>>(*array))
        .collect::<Result<Vec<_>>>()?;
    let active = runs
        .iter()
        .copied()
        .filter(|run| !run.run_ends().is_empty())
        .collect::<Vec<_>>();
    joined::<R>(&active, check)
}
fn joined<R: RunEndIndexType>(active: &[&RunArray<R>], check: Check<'_>) -> Result<ArrayRef> {
    let first = active
        .first()
        .ok_or_else(|| super::invalid("empty normal run concat"))?;
    let ends = ends::<R>(active, check)?;
    let children = slices(active, check)?;
    let values = concatenate(&child_refs(&children), check)?;
    let len = ends.values().last().map_or(0, |end| end.as_usize());
    let data = ArrayData::builder(first.data_type().clone())
        .len(len)
        .child_data(vec![ends.into_data(), values.to_data()])
        .build()
        .map_err(|_| super::invalid("normal run constructor"))?;
    Ok(make_array(data))
}
fn slices<R: RunEndIndexType>(runs: &[&RunArray<R>], check: Check<'_>) -> Result<Vec<ArrayRef>> {
    let mut children = Vec::with_capacity(runs.len());
    for run in runs {
        check()?;
        children.push(run.values_slice());
    }
    Ok(children)
}
fn ends<R: RunEndIndexType>(runs: &[&RunArray<R>], check: Check<'_>) -> Result<PrimitiveArray<R>> {
    let mut buffer = MutableBuffer::new(0);
    let mut previous = 0usize;
    for run in runs {
        check()?;
        for end in run.run_ends().sliced_values() {
            let end = end
                .as_usize()
                .checked_add(previous)
                .and_then(R::Native::from_usize)
                .ok_or_else(|| super::invalid("normal run end"))?;
            buffer.push(end);
        }
        previous = previous
            .checked_add(run.len())
            .ok_or_else(|| super::invalid("normal run length"))?;
    }
    Ok(PrimitiveArray::<R>::new(buffer.into(), None))
}
