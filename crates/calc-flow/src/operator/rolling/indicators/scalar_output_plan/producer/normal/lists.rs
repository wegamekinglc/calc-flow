use super::super::super::{math::size, normal::view_offsets};
use super::{
    Array, ArrayRef, Check, FieldRef, OffsetSizeTrait, Result, child_refs, concat, field_array,
    length, nulls, offsets, typed,
};
use datafusion::arrow::{
    array::{GenericListArray, GenericListViewArray},
    buffer::{MutableBuffer, ScalarBuffer},
};

pub(super) fn list<O: OffsetSizeTrait>(
    arrays: &[&dyn Array],
    field: &FieldRef,
    check: Check<'_>,
) -> Result<ArrayRef> {
    let lists = arrays
        .iter()
        .map(|array| typed::<GenericListArray<O>>(*array))
        .collect::<Result<Vec<_>>>()?;
    let len = length(arrays, check)?;
    let nulls = nulls(arrays, len, check)?;
    let mut children = Vec::with_capacity(arrays.len());
    for list in &lists {
        check()?;
        let start = list.value_offsets()[0].as_usize();
        let end = list.value_offsets()[list.len()].as_usize();
        children.push(list.values().slice(start, end - start));
    }
    let values = concat(&child_refs(&children), check)?;
    let offsets = offsets::<O>(lists.iter().flat_map(|list| list.offsets().lengths()));
    GenericListArray::<O>::try_new(field.clone(), offsets, values, nulls)
        .map(field_array)
        .map_err(|_| super::invalid("normal list constructor"))
}
pub(super) fn view<O: OffsetSizeTrait>(
    arrays: &[&dyn Array],
    field: &FieldRef,
    check: Check<'_>,
) -> Result<ArrayRef> {
    let lists = arrays
        .iter()
        .map(|array| typed::<GenericListViewArray<O>>(*array))
        .collect::<Result<Vec<_>>>()?;
    let len = length(arrays, check)?;
    let nulls = nulls(arrays, len, check)?;
    let values = lists
        .iter()
        .map(|list| list.values().as_ref())
        .collect::<Vec<_>>();
    let values = concat(&values, check)?;
    let sizes: ScalarBuffer<O> = lists
        .iter()
        .flat_map(|list| list.value_sizes())
        .copied()
        .collect();
    let offsets = view_offsets::<O>(&lists, len, check)?;
    GenericListViewArray::<O>::try_new(field.clone(), offsets, sizes, values, nulls)
        .map(field_array)
        .map_err(|_| super::invalid("normal view constructor"))
}
fn view_offsets<O: OffsetSizeTrait>(
    lists: &[&GenericListViewArray<O>],
    rows: usize,
    check: Check<'_>,
) -> Result<ScalarBuffer<O>> {
    let mut offsets = MutableBuffer::with_capacity(rows);
    let mut global = 0;
    let last = lists.iter().rposition(|list| !list.is_empty());
    for (index, list) in lists.iter().enumerate() {
        check()?;
        append_offsets::<O>(&mut offsets, list, global, check)?;
        global = view_offsets::advance::<O>(
            global,
            size(list.values().len())?,
            last.is_some_and(|last| index < last),
        )?;
    }
    Ok(offsets.into())
}

fn append_offsets<O: OffsetSizeTrait>(
    buffer: &mut MutableBuffer,
    list: &GenericListViewArray<O>,
    global: u64,
    check: Check<'_>,
) -> Result<()> {
    for offset in list.value_offsets() {
        check()?;
        buffer.push(view_offsets::native::<O>(global, *offset)?);
    }
    Ok(())
}
