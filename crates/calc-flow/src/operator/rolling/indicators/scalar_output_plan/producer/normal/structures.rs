use super::{
    Array, ArrayRef, Check, FieldRef, Result, child_refs, concat, field_array, length, nulls,
    offsets, typed,
};
use datafusion::arrow::{
    array::{MapArray, StructArray},
    datatypes::Fields,
};

pub(super) fn structure(
    arrays: &[&dyn Array],
    fields: &Fields,
    check: Check<'_>,
) -> Result<ArrayRef> {
    let structs = arrays
        .iter()
        .map(|array| typed::<StructArray>(*array))
        .collect::<Result<Vec<_>>>()?;
    let len = length(arrays, check)?;
    let nulls = nulls(arrays, len, check)?;
    let mut columns = Vec::with_capacity(fields.len());
    for index in 0..fields.len() {
        check()?;
        let children = structs
            .iter()
            .map(|array| array.column(index).as_ref())
            .collect::<Vec<_>>();
        columns.push(concat(&children, check)?);
    }
    StructArray::try_new_with_length(fields.clone(), columns, nulls, len)
        .map(field_array)
        .map_err(|_| super::invalid("normal struct constructor"))
}
pub(super) fn map(
    arrays: &[&dyn Array],
    field: &FieldRef,
    ordered: bool,
    check: Check<'_>,
) -> Result<ArrayRef> {
    let maps = arrays
        .iter()
        .map(|array| typed::<MapArray>(*array))
        .collect::<Result<Vec<_>>>()?;
    let len = length(arrays, check)?;
    let nulls = nulls(arrays, len, check)?;
    let entries = entries(&maps, check)?;
    let values = concat(&child_refs(&entries), check)?;
    let offsets = offsets::<i32>(maps.iter().flat_map(|map| map.offsets().lengths()));
    MapArray::try_new(
        field.clone(),
        offsets,
        typed::<StructArray>(values.as_ref())?.clone(),
        nulls,
        ordered,
    )
    .map(field_array)
    .map_err(|_| super::invalid("normal map constructor"))
}

fn entries(maps: &[&MapArray], check: Check<'_>) -> Result<Vec<ArrayRef>> {
    let mut entries = Vec::with_capacity(maps.len());
    for map in maps {
        check()?;
        let start = usize::try_from(map.value_offsets()[0])
            .map_err(|_| super::invalid("normal map offset"))?;
        let end = usize::try_from(map.value_offsets()[map.len()])
            .map_err(|_| super::invalid("normal map offset"))?;
        let array: &dyn Array = map.entries();
        entries.push(array.slice(start, end - start));
    }
    Ok(entries)
}
