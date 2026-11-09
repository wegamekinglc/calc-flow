use super::{Check, Result, concatenate, invalid, null_array};
use datafusion::{
    arrow::{
        array::{ArrayRef, UnionArray},
        datatypes::{DataType, UnionFields, UnionMode},
    },
    common::ScalarValue,
};
use std::sync::Arc;

pub(super) fn column(values: Vec<ScalarValue>, check: Check<'_>) -> Result<ArrayRef> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        arrays.push(super::single(&value, check)?);
    }
    concatenate(&arrays, check)
}
pub(super) fn single(
    value: Option<&(i8, Box<ScalarValue>)>,
    fields: &UnionFields,
    mode: UnionMode,
    check: Check<'_>,
) -> Result<ArrayRef> {
    let Some((id, value)) = value else {
        return null_array(&DataType::Union(fields.clone(), mode), 1, check);
    };
    if !fields.iter().any(|(field_id, _)| field_id == *id) {
        return Err(invalid("union type ID"));
    }
    let children = children(*id, value, fields, mode, check)?;
    let offsets = (mode == UnionMode::Dense).then(|| vec![0_i32].into());
    let array = UnionArray::try_new(fields.clone(), vec![*id].into(), offsets, children)
        .map_err(|_| invalid("union"))?;
    Ok(Arc::new(array))
}
fn children(
    id: i8,
    value: &ScalarValue,
    fields: &UnionFields,
    mode: UnionMode,
    check: Check<'_>,
) -> Result<Vec<ArrayRef>> {
    let mut arrays = Vec::with_capacity(fields.len());
    for (field_id, field) in fields.iter() {
        check()?;
        let array = if field_id == id {
            super::super::single_ctor::plan(field.data_type(), value, check)?;
            super::single(value, check)?
        } else {
            null_array(
                field.data_type(),
                usize::from(mode == UnionMode::Sparse),
                check,
            )?
        };
        arrays.push(array);
    }
    Ok(arrays)
}
