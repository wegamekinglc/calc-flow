use super::{Check, Result, invalid};
use datafusion::arrow::{
    array::{ArrayData, ByteView},
    datatypes::{DataType, UnionMode},
};
#[path = "views/nested.rs"]
mod nested;
#[path = "views/ranges.rs"]
mod ranges;
type At<'a, 'b> = dyn Fn(usize) -> &'a ArrayData + 'b;
type Emit<'a> = dyn FnMut(usize, usize, Check<'_>) -> Result<()> + 'a;
type Copies<'a> = dyn Fn(usize, &mut Emit<'_>, Check<'_>) -> Result<()> + 'a;

pub(super) fn validate(data: &[ArrayData], check: Check<'_>) -> Result<()> {
    let at = |index: usize| &data[index];
    let copies =
        |index: usize, emit: &mut Emit<'_>, check: Check<'_>| emit(0, data[index].len(), check);
    walk(data.len(), &at, &copies, check)
}
fn walk(count: usize, at: &At<'_, '_>, copies: &Copies<'_>, check: Check<'_>) -> Result<()> {
    check()?;
    match at(0).data_type() {
        DataType::Utf8View | DataType::BinaryView => views(count, at, copies, check),
        DataType::Dictionary(_, _) => dictionary(count, at, check),
        DataType::Struct(fields) => structure(fields.len(), count, at, copies, check),
        DataType::Union(fields, mode) => union(fields, *mode, count, at, copies, check),
        _ => nested::walk(count, at, copies, check),
    }
}
fn views(count: usize, at: &At<'_, '_>, copies: &Copies<'_>, check: Check<'_>) -> Result<()> {
    let mut preceding = 0_u64;
    let mut pinned = 0_u32;
    for index in 0..count {
        check()?;
        let array = at(index);
        let buffers =
            u64::try_from(array.buffers().len() - 1).map_err(|_| invalid("view buffers"))?;
        pinned = super::super::fixed::advance_view_offset(pinned, buffers)?;
        copies(
            index,
            &mut |start, len, check| referenced(array, start, len, preceding, check),
            check,
        )?;
        preceding = super::super::math::add(preceding, buffers)?;
    }
    Ok(())
}
fn referenced(
    array: &ArrayData,
    start: usize,
    len: usize,
    preceding: u64,
    check: Check<'_>,
) -> Result<()> {
    for value in &array.buffer::<u128>(0)[start..start + len] {
        check()?;
        let view = ByteView::from(*value);
        if view.length > 12 {
            reference_index(preceding, view.buffer_index)?;
        }
    }
    Ok(())
}
pub(super) fn reference_index(preceding: u64, index: u32) -> Result<u32> {
    let index = super::super::math::add(preceding, u64::from(index))?;
    u32::try_from(index).map_err(|_| invalid("mutable referenced view buffer index"))
}
fn dictionary(count: usize, at: &At<'_, '_>, check: Check<'_>) -> Result<()> {
    if super::fallback::same_values(count, at, check)? {
        return Ok(());
    }
    let child = |index| &at(index).child_data()[0];
    let copies =
        |index: usize, emit: &mut Emit<'_>, check: Check<'_>| emit(0, child(index).len(), check);
    walk(count, &child, &copies, check)
}
fn structure(
    fields: usize,
    count: usize,
    at: &At<'_, '_>,
    copies: &Copies<'_>,
    check: Check<'_>,
) -> Result<()> {
    for index in 0..fields {
        let child = |ordinal| &at(ordinal).child_data()[index];
        walk(count, &child, copies, check)?;
    }
    Ok(())
}
fn union(
    fields: &datafusion::arrow::datatypes::UnionFields,
    mode: UnionMode,
    count: usize,
    at: &At<'_, '_>,
    copies: &Copies<'_>,
    check: Check<'_>,
) -> Result<()> {
    for (index, (id, _)) in fields.iter().enumerate() {
        let child = |ordinal| &at(ordinal).child_data()[index];
        let selected = |ordinal, emit: &mut Emit<'_>, check: Check<'_>| {
            copies(
                ordinal,
                &mut |start, len, check| {
                    ranges::union(at(ordinal), id, mode, start, len, emit, check)
                },
                check,
            )
        };
        if mode == UnionMode::Dense {
            dense_length(copied_rows(count, &selected, check)?)?;
        }
        walk(count, &child, &selected, check)?;
    }
    Ok(())
}

pub(super) fn dense_length(rows: u64) -> Result<()> {
    super::fallback::span(&DataType::Int32, 0, rows)
}
fn copied_rows(count: usize, copies: &Copies<'_>, check: Check<'_>) -> Result<u64> {
    let mut rows = 0;
    for index in 0..count {
        check()?;
        copies(
            index,
            &mut |_, len, _| {
                rows = super::super::math::add(rows, super::super::math::size(len)?)?;
                Ok(())
            },
            check,
        )?;
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_dense_child_offset_guard_preserves_last_representable_index() {
        let rows = u64::try_from(i32::MAX).unwrap() + 1;
        assert!(super::dense_length(0).is_ok());
        assert!(super::dense_length(rows).is_ok());
        assert!(super::dense_length(rows + 1).is_err());
        assert!(super::dense_length(u64::MAX).is_err());
    }
}
