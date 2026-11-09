use super::super::math::{add, grow, retained, size, sum, vector};
use super::{At, Check, Cost, Piece, Result, invalid, node, slice, typed};
use datafusion::arrow::{
    array::{GenericListArray, GenericListViewArray, OffsetSizeTrait},
    datatypes::DataType,
};

pub(super) fn plan(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    match dtype {
        DataType::List(field) => list::<i32>(field.data_type(), count, rows, at, check),
        DataType::LargeList(field) => list::<i64>(field.data_type(), count, rows, at, check),
        DataType::ListView(field) => view::<i32>(field.data_type(), count, rows, at, check),
        DataType::LargeListView(field) => view::<i64>(field.data_type(), count, rows, at, check),
        _ => composite(dtype, count, rows, at, check),
    }
}
fn composite(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    match dtype {
        DataType::RunEndEncoded(..) => super::run::plan(dtype, count, rows, at, check),
        DataType::Struct(fields) => super::structures::structure(fields, count, at, check),
        DataType::Map(field, _) => {
            super::structures::map(field.data_type(), count, rows, at, check)
        }
        DataType::FixedSizeList(..) | DataType::Union(..) => {
            super::super::fixed::normal(dtype, count, rows, at, check)
        }
        DataType::Dictionary(..) => {
            super::super::normal::dictionary::plan(dtype, count, rows, at, check)
        }
        _ => Err(invalid("nested concat datatype")),
    }
}

fn list<O: OffsetSizeTrait>(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    super::super::normal::list_length::<O>(count, at, check)?;
    copy_list::<O>(dtype, count, rows, at, check)
}
fn copy_list<O: OffsetSizeTrait>(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let mut child = |index| selected::<O>(at(index)?);
    let cost = node(dtype, count, &mut child, check)?;
    let offsets = retained(add(rows, 1)?, size(size_of::<O>())?)?;
    cost.combine(Cost {
        owned: offsets,
        temporary: sum([Ok(offsets), slices::<O>(dtype, count, at, check)])?,
    })
}
pub(in crate::operator::rolling::indicators::scalar_output_plan) fn selected<O: OffsetSizeTrait>(
    piece: Piece<'_>,
) -> Result<Piece<'_>> {
    let Some(array) = piece.array else {
        return Ok(Piece::null(0));
    };
    let list = typed::<GenericListArray<O>>(array)?;
    let start = list.value_offsets()[piece.start].as_usize();
    let end = list.value_offsets()[piece.start + piece.len].as_usize();
    slice(
        list.values().as_ref(),
        start,
        end.checked_sub(start)
            .ok_or_else(|| invalid("list range"))?,
    )
}
fn slices<O: OffsetSizeTrait>(
    dtype: &DataType,
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<u64> {
    let mut total = vector::<datafusion::arrow::array::ArrayRef>(count)?;
    for index in 0..count {
        check()?;
        total = add(total, slice_descriptor::<O>(dtype, at(index)?, check)?)?;
    }
    Ok(total)
}

fn slice_descriptor<O: OffsetSizeTrait>(
    dtype: &DataType,
    piece: Piece<'_>,
    check: Check<'_>,
) -> Result<u64> {
    let child = selected::<O>(piece)?;
    match child.array {
        Some(array) => super::super::single::sliced(array, check),
        None => super::super::super::scalar_materialization_plan::null_array_requirement(
            dtype, 0, check,
        ),
    }
}
fn view<O: OffsetSizeTrait>(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    super::super::normal::view_length::<O>(count, at, check)?;
    let mut child = |index| full::<O>(at(index)?);
    let cost = node(dtype, count, &mut child, check)?;
    cost.combine(view_buffers::<O>(rows, check)?)
}
fn view_buffers<O: OffsetSizeTrait>(rows: u64, check: Check<'_>) -> Result<Cost> {
    let width = size(size_of::<O>())?;
    let offsets = grown_offsets(rows, width, check)?;
    Ok(Cost {
        owned: add(offsets.capacity, retained(rows, width)?)?,
        temporary: sum([Ok(offsets.capacity), Ok(offsets.old), grow(rows, width)])?,
    })
}
fn grown_offsets(rows: u64, width: u64, check: Check<'_>) -> Result<super::super::fixed::Growing> {
    let mut offsets = super::super::fixed::Growing::new(rows, 0)?;
    for _ in 0..rows {
        check()?;
        offsets.append(width, 0)?;
    }
    Ok(offsets)
}
pub(in crate::operator::rolling::indicators::scalar_output_plan) fn full<O: OffsetSizeTrait>(
    piece: Piece<'_>,
) -> Result<Piece<'_>> {
    let Some(array) = piece.array else {
        return Ok(Piece::null(0));
    };
    let list = typed::<GenericListViewArray<O>>(array)?;
    slice(list.values().as_ref(), 0, list.values().len())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_normal_list_view_retains_real_mutable_offset_capacity() {
        use datafusion::arrow::{
            array::{Array, ListViewArray, NullArray},
            datatypes::{DataType, Field},
        };
        use std::sync::Arc;
        let array = ListViewArray::new(
            Arc::new(Field::new("item", DataType::Null, true)),
            vec![0].into(),
            vec![1].into(),
            Arc::new(NullArray::new(1)),
            None,
        );
        let actual = datafusion::arrow::compute::concat(&[&array, &array]).unwrap();
        let payload = actual
            .to_data()
            .buffers()
            .iter()
            .map(|buffer| u64::try_from(buffer.capacity()).unwrap())
            .sum::<u64>();
        let piece = super::Piece {
            array: Some(&array),
            start: 0,
            len: 1,
        };
        let quote =
            super::view::<i32>(&DataType::Null, 2, 2, &mut |_| Ok(piece), &mut || Ok(())).unwrap();
        assert!(
            quote.owned >= payload,
            "quoted {} < retained {payload}",
            quote.owned
        );
    }
}
