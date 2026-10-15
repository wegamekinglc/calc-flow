use super::{add, cell_sum, range_sum, typed};
use crate::{CalcFlowError, Result};
use datafusion::arrow::{
    array::{
        Array, DictionaryArray, FixedSizeListArray, LargeListArray, LargeListViewArray, ListArray,
        ListViewArray, MapArray, RunArray, StructArray, UnionArray,
    },
    datatypes::{DataType, Int32Type},
};

pub(super) fn non_null(array: &dyn Array, row: usize) -> Result<u64> {
    #[cfg(test)]
    super::super::note_join_work(|work| work.charge_nested_rows += 1);
    match array.data_type() {
        DataType::List(_) => {
            let list = typed::<ListArray>(array)?;
            offsets(list.values().as_ref(), list.value_offsets(), row, 4)
        }
        DataType::LargeList(_) => {
            let list = typed::<LargeListArray>(array)?;
            offsets(list.values().as_ref(), list.value_offsets(), row, 8)
        }
        DataType::Map(..) => {
            let map = typed::<MapArray>(array)?;
            offsets(map.entries(), map.value_offsets(), row, 4)
        }
        DataType::ListView(_) | DataType::LargeListView(_) | DataType::FixedSizeList(..) => {
            view(array, row)
        }
        _ => encoded(array, row),
    }
}

fn offsets<O: Copy + TryInto<usize>>(
    child: &dyn Array,
    offsets: &[O],
    row: usize,
    prefix: u64,
) -> Result<u64> {
    let start = index(offsets[row])?;
    let end = index(offsets[row + 1])?;
    add(prefix, range_sum(child, start..end)?)
}

fn view(array: &dyn Array, row: usize) -> Result<u64> {
    let (child, start, length, prefix) = match array.data_type() {
        DataType::ListView(_) => {
            let list = typed::<ListViewArray>(array)?;
            (
                list.values(),
                index(list.value_offset(row))?,
                index(list.value_size(row))?,
                8,
            )
        }
        DataType::LargeListView(_) => {
            let list = typed::<LargeListViewArray>(array)?;
            (
                list.values(),
                index(list.value_offset(row))?,
                index(list.value_size(row))?,
                16,
            )
        }
        _ => {
            let list = typed::<FixedSizeListArray>(array)?;
            let length = index(list.value_length())?;
            (list.values(), row * length, length, 0)
        }
    };
    add(prefix, range_sum(child.as_ref(), start..start + length)?)
}

fn encoded(array: &dyn Array, row: usize) -> Result<u64> {
    match array.data_type() {
        DataType::Struct(_) => typed::<StructArray>(array)?
            .columns()
            .iter()
            .try_fold(0, |total, child| add(total, cell_sum(child.as_ref(), row)?)),
        DataType::Union(..) => {
            let union = typed::<UnionArray>(array)?;
            let offset = union.value_offset(row);
            add(
                1,
                cell_sum(union.child(union.type_id(row)).as_ref(), offset)?,
            )
        }
        DataType::Dictionary(..) => {
            // V1 downcasts only Int32 physical keys. Keep all other
            // key widths on its original, tested type-mismatch path.
            let dictionary = typed::<DictionaryArray<Int32Type>>(array)?;
            let offset = index(dictionary.keys().value(row))?;
            cell_sum(dictionary.values().as_ref(), offset)
        }
        DataType::RunEndEncoded(..) => {
            let run = typed::<RunArray<Int32Type>>(array)?;
            let offset = run.get_physical_index(row);
            cell_sum(run.values().as_ref(), offset)
        }
        _ => Err(super::super::unsupported_payload_type(array.data_type())),
    }
}

fn index<T: TryInto<usize>>(value: T) -> Result<usize> {
    value.try_into().map_err(|_| CalcFlowError::Internal {
        message: "nested charge offset does not fit usize".into(),
    })
}
