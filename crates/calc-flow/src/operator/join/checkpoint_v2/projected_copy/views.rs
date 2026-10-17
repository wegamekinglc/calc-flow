use std::sync::Arc;

use arrow_data::ArrayData;
use datafusion::arrow::{
    array::{
        Array, ArrayRef, BinaryViewArray, GenericByteViewArray, StringViewArray,
        builder::GenericByteViewBuilder,
        make_array,
        types::{BinaryViewType, ByteViewType, StringViewType},
    },
    buffer::Buffer,
    datatypes::DataType,
};
use datafusion::execution::memory_pool::MemoryReservation;

use super::{Funding, Result, StrideStop, accounting, add, concat, invalid, product, reserve, sum};

mod dictionaries;

#[cfg(test)]
pub(in crate::operator::join) fn assert_dictionary_controls() {
    dictionaries::tests::controls();
}

pub(super) fn independent(
    array: ArrayRef,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    if !requires_copy(array.data_type()) {
        return Ok(array);
    }
    reserve(workspace, concat::slice_controls(array.as_ref(), check)?)?;
    rebuild(&array.to_data(), workspace, funding, check).map(make_array)
}

fn requires_copy(data_type: &DataType) -> bool {
    match data_type {
        DataType::Utf8View | DataType::BinaryView | DataType::Dictionary(_, _) => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => requires_copy(field.data_type()),
        DataType::Struct(fields) => fields.iter().any(|field| requires_copy(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| requires_copy(field.data_type())),
        DataType::RunEndEncoded(ends, values) => {
            requires_copy(ends.data_type()) || requires_copy(values.data_type())
        }
        _ => false,
    }
}

fn rebuild(
    data: &ArrayData,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayData> {
    check()?;
    reserve(
        workspace,
        accounting::array_controls(2, add(data.buffers().len(), data.child_data().len())?)?,
    )?;
    if let Some(copied) = copy_leaf(data, workspace, funding, check) {
        return copied;
    }
    let children = data
        .child_data()
        .iter()
        .map(|child| rebuild(child, workspace, funding, check))
        .collect::<Result<Vec<_>>>()?;
    data.clone()
        .into_builder()
        .child_data(children)
        .build()
        .map_err(|_| invalid("selected payload reconstruction failed"))
}

fn copy_leaf(
    data: &ArrayData,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Option<Result<ArrayData>> {
    match data.data_type() {
        DataType::Dictionary(key, _) => {
            Some(dictionaries::copy(data, key, workspace, funding, check))
        }
        DataType::Utf8View => Some(copy_view::<StringViewType>(
            &StringViewArray::from(data.clone()),
            workspace,
            funding,
            check,
        )),
        DataType::BinaryView => Some(copy_view::<BinaryViewType>(
            &BinaryViewArray::from(data.clone()),
            workspace,
            funding,
            check,
        )),
        _ => None,
    }
}

fn copy_view<T: ByteViewType>(
    array: &GenericByteViewArray<T>,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayData> {
    let bytes = long_values(array, check)?;
    admit_builder::<T>(array.len(), bytes, workspace, funding)?;
    // A known fixed block prevents geometric growth from copying unused source backing.
    let block =
        u32::try_from(bytes.min(u32::MAX as usize - 1).max(1)).expect("bounded nonzero view block");
    let mut builder =
        GenericByteViewBuilder::<T>::with_capacity(array.len()).with_fixed_block_size(block);
    let mut stop = StrideStop::new(check);
    for value in array {
        stop.visit()?;
        append_view(&mut builder, value)?;
    }
    check()?;
    Ok(builder.finish().to_data())
}

fn append_view<T: ByteViewType>(
    builder: &mut GenericByteViewBuilder<T>,
    value: Option<&T::Native>,
) -> Result<()> {
    if let Some(value) = value {
        builder
            .try_append_value(value)
            .map_err(|error| crate::CalcFlowError::DataFusion {
                node_id: None,
                message: format!("retained view copy failed: {error}"),
            })
    } else {
        builder.append_null();
        Ok(())
    }
}

fn admit_builder<T: ByteViewType>(
    rows: usize,
    values: usize,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
) -> Result<()> {
    funding.grow(view_bytes(rows, values)?)?;
    reserve(
        workspace,
        sum(&[
            size_of::<GenericByteViewBuilder<T>>(),
            accounting::array_controls(3, add(rows, 2)?)?,
        ])?,
    )
}

fn long_values<T: ByteViewType>(
    array: &GenericByteViewArray<T>,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let mut stop = StrideStop::new(check);
    (0..array.len()).try_fold(0, |bytes, row| {
        stop.visit()?;
        // The low u32 of a view is the byte length, including inline values.
        let length = usize::try_from(arrow_data::ByteView::from(array.views()[row]).length)
            .expect("u32 view length fits supported usize");
        if array.is_null(row) || length <= 12 {
            Ok(bytes)
        } else {
            add(bytes, validate_length(length)?)
        }
    })
}

pub(in crate::operator::join) fn validate_length(length: usize) -> Result<usize> {
    if length >= u32::MAX as usize {
        return Err(crate::CalcFlowError::DataFusion {
            node_id: None,
            message: "retained view value exceeds Arrow's owned block limit".into(),
        });
    }
    Ok(length)
}

fn view_bytes(rows: usize, values: usize) -> Result<usize> {
    sum(&[
        product(values, 3)?,
        view_controls(rows)?,
        view_buffers(rows)?,
    ])
}

fn view_controls(rows: usize) -> Result<usize> {
    let views = product(add(rows, 4)?, size_of::<u128>())?;
    let validity = add(rows, 7)? / 8;
    product(add(views, add(validity, 128)?)?, 3)
}

fn view_buffers(rows: usize) -> Result<usize> {
    sum(&[
        accounting::vector_peak::<Buffer>(add(rows, 1)?)?,
        accounting::array_controls(2, add(rows, 2)?)?,
    ])
}
