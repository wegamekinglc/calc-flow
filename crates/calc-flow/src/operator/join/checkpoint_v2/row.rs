use std::ops::Range;

use datafusion::arrow::{
    array::{
        Array, BinaryViewArray, DictionaryArray, FixedSizeListArray, GenericListArray,
        GenericListViewArray, LargeListArray, LargeListViewArray, LargeStringArray, ListArray,
        ListViewArray, MapArray, OffsetSizeTrait, RunArray, StringArray, StringViewArray,
        StructArray, UnionArray,
    },
    buffer::Buffer,
    datatypes::{DataType, Int32Type},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryReservation;

use super::super::{JoinSide, encode_join_key_v1, event_time_at, state_row_charge_with_key};
use super::{geometry::invalid, history::Locator, ipc::accounting};
use crate::{EventTime, Result};

pub(super) struct ValidationContext<'a> {
    pub(super) key_indices: &'a [usize],
    pub(super) event_time_index: usize,
    pub(super) name: &'a str,
    pub(super) side: JoinSide,
    pub(super) workspace: &'a MemoryReservation,
    pub(super) check: &'a dyn Fn() -> Result<()>,
}

pub(super) struct ValidatedRow {
    pub(super) event_time: EventTime,
    pub(super) encoded_key: Vec<u8>,
}

pub(super) fn validate(
    record: &RecordBatch,
    row: usize,
    locator: &Locator<'_>,
    context: &ValidationContext<'_>,
) -> Result<ValidatedRow> {
    (context.check)()?;
    accounting::reserve(context.workspace, diagnostic_credit(context.name)?)?;
    let encoded_key = validated_key(record, row, locator, context)?;
    let event_time = validated_time(record, row, locator, context)?;
    validate_charge(record, row, encoded_key.len(), locator, context)?;
    (context.check)()?;
    Ok(ValidatedRow {
        event_time,
        encoded_key,
    })
}

fn validated_key(
    record: &RecordBatch,
    row: usize,
    locator: &Locator<'_>,
    context: &ValidationContext<'_>,
) -> Result<Vec<u8>> {
    let (length, temporary) = key_layout(record, row, context)?;
    reserve_key(length, temporary, context)?;
    (context.check)()?;
    let key = encode_join_key_v1(record, row, context.key_indices)
        .map_err(|_| invalid("V2 historical key encoding failed"))?;
    (context.check)()?;
    if key.as_slice() != locator.key {
        return Err(invalid("V2 historical key differs from its payload"));
    }
    Ok(key)
}

fn reserve_key(length: usize, temporary: usize, context: &ValidationContext<'_>) -> Result<()> {
    accounting::reserve(
        context.workspace,
        accounting::add(bulk_vec_peak::<u8>(length)?, temporary)?,
    )
}

fn key_layout(
    record: &RecordBatch,
    row: usize,
    context: &ValidationContext<'_>,
) -> Result<(usize, usize)> {
    context
        .key_indices
        .iter()
        .try_fold((0, 0), |(length, temporary), &index| {
            (context.check)()?;
            let array = column(record, index, row)?;
            let value = key_value_length(array, row)?;
            let block = key_block_length(array.data_type(), value)?;
            Ok((accounting::add(length, block)?, temporary.max(value)))
        })
}

fn key_value_length(array: &dyn Array, row: usize) -> Result<usize> {
    if array.is_null(row) {
        return Err(invalid("V2 historical key cannot contain nulls"));
    }
    match array.data_type() {
        DataType::Boolean => Ok(1),
        DataType::Utf8 => Ok(typed::<StringArray>(array)?.value(row).len()),
        DataType::LargeUtf8 => Ok(typed::<LargeStringArray>(array)?.value(row).len()),
        data_type => data_type
            .primitive_width()
            .ok_or_else(|| invalid("V2 historical key type has no V1 width")),
    }
}

fn key_block_length(data_type: &DataType, value: usize) -> Result<usize> {
    let timezone = match data_type {
        DataType::Timestamp(_, Some(timezone)) => timezone.len(),
        _ => 0,
    };
    u32::try_from(timezone).map_err(|_| invalid("V2 key timezone exceeds u32"))?;
    u32::try_from(value).map_err(|_| invalid("V2 key value exceeds u32"))?;
    accounting::sum(&[9, timezone, value])
}

fn validated_time(
    record: &RecordBatch,
    row: usize,
    locator: &Locator<'_>,
    context: &ValidationContext<'_>,
) -> Result<EventTime> {
    column(record, context.event_time_index, row)?;
    (context.check)()?;
    let time = event_time_at(
        record,
        context.event_time_index,
        row,
        context.name,
        context.side.as_str(),
    )
    .map_err(|_| invalid("V2 historical event time conversion failed"))?
    .ok_or_else(|| invalid("V2 historical event time cannot be null"))?;
    (context.check)()?;
    if time.as_micros() != locator.time {
        return Err(invalid("V2 historical time differs from its payload"));
    }
    Ok(time)
}

fn validate_charge(
    record: &RecordBatch,
    row: usize,
    key_bytes: usize,
    locator: &Locator<'_>,
    context: &ValidationContext<'_>,
) -> Result<()> {
    let scratch = record.columns().iter().try_fold(0, |bytes, array| {
        accounting::add(bytes, charge_scratch(array.as_ref(), row, context.check)?)
    })?;
    accounting::reserve(context.workspace, scratch)?;
    (context.check)()?;
    let charge = state_row_charge_with_key(record, row, key_bytes, context.name)
        .map_err(|_| invalid("V2 historical charge decoding failed"))?;
    (context.check)()?;
    if charge != locator.charge {
        return Err(invalid("V2 historical charge differs from its payload"));
    }
    Ok(())
}

fn column(record: &RecordBatch, index: usize, row: usize) -> Result<&dyn Array> {
    let array = record
        .columns()
        .get(index)
        .ok_or_else(|| invalid("V2 historical column index is out of range"))?;
    if row >= array.len() {
        return Err(invalid("V2 historical row index is out of range"));
    }
    Ok(array.as_ref())
}

fn typed<T: Array + 'static>(array: &dyn Array) -> Result<&T> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| invalid("V2 historical array does not match its type"))
}

struct Selection<'a> {
    values: &'a dyn Array,
    range: Range<usize>,
}

fn charge_scratch(array: &dyn Array, row: usize, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    check()?;
    if row >= array.len() {
        return Err(invalid("V2 nested historical row is out of range"));
    }
    if array.is_null(row) {
        return Ok(0);
    }
    if let Some(mut selection) = list_selection(array, row)? {
        let controls = slice_controls(selection.values, check)?;
        return selection.range.try_fold(controls, |bytes, index| {
            accounting::add(bytes, charge_scratch(selection.values, index, check)?)
        });
    }
    scalar_children_scratch(array, row, check)
}

fn scalar_children_scratch(
    array: &dyn Array,
    row: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    match array.data_type() {
        DataType::Struct(_) => typed::<StructArray>(array)?
            .columns()
            .iter()
            .try_fold(0, |bytes, child| {
                accounting::add(bytes, charge_scratch(child.as_ref(), row, check)?)
            }),
        DataType::Union(_, _) => union_scratch(typed::<UnionArray>(array)?, row, check),
        DataType::Dictionary(_, _) => dictionary_scratch(array, row, check),
        DataType::RunEndEncoded(..) => run_scratch(array, row, check),
        _ => Ok(0),
    }
}

fn union_scratch(array: &UnionArray, row: usize, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    charge_scratch(
        array.child(array.type_id(row)).as_ref(),
        array.value_offset(row),
        check,
    )
}

fn dictionary_scratch(
    array: &dyn Array,
    row: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let array = typed::<DictionaryArray<Int32Type>>(array)?;
    let index = usize::try_from(array.keys().value(row))
        .map_err(|_| invalid("V2 historical dictionary key is negative"))?;
    charge_scratch(array.values().as_ref(), index, check)
}

fn run_scratch(array: &dyn Array, row: usize, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    let array = typed::<RunArray<Int32Type>>(array)?;
    charge_scratch(
        array.values().as_ref(),
        array.get_physical_index(row),
        check,
    )
}

fn list_selection(array: &dyn Array, row: usize) -> Result<Option<Selection<'_>>> {
    let selection = match array.data_type() {
        DataType::List(_) => select_list(array.as_any().downcast_ref::<ListArray>(), row),
        DataType::LargeList(_) => select_list(array.as_any().downcast_ref::<LargeListArray>(), row),
        DataType::ListView(_) => select_view(array.as_any().downcast_ref::<ListViewArray>(), row),
        DataType::LargeListView(_) => {
            select_view(array.as_any().downcast_ref::<LargeListViewArray>(), row)
        }
        DataType::Map(_, _) => select_map(array, row),
        DataType::FixedSizeList(_, _) => select_fixed(array, row),
        _ => return Ok(None),
    };
    selection.map(Some)
}

fn select_list<O: OffsetSizeTrait>(
    array: Option<&GenericListArray<O>>,
    row: usize,
) -> Result<Selection<'_>> {
    let array = array.ok_or_else(|| invalid("V2 historical list type mismatch"))?;
    selection(
        array.values().as_ref(),
        offsets_range(array.value_offsets(), row)?,
    )
}

fn select_view<O: OffsetSizeTrait>(
    array: Option<&GenericListViewArray<O>>,
    row: usize,
) -> Result<Selection<'_>> {
    let array = array.ok_or_else(|| invalid("V2 historical list-view type mismatch"))?;
    selection(
        array.values().as_ref(),
        view_range(array.value_offsets(), array.value_sizes(), row)?,
    )
}

fn select_map(array: &dyn Array, row: usize) -> Result<Selection<'_>> {
    let array = typed::<MapArray>(array)?;
    selection(array.entries(), offsets_range(array.value_offsets(), row)?)
}

fn select_fixed(array: &dyn Array, row: usize) -> Result<Selection<'_>> {
    let array = typed::<FixedSizeListArray>(array)?;
    let width = usize::try_from(array.value_length())
        .map_err(|_| invalid("V2 fixed-size list width is negative"))?;
    let start = accounting::product(row, width)?;
    let end = accounting::add(start, width)?;
    selection(array.values().as_ref(), start..end)
}

fn offsets_range<O: OffsetSizeTrait>(offsets: &[O], row: usize) -> Result<Range<usize>> {
    let next = accounting::add(row, 1)?;
    let start = offset(offsets, row)?;
    let end = offset(offsets, next)?;
    Ok(start..end)
}

fn view_range<O: OffsetSizeTrait>(offsets: &[O], sizes: &[O], row: usize) -> Result<Range<usize>> {
    let start = offset(offsets, row)?;
    let end = accounting::add(start, offset(sizes, row)?)?;
    Ok(start..end)
}

fn offset<O: OffsetSizeTrait>(offsets: &[O], row: usize) -> Result<usize> {
    let value = offsets
        .get(row)
        .ok_or_else(|| invalid("V2 historical list offset is missing"))?;
    value
        .to_usize()
        .ok_or_else(|| invalid("V2 historical list offset is negative or exceeds usize"))
}

fn selection(values: &dyn Array, range: Range<usize>) -> Result<Selection<'_>> {
    if range.start > range.end || range.end > values.len() {
        return Err(invalid("V2 historical list range is invalid"));
    }
    Ok(Selection { values, range })
}

fn slice_controls(array: &dyn Array, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    let (nodes, buffers, variadic) = slice_geometry(array, check)?;
    accounting::sum(&[
        accounting::array_controls(nodes, buffers)?,
        accounting::data_type_bytes(array.data_type())?,
        // ByteView to_data can clone an exact-capacity Vec, then insert its views buffer.
        bulk_vec_peak::<Buffer>(accounting::add(variadic, 1)?)?,
    ])
}

fn slice_geometry(
    array: &dyn Array,
    check: &dyn Fn() -> Result<()>,
) -> Result<(usize, usize, usize)> {
    let nodes = accounting::shape_nodes(array.data_type())?;
    let variadic = variadic_buffers(array, check)?;
    let buffers = accounting::add(accounting::product(nodes, 3)?, variadic)?;
    Ok((nodes, buffers, variadic))
}

fn variadic_buffers(array: &dyn Array, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    check()?;
    match array.data_type() {
        DataType::Utf8View => Ok(typed::<StringViewArray>(array)?.data_buffers().len()),
        DataType::BinaryView => Ok(typed::<BinaryViewArray>(array)?.data_buffers().len()),
        _ => child_variadic(array, check),
    }
}

fn child_variadic(array: &dyn Array, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    if let Some(child) = list_values(array) {
        return variadic_buffers(child, check);
    }
    match array.data_type() {
        DataType::Struct(_) => typed::<StructArray>(array)?
            .columns()
            .iter()
            .try_fold(0, |bytes, child| {
                accounting::add(bytes, variadic_buffers(child.as_ref(), check)?)
            }),
        DataType::Union(fields, _) => union_variadic(typed::<UnionArray>(array)?, fields, check),
        DataType::Dictionary(_, _) => variadic_buffers(
            typed::<DictionaryArray<Int32Type>>(array)?
                .values()
                .as_ref(),
            check,
        ),
        DataType::RunEndEncoded(..) => variadic_buffers(
            typed::<RunArray<Int32Type>>(array)?.values().as_ref(),
            check,
        ),
        _ => Ok(0),
    }
}

fn union_variadic(
    array: &UnionArray,
    fields: &datafusion::arrow::datatypes::UnionFields,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    fields.iter().try_fold(0, |bytes, (id, _)| {
        accounting::add(bytes, variadic_buffers(array.child(id).as_ref(), check)?)
    })
}

fn list_values(array: &dyn Array) -> Option<&dyn Array> {
    match array.data_type() {
        DataType::List(_) => array
            .as_any()
            .downcast_ref::<ListArray>()
            .map(|array| array.values().as_ref()),
        DataType::LargeList(_) => array
            .as_any()
            .downcast_ref::<LargeListArray>()
            .map(|array| array.values().as_ref()),
        DataType::ListView(_) => array
            .as_any()
            .downcast_ref::<ListViewArray>()
            .map(|array| array.values().as_ref()),
        DataType::LargeListView(_) => array
            .as_any()
            .downcast_ref::<LargeListViewArray>()
            .map(|array| array.values().as_ref()),
        DataType::Map(_, _) => array
            .as_any()
            .downcast_ref::<MapArray>()
            .map(|array| array.entries() as &dyn Array),
        DataType::FixedSizeList(_, _) => array
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .map(|array| array.values().as_ref()),
        _ => None,
    }
}

fn bulk_vec_peak<T>(length: usize) -> Result<usize> {
    let minimum = if size_of::<T>() == 1 { 8 } else { 4 };
    // A bulk-growth request has old <= final length and new <= twice final length.
    accounting::product(accounting::product(length, 3)?.max(minimum), size_of::<T>())
}

fn diagnostic_credit(name: &str) -> Result<usize> {
    let bytes = accounting::sum(&[
        name.len(),
        2 * "stream_join.right_event_time".len(),
        2 * "right event time cannot be represented".len(),
        "timestamp value overflows the microsecond event-time range".len(),
        2 * "logical payload bytes counter overflowed".len(),
        "int8 key does not fit u8".len(),
        "FixedSizeList array type mismatch in logical charge".len(),
        "V2 historical array does not match its type".len(),
        "V2 historical charge differs from its payload".len(),
    ])?;
    bulk_vec_peak::<u8>(bytes)
}
