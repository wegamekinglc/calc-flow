use std::collections::HashMap;

use arrow_data::{ArrayData, ArrayDataBuilder};
use datafusion::arrow::{
    array::{
        ArrayRef, BooleanArray, DictionaryArray, FixedSizeBinaryArray, FixedSizeListArray,
        GenericListArray, GenericListViewArray, GenericStringArray, MapArray, NullArray,
        PrimitiveArray, RunArray, StringViewArray, StructArray, UnionArray,
    },
    buffer::{Buffer, MutableBuffer},
    datatypes::{DataType, Field, FieldRef, Int64Type, Schema, UnionFields},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryReservation;

use super::super::geometry::invalid;
use super::Plan;
use crate::{CalcFlowError, Result};

pub(in crate::operator::join::checkpoint_v2) fn reserve(
    credit: &MemoryReservation,
    bytes: usize,
) -> Result<()> {
    credit.try_grow(bytes).map_err(|_| CalcFlowError::Internal {
        message: "V2 IPC constructor credit admission failed".into(),
    })
}

pub(in crate::operator::join::checkpoint_v2) fn schema_bytes(schema: &Schema) -> Result<usize> {
    sum(&[
        arc::<Schema>()?,
        fields_bytes(
            schema.fields().len(),
            schema.fields().iter().map(AsRef::as_ref),
        )?,
        map_bytes(schema.metadata())?,
    ])
}

fn fields_bytes<'a>(count: usize, fields: impl Iterator<Item = &'a Field>) -> Result<usize> {
    let mut bytes = sum(&[
        vector_peak::<Field>(count)?,
        vector_peak::<FieldRef>(count)?,
        arc_slice::<FieldRef>(count)?,
    ])?;
    for field in fields {
        bytes = add(bytes, field_bytes(field)?)?;
    }
    Ok(bytes)
}

fn field_bytes(field: &Field) -> Result<usize> {
    sum(&[
        arc::<Field>()?,
        field.name().len(),
        map_bytes(field.metadata())?,
        data_type_bytes(field.data_type())?,
    ])
}

pub(in crate::operator::join::checkpoint_v2) fn data_type_bytes(
    data_type: &DataType,
) -> Result<usize> {
    if let Some(field) = unary_field(data_type) {
        return field_bytes(field);
    }
    match data_type {
        DataType::Dictionary(key, value) => dictionary_type_bytes(key, value),
        DataType::Struct(fields) => fields_bytes(fields.len(), fields.iter().map(AsRef::as_ref)),
        DataType::Union(fields, _) => union_type_bytes(fields),
        DataType::RunEndEncoded(run_ends, values) => {
            add(field_bytes(run_ends)?, field_bytes(values)?)
        }
        DataType::Timestamp(_, Some(timezone)) => {
            add(timezone.len(), arc_layout(timezone.len(), 1)?)
        }
        _ => Ok(0),
    }
}

fn dictionary_type_bytes(key: &DataType, value: &DataType) -> Result<usize> {
    sum(&[
        product(2, size_of::<DataType>())?,
        data_type_bytes(key)?,
        data_type_bytes(value)?,
    ])
}

fn union_type_bytes(fields: &UnionFields) -> Result<usize> {
    sum(&[
        fields_bytes(fields.len(), fields.iter().map(|(_, field)| field.as_ref()))?,
        vector_peak::<(i8, FieldRef)>(fields.len())?,
        vector_peak::<i8>(fields.len())?,
        arc_slice::<(i8, FieldRef)>(fields.len())?,
    ])
}

fn unary_field(data_type: &DataType) -> Option<&Field> {
    match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => Some(field),
        _ => None,
    }
}

fn map_bytes(metadata: &HashMap<String, String>) -> Result<usize> {
    let table = hash_table_peak::<(String, String)>(metadata.len())?;
    metadata.iter().try_fold(table, |bytes, (key, value)| {
        sum(&[bytes, key.len(), value.len()])
    })
}

pub(in crate::operator::join::checkpoint_v2) fn reader_controls(
    plan: &Plan<'_>,
    schema: &Schema,
) -> Result<usize> {
    let arrays = reader_array_controls(plan, schema)?;
    sum(&[
        schema_bytes(schema)?,
        arrays,
        hash_table_peak::<(i64, ArrayRef)>(plan.dictionary_messages)?,
        vector_peak::<i64>(plan.variadic_counts)?,
        vector_peak::<ArrayRef>(schema.fields().len())?,
        size_of::<RecordBatch>(),
        size_of::<MutableBuffer>(),
        // Every selected buffer may need an independent aligned copy.
        plan.buffer_bytes,
        product(plan.buffers, 63)?,
    ])
}

fn reader_array_controls(plan: &Plan<'_>, schema: &Schema) -> Result<usize> {
    let schema_nodes = schema_shape_nodes(schema)?;
    let expanded = plan
        .nodes
        .max(product(schema_nodes, add(plan.dictionary_messages, 1)?)?);
    let buffer_slots = add(plan.buffers, product(expanded, 3)?)?;
    array_controls(expanded, buffer_slots)
}

fn schema_shape_nodes(schema: &Schema) -> Result<usize> {
    schema.fields().iter().try_fold(0, |nodes, field| {
        add(nodes, shape_nodes(field.data_type())?)
    })
}

pub(in crate::operator::join::checkpoint_v2) fn dictionary_schema_bytes(
    data_type: &DataType,
) -> Result<usize> {
    sum(&[
        arc::<Schema>()?,
        arc::<Field>()?,
        arc_slice::<FieldRef>(1)?,
        size_of::<Field>(),
        size_of::<FieldRef>(),
        data_type_bytes(data_type)?,
    ])
}

pub(in crate::operator::join::checkpoint_v2) fn body_backing(plan: &Plan<'_>) -> Result<usize> {
    sum(&[
        plan.body_bytes,
        product(add(plan.dictionary_messages, 1)?, 7 * size_of::<usize>())?,
        plan.buffer_bytes,
        product(plan.buffers, 63 + 7 * size_of::<usize>())?,
    ])
}

pub(in crate::operator::join::checkpoint_v2) fn empty_dictionary_backing(
    schema: &Schema,
) -> Result<usize> {
    schema.fields().iter().try_fold(0, |bytes, field| {
        add(bytes, missing_values_backing(field.data_type())?)
    })
}

fn missing_values_backing(data_type: &DataType) -> Result<usize> {
    if let DataType::Dictionary(_, values) = data_type {
        return empty_backing(values);
    }
    child_backing(data_type, missing_values_backing)
}

fn empty_backing(data_type: &DataType) -> Result<usize> {
    let offset = match data_type {
        DataType::Utf8
        | DataType::Binary
        | DataType::LargeUtf8
        | DataType::LargeBinary
        | DataType::List(_)
        | DataType::LargeList(_)
        | DataType::Map(_, _) => 64,
        _ => 0,
    };
    // Each synthesized offset requests four/eight bytes; 64 is a conservative allowance.
    add(offset, child_backing(data_type, empty_backing)?)
}

fn child_backing(data_type: &DataType, visit: fn(&DataType) -> Result<usize>) -> Result<usize> {
    if let Some(field) = unary_field(data_type) {
        return visit(field.data_type());
    }
    match data_type {
        DataType::Dictionary(_, values) => visit(values),
        DataType::Struct(fields) => fields
            .iter()
            .try_fold(0, |bytes, field| add(bytes, visit(field.data_type())?)),
        DataType::Union(fields, _) => fields
            .iter()
            .try_fold(0, |bytes, (_, field)| add(bytes, visit(field.data_type())?)),
        DataType::RunEndEncoded(run_ends, values) => {
            add(visit(run_ends.data_type())?, visit(values.data_type())?)
        }
        _ => Ok(0),
    }
}

pub(in crate::operator::join::checkpoint_v2) fn resident_controls(
    plan: &Plan<'_>,
    schema: &Schema,
) -> Result<usize> {
    let nodes = schema_shape_nodes(schema)?;
    let buffers = add(plan.buffers, product(nodes, 3)?)?;
    sum(&[
        array_controls(nodes, buffers)?,
        vector_peak::<ArrayRef>(schema.fields().len())?,
        product(buffers, super::super::payload::buffer_owner_bytes())?,
        arc::<super::super::payload::OwnedPayload>()?,
    ])
}

pub(in crate::operator::join::checkpoint_v2) fn shape_nodes(data_type: &DataType) -> Result<usize> {
    if let Some(field) = unary_field(data_type) {
        return add(1, shape_nodes(field.data_type())?);
    }
    let children = shape_children(data_type)?;
    add(1, children)
}

fn shape_children(data_type: &DataType) -> Result<usize> {
    match data_type {
        DataType::Dictionary(_, value) => shape_nodes(value),
        DataType::Struct(fields) => fields.iter().try_fold(0, |nodes, field| {
            add(nodes, shape_nodes(field.data_type())?)
        }),
        DataType::Union(fields, _) => fields.iter().try_fold(0, |nodes, (_, field)| {
            add(nodes, shape_nodes(field.data_type())?)
        }),
        DataType::RunEndEncoded(run_ends, values) => add(
            shape_nodes(run_ends.data_type())?,
            shape_nodes(values.data_type())?,
        ),
        _ => Ok(0),
    }
}

pub(in crate::operator::join::checkpoint_v2) fn array_controls(
    nodes: usize,
    buffers: usize,
) -> Result<usize> {
    let (per_array, graph_vectors, graph_copies) = array_control_geometry(nodes, buffers)?;
    sum(&[
        product(nodes, per_array)?,
        product(graph_copies, graph_vectors)?,
        product(graph_copies, size_of::<ArrayData>())?,
        vector_peak::<ArrayRef>(nodes)?,
    ])
}

fn array_control_geometry(nodes: usize, buffers: usize) -> Result<(usize, usize, usize)> {
    let per_array = sum(&[
        max_array_layout()?,
        // UnionArray materializes a lookup indexed by nonnegative i8 type ID.
        product(128, size_of::<Option<ArrayRef>>())?,
        size_of::<ArrayDataBuilder>(),
    ])?;
    let graph_vectors = add(
        vector_peak::<ArrayData>(nodes)?,
        vector_peak::<Buffer>(buffers)?,
    )?;
    // Decoder construction, to_data, clone/into_builder and canonical make_array overlap.
    let graph_copies = product(4, nodes)?;
    Ok((per_array, graph_vectors, graph_copies))
}

fn max_array_layout() -> Result<usize> {
    let controls = [
        size_of::<PrimitiveArray<Int64Type>>(),
        size_of::<BooleanArray>(),
        size_of::<NullArray>(),
        size_of::<GenericStringArray<i64>>(),
        size_of::<StringViewArray>(),
        size_of::<FixedSizeBinaryArray>(),
        size_of::<GenericListArray<i64>>(),
        size_of::<GenericListViewArray<i64>>(),
        size_of::<FixedSizeListArray>(),
        size_of::<MapArray>(),
        size_of::<StructArray>(),
        size_of::<UnionArray>(),
        size_of::<DictionaryArray<Int64Type>>(),
        size_of::<RunArray<Int64Type>>(),
    ];
    arc_layout(
        *controls.iter().max().expect("array controls"),
        align_of::<usize>(),
    )
}

pub(in crate::operator::join::checkpoint_v2) fn vector_peak<T>(length: usize) -> Result<usize> {
    if length == 0 || size_of::<T>() == 0 {
        return Ok(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    let capacity = length
        .max(minimum)
        .checked_next_power_of_two()
        .ok_or_else(overflow)?;
    let old = if length <= minimum { 0 } else { capacity / 2 };
    product(add(capacity, old)?, size_of::<T>())
}

fn hash_table_peak<T>(length: usize) -> Result<usize> {
    if length == 0 {
        return Ok(0);
    }
    let requested = add(product(length, 8)? / 7, 1)?.max(4);
    let buckets = requested.checked_next_power_of_two().ok_or_else(overflow)?;
    let old = if buckets <= 4 { 0 } else { buckets / 2 };
    hash_table_backing::<T>(buckets, old)
}

fn hash_table_backing<T>(buckets: usize, old: usize) -> Result<usize> {
    let entries = product(add(buckets, old)?, size_of::<T>())?;
    let controls = add(add(buckets, old)?, 32)?;
    sum(&[entries, controls, 2 * (align_of::<T>() - 1)])
}

pub(in crate::operator::join::checkpoint_v2) fn arc<T>() -> Result<usize> {
    arc_layout(size_of::<T>(), align_of::<T>())
}

pub(in crate::operator::join::checkpoint_v2) fn arc_slice<T>(length: usize) -> Result<usize> {
    arc_layout(product(length, size_of::<T>())?, align_of::<T>())
}

fn arc_layout(bytes: usize, alignment: usize) -> Result<usize> {
    let header = align_up(2 * size_of::<usize>(), alignment)?;
    align_up(add(header, bytes)?, alignment.max(align_of::<usize>()))
}

fn align_up(bytes: usize, alignment: usize) -> Result<usize> {
    product(add(bytes, alignment - 1)? / alignment, alignment)
}

pub(in crate::operator::join::checkpoint_v2) fn sum(bytes: &[usize]) -> Result<usize> {
    bytes.iter().try_fold(0, |total, bytes| add(total, *bytes))
}

pub(in crate::operator::join::checkpoint_v2) fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or_else(overflow)
}

pub(in crate::operator::join::checkpoint_v2) fn product(
    length: usize,
    width: usize,
) -> Result<usize> {
    length.checked_mul(width).ok_or_else(overflow)
}

fn overflow() -> CalcFlowError {
    invalid("V2 IPC constructor layout overflow")
}
