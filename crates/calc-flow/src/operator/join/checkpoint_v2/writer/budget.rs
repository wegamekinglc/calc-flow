use arrow_data::{ArrayData, BufferSpec};
use datafusion::arrow::{
    array::{
        Array, AsArray, BinaryViewArray, FixedSizeListArray, GenericListArray,
        GenericListViewArray, MapArray, RunArray, StringViewArray, StructArray, UnionArray,
    },
    datatypes::{DataType, Field, Int16Type, Int32Type, Int64Type, RunEndIndexType},
    ipc::{Buffer as IpcBuffer, FieldNode, writer::EncodedData},
    record_batch::RecordBatch,
};

use super::super::ipc::accounting::{
    add, array_controls, product, schema_bytes, shape_nodes, sum, vector_peak,
};
use crate::{CalcFlowError, Result};

#[derive(Clone, Copy, Default)]
struct Arrays {
    nodes: usize,
    buffers: usize,
    dictionaries: usize,
    validity: usize,
    normalization: usize,
}

#[derive(Clone, Copy, Default)]
struct Schema {
    entries: usize,
    strings: usize,
    text: usize,
}

pub(super) fn diagnostic_bytes() -> usize {
    "V2 IPC writer array representation is inconsistent"
        .len()
        .max("V2 IPC writer constructor layout overflow".len())
}

pub(super) fn stream_scratch(record: &RecordBatch) -> Result<usize> {
    count_i64(record.num_rows())?;
    let arrays = record_arrays(record)?;
    let nodes = schema_nodes(record)?;
    let backing = record_backing(record)?;
    let bytes = scratch_components(record, arrays, nodes, backing)?;
    allocation_size(bytes)
}

fn scratch_components(
    record: &RecordBatch,
    arrays: Arrays,
    nodes: usize,
    backing: usize,
) -> Result<usize> {
    sum(&[
        schema_scratch(record, nodes)?,
        tracker_scratch(arrays, nodes)?,
        descriptor_scratch(record, arrays)?,
        message_scratch(arrays, backing)?,
    ])
}

fn allocation_size(bytes: usize) -> Result<usize> {
    if bytes > isize::MAX as usize {
        return Err(overflow());
    }
    Ok(bytes)
}

fn record_backing(record: &RecordBatch) -> Result<usize> {
    record
        .columns()
        .iter()
        .try_fold(0, |total, array| add(total, array.get_buffer_memory_size()))
}

fn record_arrays(record: &RecordBatch) -> Result<Arrays> {
    record
        .columns()
        .iter()
        .try_fold(Arrays::default(), |left, array| {
            merge_arrays(left, array_facts(array.as_ref())?)
        })
}

fn schema_nodes(record: &RecordBatch) -> Result<usize> {
    record
        .schema_ref()
        .fields()
        .iter()
        .try_fold(0, |total, field| {
            add(total, shape_nodes(field.data_type())?)
        })
}

fn merge_arrays(left: Arrays, right: Arrays) -> Result<Arrays> {
    Ok(Arrays {
        nodes: add(left.nodes, right.nodes)?,
        buffers: add(left.buffers, right.buffers)?,
        dictionaries: add(left.dictionaries, right.dictionaries)?,
        validity: add(left.validity, right.validity)?,
        normalization: add(left.normalization, right.normalization)?,
    })
}

fn array_facts(array: &dyn Array) -> Result<Arrays> {
    count_i64(array.len())?;
    let own = Arrays {
        nodes: 1,
        buffers: add(3, view_buffers(array)?)?,
        dictionaries: usize::from(matches!(array.data_type(), DataType::Dictionary(_, _))),
        validity: validity_bytes(array)?,
        normalization: normalization_bytes(array)?,
    };
    merge_arrays(own, array_children(array)?)
}

fn validity_bytes(array: &dyn Array) -> Result<usize> {
    match array.data_type() {
        DataType::Null | DataType::Union(_, _) | DataType::RunEndEncoded(_, _) => Ok(0),
        _ => rounded(add(array.len(), 7)? / 8, 64),
    }
}

fn normalization_bytes(array: &dyn Array) -> Result<usize> {
    let offsets = match array.data_type() {
        DataType::Binary | DataType::Utf8 | DataType::List(_) | DataType::Map(_, _) => 4,
        DataType::LargeBinary | DataType::LargeUtf8 | DataType::LargeList(_) => 8,
        DataType::Boolean => return rounded(add(array.len(), 7)? / 8, 64),
        _ => return Ok(0),
    };
    rounded(product(add(array.len(), 1)?, offsets)?, 64)
}

fn view_buffers(array: &dyn Array) -> Result<usize> {
    match array.data_type() {
        DataType::Utf8View => Ok(typed::<StringViewArray>(array)?.data_buffers().len()),
        DataType::BinaryView => Ok(typed::<BinaryViewArray>(array)?.data_buffers().len()),
        _ => Ok(0),
    }
}

fn array_children(array: &dyn Array) -> Result<Arrays> {
    if let Some(child) = unary(array)? {
        return array_facts(child);
    }
    match array.data_type() {
        DataType::Struct(_) => typed::<StructArray>(array)?
            .columns()
            .iter()
            .try_fold(Arrays::default(), |total, child| {
                merge_arrays(total, array_facts(child.as_ref())?)
            }),
        DataType::Union(fields, _) => union_children(array, fields),
        DataType::Dictionary(_, _) => dictionary_child(array),
        DataType::RunEndEncoded(run_ends, _) => run_children(array, run_ends.data_type()),
        _ => Ok(Arrays::default()),
    }
}

fn unary(array: &dyn Array) -> Result<Option<&dyn Array>> {
    if let Some(child) = list_child(array)? {
        return Ok(Some(child));
    }
    Ok(match array.data_type() {
        DataType::ListView(_) => Some(typed::<GenericListViewArray<i32>>(array)?.values().as_ref()),
        DataType::LargeListView(_) => {
            Some(typed::<GenericListViewArray<i64>>(array)?.values().as_ref())
        }
        DataType::FixedSizeList(_, _) => {
            Some(typed::<FixedSizeListArray>(array)?.values().as_ref())
        }
        DataType::Map(_, _) => Some(typed::<MapArray>(array)?.entries()),
        _ => None,
    })
}

fn list_child(array: &dyn Array) -> Result<Option<&dyn Array>> {
    Ok(match array.data_type() {
        DataType::List(_) => Some(typed::<GenericListArray<i32>>(array)?.values().as_ref()),
        DataType::LargeList(_) => Some(typed::<GenericListArray<i64>>(array)?.values().as_ref()),
        _ => None,
    })
}

fn union_children(
    array: &dyn Array,
    fields: &datafusion::arrow::datatypes::UnionFields,
) -> Result<Arrays> {
    let array = typed::<UnionArray>(array)?;
    fields.iter().try_fold(Arrays::default(), |total, (id, _)| {
        merge_arrays(total, array_facts(array.child(id).as_ref())?)
    })
}

fn dictionary_child(array: &dyn Array) -> Result<Arrays> {
    let dictionary = array.as_any_dictionary_opt().ok_or_else(invalid_array)?;
    array_facts(dictionary.values().as_ref())
}

fn run_children(array: &dyn Array, run_ends: &DataType) -> Result<Arrays> {
    match run_ends {
        DataType::Int16 => run_facts::<Int16Type>(array),
        DataType::Int32 => run_facts::<Int32Type>(array),
        DataType::Int64 => run_facts::<Int64Type>(array),
        _ => Err(invalid_array()),
    }
}

fn run_facts<R: RunEndIndexType>(array: &dyn Array) -> Result<Arrays> {
    let array = typed::<RunArray<R>>(array)?;
    let count = array.run_ends().values().len();
    count_i64(count)?;
    let run_ends = Arrays {
        nodes: 1,
        buffers: 3,
        dictionaries: 0,
        validity: rounded(add(count, 7)? / 8, 64)?,
        normalization: rounded(product(count, size_of::<R::Native>())?, 64)?,
    };
    merge_arrays(run_ends, array_facts(array.values().as_ref())?)
}

fn schema_scratch(record: &RecordBatch, nodes: usize) -> Result<usize> {
    let schema = record_schema(record)?;
    let tables = sum(&[2, product(nodes, 4)?, schema.entries])?;
    sum(&[
        flatbuffer_scratch(schema_flatbuffer(schema, nodes)?, tables)?,
        schema_auxiliary(nodes, schema.entries)?,
    ])
}

fn record_schema(record: &RecordBatch) -> Result<Schema> {
    let metadata = map_schema(record.schema_ref().metadata())?;
    record
        .schema_ref()
        .fields()
        .iter()
        .try_fold(metadata, |total, field| {
            merge_schema(total, field_schema(field)?)
        })
}

fn schema_flatbuffer(schema: Schema, nodes: usize) -> Result<usize> {
    sum(&[
        schema_tables(nodes, schema.entries)?,
        schema.text,
        product(schema.strings, 8)?,
        schema_vectors(nodes, schema.entries)?,
        11,
    ])
}

fn schema_tables(nodes: usize, entries: usize) -> Result<usize> {
    let field = field_tables()?;
    sum(&[
        fb_table(4)?,
        fb_table(5)?,
        product(nodes, field)?,
        key_value_tables(entries)?,
    ])
}

fn field_tables() -> Result<usize> {
    sum(&[fb_table(7)?, fb_table(3)?, fb_table(2)?, fb_table(4)?])
}

fn key_value_tables(entries: usize) -> Result<usize> {
    product(entries, fb_table(2)?)
}

fn merge_schema(left: Schema, right: Schema) -> Result<Schema> {
    Ok(Schema {
        entries: add(left.entries, right.entries)?,
        strings: add(left.strings, right.strings)?,
        text: add(left.text, right.text)?,
    })
}

fn map_schema(metadata: &std::collections::HashMap<String, String>) -> Result<Schema> {
    let text = metadata.iter().try_fold(0, |total, (key, value)| {
        sum(&[total, key.len(), value.len()])
    })?;
    Ok(Schema {
        entries: metadata.len(),
        strings: product(metadata.len(), 2)?,
        text,
    })
}

fn field_schema(field: &Field) -> Result<Schema> {
    let own = merge_schema(
        map_schema(field.metadata())?,
        Schema {
            entries: 0,
            strings: 1,
            text: field.name().len(),
        },
    )?;
    merge_schema(own, type_schema(field.data_type())?)
}

fn type_schema(data_type: &DataType) -> Result<Schema> {
    if let Some(field) = unary_field(data_type) {
        return field_schema(field);
    }
    match data_type {
        DataType::Struct(fields) => fields.iter().try_fold(Schema::default(), |total, field| {
            merge_schema(total, field_schema(field)?)
        }),
        DataType::Union(fields, _) => fields
            .iter()
            .try_fold(Schema::default(), |total, (_, field)| {
                merge_schema(total, field_schema(field)?)
            }),
        DataType::Dictionary(_, value) => type_schema(value),
        DataType::RunEndEncoded(ends, values) => {
            merge_schema(field_schema(ends)?, field_schema(values)?)
        }
        DataType::Timestamp(_, timezone) => Ok(Schema {
            entries: 0,
            strings: 1,
            text: timezone.as_ref().map_or(0, |text| text.len()),
        }),
        _ => Ok(Schema::default()),
    }
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

fn schema_vectors(nodes: usize, entries: usize) -> Result<usize> {
    let groups = add(nodes, 1)?;
    sum(&[
        vector_groups(groups, nodes, 4)?,
        vector_groups(groups, entries, 4)?,
        vector_groups(nodes, nodes, 4)?,
    ])
}

fn vector_groups(groups: usize, length: usize, width: usize) -> Result<usize> {
    add(product(groups, 11)?, product(length, width)?)
}

fn schema_auxiliary(nodes: usize, entries: usize) -> Result<usize> {
    sum(&[
        // Every recursive fields/_fields result and the outer flattened_fields result.
        schema_recursive_vectors::<&Field>(nodes)?,
        // Schema/Struct child-offset collection and Union type-ID collection.
        schema_recursive_vectors::<usize>(nodes)?,
        // Metadata keys, stable-sort scratch, and KeyValue offsets.
        product(3, bulk_peak::<&String>(entries)?)?,
    ])
}

fn schema_recursive_vectors<T>(nodes: usize) -> Result<usize> {
    product(add(product(nodes, 2)?, 1)?, bulk_peak::<T>(nodes)?)
}

fn tracker_scratch(arrays: Arrays, schema_nodes: usize) -> Result<usize> {
    sum(&[
        vector_peak::<i64>(arrays.dictionaries)?,
        product(arrays.dictionaries, size_of::<i64>())?,
        hash_peak::<(i64, ArrayData)>(arrays.dictionaries)?,
        // encode allocates this capacity from flattened schema fields, even with no dictionaries.
        bulk_peak::<EncodedData>(schema_nodes)?,
    ])
}

fn descriptor_scratch(record: &RecordBatch, arrays: Arrays) -> Result<usize> {
    // Tracker-retained graphs, recursive dictionary to_data stack, record/normalization graph.
    let graphs = sum(&[arrays.dictionaries, arrays.nodes, 1])?;
    let clone_calls = product(4, arrays.nodes)?;
    let graph = sum(&[
        array_controls(arrays.nodes, arrays.buffers)?,
        product(clone_calls, schema_bytes(record.schema_ref())?)?,
        // View to_data clones an exact Vec and then inserts the views buffer.
        view_descriptor_scratch(arrays)?,
    ])?;
    product(graphs, graph)
}

fn view_descriptor_scratch(arrays: Arrays) -> Result<usize> {
    let vector = bulk_peak::<datafusion::arrow::buffer::Buffer>(add(arrays.buffers, 1)?)?;
    product(arrays.nodes, vector)
}

fn message_scratch(arrays: Arrays, backing: usize) -> Result<usize> {
    let body = sum(&[
        backing,
        arrays.validity,
        arrays.normalization,
        product(arrays.buffers, 7)?,
    ])?;
    count_i64(body)?;
    let one = encoded_message_scratch(arrays, body)?;
    // encode returns all dictionary bodies and the record before any output writes.
    product(add(arrays.dictionaries, 1)?, one)
}

fn encoded_message_scratch(arrays: Arrays, body: usize) -> Result<usize> {
    let one = sum(&[
        bulk_peak::<u8>(body)?,
        arrays.validity,
        arrays.normalization,
        // New immutable Buffer's Arc<Bytes> header, including both Arc counters.
        product(arrays.buffers, 7 * size_of::<usize>())?,
        message_vectors(arrays)?,
        // get_or_truncate_buffer creates one layout Vec with at most two BufferSpecs.
        product(2, size_of::<BufferSpec>())?,
        flatbuffer_scratch(message_metadata(arrays)?, 3)?,
    ])?;
    Ok(one)
}

fn message_vectors(arrays: Arrays) -> Result<usize> {
    sum(&[
        vector_peak::<FieldNode>(arrays.nodes)?,
        vector_peak::<IpcBuffer>(arrays.buffers)?,
        vector_peak::<i64>(arrays.nodes)?,
    ])
}

fn message_metadata(arrays: Arrays) -> Result<usize> {
    sum(&[
        message_tables()?,
        fb_vector(arrays.nodes, 16)?,
        fb_vector(arrays.buffers, 16)?,
        fb_vector(arrays.nodes, 8)?,
        11,
    ])
}

fn message_tables() -> Result<usize> {
    sum(&[fb_table(5)?, fb_table(5)?, fb_table(3)?])
}

fn fb_table(slots: usize) -> Result<usize> {
    sum(&[4, product(slots, 8)?, 4, product(slots, 2)?, 7])
}

fn fb_vector(length: usize, width: usize) -> Result<usize> {
    sum(&[4, product(length, width)?, 7])
}

fn flatbuffer_scratch(serialized: usize, tables: usize) -> Result<usize> {
    let capacity = serialized
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(overflow)?;
    if capacity >= 1_usize << 31 {
        return Err(overflow());
    }
    sum(&[
        vector_peak::<u8>(serialized)?,
        serialized,
        vector_peak::<(u16, u32)>(7)?,
        vector_peak::<u32>(tables)?,
    ])
}

fn bulk_peak<T>(length: usize) -> Result<usize> {
    if length == 0 || size_of::<T>() == 0 {
        return Ok(0);
    }
    let minimum = if size_of::<T>() == 1 { 8 } else { 4 };
    // For bulk extend: old <= final length and new <= max(2*final length, minimum).
    product(product(length, 3)?.max(minimum), size_of::<T>())
}

fn hash_peak<T>(length: usize) -> Result<usize> {
    if length == 0 {
        return Ok(0);
    }
    let requested = add(product(length, 8)? / 7, 1)?.max(4);
    let buckets = requested.checked_next_power_of_two().ok_or_else(overflow)?;
    let old = if buckets <= 4 { 0 } else { buckets / 2 };
    let slots = add(buckets, old)?;
    sum(&[
        product(slots, size_of::<T>())?,
        slots,
        32,
        2 * (align_of::<T>() - 1),
    ])
}

fn rounded(bytes: usize, alignment: usize) -> Result<usize> {
    let bytes = product(add(bytes, alignment - 1)? / alignment, alignment)?;
    if bytes > isize::MAX as usize {
        return Err(overflow());
    }
    Ok(bytes)
}

fn count_i64(count: usize) -> Result<()> {
    i64::try_from(count).map(|_| ()).map_err(|_| overflow())
}

fn typed<T: Array + 'static>(array: &dyn Array) -> Result<&T> {
    array.as_any().downcast_ref().ok_or_else(invalid_array)
}

fn invalid_array() -> CalcFlowError {
    super::super::geometry::invalid("V2 IPC writer array representation is inconsistent")
}

fn overflow() -> CalcFlowError {
    super::super::geometry::invalid("V2 IPC writer constructor layout overflow")
}
