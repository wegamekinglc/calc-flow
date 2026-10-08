use datafusion::arrow::{
    datatypes::{DataType, Field, Schema, TimeUnit},
    ipc::{self, Message},
};

const VERIFIER_DEPTH: usize = 64;

#[derive(Default)]
pub(super) struct StreamFacts {
    pub metadata_bytes: usize,
    pub body_bytes: usize,
    pub batches: usize,
}

pub(super) fn trace_bytes() -> Option<usize> {
    let count = VERIFIER_DEPTH.checked_mul(2)?;
    let detail = size_of::<flatbuffers::ErrorTraceDetail>();
    let capacity = count.checked_next_power_of_two()?;
    let bytes = capacity.checked_add(capacity / 2)?.checked_mul(detail)?;
    std::alloc::Layout::from_size_align(bytes, align_of::<flatbuffers::ErrorTraceDetail>()).ok()?;
    Some(bytes)
}

struct MessageView<'a> {
    message: Message<'a>,
    metadata_bytes: usize,
    body_bytes: usize,
}

enum StreamItem<T> {
    End,
    Message(T),
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let value = u32::from_le_bytes(bytes.get(*offset..end)?.try_into().ok()?);
    *offset = end;
    Some(value)
}

fn message_metadata<'a>(bytes: &'a [u8], offset: &mut usize) -> Option<StreamItem<&'a [u8]>> {
    let length = message_length(bytes, offset)?;
    if length == 0 {
        return (*offset == bytes.len()).then_some(StreamItem::End);
    }
    let length = usize::try_from(i32::try_from(length).ok()?).ok()?;
    let metadata_end = offset.checked_add(length)?;
    let metadata = bytes.get(*offset..metadata_end)?;
    *offset = metadata_end;
    Some(StreamItem::Message(metadata))
}

fn message_length(bytes: &[u8], offset: &mut usize) -> Option<u32> {
    let prefix = read_u32(bytes, offset)?;
    Some(if prefix == u32::MAX {
        read_u32(bytes, offset)?
    } else {
        prefix
    })
}

fn next_message<'a>(bytes: &'a [u8], offset: &mut usize) -> Option<StreamItem<MessageView<'a>>> {
    let StreamItem::Message(metadata) = message_metadata(bytes, offset)? else {
        return Some(StreamItem::End);
    };
    let options = flatbuffers::VerifierOptions::default();
    let message = ipc::root_as_message_with_opts(&options, metadata).ok()?;
    let body_bytes = usize::try_from(message.bodyLength()).ok()?;
    let metadata_end = *offset;
    *offset = offset.checked_add(body_bytes)?;
    bytes.get(metadata_end..*offset)?;
    Some(StreamItem::Message(MessageView {
        message,
        metadata_bytes: metadata.len(),
        body_bytes,
    }))
}

pub(super) fn inspect(ipc: &[u8], expected: &Schema) -> Option<StreamFacts> {
    let mut offset = 0;
    let StreamItem::Message(first) = next_message(ipc, &mut offset)? else {
        return None;
    };
    if !schema_message(&first, expected) {
        return None;
    }
    record_facts(ipc, &mut offset, expected, first.metadata_bytes)
}

fn record_facts(
    ipc: &[u8],
    offset: &mut usize,
    expected: &Schema,
    metadata_bytes: usize,
) -> Option<StreamFacts> {
    let mut facts = StreamFacts {
        metadata_bytes,
        ..StreamFacts::default()
    };
    while let StreamItem::Message(view) = next_message(ipc, offset)? {
        if !record_message(&view, expected) {
            return None;
        }
        facts.metadata_bytes = facts.metadata_bytes.max(view.metadata_bytes);
        facts.body_bytes = facts.body_bytes.checked_add(view.body_bytes)?;
        facts.batches = facts.batches.checked_add(1)?;
        if facts.batches > 2 {
            return None;
        }
    }
    (facts.batches != 0).then_some(facts)
}

fn schema_message(view: &MessageView<'_>, expected: &Schema) -> bool {
    if !schema_header(view) {
        return false;
    }
    let Some(schema) = view.message.header_as_schema() else {
        return false;
    };
    let Some(fields) = schema.fields() else {
        return false;
    };
    if !schema_fields(&schema, fields.len(), expected.fields().len()) {
        return false;
    }
    fields
        .iter()
        .zip(expected.fields())
        .all(|(actual, expected)| field_matches(actual, expected))
}

fn schema_header(view: &MessageView<'_>) -> bool {
    cfg!(target_endian = "little")
        && view.body_bytes == 0
        && view.message.version() == ipc::MetadataVersion::V5
}

fn schema_fields(schema: &ipc::Schema<'_>, actual: usize, expected: usize) -> bool {
    schema.endianness() == ipc::Endianness::Little
        && actual == expected
        && schema
            .custom_metadata()
            .is_none_or(|values| values.is_empty())
}

fn field_matches(actual: ipc::Field<'_>, expected: &Field) -> bool {
    if actual.name() != Some(expected.name().as_str())
        || actual.nullable() != expected.is_nullable()
    {
        return false;
    }
    field_structure(&actual) && type_matches(actual, expected.data_type())
}

fn field_structure(field: &ipc::Field<'_>) -> bool {
    field.dictionary().is_none()
        && field.children().is_none_or(|values| values.is_empty())
        && field
            .custom_metadata()
            .is_none_or(|values| values.is_empty())
}

fn integer_profile(data_type: &DataType) -> Option<(i32, bool)> {
    match data_type {
        DataType::Int16 => Some((16, true)),
        DataType::Int32 => Some((32, true)),
        DataType::Int64 => Some((64, true)),
        DataType::UInt8 => Some((8, false)),
        DataType::UInt16 => Some((16, false)),
        DataType::UInt32 => Some((32, false)),
        DataType::UInt64 => Some((64, false)),
        _ => None,
    }
}

fn type_matches(actual: ipc::Field<'_>, expected: &DataType) -> bool {
    if let Some((width, signed)) = integer_profile(expected) {
        return actual
            .type_as_int()
            .is_some_and(|integer| integer.bitWidth() == width && integer.is_signed() == signed);
    }
    let DataType::Timestamp(unit, timezone) = expected else {
        return false;
    };
    if timezone.as_deref() == Some("") {
        return false;
    }
    actual.type_as_timestamp().is_some_and(|timestamp| {
        timestamp.unit() == ipc_unit(*unit) && timestamp.timezone() == timezone.as_deref()
    })
}

fn ipc_unit(unit: TimeUnit) -> ipc::TimeUnit {
    match unit {
        TimeUnit::Second => ipc::TimeUnit::SECOND,
        TimeUnit::Millisecond => ipc::TimeUnit::MILLISECOND,
        TimeUnit::Microsecond => ipc::TimeUnit::MICROSECOND,
        TimeUnit::Nanosecond => ipc::TimeUnit::NANOSECOND,
    }
}

fn record_message(view: &MessageView<'_>, expected: &Schema) -> bool {
    let Some(record) = view.message.header_as_record_batch() else {
        return false;
    };
    if !record_layout(view, &record) {
        return false;
    }
    let Some(nodes) = record.nodes() else {
        return false;
    };
    let Some(buffers) = record.buffers() else {
        return false;
    };
    let Ok(rows) = usize::try_from(record.length()) else {
        return false;
    };
    columns_match(nodes, buffers, expected, rows, view.body_bytes)
}

fn columns_match(
    nodes: flatbuffers::Vector<'_, ipc::FieldNode>,
    buffers: flatbuffers::Vector<'_, ipc::Buffer>,
    expected: &Schema,
    rows: usize,
    body_bytes: usize,
) -> bool {
    if nodes.len() != expected.fields().len() || buffers.len() != 2 * nodes.len() {
        return false;
    }
    expected.fields().iter().enumerate().all(|(index, field)| {
        node_matches(nodes.get(index), rows)
            && null_buffer(buffers.get(2 * index), body_bytes)
            && values_buffer(buffers.get(2 * index + 1), field, rows, body_bytes)
    })
}

fn node_matches(node: &ipc::FieldNode, rows: usize) -> bool {
    usize::try_from(node.length()) == Ok(rows) && node.null_count() == 0
}

fn null_buffer(buffer: &ipc::Buffer, body_bytes: usize) -> bool {
    let Ok(offset) = usize::try_from(buffer.offset()) else {
        return false;
    };
    let Ok(length) = usize::try_from(buffer.length()) else {
        return false;
    };
    offset
        .checked_add(length)
        .is_some_and(|end| end <= body_bytes)
}

fn values_buffer(buffer: &ipc::Buffer, field: &Field, rows: usize, body_bytes: usize) -> bool {
    let Some(width) = crate::operator::join::columnar::restored::width(field.data_type()) else {
        return false;
    };
    let Ok(offset) = usize::try_from(buffer.offset()) else {
        return false;
    };
    let Ok(length) = usize::try_from(buffer.length()) else {
        return false;
    };
    offset.is_multiple_of(width)
        && rows.checked_mul(width) == Some(length)
        && offset
            .checked_add(length)
            .is_some_and(|end| end <= body_bytes)
}

fn record_layout(view: &MessageView<'_>, record: &ipc::RecordBatch<'_>) -> bool {
    view.message.version() == ipc::MetadataVersion::V5
        && record.compression().is_none()
        && record
            .variadicBufferCounts()
            .is_none_or(|values| values.is_empty())
}
