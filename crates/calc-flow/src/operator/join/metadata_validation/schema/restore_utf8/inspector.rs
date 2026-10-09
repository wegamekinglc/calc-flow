use super::super::restore_bases::inspector::{self as scalar, StreamItem};
use crate::Result;
use crate::operator::join::columnar::restored;
use crate::runtime::streaming::gather_work::GatherStop;
use datafusion::arrow::{
    datatypes::{DataType, Field, Schema},
    ipc,
};
use std::ops::Range;

#[derive(Clone, Copy, Default)]
pub(super) struct Facts {
    pub key_total: usize,
    pub key_max: usize,
    pub value_max: usize,
}

impl Facts {
    pub(super) fn add(&mut self, next: Self) -> Option<()> {
        self.key_total = self.key_total.checked_add(next.key_total)?;
        self.key_max = self.key_max.max(next.key_max);
        self.value_max = self.value_max.max(next.value_max);
        Some(())
    }
}

struct MessageView<'a> {
    message: ipc::Message<'a>,
    body: &'a [u8],
}

fn next_message<'a>(bytes: &'a [u8], offset: &mut usize) -> Option<StreamItem<MessageView<'a>>> {
    let StreamItem::Message(metadata) = scalar::message_metadata(bytes, offset)? else {
        return Some(StreamItem::End);
    };
    let options = flatbuffers::VerifierOptions::default();
    let message = ipc::root_as_message_with_opts(&options, metadata).ok()?;
    let body_end = offset.checked_add(usize::try_from(message.bodyLength()).ok()?)?;
    let body = bytes.get(*offset..body_end)?;
    *offset = body_end;
    Some(StreamItem::Message(MessageView { message, body }))
}

pub(super) fn inspect(
    bytes: &[u8],
    expected: &Schema,
    keys: &[usize],
    stop: &GatherStop,
) -> Result<Option<Facts>> {
    stop.check()?;
    let mut offset = 0;
    let Some(StreamItem::Message(first)) = next_message(bytes, &mut offset) else {
        return Ok(None);
    };
    if !schema_matches(&first, expected) {
        return Ok(None);
    }
    let mut facts = Facts::default();
    let mut batches = 0;
    loop {
        stop.check()?;
        match next_message(bytes, &mut offset) {
            Some(StreamItem::End) => return Ok((batches != 0).then_some(facts)),
            Some(StreamItem::Message(view)) => {
                let Some(row) = record_facts(&view, expected, keys) else {
                    return Ok(None);
                };
                if facts.add(row).is_none() {
                    return Ok(None);
                }
                batches += 1;
                if batches > 2 {
                    return Ok(None);
                }
            }
            None => return Ok(None),
        }
    }
}

fn schema_matches(view: &MessageView<'_>, expected: &Schema) -> bool {
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
        && view.body.is_empty()
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
    if expected.data_type() != &DataType::Utf8 {
        return scalar::field_matches(actual, expected);
    }
    actual.name() == Some(expected.name().as_str())
        && actual.nullable() == expected.is_nullable()
        && scalar::field_structure(&actual)
        && actual.type_as_utf_8().is_some()
}

fn record_facts(view: &MessageView<'_>, expected: &Schema, keys: &[usize]) -> Option<Facts> {
    let record = view.message.header_as_record_batch()?;
    if !record_layout(view, &record) {
        return None;
    }
    let nodes = record.nodes()?;
    let buffers = record.buffers()?;
    if !columns_layout(nodes.len(), buffers.len(), expected)? {
        return None;
    }
    columns_facts(view.body, expected, keys, nodes, buffers)
}

fn columns_facts(
    body: &[u8],
    expected: &Schema,
    keys: &[usize],
    nodes: flatbuffers::Vector<'_, ipc::FieldNode>,
    buffers: flatbuffers::Vector<'_, ipc::Buffer>,
) -> Option<Facts> {
    let mut cursor = 0;
    let mut facts = Facts::default();
    for (index, field) in expected.fields().iter().enumerate() {
        let length = inspect_column(nodes.get(index), field, buffers, &mut cursor, body)?;
        if keys.contains(&index) {
            add_key(&mut facts, field.data_type(), length)?;
        }
    }
    if cursor != buffers.len() {
        return None;
    }
    facts.key_max = facts.key_total;
    Some(facts)
}

fn inspect_column(
    node: &ipc::FieldNode,
    field: &Field,
    buffers: flatbuffers::Vector<'_, ipc::Buffer>,
    cursor: &mut usize,
    body: &[u8],
) -> Option<usize> {
    if node.length() != 1 {
        return None;
    }
    let bitmap = buffer_range(buffers.get(*cursor), body)?;
    *cursor += 1;
    if !restored::validity::certified(field, 1, node.null_count(), &body[bitmap]) {
        return None;
    }
    column_length(field, buffers, cursor, body)
}

fn columns_layout(nodes: usize, buffers: usize, expected: &Schema) -> Option<bool> {
    let fields_match = nodes == expected.fields().len();
    let buffers_match = buffers == buffer_count(expected)?;
    Some(fields_match && buffers_match)
}

fn record_layout(view: &MessageView<'_>, record: &ipc::RecordBatch<'_>) -> bool {
    view.message.version() == ipc::MetadataVersion::V5
        && record.length() == 1
        && record.compression().is_none()
        && record
            .variadicBufferCounts()
            .is_none_or(|values| values.is_empty())
}

fn buffer_count(schema: &Schema) -> Option<usize> {
    schema.fields().iter().try_fold(0_usize, |count, field| {
        count.checked_add(if field.data_type() == &DataType::Utf8 {
            3
        } else {
            2
        })
    })
}

fn add_key(facts: &mut Facts, data_type: &DataType, length: usize) -> Option<()> {
    let timezone = match data_type {
        DataType::Timestamp(_, timezone) => timezone.as_ref().map_or(0, |text| text.len()),
        _ => 0,
    };
    if data_type != &DataType::Utf8 {
        restored::key_width(data_type)?;
    }
    facts.key_total = facts
        .key_total
        .checked_add(9)?
        .checked_add(timezone)?
        .checked_add(length)?;
    facts.value_max = facts.value_max.max(length);
    Some(())
}

fn buffer_range(buffer: &ipc::Buffer, body: &[u8]) -> Option<Range<usize>> {
    let start = usize::try_from(buffer.offset()).ok()?;
    let end = start.checked_add(usize::try_from(buffer.length()).ok()?)?;
    body.get(start..end)?;
    Some(start..end)
}

fn column_length(
    field: &Field,
    buffers: flatbuffers::Vector<'_, ipc::Buffer>,
    cursor: &mut usize,
    body: &[u8],
) -> Option<usize> {
    let range = buffer_range(buffers.get(*cursor), body)?;
    *cursor += 1;
    if field.data_type() == &DataType::Utf8 {
        if !range.start.is_multiple_of(size_of::<i32>()) || range.len() != 2 * size_of::<i32>() {
            return None;
        }
        let values = buffer_range(buffers.get(*cursor), body)?;
        *cursor += 1;
        return string_length(&body[range], &body[values]);
    }
    let width = restored::width(field.data_type())?;
    (range.start.is_multiple_of(width) && range.len() == width).then_some(width)
}

fn string_length(offsets: &[u8], values: &[u8]) -> Option<usize> {
    let start = usize::try_from(i32::from_le_bytes(offsets.get(..4)?.try_into().ok()?)).ok()?;
    let end = usize::try_from(i32::from_le_bytes(offsets.get(4..8)?.try_into().ok()?)).ok()?;
    let text = std::str::from_utf8(values).ok()?;
    text.get(start..end).map(str::len)
}
