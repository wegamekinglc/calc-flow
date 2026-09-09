use super::framing_error;
use crate::Result;
use datafusion::arrow::ipc::{self, Message, MessageHeader};

pub(super) fn validate_ipc_framing(mut bytes: &[u8]) -> Result<()> {
    let mut stage = 0;
    let mut fields = 0;
    loop {
        let length = message_prefix(&mut bytes)?;
        if length == 0 {
            return validate_end(stage, bytes);
        }
        if stage >= 2 {
            return Err(framing_error("unexpected additional row IPC message"));
        }
        let (message, body) = read_message(&mut bytes, length)?;
        fields = validate_stage(stage, &message, body, fields)?;
        bytes = &bytes[body..];
        stage += 1;
    }
}

fn message_prefix(bytes: &mut &[u8]) -> Result<i32> {
    let prefix = bytes
        .get(..8)
        .ok_or_else(|| framing_error("missing canonical message prefix"))?;
    if prefix[..4] != [255; 4] {
        return Err(framing_error("noncanonical continuation marker"));
    }
    let length = i32::from_le_bytes(prefix[4..].try_into().expect("four bytes"));
    *bytes = &bytes[8..];
    Ok(length)
}

fn validate_end(stage: usize, bytes: &[u8]) -> Result<()> {
    if stage == 2 && bytes.is_empty() {
        Ok(())
    } else {
        Err(framing_error(
            "row IPC requires schema, one record batch and end marker",
        ))
    }
}

fn read_message<'a>(bytes: &mut &'a [u8], length: i32) -> Result<(Message<'a>, usize)> {
    let length = usize::try_from(length).map_err(|_| framing_error("negative metadata length"))?;
    let metadata = bytes
        .get(..length)
        .ok_or_else(|| framing_error("metadata exceeds encoded payload"))?;
    let message =
        ipc::root_as_message(metadata).map_err(|_| framing_error("invalid message metadata"))?;
    let body = usize::try_from(message.bodyLength())
        .map_err(|_| framing_error("negative message body length"))?;
    *bytes = &bytes[length..];
    if body > bytes.len() {
        return Err(framing_error("message body exceeds encoded payload"));
    }
    Ok((message, body))
}

fn validate_stage(
    stage: usize,
    message: &Message<'_>,
    body: usize,
    fields: usize,
) -> Result<usize> {
    if stage == 0 {
        return schema_fields(message, body);
    }
    if message.header_type() != MessageHeader::RecordBatch {
        return Err(framing_error(
            "only one flat record batch may follow schema",
        ));
    }
    validate_message(message, body, fields)?;
    Ok(fields)
}

fn schema_fields(message: &Message<'_>, body: usize) -> Result<usize> {
    let schema = message
        .header_as_schema()
        .ok_or_else(|| framing_error("first message must be trusted schema"))?;
    let fields = schema
        .fields()
        .ok_or_else(|| framing_error("schema fields are absent"))?
        .len();
    if body != 0 {
        return Err(framing_error("schema must have no body"));
    }
    Ok(fields)
}

pub(super) fn validate_message(message: &Message<'_>, body: usize, fields: usize) -> Result<()> {
    let batch = message
        .header_as_record_batch()
        .ok_or_else(|| framing_error("expected record batch"))?;
    validate_record_header(&batch)?;
    validate_field_nodes(&batch, fields)?;
    validate_buffers(&batch, body)
}

fn validate_record_header(batch: &ipc::RecordBatch<'_>) -> Result<()> {
    if batch.compression().is_some() {
        return Err(framing_error("compressed row state is not supported"));
    }
    if batch.length() != 1 {
        return Err(framing_error("row state must have one top-level row"));
    }
    if batch
        .variadicBufferCounts()
        .is_some_and(|counts| !counts.is_empty())
    {
        return Err(framing_error(
            "flat row state cannot contain variadic buffers",
        ));
    }
    Ok(())
}

fn validate_field_nodes(batch: &ipc::RecordBatch<'_>, fields: usize) -> Result<()> {
    let nodes = batch
        .nodes()
        .ok_or_else(|| framing_error("row field nodes are absent"))?;
    if nodes.len() != fields
        || nodes
            .iter()
            .any(|node| node.length() != 1 || !(0..=1).contains(&node.null_count()))
    {
        return Err(framing_error(
            "flat row field nodes differ from one-row schema",
        ));
    }
    Ok(())
}

fn validate_buffers(batch: &ipc::RecordBatch<'_>, body: usize) -> Result<()> {
    let mut previous_end = 0;
    if let Some(buffers) = batch.buffers() {
        for buffer in buffers {
            previous_end = buffer_end(buffer, body, previous_end)?;
        }
    }
    Ok(())
}

fn buffer_end(buffer: &ipc::Buffer, body: usize, previous_end: usize) -> Result<usize> {
    let offset =
        usize::try_from(buffer.offset()).map_err(|_| framing_error("negative buffer offset"))?;
    let length =
        usize::try_from(buffer.length()).map_err(|_| framing_error("negative buffer length"))?;
    let end = offset
        .checked_add(length)
        .filter(|end| *end <= body)
        .ok_or_else(|| framing_error("buffer exceeds bounded body"))?;
    if offset < previous_end || offset % 64 != 0 {
        return Err(framing_error(
            "row buffers overlap or are not canonically aligned",
        ));
    }
    Ok(end)
}
