use crate::{CalcFlowError, Result};
use datafusion::arrow::{
    datatypes::Schema,
    ipc::{reader::StreamReader, writer::StreamWriter},
    record_batch::RecordBatch,
};
use sha2::{Digest as _, Sha256};
use std::io::{self, Cursor, Write};

pub(super) struct BoundedWriter {
    pub bytes: Vec<u8>,
    limit: usize,
}
impl BoundedWriter {
    pub fn with_capacity(capacity: usize, limit: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
            limit: limit.min(capacity),
        }
    }
}
impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|size| *size <= self.limit)
            .ok_or_else(|| io::Error::other("ASOF bounded encoding workspace exhausted"))?;
        debug_assert!(next <= self.bytes.capacity());
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct CountingWriter {
    size: usize,
    limit: usize,
}
impl Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.size = self
            .size
            .checked_add(bytes.len())
            .filter(|size| *size <= self.limit)
            .ok_or_else(|| io::Error::other("ASOF bounded encoding workspace exhausted"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn write_batch(batch: &RecordBatch, output: &mut impl Write) -> Result<()> {
    let mut writer = StreamWriter::try_new(output, batch.schema().as_ref())
        .map_err(|error| super::arrow_error(&error))?;
    writer
        .write(batch)
        .map_err(|error| super::arrow_error(&error))?;
    writer.finish().map_err(|error| super::arrow_error(&error))
}
pub(super) fn encode_batch(batch: &RecordBatch, limit: usize) -> Result<Vec<u8>> {
    let mut counter = CountingWriter { size: 0, limit };
    write_batch(batch, &mut counter)?;
    let mut bytes = BoundedWriter::with_capacity(counter.size, limit);
    write_batch(batch, &mut bytes)?;
    Ok(bytes.bytes)
}

pub(super) fn decode_batch(bytes: &[u8], schema_digest: &[u8; 32]) -> Result<RecordBatch> {
    validate_schema_digest(bytes, schema_digest)?;
    validate_ipc_framing(bytes)?;
    let mut reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| super::arrow_error(&error))?;
    let batch = reader
        .next()
        .transpose()
        .map_err(|error| super::arrow_error(&error))?
        .ok_or_else(|| CalcFlowError::Format {
            message: "ASOF row payload is empty".into(),
        })?;
    if batch.num_rows() != 1 || reader.next().is_some() {
        return Err(CalcFlowError::Format {
            message: "ASOF payload must contain exactly one row".into(),
        });
    }
    Ok(batch)
}

fn validate_ipc_framing(mut bytes: &[u8]) -> Result<()> {
    use datafusion::arrow::ipc::MessageHeader;
    let mut stage = 0;
    let mut fields = 0;
    loop {
        let prefix = bytes
            .get(..8)
            .ok_or_else(|| framing_error("missing canonical message prefix"))?;
        if prefix[..4] != [255; 4] {
            return Err(framing_error("noncanonical continuation marker"));
        }
        let length = i32::from_le_bytes(prefix[4..].try_into().expect("four bytes"));
        bytes = &bytes[8..];
        if length == 0 {
            return if stage == 2 && bytes.is_empty() {
                Ok(())
            } else {
                Err(framing_error(
                    "row IPC requires schema, one record batch and end marker",
                ))
            };
        }
        if stage >= 2 {
            return Err(framing_error("unexpected additional row IPC message"));
        }
        let length =
            usize::try_from(length).map_err(|_| framing_error("negative metadata length"))?;
        let metadata = bytes
            .get(..length)
            .ok_or_else(|| framing_error("metadata exceeds encoded payload"))?;
        let message = datafusion::arrow::ipc::root_as_message(metadata)
            .map_err(|_| framing_error("invalid message metadata"))?;
        let body = usize::try_from(message.bodyLength())
            .map_err(|_| framing_error("negative message body length"))?;
        bytes = &bytes[length..];
        if body > bytes.len() {
            return Err(framing_error("message body exceeds encoded payload"));
        }
        if stage == 0 {
            let schema = message
                .header_as_schema()
                .ok_or_else(|| framing_error("first message must be trusted schema"))?;
            fields = schema
                .fields()
                .ok_or_else(|| framing_error("schema fields are absent"))?
                .len();
            if body != 0 {
                return Err(framing_error("schema must have no body"));
            }
        } else {
            if message.header_type() != MessageHeader::RecordBatch {
                return Err(framing_error(
                    "only one flat record batch may follow schema",
                ));
            }
            validate_message(&message, body, fields)?;
        }
        bytes = &bytes[body..];
        stage += 1;
    }
}

fn validate_message(
    message: &datafusion::arrow::ipc::Message<'_>,
    body: usize,
    fields: usize,
) -> Result<()> {
    let batch = message
        .header_as_record_batch()
        .ok_or_else(|| framing_error("expected record batch"))?;
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
    let mut previous_end = 0;
    if let Some(buffers) = batch.buffers() {
        for buffer in buffers {
            let offset = usize::try_from(buffer.offset())
                .map_err(|_| framing_error("negative buffer offset"))?;
            let length = usize::try_from(buffer.length())
                .map_err(|_| framing_error("negative buffer length"))?;
            let end = offset
                .checked_add(length)
                .filter(|end| *end <= body)
                .ok_or_else(|| framing_error("buffer exceeds bounded body"))?;
            if offset < previous_end || offset % 64 != 0 {
                return Err(framing_error(
                    "row buffers overlap or are not canonically aligned",
                ));
            }
            previous_end = end;
        }
    }
    Ok(())
}

fn framing_error(message: &str) -> CalcFlowError {
    CalcFlowError::Format {
        message: format!("ASOF IPC framing: {message}"),
    }
}

struct SchemaHasher(Sha256);
impl Write for SchemaHasher {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(super) fn schema_digest(schema: &Schema) -> Result<[u8; 32]> {
    let mut writer = SchemaHasher(Sha256::new());
    StreamWriter::try_new(&mut writer, schema).map_err(|error| super::arrow_error(&error))?;
    Ok(writer.0.finalize().into())
}
fn validate_schema_digest(bytes: &[u8], expected: &[u8; 32]) -> Result<()> {
    if bytes.get(..4) != Some(&[255; 4]) {
        return Err(framing_error("noncanonical schema prefix"));
    }
    let length = bytes
        .get(4..8)
        .ok_or_else(|| framing_error("missing schema length"))?;
    let length = usize::try_from(i32::from_le_bytes(length.try_into().expect("four bytes")))
        .map_err(|_| framing_error("negative schema length"))?;
    let end = length
        .checked_add(8)
        .ok_or_else(|| framing_error("schema length overflow"))?;
    let schema = bytes
        .get(..end)
        .ok_or_else(|| framing_error("truncated schema"))?;
    let actual: [u8; 32] = Sha256::digest(schema).into();
    if &actual != expected {
        return Err(framing_error(
            "schema encoding differs from trusted declaration",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::ipc;

    #[test]
    fn asof_codec_rejects_variadic_buffers_for_flat_rows() {
        let mut builder = ipc::convert::IpcSchemaEncoder::new().schema_to_fb(&Schema::empty());
        builder.reset();
        let variadic = builder.create_vector(&[1_i64]);
        let nodes = builder.create_vector(&[ipc::FieldNode::new(1, 0)]);
        let record = ipc::RecordBatch::create(
            &mut builder,
            &ipc::RecordBatchArgs {
                length: 1,
                nodes: Some(nodes),
                variadicBufferCounts: Some(variadic),
                ..Default::default()
            },
        );
        let message = ipc::Message::create(
            &mut builder,
            &ipc::MessageArgs {
                header_type: ipc::MessageHeader::RecordBatch,
                header: Some(record.as_union_value()),
                ..Default::default()
            },
        );
        builder.finish(message, None);
        let message = ipc::root_as_message(builder.finished_data()).unwrap();
        assert!(validate_message(&message, 0, 1).is_err());
    }
}
