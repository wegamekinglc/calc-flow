mod framing;

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
    framing::validate_ipc_framing(bytes)?;
    let mut reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| super::arrow_error(&error))?;
    decoded_row(&mut reader)
}

fn decoded_row(reader: &mut StreamReader<Cursor<&[u8]>>) -> Result<RecordBatch> {
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
    let schema = encoded_schema(bytes)?;
    let actual: [u8; 32] = Sha256::digest(schema).into();
    if &actual != expected {
        return Err(framing_error(
            "schema encoding differs from trusted declaration",
        ));
    }
    Ok(())
}

fn encoded_schema(bytes: &[u8]) -> Result<&[u8]> {
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
    bytes
        .get(..end)
        .ok_or_else(|| framing_error("truncated schema"))
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
        assert!(framing::validate_message(&message, 0, 1).is_err());
    }
}
