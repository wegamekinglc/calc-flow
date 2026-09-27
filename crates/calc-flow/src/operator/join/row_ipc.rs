//! Per-row Arrow IPC streams that build the schema message once per schema.

use crate::{CalcFlowError, Result};
use datafusion::arrow::{
    datatypes::{DataType, Schema, SchemaRef},
    error::ArrowError,
    ipc::writer::{
        CompressionContext, DictionaryTracker, IpcDataGenerator, IpcWriteOptions, StreamWriter,
        write_message,
    },
    record_batch::RecordBatch,
};
use std::sync::Arc;

/// Encodes stored rows as standalone IPC streams byte-identical to a fresh
/// default `StreamWriter` per row. For flat, dictionary-free schemas the
/// schema message and end-of-stream marker are reused across rows; other
/// schemas carry dictionary state and take the writer path unchanged.
#[derive(Default)]
pub(super) struct RowIpcEncoder {
    framing: Option<Framing>,
}

/// The bytes a fresh writer emits around one record-batch message.
struct Framing {
    schema: SchemaRef,
    prefix: Vec<u8>,
    suffix: Vec<u8>,
}

impl RowIpcEncoder {
    pub(super) fn encode(
        &mut self,
        record: &RecordBatch,
        operator_id: &str,
        side: &str,
    ) -> Result<Vec<u8>> {
        if !reusable_framing(record.schema_ref()) {
            return writer_ipc(record, operator_id, side);
        }
        let framing = self.framing(record.schema_ref(), operator_id, side)?;
        framed_ipc(framing, record).map_err(|error| encoding_error(operator_id, side, &error))
    }

    fn framing(&mut self, schema: &SchemaRef, operator_id: &str, side: &str) -> Result<&Framing> {
        let stale = self.framing.as_ref().is_none_or(|framing| {
            !Arc::ptr_eq(&framing.schema, schema) && framing.schema != *schema
        });
        if stale {
            self.framing = Some(Framing::new(schema, operator_id, side)?);
        }
        Ok(self.framing.as_ref().expect("framing was just built"))
    }
}

impl Framing {
    fn new(schema: &SchemaRef, operator_id: &str, side: &str) -> Result<Self> {
        let mut writer = StreamWriter::try_new(Vec::new(), schema.as_ref())
            .map_err(|error| writer_error(operator_id, side, &error))?;
        let prefix = writer.get_ref().clone();
        writer
            .finish()
            .map_err(|error| encoding_error(operator_id, side, &error))?;
        let suffix = writer.get_ref()[prefix.len()..].to_vec();
        Ok(Self {
            schema: Arc::clone(schema),
            prefix,
            suffix,
        })
    }
}

/// Whether every column is flat and dictionary-free, so a fresh writer's
/// dictionary tracker never contributes bytes.
fn reusable_framing(schema: &Schema) -> bool {
    schema.fields().iter().all(|field| {
        let data_type = field.data_type();
        !data_type.is_nested()
            && !matches!(
                data_type,
                DataType::Dictionary(..) | DataType::RunEndEncoded(..)
            )
    })
}

fn framed_ipc(framing: &Framing, record: &RecordBatch) -> std::result::Result<Vec<u8>, ArrowError> {
    let options = IpcWriteOptions::default();
    let (dictionaries, message) = IpcDataGenerator::default().encode(
        record,
        &mut DictionaryTracker::new(false),
        &options,
        &mut CompressionContext::default(),
    )?;
    debug_assert!(
        dictionaries.is_empty(),
        "flat schemas carry no dictionaries"
    );
    let mut ipc = framing.prefix.clone();
    write_message(&mut ipc, message, &options)?;
    ipc.extend_from_slice(&framing.suffix);
    Ok(ipc)
}

/// Encodes one stored row's Arrow IPC payload with a fresh stream writer.
fn writer_ipc(record: &RecordBatch, operator_id: &str, side: &str) -> Result<Vec<u8>> {
    let mut ipc = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut ipc, record.schema().as_ref())
            .map_err(|error| writer_error(operator_id, side, &error))?;
        writer
            .write(record)
            .and_then(|()| writer.finish())
            .map_err(|error| encoding_error(operator_id, side, &error))?;
    }
    Ok(ipc)
}

fn writer_error(operator_id: &str, side: &str, error: &ArrowError) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("stream Join {operator_id:?} {side} IPC writer failed: {error}"),
    }
}

fn encoding_error(operator_id: &str, side: &str, error: &ArrowError) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("stream Join {operator_id:?} {side} IPC encoding failed: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::{
        array::{
            ArrayRef, DictionaryArray, Float64Array, Int64Array, StringArray, StructArray,
            TimestampMicrosecondArray,
        },
        datatypes::{Field, Int32Type, TimeUnit},
    };
    use std::collections::HashMap;

    fn writer_bytes(record: &RecordBatch) -> Vec<u8> {
        let mut ipc = Vec::new();
        let mut writer = StreamWriter::try_new(&mut ipc, record.schema().as_ref()).unwrap();
        writer.write(record).unwrap();
        writer.finish().unwrap();
        drop(writer);
        ipc
    }

    fn flat_record(label: &str) -> RecordBatch {
        let schema = Schema::new_with_metadata(
            vec![
                Field::new("key", DataType::Utf8, false)
                    .with_metadata(HashMap::from([("role".to_owned(), "key".to_owned())])),
                Field::new(
                    "time",
                    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                    false,
                ),
                Field::new("value", DataType::Float64, true),
                Field::new("count", DataType::Int64, true),
            ],
            HashMap::from([("origin".to_owned(), label.to_owned())]),
        );
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["alpha", "b", "", "delta-long-key"])),
            Arc::new(TimestampMicrosecondArray::from(vec![1, 2, 3, 4]).with_timezone("UTC")),
            Arc::new(Float64Array::from(vec![
                Some(1.5),
                None,
                Some(-0.0),
                Some(9.25),
            ])),
            Arc::new(Int64Array::from(vec![
                None,
                Some(7),
                Some(-1),
                Some(i64::MAX),
            ])),
        ];
        RecordBatch::try_new(Arc::new(schema), columns).unwrap()
    }

    fn fallback_records() -> Vec<RecordBatch> {
        let dictionary: DictionaryArray<Int32Type> = vec!["x", "y", "x"].into_iter().collect();
        let dictionary = RecordBatch::try_from_iter([("d", Arc::new(dictionary) as ArrayRef)]);
        let nested = StructArray::from(vec![(
            Arc::new(Field::new("inner", DataType::Int64, false)),
            Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
        )]);
        let nested = RecordBatch::try_from_iter([("s", Arc::new(nested) as ArrayRef)]);
        vec![dictionary.unwrap(), nested.unwrap()]
    }

    #[test]
    fn row_streams_match_a_fresh_stream_writer_per_row() {
        let first = flat_record("first");
        let second = flat_record("second");
        let rows = (0..first.num_rows())
            .flat_map(|row| [first.slice(row, 1), second.slice(row, 1)])
            .chain([first.slice(0, 0), first.clone(), first.slice(1, 1)])
            .chain(fallback_records())
            .chain([second.slice(3, 1)])
            .collect::<Vec<_>>();
        let mut encoder = RowIpcEncoder::default();
        for record in &rows {
            assert_eq!(
                encoder.encode(record, "join", "left").unwrap(),
                writer_bytes(record),
                "schema {:?}",
                record.schema()
            );
        }
    }
}
