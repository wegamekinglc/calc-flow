//! The CSV format codec.
//!
//! Decoding accepts an optional explicit schema; without one the header
//! row drives schema inference. Every expansion is bounded by the decode
//! row and byte limits before a batch is returned, and an explicit schema
//! that disagrees with the payload fails closed before any row is
//! emitted.

use std::io::Cursor;

use arrow::datatypes::SchemaRef;
use arrow_csv::reader::Format as CsvFormat;
use arrow_csv::{ReaderBuilder, WriterBuilder};
use calc_flow::{
    ArrowFieldSpec, Batch, DecodeBounds, FormatDecoder, FormatEncoder, FormatIdentity, Result,
};

use crate::arrow_schema::{bounded_table_batch, codec_error, schema_from_spec};

/// The CSV codec identity.
pub const IDENTITY: &str = "csv";

/// The codec implementation version.
pub const IDENTITY_VERSION: &str = "1";

/// Builds the CSV codec identity.
///
/// # Errors
///
/// Returns [`calc_flow::CalcFlowError::InvalidArgument`] when the version is empty.
pub fn identity(version: &str) -> Result<FormatIdentity> {
    FormatIdentity::new(IDENTITY, version)
}

/// The CSV codec.
#[derive(Clone, Debug)]
pub struct CsvCodec {
    identity: FormatIdentity,
    header: bool,
}

impl CsvCodec {
    /// Creates a codec; `header` selects whether the first record names
    /// the columns.
    ///
    /// # Errors
    ///
    /// Returns [`calc_flow::CalcFlowError::InvalidArgument`] when `version` is empty.
    pub fn new(version: &str, header: bool) -> Result<Self> {
        Ok(Self {
            identity: identity(version)?,
            header,
        })
    }

    fn resolve_schema(&self, bytes: &[u8], schema: &[ArrowFieldSpec]) -> Result<SchemaRef> {
        let inferred = self.infer_schema(bytes)?;
        if schema.is_empty() {
            return Ok(inferred);
        }
        let expected = schema_from_spec(schema)?;
        if self.header {
            // The header row names the columns, so an explicit schema
            // must agree with it; headerless payloads map positionally.
            let fields_agree = expected.fields().len() == inferred.fields().len()
                && expected
                    .fields()
                    .iter()
                    .zip(inferred.fields())
                    .all(|(left, right)| left.name() == right.name());
            if !fields_agree {
                return Err(codec_error(
                    &self.identity,
                    "decode",
                    "payload columns do not match the explicit schema",
                ));
            }
        }
        Ok(expected)
    }

    fn infer_schema(&self, bytes: &[u8]) -> Result<SchemaRef> {
        let format = CsvFormat::default().with_header(self.header);
        let (inferred, _) = format
            .infer_schema(bytes, None)
            .map_err(|error| codec_error(&self.identity, "decode", &error.to_string()))?;
        Ok(SchemaRef::from(inferred))
    }
}

impl FormatDecoder for CsvCodec {
    fn identity(&self) -> FormatIdentity {
        self.identity.clone()
    }

    fn decode(
        &self,
        bytes: &[u8],
        bounds: &DecodeBounds,
        schema: &[ArrowFieldSpec],
    ) -> Result<Batch> {
        let arrow_schema = self.resolve_schema(bytes, schema)?;
        let batch_rows = usize::try_from(bounds.max_rows).unwrap_or(usize::MAX);
        let mut reader = ReaderBuilder::new(arrow_schema)
            .with_header(self.header)
            .with_batch_size(batch_rows.max(1))
            .build(Cursor::new(bytes))
            .map_err(|error| codec_error(&self.identity, "decode", &error.to_string()))?;
        let mut batches = Vec::new();
        while let Some(batch) = reader
            .next()
            .transpose()
            .map_err(|error| codec_error(&self.identity, "decode", &error.to_string()))?
        {
            batches.push(batch);
        }
        bounded_table_batch(&self.identity, batches, bounds, IDENTITY, 0)
    }
}

impl FormatEncoder for CsvCodec {
    fn identity(&self) -> FormatIdentity {
        self.identity.clone()
    }

    fn encode(&self, batch: &Batch) -> Result<Vec<u8>> {
        let payload = batch
            .table_payload()
            .map_err(|_| codec_error(&self.identity, "encode", "CSV encodes table batches only"))?;
        let mut writer = WriterBuilder::new().build(Vec::<u8>::new());
        for record in payload.batches() {
            writer
                .write(record)
                .map_err(|error| codec_error(&self.identity, "encode", &error.to_string()))?;
        }
        Ok(writer.into_inner())
    }
}
