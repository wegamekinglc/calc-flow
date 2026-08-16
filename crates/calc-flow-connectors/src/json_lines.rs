//! The newline-delimited JSON format codec.
//!
//! Each payload line is one JSON object. Decoding accepts an optional
//! explicit schema; without one the payload infers its schema. Expansion
//! is bounded by the decode row and byte limits before a batch is
//! returned, and an explicit schema that disagrees with the payload fails
//! closed.

use std::io::Cursor;

use arrow::datatypes::SchemaRef;
use arrow_json::reader::infer_json_schema_from_iterator;
use arrow_json::{LineDelimitedWriter, ReaderBuilder};
use calc_flow::{
    ArrowFieldSpec, Batch, DecodeBounds, FormatDecoder, FormatEncoder, FormatIdentity, Result,
};
use serde_json::Deserializer;

use crate::arrow_schema::{bounded_table_batch, codec_error, schema_from_spec};

/// The newline JSON codec identity.
pub const IDENTITY: &str = "json";

/// The codec implementation version.
pub const IDENTITY_VERSION: &str = "1";

/// Builds the newline JSON codec identity.
///
/// # Errors
///
/// Returns [`calc_flow::CalcFlowError::InvalidArgument`] when the version is empty.
pub fn identity(version: &str) -> Result<FormatIdentity> {
    FormatIdentity::new(IDENTITY, version)
}

/// The newline-delimited JSON codec.
#[derive(Clone, Debug)]
pub struct JsonLinesCodec {
    identity: FormatIdentity,
}

impl JsonLinesCodec {
    /// Creates the codec.
    ///
    /// # Errors
    ///
    /// Returns [`calc_flow::CalcFlowError::InvalidArgument`] when `version` is empty.
    pub fn new(version: &str) -> Result<Self> {
        Ok(Self {
            identity: identity(version)?,
        })
    }

    fn resolve_schema(&self, bytes: &[u8], schema: &[ArrowFieldSpec]) -> Result<SchemaRef> {
        let values = Deserializer::from_slice(bytes)
            .into_iter::<serde_json::Value>()
            .map(|result| {
                result.map_err(|error| arrow::error::ArrowError::ParseError(error.to_string()))
            });
        let inferred = infer_json_schema_from_iterator(values)
            .map_err(|error| codec_error(&self.identity, "decode", &error.to_string()))?;
        if schema.is_empty() {
            return Ok(SchemaRef::from(inferred));
        }
        let expected = schema_from_spec(schema)?;
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
                "payload fields do not match the explicit schema",
            ));
        }
        Ok(expected)
    }
}

impl FormatDecoder for JsonLinesCodec {
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

impl FormatEncoder for JsonLinesCodec {
    fn identity(&self) -> FormatIdentity {
        self.identity.clone()
    }

    fn encode(&self, batch: &Batch) -> Result<Vec<u8>> {
        let payload = batch.table_payload().map_err(|_| {
            codec_error(
                &self.identity,
                "encode",
                "newline JSON encodes table batches only",
            )
        })?;
        let mut writer = LineDelimitedWriter::new(Vec::<u8>::new());
        for record in payload.batches() {
            writer
                .write(record)
                .map_err(|error| codec_error(&self.identity, "encode", &error.to_string()))?;
        }
        Ok(writer.into_inner())
    }
}
