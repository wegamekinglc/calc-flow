//! The Parquet format codec (feature `file`).
//!
//! Decoding reads the file's row groups as Arrow record batches, enforces
//! the row and byte limits per decode expansion — including one row group
//! whose row count exceeds the limit — and compares an explicit schema
//! against the file's stored Arrow schema before any row is emitted.

use bytes::Bytes;
use calc_flow::{
    ArrowFieldSpec, Batch, DecodeBounds, FormatDecoder, FormatEncoder, FormatIdentity, Result,
};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;

use crate::arrow_schema::{bounded_table_batch, codec_error, schema_from_spec};

/// The Parquet codec identity.
pub const IDENTITY: &str = "parquet";

/// The codec implementation version.
pub const IDENTITY_VERSION: &str = "1";

/// Builds the Parquet codec identity.
///
/// # Errors
///
/// Returns [`calc_flow::CalcFlowError::InvalidArgument`] when the version is empty.
pub fn identity(version: &str) -> Result<FormatIdentity> {
    FormatIdentity::new(IDENTITY, version)
}

/// The Parquet codec.
#[derive(Clone, Debug)]
pub struct ParquetCodec {
    identity: FormatIdentity,
}

impl ParquetCodec {
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
}

impl FormatDecoder for ParquetCodec {
    fn identity(&self) -> FormatIdentity {
        self.identity.clone()
    }

    fn decode(
        &self,
        bytes: &[u8],
        bounds: &DecodeBounds,
        schema: &[ArrowFieldSpec],
    ) -> Result<Batch> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::copy_from_slice(bytes))
            .map_err(|error| codec_error(&self.identity, "decode", &error.to_string()))?;
        verify_stored_schema(&self.identity, builder.schema(), schema)?;
        verify_row_groups(&self.identity, builder.metadata(), bounds)?;
        let batch_rows = usize::try_from(bounds.max_rows).unwrap_or(usize::MAX);
        let reader = builder
            .with_batch_size(batch_rows.max(1))
            .build()
            .map_err(|error| codec_error(&self.identity, "decode", &error.to_string()))?;
        let batches: Vec<_> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| codec_error(&self.identity, "decode", &error.to_string()))?;
        bounded_table_batch(&self.identity, batches, bounds, IDENTITY, 0)
    }
}

/// Compares the file's stored Arrow schema with an explicit expectation.
fn verify_stored_schema(
    identity: &FormatIdentity,
    stored: &arrow::datatypes::SchemaRef,
    schema: &[ArrowFieldSpec],
) -> Result<()> {
    if schema.is_empty() {
        return Ok(());
    }
    let expected = schema_from_spec(schema)?;
    let fields_agree = expected.fields().len() == stored.fields().len()
        && expected
            .fields()
            .iter()
            .zip(stored.fields())
            .all(|(left, right)| left.name() == right.name());
    if !fields_agree {
        return Err(codec_error(
            identity,
            "decode",
            "stored schema does not match the explicit schema",
        ));
    }
    Ok(())
}

/// Rejects any single row group whose row count exceeds the bound.
fn verify_row_groups(
    identity: &FormatIdentity,
    metadata: &parquet::file::metadata::ParquetMetaData,
    bounds: &DecodeBounds,
) -> Result<()> {
    for (index, row_group) in metadata.row_groups().iter().enumerate() {
        let rows = u64::try_from(row_group.num_rows()).unwrap_or(u64::MAX);
        if rows > bounds.max_rows {
            return Err(codec_error(
                identity,
                "decode",
                &format!(
                    "row group {index} carries {rows} rows above the {} row limit",
                    bounds.max_rows
                ),
            ));
        }
    }
    Ok(())
}

impl FormatEncoder for ParquetCodec {
    fn identity(&self) -> FormatIdentity {
        self.identity.clone()
    }

    fn encode(&self, batch: &Batch) -> Result<Vec<u8>> {
        let payload = batch.table_payload().map_err(|_| {
            codec_error(
                &self.identity,
                "encode",
                "Parquet encodes table batches only",
            )
        })?;
        let mut writer = ArrowWriter::try_new(
            Vec::<u8>::new(),
            payload.schema().clone(),
            Some(WriterProperties::default()),
        )
        .map_err(|error| codec_error(&self.identity, "encode", &error.to_string()))?;
        for record in payload.batches() {
            writer
                .write(record)
                .map_err(|error| codec_error(&self.identity, "encode", &error.to_string()))?;
        }
        writer
            .into_inner()
            .map_err(|error| codec_error(&self.identity, "encode", &error.to_string()))
    }
}
