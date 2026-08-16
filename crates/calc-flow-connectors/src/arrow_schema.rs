//! Shared Arrow helpers for the format codecs.
//!
//! Schema conversion accepts exactly the data-only `ArrowFieldSpec`
//! vocabulary mirrored from the core crate; an unknown type string fails
//! closed instead of guessing. Bounded assembly wraps decoded record
//! batches into immutable [`Batch`] values after enforcing the row and
//! byte limits of one decode expansion.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use arrow::record_batch::RecordBatch;
use calc_flow::{
    ArrowFieldSpec, Batch, BatchMetadata, CalcFlowError, ConnectorError, ConnectorIdentity,
    ConnectorOperation, DecodeBounds, FormatIdentity, Result,
};

/// Builds a schema from explicit field specifications.
///
/// # Errors
///
/// Returns [`calc_flow::CalcFlowError::InvalidArgument`] naming the field whose type
/// string is outside the supported vocabulary or whose name is empty.
pub fn schema_from_spec(fields: &[ArrowFieldSpec]) -> Result<SchemaRef> {
    let mut converted = Vec::with_capacity(fields.len());
    for field in fields {
        if field.name.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "schema field name".into(),
                message: "schema field names must not be empty".into(),
            });
        }
        let data_type =
            arrow_data_type(&field.data_type).ok_or_else(|| CalcFlowError::InvalidArgument {
                field: format!("schema field {}", field.name),
                message: format!("unsupported data type {:?}", field.data_type),
            })?;
        converted.push(Field::new(&field.name, data_type, field.nullable));
    }
    Ok(Arc::new(Schema::new(converted)))
}

/// The trusted provider namespace of this crate.
pub const PROVIDER: &str = "calc-flow-connectors";

/// The connector identity codecs report in their safe error projections.
///
/// A [`FormatIdentity`] guarantees non-empty name and version components
/// and the provider is a non-empty constant, so construction cannot fail.
///
/// # Panics
///
/// Never in practice; the `expect` documents the non-emptiness invariant
/// shared by every accepted format identity.
pub fn codec_connector_identity(identity: &FormatIdentity) -> ConnectorIdentity {
    ConnectorIdentity::new(PROVIDER, &identity.name, &identity.version)
        .expect("format identities carry non-empty components")
}

/// Projects a codec failure through the typed connector error surface.
///
/// Empty operation names fall back to `decode`; the projection never
/// carries payload bytes.
///
/// # Panics
///
/// Never in practice; the operation name is non-empty after the
/// fallback substitution.
pub fn codec_error(identity: &FormatIdentity, operation: &str, detail: &str) -> CalcFlowError {
    let operation = if operation.is_empty() {
        "decode"
    } else {
        operation
    };
    let operation = ConnectorOperation::new(operation).expect("the operation name is non-empty");
    CalcFlowError::Connector(ConnectorError::new(
        codec_connector_identity(identity),
        operation,
        detail,
    ))
}

/// Wraps decoded record batches into an immutable batch after bounds.
///
/// # Errors
///
/// Returns the codec's safe error when the decoded expansion exceeds the
/// row or byte limit, before any batch reaches an edge.
pub fn bounded_table_batch(
    identity: &FormatIdentity,
    batches: Vec<RecordBatch>,
    bounds: &DecodeBounds,
    source: &str,
    sequence: u64,
) -> Result<Batch> {
    let rows: u64 = batches
        .iter()
        .map(|batch| u64::try_from(batch.num_rows()).unwrap_or(u64::MAX))
        .sum();
    let bytes: u64 = batches
        .iter()
        .map(|batch| u64::try_from(batch.get_array_memory_size()).unwrap_or(u64::MAX))
        .sum();
    bounds
        .check(identity, rows, bytes)
        .map_err(|error| match error {
            CalcFlowError::InvalidArgument { field, message } => {
                codec_error(identity, "decode", &format!("{field}: {message}"))
            }
            other => other,
        })?;
    let metadata = BatchMetadata::new(source, sequence, BTreeMap::new())?;
    Batch::table(batches, metadata)
}

fn arrow_data_type(value: &str) -> Option<DataType> {
    Some(match value {
        "bool" => DataType::Boolean,
        "date32" => DataType::Date32,
        "date64" => DataType::Date64,
        "float32" => DataType::Float32,
        "float64" => DataType::Float64,
        "int8" => DataType::Int8,
        "int16" => DataType::Int16,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "large_string" => DataType::LargeUtf8,
        "string" => DataType::Utf8,
        "time32[s]" => DataType::Time32(TimeUnit::Second),
        "time64[us]" => DataType::Time64(TimeUnit::Microsecond),
        "timestamp[ms]" => DataType::Timestamp(TimeUnit::Millisecond, None),
        "timestamp[us]" => DataType::Timestamp(TimeUnit::Microsecond, None),
        "uint8" => DataType::UInt8,
        "uint16" => DataType::UInt16,
        "uint32" => DataType::UInt32,
        "uint64" => DataType::UInt64,
        _ => return None,
    })
}
