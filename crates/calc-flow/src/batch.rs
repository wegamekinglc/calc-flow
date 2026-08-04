use std::{any::Any, collections::BTreeMap, fmt::Debug, sync::Arc};

use datafusion::arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{CalcFlowError, JsonMap, Result};

/// Adds `addend` to `total`, reporting overflow as a typed error (spec S10.2:
/// byte and row sums use checked arithmetic; overflow is a typed error).
pub(crate) fn checked_accumulate(
    total: usize,
    addend: usize,
    field: &'static str,
) -> Result<usize> {
    total
        .checked_add(addend)
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "size sum overflowed usize".into(),
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BatchKind {
    Table,
    Array,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BatchMetadata {
    source: String,
    sequence: u64,
    attributes: JsonMap,
}

impl BatchMetadata {
    /// Creates metadata for a batch.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when `source` contains a NUL
    /// character.
    pub fn new(
        source: impl Into<String>,
        sequence: u64,
        attributes: BTreeMap<String, Value>,
    ) -> Result<Self> {
        let source = source.into();
        if source.contains('\0') {
            return Err(CalcFlowError::InvalidArgument {
                field: "metadata.source".into(),
                message: "must not contain NUL".into(),
            });
        }
        Ok(Self {
            source,
            sequence,
            attributes,
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn attributes(&self) -> &JsonMap {
        &self.attributes
    }
}

#[derive(Clone, Debug)]
pub struct TableBatch {
    schema: SchemaRef,
    batches: Arc<[RecordBatch]>,
    rows: usize,
}

impl TableBatch {
    fn new(batches: Vec<RecordBatch>) -> Result<Self> {
        let schema = batches.first().map(RecordBatch::schema).ok_or_else(|| {
            CalcFlowError::InvalidArgument {
                field: "batches".into(),
                message: "must contain at least one RecordBatch; represent an empty table with one zero-row batch".into(),
            }
        })?;
        if batches.iter().any(|batch| batch.schema() != schema) {
            return Err(CalcFlowError::InvalidArgument {
                field: "batches".into(),
                message: "schemas must match".into(),
            });
        }
        let rows = batches.iter().try_fold(0_usize, |rows, batch| {
            checked_accumulate(rows, batch.num_rows(), "batches")
        })?;
        Ok(Self {
            schema,
            batches: batches.into(),
            rows,
        })
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// Estimates the in-memory cost of the visible Arrow slices in bytes.
    ///
    /// Each column of each record batch is charged its Arrow slice memory
    /// size, so sliced arrays sharing a larger backing allocation are charged
    /// only for their visible window. The estimate is a logical queue charge,
    /// not a process RSS measurement.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when Arrow cannot measure a
    /// column or the summed size overflows `usize`.
    pub fn estimated_bytes(&self) -> Result<usize> {
        self.batches.iter().try_fold(0_usize, |total, batch| {
            batch.columns().iter().try_fold(total, |total, column| {
                let bytes = column.to_data().get_slice_memory_size().map_err(|error| {
                    CalcFlowError::InvalidArgument {
                        field: "batch".into(),
                        message: format!("Arrow slice memory could not be measured: {error}"),
                    }
                })?;
                checked_accumulate(total, bytes, "batch")
            })
        })
    }
}

#[allow(clippy::len_without_is_empty)]
pub trait ExternalPayload: Any + Debug + Send + Sync {
    fn backend(&self) -> &str;
    fn len(&self) -> usize;
    /// Returns an exact or conservative estimate of the payload's visible
    /// in-memory cost in bytes.
    ///
    /// Implementations must never under-report the visible payload cost;
    /// there is no opt-out (spec S10.2). The estimate is a logical queue
    /// charge used for backpressure accounting, not a process RSS
    /// measurement, and shared payloads are charged per consumer.
    fn estimated_bytes(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone, Debug)]
enum BatchPayload {
    Table(TableBatch),
    External(Arc<dyn ExternalPayload>),
}

#[derive(Clone, Debug)]
pub struct Batch {
    payload: BatchPayload,
    metadata: BatchMetadata,
}

impl Batch {
    /// Creates a table batch from one or more identically shaped Arrow batches.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when `batches` is empty or
    /// contains mismatched schemas.
    pub fn table(batches: Vec<RecordBatch>, metadata: BatchMetadata) -> Result<Self> {
        Ok(Self {
            payload: BatchPayload::Table(TableBatch::new(batches)?),
            metadata,
        })
    }

    /// Creates an array batch from an externally owned payload.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the payload backend is
    /// empty.
    pub fn external(payload: Arc<dyn ExternalPayload>, metadata: BatchMetadata) -> Result<Self> {
        if payload.backend().is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "backend".into(),
                message: "must not be empty".into(),
            });
        }
        Ok(Self {
            payload: BatchPayload::External(payload),
            metadata,
        })
    }

    pub fn kind(&self) -> BatchKind {
        match &self.payload {
            BatchPayload::Table(_) => BatchKind::Table,
            BatchPayload::External(_) => BatchKind::Array,
        }
    }

    pub fn num_rows(&self) -> usize {
        match &self.payload {
            BatchPayload::Table(table) => table.rows,
            BatchPayload::External(payload) => payload.len(),
        }
    }

    /// Estimates this batch's in-memory cost in bytes before it enters a
    /// stream queue.
    ///
    /// Table batches are charged the Arrow memory size of their visible
    /// slices; external batches are charged the payload-provided exact or
    /// conservative estimate (spec S10.2). The result is a logical queue
    /// charge, not a process RSS measurement.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when a table batch cannot
    /// be measured or the summed size overflows `usize`.
    pub fn estimated_bytes(&self) -> Result<usize> {
        match &self.payload {
            BatchPayload::Table(table) => table.estimated_bytes(),
            BatchPayload::External(payload) => Ok(payload.estimated_bytes()),
        }
    }

    pub fn metadata(&self) -> &BatchMetadata {
        &self.metadata
    }

    /// Returns this batch's Arrow table payload.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when this is an array batch.
    pub fn table_payload(&self) -> Result<&TableBatch> {
        match &self.payload {
            BatchPayload::Table(table) => Ok(table),
            BatchPayload::External(_) => Err(CalcFlowError::InvalidArgument {
                field: "batch".into(),
                message: "expected table batch".into(),
            }),
        }
    }

    /// Returns this batch's externally owned payload.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when this is a table batch.
    pub fn external_payload(&self) -> Result<&Arc<dyn ExternalPayload>> {
        match &self.payload {
            BatchPayload::External(payload) => Ok(payload),
            BatchPayload::Table(_) => Err(CalcFlowError::InvalidArgument {
                field: "batch".into(),
                message: "expected array batch".into(),
            }),
        }
    }

    #[must_use]
    pub fn with_metadata(&self, metadata: BatchMetadata) -> Self {
        Self {
            payload: self.payload.clone(),
            metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_accumulate_sums_within_usize() {
        assert_eq!(checked_accumulate(2, 3, "batch").unwrap(), 5);
        assert_eq!(checked_accumulate(0, 0, "batch").unwrap(), 0);
    }

    #[test]
    fn checked_accumulate_rejects_usize_overflow_with_a_typed_error() {
        assert!(matches!(
            checked_accumulate(usize::MAX, 1, "batch"),
            Err(CalcFlowError::InvalidArgument { ref field, .. }) if field == "batch"
        ));
    }
}
