use std::{any::Any, collections::BTreeMap, fmt::Debug, sync::Arc};

use datafusion::arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{CalcFlowError, JsonMap, Result};

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
        let rows = batches.iter().map(RecordBatch::num_rows).sum();
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
}

#[allow(clippy::len_without_is_empty)]
pub trait ExternalPayload: Any + Debug + Send + Sync {
    fn backend(&self) -> &str;
    fn len(&self) -> usize;
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
