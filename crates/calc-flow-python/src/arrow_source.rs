//! Materialized Arrow input: import once, then poll without the Python source bridge.

use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, Cursor, EdgeBudget, JsonMap, NativeWatermarkCapability,
    ReplayPositioning, Result, SourceCapabilities, SourceDeliveryCapability, SourceEvent,
    SourceSchema, StreamSource,
};
use datafusion::arrow::{
    array::{Array, Int64Array},
    compute::cast,
    datatypes::{DataType, SchemaRef},
    record_batch::RecordBatch,
};
use pyo3::prelude::*;

use crate::batch::PyBatch;

/// A reusable immutable descriptor. Each runner owns an independent native cursor.
#[pyclass(name = "_ArrowSource", frozen, module = "calc_flow._native")]
pub(crate) struct PyArrowSource {
    batch: Batch,
    budget: EdgeBudget,
    ordered_column: Option<String>,
}

#[pymethods]
impl PyArrowSource {
    #[new]
    #[pyo3(signature = (batch, max_rows, max_bytes, ordered_column=None))]
    fn new(
        batch: &Bound<'_, PyBatch>,
        max_rows: usize,
        max_bytes: usize,
        ordered_column: Option<String>,
    ) -> PyResult<Self> {
        let batch = batch.borrow().clone_inner()?;
        batch.table_payload().map_err(crate::error::to_py_err)?;
        let budget = EdgeBudget::new(max_rows, max_bytes).map_err(crate::error::to_py_err)?;
        Ok(Self {
            batch,
            budget,
            ordered_column,
        })
    }
}

impl PyArrowSource {
    pub(crate) fn source(&self) -> Result<ArrowSource> {
        let table = self.batch.table_payload()?;
        Ok(ArrowSource {
            records: table.batches().to_vec().into_iter(),
            current: None,
            schema: table.schema().clone(),
            metadata: self.batch.metadata().clone(),
            budget: self.budget,
            ordered_column: self.ordered_column.clone(),
            last_time: None,
            position: 0,
        })
    }
}

pub(crate) struct ArrowSource {
    records: std::vec::IntoIter<RecordBatch>,
    current: Option<RecordBatch>,
    schema: SchemaRef,
    metadata: BatchMetadata,
    budget: EdgeBudget,
    ordered_column: Option<String>,
    last_time: Option<i64>,
    position: u128,
}

fn invalid(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

impl ArrowSource {
    fn slice(&self, record: &RecordBatch, rows: usize) -> Result<Batch> {
        Batch::table(vec![record.slice(0, rows)], self.metadata.clone())
    }

    fn bounded_batch(&self, record: &RecordBatch) -> Result<Batch> {
        let rows = record.num_rows().min(self.budget.max_rows);
        let batch = self.slice(record, rows)?;
        if batch.estimated_bytes()? <= self.budget.max_bytes {
            return Ok(batch);
        }
        // Visible Arrow slice charges are monotone; find a fitting prefix without copying.
        let (mut low, mut high) = (0, rows);
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if self.slice(record, middle)?.estimated_bytes()? <= self.budget.max_bytes {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        if low == 0 {
            return Err(invalid(
                "source.arrow.max_bytes",
                "one row exceeds the source byte budget",
            ));
        }
        self.slice(record, low)
    }

    fn validate_time(&mut self, batch: &Batch) -> Result<()> {
        let Some(column) = &self.ordered_column else {
            return Ok(());
        };
        let record = &batch.table_payload()?.batches()[0];
        let column = record
            .column_by_name(column)
            .ok_or_else(|| invalid("source.arrow.event_time", "missing event time column"))?;
        if column.null_count() != 0 {
            return Err(invalid(
                "source.arrow.event_time",
                "expected non-null event time",
            ));
        }
        let values = cast(column, &DataType::Int64)
            .map_err(|_| invalid("source.arrow.event_time", "expected timestamp column"))?;
        let values = values
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| invalid("source.arrow.event_time", "expected timestamp column"))?;
        let mut previous = self.last_time;
        for &current in values.values() {
            if previous.is_some_and(|previous| current < previous) {
                return Err(invalid(
                    "source.arrow.event_time",
                    "expected nondecreasing event time; select an explicit watermark policy for out-of-order input",
                ));
            }
            previous = Some(current);
        }
        self.last_time = previous;
        Ok(())
    }
}

#[async_trait::async_trait]
impl StreamSource for ArrowSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::Unsupported,
            delivery: SourceDeliveryCapability::Lossy,
            max_batch_rows: self.budget.max_rows,
            max_batch_bytes: self.budget.max_bytes,
            schema: SourceSchema::Exact(self.schema.clone()),
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if cursor.is_some() {
            return Err(invalid(
                "source.arrow.cursor",
                "materialized input does not support replay",
            ));
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let Some(record) = self.current.take().or_else(|| self.records.next()) else {
            return Ok(None);
        };
        let batch = self.bounded_batch(&record)?;
        self.validate_time(&batch)?;
        let rows = batch.num_rows();
        if rows < record.num_rows() {
            self.current = Some(record.slice(rows, record.num_rows() - rows));
        }
        self.position += 1;
        Ok(Some(SourceEvent::Data {
            batch,
            cursor: Cursor::unbound(self.position.to_be_bytes().to_vec(), JsonMap::new())?,
        }))
    }

    async fn close(&mut self) -> Result<()> {
        self.current = None;
        self.records = Vec::new().into_iter();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{
        array::ArrayRef,
        datatypes::{Field, Schema},
    };

    use super::*;

    #[tokio::test]
    async fn close_releases_unread_and_partial_arrow_owners_and_rejects_replay() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("value", DataType::Int64, false).with_metadata(
                    std::collections::HashMap::from([("unit".into(), "count".into())]),
                ),
            ])
            .with_metadata(std::collections::HashMap::from([(
                "origin".into(),
                "caller".into(),
            )])),
        );
        let record = RecordBatch::try_new(schema, vec![values.clone()]).unwrap();
        let metadata = BatchMetadata::new("caller", 17, JsonMap::new()).unwrap();
        let descriptor = PyArrowSource {
            batch: Batch::table(vec![record.clone(), record], metadata.clone()).unwrap(),
            budget: EdgeBudget::new(1, 1024).unwrap(),
            ordered_column: None,
        };
        let mut source = descriptor.source().unwrap();
        drop(descriptor);
        let cursor = Cursor::unbound(vec![1], JsonMap::new()).unwrap();
        assert!(source.open(Some(cursor)).await.is_err());
        source.open(None).await.unwrap();
        let Some(SourceEvent::Data { batch, .. }) = source.next().await.unwrap() else {
            panic!("expected first slice");
        };
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.metadata(), &metadata);
        assert_eq!(batch.table_payload().unwrap().schema(), &source.schema);
        drop(batch);
        assert!(Arc::strong_count(&values) > 1);
        source.close().await.unwrap();
        source.close().await.unwrap();
        assert_eq!(Arc::strong_count(&values), 1);
        assert!(source.next().await.unwrap().is_none());
    }
}
