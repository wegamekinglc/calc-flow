//! Materialized Arrow input: import once, then poll without the Python source bridge.

use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, Cursor, EdgeBudget, JsonMap, NativeWatermarkCapability,
    ReplayPositioning, Result, SourceCapabilities, SourceDeliveryCapability, SourceEvent,
    SourceSchema, StreamSource,
};
use datafusion::arrow::{
    array::AsArray,
    compute::cast,
    datatypes::{DataType, Int64Type, SchemaRef},
    record_batch::RecordBatch,
};
use pyo3::{exceptions::PyValueError, prelude::*};

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

fn last_ordered_time(
    batch: &Batch,
    column: &str,
    mut previous: Option<i64>,
) -> std::result::Result<Option<i64>, &'static str> {
    let table = batch.table_payload().map_err(|_| "expected table batch")?;
    let index = table
        .schema()
        .index_of(column)
        .map_err(|_| "missing event time column")?;
    let columns = table.batches().iter().map(|record| record.column(index));
    if columns.clone().any(|column| column.null_count() != 0) {
        return Err("expected non-null event time");
    }
    for column in columns {
        let integers = cast(column, &DataType::Int64).map_err(|_| "expected timestamp column")?;
        let values = integers.as_primitive::<Int64Type>().values();
        if let Some(&first) = values.first() {
            if previous.is_some_and(|last| first < last) || !values.is_sorted() {
                return Err(
                    "expected nondecreasing event time; select an explicit watermark policy for out-of-order input",
                );
            }
            previous = values.last().copied();
        }
    }
    Ok(previous)
}

#[pyfunction]
#[pyo3(signature = (batch, column, previous))]
pub(crate) fn _ordered_event_time(
    batch: &PyBatch,
    column: &str,
    previous: Option<i64>,
) -> PyResult<Option<i64>> {
    last_ordered_time(&batch.clone_inner()?, column, previous).map_err(PyValueError::new_err)
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
        let rows = self.fitting_prefix_rows(record, rows)?;
        if rows == 0 {
            return Err(invalid(
                "source.arrow.max_bytes",
                "one row exceeds the source byte budget",
            ));
        }
        self.slice(record, rows)
    }

    fn fitting_prefix_rows(&self, record: &RecordBatch, upper: usize) -> Result<usize> {
        // Visible Arrow slice charges are monotone; find a fitting prefix without copying.
        let (mut low, mut high) = (0, upper);
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if self.slice(record, middle)?.estimated_bytes()? <= self.budget.max_bytes {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        Ok(low)
    }

    fn validate_time(&mut self, batch: &Batch) -> Result<()> {
        let Some(column) = &self.ordered_column else {
            return Ok(());
        };
        self.last_time = last_ordered_time(batch, column, self.last_time)
            .map_err(|message| invalid("source.arrow.event_time", message))?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl StreamSource for ArrowSource {
    fn boundedness(&self) -> calc_flow::SourceBoundedness {
        calc_flow::SourceBoundedness::Bounded
    }
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
        array::{ArrayRef, Int64Array},
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
