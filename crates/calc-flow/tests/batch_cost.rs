mod support;

use std::{any::Any, sync::Arc};

use calc_flow::{Batch, BatchMetadata, BatchingSource, CalcFlowError, ExternalPayload, Source};
use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use serde_json::json;
use support::{QueueSource, source_item};

#[derive(Debug)]
struct MeasuredArray;

impl ExternalPayload for MeasuredArray {
    fn backend(&self) -> &'static str {
        "test"
    }

    fn len(&self) -> usize {
        3
    }

    fn estimated_bytes(&self) -> usize {
        41
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn table_batch_estimates_the_visible_slice_bytes_of_every_column_and_record() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    let base = Arc::new(Int64Array::from((0..64).collect::<Vec<_>>())) as ArrayRef;
    let record = |offset, len| {
        RecordBatch::try_new(Arc::clone(&schema), vec![base.slice(offset, len)]).unwrap()
    };

    let batch = Batch::table(vec![record(8, 2), record(32, 1)], BatchMetadata::default()).unwrap();

    // Three visible i64 values occupy exactly 24 bytes even though both
    // records share a much larger backing allocation.
    assert_eq!(batch.estimated_bytes().unwrap(), 24);
    assert_eq!(
        batch.table_payload().unwrap().estimated_bytes().unwrap(),
        24
    );
    assert!(batch.estimated_bytes().unwrap() < 64 * 8);
}

#[test]
fn table_batch_estimates_zero_rows_as_zero_bytes() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    let record = RecordBatch::new_empty(schema);

    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();

    assert_eq!(batch.num_rows(), 0);
    assert_eq!(batch.estimated_bytes().unwrap(), 0);
}

#[test]
fn external_batch_uses_the_payload_provided_byte_estimate() {
    let batch = Batch::external(Arc::new(MeasuredArray), BatchMetadata::default()).unwrap();

    assert_eq!(batch.estimated_bytes().unwrap(), 41);
    assert_eq!(batch.num_rows(), 3);
}

#[tokio::test]
async fn batching_source_fails_a_single_item_over_the_row_limit_and_latches() {
    let (source, _) = QueueSource::new(vec![source_item(&[1, 2], json!(2), 1)]);
    let mut source = BatchingSource::new(source, 1, usize::MAX).unwrap();
    source.open(None).await.unwrap();

    assert!(matches!(
        source.next().await,
        Err(CalcFlowError::InvalidArgument { ref field, ref message })
            if field == "source.batch" && message.contains("rows")
    ));
    assert!(matches!(
        source.next().await,
        Err(CalcFlowError::InvalidArgument { ref field, .. }) if field == "source"
    ));
}

#[tokio::test]
async fn batching_source_delivers_valid_groups_before_failing_an_oversize_item() {
    let small = source_item(&[1, 2], json!(2), 1);
    let bytes = arrow_bytes(&small.batch);
    let (source, _) = QueueSource::new(vec![small, source_item(&[3, 4, 5, 6], json!(6), 2)]);
    let mut source = BatchingSource::new(source, usize::MAX, bytes + 1).unwrap();
    source.open(None).await.unwrap();

    let first = source.next().await.unwrap().unwrap();
    assert_eq!(first.batch.num_rows(), 2);
    assert!(matches!(
        source.next().await,
        Err(CalcFlowError::InvalidArgument { ref field, ref message })
            if field == "source.batch" && message.contains("bytes")
    ));
}

#[tokio::test]
async fn batching_source_reports_every_exceeded_limit_in_the_oversize_error() {
    let item = source_item(&[1, 2, 3, 4], json!(4), 1);
    let bytes = arrow_bytes(&item.batch);
    let (source, _) = QueueSource::new(vec![item]);
    let mut source = BatchingSource::new(source, 3, bytes - 1).unwrap();
    source.open(None).await.unwrap();

    match source.next().await {
        Err(CalcFlowError::InvalidArgument { field, message }) => {
            assert_eq!(field, "source.batch");
            assert!(message.contains("rows"), "{message}");
            assert!(message.contains("bytes"), "{message}");
            let rows_at = message.find("rows").unwrap();
            let bytes_at = message.find("bytes").unwrap();
            assert!(rows_at < bytes_at, "{message}");
        }
        other => panic!("expected an oversize error, got {other:?}"),
    }
}

fn arrow_bytes(batch: &Batch) -> usize {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(RecordBatch::columns)
        .map(|column| column.to_data().get_slice_memory_size().unwrap())
        .sum()
}
