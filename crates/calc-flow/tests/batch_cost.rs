mod support;

use std::{any::Any, sync::Arc};

use calc_flow::{Batch, BatchMetadata, ExternalPayload};
use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Array, ListArray},
    buffer::{OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

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
fn table_batch_estimates_the_visible_window_of_a_sliced_list_column() {
    let item_field = Arc::new(Field::new("item", DataType::Int64, false));
    let offsets = OffsetBuffer::new(ScalarBuffer::from(
        (0..=16_i32).map(|index| index * 4).collect::<Vec<_>>(),
    ));
    let base = Arc::new(ListArray::new(
        Arc::clone(&item_field),
        offsets,
        Arc::new(Int64Array::from((0..64).collect::<Vec<_>>())),
        None,
    )) as ArrayRef;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::List(item_field),
        false,
    )]));
    let record = RecordBatch::try_new(schema, vec![base.slice(2, 1)]).unwrap();
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();

    let estimated = batch.estimated_bytes().unwrap();
    let column = batch.table_payload().unwrap().batches()[0]
        .column(0)
        .clone();
    assert_eq!(estimated, column.to_data().get_slice_memory_size().unwrap());
    // The visible child window holds 4 i64 values; the estimate must never
    // under-report them, and must stay below the shared base allocation.
    assert!(estimated >= 4 * 8, "{estimated}");
    assert!(estimated < base.to_data().get_array_memory_size());
}

#[test]
fn external_batch_uses_the_payload_provided_byte_estimate() {
    let batch = Batch::external(Arc::new(MeasuredArray), BatchMetadata::default()).unwrap();

    assert_eq!(batch.estimated_bytes().unwrap(), 41);
    assert_eq!(batch.num_rows(), 3);
}
