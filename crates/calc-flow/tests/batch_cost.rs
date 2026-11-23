mod support;

use std::{any::Any, sync::Arc};

use calc_flow::{Batch, BatchMetadata, ExternalPayload};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Decimal128Array, Int64Array, ListArray, StringArray,
        TimestampNanosecondArray, UInt16Array,
    },
    buffer::{OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Int64Type, Schema},
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
    let record = RecordBatch::try_from_iter([
        (
            "integer",
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
        ),
        (
            "string",
            Arc::new(StringArray::from(vec!["one", "two"])) as ArrayRef,
        ),
        (
            "list",
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>([
                Some(vec![Some(1)]),
                Some(vec![Some(2), Some(3)]),
            ])) as ArrayRef,
        ),
    ])
    .unwrap();

    for empty in [RecordBatch::new_empty(record.schema()), record.slice(1, 0)] {
        let batch = Batch::table(vec![empty.clone()], BatchMetadata::default()).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.estimated_bytes().unwrap(), 0);

        let expected = record
            .columns()
            .iter()
            .map(|column| column.to_data().get_slice_memory_size().unwrap())
            .sum::<usize>();
        let mixed = Batch::table(
            vec![empty.clone(), record.clone(), empty],
            BatchMetadata::default(),
        )
        .unwrap();
        assert_eq!(mixed.num_rows(), record.num_rows());
        assert_eq!(mixed.estimated_bytes().unwrap(), expected);
    }
}

#[test]
fn primitive_slice_cost_only_allocates_for_an_ambiguous_validity_bitmap() {
    let quantities = (0..20_u16)
        .map(|i| (i != 9).then_some(i))
        .collect::<UInt16Array>();
    let times = TimestampNanosecondArray::from_iter_values(0..20).with_timezone("UTC");
    let prices = Decimal128Array::from_iter_values(0..20)
        .with_precision_and_scale(12, 2)
        .unwrap();
    let record = RecordBatch::try_from_iter([
        ("quantity", Arc::new(quantities) as ArrayRef),
        ("time", Arc::new(times) as ArrayRef),
        ("price", Arc::new(prices) as ArrayRef),
    ])
    .unwrap();
    let records = vec![record.slice(1, 9), record.slice(10, 8), record.slice(5, 0)];
    assert_eq!(records[0].column(0).null_count(), 1);
    assert_eq!(records[1].column(0).null_count(), 0);
    let all_valid = records[1].column(0);
    assert!(all_valid.nulls().is_some());
    // Slicing before or after exporting ArrayData can retain or discard the bitmap.
    let retained = record.column(0).to_data().slice(10, 8);
    assert_eq!(retained.get_slice_memory_size().unwrap(), 17);
    let fallback = allocation_counter::measure(|| {
        assert_eq!(all_valid.to_data().get_slice_memory_size().unwrap(), 16);
    });
    let expected = records
        .iter()
        .filter(|r| r.num_rows() > 0)
        .flat_map(RecordBatch::columns)
        .map(|column| column.to_data().get_slice_memory_size().unwrap())
        .sum::<usize>();
    assert_eq!(expected, 17 * (2 + 8 + 16) + 2);
    let batch = Batch::table(records, BatchMetadata::default()).unwrap();
    let measured =
        allocation_counter::measure(|| assert_eq!(batch.estimated_bytes().unwrap(), expected));
    eprintln!("primitive slice charge {measured:?}; all-valid Arrow fallback {fallback:?}");
    assert_eq!(measured.count_total, fallback.count_total, "{measured:?}");
    assert_eq!(measured.bytes_total, fallback.bytes_total, "{measured:?}");
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
