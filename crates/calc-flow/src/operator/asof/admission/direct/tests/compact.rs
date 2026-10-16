use super::*;
use datafusion::{
    arrow::{
        array::{
            ArrayRef, BinaryArray, BooleanArray, Float64Array, LargeBinaryArray, LargeStringArray,
            StringArray, StringViewArray,
        },
        buffer::NullBuffer,
        datatypes::{Field, IntervalUnit, Schema, TimeUnit},
    },
    common::ScalarValue,
};

fn encode(record: &RecordBatch) -> Vec<u8> {
    super::super::super::super::codec::encode_batch(record, usize::MAX, &mut Vec::new()).unwrap()
}

fn reference(record: &RecordBatch) -> Result<RecordBatch> {
    if can_share_batch(record)? {
        return Ok(record.clone());
    }
    let indices = UInt64Array::from_iter_values(0..record.num_rows() as u64);
    let columns = record
        .columns()
        .iter()
        .map(|column| {
            if column.data_type() == &DataType::FixedSizeBinary(0) {
                datafusion::arrow::array::FixedSizeBinaryArray::try_new_with_len(
                    0,
                    datafusion::arrow::buffer::Buffer::from(Vec::<u8>::new()),
                    column.nulls().cloned(),
                    column.len(),
                )
                .map(|array| Arc::new(array) as ArrayRef)
            } else {
                datafusion::arrow::compute::take(column, &indices, None)
            }
            .map_err(|error| super::super::super::super::arrow_error(&error))
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(record.schema(), columns)
        .map_err(|error| super::super::super::super::arrow_error(&error))
}

fn with_payload(payload: ArrayRef) -> RecordBatch {
    let input = string_unsigned_input(payload.len());
    let base = &input.table_payload().unwrap().batches()[0];
    let mut fields = base.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "payload",
        payload.data_type().clone(),
        true,
    )));
    let mut columns = base.columns().to_vec();
    columns.push(payload);
    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            fields,
            std::collections::HashMap::from([("note".into(), "copy界".into())]),
        )),
        columns,
    )
    .unwrap()
}

fn assert_independent(record: &RecordBatch, copied: &RecordBatch) {
    for original in record.columns() {
        for source in original.to_data().buffers() {
            if source.is_empty() {
                continue;
            }
            let start = source.as_ptr() as usize;
            let end = start + source.len();
            for column in copied.columns() {
                for buffer in column.to_data().buffers() {
                    let other_start = buffer.as_ptr() as usize;
                    let other_end = other_start + buffer.len();
                    assert!(
                        buffer.is_empty() || end <= other_start || other_end <= start,
                        "copied payload aliases a caller buffer"
                    );
                }
            }
        }
    }
}

fn assert_compact(record: &RecordBatch, expected_indices: usize, owned: bool) {
    assert!(!can_share_batch(record).unwrap());
    let before = encode(record);
    let expected = encode(&reference(record).unwrap());
    COMPLETE_INDICES.with(|count| count.set(0));
    let actual = compact_complete_record(record).unwrap();
    let indices = COMPLETE_INDICES.with(|count| count.replace(0));
    assert_eq!(indices, expected_indices);
    assert_eq!(actual.schema(), record.schema());
    assert_eq!(encode(&actual), expected);
    assert_eq!(encode(record), before);
    if owned {
        assert_independent(record, &actual);
    }
}

fn primitive_payloads() {
    let types = [
        DataType::Boolean,
        DataType::Int8,
        DataType::Int64,
        DataType::UInt64,
        DataType::Float16,
        DataType::Float32,
        DataType::Float64,
        DataType::Date32,
        DataType::Date64,
        DataType::Time32(TimeUnit::Second),
        DataType::Time64(TimeUnit::Nanosecond),
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        DataType::Duration(TimeUnit::Microsecond),
        DataType::Interval(IntervalUnit::MonthDayNano),
        DataType::Decimal32(8, 2),
        DataType::Decimal64(16, 2),
        DataType::Decimal128(30, 2),
        DataType::Decimal256(70, 2),
        DataType::FixedSizeBinary(0),
        DataType::FixedSizeBinary(8),
        DataType::Utf8,
        DataType::LargeUtf8,
        DataType::Binary,
        DataType::LargeBinary,
    ];
    for data_type in types {
        let array = ScalarValue::new_default(&data_type)
            .unwrap()
            .to_array_of_size(80)
            .unwrap();
        assert!(array.nulls().is_none(), "{data_type:?}");
        assert_compact(&with_payload(array).slice(7, 65), 0, true);
    }
}

fn variable_and_bit_payloads() {
    let values = (0..80)
        .map(|row| "界abc".repeat(row % 17))
        .collect::<Vec<_>>();
    let bytes = values.iter().map(String::as_bytes).collect::<Vec<_>>();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(&values)),
        Arc::new(LargeStringArray::from_iter_values(&values)),
        Arc::new(BinaryArray::from_iter_values(bytes.clone())),
        Arc::new(LargeBinaryArray::from_iter_values(bytes)),
        Arc::new(BooleanArray::from(
            (0..80).map(|row| row % 3 == 1).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from_iter_values((0..80).map(|row| {
            f64::from_bits([0, 1 << 63, 0x7ff8_0000_0000_0042, 0x7ff0_0000_0000_0000][row % 4])
        }))),
    ];
    for array in arrays {
        assert_compact(&with_payload(array).slice(7, 65), 0, true);
    }
}

fn null_and_view_payloads() {
    let strings = StringArray::from_iter_values((0..80).map(|row| format!("hidden-{row}")));
    for validity in [
        NullBuffer::new_valid(80),
        NullBuffer::from((0..80).map(|row| row % 3 != 1).collect::<Vec<_>>()),
    ] {
        let array = Arc::new(StringArray::new(
            strings.offsets().clone(),
            strings.values().clone(),
            Some(validity),
        ));
        assert!(array.nulls().is_some());
        assert_compact(&with_payload(array).slice(7, 65), 65, true);
    }
    let view = Arc::new(StringViewArray::from_iter_values(
        (0..80).map(|row| format!("long-view-value-{row}")),
    ));
    assert_compact(&with_payload(view).slice(7, 65), 65, false);
}

fn allocation_fits_existing_workspace(rows: usize) {
    let input = string_unsigned_input(rows + 16);
    let record = input.table_payload().unwrap().batches()[0].slice(7, rows);
    let input = Batch::table(vec![record.clone()], crate::BatchMetadata::default()).unwrap();
    let mut operator = fixture(&record.schema());
    let validated = operator.validate_admission("left", &input).unwrap();
    let reservation = operator.input_workspace(&input, validated).unwrap();
    let mut previous = None;
    let original_allocation = allocation_counter::measure(|| {
        previous = Some(reference(&record).unwrap());
    });
    let expected = encode(&previous.unwrap());
    let mut copied = None;
    let allocation = allocation_counter::measure(|| {
        copied = Some(compact_complete_record(&record).unwrap());
    });
    assert!(usize::try_from(allocation.bytes_max).unwrap() <= reservation.size());
    assert_eq!(encode(&copied.unwrap()), expected);
    eprintln!(
        "complete-copy rows={rows}; previous={original_allocation:?}; allocation={allocation:?}; reservation={}",
        reservation.size()
    );
}

fn complete_records_share_exact_buffers() {
    let array: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..128));
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::UInt64,
            false,
        )])),
        vec![array.clone()],
    )
    .unwrap();
    assert!(can_share_batch(&record).unwrap());
    COMPLETE_INDICES.with(|count| count.set(0));
    let shared = compact_complete_record(&record).unwrap();
    assert!(Arc::ptr_eq(shared.column(0), &array));
    assert_eq!(COMPLETE_INDICES.with(|count| count.replace(0)), 0);
}

pub(super) fn complete_records_avoid_identity_indices() {
    let input = string_unsigned_input(4_097);
    let original = &input.table_payload().unwrap().batches()[0];
    let selected = original.slice(1, original.num_rows() - 2);
    assert!(!can_share_batch(&selected).unwrap());
    COMPLETE_INDICES.with(|count| count.set(0));
    let compact = compact_complete_record(&selected).unwrap();
    let indices = COMPLETE_INDICES.with(|count| count.replace(0));
    assert_eq!(compact, selected);
    assert_eq!(
        indices, 0,
        "complete batch built {indices} identity indices"
    );
    complete_records_share_exact_buffers();
    for rows in [0, 1, 63, 64, 65, 127, 1_024] {
        let input = string_unsigned_input(rows + 16);
        let record = input.table_payload().unwrap().batches()[0].slice(7, rows);
        assert_compact(&record, if rows < 64 { rows } else { 0 }, true);
    }
    primitive_payloads();
    variable_and_bit_payloads();
    null_and_view_payloads();
    for rows in [64, 1_024] {
        allocation_fits_existing_workspace(rows);
    }
}

#[test]
fn test_zero_width_payload_compaction_preserves_rows_and_selected_validity() {
    use datafusion::arrow::{array::FixedSizeBinaryArray, buffer::Buffer};

    for rows in [1, 63, 64, 65] {
        for validity in [
            None,
            Some(NullBuffer::new_valid(rows)),
            Some(NullBuffer::from(
                (0..rows).map(|row| row % 3 != 0).collect::<Vec<_>>(),
            )),
        ] {
            let payload: ArrayRef = Arc::new(
                FixedSizeBinaryArray::try_new_with_len(
                    0,
                    Buffer::from(Vec::<u8>::new()),
                    validity,
                    rows,
                )
                .unwrap(),
            );
            let input = with_payload(payload);
            let before = encode(&input);
            let record = input.slice(0, rows);
            let actual = compact_complete_record(&record).unwrap();
            assert_eq!(actual.num_rows(), rows);
            assert_eq!(encode(&actual), before);
            assert_independent(&record, &actual);
            let selections = (0..rows).filter(|row| row % 2 == 0).collect::<Vec<_>>();
            let accepted = selections
                .iter()
                .map(|&row| {
                    (
                        (
                            0,
                            state::Encoding::from_slice(&[]),
                            state::Encoding::from_slice(&[]),
                        ),
                        &record,
                        row,
                        0,
                    )
                })
                .collect::<Vec<_>>();
            let selected = super::super::super::compact_accepted_rows(&accepted, &record).unwrap();
            assert_eq!(selected.num_rows(), selections.len());
            for (index, &row) in selections.iter().enumerate() {
                assert_eq!(
                    selected.columns().last().unwrap().is_null(index),
                    record.columns().last().unwrap().is_null(row)
                );
            }
            assert_independent(&record, &selected);
            assert_eq!(encode(&input), before);
        }
    }
}
