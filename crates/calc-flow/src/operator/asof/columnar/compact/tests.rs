use super::compact_accepted;
use datafusion::arrow::{
    array::{Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use std::sync::Arc;

#[test]
fn full_admission_copies_only_the_logical_primitive_range() {
    const ROWS: usize = 32_768;
    let values = Int64Array::from_iter_values(0..i64::try_from(3 * ROWS).unwrap());
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(values)],
    )
    .unwrap()
    .slice(ROWS, ROWS);
    let accepted = vec![true; ROWS];
    let mut result = None;
    let memory = allocation_counter::measure(|| {
        result = Some(compact_accepted(&input, &accepted, ROWS).unwrap());
    });
    let result = result.unwrap();
    assert_eq!(result, input);
    let data = result.column(0).to_data();
    let source = input.column(0).to_data();
    assert_ne!(data.buffers()[0].data_ptr(), source.buffers()[0].data_ptr());
    assert_eq!(data.buffers()[0].capacity(), ROWS * size_of::<i64>());
    assert!(
        memory.bytes_max < u64::try_from(ROWS * size_of::<i64>() + 16 * 1024).unwrap(),
        "full admission should allocate the output without a per-row index: {memory:?}"
    );
}

#[test]
fn full_copy_rebases_sliced_strings_and_preserves_temporal_metadata() {
    use datafusion::arrow::array::{LargeBinaryArray, StringArray, TimestampMicrosecondArray};

    let prefix = "p".repeat(65_536);
    let strings = StringArray::from(vec![prefix.as_str(), "月", "", "è", prefix.as_str()]);
    let bytes = LargeBinaryArray::from_iter_values(["skip", "", "a", "bc", "skip"]);
    let times = TimestampMicrosecondArray::from_iter_values(0..5).with_timezone("UTC");
    let input = RecordBatch::try_from_iter([
        (
            "text",
            Arc::new(strings) as datafusion::arrow::array::ArrayRef,
        ),
        ("bytes", Arc::new(bytes)),
        ("time", Arc::new(times)),
    ])
    .unwrap()
    .slice(1, 3);
    let copied = compact_accepted(&input, &[true; 3], 3).unwrap();
    assert_eq!(copied, input);
    for (expected, actual) in input.columns().iter().zip(copied.columns()) {
        let expected = expected.to_data();
        let actual = actual.to_data();
        for (source, retained) in expected.buffers().iter().zip(actual.buffers()) {
            assert_ne!(source.data_ptr(), retained.data_ptr());
        }
    }
    let text = copied
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(text.value_offsets(), &[0, 3, 3, 5]);
    assert_eq!(text.to_data().buffers()[1].capacity(), 5);
}

#[test]
fn nullable_and_partial_admission_preserve_selected_values() {
    use datafusion::arrow::array::StringArray;

    let strings = Arc::new(StringArray::from(vec![
        Some("x"),
        None,
        Some("y"),
        Some("z"),
    ]));
    let input =
        RecordBatch::try_from_iter([("text", strings as datafusion::arrow::array::ArrayRef)])
            .unwrap();
    let all = compact_accepted(&input, &[true; 4], 4).unwrap();
    assert_eq!(all, input);
    let some = compact_accepted(&input, &[false, true, false, true], 2).unwrap();
    let expected = StringArray::from(vec![None, Some("z")]);
    assert_eq!(some.column(0).to_data(), expected.to_data());
}
