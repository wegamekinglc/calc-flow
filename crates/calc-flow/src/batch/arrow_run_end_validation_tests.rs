use super::{Arc, Buffer, DataType, Field};
use arrow_data::ArrayData;

fn run_end_data() -> ArrayData {
    let child = |values: [i32; 2]| {
        ArrayData::builder(DataType::Int32)
            .len(values.len())
            .add_buffer(Buffer::from_slice_ref(values))
            .build()
            .unwrap()
    };
    let dtype = DataType::RunEndEncoded(
        Arc::new(Field::new("run_ends", DataType::Int32, false)),
        Arc::new(Field::new("values", DataType::Int32, true)),
    );
    ArrayData::builder(dtype)
        .len(3)
        .child_data(vec![child([2, 3]), child([7, 9])])
        .build()
        .unwrap()
}

#[test]
fn test_run_end_validation_rejects_logical_length_beyond_last_run() {
    let data = run_end_data();
    data.validate_full().unwrap();
    let error = data
        .into_builder()
        .len(5)
        .build()
        .expect_err("logical rows beyond the last run must be rejected");
    assert!(error.to_string().contains("offset + length"));
}

#[test]
fn test_run_end_validation_checks_parent_offset_and_preserves_valid_slice() {
    let data = run_end_data();
    let valid = data
        .clone()
        .into_builder()
        .offset(1)
        .len(2)
        .build()
        .unwrap();
    valid.validate_full().unwrap();
    assert_eq!((valid.offset(), valid.len()), (1, 2));
    let error = data
        .into_builder()
        .offset(1)
        .len(3)
        .build()
        .expect_err("offset plus length beyond the last run must be rejected");
    assert!(error.to_string().contains("offset + length"));
}
