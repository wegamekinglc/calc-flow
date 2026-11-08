use super::{ArrayDType, NullBuffer, validate_parts, validate_vec, visible_bytes};
use crate::{CalcFlowError, Result};

fn assert_bytes_overflow(result: Result<impl Sized>) {
    assert!(matches!(
        result,
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "input.bytes"
    ));
}

#[test]
fn test_visible_byte_accounting_checks_products_and_sums() {
    assert_bytes_overflow(visible_bytes(usize::MAX, ArrayDType::Float64, None, 0));
    assert_bytes_overflow(visible_bytes(0, ArrayDType::Float32, None, usize::MAX));
    let length = usize::MAX / size_of::<f32>();
    assert_bytes_overflow(visible_bytes(length, ArrayDType::Float32, None, 1));
    let validity = NullBuffer::new_valid(1);
    assert_bytes_overflow(visible_bytes(
        length,
        ArrayDType::Float32,
        Some(&validity),
        0,
    ));
}

#[test]
fn test_retained_byte_accounting_checks_capacity_products_and_sums() {
    assert_bytes_overflow(validate_vec(0, usize::MAX, ArrayDType::Float64, &[0], None));
    assert_bytes_overflow(validate_parts(
        1,
        ArrayDType::Float32,
        &[1],
        None,
        usize::MAX,
    ));
    let validity = NullBuffer::new_valid(1);
    assert_bytes_overflow(validate_parts(
        1,
        ArrayDType::Float32,
        &[1],
        Some(&validity),
        usize::MAX,
    ));
}
