use std::{collections::BTreeMap, mem::size_of, sync::Arc};

use calc_flow::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, ExternalPayload,
    array::{ArrayDType, FloatValues, FloatView, NativeArray},
};
use datafusion::arrow::{
    array::{Array, ArrayRef, Float32Array, Int64Array},
    buffer::{NullBuffer, ScalarBuffer},
};

#[test]
fn test_float_view_borrows_exact_float_storage() {
    let values = [1.0_f64, f64::NAN, f64::INFINITY];
    let view = FloatView::from_f64(&values);
    assert_eq!(view.dtype(), ArrayDType::Float64);
    assert_eq!(view.len(), values.len());
    assert!(!view.is_empty());
    assert!(view.validity().is_none());
    let FloatValues::Float64(borrowed) = view.values() else {
        panic!("expected Float64 view");
    };
    assert!(std::ptr::eq(borrowed.as_ptr(), values.as_ptr()));
    assert!(borrowed[1].is_nan());
    assert_eq!(borrowed[2].to_bits(), f64::INFINITY.to_bits());
    assert!(FloatView::from_f32(&[]).is_empty());
}

#[test]
fn test_native_array_takes_owned_values_and_preserves_ieee_bits() {
    let nan_bits = 0x7ff8_0000_0000_0042;
    let mut values = Vec::with_capacity(32);
    values.extend([-0.0_f64, f64::from_bits(nan_bits), f64::INFINITY]);
    let retained = values.capacity() * size_of::<f64>() + size_of::<usize>();
    let pointer = values.as_ptr();
    let array = NativeArray::from_f64(values, vec![3], None).unwrap();
    assert_eq!(array.dtype(), ArrayDType::Float64);
    assert_eq!(array.shape(), &[3]);
    assert_eq!(array.len(), 3);
    assert!(!array.is_empty());
    assert_eq!(array.estimated_bytes(), 24 + size_of::<usize>());
    assert_eq!(array.retained_buffer_bytes(), retained);
    let cloned = array.clone();
    assert!(Arc::ptr_eq(array.as_arrow(), cloned.as_arrow()));
    drop(array);
    let FloatValues::Float64(values) = cloned.float_view().unwrap().values() else {
        panic!("expected Float64 storage");
    };
    assert!(std::ptr::eq(values.as_ptr(), pointer));
    assert_eq!(values[0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(values[1].to_bits(), nan_bits);
    assert_eq!(values[2].to_bits(), f64::INFINITY.to_bits());

    let scalar = NativeArray::from_f32(vec![-0.0], vec![], None).unwrap();
    assert_eq!(scalar.dtype(), ArrayDType::Float32);
    assert!(scalar.shape().is_empty());
    assert_eq!(scalar.len(), 1);
    assert_eq!(scalar.estimated_bytes(), 4);
    let FloatValues::Float32(values) = scalar.float_view().unwrap().values() else {
        panic!("expected Float32 scalar");
    };
    assert_eq!(values[0].to_bits(), (-0.0_f32).to_bits());
}

#[test]
fn test_native_array_shares_arrow_slices_and_accounts_for_validity_offsets() {
    let values = ScalarBuffer::from(vec![3.0_f32; 16]);
    let mut validity = vec![true; 16];
    validity[8] = false;
    let original = Float32Array::new(values, Some(NullBuffer::from(validity)));
    let sliced: ArrayRef = Arc::new(original.slice(7, 3));
    let retained = original.values().inner().capacity()
        + original.nulls().unwrap().buffer().capacity()
        + size_of::<usize>();
    let array = NativeArray::from_arrow(Arc::clone(&sliced), vec![3]).unwrap();
    assert!(Arc::ptr_eq(array.as_arrow(), &sliced));
    assert_eq!(array.estimated_bytes(), 12 + 2 + size_of::<usize>());
    assert_eq!(array.retained_buffer_bytes(), retained);
    assert!(array.retained_buffer_bytes() > array.estimated_bytes());
    let view = array.float_view().unwrap();
    assert_eq!(view.validity().unwrap().offset(), 7);
    assert!(view.validity().unwrap().is_valid(0));
    assert!(view.validity().unwrap().is_null(1));
    assert!(view.validity().unwrap().is_valid(2));
    assert_eq!(original.len(), 16);
    assert!(original.is_null(8));
}

#[test]
fn test_native_array_preserves_explicit_empty_and_all_valid_bitmaps() {
    let empty = NativeArray::from_f64(vec![], vec![0], Some(NullBuffer::new_valid(0))).unwrap();
    assert!(empty.is_empty());
    assert!(empty.float_view().unwrap().validity().is_some());
    assert_eq!(empty.estimated_bytes(), size_of::<usize>());

    let validity = NullBuffer::new_valid(9).slice(7, 2);
    let array = NativeArray::from_f32(vec![1.0, 2.0], vec![2], Some(validity)).unwrap();
    assert!(array.float_view().unwrap().validity().is_some());
    assert_eq!(array.estimated_bytes(), 8 + 2 + size_of::<usize>());
}

fn assert_invalid_field(error: CalcFlowError, expected: &str) {
    match error {
        CalcFlowError::InvalidArgument { field, .. } => assert_eq!(field, expected),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn test_native_array_rejects_invalid_descriptors() {
    let integers: ArrayRef = Arc::new(Int64Array::from(vec![1]));
    assert_invalid_field(
        NativeArray::from_arrow(integers, vec![1]).unwrap_err(),
        "input.dtype",
    );
    assert_invalid_field(
        NativeArray::from_f64(vec![1.0], vec![1, 1], None).unwrap_err(),
        "input.shape",
    );
    assert_invalid_field(
        NativeArray::from_f32(vec![1.0], vec![usize::MAX, 2], None).unwrap_err(),
        "input.shape",
    );
    assert_invalid_field(
        NativeArray::from_f64(vec![], vec![], None).unwrap_err(),
        "input.shape",
    );
    assert_invalid_field(
        NativeArray::from_f64(vec![1.0], vec![2], None).unwrap_err(),
        "input.shape",
    );
    assert_invalid_field(
        NativeArray::from_f32(vec![1.0], vec![1], Some(NullBuffer::new_valid(2))).unwrap_err(),
        "input.validity",
    );
}

#[test]
fn test_native_array_enters_batch_without_static_snapshot_and_hides_values() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<NativeArray>();
    let array = NativeArray::from_f64(vec![123_456.75], vec![], None).unwrap();
    let debug = format!("{array:?}");
    assert!(!debug.contains("123456"));
    let expected_bytes = array.estimated_bytes();
    let expected_retained = array.retained_buffer_bytes();
    let metadata = BatchMetadata::new("native-array-test", 4, BTreeMap::new()).unwrap();
    let batch = Batch::external(Arc::new(array), metadata.clone()).unwrap();
    assert_eq!(batch.kind(), BatchKind::Array);
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.estimated_bytes().unwrap(), expected_bytes);
    assert_eq!(batch.metadata(), &metadata);
    assert!(batch.static_array_snapshot().is_none());
    let payload = batch.external_payload().unwrap();
    assert_eq!(payload.backend(), "native");
    let native = payload.as_any().downcast_ref::<NativeArray>().unwrap();
    assert_eq!(ExternalPayload::len(native), 1);
    assert_eq!(native.retained_buffer_bytes(), expected_retained);
    let exported: ArrayRef = Arc::clone(native.as_arrow());
    drop(batch);
    assert_eq!(exported.len(), 1);
}

#[test]
fn test_float_view_preserves_sliced_validity_and_rejects_wrong_length() {
    let values = [5.0_f32, 7.0];
    let original = NullBuffer::from(vec![true, false, true, true]);
    let validity = original.slice(1, values.len());
    let view = FloatView::try_new(FloatValues::Float32(&values), Some(&validity)).unwrap();
    assert_eq!(view.dtype(), ArrayDType::Float32);
    assert_eq!(view.validity().unwrap().offset(), 1);
    assert!(view.validity().unwrap().is_null(0));
    assert!(view.validity().unwrap().is_valid(1));
    let error = FloatView::try_new(FloatValues::Float32(&values), Some(&original)).unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { field, .. } if field == "input.validity"
    ));
    let debug = format!("{view:?}");
    assert!(debug.contains("Float32"));
    assert!(!debug.contains("5.0"));
    assert!(!debug.contains("7.0"));
}
