use std::{mem::size_of, sync::Arc};

use calc_flow::{
    CalcFlowError,
    array::{
        ArrayDType, ArrayExecutionOptions, ArrayView, FloatView, NativeArray, abs, add, divide,
        equal, exp, greater, greater_equal, less, less_equal, log, logical_and, logical_not,
        logical_or, logical_xor, multiply, neg, not_equal, pow, sqrt, subtract, where_,
    },
};
use datafusion::arrow::{
    array::{Array, ArrayRef, BooleanArray, Float32Array, Float64Array},
    buffer::{BooleanBuffer, Buffer, NullBuffer},
};

#[test]
fn test_boolean_storage_and_views_preserve_independent_bit_offsets() {
    let values = BooleanBuffer::new(Buffer::from_vec(vec![0b1000_0000_u8, 0b0000_0010]), 7, 3);
    let validity = NullBuffer::new(BooleanBuffer::new(
        Buffer::from_vec(vec![0b0001_0100_u8]),
        2,
        3,
    ));
    let retained = values.inner().capacity() + validity.buffer().capacity() + size_of::<usize>();
    let pointer = values.inner().as_ptr();
    let array = NativeArray::from_boolean(values.clone(), vec![3], Some(validity.clone())).unwrap();
    assert_eq!(array.dtype(), ArrayDType::Boolean);
    assert_eq!(array.shape(), &[3]);
    assert_eq!(array.estimated_bytes(), 3 + size_of::<usize>());
    assert_eq!(array.retained_buffer_bytes(), retained);
    let arrow = array
        .as_arrow()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert_eq!(arrow.values().offset(), 7);
    assert_eq!(arrow.nulls().unwrap().offset(), 2);
    assert!(std::ptr::eq(arrow.values().inner().as_ptr(), pointer));
    assert!(arrow.value(0));
    assert!(arrow.is_null(1));
    assert!(arrow.value(2));
    let view = array.view();
    assert_eq!(view.dtype(), ArrayDType::Boolean);
    assert_eq!(view.shape(), &[3]);
    assert_eq!(view.len(), 3);
    assert!(!view.is_empty());
    assert!(!format!("{view:?}").contains("BooleanBuffer"));

    let source: ArrayRef = Arc::new(BooleanArray::new(values, Some(validity)));
    let adopted = NativeArray::from_arrow(Arc::clone(&source), vec![3]).unwrap();
    assert!(Arc::ptr_eq(adopted.as_arrow(), &source));
    let CalcFlowError::InvalidArgument { field, .. } = adopted.float_view().unwrap_err() else {
        panic!("expected non-floating dtype error");
    };
    assert_eq!(field, "input.dtype");
}

#[test]
fn test_scalar_and_singleton_vector_identity_are_explicit() {
    const FLOAT: ArrayView<'static> = ArrayView::scalar_f32(Some(2.0));
    const NULL: ArrayView<'static> = ArrayView::scalar_f64(None);
    const BOOLEAN: ArrayView<'static> = ArrayView::scalar_boolean(Some(false));
    for view in [FLOAT, NULL, BOOLEAN] {
        assert!(view.shape().is_empty());
        assert_eq!(view.len(), 1);
        assert!(!view.is_empty());
    }
    assert_eq!(FLOAT.dtype(), ArrayDType::Float32);
    assert_eq!(NULL.dtype(), ArrayDType::Float64);
    assert_eq!(BOOLEAN.dtype(), ArrayDType::Boolean);
    let values = [1.0];
    assert_eq!(
        ArrayView::from_float(FloatView::from_f64(&values)).shape(),
        &[1]
    );
    let scalar = NativeArray::from_f64(vec![1.0], vec![], None).unwrap();
    assert!(scalar.view().shape().is_empty());
    let vector = NativeArray::from_f64(vec![1.0], vec![1], None).unwrap();
    assert_eq!(vector.view().shape(), &[1]);

    let values = BooleanBuffer::from(vec![true]);
    assert_eq!(
        ArrayView::from_boolean(&values, None).unwrap().shape(),
        &[1]
    );
    let validity = NullBuffer::new_valid(2);
    assert!(ArrayView::from_boolean(&values, Some(&validity)).is_err());
}

#[test]
fn test_numeric_scalar_broadcast_and_promotion_preserve_shape_and_precision() {
    let execution = ArrayExecutionOptions::default();
    let source = NativeArray::from_f32(vec![-0.0, 1.0, 4.0], vec![3], None).unwrap();
    let result = add(source.view(), ArrayView::scalar_f32(Some(2.0)), &execution).unwrap();
    assert_eq!(result.shape(), &[3]);
    assert_f32(&result, &[Some(2.0), Some(3.0), Some(6.0)]);
    let result = multiply(source.view(), ArrayView::scalar_f64(Some(0.5)), &execution).unwrap();
    assert_eq!(result.dtype(), ArrayDType::Float64);
    assert_f64(&result, &[Some(-0.0), Some(0.5), Some(2.0)]);
    let result = add(
        ArrayView::scalar_f32(Some(2.0)),
        ArrayView::scalar_f64(Some(3.0)),
        &execution,
    )
    .unwrap();
    assert!(result.shape().is_empty());
    assert_f64(&result, &[Some(5.0)]);

    let source = NativeArray::from_f32(vec![8.0, 4.0], vec![2], None).unwrap();
    let two = ArrayView::scalar_f32(Some(2.0));
    assert_f32(
        &subtract(source.view(), two, &execution).unwrap(),
        &[Some(6.0), Some(2.0)],
    );
    assert_f32(
        &multiply(source.view(), two, &execution).unwrap(),
        &[Some(16.0), Some(8.0)],
    );
    assert_f32(
        &divide(source.view(), two, &execution).unwrap(),
        &[Some(4.0), Some(2.0)],
    );
    assert_f32(
        &pow(source.view(), two, &execution).unwrap(),
        &[Some(64.0), Some(16.0)],
    );
}

#[test]
fn test_unary_math_and_nonfinite_arithmetic_keep_ieee_results_valid() {
    let execution = ArrayExecutionOptions::default();
    let source = NativeArray::from_f32(vec![-0.0, -4.0, 4.0], vec![3], None).unwrap();
    assert_f32(
        &neg(source.view(), &execution).unwrap(),
        &[Some(0.0), Some(4.0), Some(-4.0)],
    );
    assert_f32(
        &abs(source.view(), &execution).unwrap(),
        &[Some(0.0), Some(4.0), Some(4.0)],
    );
    let result = sqrt(source.view(), &execution).unwrap();
    let values = result
        .as_arrow()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(values.value(0).to_bits(), (-0.0_f32).to_bits());
    assert!(values.value(1).is_nan());
    assert!(values.is_valid(1));
    assert_eq!(values.value(2).to_bits(), 2.0_f32.to_bits());

    let zero = ArrayView::scalar_f64(Some(0.0));
    let one = ArrayView::scalar_f64(Some(1.0));
    assert_eq!(scalar_f64(&log(one, &execution).unwrap()).to_bits(), 0);
    assert_eq!(
        scalar_f64(&exp(zero, &execution).unwrap()).to_bits(),
        1.0_f64.to_bits()
    );
    assert!(scalar_f64(&divide(one, zero, &execution).unwrap()).is_infinite());
    assert!(scalar_f64(&divide(zero, zero, &execution).unwrap()).is_nan());
    let result = scalar_f64(&log(zero, &execution).unwrap());
    assert!(result.is_infinite() && result.is_sign_negative());
    assert!(
        scalar_f64(&exp(ArrayView::scalar_f64(Some(1e308)), &execution).unwrap()).is_infinite()
    );
}

#[test]
fn test_numeric_nulls_empty_vectors_and_limits_are_explicit() {
    let execution = ArrayExecutionOptions::default();
    let source = NativeArray::from_f32(
        vec![f32::from_bits(0x7f80_0042), 4.0],
        vec![2],
        Some(NullBuffer::from(vec![false, true])),
    )
    .unwrap();
    assert_f32(
        &add(source.view(), ArrayView::scalar_f32(Some(2.0)), &execution).unwrap(),
        &[None, Some(6.0)],
    );
    let empty = NativeArray::from_f32(vec![], vec![0], None).unwrap();
    let result = add(
        ArrayView::scalar_f32(None),
        empty.view(),
        &execution
            .with_max_elements(0)
            .with_max_allocation_bytes(size_of::<usize>()),
    )
    .unwrap();
    assert_eq!(result.shape(), &[0]);
    assert!(result.is_empty());
    let one = ArrayView::scalar_f32(Some(1.0));
    assert_error_field(
        add(one, one, &execution.with_max_elements(0)),
        "outputs.output.elements",
    );
    assert_error_field(
        add(one, one, &execution.with_max_allocation_bytes(3)),
        "outputs.output.bytes",
    );
    let result = add(one, one, &execution.with_max_allocation_bytes(4)).unwrap();
    assert_f32(&result, &[Some(2.0)]);
}

#[test]
fn test_numeric_shape_dtype_and_exponent_rejections_precede_execution() {
    let execution = ArrayExecutionOptions::default();
    let values = [1.0, 2.0];
    let singleton = [3.0];
    let left = ArrayView::from_float(FloatView::from_f64(&values));
    let right = ArrayView::from_float(FloatView::from_f64(&singleton));
    assert_error_field(add(left, right, &execution), "inputs.right.shape");
    assert_error_field(
        add(ArrayView::scalar_boolean(Some(true)), left, &execution),
        "inputs.left.dtype",
    );
    assert_error_field(
        neg(ArrayView::scalar_boolean(None), &execution),
        "inputs.input.dtype",
    );
    assert_error_field(pow(left, right, &execution), "inputs.right.shape");
    for value in [None, Some(f64::NAN), Some(f64::INFINITY), Some(65.0)] {
        assert_error_field(
            pow(left, ArrayView::scalar_f64(value), &execution),
            "right.value",
        );
    }
}

#[test]
fn test_comparisons_keep_nan_values_valid_and_reject_boolean_numeric_coercion() {
    let execution = ArrayExecutionOptions::default();
    let source =
        NativeArray::from_f64(vec![f64::NAN, 1.0, -0.0, f64::INFINITY], vec![4], None).unwrap();
    let one = ArrayView::scalar_f32(Some(1.0));
    assert_boolean(
        &equal(source.view(), one, &execution).unwrap(),
        &[Some(false), Some(true), Some(false), Some(false)],
    );
    assert_boolean(
        &not_equal(source.view(), one, &execution).unwrap(),
        &[Some(true), Some(false), Some(true), Some(true)],
    );
    assert_boolean(
        &less(source.view(), one, &execution).unwrap(),
        &[Some(false), Some(false), Some(true), Some(false)],
    );
    assert_boolean(
        &less_equal(source.view(), one, &execution).unwrap(),
        &[Some(false), Some(true), Some(true), Some(false)],
    );
    assert_boolean(
        &greater(source.view(), one, &execution).unwrap(),
        &[Some(false), Some(false), Some(false), Some(true)],
    );
    assert_boolean(
        &greater_equal(source.view(), one, &execution).unwrap(),
        &[Some(false), Some(true), Some(false), Some(true)],
    );
    let boolean = ArrayView::scalar_boolean(Some(true));
    assert_boolean(&equal(boolean, boolean, &execution).unwrap(), &[Some(true)]);
    let null = ArrayView::scalar_boolean(None);
    assert_boolean(&equal(null, null, &execution).unwrap(), &[None]);
    assert_error_field(equal(boolean, one, &execution), "inputs.right.dtype");
    assert_error_field(less(boolean, boolean, &execution), "inputs.left.dtype");
    assert_error_field(logical_and(one, boolean, &execution), "inputs.left.dtype");
}

#[test]
fn test_boolean_kleene_tables_and_not_cover_all_three_values() {
    let execution = ArrayExecutionOptions::default();
    let values = [None, Some(false), Some(true)];
    let and = [
        [None, Some(false), None],
        [Some(false); 3],
        [None, Some(false), Some(true)],
    ];
    let or = [
        [None, None, Some(true)],
        [None, Some(false), Some(true)],
        [Some(true); 3],
    ];
    let xor = [
        [None; 3],
        [None, Some(false), Some(true)],
        [None, Some(true), Some(false)],
    ];
    for (left_index, left) in values.iter().copied().enumerate() {
        let left = ArrayView::scalar_boolean(left);
        let expected = [None, Some(true), Some(false)][left_index];
        assert_boolean(&logical_not(left, &execution).unwrap(), &[expected]);
        for (right_index, right) in values.iter().copied().enumerate() {
            let right = ArrayView::scalar_boolean(right);
            assert_boolean(
                &logical_and(left, right, &execution).unwrap(),
                &[and[left_index][right_index]],
            );
            assert_boolean(
                &logical_or(left, right, &execution).unwrap(),
                &[or[left_index][right_index]],
            );
            assert_boolean(
                &logical_xor(left, right, &execution).unwrap(),
                &[xor[left_index][right_index]],
            );
        }
    }
}

#[test]
fn test_boolean_offsets_vector_logic_and_packed_output_limits() {
    let execution = ArrayExecutionOptions::default();
    let values = BooleanBuffer::new(Buffer::from_vec(vec![0b1000_0000_u8, 0b0000_0010]), 7, 3);
    let validity = NullBuffer::new(BooleanBuffer::new(
        Buffer::from_vec(vec![0b0001_0100_u8]),
        2,
        3,
    ));
    let source = NativeArray::from_boolean(values, vec![3], Some(validity)).unwrap();
    assert_boolean(
        &logical_not(source.view(), &execution).unwrap(),
        &[Some(false), None, Some(false)],
    );
    assert_boolean(
        &logical_and(
            source.view(),
            ArrayView::scalar_boolean(Some(false)),
            &execution,
        )
        .unwrap(),
        &[Some(false); 3],
    );
    assert_boolean(
        &logical_or(
            source.view(),
            ArrayView::scalar_boolean(Some(true)),
            &execution,
        )
        .unwrap(),
        &[Some(true); 3],
    );
    assert_boolean(
        &logical_xor(
            source.view(),
            ArrayView::scalar_boolean(Some(true)),
            &execution,
        )
        .unwrap(),
        &[Some(false), None, Some(false)],
    );
    let scalar = ArrayView::scalar_boolean(Some(false));
    let result = logical_not(scalar, &execution.with_max_allocation_bytes(1)).unwrap();
    assert!(result.shape().is_empty());
    assert_eq!(result.estimated_bytes(), 1);
    assert_eq!(result.retained_buffer_bytes(), 1);
    assert_error_field(
        logical_not(scalar, &execution.with_max_allocation_bytes(0)),
        "outputs.output.bytes",
    );
    let null = logical_not(
        ArrayView::scalar_boolean(None),
        &execution.with_max_allocation_bytes(2),
    )
    .unwrap();
    assert_boolean(&null, &[None]);
    assert_eq!(null.retained_buffer_bytes(), 2);
}

#[test]
fn test_where_selects_nullable_branches_with_numeric_promotion_and_scalar_broadcast() {
    let execution = ArrayExecutionOptions::default();
    let condition = NativeArray::from_boolean(
        BooleanBuffer::from(vec![true, false, true, false]),
        vec![4],
        Some(NullBuffer::from(vec![true, true, false, true])),
    )
    .unwrap();
    let values = NativeArray::from_f32(
        vec![2.0, f32::NAN, 7.0, f32::INFINITY],
        vec![4],
        Some(NullBuffer::from(vec![true, false, true, true])),
    )
    .unwrap();
    let result = where_(
        condition.view(),
        values.view(),
        ArrayView::scalar_f64(Some(-0.0)),
        &execution,
    )
    .unwrap();
    assert_eq!(result.shape(), &[4]);
    assert_f64(&result, &[Some(2.0), Some(-0.0), None, Some(-0.0)]);
    let scalar = where_(
        ArrayView::scalar_boolean(Some(false)),
        ArrayView::scalar_f32(None),
        ArrayView::scalar_f32(Some(f32::INFINITY)),
        &execution,
    )
    .unwrap();
    assert!(scalar.shape().is_empty());
    assert_f32(&scalar, &[Some(f32::INFINITY)]);
    let null = where_(
        ArrayView::scalar_boolean(None),
        ArrayView::scalar_f64(Some(f64::NAN)),
        ArrayView::scalar_f64(Some(3.0)),
        &execution,
    )
    .unwrap();
    assert_f64(&null, &[None]);
    assert_eq!(
        null.as_arrow()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0)
            .to_bits(),
        0
    );
    let boolean = where_(
        condition.view(),
        ArrayView::scalar_boolean(Some(true)),
        ArrayView::scalar_boolean(Some(false)),
        &execution,
    )
    .unwrap();
    assert_boolean(&boolean, &[Some(true), Some(false), None, Some(false)]);
}

#[test]
fn test_where_checks_unselected_branches_shape_dtype_and_complete_peak_budget() {
    let execution = ArrayExecutionOptions::default();
    let condition = ArrayView::scalar_boolean(Some(true));
    let vector = NativeArray::from_f64(vec![1.0, 2.0], vec![2], None).unwrap();
    let singleton = NativeArray::from_f64(vec![3.0], vec![1], None).unwrap();
    assert_error_field(
        where_(condition, vector.view(), singleton.view(), &execution),
        "if_false.shape",
    );
    assert_error_field(
        where_(
            condition,
            vector.view(),
            ArrayView::scalar_boolean(None),
            &execution,
        ),
        "if_false.dtype",
    );
    assert_error_field(
        where_(
            ArrayView::scalar_f32(Some(1.0)),
            vector.view(),
            vector.view(),
            &execution,
        ),
        "condition.dtype",
    );
    let condition_vector =
        NativeArray::from_boolean(BooleanBuffer::from(vec![true, false, true]), vec![3], None)
            .unwrap();
    assert_error_field(
        where_(
            condition_vector.view(),
            vector.view(),
            vector.view(),
            &execution,
        ),
        "if_true.shape",
    );
    let empty = NativeArray::from_f32(vec![], vec![0], None).unwrap();
    let result = where_(
        condition,
        empty.view(),
        ArrayView::scalar_f32(None),
        &execution
            .with_max_elements(0)
            .with_max_allocation_bytes(size_of::<usize>()),
    )
    .unwrap();
    assert_eq!(result.shape(), &[0]);
    assert!(result.is_empty());
    let peak = 2 * size_of::<f64>() + 1 + size_of::<usize>();
    let nullable_unselected = ArrayView::scalar_f64(None);
    let result = where_(
        condition,
        vector.view(),
        nullable_unselected,
        &execution.with_max_allocation_bytes(peak),
    )
    .unwrap();
    assert_f64(&result, &[Some(1.0), Some(2.0)]);
    assert!(result.as_arrow().nulls().is_none());
    assert_error_field(
        where_(
            condition,
            vector.view(),
            nullable_unselected,
            &execution.with_max_allocation_bytes(peak - 1),
        ),
        "outputs.output.bytes",
    );
    assert_error_field(
        where_(
            condition,
            vector.view(),
            vector.view(),
            &execution.with_max_elements(1),
        ),
        "outputs.output.elements",
    );
}

fn assert_f32(array: &NativeArray, expected: &[Option<f32>]) {
    assert_eq!(array.dtype(), ArrayDType::Float32);
    let values = array
        .as_arrow()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    let actual: Vec<_> = values.iter().map(|value| value.map(f32::to_bits)).collect();
    let expected: Vec<_> = expected
        .iter()
        .map(|value| value.map(f32::to_bits))
        .collect();
    assert_eq!(actual, expected);
}

fn assert_f64(array: &NativeArray, expected: &[Option<f64>]) {
    assert_eq!(array.dtype(), ArrayDType::Float64);
    let values = array
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let actual: Vec<_> = values.iter().map(|value| value.map(f64::to_bits)).collect();
    let expected: Vec<_> = expected
        .iter()
        .map(|value| value.map(f64::to_bits))
        .collect();
    assert_eq!(actual, expected);
}

fn scalar_f64(array: &NativeArray) -> f64 {
    assert!(array.shape().is_empty());
    let values = array
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(values.is_valid(0));
    values.value(0)
}

fn assert_boolean(array: &NativeArray, expected: &[Option<bool>]) {
    assert_eq!(array.dtype(), ArrayDType::Boolean);
    let values = array
        .as_arrow()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert_eq!(values.iter().collect::<Vec<_>>(), expected);
    for (index, value) in expected.iter().enumerate() {
        if value.is_none() {
            assert!(!values.value(index));
        }
    }
}

fn assert_error_field(result: calc_flow::Result<NativeArray>, expected: &str) {
    let CalcFlowError::InvalidArgument { field, .. } = result.unwrap_err() else {
        panic!("expected invalid argument");
    };
    assert_eq!(field, expected);
}
