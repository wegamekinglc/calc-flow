use std::sync::Arc;

use calc_flow::{
    Batch, BatchMetadata,
    array::{self, ArrayExecutionOptions, NativeArray},
};
use datafusion::arrow::{
    array::{Array, BooleanArray, Float64Array},
    buffer::{BooleanBuffer, NullBuffer},
};

fn values(array: &NativeArray) -> Vec<Option<f64>> {
    array
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect()
}

#[test]
fn matrix_storage_counts_rows_separately_and_preserves_flat_arrow() {
    let array =
        NativeArray::from_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None).unwrap();
    assert_eq!(array.len(), 2);
    assert_eq!(array.element_count(), 6);
    assert_eq!(array.view().len(), 6);
    assert_eq!(array.view().shape(), &[2, 3]);
    assert!(
        array
            .float_view()
            .unwrap_err()
            .to_string()
            .contains("input.shape")
    );
    let batch = Batch::external(Arc::new(array), BatchMetadata::default()).unwrap();
    assert_eq!(batch.num_rows(), 2);
    let no_columns = NativeArray::from_f64(vec![], vec![3, 0], None).unwrap();
    assert_eq!(no_columns.len(), 3);
    assert!(!no_columns.is_empty());
    assert!(no_columns.view().is_empty());
    let no_rows = NativeArray::from_f64(vec![], vec![0, 3], None).unwrap();
    assert!(no_rows.is_empty());
    assert_eq!(no_rows.element_count(), 0);
}

#[test]
fn matrix_view_reshape_validates_descriptor_without_flattening_implicitly() {
    let array = NativeArray::from_f64(vec![1.0, 2.0, 3.0, 4.0], vec![4], None).unwrap();
    let view = array.view().reshape(&[2, 2]).unwrap();
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.len(), 4);
    assert_eq!(view.reshape(&[4]).unwrap().shape(), &[4]);
    for invalid in [&[3, 2][..], &[1, 2, 2], &[usize::MAX, usize::MAX], &[]] {
        assert!(
            view.reshape(invalid)
                .unwrap_err()
                .to_string()
                .contains("input.shape")
        );
    }
}

#[test]
fn matrix_elementwise_requires_identical_shapes_or_a_scalar() {
    let matrix = NativeArray::from_f64(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        Some(NullBuffer::from(vec![true, false, true, true])),
    )
    .unwrap();
    let scalar = NativeArray::from_f64(vec![10.0], vec![], None).unwrap();
    let output = array::add(
        matrix.view(),
        scalar.view(),
        &ArrayExecutionOptions::default(),
    )
    .unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(
        values(&output),
        vec![Some(11.0), None, Some(13.0), Some(14.0)]
    );
    let vector = NativeArray::from_f64(vec![1.0; 4], vec![4], None).unwrap();
    assert!(
        array::add(
            matrix.view(),
            vector.view(),
            &ArrayExecutionOptions::default()
        )
        .unwrap_err()
        .to_string()
        .contains("inputs.right.shape")
    );
    assert!(
        array::add(
            matrix.view(),
            scalar.view(),
            &ArrayExecutionOptions::default().with_max_elements(3)
        )
        .unwrap_err()
        .to_string()
        .contains("outputs.output.elements")
    );
}

#[test]
fn boolean_matrix_storage_preserves_sliced_value_and_validity_bits() {
    let original = BooleanArray::from(vec![
        Some(false),
        Some(true),
        None,
        Some(false),
        Some(true),
        Some(false),
        None,
        Some(true),
    ]);
    let sliced = original.slice(1, 6);
    let array = NativeArray::from_arrow(Arc::new(sliced), vec![2, 3]).unwrap();
    let output = array::logical_not(array.view(), &ArrayExecutionOptions::default()).unwrap();
    assert_eq!(output.shape(), &[2, 3]);
    let output = output
        .as_arrow()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert_eq!(
        output.iter().collect::<Vec<_>>(),
        vec![Some(false), None, Some(true), Some(false), Some(true), None]
    );
    assert_eq!(array.estimated_bytes(), 2 + 2 * size_of::<usize>());
}

#[test]
fn matmul_uses_explicit_inner_order_and_promoted_precision() {
    let limits = ArrayExecutionOptions::default();
    let left = NativeArray::from_f64(vec![1e16, 1.0, -1e16], vec![1, 3], None).unwrap();
    let right = NativeArray::from_f64(vec![1.0; 3], vec![3, 1], None).unwrap();
    let result = array::matmul(left.view(), right.view(), &limits).unwrap();
    assert_eq!(result.shape(), &[1, 1]);
    assert_eq!(values(&result), vec![Some(0.0)]);
    let left =
        NativeArray::from_f32(vec![16_777_216.0, 1.0, -16_777_216.0], vec![1, 3], None).unwrap();
    let right32 = NativeArray::from_f32(vec![1.0; 3], vec![3, 1], None).unwrap();
    let result32 = array::matmul(left.view(), right32.view(), &limits).unwrap();
    assert_eq!(
        result32
            .as_arrow()
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Float32Array>()
            .unwrap()
            .value(0)
            .to_bits(),
        0.0_f32.to_bits()
    );
    assert_eq!(
        values(&array::matmul(left.view(), right.view(), &limits).unwrap()),
        vec![Some(1.0)]
    );
    let delta = 2.0_f64.powi(-27);
    let left = NativeArray::from_f64(vec![1.0, 1.0 + delta], vec![1, 2], None).unwrap();
    let right = NativeArray::from_f64(vec![-1.0, 1.0 - delta], vec![2, 1], None).unwrap();
    assert_eq!(
        values(&array::matmul(left.view(), right.view(), &limits).unwrap()),
        vec![Some(0.0)]
    );
}

#[test]
fn matmul_zero_dimensions_still_enforce_operand_admission() {
    let limits = ArrayExecutionOptions::default();
    let left = NativeArray::from_f64(vec![], vec![2, 0], None).unwrap();
    let right = NativeArray::from_f64(vec![], vec![0, 3], None).unwrap();
    let result = array::matmul(left.view(), right.view(), &limits).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    assert!(
        values(&result)
            .iter()
            .all(|value| value.unwrap().to_bits() == 0)
    );
    let left = NativeArray::from_f64(vec![], vec![0, 2], None).unwrap();
    let right = NativeArray::from_f64(vec![1.0; 6], vec![2, 3], None).unwrap();
    assert_eq!(
        array::matmul(left.view(), right.view(), &limits)
            .unwrap()
            .shape(),
        &[0, 3]
    );
    let nullable = NativeArray::from_f64(
        vec![1.0; 6],
        vec![2, 3],
        Some(NullBuffer::from(vec![false, true, true, true, true, true])),
    )
    .unwrap();
    assert!(
        array::matmul(left.view(), nullable.view(), &limits)
            .unwrap_err()
            .to_string()
            .contains("inputs.right.validity")
    );
    let zero_width = NativeArray::from_f64(vec![], vec![2, 0], None).unwrap();
    assert!(
        array::matmul(left.view(), zero_width.view(), &limits)
            .unwrap_err()
            .to_string()
            .contains("inputs.right.shape")
    );
    assert!(
        array::matmul(right.view(), right.view(), &limits)
            .unwrap_err()
            .to_string()
            .contains("inputs.right.shape")
    );
    assert!(
        array::matmul(
            left.view(),
            array::ArrayView::scalar_f64(Some(1.0)),
            &limits
        )
        .is_err()
    );
}

#[test]
fn reshape_and_degenerate_transpose_alias_buffers_with_descriptor_budgets() {
    let input = NativeArray::from_f64(vec![1.0, 2.0, 3.0, 4.0], vec![4], None).unwrap();
    let limits = ArrayExecutionOptions::default().with_max_allocation_bytes(2 * size_of::<usize>());
    let result = array::reshape(&input, &[2, 2], &limits).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert!(Arc::ptr_eq(input.as_arrow(), result.as_arrow()));
    assert!(
        array::reshape(
            &input,
            &[2, 2],
            &limits.with_max_allocation_bytes(2 * size_of::<usize>() - 1)
        )
        .is_err()
    );
    assert!(array::reshape(&input, &[2, 2], &limits.with_max_elements(3)).is_err());
    let row = array::reshape(&input, &[1, 4], &limits).unwrap();
    let column = array::transpose(&row, &limits).unwrap();
    assert_eq!(column.shape(), &[4, 1]);
    assert!(Arc::ptr_eq(input.as_arrow(), column.as_arrow()));
}

#[test]
fn transpose_reorders_raw_float_bits_and_sliced_boolean_masks() {
    let input = NativeArray::from_f64(
        vec![
            1.0,
            -0.0,
            f64::from_bits(0x7ff8_0000_0000_0123),
            4.0,
            5.0,
            6.0,
        ],
        vec![2, 3],
        Some(NullBuffer::from(vec![true, false, true, true, true, false])),
    )
    .unwrap();
    let result = array::transpose(&input, &ArrayExecutionOptions::default()).unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    let result = result
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let original = input
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for (index, source) in [0, 3, 1, 4, 2, 5].into_iter().enumerate() {
        assert_eq!(
            result.value(index).to_bits(),
            original.value(source).to_bits()
        );
        assert_eq!(result.is_valid(index), original.is_valid(source));
    }
    let required = 6 * size_of::<f64>() + 1 + 2 * size_of::<usize>();
    assert!(
        array::transpose(
            &input,
            &ArrayExecutionOptions::default().with_max_allocation_bytes(required - 1)
        )
        .is_err()
    );
    let bits =
        BooleanBuffer::from(vec![false, false, true, true, false, true, false, true]).slice(2, 6);
    let validity = NullBuffer::from(vec![true, true, false, true, true, true, false]).slice(1, 6);
    let original = BooleanArray::new(bits, Some(validity));
    let input = NativeArray::from_arrow(Arc::new(original.clone()), vec![2, 3]).unwrap();
    let result = array::transpose(&input, &ArrayExecutionOptions::default()).unwrap();
    let result = result
        .as_arrow()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    for (index, source) in [0, 3, 1, 4, 2, 5].into_iter().enumerate() {
        assert_eq!(result.value(index), original.value(source));
        assert_eq!(result.is_valid(index), original.is_valid(source));
    }
}

#[test]
fn axis_reduction_preserves_exact_lanes_and_keepdims_shapes() {
    let input = NativeArray::from_f64(
        vec![1e308, 1.0, -1e308, 1e308, 3.0, -1e308],
        vec![2, 3],
        None,
    )
    .unwrap();
    let limits = ArrayExecutionOptions::default();
    let options = array::ReductionOptions::default().with_axis(Some(1));
    let sum = array::reduce(input.view(), array::Reduction::Sum, options, &limits).unwrap();
    assert_eq!(sum.shape(), &[2]);
    assert_eq!(values(&sum), vec![Some(1.0), Some(3.0)]);
    let mean = array::reduce(
        input.view(),
        array::Reduction::Mean,
        options.with_keepdims(true),
        &limits,
    )
    .unwrap();
    assert_eq!(mean.shape(), &[2, 1]);
    assert_eq!(values(&mean), vec![Some(1.0 / 3.0), Some(1.0)]);
    for (axis, keep, expected) in [
        (None, false, vec![]),
        (None, true, vec![1, 1]),
        (Some(0), false, vec![3]),
        (Some(0), true, vec![1, 3]),
        (Some(1), false, vec![2]),
    ] {
        let result = array::reduce(
            input.view(),
            array::Reduction::Sum,
            array::ReductionOptions::default()
                .with_axis(axis)
                .with_keepdims(keep),
            &limits,
        )
        .unwrap();
        assert_eq!(result.shape(), expected);
    }
    let large = NativeArray::from_f64(vec![1e308, 1e308, 3.0, 5.0], vec![2, 2], None).unwrap();
    assert_eq!(
        values(&array::reduce(large.view(), array::Reduction::Mean, options, &limits).unwrap()),
        vec![Some(1e308), Some(4.0)]
    );
}

#[test]
fn axis_reduction_empty_lanes_are_null_and_empty_outputs_are_bounded() {
    let limits = ArrayExecutionOptions::default();
    for (shape, axis, expected_len) in [(vec![0, 3], 0, 3), (vec![2, 0], 1, 2)] {
        let input = NativeArray::from_f64(vec![], shape, None).unwrap();
        let result = array::reduce(
            input.view(),
            array::Reduction::Sum,
            array::ReductionOptions::default().with_axis(Some(axis)),
            &limits,
        )
        .unwrap();
        assert_eq!(values(&result), vec![None; expected_len]);
    }
    let input = NativeArray::from_f64(vec![], vec![0, 3], None).unwrap();
    let result = array::reduce(
        input.view(),
        array::Reduction::Sum,
        array::ReductionOptions::default().with_axis(Some(1)),
        &limits
            .with_max_elements(0)
            .with_max_allocation_bytes(size_of::<usize>()),
    )
    .unwrap();
    assert_eq!(result.shape(), &[0]);
    assert_eq!(result.element_count(), 0);
}

#[test]
fn axis_reduction_reuses_null_nan_and_signed_zero_contracts() {
    let limits = ArrayExecutionOptions::default();
    let input = NativeArray::from_f32(vec![-0.0, 0.0, 0.0, -0.0], vec![2, 2], None).unwrap();
    for (operation, expected) in [
        (array::Reduction::Min, (-0.0_f32).to_bits()),
        (array::Reduction::Max, 0.0_f32.to_bits()),
    ] {
        let result = array::reduce(
            input.view(),
            operation,
            array::ReductionOptions::default().with_axis(Some(0)),
            &limits,
        )
        .unwrap();
        let result = result
            .as_arrow()
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Float32Array>()
            .unwrap();
        assert!(
            result
                .values()
                .iter()
                .all(|value| value.to_bits() == expected)
        );
    }
    let input = NativeArray::from_f64(
        vec![2.0, 99.0, f64::from_bits(0x7ff8_0000_0000_0123), 4.0],
        vec![2, 2],
        Some(NullBuffer::from(vec![true, false, true, true])),
    )
    .unwrap();
    let options = array::ReductionOptions::default().with_axis(Some(1));
    let result = array::reduce(input.view(), array::Reduction::Mean, options, &limits).unwrap();
    assert_eq!(values(&result)[0], None);
    assert_eq!(values(&result)[1].unwrap().to_bits(), 0x7ff8_0000_0000_0000);
    let result = array::reduce(
        input.view(),
        array::Reduction::Mean,
        options.with_skip_nulls(true),
        &limits,
    )
    .unwrap();
    assert_eq!(values(&result)[0], Some(2.0));
    assert_eq!(values(&result)[1].unwrap().to_bits(), 0x7ff8_0000_0000_0000);
}

#[test]
fn reduction_axes_and_output_budgets_are_validated_without_changing_m1_defaults() {
    let limits = ArrayExecutionOptions::default();
    let input = NativeArray::from_f64(vec![1.0; 6], vec![2, 3], None).unwrap();
    let options = array::ReductionOptions::default()
        .with_axis(Some(0))
        .with_keepdims(true);
    assert!(
        array::reduce(
            input.view(),
            array::Reduction::Sum,
            options,
            &limits.with_max_elements(2)
        )
        .unwrap_err()
        .to_string()
        .contains("outputs.output.elements")
    );
    let bytes = 3 * size_of::<f64>() + 2 * size_of::<usize>();
    assert!(
        array::reduce(
            input.view(),
            array::Reduction::Sum,
            options,
            &limits.with_max_allocation_bytes(bytes)
        )
        .unwrap_err()
        .to_string()
        .contains("scratch.bytes")
    );
    assert!(
        array::reduce(
            input.view(),
            array::Reduction::Min,
            options,
            &limits.with_max_allocation_bytes(bytes)
        )
        .is_ok()
    );
    assert!(
        array::reduce(
            input.view(),
            array::Reduction::Sum,
            options.with_axis(Some(2)),
            &limits
        )
        .unwrap_err()
        .to_string()
        .contains("options.axis")
    );
    let scalar = NativeArray::from_f64(vec![1.0], vec![], None).unwrap();
    assert!(
        array::reduce(
            scalar.view(),
            array::Reduction::Sum,
            options.with_axis(Some(0)),
            &limits
        )
        .is_err()
    );
    let old = array::sum(
        scalar.float_view().unwrap(),
        array::ReductionOptions::default(),
        &limits,
    )
    .unwrap();
    assert!(old.shape().is_empty());
    let vector = array::sum(
        scalar.float_view().unwrap(),
        array::ReductionOptions::default()
            .with_axis(Some(0))
            .with_keepdims(true),
        &limits,
    )
    .unwrap();
    assert_eq!(vector.shape(), &[1]);
    let boolean =
        NativeArray::from_arrow(Arc::new(BooleanArray::from(vec![true])), vec![1, 1]).unwrap();
    assert!(
        array::reduce(
            boolean.view(),
            array::Reduction::Sum,
            array::ReductionOptions::default(),
            &limits
        )
        .unwrap_err()
        .to_string()
        .contains("input.dtype")
    );
}

#[tokio::test]
async fn version_one_provider_rejects_a_real_matrix_payload_before_execution() {
    use calc_flow::{
        BatchKind, BatchOperatorContext, BatchOperatorFactory, CancellationToken,
        ExternalOperatorSpec, Port, RunContext,
    };
    use std::collections::BTreeMap;
    let options = serde_json::from_value(serde_json::json!({
        "ir_version": 1,
        "inputs": [{"name":"x","dtype":"float64","shape":[null]}],
        "nodes": [{"id":"x","op":"input","name":"x"}],
        "outputs": [{"name":"output","node":"x","dtype":"float64","shape":[null]}]
    }))
    .unwrap();
    let spec = ExternalOperatorSpec::new("native", "array_expr", "1", options).unwrap();
    let mut operator = array::NativeArrayFactory::default()
        .create(
            &spec,
            vec![Port::new("x", BatchKind::Array, true, None).unwrap()],
            vec![Port::new("output", BatchKind::Array, true, None).unwrap()],
        )
        .unwrap();
    let matrix = NativeArray::from_f64(vec![1.0; 4], vec![2, 2], None).unwrap();
    let inputs = BTreeMap::from([(
        "x".into(),
        Batch::external(Arc::new(matrix), BatchMetadata::default()).unwrap(),
    )]);
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let error = operator
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap_err();
    assert!(error.to_string().contains("inputs.x.shape"));
}

#[test]
fn matmul_nonfinite_results_remain_valid_and_preserve_promoted_dtype() {
    use datafusion::arrow::array::Float32Array;
    let limits = ArrayExecutionOptions::default();
    let left = NativeArray::from_f32(
        vec![f32::MAX, 0.0, f32::MAX, f32::MAX, 0.0, 0.0, f32::NAN, 1.0],
        vec![4, 2],
        None,
    )
    .unwrap();
    let right = NativeArray::from_f32(vec![2.0, 1.0, 0.0, 1.0], vec![2, 2], None).unwrap();
    let result = array::matmul(left.view(), right.view(), &limits).unwrap();
    let result = result
        .as_arrow()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(result.null_count(), 0);
    assert!(result.value(0).is_infinite() && result.value(0).is_sign_positive());
    assert!(result.value(3).is_infinite() && result.value(3).is_sign_positive());
    assert_eq!(result.value(1).to_bits(), f32::MAX.to_bits());
    assert!(result.value(6).is_nan());
    assert!(result.value(7).is_nan());
    let zero = NativeArray::from_f64(vec![0.0], vec![1, 1], None).unwrap();
    let infinite = NativeArray::from_f32(vec![f32::INFINITY], vec![1, 1], None).unwrap();
    let mixed = array::matmul(zero.view(), infinite.view(), &limits).unwrap();
    let mixed = mixed
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(mixed.null_count(), 0);
    assert!(mixed.value(0).is_nan());
}
