use calc_flow::{
    CalcFlowError,
    array::{
        ArrayDType, ArrayExecutionOptions, FloatValues, FloatView, NativeArray, ReductionOptions,
        max, mean, min, sum,
    },
};
use datafusion::arrow::buffer::NullBuffer;

#[test]
fn test_sum_and_mean_round_exactly_without_intermediate_overflow() {
    let execution = ArrayExecutionOptions::default();
    let sum = sum(
        FloatView::from_f64(&[1e308, 1.0, -1e308]),
        ReductionOptions::default(),
        &execution,
    )
    .unwrap();
    assert_eq!(sum.dtype(), ArrayDType::Float64);
    assert!(sum.shape().is_empty());
    let FloatValues::Float64(values) = sum.float_view().unwrap().values() else {
        panic!("expected float64 scalar");
    };
    assert_eq!(values[0].to_bits(), 1.0_f64.to_bits());

    let average = mean(
        FloatView::from_f64(&[1e308, 1e308]),
        ReductionOptions::default(),
        &execution,
    )
    .unwrap();
    let FloatValues::Float64(values) = average.float_view().unwrap().values() else {
        panic!("expected float64 scalar");
    };
    assert_eq!(values[0].to_bits(), 1e308_f64.to_bits());
}

#[test]
fn test_reductions_preserve_nulls_or_explicitly_skip_them() {
    let execution = ArrayExecutionOptions::default();
    let values = [1.0_f64, f64::from_bits(0x7ff0_0000_0000_0042), 3.0];
    let validity = NullBuffer::from(vec![
        false, false, false, false, false, false, false, true, false, true,
    ])
    .slice(7, 3);
    let input = FloatView::try_new(FloatValues::Float64(&values), Some(&validity)).unwrap();
    let preserved = sum(input, ReductionOptions::default(), &execution).unwrap();
    assert_null_scalar(&preserved, ArrayDType::Float64);
    assert_eq!(preserved.estimated_bytes(), 9);
    assert_eq!(preserved.retained_buffer_bytes(), 9);

    let options = ReductionOptions::default().with_skip_nulls(true);
    assert!(options.skip_nulls());
    assert_eq!(
        scalar_f64(&sum(input, options, &execution).unwrap()).to_bits(),
        4.0_f64.to_bits()
    );
    assert_eq!(
        scalar_f64(&mean(input, options, &execution).unwrap()).to_bits(),
        2.0_f64.to_bits()
    );
    assert_eq!(values[1].to_bits(), 0x7ff0_0000_0000_0042);

    let empty = sum(FloatView::from_f32(&[]), options, &execution).unwrap();
    assert_null_scalar(&empty, ArrayDType::Float64);
    let validity = NullBuffer::new_null(1);
    let null = FloatView::try_new(FloatValues::Float32(&[f32::NAN]), Some(&validity)).unwrap();
    assert_null_scalar(
        &mean(null, options, &execution).unwrap(),
        ArrayDType::Float64,
    );
}

#[test]
fn test_scalar_limits_charge_output_and_kernel_scratch_without_limiting_input_length() {
    let input = FloatView::from_f64(&[1.0, 2.0, 3.0]);
    let options = ReductionOptions::default();
    let execution = ArrayExecutionOptions::default();
    assert_eq!(execution.max_elements(), 10_000_000);
    assert_eq!(execution.max_allocation_bytes(), 256 * 1024 * 1024);
    assert!(!options.skip_nulls());
    let one_result = execution.with_max_elements(1);
    assert_eq!(
        scalar_f64(&sum(input, options, &one_result).unwrap()).to_bits(),
        6.0_f64.to_bits()
    );

    assert_error_field(
        sum(input, options, &execution.with_max_elements(0)),
        "outputs.output.elements",
    );
    assert_error_field(
        mean(input, options, &execution.with_max_allocation_bytes(0)),
        "outputs.output.bytes",
    );
    assert_error_field(
        sum(input, options, &execution.with_max_allocation_bytes(8)),
        "scratch.bytes",
    );
    assert_eq!(execution.max_elements(), 10_000_000);
    assert_eq!(execution.max_allocation_bytes(), 256 * 1024 * 1024);
}

#[test]
fn test_float32_sum_and_mean_promote_without_losing_their_exact_input_values() {
    let options = ReductionOptions::default();
    let execution = ArrayExecutionOptions::default();
    let tiny = f32::from_bits(1);
    let values = [f32::MAX, tiny, -f32::MAX];
    let result = sum(FloatView::from_f32(&values), options, &execution).unwrap();
    assert_eq!(scalar_f64(&result).to_bits(), f64::from(tiny).to_bits());

    let result = mean(
        FloatView::from_f32(&[f32::MAX, f32::MAX]),
        options,
        &execution,
    )
    .unwrap();
    assert_eq!(scalar_f64(&result).to_bits(), f64::from(f32::MAX).to_bits());
}

#[test]
fn test_extrema_preserve_dtype_and_signed_zero_rules() {
    let options = ReductionOptions::default();
    let execution = ArrayExecutionOptions::default();
    for (values, minimum, maximum) in [
        (vec![-0.0_f32, 0.0], (-0.0_f32).to_bits(), 0),
        (vec![-0.0, -0.0], (-0.0_f32).to_bits(), (-0.0_f32).to_bits()),
        (vec![0.0, 0.0], 0, 0),
        (
            vec![3.0, -2.0, 7.0],
            (-2.0_f32).to_bits(),
            7.0_f32.to_bits(),
        ),
    ] {
        let input = FloatView::from_f32(&values);
        assert_eq!(
            scalar_f32(&min(input, options, &execution).unwrap()).to_bits(),
            minimum
        );
        assert_eq!(
            scalar_f32(&max(input, options, &execution).unwrap()).to_bits(),
            maximum
        );
    }
    let input = FloatView::from_f64(&[f64::NEG_INFINITY, 0.0, f64::INFINITY]);
    assert_eq!(
        scalar_f64(&min(input, options, &execution).unwrap()).to_bits(),
        f64::NEG_INFINITY.to_bits()
    );
    assert_eq!(
        scalar_f64(&max(input, options, &execution).unwrap()).to_bits(),
        f64::INFINITY.to_bits()
    );
}

#[test]
fn test_extrema_keep_nan_valid_and_charge_nullable_output_without_accumulator_scratch() {
    let options = ReductionOptions::default();
    let execution = ArrayExecutionOptions::default();
    let values = [1.0, f32::from_bits(0xff80_0042)];
    let input = FloatView::from_f32(&values);
    for result in [
        min(input, options, &execution),
        max(input, options, &execution),
    ] {
        assert_eq!(scalar_f32(&result.unwrap()).to_bits(), 0x7fc0_0000);
    }
    let values = [f64::from_bits(0xfff0_0000_0000_0042), 1.0];
    let input = FloatView::from_f64(&values);
    for result in [
        min(input, options, &execution),
        max(input, options, &execution),
    ] {
        assert_eq!(
            scalar_f64(&result.unwrap()).to_bits(),
            0x7ff8_0000_0000_0000
        );
    }

    let values = [-3.0_f32, f32::NAN, 7.0];
    let validity = NullBuffer::from(vec![true, false, true]);
    let input = FloatView::try_new(FloatValues::Float32(&values), Some(&validity)).unwrap();
    let nullable_budget = execution.with_max_allocation_bytes(5);
    assert_null_scalar(
        &min(input, options, &nullable_budget).unwrap(),
        ArrayDType::Float32,
    );
    assert_error_field(
        max(input, options, &execution.with_max_allocation_bytes(4)),
        "outputs.output.bytes",
    );
    let skip = options.with_skip_nulls(true);
    let scalar_budget = execution.with_max_elements(1).with_max_allocation_bytes(4);
    assert_eq!(
        scalar_f32(&min(input, skip, &scalar_budget).unwrap()).to_bits(),
        (-3.0_f32).to_bits()
    );
    assert_eq!(
        scalar_f32(&max(input, skip, &scalar_budget).unwrap()).to_bits(),
        7.0_f32.to_bits()
    );
    assert_null_scalar(
        &max(FloatView::from_f64(&[]), skip, &execution).unwrap(),
        ArrayDType::Float64,
    );
}

fn scalar_f64(array: &NativeArray) -> f64 {
    assert!(array.shape().is_empty());
    assert_eq!(array.as_arrow().null_count(), 0);
    let FloatValues::Float64(values) = array.float_view().unwrap().values() else {
        panic!("expected float64 scalar");
    };
    values[0]
}

fn scalar_f32(array: &NativeArray) -> f32 {
    assert!(array.shape().is_empty());
    assert_eq!(array.as_arrow().null_count(), 0);
    let FloatValues::Float32(values) = array.float_view().unwrap().values() else {
        panic!("expected float32 scalar");
    };
    values[0]
}

fn assert_null_scalar(array: &NativeArray, dtype: ArrayDType) {
    assert!(array.shape().is_empty());
    assert_eq!(array.dtype(), dtype);
    assert_eq!(array.as_arrow().null_count(), 1);
    match array.float_view().unwrap().values() {
        FloatValues::Float32(values) => assert_eq!(values[0].to_bits(), 0),
        FloatValues::Float64(values) => assert_eq!(values[0].to_bits(), 0),
        _ => panic!("expected float storage"),
    }
}

fn assert_error_field(result: calc_flow::Result<NativeArray>, expected: &str) {
    let error = result.unwrap_err();
    let CalcFlowError::InvalidArgument { field, .. } = error else {
        panic!("expected invalid argument, received {error}");
    };
    assert_eq!(field, expected);
}
