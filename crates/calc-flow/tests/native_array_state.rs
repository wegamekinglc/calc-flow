use calc_flow::array::{
    ArrayExecutionOptions, ArrayStateOptions, ArrayView, EwmaOptions, EwmaState, FloatValues,
    FloatView, NativeArray, RollingMeanOptions, RollingMeanState,
};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}

fn bits(output: &NativeArray) -> Vec<Option<u64>> {
    let floats = output.float_view().unwrap();
    let FloatValues::Float64(values) = floats.values() else {
        panic!("expected f64");
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            floats
                .validity()
                .is_none_or(|mask| mask.is_valid(index))
                .then(|| value.to_bits())
        })
        .collect()
}

#[test]
fn test_owned_sequence_states_continue_across_chunks_and_reset() {
    let execution = ArrayExecutionOptions::default();
    let mut mean = RollingMeanState::new(
        RollingMeanOptions::new(2).unwrap(),
        ArrayStateOptions::default(),
    )
    .unwrap();
    let mut ewma =
        EwmaState::new(EwmaOptions::new(3).unwrap(), ArrayStateOptions::default()).unwrap();
    assert_eq!(mean.state_bytes(), 64);
    assert_eq!(ewma.state_bytes(), 24);
    assert_eq!(
        bits(&mean.process(view(&[1.0, 3.0]), &execution).unwrap()),
        [Some(1.0_f64.to_bits()), Some(2.0_f64.to_bits())]
    );
    assert_eq!(
        bits(&ewma.process(view(&[1.0, 3.0]), &execution).unwrap()),
        [Some(1.0_f64.to_bits()), Some(2.0_f64.to_bits())]
    );
    let retained = mean.process(view(&[5.0]), &execution).unwrap();
    assert_eq!(bits(&retained), [Some(4.0_f64.to_bits())]);
    assert_eq!(
        bits(&ewma.process(view(&[5.0]), &execution).unwrap()),
        [Some(3.5_f64.to_bits())]
    );
    mean.reset();
    ewma.reset();
    assert_eq!(mean.state_bytes(), 64);
    assert_eq!(ewma.state_bytes(), 24);
    assert_eq!(bits(&retained), [Some(4.0_f64.to_bits())]);
    assert_eq!(
        bits(&mean.process(view(&[7.0]), &execution).unwrap()),
        [Some(7.0_f64.to_bits())]
    );
    assert_eq!(
        bits(&ewma.process(view(&[7.0]), &execution).unwrap()),
        [Some(7.0_f64.to_bits())]
    );
}

fn assert_field(result: calc_flow::Result<NativeArray>, expected: &str) {
    assert!(
        matches!(result, Err(calc_flow::CalcFlowError::InvalidArgument { field, .. }) if field == expected)
    );
}

#[test]
fn test_state_constructors_admit_fixed_storage_without_window_reservation() {
    let limits = ArrayStateOptions::default();
    assert_eq!(limits.max_state_bytes(), 256 * 1024 * 1024);
    assert!(
        matches!(RollingMeanState::new(RollingMeanOptions::new(u64::MAX).unwrap(), limits.with_max_state_bytes(63)),
        Err(calc_flow::CalcFlowError::InvalidArgument { field, .. }) if field == "state.max_state_bytes")
    );
    assert!(
        matches!(EwmaState::new(EwmaOptions::new(u64::MAX).unwrap(), limits.with_max_state_bytes(23)),
        Err(calc_flow::CalcFlowError::InvalidArgument { field, .. }) if field == "state.max_state_bytes")
    );
    let mut mean = RollingMeanState::new(
        RollingMeanOptions::new(u64::MAX).unwrap(),
        limits.with_max_state_bytes(80),
    )
    .unwrap();
    assert_eq!(mean.state_bytes(), 64);
    assert_eq!(
        bits(
            &mean
                .process(view(&[3.0]), &ArrayExecutionOptions::default())
                .unwrap()
        ),
        [Some(3.0_f64.to_bits())]
    );
    assert_eq!(mean.state_bytes(), 80);
    assert_field(
        mean.process(view(&[5.0]), &ArrayExecutionOptions::default()),
        "state.max_state_bytes",
    );
    assert_eq!(mean.state_bytes(), 80);
}

#[test]
fn test_state_peak_admission_counts_old_candidate_and_output_then_allows_retry() {
    let execution = ArrayExecutionOptions::default();
    let mut mean = RollingMeanState::new(
        RollingMeanOptions::new(2).unwrap(),
        ArrayStateOptions::default(),
    )
    .unwrap();
    mean.process(view(&[1.0, 3.0]), &execution).unwrap();
    let old = mean.state_bytes();
    // Old history has two slots, the isolated candidate needs the third insertion slot.
    let peak = old + 64 + 16 * 3 + 8 + 1 + size_of::<usize>();
    assert_field(
        mean.process(view(&[5.0]), &execution.with_max_allocation_bytes(peak - 1)),
        "state.peak_bytes",
    );
    assert_eq!(mean.state_bytes(), old);
    assert_eq!(
        bits(
            &mean
                .process(view(&[5.0]), &execution.with_max_allocation_bytes(peak))
                .unwrap()
        ),
        [Some(4.0_f64.to_bits())]
    );
    let mut ewma =
        EwmaState::new(EwmaOptions::new(3).unwrap(), ArrayStateOptions::default()).unwrap();
    ewma.process(view(&[1.0]), &execution).unwrap();
    let peak = 24 + 24 + 8 + 1 + size_of::<usize>();
    assert_field(
        ewma.process(view(&[3.0]), &execution.with_max_allocation_bytes(peak - 1)),
        "state.peak_bytes",
    );
    assert_eq!(
        bits(
            &ewma
                .process(view(&[3.0]), &execution.with_max_allocation_bytes(peak))
                .unwrap()
        ),
        [Some(2.0_f64.to_bits())]
    );
}

#[test]
fn test_state_rejects_descriptors_and_output_limits_even_for_empty_input() {
    let execution = ArrayExecutionOptions::default();
    let mut mean = RollingMeanState::new(
        RollingMeanOptions::new(2).unwrap(),
        ArrayStateOptions::default(),
    )
    .unwrap();
    let mut ewma =
        EwmaState::new(EwmaOptions::new(3).unwrap(), ArrayStateOptions::default()).unwrap();
    let empty = NativeArray::from_f64(vec![], vec![0, 2], None).unwrap();
    let boolean = datafusion::arrow::buffer::BooleanBuffer::from(vec![]);
    for input in [ArrayView::scalar_f64(Some(1.0)), empty.view()] {
        assert_field(mean.process(input, &execution), "input.shape");
        assert_field(ewma.process(input, &execution), "input.shape");
    }
    let input = ArrayView::from_boolean(&boolean, None).unwrap();
    assert_field(mean.process(input, &execution), "input.dtype");
    assert_field(ewma.process(input, &execution), "input.dtype");
    assert_field(
        mean.process(view(&[1.0]), &execution.with_max_elements(0)),
        "outputs.output.elements",
    );
    assert_field(
        ewma.process(view(&[1.0]), &execution.with_max_allocation_bytes(0)),
        "outputs.output.bytes",
    );
    assert_field(
        mean.process(view(&[]), &execution.with_max_allocation_bytes(0)),
        "outputs.output.bytes",
    );
    assert_eq!(mean.state_bytes(), 64);
    let result = mean.process(view(&[]), &execution).unwrap();
    assert!(result.float_view().unwrap().validity().is_none());
    let result = ewma.process(view(&[]), &execution).unwrap();
    assert!(result.float_view().unwrap().validity().is_some());
    assert_eq!(
        bits(&mean.process(view(&[7.0]), &execution).unwrap()),
        [Some(7.0_f64.to_bits())]
    );
    assert_eq!(
        bits(&ewma.process(view(&[7.0]), &execution).unwrap()),
        [Some(7.0_f64.to_bits())]
    );
}

#[test]
fn test_state_debug_reports_descriptors_without_sample_values() {
    let execution = ArrayExecutionOptions::default();
    let mut mean = RollingMeanState::new(
        RollingMeanOptions::new(2).unwrap(),
        ArrayStateOptions::default(),
    )
    .unwrap();
    let mut ewma =
        EwmaState::new(EwmaOptions::new(3).unwrap(), ArrayStateOptions::default()).unwrap();
    mean.process(view(&[123_456.75]), &execution).unwrap();
    ewma.process(view(&[123_456.75]), &execution).unwrap();
    for description in [format!("{mean:?}"), format!("{ewma:?}")] {
        assert!(!description.contains("123456"));
        assert!(description.contains("state_bytes"));
    }
}

#[test]
fn test_sliced_nullable_chunks_keep_global_mean_cadence_and_eager_bits() {
    use calc_flow::{RollingNumericalProfile, array::rolling_mean};
    use datafusion::arrow::array::{Array, Float64Array};
    use std::sync::Arc;
    let prices = Float64Array::from(
        (0..135)
            .map(|index| match index {
                0 | 134 => Some(9e99),
                17 | 66 => None,
                4 => Some(f64::NAN),
                29 => Some(f64::INFINITY),
                30 => Some(f64::NEG_INFINITY),
                31 => Some(-0.0),
                _ => Some(1e12 + f64::from(index % 7) * 0.125),
            })
            .collect::<Vec<_>>(),
    );
    let sliced = prices.slice(1, 133);
    let input = NativeArray::from_arrow(Arc::new(sliced), vec![133]).unwrap();
    let execution = ArrayExecutionOptions::default();
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        let options = RollingMeanOptions::new(7)
            .unwrap()
            .with_min_periods(3)
            .unwrap()
            .with_numerical_profile(profile);
        let expected =
            bits(&rolling_mean(input.float_view().unwrap(), options, &execution).unwrap());
        let mut state = RollingMeanState::new(options, ArrayStateOptions::default()).unwrap();
        let mut observed = Vec::new();
        for (offset, len) in [(0, 13), (13, 50), (63, 0), (63, 1), (64, 69)] {
            let chunk =
                NativeArray::from_arrow(input.as_arrow().slice(offset, len), vec![len]).unwrap();
            observed.extend(bits(&state.process(chunk.view(), &execution).unwrap()));
        }
        assert_eq!(observed, expected);
        assert_eq!(input.as_arrow().null_count(), 2);
    }
}

#[test]
fn test_states_preserve_independent_extreme_mean_and_ewma_oracles() {
    let execution = ArrayExecutionOptions::default();
    let mut mean = RollingMeanState::new(
        RollingMeanOptions::new(2).unwrap(),
        ArrayStateOptions::default(),
    )
    .unwrap();
    mean.process(view(&[-1e308]), &execution).unwrap();
    assert_eq!(
        bits(&mean.process(view(&[1e308]), &execution).unwrap()),
        [Some(f64::INFINITY.to_bits())]
    );
    mean.reset();
    mean.process(view(&[1e155, 1e155, 1e155]), &execution)
        .unwrap();
    assert_eq!(
        bits(&mean.process(view(&[-1e155]), &execution).unwrap()),
        [Some(0.0_f64.to_bits())]
    );
    let mut ewma = EwmaState::new(
        EwmaOptions::new(1).unwrap().with_min_periods(3).unwrap(),
        ArrayStateOptions::default(),
    )
    .unwrap();
    let first = ArrayView::from_float(FloatView::from_f32(&[-0.0, 1.0]));
    assert_eq!(
        bits(&ewma.process(first, &execution).unwrap()),
        [None, None]
    );
    let values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 4.0];
    assert_eq!(
        bits(&ewma.process(view(&values), &execution).unwrap()),
        [
            None,
            Some(f64::INFINITY.to_bits()),
            Some(f64::NEG_INFINITY.to_bits()),
            Some(4.0_f64.to_bits())
        ]
    );
    let mut ewma =
        EwmaState::new(EwmaOptions::new(3).unwrap(), ArrayStateOptions::default()).unwrap();
    assert_eq!(
        bits(
            &ewma
                .process(
                    ArrayView::from_float(FloatView::from_f32(&[-0.0])),
                    &execution
                )
                .unwrap()
        ),
        [Some((-0.0_f64).to_bits())]
    );
    ewma.process(view(&[f64::INFINITY]), &execution).unwrap();
    let last = ewma
        .process(view(&[f64::INFINITY, f64::NAN]), &execution)
        .unwrap();
    assert!(
        bits(&last)
            .iter()
            .all(|value| value.is_some_and(|value| f64::from_bits(value).is_nan()))
    );
}

#[test]
fn test_direct_states_accept_widened_f32_then_f64_without_binding_dtype() {
    use calc_flow::array::{ewma, rolling_mean};
    let execution = ArrayExecutionOptions::default();
    let first = [1.25_f32, 3.5];
    let second = [5.75_f64, 7.0];
    let joined = [1.25_f64, 3.5, 5.75, 7.0];
    let options = RollingMeanOptions::new(u64::MAX).unwrap();
    let mut mean = RollingMeanState::new(options, ArrayStateOptions::default()).unwrap();
    let mut actual = bits(
        &mean
            .process(
                ArrayView::from_float(FloatView::from_f32(&first)),
                &execution,
            )
            .unwrap(),
    );
    actual.extend(bits(&mean.process(view(&second), &execution).unwrap()));
    assert_eq!(
        actual,
        bits(&rolling_mean(FloatView::from_f64(&joined), options, &execution).unwrap())
    );
    let options = EwmaOptions::new(u64::MAX).unwrap();
    let mut state = EwmaState::new(options, ArrayStateOptions::default()).unwrap();
    let mut actual = bits(
        &state
            .process(
                ArrayView::from_float(FloatView::from_f32(&first)),
                &execution,
            )
            .unwrap(),
    );
    actual.extend(bits(&state.process(view(&second), &execution).unwrap()));
    assert_eq!(
        actual,
        bits(&ewma(FloatView::from_f64(&joined), options, &execution).unwrap())
    );
}

#[test]
fn test_retained_limit_failure_allows_a_smaller_chunk_without_double_counting() {
    let execution = ArrayExecutionOptions::default();
    let mut state = RollingMeanState::new(
        RollingMeanOptions::new(2).unwrap(),
        ArrayStateOptions::default().with_max_state_bytes(96),
    )
    .unwrap();
    state.process(view(&[1.0]), &execution).unwrap();
    assert_field(
        state.process(view(&[3.0, 5.0]), &execution),
        "state.max_state_bytes",
    );
    assert_eq!(state.state_bytes(), 80);
    assert_eq!(
        bits(&state.process(view(&[3.0]), &execution).unwrap()),
        [Some(2.0_f64.to_bits())]
    );
    assert_eq!(state.state_bytes(), 96);
}

#[test]
fn test_ewma_offset_chunks_preserve_null_carry_and_result_lifetime() {
    use calc_flow::array::ewma;
    use datafusion::arrow::array::{Array, Float32Array};
    use std::sync::Arc;
    let execution = ArrayExecutionOptions::default();
    let options = EwmaOptions::new(3).unwrap().with_min_periods(2).unwrap();
    let values = Float32Array::from(vec![
        Some(99.0),
        None,
        Some(-0.0),
        Some(f32::NAN),
        Some(4.0),
        None,
        Some(f32::INFINITY),
        Some(f32::INFINITY),
        Some(99.0),
    ]);
    let input = NativeArray::from_arrow(Arc::new(values.slice(1, 7)), vec![7]).unwrap();
    let expected = bits(&ewma(input.float_view().unwrap(), options, &execution).unwrap());
    let (mut actual, retained) = {
        let mut state = EwmaState::new(options, ArrayStateOptions::default()).unwrap();
        let first = NativeArray::from_arrow(input.as_arrow().slice(0, 3), vec![3]).unwrap();
        let actual = bits(&state.process(first.view(), &execution).unwrap());
        state.process(view(&[]), &execution).unwrap();
        let second = NativeArray::from_arrow(input.as_arrow().slice(3, 4), vec![4]).unwrap();
        let retained = state.process(second.view(), &execution).unwrap();
        state.reset();
        (actual, retained)
    };
    drop(input);
    actual.extend(bits(&retained));
    assert_eq!(actual, expected);
    assert!(f64::from_bits(actual[6].unwrap()).is_nan());
}
