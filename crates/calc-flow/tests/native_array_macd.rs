use calc_flow::{
    Result,
    array::{
        ArrayExecutionOptions, ArrayStateOptions, ArrayView, FloatValues, FloatView, MacdOptions,
        MacdOutputs, MacdState, NativeArray, macd_components,
    },
};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}
fn values(array: &NativeArray) -> Vec<Option<f64>> {
    let view = array.float_view().unwrap();
    let FloatValues::Float64(values) = view.values() else {
        panic!("expected float64")
    };
    let validity = view.validity();
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if validity.is_some_and(|mask| mask.is_null(index)) {
                assert_eq!(value.to_bits(), 0);
                None
            } else {
                Some(*value)
            }
        })
        .collect()
}
fn assert_values(array: &NativeArray, expected: &[Option<f64>]) {
    assert!(
        array.float_view().unwrap().validity().is_some(),
        "explicit validity, even empty"
    );
    let actual = values(array);
    assert_eq!(array.shape(), &[expected.len()]);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        match (actual, expected) {
            (Some(actual), Some(expected)) if expected.is_nan() => assert!(actual.is_nan()),
            (Some(actual), Some(expected)) => assert_eq!(actual.to_bits(), expected.to_bits()),
            (None, None) => {}
            _ => panic!("validity differs"),
        }
    }
}
fn assert_components(output: &MacdOutputs, expected: [&[Option<f64>]; 3]) {
    for (array, expected) in [output.line(), output.signal(), output.histogram()]
        .into_iter()
        .zip(expected)
    {
        assert_values(array, expected);
    }
}

#[test]
fn test_macd_defaults_and_positive_independent_warmups() -> Result<()> {
    let options = MacdOptions::default();
    assert_eq!(
        (
            options.fast_span(),
            options.slow_span(),
            options.signal_span()
        ),
        (12, 26, 9)
    );
    assert_eq!(
        (options.min_periods(), options.signal_min_periods()),
        (1, 1)
    );
    for spans in [(0, 2, 1), (1, 0, 1), (1, 2, 0), (2, 2, 1), (3, 2, 1)] {
        assert!(MacdOptions::new(spans.0, spans.1, spans.2).is_err());
    }
    assert!(options.with_min_periods(0).is_err());
    assert!(options.with_signal_min_periods(0).is_err());
    let large = MacdOptions::new(1, 2, u64::MAX)?
        .with_min_periods(u64::MAX)?
        .with_signal_min_periods(u64::MAX)?;
    let output = macd_components(view(&[1.0, 2.0]), large, &ArrayExecutionOptions::default())?;
    assert_components(&output, [&[None, None]; 3]);
    Ok(())
}

#[test]
fn test_macd_literally_composes_three_ewma_transitions() -> Result<()> {
    let output = macd_components(
        view(&[10.0, 14.0, 18.0]),
        MacdOptions::new(1, 3, 3)?,
        &ArrayExecutionOptions::default(),
    )?;
    assert_components(
        &output,
        [
            &[Some(0.0), Some(2.0), Some(3.0)],
            &[Some(0.0), Some(1.0), Some(2.0)],
            &[Some(0.0), Some(1.0), Some(1.0)],
        ],
    );
    Ok(())
}

#[test]
fn test_missing_price_still_updates_signal_from_carried_line() -> Result<()> {
    use datafusion::arrow::buffer::NullBuffer;
    let data = [10.0, 14.0, f64::INFINITY, f64::NAN];
    let mask = NullBuffer::from(vec![true, true, false, true]);
    let input = ArrayView::from_float(FloatView::try_new(
        FloatValues::Float64(&data),
        Some(&mask),
    )?);
    let output = macd_components(
        input,
        MacdOptions::new(1, 3, 3)?,
        &ArrayExecutionOptions::default(),
    )?;
    assert_components(
        &output,
        [
            &[Some(0.0), Some(2.0), Some(2.0), Some(2.0)],
            &[Some(0.0), Some(1.0), Some(1.5), Some(1.75)],
            &[Some(0.0), Some(1.0), Some(0.5), Some(0.25)],
        ],
    );
    Ok(())
}

#[test]
fn test_nonfinite_line_is_valid_and_only_nan_skips_signal() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    let options = MacdOptions::new(1, 3, 3)?;
    let output = macd_components(view(&[10.0, 14.0, f64::INFINITY, 3.0]), options, &execution)?;
    assert_components(
        &output,
        [
            &[Some(0.0), Some(2.0), Some(f64::NAN), Some(f64::NAN)],
            &[Some(0.0), Some(1.0), Some(1.0), Some(1.0)],
            &[Some(0.0), Some(1.0), Some(f64::NAN), Some(f64::NAN)],
        ],
    );
    let output = macd_components(view(&[-1e308, 1e308, f64::NAN]), options, &execution)?;
    assert_components(
        &output,
        [
            &[Some(0.0), Some(f64::NEG_INFINITY), Some(f64::NEG_INFINITY)],
            &[Some(0.0), Some(f64::NEG_INFINITY), Some(f64::NAN)],
            &[Some(0.0), Some(f64::NAN), Some(f64::NAN)],
        ],
    );
    Ok(())
}

#[test]
fn test_owned_macd_errors_are_atomic_and_outputs_survive_reset() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    let options = MacdOptions::new(1, 3, 3)?;
    let mut state = MacdState::new(options, ArrayStateOptions::default())?;
    state.process(view(&[10.0]), &execution)?;
    assert!(
        state
            .process(view(&[14.0]), &execution.with_max_elements(2))
            .is_err()
    );
    assert!(
        state
            .process(view(&[14.0, 18.0]).reshape(&[1, 2])?, &execution)
            .is_err()
    );
    let output = state.process(view(&[14.0, 18.0]), &execution)?;
    assert_eq!(state.state_bytes(), 56);
    state.reset();
    assert_components(
        &output,
        [
            &[Some(2.0), Some(3.0)],
            &[Some(1.0), Some(2.0)],
            &[Some(1.0), Some(1.0)],
        ],
    );
    assert_values(
        state.process(view(&[14.0]), &execution)?.line(),
        &[Some(0.0)],
    );
    Ok(())
}

#[test]
fn test_macd_charges_three_outputs_and_fixed_state_at_exact_peak() -> Result<()> {
    let options = MacdOptions::new(1, 3, 3)?;
    let execution = ArrayExecutionOptions::default();
    let output = 3 * (8 + 1 + size_of::<usize>());
    assert!(
        MacdState::new(
            options,
            ArrayStateOptions::default().with_max_state_bytes(55)
        )
        .is_err()
    );
    assert!(macd_components(view(&[1.0]), options, &execution.with_max_elements(2)).is_err());
    assert!(
        macd_components(
            view(&[1.0]),
            options,
            &execution.with_max_allocation_bytes(56 + output - 1)
        )
        .is_err()
    );
    macd_components(
        view(&[1.0]),
        options,
        &execution
            .with_max_elements(3)
            .with_max_allocation_bytes(56 + output),
    )?;
    let mut state = MacdState::new(options, ArrayStateOptions::default())?;
    assert!(
        state
            .process(
                view(&[1.0]),
                &execution.with_max_allocation_bytes(112 + output - 1)
            )
            .is_err()
    );
    state.process(
        view(&[1.0]),
        &execution.with_max_allocation_bytes(112 + output),
    )?;
    let empty = state.process(
        view(&[]),
        &execution
            .with_max_elements(0)
            .with_max_allocation_bytes(56 + 3 * size_of::<usize>()),
    )?;
    assert_components(&empty, [&[]; 3]);
    Ok(())
}

#[path = "fixtures/macd/golden.rs"]
mod golden;

fn fixture_options(fixture: &golden::Fixture) -> Result<MacdOptions> {
    let [fast, slow, signal, minimum, signal_minimum] = fixture.parameters;
    MacdOptions::new(fast, slow, signal)?
        .with_min_periods(minimum)?
        .with_signal_min_periods(signal_minimum)
}
fn fixture_input(fixture: &golden::Fixture) -> Result<NativeArray> {
    use datafusion::arrow::buffer::NullBuffer;
    let values = fixture
        .input
        .iter()
        .map(|value| value.map_or(f64::INFINITY, f64::from_bits))
        .collect();
    let validity = NullBuffer::from(
        fixture
            .input
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>(),
    );
    NativeArray::from_f64(values, vec![fixture.input.len()], Some(validity))
}
fn assert_reference(output: &MacdOutputs, rows: &[[Option<u64>; 3]]) {
    for (component, array) in [output.line(), output.signal(), output.histogram()]
        .into_iter()
        .enumerate()
    {
        let expected: Vec<_> = rows
            .iter()
            .map(|row| row[component].map(f64::from_bits))
            .collect();
        assert_values(array, &expected);
    }
}

#[test]
fn test_independent_ieee_macd_fixtures_match_eager() -> Result<()> {
    for fixture in golden::FIXTURES {
        let input = fixture_input(fixture)?;
        let output = macd_components(
            input.view(),
            fixture_options(fixture)?,
            &ArrayExecutionOptions::default(),
        )
        .expect(fixture.name);
        assert_reference(&output, fixture.expected);
    }
    Ok(())
}

#[test]
fn test_independent_ieee_macd_fixtures_match_owned_chunks() -> Result<()> {
    for fixture in golden::FIXTURES {
        assert_chunked_fixture(fixture)?;
    }
    Ok(())
}
fn assert_chunked_fixture(fixture: &golden::Fixture) -> Result<()> {
    let input = fixture_input(fixture)?;
    let mut state = MacdState::new(fixture_options(fixture)?, ArrayStateOptions::default())?;
    for offset in (0..input.len()).step_by(3) {
        let len = 3.min(input.len() - offset);
        let chunk = NativeArray::from_arrow(input.as_arrow().slice(offset, len), vec![len])?;
        let output = state
            .process(chunk.view(), &ArrayExecutionOptions::default())
            .expect(fixture.name);
        assert_reference(&output, &fixture.expected[offset..offset + len]);
    }
    Ok(())
}

fn legacy_average(
    input: &NativeArray,
    span: u64,
    minimum: u64,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    use calc_flow::array::{EwmaOptions, ewma};
    ewma(
        input.float_view()?,
        EwmaOptions::new(span)?.with_min_periods(minimum)?,
        execution,
    )
}
fn legacy_components(
    input: &NativeArray,
    options: MacdOptions,
    execution: &ArrayExecutionOptions,
) -> Result<[NativeArray; 3]> {
    use calc_flow::array::subtract;
    let fast = legacy_average(input, options.fast_span(), options.min_periods(), execution)?;
    let slow = legacy_average(input, options.slow_span(), options.min_periods(), execution)?;
    let line = subtract(fast.view(), slow.view(), execution)?;
    let signal = legacy_average(
        &line,
        options.signal_span(),
        options.signal_min_periods(),
        execution,
    )?;
    let histogram = subtract(line.view(), signal.view(), execution)?;
    Ok([line, signal, histogram])
}
#[test]
fn test_macd_retains_legacy_ewma_and_subtraction_semantics() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    for fixture in golden::FIXTURES {
        let input = fixture_input(fixture)?;
        let options = fixture_options(fixture)?;
        let expected = legacy_components(&input, options, &execution)?;
        let actual = macd_components(input.view(), options, &execution)?;
        for (array, expected) in [actual.line(), actual.signal(), actual.histogram()]
            .into_iter()
            .zip(&expected)
        {
            assert_values(array, &values(expected));
        }
    }
    Ok(())
}

#[test]
fn test_output_owners_move_without_copying_and_accept_float32_then_float64() -> Result<()> {
    use std::sync::Arc;
    let execution = ArrayExecutionOptions::default();
    let mut state = MacdState::new(MacdOptions::new(1, 3, 1)?, ArrayStateOptions::default())?;
    let output = state.process(
        ArrayView::from_float(FloatView::from_f32(&[-0.0, 1.0])),
        &execution,
    )?;
    let retained = Arc::clone(output.line().as_arrow());
    let (line, signal, histogram) = output.into_arrays();
    assert!(Arc::ptr_eq(&retained, line.as_arrow()));
    assert_values(&line, &[Some(0.0), Some(0.5)]);
    assert_values(&signal, &[Some(0.0), Some(0.5)]);
    assert_values(&histogram, &[Some(0.0), Some(0.0)]);
    assert_values(
        state.process(view(&[3.0]), &execution)?.line(),
        &[Some(1.25)],
    );
    Ok(())
}

#[test]
fn test_macd_preserves_signed_zero_of_each_subtraction() -> Result<()> {
    let output = macd_components(
        view(&[0.0, -0.0]),
        MacdOptions::new(1, 3, 3)?,
        &ArrayExecutionOptions::default(),
    )?;
    assert_components(
        &output,
        [
            &[Some(0.0), Some(-0.0)],
            &[Some(0.0), Some(0.0)],
            &[Some(0.0), Some(-0.0)],
        ],
    );
    Ok(())
}
