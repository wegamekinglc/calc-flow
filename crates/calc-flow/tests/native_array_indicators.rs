use calc_flow::{
    Result,
    array::{ArrayExecutionOptions, ArrayView, AtrOptions, FloatView, RsiOptions, atr, rsi},
};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}

fn assert_values(output: &calc_flow::array::NativeArray, expected: &[Option<f64>]) {
    let input = output.float_view().unwrap();
    let calc_flow::array::FloatValues::Float64(values) = input.values() else {
        panic!("indicators return float64");
    };
    let validity = input.validity().expect("explicit validity is required");
    assert_eq!(output.shape(), &[expected.len()]);
    for (index, expected) in expected.iter().enumerate() {
        assert_eq!(validity.is_valid(index), expected.is_some(), "row {index}");
        assert_eq!(
            values[index].to_bits(),
            expected.unwrap_or(0.0).to_bits(),
            "row {index}"
        );
    }
}

#[test]
fn test_rsi_seed_and_wilder_follow_exact_operation_order() -> Result<()> {
    let output = rsi(
        view(&[1.0, 2.0, 1.0, 3.0]),
        RsiOptions::new(2)?,
        &ArrayExecutionOptions::default(),
    )?;
    assert_values(&output, &[None, None, Some(50.0), Some(100.0 / 1.2)]);
    Ok(())
}

#[test]
fn test_atr_first_row_only_establishes_previous_close() -> Result<()> {
    let output = atr(
        view(&[10.0, 12.0, 11.0, 14.0]),
        view(&[8.0, 9.0, 8.0, 11.0]),
        view(&[9.0, 10.0, 9.0, 13.0]),
        AtrOptions::new(2)?,
        &ArrayExecutionOptions::default(),
    )?;
    assert_values(&output, &[None, None, Some(3.0), Some(4.0)]);
    Ok(())
}

#[test]
fn test_underflow_flat_up_down_and_period_one_are_explicit() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    let tiny = f64::from_bits(1);
    assert_values(
        &rsi(view(&[0.0, tiny, 0.0]), RsiOptions::new(2)?, &execution)?,
        &[None, None, Some(100.0)],
    );
    assert_values(
        &rsi(view(&[1.0, 2.0]), RsiOptions::new(u64::MAX)?, &execution)?,
        &[None, None],
    );
    Ok(())
}

#[test]
fn test_period_one_flat_up_and_down_segments() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    for (input, expected) in [
        ([0.0, -0.0, 0.0], 50.0),
        ([0.0, 1.0, 2.0], 100.0),
        ([2.0, 1.0, 0.0], 0.0),
    ] {
        assert_values(
            &rsi(view(&input), RsiOptions::new(1)?, &execution)?,
            &[None, Some(expected), Some(expected)],
        );
    }

    Ok(())
}

#[test]
fn test_missing_resets_whole_segment_and_null_payload_is_not_inspected() -> Result<()> {
    use calc_flow::array::{ArrayStateOptions, FloatValues, RsiState};
    use datafusion::arrow::buffer::NullBuffer;
    let input = [1.0, 2.0, f64::INFINITY, 3.0, 2.0, f64::NAN, 8.0, 8.0];
    let validity = NullBuffer::from(vec![true, true, false, true, true, true, true, true]);
    let input = ArrayView::from_float(FloatView::try_new(
        FloatValues::Float64(&input),
        Some(&validity),
    )?);
    let mut state = RsiState::new(RsiOptions::new(1)?, ArrayStateOptions::default())?;
    assert_values(
        &state.process(input, &ArrayExecutionOptions::default())?,
        &[
            None,
            Some(100.0),
            None,
            None,
            Some(0.0),
            None,
            None,
            Some(50.0),
        ],
    );
    Ok(())
}

#[test]
fn test_atr_checks_valid_infinity_and_domain_before_missing() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    for (high, low, close, path) in [
        (f64::NAN, f64::INFINITY, 0.0, "inputs.low[0]"),
        (2.0, 3.0, f64::NAN, "inputs.high[0]"),
        (2.0, 3.0, f64::INFINITY, "inputs.close[0]"),
        (1e308, -1e308, f64::NAN, "inputs.high[0]"),
    ] {
        let error = atr(
            view(&[high]),
            view(&[low]),
            view(&[close]),
            AtrOptions::new(1)?,
            &execution,
        )
        .unwrap_err();
        assert!(error.to_string().contains(path), "{error}");
    }
    Ok(())
}

#[test]
fn test_owned_chunk_failure_is_atomic_and_reset_retains_constant_charge() -> Result<()> {
    use calc_flow::array::{ArrayStateOptions, RsiState};
    let execution = ArrayExecutionOptions::default();
    let mut state = RsiState::new(RsiOptions::new(2)?, ArrayStateOptions::default())?;
    state.process(view(&[1.0, 2.0]), &execution)?;
    assert!(
        state
            .process(view(&[3.0, f64::INFINITY]), &execution)
            .is_err()
    );
    assert_values(
        &state.process(view(&[1.0, 3.0]), &execution)?,
        &[Some(50.0), Some(100.0 / 1.2)],
    );
    assert_eq!(state.state_bytes(), 48);
    state.reset();
    assert_values(
        &state.process(view(&[1.0, 2.0]), &execution)?,
        &[None, None],
    );
    Ok(())
}

#[test]
fn test_atr_gap_failure_is_atomic_and_reset_retains_fixed_charge() -> Result<()> {
    use calc_flow::array::{ArrayStateOptions, AtrState};
    let execution = ArrayExecutionOptions::default();
    let mut state = AtrState::new(AtrOptions::new(1)?, ArrayStateOptions::default())?;
    state.process(view(&[1.0]), view(&[0.0]), view(&[-1e308]), &execution)?;
    assert!(
        state
            .process(view(&[1e308]), view(&[1e308]), view(&[1.0]), &execution)
            .is_err()
    );
    assert_values(
        &state.process(view(&[1.0]), view(&[0.0]), view(&[0.0]), &execution)?,
        &[Some(1e308)],
    );
    state.reset();
    assert_eq!(state.state_bytes(), 40);
    Ok(())
}

#[test]
fn test_descriptors_period_and_peak_limits_reject_before_values() -> Result<()> {
    use calc_flow::array::{ArrayStateOptions, RsiState};
    assert!(RsiOptions::new(0).is_err());
    assert!(AtrOptions::new(0).is_err());
    let options = RsiOptions::new(1)?;
    let execution = ArrayExecutionOptions::default();
    assert!(rsi(ArrayView::scalar_f64(Some(1.0)), options, &execution).is_err());
    assert!(rsi(ArrayView::scalar_boolean(Some(true)), options, &execution).is_err());
    assert!(
        atr(
            view(&[1.0]),
            view(&[]),
            view(&[1.0]),
            AtrOptions::new(1)?,
            &execution
        )
        .is_err()
    );
    assert!(
        RsiState::new(
            options,
            ArrayStateOptions::default().with_max_state_bytes(47)
        )
        .is_err()
    );
    let output = 2 * 8 + 1 + size_of::<usize>();
    assert!(
        rsi(
            view(&[0.0, 1.0]),
            options,
            &execution.with_max_allocation_bytes(48 + output - 1)
        )
        .is_err()
    );
    rsi(
        view(&[0.0, 1.0]),
        options,
        &execution.with_max_allocation_bytes(48 + output),
    )?;
    let mut state = RsiState::new(options, ArrayStateOptions::default())?;
    assert!(
        state
            .process(
                view(&[0.0, 1.0]),
                &execution.with_max_allocation_bytes(96 + output - 1)
            )
            .is_err()
    );
    assert_values(
        &state.process(
            view(&[0.0, 1.0]),
            &execution.with_max_allocation_bytes(96 + output),
        )?,
        &[None, Some(100.0)],
    );
    assert_values(&rsi(view(&[]), options, &execution)?, &[]);
    Ok(())
}

#[path = "fixtures/indicators/golden.rs"]
mod golden;

#[test]
fn test_independent_ieee_and_fraction_fixtures_match_eager() -> Result<()> {
    for fixture in golden::FIXTURES {
        assert_eager_fixture(fixture)?;
    }
    Ok(())
}

fn assert_eager_fixture(fixture: &golden::Fixture) -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    let options = RsiOptions::new(fixture.period)?;
    let atr_options = AtrOptions::new(fixture.period)?;
    let output = rsi(view(golden::PRICE), options, &execution)?;
    assert_golden(&output, fixture.rsi, fixture.fraction_rsi, false);
    let output = atr(
        view(golden::HIGH),
        view(golden::LOW),
        view(golden::CLOSE),
        atr_options,
        &execution,
    )?;
    assert_golden(&output, fixture.atr, fixture.fraction_atr, true);
    Ok(())
}

#[test]
fn test_independent_fixtures_match_chunked_states() -> Result<()> {
    for fixture in golden::FIXTURES {
        assert_chunked_fixture(fixture)?;
    }
    Ok(())
}

fn fixture_states(period: u64) -> Result<(calc_flow::array::RsiState, calc_flow::array::AtrState)> {
    use calc_flow::array::{ArrayStateOptions, AtrState, RsiState};
    Ok((
        RsiState::new(RsiOptions::new(period)?, ArrayStateOptions::default())?,
        AtrState::new(AtrOptions::new(period)?, ArrayStateOptions::default())?,
    ))
}

fn assert_chunked_fixture(fixture: &golden::Fixture) -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    let (mut rsi_state, mut atr_state) = fixture_states(fixture.period)?;
    for start in (0..golden::PRICE.len()).step_by(5) {
        let end = (start + 5).min(golden::PRICE.len());
        let output = rsi_state.process(view(&golden::PRICE[start..end]), &execution)?;
        assert_golden(
            &output,
            &fixture.rsi[start..end],
            &fixture.fraction_rsi[start..end],
            false,
        );
        let output = atr_state.process(
            view(&golden::HIGH[start..end]),
            view(&golden::LOW[start..end]),
            view(&golden::CLOSE[start..end]),
            &execution,
        )?;
        assert_golden(
            &output,
            &fixture.atr[start..end],
            &fixture.fraction_atr[start..end],
            true,
        );
    }
    Ok(())
}

fn assert_golden(
    output: &calc_flow::array::NativeArray,
    bits: &[Option<u64>],
    rational: &[Option<f64>],
    relative: bool,
) {
    let expected: Vec<_> = bits.iter().map(|bits| bits.map(f64::from_bits)).collect();
    assert_values(output, &expected);
    for (actual, expected) in expected.into_iter().zip(rational) {
        if let (Some(actual), Some(expected)) = (actual, expected) {
            let scale = if relative { expected.abs() } else { 100.0 };
            assert!((actual - expected).abs() <= scale * 256.0 * f64::EPSILON);
        }
    }
}

#[test]
fn test_atr_mixed_float_sliced_nulls_preserve_inputs_and_output_ownership() -> Result<()> {
    use calc_flow::array::FloatValues;
    use datafusion::arrow::buffer::NullBuffer;
    let output = {
        let high = [999.0_f32, 10.0, 12.0, f32::INFINITY, 14.0, 15.0, 999.0];
        let low = [8.0, 9.0, f64::INFINITY, 11.0, 12.0];
        let close = [9.0, 10.0, f64::INFINITY, 13.0, 14.0];
        let mask = NullBuffer::from(vec![true, true, true, false, true, true, true]).slice(1, 5);
        let high_view = ArrayView::from_float(FloatView::try_new(
            FloatValues::Float32(&high[1..6]),
            Some(&mask),
        )?);
        let low_view =
            ArrayView::from_float(FloatView::try_new(FloatValues::Float64(&low), Some(&mask))?);
        let close_view = ArrayView::from_float(FloatView::try_new(
            FloatValues::Float64(&close),
            Some(&mask),
        )?);
        let output = atr(
            high_view,
            low_view,
            close_view,
            AtrOptions::new(1)?,
            &ArrayExecutionOptions::default(),
        )?;
        assert_eq!(high[3].to_bits(), f32::INFINITY.to_bits());
        assert_eq!(mask.null_count(), 1);
        output
    };
    assert_values(&output, &[None, Some(3.0), None, None, Some(3.0)]);
    Ok(())
}

#[test]
fn test_finite_rsi_difference_overflow_during_warmup_is_atomic() -> Result<()> {
    use calc_flow::array::{ArrayStateOptions, RsiState};
    let execution = ArrayExecutionOptions::default();
    let options = RsiOptions::new(2)?;
    let mut state = RsiState::new(options, ArrayStateOptions::default())?;
    state.process(view(&[1e308]), &execution)?;
    let error = state
        .process(view(&[1e308, -1e308]), &execution)
        .unwrap_err();
    assert!(error.to_string().contains("input[1]"));
    assert!(error.to_string().contains("RSI price difference"));
    assert_values(
        &state.process(view(&[1e308, 1e308]), &execution)?,
        &[None, Some(50.0)],
    );
    Ok(())
}

#[test]
fn test_large_representable_means_do_not_use_an_overflowing_sum_seed() -> Result<()> {
    let output = rsi(
        view(&[0.0, 1e308, 0.0, 1e308, 0.0]),
        RsiOptions::new(2)?,
        &ArrayExecutionOptions::default(),
    )?;
    // Independent Python float64 operation-order reference, outside Fraction's domain.
    assert_values(
        &output,
        &[
            None,
            None,
            Some(50.0),
            Some(75.0),
            Some(f64::from_bits(0x4042_bfff_ffff_ffff)),
        ],
    );
    Ok(())
}
