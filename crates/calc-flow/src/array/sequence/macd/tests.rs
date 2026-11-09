use super::*;
use crate::{CalcFlowError, array::FloatView};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}
fn parts(kernel: &MacdKernel) -> [u64; 7] {
    [
        kernel.positions,
        kernel.fast_count,
        kernel.fast_value.to_bits(),
        kernel.slow_count,
        kernel.slow_value.to_bits(),
        kernel.signal_count,
        kernel.signal_value.to_bits(),
    ]
}
fn cancelled() -> CalcFlowError {
    CalcFlowError::Cancelled {
        run_id: "macd-control".into(),
    }
}
fn assert_field(result: Result<()>, expected: &str) {
    let Err(CalcFlowError::InvalidArgument { field, .. }) = result else {
        panic!("expected invalid state")
    };
    assert_eq!(field, expected);
}

#[test]
fn test_zero_count_state_identifies_the_failed_component_without_normalizing_bits() {
    let options = MacdOptions::default();
    assert_field(
        MacdKernel {
            fast_value: -0.0,
            ..MacdKernel::default()
        }
        .validate(options),
        "state.fast_value",
    );
    assert_field(
        MacdKernel {
            slow_value: f64::NAN,
            ..MacdKernel::default()
        }
        .validate(options),
        "state.slow_value",
    );
    assert_field(
        MacdKernel {
            signal_value: f64::INFINITY,
            ..MacdKernel::default()
        }
        .validate(options),
        "state.signal_value",
    );
}

#[test]
fn test_signal_count_may_exceed_prices_but_respects_warmup_and_positions() -> Result<()> {
    let options = MacdOptions::new(1, 3, 3)?;
    let kernel = MacdKernel {
        positions: 5,
        fast_count: 2,
        fast_value: -0.0,
        slow_count: 2,
        slow_value: f64::INFINITY,
        signal_count: 5,
        signal_value: f64::from_bits(0x7ff8_0000_0000_0042),
    };
    kernel.validate(options)?;
    assert_field(
        MacdKernel {
            fast_count: 3,
            ..kernel
        }
        .validate(options),
        "state.fast_count",
    );
    assert_field(
        MacdKernel {
            signal_count: 6,
            ..kernel
        }
        .validate(options),
        "state.signal_count",
    );
    assert_field(
        kernel.validate(options.with_min_periods(2)?),
        "state.signal_count",
    );
    assert_field(
        kernel.validate(options.with_min_periods(3)?),
        "state.signal_count",
    );
    Ok(())
}

#[test]
fn test_maximum_positions_and_counts_fail_atomically_without_forming_warmup_sum() -> Result<()> {
    let options = MacdOptions::new(1, 3, 3)?;
    let mut kernel = MacdKernel {
        positions: u64::MAX - 1,
        fast_count: u64::MAX - 1,
        slow_count: u64::MAX - 1,
        signal_count: u64::MAX - 1,
        ..MacdKernel::default()
    };
    kernel.push(Some(0.0), &MacdParameters::new(options))?;
    kernel.validate(options)?;
    assert_eq!(
        (kernel.positions, kernel.fast_count, kernel.signal_count),
        (u64::MAX, u64::MAX, u64::MAX)
    );
    let previous = parts(&kernel);
    assert!(kernel.push(None, &MacdParameters::new(options)).is_err());
    assert_eq!(parts(&kernel), previous);
    let mut state = MacdState::new(options, ArrayStateOptions::default())?;
    state.kernel = kernel;
    state.process(view(&[]), &ArrayExecutionOptions::default())?;
    assert!(
        state
            .process(view(&[0.0]), &ArrayExecutionOptions::default())
            .is_err()
    );
    assert_eq!(parts(&state.kernel), previous);
    Ok(())
}

#[test]
fn test_cancelled_candidate_keeps_all_recurrences_and_retained_charge() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    let mut state = MacdState::new(MacdOptions::new(1, 3, 3)?, ArrayStateOptions::default())?;
    state.process(view(&[10.0, 14.0]), &execution)?;
    let previous = parts(&state.kernel);
    let mut calls = 0;
    let input = vec![1.0; 20_000];
    let result = state.prepare_checked(view(&input), &execution, 17, &mut || {
        calls += 1;
        if calls == 7 { Err(cancelled()) } else { Ok(()) }
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(calls, 7);
    assert_eq!(parts(&state.kernel), previous);
    assert_eq!(state.state_bytes(), 56);
    Ok(())
}

#[test]
fn test_empty_candidate_checks_final_cancellation_and_keeps_counts() -> Result<()> {
    let state = MacdState::new(MacdOptions::default(), ArrayStateOptions::default())?;
    let mut calls = 0;
    let result =
        state.prepare_checked(view(&[]), &ArrayExecutionOptions::default(), 0, &mut || {
            calls += 1;
            if calls == 2 { Err(cancelled()) } else { Ok(()) }
        });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(parts(&state.kernel), [0; 7]);
    Ok(())
}
