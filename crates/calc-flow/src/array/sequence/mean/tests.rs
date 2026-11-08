use super::*;
use crate::{
    CalcFlowError,
    array::{FloatValues, FloatView, rolling_mean},
};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}

fn bits(output: &NativeArray) -> Vec<Option<u64>> {
    let view = output.float_view().unwrap();
    let FloatValues::Float64(values) = view.values() else {
        panic!("expected f64")
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            view.validity()
                .is_none_or(|mask| mask.is_valid(index))
                .then(|| value.to_bits())
        })
        .collect()
}

fn cancel_at(limit: usize) -> impl FnMut() -> Result<()> {
    let mut count = 0;
    move || {
        count += 1;
        if count == limit {
            Err(CalcFlowError::Cancelled {
                run_id: "owned-state".into(),
            })
        } else {
            Ok(())
        }
    }
}

#[test]
fn test_owned_mean_position_overflow_rolls_back_then_accepts_last_row() {
    let execution = ArrayExecutionOptions::default();
    let mut state = RollingMeanState::new(
        RollingMeanOptions::new(2).unwrap(),
        ArrayStateOptions::default(),
    )
    .unwrap();
    state.process(view(&[1.0]), &execution).unwrap();
    state.kernel.set_test_positions(u64::MAX - 1);
    let bytes = state.state_bytes();
    assert!(matches!(state.process(view(&[3.0, 5.0]), &execution),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "input.rows"));
    assert_eq!(state.state_bytes(), bytes);
    assert_eq!(
        bits(&state.process(view(&[3.0]), &execution).unwrap()),
        [Some(2.0_f64.to_bits())]
    );
    assert!(state.process(view(&[]), &execution).is_ok());
    assert!(state.process(view(&[5.0]), &execution).is_err());
}

#[test]
fn test_owned_mean_cancels_long_history_copy_without_committing_capacity() {
    let execution = ArrayExecutionOptions::default();
    let mut state = RollingMeanState::new(
        RollingMeanOptions::new(20_000).unwrap(),
        ArrayStateOptions::default(),
    )
    .unwrap();
    state.process(view(&vec![1.0; 10_000]), &execution).unwrap();
    let bytes = state.state_bytes();
    // Checks 5 and 6 visit the existing history, after candidate initialization.
    let result = state.process_checked(view(&[3.0]), &execution, &mut cancel_at(6));
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(state.state_bytes(), bytes);
    assert_eq!(
        bits(&state.process(view(&[3.0]), &execution).unwrap()),
        [Some((1.0_f64 + 2.0 / 10_001.0).to_bits())]
    );
}

#[test]
fn test_owned_mean_cancels_long_m2_repair_in_each_profile_then_retries() {
    let execution = ArrayExecutionOptions::default();
    let prefix = vec![1e155; 8193];
    for (profile, stop) in [
        (RollingNumericalProfile::StableV1, 12),
        (RollingNumericalProfile::StableV2Preview, 13),
    ] {
        let options = RollingMeanOptions::new(9000)
            .unwrap()
            .with_numerical_profile(profile);
        let mut state = RollingMeanState::new(options, ArrayStateOptions::default()).unwrap();
        state.process(view(&prefix), &execution).unwrap();
        let bytes = state.state_bytes();
        assert!(matches!(
            state.process_checked(view(&[-1e155]), &execution, &mut cancel_at(stop)),
            Err(CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(state.state_bytes(), bytes);
        let actual = state.process(view(&[-1e155]), &execution).unwrap();
        let mut full = prefix.clone();
        full.push(-1e155);
        let expected = rolling_mean(FloatView::from_f64(&full), options, &execution).unwrap();
        assert_eq!(bits(&actual)[0], bits(&expected).last().copied().unwrap());
    }
}

#[test]
fn test_owned_mean_peak_includes_stable_fold_scratch() {
    let options = RollingMeanOptions::new(2)
        .unwrap()
        .with_numerical_profile(RollingNumericalProfile::StableV2Preview);
    let mut state = RollingMeanState::new(options, ArrayStateOptions::default()).unwrap();
    let peak = 64 + 80 + 17 + rolling_mean_math::stable_scratch_bytes();
    let execution = ArrayExecutionOptions::default();
    assert!(
        matches!(state.process(view(&[1.0]), &execution.with_max_allocation_bytes(peak - 1)),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "state.peak_bytes")
    );
    assert_eq!(state.state_bytes(), 64);
    assert!(
        state
            .process(view(&[1.0]), &execution.with_max_allocation_bytes(peak))
            .is_ok()
    );
}
