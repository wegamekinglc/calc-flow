use super::*;
use crate::array::{EwmaOptions, FloatValues, FloatView, RollingMeanOptions};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}

fn last(output: &NativeArray) -> u64 {
    let FloatValues::Float64(values) = output.float_view().unwrap().values() else {
        panic!("expected f64")
    };
    values.last().unwrap().to_bits()
}

#[test]
fn test_private_candidates_can_be_discarded_without_advancing_owned_states() {
    let execution = ArrayExecutionOptions::default();
    let mut mean = RollingMeanState::new(
        RollingMeanOptions::new(2).unwrap(),
        ArrayStateOptions::default(),
    )
    .unwrap();
    let mut ewma =
        EwmaState::new(EwmaOptions::new(3).unwrap(), ArrayStateOptions::default()).unwrap();
    mean.process(view(&[1.0]), &execution).unwrap();
    ewma.process(view(&[1.0]), &execution).unwrap();
    let prepared = mean
        .prepare_checked(view(&[3.0]), &execution, 0, &mut || Ok(()))
        .unwrap();
    assert_eq!(last(&prepared.output), 2.0_f64.to_bits());
    assert_eq!(mean.state_bytes(), 80);
    drop(prepared);
    let prepared = ewma
        .prepare_checked(view(&[3.0]), &execution, 0, &mut || Ok(()))
        .unwrap();
    assert_eq!(last(&prepared.output), 2.0_f64.to_bits());
    drop(prepared);
    assert_eq!(
        last(&mean.process(view(&[5.0]), &execution).unwrap()),
        3.0_f64.to_bits()
    );
    assert_eq!(
        last(&ewma.process(view(&[5.0]), &execution).unwrap()),
        3.0_f64.to_bits()
    );
}

#[test]
fn test_private_candidate_admission_includes_retained_captures_and_empty_avoids_copy() {
    let state = EwmaState::new(EwmaOptions::new(3).unwrap(), ArrayStateOptions::default()).unwrap();
    let peak = 24 + 24 + 17 + 100;
    let execution = ArrayExecutionOptions::default();
    assert!(
        state
            .prepare_checked(
                view(&[1.0]),
                &execution.with_max_allocation_bytes(peak - 1),
                100,
                &mut || Ok(())
            )
            .is_err()
    );
    let prepared = state
        .prepare_checked(
            view(&[1.0]),
            &execution.with_max_allocation_bytes(peak),
            100,
            &mut || Ok(()),
        )
        .unwrap();
    assert_eq!(last(&prepared.output), 1.0_f64.to_bits());
    assert!(prepared.candidate.is_some());
    let prepared = state
        .prepare_checked(
            view(&[]),
            &execution.with_max_allocation_bytes(24 + 8 + 100),
            100,
            &mut || Ok(()),
        )
        .unwrap();
    assert!(prepared.candidate.is_none());
    let mut mean = RollingMeanState::new(
        RollingMeanOptions::new(u64::MAX).unwrap(),
        ArrayStateOptions::default(),
    )
    .unwrap();
    mean.process(view(&[1.0, 3.0]), &execution).unwrap();
    let prepared = mean
        .prepare_checked(
            view(&[]),
            &execution.with_max_allocation_bytes(mean.state_bytes() + 8 + 100),
            100,
            &mut || Ok(()),
        )
        .unwrap();
    assert!(prepared.candidate.is_none());
}
