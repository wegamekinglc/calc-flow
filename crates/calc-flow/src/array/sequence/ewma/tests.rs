use super::*;
use crate::{
    CalcFlowError,
    array::{FloatValues, FloatView},
};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}

fn last(output: &NativeArray) -> f64 {
    let FloatValues::Float64(values) = output.float_view().unwrap().values() else {
        panic!("expected f64")
    };
    *values.last().unwrap()
}

#[test]
fn test_owned_ewma_checked_position_and_valid_count_rollback() {
    let execution = ArrayExecutionOptions::default();
    let mut state =
        EwmaState::new(EwmaOptions::new(3).unwrap(), ArrayStateOptions::default()).unwrap();
    state.process(view(&[1.0]), &execution).unwrap();
    state.positions = u64::MAX - 1;
    assert!(
        matches!(state.process(view(&[3.0, 5.0]), &execution), Err(CalcFlowError::InvalidArgument { field, .. }) if field == "input.rows")
    );
    assert_eq!(
        last(&state.process(view(&[3.0]), &execution).unwrap()).to_bits(),
        2.0_f64.to_bits()
    );
    assert!(state.process(view(&[]), &execution).is_ok());
    assert!(state.process(view(&[5.0]), &execution).is_err());
    state.positions = 1;
    state.kernel.set_test_valid_count(u64::MAX);
    assert!(state.process(view(&[5.0]), &execution).is_err());
    // A skipped sample may still advance positions at the maximum valid count.
    assert_eq!(
        last(&state.process(view(&[f64::NAN]), &execution).unwrap()).to_bits(),
        2.0_f64.to_bits()
    );
}

#[test]
fn test_owned_ewma_final_cancellation_rolls_back_finished_output() {
    let execution = ArrayExecutionOptions::default();
    let mut state =
        EwmaState::new(EwmaOptions::new(3).unwrap(), ArrayStateOptions::default()).unwrap();
    state.process(view(&[1.0]), &execution).unwrap();
    let mut checks = 0;
    let error = state.process_checked(view(&[3.0]), &execution, &mut || {
        checks += 1;
        if checks == 4 {
            Err(CalcFlowError::Cancelled {
                run_id: "final".into(),
            })
        } else {
            Ok(())
        }
    });
    assert!(matches!(error, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(
        last(&state.process(view(&[3.0]), &execution).unwrap()).to_bits(),
        2.0_f64.to_bits()
    );
}
