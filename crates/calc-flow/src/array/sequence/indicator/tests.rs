use crate::{
    CalcFlowError, Result,
    array::{
        ArrayExecutionOptions, ArrayStateOptions, ArrayView, AtrOptions, AtrState, FloatView,
        RsiOptions, RsiState,
    },
};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}

#[test]
fn test_controlled_eager_calls_cancel_in_long_chunks_and_before_empty_publish() {
    let values = vec![1.0; 20_000];
    let execution = ArrayExecutionOptions::default();
    let mut calls = 0;
    let mut check = || {
        calls += 1;
        if calls == 5 {
            Err(CalcFlowError::Cancelled {
                run_id: "indicator-test".to_owned(),
            })
        } else {
            Ok(())
        }
    };
    let error = super::rsi::rsi_checked(
        view(&values),
        RsiOptions::new(2).unwrap(),
        &execution,
        &mut check,
    )
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(calls, 5);
    let mut calls = 0;
    let mut check = || {
        calls += 1;
        if calls == 2 {
            Err(CalcFlowError::Cancelled {
                run_id: "indicator-test".to_owned(),
            })
        } else {
            Ok(())
        }
    };
    let error = super::atr::atr_checked(
        [view(&[]); 3],
        AtrOptions::new(2).unwrap(),
        &execution,
        &mut check,
    )
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(calls, 2);
}

#[test]
fn test_prepared_cancellation_and_position_overflow_preserve_committed_bits() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    let mut state = RsiState::new(RsiOptions::new(2)?, ArrayStateOptions::default())?;
    state.process(view(&[1.0, 2.0, 0.0]), &execution)?;
    let before = state.kernel;
    let values = vec![1.0; 20_000];
    let mut calls = 0;
    let result = state.prepare_checked(view(&values), &execution, 0, &mut || {
        calls += 1;
        if calls == 5 {
            Err(CalcFlowError::Cancelled {
                run_id: "indicator-test".to_owned(),
            })
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(state.kernel.positions, before.positions);
    assert_eq!(state.kernel.mean_gain.to_bits(), before.mean_gain.to_bits());
    assert_eq!(state.kernel.mean_loss.to_bits(), before.mean_loss.to_bits());
    Ok(())
}

#[test]
fn test_position_exhaustion_rejects_nonempty_but_admits_empty_chunks() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    let mut state = RsiState::new(RsiOptions::new(2)?, ArrayStateOptions::default())?;
    state.kernel.positions = u64::MAX;
    assert!(state.process(view(&[1.0]), &execution).is_err());
    state.process(view(&[]), &execution)?;
    assert_eq!(state.kernel.positions, u64::MAX);
    let mut atr = AtrState::new(AtrOptions::new(1)?, ArrayStateOptions::default())?;
    atr.kernel.positions = u64::MAX;
    assert!(
        atr.process(view(&[1.0]), view(&[0.0]), view(&[1.0]), &execution)
            .is_err()
    );
    atr.process(view(&[]), view(&[]), view(&[]), &execution)?;
    assert_eq!(atr.kernel.positions, u64::MAX);
    Ok(())
}
