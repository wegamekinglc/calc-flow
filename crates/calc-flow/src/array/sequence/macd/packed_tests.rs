use super::*;
use crate::{CalcFlowError, array::FloatView};
use datafusion::arrow::array::{Array, Float64Array};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}
#[test]
fn test_packed_candidate_writes_one_row_major_output_at_exact_peak() -> Result<()> {
    let state = MacdState::new(MacdOptions::new(1, 3, 3)?, ArrayStateOptions::default())?;
    let execution = ArrayExecutionOptions::default()
        .with_max_elements(3)
        .with_max_allocation_bytes(153);
    assert!(
        state
            .prepare_packed_checked(
                view(&[10.0]),
                &execution.with_max_allocation_bytes(152),
                0,
                &mut || Ok(())
            )
            .is_err()
    );
    let prepared = state.prepare_packed_checked(view(&[10.0]), &execution, 0, &mut || Ok(()))?;
    assert_eq!(prepared.output.shape(), &[1, 3]);
    assert_eq!(prepared.output.element_count(), 3);
    let array = prepared
        .output
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(array.values(), &[0.0, 0.0, 0.0]);
    assert!(array.nulls().is_some());
    assert_eq!(state.kernel.positions, 0);
    assert_eq!(prepared.candidate.unwrap().kernel.positions, 1);
    Ok(())
}
#[test]
fn test_packed_warmup_masks_and_empty_shape_do_not_require_three_staging_arrays() -> Result<()> {
    let state = warmup_state()?;
    let prepared = state.prepare_packed_checked(
        view(&[10.0, 14.0]),
        &ArrayExecutionOptions::default(),
        0,
        &mut || Ok(()),
    )?;
    let array = prepared
        .output
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(array.values(), &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    assert_eq!(
        (0..6).map(|i| array.is_valid(i)).collect::<Vec<_>>(),
        [false, false, false, true, false, false]
    );
    assert_empty(&state)
}
fn warmup_state() -> Result<MacdState> {
    MacdState::new(
        MacdOptions::new(1, 3, 3)?
            .with_min_periods(2)?
            .with_signal_min_periods(2)?,
        ArrayStateOptions::default(),
    )
}
fn assert_empty(state: &MacdState) -> Result<()> {
    let empty = state.prepare_packed_checked(
        view(&[]),
        &ArrayExecutionOptions::default()
            .with_max_elements(0)
            .with_max_allocation_bytes(72),
        0,
        &mut || Ok(()),
    )?;
    assert_eq!(empty.output.shape(), &[0, 3]);
    assert!(empty.candidate.is_none());
    Ok(())
}
#[test]
fn test_packed_candidate_cancellation_never_commits_partial_signal() -> Result<()> {
    let state = MacdState::new(MacdOptions::default(), ArrayStateOptions::default())?;
    let mut calls = 0;
    let result = state.prepare_packed_checked(
        view(&vec![1.0; 20_000]),
        &ArrayExecutionOptions::default(),
        0,
        &mut || {
            calls += 1;
            if calls == 5 {
                Err(CalcFlowError::Cancelled {
                    run_id: "packed".into(),
                })
            } else {
                Ok(())
            }
        },
    );
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!((state.kernel.positions, state.kernel.signal_count), (0, 0));
    Ok(())
}
