use super::*;
use crate::CalcFlowError;
#[test]
fn test_checked_rebuild_fold_preserves_cancellation_after_row_boundary() {
    let mut calls = 0;
    let error = fold(
        0_usize..8_193,
        &mut || {
            calls += 1;
            if calls == 4_096 {
                return Err(CalcFlowError::Cancelled {
                    run_id: "restore".into(),
                });
            }
            Ok(())
        },
        |rows: &mut dyn Iterator<Item = usize>| Ok(rows.count()),
    )
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(calls, 4_096);
}
#[test]
fn test_checked_projection_visits_filtered_columns_without_allocating_them() {
    let mut calls = 0;
    let projection = projection(std::iter::repeat_n(None, 8_193), &mut || {
        calls += 1;
        Ok::<_, CalcFlowError>(())
    })
    .unwrap();
    assert_eq!(calls, 8_193);
    assert!(projection.is_empty());
}
