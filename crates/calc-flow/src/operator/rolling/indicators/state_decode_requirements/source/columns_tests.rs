use super::*;
use crate::CalcFlowError;
#[test]
fn test_partition_lookup_checks_every_borrowed_column_and_preserves_cancel() {
    let mut checks = 0;
    let error = contains(std::iter::repeat_n(7, 8_193), 8, &mut || {
        checks += 1;
        if checks == 4_096 {
            return Err(CalcFlowError::Cancelled {
                run_id: "columns".into(),
            });
        }
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(checks, 4_096);
}
#[test]
fn test_partition_lookup_borrows_without_allocation() {
    let mut result = false;
    let measured = allocation_counter::measure(|| {
        result = contains([1, 2, 3].into_iter(), 2, &mut || Ok(())).unwrap();
    });
    assert!(result);
    assert_eq!(measured.count_total, 0);
}
