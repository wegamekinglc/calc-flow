use super::*;
use crate::CalcFlowError;
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
