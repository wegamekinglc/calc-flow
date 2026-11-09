use crate::CalcFlowError;
use crate::operator::rolling::{ExtremaAccumulator, ExtremaKey};
use datafusion::common::ScalarValue;
fn key(position: i64) -> ExtremaKey {
    ExtremaKey {
        event_time: position,
        sequence: vec![],
    }
}
#[test]
fn test_restore_extrema_dominated_refold_checks_each_pop() {
    let mut accumulator = ExtremaAccumulator::new(false);
    for index in 0..8_193 {
        accumulator.add(key(index), ScalarValue::Int64(Some(index)));
    }
    let mut calls = 0;
    let error = accumulator
        .add_checked(key(8_194), ScalarValue::Int64(Some(-1)), &mut || {
            calls += 1;
            if calls == 4_096 {
                return Err(CalcFlowError::Cancelled {
                    run_id: "restore".into(),
                });
            }
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(calls, 4_096);
}
#[test]
fn test_restore_extrema_expiration_checks_each_pop() {
    for by_key in [false, true] {
        let mut accumulator = ExtremaAccumulator::new(false);
        for index in 0..8_193 {
            accumulator.add(key(index), ScalarValue::Int64(Some(index)));
        }
        let mut calls = 0;
        let mut check = || {
            calls += 1;
            if calls == 4_096 {
                return Err(CalcFlowError::Cancelled {
                    run_id: "restore".into(),
                });
            }
            Ok(())
        };
        let result = if by_key {
            accumulator.expire_through_key_checked(&key(8_194), &mut check)
        } else {
            accumulator.expire_through_time_checked(8_194, &mut check)
        };
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
        assert_eq!(calls, 4_096);
    }
}
