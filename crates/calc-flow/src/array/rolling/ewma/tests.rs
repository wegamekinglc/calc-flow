use crate::array::rolling::next_ewma;

#[test]
fn test_ewma_transition_checks_count_without_advancing_missing_samples() {
    assert!(next_ewma(u64::MAX, 7.0, Some(8.0), 0.5, false).is_none());
    let (count, value) = next_ewma(u64::MAX, -0.0, None, 0.5, false).unwrap();
    assert_eq!(count, u64::MAX);
    assert_eq!(value.to_bits(), (-0.0_f64).to_bits());
}

#[test]
fn test_ewma_transition_preserves_seeding_and_span_one_bits() {
    let (count, value) = next_ewma(0, 0.0, Some(-0.0), 0.5, false).unwrap();
    assert_eq!(count, 1);
    assert_eq!(value.to_bits(), (-0.0_f64).to_bits());
    let (_, value) = next_ewma(1, f64::INFINITY, Some(2.0), 1.0, false).unwrap();
    assert_eq!(value.to_bits(), 2.0_f64.to_bits());
}

#[test]
fn test_shared_cumulative_transition_keeps_overflow_repair_and_infinity_rules() {
    let (_, mean) = next_ewma(1, 1.0e308, Some(-1.0e308), 0.0, true).unwrap();
    assert_eq!(mean.to_bits(), 0.0_f64.to_bits());
    let (_, mean) = next_ewma(1, f64::INFINITY, Some(3.0), 0.0, true).unwrap();
    assert_eq!(mean.to_bits(), f64::INFINITY.to_bits());
    let (_, mean) = next_ewma(1, 3.0, Some(f64::INFINITY), 0.0, true).unwrap();
    assert_eq!(mean.to_bits(), f64::INFINITY.to_bits());
    let (_, mean) = next_ewma(1, f64::INFINITY, Some(f64::NEG_INFINITY), 0.0, true).unwrap();
    assert!(mean.is_nan());
}
