use super::{RESTORE_MEMBERSHIP_READS, RunState, RunStateSnapshot};
use crate::operator::join::{
    JoinSide, JoinStateLimits, JoinTimeBounds,
    run_tests::{eviction_state, fanout_input, output_schema},
};
use crate::{EdgeBudget, EventTime};
use std::sync::Arc;

fn snapshot(rows: usize) -> RunStateSnapshot {
    let mut state = RunState::new(
        "restore",
        JoinStateLimits::new(256, 1_000_000, 256).unwrap(),
        JoinTimeBounds::from_micros(0, 0).unwrap(),
    )
    .unwrap();
    state
        .apply(
            JoinSide::Left,
            fanout_input(rows, 0, true),
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    state.snapshot()
}

#[test]
fn test_restore_validates_live_membership_once_per_locator() {
    let saved = snapshot(128);
    let fingerprint = saved.fingerprint();
    RESTORE_MEMBERSHIP_READS.set(0);
    let restored = RunState::restore(saved).unwrap();
    let reads = RESTORE_MEMBERSHIP_READS.get();
    assert_eq!(restored.rows(JoinSide::Left), 128);
    assert_eq!(restored.next_row_id(JoinSide::Left), 128);
    assert_eq!(restored.snapshot().fingerprint(), fingerprint);
    eprintln!("live membership ordinal reads for 128 restored locators: {reads}");
    assert!((1..=128).contains(&reads));
}

#[test]
fn test_restore_rejects_reordered_equal_time_ordinals() {
    let mut saved = snapshot(3);
    let live = saved.sides[0].runs.values_mut().next().unwrap();
    Arc::get_mut(&mut live.run)
        .expect("snapshot fixture owns the run")
        .order
        .swap(1, 2);
    assert!(RunState::restore(saved).is_err());
}

#[test]
fn test_restore_requires_a_complete_in_range_permutation() {
    for replacement in [0, 3] {
        let mut saved = snapshot(3);
        let live = saved.sides[0].runs.values_mut().next().unwrap();
        Arc::get_mut(&mut live.run)
            .expect("snapshot fixture owns the run")
            .order[1] = replacement;
        assert!(matches!(
            RunState::restore(saved),
            Err(crate::CalcFlowError::CheckpointMismatch { .. })
        ));
    }
}

#[test]
fn test_restore_checks_the_run_head_before_charging_its_prefix() {
    let mut saved = snapshot(3);
    saved.sides[0].runs.values_mut().next().unwrap().head = 4;
    assert!(RunState::restore(saved).is_err());
}

#[test]
fn test_restore_rejects_an_expired_row_in_both_active_indexes() {
    let mut state = eviction_state();
    let evicted = state
        .evict(JoinSide::Left, 0, EventTime::from_micros(11))
        .unwrap();
    assert_eq!(evicted.expired_rows, 1);
    let mut saved = state.snapshot();
    let side = &mut saved.sides[0];
    let expired = side.expiration[side.expiration_head - 1];
    let bucket = side.buckets.get_mut(&expired.group).unwrap();
    bucket.entries[bucket.head] = expired;
    side.expiration[side.expiration_head] = expired;
    assert!(
        matches!(RunState::restore(saved), Err(crate::CalcFlowError::InvalidArgument { message, .. }) if message == "restored key locator differs from its run")
    );
}
