use super::{Encoding, RightBucket, State};
use crate::{EventTime, StreamAsofJoinStatus};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default)]
struct Visits {
    preview: usize,
    dictionary: usize,
    minima: usize,
    compaction: usize,
}

thread_local! {
    static VISITS: Cell<Visits> = const { Cell::new(Visits {
        preview: 0, dictionary: 0, minima: 0, compaction: 0,
    }) };
}

pub(super) fn record_preview() {
    VISITS.with(|counter| {
        let mut visits = counter.get();
        visits.preview += 1;
        counter.set(visits);
    });
}

pub(super) fn record_dictionary() {
    VISITS.with(|counter| {
        let mut visits = counter.get();
        visits.dictionary += 1;
        counter.set(visits);
    });
}

pub(super) fn record_minima() {
    VISITS.with(|counter| {
        let mut visits = counter.get();
        visits.minima += 1;
        counter.set(visits);
    });
}

pub(super) fn record_compaction() {
    VISITS.with(|counter| {
        let mut visits = counter.get();
        visits.compaction += 1;
        counter.set(visits);
    });
}

pub(super) fn take_compaction_visits() -> usize {
    VISITS.with(|visits| visits.replace(Visits::default()).compaction)
}

fn sparse_sweep(keys: u32) {
    let mut state = State::default();
    for key in 0..keys {
        let time = if key == 0 { 0 } else { 100 };
        state.right.insert(
            Encoding::from_slice(&key.to_le_bytes()),
            RightBucket::from_iter([((time, Encoding::from_slice(&[1])), None)]),
        );
    }
    let mut status = StreamAsofJoinStatus::default();
    status.left.watermark_micros = Some(EventTime::from_micros(1));
    status.right.watermark_micros = status.left.watermark_micros;
    VISITS.with(|visits| visits.set(Visits::default()));
    let preview = state.preview_eviction(&status, 0, "asof").unwrap();
    let compaction = state.batches.prepare_removal(&preview.batches);
    let dictionary = state
        .right
        .prepare_fixture_compaction(state.right.len() - preview.projected_right.0);
    state.evict_prepared(&status, 0, compaction, dictionary, &preview);
    let visits = VISITS.with(|visits| visits.replace(Visits::default()));
    assert_eq!(preview.selected.len(), 1);
    assert_eq!(preview.removed_identities, 1);
    assert_eq!(state.right.len(), keys as usize - 1);
    assert_eq!(state.right_identity_min, Some(100));
    assert!(
        visits.preview <= 1
            && visits.dictionary <= 2
            && visits.minima <= 2
            && visits.compaction == 0,
        "one-key expiration must avoid whole-dictionary traversal: {visits:?}"
    );
}

#[test]
fn test_asof_sparse_expiration_visits_4096_keys_locally() {
    sparse_sweep(4_096);
}

#[test]
fn test_asof_sparse_expiration_visits_65536_keys_locally() {
    sparse_sweep(65_536);
}

#[test]
fn test_asof_zero_due_expiration_does_no_dictionary_work() {
    let mut state = State::default();
    for key in 0_u32..4_096 {
        state.right.insert(
            Encoding::from_slice(&key.to_le_bytes()),
            RightBucket::from_iter([((100, Encoding::from_slice(&[1])), None)]),
        );
    }
    let mut status = StreamAsofJoinStatus::default();
    status.right.watermark_micros = Some(EventTime::from_micros(100));
    VISITS.with(|visits| visits.set(Visits::default()));
    let preview = state.preview_eviction(&status, 0, "asof").unwrap();
    let compaction = state.batches.prepare_removal(&preview.batches);
    let dictionary = state
        .right
        .prepare_fixture_compaction(state.right.len() - preview.projected_right.0);
    state.evict_prepared(&status, 0, compaction, dictionary, &preview);
    let visits = VISITS.with(|visits| visits.replace(Visits::default()));
    assert!(preview.selected.is_empty());
    assert_eq!(state.right.len(), 4_096);
    assert_eq!(state.right_identity_min, Some(100));
    assert_eq!(visits.preview, 0);
    assert_eq!(visits.dictionary, 0);
    assert_eq!(visits.minima, 0);
    assert_eq!(visits.compaction, 0);
}

#[test]
fn test_asof_all_due_expiration_releases_index_capacity() {
    let mut state = State::default();
    for key in 0_u32..4_096 {
        state.right.insert(
            Encoding::from_slice(&key.to_le_bytes()),
            RightBucket::from_iter([((100, Encoding::from_slice(&[1])), None)]),
        );
    }
    let mut status = StreamAsofJoinStatus::default();
    status.right.ended = true;
    VISITS.with(|visits| visits.set(Visits::default()));
    let preview = state.preview_eviction(&status, 0, "asof").unwrap();
    let compaction = state.batches.prepare_removal(&preview.batches);
    let dictionary = state
        .right
        .prepare_fixture_compaction(state.right.len() - preview.projected_right.0);
    state.evict_prepared(&status, 0, compaction, dictionary, &preview);
    let visits = VISITS.with(|visits| visits.replace(Visits::default()));
    assert_eq!(preview.selected.len(), 4_096);
    assert_eq!(preview.removed_identities, 4_096);
    assert!(state.right.is_empty());
    assert_eq!(state.right.checkpoint_capacities(), [0, 0]);
    assert_eq!(state.right.heap_capacities(), [0, 0]);
    assert_eq!(state.right_identity_min, None);
    assert_eq!(visits.preview, 4_096);
    assert_eq!(visits.dictionary, 4_096);
    assert_eq!(visits.compaction, 0);
}
