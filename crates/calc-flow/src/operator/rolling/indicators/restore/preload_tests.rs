use super::state_tests;
use crate::operator::rolling::{IndicatorRollingRecord, RollingOperator};
use crate::pipeline::OperatorCheckpointCapability;
use crate::{CancellationToken, JsonMap, OperatorStateSnapshot, StateBudget, StreamJobContext};
use serde_json::json;
use std::sync::Arc;

fn fixture(
    empty: bool,
) -> (
    RollingOperator,
    IndicatorRollingRecord,
    OperatorStateSnapshot,
) {
    let operator = state_tests::operator();
    let mut snapshot = state_tests::snapshot(&operator);
    if empty {
        snapshot.segments.clear();
        snapshot
            .inline_metadata
            .insert("segment_inventory".into(), json!([]));
        snapshot
            .inline_metadata
            .insert("next_output_sequence".into(), json!(0));
    }
    let capability = OperatorCheckpointCapability::CheckpointedStateful { state_version: 5 };
    let snapshot = capability.encode_snapshot("features", snapshot).unwrap();
    let job = StreamJobContext::new(
        41,
        "f".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let record = operator
        .native_checkpoint_record(
            "features",
            job.fingerprint().into(),
            capability,
            Arc::new(job.native_control()),
        )
        .unwrap();
    (operator, record, snapshot)
}

#[test]
fn test_native_preloaded_metadata_receipt_survives_until_last_snapshot_owner() {
    let (operator, record, snapshot) = fixture(true);
    let resources = operator.indicator_resources.as_ref().unwrap();
    let initial = resources.begin_work().source().bytes("features").unwrap();
    let small = StateBudget::new(32, initial).unwrap();
    let copied = record
        .preloaded_snapshot(&snapshot.inline_metadata, None)
        .unwrap();
    assert_eq!(copied.inline_metadata, snapshot.inline_metadata);
    assert!(copied.clone_legacy().is_err());
    let first = Arc::new(copied);
    let last = Arc::clone(&first);
    assert!(resources.check(0, small, "features").is_err());
    drop(first);
    assert!(resources.check(0, small, "features").is_err());
    drop(last);
    resources.check(0, small, "features").unwrap();
}

#[test]
fn test_native_preloaded_snapshot_moves_one_funded_segment_without_payload_copy() {
    let (operator, record, mut snapshot) = fixture(false);
    let (id, segment) = snapshot.segments.pop_first().unwrap();
    let data = segment.bytes().as_ptr();
    let resources = operator.indicator_resources.as_ref().unwrap();
    let segment = resources
        .retain(segment, 0, operator.state_budget, "features")
        .unwrap();
    let copied = record
        .preloaded_snapshot(&snapshot.inline_metadata, Some((&id, segment)))
        .unwrap();
    assert_eq!(copied.segments.len(), 1);
    assert_eq!(copied.segments[&id].bytes().as_ptr(), data);
    assert_eq!(copied.inline_metadata, snapshot.inline_metadata);
}
