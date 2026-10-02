use super::*;

fn legacy_snapshot(fixture_json: &str) -> (StreamAsofJoinOperator, crate::OperatorStateSnapshot) {
    let value: serde_json::Value = serde_json::from_str(fixture_json).unwrap();
    let (template, _) = if value["schema_kind"] == "utf8" {
        shared_string_identity_fixture()
    } else {
        fixture()
    };
    let spec = serde_json::from_value(value["spec"].clone()).unwrap();
    let operator = StreamAsofJoinOperator::new(
        "asof",
        template.schemas[0].clone(),
        template.schemas[1].clone(),
        spec,
    )
    .unwrap();
    let segments = value["segments"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, segment)| {
            let encoded = segment["hex"].as_str().unwrap();
            let mut bytes = vec![0; encoded.len() / 2];
            hex::decode_to_slice(encoded, &mut bytes).unwrap();
            let segment = crate::StateSegment::new(bytes);
            assert_eq!(segment.sha256(), value["segments"][name]["sha256"]);
            (name.clone(), segment)
        })
        .collect();
    let snapshot = crate::OperatorStateSnapshot {
        inline_metadata: serde_json::from_value(value["metadata"].clone()).unwrap(),
        segments,
    };
    (operator, snapshot)
}

fn assert_legacy_migration(fixture: &str) {
    let (mut operator, snapshot) = legacy_snapshot(fixture);
    let previous_rows = snapshot.inline_metadata["metrics"]["state_rows"].clone();
    let previous_bytes = snapshot.inline_metadata["metrics"]["state_bytes"]
        .as_u64()
        .unwrap();
    operator.restore(&snapshot).unwrap();
    assert_eq!(serde_json::json!(operator.status.state_rows), previous_rows);
    let expected = operator.state.right.auxiliary_bytes();
    assert!(expected > 0);
    assert_eq!(operator.runtime.pool.reserved(), expected);
    assert_eq!(
        operator.status.state_bytes,
        previous_bytes + expected as u64 + 16
    );
    let current = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        current.inline_metadata["layout_version"],
        serde_json::json!(4)
    );
    assert_eq!(
        current.inline_metadata["accounting_version"],
        serde_json::json!(4)
    );
    assert!(current.segments.contains_key("asof-index-v4"));
    operator.restore(&current).unwrap();
    assert_eq!(operator.runtime.pool.reserved(), expected);
    let repeated = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(repeated.inline_metadata, current.inline_metadata);
    assert_eq!(repeated.segments, current.segments);
}

#[test]
fn test_a03_legacy_integer_snapshot_migrates_inventory_once() {
    assert_legacy_migration(include_str!("fixtures/legacy-v3/populated.json"));
}

#[test]
fn test_a03_legacy_general_spare_snapshot_migrates_inventory_once() {
    assert_legacy_migration(include_str!("fixtures/legacy-v3/general-spare-shared.json"));
}

#[test]
fn test_a03_legacy_empty_and_terminal_snapshots_preserve_lifecycle() {
    for fixture in [
        include_str!("fixtures/legacy-v3/empty.json"),
        include_str!("fixtures/legacy-v3/terminal.json"),
    ] {
        let (mut operator, previous) = legacy_snapshot(fixture);
        operator.restore(&previous).unwrap();
        assert_eq!(operator.runtime.pool.reserved(), 0);
        let current = operator.capture(Epoch::INITIAL).unwrap();
        assert!(current.segments.is_empty());
        assert_eq!(
            current.inline_metadata["layout_version"],
            serde_json::json!(4)
        );
        assert_eq!(
            current.inline_metadata["metrics"],
            previous.inline_metadata["metrics"]
        );
        assert_eq!(
            current.inline_metadata["terminal"],
            previous.inline_metadata["terminal"]
        );
    }
}

#[test]
fn test_a03_mixed_and_unknown_snapshot_versions_fail_before_allocation() {
    let (mut operator, snapshot) =
        legacy_snapshot(include_str!("fixtures/legacy-v3/populated.json"));
    let before = operator.status();
    for (state, layout, accounting) in [(3, 3, 4), (3, 4, 3), (3, 5, 5), (4, 4, 4)] {
        let mut invalid = snapshot.clone();
        invalid
            .inline_metadata
            .insert("state_version".into(), serde_json::json!(state));
        invalid
            .inline_metadata
            .insert("layout_version".into(), serde_json::json!(layout));
        invalid
            .inline_metadata
            .insert("accounting_version".into(), serde_json::json!(accounting));
        assert!(operator.restore(&invalid).is_err());
        assert_eq!(operator.status(), before);
        assert_eq!(operator.runtime.pool.reserved(), 0);
    }
}

#[test]
fn test_a03_guard_preparation_workspace_failure_releases_auxiliary_growth() {
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

    let state = state::RightState::default();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(40));
    let result = state.prepare_storage(&[(state::Encoding::from_slice(b"key"), 1)], &pool, "asof");
    assert!(matches!(
        result,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    ));
    assert_eq!(pool.reserved(), 0);
    assert_eq!(state.checkpoint_capacities(), [0, 0]);
    assert_eq!(state.heap_capacities(), [0, 0]);
}

#[test]
fn test_a03_guard_dropped_preparation_preserves_installed_lease() {
    let (mut operator, snapshot) =
        legacy_snapshot(include_str!("fixtures/legacy-v3/populated.json"));
    operator.restore(&snapshot).unwrap();
    let before = operator.status();
    let capacities = operator.state.right.checkpoint_capacities();
    let expected = operator.state.right.auxiliary_bytes();
    let prepared = operator
        .state
        .right
        .prepare_storage(
            &[(state::Encoding::from_slice(b"new-key"), 1)],
            &operator.runtime.pool,
            "asof",
        )
        .unwrap();
    assert!(operator.runtime.pool.reserved() > expected);
    assert_eq!(operator.status(), before);
    assert_eq!(operator.state.right.checkpoint_capacities(), capacities);
    drop(prepared);
    assert_eq!(operator.runtime.pool.reserved(), expected);
    let pool = operator.runtime.pool.clone();
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_a03_guard_failed_restore_preserves_live_state_and_lease() {
    let (mut operator, legacy) = legacy_snapshot(include_str!("fixtures/legacy-v3/populated.json"));
    operator.restore(&legacy).unwrap();
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    let before = operator.status();
    let expected = operator.state.right.auxiliary_bytes();
    let mut invalid = snapshot.clone();
    invalid.inline_metadata.get_mut("metrics").unwrap()["state_bytes"] =
        serde_json::json!(before.state_bytes + 1);
    assert!(matches!(
        operator.restore(&invalid),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(operator.status(), before);
    assert_eq!(operator.runtime.pool.reserved(), expected);
    assert_eq!(
        operator.capture(Epoch::INITIAL).unwrap().segments,
        snapshot.segments
    );
}

#[test]
fn test_a03_guard_progress_rejection_preserves_installed_lease() {
    let (mut operator, legacy) = legacy_snapshot(include_str!("fixtures/legacy-v3/populated.json"));
    operator.restore(&legacy).unwrap();
    let current = operator.capture(Epoch::INITIAL).unwrap();
    let before = operator.status();
    let expected = operator.state.right.auxiliary_bytes();
    for snapshot in [&legacy, &current] {
        assert!(matches!(
            operator.restore_with_progress(snapshot, &asymmetric_progress(150, 150), None),
            Err(CalcFlowError::CheckpointMismatch { .. })
        ));
        assert_eq!(operator.status(), before);
        assert_eq!(operator.runtime.pool.reserved(), expected);
        let repeated = operator.capture(Epoch::INITIAL).unwrap();
        assert_eq!(repeated.inline_metadata, current.inline_metadata);
        assert_eq!(repeated.segments, current.segments);
    }
}

#[tokio::test]
async fn test_a03_guard_end_releases_all_expiration_storage() {
    let (mut operator, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", input, &context, &mut output)
        .await
        .unwrap();
    assert!(operator.runtime.pool.reserved() > 0);
    operator.on_end(&context, &mut output).await.unwrap();
    assert_eq!(operator.status.state_rows, 0);
    assert_eq!(operator.status.state_bytes, 0);
    assert_eq!(operator.state.right.checkpoint_capacities(), [0, 0]);
    assert_eq!(operator.state.right.heap_capacities(), [0, 0]);
    assert_eq!(operator.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn test_a03_guard_tight_legacy_migration_preserves_existing_state() {
    let (mut operator, legacy) =
        legacy_snapshot(include_str!("fixtures/legacy-v3/tight-state-cap.json"));
    operator.runtime.pool = Arc::new(datafusion::execution::memory_pool::GreedyMemoryPool::new(
        1 << 20,
    ));
    let (_, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", input, &context, &mut output)
        .await
        .unwrap();
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    let before = operator.status();
    let expected = operator.state.right.auxiliary_bytes();
    assert_eq!(
        legacy.inline_metadata["metrics"]["state_bytes"],
        serde_json::json!(7_127)
    );
    assert_eq!(operator.spec.limits().max_state_bytes(), 7_127);
    assert!(matches!(
        operator.restore(&legacy),
        Err(CalcFlowError::CheckpointMismatch { message })
            if message == "ASOF restored state with expiration index exceeds limits.max_state_bytes"
    ));
    assert_eq!(operator.status(), before);
    assert_eq!(operator.runtime.pool.reserved(), expected);
    let repeated = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(repeated.inline_metadata, snapshot.inline_metadata);
    assert_eq!(repeated.segments, snapshot.segments);
}

#[tokio::test]
async fn test_a03_expiration_index_keeps_live_capacity_funded() {
    let (mut operator, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", input, &context, &mut output)
        .await
        .unwrap();
    let dictionary = operator.state.right.checkpoint_capacities()[0];
    let heaps = operator.state.right.heap_capacities();
    let expected = dictionary * 8 + heaps.into_iter().sum::<usize>() * 16;
    assert!(expected > 0);
    assert_eq!(
        operator.runtime.pool.reserved(),
        expected,
        "installed queue and inverse storage must retain its lifetime reservation"
    );
    operator.reset().unwrap();
    assert_eq!(operator.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn test_a03_current_layout_restore_keeps_live_capacity_funded() {
    let (mut operator, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", input, &context, &mut output)
        .await
        .unwrap();
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    let (mut restored, _) = fixture();
    restored.restore(&snapshot).unwrap();
    let expected = restored.state.right.auxiliary_bytes();
    assert!(expected > 0);
    assert_eq!(restored.runtime.pool.reserved(), expected);
    let pool = restored.runtime.pool.clone();
    drop(restored);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_a03_cold_compaction_releases_sparse_dictionary_and_queue_capacity() {
    let (mut operator, _) = fixture();
    let keys = (0..64).map(|key| format!("key-{key}")).collect::<Vec<_>>();
    let record = RecordBatch::try_new(
        operator.schemas[1].clone(),
        vec![
            Arc::new(StringArray::from_iter_values(
                keys.iter().map(String::as_str),
            )),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(
                    (0..64).map(|key| if key == 0 { 100 } else { 0 }),
                )
                .with_timezone("UTC"),
            ),
            Arc::new(Int64Array::from_iter_values(0..64)),
        ],
    )
    .unwrap();
    let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", input, &context, &mut output)
        .await
        .unwrap();
    let progress = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        None,
        asymmetric_progress(200, 1),
    );
    operator
        .on_ingress_progress_with_output("right", &progress, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.state.right.len(), 1);
    assert_eq!(operator.status.identity_only_rows, 1);
    assert_eq!(operator.status.retained_right_rows, 0);
    assert!(operator.state.right.checkpoint_capacities()[0] <= 4);
    assert!(
        operator
            .state
            .right
            .heap_capacities()
            .into_iter()
            .all(|capacity| capacity <= 4)
    );
    assert_eq!(
        operator.runtime.pool.reserved(),
        operator.state.right.auxiliary_bytes()
    );
    assert_eq!(
        operator.status.state_bytes,
        operator.current_inventory(None).unwrap().bytes
            + operator.deferred_index_len.unwrap()
            + 256,
    );
    operator.reset().unwrap();
    assert_eq!(operator.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn test_a03_expiration_index_captures_distinct_accounting_layout() {
    let (mut operator, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", input, &context, &mut output)
        .await
        .unwrap();
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        snapshot.inline_metadata["state_version"],
        serde_json::json!(3)
    );
    assert_eq!(
        snapshot.inline_metadata["layout_version"],
        serde_json::json!(4)
    );
    assert_eq!(
        snapshot.inline_metadata["accounting_version"],
        serde_json::json!(4)
    );
    assert_eq!(
        &snapshot.segments["asof-index-v4"].bytes()[..8],
        b"CFASOF04"
    );
    assert!(!snapshot.segments.contains_key("asof-index-v3"));
    let bytes = snapshot.segments["asof-index-v4"].bytes();
    let declared = [
        u64::from_le_bytes(bytes[72..80].try_into().unwrap()),
        u64::from_le_bytes(bytes[80..88].try_into().unwrap()),
    ];
    assert_eq!(
        declared,
        operator
            .state
            .right
            .heap_capacities()
            .map(|capacity| capacity as u64)
    );
}
