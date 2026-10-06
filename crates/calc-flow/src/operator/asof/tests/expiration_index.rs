use super::*;

async fn populated_current_snapshot() -> (StreamAsofJoinOperator, crate::OperatorStateSnapshot) {
    let (mut operator, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", input.clone(), &context, &mut output)
        .await
        .unwrap();
    operator
        .process_data("left", input, &context, &mut output)
        .await
        .unwrap();
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        snapshot.inline_metadata["layout_version"],
        serde_json::json!(10)
    );
    (operator, snapshot)
}

#[tokio::test]
async fn test_a03_mixed_and_unknown_snapshot_versions_fail_before_allocation() {
    let (mut operator, snapshot) = populated_current_snapshot().await;
    let before = operator.status();
    let reserved = operator.runtime.pool.reserved();
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
        assert_eq!(operator.runtime.pool.reserved(), reserved);
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
    assert_eq!(state.heap_capacities(), [0, 0, 0]);
}

#[tokio::test]
async fn test_a03_guard_dropped_preparation_preserves_installed_lease() {
    let (mut operator, snapshot) = populated_current_snapshot().await;
    operator.restore(&snapshot).unwrap();
    let before = operator.status();
    let capacities = operator.state.right.checkpoint_capacities();
    let expected = operator.runtime.pool.reserved();
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
    drop(snapshot);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_a03_guard_failed_restore_preserves_live_state_and_lease() {
    let (mut operator, snapshot) = populated_current_snapshot().await;
    let before = operator.status();
    let expected = operator.runtime.pool.reserved();
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

#[tokio::test]
async fn test_a03_guard_progress_rejection_preserves_installed_lease() {
    let (mut operator, current) = populated_current_snapshot().await;
    let before = operator.status();
    let expected = operator.runtime.pool.reserved();
    assert!(matches!(
        operator.restore_with_progress(&current, &asymmetric_progress(150, 150), None),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(operator.status(), before);
    assert_eq!(operator.runtime.pool.reserved(), expected);
    let repeated = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(repeated.inline_metadata, current.inline_metadata);
    assert_eq!(repeated.segments, current.segments);
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
    assert_eq!(operator.state.right.heap_capacities(), [0, 0, 0]);
    assert_eq!(operator.runtime.pool.reserved(), 0);
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
    let expected = dictionary * 8
        + operator
            .state
            .right
            .shard_capacities()
            .into_iter()
            .sum::<usize>()
            * 24
        + heaps.into_iter().sum::<usize>() * 16;
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
    let expected = restored.runtime.pool.reserved();
    assert!(expected > 0);
    assert_log_funded(&restored);
    assert!(restored.state.right.auxiliary_bytes() > 0);
    let pool = restored.runtime.pool.clone();
    drop(restored);
    drop(snapshot);
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
        operator.current_inventory(None).unwrap().bytes,
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
        serde_json::json!(10)
    );
    assert_eq!(
        snapshot.inline_metadata["accounting_version"],
        serde_json::json!(10)
    );
    assert_eq!(
        &snapshot.segments["asof-log-v10-1-0-1"].bytes()[..8],
        b"CFASDL10"
    );
    assert!(!snapshot.segments.contains_key("asof-index-v3"));
    let bytes = &snapshot.segments["asof-log-v10-1-0-1"].bytes()[128..];
    let declared = [
        u64::from_le_bytes(bytes[72..80].try_into().unwrap()),
        u64::from_le_bytes(bytes[80..88].try_into().unwrap()),
        u64::from_le_bytes(bytes[88..96].try_into().unwrap()),
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
