use super::*;

async fn v1_snapshot() -> OperatorStateSnapshot {
    let mut source = operator();
    let original = snapshot(&source);
    source.restore(&original).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "v2-match", None);
    source.prepare_compaction(&context).await.unwrap();
    let captured = source.checkpoint_v1(Epoch::new(3).unwrap()).unwrap();
    assert_eq!(captured.inline_metadata["layout_version"], 1);
    assert_eq!(
        captured
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["left-base", "right-base"]
    );
    captured
}

fn assert_v1_rows(restored: &StreamJoinOperator) {
    assert_eq!(
        (restored.state.left.len(), restored.state.right.len()),
        (2, 1)
    );
    assert_row(
        &restored.state.left[0],
        7,
        96,
        148,
        "猫",
        "blue",
        Some(&[2, 3]),
    );
    assert_row(&restored.state.left[1], 9, 97, 135, "新", "green", None);
    assert_row(
        &restored.state.right[0],
        4,
        100,
        141,
        "ok",
        "red",
        Some(&[9]),
    );
    assert_eq!(
        (
            restored.state.next_left_row_id,
            restored.state.next_right_row_id,
            restored.state.next_output_sequence,
        ),
        (10, 5, 6)
    );
    assert_eq!(restored.state.last_checkpoint_epoch, Epoch::new(3));
    assert!(!restored.state.ended);
}

#[tokio::test]
async fn test_join_v1_restore_releases_previous_v2_containers_atomically() {
    let v1 = v1_snapshot().await;
    let mut restored = operator();
    let v2 = snapshot(&restored);
    restored.restore(&v2).unwrap();
    assert_restored(&restored);
    let pool = restored
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    let paid = pool.reserved();
    assert!(paid > 0);
    let containers = Arc::downgrade(restored.v2_containers.as_ref().unwrap());
    let left = Arc::downgrade(&restored.state.left.0);
    let right = Arc::downgrade(&restored.state.right.0);
    let metrics = restored.state.metrics.clone();

    let mut invalid = v1.clone();
    invalid
        .inline_metadata
        .insert("next_left_row_id".into(), 0.into());
    assert!(matches!(
        restored.restore(&invalid),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_restored(&restored);
    assert!(Arc::ptr_eq(
        &restored.state.left.0,
        &left.upgrade().unwrap()
    ));
    assert!(Arc::ptr_eq(
        &restored.state.right.0,
        &right.upgrade().unwrap()
    ));
    assert!(Arc::ptr_eq(
        restored.v2_containers.as_ref().unwrap(),
        &containers.upgrade().unwrap()
    ));
    assert_eq!(pool.reserved(), paid);

    restored.restore(&v1).unwrap();
    assert_v1_rows(&restored);
    assert_eq!(restored.state.metrics, metrics);
    assert!(left.upgrade().is_none());
    assert!(right.upgrade().is_none());
    assert!(
        restored.v2_containers.is_none(),
        "V1 restore retained the previous V2 owner slot"
    );
    assert!(containers.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
}
