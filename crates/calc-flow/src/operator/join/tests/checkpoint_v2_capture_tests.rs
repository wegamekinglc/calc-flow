use super::*;

fn assert_v1_base(segment: &StateSegment, rows: u64) {
    let bytes = segment.bytes();
    assert_eq!(&bytes[..8], b"CFJOIN1\0");
    assert_eq!(&bytes[8..16], &rows.to_le_bytes());
}

fn assert_recaptured_rows(operator: &StreamJoinOperator) {
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (2, 1)
    );
    assert_row(
        &operator.state.left[0],
        7,
        96,
        148,
        "猫",
        "blue",
        Some(&[2, 3]),
    );
    assert_row(&operator.state.left[1], 9, 97, 135, "新", "green", None);
    assert_row(
        &operator.state.right[0],
        4,
        100,
        141,
        "ok",
        "red",
        Some(&[9]),
    );
    assert_eq!(
        (
            operator.state.next_left_row_id,
            operator.state.next_right_row_id,
            operator.state.next_output_sequence,
        ),
        (10, 5, 6)
    );
    assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(3));
    assert!(!operator.state.ended);
}

#[tokio::test]
async fn test_join_v2_restore_prepares_v1_capture_and_restores_rows() {
    let mut restored = operator();
    let original = snapshot(&restored);
    assert_eq!(original.segments.len(), 6);
    restored.restore(&original).unwrap();
    assert_restored(&restored);
    let metrics = restored.state.metrics.clone();

    let job = job();
    let context = StreamOperatorContext::new(&job, "v2-match", None);
    restored.prepare_compaction(&context).await.unwrap();
    let captured = restored.checkpoint_v1(Epoch::new(3).unwrap()).unwrap();
    assert_eq!(captured.inline_metadata["layout_version"], 1);
    assert_eq!(captured.inline_metadata["epoch"], 3);
    assert!(!captured.inline_metadata.contains_key("v2_inventory"));
    assert_eq!(
        captured
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["left-base", "right-base"]
    );
    assert_v1_base(&captured.segments["left-base"], 2);
    assert_v1_base(&captured.segments["right-base"], 1);

    let mut recaptured = operator();
    recaptured.restore(&captured).unwrap();
    assert_recaptured_rows(&recaptured);
    assert_eq!(recaptured.state.metrics, metrics);
    assert_eq!(original.inline_metadata["layout_version"], 2);
    assert_eq!(&original.segments["left-base"].bytes()[..8], b"CFJIDX2\0");
}
