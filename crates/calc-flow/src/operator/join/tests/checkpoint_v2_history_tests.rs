use super::*;

fn corrupt_removed_charge(snapshot: &OperatorStateSnapshot) -> OperatorStateSnapshot {
    let original = snapshot.segments.get("left-base").unwrap().bytes();
    assert_eq!(&original[..8], b"CFJIDX2\0");
    assert_eq!(&original[16..24], &2_u64.to_le_bytes());
    assert_eq!(&original[24..32], &0_u64.to_le_bytes());
    assert_eq!(&original[32..40], &2_u64.to_le_bytes());
    assert_eq!(&original[40..48], &95_i64.to_le_bytes());
    assert_eq!(&original[48..56], &136_u64.to_le_bytes());
    assert_eq!(&original[88..96], &0_u64.to_le_bytes());
    assert_eq!(&original[104..121], KEY);
    assert_removed_locator(snapshot, &original[56..88]);

    let delta = snapshot.segments.get("left-delta-2").unwrap().bytes();
    assert_eq!(&delta[..8], b"CFJDIX2\0");
    assert_eq!(&delta[24..32], &1_u64.to_le_bytes());
    assert_eq!(&delta[32..40], &1_u64.to_le_bytes());
    assert_eq!(&delta[129..137], &2_u64.to_le_bytes());
    assert_eq!(&delta[137..145], &95_i64.to_le_bytes());
    assert_eq!(&delta[153..170], KEY);

    let mut bytes = original.to_vec();
    bytes[48..56].copy_from_slice(&137_u64.to_le_bytes());
    let mut invalid = snapshot.clone();
    invalid
        .segments
        .insert("left-base".into(), StateSegment::new(bytes));
    assert_eq!(invalid.inline_metadata, snapshot.inline_metadata);
    assert_eq!(invalid.segments.len(), 6);
    for (name, segment) in &snapshot.segments {
        if name != "left-base" {
            assert_eq!(invalid.segments[name].bytes(), segment.bytes());
        }
    }
    assert_eq!(
        &snapshot.segments["left-base"].bytes()[48..56],
        &136_u64.to_le_bytes()
    );
    invalid
}

fn assert_removed_locator(snapshot: &OperatorStateSnapshot, digest: &[u8]) {
    let sha256 = hex::encode(digest);
    let name = format!("left-payload-{sha256}");
    let payload = snapshot.segments.get(&name).unwrap().bytes();
    let actual: [u8; 32] = Sha256::digest(payload).into();
    assert_eq!(actual.as_slice(), digest);
    assert_eq!(&payload[..8], b"CFJPAY2\0");
    assert_eq!(&payload[16..24], &2_u64.to_le_bytes());
    assert_eq!(&payload[32..40], &2_u64.to_le_bytes());
    assert_eq!(&payload[40..48], &7_u64.to_le_bytes());
    let inventory = snapshot.inline_metadata["v2_inventory"]["payloads"]
        .as_array()
        .unwrap();
    let entry = inventory
        .iter()
        .find(|entry| entry["sha256"].as_str() == Some(sha256.as_str()))
        .unwrap();
    assert_eq!(entry["side"].as_str(), Some("left"));
    assert_eq!(entry["rows"].as_u64(), Some(2));
    assert_eq!(
        entry["bytes"].as_u64(),
        Some(u64::try_from(payload.len()).unwrap())
    );
}

fn assert_seed_state(operator: &StreamJoinOperator) {
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (1, 1)
    );
    assert_row(
        &operator.state.left[0],
        0,
        96,
        148,
        "猫",
        "blue",
        Some(&[2, 3]),
    );
    assert_row(
        &operator.state.right[0],
        0,
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
        (1, 1, 1)
    );
    assert_eq!(operator.state.last_checkpoint_epoch, None);
    assert!(!operator.state.ended);
    assert_eq!(
        (
            operator.state.metrics.left.retained_rows,
            operator.state.metrics.left.retained_bytes
        ),
        (1, 148)
    );
    assert_eq!(
        (
            operator.state.metrics.right.retained_rows,
            operator.state.metrics.right.retained_bytes
        ),
        (1, 141)
    );
    assert_eq!(operator.state.metrics.emitted_match_rows, 1);
}

#[tokio::test]
async fn test_join_restore_rejects_removed_v2_upsert_charge_atomically() {
    let mut operator = operator();
    let original = snapshot(&operator);
    let invalid = corrupt_removed_charge(&original);
    let job = job();
    let context = StreamOperatorContext::new(&job, "v2-match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let left = record(
        &[96],
        &[Some("猫")],
        &["blue"],
        vec![Some(vec![Some(2), Some(3)])],
    );
    let right = record(&[100], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]);
    operator
        .process_data(
            "left",
            Batch::table(vec![left], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .process_data(
            "right",
            Batch::table(vec![right], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert_eq!(collector.drain("output").len(), 1);
    assert_seed_state(&operator);
    let before_metrics = operator.state.metrics.clone();

    let failure = operator.restore(&invalid).unwrap_err();
    let CalcFlowError::CheckpointMismatch { message } = failure else {
        panic!("historical charge must fail the checkpoint contract: {failure}");
    };
    assert!(
        message.contains("charge"),
        "expected decoded historical charge rejection: {message}"
    );
    assert_seed_state(&operator);
    assert_eq!(operator.state.metrics, before_metrics);
}
