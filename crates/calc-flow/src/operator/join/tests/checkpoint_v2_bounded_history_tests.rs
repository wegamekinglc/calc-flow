use super::*;

fn history_delta(
    epoch: u64,
    added: &Payload,
    time: i64,
    charge: u64,
    deleted: (u64, i64),
) -> StateSegment {
    let mut bytes = header(*b"CFJDIX2\0", 0);
    bytes.extend_from_slice(&epoch.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    append_upsert(&mut bytes, added, 0, time, charge);
    bytes.extend_from_slice(&deleted.0.to_le_bytes());
    bytes.extend_from_slice(&deleted.1.to_le_bytes());
    bytes.extend_from_slice(&17_u64.to_le_bytes());
    bytes.extend_from_slice(&KEY);
    StateSegment::new(bytes)
}

pub(super) fn interleaved_snapshot(operator: &StreamJoinOperator) -> OperatorStateSnapshot {
    let mut snapshot = snapshot(operator);
    let old_digest = hex::encode(&snapshot.segments["left-delta-2"].bytes()[64..96]);
    snapshot.segments.remove("left-delta-2");
    snapshot
        .segments
        .remove(&format!("left-payload-{old_digest}"));
    let first = payload(
        0,
        &[1],
        &record(&[97], &[Some("新")], &["green"], vec![None]),
        &[135],
    );
    let second = payload(
        0,
        &[4],
        &record(&[98], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]),
        &[141],
    );
    snapshot.segments.insert(
        "left-delta-5".into(),
        history_delta(5, &first, 97, 135, (2, 95)),
    );
    snapshot.segments.insert(
        "left-delta-9".into(),
        history_delta(9, &second, 98, 141, (7, 96)),
    );
    snapshot.inline_metadata.insert("epoch".into(), 9.into());
    snapshot.inline_metadata.get_mut("metrics").unwrap()["left"]["retained_bytes"] = 276.into();
    snapshot.inline_metadata.get_mut("metrics").unwrap()["left"]["evicted_rows"] = 2.into();
    let inventory = snapshot.inline_metadata.get_mut("v2_inventory").unwrap();
    inventory["deltas"] = serde_json::json!([
        {"epoch": 5, "sides": ["left"]}, {"epoch": 9, "sides": ["left"]},
    ]);
    let entries = inventory["payloads"].as_array_mut().unwrap();
    entries.retain(|entry| entry["sha256"] != old_digest);
    for payload in [first, second] {
        let digest = hex::encode(payload.digest);
        entries.push(serde_json::json!({
            "side": "left", "sha256": digest, "rows": 1,
            "bytes": payload.segment.bytes().len(),
        }));
        snapshot
            .segments
            .insert(format!("left-payload-{digest}"), payload.segment);
    }
    entries.sort_by(|a, b| {
        (a["side"].as_str(), a["sha256"].as_str()).cmp(&(b["side"].as_str(), b["sha256"].as_str()))
    });
    snapshot
}

#[test]
fn test_v2_restore_merges_gapped_epochs_and_interleaved_both_side_ids() {
    let mut operator = operator();
    operator.restore(&interleaved_snapshot(&operator)).unwrap();
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (2, 1)
    );
    assert_row(&operator.state.left[0], 1, 97, 135, "新", "green", None);
    assert_row(&operator.state.left[1], 4, 98, 141, "ok", "red", Some(&[9]));
    assert_row(
        &operator.state.right[0],
        4,
        100,
        141,
        "ok",
        "red",
        Some(&[9]),
    );
    assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(9));
    assert_eq!(operator.status().left.retained_bytes, 276);
}

fn changed_segment(
    snapshot: &OperatorStateSnapshot,
    name: &str,
    offset: usize,
    value: u64,
) -> OperatorStateSnapshot {
    let mut changed = snapshot.clone();
    let mut bytes = changed.segments[name].bytes().to_vec();
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    changed
        .segments
        .insert(name.into(), StateSegment::new(bytes));
    changed
}

#[test]
fn test_v2_restore_rejects_history_aliases_and_invalid_tombstone_order_atomically() {
    let mut operator = operator();
    operator.restore(&snapshot(&operator)).unwrap();
    let valid = interleaved_snapshot(&operator);
    let pool = operator.checkpoint_preload_test_pool().unwrap();
    let paid = pool.reserved();
    let left = Arc::clone(&operator.state.left.0);
    let right = Arc::clone(&operator.state.right.0);
    let containers = Arc::clone(operator.v2_containers.as_ref().unwrap());
    let metrics = operator.status();
    for (name, offset, value) in [
        ("left-delta-5", 40, 7),
        ("left-delta-9", 40, 1),
        ("left-delta-5", 96, 99),
        ("left-delta-5", 129, 4),
        ("left-delta-5", 129, 6),
        ("left-delta-9", 129, 2),
        ("left-delta-5", 137, 96),
        ("left-base", 177, 0),
    ] {
        let invalid = changed_segment(&valid, name, offset, value);
        let result = operator.restore(&invalid);
        assert!(
            matches!(result, Err(CalcFlowError::CheckpointMismatch { .. })),
            "{name}:{offset}={value}: {result:?}"
        );
        assert!(Arc::ptr_eq(&operator.state.left.0, &left));
        assert!(Arc::ptr_eq(&operator.state.right.0, &right));
        assert!(Arc::ptr_eq(
            operator.v2_containers.as_ref().unwrap(),
            &containers
        ));
        assert_restored(&operator);
        assert_eq!(operator.status(), metrics);
        assert_eq!(pool.reserved(), paid);
    }
}

#[test]
fn test_v2_restore_validates_deleted_key_and_time_against_payload() {
    let mut operator = operator();
    operator.restore(&snapshot(&operator)).unwrap();
    let valid = interleaved_snapshot(&operator);
    let pool = operator.checkpoint_preload_test_pool().unwrap();
    let paid = pool.reserved();
    let left = Arc::clone(&operator.state.left.0);
    let right = Arc::clone(&operator.state.right.0);
    for (base_offset, tombstone_offset, value, expected) in [
        (40, 137, 94, "historical time"),
        (113, 162, 8, "historical key"),
    ] {
        let changed = changed_segment(&valid, "left-base", base_offset, value);
        let changed = changed_segment(&changed, "left-delta-5", tombstone_offset, value);
        let error = operator.restore(&changed).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        assert!(Arc::ptr_eq(&operator.state.left.0, &left));
        assert!(Arc::ptr_eq(&operator.state.right.0, &right));
        assert_restored(&operator);
        assert_eq!(pool.reserved(), paid);
    }
}

#[test]
fn test_v2_restore_rejects_cross_side_payload_locators_before_decoding() {
    let mut operator = operator();
    operator.restore(&snapshot(&operator)).unwrap();
    let mut invalid = interleaved_snapshot(&operator);
    let mut bytes = invalid.segments["left-delta-5"].bytes().to_vec();
    bytes[64..96].copy_from_slice(&invalid.segments["right-base"].bytes()[56..88]);
    invalid
        .segments
        .insert("left-delta-5".into(), StateSegment::new(bytes));
    operator.set_checkpoint_decoded_row_test_hook(Arc::new(|_, _, _, _| {
        panic!("cross-side payload locator must fail before IPC decode");
    }));
    let pool = operator.checkpoint_preload_test_pool().unwrap();
    let paid = pool.reserved();
    let error = operator.restore(&invalid).unwrap_err();
    assert!(error.to_string().contains("unknown payload"), "{error}");
    assert_restored(&operator);
    assert_eq!(pool.reserved(), paid);
}
