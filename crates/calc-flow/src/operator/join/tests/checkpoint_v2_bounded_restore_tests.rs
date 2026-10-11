use super::*;

#[path = "checkpoint_v2_bounded_history_tests.rs"]
mod history_controls;

#[path = "checkpoint_v2_restore_admission_tests.rs"]
mod admission_controls;

const ROWS: usize = 4_096;
const RESTORE_BUDGET: usize = 4 << 20;

fn bounded_operator() -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "at",
        "at",
        JoinTimeBounds::new(Duration::ZERO, Duration::from_micros(10)).unwrap(),
        JoinStateLimits::new(10_000, 10 << 20, 10_000).unwrap(),
    )
    .unwrap();
    StreamJoinOperator::new("v2-match", schema(), schema(), spec).unwrap()
}

fn bounded_snapshot(operator: &StreamJoinOperator) -> OperatorStateSnapshot {
    let mut snapshot = snapshot(operator);
    snapshot
        .segments
        .retain(|name, _| name.starts_with("right-"));
    let times = vec![95; ROWS];
    let charges = vec![141; ROWS];
    let ids = (0..u64::try_from(ROWS).unwrap()).collect::<Vec<_>>();
    let payload = payload(
        0,
        &ids,
        &record(
            &times,
            &vec![Some("ok"); ROWS],
            &vec!["red"; ROWS],
            vec![Some(vec![Some(9)]); ROWS],
        ),
        &charges,
    );
    snapshot
        .segments
        .insert("left-base".into(), base(&payload, &times, &charges));
    let digest = hex::encode(payload.digest);
    let entry = serde_json::json!({
        "side": "left", "sha256": digest, "rows": ROWS,
        "bytes": payload.segment.bytes().len(),
    });
    snapshot
        .segments
        .insert(format!("left-payload-{digest}"), payload.segment);
    let metadata = &mut snapshot.inline_metadata;
    metadata.insert("next_left_row_id".into(), ROWS.into());
    metadata.get_mut("metrics").unwrap()["left"]["retained_rows"] = ROWS.into();
    metadata.get_mut("metrics").unwrap()["left"]["retained_bytes"] = (141 * ROWS).into();
    let inventory = metadata.get_mut("v2_inventory").unwrap();
    inventory["base_epoch"] = 2.into();
    inventory["deltas"] = serde_json::json!([]);
    let payloads = inventory["payloads"].as_array_mut().unwrap();
    payloads.retain(|entry| entry["side"] == "right");
    payloads.insert(0, entry);
    snapshot
}

#[test]
fn test_v2_restore_4096_rows_fits_fixed_four_mib_workspace() {
    let mut operator = with_one_gib_pool(bounded_operator());
    operator.restore(&snapshot(&operator)).unwrap();
    assert_restored(&operator);
    let snapshot = bounded_snapshot(&operator);
    let original = snapshot.clone();
    let pool = operator.checkpoint_preload_test_pool().unwrap();
    let pressure = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_reservation("bounded-v2-restore-pressure");
    pressure
        .try_grow((1 << 30) - pool.reserved() - RESTORE_BUDGET)
        .unwrap();
    let result = operator.restore(&snapshot);
    assert!(
        result.is_ok(),
        "4,096 historical rows must fit 4 MiB of additional restore credit: {result:?}"
    );
    assert_eq!(operator.state.left.len(), ROWS);
    assert_eq!(
        operator.status().left.retained_bytes,
        141 * u64::try_from(ROWS).unwrap()
    );
    for (index, row) in operator.state.left.iter().enumerate() {
        assert_row(
            row,
            u64::try_from(index).unwrap(),
            95,
            141,
            "ok",
            "red",
            Some(&[9]),
        );
    }
    assert_eq!(operator.state.right[0].row_id, 4);
    assert_eq!(snapshot.inline_metadata, original.inline_metadata);
    assert_eq!(snapshot.segments, original.segments);
    drop((operator, snapshot, original, pressure));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_v2_restore_payload_scratch_is_scoped_to_each_ipc_decode() {
    let mut operator = operator();
    let observations = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = Arc::clone(&observations);
    operator.set_checkpoint_decoded_row_test_hook(Arc::new(move |credit, _, copied, _| {
        let credit = credit.expect("V2 decode has admitted workspace");
        observed
            .lock()
            .unwrap()
            .push((copied, credit.size(), std::ptr::from_ref(credit) as usize));
    }));
    operator.restore(&snapshot(&operator)).unwrap();
    assert_restored(&operator);
    let observations = observations.lock().unwrap();
    assert_eq!(observations.len(), 6);
    let parent_bytes = observations[0].1;
    assert!(
        parent_bytes > 0,
        "entry instrumentation retains its funded metadata workspace"
    );
    for pair in observations.chunks_exact(2) {
        assert!(!pair[0].0 && pair[1].0);
        assert_eq!(
            pair[0].1, parent_bytes,
            "previous reader scratch must have retired"
        );
        assert!(
            pair[1].1 > 0,
            "reader scratch remains funded until the resident owner returns"
        );
        assert_ne!(
            pair[0].2, pair[1].2,
            "each IPC decode has a child workspace"
        );
    }
}

#[test]
fn test_v2_restore_refunds_scratch_before_returned_payload_and_key_aliases() {
    let mut operator = operator();
    operator.restore(&snapshot(&operator)).unwrap();
    operator.assert_v2_scratch_retirement_for_test(&snapshot(&operator));
    assert_restored(&operator);
}

#[test]
fn test_v2_restore_rejects_live_summaries_before_decoding_payloads() {
    let mut operator = operator();
    operator.restore(&snapshot(&operator)).unwrap();
    operator.set_checkpoint_decoded_row_test_hook(Arc::new(|_, _, _, _| {
        panic!("invalid live summary must fail before IPC decode");
    }));
    let pool = operator.checkpoint_preload_test_pool().unwrap();
    let paid = pool.reserved();
    let oversized = bounded_snapshot(&operator);
    let error = operator.restore(&oversized).unwrap_err();
    assert!(
        error.to_string().contains("exceeds configured limits"),
        "{error}"
    );
    assert_restored(&operator);
    assert_eq!(pool.reserved(), paid);
    let mut inconsistent = snapshot(&operator);
    inconsistent.inline_metadata.get_mut("metrics").unwrap()["left"]["retained_bytes"] = 284.into();
    let error = operator.restore(&inconsistent).unwrap_err();
    assert!(
        error.to_string().contains("charge is inconsistent"),
        "{error}"
    );
    assert_restored(&operator);
    assert_eq!(pool.reserved(), paid);
}
