use super::*;
use std::sync::Mutex;

const CAPTURED: u64 = 6;
const NEXT: u64 = 12;

async fn admit(
    join: &mut StreamJoinOperator,
    side: &str,
    input: RecordBatch,
    charges: &[u64],
    context: &StreamOperatorContext<'_>,
) {
    assert_eq!(input.num_rows(), charges.len());
    for (row, charge) in charges.iter().enumerate() {
        assert_eq!(
            state_row_charge(&input, row, &[0], "v2-match").unwrap(),
            *charge
        );
    }
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    join.process_data(
        side,
        Batch::table(vec![input], BatchMetadata::default()).unwrap(),
        context,
        &mut output,
    )
    .await
    .unwrap();
}

async fn produce_v1(
    join: &mut StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
) -> OperatorStateSnapshot {
    admit(
        join,
        "left",
        record(&[95], &[None], &["red"], vec![Some(vec![Some(1), None])]),
        &[136],
        context,
    )
    .await;
    join.checkpoint_v1(Epoch::INITIAL).unwrap();
    admit(
        join,
        "left",
        record(
            &[96],
            &[Some("猫")],
            &["blue"],
            vec![Some(vec![Some(2), Some(3)])],
        ),
        &[148],
        context,
    )
    .await;
    join.checkpoint_v1(Epoch::new(2).unwrap()).unwrap();
    admit(
        join,
        "right",
        record(&[100], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]),
        &[141],
        context,
    )
    .await;
    join.checkpoint_v1(Epoch::new(3).unwrap()).unwrap();
    admit(
        join,
        "left",
        record(&[97], &[Some("新")], &["green"], vec![None]),
        &[135],
        context,
    )
    .await;
    join.checkpoint_v1(Epoch::new(4).unwrap()).unwrap();
    assert_eq!(join.state.deltas.segments_since_base, 4);
    assert!(join.state.deltas.needs_compaction);
    join.prepare_compaction(context).await.unwrap();
    assert!(!join.state.deltas.needs_compaction);
    let snapshot = join.checkpoint_v1(Epoch::new(CAPTURED).unwrap()).unwrap();
    assert_eq!(snapshot.inline_metadata["layout_version"], 1);
    assert_eq!(snapshot.inline_metadata["epoch"], CAPTURED);
    assert!(!snapshot.inline_metadata.contains_key("v2_inventory"));
    assert_eq!(snapshot.segments.len(), 2);
    assert_v1_base(&snapshot.segments["left-base"], 3);
    assert_v1_base(&snapshot.segments["right-base"], 1);
    snapshot
}

fn assert_v1_base(segment: &StateSegment, rows: u64) {
    assert_eq!(&segment.bytes()[..8], b"CFJOIN1\0");
    assert_eq!(&segment.bytes()[8..16], &rows.to_le_bytes());
}

fn pending(join: &StreamJoinOperator) -> Vec<(JoinSide, u64, i64, Option<u64>)> {
    join.state
        .deltas
        .pending
        .iter()
        .map(|op| match op {
            PendingOp::Upsert {
                side,
                row_id,
                event_time,
                charge,
                ..
            } => (*side, *row_id, event_time.as_micros(), Some(*charge)),
            PendingOp::Tombstone {
                side,
                row_id,
                event_time,
                ..
            } => (*side, *row_id, event_time.as_micros(), None),
        })
        .collect()
}

fn assert_migration_refused(join: &mut StreamJoinOperator) {
    let status = join.status();
    let metrics = join.state.metrics.clone();
    let left = Arc::as_ptr(&join.state.left.0);
    let right = Arc::as_ptr(&join.state.right.0);
    let old_base = join.state.deltas.base.clone();
    let dirty = pending(join);
    assert_eq!(dirty, [(JoinSide::Left, 3, 97, Some(141))]);
    let failure = join.checkpoint(Epoch::new(NEXT).unwrap()).unwrap_err();
    assert!(matches!(failure, CalcFlowError::Internal { message }
        if message == "V2 checkpoint requires successful migration preparation"));
    assert_eq!(join.status(), status);
    assert_eq!(join.state.metrics, metrics);
    assert_eq!(Arc::as_ptr(&join.state.left.0), left);
    assert_eq!(Arc::as_ptr(&join.state.right.0), right);
    assert_eq!(join.state.deltas.base, old_base);
    assert!(join.state.deltas.segments.is_empty());
    assert_eq!(pending(join), dirty);
    assert_eq!(join.state.last_checkpoint_epoch, Epoch::new(CAPTURED));
    assert_eq!(
        (
            join.state.next_left_row_id,
            join.state.next_right_row_id,
            join.state.next_output_sequence
        ),
        (4, 1, 3)
    );
}

async fn migrate(join: &mut StreamJoinOperator, context: &StreamOperatorContext<'_>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&events);
    join.set_checkpoint_writer_test_hook(Arc::new(move |credit| {
        let native = std::thread::current()
            .name()
            .is_some_and(|name| name == "calc-flow-gather");
        observed.lock().unwrap().push((credit.size(), native));
    }));
    let state = join.status();
    join.prepare_checkpoint_async(context).await.unwrap();
    assert_eq!(join.status(), state);
    assert_eq!(join.state.last_checkpoint_epoch, Epoch::new(CAPTURED));
    assert!(join.state.deltas.pending.is_empty());
    assert!(
        join.state.deltas.base.is_empty(),
        "successful V2 migration must retire the absorbed V1 base"
    );
    assert!(join.state.deltas.segments.is_empty());
    assert_eq!(join.state.deltas.segments_since_base, 0);
    assert!(!join.state.deltas.needs_compaction);
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert!(events[0].0 > 0);
    assert!(events[0].1);
}

async fn continue_and_evict(join: &mut StreamJoinOperator, job: &StreamJobContext) {
    let context = StreamOperatorContext::new(job, "v2-match", None);
    admit(
        join,
        "left",
        record(&[98], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]),
        &[141],
        &context,
    )
    .await;
    let watermark = progress_context(
        job,
        (IngressState::Active, None),
        (IngressState::Active, Some(106)),
    );
    join.on_ingress_progress("right", &watermark).await.unwrap();
    assert_eq!(
        pending(join),
        [
            (JoinSide::Left, 4, 98, Some(141)),
            (JoinSide::Left, 0, 95, None)
        ]
    );
    assert_eq!(
        join.status().right.watermark_micros,
        Some(EventTime::from_micros(106))
    );
}

fn assert_final(join: &StreamJoinOperator) {
    let mut left = join.state.left.iter().collect::<Vec<_>>();
    left.sort_by_key(|row| row.row_id);
    assert_eq!(left.len(), 4);
    assert_row(left[0], 1, 96, 148, "猫", "blue", Some(&[2, 3]));
    assert_row(left[1], 2, 97, 135, "新", "green", None);
    assert_row(left[2], 3, 97, 141, "ok", "red", Some(&[9]));
    assert_row(left[3], 4, 98, 141, "ok", "red", Some(&[9]));
    assert_eq!(join.state.right.len(), 1);
    assert_row(&join.state.right[0], 0, 100, 141, "ok", "red", Some(&[9]));
    assert_eq!(
        (
            join.state.next_left_row_id,
            join.state.next_right_row_id,
            join.state.next_output_sequence
        ),
        (5, 1, 4)
    );
    assert_eq!(
        join.state.metrics,
        JoinMetrics {
            left: SideMetrics {
                retained_rows: 4,
                retained_bytes: 565,
                evicted_rows: 1,
                ..SideMetrics::default()
            },
            right: SideMetrics {
                retained_rows: 1,
                retained_bytes: 141,
                ..SideMetrics::default()
            },
            emitted_match_rows: 5,
            ..JoinMetrics::default()
        }
    );
    assert!(!join.state.ended);
}

fn integer(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn assert_payload(
    snapshot: &OperatorStateSnapshot,
    side: &str,
    side_id: u8,
    ids: &[u64],
) -> [u8; 32] {
    let inventory = snapshot.inline_metadata["v2_inventory"]["payloads"]
        .as_array()
        .unwrap();
    let entry = inventory
        .iter()
        .find(|entry| entry["side"] == side && entry["rows"] == ids.len())
        .unwrap();
    let text = entry["sha256"].as_str().unwrap();
    let bytes = snapshot.segments[&format!("{side}-payload-{text}")].bytes();
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    assert_eq!(hex::encode(digest), text);
    assert_eq!(entry["bytes"], bytes.len());
    assert_eq!(&bytes[..16], &header(*b"CFJPAY2\0", side_id));
    assert_eq!(integer(bytes, 16), u64::try_from(ids.len()).unwrap());
    assert_eq!(
        integer(bytes, 24),
        u64::try_from(bytes.len() - 32 - ids.len() * 8).unwrap()
    );
    for (position, id) in ids.iter().enumerate() {
        assert_eq!(integer(bytes, 32 + position * 8), *id);
    }
    digest
}

fn assert_upsert(bytes: &[u8], start: usize, expected: (u64, i64, u64, u64), digest: &[u8; 32]) {
    let (id, time, charge, payload_row) = expected;
    assert_eq!(integer(bytes, start), id);
    assert_eq!(&bytes[start + 8..start + 16], &time.to_le_bytes());
    assert_eq!(integer(bytes, start + 16), charge);
    assert_eq!(&bytes[start + 24..start + 56], digest);
    assert_eq!(integer(bytes, start + 56), payload_row);
    assert_eq!(integer(bytes, start + 64), 17);
    assert_eq!(&bytes[start + 72..start + 89], &KEY);
}

fn assert_indexes(snapshot: &OperatorStateSnapshot) {
    let left_payload = assert_payload(snapshot, "left", 0, &[0, 1, 2, 3]);
    let right_payload = assert_payload(snapshot, "right", 1, &[0]);
    let delta_payload = assert_payload(snapshot, "left", 0, &[4]);
    let left = snapshot.segments["left-base"].bytes();
    assert_eq!(&left[..16], &header(*b"CFJIDX2\0", 0));
    assert_eq!(
        (integer(left, 16), integer(left, 24), left.len()),
        (4, 0, 388)
    );
    for (position, expected) in [
        (0, 95, 136, 0),
        (1, 96, 148, 1),
        (2, 97, 135, 2),
        (3, 97, 141, 3),
    ]
    .into_iter()
    .enumerate()
    {
        assert_upsert(left, 32 + position * 89, expected, &left_payload);
    }
    let right = snapshot.segments["right-base"].bytes();
    assert_eq!(&right[..16], &header(*b"CFJIDX2\0", 1));
    assert_eq!(
        (integer(right, 16), integer(right, 24), right.len()),
        (1, 0, 121)
    );
    assert_upsert(right, 32, (0, 100, 141, 0), &right_payload);
    let delta = snapshot.segments["left-delta-12"].bytes();
    assert_eq!(&delta[..16], &header(*b"CFJDIX2\0", 0));
    assert_eq!(
        (
            integer(delta, 16),
            integer(delta, 24),
            integer(delta, 32),
            delta.len()
        ),
        (NEXT, 1, 1, 170)
    );
    assert_upsert(delta, 40, (4, 98, 141, 0), &delta_payload);
    assert_eq!(integer(delta, 129), 0);
    assert_eq!(&delta[137..145], &95_i64.to_le_bytes());
    assert_eq!(integer(delta, 145), 17);
    assert_eq!(&delta[153..], &KEY);
}

fn assert_capture(snapshot: &OperatorStateSnapshot) {
    assert_eq!(snapshot.inline_metadata["layout_version"], 2);
    assert_eq!(snapshot.inline_metadata["epoch"], NEXT);
    let inventory = &snapshot.inline_metadata["v2_inventory"];
    assert_eq!(inventory["codec_version"], 2);
    assert_eq!(inventory["base_epoch"], CAPTURED);
    assert_eq!(
        inventory["deltas"],
        serde_json::json!([{ "epoch": NEXT, "sides": ["left"] }])
    );
    assert_eq!(inventory["payloads"].as_array().unwrap().len(), 3);
    assert_eq!(snapshot.segments.len(), 6);
    assert_indexes(snapshot);
    let mut restored = operator();
    restored.restore(snapshot).unwrap();
    assert_eq!(
        restored
            .state
            .left
            .iter()
            .map(|row| row.row_id)
            .collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
    assert_final(&restored);
    assert_eq!(restored.state.last_checkpoint_epoch, Epoch::new(NEXT));
    assert!(restored.state.deltas.pending.is_empty());
}

#[tokio::test]
async fn test_v2_writer_migrates_rich_v1_before_skipped_capture_and_eviction() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "v2-match", None);
    let mut join = operator();
    let v1 = produce_v1(&mut join, &context).await;
    let original = v1.clone();
    join.restore(&v1).unwrap();
    admit(
        &mut join,
        "left",
        record(&[97], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]),
        &[141],
        &context,
    )
    .await;
    assert_migration_refused(&mut join);
    migrate(&mut join, &context).await;
    continue_and_evict(&mut join, &job).await;
    assert_final(&join);
    let captured = join.checkpoint(Epoch::new(NEXT).unwrap()).unwrap();
    assert_eq!(join.state.last_checkpoint_epoch, Epoch::new(NEXT));
    assert!(join.state.deltas.pending.is_empty());
    assert_capture(&captured);
    assert_eq!(v1.inline_metadata, original.inline_metadata);
    assert_eq!(v1.segments, original.segments);
    drop(join);
    drop(captured);
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}
