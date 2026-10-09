use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

async fn admit(
    join: &mut StreamJoinOperator,
    side: &str,
    record: RecordBatch,
    context: &StreamOperatorContext<'_>,
) {
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    join.process_data(
        side,
        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
        context,
        &mut output,
    )
    .await
    .unwrap();
}

async fn seed(join: &mut StreamJoinOperator, context: &StreamOperatorContext<'_>) {
    admit(
        join,
        "left",
        record(
            &[95, 96],
            &[None, Some("猫")],
            &["red", "blue"],
            vec![Some(vec![Some(1), None]), Some(vec![Some(2), Some(3)])],
        ),
        context,
    )
    .await;
    admit(
        join,
        "right",
        record(&[100], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]),
        context,
    )
    .await;
}

fn assert_old_bases(first: &OperatorStateSnapshot, next: &OperatorStateSnapshot) {
    for name in ["left-base", "right-base"] {
        assert!(Arc::ptr_eq(
            &first.segments[name].bytes_arc(),
            &next.segments[name].bytes_arc()
        ));
    }
    assert_eq!(next.inline_metadata["v2_inventory"]["base_epoch"], 1);
}

fn assert_dirty(snapshot: &OperatorStateSnapshot, epoch: u64, id: u64, deltas: usize) {
    let inventory = &snapshot.inline_metadata["v2_inventory"];
    assert_eq!(inventory["deltas"].as_array().unwrap().len(), deltas);
    let delta = snapshot.segments[&format!("left-delta-{epoch}")].bytes();
    assert_eq!(&delta[..8], b"CFJDIX2\0");
    assert_eq!(&delta[16..24], epoch.to_le_bytes());
    assert_eq!(&delta[24..32], 1_u64.to_le_bytes());
    assert_eq!(&delta[32..40], 0_u64.to_le_bytes());
    assert_eq!(&delta[40..48], id.to_le_bytes());
    let mut restored = operator();
    restored.restore(snapshot).unwrap();
    let added = restored
        .state
        .left
        .iter()
        .find(|row| row.row_id == id)
        .unwrap();
    assert_row(added, id, 97, 141, "ok", "red", Some(&[9]));
    assert_eq!(restored.state.metrics.emitted_match_rows, id + 1);
}

fn assert_compacted_rows(snapshot: &OperatorStateSnapshot) {
    let mut restored = operator();
    restored.restore(snapshot).unwrap();
    assert_eq!(
        (restored.state.left.len(), restored.state.right.len()),
        (6, 1)
    );
    let first = &restored.state.left[0];
    assert_eq!(
        (first.row_id, first.event_time.as_micros(), first.charge),
        (0, 95, 136)
    );
    assert_eq!(first.encoded_key.as_slice(), KEY);
    assert!(first.record.view().column(2).is_null(0));
    assert_row(
        &restored.state.left[1],
        1,
        96,
        148,
        "猫",
        "blue",
        Some(&[2, 3]),
    );
    for (row, id) in restored.state.left[2..].iter().zip(2..6) {
        assert_row(row, id, 97, 141, "ok", "red", Some(&[9]));
    }
    assert_row(
        &restored.state.right[0],
        0,
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
            restored.state.next_output_sequence
        ),
        (6, 1, 5)
    );
    assert_eq!(restored.state.metrics.left.retained_bytes, 848);
    assert_eq!(restored.state.metrics.right.retained_bytes, 141);
    assert_eq!(restored.state.metrics.emitted_match_rows, 6);
}

#[tokio::test]
async fn test_v2_writer_prepares_only_dirty_payloads_until_four_dirty_epochs() {
    let mut join = operator();
    let bases = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&bases);
    join.set_checkpoint_writer_base_test_hook(Arc::new(move |credit| {
        assert!(credit.size() > 0);
        assert_eq!(std::thread::current().name(), Some("calc-flow-gather"));
        observed.fetch_add(1, Ordering::SeqCst);
    }));
    let job = job();
    let context = StreamOperatorContext::new(&job, "v2-match", None);
    seed(&mut join, &context).await;
    join.prepare_checkpoint_async(&context).await.unwrap();
    let first = join.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(bases.load(Ordering::SeqCst), 1);
    let mut latest = first.clone();
    for (offset, epoch) in [2, 4, 7, 9].into_iter().enumerate() {
        admit(
            &mut join,
            "left",
            record(&[97], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]),
            &context,
        )
        .await;
        join.prepare_checkpoint_async(&context).await.unwrap();
        latest = join.checkpoint(Epoch::new(epoch).unwrap()).unwrap();
        assert_old_bases(&first, &latest);
        assert_dirty(
            &latest,
            epoch,
            u64::try_from(offset).unwrap() + 2,
            offset + 1,
        );
        assert_eq!(bases.load(Ordering::SeqCst), 1);
    }
    join.prepare_checkpoint_async(&context).await.unwrap();
    let compacted = join.checkpoint(Epoch::new(11).unwrap()).unwrap();
    assert_eq!(compacted.inline_metadata["v2_inventory"]["base_epoch"], 9);
    assert_eq!(
        compacted.inline_metadata["v2_inventory"]["deltas"],
        serde_json::json!([])
    );
    assert_eq!(compacted.segments.len(), 4);
    assert_eq!(bases.load(Ordering::SeqCst), 2);
    assert_compacted_rows(&compacted);
    assert_eq!(latest.inline_metadata["v2_inventory"]["base_epoch"], 1);
    join.prepare_checkpoint_async(&context).await.unwrap();
    let clean = join.checkpoint(Epoch::new(13).unwrap()).unwrap();
    assert_eq!(bases.load(Ordering::SeqCst), 2);
    assert_compacted_rows(&clean);
    assert_eq!(
        clean.inline_metadata["v2_inventory"],
        compacted.inline_metadata["v2_inventory"]
    );
    for (name, segment) in &compacted.segments {
        assert!(Arc::ptr_eq(
            &segment.bytes_arc(),
            &clean.segments[name].bytes_arc()
        ));
    }
    let maximum = join.checkpoint(Epoch::new(u64::MAX).unwrap()).unwrap();
    let unchanged = maximum.inline_metadata.clone();
    assert!(join.prepare_checkpoint_async(&context).await.is_err());
    assert!(join.checkpoint(Epoch::new(u64::MAX).unwrap()).is_err());
    assert_eq!(join.state.last_checkpoint_epoch, Epoch::new(u64::MAX));
    assert_eq!(maximum.inline_metadata, unchanged);
    assert_eq!(bases.load(Ordering::SeqCst), 2);
}
