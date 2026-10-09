use super::*;

fn assert_shared_history(previous: &OperatorStateSnapshot, next: &OperatorStateSnapshot) {
    for (name, segment) in &previous.segments {
        let retained = next
            .segments
            .get(name)
            .expect("historical segment retained");
        assert_eq!(segment.bytes(), retained.bytes());
        assert!(Arc::ptr_eq(&segment.bytes_arc(), &retained.bytes_arc()));
    }
}

fn assert_history_rows(join: &StreamJoinOperator, epoch: Epoch, continued: bool) {
    let (rows, bytes, next_left, sequence, emitted) = if continued {
        (3, 424, 11, 7, 6)
    } else {
        (2, 283, 10, 6, 5)
    };
    assert_eq!((join.state.left.len(), join.state.right.len()), (rows, 1));
    assert_row(&join.state.left[0], 7, 96, 148, "猫", "blue", Some(&[2, 3]));
    assert_row(&join.state.left[1], 9, 97, 135, "新", "green", None);
    assert_row(&join.state.right[0], 4, 100, 141, "ok", "red", Some(&[9]));
    if continued {
        assert_row(&join.state.left[2], 10, 98, 141, "ok", "red", Some(&[9]));
    }
    assert_eq!(
        (
            join.state.next_left_row_id,
            join.state.next_right_row_id,
            join.state.next_output_sequence
        ),
        (next_left, 5, sequence)
    );
    assert_eq!(join.state.last_checkpoint_epoch, Some(epoch));
    assert_eq!(
        join.state.metrics,
        JoinMetrics {
            left: SideMetrics {
                retained_rows: u64::try_from(rows).unwrap(),
                retained_bytes: bytes,
                evicted_rows: 1,
                ..SideMetrics::default()
            },
            right: SideMetrics {
                retained_rows: 1,
                retained_bytes: 141,
                ..SideMetrics::default()
            },
            emitted_match_rows: emitted,
            ..JoinMetrics::default()
        }
    );
    assert!(!join.state.ended);
}

fn assert_roundtrip(snapshot: &OperatorStateSnapshot, epoch: Epoch, continued: bool) {
    let mut restored = operator();
    restored.restore(snapshot).unwrap();
    assert_history_rows(&restored, epoch, continued);
}

fn assert_clean_history(source: &OperatorStateSnapshot, clean: &OperatorStateSnapshot) {
    assert_eq!(clean.inline_metadata["layout_version"], 2);
    assert_eq!(clean.inline_metadata["epoch"], 3);
    assert_eq!(
        clean.inline_metadata["v2_inventory"],
        source.inline_metadata["v2_inventory"]
    );
    assert_eq!(clean.inline_metadata["v2_inventory"]["base_epoch"], 1);
    assert_eq!(
        clean.inline_metadata["v2_inventory"]["deltas"],
        serde_json::json!([
            {"epoch": 2, "sides": ["left"]}
        ])
    );
    assert_eq!(clean.segments.len(), 6);
    assert_shared_history(source, clean);
    assert_roundtrip(clean, Epoch::new(3).unwrap(), false);
}

fn assert_dirty_history(clean: &OperatorStateSnapshot, dirty: &OperatorStateSnapshot) {
    assert_eq!(dirty.inline_metadata["layout_version"], 2);
    assert_eq!(dirty.inline_metadata["epoch"], 4);
    assert_eq!(dirty.inline_metadata["v2_inventory"]["base_epoch"], 1);
    assert_eq!(
        dirty.inline_metadata["v2_inventory"]["deltas"],
        serde_json::json!([
            {"epoch": 2, "sides": ["left"]},
            {"epoch": 4, "sides": ["left"]}
        ])
    );
    assert_eq!(dirty.segments.len(), 8);
    assert_shared_history(clean, dirty);
    let added = dirty
        .segments
        .keys()
        .filter(|name| !clean.segments.contains_key(*name))
        .collect::<Vec<_>>();
    assert_eq!(added.len(), 2);
    assert!(added.iter().any(|name| name.as_str() == "left-delta-4"));
    assert!(added.iter().any(|name| name.starts_with("left-payload-")));
    let delta = dirty.segments["left-delta-4"].bytes();
    assert_eq!(&delta[..8], b"CFJDIX2\0");
    assert_eq!(&delta[16..24], &4_u64.to_le_bytes());
    assert_eq!(&delta[24..32], &1_u64.to_le_bytes());
    assert_eq!(&delta[32..40], &0_u64.to_le_bytes());
    assert_eq!(&delta[40..48], &10_u64.to_le_bytes());
    assert_eq!(
        dirty.inline_metadata["v2_inventory"]["payloads"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert_roundtrip(dirty, Epoch::new(4).unwrap(), true);
}

#[tokio::test]
async fn test_v2_writer_continues_restored_history_without_reencoding() {
    let mut join = operator();
    let source = snapshot(&join);
    let original = source.clone();
    join.restore(&source).unwrap();
    assert_restored(&join);
    let clean = join.checkpoint(Epoch::new(3).unwrap()).unwrap();
    assert_history_rows(&join, Epoch::new(3).unwrap(), false);
    assert_clean_history(&source, &clean);

    let job = job();
    let context = StreamOperatorContext::new(&job, "v2-match", None);
    let mut collector = EdgeCollector::new(join.output_ports().to_vec());
    let left = record(&[98], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]);
    assert_eq!(state_row_charge(&left, 0, &[0], "v2-match").unwrap(), 141);
    join.process_data(
        "left",
        Batch::table(vec![left], BatchMetadata::default()).unwrap(),
        &context,
        &mut collector,
    )
    .await
    .unwrap();
    assert_history_rows(&join, Epoch::new(3).unwrap(), true);
    let dirty = join.checkpoint(Epoch::new(4).unwrap()).unwrap();
    assert_history_rows(&join, Epoch::new(4).unwrap(), true);
    assert_dirty_history(&clean, &dirty);
    assert_eq!(source.inline_metadata, original.inline_metadata);
    assert_eq!(source.segments, original.segments);
}
