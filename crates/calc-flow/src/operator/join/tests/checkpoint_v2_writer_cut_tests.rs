use super::*;

async fn admit_left(operator: &mut StreamJoinOperator, context: &StreamOperatorContext<'_>) {
    let record = record(
        &[95, 96],
        &[None, Some("猫")],
        &["red", "blue"],
        vec![Some(vec![Some(1), None]), Some(vec![Some(2), Some(3)])],
    );
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            context,
            &mut output,
        )
        .await
        .unwrap();
}

async fn seed(operator: &mut StreamJoinOperator, context: &StreamOperatorContext<'_>) {
    admit_left(operator, context).await;
    let right = record(&[100], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            Batch::table(vec![right], BatchMetadata::default()).unwrap(),
            context,
            &mut output,
        )
        .await
        .unwrap();
}

fn assert_live(snapshot: &OperatorStateSnapshot, left_ids: &[u64], emitted: u64) {
    let mut restored = operator();
    restored.restore(snapshot).unwrap();
    assert_eq!(restored.state.left.len(), left_ids.len());
    assert_eq!(restored.state.right.len(), 1);
    for (row, id) in restored.state.left.iter().zip(left_ids) {
        assert_eq!(row.row_id, *id);
        assert_eq!(row.encoded_key.as_slice(), KEY);
        assert_eq!(
            row.event_time.as_micros(),
            95 + i64::try_from(id % 2).unwrap()
        );
        assert_eq!(row.charge, if id % 2 == 0 { 136 } else { 148 });
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
        restored.state.next_left_row_id,
        u64::try_from(left_ids.len()).unwrap()
    );
    assert_eq!(restored.state.next_right_row_id, 1);
    assert_eq!(restored.state.metrics.emitted_match_rows, emitted);
}

fn assert_shared(previous: &OperatorStateSnapshot, next: &OperatorStateSnapshot) {
    assert_eq!(previous.segments.len(), next.segments.len());
    for (name, segment) in &previous.segments {
        assert!(Arc::ptr_eq(
            &segment.bytes_arc(),
            &next.segments[name].bytes_arc()
        ));
    }
}

async fn assert_positive_anchor(context: &StreamOperatorContext<'_>) {
    let mut join = operator();
    seed(&mut join, context).await;
    join.prepare_v2_checkpoint(context).await.unwrap();
    let first = join.checkpoint(Epoch::INITIAL).unwrap();
    assert_live(&first, &[0, 1], 2);
    let clean = join.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(clean.inline_metadata["v2_inventory"]["base_epoch"], 1);
    assert_shared(&first, &clean);
    admit_left(&mut join, context).await;
    join.prepare_v2_checkpoint(context).await.unwrap();
    admit_left(&mut join, context).await;
    let later = join.checkpoint(Epoch::new(3).unwrap()).unwrap();
    assert_eq!(later.inline_metadata["v2_inventory"]["base_epoch"], 2);
    assert_eq!(
        later.inline_metadata["v2_inventory"]["deltas"],
        serde_json::json!([
            {"epoch": 3, "sides": ["left"]}
        ])
    );
    assert_eq!(
        &later.segments["left-delta-3"].bytes()[24..32],
        &2_u64.to_le_bytes()
    );
    assert_live(&later, &[0, 1, 2, 3, 4, 5], 6);
}

async fn assert_stale_initial(context: &StreamOperatorContext<'_>) {
    let mut join = operator();
    seed(&mut join, context).await;
    join.prepare_v2_checkpoint(context).await.unwrap();
    admit_left(&mut join, context).await;
    let captured = join.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(captured.inline_metadata["v2_inventory"]["base_epoch"], 0);
    assert_eq!(
        &captured.segments["left-base"].bytes()[16..24],
        &0_u64.to_le_bytes()
    );
    assert_eq!(
        &captured.segments["right-base"].bytes()[16..24],
        &0_u64.to_le_bytes()
    );
    assert_eq!(
        captured.inline_metadata["v2_inventory"]["deltas"],
        serde_json::json!([
            {"epoch": 1, "sides": ["left", "right"]}
        ])
    );
    assert_live(&captured, &[0, 1, 2, 3], 4);
}

#[tokio::test]
async fn test_v2_writer_preserves_prepared_cut_anchor_and_dirty_history() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "v2-match", None);
    assert_positive_anchor(&context).await;
    assert_stale_initial(&context).await;
}
