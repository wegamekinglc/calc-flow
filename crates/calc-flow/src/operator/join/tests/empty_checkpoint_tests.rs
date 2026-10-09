use super::*;

#[tokio::test]
async fn test_restore_accepts_empty_v1_history_after_uncaptured_eviction() {
    let mut source =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job_context = job();
    let context = StreamOperatorContext::new(&job_context, "match", None);
    let mut collector = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data("left", left_batch(vec![0]), &context, &mut collector)
        .await
        .unwrap();
    source
        .process_data("right", right_batch(vec![1]), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(collector.drain("output").len(), 1);
    let ended = progress_context(
        &job_context,
        (IngressState::Ended, Some(0)),
        (IngressState::Ended, Some(1)),
    );
    source.on_ingress_progress("left", &ended).await.unwrap();
    source.on_ingress_progress("right", &ended).await.unwrap();
    source.on_end(&ended, &mut collector).await.unwrap();
    let snapshot = source.checkpoint_v1(Epoch::new(7).unwrap()).unwrap();
    assert!(snapshot.segments.is_empty());
    assert_empty_history(&source);

    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored.restore(&snapshot).unwrap();
    assert_empty_history(&restored);
    let round_trip = restored.checkpoint_v1(Epoch::new(8).unwrap()).unwrap();
    let mut expected = snapshot.inline_metadata.clone();
    expected.insert("epoch".into(), 8.into());
    assert_eq!(round_trip.inline_metadata, expected);
    assert!(round_trip.segments.is_empty());
    reject_missing_retained_rows(&mut restored, &snapshot);
}

fn assert_empty_history(operator: &StreamJoinOperator) {
    assert!(operator.state.left.is_empty());
    assert!(operator.state.right.is_empty());
    assert_eq!(operator.state.next_left_row_id, 1);
    assert_eq!(operator.state.next_right_row_id, 1);
    assert_eq!(operator.state.next_output_sequence, 1);
    assert_eq!(operator.state.metrics.emitted_match_rows, 1);
    assert_eq!(operator.state.metrics.left.evicted_rows, 1);
    assert_eq!(operator.state.metrics.right.evicted_rows, 1);
    assert_eq!(operator.state.metrics.left.retained_rows, 0);
    assert_eq!(operator.state.metrics.left.retained_bytes, 0);
    assert_eq!(operator.state.metrics.right.retained_rows, 0);
    assert_eq!(operator.state.metrics.right.retained_bytes, 0);
    assert!(operator.state.ended);
}

fn reject_missing_retained_rows(
    operator: &mut StreamJoinOperator,
    snapshot: &OperatorStateSnapshot,
) {
    for (side, gauge) in [
        ("left", "retained_rows"),
        ("left", "retained_bytes"),
        ("right", "retained_rows"),
        ("right", "retained_bytes"),
    ] {
        let mut invalid = snapshot.clone();
        invalid.inline_metadata.get_mut("metrics").unwrap()[side][gauge] = 1.into();
        let error = operator.restore(&invalid).unwrap_err();
        assert!(error.to_string().contains("segment inventory is empty"));
        assert_empty_history(operator);
        assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(8));
    }
}
