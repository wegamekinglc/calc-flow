use super::*;

fn operator(kind: StreamJoinType) -> StreamJoinOperator {
    let mut config = spec();
    config.join_type = kind;
    config.bounds = JoinTimeBounds::from_micros(0, 0).unwrap();
    let mut op = StreamJoinOperator::new("match", left_schema(), right_schema(), config).unwrap();
    op.enable_run_core();
    op
}

async fn verify_direct_end(kind: StreamJoinType) {
    let mut op = operator(kind);
    let job = job().with_checkpointing(false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", left_batch(vec![10, 11]), &context, &mut collector)
        .await
        .unwrap();
    op.process_data("right", right_batch(vec![10]), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(
        collector.drain("output")[0].as_data().unwrap().num_rows(),
        1
    );
    let temp = tempfile::tempdir().unwrap();
    let mut store = crate::state::v4::V4Store::new(temp.path().join("terminal-cut"));
    let before = persist_native_epoch(&mut op, &mut store, 1).await;
    let tiny = StreamOperatorContext::new(&job, "match", None)
        .with_output_budget(EdgeBudget::new(1, 1).unwrap());
    (&mut op as &mut dyn StreamOperator)
        .on_end(&tiny, &mut collector)
        .await
        .unwrap();
    assert!(collector.drain("output").is_empty());
    let end = persist_native_epoch(&mut op, &mut store, 2).await;
    let mut restored = operator(kind);
    let cut = crate::state::v4::native::load_native(&mut store, &end, "match").unwrap();
    restored.restore(&cut).unwrap();
    restored.validate_terminal_recovery_state().unwrap();
    let status = restored.status();
    assert_eq!(
        (status.left.evicted_rows, status.right.evicted_rows),
        (2, 1)
    );
    assert_eq!(status.emitted_match_rows, 1);
    assert_eq!(
        (
            cut.inline_metadata["next_left_row_id"].as_u64(),
            cut.inline_metadata["next_right_row_id"].as_u64()
        ),
        (Some(2), Some(1))
    );
    let cut = crate::state::v4::native::load_native(&mut store, &before, "match").unwrap();
    restored.restore(&cut).unwrap();
    restored
        .process_data("right", right_batch(vec![11]), &context, &mut collector)
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output[0].as_data().unwrap().metadata().sequence(), 1);
    assert_eq!(output[0].as_data().unwrap().num_rows(), 1);
}

#[tokio::test]
async fn test_inner_direct_end_releases_both_sides_for_native_restore() {
    verify_direct_end(StreamJoinType::Inner).await;
}

#[tokio::test]
async fn test_semi_direct_end_releases_both_sides_for_native_restore() {
    verify_direct_end(StreamJoinType::LeftSemi).await;
}

#[tokio::test]
async fn test_direct_end_right_counter_overflow_keeps_both_sides_open() {
    let mut op = operator(StreamJoinType::Inner);
    let job = job().with_checkpointing(false);
    let ctx = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", left_batch(vec![20]), &ctx, &mut output)
        .await
        .unwrap();
    op.process_data("right", right_batch(vec![10]), &ctx, &mut output)
        .await
        .unwrap();
    op.state.metrics.right.evicted_rows = u64::MAX;
    let before = op.status();
    let error = (&mut op as &mut dyn StreamOperator)
        .on_end(&ctx, &mut output)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("evicted_rows"));
    assert_eq!(op.status(), before);
    assert!(output.drain("output").is_empty());
    op.state.metrics.right.evicted_rows = 0;
    op.on_end(&ctx, &mut output).await.unwrap();
    op.validate_terminal_recovery_state().unwrap();
}
