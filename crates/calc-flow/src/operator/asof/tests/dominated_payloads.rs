use super::*;

fn dominated_operator() -> StreamAsofJoinOperator {
    let (template, _) = fixture();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(1_000),
        template.spec.limits(),
    )
    .unwrap();
    StreamAsofJoinOperator::new(
        "asof",
        template.schemas[0].clone(),
        template.schemas[1].clone(),
        spec,
    )
    .unwrap()
}

fn dominated_input(operator: &StreamAsofJoinOperator, rows: &[(&str, i64, i64)]) -> Batch {
    Batch::table(
        vec![indexed_input(&operator.schemas[1], rows)],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn right_sequences(messages: &[crate::StreamMessage]) -> Vec<Option<i64>> {
    messages
        .iter()
        .filter_map(crate::StreamMessage::as_data)
        .flat_map(|batch| batch.table_payload().unwrap().batches())
        .flat_map(|record| {
            record
                .column_by_name("right__seq")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
        })
        .collect()
}

#[tokio::test]
async fn dominated_payload_progress_runs_when_tolerance_and_identity_are_not_due() {
    let mut operator = dominated_operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let initial = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let input = dominated_input(&operator, &[("A", 90, 1), ("A", 100, 2)]);
    operator
        .process_data("right", input, &initial, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status.retained_right_rows, 2);
    let progress = asymmetric_progress(100, 90);
    let context = StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress);
    operator
        .on_ingress_progress_with_output("left", &context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    assert_eq!(operator.status.pending_left_rows, 0);
    assert_eq!(operator.status.retained_right_rows, 1);
    assert_eq!(operator.status.identity_only_rows, 1);
    assert_eq!(operator.status.evicted_right_rows, 1);
    assert_eq!(operator.status.state_rows, 2);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    let before = operator.status();
    operator
        .on_ingress_progress_with_output("left", &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status(), before);
    let repeated = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(repeated.inline_metadata, snapshot.inline_metadata);
    assert_eq!(repeated.segments, snapshot.segments);
    let mut restored = dominated_operator();
    restored
        .restore_with_progress(&snapshot, &asymmetric_progress(100, 90), None)
        .unwrap();
    let right = dominated_input(&restored, &[("A", 110, 3)]);
    restored
        .process_data("right", right, &context, &mut output)
        .await
        .unwrap();
    let left = dominated_input(&restored, &[("A", 105, 123)]);
    restored
        .process_data("left", left, &context, &mut output)
        .await
        .unwrap();
    let ready = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        None,
        asymmetric_progress(111, 111),
    );
    restored
        .on_ingress_progress_with_output("right", &ready, &mut output)
        .await
        .unwrap();
    assert_eq!(right_sequences(&output.drain("output")), vec![Some(2)]);
    restored.prepare_checkpoint_async(&ready).await.unwrap();
    let next = restored.capture(Epoch::INITIAL).unwrap();
    operator
        .restore_with_progress(&next, &asymmetric_progress(111, 111), None)
        .unwrap();
    assert_eq!(operator.status.retained_right_rows, 1);
    assert_eq!(operator.status.evicted_right_rows, 2);
    assert_eq!(operator.status.identity_only_rows, 0);
    assert_eq!(
        operator.runtime.pool.reserved(),
        operator.state.right.auxiliary_bytes()
    );
}

#[tokio::test]
async fn dominated_payload_same_time_uses_typed_sequence_and_retains_duplicate_identity() {
    let mut operator = dominated_operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let initial = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let input = dominated_input(&operator, &[("A", 100, 9), ("A", 100, 1)]);
    operator
        .process_data("right", input, &initial, &mut output)
        .await
        .unwrap();
    let context = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        None,
        asymmetric_progress(100, 100),
    );
    operator
        .on_ingress_progress_with_output("left", &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status.retained_right_rows, 1);
    assert_eq!(operator.status.identity_only_rows, 1);
    let duplicate = dominated_input(&operator, &[("A", 100, 1)]);
    let failure = operator
        .process_data("right", duplicate, &context, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        failure,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        }
    ));
    assert_eq!(operator.status.right.accepted_rows, 2);
    assert_eq!(operator.status.retained_right_rows, 1);
    let left = dominated_input(&operator, &[("A", 100, 123)]);
    operator
        .process_data("left", left, &context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    let ready = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        None,
        asymmetric_progress(101, 101),
    );
    operator
        .on_ingress_progress_with_output("right", &ready, &mut output)
        .await
        .unwrap();
    assert_eq!(right_sequences(&output.drain("output")), vec![Some(9)]);
    assert_eq!(operator.status.matched_rows, 1);
}

#[tokio::test]
async fn dominated_payload_pending_left_protects_its_older_match() {
    let mut operator = dominated_operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let initial = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let right = dominated_input(&operator, &[("A", 90, 1), ("A", 100, 2)]);
    operator
        .process_data("right", right, &initial, &mut output)
        .await
        .unwrap();
    let left = dominated_input(&operator, &[("A", 95, 123)]);
    operator
        .process_data("left", left, &initial, &mut output)
        .await
        .unwrap();
    let blocked = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        None,
        asymmetric_progress(100, 90),
    );
    operator
        .on_ingress_progress_with_output("left", &blocked, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    assert_eq!(operator.status.pending_left_rows, 1);
    assert_eq!(operator.status.retained_right_rows, 2);
    let ready = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        None,
        asymmetric_progress(110, 110),
    );
    operator
        .on_ingress_progress_with_output("right", &ready, &mut output)
        .await
        .unwrap();
    assert_eq!(right_sequences(&output.drain("output")), vec![Some(1)]);
    assert_eq!(operator.status.matched_rows, 1);
    assert_eq!(operator.status.pending_left_rows, 0);
}

#[tokio::test]
async fn dominated_payload_unknown_left_frontier_preserves_both_candidates() {
    let mut operator = dominated_operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let right = dominated_input(&operator, &[("A", 90, 1), ("A", 100, 2)]);
    operator
        .process_data("right", right, &context, &mut output)
        .await
        .unwrap();
    operator
        .on_ingress_progress_with_output("right", &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status.retained_right_rows, 2);
    assert_eq!(operator.status.identity_only_rows, 0);
    assert_eq!(operator.status.evicted_right_rows, 0);
    assert!(output.drain("output").is_empty());
}

#[tokio::test]
async fn dominated_payload_checkpoint_pays_for_current_layout_six() {
    let mut operator = dominated_operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let right = dominated_input(&operator, &[("A", 90, 1), ("A", 100, 2)]);
    operator
        .process_data("right", right, &context, &mut output)
        .await
        .unwrap();
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        snapshot.inline_metadata["state_version"],
        serde_json::json!(3)
    );
    assert_eq!(
        snapshot.inline_metadata["layout_version"],
        serde_json::json!(6)
    );
    assert_eq!(
        snapshot.inline_metadata["accounting_version"],
        serde_json::json!(6)
    );
    assert!(snapshot.segments.contains_key("asof-index-v6"));
    assert_eq!(
        operator.runtime.pool.reserved(),
        operator.state.right.auxiliary_bytes()
    );
}

#[tokio::test]
async fn dominated_payload_inclusive_threshold_handles_both_time_extremes() {
    for (older, newer) in [(i64::MIN, i64::MIN + 1), (i64::MAX - 1, i64::MAX)] {
        let mut operator = dominated_operator();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let initial = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let right = dominated_input(&operator, &[("A", older, 1), ("A", newer, 2)]);
        operator
            .process_data("right", right, &initial, &mut output)
            .await
            .unwrap();
        let progress = StreamOperatorContext::with_ingress_progress(
            &job,
            "asof",
            None,
            asymmetric_progress(newer, older),
        );
        operator
            .on_ingress_progress_with_output("left", &progress, &mut output)
            .await
            .unwrap();
        assert_eq!(operator.status.retained_right_rows, 1);
        assert_eq!(operator.status.identity_only_rows, 1);
        assert_eq!(operator.status.evicted_right_rows, 1);
        assert!(output.drain("output").is_empty());
    }
}
