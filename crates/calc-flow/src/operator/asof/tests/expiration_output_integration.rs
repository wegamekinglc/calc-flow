use super::*;
use crate::OperatorStateSnapshot;
use datafusion::arrow::array::Array;

#[tokio::test(flavor = "current_thread")]
async fn test_a03_a10_projected_restore_preserves_multiplicity_and_funding() {
    let (mut operator, left, right) = prefix_fixture();
    operator.set_output_projection(vec![2, 5, 5]).unwrap();
    let configured = operator.runtime.pool.reserved();
    assert!(configured > 0);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right, &context, &mut output)
        .await
        .unwrap();
    operator
        .process_data("left", left, &context, &mut output)
        .await
        .unwrap();
    assert_eq!(
        operator.runtime.pool.reserved(),
        configured + operator.state.right.auxiliary_bytes()
    );
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        snapshot.inline_metadata["layout_version"],
        serde_json::json!(10)
    );
    let (mut restored, _, _) = prefix_fixture();
    restored.set_output_projection(vec![2, 5, 5]).unwrap();
    let restored_configured = restored.runtime.pool.reserved();
    assert!(restored_configured > 0);
    restored.restore(&snapshot).unwrap();
    assert!(restored.state.right.auxiliary_bytes() > 0);
    assert_log_funded(&restored);
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    workspace::take_output_source_registrations();
    restored.on_end(&context, &mut output).await.unwrap();
    assert_eq!(workspace::take_output_source_registrations(), 2);
    let delivered = output.drain("output");
    assert_eq!(delivered.len(), 1);
    let record = &delivered[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!((record.num_columns(), record.num_rows()), (3, 3));
    let left = record
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(left.values().as_ref(), &[1, 2, 3]);
    for index in [1, 2] {
        let right = record
            .column(index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(right.values().as_ref(), &[1, 1, 1]);
        assert_eq!(right.null_count(), 0);
    }
    assert_eq!(restored.status.matched_rows, 3);
    assert_eq!(restored.status.state_bytes, 0);
    let funding = job.gather_owner().funding();
    let retained = restored.runtime.pool.reserved();
    let failures = job.gather_owner().close_and_drain().await;
    let drained = job.gather_owner().funding();
    drop(context);
    drop(job);
    assert_eq!(funding, (16_384, 16_384, 0));
    assert_eq!(retained, restored_configured + funding.0 + funding.1);
    assert_eq!((drained.1, drained.2), (0, 0));
    assert!(failures.is_empty());
    assert_eq!(restored.runtime.pool.reserved(), restored_configured);
    let restored_pool = restored.runtime.pool.clone();
    drop(restored);
    assert_eq!(restored_pool.reserved(), 0);
    let original_pool = operator.runtime.pool.clone();
    drop(operator);
    drop(snapshot);
    assert_eq!(original_pool.reserved(), 0);
}

#[tokio::test]
async fn test_a03_a10_zero_column_cancelled_prefix_recovers_exactly() {
    let (mut operator, left, right) = prefix_fixture();
    operator.set_output_projection(Vec::new()).unwrap();
    let configured = operator.runtime.pool.reserved();
    assert!(configured > 0);
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancellation.clone());
    let context = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut preload = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right, &context, &mut preload)
        .await
        .unwrap();
    operator
        .process_data("left", left, &context, &mut preload)
        .await
        .unwrap();
    let mut stopped = CancelPrefixCollector {
        cancel: cancellation,
        accepted: Vec::new(),
    };
    assert!(matches!(
        operator
            .on_watermark(EventTime::from_micros(103), &context, &mut stopped)
            .await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(operator.status.pending_left_rows, 2);
    assert_eq!(operator.next_output_sequence, 1);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(
        operator.runtime.pool.reserved(),
        configured + operator.state.right.auxiliary_bytes()
    );
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    let (mut restored, _, _) = prefix_fixture();
    restored.set_output_projection(Vec::new()).unwrap();
    let restored_configured = restored.runtime.pool.reserved();
    assert!(restored_configured > 0);
    restored.restore(&snapshot).unwrap();
    assert_log_funded(&restored);
    let job = StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    restored.on_end(&context, &mut output).await.unwrap();
    stopped.accepted.extend(
        output
            .drain("output")
            .into_iter()
            .map(|message| message.as_data().unwrap().clone()),
    );
    assert_eq!(stopped.accepted.len(), 3);
    for (index, batch) in stopped.accepted.iter().enumerate() {
        assert_eq!(batch.metadata().sequence(), index as u64);
        let record = &batch.table_payload().unwrap().batches()[0];
        assert_eq!((record.num_columns(), record.num_rows()), (0, 1));
    }
    assert_eq!(restored.status.emitted_left_rows, 3);
    assert_eq!(restored.status.matched_rows, 3);
    assert_eq!(restored.status.state_bytes, 0);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(restored.runtime.pool.reserved(), restored_configured);
    let restored_pool = restored.runtime.pool.clone();
    drop(restored);
    assert_eq!(restored_pool.reserved(), 0);
    let original_pool = operator.runtime.pool.clone();
    drop(operator);
    drop(snapshot);
    assert_eq!(original_pool.reserved(), 0);
}

fn cropped_operator() -> StreamAsofJoinOperator {
    let (template, _) = fixture();
    let mut fields = template.schemas[0].fields().to_vec();
    fields.push(Arc::new(Field::new("unused_value", DataType::Int64, false)));
    let schema = Arc::new(Schema::new(fields));
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(1_000),
        template.spec.limits(),
    )
    .unwrap();
    let mut operator = StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
    operator.set_output_projection(vec![2, 6, 6]).unwrap();
    operator
}

fn cropped_input(operator: &StreamAsofJoinOperator, rows: &[(&str, i64, i64)]) -> Batch {
    let record = RecordBatch::try_new(
        operator.schemas[0].clone(),
        vec![
            Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.0))),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(rows.iter().map(|row| row.1))
                    .with_timezone("UTC"),
            ),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.2))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|_| 999))),
        ],
    )
    .unwrap();
    Batch::table(
        vec![record],
        BatchMetadata::new(
            "caller",
            77,
            JsonMap::from([("immutable".into(), serde_json::json!(true))]),
        )
        .unwrap(),
    )
    .unwrap()
}

async fn assert_cropped_refusals(
    operator: &mut StreamAsofJoinOperator,
    snapshot: &OperatorStateSnapshot,
    cancellation: &CancellationToken,
    job: &StreamJobContext,
    output: &mut EdgeCollector,
) {
    let stalled = StreamOperatorContext::with_ingress_progress(
        job,
        "asof",
        None,
        asymmetric_progress(100, 90),
    );
    let mut before = operator.status();
    let duplicate = cropped_input(operator, &[("A", 100, 2)]);
    let error = operator
        .process_data("right", duplicate, &stalled, output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        }
    ));
    before.right.duplicate_rows += 1;
    assert_eq!(operator.status(), before);
    assert_eq!(
        operator.capture(Epoch::INITIAL).unwrap().segments,
        snapshot.segments
    );
    let reserved = operator.runtime.pool.reserved();
    cancellation.cancel();
    let refused = cropped_input(operator, &[("A", 105, 123)]);
    assert!(matches!(
        operator
            .process_data("left", refused, &stalled, output)
            .await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(operator.status(), before);
    assert_eq!(operator.runtime.pool.reserved(), reserved);
}

async fn assert_cropped_continuation(
    restored: &mut StreamAsofJoinOperator,
    restored_configured: usize,
    job: &StreamJobContext,
    output: &mut EdgeCollector,
    left_caller: &Batch,
    left_original: &RecordBatch,
) -> Vec<crate::StreamMessage> {
    let duplicate = cropped_input(restored, &[("A", 100, 2)]);
    let stalled = StreamOperatorContext::with_ingress_progress(
        job,
        "asof",
        None,
        asymmetric_progress(100, 90),
    );
    assert!(matches!(
        restored
            .process_data("right", duplicate, &stalled, output)
            .await,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        })
    ));
    let right = cropped_input(restored, &[("A", 110, 10)]);
    restored
        .process_data("right", right, &stalled, output)
        .await
        .unwrap();
    restored
        .process_data("left", left_caller.clone(), &stalled, output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    let ready = StreamOperatorContext::with_ingress_progress(
        job,
        "asof",
        None,
        asymmetric_progress(111, 111),
    );
    restored
        .on_ingress_progress_with_output("right", &ready, output)
        .await
        .unwrap();
    let delivered = output.drain("output");
    assert_eq!(delivered.len(), 1);
    let actual = delivered[0].as_data().unwrap();
    assert_eq!(
        actual.metadata(),
        &BatchMetadata::new("asof", 0, JsonMap::new()).unwrap()
    );
    let expected = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("left__seq", DataType::Int64, false),
            Field::new("right__seq", DataType::Int64, true),
            Field::new("right__seq", DataType::Int64, true),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![123, 124])),
            Arc::new(Int64Array::from(vec![Some(9), None])),
            Arc::new(Int64Array::from(vec![Some(9), None])),
        ],
    )
    .unwrap();
    assert_eq!(actual.table_payload().unwrap().batches(), &[expected]);
    assert_eq!(
        left_caller.table_payload().unwrap().batches()[0],
        *left_original
    );
    assert_eq!(restored.status.matched_rows, 1);
    assert_eq!(restored.status.unmatched_rows, 1);
    restored.on_end(&ready, output).await.unwrap();
    assert!(output.drain("output").is_empty());
    assert_eq!(restored.status.state_rows, 0);
    let funding = job.gather_owner().funding();
    assert_eq!(funding, (16_384, 16_384, 0));
    assert_eq!(
        restored.runtime.pool.reserved(),
        restored_configured + funding.0 + funding.1,
    );
    let failures = job.gather_owner().close_and_drain().await;
    let drained = job.gather_owner().funding();
    assert!(failures.is_empty());
    assert_eq!((drained.1, drained.2), (0, 0));
    assert_eq!(
        restored.runtime.pool.reserved(),
        restored_configured + drained.0,
    );
    delivered
}

fn assert_cropped_caller(caller: &Batch, original_record: &RecordBatch) {
    assert_eq!(
        caller.table_payload().unwrap().batches()[0],
        *original_record
    );
    assert_eq!(caller.metadata().source(), "caller");
    assert_eq!(caller.metadata().sequence(), 77);
    assert_eq!(
        caller.metadata().attributes(),
        &JsonMap::from([("immutable".into(), serde_json::json!(true))])
    );
}

#[tokio::test]
async fn test_cropped_dominance_only_progress_restores_typed_identity_and_complete_output() {
    let mut operator = cropped_operator();
    let configured = operator.runtime.pool.reserved();
    assert!(configured > 0);
    let right = cropped_input(&operator, &[("A", 90, 1), ("A", 100, 9), ("A", 100, 2)]);
    let caller = right.clone();
    let original_record = caller.table_payload().unwrap().batches()[0].clone();
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancellation.clone());
    let initial = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right, &initial, &mut output)
        .await
        .unwrap();
    let stalled = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        None,
        asymmetric_progress(100, 90),
    );
    operator
        .on_ingress_progress_with_output("left", &stalled, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    assert_eq!(operator.status.retained_right_rows, 1);
    assert_eq!(operator.status.identity_only_rows, 2);
    assert_eq!(operator.status.evicted_right_rows, 2);
    assert_eq!(operator.status.state_rows, 3);
    assert_eq!(
        operator.runtime.pool.reserved(),
        configured + operator.state.right.auxiliary_bytes()
    );
    for (_, (payload, _)) in operator.state.batches.iter() {
        assert_eq!(payload.record.num_columns(), 3);
    }
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        snapshot.inline_metadata["layout_version"],
        serde_json::json!(10)
    );
    assert_eq!(
        snapshot.inline_metadata["retained_payloads"]["columns"],
        serde_json::json!([[0, 1, 2], [0, 1, 2]])
    );
    assert_cropped_refusals(&mut operator, &snapshot, &cancellation, &job, &mut output).await;
    assert_cropped_caller(&caller, &original_record);
    let mut restored = cropped_operator();
    let restored_configured = restored.runtime.pool.reserved();
    restored
        .restore_with_progress(&snapshot, &asymmetric_progress(100, 90), None)
        .unwrap();
    assert_eq!(restored.status.retained_right_rows, 1);
    assert_eq!(restored.status.identity_only_rows, 2);
    assert_log_funded(&restored);
    let job = StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new());
    let left_caller = cropped_input(&restored, &[("A", 105, 123), ("B", 105, 124)]);
    let left_original = left_caller.table_payload().unwrap().batches()[0].clone();
    let _delivered = assert_cropped_continuation(
        &mut restored,
        restored_configured,
        &job,
        &mut output,
        &left_caller,
        &left_original,
    )
    .await;
    drop(job);
    assert_eq!(restored.runtime.pool.reserved(), restored_configured);
    let pool = restored.runtime.pool.clone();
    drop(restored);
    assert_eq!(pool.reserved(), 0);
    let pool = operator.runtime.pool.clone();
    drop(operator);
    drop(snapshot);
    assert_eq!(pool.reserved(), 0);
}

fn tiny_empty_operator() -> StreamAsofJoinOperator {
    let (template, _) = fixture();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::ZERO,
        AsofStateLimits::new(1, 1).unwrap(),
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

async fn assert_empty_terminal_wire_roundtrips(
    terminal: &OperatorStateSnapshot,
    context: &StreamOperatorContext<'_>,
    output: &mut EdgeCollector,
) {
    let ended = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Ended, Some(EventTime::from_micros(101))),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Ended, Some(EventTime::from_micros(101))),
        ),
    ]));
    assert_eq!(
        terminal.inline_metadata["layout_version"],
        serde_json::json!(10)
    );
    let wire = terminal.clone();
    let mut target = tiny_empty_operator();
    target.restore(&wire).unwrap();
    target.restore_with_progress(&wire, &ended, None).unwrap();
    assert_eq!(target.status.right.late_rows, 1);
    assert_eq!(target.next_output_sequence, 0);
    assert!(target.terminal);
    target.on_end(context, output).await.unwrap();
    let repeated = target.capture(Epoch::INITIAL).unwrap();
    assert_eq!(repeated.inline_metadata, terminal.inline_metadata);
    assert_eq!(repeated.segments, terminal.segments);
    assert_eq!(target.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn test_empty_default_checkpoint_keeps_tiny_budget_and_managed_progress() {
    let (_, input) = fixture();
    let mut source = tiny_empty_operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let initial = StreamOperatorContext::new(&job, "asof", None);
    let snapshot = source.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        snapshot.inline_metadata["layout_version"],
        serde_json::json!(10)
    );
    assert_eq!(
        snapshot.inline_metadata["accounting_version"],
        serde_json::json!(10)
    );
    assert!(!snapshot.inline_metadata.contains_key("retained_payloads"));
    assert!(snapshot.segments.is_empty());
    assert_eq!(source.runtime.pool.reserved(), 0);
    let mut restored = tiny_empty_operator();
    restored.restore(&snapshot).unwrap();
    let repeated = restored.capture(Epoch::INITIAL).unwrap();
    assert_eq!(repeated.inline_metadata, snapshot.inline_metadata);
    assert_eq!(repeated.segments, snapshot.segments);
    let progress = asymmetric_progress(101, 101);
    restored
        .restore_with_progress(&snapshot, &progress, Some(EventTime::from_micros(100)))
        .unwrap();
    assert_eq!(
        restored.status.left.watermark_micros,
        Some(EventTime::from_micros(101))
    );
    assert_eq!(
        restored.status.output_watermark_micros,
        Some(EventTime::from_micros(100))
    );
    let managed = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        Some(EventTime::from_micros(100)),
        progress.clone(),
    );
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    assert!(matches!(
        restored
            .process_data("right", input, &managed, &mut output)
            .await,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofLateRow,
            ..
        })
    ));
    assert_eq!(restored.status.right.late_rows, 1);
    assert_eq!(restored.status.state_bytes, 0);
    let diagnostic = restored.capture(Epoch::INITIAL).unwrap();
    restored
        .restore_with_progress(&diagnostic, &progress, Some(EventTime::from_micros(100)))
        .unwrap();
    assert_eq!(restored.status.right.late_rows, 1);
    assert_eq!(
        restored.status.output_watermark_micros,
        Some(EventTime::from_micros(100))
    );
    let invalid = IngressProgressSnapshot::new(BTreeMap::new());
    let before = restored.status();
    assert!(matches!(
        restored.restore_with_progress(&diagnostic, &invalid, None),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(restored.status(), before);
    restored.on_end(&initial, &mut output).await.unwrap();
    let terminal = restored.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        terminal.inline_metadata["terminal"],
        serde_json::json!(true)
    );
    assert_empty_terminal_wire_roundtrips(&terminal, &initial, &mut output).await;
    assert!(output.drain("output").is_empty());
    assert_eq!(restored.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn test_empty_checkpoint_preserves_history_and_configured_descriptor() {
    for configured in [false, true] {
        let (template, _) = fixture();
        let spec = StreamAsofJoinSpec::new(
            template.spec.left().clone(),
            template.spec.right().clone(),
            Duration::from_micros(1000),
            template.spec.limits(),
        )
        .unwrap();
        let create = || {
            StreamAsofJoinOperator::new(
                "asof",
                template.schemas[0].clone(),
                template.schemas[1].clone(),
                spec.clone(),
            )
            .unwrap()
        };
        let mut source = create();
        if configured {
            source.set_output_projection(vec![2, 5]).unwrap();
        }
        let baseline = source.runtime.pool.reserved();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let initial = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(source.output_ports().to_vec());
        let right = Batch::table(
            vec![indexed_input(
                &source.schemas[1],
                &[("A", 90, 1), ("A", 100, 2)],
            )],
            BatchMetadata::default(),
        )
        .unwrap();
        source
            .process_data("right", right, &initial, &mut output)
            .await
            .unwrap();
        let progress = StreamOperatorContext::with_ingress_progress(
            &job,
            "asof",
            None,
            asymmetric_progress(100, 90),
        );
        source
            .on_ingress_progress_with_output("left", &progress, &mut output)
            .await
            .unwrap();
        assert_eq!(source.status.identity_only_rows, 1);
        source.on_end(&initial, &mut output).await.unwrap();
        assert_eq!(source.status.evicted_right_rows, 2);
        assert_eq!(source.status.state_bytes, 0);
        assert_eq!(source.runtime.pool.reserved(), baseline);
        let snapshot = source.capture(Epoch::INITIAL).unwrap();
        assert_eq!(
            snapshot.inline_metadata["layout_version"],
            serde_json::json!(10)
        );
        assert_eq!(
            snapshot.inline_metadata.contains_key("retained_payloads"),
            configured
        );
        assert!(snapshot.segments.is_empty());
        let mut target = create();
        if configured {
            target.set_output_projection(vec![2, 5]).unwrap();
        }
        target.restore(&snapshot).unwrap();
        let repeated = target.capture(Epoch::INITIAL).unwrap();
        assert_eq!(repeated.inline_metadata, snapshot.inline_metadata);
        assert_eq!(repeated.segments, snapshot.segments);
        assert_eq!(target.status.evicted_right_rows, 2);
        assert_eq!(target.runtime.pool.reserved(), baseline);
        if configured {
            let mut missing = snapshot.clone();
            missing.inline_metadata.remove("retained_payloads");
            let before = target.status();
            assert!(matches!(
                target.restore(&missing),
                Err(CalcFlowError::CheckpointMismatch { .. })
            ));
            assert_eq!(target.status(), before);
            let mut invalid_columns = snapshot.clone();
            invalid_columns
                .inline_metadata
                .get_mut("retained_payloads")
                .unwrap()["columns"][0] = serde_json::json!([0, 1]);
            assert!(matches!(
                target.restore(&invalid_columns),
                Err(CalcFlowError::CheckpointMismatch { .. })
            ));
            assert_eq!(target.status(), before);
        }
        assert!(output.drain("output").is_empty());
        for operator in [source, target] {
            let pool = operator.runtime.pool.clone();
            drop(operator);
            assert_eq!(pool.reserved(), 0);
        }
    }
}

#[test]
fn test_current_empty_wire_restore_needs_no_copy_reservation() {
    let (template, _) = fixture();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::ZERO,
        AsofStateLimits::new(1, 1).unwrap(),
    )
    .unwrap();
    let create = || {
        StreamAsofJoinOperator::new(
            "asof",
            template.schemas[0].clone(),
            template.schemas[1].clone(),
            spec.clone(),
        )
        .unwrap()
    };
    let mut source = create();
    source.runtime.pool = Arc::new(datafusion::execution::memory_pool::GreedyMemoryPool::new(
        1 << 20,
    ));
    let wire = source.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        wire.inline_metadata["layout_version"],
        serde_json::json!(10)
    );
    assert_eq!(
        wire.inline_metadata["accounting_version"],
        serde_json::json!(10)
    );
    assert!(!wire.inline_metadata.contains_key("retained_payloads"));
    assert!(wire.segments.is_empty());
    let mut target = create();
    let decoded = target.decoded_snapshot(&wire).unwrap();
    drop(decoded);
    assert_eq!(target.runtime.pool.reserved(), 0);
    let invalid = IngressProgressSnapshot::new(BTreeMap::new());
    assert!(matches!(
        target.restore_with_progress(&wire, &invalid, None),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(target.status.state_bytes, 0);
    assert_eq!(target.runtime.pool.reserved(), 0);
    target
        .restore_with_progress(
            &wire,
            &asymmetric_progress(101, 101),
            Some(EventTime::from_micros(100)),
        )
        .unwrap();
    assert_eq!(
        target.status.output_watermark_micros,
        Some(EventTime::from_micros(100))
    );
    assert_eq!(target.runtime.pool.reserved(), 0);
    target.restore(&wire).unwrap();
    assert_eq!(target.runtime.pool.reserved(), 0);
}
