mod asof_support;
use asof_support::{batch, operator};
use calc_flow::{
    CalcFlowError, CancellationToken, EdgeCollector, EventTime, IngressProgress,
    IngressProgressSnapshot, IngressState, JsonMap, OperatorMetadata, StreamJobContext,
    StreamOperator, StreamOperatorContext, StreamingFailureReason,
};
use std::collections::BTreeMap;

#[tokio::test]
async fn duplicate_in_batch_is_rejected_atomically() {
    let mut op = operator(10);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    let result = op
        .process_data(
            "right",
            batch(&[("A", 100, 1, 5), ("A", 100, 1, 6)]),
            &cx,
            &mut out,
        )
        .await;
    assert!(matches!(
        result,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        })
    ));
    assert_eq!(op.status().right.accepted_rows, 0);
    assert_eq!(op.status().right.duplicate_rows, 1);
    assert_eq!(op.status().retained_right_rows, 0);
    op.process_data("right", batch(&[("A", 100, 1, 7)]), &cx, &mut out)
        .await
        .unwrap();
    assert_eq!(op.status().right.accepted_rows, 1);
}

#[tokio::test]
async fn late_error_precedes_duplicate_and_keeps_admission_atomic() {
    let mut op = operator(10);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, None),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(100))),
        ),
    ]));
    let cx = StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    let result = op
        .process_data(
            "right",
            batch(&[("A", 100, 1, 5), ("A", 99, 2, 6)]),
            &cx,
            &mut out,
        )
        .await;
    assert!(matches!(
        result,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofLateRow,
            ..
        })
    ));
    assert_eq!(op.status().right.accepted_rows, 0);
    assert_eq!(op.status().right.late_rows, 1);
    assert_eq!(op.status().retained_right_rows, 0);
}

#[tokio::test]
async fn expired_payload_retains_on_time_identity_until_own_watermark() {
    let mut op = operator(10);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let initial = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", batch(&[("A", 100, 1, 5)]), &initial, &mut out)
        .await
        .unwrap();
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1000))),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(0))),
        ),
    ]));
    let cx = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        Some(EventTime::from_micros(0)),
        progress,
    );
    op.on_watermark(EventTime::from_micros(0), &cx, &mut out)
        .await
        .unwrap();
    assert_eq!(op.status().retained_right_rows, 0);
    assert_eq!(op.status().identity_only_rows, 1);
    assert_eq!(op.status().state_rows, 1);
    assert!(matches!(
        op.process_data("right", batch(&[("A", 100, 1, 6)]), &cx, &mut out)
            .await,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        })
    ));
}

#[tokio::test]
async fn state_rows_are_shared_and_rejected_batch_leaves_no_prefix() {
    use calc_flow::{AsofStateLimits, StreamAsofJoinOperator, StreamAsofJoinSpec};
    let template = asof_support::spec(10);
    let spec = StreamAsofJoinSpec::new(
        template.left().clone(),
        template.right().clone(),
        std::time::Duration::from_micros(10),
        AsofStateLimits::new(2, 1024 * 1024).unwrap(),
    )
    .unwrap();
    let mut op =
        StreamAsofJoinOperator::new("asof", asof_support::schema(), asof_support::schema(), spec)
            .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", batch(&[("A", 100, 1, 5)]), &cx, &mut out)
        .await
        .unwrap();
    let error = op
        .process_data(
            "left",
            batch(&[("A", 101, 1, 8), ("A", 102, 2, 9)]),
            &cx,
            &mut out,
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofStateLimitExceeded,
            ..
        }
    ));
    assert_eq!(op.status().left.accepted_rows, 0);
    assert_eq!(op.status().state_rows, 1);
    assert_eq!(op.status().state_limit_failures, 1);
    assert!(op.status().state_bytes > 0);
}

#[tokio::test]
async fn repeated_snapshot_restore_keeps_pending_and_terminal_does_not_flush() {
    use calc_flow::Epoch;
    let mut op = operator(10);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", batch(&[("A", 100, 1, 5)]), &cx, &mut out)
        .await
        .unwrap();
    op.process_data("left", batch(&[("A", 105, 1, 8)]), &cx, &mut out)
        .await
        .unwrap();
    let first = op.checkpoint(Epoch::INITIAL).unwrap();
    assert!(!first.segments.is_empty());
    let second = op.checkpoint(Epoch::INITIAL).unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &first.segments.values().next().unwrap().bytes_arc(),
        &second.segments.values().next().unwrap().bytes_arc()
    ));
    let mut restored = operator(10);
    restored.restore(&first).unwrap();
    let repeated = restored.checkpoint(Epoch::INITIAL).unwrap();
    let mut final_op = operator(10);
    final_op.restore(&repeated).unwrap();
    final_op.on_end(&cx, &mut out).await.unwrap();
    assert_eq!(out.drain("output").len(), 1);
    assert_eq!(final_op.status().matched_rows, 1);
    let terminal = final_op.checkpoint(Epoch::INITIAL).unwrap();
    restored.restore(&terminal).unwrap();
    restored.on_end(&cx, &mut out).await.unwrap();
    assert!(out.drain("output").is_empty());
    assert_eq!(restored.status().emitted_left_rows, 1);
}

#[tokio::test]
async fn wide_admission_reserves_workspace_before_committing_state() {
    use calc_flow::{
        AsofStateLimits, Batch, BatchMetadata, StreamAsofJoinOperator, StreamAsofJoinSpec,
    };
    use datafusion::arrow::{
        array::{Int64Array, StringArray, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use std::{sync::Arc, time::Duration};
    let mut fields = asof_support::schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[3] = Field::new("value", DataType::Utf8, false);
    let schema = Arc::new(Schema::new(fields));
    let template = asof_support::spec(10);
    let spec = StreamAsofJoinSpec::new(
        template.left().clone(),
        template.right().clone(),
        Duration::from_micros(10),
        AsofStateLimits::new(100, 40_000).unwrap(),
    )
    .unwrap();
    let mut op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    // The row must genuinely exceed the 40 KB workspace limit under the
    // allocation-free charge, which tracks the actual encoded bytes.
    let input = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A"])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![1])),
                    Arc::new(StringArray::from(vec!["x".repeat(48_000)])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    let result = op.process_data("right", input.clone(), &cx, &mut out).await;
    assert!(matches!(
        result,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    ));
    assert_eq!(op.status().right.accepted_rows, 0);
    assert_eq!(op.status().state_rows, 0);
    assert_eq!(op.status().workspace_limit_failures, 1);
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, None),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(101))),
        ),
    ]));
    let cx = StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress);
    assert!(matches!(
        op.process_data("right", input, &cx, &mut out).await,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofLateRow,
            ..
        })
    ));
    assert_eq!(op.status().right.late_rows, 1);
    assert_eq!(op.status().workspace_limit_failures, 1);
}

#[tokio::test]
async fn repeated_wide_candidate_is_finalized_in_bounded_chunks() {
    use calc_flow::{
        AsofStateLimits, Batch, BatchMetadata, StreamAsofJoinOperator, StreamAsofJoinSpec,
    };
    use datafusion::arrow::{
        array::{Int64Array, StringArray, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use std::{sync::Arc, time::Duration};
    let mut fields = asof_support::schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[3] = Field::new("value", DataType::Utf8, false);
    let schema = Arc::new(Schema::new(fields));
    let template = asof_support::spec(10);
    let spec = StreamAsofJoinSpec::new(
        template.left().clone(),
        template.right().clone(),
        Duration::from_micros(10),
        AsofStateLimits::new(1000, 1024 * 1024).unwrap(),
    )
    .unwrap();
    let mut op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let input = |time, sequence, value: &str| {
        Batch::table(
            vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StringArray::from(vec!["A"])),
                        Arc::new(TimestampMicrosecondArray::from(vec![time]).with_timezone("UTC")),
                        Arc::new(Int64Array::from(vec![sequence])),
                        Arc::new(StringArray::from(vec![value])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    };
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input(100, 1, &"x".repeat(16_384)), &cx, &mut out)
        .await
        .unwrap();
    for sequence in 0..100 {
        op.process_data("left", input(105, sequence, ""), &cx, &mut out)
            .await
            .unwrap();
    }
    op.on_end(&cx, &mut out).await.unwrap();
    let messages = out.drain("output");
    assert!(
        messages.len() > 1,
        "one repeated candidate chunk would exceed its workspace"
    );
    assert_eq!(
        messages
            .iter()
            .map(|message| message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>())
            .sum::<usize>(),
        100
    );
    assert_eq!(op.status().matched_rows, 100);
    assert_eq!(op.status().state_rows, 0);
}

#[tokio::test]
async fn failed_restore_is_atomic_and_reset_does_not_mutate_shared_segments() {
    use calc_flow::{Epoch, StateSegment};
    let mut op = operator(10);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", batch(&[("A", 100, 1, 5)]), &cx, &mut out)
        .await
        .unwrap();
    let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    let original = op.status();
    let mut damaged = snapshot.clone();
    damaged
        .inline_metadata
        .insert("kind".into(), serde_json::json!("stream_join"));
    assert!(matches!(
        op.restore(&damaged),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(op.status(), original);
    damaged = snapshot.clone();
    let key = damaged.segments.keys().next().unwrap().clone();
    damaged
        .segments
        .insert(key, StateSegment::new(b"CFASOF01".to_vec()));
    assert!(matches!(
        op.restore(&damaged),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(op.status(), original);
    op.reset().unwrap();
    assert_eq!(op.status().state_rows, 0);
    op.restore(&snapshot).unwrap();
    assert_eq!(op.status().retained_right_rows, 1);
}

#[tokio::test]
async fn malformed_ipc_body_length_is_rejected_before_body_allocation() {
    use calc_flow::{Epoch, StateSegment};
    let mut op = operator(10);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", batch(&[("A", 105, 1, 8)]), &cx, &mut out)
        .await
        .unwrap();
    let mut snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    let segment_id = snapshot
        .segments
        .keys()
        .find(|name| name.starts_with("asof-batch-"))
        .unwrap()
        .clone();
    let mut bytes = snapshot.segments[&segment_id].bytes().to_vec();
    let mut cursor = 0;
    loop {
        assert_eq!(&bytes[cursor..cursor + 4], &[255; 4]);
        let metadata_length =
            u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let start = cursor + 8;
        let message =
            datafusion::arrow::ipc::root_as_message(&bytes[start..start + metadata_length])
                .unwrap();
        if message.header_type() == datafusion::arrow::ipc::MessageHeader::RecordBatch {
            let position = start
                + message._tab.loc()
                + usize::from(
                    message
                        ._tab
                        .vtable()
                        .get(datafusion::arrow::ipc::Message::VT_BODYLENGTH),
                );
            bytes[position..position + 8].copy_from_slice(&(16_i64 * 1024 * 1024).to_le_bytes());
            break;
        }
        cursor = start + metadata_length + usize::try_from(message.bodyLength()).unwrap();
    }
    snapshot
        .segments
        .insert(segment_id, StateSegment::new(bytes));
    let original = op.status();
    let allocations = allocation_counter::measure(|| {
        assert!(matches!(
            op.restore(&snapshot),
            Err(CalcFlowError::CheckpointMismatch { .. })
        ));
    });
    assert!(
        allocations.bytes_max < 1024 * 1024,
        "declared body length allocated before framing validation: {allocations:?}"
    );
    assert_eq!(op.status(), original);
}

#[tokio::test]
async fn tiny_slice_does_not_retain_its_large_utf8_backing_allocation() {
    use calc_flow::{
        AsofStateLimits, Batch, BatchMetadata, StreamAsofJoinOperator, StreamAsofJoinSpec,
    };
    use datafusion::arrow::{
        array::{Int64Array, StringArray, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use std::{sync::Arc, time::Duration};
    let mut fields = asof_support::schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[3] = Field::new("value", DataType::Utf8, false);
    let schema = Arc::new(Schema::new(fields));
    let template = asof_support::spec(10);
    let spec = StreamAsofJoinSpec::new(
        template.left().clone(),
        template.right().clone(),
        Duration::from_micros(10),
        AsofStateLimits::new(100, 65_536).unwrap(),
    )
    .unwrap();
    let mut op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let input = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["A", "A", "A"])),
            Arc::new(TimestampMicrosecondArray::from(vec![99, 100, 101]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![
                "x".repeat(4 * 1024 * 1024),
                "z".into(),
                "x".repeat(4 * 1024 * 1024),
            ])),
        ],
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data(
        "right",
        Batch::table(vec![input.slice(1, 1)], BatchMetadata::default()).unwrap(),
        &cx,
        &mut out,
    )
    .await
    .unwrap();
    assert_eq!(op.status().right.accepted_rows, 1);
    assert!(op.status().state_bytes < 16_384);
}

#[tokio::test]
async fn terminal_restore_rejects_new_input() {
    use calc_flow::Epoch;
    let mut op = operator(10);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.on_end(&cx, &mut out).await.unwrap();
    let checkpoint = op.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = operator(10);
    restored.restore(&checkpoint).unwrap();
    assert!(matches!(
        restored
            .process_data("left", batch(&[("A", 105, 1, 8)]), &cx, &mut out)
            .await,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofProtocolError,
            ..
        })
    ));
    assert_eq!(restored.status().left.accepted_rows, 0);
}

#[tokio::test]
async fn restored_identity_only_encoding_is_validated_without_payload() {
    use calc_flow::{Epoch, StateSegment};
    let mut op = operator(10);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let initial = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", batch(&[("A", 100, 1, 5)]), &initial, &mut out)
        .await
        .unwrap();
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1000))),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(0))),
        ),
    ]));
    let cx = StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress);
    op.on_watermark(EventTime::from_micros(0), &cx, &mut out)
        .await
        .unwrap();
    assert_eq!(op.status().identity_only_rows, 1);
    let mut snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    let key = "asof-index-v4";
    let mut bytes = snapshot.segments[key].bytes().to_vec();
    // Corrupt the inline key marker without an Arrow payload.
    assert_eq!(&bytes[..8], b"CFASOF04");
    assert_eq!(&bytes[88..96], &[0; 8]);
    assert_eq!(bytes[96], 0);
    assert_eq!(bytes[98], 2);
    bytes[98] = 0;
    snapshot
        .segments
        .insert(key.into(), StateSegment::new(bytes));
    let previous = op.status();
    assert!(matches!(
        op.restore(&snapshot),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(op.status(), previous);
}

#[tokio::test]
async fn empty_input_and_eof_require_no_identity_state_budget() {
    use calc_flow::{AsofStateLimits, StreamAsofJoinOperator, StreamAsofJoinSpec};
    let template = asof_support::spec(0);
    let spec = StreamAsofJoinSpec::new(
        template.left().clone(),
        template.right().clone(),
        std::time::Duration::ZERO,
        AsofStateLimits::new(1, 1).unwrap(),
    )
    .unwrap();
    let mut op =
        StreamAsofJoinOperator::new("asof", asof_support::schema(), asof_support::schema(), spec)
            .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", batch(&[]), &cx, &mut out)
        .await
        .unwrap();
    op.on_end(&cx, &mut out).await.unwrap();
    assert_eq!(op.status().state_bytes, 0);
    assert!(out.drain("output").is_empty());
    let checkpoint = op.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    op.reset().unwrap();
    op.restore(&checkpoint).unwrap();
    op.on_end(&cx, &mut out).await.unwrap();
    assert!(out.drain("output").is_empty());
}

#[test]
fn schema_metadata_is_reserved_before_encoding_and_after_duplicate_validation() {
    use calc_flow::{
        AsofStateLimits, Batch, BatchMetadata, StreamAsofJoinOperator, StreamAsofJoinSpec,
    };
    use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};
    use std::{collections::HashMap, sync::Arc, time::Duration};
    let source = batch(&[("A", 100, 1, 5)]);
    let source = &source.table_payload().unwrap().batches()[0];
    let schema = Arc::new(Schema::new_with_metadata(
        source.schema().fields().clone(),
        HashMap::from([("note".into(), "x".repeat(2 * 1024 * 1024))]),
    ));
    let record = RecordBatch::try_new(schema.clone(), source.columns().to_vec()).unwrap();
    let input = Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap();
    let duplicate = Batch::table(vec![record.clone(), record], BatchMetadata::default()).unwrap();
    let template = asof_support::spec(10);
    let spec = StreamAsofJoinSpec::new(
        template.left().clone(),
        template.right().clone(),
        Duration::from_micros(10),
        AsofStateLimits::new(100, 1024 * 1024).unwrap(),
    )
    .unwrap();
    let mut op = StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let allocations = allocation_counter::measure(|| {
        let error = runtime
            .block_on(op.process_data("right", input, &cx, &mut out))
            .unwrap_err();
        assert!(matches!(
            error,
            CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
                ..
            }
        ));
    });
    assert!(
        allocations.bytes_max < 1024 * 1024,
        "schema FlatBuffer allocated before reservation: {allocations:?}"
    );
    let error = runtime
        .block_on(op.process_data("right", duplicate, &cx, &mut out))
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        }
    ));
}

#[tokio::test]
async fn duplicate_identity_precedes_even_unavailable_identity_workspace() {
    use calc_flow::{AsofStateLimits, StreamAsofJoinOperator, StreamAsofJoinSpec};
    let template = asof_support::spec(0);
    let spec = StreamAsofJoinSpec::new(
        template.left().clone(),
        template.right().clone(),
        std::time::Duration::ZERO,
        AsofStateLimits::new(10, 1).unwrap(),
    )
    .unwrap();
    let mut op =
        StreamAsofJoinOperator::new("asof", asof_support::schema(), asof_support::schema(), spec)
            .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    let error = op
        .process_data(
            "right",
            batch(&[("A", 100, 1, 5), ("A", 100, 1, 6)]),
            &cx,
            &mut out,
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        }
    ));
    assert_eq!(op.status().right.duplicate_rows, 1);
    assert_eq!(op.status().workspace_limit_failures, 0);
}

#[tokio::test]
async fn malformed_ipc_schema_is_rejected_before_arrow_schema_conversion() {
    use calc_flow::{Epoch, StateSegment};
    let mut op = operator(10);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", batch(&[("A", 105, 1, 8)]), &cx, &mut out)
        .await
        .unwrap();
    let mut snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    let segment_id = snapshot
        .segments
        .keys()
        .find(|name| name.starts_with("asof-batch-"))
        .unwrap()
        .clone();
    let mut bytes = snapshot.segments[&segment_id].bytes().to_vec();
    let start = 8;
    let message = datafusion::arrow::ipc::root_as_message(&bytes[start..]).unwrap();
    let schema = message.header_as_schema().unwrap();
    let table = start + schema._tab.loc();
    let distance = i32::from_le_bytes(bytes[table..table + 4].try_into().unwrap());
    let vtable = usize::try_from(i64::try_from(table).unwrap() - i64::from(distance)).unwrap();
    let entry = vtable + usize::from(datafusion::arrow::ipc::Schema::VT_FIELDS);
    bytes[entry..entry + 2].fill(0);
    snapshot
        .segments
        .insert(segment_id, StateSegment::new(bytes));
    let before = op.status();
    assert!(matches!(
        op.restore(&snapshot),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(op.status(), before);
}

#[test]
fn nested_payload_without_bounded_materialization_accounting_is_rejected() {
    use calc_flow::StreamAsofJoinOperator;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    let fields = asof_support::schema()
        .fields()
        .iter()
        .take(3)
        .cloned()
        .chain([Arc::new(Field::new(
            "value",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Null, true)),
                1_000_000,
            ),
            false,
        ))])
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    assert!(matches!(
        StreamAsofJoinOperator::new("asof", schema.clone(), schema, asof_support::spec(10)),
        Err(CalcFlowError::InvalidArgument { .. })
    ));
}

#[tokio::test]
async fn unmatched_fixed_binary_output_respects_workspace_limit() {
    use calc_flow::{AsofStateLimits, StreamAsofJoinOperator, StreamAsofJoinSpec};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    let left = asof_support::schema();
    let right = Arc::new(Schema::new(
        left.fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain([Field::new(
                "wide",
                DataType::FixedSizeBinary(256 * 1024),
                true,
            )])
            .collect::<Vec<_>>(),
    ));
    let template = asof_support::spec(0);
    let spec = StreamAsofJoinSpec::new(
        template.left().clone(),
        template.right().clone(),
        std::time::Duration::ZERO,
        AsofStateLimits::new(10, 128 * 1024).unwrap(),
    )
    .unwrap();
    let mut op = StreamAsofJoinOperator::new("asof", left, right, spec).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", batch(&[("A", 100, 1, 8)]), &context, &mut output)
        .await
        .unwrap();
    assert!(matches!(
        op.on_end(&context, &mut output).await,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    ));
    assert!(output.drain("output").is_empty());
    assert_eq!(op.status().pending_left_rows, 1);
}

fn flat_payload_types() -> Vec<datafusion::arrow::datatypes::DataType> {
    use datafusion::arrow::datatypes::{DataType, IntervalUnit, TimeUnit};
    vec![
        DataType::Null,
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float16,
        DataType::Float32,
        DataType::Float64,
        DataType::Date32,
        DataType::Date64,
        DataType::Time32(TimeUnit::Second),
        DataType::Time32(TimeUnit::Millisecond),
        DataType::Time64(TimeUnit::Microsecond),
        DataType::Time64(TimeUnit::Nanosecond),
        DataType::Duration(TimeUnit::Microsecond),
        DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
        DataType::Interval(IntervalUnit::YearMonth),
        DataType::Interval(IntervalUnit::DayTime),
        DataType::Interval(IntervalUnit::MonthDayNano),
        DataType::Decimal32(9, 2),
        DataType::Decimal64(18, 2),
        DataType::Decimal128(38, 2),
        DataType::Decimal256(76, 2),
        DataType::Utf8,
        DataType::LargeUtf8,
        DataType::Binary,
        DataType::LargeBinary,
        DataType::FixedSizeBinary(7),
    ]
}

fn sample_flat_payload_column(
    data_type: &datafusion::arrow::datatypes::DataType,
) -> datafusion::arrow::array::ArrayRef {
    use datafusion::arrow::datatypes::{
        DataType, IntervalDayTime, IntervalMonthDayNano, IntervalUnit, TimeUnit, i256,
    };
    use datafusion::scalar::ScalarValue;
    let value = match data_type {
        DataType::Null => ScalarValue::Null,
        DataType::Boolean => ScalarValue::Boolean(Some(true)),
        DataType::Int8 => ScalarValue::Int8(Some(-8)),
        DataType::Int16 => ScalarValue::Int16(Some(-16)),
        DataType::Int32 => ScalarValue::Int32(Some(-32)),
        DataType::Int64 => ScalarValue::Int64(Some(-64)),
        DataType::UInt8 => ScalarValue::UInt8(Some(8)),
        DataType::UInt16 => ScalarValue::UInt16(Some(16)),
        DataType::UInt32 => ScalarValue::UInt32(Some(32)),
        DataType::UInt64 => ScalarValue::UInt64(Some(64)),
        DataType::Float16 => ScalarValue::Float16(Some(3_i8.into())),
        DataType::Float32 => ScalarValue::Float32(Some(3.25)),
        DataType::Float64 => ScalarValue::Float64(Some(6.5)),
        DataType::Date32 => ScalarValue::Date32(Some(20_000)),
        DataType::Date64 => ScalarValue::Date64(Some(1_728_000_000)),
        DataType::Time32(TimeUnit::Second) => ScalarValue::Time32Second(Some(123)),
        DataType::Time32(TimeUnit::Millisecond) => ScalarValue::Time32Millisecond(Some(123_456)),
        DataType::Time64(TimeUnit::Microsecond) => {
            ScalarValue::Time64Microsecond(Some(123_456_789))
        }
        DataType::Time64(TimeUnit::Nanosecond) => ScalarValue::Time64Nanosecond(Some(123_456_789)),
        DataType::Duration(TimeUnit::Microsecond) => ScalarValue::DurationMicrosecond(Some(777)),
        DataType::Timestamp(TimeUnit::Nanosecond, timezone) => {
            ScalarValue::TimestampNanosecond(Some(123_456_789), timezone.clone())
        }
        DataType::Interval(IntervalUnit::YearMonth) => ScalarValue::IntervalYearMonth(Some(14)),
        DataType::Interval(IntervalUnit::DayTime) => {
            ScalarValue::IntervalDayTime(Some(IntervalDayTime::default()))
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            ScalarValue::IntervalMonthDayNano(Some(IntervalMonthDayNano::default()))
        }
        DataType::Decimal32(precision, scale) => {
            ScalarValue::Decimal32(Some(123), *precision, *scale)
        }
        DataType::Decimal64(precision, scale) => {
            ScalarValue::Decimal64(Some(1234), *precision, *scale)
        }
        DataType::Decimal128(precision, scale) => {
            ScalarValue::Decimal128(Some(12_345), *precision, *scale)
        }
        DataType::Decimal256(precision, scale) => {
            ScalarValue::Decimal256(Some(i256::default()), *precision, *scale)
        }
        DataType::Utf8 => ScalarValue::Utf8(Some("alpha".into())),
        DataType::LargeUtf8 => ScalarValue::LargeUtf8(Some("long-alpha".into())),
        DataType::Binary => ScalarValue::Binary(Some(b"bytes".to_vec())),
        DataType::LargeBinary => ScalarValue::LargeBinary(Some(b"large-bytes".to_vec())),
        DataType::FixedSizeBinary(7) => ScalarValue::FixedSizeBinary(7, Some(b"payload".to_vec())),
        _ => panic!("missing sample for {data_type:?}"),
    };
    value.to_array().unwrap()
}

#[tokio::test]
async fn flat_payload_types_roundtrip_through_checkpoint_and_arrow_output() {
    use calc_flow::{Batch, BatchMetadata, Epoch, StreamAsofJoinOperator};
    use datafusion::arrow::{
        datatypes::{Field, Schema},
        record_batch::RecordBatch,
    };
    use std::sync::Arc;
    let types = flat_payload_types();
    let base = batch(&[("A", 100, -1, 8)]);
    let record = &base.table_payload().unwrap().batches()[0];
    let fields = record
        .schema()
        .fields()
        .iter()
        .cloned()
        .chain(types.iter().enumerate().map(|(index, data_type)| {
            Arc::new(Field::new(
                format!("payload_{index}"),
                data_type.clone(),
                true,
            ))
        }))
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let columns = record
        .columns()
        .iter()
        .cloned()
        .chain(types.iter().map(sample_flat_payload_column))
        .collect();
    let input = Batch::table(
        vec![RecordBatch::try_new(schema.clone(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut op =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema, asof_support::spec(10))
            .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    for side in ["left", "right"] {
        op.process_data(side, input.clone(), &cx, &mut out)
            .await
            .unwrap();
    }
    let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    op.reset().unwrap();
    op.restore(&snapshot).unwrap();
    op.on_end(&cx, &mut out).await.unwrap();
    let emitted = out.drain("output");
    assert_eq!(emitted.len(), 1);
    let output = &emitted[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    let source = &input.table_payload().unwrap().batches()[0];
    let source_fields = source.num_columns();
    for (index, data_type) in types.iter().enumerate() {
        let column = source.column(4 + index).to_data();
        assert_eq!(
            output.column(4 + index).to_data(),
            column,
            "left {data_type:?}"
        );
        assert_eq!(
            output.column(source_fields + 4 + index).to_data(),
            column,
            "right {data_type:?}"
        );
    }
    assert_eq!(op.status().matched_rows, 1);
    assert_eq!(op.status().state_bytes, 0);
}

#[tokio::test]
async fn a_second_ipc_schema_is_rejected_before_arrow_conversion() {
    use calc_flow::{Epoch, StateSegment};
    let mut op = operator(10);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", batch(&[("A", 105, 1, 8)]), &cx, &mut out)
        .await
        .unwrap();
    let mut snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    let segment_id = snapshot
        .segments
        .keys()
        .find(|name| name.starts_with("asof-batch-"))
        .unwrap()
        .clone();
    let mut bytes = snapshot.segments[&segment_id].bytes().to_vec();
    let schema_length = 8 + u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut malicious = bytes[..schema_length].to_vec();
    let message = datafusion::arrow::ipc::root_as_message(&malicious[8..]).unwrap();
    let schema = message.header_as_schema().unwrap();
    let table = 8 + schema._tab.loc();
    let distance = i32::from_le_bytes(malicious[table..table + 4].try_into().unwrap());
    let vtable = usize::try_from(i64::try_from(table).unwrap() - i64::from(distance)).unwrap();
    let entry = vtable + usize::from(datafusion::arrow::ipc::Schema::VT_FIELDS);
    malicious[entry..entry + 2].fill(0);
    bytes.splice(schema_length..schema_length, malicious);
    snapshot
        .segments
        .insert(segment_id, StateSegment::new(bytes));
    let before = op.status();
    assert!(matches!(
        op.restore(&snapshot),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(op.status(), before);
}

#[test]
fn invalid_flat_payload_parameters_are_rejected_at_declaration() {
    use calc_flow::StreamAsofJoinOperator;
    use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use std::sync::Arc;
    for data_type in [
        DataType::Time32(TimeUnit::Microsecond),
        DataType::Time64(TimeUnit::Second),
        DataType::FixedSizeBinary(-1),
    ] {
        let fields = asof_support::schema()
            .fields()
            .iter()
            .take(3)
            .cloned()
            .chain([Arc::new(Field::new("value", data_type.clone(), true))])
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(fields));
        assert!(
            StreamAsofJoinOperator::new("asof", schema.clone(), schema, asof_support::spec(10))
                .is_err(),
            "accepted {data_type:?}"
        );
    }
}

#[test]
fn malformed_metadata_does_not_copy_unbounded_error_values() {
    use calc_flow::Epoch;
    let mut op = operator(10);
    let mut snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    snapshot.inline_metadata.get_mut("metrics").unwrap()["state_rows"] =
        serde_json::Value::String("x".repeat(2 * 1024 * 1024));
    let allocation = allocation_counter::measure(|| {
        assert!(matches!(
            op.restore(&snapshot),
            Err(CalcFlowError::CheckpointMismatch { .. })
        ));
    });
    assert!(
        allocation.bytes_max < 16_384,
        "malformed metadata copied into an error: {allocation:?}"
    );
    assert_eq!(op.status().state_rows, 0);
}
