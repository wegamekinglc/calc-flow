use super::*;

fn progress_context(
    job: &StreamJobContext,
    watermark: i64,
    max_rows: usize,
) -> StreamOperatorContext<'_> {
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(
                IngressState::Active,
                Some(EventTime::from_micros(watermark)),
            ),
        ),
        (
            "right".into(),
            IngressProgress::new(
                IngressState::Active,
                Some(EventTime::from_micros(watermark)),
            ),
        ),
    ]));
    StreamOperatorContext::for_task(
        job,
        "asof",
        None,
        progress,
        EdgeBudget {
            max_rows,
            ..EdgeBudget::default()
        },
        Arc::new(LateMetrics),
    )
}

fn right_sequences(messages: &[crate::StreamMessage]) -> Vec<Option<i64>> {
    messages
        .iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .iter()
                .flat_map(|record| {
                    record
                        .column_by_name("right__seq")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .iter()
                })
        })
        .collect()
}

#[tokio::test]
async fn test_watermark_finds_a_sorted_asof_prefix_without_scanning_every_row() {
    let (mut operator, _) = fixture();
    let schema = operator.schemas[0].clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let batch = |rows: &[(&str, i64, i64)]| {
        Batch::table(vec![indexed_input(&schema, rows)], BatchMetadata::default()).unwrap()
    };
    operator
        .process_data("right", batch(&[("A", 31, 999)]), &context, &mut output)
        .await
        .unwrap();
    let rows = (0..96)
        .rev()
        .map(|sequence| ("A", sequence / 2, sequence))
        .collect::<Vec<_>>();
    for run in rows.chunks(48) {
        operator
            .process_data("left", batch(run), &context, &mut output)
            .await
            .unwrap();
    }
    columnar::take_finality_probes();
    operator
        .on_ingress_progress_with_output("left", &progress_context(&job, 32, 80), &mut output)
        .await
        .unwrap();
    let probes = columnar::take_finality_probes();
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    let data = messages[0].as_data().unwrap().table_payload().unwrap();
    let sequences = data.batches()[0]
        .column_by_name("left__seq")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(sequences.values().as_ref(), (0..64).collect::<Vec<_>>());
    assert_eq!(
        right_sequences(&messages),
        (0..64)
            .map(|row| (row >= 62).then_some(999))
            .collect::<Vec<_>>()
    );
    assert_eq!(operator.status.pending_left_rows, 32);
    println!("ASOF readiness timestamp probes for 64 ready of 96: {probes}");
    assert!(
        probes <= 12,
        "expected a bounded search, got {probes} probes"
    );
}

#[tokio::test]
async fn test_columnar_checkpoint_restores_partial_prefix_and_continues() {
    let (mut operator, _) = fixture();
    let schema = operator.schemas[0].clone();
    let right = Batch::table(
        vec![indexed_input(
            &schema,
            &[("A", 5, 1), ("A", 5, 2), ("A", 10, 1)],
        )],
        BatchMetadata::default(),
    )
    .unwrap();
    let left = Batch::table(
        vec![indexed_input(
            &schema,
            &[("A", 5, 10), ("A", 10, 10), ("A", 12, 10)],
        )],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let admission = StreamOperatorContext::for_task(
        &job,
        "asof",
        None,
        IngressProgressSnapshot::default(),
        EdgeBudget {
            max_rows: 1,
            ..EdgeBudget::default()
        },
        Arc::new(LateMetrics),
    );
    let mut first = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right, &admission, &mut first)
        .await
        .unwrap();
    operator
        .process_data("left", left, &admission, &mut first)
        .await
        .unwrap();
    let progress = progress_context(&job, 6, 1);
    operator
        .on_ingress_progress_with_output("left", &progress, &mut first)
        .await
        .unwrap();
    let first_rows = first.drain("output");
    assert_eq!(first_rows.len(), 1);
    assert_eq!(right_sequences(&first_rows), [Some(2)]);
    assert_eq!(operator.status.pending_left_rows, 2);

    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(
        snapshot
            .inline_metadata
            .get("layout_version")
            .and_then(serde_json::Value::as_u64),
        Some(11)
    );
    let snapshot_capacity = snapshot
        .segments
        .values()
        .map(|segment| segment.bytes_arc().capacity())
        .sum::<usize>();
    assert!(u64::try_from(snapshot_capacity).unwrap() <= operator.status.state_bytes);
    let mut restored = StreamAsofJoinOperator::new(
        "asof",
        schema.clone(),
        schema.clone(),
        operator.spec.clone(),
    )
    .unwrap();
    restored.restore(&snapshot).unwrap();
    assert_eq!(
        restored.current_inventory().unwrap().bytes,
        restored.status.state_bytes
    );
    assert_eq!(restored.checkpoint(Epoch::INITIAL).unwrap(), snapshot);

    let later_right = Batch::table(
        vec![indexed_input(&schema, &[("A", 12, 9)])],
        BatchMetadata::default(),
    )
    .unwrap();
    restored
        .process_data("right", later_right, &admission, &mut first)
        .await
        .unwrap();
    let mut remaining = EdgeCollector::new(restored.output_ports().to_vec());
    restored.on_end(&admission, &mut remaining).await.unwrap();
    let remaining = remaining.drain("output");
    assert_eq!(right_sequences(&remaining), [Some(1), Some(9)]);
    assert_eq!(
        remaining
            .iter()
            .map(|message| message.as_data().unwrap().metadata().sequence())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    assert_eq!(restored.status.emitted_left_rows, 3);
    assert_eq!(restored.status.matched_rows, 3);
    assert_eq!(restored.status.pending_left_rows, 0);
    assert_eq!(restored.next_output_sequence, 3);
}

#[tokio::test]
async fn test_columnar_multirecord_late_drop_restores_payload_only_history() {
    let (template, _) = fixture();
    let schema = template.schemas[0].clone();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(5),
        template.spec.limits(),
    )
    .unwrap()
    .with_late_policy(AsofLatePolicy::Drop);
    let mut operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let watermark = progress_context(&job, 5, 1);
    let right = Batch::table(
        vec![
            indexed_input(&schema, &[("A", 4, 9), ("A", 5, 1)]),
            indexed_input(&schema, &[("A", 10, 2)]),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let left = Batch::table(
        vec![
            indexed_input(&schema, &[("A", 4, 9), ("A", 5, 1)]),
            indexed_input(&schema, &[("A", 9, 1)]),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right, &watermark, &mut output)
        .await
        .unwrap();
    operator
        .process_data("left", left, &watermark, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status.right.accepted_rows, 2);
    assert_eq!(operator.status.right.late_rows, 1);
    assert_eq!(operator.status.left.accepted_rows, 2);
    assert_eq!(operator.status.left.late_rows, 1);

    let closed = progress_context(&job, 6, 1);
    operator
        .on_ingress_progress_with_output("left", &closed, &mut output)
        .await
        .unwrap();
    assert_eq!(right_sequences(&output.drain("output")), [Some(1)]);
    assert_eq!(operator.status.retained_right_rows, 2);
    assert_eq!(operator.status.identity_only_rows, 0);
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema, operator.spec.clone()).unwrap();
    restored.restore(&snapshot).unwrap();
    assert_eq!(
        restored.current_inventory().unwrap().bytes,
        restored.status.state_bytes
    );
    assert_eq!(restored.checkpoint(Epoch::INITIAL).unwrap(), snapshot);
    let mut remaining = EdgeCollector::new(restored.output_ports().to_vec());
    restored.on_end(&watermark, &mut remaining).await.unwrap();
    assert_eq!(right_sequences(&remaining.drain("output")), [Some(1)]);
    assert_eq!(restored.status.state_rows, 0);
    assert_eq!(restored.status.state_bytes, 0);
}

#[tokio::test]
async fn test_columnar_multirecord_duplicate_installs_no_partial_run() {
    let (mut operator, _) = fixture();
    let schema = operator.schemas[0].clone();
    let input = Batch::table(
        vec![
            indexed_input(&schema, &[("A", 5, 1)]),
            indexed_input(&schema, &[("A", 5, 1)]),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let error = operator
        .process_data("right", input, &context, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        }
    ));
    assert_eq!(operator.status.right.accepted_rows, 0);
    assert_eq!(operator.status.right.duplicate_rows, 1);
    assert_eq!(operator.status.state_rows, 0);
    assert!(
        operator
            .checkpoint(Epoch::INITIAL)
            .unwrap()
            .segments
            .is_empty()
    );
}

#[tokio::test]
async fn test_columnar_output_byte_limit_preserves_pending_state() {
    let (mut operator, batch) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::for_task(
        &job,
        "asof",
        None,
        IngressProgressSnapshot::default(),
        EdgeBudget {
            max_bytes: 1,
            ..EdgeBudget::default()
        },
        Arc::new(LateMetrics),
    );
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", batch, &context, &mut output)
        .await
        .unwrap();
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    let error = operator
        .on_watermark(EventTime::from_micros(101), &context, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofOutputLimitExceeded,
            ..
        }
    ));
    assert_eq!(operator.status.pending_left_rows, 1);
    assert_eq!(operator.status.output_limit_failures, 1);
    assert_eq!(
        operator.checkpoint(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
    assert!(output.drain("output").is_empty());
}

#[tokio::test]
async fn test_columnar_restore_rejects_out_of_range_identity_only_group() {
    let (mut operator, _) = fixture();
    let schema = operator.schemas[0].clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let admission = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for (side, row) in [("right", ("A", 5, 1)), ("left", ("A", 10, 1))] {
        let input = Batch::table(
            vec![indexed_input(&schema, &[row])],
            BatchMetadata::default(),
        )
        .unwrap();
        operator
            .process_data(side, input, &admission, &mut output)
            .await
            .unwrap();
    }
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(11))),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(5))),
        ),
    ]));
    let context = StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress);
    operator
        .on_ingress_progress_with_output("left", &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status.identity_only_rows, 1);
    let mut snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let identity = snapshot
        .segments
        .keys()
        .find(|name| name.contains("-1-") && name.ends_with("-identity"))
        .unwrap()
        .clone();
    let segment = &snapshot.segments[&identity];
    let mut reader = datafusion::arrow::ipc::reader::StreamReader::try_new(
        std::io::Cursor::new(segment.bytes()),
        None,
    )
    .unwrap();
    let record = reader.next().unwrap().unwrap();
    let mut columns = record.columns().to_vec();
    columns[0] = Arc::new(datafusion::arrow::array::UInt32Array::from(vec![u32::MAX]));
    let record = RecordBatch::try_new(record.schema(), columns).unwrap();
    let mut bytes = Vec::new();
    let mut writer =
        datafusion::arrow::ipc::writer::StreamWriter::try_new(&mut bytes, record.schema().as_ref())
            .unwrap();
    writer.write(&record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    snapshot
        .segments
        .insert(identity, crate::StateSegment::new(bytes));
    let mut restored =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema, operator.spec.clone()).unwrap();
    assert!(restored.restore(&snapshot).is_err());
}

#[tokio::test]
async fn test_columnar_watermark_equality_remains_pending() {
    let (mut operator, _) = fixture();
    let schema = operator.schemas[0].clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let admission = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for side in ["right", "left"] {
        operator
            .process_data(
                side,
                Batch::table(
                    vec![indexed_input(&schema, &[("A", 100, 1)])],
                    BatchMetadata::default(),
                )
                .unwrap(),
                &admission,
                &mut output,
            )
            .await
            .unwrap();
    }
    let equal = progress_context(&job, 100, 10);
    (&mut operator as &mut dyn StreamOperator)
        .on_ingress_progress_with_output("left", &equal, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    assert_eq!(operator.status.pending_left_rows, 1);
    let closed = progress_context(&job, 101, 10);
    (&mut operator as &mut dyn StreamOperator)
        .on_ingress_progress_with_output("left", &closed, &mut output)
        .await
        .unwrap();
    assert_eq!(right_sequences(&output.drain("output")), [Some(1)]);
    assert_eq!(operator.status.pending_left_rows, 0);
}

#[tokio::test]
async fn test_columnar_accepted_counter_overflow_installs_no_state() {
    let (mut template, batch) = fixture();
    let mut snapshot = template.checkpoint(Epoch::INITIAL).unwrap();
    let mut metadata = serde_json::to_value(&snapshot.inline_metadata).unwrap();
    let metrics = metadata.pointer_mut("/metrics").unwrap();
    metrics["left"]["accepted_rows"] = serde_json::json!(u64::MAX);
    metrics["emitted_left_rows"] = serde_json::json!(u64::MAX);
    metrics["matched_rows"] = serde_json::json!(u64::MAX);
    metadata["next_output_sequence"] = serde_json::json!(u64::MAX);
    snapshot.inline_metadata = serde_json::from_value(metadata).unwrap();
    let schema = template.schemas[0].clone();
    let mut operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema, template.spec.clone()).unwrap();
    operator.restore(&snapshot).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let error = operator
        .process_data("left", batch, &context, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofCounterOverflow,
            ..
        }
    ));
    assert!(operator.state.is_empty());
    assert_eq!(operator.checkpoint(Epoch::INITIAL).unwrap(), snapshot);
}

#[test]
fn test_columnar_multirecord_preflight_precedes_concatenation() {
    let (template, _) = fixture();
    let mut fields = template.schemas[0]
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields.push(Field::new("value", DataType::Utf8, false));
    let schema = Arc::new(Schema::new(fields));
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::ZERO,
        AsofStateLimits::new(100, 40_000).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let record = |sequence: i64| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["A"])),
                Arc::new(TimestampMicrosecondArray::from(vec![5]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![sequence])),
                Arc::new(StringArray::from(vec!["x".repeat(30_000)])),
            ],
        )
        .unwrap()
    };
    let input = Batch::table(vec![record(1), record(2)], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let allocations = allocation_counter::measure(|| {
        let error = runtime
            .block_on(operator.process_data("right", input, &context, &mut output))
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
        allocations.bytes_max < 40_000,
        "multi-record payload concatenated before refusal: {allocations:?}"
    );
    assert_eq!(operator.status.state_rows, 0);
}

#[test]
fn test_columnar_output_preflight_precedes_materialization() {
    let (template, _) = fixture();
    let mut fields = template.schemas[0]
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields.push(Field::new("value", DataType::Utf8, false));
    let schema = Arc::new(Schema::new(fields));
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(10),
        AsofStateLimits::new(1_000, 1024 * 1024).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let input = |time, sequence, value: String| {
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
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(async {
        operator
            .process_data(
                "right",
                input(100, 1, "x".repeat(16_384)),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        for sequence in 0..100 {
            operator
                .process_data(
                    "left",
                    input(105, sequence, String::new()),
                    &context,
                    &mut output,
                )
                .await
                .unwrap();
        }
    });
    finalize::take_materializations();
    runtime
        .block_on(operator.on_end(&context, &mut output))
        .unwrap();
    let materialized = finalize::take_materializations();
    let emitted = output.drain("output").len();
    assert!(emitted > 1, "fixture must retry a smaller output prefix");
    assert_eq!(materialized, emitted);
}

#[tokio::test]
async fn test_columnar_output_scratch_preflight_precedes_selection() {
    let (mut operator, _) = fixture();
    let schema = operator.schemas[0].clone();
    let rows = (0..16).map(|row| ("A", 100, row)).collect::<Vec<_>>();
    let input = Batch::table(
        vec![indexed_input(&schema, &rows)],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", input, &context, &mut output)
        .await
        .unwrap();
    operator.runtime_pool = Arc::new(GreedyMemoryPool::new(1));
    let progress = progress_context(&job, 101, 16);
    for ended in [false, true] {
        columnar::take_selections();
        columnar::take_finality_probes();
        let result = if ended {
            operator.on_end(&context, &mut output).await
        } else {
            operator
                .on_ingress_progress_with_output("left", &progress, &mut output)
                .await
        };
        assert!(matches!(
            result.unwrap_err(),
            CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
                ..
            }
        ));
        assert_eq!(columnar::take_selections(), 0);
        let probes = columnar::take_finality_probes();
        assert!(
            probes <= if ended { 0 } else { 2 },
            "shrinking the prefix took {probes} probes"
        );
        assert_eq!(operator.status.pending_left_rows, 16);
        assert!(output.drain("output").is_empty());
    }
}
