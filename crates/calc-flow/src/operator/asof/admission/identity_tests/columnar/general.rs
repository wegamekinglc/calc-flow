use super::*;
use crate::{EdgeCollector, OperatorMetadata, StreamOperator};

#[tokio::test]
async fn test_general_batch_admission_avoids_canonical_row_descriptors() {
    let schema = schema(DataType::Int64, DataType::Int64);
    let input = batch(vec![integer_record(&schema, 0, vec![20, 10, 20, 9])]);
    let mut operator = fixture(&schema);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    for index in [0, 1] {
        let side = ValidatedInput {
            index,
            watermark: Some(10),
        };
        let (mut legacy, _) = prepared(&mut operator, &input, side, &context, true).await;
        batch::take_work();
        let (mut admission, _) = prepared(&mut operator, &input, side, &context, false).await;
        assert_eq!(
            batch::take_work(),
            batch::Work {
                records: 1,
                rows: 3,
                ranks: 1,
                sorted_rows: 3,
                sequence_fallbacks: 0,
                canonical_rows: 0,
            }
        );
        assert_eq!(admission.len(), 3);
        assert!(
            admission.rows.is_empty(),
            "general admission expanded {} canonical rows",
            admission.rows.len()
        );
        if index == 0 {
            assert_chunks_equal(
                admission.left_chunks.take().unwrap().pop().unwrap(),
                legacy.left_chunks.take().unwrap().pop().unwrap(),
            );
        } else {
            let mut actual = admission
                .right_columns
                .as_ref()
                .unwrap()
                .identities(&admission.right_capacities)
                .collect::<Vec<_>>();
            actual.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            legacy.rows.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            for ((identity, reference), (old, old_ref)) in actual.iter().zip(&legacy.rows) {
                assert_eq!(identity, old);
                assert_eq!(
                    (reference.batch_index, reference.row, reference.key_index),
                    (old_ref.batch_index, old_ref.row, old_ref.key_index)
                );
            }
        }
    }
    assert_eq!(operator.runtime.pool.reserved(), 0);
    // Preserve the original 4096-row/128KiB inline bound for ordered general
    // keys; introducing worker registration would change tight pool admission.
    let schema = super::schema(DataType::Boolean, DataType::Int64);
    let mut operator = fixture(&schema);
    let input = batch(vec![record(
        &schema,
        Arc::new(datafusion::arrow::array::BooleanArray::from(vec![
            false;
            1_024
        ])),
        Arc::new(Int64Array::from_iter_values(0..1_024)),
        vec![12; 1_024],
    )]);
    let (admission, _) = prepared(
        &mut operator,
        &input,
        ValidatedInput {
            index: 0,
            watermark: None,
        },
        &context,
        false,
    )
    .await;
    assert!(admission.rows.is_empty());
    drop(admission);
    assert_eq!(
        operator.runtime.pool.reserved(),
        0,
        "bounded ordered admission must not register a worker"
    );
}

fn integer_record(schema: &SchemaRef, start: i64, times: Vec<i64>) -> RecordBatch {
    let count = times.len();
    record(
        schema,
        Arc::new(Int64Array::from(vec![7; count])),
        Arc::new(Int64Array::from_iter_values(
            start..start + i64::try_from(count).unwrap(),
        )),
        times,
    )
}

type Row<'a> = (&'a str, i64, i64, i64);

fn composite_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
        Field::new("extra", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, true),
    ]))
}

fn composite_record(rows: &[Row<'_>]) -> RecordBatch {
    RecordBatch::try_new(
        composite_schema(),
        vec![
            Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.0))),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(rows.iter().map(|r| r.2))
                    .with_timezone("UTC"),
            ),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.3))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.1))),
            Arc::new(StringArray::from(
                rows.iter()
                    .enumerate()
                    .map(|(i, r)| (!i.is_multiple_of(2)).then_some(r.0))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn composite_operator(projection: Option<&[usize]>) -> StreamAsofJoinOperator {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into(), "extra".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        std::time::Duration::from_micros(2),
        super::super::super::super::AsofStateLimits::new(100_000, 128 << 20).unwrap(),
    )
    .unwrap()
    .with_late_policy(AsofLatePolicy::Drop);
    let mut operator =
        StreamAsofJoinOperator::new("asof", composite_schema(), composite_schema(), spec).unwrap();
    if let Some(projection) = projection {
        operator.set_output_projection(projection.to_vec()).unwrap();
    }
    operator.status.left.watermark_micros = Some(crate::EventTime::from_micros(10));
    operator.status.right.watermark_micros = Some(crate::EventTime::from_micros(10));
    operator
}

async fn install(
    operator: &mut StreamAsofJoinOperator,
    input: &Batch,
    ingress: &str,
    context: &StreamOperatorContext<'_>,
    legacy: bool,
) {
    let validated = operator.validate_admission(ingress, input).unwrap();
    let (admission, _) = prepared(operator, input, validated, context, legacy).await;
    assert!(admission.rows.is_empty() || legacy);
    operator
        .install_admission(ingress, admission, validated, context)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_general_rank_checkpoint_restore_and_tolerance_match_legacy_and_oracle() {
    let wide = "z-key-outside-inline-非ASCII";
    let right = [
        (wide, 1, 10, 1),
        ("a", -1, 11, 2),
        ("a", -1, 11, 3),
        ("é", 7, 14, 4),
        ("right-only", 0, 12, 5),
    ];
    let left = [
        ("é", 7, 13, 3),
        ("a", -1, 12, 8),
        (wide, 1, 12, 6),
        ("probe-only", 0, 12, 4),
        ("a", -1, 20, 9),
        ("a", i64::MIN, 12, 10),
    ];
    let late = composite_record(&[("late", 0, 8, 0)]);
    let padded = composite_record(&[
        ("pad", 0, 0, 0),
        right[0],
        right[1],
        right[2],
        ("pad", 0, 0, 0),
    ]);
    let right_batch = batch(vec![
        late.slice(0, 0),
        padded.slice(1, 3),
        late.clone(),
        composite_record(&right[3..]),
    ]);
    let left_batch = batch(vec![
        composite_record(&left[..3]),
        late,
        composite_record(&left[3..]),
    ]);
    let before = right_batch.table_payload().unwrap().batches().to_vec();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let expected = sql_oracle(&left, &right).await;
    let mut pools = Vec::new();
    for projection in [None, Some(vec![2, 7]), Some(Vec::new())] {
        pools.extend(
            compare_projection(projection, &left_batch, &right_batch, &expected, &context).await,
        );
    }
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
    assert_eq!(right_batch.table_payload().unwrap().batches(), &before);
}

#[tokio::test]
async fn test_general_rejection_duplicate_priority_cancellation_and_refunds() {
    for ingress in ["left", "right"] {
        for (duplicate, tight, cancelled) in [
            (true, false, false),
            (true, true, false),
            (false, true, false),
            (false, false, true),
        ] {
            for old in [false, true] {
                let mut operator = composite_operator(None);
                let pool = operator.runtime.pool.clone();
                let cancellation = CancellationToken::new();
                let job =
                    StreamJobContext::new(1, "asof", JsonMap::new(), None, cancellation.clone())
                        .with_checkpointing(false);
                let context = StreamOperatorContext::new(&job, "asof", None);
                let mut output = EdgeCollector::new(operator.output_ports().to_vec());
                let seed = batch(vec![composite_record(&[("resident-wide-key", -1, 12, 1)])]);
                install(&mut operator, &seed, ingress, &context, old).await;
                let before = operator.status.clone();
                let reserved = pool.reserved();
                let hold = tight.then(|| {
                    operator
                        .reserve_workspace((128 << 20) - reserved as u64)
                        .unwrap()
                });
                if cancelled {
                    cancellation.cancel();
                }
                let input = if duplicate {
                    batch(vec![
                        composite_record(&[("resident-wide-key", -1, 12, 1), ("new", 0, 13, 2)]),
                        composite_record(&[
                            ("resident-wide-key", -1, 12, 1),
                            ("new", 0, 13, 2),
                            ("new", 0, 13, 2),
                        ]),
                    ])
                } else {
                    batch(vec![composite_record(&[("new-wide-key", 1, 13, 2)])])
                };
                let guard = LegacyGuard::new(old);
                let error = operator
                    .process_data(ingress, input, &context, &mut output)
                    .await
                    .unwrap_err();
                drop(guard);
                assert_general_rejection(&error, &operator, ingress, duplicate, cancelled);
                assert_eq!(operator.status.state_rows, before.state_rows);
                assert_eq!(
                    operator.status.left.accepted_rows,
                    before.left.accepted_rows
                );
                assert_eq!(
                    operator.status.right.accepted_rows,
                    before.right.accepted_rows
                );
                assert_eq!(
                    operator.current_inventory(None).unwrap().bytes,
                    before.state_bytes
                );
                assert!(output.drain("output").is_empty());
                drop(hold);
                assert_eq!(pool.reserved(), reserved);
                drop((operator, output));
                assert_eq!(pool.reserved(), 0);
            }
        }
    }
}

fn assert_general_rejection(
    error: &crate::CalcFlowError,
    operator: &StreamAsofJoinOperator,
    ingress: &str,
    duplicate: bool,
    cancelled: bool,
) {
    if cancelled {
        assert!(matches!(error, crate::CalcFlowError::Cancelled { .. }));
    } else {
        let crate::CalcFlowError::OperatorReason { reason_code, .. } = error else {
            panic!("unexpected error: {error}")
        };
        assert_eq!(
            *reason_code,
            if duplicate {
                StreamingFailureReason::AsofDuplicateIdentity
            } else {
                StreamingFailureReason::AsofWorkspaceLimitExceeded
            }
        );
        let status = if ingress == "left" {
            &operator.status.left
        } else {
            &operator.status.right
        };
        assert_eq!(status.duplicate_rows, if duplicate { 4 } else { 0 });
    }
}

#[tokio::test]
async fn test_general_types_and_explicit_fallbacks() {
    use datafusion::arrow::array::{BooleanArray, Date32Array, Date64Array};
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    for keys in [
        Arc::new(BooleanArray::from(vec![true, false, true])) as ArrayRef,
        Arc::new(Date32Array::from(vec![i32::MAX, i32::MIN, -1])),
        Arc::new(Date64Array::from(vec![i64::MAX, i64::MIN, -1])),
    ] {
        let schema = schema(keys.data_type().clone(), DataType::Int64);
        let input = batch(vec![record(
            &schema,
            keys,
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            vec![12; 3],
        )]);
        for index in [0, 1] {
            let mut operator = fixture(&schema);
            let validated = ValidatedInput {
                index,
                watermark: None,
            };
            let (mut fast, _) = prepared(&mut operator, &input, validated, &context, false).await;
            let (mut legacy, _) = prepared(&mut operator, &input, validated, &context, true).await;
            assert!(fast.rows.is_empty());
            if index == 0 {
                assert_chunks_equal(
                    fast.left_chunks.take().unwrap().pop().unwrap(),
                    legacy.left_chunks.take().unwrap().pop().unwrap(),
                );
            } else {
                operator
                    .install_admission("right", fast, validated, &context)
                    .await
                    .unwrap();
            }
        }
    }
    let schema = schema(DataType::Utf8, DataType::Utf8);
    let mut operator = fixture(&schema);
    let input = batch(vec![record(
        &schema,
        Arc::new(StringArray::from(vec!["wide-key-value"; 2])),
        Arc::new(StringArray::from(vec!["b", "a"])),
        vec![12; 2],
    )]);
    batch::take_work();
    let (admission, _) = prepared(
        &mut operator,
        &input,
        ValidatedInput {
            index: 1,
            watermark: None,
        },
        &context,
        false,
    )
    .await;
    assert_eq!(admission.rows.len(), 2);
    assert_eq!(batch::take_work().sequence_fallbacks, 1);
    for data_type in [DataType::Float64, DataType::Binary] {
        assert!(
            StreamAsofJoinOperator::new(
                "asof",
                super::schema(data_type.clone(), DataType::Int64),
                super::schema(data_type, DataType::Int64),
                fixture(&super::schema(DataType::Int64, DataType::Int64)).spec
            )
            .is_err()
        );
    }
    let nullable = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, true),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
    ]));
    assert!(
        StreamAsofJoinOperator::new(
            "asof",
            nullable.clone(),
            nullable,
            fixture(&super::schema(DataType::Int64, DataType::Int64)).spec
        )
        .is_err()
    );
}

#[tokio::test]
async fn test_general_canonical_sequence_owner_runs_match_legacy() {
    let schema = schema(DataType::Utf8, DataType::Utf8);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let records = (0..2)
        .map(|batch| {
            record(
                &schema,
                Arc::new(StringArray::from(vec!["wide-key-value-非ASCII"; 6])),
                Arc::new(StringArray::from_iter_values(
                    (0..6).map(|row| format!("sequence-{batch}-{row}-outside-inline")),
                )),
                vec![8, 12, 13, 8, 12, 14],
            )
        })
        .collect::<Vec<_>>();
    let input = batch(records);
    let mut pools = Vec::new();
    for ingress in ["left", "right"] {
        let make_operator = || {
            let mut operator = shaped_fixture(
                &schema,
                &["key".into()],
                &["seq".into()],
                AsofLatePolicy::Drop,
            );
            operator.status.left.watermark_micros = Some(crate::EventTime::from_micros(10));
            operator.status.right.watermark_micros = Some(crate::EventTime::from_micros(10));
            operator
        };
        let mut fast = make_operator();
        let mut legacy = make_operator();
        let validated = fast.validate_admission(ingress, &input).unwrap();
        legacy.validate_admission(ingress, &input).unwrap();
        let (fast_admission, _) = prepared(&mut fast, &input, validated, &context, false).await;
        let (legacy_admission, _) = prepared(&mut legacy, &input, validated, &context, true).await;
        assert_eq!(
            fast_admission.rows.len(),
            8,
            "only the canonical ledger boundary materializes rows"
        );
        assert_eq!(
            fast_admission
                .rows
                .iter()
                .map(|(identity, _)| identity)
                .collect::<Vec<_>>(),
            legacy_admission
                .rows
                .iter()
                .map(|(identity, _)| identity)
                .collect::<Vec<_>>()
        );
        fast.install_admission(ingress, fast_admission, validated, &context)
            .await
            .unwrap();
        legacy
            .install_admission(ingress, legacy_admission, validated, &context)
            .await
            .unwrap();
        let snapshot = assert_capture_equal(&mut fast, &mut legacy, &context).await;
        let mut restored = make_operator();
        restored.restore(&snapshot).unwrap();
        pools.extend([&fast, &legacy, &restored].map(|operator| operator.runtime.pool.clone()));
        drop((snapshot, fast, legacy, restored));
    }
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

fn output_sequences(records: &[RecordBatch]) -> Vec<(i64, Option<i64>)> {
    records
        .iter()
        .flat_map(|record| {
            let l = record
                .column_by_name("left__seq")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let r = record
                .column_by_name("right__seq")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..record.num_rows())
                .map(|i| (l.value(i), (!r.is_null(i)).then(|| r.value(i))))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

async fn compare_projection(
    projection: Option<Vec<usize>>,
    left_batch: &Batch,
    right_batch: &Batch,
    expected: &[(i64, Option<i64>)],
    context: &StreamOperatorContext<'_>,
) -> [Arc<dyn MemoryPool>; 3] {
    let mut fast = composite_operator(projection.as_deref());
    let mut legacy = composite_operator(projection.as_deref());
    for (ingress, input) in [("right", right_batch), ("left", left_batch)] {
        install(&mut fast, input, ingress, context, false).await;
        install(&mut legacy, input, ingress, context, true).await;
    }
    assert_eq!(
        fast.state.right.len(),
        4,
        "probe-only keys must not become resident right keys"
    );
    let snapshot = assert_capture_equal(&mut fast, &mut legacy, context).await;
    let mut restored = composite_operator(projection.as_deref());
    restored.restore(&snapshot).unwrap();
    let continuation = batch(vec![composite_record(&[("a", -1, 24, i64::MAX)])]);
    for (operator, old) in [
        (&mut fast, false),
        (&mut legacy, true),
        (&mut restored, false),
    ] {
        for ingress in ["right", "left"] {
            install(operator, &continuation, ingress, context, old).await;
        }
    }
    let continued = assert_capture_equal(&mut fast, &mut legacy, context).await;
    let restored_snapshot = assert_capture_equal(&mut restored, &mut legacy, context).await;
    let mut outputs = Vec::new();
    for operator in [&mut fast, &mut legacy, &mut restored] {
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator.on_end(context, &mut output).await.unwrap();
        outputs.push(output_records(&mut output));
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[0], outputs[2]);
    assert_eq!(
        outputs[0].iter().map(RecordBatch::num_rows).sum::<usize>(),
        expected.len()
    );
    if projection.as_ref().is_none_or(|p| !p.is_empty()) {
        assert_eq!(output_sequences(&outputs[0]), expected);
    }
    let pools = [&fast, &legacy, &restored].map(|op| op.runtime.pool.clone());
    drop((
        snapshot,
        continued,
        restored_snapshot,
        fast,
        legacy,
        restored,
    ));
    pools
}

async fn sql_oracle(left: &[Row<'_>], right: &[Row<'_>]) -> Vec<(i64, Option<i64>)> {
    let continuation = ("a", -1, 24, i64::MAX);
    let left = [left, &[continuation]].concat();
    let right = [right, &[continuation]].concat();
    let session = datafusion::prelude::SessionContext::new();
    session
        .register_batch("left_input", composite_record(&left))
        .unwrap();
    session
        .register_batch("right_input", composite_record(&right))
        .unwrap();
    let records = session.sql(
        "SELECT left__seq, right__seq FROM (
            SELECT l.time, l.key, l.extra, l.seq AS left__seq, r.seq AS right__seq,
                ROW_NUMBER() OVER (PARTITION BY l.seq ORDER BY r.time DESC NULLS LAST, r.seq DESC NULLS LAST) AS rn
            FROM left_input l LEFT JOIN right_input r
            ON l.key = r.key AND l.extra = r.extra AND r.time <= l.time
                AND r.time >= l.time - INTERVAL '2 microseconds'
        ) WHERE rn = 1 ORDER BY time, key, extra, left__seq"
    ).await.unwrap().collect().await.unwrap();
    output_sequences(&records)
}
