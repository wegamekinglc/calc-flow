use super::*;
use crate::{EdgeCollector, OperatorMetadata, StreamOperator};

pub(super) async fn run_cases(service: &TestService) {
    let schema = schema(DataType::Utf8, DataType::UInt64);
    let input = |keys, seq, times| batch(vec![unsigned_record(&schema, keys, seq, times)]);
    let ordered = input(vec!["A", "B"], vec![1, 2], vec![0, 0]);
    let reversed_ids = input(vec!["B", "A"], vec![3, 4], vec![1, 2]);
    let empty = ordered.table_payload().unwrap().batches()[0].slice(0, 0);
    let records = batch(vec![
        empty.clone(),
        ordered.table_payload().unwrap().batches()[0].clone(),
        empty,
        reversed_ids.table_payload().unwrap().batches()[0].clone(),
    ]);
    let sliced = batch(vec![
        string_unsigned_input(1_024)
            .table_payload()
            .unwrap()
            .batches()[0]
            .slice(17, 3),
    ]);
    let cases = vec![
        (string_unsigned_input(4_097), None, None, true),
        (records, None, None, true),
        (sliced, None, None, true),
        (
            input(vec!["A"; 3], vec![0, u64::MAX - 1, u64::MAX], vec![10; 3]),
            None,
            None,
            true,
        ),
        (
            input(vec!["B", "A"], vec![1, 2], vec![0, 0]),
            None,
            None,
            false,
        ),
        (
            input(vec!["long-key-outside-inline"; 2], vec![1, 2], vec![0, 1]),
            None,
            None,
            false,
        ),
        (
            input(vec!["A"; 3], vec![1, 2, 3], vec![9, 10, 11]),
            None,
            Some(10),
            false,
        ),
        (
            input(vec!["A"], vec![2], vec![50]),
            Some(input(vec!["A"; 2], vec![0, 1], vec![100, 0])),
            None,
            false,
        ),
        (
            input(vec!["A"], vec![2], vec![100]),
            Some(input(vec!["A", "B"], vec![1, 1], vec![100, 200])),
            None,
            true,
        ),
    ];
    for checkpointing in [false, true] {
        for (data, seed, watermark, direct) in &cases {
            compare(
                data,
                seed.as_ref(),
                *watermark,
                *direct,
                checkpointing,
                service,
                None,
            )
            .await;
        }
    }
    retained_maxima();
    delta_replay(service).await;
    failure_controls().await;
}

pub(super) async fn projected_cases(input: &Batch, service: &TestService) {
    let original = &input.table_payload().unwrap().batches()[0];
    let sliced = batch(vec![original.slice(17, 3)]);
    for projection in [None, Some(vec![3]), Some(vec![7]), Some(Vec::new())] {
        for data in [input, &sliced] {
            compare(data, None, None, true, true, service, projection.clone()).await;
        }
    }
}

fn retained_maxima() {
    use crate::operator::asof::admission::right::plan;
    let schema = schema(DataType::Utf8, DataType::UInt64);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    for tag in [0, 2] {
        let mut operator = fixture(&schema);
        let seed = unsigned_record(&schema, vec!["A"], vec![1], vec![100]);
        let key = state::encoded_columns(&seed, 0, operator.spec.right().keys()).unwrap();
        let seq = state::encoded_columns(&seed, 0, operator.spec.right().sequence_by()).unwrap();
        let mut bucket = state::RightBucket::with_index_capacities(
            [0, 0, 0, 1, 1],
            state::SequenceKind::Unsigned(8),
        );
        bucket.push_index((100, seq), tag, None);
        operator.state.right.insert(key, bucket);
        for (time, seq, qualifies) in [
            (50, 2, false),
            (100, 0, false),
            (100, 1, false),
            (100, 2, true),
            (200, 0, true),
        ] {
            let input = batch(vec![unsigned_record(
                &schema,
                vec!["A"],
                vec![seq],
                vec![time],
            )]);
            assert_eq!(
                plan(
                    &operator,
                    &input,
                    ValidatedInput {
                        index: 1,
                        watermark: None
                    },
                    &context
                )
                .unwrap()
                .is_some(),
                qualifies,
                "history tag {tag}, time {time}, seq {seq}"
            );
        }
    }
}

async fn delta_replay(service: &TestService) {
    let seed = string_unsigned_input(1_024);
    let schema = seed.table_payload().unwrap().batches()[0].schema();
    let mut fast = fixture(&schema);
    let mut legacy = fixture(&schema);
    let mut restored = fixture(&schema);
    let pools = [&fast, &legacy, &restored].map(|op| op.runtime.pool.clone());
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("direct-right-delta".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    for op in [&mut fast, &mut legacy] {
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        op.process_data("right", seed.clone(), &context, &mut output)
            .await
            .unwrap();
    }
    assert_capture_equal(&mut fast, &mut legacy, &context).await;
    let input = batch(vec![unsigned_record(
        &schema,
        vec!["S000", "S000", "S063"],
        vec![2_048, 2_049, 2_050],
        vec![16; 3],
    )]);
    for (op, old) in [(&mut fast, false), (&mut legacy, true)] {
        let guard = LegacyGuard::new(old);
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        op.process_data("right", input.clone(), &context, &mut output)
            .await
            .unwrap();
        drop(guard);
        assert_eq!(
            op.checkpoint_log.journal.changes().len(),
            3,
            "columnar rows were absent from the delta journal"
        );
    }
    for op in [&mut fast, &mut legacy] {
        op.prepare_checkpoint_async(&context).await.unwrap();
    }
    let epoch = crate::Epoch::new(2).unwrap();
    let snapshot = fast.checkpoint(epoch).unwrap();
    let old = legacy.checkpoint(epoch).unwrap();
    assert_eq!(snapshot.inline_metadata, old.inline_metadata);
    assert_eq!(snapshot.segments, old.segments);
    assert_eq!(
        snapshot.inline_metadata["checkpoint_log"]["frames"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status, fast.status);
    let mut results = Vec::new();
    for op in [&mut fast, &mut legacy, &mut restored] {
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        op.process_data("left", input.clone(), &context, &mut output)
            .await
            .unwrap();
        op.on_end(&context, &mut output).await.unwrap();
        let records = output_records(&mut output);
        let sequences = records
            .iter()
            .flat_map(|record| {
                record
                    .column(5)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(sequences, [2_049, 2_049, 2_050]);
        results.push(records);
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[0], results[2]);
    drop((fast, legacy, restored, results, snapshot, old));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

async fn failure_controls() {
    let seed = string_unsigned_input(1);
    let schema = seed.table_payload().unwrap().batches()[0].schema();
    for (duplicate, tight, cancelled) in [
        (true, false, false),
        (true, true, false),
        (false, true, false),
        (false, false, true),
    ] {
        for old in [false, true] {
            let mut op = fixture(&schema);
            let pool = op.runtime.pool.clone();
            let cancellation = CancellationToken::new();
            let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancellation.clone())
                .with_checkpointing(false);
            let context = StreamOperatorContext::new(&job, "asof", None);
            let mut output = EdgeCollector::new(op.output_ports().to_vec());
            op.process_data("right", seed.clone(), &context, &mut output)
                .await
                .unwrap();
            let before = op.status.clone();
            let reservation = pool.reserved();
            let hold = tight.then(|| {
                op.reserve_workspace((128 << 20) - reservation as u64)
                    .unwrap()
            });
            if cancelled {
                cancellation.cancel();
            }
            let next = if duplicate {
                seed.clone()
            } else {
                batch(vec![unsigned_record(
                    &schema,
                    vec!["S000"],
                    vec![1],
                    vec![1],
                )])
            };
            let guard = LegacyGuard::new(old);
            let error = op
                .process_data("right", next, &context, &mut output)
                .await
                .unwrap_err();
            drop(guard);
            if !cancelled {
                let crate::CalcFlowError::OperatorReason { reason_code, .. } = error else {
                    panic!("unexpected error: {error}");
                };
                assert_eq!(
                    reason_code,
                    if duplicate {
                        StreamingFailureReason::AsofDuplicateIdentity
                    } else {
                        StreamingFailureReason::AsofWorkspaceLimitExceeded
                    }
                );
            }
            assert_eq!(op.status.right.accepted_rows, before.right.accepted_rows);
            assert_eq!(op.status.state_rows, before.state_rows);
            assert_eq!(
                op.current_inventory(None).unwrap().bytes,
                before.state_bytes
            );
            drop(hold);
            assert_eq!(pool.reserved(), reservation);
            drop((op, output, context));
            drop(job);
            assert_eq!(pool.reserved(), 0);
        }
    }
}

async fn compare(
    input: &Batch,
    seed: Option<&Batch>,
    watermark: Option<i64>,
    direct: bool,
    checkpointing: bool,
    service: &TestService,
    projection: Option<Vec<usize>>,
) {
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut fast = fixture(&schema);
    let mut legacy = fixture(&schema);
    if let Some(columns) = projection {
        fast.set_output_projection(columns.clone()).unwrap();
        legacy.set_output_projection(columns).unwrap();
    }
    let pools = [&fast, &legacy].map(|op| op.runtime.pool.clone());
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_checkpointing(checkpointing)
        .with_gather_owner(service.owner("direct-right-differential".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    if let Some(seed) = seed {
        for op in [&mut fast, &mut legacy] {
            let mut output = EdgeCollector::new(op.output_ports().to_vec());
            op.process_data("right", seed.clone(), &context, &mut output)
                .await
                .unwrap();
        }
        if checkpointing {
            let snapshot = assert_capture_equal(&mut fast, &mut legacy, &context).await;
            fast.restore(&snapshot).unwrap();
            legacy.restore(&snapshot).unwrap();
        }
    }
    let validated = ValidatedInput {
        index: 1,
        watermark,
    };
    let frozen = [&fast, &legacy].map(|op| {
        op.state
            .right
            .owned_buckets()
            .map(|(_, bucket)| {
                let rows = bucket.len();
                (bucket, rows)
            })
            .collect::<Vec<_>>()
    });
    for (op, old) in [(&mut fast, false), (&mut legacy, true)] {
        CANONICAL_ROWS.set(0);
        let (admission, _) = prepared(op, input, validated, &context, old).await;
        assert_eq!(
            admission.rows.is_empty(),
            direct && !old,
            "unexpected admission path"
        );
        assert_eq!(CANONICAL_ROWS.replace(0) == 0, direct && !old);
        op.install_admission("right", admission, validated, &context)
            .await
            .unwrap();
    }
    assert_eq!(fast.status, legacy.status);
    assert_eq!(
        fast.current_inventory(None).unwrap().bytes,
        fast.status.state_bytes
    );
    for (key, bucket) in &fast.state.right {
        let rows = |bucket: &state::RightBucket| {
            bucket
                .iter()
                .map(|((t, s), row)| (*t, s.into_owned(), row.copied()))
                .collect::<Vec<_>>()
        };
        assert_eq!(rows(bucket), rows(legacy.state.right.get(key).unwrap()));
    }
    assert!(
        frozen
            .iter()
            .flatten()
            .all(|(bucket, count)| bucket.len() == *count)
    );
    drop(frozen);
    if checkpointing {
        assert_capture_equal(&mut fast, &mut legacy, &context).await;
    }
    let mut outputs = Vec::new();
    for op in [&mut fast, &mut legacy] {
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        // Prepare accepted rows explicitly for the mixed-late case.
        let accepted = if watermark.is_some() {
            let record = &input.table_payload().unwrap().batches()[0];
            batch(vec![record.slice(1, record.num_rows() - 1)])
        } else {
            input.clone()
        };
        op.process_data("left", accepted, &context, &mut output)
            .await
            .unwrap();
        op.on_end(&context, &mut output).await.unwrap();
        outputs.push(output_records(&mut output));
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(fast.status, legacy.status);
    drop((fast, legacy, outputs));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}
