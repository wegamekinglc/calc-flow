use super::*;
mod source_replay;
use crate::{
    BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, Epoch, IngressProgress,
    IngressState, StreamJobContext,
};
use datafusion::arrow::{
    array::{Float64Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

struct LateMetrics;

fn assert_log_funded(operator: &StreamAsofJoinOperator) {
    let credit = operator
        .checkpoint_log
        .credit
        .as_ref()
        .map_or(0, |credit| credit.size());
    assert!(operator.runtime.pool.reserved() >= operator.state.right.auxiliary_bytes() + credit);
    assert!(
        operator
            .checkpoint_log
            .segments
            .values()
            .all(crate::StateSegment::has_owner)
    );
    assert_eq!(
        operator.status.state_bytes,
        operator.current_inventory(None).unwrap().bytes
    );
}

fn assert_shared_segments(
    left: &crate::OperatorStateSnapshot,
    right: &crate::OperatorStateSnapshot,
) {
    assert_eq!(left.segments, right.segments);
    for (name, segment) in &left.segments {
        assert!(Arc::ptr_eq(
            &segment.bytes_arc(),
            &right.segments[name].bytes_arc()
        ));
    }
}

fn indexed_input(schema: &SchemaRef, rows: &[(&str, i64, i64)]) -> RecordBatch {
    RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.0))),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(rows.iter().map(|row| row.1))
                    .with_timezone("UTC"),
            ),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.2))),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn indexed_admission_preserves_compacted_multibatch_payloads_after_sort_and_restore() {
    let (template, _) = fixture();
    let schema = template.schemas[0].clone();
    let spec = template.spec.with_late_policy(AsofLatePolicy::Drop);
    let mut op =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec.clone()).unwrap();
    op.status.left.watermark_micros = Some(EventTime::from_micros(100));
    op.status.right.watermark_micros = Some(EventTime::from_micros(100));
    let right = Batch::table(
        vec![
            indexed_input(&schema, &[]),
            indexed_input(&schema, &[("B", 103, 33), ("A", 100, 10), ("A", 102, 22)]),
            indexed_input(&schema, &[("A", 99, 9), ("A", 101, 11)]),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let left = Batch::table(
        vec![
            indexed_input(&schema, &[]),
            indexed_input(&schema, &[("A", 102, 202), ("B", 103, 303), ("A", 99, 199)]),
            indexed_input(
                &schema,
                &[("A", 100, 100), ("A", 101, 101), ("A", 104, 104)],
            ),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut preload = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut preload)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut preload)
        .await
        .unwrap();
    assert_eq!(op.status.right.accepted_rows, 4);
    assert_eq!(op.status.left.accepted_rows, 5);
    assert_eq!(op.status.right.late_rows, 1);
    assert_eq!(op.status.left.late_rows, 1);
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    let mut restored = StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
    restored.restore(&snapshot).unwrap();
    let mut expected_status = op.status.clone();
    expected_status.left.watermark_micros = None;
    expected_status.right.watermark_micros = None;
    assert_eq!(restored.status, expected_status);
    let repeated = restored.capture(Epoch::INITIAL).unwrap();
    assert_eq!(repeated.inline_metadata, snapshot.inline_metadata);
    assert_eq!(repeated.segments, snapshot.segments);
    for candidate in [&mut op, &mut restored] {
        let mut output = EdgeCollector::new(candidate.output_ports().to_vec());
        candidate.on_end(&cx, &mut output).await.unwrap();
        let rows = output.drain("output");
        let right_sequences = rows
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
            .collect::<Vec<_>>();
        assert_eq!(
            right_sequences,
            vec![Some(10), Some(11), Some(22), Some(33), None]
        );
        assert_eq!(candidate.status.matched_rows, 4);
        assert_eq!(candidate.status.unmatched_rows, 1);
        assert_eq!(candidate.status.state_bytes, 0);
    }
}
impl crate::operator::stream::LateMetricSink for LateMetrics {
    fn record(&self, _delta: crate::operator::stream::LateMetricDelta) -> Result<()> {
        Ok(())
    }
}

struct RetainedPayloadCollector {
    payload: Arc<state::PayloadBatch>,
    owners: Vec<usize>,
}

struct GatedCollector {
    entered: Arc<std::sync::atomic::AtomicBool>,
    allowed: Arc<std::sync::atomic::AtomicBool>,
    batch: Option<Batch>,
}

#[async_trait]
impl StreamCollector for GatedCollector {
    async fn emit(&mut self, _port: &str, batch: Batch) -> Result<()> {
        use std::sync::atomic::Ordering;
        self.batch = Some(batch);
        self.entered.store(true, Ordering::SeqCst);
        futures::future::poll_fn(|_| {
            if self.allowed.load(Ordering::SeqCst) {
                std::task::Poll::Ready(Ok(()))
            } else {
                std::task::Poll::Pending
            }
        })
        .await
    }
}

#[tokio::test(flavor = "current_thread")]
async fn accepted_prefix_installs_pool_compaction_without_allocating() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let (template, _) = fixture();
    let schema = template.schemas[0].clone();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::ZERO,
        AsofStateLimits::new(10_000, 64 << 20).unwrap(),
    )
    .unwrap();
    let mut op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let input = Batch::table(
        (0..1_024)
            .map(|row| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StringArray::from(vec!["A"])),
                        Arc::new(TimestampMicrosecondArray::from(vec![row]).with_timezone("UTC")),
                        Arc::new(Int64Array::from(vec![row])),
                    ],
                )
                .unwrap()
            })
            .collect(),
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut preload = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", input, &cx, &mut preload)
        .await
        .unwrap();
    let progress = IngressProgressSnapshot::new(
        ["left", "right"]
            .map(|side| {
                (
                    side.into(),
                    IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(800))),
                )
            })
            .into_iter()
            .collect(),
    );
    let progress_cx = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        Some(EventTime::from_micros(800)),
        progress,
    );
    let entered = Arc::new(AtomicBool::new(false));
    let allowed = Arc::new(AtomicBool::new(false));
    let mut collector = GatedCollector {
        entered: entered.clone(),
        allowed: allowed.clone(),
        batch: None,
    };
    let waker = futures::task::noop_waker();
    let mut poll_context = std::task::Context::from_waker(&waker);
    let mut operation =
        Box::pin(op.on_ingress_progress_with_output("left", &progress_cx, &mut collector));
    tokio::time::timeout(Duration::from_secs(5), async {
        while !entered.load(Ordering::SeqCst) {
            assert!(operation.as_mut().poll(&mut poll_context).is_pending());
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    allowed.store(true, Ordering::SeqCst);
    let mut result = None;
    let allocation = allocation_counter::measure(|| {
        result = Some(operation.as_mut().poll(&mut poll_context));
    });
    assert_eq!(
        allocation.count_total, 0,
        "post-delivery commit allocated: {allocation:?}"
    );
    assert!(matches!(result.unwrap(), std::task::Poll::Pending));
    drop(operation);
    assert_eq!(op.status.emitted_left_rows, 800);
    assert_eq!(op.state.batches.len(), 224);
    assert_eq!(collector.batch.unwrap().num_rows(), 800);
    op.retirement.wait(&cx).await.unwrap();
}

#[test]
fn shared_right_columns_are_copied_outside_the_executor() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    runtime.block_on(async {
        for admission in [false, true] {
            assert_shared_right_copy_preparation(admission).await;
        }
    });
}

fn shared_right_copy_fixture() -> (
    StreamAsofJoinOperator,
    impl Fn(std::ops::Range<i64>) -> Batch,
) {
    let (template, _) = fixture();
    let schema = template.schemas[0].clone();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::ZERO,
        AsofStateLimits::new(10_000, 64 << 20).unwrap(),
    )
    .unwrap();
    let op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let input = move |range: std::ops::Range<i64>| {
        let rows = usize::try_from(range.end - range.start).unwrap();
        Batch::table(
            vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StringArray::from(vec!["A"; rows])),
                        Arc::new(
                            TimestampMicrosecondArray::from_iter_values(range.clone())
                                .with_timezone("UTC"),
                        ),
                        Arc::new(Int64Array::from_iter_values(range)),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    };
    (op, input)
}

async fn assert_shared_right_copy_preparation(admission: bool) {
    let (mut op, input) = shared_right_copy_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input(0..4_096), &cx, &mut out)
        .await
        .unwrap();
    let frozen = op.state.right.owned_buckets().next().unwrap().1;
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
        ),
    ]));
    let progress_cx = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        Some(EventTime::from_micros(1)),
        progress,
    );
    // Keep the only blocking worker occupied so this poll measures preparation
    // and scheduling, regardless of how quickly a platform can copy columns.
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let blocker = tokio::task::spawn_blocking(move || {
        started_tx.send(()).unwrap();
        let _released = release_rx.recv();
    });
    started_rx.await.unwrap();
    let mut operation: std::pin::Pin<Box<dyn Future<Output = Result<()>> + '_>> = if admission {
        Box::pin(op.process_data("right", input(4_096..4_097), &cx, &mut out))
    } else {
        Box::pin(op.on_ingress_progress_with_output("right", &progress_cx, &mut out))
    };
    let waker = futures::task::noop_waker();
    let mut poll_context = std::task::Context::from_waker(&waker);
    let mut first = None;
    let allocations = allocation_counter::measure(|| {
        first = Some(operation.as_mut().poll(&mut poll_context));
    });
    assert!(
        allocations.bytes_max < 32_768,
        "shared bucket copied on the executor: admission={admission}, {allocations:?}"
    );
    assert!(
        first.unwrap().is_pending(),
        "shared column preparation must yield to its worker"
    );
    release_tx.send(()).unwrap();
    blocker.await.unwrap();
    operation.await.unwrap();
    assert_eq!(frozen.len(), 4_096);
    assert_eq!(
        op.status.retained_right_rows,
        if admission { 4_097 } else { 4_095 }
    );
}

#[tokio::test]
async fn test_checkpoint_disabled_asof_keeps_live_state_without_log_owners() {
    let (mut operator, left, right) = prefix_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_checkpointing(false);
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for (side, batch) in [("left", left), ("right", right)] {
        operator
            .process_data(side, batch, &context, &mut output)
            .await
            .unwrap();
        assert!(operator.status.state_rows > 0);
        assert!(operator.status.state_bytes > 0);
        assert_eq!(
            operator.current_inventory(None).unwrap().bytes,
            operator.status.state_bytes
        );
        assert!(operator.checkpoint_log.credit.is_none());
        assert!(operator.checkpoint_log.registry.is_none());
        assert!(operator.checkpoint_log.journal.is_empty());
        assert_eq!(operator.checkpoint_log.bytes(), 0);
    }
    operator
        .on_watermark(EventTime::from_micros(102), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status.emitted_left_rows, 2);
    assert_eq!(operator.status.pending_left_rows, 1);
    assert!(operator.checkpoint_log.credit.is_none());
    operator.on_end(&context, &mut output).await.unwrap();
    assert_eq!(operator.status.emitted_left_rows, 3);
    assert_eq!(operator.status.matched_rows, 3);
    assert_eq!(operator.status.state_rows, 0);
    assert_eq!(operator.status.state_bytes, 0);
    assert!(operator.checkpoint_log.credit.is_none());
    assert!(operator.checkpoint_log.journal.is_empty());
    assert_eq!(
        output
            .drain("output")
            .iter()
            .map(|message| message.as_data().unwrap().num_rows())
            .sum::<usize>(),
        3
    );
}

#[tokio::test]
async fn new_checkpoint_uses_columnar_v6_and_restores_the_same_state_charge() {
    let (mut operator, left, right) = prefix_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left, &context, &mut output)
        .await
        .unwrap();
    operator
        .process_data("right", right, &context, &mut output)
        .await
        .unwrap();
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        snapshot.inline_metadata["state_version"],
        serde_json::json!(3)
    );
    for field in ["layout_version", "accounting_version"] {
        assert_eq!(snapshot.inline_metadata[field], serde_json::json!(10));
    }
    let index = snapshot
        .segments
        .get("asof-log-v10-1-0-1")
        .expect("columnar v6 index");
    assert_eq!(&index.bytes()[..8], b"CFASDL10");
    let (mut restored, _, _) = prefix_fixture();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status.state_bytes, operator.status.state_bytes);
    let repeated = restored.capture(Epoch::INITIAL).unwrap();
    assert_eq!(repeated.inline_metadata, snapshot.inline_metadata);
    assert_eq!(repeated.segments, snapshot.segments);
    restored.on_end(&context, &mut output).await.unwrap();
    assert_eq!(restored.status.emitted_left_rows, 3);
    assert_eq!(restored.status.matched_rows, 3);
    assert_eq!(restored.status.state_bytes, 0);
}

fn integer_sequences(data_type: &DataType, values: [u64; 3]) -> datafusion::arrow::array::ArrayRef {
    use datafusion::arrow::array::{
        Int8Array, Int16Array, Int32Array, UInt8Array, UInt16Array, UInt32Array,
    };
    match data_type {
        DataType::Int8 => Arc::new(Int8Array::from_iter_values(
            values.map(|value| i8::from_le_bytes([u8::try_from(value).unwrap()])),
        )),
        DataType::Int16 => {
            Arc::new(Int16Array::from_iter_values(values.map(|value| {
                i16::from_le_bytes(u16::try_from(value).unwrap().to_le_bytes())
            })))
        }
        DataType::Int32 => {
            Arc::new(Int32Array::from_iter_values(values.map(|value| {
                i32::from_le_bytes(u32::try_from(value).unwrap().to_le_bytes())
            })))
        }
        DataType::Int64 => Arc::new(Int64Array::from_iter_values(
            values.map(|value| i64::from_le_bytes(value.to_le_bytes())),
        )),
        DataType::UInt8 => Arc::new(UInt8Array::from_iter_values(
            values.map(|value| u8::try_from(value).unwrap()),
        )),
        DataType::UInt16 => Arc::new(UInt16Array::from_iter_values(
            values.map(|value| u16::try_from(value).unwrap()),
        )),
        DataType::UInt32 => Arc::new(UInt32Array::from_iter_values(
            values.map(|value| u32::try_from(value).unwrap()),
        )),
        DataType::UInt64 => Arc::new(UInt64Array::from_iter_values(values)),
        _ => unreachable!("integer sequence fixture"),
    }
}

fn integer_checkpoint_inputs(
    data_type: &DataType,
    sequences: datafusion::arrow::array::ArrayRef,
) -> (StreamAsofJoinOperator, impl Fn(Vec<i64>) -> Batch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", data_type.clone(), false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left_"),
        side("right_"),
        Duration::from_micros(10),
        AsofStateLimits::new(100, 128 * 1024).unwrap(),
    )
    .unwrap();
    let op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let input = move |times: Vec<i64>| {
        Batch::table(
            vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StringArray::from(vec!["a long shared key 中文"; 3])),
                        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
                        sequences.clone(),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    };
    (op, input)
}

fn integer_sequence_types() -> [DataType; 8] {
    [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
    ]
}

#[tokio::test]
async fn integer_v3_checkpoints_preserve_extremes_capacities_and_answers() {
    for (flag, data_type) in (1..=8).zip(integer_sequence_types()) {
        let kind = state::SequenceKind::from_flag(flag).unwrap();
        let width = kind.width().unwrap();
        let maximum = u64::MAX >> (64 - width * 8);
        let sign = 1_u64 << (width * 8 - 1);
        let values = if matches!(kind, state::SequenceKind::Signed(_)) {
            [sign, maximum, sign - 1]
        } else {
            [0, 1, maximum]
        };
        let sequences = integer_sequences(&data_type, values);
        let expected = datafusion::common::ScalarValue::try_from_array(&sequences, 2).unwrap();
        for identity_only in [false, true] {
            let (mut op, input) = integer_checkpoint_inputs(&data_type, sequences.clone());
            let job =
                StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
            let cx = StreamOperatorContext::new(&job, "asof", None);
            let mut out = EdgeCollector::new(op.output_ports().to_vec());
            op.process_data("right", input(vec![100; 3]), &cx, &mut out)
                .await
                .unwrap();
            if identity_only {
                let progress = IngressProgressSnapshot::new(
                    [("left", 200), ("right", 100)]
                        .map(|(side, time)| {
                            (
                                side.into(),
                                IngressProgress::new(
                                    IngressState::Active,
                                    Some(EventTime::from_micros(time)),
                                ),
                            )
                        })
                        .into_iter()
                        .collect(),
                );
                let progress_cx =
                    StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress);
                op.on_ingress_progress_with_output("left", &progress_cx, &mut out)
                    .await
                    .unwrap();
                assert_eq!(op.status.identity_only_rows, 3);
            } else {
                op.process_data("left", input(vec![101, 102, 103]), &cx, &mut out)
                    .await
                    .unwrap();
            }
            let snapshot = op.capture(Epoch::INITIAL).unwrap();
            let before = op.status();
            let mut restored = StreamAsofJoinOperator::new(
                "asof",
                op.schemas[0].clone(),
                op.schemas[1].clone(),
                op.spec.clone(),
            )
            .unwrap();
            restored.restore(&snapshot).unwrap();
            assert_eq!(
                restored.status.state_bytes, before.state_bytes,
                "{data_type:?}"
            );
            assert_eq!(restored.status.state_rows, before.state_rows);
            assert_eq!(
                restored.status.identity_only_rows,
                before.identity_only_rows
            );
            assert_eq!(
                restored.capture(Epoch::INITIAL).unwrap().segments,
                snapshot.segments
            );
            restored.on_end(&cx, &mut out).await.unwrap();
            if !identity_only {
                let outputs = out.drain("output");
                let record = &outputs[0]
                    .as_data()
                    .unwrap()
                    .table_payload()
                    .unwrap()
                    .batches()[0];
                assert_eq!(record.num_rows(), 3);
                for row in 0..3 {
                    assert_eq!(
                        datafusion::common::ScalarValue::try_from_array(record.column(5), row)
                            .unwrap(),
                        expected
                    );
                }
            }
            assert_eq!(restored.status.state_rows, 0);
            assert_eq!(restored.status.state_bytes, 0);
        }
    }
}

#[tokio::test]
async fn retained_payload_batch_has_one_state_owner() {
    let (mut op, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    for (batch, _) in op.state.batches.values() {
        assert_eq!(
            Arc::strong_count(batch),
            1,
            "retained rows must refer to a batch through compact pool handles"
        );
    }
}

#[tokio::test]
async fn left_rows_share_one_key_owner_within_their_arrow_chunk() {
    let (mut op, _) = fixture();
    let key = "a repeated long left key with Unicode 字符";
    let input = Batch::table(
        vec![
            RecordBatch::try_new(
                op.schemas[0].clone(),
                vec![
                    Arc::new(StringArray::from(vec![key; 3])),
                    Arc::new(TimestampMicrosecondArray::from(vec![10; 3]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![3, 1, 2])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", input, &cx, &mut output)
        .await
        .unwrap();
    let (_, key, _) = op.state.left.keys().next().unwrap();
    let state::Encoding::Shared(bytes) = key else {
        panic!("long key must use shared encoding");
    };
    assert_eq!(
        Arc::strong_count(bytes),
        1,
        "the left chunk must own each distinct canonical key once"
    );
}

#[tokio::test]
async fn overlapping_left_chunks_restore_only_live_sparse_rows() {
    let (mut op, _, _) = prefix_fixture();
    let input = |times: Vec<i64>, sequences: Vec<i64>| {
        Batch::table(
            vec![
                RecordBatch::try_new(
                    op.schemas[0].clone(),
                    vec![
                        Arc::new(StringArray::from(vec!["A"; times.len()])),
                        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
                        Arc::new(Int64Array::from(sequences)),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    };
    let first = input(vec![102, 100, 101], vec![3, 1, 2]);
    let second = input(vec![104, 101], vec![5, 4]);
    let next = input(vec![103], vec![6]);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", first, &cx, &mut output)
        .await
        .unwrap();
    op.process_data("left", second, &cx, &mut output)
        .await
        .unwrap();
    op.on_watermark(EventTime::from_micros(102), &cx, &mut output)
        .await
        .unwrap();
    let emitted = output.drain("output");
    let record = &emitted[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(
        record
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[1, 2, 4]
    );
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    let (mut restored, _, _) = prefix_fixture();
    restored.restore(&snapshot).unwrap();
    let live = restored
        .state
        .left
        .iter()
        .map(|(identity, row)| (*identity.0, row.row))
        .collect::<Vec<_>>();
    assert_eq!(live, vec![(102, 0), (104, 0)]);
    restored
        .process_data("left", next, &cx, &mut output)
        .await
        .unwrap();
    restored
        .on_watermark(EventTime::from_micros(105), &cx, &mut output)
        .await
        .unwrap();
    let remaining = output.drain("output");
    let record = &remaining[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(
        record
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[3, 6, 5]
    );
    assert_eq!(restored.status.emitted_left_rows, 6);
    assert_eq!(restored.status.pending_left_rows, 0);
}

#[tokio::test]
async fn integer_left_chunk_checkpoint_omits_consumed_prefix_without_compaction() {
    let (mut op, left, right) = prefix_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut output)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut output)
        .await
        .unwrap();
    op.on_watermark(EventTime::from_micros(101), &cx, &mut output)
        .await
        .unwrap();
    assert_eq!(output.drain("output").len(), 1);
    let (_, _, head) = op
        .state
        .left
        .checkpoint_chunks(&op.state.batches)
        .next()
        .unwrap();
    assert_eq!(head, 1, "the consumed integer prefix must remain uncompact");
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    let expected_charge = op.status.state_bytes;
    let (mut restored, _, _) = prefix_fixture();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status.state_bytes, expected_charge);
    assert_eq!(
        restored
            .state
            .left
            .iter()
            .map(|(order, _)| *order.0)
            .collect::<Vec<_>>(),
        [101, 102]
    );
    let repeated = restored.capture(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.segments, repeated.segments);
    restored
        .on_watermark(EventTime::from_micros(103), &cx, &mut output)
        .await
        .unwrap();
    let remaining = output.drain("output");
    assert_eq!(remaining.len(), 1);
    let record = &remaining[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    let left = record
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let right = record
        .column(5)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(left.values().as_ref(), &[2, 3]);
    assert_eq!(right.values().as_ref(), &[1, 1]);
    assert_eq!(restored.status.pending_left_rows, 0);
    assert_eq!(restored.status.matched_rows, 3);
}

#[tokio::test]
async fn later_admission_reuses_the_resident_right_key_bytes() {
    let (mut op, initial) = fixture();
    let schema = initial.table_payload().unwrap().batches()[0].schema();
    let key = "a persistent symbol longer than the inline encoding";
    let input = |sequence| {
        Batch::table(
            vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StringArray::from(vec![key])),
                        Arc::new(TimestampMicrosecondArray::from(vec![10]).with_timezone("UTC")),
                        Arc::new(Int64Array::from(vec![sequence])),
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
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input(1), &cx, &mut output)
        .await
        .unwrap();
    let resident = op
        .state
        .right
        .ordered_iter()
        .next()
        .unwrap()
        .0
        .as_slice()
        .as_ptr();
    let batch = input(2);
    let validated = op.validate_admission("right", &batch).unwrap();
    let admission = op.prepare_admission(validated, &batch, &cx).await.unwrap();
    assert_eq!(
        admission.rows[0].0.1.as_slice().as_ptr(),
        resident,
        "each canonical key buffer must be owned once across batches"
    );
    drop(admission);
    assert_eq!(op.runtime.pool.reserved(), op.state.right.auxiliary_bytes());
}

#[tokio::test]
async fn admission_prepares_index_only_when_checkpoint_is_captured() {
    let (mut op, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    assert!(op.prepared.is_none(), "admission must defer index encoding");
    let charged = op.status.state_bytes;
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    assert!(snapshot.segments.contains_key("asof-log-v10-1-0-1"));
    assert!(op.status.state_bytes > charged);
    assert_log_funded(&op);
    assert!(!op.checkpoint_log.frames.is_empty());
}

#[tokio::test]
async fn admitted_payload_is_encoded_only_for_a_checkpoint() {
    let (mut op, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    assert!(!op.state.batches.is_empty());
    assert!(
        op.state
            .batches
            .values()
            .all(|(batch, _)| !batch.has_encoded())
    );
    let charged = op.status.state_bytes;
    op.prepare_checkpoint_async(&cx).await.unwrap();
    assert!(
        op.state
            .batches
            .values()
            .all(|(batch, _)| batch.has_encoded())
    );
    for (batch, _) in op.state.batches.values() {
        let encoded = batch.encoded.get().unwrap();
        let eager = codec::encode_batch(&batch.record, usize::MAX, &mut Vec::new()).unwrap();
        assert_eq!(encoded.bytes(), eager);
        assert_eq!(
            encoded.bytes_arc().capacity() as u64,
            batch.encoded_charge_bytes
        );
    }
    let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    assert!(
        snapshot
            .segments
            .keys()
            .any(|name| name.starts_with("asof-batch"))
    );
    assert!(op.status.state_bytes > charged);
    assert_log_funded(&op);
}

#[tokio::test]
async fn async_checkpoint_preparation_keeps_capture_on_shared_bytes() {
    let (mut op, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    op.prepare_checkpoint_async(&cx).await.unwrap();
    assert!(op.checkpoint_log.pending.is_some());
    let payloads = op
        .state
        .batches
        .iter()
        .map(|(key, (batch, _))| (*key, batch.encoded.get().unwrap().bytes_arc()))
        .collect::<BTreeMap<_, _>>();
    let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    for (key, bytes) in payloads {
        assert!(Arc::ptr_eq(
            &bytes,
            &snapshot.segments[&format!("asof-batch-{}-{}", key.0, key.1)].bytes_arc()
        ));
    }
    let repeated = op.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_shared_segments(&snapshot, &repeated);
}

#[tokio::test]
async fn cancelled_checkpoint_preparation_preserves_deferred_state_for_retry() {
    let (mut op, input) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    let deferred = op.deferred_index_len;
    let charged = op.status.state_bytes;
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let cancelled_job = StreamJobContext::new(2, "asof", JsonMap::new(), None, cancelled);
    let cancelled_cx = StreamOperatorContext::new(&cancelled_job, "asof", None);
    assert!(matches!(
        op.prepare_checkpoint_async(&cancelled_cx).await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(op.deferred_index_len, deferred);
    assert_eq!(op.status.state_bytes, charged);
    assert!(op.prepared.is_none());
    op.prepare_checkpoint_async(&cx).await.unwrap();
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    let (mut restored, _) = fixture();
    restored.restore(&snapshot).unwrap();
    let mut expected_status = op.status();
    expected_status.output_watermark_micros = None;
    assert_eq!(restored.status(), expected_status);
    assert_log_funded(&restored);
    assert_eq!(restored.status.retained_right_rows, 1);
}

#[tokio::test]
async fn deferred_index_length_tracks_existing_and_new_right_buckets() {
    let (mut op, first) = fixture();
    let schema = op.schemas[1].clone();
    let second = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A", "B"])),
                    Arc::new(TimestampMicrosecondArray::from(vec![101, 99]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![2, 1])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", first, &cx, &mut output)
        .await
        .unwrap();
    op.process_data("right", second, &cx, &mut output)
        .await
        .unwrap();
    assert_eq!(
        op.deferred_index_len,
        Some(checkpoint::v3_encoded_length(&op.state, &op.name).unwrap())
    );
    assert!(op.prepared.is_none());
    let before = op.status.state_bytes;
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    assert!(op.status.state_bytes > before);
    let before = op.status.state_bytes;
    assert_log_funded(&op);
    let (mut restored, _) = fixture();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status.state_bytes, before);
}

#[tokio::test]
async fn finalizable_prefix_can_use_full_edge_row_budget() {
    let (template, _) = fixture();
    let schema = template.schemas[0].clone();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::ZERO,
        AsofStateLimits::new(10_000, 64 << 20).unwrap(),
    )
    .unwrap();
    let mut op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let rows = 5_000;
    let left = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A"; rows])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100; rows]).with_timezone("UTC")),
                    Arc::new(Int64Array::from_iter_values(
                        0..i64::try_from(rows).unwrap(),
                    )),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(8_192, 64 << 20).unwrap());
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", left, &cx, &mut output)
        .await
        .unwrap();
    op.on_watermark(EventTime::from_micros(101), &cx, &mut output)
        .await
        .unwrap();
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    assert_eq!(
        messages[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0]
            .num_rows(),
        rows
    );
    assert!(
        !crate::pipeline::CompiledStreamOperator::StreamAsofJoin(Box::new(op))
            .datafusion_runtime_initialized()
    );
}

#[tokio::test]
async fn full_admission_of_a_tiny_slice_compacts_large_backing_buffers() {
    let (mut op, _) = fixture();
    let schema = op.schemas[1].clone();
    let large = "x".repeat(4 * 1024 * 1024);
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![large.as_str(), "A", large.as_str()])),
            Arc::new(TimestampMicrosecondArray::from(vec![99, 100, 101]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    let input = Batch::table(vec![record.slice(1, 1)], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    let retained = op
        .state
        .right
        .values()
        .next()
        .unwrap()
        .values()
        .next()
        .unwrap();
    let bytes = op
        .state
        .batches
        .view(**retained.as_ref().unwrap())
        .batch
        .record
        .column(0)
        .to_data()
        .get_buffer_memory_size();
    assert!(bytes < 16_384, "tiny row retained {bytes} backing bytes");
}

#[tokio::test]
async fn full_admission_does_not_retain_unbilled_small_slice_tail() {
    let (mut op, _) = fixture();
    let schema = op.schemas[1].clone();
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["A", &"x".repeat(3_000)])),
            Arc::new(TimestampMicrosecondArray::from(vec![99, 100]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![1, 2])),
        ],
    )
    .unwrap();
    let input = Batch::table(vec![record.slice(0, 1)], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    let retained = op
        .state
        .right
        .values()
        .next()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .unwrap();
    let data = op
        .state
        .batches
        .view(*retained)
        .batch
        .record
        .column(0)
        .to_data();
    assert!(
        data.get_buffer_memory_size() < 64,
        "admitted slice retained backing bytes outside its charged payload"
    );
}

#[async_trait]
impl StreamCollector for RetainedPayloadCollector {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        self.owners.push(Arc::strong_count(&self.payload));
        Ok(())
    }
}

#[tokio::test]
async fn finalization_without_eviction_does_not_clone_retained_state() {
    let (mut op, left, right) = prefix_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut preload = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut preload)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut preload)
        .await
        .unwrap();
    let payload = op.state.batches.values().next().unwrap().0.clone();
    let owners = Arc::strong_count(&payload);
    let mut output = RetainedPayloadCollector {
        payload,
        owners: Vec::new(),
    };
    op.on_watermark(EventTime::from_micros(103), &cx, &mut output)
        .await
        .unwrap();
    assert_eq!(
        output.owners, [owners; 3],
        "each chunk must borrow unchanged right state"
    );
    assert_eq!(op.status.matched_rows, 3);
    assert_eq!(op.status.pending_left_rows, 0);
    let expected = op.prepare_checkpoint(&op.state, &cx).await.unwrap();
    let captured = op.capture(Epoch::INITIAL).unwrap();
    let frame = checkpoint::index_v3::log::chain::decode(
        &captured.segments["asof-log-v10-1-0-1"],
        &op.fingerprint,
    )
    .unwrap();
    assert_eq!(frame.body, expected.segment.unwrap().canonical().bytes());
}
#[tokio::test]
async fn finalized_prefixes_release_old_checkpoint_bytes_and_defer_new_encoding() {
    let (mut op, left, right) = prefix_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut collector)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut collector)
        .await
        .unwrap();
    let committed = op.capture(Epoch::INITIAL).unwrap().segments["asof-log-v10-1-0-1"].bytes_arc();
    let owners = Arc::strong_count(&committed);
    op.on_watermark(EventTime::from_micros(103), &cx, &mut collector)
        .await
        .unwrap();
    assert_eq!(collector.drain("output").len(), 3);
    assert_eq!(Arc::strong_count(&committed), owners);
    assert!(op.checkpoint_log.pending.is_none());
    assert!(!op.checkpoint_log.journal.is_empty());
    op.prepare_checkpoint_async(&cx).await.unwrap();
    let captured = op.capture(Epoch::new(2).unwrap()).unwrap();
    let expected = op.prepare_checkpoint(&op.state, &cx).await.unwrap();
    let (mut restored, _, _) = prefix_fixture();
    restored.restore(&captured).unwrap();
    let mut expected_status = op.status();
    expected_status.output_watermark_micros = None;
    assert_eq!(restored.status(), expected_status);
    let actual = restored
        .prepare_checkpoint(&restored.state, &cx)
        .await
        .unwrap();
    assert_eq!(
        actual.segment.map(|segment| segment.canonical()),
        expected.segment.map(|segment| segment.canonical())
    );
    let repeated = op.capture(Epoch::new(2).unwrap()).unwrap();
    assert_shared_segments(&captured, &repeated);
}

struct CancelPrefixCollector {
    cancel: CancellationToken,
    accepted: Vec<Batch>,
}

#[async_trait]
impl StreamCollector for CancelPrefixCollector {
    async fn emit(&mut self, _port: &str, batch: Batch) -> Result<()> {
        if self.accepted.is_empty() {
            self.accepted.push(batch);
            Ok(())
        } else {
            self.cancel.cancel();
            std::future::pending().await
        }
    }
}

#[tokio::test]
async fn cancelled_prefix_has_canonical_checkpoint_and_resumes_exactly() {
    let (mut op, left, right) = prefix_fixture();
    let cancel = CancellationToken::new();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancel.clone());
    let cx = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut preload = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut preload)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut preload)
        .await
        .unwrap();
    let before = op.capture(Epoch::INITIAL).unwrap();
    let mut stopped = CancelPrefixCollector {
        cancel,
        accepted: Vec::new(),
    };
    let result = op
        .on_watermark(EventTime::from_micros(103), &cx, &mut stopped)
        .await;
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(op.status.pending_left_rows, 2);
    assert_eq!(op.status.matched_rows, 1);
    assert_eq!(op.next_output_sequence, 1);
    assert_log_funded(&op);
    let snapshot = op.capture(Epoch::new(2).unwrap()).unwrap();
    let fresh_job =
        StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new());
    let resumed_cx = StreamOperatorContext::new(&fresh_job, "asof", None)
        .with_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let canonical = op.prepare_checkpoint(&op.state, &resumed_cx).await.unwrap();
    assert!(canonical.segment.is_some());
    let (mut restored, _, _) = prefix_fixture();
    restored.restore(&before).unwrap();
    assert_eq!(
        restored.status.pending_left_rows, 3,
        "older shared snapshot is unchanged"
    );
    restored.restore(&snapshot).unwrap();
    let repeated = restored.capture(Epoch::new(2).unwrap()).unwrap();
    assert_shared_segments(&snapshot, &repeated);
    restored.restore(&repeated).unwrap();
    let mut remaining = EdgeCollector::new(restored.output_ports().to_vec());
    restored
        .on_watermark(EventTime::from_micros(103), &resumed_cx, &mut remaining)
        .await
        .unwrap();
    let mut batches = stopped.accepted;
    batches.extend(
        remaining
            .drain("output")
            .into_iter()
            .map(|message| message.as_data().unwrap().clone()),
    );
    for (index, batch) in batches.iter().enumerate() {
        assert_eq!(batch.metadata().sequence(), index as u64);
        let record = &batch.table_payload().unwrap().batches()[0];
        let left = record
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let right = record
            .column(5)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(left.value(0), i64::try_from(index).unwrap() + 1);
        assert_eq!(right.value(0), 1);
    }
    assert_eq!(batches.len(), 3);
    assert_eq!(restored.status.emitted_left_rows, 3);
    assert_eq!(restored.status.pending_left_rows, 0);
}

#[tokio::test]
async fn evictable_right_state_survives_a_cancelled_output_prefix() {
    let (template, left, right) = prefix_fixture();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::ZERO,
        template.spec.limits(),
    )
    .unwrap();
    let schema = template.schemas[0].clone();
    let mut op =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec.clone()).unwrap();
    let cancel = CancellationToken::new();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancel.clone());
    let cx = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut preload = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut preload)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut preload)
        .await
        .unwrap();
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(103))),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(103))),
        ),
    ]));
    let tick_cx =
        StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress.clone())
            .with_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut stopped = CancelPrefixCollector {
        cancel,
        accepted: Vec::new(),
    };
    assert!(matches!(
        op.on_watermark(EventTime::from_micros(103), &tick_cx, &mut stopped)
            .await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(op.status.pending_left_rows, 2);
    assert_eq!(op.status.retained_right_rows, 1);
    assert!(op.prepared.is_none());
    assert!(op.deferred_index_len.is_some());
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    let mut restored = StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
    restored.restore(&snapshot).unwrap();
    let resumed_job =
        StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new());
    let resumed_cx =
        StreamOperatorContext::with_ingress_progress(&resumed_job, "asof", None, progress)
            .with_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut remaining = EdgeCollector::new(restored.output_ports().to_vec());
    restored
        .on_watermark(EventTime::from_micros(103), &resumed_cx, &mut remaining)
        .await
        .unwrap();
    assert_eq!(remaining.drain("output").len(), 2);
    assert_eq!(restored.status.retained_right_rows, 0);
    assert_eq!(restored.status.evicted_right_rows, 1);
}

fn prefix_fixture() -> (StreamAsofJoinOperator, Batch, Batch) {
    let (template, right) = fixture();
    let schema = template.schemas[0].clone();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(10),
        template.spec.limits(),
    )
    .unwrap();
    let op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let left = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A"; 3])),
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![100, 101, 102]).with_timezone("UTC"),
                    ),
                    Arc::new(Int64Array::from(vec![1, 2, 3])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    (op, left, right)
}

fn shared_string_identity_fixture() -> (StreamAsofJoinOperator, Batch) {
    let (template, _) = fixture();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Utf8, false),
    ]));
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(2),
        template.spec.limits(),
    )
    .unwrap();
    let op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["key".repeat(32); 3])),
            Arc::new(TimestampMicrosecondArray::from(vec![10, 20, 30]).with_timezone("UTC")),
            Arc::new(StringArray::from(vec!["sequence".repeat(32); 3])),
        ],
    )
    .unwrap();
    (
        op,
        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
    )
}

fn asymmetric_progress(left: i64, right: i64) -> IngressProgressSnapshot {
    IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(left))),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(right))),
        ),
    ]))
}

#[tokio::test]
async fn bulk_eviction_preserves_shared_string_owners_until_all_live_references_expire() {
    let (mut op, right) = shared_string_identity_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut output)
        .await
        .unwrap();
    let allocation = op.state.encoding_owner_allocation();
    assert!(allocation.0 > 0 && allocation.1 > 0);
    let progress = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        None,
        asymmetric_progress(25, 15),
    );
    op.on_ingress_progress_with_output("left", &progress, &mut output)
        .await
        .unwrap();
    assert_eq!(op.status.retained_right_rows, 1);
    assert_eq!(op.status.identity_only_rows, 1);
    assert_eq!(op.status.evicted_right_rows, 2);
    assert_eq!(op.state.batches.values().next().unwrap().1, 1);
    assert_eq!(op.state.encoding_owner_allocation(), allocation);
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    let expected_charge = op.status.state_bytes;
    let (mut restored, _) = shared_string_identity_fixture();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status.state_bytes, expected_charge);
    assert_eq!(restored.state.encoding_owner_allocation(), allocation);
    assert_eq!(
        restored.capture(Epoch::INITIAL).unwrap().segments,
        snapshot.segments
    );
    let progress = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        None,
        asymmetric_progress(35, 31),
    );
    restored
        .on_ingress_progress_with_output("left", &progress, &mut output)
        .await
        .unwrap();
    assert_eq!(restored.status.retained_right_rows, 0);
    assert_eq!(restored.status.identity_only_rows, 0);
    assert_eq!(restored.status.evicted_right_rows, 3);
    assert_eq!(restored.state.encoding_owner_allocation(), (0, 0));
    assert!(restored.state.batches.values().next().is_none());
    assert_eq!(
        restored.status.state_bytes,
        restored.current_inventory(None).unwrap().bytes
    );
}

#[tokio::test]
async fn eviction_preview_workspace_covers_many_owned_keys_in_one_payload_batch() {
    let (mut op, _) = fixture();
    let count = 64;
    let input = Batch::table(
        vec![
            RecordBatch::try_new(
                op.schemas[1].clone(),
                vec![
                    Arc::new(StringArray::from_iter_values(
                        (0..count).map(|row| format!("key-{row:04}-{}", "x".repeat(32))),
                    )),
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![100; count]).with_timezone("UTC"),
                    ),
                    Arc::new(Int64Array::from(vec![1; count])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", input, &cx, &mut output)
        .await
        .unwrap();
    assert_eq!(op.state.batches.values().count(), 1);
    let mut ended = op.status();
    ended.left.ended = true;
    ended.right.ended = true;
    let charge = op
        .state
        .eviction_workspace_bytes(&ended, op.spec.tolerance_micros(), "asof")
        .unwrap();
    let workspace = op.reserve_workspace(charge).unwrap();
    let mut preview = None;
    let allocation = allocation_counter::measure(|| {
        preview = Some(op.state.preview_eviction(&ended, 0, "asof").unwrap());
    });
    assert_eq!(preview.unwrap().evicted_payloads, count as u64);
    assert!(
        allocation.bytes_max <= charge,
        "eviction preview allocated {}, reserved {charge}",
        allocation.bytes_max
    );
    drop(workspace);
    assert_eq!(op.runtime.pool.reserved(), op.state.right.auxiliary_bytes());
}

#[tokio::test]
async fn finalization_reads_a_ready_left_prefix_as_one_run() {
    let (mut op, left, _) = prefix_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", left, &cx, &mut output)
        .await
        .unwrap();
    state::take_left_visits();
    op.on_end(&cx, &mut output).await.unwrap();
    assert_eq!(
        state::take_left_visits(),
        1,
        "ready prefix required more than one heap run"
    );
    assert_eq!(op.status.emitted_left_rows, 3);
}

#[tokio::test]
async fn finalized_prefix_keeps_index_deferred_until_capture() {
    let (mut op, left, right) = prefix_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut output)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut output)
        .await
        .unwrap();
    op.on_watermark(EventTime::from_micros(103), &cx, &mut output)
        .await
        .unwrap();
    assert_eq!(op.status.emitted_left_rows, 3);
    assert!(op.prepared.is_none(), "output must not serialize the index");
    assert_eq!(
        op.deferred_index_len,
        Some(checkpoint::v3_encoded_length(&op.state, &op.name).unwrap())
    );
    let snapshot = op.capture(Epoch::INITIAL).unwrap();
    assert!(snapshot.segments.contains_key("asof-log-v10-1-0-1"));
}

#[tokio::test]
async fn finalizes_large_ready_prefix_in_one_bounded_batch() {
    let (template, right) = fixture();
    let schema = template.schemas[0].clone();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::ZERO,
        AsofStateLimits::new(1_000, 64 * 1024 * 1024).unwrap(),
    )
    .unwrap();
    let mut op = StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let rows = 300;
    let left = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A"; rows])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100; rows]).with_timezone("UTC")),
                    Arc::new(Int64Array::from_iter_values(
                        0..i64::try_from(rows).unwrap(),
                    )),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(512, 8 << 20).unwrap());
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right, &cx, &mut collector)
        .await
        .unwrap();
    op.process_data("left", left, &cx, &mut collector)
        .await
        .unwrap();
    op.on_watermark(EventTime::from_micros(101), &cx, &mut collector)
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        rows
    );
    assert_eq!(op.status.matched_rows, rows as u64);
    assert_eq!(op.status.pending_left_rows, 0);
}

pub(super) fn fixture() -> (StreamAsofJoinOperator, Batch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(100, 1_048_576).unwrap(),
    )
    .unwrap();
    let operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let batch = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A"])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![1])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    (operator, batch)
}

#[tokio::test]
async fn one_row_output_byte_limit_preserves_pending_state_and_releases_workspace() {
    let (mut op, batch) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::for_task(
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
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", batch, &cx, &mut output)
        .await
        .unwrap();
    let before = op.capture(Epoch::INITIAL).unwrap();
    assert_log_funded(&op);
    let reserved = op.runtime.pool.reserved();
    let error = op
        .on_watermark(EventTime::from_micros(101), &cx, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofOutputLimitExceeded,
            ..
        }
    ));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(cx);
    drop(job);
    assert_eq!(op.runtime.pool.reserved(), reserved);
    assert_eq!(op.status.pending_left_rows, 1);
    assert_eq!(op.status.output_limit_failures, 1);
    assert_eq!(
        op.capture(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
    assert!(output.drain("output").is_empty());
    assert_log_funded(&op);
    op.reset().unwrap();
    drop(before);
    assert_eq!(op.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn ten_thousand_row_admission_fits_the_recommended_workspace_limits() {
    let operator = admit_repro_rows(10_000).await;
    assert!(operator.status.state_bytes < 16 * 1024 * 1024);
}

#[tokio::test]
async fn sixty_thousand_row_batch_admission_uses_columnar_workspace() {
    let operator = admit_repro_rows(60_000).await;
    assert!(operator.status.state_bytes < 64 * 1024 * 1024);
}

async fn admit_repro_rows(rows: u64) -> StreamAsofJoinOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["symbol".into()],
            "event_time".into(),
            vec!["sequence".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::from_secs(10_000_000),
        AsofStateLimits::new(100_000, 64 * 1024 * 1024).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let indexes = 0..i32::try_from(rows).unwrap();
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(
                    indexes.clone().map(|row| 1_000_000 + i64::from(row)),
                )
                .with_timezone("UTC"),
            ),
            Arc::new(UInt64Array::from_iter_values(0..rows)),
            Arc::new(StringArray::from_iter_values(
                indexes.clone().map(|row| format!("S{:03}", row % 64)),
            )),
            Arc::new(Float64Array::from_iter_values(
                indexes.map(|row| 100.0 + f64::from(row % 257)),
            )),
        ],
    )
    .unwrap();
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());

    operator
        .process_data("left", batch, &cx, &mut output)
        .await
        .unwrap();

    assert_eq!(operator.status.left.accepted_rows, rows);
    assert_eq!(operator.status.pending_left_rows, rows);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(cx);
    drop(job);
    assert_eq!(operator.runtime.pool.reserved(), 0);
    operator
}

#[tokio::test]
async fn watermark_tick_without_eviction_reuses_the_prepared_segment() {
    // A watermark tick that neither admits, removes nor evicts anything must
    // not re-encode state: the previously installed segment stays installed.
    let (mut op, _) = fixture();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
    ]));
    let right = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A", "A"])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100, 200]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![1, 2])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    let progress = |watermark: i64| {
        IngressProgressSnapshot::new(BTreeMap::from([
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
        ]))
    };
    let idle = StreamOperatorContext::new(&job, "asof", None);
    op.process_data("right", right, &idle, &mut output)
        .await
        .unwrap();
    let admitted = op.capture(Epoch::INITIAL).unwrap();

    // The first sweep at 150 evicts the row at 100: state changes and is
    // re-encoded into a fresh segment allocation.
    let cx = StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress(150));
    op.on_watermark(EventTime::from_micros(150), &cx, &mut output)
        .await
        .unwrap();
    let swept = op.capture(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(op.status().evicted_right_rows, 1);
    assert_ne!(admitted.segments, swept.segments);

    // An identical tick changes nothing: status is untouched and the prepared
    // segment allocation is reused instead of re-encoded.
    let status = op.status();
    op.on_watermark(EventTime::from_micros(150), &cx, &mut output)
        .await
        .unwrap();
    let repeated = op.capture(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(op.status(), status);
    assert_shared_segments(&swept, &repeated);

    // Watermark progress below the next evictable row also reuses it.
    let cx = StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress(199));
    op.on_watermark(EventTime::from_micros(199), &cx, &mut output)
        .await
        .unwrap();
    let advanced = op.capture(Epoch::new(3).unwrap()).unwrap();
    assert_shared_segments(&swept, &advanced);
    assert_eq!(
        op.status().output_watermark_micros,
        Some(EventTime::from_micros(198))
    );
}

#[tokio::test]
async fn restored_logical_counter_overflow_fails_before_admission_or_emit() {
    let (mut op, batch) = fixture();
    let mut snapshot = op.capture(Epoch::INITIAL).unwrap();
    let metrics = snapshot.inline_metadata.get_mut("metrics").unwrap();
    metrics["right"]["accepted_rows"] = u64::MAX.into();
    metrics["evicted_right_rows"] = u64::MAX.into();
    op.restore(&snapshot).unwrap();
    let before = op.status();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    let error = op
        .process_data("right", batch, &cx, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofCounterOverflow,
            ..
        }
    ));
    assert_eq!(op.status(), before);
    assert!(output.drain("output").is_empty());
    assert_eq!(op.runtime.pool.reserved(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn output_plan_registers_each_side_source_once() {
    let (mut op, batch) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", batch.clone(), &cx, &mut collector)
        .await
        .unwrap();
    op.process_data("left", batch, &cx, &mut collector)
        .await
        .unwrap();
    workspace::take_output_source_registrations();
    op.on_end(&cx, &mut collector).await.unwrap();
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    assert_eq!(op.status().matched_rows, 1);
    assert_eq!(workspace::take_output_source_registrations(), 2);
}

#[tokio::test]
async fn output_plan_zero_columns_preserve_operator_delivery_and_status() {
    let (mut op, batch) = fixture();
    op.set_output_projection(Vec::new()).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", batch.clone(), &cx, &mut collector)
        .await
        .unwrap();
    op.process_data("left", batch, &cx, &mut collector)
        .await
        .unwrap();
    op.on_end(&cx, &mut collector).await.unwrap();
    let output = collector.drain("output");
    let batch = output[0].as_data().unwrap();
    let record = &batch.table_payload().unwrap().batches()[0];
    assert_eq!((record.num_columns(), record.num_rows()), (0, 1));
    assert_eq!(op.status().emitted_left_rows, 1);
    assert_eq!(op.status().matched_rows, 1);
    assert_eq!(op.status().pending_left_rows, 0);
}

#[path = "tests/expiration_index.rs"]
mod expiration_index;

#[path = "tests/expiration_output_integration.rs"]
mod expiration_output_integration;

#[path = "tests/restored_workspace.rs"]
mod restored_workspace;

#[path = "tests/dominated_payloads.rs"]
mod dominated_payloads;

#[path = "tests/retained_projection.rs"]
mod retained_projection;

#[path = "tests/gather_admission.rs"]
mod gather_admission;

#[path = "tests/checkpoint_delta.rs"]
mod checkpoint_delta;
