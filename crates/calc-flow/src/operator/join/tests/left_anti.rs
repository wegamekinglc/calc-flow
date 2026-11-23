use super::*;

fn anti() -> StreamJoinOperator {
    let mut config = spec();
    config.join_type = StreamJoinType::LeftAnti;
    config.bounds = JoinTimeBounds::new(Duration::ZERO, Duration::from_micros(2)).unwrap();
    config.limits = JoinStateLimits::new(10_000, 8 << 20, 1).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), config).unwrap();
    operator.enable_run_core();
    operator
}

fn drain_times(output: &mut EdgeCollector) -> Vec<(u64, Vec<i64>)> {
    output
        .drain("output")
        .iter()
        .map(|message| {
            let batch = message.as_data().unwrap();
            let times = batch
                .table_payload()
                .unwrap()
                .batches()
                .iter()
                .flat_map(|record| {
                    record
                        .column(1)
                        .as_any()
                        .downcast_ref::<TimestampMicrosecondArray>()
                        .unwrap()
                        .values()
                        .iter()
                        .copied()
                })
                .collect();
            (batch.metadata().sequence(), times)
        })
        .collect()
}

#[tokio::test]
async fn test_anti_strict_closure_eof_then_new_left_probe() {
    let mut op = anti();
    let job = job().with_checkpointing(false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let limits = EdgeBudget::new(2, 4096).unwrap();
    let right_progress = |state, time| {
        progress_context(&job, (IngressState::Active, None), (state, time))
            .with_output_budget(limits)
    };
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data(
        "left",
        left_batch(vec![30, 11, 10, 11, 12, 10]),
        &context,
        &mut out,
    )
    .await
    .unwrap();
    op.process_data("right", right_batch(vec![10, 10]), &context, &mut out)
        .await
        .unwrap();
    assert!(drain_times(&mut out).is_empty());
    for (state, time) in [(IngressState::Idle, None), (IngressState::Active, Some(13))] {
        op.on_ingress_progress_with_output("right", &right_progress(state, time), &mut out)
            .await
            .unwrap();
        assert!(drain_times(&mut out).is_empty());
    }
    op.on_ingress_progress_with_output(
        "right",
        &right_progress(IngressState::Active, Some(14)),
        &mut out,
    )
    .await
    .unwrap();
    assert_eq!(drain_times(&mut out), [(0, vec![11, 11])]);
    let ended = right_progress(IngressState::Ended, Some(14));
    op.on_ingress_progress_with_output("right", &ended, &mut out)
        .await
        .unwrap();
    assert_eq!(drain_times(&mut out), [(1, vec![12, 30])]);
    op.process_data("left", left_batch(vec![10, 90]), &ended, &mut out)
        .await
        .unwrap();
    assert_eq!(drain_times(&mut out), [(2, vec![90])]);
    assert_eq!(op.status().left.retained_rows, 0);
}

struct Interrupt {
    accepted: Vec<u64>,
    cancel: Option<CancellationToken>,
}

#[async_trait::async_trait]
impl StreamCollector for Interrupt {
    async fn emit(&mut self, _: &str, batch: Batch) -> Result<()> {
        if !self.accepted.is_empty() {
            return Err(operator_error("test", "sink refused second chunk"));
        }
        self.accepted.push(batch.metadata().sequence());
        if let Some(token) = &self.cancel {
            token.cancel();
        }
        Ok(())
    }
}

#[tokio::test]
async fn test_anti_release_retries_refused_and_cancelled_prefixes_using_native_cuts() {
    let mut op = anti();
    let job = job().with_checkpointing(false);
    let open = StreamOperatorContext::new(&job, "match", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", left_batch(vec![13, 12, 11, 10]), &open, &mut out)
        .await
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let mut store = crate::state::v4::V4Store::new(temp.path().join("lineage"));
    let previous = persist_native_epoch(&mut op, &mut store, 1).await;
    op.process_data("right", right_batch(vec![11]), &open, &mut out)
        .await
        .unwrap();
    assert!(drain_times(&mut out).is_empty());
    let marked = persist_native_epoch(&mut op, &mut store, 2).await;
    let load = |store: &mut _, epoch| {
        crate::state::v4::native::load_native(store, epoch, "match").unwrap()
    };
    let previous = load(&mut store, &previous);
    let marked = load(&mut store, &marked);
    assert_eq!(marked.inline_metadata["layout_version"], 14);
    assert_ne!(
        previous.segments["left/group-0"].sha256(),
        marked.segments["left/group-0"].sha256()
    );
    let end = |job| {
        progress_context(
            job,
            (IngressState::Active, None),
            (IngressState::Ended, None),
        )
        .with_output_budget(EdgeBudget::new(1, 4096).unwrap())
    };
    for (cut, times) in [(&previous, vec![10, 11, 12, 13]), (&marked, vec![12, 13])] {
        for cancel in [false, true] {
            let mut restored = anti();
            restored.restore(cut).unwrap();
            let token = CancellationToken::new();
            let failing_job =
                StreamJobContext::new(1, "retry", JsonMap::new(), None, token.clone())
                    .with_checkpointing(false);
            let failed = progress_context(
                &failing_job,
                (IngressState::Active, None),
                (IngressState::Ended, None),
            )
            .with_output_budget(EdgeBudget::new(1, 4096).unwrap());
            let before = restored.status();
            let mut interrupted = Interrupt {
                accepted: Vec::new(),
                cancel: cancel.then_some(token),
            };
            assert!(
                restored
                    .on_ingress_progress_with_output("right", &failed, &mut interrupted)
                    .await
                    .is_err()
            );
            assert_eq!(interrupted.accepted, [0]);
            assert_eq!(restored.status(), before);
            restored
                .on_ingress_progress_with_output("right", &end(&job), &mut out)
                .await
                .unwrap();
            assert_eq!(
                drain_times(&mut out),
                times
                    .iter()
                    .enumerate()
                    .map(|(i, time)| (i as u64, vec![*time]))
                    .collect::<Vec<_>>()
            );
        }
    }
    op.on_ingress_progress_with_output("right", &end(&job), &mut out)
        .await
        .unwrap();
    assert_eq!(drain_times(&mut out), [(0, vec![12]), (1, vec![13])]);
    let released = persist_native_epoch(&mut op, &mut store, 3).await;
    let mut restored = anti();
    restored.restore(&load(&mut store, &released)).unwrap();
    restored
        .on_ingress_progress_with_output("right", &end(&job), &mut out)
        .await
        .unwrap();
    assert!(drain_times(&mut out).is_empty());
    restored
        .process_data("left", left_batch(vec![80]), &end(&job), &mut out)
        .await
        .unwrap();
    assert_eq!(drain_times(&mut out), [(2, vec![80])]);
}

#[tokio::test]
async fn test_anti_all_witnessed_release_yields_before_committing() {
    let mut op = anti();
    let token = CancellationToken::new();
    let job = StreamJobContext::new(1, "release", JsonMap::new(), None, token.clone())
        .with_checkpointing(false);
    let open = StreamOperatorContext::new(&job, "match", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", left_batch(vec![10; 8193]), &open, &mut out)
        .await
        .unwrap();
    op.process_data("right", right_batch(vec![10]), &open, &mut out)
        .await
        .unwrap();
    let before = op.status();
    let ended = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Ended, None),
    );
    let (result, ()) = tokio::join!(biased;
        op.on_ingress_progress_with_output("right",&ended,&mut out),
        async {token.cancel()}
    );
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(op.status(), before);
    assert!(drain_times(&mut out).is_empty());
    let retry = super::job().with_checkpointing(false);
    let ended = progress_context(
        &retry,
        (IngressState::Active, None),
        (IngressState::Ended, None),
    );
    runs::CLOSED_READS.set(0);
    op.on_ingress_progress_with_output("right", &ended, &mut out)
        .await
        .unwrap();
    assert_eq!(runs::CLOSED_READS.get(), 8193);
    assert_eq!(op.status().left.retained_rows, 0);
    assert!(drain_times(&mut out).is_empty());
}

fn nullable_seconds_operator(schema: Arc<Schema>) -> StreamJoinOperator {
    let spec = StreamJoinSpec::new(
        StreamJoinType::LeftAnti,
        ["k"],
        ["k"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 3).unwrap(),
    )
    .unwrap();
    let mut op = StreamJoinOperator::new("match", schema.clone(), schema, spec).unwrap();
    op.enable_run_core();
    op
}

fn seconds_batch(schema: &Arc<Schema>, keys: Vec<Option<i64>>, times: Vec<Option<i64>>) -> Batch {
    use datafusion::arrow::array::{ArrayRef, TimestampSecondArray};
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(keys)),
        Arc::new(TimestampSecondArray::from(times)),
    ];
    Batch::table(
        vec![RecordBatch::try_new(Arc::clone(schema), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_anti_immediate_rows_merge_in_physical_order_after_admission() {
    use datafusion::arrow::array::TimestampSecondArray;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Second, None), true),
    ]));
    let mut op = nullable_seconds_operator(schema.clone());
    let batch = |keys, times| seconds_batch(&schema, keys, times);
    let job = job().with_checkpointing(false);
    let open = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data(
        "right",
        batch(vec![Some(7), None], vec![Some(1), Some(1)]),
        &open,
        &mut output,
    )
    .await
    .unwrap();
    let ended = progress_context(
        &job,
        (IngressState::Active, Some(1_000_000)),
        (IngressState::Ended, None),
    );
    op.on_ingress_progress_with_output("right", &ended, &mut output)
        .await
        .unwrap();
    let before = op.status();
    let invalid = op
        .process_data(
            "left",
            batch(vec![None, None], vec![None, Some(i64::MAX)]),
            &ended,
            &mut output,
        )
        .await
        .unwrap_err();
    assert_eq!(
        reason_of(&invalid),
        Some(crate::StreamingFailureReason::JoinTimeConversionFailed)
    );
    assert_eq!(op.status(), before);
    let rejected = op
        .process_data(
            "left",
            Batch::table(
                [
                    batch(vec![None, None], vec![None, Some(1)]),
                    batch(vec![Some(8), Some(9)], vec![Some(1), Some(1)]),
                ]
                .into_iter()
                .flat_map(|batch| batch.table_payload().unwrap().batches().to_vec())
                .collect(),
                BatchMetadata::default(),
            )
            .unwrap(),
            &ended,
            &mut output,
        )
        .await
        .unwrap_err();
    assert_eq!(
        reason_of(&rejected),
        Some(crate::StreamingFailureReason::JoinMatchLimitExceeded)
    );
    assert!(output.drain("output").is_empty());
    op.process_data(
        "left",
        batch(
            vec![Some(8), None, None, Some(7), None],
            vec![Some(1), Some(0), Some(1), Some(1), None],
        ),
        &ended,
        &mut output,
    )
    .await
    .unwrap();
    let seen = output.drain("output");
    let columns = seen[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0]
        .columns();
    let keys = columns[0].as_any().downcast_ref::<Int64Array>().unwrap();
    let times = columns[1]
        .as_any()
        .downcast_ref::<TimestampSecondArray>()
        .unwrap();
    assert_eq!(
        keys.iter().zip(times.iter()).collect::<Vec<_>>(),
        [(Some(8), Some(1)), (None, Some(1)), (None, None)]
    );
    assert_eq!(op.status().left.late_rows, 1);
    assert_eq!(op.status().left.null_key_rows, 0);
    assert_eq!(op.status().left.null_event_time_rows, 0);
}

#[tokio::test]
async fn test_anti_unaffordable_final_row_preflights_before_any_control_output() {
    use datafusion::arrow::array::ArrayRef;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("value", DataType::Utf8, false),
    ]));
    let spec = StreamJoinSpec::new(
        StreamJoinType::LeftAnti,
        ["k"],
        ["k"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(32, 1_000_000, 1).unwrap(),
    )
    .unwrap();
    let mut op = StreamJoinOperator::new("match", schema.clone(), schema.clone(), spec).unwrap();
    op.enable_run_core();
    let job = job().with_checkpointing(false);
    let open = StreamOperatorContext::new(&job, "match", None);
    let values = ["a", "bb", "x"]
        .into_iter()
        .map(str::to_owned)
        .chain(std::iter::once("z".repeat(1024)))
        .collect::<Vec<_>>();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![1; 4])),
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2, 3, 4])),
        Arc::new(StringArray::from(values)),
    ];
    let batch = Batch::table(
        vec![RecordBatch::try_new(schema, columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", batch, &open, &mut out)
        .await
        .unwrap();
    let cut = |budget| {
        progress_context(
            &job,
            (IngressState::Active, None),
            (IngressState::Ended, None),
        )
        .with_output_budget(EdgeBudget::new(1, budget).unwrap())
    };
    let before = op.status();
    let error = op
        .on_ingress_progress_with_output("right", &cut(64), &mut out)
        .await
        .unwrap_err();
    assert!(matches!(error,CalcFlowError::InvalidArgument {field,..} if field=="message.bytes"));
    assert!(out.drain("output").is_empty());
    assert_eq!(op.status(), before);
    op.on_ingress_progress_with_output("right", &cut(4096), &mut out)
        .await
        .unwrap();
    assert_eq!(
        drain_times(&mut out),
        [(0, vec![1]), (1, vec![2]), (2, vec![3]), (3, vec![4])]
    );
}

#[derive(Default)]
struct CountReleased {
    batches: u64,
    rows: usize,
    peak_rows: usize,
}

#[async_trait::async_trait]
impl StreamCollector for CountReleased {
    async fn emit(&mut self, _: &str, batch: Batch) -> Result<()> {
        assert_eq!(self.batches, batch.metadata().sequence());
        self.batches += 1;
        self.rows += batch.num_rows();
        self.peak_rows = self.peak_rows.max(batch.num_rows());
        Ok(())
    }
}

#[test]
fn test_anti_release_heap_is_bounded_across_three_retained_morsels() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut op = anti();
    let job = job().with_checkpointing(false);
    let open = StreamOperatorContext::new(&job, "match", None);
    let mut collector = CountReleased::default();
    runtime
        .block_on(op.process_data("left", left_batch(vec![10; 8193]), &open, &mut collector))
        .unwrap();
    let end = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Ended, None),
    )
    .with_output_budget(EdgeBudget::new(32, 4096).unwrap());
    let heap = allocation_counter::measure(|| {
        runtime
            .block_on(op.on_ingress_progress_with_output("right", &end, &mut collector))
            .unwrap();
    });
    assert_eq!(
        (collector.batches, collector.rows, collector.peak_rows),
        (257, 8193, 32)
    );
    assert!(
        heap.bytes_max < 65_536,
        "control workspace peaked at {} bytes",
        heap.bytes_max
    );
    eprintln!(
        "8193 physical rows, {} chunks, max {} rows, peak heap {} bytes",
        collector.batches, collector.peak_rows, heap.bytes_max
    );
}

#[tokio::test]
async fn test_anti_direct_terminal_flush_restores_an_empty_native_cut() {
    let mut op = anti();
    let job = job().with_checkpointing(false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("right", right_batch(vec![10]), &context, &mut output)
        .await
        .unwrap();
    op.process_data("left", left_batch(vec![10, 90]), &context, &mut output)
        .await
        .unwrap();
    let before = op.status();
    let tight = StreamOperatorContext::new(&job, "match", None)
        .with_output_budget(EdgeBudget::new(1, 1).unwrap());
    assert!(op.on_end(&tight, &mut output).await.is_err());
    assert!(drain_times(&mut output).is_empty());
    assert_eq!(op.status(), before);
    op.on_end(&context, &mut output).await.unwrap();
    assert_eq!(drain_times(&mut output), [(0, vec![90])]);
    let temp = tempfile::tempdir().unwrap();
    let mut store = crate::state::v4::V4Store::new(temp.path().join("terminal"));
    let terminal = persist_native_epoch(&mut op, &mut store, 1).await;
    let cut = crate::state::v4::native::load_native(&mut store, &terminal, "match").unwrap();
    assert_eq!(cut.inline_metadata["ended"], true);
    let mut restored = anti();
    restored.restore(&cut).unwrap();
    assert_eq!(restored.status().left.retained_rows, 0);
    assert_eq!(restored.status().right.retained_rows, 0);
    assert_eq!(restored.status().emitted_match_rows, 1);
}

#[tokio::test]
async fn test_anti_repeated_right_message_scans_committed_witnesses_once() {
    let mut op = anti();
    let job = job().with_checkpointing(false);
    let ctx = StreamOperatorContext::new(&job, "match", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", left_batch(vec![10; 8193]), &ctx, &mut out)
        .await
        .unwrap();
    for count in [1, 4096] {
        runs::ELIGIBILITY_READS.set(0);
        op.process_data("right", right_batch(vec![10; count]), &ctx, &mut out)
            .await
            .unwrap();
        assert!(drain_times(&mut out).is_empty());
        assert_eq!(runs::ELIGIBILITY_READS.get(), 8193);
    }
    assert_eq!(op.status().left.retained_rows, 8193);
    assert_eq!(op.status().emitted_match_rows, 0);
}
