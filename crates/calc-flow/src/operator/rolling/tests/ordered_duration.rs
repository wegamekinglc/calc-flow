use crate::operator::rolling_metrics::{RollingCallback, RollingMetricsStore};

use super::*;

#[tokio::test]
async fn test_deferred_duration_windows_retain_one_timed_sample_per_row() {
    const SECOND: i64 = 1_000_000;
    let spec = kernel_spec(json!([
        duration_output("sum", "price", "sum30", 30_000_000),
        duration_output("mean", "price", "mean60", 60_000_000),
        duration_output("sum", "price", "sum90", 90_000_000),
    ]));
    let rows = (0..100u32)
        .map(|i| {
            (
                i64::from(i) * SECOND,
                "a",
                u64::from(i),
                (i % 7 != 0).then_some(f64::from(i)),
            )
        })
        .collect::<Vec<_>>();
    let input = float64_fast_record(&rows);
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    operator
        .set_state_budget(StateBudget::new(90 + 54, 1 << 24).unwrap())
        .unwrap();
    let (expected, _) = reference_output(&operator, &input);
    let (job, store, mut output) = fixture(&operator);
    let job = job.with_checkpointing(false);
    let mut emitted = Vec::new();
    for (offset, len) in [(0, 46), (46, 54)] {
        assert!(
            observe_data(
                &mut operator,
                &job,
                &store,
                &mut output,
                input.slice(offset, len)
            )
            .await
        );
        emitted.extend(
            close(
                &mut operator,
                &job,
                &store,
                &mut output,
                rows[offset + len - 1].0,
            )
            .await,
        );
    }
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
    assert!(operator.state.histories.by_entity.is_empty());
    assert_deferred_charge(&operator);
    assert_eq!(
        operator
            .state
            .typed_kernel_state
            .as_deref()
            .unwrap()
            .physical_duration_samples(),
        90
    );
    assert_eq!(operator.state.charge.rows, 90);
}

#[tokio::test]
async fn test_shared_duration_sum_and_mean_match_independent_ieee_results() {
    let spec = kernel_spec(json!([
        duration_output("sum", "price", "sum5", 5),
        duration_output("mean", "price", "mean80", 80),
    ]));
    let special = [
        None,
        Some(-0.0),
        Some(0.0),
        Some(1e155),
        Some(-1e155),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
        Some(f64::from_bits(0x7ff8_0000_0000_1234)),
        Some(1e-300),
    ];
    let rows = (0..300_u32)
        .map(|i| {
            let value = if i % 150 < 30 {
                special[i as usize % special.len()]
            } else {
                Some([1e16, 1.0, -1e16][i as usize % 3])
            };
            (i64::from(i), "a", u64::from(i), value)
        })
        .collect::<Vec<_>>();
    let input = float64_fast_record(&rows);
    let mut results = Vec::new();
    for checkpointing in [true, false] {
        let mut operator = RollingOperator::new("rolling", input.schema(), spec.clone()).unwrap();
        let (job, store, mut output) = fixture(&operator);
        let job = job.with_checkpointing(checkpointing);
        let mut emitted = Vec::new();
        for offset in [0, 100, 200] {
            assert!(
                observe_data(
                    &mut operator,
                    &job,
                    &store,
                    &mut output,
                    input.slice(offset, 100)
                )
                .await
            );
            emitted.extend(
                close(
                    &mut operator,
                    &job,
                    &store,
                    &mut output,
                    rows[offset + 99].0,
                )
                .await,
            );
        }
        assert_eq!(operator.state.deferred_history, !checkpointing);
        results.push(concat_batches(&emitted[0].schema(), &emitted).unwrap());
    }
    capability_profiles::assert_complete_bits(&results[0], &results[1]);
}

#[tokio::test]
async fn test_shared_duration_eviction_retries_and_closes_out_of_order_rows() {
    let spec = kernel_spec(json!([
        duration_output("sum", "price", "short", 3),
        duration_output("mean", "price", "long", 8),
    ]));
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, None),
        (3, "a", 3, Some(3.0)),
        (6, "a", 4, Some(4.0)),
        (6, "a", 5, Some(5.0)),
        (20, "a", 6, Some(6.0)),
        (21, "a", 7, None),
        (25, "a", 9, Some(9.0)),
        (23, "a", 8, Some(8.0)),
    ]);
    for alias in [false, true] {
        let mut operator = RollingOperator::new("rolling", input.schema(), spec.clone()).unwrap();
        let (expected, _) = reference_output(&operator, &input);
        let (job, store, mut output) = fixture(&operator);
        let job = job.with_checkpointing(false);
        assert!(observe_data(&mut operator, &job, &store, &mut output, input.slice(0, 5)).await);
        let mut emitted = close(&mut operator, &job, &store, &mut output, 6).await;
        let prior = format!("{:?}", operator.state.typed_kernel_state);
        let _alias = alias.then(|| operator.state.typed_kernel_state.clone());
        assert!(observe_data(&mut operator, &job, &store, &mut output, input.slice(5, 2)).await);
        let charge = operator.state.charge;
        let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(6)));
        let error = operator
            .on_watermark(
                EventTime::from_micros(21),
                &context,
                &mut crate::EdgeCollector::new(Vec::new()),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("unknown output port"), "{error}");
        assert_eq!(format!("{:?}", operator.state.typed_kernel_state), prior);
        assert_eq!(operator.state.charge, charge);
        emitted.extend(close(&mut operator, &job, &store, &mut output, 21).await);
        assert!(!observe_data(&mut operator, &job, &store, &mut output, input.slice(7, 2)).await);
        emitted.extend(close(&mut operator, &job, &store, &mut output, 25).await);
        assert_eq!(
            concat_batches(&expected.schema(), &emitted).unwrap(),
            expected
        );
        assert_eq!(operator.state.charge.rows, 4);
        assert_deferred_charge(&operator);
        let error = operator.checkpoint(Epoch::new(1).unwrap()).unwrap_err();
        assert!(
            error.to_string().contains("checkpointing is disabled"),
            "{error}"
        );
    }
}

#[tokio::test]
async fn test_full_duration_tape_expires_before_growing_under_a_byte_budget() {
    let spec = kernel_spec(json!([
        duration_output("sum", "price", "short", 1),
        duration_output("sum", "price", "long", 64),
    ]));
    let rows = (0..65_u32)
        .map(|row| (i64::from(row), "a", u64::from(row), Some(f64::from(row))))
        .collect::<Vec<_>>();
    let input = float64_fast_record(&rows);
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let (expected, _) = reference_output(&operator, &input);
    let (job, store, mut output) = fixture(&operator);
    let job = job.with_checkpointing(false);
    assert!(observe_data(&mut operator, &job, &store, &mut output, input.slice(0, 64)).await);
    let mut emitted = close(&mut operator, &job, &store, &mut output, 63).await;
    let initial = operator.state.charge;
    assert_eq!(initial.rows, 64);
    assert!(observe_data(&mut operator, &job, &store, &mut output, input.slice(64, 1)).await);
    operator
        .set_state_budget(StateBudget::new(128, operator.state.charge.bytes).unwrap())
        .unwrap();
    emitted.extend(close(&mut operator, &job, &store, &mut output, 64).await);
    assert_eq!(operator.state.charge, initial);
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
}

#[tokio::test]
async fn test_aborted_first_duration_delivery_keeps_durable_unordered_history() {
    struct Pending;
    #[async_trait::async_trait]
    impl StreamCollector for Pending {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            std::future::pending().await
        }
    }
    let spec = kernel_spec(json!([
        duration_output("mean", "price", "short", 3),
        duration_output("sum", "price", "long", 8),
    ]));
    let input = float64_fast_record(&[
        (1, "a", 1, Some(100.0)),
        (6, "a", 3, Some(6.0)),
        (5, "a", 2, Some(5.0)),
        (7, "a", 4, Some(7.0)),
    ]);
    let mut operator = RollingOperator::new("rolling", input.schema(), spec.clone()).unwrap();
    let (expected, _) = reference_output(&operator, &input.slice(1, 3));
    let (durable, store, mut output) = fixture(&operator);
    let ephemeral = durable.clone().with_checkpointing(false);
    assert!(
        observe_data(
            &mut operator,
            &ephemeral,
            &store,
            &mut output,
            input.slice(0, 1)
        )
        .await
    );
    let context = StreamOperatorContext::new(&ephemeral, "rolling", None);
    let mut collector = Pending;
    {
        let mut callback =
            Box::pin(operator.on_watermark(EventTime::from_micros(1), &context, &mut collector));
        assert!(futures::poll!(callback.as_mut()).is_pending());
    }
    assert!(!operator.state.deferred_history);
    assert!(
        !observe_data(
            &mut operator,
            &durable,
            &store,
            &mut output,
            input.slice(1, 2)
        )
        .await
    );
    let mut emitted = close(&mut operator, &durable, &store, &mut output, 6).await;
    let checkpoint = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let mut restored = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    StreamOperator::restore(&mut restored, &checkpoint).unwrap();
    assert!(
        observe_data(
            &mut restored,
            &durable,
            &store,
            &mut output,
            input.slice(3, 1)
        )
        .await
    );
    emitted.extend(close(&mut restored, &durable, &store, &mut output, 7).await);
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
}

#[tokio::test]
async fn test_failed_first_deferred_duration_output_allows_durable_retry_and_restore() {
    let spec = kernel_spec(json!([
        duration_output("mean", "price", "short", 3),
        duration_output("sum", "price", "long", 8),
    ]));
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, None),
        (3, "a", 3, Some(3.0)),
        (9, "a", 4, Some(9.0)),
    ]);
    for checkpointing in [false, true] {
        let mut operator = RollingOperator::new("rolling", input.schema(), spec.clone()).unwrap();
        let (expected, _) = reference_output(&operator, &input);
        let (durable, store, mut output) = fixture(&operator);
        let (ephemeral, _, _) = fixture(&operator);
        let ephemeral = ephemeral.with_checkpointing(false);
        assert!(
            observe_data(
                &mut operator,
                &ephemeral,
                &store,
                &mut output,
                input.slice(0, 3)
            )
            .await
        );
        let context = StreamOperatorContext::new(&ephemeral, "rolling", None);
        assert!(
            operator
                .on_watermark(
                    EventTime::from_micros(3),
                    &context,
                    &mut crate::EdgeCollector::new(vec![])
                )
                .await
                .is_err()
        );
        let retry = if checkpointing { &durable } else { &ephemeral };
        let mut emitted = close(&mut operator, retry, &store, &mut output, 3).await;
        assert_eq!(
            operator
                .state
                .typed_kernel_state
                .as_deref()
                .unwrap()
                .physical_duration_samples(),
            if checkpointing { 6 } else { 3 }
        );
        if !checkpointing {
            assert_deferred_charge(&operator);
            assert!(operator.state.deferred_history);
            continue;
        }
        let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
        let mut restored = RollingOperator::new("rolling", input.schema(), spec.clone()).unwrap();
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        assert!(
            observe_data(
                &mut restored,
                &durable,
                &store,
                &mut output,
                input.slice(3, 1)
            )
            .await
        );
        emitted.extend(close(&mut restored, &durable, &store, &mut output, 9).await);
        assert_eq!(
            concat_batches(&expected.schema(), &emitted).unwrap(),
            expected
        );
    }
}

#[tokio::test]
async fn test_ordered_duration_mixed_input_stays_arrow_across_checkpoint() {
    let spec = kernel_spec(json!([
        duration_output("mean", "price", "time_mean", 6),
        aggregate_output("min", "price", "row_min", 2),
    ]));
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, Some(2.0)),
        (3, "a", 3, Some(3.0)),
        (4, "a", 4, Some(4.0)),
        (7, "a", 5, Some(7.0)),
        (8, "a", 6, Some(8.0)),
        (9, "a", 7, Some(9.0)),
        (10, "a", 8, Some(10.0)),
    ]);
    let mut operator = RollingOperator::new("rolling", input.schema(), spec.clone()).unwrap();
    let (expected, _) = reference_output(&operator, &input);
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let store = RollingMetricsStore::default();
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let first_arrow =
        observe_data(&mut operator, &job, &store, &mut output, input.slice(0, 4)).await;
    let mut emitted = close(&mut operator, &job, &store, &mut output, 4).await;
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let mut restored = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let second_arrow =
        observe_data(&mut restored, &job, &store, &mut output, input.slice(4, 4)).await;
    emitted.extend(close(&mut restored, &job, &store, &mut output, 10).await);
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
    assert_cached_rolling_charge(&restored);
    let observed = store.snapshot();
    assert!(
        first_arrow && second_arrow && observed.data.scalar_value_conversions == 0,
        "ordered Duration/mixed input must preserve Arrow owners without the scalar input bridge: first={first_arrow}, second={second_arrow}, data scalar conversions={}",
        observed.data.scalar_value_conversions
    );
}

#[tokio::test]
async fn test_ordered_duration_hybrid_chunks_expire_and_fallback_in_order() {
    let spec = kernel_spec(json!([
        duration_output("mean", "price", "time_mean", 100),
        aggregate_output("min", "price", "row_min", 2),
    ]));
    let tiny_label = "owned-tiny-payload".repeat(64);
    let steps = [
        labeled_range(1, 8, Some("dense-first")),
        labeled_range(9, 1, Some(&tiny_label)),
        labeled_range(10, 1, Some("tiny-second")),
        labeled_range(11, 8, Some("dense-last")),
    ];
    let gap = labeled_range(104, 1, None);
    let unordered =
        float64_fast_record(&[(106, "a", 20, Some(106.0)), (105, "a", 19, Some(105.0))]);
    let mut inputs = steps.to_vec();
    inputs.extend([gap.clone(), unordered.clone()]);
    let input = concat_batches(&steps[0].schema(), &inputs).unwrap();
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let (expected, reference) = reference_output(&operator, &input);
    let (job, store, mut output) = fixture(&operator);
    let mut emitted = Vec::new();
    for (record, watermark) in steps.into_iter().zip([8, 9, 10, 18]) {
        assert!(observe_data(&mut operator, &job, &store, &mut output, record).await);
        emitted.extend(close(&mut operator, &job, &store, &mut output, watermark).await);
        assert_cached_rolling_charge(&operator);
    }
    let retained_label = assert_hybrid_payload(&operator);
    incremental_accounting::assert_exact_byte_boundary(&mut operator);
    assert!(observe_data(&mut operator, &job, &store, &mut output, gap).await);
    emitted.extend(close(&mut operator, &job, &store, &mut output, 104).await);
    assert_cached_rolling_charge(&operator);
    assert_eq!(
        operator
            .state
            .histories
            .by_entity
            .values()
            .next()
            .unwrap()
            .columnar
            .records[0]
            .len(),
        4
    );
    assert!(!observe_data(&mut operator, &job, &store, &mut output, unordered).await);
    emitted.extend(close(&mut operator, &job, &store, &mut output, 106).await);
    assert_cached_rolling_charge(&operator);
    incremental_accounting::assert_exact_byte_boundary(&mut operator);
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
    let history = operator.state.histories.by_entity.values().next().unwrap();
    assert_eq!(history.transition_count, 21);
    assert!(history.columnar.records.is_empty());
    let tiny = history
        .rows
        .iter()
        .find(|row| history_event_time(row, &operator.compiled) == 9)
        .unwrap();
    assert!(matches!(&tiny[5], ScalarValue::Utf8(Some(label)) if label.as_ptr() == retained_label));
    assert_canonical_histories(&mut operator, &input, &reference);
    let observed = store.snapshot().watermark;
    assert!(observed.history_charge_scalar_rows > 0);
    assert!(observed.history_charge_columnar_chunks > 0);
    assert!(observed.history_append_rows_prepared > 0);
    assert!(observed.history_evict_rows_prepared > 0);
}

#[tokio::test]
async fn test_ordered_duration_late_duplicate_drops_before_duplicate_validation() {
    let mut spec = kernel_spec(json!([duration_output("mean", "price", "time_mean", 6)]));
    spec.late_policy = LatePolicySpec::Drop { metrics_version: 1 };
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let (job, store, mut output) = fixture(&operator);
    let first = labeled_range(1, 2, None);
    assert!(observe_data(&mut operator, &job, &store, &mut output, first).await);
    close(&mut operator, &job, &store, &mut output, 2).await;
    let mixed = float64_fast_record(&[(2, "a", 2, Some(2.0)), (3, "a", 3, Some(3.0))]);
    assert!(!observe_data(&mut operator, &job, &store, &mut output, mixed).await);
    let emitted = close(&mut operator, &job, &store, &mut output, 3).await;
    assert_eq!(emitted.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    assert_eq!(
        emitted[0]
            .column(6)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0)
            .to_bits(),
        2.0_f64.to_bits()
    );
    assert_cached_rolling_charge(&operator);
    assert_eq!(
        operator
            .state
            .histories
            .by_entity
            .values()
            .next()
            .unwrap()
            .transition_count,
        3
    );
    let before = operator.state.charge;
    let cache = operator
        .state
        .histories
        .by_entity
        .values()
        .next()
        .unwrap()
        .charge_cache;
    let duplicate = float64_fast_record(&[(4, "a", 4, Some(4.0)), (4, "a", 4, Some(4.0))]);
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(3)));
    let error = operator
        .process_data(
            "input",
            Batch::table(vec![duplicate], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("duplicate"), "{error}");
    assert_eq!(operator.state.charge, before);
    assert_eq!(
        operator
            .state
            .histories
            .by_entity
            .values()
            .next()
            .unwrap()
            .charge_cache,
        cache
    );
    assert!(operator.state.buffer.is_empty() && operator.state.ordered.is_empty());
}

#[tokio::test]
async fn test_checkpoint_disabled_small_batch_reuses_resident_numeric_state() {
    let spec = kernel_spec(json!([
        duration_output("sum", "price", "sum", 6),
        duration_output("mean", "price", "mean", 10),
    ]));
    let first = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "b", 2, Some(2.0)),
        (3, "a", 3, Some(3.0)),
    ]);
    let second = float64_fast_record(&[
        (4, "b", 4, Some(4.0)),
        (5, "a", 5, Some(5.0)),
        (10, "a", 6, Some(10.0)),
    ]);
    let input = concat_batches(&first.schema(), [&first, &second]).unwrap();
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let (expected, _) = reference_output(&operator, &input);
    let (job, store, mut output) = fixture(&operator);
    let job = job.with_checkpointing(false);
    let mut emitted = Vec::new();
    for (record, watermark) in [(first, 3), (second, 10)] {
        assert!(observe_data(&mut operator, &job, &store, &mut output, record).await);
        emitted.extend(close(&mut operator, &job, &store, &mut output, watermark).await);
    }
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
    assert_eq!(store.snapshot().watermark.copied_entities, 0);
    assert_deferred_charge(&operator);
}

#[tokio::test]
async fn test_checkpoint_disabled_rows_mean_rejection_restores_full_window() {
    struct Reject;
    #[async_trait::async_trait]
    impl StreamCollector for Reject {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            Err(internal_error("expected row mean collector rejection"))
        }
    }

    let spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 2)]));
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, Some(2.0)),
        (3, "a", 3, Some(3.0)),
        (10, "a", 4, Some(10.0)),
    ]);
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let (expected, _) = reference_output(&operator, &input);
    let (job, store, mut output) = fixture(&operator);
    let job = job.with_checkpointing(false);
    assert!(observe_data(&mut operator, &job, &store, &mut output, input.slice(0, 2)).await);
    let mut emitted = close(&mut operator, &job, &store, &mut output, 2).await;
    let prior = format!("{:?}", operator.state.typed_kernel_state);
    let shapes = operator
        .state
        .typed_kernel_state
        .as_deref()
        .unwrap()
        .numeric_queue_shapes_for_test();
    let sequence = operator.state.next_output_sequence;
    assert!(observe_data(&mut operator, &job, &store, &mut output, input.slice(2, 2)).await);
    let charge = operator.state.charge;
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(2)));
    let error = operator
        .on_watermark(EventTime::from_micros(10), &context, &mut Reject)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expected row mean collector rejection")
    );
    assert_eq!(format!("{:?}", operator.state.typed_kernel_state), prior);
    assert_eq!(
        operator
            .state
            .typed_kernel_state
            .as_deref()
            .unwrap()
            .numeric_queue_shapes_for_test(),
        shapes
    );
    assert_eq!(operator.state.charge, charge);
    assert_eq!(operator.state.next_output_sequence, sequence);
    emitted.extend(close(&mut operator, &job, &store, &mut output, 10).await);
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
}

#[tokio::test]
async fn test_warm_direct_entities_skip_canonical_key_encoding() {
    let spec = kernel_spec(json!([duration_output("sum", "price", "sum", 6)]));
    let first = float64_fast_record(&[(1, "a", 1, Some(1.0)), (2, "b", 2, Some(2.0))]);
    let second = float64_fast_record(&[(3, "a", 3, Some(3.0)), (4, "b", 4, Some(4.0))]);
    let input = concat_batches(&first.schema(), [&first, &second]).unwrap();
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let (expected, _) = reference_output(&operator, &input);
    let (job, store, mut output) = fixture(&operator);
    let job = job.with_checkpointing(false);
    let mut emitted = Vec::new();
    for (record, watermark) in [(first, 2), (second, 4)] {
        assert!(observe_data(&mut operator, &job, &store, &mut output, record).await);
        emitted.extend(close(&mut operator, &job, &store, &mut output, watermark).await);
        if watermark == 2 {
            assert_eq!(kernel::take_direct_entity_key_encodings(), 2);
        }
    }
    assert_eq!(kernel::take_direct_entity_key_encodings(), 0);
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
}

#[tokio::test]
async fn test_checkpoint_disabled_numeric_stream_skips_duplicate_recovery_rows() {
    let spec = kernel_spec(json!([duration_output("sum", "price", "sum", 6)]));
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, Some(2.0)),
        (3, "a", 3, Some(3.0)),
        (10, "a", 4, Some(10.0)),
    ]);
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let (expected, _) = reference_output(&operator, &input);
    let (job, store, mut output) = fixture(&operator);
    let job = job.with_checkpointing(false);
    let mut emitted = Vec::new();
    for (record, watermark) in [(input.slice(0, 2), 2), (input.slice(2, 2), 10)] {
        assert!(observe_data(&mut operator, &job, &store, &mut output, record).await);
        emitted.extend(close(&mut operator, &job, &store, &mut output, watermark).await);
        assert!(operator.state.charge.rows > 0);
    }
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
    assert!(operator.state.histories.by_entity.is_empty());
    assert_eq!(store.snapshot().watermark.history_rows_materialized, 0);
    let error = operator.checkpoint(Epoch::new(1).unwrap()).unwrap_err();
    assert!(
        error.to_string().contains("checkpointing is disabled"),
        "{error}"
    );
}

#[tokio::test]
async fn test_deferred_numeric_history_matches_reference_after_unordered_fallback() {
    let spec = kernel_spec(json!([duration_output("sum", "price", "sum", 6)]));
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, Some(2.0)),
        (4, "a", 4, Some(4.0)),
        (3, "a", 3, Some(3.0)),
    ]);
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let (expected, _) = reference_output(&operator, &input);
    let (job, store, mut output) = fixture(&operator);
    let job = job.with_checkpointing(false);
    let mut emitted = Vec::new();
    assert!(observe_data(&mut operator, &job, &store, &mut output, input.slice(0, 2)).await);
    emitted.extend(close(&mut operator, &job, &store, &mut output, 2).await);
    assert!(!observe_data(&mut operator, &job, &store, &mut output, input.slice(2, 2)).await);
    emitted.extend(close(&mut operator, &job, &store, &mut output, 4).await);
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
    assert!(!operator.state.histories.by_entity.is_empty());
    assert!(operator.state.deferred_history);
    assert_deferred_charge(&operator);
    let next = float64_fast_record(&[(5, "a", 5, Some(5.0))]);
    assert!(observe_data(&mut operator, &job, &store, &mut output, next).await);
    close(&mut operator, &job, &store, &mut output, 5).await;
    assert_deferred_charge(&operator);
}

#[tokio::test]
async fn test_deferred_numeric_history_restores_state_after_output_rejection() {
    struct Reject;
    #[async_trait::async_trait]
    impl StreamCollector for Reject {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            Err(internal_error("expected deferred collector rejection"))
        }
    }

    let spec = kernel_spec(json!([
        duration_output("sum", "price", "sum", 6),
        duration_output("mean", "price", "mean", 6),
    ]));
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, Some(2.0)),
        (3, "a", 3, Some(3.0)),
        (10, "a", 4, Some(10.0)),
        (20, "a", 5, Some(20.0)),
        (21, "b", 6, Some(21.0)),
    ]);
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let (expected, _) = reference_output(&operator, &input);
    let (job, store, mut output) = fixture(&operator);
    let job = job.with_checkpointing(false);
    assert!(observe_data(&mut operator, &job, &store, &mut output, input.slice(0, 3)).await);
    let mut emitted = close(&mut operator, &job, &store, &mut output, 3).await;
    let prior = format!("{:?}", operator.state.typed_kernel_state);
    let shapes = operator
        .state
        .typed_kernel_state
        .as_deref()
        .unwrap()
        .numeric_queue_shapes_for_test();
    let sequence = operator.state.next_output_sequence;
    assert!(observe_data(&mut operator, &job, &store, &mut output, input.slice(3, 3)).await);
    let charge = operator.state.charge;
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(3)));
    let error = operator
        .on_watermark(EventTime::from_micros(21), &context, &mut Reject)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expected deferred collector rejection")
    );
    assert_eq!(format!("{:?}", operator.state.typed_kernel_state), prior);
    assert_eq!(
        operator
            .state
            .typed_kernel_state
            .as_deref()
            .unwrap()
            .numeric_queue_shapes_for_test(),
        shapes
    );
    assert_eq!(operator.state.charge, charge);
    assert_eq!(operator.state.next_output_sequence, sequence);
    assert_eq!(operator.state.ordered.records().count(), 1);
    assert_deferred_charge(&operator);
    emitted.extend(close(&mut operator, &job, &store, &mut output, 21).await);
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
}

#[tokio::test]
async fn test_deferred_numeric_history_rolls_back_when_output_future_is_dropped() {
    struct Pending {
        entered: Arc<tokio::sync::Notify>,
    }

    #[async_trait::async_trait]
    impl StreamCollector for Pending {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            self.entered.notify_one();
            std::future::pending::<Result<()>>().await
        }
    }

    let spec = kernel_spec(json!([duration_output("sum", "price", "sum", 6)]));
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (10, "a", 2, Some(10.0)),
        (21, "b", 3, Some(21.0)),
    ]);
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let (job, store, mut output) = fixture(&operator);
    let job = job.with_checkpointing(false);
    assert!(observe_data(&mut operator, &job, &store, &mut output, input.slice(0, 1)).await);
    close(&mut operator, &job, &store, &mut output, 1).await;
    assert!(observe_data(&mut operator, &job, &store, &mut output, input.slice(1, 2)).await);
    let prior = operator
        .state
        .typed_kernel_state
        .as_deref()
        .unwrap()
        .canonical_state_for_test();
    let sequence = operator.state.next_output_sequence;
    let operator = Arc::new(tokio::sync::Mutex::new(operator));
    let entered = Arc::new(tokio::sync::Notify::new());
    let task = {
        let operator = Arc::clone(&operator);
        let entered = Arc::clone(&entered);
        let job = job.clone();
        tokio::spawn(async move {
            let mut operator = operator.lock().await;
            let context =
                StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(1)));
            operator
                .on_watermark(
                    EventTime::from_micros(21),
                    &context,
                    &mut Pending { entered },
                )
                .await
        })
    };
    entered.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let operator = operator.lock().await;
    assert_eq!(
        operator
            .state
            .typed_kernel_state
            .as_deref()
            .unwrap()
            .canonical_state_for_test(),
        prior
    );
    assert_eq!(operator.state.next_output_sequence, sequence);
}

#[tokio::test]
async fn test_aliased_duration_state_uses_transactional_update_and_preserves_capacity() {
    struct Reject;
    #[async_trait::async_trait]
    impl StreamCollector for Reject {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            Err(internal_error("expected aliased collector rejection"))
        }
    }

    let spec = kernel_spec(json!([duration_output("sum", "price", "sum", 100)]));
    let burst = float64_fast_record(
        &(1_u32..=32)
            .map(|time| (i64::from(time), "a", u64::from(time), Some(f64::from(time))))
            .collect::<Vec<_>>(),
    );
    let gap = float64_fast_record(&[(1_000, "a", 33, Some(1_000.0))]);
    let next = float64_fast_record(&[(1_001, "a", 34, Some(1_001.0))]);
    let input = concat_batches(&burst.schema(), [&burst, &gap, &next]).unwrap();
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let (expected, _) = reference_output(&operator, &input);
    let (job, store, mut output) = fixture(&operator);
    let job = job.with_checkpointing(false);
    let mut emitted = Vec::new();
    for (record, watermark) in [(burst, 32), (gap, 1_000)] {
        assert!(observe_data(&mut operator, &job, &store, &mut output, record).await);
        emitted.extend(close(&mut operator, &job, &store, &mut output, watermark).await);
    }
    let kernel = operator.state.typed_kernel_state.as_deref().unwrap();
    let prior = kernel.canonical_state_for_test();
    let shapes = kernel.numeric_queue_shapes_for_test();
    assert!(shapes[0][0].1 > shapes[0][0].0 + 1, "{shapes:?}");
    let _alias = operator.state.typed_kernel_state.clone();
    let copied = store.snapshot().watermark.copied_entities;
    assert!(observe_data(&mut operator, &job, &store, &mut output, next).await);
    let charge = operator.state.charge;
    let callback = store.begin(RollingCallback::Watermark, crate::CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(1_000)))
        .with_rolling_metrics(callback.recorder());
    let result = operator
        .on_watermark(EventTime::from_micros(1_001), &context, &mut Reject)
        .await;
    callback.complete(&result);
    let error = result.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expected aliased collector rejection")
    );
    let kernel = operator.state.typed_kernel_state.as_deref().unwrap();
    assert_eq!(kernel.canonical_state_for_test(), prior);
    assert_eq!(kernel.numeric_queue_shapes_for_test(), shapes);
    assert_eq!(operator.state.charge, charge);
    assert_deferred_charge(&operator);
    assert!(store.snapshot().watermark.copied_entities > copied);
    emitted.extend(close(&mut operator, &job, &store, &mut output, 1_001).await);
    assert_eq!(
        concat_batches(&expected.schema(), &emitted).unwrap(),
        expected
    );
}

#[tokio::test]
async fn test_deferred_shared_rows_keep_their_retained_row_charge() {
    let spec = kernel_spec(json!([
        aggregate_output("mean", "price", "fast", 2),
        aggregate_output("mean", "price", "slow", 8),
    ]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let (job, store, mut output) = fixture(&operator);
    let job = job.with_checkpointing(false);
    let first = float64_fast_record(&[(1, "a", 1, Some(1.0)), (2, "a", 2, Some(2.0))]);
    assert!(observe_data(&mut operator, &job, &store, &mut output, first).await);
    close(&mut operator, &job, &store, &mut output, 2).await;
    assert_deferred_charge(&operator);
    let next = float64_fast_record(&[(3, "a", 3, Some(3.0)), (4, "a", 4, Some(4.0))]);
    assert!(observe_data(&mut operator, &job, &store, &mut output, next).await);
    close(&mut operator, &job, &store, &mut output, 4).await;
    assert_eq!(operator.state.charge.rows, 4);
    assert_eq!(
        RollingKernelPlan::slab_window_count_for_test(
            operator.state.typed_kernel_state.as_deref().unwrap()
        ),
        Some(2)
    );
    assert_deferred_charge(&operator);
}

fn assert_deferred_charge(operator: &RollingOperator) {
    let typed_state = operator.state.typed_kernel_state.as_deref().unwrap();
    let typed = RollingKernelPlan::slab_charge(typed_state, "rolling")
        .unwrap()
        .unwrap_or_else(|| typed_state.total_charge_for_test("rolling").unwrap());
    let history = budget::full_state_charge(
        &operator.state.buffer,
        &operator.state.ordered,
        &operator.state.histories,
        &operator.compiled,
        "rolling",
    )
    .unwrap();
    assert_eq!(
        operator.state.charge,
        history.checked_add(typed, "rolling").unwrap()
    );
}

fn assert_hybrid_payload(operator: &RollingOperator) -> *const u8 {
    let history = operator.state.histories.by_entity.values().next().unwrap();
    assert_eq!(
        history
            .columnar
            .records
            .iter()
            .map(|chunk| matches!(chunk, history::RetainedRows::Columnar(_)))
            .collect::<Vec<_>>(),
        [true, false, true]
    );
    let history::RetainedRows::Scalar(rows) = &history.columnar.records[1] else {
        panic!("the two tiny tails must share one scalar block");
    };
    assert_eq!(rows.len(), 2);
    let ScalarValue::Utf8(Some(label)) = &rows[0][5] else {
        panic!("missing retained tiny payload");
    };
    label.as_ptr()
}

fn assert_canonical_histories(
    operator: &mut RollingOperator,
    input: &RecordBatch,
    reference: &RollingHistories,
) {
    operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let encode = |histories: &RollingHistories| {
        encode_state_segment(
            histories,
            &BTreeMap::new(),
            input.schema().as_ref(),
            &operator.compiled,
            TEST_FINGERPRINT,
            "rolling",
        )
        .unwrap()
    };
    assert_eq!(encode(&operator.state.histories), encode(reference));
}

fn fixture(
    operator: &RollingOperator,
) -> (
    crate::StreamJobContext,
    RollingMetricsStore,
    crate::EdgeCollector,
) {
    (
        crate::StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        ),
        RollingMetricsStore::default(),
        crate::EdgeCollector::new(operator.output_ports().to_vec()),
    )
}

fn labeled_range(start: u32, count: u32, label: Option<&str>) -> RecordBatch {
    use datafusion::arrow::array::StringArray;

    let rows = (start..start + count)
        .map(|time| (i64::from(time), "a", u64::from(time), Some(f64::from(time))))
        .collect::<Vec<_>>();
    let record = float64_fast_record(&rows);
    let mut columns = record.columns().to_vec();
    columns[5] = Arc::new(StringArray::from(vec![
        label;
        usize::try_from(count).unwrap()
    ]));
    RecordBatch::try_new(record.schema(), columns).unwrap()
}

fn reference_output(
    operator: &RollingOperator,
    input: &RecordBatch,
) -> (RecordBatch, RollingHistories) {
    let batch = Batch::table(vec![input.clone()], BatchMetadata::default()).unwrap();
    let mut rows = read_buffered_rows(
        batch.table_payload().unwrap(),
        &operator.compiled,
        "rolling",
    )
    .unwrap();
    rows.sort_by(|left, right| left.identity.cmp(&right.identity));
    let computed = compute_output_columns(
        &rows,
        &RollingHistories::default(),
        &operator.compiled,
        "rolling",
    )
    .unwrap();
    let record = build_output_record(
        &rows,
        computed.columns,
        operator.output_ports()[0].schema().unwrap(),
        "rolling",
    )
    .unwrap();
    let mut histories = RollingHistories::default();
    histories.apply(computed.touched);
    (record, histories)
}

async fn observe_data(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    store: &RollingMetricsStore,
    output: &mut crate::EdgeCollector,
    input: RecordBatch,
) -> bool {
    let callback = store.begin(RollingCallback::Data, crate::CancellationToken::new());
    let context = StreamOperatorContext::new(job, "rolling", operator.state.last_input_watermark)
        .with_rolling_metrics(callback.recorder());
    let result = operator
        .process_data(
            "input",
            Batch::table(vec![input.clone()], BatchMetadata::default()).unwrap(),
            &context,
            output,
        )
        .await;
    callback.complete(&result);
    result.unwrap();
    operator.state.buffer.is_empty()
        && operator.state.ordered.records().any(|record| {
            record.num_rows() == input.num_rows()
                && record
                    .columns()
                    .iter()
                    .zip(input.columns())
                    .all(|(actual, expected)| Arc::ptr_eq(actual, expected))
        })
}

async fn close(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    store: &RollingMetricsStore,
    output: &mut crate::EdgeCollector,
    watermark: i64,
) -> Vec<RecordBatch> {
    let callback = store.begin(RollingCallback::Watermark, crate::CancellationToken::new());
    let context = StreamOperatorContext::new(job, "rolling", operator.state.last_input_watermark)
        .with_rolling_metrics(callback.recorder());
    let result = operator
        .on_watermark(EventTime::from_micros(watermark), &context, output)
        .await;
    callback.complete(&result);
    result.unwrap();
    output
        .drain("output")
        .into_iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .to_vec()
        })
        .collect()
}

#[tokio::test]
async fn test_pending_window_records_validate_finality_once_across_watermarks() {
    let mut spec = kernel_spec(json!([duration_output("sum", "price", "sum", 5)]));
    spec.allowed_lateness_micros = 1;
    let rows = (0..64_u32)
        .map(|row| {
            (
                100 + i64::from(row),
                "a",
                u64::from(row),
                Some(f64::from(row)),
            )
        })
        .collect::<Vec<_>>();
    let input = float64_fast_record(&rows);
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let (expected, _) = reference_output(&operator, &input);
    let (job, store, mut output) = fixture(&operator);
    for offset in 0..32_u32 {
        assert!(
            observe_data(
                &mut operator,
                &job,
                &store,
                &mut output,
                input.slice(2 * offset as usize, 2)
            )
            .await
        );
        assert!(
            close(&mut operator, &job, &store, &mut output, i64::from(offset))
                .await
                .is_empty()
        );
    }
    let mut emitted = close(&mut operator, &job, &store, &mut output, 101).await;
    assert_eq!(emitted.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    emitted.extend(close(&mut operator, &job, &store, &mut output, i64::MAX).await);
    let actual = concat_batches(&expected.schema(), &emitted).unwrap();
    capability_profiles::assert_complete_bits(&actual, &expected);
    assert_eq!(operator.state.ordered.closing_coordinates_checked, 32);
}

#[test]
fn test_pending_window_records_recheck_after_retry_and_drain() {
    let mut spec = kernel_spec(json!([duration_output("sum", "price", "sum", 5)]));
    spec.allowed_lateness_micros = 1;
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, Some(2.0)),
        (3, "a", 3, Some(3.0)),
    ]);
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let batch = Batch::table(
        vec![input.slice(0, 2), input.slice(2, 1)],
        BatchMetadata::default(),
    )
    .unwrap();
    assert!(
        operator
            .try_buffer_ordered(batch.table_payload().unwrap(), None, None)
            .unwrap()
    );
    let buffer = &mut operator.state.owned_mut().unwrap().ordered;
    let close = |buffer: &mut ordered_stream::OrderedStreamBuffer, time| {
        buffer.take_closed(
            EventTime::from_micros(time),
            &operator.compiled,
            1,
            "rolling",
        )
    };
    let identity = buffer.last_identity();
    let prior = close(buffer, 2).unwrap();
    assert_eq!(prior, [input.slice(0, 1)]);
    buffer.restore_front(prior, identity);
    assert_eq!(close(buffer, 2).unwrap(), [input.slice(0, 1)]);
    assert_eq!(buffer.closing_coordinates_checked, 5);
    let retained = buffer.take_all();
    assert_eq!(retained, [input.slice(1, 1), input.slice(2, 1)]);
    let last = float64_fast_record(&[(i64::MAX, "b", 4, None)]);
    buffer.restore_pending(
        vec![retained[0].clone(), last.clone(), retained[1].clone()],
        None,
    );
    let error = close(buffer, 3).unwrap_err();
    assert!(
        error.to_string().contains("finality coordinate overflowed"),
        "{error}"
    );
    assert_eq!(
        buffer.records().cloned().collect::<Vec<_>>(),
        [retained[0].clone(), last, retained[1].clone()]
    );
    buffer.clear();
    assert!(close(buffer, 3).unwrap().is_empty());
}
