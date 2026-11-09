use super::*;

fn decay_output(window: u64) -> Value {
    json!({
        "kind": "decay", "primitive_version": 1, "input": "price",
        "output": "decayed", "frame": {"kind": "rows", "size": window},
        "min_periods": 1
    })
}

fn decay_spec(window: u64, with_mean: bool) -> RollingSpec {
    let mut outputs = vec![decay_output(window)];
    if with_mean {
        outputs.push(aggregate_output("mean", "price", "mean", window));
    }
    kernel_spec(json!(outputs))
}

fn decay_job() -> crate::StreamJobContext {
    crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    )
}

#[test]
fn p2_decay_and_mean_use_the_typed_kernel() {
    for with_mean in [false, true] {
        let spec = decay_spec(3, with_mean);
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        assert_eq!(
            compiled.kernel_plan.selection(),
            KernelSelection::OrderedPrimitive
        );
        let input = float64_fast_record(&[
            (1, "a", 1, Some(1.0)),
            (2, "a", 2, None),
            (3, "a", 3, Some(4.0)),
            (4, "a", 4, Some(f64::NAN)),
            (5, "a", 5, Some(7.0)),
        ]);
        assert_typed_matches_general(&spec, &kernel_schema(), &input);
    }
}

fn assert_equivalent_columns(actual: &RecordBatch, expected: &RecordBatch) {
    assert_eq!(actual.schema(), expected.schema());
    assert_eq!(actual.num_rows(), expected.num_rows());
    for (index, (actual, expected)) in actual.columns().iter().zip(expected.columns()).enumerate() {
        if index >= kernel_schema().fields().len() && actual.data_type() == &DataType::Float64 {
            assert_equivalent_float_columns(actual, expected);
        } else {
            assert_eq!(actual.to_data(), expected.to_data());
        }
    }
}

fn assert_equivalent_float_columns(actual: &ArrayRef, expected: &ArrayRef) {
    let actual = actual.as_any().downcast_ref::<Float64Array>().unwrap();
    let expected = expected.as_any().downcast_ref::<Float64Array>().unwrap();
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_equivalent_float_value(actual, expected);
    }
}

fn assert_equivalent_float_value(actual: Option<f64>, expected: Option<f64>) {
    match (actual, expected) {
        (None, None) => {}
        (Some(actual), Some(expected)) if expected.is_nan() => assert!(actual.is_nan()),
        (Some(actual), Some(expected)) if expected.is_infinite() || expected == 0.0 => {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        (Some(actual), Some(expected)) => assert!(
            (actual - expected).abs() <= 1e-12 + 1e-12 * actual.abs().max(expected.abs()),
            "actual={actual}, expected={expected}"
        ),
        values => panic!("different null positions: {values:?}"),
    }
}

async fn step(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    input: RecordBatch,
    watermark: i64,
) -> RecordBatch {
    let context = StreamOperatorContext::new(job, "rolling", operator.state.last_input_watermark);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            Batch::table(vec![input], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(watermark), &context, &mut output)
        .await
        .unwrap();
    let records = output
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
        .collect::<Vec<_>>();
    concat_batches(operator.output_ports()[0].schema().unwrap(), &records).unwrap()
}

#[tokio::test]
async fn p2_old_source_snapshots_restore_and_continue_mixed_outputs() {
    let captured: Value =
        serde_json::from_str(include_str!("fixtures/decay-main-8d8be7d9.json")).unwrap();
    assert_eq!(
        captured["runtime_baseline"],
        "8d8be7d907a095dbc0dc2b73dafcfdd842979cdf"
    );
    let prefix = [
        (1, "a", 1, Some(1e20)),
        (2, "a", 2, None),
        (3, "a", 3, Some(1.0)),
    ];
    let suffix = [
        (4, "a", 4, Some(1.0)),
        (5, "a", 5, None),
        (6, "a", 6, Some(-2.0)),
        (7, "a", 7, Some(f64::INFINITY)),
        (8, "a", 8, Some(3.0)),
        (9, "a", 9, None),
        (10, "a", 10, Some(4.0)),
    ];
    for fixture in captured["fixtures"].as_array().unwrap() {
        let spec: RollingSpec = serde_json::from_value(fixture["spec"].clone()).unwrap();
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let mut uninterrupted =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
        let mut restored =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
        let snapshot = crate::OperatorStateSnapshot {
            inline_metadata: serde_json::from_value(fixture["metadata"].clone()).unwrap(),
            segments: fixture["segments"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(id, value)| {
                    (
                        id.clone(),
                        crate::StateSegment::new(hex::decode(value.as_str().unwrap()).unwrap()),
                    )
                })
                .collect(),
        };
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        let job = crate::StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        step(&mut uninterrupted, &job, float64_fast_record(&prefix), 3).await;
        for chunk in suffix.chunks(2) {
            let end = chunk.last().unwrap().0;
            let actual = step(&mut restored, &job, float64_fast_record(chunk), end).await;
            let continuous = step(&mut uninterrupted, &job, float64_fast_record(chunk), end).await;
            assert_equivalent_columns(&actual, &continuous);
            let complete = prefix
                .iter()
                .chain(suffix.iter().filter(|row| row.0 <= end))
                .copied()
                .collect::<Vec<_>>();
            let input = float64_fast_record(&complete);
            let rows = (0..input.num_rows())
                .map(|row| read_buffered_row(&input, row, &compiled, "rolling").unwrap())
                .collect::<Vec<_>>();
            let general =
                compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                    .unwrap();
            let expected = build_output_record(
                &rows,
                general.columns,
                &Arc::new(output_schema(&kernel_schema(), &compiled.outputs)),
                "rolling",
            )
            .unwrap();
            assert_equivalent_columns(
                &actual,
                &expected.slice(expected.num_rows() - chunk.len(), chunk.len()),
            );
        }
        let checkpoint = restored.checkpoint(Epoch::new(2).unwrap()).unwrap();
        assert_eq!(
            checkpoint.inline_metadata["kernel_fingerprint"],
            fixture["metadata"]["kernel_fingerprint"]
        );
    }
}

#[tokio::test]
async fn p2_cache_refusal_and_recovery_preserve_mean_and_logical_budget() {
    use datafusion::execution::memory_pool::MemoryConsumer;
    let spec = decay_spec(2, true);
    let mut cached =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
    let mut reference = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let budget = StateBudget::new(1000, 1 << 20).unwrap();
    cached.set_state_budget(budget).unwrap();
    reference.set_state_budget(budget).unwrap();
    Arc::get_mut(&mut reference.compiled)
        .expect("reference compiled spec is unshared")
        .kernel_plan
        .set_decay_pool_for_test(0);
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let prefix = [(1, "a", 1, Some(1e155)), (2, "a", 2, Some(1e155))];
    let left = step(&mut cached, &job, float64_fast_record(&prefix), 2).await;
    let right = step(&mut reference, &job, float64_fast_record(&prefix), 2).await;
    assert_equivalent_columns(&left, &right);
    assert_eq!(left.column(7).to_data(), right.column(7).to_data());
    assert!(reference.state.typed_kernel_state.is_some());
    assert_eq!(reference.state.charge, cached.state.charge);
    let pool = cached.compiled.kernel_plan.decay_pool_for_test();
    let retained_owner = cached.state.typed_kernel_state.clone();
    let before = pool.reserved();
    assert!(before > 0);
    let pressure = MemoryConsumer::new("decay quota test").register(&pool);
    pressure.try_grow((64 << 20) - before).unwrap();
    let next = [
        (3, "a", 3, Some(1e155)),
        (4, "a", 4, Some(-1e155)),
        (5, "a", 5, Some(3.0)),
    ];
    let left = step(&mut cached, &job, float64_fast_record(&next), 5).await;
    let right = step(&mut reference, &job, float64_fast_record(&next), 5).await;
    assert_equivalent_columns(&left, &right);
    assert_eq!(left.column(7).to_data(), right.column(7).to_data());
    assert_eq!(reference.state.charge, cached.state.charge);
    drop(pressure);
    assert_eq!(
        pool.reserved(),
        before,
        "the old owner retains its cache credit"
    );
    drop(retained_owner);
    assert_eq!(pool.reserved(), 0);
    Arc::get_mut(&mut reference.compiled)
        .expect("reference compiled spec is unshared")
        .kernel_plan
        .set_decay_pool_for_test(64 << 20);
    let final_rows = [
        (6, "a", 6, None),
        (7, "a", 7, Some(4.0)),
        (8, "a", 8, Some(5.0)),
    ];
    let left = step(&mut cached, &job, float64_fast_record(&final_rows), 8).await;
    let right = step(&mut reference, &job, float64_fast_record(&final_rows), 8).await;
    assert_equivalent_columns(&left, &right);
    assert_eq!(left.column(7).to_data(), right.column(7).to_data());
    assert!(pool.reserved() > 0);
    assert!(
        reference
            .compiled
            .kernel_plan
            .decay_pool_for_test()
            .reserved()
            > 0
    );
}

fn old_snapshot() -> (RollingSpec, crate::OperatorStateSnapshot) {
    let captured: Value =
        serde_json::from_str(include_str!("fixtures/decay-main-8d8be7d9.json")).unwrap();
    let fixture = &captured["fixtures"][0];
    let snapshot = crate::OperatorStateSnapshot {
        inline_metadata: serde_json::from_value(fixture["metadata"].clone()).unwrap(),
        segments: fixture["segments"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, value)| {
                (
                    id.clone(),
                    crate::StateSegment::new(hex::decode(value.as_str().unwrap()).unwrap()),
                )
            })
            .collect(),
    };
    (
        serde_json::from_value(fixture["spec"].clone()).unwrap(),
        snapshot,
    )
}

#[tokio::test]
async fn p2_extra_wrong_checkpoint_identity_profile_and_budget_preserve_installed_state() {
    let (spec, snapshot) = old_snapshot();
    let mut operator =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    step(
        &mut operator,
        &job,
        float64_fast_record(&[(1, "a", 1, Some(2.0))]),
        1,
    )
    .await;
    let before = format!("{:?}", operator.state.typed_kernel_state);
    let charge = operator.state.charge;
    for (field, value) in [
        ("kernel_fingerprint", "foreign"),
        ("numerical_profile", "stable_v2"),
        ("state_schema_fingerprint", "foreign"),
    ] {
        let mut bad = snapshot.clone();
        bad.inline_metadata.insert(field.into(), json!(value));
        assert!(StreamOperator::restore(&mut operator, &bad).is_err());
        assert_eq!(format!("{:?}", operator.state.typed_kernel_state), before);
        assert_eq!(operator.state.charge, charge);
        assert_eq!(
            operator.state.last_input_watermark,
            Some(EventTime::from_micros(1))
        );
    }
    let mut limited = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    limited
        .set_state_budget(StateBudget::new(1, 1).unwrap())
        .unwrap();
    let error = StreamOperator::restore(&mut limited, &snapshot).unwrap_err();
    assert!(error.to_string().contains("state budget"), "{error}");
    assert!(limited.state.histories.by_entity.is_empty());
    assert!(limited.state.typed_kernel_state.is_none());
    assert_eq!(
        limited
            .compiled
            .kernel_plan
            .decay_pool_for_test()
            .reserved(),
        0
    );
}

#[tokio::test]
async fn p2_extra_rejected_or_cancelled_repair_releases_private_credit_and_retries() {
    struct Reject(bool);
    #[async_trait]
    impl StreamCollector for Reject {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            if self.0 {
                Err(CalcFlowError::Cancelled { run_id: "7".into() })
            } else {
                Err(internal_error("expected decay output refusal"))
            }
        }
    }
    for cancelled in [false, true] {
        let mut operator =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), decay_spec(2, true))
                .unwrap();
        let job = crate::StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        step(
            &mut operator,
            &job,
            float64_fast_record(&[(1, "a", 1, Some(1e20)), (2, "a", 2, Some(1.0))]),
            2,
        )
        .await;
        let prior = format!("{:?}", operator.state.typed_kernel_state);
        let pool = operator.compiled.kernel_plan.decay_pool_for_test();
        let reserved = pool.reserved();
        let sequence = operator.state.next_output_sequence;
        let input = float64_fast_record(&[(3, "a", 3, Some(1.0))]);
        let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(2)));
        let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "input",
                Batch::table(vec![input.clone()], BatchMetadata::default()).unwrap(),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        let error = operator
            .on_watermark(EventTime::from_micros(3), &context, &mut Reject(cancelled))
            .await
            .unwrap_err();
        if cancelled {
            assert!(matches!(error, CalcFlowError::Cancelled { .. }));
        } else {
            assert!(error.to_string().contains("expected decay output refusal"));
        }
        assert_eq!(pool.reserved(), reserved);
        assert_eq!(format!("{:?}", operator.state.typed_kernel_state), prior);
        assert_eq!(operator.state.next_output_sequence, sequence);
        assert_eq!(
            operator.state.last_input_watermark,
            Some(EventTime::from_micros(2))
        );
        let retried = step(&mut operator, &job, input, 3).await;
        let decayed = retried
            .column(6)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(decayed.value(0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(operator.state.next_output_sequence, sequence + 1);
    }
}

async fn buffered_step(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    input: RecordBatch,
    watermark: i64,
    store: &crate::operator::rolling_metrics::RollingMetricsStore,
) -> RecordBatch {
    use crate::operator::rolling_metrics::RollingCallback;
    let context = StreamOperatorContext::new(job, "rolling", operator.state.last_input_watermark);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            Batch::table(vec![input], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert!(!operator.state.buffer.is_empty());
    assert!(operator.state.ordered.is_empty());
    let callback = store.begin(RollingCallback::Watermark, crate::CancellationToken::new());
    let context = context.with_rolling_metrics(callback.recorder());
    let result = operator
        .on_watermark(EventTime::from_micros(watermark), &context, &mut output)
        .await;
    callback.complete(&result);
    result.unwrap();
    let records = output
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
        .collect::<Vec<_>>();
    concat_batches(operator.output_ports()[0].schema().unwrap(), &records).unwrap()
}

#[tokio::test]
async fn p2_buffered_profiles_mix_cache_reference_and_recover_without_rephasing_mean() {
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        assert_mixed_buffered_decay_profile(profile).await;
    }
}

async fn assert_mixed_buffered_decay_profile(profile: RollingNumericalProfile) {
    use crate::operator::rolling_metrics::RollingMetricsStore;
    use datafusion::execution::memory_pool::MemoryConsumer;
    let mut spec = decay_spec(2, true);
    spec.numerical_profile = profile;
    let mut cached =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
    let mut reference = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    Arc::get_mut(&mut reference.compiled)
        .expect("reference compiled spec is unshared")
        .kernel_plan
        .set_decay_pool_for_test(0);
    let budget = StateBudget::new(1000, 1 << 20).unwrap();
    cached.set_state_budget(budget).unwrap();
    reference.set_state_budget(budget).unwrap();
    let fingerprint = cached.compiled.kernel_plan.fingerprint().to_owned();
    let job = decay_job();
    let store = RollingMetricsStore::default();
    let reference_store = RollingMetricsStore::default();
    let prefix = (1..=63_u32)
        .flat_map(|time| {
            [
                (i64::from(time), "a", u64::from(time), Some(1e155)),
                (i64::from(time), "b", u64::from(time), Some(2e155)),
            ]
        })
        .rev()
        .collect::<Vec<_>>();
    let left = buffered_step(&mut cached, &job, float64_fast_record(&prefix), 63, &store).await;
    let right = buffered_step(
        &mut reference,
        &job,
        float64_fast_record(&prefix),
        63,
        &reference_store,
    )
    .await;
    assert_equivalent_columns(&left, &right);
    assert_eq!(left.column(7).to_data(), right.column(7).to_data());
    let pool = cached.compiled.kernel_plan.decay_pool_for_test();
    let old_owners = cached.state.typed_kernel_state.clone();
    let before = pool.reserved();
    assert!(before > 0);
    let pressure = MemoryConsumer::new("mixed decay quota").register(&pool);
    pressure
        .try_grow((64 << 20) - before - (before - 1))
        .unwrap();
    let next = [
        (65, "b", 65, Some(4.0)),
        (64, "b", 64, Some(-2e155)),
        (65, "a", 65, Some(3.0)),
        (64, "a", 64, Some(-1e155)),
    ];
    let left = buffered_step(&mut cached, &job, float64_fast_record(&next), 65, &store).await;
    let right = buffered_step(
        &mut reference,
        &job,
        float64_fast_record(&next),
        65,
        &reference_store,
    )
    .await;
    assert_equivalent_columns(&left, &right);
    assert_eq!(left.column(7).to_data(), right.column(7).to_data());
    assert_eq!(cached.state.charge, reference.state.charge);
    drop(pressure);
    assert_eq!(
        pool.reserved(),
        before + before / 2,
        "two retained owners and one new cached entity"
    );
    drop(old_owners);
    assert_eq!(
        pool.reserved(),
        before / 2,
        "one entity cached while the other uses reference"
    );
    let next = [
        (67, "b", 67, Some(8.0)),
        (66, "b", 66, Some(6.0)),
        (67, "a", 67, Some(7.0)),
        (66, "a", 66, Some(5.0)),
    ];
    let left = buffered_step(&mut cached, &job, float64_fast_record(&next), 67, &store).await;
    let right = buffered_step(
        &mut reference,
        &job,
        float64_fast_record(&next),
        67,
        &reference_store,
    )
    .await;
    assert_equivalent_columns(&left, &right);
    assert_eq!(left.column(7).to_data(), right.column(7).to_data());
    assert_eq!(cached.state.charge, reference.state.charge);
    assert_eq!(pool.reserved(), before, "both entity caches admitted again");
    assert_eq!(cached.compiled.kernel_plan.fingerprint(), fingerprint);
    assert_eq!(store.snapshot().watermark.numeric_rows, 134);
}

#[tokio::test]
async fn p2_work_counts_actual_copy_reference_seed_and_certified_attempts() {
    use crate::operator::rolling_metrics::RollingMetricsStore;
    use datafusion::execution::memory_pool::MemoryConsumer;
    let mut operator =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), decay_spec(3, false)).unwrap();
    let job = decay_job();
    let store = RollingMetricsStore::default();
    for (rows, watermark) in [
        (vec![(2, "a", 2, Some(2.0)), (1, "a", 1, Some(1.0))], 2),
        (vec![(4, "a", 4, Some(4.0)), (3, "a", 3, Some(3.0))], 4),
    ] {
        buffered_step(
            &mut operator,
            &job,
            float64_fast_record(&rows),
            watermark,
            &store,
        )
        .await;
    }
    let value = serde_json::to_value(store.snapshot()).unwrap();
    let work = &value["watermark"]["_decay_work"];
    assert_eq!(work["transitions"], 4);
    assert_eq!(work["certified_rows"], 3);
    assert_eq!(work["scheduled_reseed_rows"], 1);
    assert_eq!(work["scheduled_reseed_samples"], 3);
    assert_eq!(work["cache_copied_samples"], 2);
    assert_eq!(work["reference_scan_samples"], 0);
    let pool = operator.compiled.kernel_plan.decay_pool_for_test();
    let pressure = MemoryConsumer::new("count reference work").register(&pool);
    pressure.try_grow((64 << 20) - pool.reserved()).unwrap();
    buffered_step(
        &mut operator,
        &job,
        float64_fast_record(&[(6, "a", 6, Some(6.0)), (5, "a", 5, Some(5.0))]),
        6,
        &store,
    )
    .await;
    let value = serde_json::to_value(store.snapshot()).unwrap();
    let work = &value["watermark"]["_decay_work"];
    assert_eq!(work["transitions"], 6);
    assert_eq!(work["quota_reference_rows"], 2);
    assert_eq!(work["reference_history_samples"], 3);
    assert_eq!(work["reference_scan_samples"], 12);
    drop(pressure);
    buffered_step(
        &mut operator,
        &job,
        float64_fast_record(&[(8, "a", 8, Some(8.0)), (7, "a", 7, Some(7.0))]),
        8,
        &store,
    )
    .await;
    let value = serde_json::to_value(store.snapshot()).unwrap();
    let work = &value["watermark"]["_decay_work"];
    assert_eq!(work["transitions"], 8);
    assert_eq!(work["initial_seed_samples"], 9);
    assert_eq!(work["reference_history_samples"], 6);
    assert_eq!(work["cache_copied_samples"], 2);
    assert_rejected_work_is_counted_and_replay_commits_once(&mut operator, &job, &store, &pool)
        .await;
}

async fn assert_rejected_work_is_counted_and_replay_commits_once(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    store: &crate::operator::rolling_metrics::RollingMetricsStore,
    pool: &Arc<dyn datafusion::execution::memory_pool::MemoryPool>,
) {
    use crate::operator::rolling_metrics::RollingCallback;
    struct Reject;
    #[async_trait]
    impl StreamCollector for Reject {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            Err(internal_error("expected work-count output refusal"))
        }
    }
    let before_state = format!("{:?}", operator.state.typed_kernel_state);
    let before_credit = pool.reserved();
    let before_sequence = operator.state.next_output_sequence;
    let context = StreamOperatorContext::new(job, "rolling", Some(EventTime::from_micros(8)));
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let retry_input = float64_fast_record(&[
        (13, "a", 13, Some(1.0)),
        (12, "a", 12, Some(1.0)),
        (11, "a", 11, Some(1.0)),
        (10, "a", 10, Some(1e20)),
        (9, "a", 9, Some(1.0)),
    ]);
    operator
        .process_data(
            "input",
            Batch::table(vec![retry_input.clone()], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let callback = store.begin(RollingCallback::Watermark, crate::CancellationToken::new());
    let observed = StreamOperatorContext::new(job, "rolling", Some(EventTime::from_micros(8)))
        .with_rolling_metrics(callback.recorder());
    let result = operator
        .on_watermark(EventTime::from_micros(13), &observed, &mut Reject)
        .await;
    callback.complete(&result);
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("expected work-count output refusal")
    );
    assert_eq!(
        format!("{:?}", operator.state.typed_kernel_state),
        before_state
    );
    assert_eq!(pool.reserved(), before_credit);
    assert_eq!(operator.state.next_output_sequence, before_sequence);
    let failed = serde_json::to_value(store.snapshot()).unwrap();
    assert_eq!(failed["watermark"]["failed"], 1);
    assert_eq!(failed["watermark"]["_decay_work"]["transitions"], 13);
    assert_eq!(failed["watermark"]["_decay_work"]["guard_fallback_rows"], 1);
    assert_eq!(
        failed["watermark"]["_decay_work"]["guard_reseed_samples"],
        3
    );
    operator
        .process_data(
            "input",
            Batch::table(vec![retry_input], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let callback = store.begin(RollingCallback::Watermark, crate::CancellationToken::new());
    let observed = context.with_rolling_metrics(callback.recorder());
    let result = operator
        .on_watermark(EventTime::from_micros(13), &observed, &mut output)
        .await;
    callback.complete(&result);
    result.unwrap();
    let retried = serde_json::to_value(store.snapshot()).unwrap();
    assert_eq!(retried["watermark"]["_decay_work"]["transitions"], 18);
    assert_eq!(
        retried["watermark"]["_decay_work"]["cache_copied_samples"],
        8
    );
    assert_eq!(
        retried["watermark"]["_decay_work"]["guard_fallback_rows"],
        2
    );
    assert_eq!(operator.state.next_output_sequence, before_sequence + 1);
    assert!(!store.snapshot().overflowed);
}
