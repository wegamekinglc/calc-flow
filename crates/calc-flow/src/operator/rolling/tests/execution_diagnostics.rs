use crate::operator::rolling_metrics::{RollingCallback, RollingMetricsStore};

use super::*;

#[tokio::test]
async fn test_execution_diagnostics_observe_incremental_history_and_serial_calls() {
    let input = float64_fast_record(&[(1, "a", 1, Some(1.0)), (2, "a", 2, Some(3.0))]);
    let spec = kernel_spec(json!([duration_output("mean", "price", "mean", 100)]));
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let expected = reference_output(&operator, &input);
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let store = RollingMetricsStore::default();
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    for (index, time) in [(0, 1), (1, 2)] {
        observe(
            &mut operator,
            &job,
            &store,
            &mut output,
            input.slice(index, 1),
            time,
        )
        .await;
        assert_cached_rolling_charge(&operator);
    }
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
    assert_eq!(
        concat_batches(&expected.schema(), &records).unwrap(),
        expected
    );
    assert_cached_rolling_charge(&operator);
    assert_eq!(store.snapshot().watermark.numeric_rows, 2);
    let snapshot = serde_json::to_value(store.snapshot()).unwrap();
    let watermark = &snapshot["watermark"];
    assert_eq!(watermark["history_charge_scalar_rows"].as_u64(), Some(2));
    assert_eq!(
        watermark["history_charge_columnar_chunks"].as_u64(),
        Some(0)
    );
    assert_eq!(watermark["history_append_rows_prepared"].as_u64(), Some(2));
    assert_eq!(watermark["history_evict_rows_prepared"].as_u64(), Some(0));
    assert_eq!(watermark["numeric_serial_calls"].as_u64(), Some(2));
    assert_eq!(watermark["numeric_general_calls"].as_u64(), Some(0));
    assert_eq!(watermark["numeric_parallel_calls"].as_u64(), Some(0));
    assert_eq!(
        watermark["parallel_fallback_no_work_client"].as_u64(),
        Some(2)
    );
}

fn reference_output(operator: &RollingOperator, input: &RecordBatch) -> RecordBatch {
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
    build_output_record(
        &rows,
        computed.columns,
        operator.output_ports()[0].schema().unwrap(),
        "rolling",
    )
    .unwrap()
}

pub(super) async fn observe(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    store: &RollingMetricsStore,
    output: &mut crate::EdgeCollector,
    input: RecordBatch,
    time: i64,
) {
    observed_data(operator, job, store, output, input)
        .await
        .unwrap();
    observed_watermark(operator, job, store, output, time)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_execution_diagnostics_distinguish_general_replacement_history() {
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, Some(3.0)),
        (3, "a", 3, Some(5.0)),
    ]);
    let spec = kernel_spec(json!([lag_price(1)]));
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let expected = reference_output(&operator, &input);
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let store = RollingMetricsStore::default();
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    observe(
        &mut operator,
        &job,
        &store,
        &mut output,
        input.slice(0, 2),
        2,
    )
    .await;
    observe(
        &mut operator,
        &job,
        &store,
        &mut output,
        input.slice(2, 1),
        3,
    )
    .await;
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
    assert_eq!(
        concat_batches(&expected.schema(), &records).unwrap(),
        expected
    );
    assert_general_charge(&operator);
    let work = store.snapshot().watermark;
    assert_eq!(work.numeric_general_calls, 2);
    assert_eq!(work.numeric_serial_calls, 0);
    assert_eq!(work.numeric_parallel_calls, 0);
    assert_eq!(work.history_charge_scalar_rows, 3);
    assert_eq!(work.history_append_rows_prepared, 2);
    assert_eq!(work.history_evict_rows_prepared, 1);
}

fn assert_general_charge(operator: &RollingOperator) {
    assert_eq!(
        operator.state.charge,
        budget::full_state_charge(
            &operator.state.buffer,
            &operator.state.ordered,
            &operator.state.histories,
            &operator.compiled,
            "rolling",
        )
        .unwrap()
    );
    assert!(operator.state.histories.by_entity.values().all(|state| {
        state.charge_cache.is_none() && state.rows.len() == 1 && state.columnar.rows == 0
    }));
    assert_eq!(
        operator.state.charge.rows,
        2 * u64::try_from(operator.state.histories.by_entity.len()).unwrap()
    );
}

#[tokio::test]
async fn test_execution_diagnostics_preserve_general_prepared_prefix_on_later_entity_error() {
    let input = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "b", 2, Some(3.0)),
        (3, "a", 3, Some(5.0)),
        (4, "b", 4, Some(7.0)),
    ]);
    let spec = kernel_spec(json!([lag_price(1)]));
    let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    observe(
        &mut operator,
        &job,
        &RollingMetricsStore::default(),
        &mut output,
        input.slice(0, 2),
        2,
    )
    .await;
    let next = Batch::table(vec![input.slice(2, 2)], BatchMetadata::default()).unwrap();
    let rows =
        read_buffered_rows(next.table_payload().unwrap(), &operator.compiled, "rolling").unwrap();
    operator
        .state
        .owned_mut()
        .unwrap()
        .histories
        .by_entity
        .get_mut(&rows[1].identity.entity)
        .unwrap()
        .transition_count = u64::MAX;
    assert_general_charge(&operator);
    let prior_ipc = retained_state_ipc(&operator);
    let prior_charge = operator.state.charge;
    let store = RollingMetricsStore::default();
    let callback = store.begin(RollingCallback::Watermark, crate::CancellationToken::new());
    let recorder = callback.recorder();
    let result = compute_output_columns_observed(
        &rows,
        &operator.state.histories,
        &operator.compiled,
        "rolling",
        Some(&recorder),
    )
    .map(|_| ());
    callback.complete(&result);
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("transition count overflowed")
    );
    assert_eq!(retained_state_ipc(&operator), prior_ipc);
    assert_eq!(operator.state.charge, prior_charge);
    assert_general_charge(&operator);
    let work = store.snapshot().watermark;
    assert_eq!(work.failed, 1);
    assert_eq!(work.numeric_rows, 1);
    assert_eq!(work.numeric_general_calls, 1);
    assert_eq!(work.history_append_rows_prepared, 1);
    assert_eq!(work.history_evict_rows_prepared, 1);
}

#[tokio::test]
async fn test_execution_diagnostics_distinguish_small_batch_from_actual_parallel() {
    for rows in [1_000, 64_000] {
        let input = parallel_rollback_record(0, 1_280 + rows);
        let spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 20)]));
        let mut operator = RollingOperator::new("rolling", input.schema(), spec).unwrap();
        let expected = reference_output(&operator, &input);
        let job = crate::StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        let store = RollingMetricsStore::default();
        let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
        observe(
            &mut operator,
            &job,
            &RollingMetricsStore::default(),
            &mut output,
            input.slice(0, 1_280),
            1_279,
        )
        .await;
        assert_cached_rolling_charge(&operator);
        observed_data(
            &mut operator,
            &job,
            &store,
            &mut output,
            input.slice(1_280, rows),
        )
        .await
        .unwrap();
        let launches = observe_managed_watermark(
            &mut operator,
            &job,
            &store,
            &mut output,
            i64::try_from(1_279 + rows).unwrap(),
        )
        .await;
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
        assert_eq!(
            concat_batches(&expected.schema(), &records).unwrap(),
            expected
        );
        assert_cached_rolling_charge(&operator);
        let actual = store.snapshot().watermark;
        let parallel = rows == 64_000;
        assert_eq!(actual.numeric_rows, u64::try_from(rows).unwrap());
        assert_eq!(
            actual.numeric_parallel_rows,
            if parallel {
                u64::try_from(rows).unwrap()
            } else {
                0
            }
        );
        assert_eq!(actual.numeric_lanes_started, if parallel { 2 } else { 0 });
        assert_eq!(actual.numeric_parallel_calls, u64::from(parallel));
        assert_eq!(actual.numeric_serial_calls, u64::from(!parallel));
        assert_eq!(actual.numeric_general_calls, 0);
        assert_eq!(actual.parallel_fallback_small_batch, u64::from(!parallel));
        assert_eq!(actual.parallel_fallback_unsupported_plan, 0);
        assert_eq!(launches, if parallel { 2 } else { 0 });
    }
}

async fn observe_managed_watermark(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    store: &RollingMetricsStore,
    output: &mut crate::EdgeCollector,
    time: i64,
) -> u64 {
    use crate::runtime::streaming::entity_work::JobEntityWorkOwner;
    use std::sync::atomic::{AtomicU64, Ordering};

    let launches = Arc::new(AtomicU64::new(0));
    let owner = JobEntityWorkOwner::new(7, Arc::clone(&launches));
    owner.set_worker_limit_for_test(2);
    let task = owner.test_client(3, "operator:rolling".into());
    let scope = task.callback_scope();
    let callback = store.begin(RollingCallback::Watermark, crate::CancellationToken::new());
    let context = StreamOperatorContext::new(job, "rolling", operator.state.last_input_watermark)
        .with_entity_work(task.context_client().unwrap())
        .with_rolling_metrics(callback.recorder());
    let result = operator
        .on_watermark(EventTime::from_micros(time), &context, output)
        .await;
    callback.complete(&result);
    result.unwrap();
    scope.settle_abandoned().await;
    drop(context);
    drop(scope);
    owner.close_admission();
    assert!(owner.drain().await.is_empty());
    launches.load(Ordering::SeqCst)
}

async fn observed_data(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    store: &RollingMetricsStore,
    output: &mut dyn StreamCollector,
    input: RecordBatch,
) -> Result<()> {
    let callback = store.begin(RollingCallback::Data, crate::CancellationToken::new());
    let context = StreamOperatorContext::new(job, "rolling", operator.state.last_input_watermark)
        .with_rolling_metrics(callback.recorder());
    let result = operator
        .process_data(
            "input",
            Batch::table(vec![input], BatchMetadata::default()).unwrap(),
            &context,
            output,
        )
        .await;
    callback.complete(&result);
    result
}

async fn observed_watermark(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    store: &RollingMetricsStore,
    output: &mut dyn StreamCollector,
    time: i64,
) -> Result<()> {
    let callback = store.begin(RollingCallback::Watermark, crate::CancellationToken::new());
    let context = StreamOperatorContext::new(job, "rolling", operator.state.last_input_watermark)
        .with_rolling_metrics(callback.recorder());
    let result = operator
        .on_watermark(EventTime::from_micros(time), &context, output)
        .await;
    callback.complete(&result);
    result
}
