use crate::operator::rolling_metrics::{RollingCallback, RollingMetricsStore};

use super::*;

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
