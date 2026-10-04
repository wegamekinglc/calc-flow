use super::*;

const AVG_QUERY: &str = "SELECT key, COUNT(value) AS valid, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key";

fn avg_inputs(value: &ScalarValue, native: bool) -> Vec<Batch> {
    let first = decimal_batch(value, &[], &[], native);
    let nullable = decimal_batch(value, &[0, 1], &[None, None], native);
    let valid = decimal_batch(value, &[0, 1, 2], &[Some(1), None, Some(2)], native);
    let multiple = Batch::table(
        vec![
            decimal_batch(value, &[0, 2, 3], &[Some(2), Some(-1), None], native)
                .table_payload()
                .unwrap()
                .batches()[0]
                .clone(),
            decimal_batch(value, &[0, 1, 4], &[Some(-2), None, None], native)
                .table_payload()
                .unwrap()
                .batches()[0]
                .clone(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    vec![
        first,
        nullable.clone(),
        nullable.clone(),
        valid,
        multiple,
        nullable,
    ]
}

async fn avg_prefixes(value: &ScalarValue, native: bool, grouped: bool) {
    let query = if grouped {
        AVG_QUERY
    } else {
        "SELECT COUNT(value) AS valid, SUM(value) AS total, AVG(value) AS mean FROM events"
    };
    let incoming = avg_inputs(value, native);
    let mut operator = assert_prefix_oracle(query, &incoming, true).await;
    assert_eq!(operator.incremental_work, (15, 1));
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = StreamJobContext::new(1, "prefix", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let prefix = incoming
        .iter()
        .flat_map(|batch| batch.table_payload().unwrap().batches().to_vec())
        .collect::<Vec<_>>();
    assert_restored_primitive_prefix(
        &mut operator,
        &runtime,
        query,
        &prefix,
        &context,
        &mut collector,
    )
    .await;
}

#[tokio::test]
async fn test_sql_incremental_decimal_avg_prefixes_preserve_types_nulls_and_work() {
    for scale in [0, 2] {
        for value in decimal_cases(scale) {
            avg_prefixes(&value, false, false).await;
            avg_prefixes(&value, false, true).await;
            avg_prefixes(&value, true, true).await;
        }
    }
}

async fn avg_output(
    query: &str,
    operator: &mut SqlOperator,
    incoming: &Batch,
    prefix: &[Batch],
    context: &StreamOperatorContext<'_>,
) {
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", incoming.clone(), context, &mut collector)
        .await
        .unwrap();
    let records = prefix
        .iter()
        .flat_map(|batch| batch.table_payload().unwrap().batches().to_vec())
        .collect::<Vec<_>>();
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([(
                "events".into(),
                Batch::table(records, BatchMetadata::default()).unwrap(),
            )]),
            None,
        )
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    let actual = output[0].as_data().unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(rows(actual), rows(&expected));
    assert!(operator.incremental.is_some());
}

fn unchanged_checkpoint(before: &OperatorStateSnapshot, operator: &mut SqlOperator) {
    let after = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(before.inline_metadata, after.inline_metadata);
    assert_eq!(
        before.segments.keys().collect::<Vec<_>>(),
        after.segments.keys().collect::<Vec<_>>()
    );
    for (name, segment) in &before.segments {
        assert!(Arc::ptr_eq(
            &segment.bytes_arc(),
            &after.segments[name].bytes_arc()
        ));
    }
}

async fn avg_transaction(value: &ScalarValue, native: bool, reject: bool) {
    let initial = decimal_batch(value, &[0, 1], &[None, Some(4)], native);
    let mut operator = assert_prefix_oracle(AVG_QUERY, std::slice::from_ref(&initial), true).await;
    let job = StreamJobContext::new(1, "prefix", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "totals", None);
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    let keys = (0..1025).collect::<Vec<_>>();
    let incoming = decimal_batch(value, &keys, &vec![Some(2); keys.len()], native);
    for _ in 0..2 {
        fail_decimal_output(&mut operator, &incoming, &context, reject).await;
        unchanged_checkpoint(&before, &mut operator);
        let pressure = operator
            .stream_state
            .runtime()
            .unwrap()
            .incremental_reservation("probe");
        pressure.try_grow((1 << 30) - (1 << 20)).unwrap();
    }
    avg_output(
        AVG_QUERY,
        &mut operator,
        &incoming,
        &[initial.clone(), incoming.clone()],
        &context,
    )
    .await;
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = operator.clone();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let next = decimal_batch(value, &[0, 1, 1026], &[Some(-1), None, None], native);
    avg_output(
        AVG_QUERY,
        &mut restored,
        &next,
        &[initial, incoming, next.clone()],
        &context,
    )
    .await;
    StreamOperator::reset(&mut restored).unwrap();
    avg_output(
        AVG_QUERY,
        &mut restored,
        &next,
        std::slice::from_ref(&next),
        &context,
    )
    .await;
    let mut cloned = operator.clone();
    avg_output(
        AVG_QUERY,
        &mut cloned,
        &next,
        std::slice::from_ref(&next),
        &context,
    )
    .await;
}

#[tokio::test]
async fn test_sql_incremental_decimal_avg_failed_emission_restores_and_retries_once() {
    for value in decimal_cases(2) {
        for native in [false, true] {
            for reject in [false, true] {
                avg_transaction(&value, native, reject).await;
            }
        }
    }
}

async fn avg_cancel(value: &ScalarValue, native: bool) {
    let keys = (0..1025).collect::<Vec<_>>();
    let input = decimal_batch(value, &keys, &vec![Some(1); keys.len()], native);
    let mut operator = assert_prefix_oracle(AVG_QUERY, std::slice::from_ref(&input), true).await;
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    let finalized = operator
        .incremental
        .as_ref()
        .unwrap()
        .finalized_groups
        .clone();
    finalized.store(0, std::sync::atomic::Ordering::SeqCst);
    let job = StreamJobContext::new(1, "prefix", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "totals", None);
    let observed = finalized.clone();
    let cancellation = job.cancellation().clone();
    let cancel = tokio::spawn(async move {
        while observed.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        cancellation.cancel();
    });
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let result = operator
        .process_data("events", input.clone(), &context, &mut collector)
        .await;
    cancel.await.unwrap();
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert!(finalized.load(std::sync::atomic::Ordering::SeqCst) <= 512);
    assert!(collector.drain("output").is_empty());
    unchanged_checkpoint(&before, &mut operator);
    let retry_job =
        StreamJobContext::new(1, "prefix", JsonMap::new(), None, CancellationToken::new());
    let retry_context = StreamOperatorContext::new(&retry_job, "totals", None);
    avg_output(
        AVG_QUERY,
        &mut operator,
        &input,
        &[input.clone(), input.clone()],
        &retry_context,
    )
    .await;
}

#[tokio::test]
async fn test_sql_incremental_decimal_avg_cancellation_discards_finalized_candidates() {
    for value in decimal_cases(2) {
        for native in [false, true] {
            avg_cancel(&value, native).await;
        }
    }
}

async fn avg_overflow(value: &ScalarValue, native: bool, grouped: bool) {
    let query = if grouped {
        AVG_QUERY
    } else {
        "SELECT AVG(value) AS mean FROM events"
    };
    let initial = decimal_batch(value, &[0, 1], &[Some(0), None], native);
    let mut operator = assert_prefix_oracle(query, std::slice::from_ref(&initial), true).await;
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    let input = repeated_decimal(value, native, 1);
    let records = vec![
        initial.table_payload().unwrap().batches()[0].clone(),
        input.table_payload().unwrap().batches()[0].clone(),
    ];
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    assert!(
        runtime
            .sql(
                query,
                &BTreeMap::from([(
                    "events".into(),
                    Batch::table(records, BatchMetadata::default()).unwrap()
                )]),
                None
            )
            .await
            .is_err()
    );
    let job = StreamJobContext::new(1, "prefix", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    assert!(
        operator
            .process_data("events", input, &context, &mut collector)
            .await
            .is_err()
    );
    assert!(collector.drain("output").is_empty());
    unchanged_checkpoint(&before, &mut operator);
    let retry = decimal_batch(value, &[0, 1], &[Some(1), None], native);
    avg_output(
        query,
        &mut operator,
        &retry,
        &[initial, retry.clone()],
        &context,
    )
    .await;
}

#[tokio::test]
async fn test_sql_incremental_decimal_avg_precision_errors_preserve_checkpoint() {
    for value in maximum_decimals() {
        for native in [false, true] {
            avg_overflow(&value, native, false).await;
            avg_overflow(&value, native, true).await;
        }
    }
}

#[tokio::test]
async fn test_sql_incremental_decimal_avg_high_positive_scales_match_prefix_oracle() {
    let values = [
        ScalarValue::Decimal32(Some(1), 9, 9),
        ScalarValue::Decimal64(Some(1), 18, 18),
        ScalarValue::Decimal128(Some(1), 38, 38),
        ScalarValue::Decimal256(Some(i256::from_i128(1)), 76, 76),
    ];
    for value in values {
        avg_prefixes(&value, false, false).await;
        avg_prefixes(&value, false, true).await;
        avg_prefixes(&value, true, true).await;
    }
}

async fn avg_negative_scale(value: &ScalarValue, native: bool, grouped: bool) {
    use futures::FutureExt;
    let query = if grouped {
        AVG_QUERY
    } else {
        "SELECT AVG(value) AS mean FROM events"
    };
    let initial = decimal_batch(value, &[0], &[None], native);
    let mut operator = assert_prefix_oracle(query, std::slice::from_ref(&initial), false).await;
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    let input = decimal_batch(value, &[0], &[Some(1)], native);
    let records = vec![
        initial.table_payload().unwrap().batches()[0].clone(),
        input.table_payload().unwrap().batches()[0].clone(),
    ];
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let tables = BTreeMap::from([(
        "events".into(),
        Batch::table(records, BatchMetadata::default()).unwrap(),
    )]);
    let reference = std::panic::AssertUnwindSafe(runtime.sql(query, &tables, None))
        .catch_unwind()
        .await;
    assert!(!matches!(reference, Ok(Ok(_))));
    let job = StreamJobContext::new(1, "prefix", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let result = std::panic::AssertUnwindSafe(operator.process_data(
        "events",
        input,
        &context,
        &mut collector,
    ))
    .catch_unwind()
    .await;
    assert_eq!(result.is_err(), reference.is_err());
    assert!(!matches!(result, Ok(Ok(()))));
    assert!(collector.drain("output").is_empty());
    assert!(operator.incremental.is_none());
    unchanged_checkpoint(&before, &mut operator);
}

#[tokio::test]
async fn test_sql_incremental_decimal_avg_negative_scales_preserve_reference_failure() {
    for scale in [-128, -2] {
        for value in decimal_cases(scale) {
            avg_negative_scale(&value, false, false).await;
            avg_negative_scale(&value, false, true).await;
            avg_negative_scale(&value, true, true).await;
        }
    }
}
