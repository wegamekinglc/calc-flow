use datafusion::{arrow::datatypes::i256, common::ScalarValue};

use super::*;

fn decimal_cases(scale: i8) -> Vec<ScalarValue> {
    vec![
        ScalarValue::Decimal32(Some(12), 4, scale),
        ScalarValue::Decimal64(Some(12), 12, scale),
        ScalarValue::Decimal128(Some(12), 28, scale),
        ScalarValue::Decimal256(Some(i256::from_i128(12)), 60, scale),
    ]
}

fn decimal_batch(value: &ScalarValue, keys: &[i64], values: &[Option<i64>], native: bool) -> Batch {
    let data_type = value.data_type();
    let values = decimal_array(&data_type, values);
    let keys = if native {
        Arc::new(Int64Array::from(keys.to_vec())) as Arc<dyn Array>
    } else {
        Arc::new(datafusion::arrow::array::StringArray::from_iter_values(
            keys.iter().map(|key| format!("key-{key}")),
        )) as Arc<dyn Array>
    };
    Batch::table(
        vec![
            RecordBatch::try_from_iter_with_nullable([
                ("key", keys, false),
                ("value", values, true),
            ])
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn decimal_array(data_type: &DataType, values: &[Option<i64>]) -> Arc<dyn Array> {
    if values.is_empty() {
        datafusion::arrow::array::new_empty_array(data_type)
    } else {
        ScalarValue::iter_to_array(values.iter().map(|value| decimal_scalar(data_type, *value)))
            .unwrap()
    }
}

fn decimal_scalar(data_type: &DataType, value: Option<i64>) -> ScalarValue {
    match *data_type {
        DataType::Decimal32(precision, scale) => ScalarValue::Decimal32(
            value.map(|value| i32::try_from(value).unwrap()),
            precision,
            scale,
        ),
        DataType::Decimal64(precision, scale) => ScalarValue::Decimal64(value, precision, scale),
        DataType::Decimal128(precision, scale) => {
            ScalarValue::Decimal128(value.map(i128::from), precision, scale)
        }
        DataType::Decimal256(precision, scale) => ScalarValue::Decimal256(
            value.map(|value| i256::from_i128(i128::from(value))),
            precision,
            scale,
        ),
        _ => panic!("expected a decimal type"),
    }
}

async fn decimal_prefixes(value: &ScalarValue, native: bool, grouped: bool) {
    let incoming = [
        decimal_batch(value, &[], &[], native),
        decimal_batch(value, &[0, 1], &[None, None], native),
        decimal_batch(value, &[0, 2], &[Some(3), None], native),
        decimal_batch(value, &[0, 1], &[Some(-2), Some(5)], native),
        decimal_batch(value, &[], &[], native),
    ];
    let aggregates = "COUNT(*) AS rows, COUNT(value) AS valid, SUM(value) AS total, MIN(value) AS lo, MAX(value) AS hi";
    let query = if grouped {
        format!("SELECT key, {aggregates} FROM events GROUP BY key")
    } else {
        format!("SELECT {aggregates} FROM events")
    };
    let mut operator = assert_prefix_oracle(&query, &incoming, true).await;
    assert_eq!(operator.incremental_work, (6, 1), "{query}");
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
        &query,
        &prefix,
        &context,
        &mut collector,
    )
    .await;
}

#[tokio::test]
async fn test_sql_incremental_decimal_prefixes_preserve_nulls_types_and_work() {
    for scale in [-2, 2] {
        for value in decimal_cases(scale) {
            decimal_prefixes(&value, false, false).await;
            decimal_prefixes(&value, false, true).await;
            decimal_prefixes(&value, true, true).await;
        }
    }
}

fn maximum_decimals() -> Vec<ScalarValue> {
    vec![
        ScalarValue::Decimal32(Some(999_999_999), 9, 0),
        ScalarValue::Decimal64(Some(999_999_999_999_999_999), 18, 0),
        ScalarValue::Decimal128(
            Some("99999999999999999999999999999999999999".parse().unwrap()),
            38,
            0,
        ),
        ScalarValue::Decimal256(Some("9".repeat(76).parse().unwrap()), 76, 0),
    ]
}

fn repeated_decimal(value: &ScalarValue, native: bool, count: usize) -> Batch {
    let values = value.to_array_of_size(count).unwrap();
    let keys: Arc<dyn Array> = if native {
        Arc::new(Int64Array::from(vec![0; count]))
    } else {
        Arc::new(datafusion::arrow::array::StringArray::from(vec![
            "key";
            count
        ]))
    };
    Batch::table(
        vec![
            RecordBatch::try_from_iter_with_nullable([
                ("key", keys, false),
                ("value", values, true),
            ])
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_sql_incremental_decimal_precision_limits_match_wrapping_oracles() {
    let query = "SELECT key, SUM(value) AS total, MIN(value) AS lo, MAX(value) AS hi FROM events GROUP BY key";
    for value in maximum_decimals() {
        for native in [false, true] {
            let batch = repeated_decimal(&value, native, 1);
            let records = vec![batch.table_payload().unwrap().batches()[0].clone(); 3];
            let multiple = Batch::table(records, BatchMetadata::default()).unwrap();
            assert_prefix_oracle(query, &[batch.clone(), multiple, batch], true).await;
        }
    }
}

const TRANSACTION_QUERY: &str =
    "SELECT key, SUM(value) AS total, MIN(value) AS lo, MAX(value) AS hi FROM events GROUP BY key";

fn nullable_numeric_batch(value: &ScalarValue, native: bool, valid: bool) -> Batch {
    let template = repeated_decimal(value, native, 3);
    let record = &template.table_payload().unwrap().batches()[0];
    let null = ScalarValue::try_from(&value.data_type()).unwrap();
    let middle = if valid { value.clone() } else { null.clone() };
    let values = ScalarValue::iter_to_array([null.clone(), middle, null]).unwrap();
    Batch::table(
        vec![
            RecordBatch::try_new(record.schema(), vec![record.column(0).clone(), values]).unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_sql_incremental_nullable_small_width_groups_reserve_aligned_bitmaps() {
    let values = vec![
        ScalarValue::Int8(Some(4)),
        ScalarValue::UInt8(Some(4)),
        ScalarValue::Int16(Some(4)),
        ScalarValue::UInt16(Some(4)),
        ScalarValue::Int32(Some(4)),
        ScalarValue::UInt32(Some(4)),
        ScalarValue::Int64(Some(4)),
        ScalarValue::UInt64(Some(4)),
    ];
    for value in values.into_iter().chain(decimal_cases(2)) {
        for native in [false, true] {
            let incoming = [
                nullable_numeric_batch(&value, native, false),
                nullable_numeric_batch(&value, native, true),
            ];
            assert_prefix_oracle(TRANSACTION_QUERY, &incoming, true).await;
        }
    }
}

fn numeric_record(
    value: &ScalarValue,
    native: bool,
    start: i64,
    count: usize,
    null_first: bool,
) -> RecordBatch {
    let keys = (start..start + i64::try_from(count).unwrap()).collect::<Vec<_>>();
    let keys = if native {
        Arc::new(Int64Array::from(keys)) as Arc<dyn Array>
    } else {
        Arc::new(datafusion::arrow::array::StringArray::from_iter_values(
            keys.iter().map(|key| format!("key-{key}")),
        )) as Arc<dyn Array>
    };
    let null = ScalarValue::try_from(&value.data_type()).unwrap();
    let values = ScalarValue::iter_to_array((0..count).map(|row| {
        if null_first && row == 0 {
            null.clone()
        } else {
            value.clone()
        }
    }))
    .unwrap();
    RecordBatch::try_from_iter_with_nullable([("key", keys, false), ("value", values, true)])
        .unwrap()
}

#[tokio::test]
async fn test_sql_incremental_native_bitmap_transition_preserves_unseen_groups() {
    for value in decimal_cases(2)
        .into_iter()
        .chain([ScalarValue::Int8(Some(4)), ScalarValue::UInt8(Some(4))])
    {
        for native in [false, true] {
            let incoming = Batch::table(
                vec![
                    numeric_record(&value, native, 0, 512, false),
                    numeric_record(&value, native, 512, 513, true),
                ],
                BatchMetadata::default(),
            )
            .unwrap();
            let operator =
                assert_prefix_oracle(TRANSACTION_QUERY, std::slice::from_ref(&incoming), true)
                    .await;
            assert_eq!(operator.incremental_work, (1025, 1));
        }
    }
}

async fn fail_decimal_output(
    operator: &mut SqlOperator,
    input: &Batch,
    context: &StreamOperatorContext<'_>,
    reject: bool,
) {
    if reject {
        assert!(matches!(
            operator.process_data("events", input.clone(), context, &mut RejectOutput).await,
            Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject"
        ));
    } else {
        let (entered, started) = tokio::sync::oneshot::channel();
        let mut pending = PendingOutput {
            entered: Some(entered),
        };
        let process = operator.process_data("events", input.clone(), context, &mut pending);
        tokio::pin!(process);
        tokio::select! {
            result = &mut process => panic!("emit finished unexpectedly: {result:?}"),
            result = started => result.unwrap(),
        }
    }
}

async fn assert_decimal_output(
    operator: &mut SqlOperator,
    incoming: &Batch,
    expected_input: &[Batch],
    context: &StreamOperatorContext<'_>,
) {
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", incoming.clone(), context, &mut collector)
        .await
        .unwrap();
    let records = expected_input
        .iter()
        .flat_map(|batch| batch.table_payload().unwrap().batches().to_vec())
        .collect::<Vec<_>>();
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            TRANSACTION_QUERY,
            &BTreeMap::from([(
                "events".into(),
                Batch::table(records, incoming.metadata().clone()).unwrap(),
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

async fn decimal_transaction(value: &ScalarValue, native: bool, reject: bool) {
    let initial = decimal_batch(value, &[0, 1], &[None, Some(4)], native);
    let mut operator =
        assert_prefix_oracle(TRANSACTION_QUERY, std::slice::from_ref(&initial), true).await;
    let job = StreamJobContext::new(1, "prefix", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "totals", None);
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    let keys = (0..1025).collect::<Vec<_>>();
    let incoming = decimal_batch(value, &keys, &vec![Some(2); keys.len()], native);
    for _ in 0..2 {
        fail_decimal_output(&mut operator, &incoming, &context, reject).await;
        let after = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(before.inline_metadata, after.inline_metadata);
        assert!(Arc::ptr_eq(
            &before.segments["group-state"].bytes_arc(),
            &after.segments["group-state"].bytes_arc()
        ));
        let pressure = operator
            .stream_state
            .runtime()
            .unwrap()
            .incremental_reservation("probe");
        pressure.try_grow((1 << 30) - (1 << 20)).unwrap();
    }
    assert_decimal_output(
        &mut operator,
        &incoming,
        &[initial.clone(), incoming.clone()],
        &context,
    )
    .await;
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = operator.clone();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let next = decimal_batch(value, &[0, 1026], &[Some(-1), None], native);
    assert_decimal_output(
        &mut restored,
        &next,
        &[initial, incoming, next.clone()],
        &context,
    )
    .await;
    StreamOperator::reset(&mut restored).unwrap();
    assert_decimal_output(&mut restored, &next, std::slice::from_ref(&next), &context).await;
    let mut cloned = operator.clone();
    assert_decimal_output(&mut cloned, &next, std::slice::from_ref(&next), &context).await;
}

#[tokio::test]
async fn test_sql_incremental_decimal_output_failure_preserves_state_and_retries_once() {
    for value in decimal_cases(2) {
        for native in [false, true] {
            for reject in [false, true] {
                decimal_transaction(&value, native, reject).await;
            }
        }
    }
}

#[tokio::test]
async fn test_sql_incremental_decimal_cancel_preserves_checkpoint() {
    for value in decimal_cases(-2) {
        for native in [false, true] {
            let initial = decimal_batch(&value, &[0, 1], &[None, Some(4)], native);
            let mut operator =
                assert_prefix_oracle(TRANSACTION_QUERY, std::slice::from_ref(&initial), true).await;
            let before = operator.checkpoint(Epoch::INITIAL).unwrap();
            let job =
                StreamJobContext::new(1, "prefix", JsonMap::new(), None, CancellationToken::new());
            job.cancellation().cancel();
            let context = StreamOperatorContext::new(&job, "totals", None);
            let incoming = decimal_batch(&value, &[0, 2], &[Some(2), None], native);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            assert!(matches!(
                operator
                    .process_data("events", incoming, &context, &mut collector)
                    .await,
                Err(CalcFlowError::Cancelled { .. })
            ));
            assert!(collector.drain("output").is_empty());
            let after = operator.checkpoint(Epoch::INITIAL).unwrap();
            assert_eq!(before.inline_metadata, after.inline_metadata);
            assert!(Arc::ptr_eq(
                &before.segments["group-state"].bytes_arc(),
                &after.segments["group-state"].bytes_arc()
            ));
        }
    }
}

#[path = "decimal_avg_tests.rs"]
mod decimal_avg_tests;
