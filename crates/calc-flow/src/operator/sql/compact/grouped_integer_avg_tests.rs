use super::*;
use datafusion::arrow::array::UInt64Array;

type IntegerPart = Vec<(Option<i64>, Option<i128>)>;
const AVG_QUERY: &str = "SELECT key, AVG(value) AS mean FROM events GROUP BY key";
const STATE_QUERY: &str = "SELECT key, COUNT(value) AS valid, SUM(CAST(value AS DOUBLE)) AS total FROM events GROUP BY key";
const INTEGER_LARGE: i128 = 1 << 54;

#[tokio::test]
async fn test_grouped_integer_mixed_sum_average_count_prefix_restore() {
    let query = "SELECT key, SUM(value) AS total, AVG(value) AS mean, COUNT(value) AS valid FROM events GROUP BY key";
    for dtype in [DataType::Int64, DataType::UInt64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = operator_query(&dtype, query);
        let mut prefix = Vec::new();
        let mut pools = Vec::new();
        for (sequence, parts) in average_arrivals(&dtype).into_iter().enumerate() {
            let batch = average_input(&dtype, &parts, sequence as u64);
            let weak = weak_arrays(&batch);
            let actual = process(&mut state, batch, &context).await;
            prefix.extend(parts);
            let expected = average_expected(query, &dtype, &prefix, sequence as u64).await;
            assert_eq!(rows(&actual), rows(&expected));
            assert_eq!(
                actual.table_payload().unwrap().schema(),
                expected.table_payload().unwrap().schema()
            );
            assert_eq!(actual.metadata(), expected.metadata());
            assert!(weak.iter().all(|array| array.upgrade().is_none()));
            let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
            assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
            assert!(
                snapshot.segments.contains_key("group-state")
                    && !snapshot.segments.contains_key("input-retained")
            );
            pools.push(
                state
                    .stream_state
                    .runtime()
                    .unwrap()
                    .incremental_memory_pool(),
            );
            drop(state);
            state = operator_query(&dtype, query);
            StreamOperator::restore(&mut state, &snapshot).unwrap();
            same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
        }
        pools.push(
            state
                .stream_state
                .runtime()
                .unwrap()
                .incremental_memory_pool(),
        );
        drop(state);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(
            pools.iter().map(|pool| pool.reserved()).collect::<Vec<_>>(),
            vec![0; pools.len()]
        );
    }
}

fn average_input(dtype: &DataType, parts: &[IntegerPart], sequence: u64) -> Batch {
    let records = parts
        .iter()
        .map(|part| {
            let values: ArrayRef = match dtype {
                DataType::Int64 => Arc::new(Int64Array::from(
                    part.iter()
                        .map(|row| row.1.map(|value| i64::try_from(value).unwrap()))
                        .collect::<Vec<_>>(),
                )),
                DataType::UInt64 => Arc::new(UInt64Array::from(
                    part.iter()
                        .map(|row| row.1.map(|value| u64::try_from(value).unwrap()))
                        .collect::<Vec<_>>(),
                )),
                _ => unreachable!(),
            };
            let arrays = (0..8)
                .map(|index| match index {
                    0 => Arc::new(Int64Array::from(
                        part.iter().map(|row| row.0).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    6 => values.clone(),
                    _ => Arc::new(StringArray::from(vec![
                        "unused integer payload";
                        part.len()
                    ])) as ArrayRef,
                })
                .collect();
            RecordBatch::try_new(schema(dtype), arrays).unwrap()
        })
        .collect();
    Batch::table(
        records,
        BatchMetadata::new(
            "grouped-integer-average",
            sequence,
            JsonMap::from([
                ("prefix".into(), json!(sequence)),
                ("case".into(), json!(format!("{dtype:?}"))),
                (
                    "nested".into(),
                    json!({"nullable": null, "values": [1, "average"]}),
                ),
            ]),
        )
        .unwrap(),
    )
    .unwrap()
}

async fn average_expected(
    query: &str,
    dtype: &DataType,
    parts: &[IntegerPart],
    sequence: u64,
) -> Batch {
    DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([("events".into(), average_input(dtype, parts, sequence))]),
            Some("integer-average-oracle"),
        )
        .await
        .unwrap()
}

async fn average_oracle(actual: &Batch, dtype: &DataType, parts: &[IntegerPart], sequence: u64) {
    let expected = average_expected(AVG_QUERY, dtype, parts, sequence).await;
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(rows(actual), rows(&expected));
    assert_eq!(actual.metadata(), expected.metadata());
    assert_eq!(
        actual.metadata(),
        average_input(dtype, parts, sequence).metadata()
    );
}

fn cancellation_rows(dtype: &DataType, key: i64) -> IntegerPart {
    match dtype {
        DataType::Int64 => vec![(Some(key), Some(-INTEGER_LARGE)), (Some(key), Some(1))],
        DataType::UInt64 => vec![(Some(key), Some(1)); 3],
        _ => unreachable!(),
    }
}

fn average_arrivals(dtype: &DataType) -> Vec<Vec<IntegerPart>> {
    let max = match dtype {
        DataType::Int64 => i128::from(i64::MAX),
        _ => i128::from(u64::MAX),
    };
    let mut next = cancellation_rows(dtype, 1);
    next.extend([(Some(3), Some(3)), (Some(4), None), (Some(2), Some(1))]);
    let mut boundary = vec![(Some(12), Some(INTEGER_LARGE))];
    boundary.extend(vec![(Some(12), Some(0)); 8191]);
    boundary.extend(cancellation_rows(dtype, 12));
    vec![
        vec![vec![
            (Some(1), Some(INTEGER_LARGE)),
            (Some(2), Some((1 << 53) + 1)),
            (Some(3), None),
            (None, None),
            (Some(5), None),
            (Some(7), Some(max)),
            (Some(8), Some(1)),
            (Some(9), Some(0)),
        ]],
        vec![
            next,
            vec![(None, Some(2)), (Some(7), Some(1)), (Some(9), None)],
        ],
        vec![
            vec![],
            boundary,
            vec![(Some(1), None), (Some(4), None), (Some(5), None)],
        ],
    ]
}

fn average_wire_rows(expected: &Batch) -> BTreeMap<Option<i64>, Vec<Cell>> {
    rows(expected)
        .into_iter()
        .map(|(key, mut values)| {
            let sum = values.pop().unwrap();
            let Cell::Other(ScalarValue::Int64(Some(count))) = values.pop().unwrap() else {
                panic!("DF COUNT(value) must be non-null Int64");
            };
            let count = (count != 0).then(|| u64::try_from(count).unwrap());
            (
                key,
                vec![
                    Cell::Other(ScalarValue::Int64(key)),
                    Cell::Other(ScalarValue::UInt64(count)),
                    sum,
                ],
            )
        })
        .collect()
}

async fn average_capture(
    state: &mut SqlOperator,
    dtype: &DataType,
    parts: &[IntegerPart],
    sequence: u64,
) -> OperatorStateSnapshot {
    assert!(
        state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
        "grouped integer AVG must own chronological native state, not retained input"
    );
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
    assert_eq!(snapshot.inline_metadata["state_accounting"], json!(3));
    assert_eq!(
        snapshot.inline_metadata["rows"],
        json!(parts.iter().map(Vec::len).sum::<usize>())
    );
    assert_eq!(
        snapshot
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["batch-metadata", "control", "group-state", "logical-schema"]
    );
    let projection = state.compact.as_ref().unwrap().projection().unwrap();
    assert_eq!(projection.columns.logical_schema(), &schema(dtype));
    assert_eq!(projection.columns.ordinals(), &[0, 6]);
    let wire = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
    let fields = wire.table_payload().unwrap().schema().fields();
    assert_eq!(fields.len(), 3);
    for (field, name, dtype) in [
        (&fields[0], "key_0", DataType::Int64),
        (&fields[1], "state_0_0", DataType::UInt64),
        (&fields[2], "state_0_1", DataType::Float64),
    ] {
        assert_eq!(field.name(), name);
        assert_eq!(field.data_type(), &dtype);
        assert!(field.is_nullable());
    }
    let expected = average_expected(STATE_QUERY, dtype, parts, sequence).await;
    assert_eq!(rows(&wire), average_wire_rows(&expected));
    snapshot
}

#[tokio::test]
async fn test_grouped_integer_avg_chronological_prefix_bits_own_native3() {
    for dtype in [DataType::Int64, DataType::UInt64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = operator_query(&dtype, AVG_QUERY);
        let mut prefix = Vec::new();
        let mut weak = Vec::new();
        for (sequence, parts) in average_arrivals(&dtype).into_iter().enumerate() {
            let batch = average_input(&dtype, &parts, sequence as u64);
            weak.extend(weak_arrays(&batch));
            prefix.extend(parts);
            let actual = process(&mut state, batch, &context).await;
            average_oracle(&actual, &dtype, &prefix, sequence as u64).await;
            if sequence == 2 {
                drop(average_capture(&mut state, &dtype, &prefix, sequence as u64).await);
            }
        }
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(state);
        assert_eq!(pool.reserved(), 0);
    }
}

async fn average_roundtrip(dtype: &DataType) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "grouped_float", None);
    let mut state = operator_query(dtype, AVG_QUERY);
    let mut prefix = Vec::new();
    for (sequence, parts) in average_arrivals(dtype).into_iter().enumerate() {
        prefix.extend(parts.clone());
        let actual = process(
            &mut state,
            average_input(dtype, &parts, sequence as u64),
            &context,
        )
        .await;
        average_oracle(&actual, dtype, &prefix, sequence as u64).await;
    }
    let before = average_capture(&mut state, dtype, &prefix, 2).await;
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(state);
    let mut restored = operator_query(dtype, AVG_QUERY);
    StreamOperator::restore(&mut restored, &before).unwrap();
    same_snapshot(&before, &restored.checkpoint(Epoch::INITIAL).unwrap());
    let empty = process(&mut restored, average_input(dtype, &[vec![]], 3), &context).await;
    average_oracle(&empty, dtype, &prefix, 3).await;
    let empty_capture = average_capture(&mut restored, dtype, &prefix, 3).await;
    let next = vec![vec![
        (Some(1), Some(2)),
        (Some(3), Some(0)),
        (Some(4), Some(5)),
        (Some(5), None),
        (Some(10), None),
        (None, Some(0)),
        (Some(7), Some(1)),
    ]];
    let batch = average_input(dtype, &next, 4);
    let weak = weak_arrays(&batch);
    let actual = process(&mut restored, batch, &context).await;
    prefix.extend(next);
    average_oracle(&actual, dtype, &prefix, 4).await;
    let after = average_capture(&mut restored, dtype, &prefix, 4).await;
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    let target = restored
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((restored, before, after, empty_capture, empty, actual));
    assert_eq!(pool.reserved(), 0);
    assert_eq!(target.reserved(), 0);
}

#[tokio::test]
async fn test_grouped_integer_avg_current_capture_restore_empty_then_continue() {
    for dtype in [DataType::Int64, DataType::UInt64] {
        average_roundtrip(&dtype).await;
    }
}

#[tokio::test]
async fn test_grouped_integer_avg_emit_failure_refunds_then_single_retry() {
    for dtype in [DataType::Int64, DataType::UInt64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = operator_query(&dtype, AVG_QUERY);
        let first = vec![vec![(Some(1), Some(INTEGER_LARGE)), (None, None)]];
        let initial = process(&mut state, average_input(&dtype, &first, 0), &context).await;
        average_oracle(&initial, &dtype, &first, 0).await;
        let before = state.checkpoint(Epoch::INITIAL).unwrap();
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let reserved = pool.reserved();
        let mut next_part = cancellation_rows(&dtype, 1);
        next_part.extend([(None, Some(2)), (Some(4), None), (Some(3), Some(5))]);
        let next = vec![next_part];
        let rejected = average_input(&dtype, &next, 1);
        let weak = weak_arrays(&rejected);
        let failure = state
            .process_data("events", rejected, &context, &mut Reject)
            .await;
        assert!(
            matches!(failure, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-grouped-float")
        );
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        assert_eq!(pool.reserved(), reserved);
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
        let retry = average_input(&dtype, &next, 1);
        let weak = weak_arrays(&retry);
        let actual = process(&mut state, retry, &context).await;
        let all = first.into_iter().chain(next).collect::<Vec<_>>();
        average_oracle(&actual, &dtype, &all, 1).await;
        let after = average_capture(&mut state, &dtype, &all, 1).await;
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        drop((state, before, after, initial, actual));
        assert_eq!(pool.reserved(), 0);
    }
}
