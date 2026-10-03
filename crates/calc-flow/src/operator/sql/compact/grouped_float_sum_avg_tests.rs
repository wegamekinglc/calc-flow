use super::*;

const FLOAT_SUM: &str = "SELECT key, SUM(value) AS total FROM events GROUP BY key";
const FLOAT_AVG: &str = "SELECT key, AVG(value) AS mean FROM events GROUP BY key";
const FLOAT_STATE: &str = "SELECT key, COUNT(value) AS valid, SUM(CAST(value AS DOUBLE)) AS total FROM events GROUP BY key";
const LARGE_FLOAT: Bits = (0x5a80_0000, 0x4350_0000_0000_0000);
const NEG_LARGE_FLOAT: Bits = (0xda80_0000, 0xc350_0000_0000_0000);
const SECOND_NAN: Bits = (0xffc0_0002, 0xfff8_0000_0000_0002);
const SUBNORMAL: Bits = (1, 1);
const NEG_SUBNORMAL: Bits = (0x8000_0001, 0x8000_0000_0000_0001);

fn cases() -> [(DataType, &'static str); 3] {
    [
        (DataType::Float32, FLOAT_SUM),
        (DataType::Float32, FLOAT_AVG),
        (DataType::Float64, FLOAT_AVG),
    ]
}

fn float_arrivals(reverse: bool) -> Vec<Vec<Part>> {
    let (first, next) = if reverse {
        (NEG_ZERO, ZERO)
    } else {
        (ZERO, NEG_ZERO)
    };
    let mut boundary = vec![(Some(12), Some(LARGE_FLOAT))];
    boundary.extend(vec![(Some(12), Some(ZERO)); 8191]);
    boundary.extend([(Some(12), Some(NEG_LARGE_FLOAT)), (Some(12), Some(ONE))]);
    vec![
        vec![vec![
            (Some(1), Some(LARGE_FLOAT)),
            (Some(2), Some(first)),
            (Some(3), None),
            (None, None),
            (Some(5), None),
            (Some(7), Some(NAN)),
            (Some(8), Some(POS_INF)),
            (Some(9), Some(NEG_INF)),
            (Some(10), Some(SNAN)),
            (Some(11), Some(SUBNORMAL)),
            (Some(13), Some(NEG_SUBNORMAL)),
        ]],
        vec![
            vec![
                (Some(1), Some(NEG_LARGE_FLOAT)),
                (Some(1), Some(ONE)),
                (Some(2), Some(next)),
                (Some(3), Some(TWO)),
                (Some(4), None),
                (Some(7), Some(SECOND_NAN)),
            ],
            vec![
                (None, Some(NEG_ZERO)),
                (Some(8), Some(NEG_INF)),
                (Some(9), Some(POS_INF)),
                (Some(10), Some(ONE)),
                (Some(11), Some(SUBNORMAL)),
                (Some(13), Some(NEG_SUBNORMAL)),
            ],
        ],
        vec![
            vec![],
            boundary,
            vec![(Some(1), None), (Some(4), None), (Some(5), None)],
        ],
    ]
}

async fn float_state_expected(dtype: &DataType, parts: &[Part], sequence: u64) -> Batch {
    DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            FLOAT_STATE,
            &BTreeMap::from([("events".into(), input(dtype, parts, sequence))]),
            Some("float-state-oracle"),
        )
        .await
        .unwrap()
}

fn float_wire_rows(expected: &Batch, query: &str) -> BTreeMap<Option<i64>, Vec<Cell>> {
    rows(expected)
        .into_iter()
        .map(|(key, mut values)| {
            let sum = values.pop().unwrap();
            let Cell::Other(ScalarValue::Int64(Some(count))) = values.pop().unwrap() else {
                panic!("DF COUNT(value) must be non-null Int64");
            };
            let mut values = vec![Cell::Other(ScalarValue::Int64(key))];
            if query == FLOAT_AVG {
                values.push(Cell::Other(ScalarValue::UInt64(
                    (count != 0).then(|| u64::try_from(count).unwrap()),
                )));
            }
            values.push(sum);
            (key, values)
        })
        .collect()
}

async fn float_capture(
    state: &mut SqlOperator,
    dtype: &DataType,
    query: &str,
    parts: &[Part],
    sequence: u64,
) -> OperatorStateSnapshot {
    assert!(
        state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
        "grouped Float32 SUM / floating AVG must own chronological native state"
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
    let expected = if query == FLOAT_AVG {
        vec![
            ("key_0", DataType::Int64),
            ("state_0_0", DataType::UInt64),
            ("state_0_1", DataType::Float64),
        ]
    } else {
        vec![("key_0", DataType::Int64), ("state_0_0", DataType::Float64)]
    };
    assert_eq!(fields.len(), expected.len());
    for (field, (name, dtype)) in fields.iter().zip(expected) {
        assert_eq!(field.name(), name);
        assert_eq!(field.data_type(), &dtype);
        assert!(field.is_nullable());
    }
    let expected = float_state_expected(dtype, parts, sequence).await;
    assert_eq!(rows(&wire), float_wire_rows(&expected, query));
    snapshot
}

#[tokio::test]
async fn test_grouped_float_sum_average_chronological_prefix_bits_own_native3() {
    for (dtype, query) in cases() {
        for reverse in [false, true] {
            let job = job();
            let context = StreamOperatorContext::new(&job, "grouped_float", None);
            let mut state = operator_query(&dtype, query);
            let mut prefix = Vec::new();
            let mut weak = Vec::new();
            for (sequence, parts) in float_arrivals(reverse).into_iter().enumerate() {
                let batch = input(&dtype, &parts, sequence as u64);
                weak.extend(weak_arrays(&batch));
                prefix.extend(parts);
                let actual = process(&mut state, batch, &context).await;
                assert_query_oracle(&actual, query, &dtype, &prefix, sequence as u64).await;
                if sequence == 2 {
                    drop(float_capture(&mut state, &dtype, query, &prefix, sequence as u64).await);
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
}

async fn float_roundtrip(dtype: &DataType, query: &str, reverse: bool) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "grouped_float", None);
    let mut state = operator_query(dtype, query);
    let mut prefix = Vec::new();
    for (sequence, parts) in float_arrivals(reverse).into_iter().enumerate() {
        prefix.extend(parts.clone());
        let actual = process(&mut state, input(dtype, &parts, sequence as u64), &context).await;
        assert_query_oracle(&actual, query, dtype, &prefix, sequence as u64).await;
    }
    let before = float_capture(&mut state, dtype, query, &prefix, 2).await;
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(state);
    let mut restored = operator_query(dtype, query);
    StreamOperator::restore(&mut restored, &before).unwrap();
    same_snapshot(&before, &restored.checkpoint(Epoch::INITIAL).unwrap());
    let empty = process(&mut restored, input(dtype, &[vec![]], 3), &context).await;
    assert_query_oracle(&empty, query, dtype, &prefix, 3).await;
    let empty_capture = float_capture(&mut restored, dtype, query, &prefix, 3).await;
    let next = vec![vec![
        (Some(1), Some(TWO)),
        (Some(3), Some(NEG_ZERO)),
        (Some(4), Some(SUBNORMAL)),
        (Some(5), None),
        (Some(7), Some(SNAN)),
        (Some(10), Some(SECOND_NAN)),
        (Some(11), Some(NEG_SUBNORMAL)),
        (None, Some(ONE)),
        (Some(14), None),
    ]];
    let batch = input(dtype, &next, 4);
    let weak = weak_arrays(&batch);
    let actual = process(&mut restored, batch, &context).await;
    prefix.extend(next);
    assert_query_oracle(&actual, query, dtype, &prefix, 4).await;
    let after = float_capture(&mut restored, dtype, query, &prefix, 4).await;
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
async fn test_grouped_float_sum_average_current_capture_restore_empty_then_continue() {
    for (dtype, query) in cases() {
        for reverse in [false, true] {
            float_roundtrip(&dtype, query, reverse).await;
        }
    }
}

#[tokio::test]
async fn test_grouped_float_sum_average_emit_failure_refunds_then_single_retry() {
    for (dtype, query) in cases() {
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = operator_query(&dtype, query);
        let first = vec![vec![(Some(1), Some(LARGE_FLOAT)), (None, None)]];
        let initial = process(&mut state, input(&dtype, &first, 0), &context).await;
        assert_query_oracle(&initial, query, &dtype, &first, 0).await;
        let before = state.checkpoint(Epoch::INITIAL).unwrap();
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let reserved = pool.reserved();
        let next = vec![vec![
            (Some(1), Some(NEG_LARGE_FLOAT)),
            (Some(1), Some(ONE)),
            (None, Some(SNAN)),
            (Some(4), None),
            (Some(3), Some(POS_INF)),
        ]];
        let rejected = input(&dtype, &next, 1);
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
        let retry = input(&dtype, &next, 1);
        let weak = weak_arrays(&retry);
        let actual = process(&mut state, retry, &context).await;
        let all = first.into_iter().chain(next).collect::<Vec<_>>();
        assert_query_oracle(&actual, query, &dtype, &all, 1).await;
        let after = float_capture(&mut state, &dtype, query, &all, 1).await;
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        drop((state, before, after, initial, actual));
        assert_eq!(pool.reserved(), 0);
    }
}
