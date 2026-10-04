use super::*;

const FLOAT_SUM: &str = "SELECT key, SUM(value) AS total FROM events GROUP BY key";
const FLOAT_AVG: &str = "SELECT key, AVG(value) AS mean FROM events GROUP BY key";
const FLOAT_MIXED: &str =
    "SELECT key, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key";
const FLOAT_MIXED_COUNT: &str = "SELECT key, COUNT(value) AS valid, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key";
const FLOAT_ALL: &str = "SELECT key, MIN(value) AS lo, MAX(value) AS hi, COUNT(value) AS valid, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key";
const FLOAT_STATE: &str = "SELECT key, COUNT(value) AS valid, SUM(CAST(value AS DOUBLE)) AS total, MIN(value) AS lo, MAX(value) AS hi FROM events GROUP BY key";
const LARGE_FLOAT: Bits = (0x5a80_0000, 0x4350_0000_0000_0000);
const NEG_LARGE_FLOAT: Bits = (0xda80_0000, 0xc350_0000_0000_0000);
const SECOND_NAN: Bits = (0xffc0_0002, 0xfff8_0000_0000_0002);
const SUBNORMAL: Bits = (1, 1);
const NEG_SUBNORMAL: Bits = (0x8000_0001, 0x8000_0000_0000_0001);

fn cases() -> [(DataType, &'static str); 8] {
    [
        (DataType::Float32, FLOAT_SUM),
        (DataType::Float32, FLOAT_AVG),
        (DataType::Float64, FLOAT_AVG),
        (DataType::Float32, FLOAT_MIXED),
        (DataType::Float64, FLOAT_MIXED),
        (DataType::Float64, FLOAT_MIXED_COUNT),
        (DataType::Float32, FLOAT_ALL),
        (DataType::Float64, FLOAT_ALL),
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

#[tokio::test]
async fn test_grouped_float_sum_average_nan_payload_chronology_native3() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in [FLOAT_MIXED, FLOAT_ALL] {
            let job = job();
            let context = StreamOperatorContext::new(&job, "grouped_float", None);
            let mut state = operator_query(&dtype, query);
            let mut prefix = Vec::new();
            let mut pools = Vec::new();
            for sequence in 0_usize..4 {
                let part = (0..100)
                    .map(|row| {
                        let index = 90_000 + sequence * 100 + row;
                        let value = match index % 17 {
                            0 => NAN,
                            1 => SECOND_NAN,
                            2 => SNAN,
                            3 => (0xff80_0004, 0xfff0_0000_0000_0004),
                            4 => POS_INF,
                            5 => NEG_INF,
                            6 => NEG_ZERO,
                            _ => ONE,
                        };
                        (
                            (row % 19 != 0).then_some(i64::try_from(index % 64).unwrap()),
                            (row % 13 != 0).then_some(value),
                        )
                    })
                    .collect::<Part>();
                let batch = input(&dtype, &[part.clone()], sequence as u64);
                let weak = weak_arrays(&batch);
                let actual = process(&mut state, batch, &context).await;
                prefix.push(part);
                assert_query_oracle(&actual, query, &dtype, &prefix, sequence as u64).await;
                let snapshot =
                    float_capture(&mut state, &dtype, query, &prefix, sequence as u64).await;
                assert!(weak.iter().all(|array| array.upgrade().is_none()));
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
            assert_eq!(pools.iter().map(|pool| pool.reserved()).sum::<usize>(), 0);
        }
    }
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
            let hi = values.pop().unwrap();
            let lo = values.pop().unwrap();
            let Cell::Float64(sum) = values.pop().unwrap() else {
                panic!("DF grouped SUM(CAST(value AS DOUBLE)) must be Float64");
            };
            let Cell::Other(ScalarValue::Int64(Some(count))) = values.pop().unwrap() else {
                panic!("DF COUNT(value) must be non-null Int64");
            };
            let mut values = vec![Cell::Other(ScalarValue::Int64(key))];
            if query == FLOAT_ALL {
                values.extend([lo, hi]);
            }
            if matches!(query, FLOAT_ALL | FLOAT_MIXED_COUNT) {
                values.push(Cell::Other(ScalarValue::Int64(Some(count))));
            }
            if query != FLOAT_AVG {
                values.push(Cell::Float64(sum));
            }
            if query != FLOAT_SUM {
                values.extend([
                    Cell::Other(ScalarValue::UInt64(
                        (count != 0).then(|| u64::try_from(count).unwrap()),
                    )),
                    Cell::Float64(sum),
                ]);
            }
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
    let descriptor = state
        .incremental
        .as_ref()
        .unwrap()
        .native_descriptor("grouped_float")
        .unwrap();
    assert_eq!(fields, descriptor.wire_schema.fields());
    for field in fields {
        let count = match query {
            FLOAT_MIXED_COUNT => field.name() == "state_0_0",
            FLOAT_ALL => field.name() == "state_2_0",
            _ => false,
        };
        assert_eq!(field.is_nullable(), !count);
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

#[tokio::test]
async fn test_grouped_mixed_float_nondefault_keeps_retained4() {
    for dtype in [DataType::Float32, DataType::Float64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = operator_query(&dtype, FLOAT_ALL);
        state.set_stream_resources(
            DataFusionConfig {
                target_partitions: 4,
                ..DataFusionConfig::default()
            },
            UdfRegistrySnapshot::default(),
            vec![],
        );
        let parts = vec![vec![(Some(1), Some(ONE)), (None, None)]];
        let actual = process(&mut state, input(&dtype, &parts, 0), &context).await;
        assert_query_oracle(&actual, FLOAT_ALL, &dtype, &parts, 0).await;
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(snapshot.inline_metadata["state_layout"], json!(4));
        assert!(snapshot.segments.contains_key("input-retained"));
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((state, snapshot, actual));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}
