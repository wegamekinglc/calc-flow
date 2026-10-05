use super::*;

const GLOBAL_CAST: &str = "SELECT SUM(CAST(value AS REAL)) AS total, AVG(CAST(value AS REAL)) AS mean, MIN(CAST(value AS REAL)) AS lo, MAX(CAST(value AS REAL)) AS hi, COUNT(CAST(value AS REAL)) AS valid, COUNT(*) AS rows FROM events";
const GROUPED_CAST: &str = "SELECT key, SUM(CAST(value AS REAL)) AS total, AVG(CAST(value AS REAL)) AS mean, COUNT(CAST(value AS REAL)) AS valid FROM events GROUP BY key";
const WHERE_CAST: &str = "SELECT SUM(CAST(value AS REAL)) AS total, AVG(CAST(value AS REAL)) AS mean, COUNT(CAST(value AS REAL)) AS valid FROM events WHERE key = 1";

#[tokio::test]
async fn test_aggregate_float_input_casts_release_input_and_preserve_exact_prefixes() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in [GLOBAL_CAST, GROUPED_CAST, WHERE_CAST] {
            let job = job();
            let context = StreamOperatorContext::new(&job, "float_extrema", None);
            let mut state = operator(&dtype, query);
            let mut history = Vec::new();
            let mut pools = Vec::new();
            let mut boundary = vec![Some(ONE); 8194];
            boundary[0] = Some((0x5a80_0000, 0x4350_0000_0000_0000));
            boundary[8192] = Some((0xda80_0000, 0xc350_0000_0000_0000));
            for (sequence, parts) in [
                vec![vec![
                    Some(NEG_ZERO),
                    None,
                    Some((0x4b80_0000, 0x4170_0000_1000_0000)),
                ]],
                vec![boundary],
                vec![vec![Some(NAN_A), Some(SNAN), Some(POS_INF)]],
                vec![vec![]],
            ]
            .into_iter()
            .enumerate()
            {
                let incoming = input(&dtype, &parts, sequence as u64);
                let weak = weak_arrays(&incoming);
                let actual = process(&mut state, incoming, &context).await;
                history.extend(parts);
                assert_oracle(&actual, query, &dtype, &history, sequence as u64).await;
                assert!(
                    state.incremental.is_some()
                        && state.compact.is_some()
                        && state.retained.is_none(),
                    "numeric float input casts must keep native cumulative state"
                );
                assert!(weak.iter().all(|array| array.upgrade().is_none()));
                let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
                if query != GROUPED_CAST {
                    let expected = DataFusionRuntime::new(DataFusionConfig::default())
                        .unwrap()
                        .global_record_accumulator_state(
                            query,
                            &input(&dtype, &history, sequence as u64),
                        )
                        .await;
                    let wire = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
                    let expected = expected
                        .iter()
                        .map(|value| cell(&value.to_array().unwrap(), 0))
                        .collect::<Vec<_>>();
                    assert_eq!(rows(&wire), vec![expected]);
                }
                pools.push(
                    state
                        .stream_state
                        .runtime()
                        .unwrap()
                        .incremental_memory_pool(),
                );
                drop(state);
                state = operator(&dtype, query);
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
            assert_eq!(pools.iter().map(|pool| pool.reserved()).sum::<usize>(), 0);
        }
    }
}

#[tokio::test]
async fn test_aggregate_float_input_casts_refusal_refunds_and_once_retry() {
    use datafusion::execution::memory_pool::MemoryConsumer;
    let dtype = DataType::Float64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, WHERE_CAST);
    let first = vec![vec![Some(ONE)]];
    drop(process(&mut state, input(&dtype, &first, 0), &context).await);
    assert!(state.incremental.is_some());
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    let next = input(&dtype, &[vec![Some(TWO)]], 1);
    let weak = weak_arrays(&next);
    assert!(
        matches!(state.process_data("events", next, &context, &mut Reject).await, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-extrema")
    );
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    assert_eq!(pool.reserved(), basis);
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    let pressure = MemoryConsumer::new("input-cast-pressure").register(&pool);
    pressure.try_grow((1 << 30) - basis - 1).unwrap();
    let held = pool.reserved();
    let next = input(&dtype, &[vec![Some(TWO)]], 1);
    let weak = weak_arrays(&next);
    let mut collector = EdgeCollector::new(state.output_ports().to_vec());
    assert!(
        state
            .process_data("events", next, &context, &mut collector)
            .await
            .is_err()
    );
    assert!(collector.drain("output").is_empty());
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    assert_eq!(pool.reserved(), held);
    drop(pressure);
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    let actual = process(&mut state, input(&dtype, &[vec![Some(TWO)]], 1), &context).await;
    assert_oracle(
        &actual,
        WHERE_CAST,
        &dtype,
        &[first[0].clone(), vec![Some(TWO)]],
        1,
    )
    .await;
    drop((state, before, actual));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn integer_input(dtype: &DataType, last: bool) -> Batch {
    let value = match dtype {
        DataType::Int8 => ScalarValue::Int8(Some(if last { i8::MIN } else { i8::MAX })),
        DataType::Int16 => ScalarValue::Int16(Some(if last { i16::MIN } else { i16::MAX })),
        DataType::Int32 => ScalarValue::Int32(Some(if last { i32::MIN } else { i32::MAX })),
        DataType::Int64 => ScalarValue::Int64(Some(if last { i64::MIN } else { i64::MAX })),
        DataType::UInt8 => ScalarValue::UInt8(Some(if last { 0 } else { u8::MAX })),
        DataType::UInt16 => ScalarValue::UInt16(Some(if last { 0 } else { u16::MAX })),
        DataType::UInt32 => ScalarValue::UInt32(Some(if last { 0 } else { u32::MAX })),
        DataType::UInt64 => ScalarValue::UInt64(Some(if last { 0 } else { u64::MAX })),
        _ => unreachable!(),
    };
    let null = ScalarValue::try_new_null(dtype).unwrap();
    let array = ScalarValue::iter_to_array([value, null]).unwrap();
    let record = RecordBatch::try_new(
        schema(dtype),
        vec![Arc::new(Int64Array::from(vec![1, 1])), array],
    )
    .unwrap();
    Batch::table(vec![record], metadata(u64::from(last))).unwrap()
}

#[tokio::test]
async fn test_aggregate_float_input_casts_cover_integer_limits_and_nested_casts() {
    let nested = (0..7).fold("value".to_owned(), |expression, depth| {
        format!(
            "CAST({expression} AS {})",
            if depth % 2 == 0 { "REAL" } else { "DOUBLE" }
        )
    });
    let query = format!(
        "SELECT SUM({nested}) AS total, AVG({nested}) AS mean, COUNT({nested}) AS valid FROM events"
    );
    for dtype in [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
    ] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut state = operator(&dtype, &query);
        let mut history = Vec::new();
        let mut pools = Vec::new();
        for last in [false, true] {
            let incoming = integer_input(&dtype, last);
            let weak = weak_arrays(&incoming);
            history.extend(
                integer_input(&dtype, last)
                    .table_payload()
                    .unwrap()
                    .batches()
                    .iter()
                    .cloned(),
            );
            let actual = process(&mut state, incoming, &context).await;
            let prefix = Batch::table(history.clone(), actual.metadata().clone()).unwrap();
            let expected = DataFusionRuntime::new(DataFusionConfig::default())
                .unwrap()
                .sql(
                    &query,
                    &BTreeMap::from([("events".into(), prefix)]),
                    Some("integer-float-cast-oracle"),
                )
                .await
                .unwrap();
            assert_eq!(rows(&actual), rows(&expected));
            assert_eq!(
                actual.table_payload().unwrap().schema(),
                expected.table_payload().unwrap().schema()
            );
            assert_eq!(actual.metadata(), expected.metadata());
            assert!(
                state.incremental.is_some() && state.compact.is_some() && state.retained.is_none()
            );
            assert!(weak.iter().all(|array| array.upgrade().is_none()));
            let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
            pools.push(
                state
                    .stream_state
                    .runtime()
                    .unwrap()
                    .incremental_memory_pool(),
            );
            drop(state);
            state = operator(&dtype, &query);
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
        assert_eq!(pools.iter().map(|pool| pool.reserved()).sum::<usize>(), 0);
    }
}

#[tokio::test]
async fn test_aggregate_input_casts_numeric_targets_and_variable_source_fallback() {
    use datafusion::arrow::array::StringArray;
    for (dtype, query, native) in [
        (
            DataType::Utf8,
            "SELECT SUM(CAST(value AS REAL)) AS total FROM events",
            false,
        ),
        (
            DataType::Float64,
            "SELECT SUM(CAST(value AS SMALLINT)) AS total FROM events",
            true,
        ),
    ] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut state = operator(&dtype, query);
        let values: ArrayRef = if dtype == DataType::Utf8 {
            Arc::new(StringArray::from(vec![Some("1"), Some("2"), None]))
        } else {
            Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), None]))
        };
        let record = RecordBatch::try_new(
            schema(&dtype),
            vec![Arc::new(Int64Array::from(vec![1; 3])), values],
        )
        .unwrap();
        let expected = DataFusionRuntime::new(DataFusionConfig::default())
            .unwrap()
            .sql(
                query,
                &BTreeMap::from([(
                    "events".into(),
                    Batch::table(vec![record.clone()], metadata(0)).unwrap(),
                )]),
                Some("unsupported-float-cast-oracle"),
            )
            .await
            .unwrap();
        let actual = process(
            &mut state,
            Batch::table(vec![record], metadata(0)).unwrap(),
            &context,
        )
        .await;
        assert_eq!(rows(&actual), rows(&expected));
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(actual.metadata(), expected.metadata());
        assert_eq!(state.incremental.is_some(), native);
        assert_eq!(state.compact.is_some(), native);
        assert_eq!(state.retained.is_none(), native);
        assert_eq!(
            state.checkpoint(Epoch::INITIAL).unwrap().inline_metadata["state_layout"],
            json!(if native { 3 } else { 4 })
        );
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(state);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}
