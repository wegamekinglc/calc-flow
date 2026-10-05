use super::*;

#[path = "post_arithmetic_tests.rs"]
mod post_arithmetic_tests;

#[path = "post_having_tests.rs"]
mod post_having_tests;

#[path = "post_order_tests.rs"]
mod post_order_tests;

#[path = "post_case_tests.rs"]
mod post_case_tests;

#[path = "input_arithmetic_tests.rs"]
mod input_arithmetic_tests;

#[path = "computed_filter_tests.rs"]
mod computed_filter_tests;

#[path = "group_key_tests.rs"]
mod group_key_tests;

const GLOBAL_CAST: &str = "SELECT CAST(SUM(value) AS REAL) AS total, CAST(AVG(value) AS REAL) AS mean, CAST(COUNT(*) AS DECIMAL(20, 0)) AS rows, 7 AS constant FROM events";
const GROUPED_CAST: &str = "SELECT CAST(key AS SMALLINT) AS bucket, CAST(SUM(value) AS REAL) AS total, CAST(COUNT(*) AS DECIMAL(20, 0)) AS rows, 7 AS constant FROM events GROUP BY key";
const TRY_CAST: &str = "SELECT TRY_CAST(SUM(value) AS SMALLINT) AS safe, CAST(COUNT(*) AS DECIMAL(20, 0)) AS rows, NULL AS nothing FROM events";

fn post_input(dtype: &DataType, parts: &[Part], sequence: u64) -> Batch {
    let original = input(dtype, parts, sequence);
    let records = original
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            let mut columns = record.columns().to_vec();
            columns[0] = Arc::new(Int64Array::from(
                (0..record.num_rows())
                    .map(|row| i64::try_from(row % 3).unwrap() + 1)
                    .collect::<Vec<_>>(),
            ));
            RecordBatch::try_new(record.schema(), columns).unwrap()
        })
        .collect();
    Batch::table(records, original.metadata().clone()).unwrap()
}

async fn post_oracle(actual: &Batch, query: &str, dtype: &DataType, parts: &[Part], sequence: u64) {
    let prefix = post_input(dtype, parts, sequence);
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([("events".into(), prefix)]),
            Some("post-oracle"),
        )
        .await
        .unwrap();
    assert_eq!(rows(actual), rows(&expected));
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(actual.metadata(), expected.metadata());
}

#[tokio::test]
async fn test_postaggregate_casts_release_input_and_restore_exact_prefixes() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in [GLOBAL_CAST, GROUPED_CAST, TRY_CAST] {
            let job = job();
            let context = StreamOperatorContext::new(&job, "float_extrema", None);
            let mut state = operator(&dtype, query);
            let mut history = Vec::new();
            let mut pools = Vec::new();
            let mut boundary = vec![Some(ONE); 8194];
            boundary[0] = Some((0x5a80_0000, 0x4350_0000_0000_0000));
            boundary[8192] = Some((0xda80_0000, 0xc350_0000_0000_0000));
            for (sequence, parts) in [
                vec![vec![Some(NEG_ZERO), None, Some(ONE)]],
                vec![boundary],
                vec![vec![Some(NAN_A), Some(SNAN), Some(POS_INF)]],
                vec![vec![]],
            ]
            .into_iter()
            .enumerate()
            {
                let incoming = post_input(&dtype, &parts, sequence as u64);
                let weak = weak_arrays(&incoming);
                let actual = process(&mut state, incoming, &context).await;
                history.extend(parts);
                post_oracle(&actual, query, &dtype, &history, sequence as u64).await;
                assert!(
                    state.incremental.is_some()
                        && state.compact.is_some()
                        && state.retained.is_none(),
                    "postaggregate casts must keep native cumulative state"
                );
                assert!(weak.iter().all(|array| array.upgrade().is_none()));
                if sequence % 2 == 0 {
                    state.prepare_compact_capture_async(&context).await.unwrap();
                }
                let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
                assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
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
async fn test_postaggregate_cast_errors_and_refusal_preserve_state_and_retry() {
    use datafusion::execution::memory_pool::MemoryConsumer;
    let dtype = DataType::Float64;
    let query = "SELECT CAST(SUM(value) AS SMALLINT) AS total FROM events";
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, query);
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
    let invalid = input(&dtype, &[vec![Some(POS_INF)]], 1);
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([(
                "events".into(),
                input(&dtype, &[vec![Some(ONE)], vec![Some(POS_INF)]], 1),
            )]),
            Some("cast-error-oracle"),
        )
        .await;
    assert!(expected.is_err());
    let weak = weak_arrays(&invalid);
    let mut collector = EdgeCollector::new(state.output_ports().to_vec());
    assert!(
        state
            .process_data("events", invalid, &context, &mut collector)
            .await
            .is_err()
    );
    assert!(collector.drain("output").is_empty());
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    assert_eq!(pool.reserved(), basis);
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    let next = input(&dtype, &[vec![Some(TWO)]], 1);
    let weak = weak_arrays(&next);
    assert!(
        matches!(state.process_data("events", next, &context, &mut Reject).await, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-extrema")
    );
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    assert_eq!(pool.reserved(), basis);
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    let pressure = MemoryConsumer::new("post-cast-pressure").register(&pool);
    pressure.try_grow((1 << 30) - basis - 1).unwrap();
    let held = pool.reserved();
    let next = input(&dtype, &[vec![Some(TWO)]], 1);
    let weak = weak_arrays(&next);
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
        query,
        &dtype,
        &[first[0].clone(), vec![Some(TWO)]],
        1,
    )
    .await;
    let accepted = state.checkpoint(Epoch::INITIAL).unwrap();
    let mut control: Value = serde_json::from_slice(accepted.segments["control"].bytes()).unwrap();
    control["identity"]["native_descriptor"]["projection"][0]["safe"] = json!(true);
    let segment = StateSegment::new(serde_json::to_vec(&control).unwrap());
    let mut corrupt = accepted.clone();
    corrupt
        .inline_metadata
        .insert("control_sha256".into(), json!(segment.sha256()));
    corrupt.segments.insert("control".into(), segment);
    assert!(StreamOperator::restore(&mut state, &corrupt).is_err());
    same_snapshot(&accepted, &state.checkpoint(Epoch::INITIAL).unwrap());
    drop((state, before, accepted, corrupt, actual));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_postaggregate_unproven_functions_keep_retained_fallback() {
    let dtype = DataType::Float64;
    let query = "SELECT SQRT(SUM(value)) AS root FROM events";
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, query);
    let mut parts = Vec::new();
    for (sequence, part) in [vec![Some(ONE)], vec![Some(TWO)]].into_iter().enumerate() {
        let actual = process(
            &mut state,
            input(&dtype, std::slice::from_ref(&part), sequence as u64),
            &context,
        )
        .await;
        parts.push(part);
        assert_oracle(&actual, query, &dtype, &parts, sequence as u64).await;
        assert!(state.incremental.is_none() && state.compact.is_none() && state.retained.is_some());
        assert_eq!(
            state.checkpoint(Epoch::INITIAL).unwrap().inline_metadata["state_layout"],
            json!(4)
        );
    }
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
