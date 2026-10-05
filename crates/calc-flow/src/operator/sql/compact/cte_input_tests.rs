use super::*;
use datafusion::execution::memory_pool::MemoryConsumer;

const QUERIES: [&str; 4] = [
    "WITH incoming AS (SELECT key, value * 2.0 AS price FROM events) SELECT key, SUM(price) AS total, AVG(price) AS mean FROM incoming GROUP BY key",
    "WITH incoming AS (SELECT key, value * 2.0 AS price, value + 1.0 AS shifted FROM events WHERE key <> 2) SELECT key, SUM(price) AS total, AVG(shifted) FILTER (WHERE key > 1) AS mean FROM incoming GROUP BY key ORDER BY key",
    "WITH incoming AS (SELECT key, value * 2.0 AS price FROM events), quotes AS (SELECT key, price + 1.0 AS adjusted FROM incoming) SELECT SUM(adjusted) AS total, AVG(adjusted) AS mean FROM quotes",
    "WITH incoming AS (SELECT TRY_CAST(value * 2.0 AS SMALLINT) AS bucket, -CAST(value AS REAL) AS price FROM events) SELECT bucket, SUM(price) AS total, AVG(price) AS mean FROM incoming GROUP BY bucket",
];

#[tokio::test]
async fn test_throwing_cte_projection_precedes_aggregate_filter_and_is_atomic() {
    let dtype = DataType::Float64;
    let query = "WITH incoming AS (SELECT key, 100 / (key - 2) AS price FROM events) SELECT SUM(price) FILTER (WHERE key <> 2) AS total FROM incoming";
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, query);
    drop(
        process(
            &mut state,
            post_input(&dtype, &[vec![Some(ONE)]], 0),
            &context,
        )
        .await,
    );
    assert!(state.incremental.is_some() && state.retained.is_none());
    let accepted = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    let prefix = post_input(&dtype, &[vec![Some(ONE)], vec![Some(TWO); 3]], 1);
    assert!(
        DataFusionRuntime::new(DataFusionConfig::default())
            .unwrap()
            .sql(
                query,
                &BTreeMap::from([("events".into(), prefix)]),
                Some("cte-error-oracle")
            )
            .await
            .is_err()
    );
    let incoming = post_input(&dtype, &[vec![Some(TWO); 3]], 1);
    let weak = weak_arrays(&incoming);
    let mut collector = EdgeCollector::new(state.output_ports().to_vec());
    assert!(
        state
            .process_data("events", incoming, &context, &mut collector)
            .await
            .is_err()
    );
    assert!(collector.drain("output").is_empty());
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    assert_eq!(pool.reserved(), basis);
    same_snapshot(&accepted, &state.checkpoint(Epoch::INITIAL).unwrap());
    let actual = process(
        &mut state,
        post_input(&dtype, &[vec![Some(TWO)]], 1),
        &context,
    )
    .await;
    post_oracle(
        &actual,
        query,
        &dtype,
        &[vec![Some(ONE)], vec![Some(TWO)]],
        1,
    )
    .await;
    drop((state, accepted, actual));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_cte_float_input_expressions_keep_native_state_and_exact_cold_prefixes() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in QUERIES {
            let job = job();
            let context = StreamOperatorContext::new(&job, "float_extrema", None);
            let mut state = operator(&dtype, query);
            let mut history = Vec::new();
            let mut pools = Vec::new();
            let mut boundary = vec![Some(ONE); 8194];
            boundary[0] = Some((0x5a80_0000, 0x4350_0000_0000_0000));
            boundary[8192] = Some((0xda80_0000, 0xc350_0000_0000_0000));
            for (sequence, part) in [
                vec![None; 3],
                vec![Some(ONE), Some(TWO), Some(NEG_ZERO)],
                boundary,
                vec![Some(NAN_A), Some(SNAN), Some(POS_INF)],
                vec![],
            ]
            .into_iter()
            .enumerate()
            {
                let incoming = post_input(&dtype, std::slice::from_ref(&part), sequence as u64);
                let weak = weak_arrays(&incoming);
                let actual = process(&mut state, incoming, &context).await;
                history.push(part);
                post_oracle(&actual, query, &dtype, &history, sequence as u64).await;
                assert!(
                    state.incremental.is_some()
                        && state.compact.is_some()
                        && state.retained.is_none(),
                    "single-input CTE must use native cumulative state: {query}, sequence {sequence}"
                );
                assert!(weak.iter().all(|array| array.upgrade().is_none()));
                if sequence % 2 == 0 {
                    state.prepare_compact_capture_async(&context).await.unwrap();
                }
                let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
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
async fn test_cte_input_expression_refusal_refunds_and_retry_is_once() {
    for query in [
        QUERIES[0],
        QUERIES[1],
        QUERIES[2],
        "WITH incoming AS (SELECT key, 100 / key AS price FROM events) SELECT key, SUM(price) AS total FROM incoming GROUP BY key",
    ] {
        let dtype = DataType::Float64;
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut state = operator(&dtype, query);
        drop(
            process(
                &mut state,
                post_input(&dtype, &[vec![Some(ONE); 3]], 0),
                &context,
            )
            .await,
        );
        assert!(state.incremental.is_some());
        let accepted = state.checkpoint(Epoch::INITIAL).unwrap();
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let basis = pool.reserved();
        let next = post_input(&dtype, &[vec![Some(TWO); 3]], 1);
        let weak = weak_arrays(&next);
        assert!(matches!(
            state.process_data("events", next, &context, &mut Reject).await,
            Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-extrema"
        ));
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        assert_eq!(pool.reserved(), basis);
        same_snapshot(&accepted, &state.checkpoint(Epoch::INITIAL).unwrap());
        let pressure = MemoryConsumer::new("cte-input-pressure").register(&pool);
        pressure.try_grow((1 << 30) - basis - 1).unwrap();
        let held = pool.reserved();
        let next = post_input(&dtype, &[vec![Some(TWO); 3]], 1);
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
        same_snapshot(&accepted, &state.checkpoint(Epoch::INITIAL).unwrap());
        let actual = process(
            &mut state,
            post_input(&dtype, &[vec![Some(TWO); 3]], 1),
            &context,
        )
        .await;
        post_oracle(
            &actual,
            query,
            &dtype,
            &[vec![Some(ONE); 3], vec![Some(TWO); 3]],
            1,
        )
        .await;
        drop((state, accepted, actual));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}
