use super::*;
use datafusion::execution::memory_pool::MemoryConsumer;

const QUERIES: [&str; 6] = [
    "WITH totals AS (SELECT key, SUM(value) AS total, AVG(value) AS mean, COUNT(*) AS n FROM events GROUP BY key) SELECT key, total * 2.0 AS adjusted, mean FROM totals WHERE n > 0 ORDER BY key DESC LIMIT 2 OFFSET 1",
    "WITH totals AS (SELECT SUM(value) AS total, AVG(value) AS mean, COUNT(*) AS n FROM events) SELECT CAST(total AS REAL) AS total, mean + 1.0 AS mean, 7 AS marker FROM totals WHERE n >= 2",
    "WITH incoming AS (SELECT key, value FROM events) SELECT key, SUM(value) AS total, AVG(value) AS mean FROM incoming GROUP BY key",
    "WITH totals AS (SELECT key, SUM(value) AS total, AVG(value) AS mean, COUNT(*) AS n FROM events GROUP BY key), visible AS (SELECT key, total, mean, n FROM totals WHERE n >= 2) SELECT key, total, mean FROM visible ORDER BY key LIMIT 2",
    "WITH incoming AS (SELECT value AS price, key AS bucket FROM events WHERE key <> 2) SELECT bucket, SUM(price) AS total, AVG(price) AS mean FROM incoming GROUP BY bucket",
    "WITH incoming AS (SELECT key, value FROM events), quotes AS (SELECT value AS price, key AS bucket FROM incoming) SELECT CAST(SUM(price) AS REAL) AS total, AVG(price) AS mean FROM quotes",
];

#[tokio::test]
async fn test_single_input_cte_wrappers_use_incremental_state_and_exact_cold_prefixes() {
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
async fn test_cte_nonincremental_shapes_keep_exact_retained_execution() {
    for query in [
        "WITH totals AS (SELECT a.key, SUM(a.value) AS total FROM events a JOIN events b ON a.key = b.key GROUP BY a.key) SELECT key, total FROM totals",
        "WITH totals AS (SELECT key, SUM(value) AS total FROM events GROUP BY key) SELECT SUM(total) AS total FROM totals",
        "WITH numbered AS (SELECT key, value, ROW_NUMBER() OVER (ORDER BY key) AS n FROM events) SELECT key, SUM(value) AS total FROM numbered WHERE n <= 2 GROUP BY key",
        "WITH ordered AS (SELECT key, value FROM events ORDER BY key DESC LIMIT 2) SELECT SUM(value) AS total FROM ordered",
    ] {
        let dtype = DataType::Float64;
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut state = operator(&dtype, query);
        let mut history = Vec::new();
        for (sequence, part) in [vec![Some(ONE); 3], vec![Some(TWO); 3]]
            .into_iter()
            .enumerate()
        {
            let actual = process(
                &mut state,
                post_input(&dtype, std::slice::from_ref(&part), sequence as u64),
                &context,
            )
            .await;
            history.push(part);
            post_oracle(&actual, query, &dtype, &history, sequence as u64).await;
            assert!(state.incremental.is_none() && state.retained.is_some());
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
}

#[tokio::test]
async fn test_cte_refusal_refunds_and_retry_is_once() {
    for query in [QUERIES[0], QUERIES[1], QUERIES[2]] {
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
        let pressure = MemoryConsumer::new("cte-pressure").register(&pool);
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
