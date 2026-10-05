use super::*;
use datafusion::execution::memory_pool::MemoryConsumer;

const QUERIES: [&str; 6] = [
    "SELECT CAST(1 AS SMALLINT) AS bucket, SUM(value) AS total, AVG(value) AS mean, COUNT(*) AS n FROM events GROUP BY CAST(1 AS SMALLINT)",
    "SELECT 2 + 3 AS bucket, SUM(value * 2.0) AS total, AVG(value + 1.0) AS mean FROM events GROUP BY 2 + 3 ORDER BY bucket LIMIT 1",
    "SELECT CAST(1 AS SMALLINT) AS bucket, SUM(value) FILTER (WHERE key > 1) AS total, AVG(value) FILTER (WHERE key % 2 = 0) AS mean, COUNT(*) AS n FROM events WHERE key <> 3 GROUP BY CAST(1 AS SMALLINT) HAVING COUNT(*) >= 2",
    "SELECT key, CAST(1 AS SMALLINT) AS bucket, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key, CAST(1 AS SMALLINT)",
    "SELECT key, key + 1 AS successor, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key, key + 1",
    "SELECT CAST(1 AS SMALLINT) AS bucket, COUNT(*) AS n FROM events GROUP BY CAST(1 AS SMALLINT)",
];

#[tokio::test]
async fn test_constant_grouping_uses_default_record_state_and_exact_cold_prefixes() {
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
                    "constant grouping must use native cumulative state: {query}, sequence {sequence}"
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
async fn test_constant_grouping_activates_after_empty_cold_prefixes() {
    for query in [QUERIES[0], QUERIES[2]] {
        let dtype = DataType::Float64;
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut state = operator(&dtype, query);
        let mut history = Vec::new();
        let mut pools = Vec::new();
        for (sequence, part) in [vec![], vec![], vec![Some(ONE); 3], vec![]]
            .into_iter()
            .enumerate()
        {
            let incoming = post_input(&dtype, std::slice::from_ref(&part), sequence as u64);
            let weak = weak_arrays(&incoming);
            let actual = process(&mut state, incoming, &context).await;
            history.push(part);
            post_oracle(&actual, query, &dtype, &history, sequence as u64).await;
            if sequence >= 2 {
                assert!(state.incremental.is_some() && state.retained.is_none());
                assert!(weak.iter().all(|array| array.upgrade().is_none()));
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

#[tokio::test]
async fn test_constant_grouping_refusal_refunds_and_retry_is_once() {
    for query in [QUERIES[0], QUERIES[2], QUERIES[4]] {
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
        let pressure = MemoryConsumer::new("constant-group-pressure").register(&pool);
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

#[tokio::test]
async fn test_constant_grouping_projection_identity_is_strict_and_atomic() {
    let dtype = DataType::Float64;
    let query = QUERIES[0];
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
    let accepted = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    for mutation in 0..2 {
        let mut control: Value =
            serde_json::from_slice(accepted.segments["control"].bytes()).unwrap();
        let descriptor = &mut control["identity"]["native_descriptor"];
        assert_eq!(descriptor["key_inputs"], json!([]));
        match mutation {
            0 => descriptor["projection"][0] = Value::Null,
            _ => descriptor["key_inputs"] = json!([{"kind": "column", "index": 0}]),
        }
        let segment = StateSegment::new(serde_json::to_vec(&control).unwrap());
        let mut corrupt = accepted.clone();
        corrupt
            .inline_metadata
            .insert("control_sha256".into(), json!(segment.sha256()));
        corrupt.segments.insert("control".into(), segment);
        assert!(StreamOperator::restore(&mut state, &corrupt).is_err());
        assert_eq!(pool.reserved(), basis);
        same_snapshot(&accepted, &state.checkpoint(Epoch::INITIAL).unwrap());
    }
    drop((state, accepted));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
