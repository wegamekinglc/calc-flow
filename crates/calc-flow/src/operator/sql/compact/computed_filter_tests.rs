use super::*;

const GLOBAL: &str = "SELECT SUM(value) FILTER (WHERE key > 1) AS total, AVG(value) FILTER (WHERE key % 2 = 0) AS mean, COUNT(*) FILTER (WHERE value IS NULL OR value >= 0) AS selected FROM events";
const GROUPED: &str = "SELECT key, SUM(value) FILTER (WHERE key > 1 AND value IS NOT NULL) AS total, AVG(value) FILTER (WHERE key % 2 = 0) AS mean, COUNT(value) FILTER (WHERE NOT (value < 0)) AS selected FROM events GROUP BY key";
const CASE_FILTER: &str = "SELECT key, SUM(value * 2.0) FILTER (WHERE CASE WHEN key = 1 THEN FALSE ELSE value IS NOT NULL END) AS total, COUNT(*) FILTER (WHERE key >= 2) AS selected FROM events GROUP BY key";

#[tokio::test]
async fn test_computed_aggregate_filters_keep_native_state_and_exact_cold_prefixes() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in [
            GLOBAL,
            GROUPED,
            CASE_FILTER,
            "SELECT SUM(value) FILTER (WHERE key % 2 = 0) AS total, AVG(value + 1.0) FILTER (WHERE key > 1) AS mean FROM events WHERE key <> 3",
            "SELECT SUM(CAST(key * 2 AS DECIMAL(20, 0))) FILTER (WHERE key > 1) AS total, COUNT(*) FILTER (WHERE key % 2 = 0) AS selected FROM events",
            "SELECT key, COUNT(*) FILTER (WHERE value IS NULL OR key > 1) AS selected FROM events GROUP BY key",
        ] {
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
                    "computed aggregate FILTER must use native cumulative state"
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
async fn test_computed_filter_error_order_matches_dataframe_execution() {
    let dtype = DataType::Float64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    for query in [
        "SELECT SUM(CAST(100 / (key - 2) AS DOUBLE)) FILTER (WHERE key <> 2) AS total FROM events",
        "SELECT SUM(CAST(100 / (key - 2) AS DECIMAL(20, 0))) FILTER (WHERE key <> 2) AS total FROM events",
        "SELECT key, SUM(CAST(100 / (key - 2) AS DOUBLE)) FILTER (WHERE key <> 2) AS total FROM events GROUP BY key",
        "SELECT key, SUM(value) FILTER (WHERE 100 / (key - 2) > 0) AS total FROM events WHERE key <> 2 GROUP BY key",
        "SELECT SUM(value) FILTER (WHERE 100 / (key - 2) > 0) AS total FROM events",
        "SELECT SUM(value) FILTER (WHERE CASE WHEN key = 2 THEN FALSE ELSE 100 / (key - 2) > 0 END) AS total FROM events",
    ] {
        let incoming = post_input(&dtype, &[vec![Some(ONE); 2]], 0);
        let expected = DataFusionRuntime::new(DataFusionConfig::default())
            .unwrap()
            .sql(
                query,
                &BTreeMap::from([("events".into(), incoming.clone())]),
                Some("computed-filter-oracle"),
            )
            .await;
        let mut state = operator(&dtype, query);
        let mut collector = EdgeCollector::new(state.output_ports().to_vec());
        let actual = state
            .process_data("events", incoming, &context, &mut collector)
            .await;
        assert_eq!(actual.is_err(), expected.is_err(), "{query}");
        if let Ok(expected) = expected {
            let outputs = collector.drain("output");
            assert_eq!(outputs.len(), 1);
            let actual = outputs[0].as_data().unwrap();
            assert_eq!(rows(actual), rows(&expected), "{query}");
            assert_eq!(
                actual.table_payload().unwrap().schema(),
                expected.table_payload().unwrap().schema()
            );
            assert_eq!(actual.metadata(), expected.metadata());
            assert!(state.incremental.is_some(), "{query}");
        } else {
            assert!(collector.drain("output").is_empty());
        }
        drop(state);
    }
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

#[tokio::test]
async fn test_computed_filter_failure_refunds_and_retry_is_once() {
    let dtype = DataType::Float64;
    let query =
        "SELECT SUM(value) FILTER (WHERE 100 / (key - 2) > 0) AS total, COUNT(*) AS n FROM events";
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
    assert!(state.incremental.is_some());
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    let invalid = post_input(&dtype, &[vec![Some(TWO); 2]], 1);
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
    let next = post_input(&dtype, &[vec![Some(TWO)]], 1);
    let weak = weak_arrays(&next);
    assert!(
        matches!(state.process_data("events", next, &context, &mut Reject).await,
        Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-extrema")
    );
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    assert_eq!(pool.reserved(), basis);
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
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
    drop((state, before, actual));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_computed_filter_identity_missing_or_corrupt_is_atomic() {
    let dtype = DataType::Float64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, GLOBAL);
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
    for mutation in 0..4 {
        let mut control: Value =
            serde_json::from_slice(accepted.segments["control"].bytes()).unwrap();
        let aggregate = &mut control["identity"]["native_descriptor"]["aggregates"][0];
        assert_eq!(aggregate["filter"]["kind"], json!("binary"));
        match mutation {
            0 => aggregate["filter"]["operator"] = json!("<"),
            1 => aggregate["filter"] = Value::Null,
            2 => aggregate["filter"]["left"]["index"] = json!(99),
            _ => {
                assert!(
                    aggregate
                        .as_object_mut()
                        .unwrap()
                        .remove("filter")
                        .is_some()
                );
            }
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
    let actual = process(
        &mut state,
        post_input(&dtype, &[vec![Some(TWO); 3]], 1),
        &context,
    )
    .await;
    post_oracle(
        &actual,
        GLOBAL,
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

#[tokio::test]
async fn test_computed_filter_large_record_uses_bounded_workspace() {
    use datafusion::execution::memory_pool::MemoryConsumer;
    let dtype = DataType::Float64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, GLOBAL);
    let first = vec![Some(ONE); 3];
    drop(
        process(
            &mut state,
            post_input(&dtype, std::slice::from_ref(&first), 0),
            &context,
        )
        .await,
    );
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let pressure = MemoryConsumer::new("computed-filter-pressure").register(&pool);
    pressure
        .try_grow((1 << 30) - pool.reserved() - (16 << 20))
        .unwrap();
    let next = vec![Some(ONE); 300_000];
    let incoming = post_input(&dtype, std::slice::from_ref(&next), 1);
    let weak = weak_arrays(&incoming);
    let actual = process(&mut state, incoming, &context).await;
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    drop(pressure);
    post_oracle(&actual, GLOBAL, &dtype, &[first, next], 1).await;
    drop((state, actual));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_unproven_aggregate_filter_keeps_retained_fallback() {
    let dtype = DataType::Float64;
    let query =
        "SELECT key, SUM(value) FILTER (WHERE ABS(key) > 1) AS total FROM events GROUP BY key";
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, query);
    let part = vec![Some(ONE), None, Some(TWO)];
    let actual = process(
        &mut state,
        post_input(&dtype, std::slice::from_ref(&part), 0),
        &context,
    )
    .await;
    post_oracle(&actual, query, &dtype, &[part], 0).await;
    assert!(state.incremental.is_none() && state.retained.is_some());
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((state, actual));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
