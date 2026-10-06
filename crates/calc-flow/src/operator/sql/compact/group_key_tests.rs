use super::*;

const QUERIES: [&str; 7] = [
    "SELECT TRY_CAST(value AS SMALLINT) AS bucket, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY TRY_CAST(value AS SMALLINT)",
    "SELECT CAST(1 AS SMALLINT) AS bucket, COUNT(value) AS valid, COUNT(*) AS n FROM events GROUP BY CAST(1 AS SMALLINT)",
    "SELECT key % 2 AS bucket, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key % 2",
    "SELECT key % 2 AS bucket, value IS NULL AS missing, SUM(value) AS total, COUNT(*) AS n FROM events GROUP BY key % 2, value IS NULL",
    "SELECT CAST(key AS SMALLINT) AS bucket, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY CAST(key AS SMALLINT)",
    "SELECT CASE WHEN key > 1 THEN 2 ELSE 1 END AS bucket, SUM(value) AS total, COUNT(*) AS n FROM events GROUP BY CASE WHEN key > 1 THEN 2 ELSE 1 END",
    "SELECT key % 2 AS bucket, SUM(value) AS total, AVG(value) FILTER (WHERE key > 1) AS mean FROM events WHERE key <> 2 GROUP BY key % 2",
];

#[tokio::test]
async fn test_group_key_expressions_keep_native_state_and_exact_cold_prefixes() {
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
                    "computed group keys must use native cumulative state: {query}, sequence {sequence}"
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
async fn test_group_key_failure_refunds_and_retry_is_once() {
    let dtype = DataType::Float64;
    let query = "SELECT 100 / (key - 2) AS bucket, SUM(value) AS total, COUNT(*) AS n FROM events GROUP BY 100 / (key - 2)";
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
async fn test_group_key_identity_missing_or_corrupt_is_atomic() {
    let dtype = DataType::Float64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, QUERIES[2]);
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
        let descriptor = &mut control["identity"]["native_descriptor"];
        assert_eq!(descriptor["key_inputs"][0]["kind"], json!("binary"));
        match mutation {
            0 => descriptor["key_inputs"][0]["operator"] = json!("+"),
            1 => descriptor["key_inputs"] = Value::Null,
            2 => descriptor["key_inputs"][0]["left"]["index"] = json!(99),
            _ => {
                assert!(
                    descriptor
                        .as_object_mut()
                        .unwrap()
                        .remove("key_inputs")
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
        QUERIES[2],
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
async fn test_unproven_group_key_keeps_retained_fallback() {
    let dtype = DataType::Float64;
    let query = "SELECT ABS(key) AS bucket, SUM(value) AS total FROM events GROUP BY ABS(key)";
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
