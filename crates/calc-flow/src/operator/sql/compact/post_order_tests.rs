use super::*;

#[tokio::test]
async fn test_order_limit_offsets_full_snapshots_without_discarding_groups() {
    let queries = [
        "SELECT key, SUM(value) AS total, COUNT(value) AS valid FROM events GROUP BY key ORDER BY total DESC NULLS LAST, key ASC",
        "SELECT key, SUM(value) AS total, COUNT(value) AS valid FROM events GROUP BY key ORDER BY total DESC NULLS LAST, key ASC LIMIT 2 OFFSET 1",
        "SELECT key, COUNT(*) AS n FROM events GROUP BY key LIMIT 2",
        "SELECT key, SUM(value) AS total FROM events GROUP BY key HAVING COUNT(value) >= 2 ORDER BY total ASC NULLS FIRST, key DESC LIMIT 1 OFFSET 1",
        "SELECT key, COUNT(*) AS n FROM events GROUP BY key OFFSET 1",
    ];
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in queries {
            let job = job();
            let context = StreamOperatorContext::new(&job, "float_extrema", None);
            let mut state = operator(&dtype, query);
            let mut history = Vec::new();
            let mut pools = Vec::new();
            for (sequence, part) in [
                vec![None; 3],
                vec![Some(ONE), Some(TWO), Some(NEG_ZERO)],
                vec![Some(TWO), Some(NEG_ZERO), Some(POS_INF)],
                vec![Some(NAN_A), Some(SNAN), Some(NEG_ZERO)],
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
                    "ORDER/LIMIT must use native full snapshots without retaining input"
                );
                assert_eq!(
                    state
                        .incremental
                        .as_ref()
                        .unwrap()
                        .native_descriptor("ordered-groups")
                        .unwrap()
                        .group_count,
                    3
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
async fn test_order_expression_error_and_rejected_output_preserve_state_and_retry() {
    let dtype = DataType::Float64;
    let query = "SELECT key, COUNT(*) AS n FROM events GROUP BY key ORDER BY 100 / (n - 3), key";
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
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    let invalid = post_input(&dtype, &[vec![Some(TWO); 6]], 1);
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([(
                "events".into(),
                post_input(&dtype, &[vec![Some(ONE); 3], vec![Some(TWO); 6]], 1),
            )]),
            Some("order-error-oracle"),
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
    let next = post_input(&dtype, &[vec![Some(TWO); 3]], 1);
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
    drop((state, before, actual));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_order_limit_identity_missing_and_corrupt_reject_atomically() {
    let dtype = DataType::Float64;
    let query = "SELECT key, SUM(value) AS total FROM events GROUP BY key ORDER BY total DESC NULLS LAST, key LIMIT 2 OFFSET 1";
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
    for mutation in 0..4 {
        let mut control: Value =
            serde_json::from_slice(accepted.segments["control"].bytes()).unwrap();
        let descriptor = &mut control["identity"]["native_descriptor"];
        match mutation {
            0 => descriptor["post_order"]["skip"] = json!(0),
            1 => descriptor["post_order"]["fetch"] = json!(1),
            2 => descriptor["post_order"]["keys"][0]["nulls_first"] = json!(true),
            _ => {
                assert!(
                    descriptor
                        .as_object_mut()
                        .unwrap()
                        .remove("post_order")
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

#[tokio::test]
async fn test_order_limit_projection_errors_match_dataframe_execution() {
    let dtype = DataType::Float64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    for query in [
        "SELECT key, 100 / (COUNT(*) - 1) AS ratio FROM events GROUP BY key ORDER BY key LIMIT 1",
        "SELECT key, 100 / (COUNT(*) - 1) AS ratio FROM events GROUP BY key LIMIT 1",
        "SELECT key, COUNT(*) AS n, 1 / 0 AS constant FROM events GROUP BY key LIMIT 0",
        "SELECT key, COUNT(*) AS n, 1 / 0 AS constant FROM events GROUP BY key LIMIT 1",
        "SELECT key, COUNT(*) AS n FROM events GROUP BY key LIMIT 0",
    ] {
        let parts = vec![vec![Some(ONE); 4]];
        let mut state = operator(&dtype, query);
        let expected = DataFusionRuntime::new(DataFusionConfig::default())
            .unwrap()
            .sql(
                query,
                &BTreeMap::from([("events".into(), post_input(&dtype, &parts, 0))]),
                Some("limit-projection-oracle"),
            )
            .await;
        let mut collector = EdgeCollector::new(state.output_ports().to_vec());
        let actual = state
            .process_data(
                "events",
                post_input(&dtype, &parts, 0),
                &context,
                &mut collector,
            )
            .await;
        assert_eq!(actual.is_err(), expected.is_err(), "{query}");
        if let Ok(expected) = expected {
            let outputs = collector.drain("output");
            assert_eq!(outputs.len(), 1);
            let actual = outputs[0].as_data().unwrap();
            assert_eq!(rows(actual), rows(&expected));
            assert_eq!(
                actual.table_payload().unwrap().schema(),
                expected.table_payload().unwrap().schema()
            );
            assert_eq!(actual.metadata(), expected.metadata());
        } else {
            assert!(collector.drain("output").is_empty());
        }
        drop(state);
    }
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}
