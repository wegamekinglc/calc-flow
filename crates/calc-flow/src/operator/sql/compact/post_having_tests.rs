use super::*;

const GROUPED_HAVING: &str = "SELECT key AS bucket, SUM(value) + 1.0 AS total FROM events GROUP BY key HAVING COUNT(value) >= 2 AND SUM(value) >= 0.0";
const GLOBAL_HAVING: &str =
    "SELECT SUM(value) + 1.0 AS total FROM events HAVING COUNT(value) >= 2 AND SUM(value) >= 0.0";
const GLOBAL_WHERE_HAVING: &str = "SELECT SUM(value) + 1.0 AS total FROM events WHERE key > 1 HAVING COUNT(value) >= 2 AND SUM(value) >= 0.0";

#[tokio::test]
async fn test_having_filters_full_snapshots_without_discarding_groups() {
    let negative_one = (0xbf80_0000, 0xbff0_0000_0000_0000);
    let negative_two = (0xc000_0000, 0xc000_0000_0000_0000);
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in [GLOBAL_HAVING, GROUPED_HAVING, GLOBAL_WHERE_HAVING] {
            let job = job();
            let context = StreamOperatorContext::new(&job, "float_extrema", None);
            let mut state = operator(&dtype, query);
            let mut history = Vec::new();
            let mut pools = Vec::new();
            for (sequence, part) in [
                vec![Some(negative_one); 3],
                vec![Some(TWO); 3],
                vec![Some(negative_two); 3],
                vec![Some(TWO); 3],
                vec![None; 3],
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
                    "HAVING must filter native full snapshots without retaining input"
                );
                let count = if query == GROUPED_HAVING { 3 } else { 1 };
                assert_eq!(
                    state
                        .incremental
                        .as_ref()
                        .unwrap()
                        .native_descriptor("having-groups")
                        .unwrap()
                        .group_count,
                    count
                );
                if sequence < 4 {
                    assert_eq!(actual.num_rows(), if sequence % 2 == 0 { 0 } else { count });
                }
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
async fn test_having_errors_and_refusals_preserve_state_and_retry() {
    use datafusion::execution::memory_pool::MemoryConsumer;
    let dtype = DataType::Float64;
    let query = "SELECT SUM(value) AS total FROM events HAVING 100 / (3 - COUNT(*)) > 0";
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
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([(
                "events".into(),
                input(&dtype, &[first[0].clone(), vec![Some(TWO), Some(TWO)]], 1),
            )]),
            Some("having-error-oracle"),
        )
        .await;
    assert!(expected.is_err());
    let invalid = input(&dtype, &[vec![Some(TWO), Some(TWO)]], 1);
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
    let pressure = MemoryConsumer::new("having-pressure").register(&pool);
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
    drop((state, before, actual));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_having_identity_missing_and_corrupt_reject_atomically() {
    let dtype = DataType::Float64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, GROUPED_HAVING);
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
    for remove in [false, true] {
        let mut control: Value =
            serde_json::from_slice(accepted.segments["control"].bytes()).unwrap();
        let descriptor = &mut control["identity"]["native_descriptor"];
        if remove {
            assert!(
                descriptor
                    .as_object_mut()
                    .unwrap()
                    .remove("post_filter")
                    .is_some()
            );
        } else {
            assert_eq!(descriptor["post_filter"]["operator"], json!("AND"));
            descriptor["post_filter"]["operator"] = json!("OR");
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
        GROUPED_HAVING,
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
async fn test_having_key_pushdown_preserves_float_record_boundaries() {
    let query = "SELECT key, SUM(value) AS total, AVG(value) AS mean, COUNT(*) AS n FROM events GROUP BY key HAVING key > 1 AND COUNT(*) >= 2";
    for dtype in [DataType::Float32, DataType::Float64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut state = operator(&dtype, query);
        let mut history = Vec::new();
        let mut pools = Vec::new();
        let mut boundary = vec![Some(ONE); 8194];
        boundary[0] = Some(SNAN);
        boundary[1] = Some((0x5a80_0000, 0x4350_0000_0000_0000));
        boundary[8191] = Some((0xda80_0000, 0xc350_0000_0000_0000));
        boundary[2] = Some(NAN_A);
        boundary[8192] = Some(POS_INF);
        for (sequence, part) in [vec![None; 3], boundary, vec![Some(TWO); 3], vec![]]
            .into_iter()
            .enumerate()
        {
            let incoming = post_input(&dtype, std::slice::from_ref(&part), sequence as u64);
            let weak = weak_arrays(&incoming);
            let actual = process(&mut state, incoming, &context).await;
            history.push(part);
            post_oracle(&actual, query, &dtype, &history, sequence as u64).await;
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
