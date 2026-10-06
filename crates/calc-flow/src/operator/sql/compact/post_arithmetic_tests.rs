use super::*;

const GLOBAL_MATH: &str = "SELECT SUM(value) + 1.0 AS shifted, AVG(value) * 2.0 AS weighted, MAX(value) - MIN(value) AS spread, COUNT(*) / 2 AS pairs, COUNT(*) % 3 AS residue, -SUM(value) AS negated, NOT (COUNT(value) = 0) AS present, SUM(value) IS NULL AS missing FROM events";
const GROUPED_MATH: &str = "SELECT key + 10 AS bucket, (SUM(value) - MIN(value)) * 2.0 AS adjusted, CAST(COUNT(*) AS DECIMAL(20, 0)) + CAST(1 AS DECIMAL(20, 0)) AS shifted_rows, COUNT(value) > 0 AND COUNT(*) > 0 AS present FROM events GROUP BY key";

#[tokio::test]
async fn test_postaggregate_arithmetic_keeps_native_state_and_exact_cold_prefixes() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in [GLOBAL_MATH, GROUPED_MATH] {
            let job = job();
            let context = StreamOperatorContext::new(&job, "float_extrema", None);
            let mut state = operator(&dtype, query);
            let mut history = Vec::new();
            let mut pools = Vec::new();
            let mut boundary = vec![Some(ONE); 8194];
            boundary[0] = Some((0x5a80_0000, 0x4350_0000_0000_0000));
            boundary[8192] = Some((0xda80_0000, 0xc350_0000_0000_0000));
            for (sequence, parts) in [
                vec![vec![None; 3]],
                vec![vec![Some(NEG_ZERO), Some(ONE), Some(TWO)]],
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
                    "postaggregate arithmetic must keep native cumulative state"
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
async fn test_postaggregate_division_errors_and_refusal_preserve_state_and_retry() {
    use datafusion::execution::memory_pool::MemoryConsumer;
    let dtype = DataType::Float64;
    let query = "SELECT 100 / (3 - COUNT(*)) AS remaining, SUM(value) AS total FROM events";
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
    let invalid = input(&dtype, &[vec![Some(TWO), Some(TWO)]], 1);
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([(
                "events".into(),
                input(&dtype, &[first[0].clone(), vec![Some(TWO), Some(TWO)]], 1),
            )]),
            Some("division-error-oracle"),
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
    let pressure = MemoryConsumer::new("post-arithmetic-pressure").register(&pool);
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
async fn test_postaggregate_binary_identity_corruption_rejects_atomically() {
    let dtype = DataType::Float64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, GLOBAL_MATH);
    drop(process(&mut state, input(&dtype, &[vec![Some(ONE)]], 0), &context).await);
    let accepted = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    for (field, value) in [("operator", json!("-")), ("fail_on_overflow", json!(true))] {
        let mut control: Value =
            serde_json::from_slice(accepted.segments["control"].bytes()).unwrap();
        let binary = &mut control["identity"]["native_descriptor"]["projection"][0];
        assert_eq!(binary["kind"], json!("binary"));
        assert_ne!(binary[field], value);
        binary[field] = value;
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
    let actual = process(&mut state, input(&dtype, &[vec![Some(TWO)]], 1), &context).await;
    assert_oracle(
        &actual,
        GLOBAL_MATH,
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
