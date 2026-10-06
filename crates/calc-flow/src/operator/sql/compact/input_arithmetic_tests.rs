use super::*;

const GLOBAL_INPUT: &str = "SELECT SUM(value * 2.0) AS total, AVG(value + 1.0) AS mean, COUNT(value - 1.0) AS valid FROM events";
const GROUPED_INPUT: &str = "SELECT key, SUM(value * CAST(key AS DOUBLE)) AS weighted, AVG(value + 1.0) AS mean, COUNT(value / 2.0) AS valid FROM events GROUP BY key";
const WHERE_INPUT: &str = "SELECT SUM(value * CAST(key AS DOUBLE)) AS weighted, AVG(value - 1.0) AS mean, COUNT(value + 1.0) AS valid FROM events WHERE key > 1";
const CASE_INPUT: &str = "SELECT key, SUM(CASE WHEN value IS NULL THEN NULL ELSE value * 2.0 END) AS total, AVG(CASE key WHEN 1 THEN value ELSE value + 1.0 END) AS mean FROM events GROUP BY key";

#[tokio::test]
async fn test_numeric_input_expressions_keep_native_state_and_exact_cold_prefixes() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in [GLOBAL_INPUT, GROUPED_INPUT, WHERE_INPUT, CASE_INPUT] {
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
                    "numeric aggregate inputs must process only new records"
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
async fn test_input_arithmetic_errors_and_refusal_preserve_state_and_retry() {
    let dtype = DataType::Float64;
    let query = "SELECT SUM(100 / (key - 2)) AS total, COUNT(value) AS valid FROM events";
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
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([(
                "events".into(),
                post_input(&dtype, &[vec![Some(ONE)], vec![Some(TWO); 2]], 1),
            )]),
            Some("input-error-oracle"),
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
async fn test_input_binary_identity_corruption_rejects_atomically() {
    let dtype = DataType::Float64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, GLOBAL_INPUT);
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
    for mutation in 0..3 {
        let mut control: Value =
            serde_json::from_slice(accepted.segments["control"].bytes()).unwrap();
        let input = &mut control["identity"]["native_descriptor"]["aggregates"][0]["inputs"][0];
        assert_eq!(input["kind"], json!("binary"));
        match mutation {
            0 => input["operator"] = json!("+"),
            1 => input["fail_on_overflow"] = json!(true),
            _ => {
                assert!(input.as_object_mut().unwrap().remove("right").is_some());
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
        GLOBAL_INPUT,
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
async fn test_input_expression_filter_error_order_matches_dataframe_execution() {
    use datafusion::arrow::array::BooleanArray;
    let dtype = DataType::Boolean;
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    for query in [
        "SELECT SUM(CAST(100 / (key - 2) AS DOUBLE)) FILTER (WHERE value) AS total, COUNT(*) AS n FROM events",
        "SELECT key, SUM(CAST(100 / (key - 2) AS DOUBLE)) AS total FROM events WHERE key <> 2 GROUP BY key",
        "SELECT SUM(CAST(100 / (key - 2) AS DECIMAL(20, 0))) FILTER (WHERE value) AS total FROM events",
        "SELECT key, SUM(CAST(100 / (key - 2) AS DOUBLE)) FILTER (WHERE value) AS total FROM events GROUP BY key",
        "SELECT SUM(CAST(100 / (key - 2) AS DECIMAL(20, 0))) AS total FROM events WHERE key <> 2",
    ] {
        let record = RecordBatch::try_new(
            schema(&dtype),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(BooleanArray::from(vec![true, false])),
            ],
        )
        .unwrap();
        let incoming = Batch::table(vec![record], metadata(0)).unwrap();
        let expected = DataFusionRuntime::new(DataFusionConfig::default())
            .unwrap()
            .sql(
                query,
                &BTreeMap::from([("events".into(), incoming.clone())]),
                Some("filter-order-oracle"),
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
            assert_eq!(rows(actual), rows(&expected));
            assert_eq!(
                actual.table_payload().unwrap().schema(),
                expected.table_payload().unwrap().schema()
            );
            assert_eq!(actual.metadata(), expected.metadata());
            assert!(state.incremental.is_some());
        } else {
            assert!(collector.drain("output").is_empty());
        }
        drop(state);
    }
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}
