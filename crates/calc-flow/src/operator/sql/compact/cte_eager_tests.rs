use super::*;

const QUERIES: [&str; 4] = [
    "WITH incoming AS (SELECT key, 100 / key AS price, value FROM events) SELECT key, SUM(price) AS total, AVG(value) FILTER (WHERE key > 1) AS mean FROM incoming GROUP BY key",
    "WITH incoming AS (SELECT key, CAST(key AS SMALLINT) AS bucket, value FROM events) SELECT bucket, SUM(value) AS total, AVG(value) AS mean FROM incoming GROUP BY bucket",
    "WITH incoming AS (SELECT key, CASE WHEN key > 1 THEN value / key ELSE value END AS price FROM events) SELECT SUM(price) FILTER (WHERE key > 1) AS total, AVG(price) AS mean FROM incoming",
    "WITH incoming AS (SELECT key, 100 / (key - 2) AS price, value FROM events WHERE key <> 2) SELECT SUM(price) FILTER (WHERE key > 1) AS total, AVG(value) AS mean FROM incoming",
];

#[tokio::test]
async fn test_cte_filtered_global_sum_ignores_unused_binary_after_cold_restore() {
    use datafusion::arrow::array::{BinaryArray, BooleanArray, Float64Array};

    let query = "WITH prepared AS (SELECT value, keep FROM events) SELECT SUM(value) AS total FROM prepared WHERE keep";
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let make = || SqlOperator::new("float_extrema", query, vec!["events".into()], vec![]).unwrap();
    let mut state = make();
    let mut history = Vec::new();
    for sequence in 0..2 {
        let record = RecordBatch::try_from_iter(vec![
            (
                "value",
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])) as ArrayRef,
            ),
            (
                "keep",
                Arc::new(BooleanArray::from(vec![true, false, true])) as ArrayRef,
            ),
            (
                "unused",
                Arc::new(BinaryArray::from(vec![b"unused".as_slice(); 3])) as ArrayRef,
            ),
        ])
        .unwrap();
        let metadata = BatchMetadata::new("cte-unused", sequence, JsonMap::new()).unwrap();
        let incoming = Batch::table(vec![record.clone()], metadata.clone()).unwrap();
        history.push(record);
        let actual = process(&mut state, incoming, &context).await;
        let expected = DataFusionRuntime::new(DataFusionConfig::default())
            .unwrap()
            .sql(
                query,
                &BTreeMap::from([(
                    "events".into(),
                    Batch::table(history.clone(), metadata).unwrap(),
                )]),
                None,
            )
            .await
            .unwrap();
        assert_eq!(rows(&actual), rows(&expected));
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(actual.metadata(), expected.metadata());
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        drop(state);
        state = make();
        StreamOperator::restore(&mut state, &snapshot).unwrap();
    }
    drop(state);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

#[tokio::test]
async fn test_eager_cte_projections_keep_native_state_and_exact_cold_prefixes() {
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
async fn test_eager_cte_pruning_preserves_default_count_evaluation() {
    for query in [
        "WITH incoming AS (SELECT key, 100 / (key - 2) AS price FROM events) SELECT COUNT(*) AS n FROM incoming",
        "WITH incoming AS (SELECT key, 100 / (key - 2) AS price FROM events) SELECT COUNT(price) AS n FROM incoming",
    ] {
        let dtype = DataType::Float64;
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut state = operator(&dtype, query);
        let mut history = Vec::new();
        for (sequence, part) in [vec![], vec![Some(ONE); 3], vec![Some(TWO); 3]]
            .into_iter()
            .enumerate()
        {
            history.push(part.clone());
            let expected = DataFusionRuntime::new(DataFusionConfig::default())
                .unwrap()
                .sql(
                    query,
                    &BTreeMap::from([(
                        "events".into(),
                        post_input(&dtype, &history, sequence as u64),
                    )]),
                    Some("cte-count-oracle"),
                )
                .await;
            let accepted = state.checkpoint(Epoch::INITIAL).unwrap();
            let mut collector = EdgeCollector::new(state.output_ports().to_vec());
            let actual = state
                .process_data(
                    "events",
                    post_input(&dtype, std::slice::from_ref(&part), sequence as u64),
                    &context,
                    &mut collector,
                )
                .await;
            assert_eq!(actual.is_err(), expected.is_err(), "{query}, {sequence}");
            if let Ok(expected) = expected {
                let output = collector.drain("output");
                assert_eq!(output.len(), 1);
                let actual = output[0].as_data().unwrap();
                assert_eq!(rows(actual), rows(&expected));
                assert_eq!(
                    actual.table_payload().unwrap().schema(),
                    expected.table_payload().unwrap().schema()
                );
                assert_eq!(actual.metadata(), expected.metadata());
            } else {
                assert!(collector.drain("output").is_empty());
                same_snapshot(&accepted, &state.checkpoint(Epoch::INITIAL).unwrap());
                history.pop();
            }
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
async fn test_eager_input_identity_missing_or_corrupt_is_atomic() {
    let dtype = DataType::Float64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(&dtype, QUERIES[0]);
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
        assert_eq!(descriptor["input_checks"][0]["kind"], json!("binary"));
        match mutation {
            0 => descriptor["input_checks"][0]["operator"] = json!("+"),
            1 => descriptor["input_checks"] = Value::Null,
            2 => descriptor["input_checks"][0]["left"]["index"] = json!(99),
            _ => {
                assert!(
                    descriptor
                        .as_object_mut()
                        .unwrap()
                        .remove("input_checks")
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
        QUERIES[0],
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
