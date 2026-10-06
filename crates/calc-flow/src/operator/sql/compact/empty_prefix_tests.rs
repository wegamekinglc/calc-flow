use super::*;

const GROUPED: &str = "SELECT key, MIN(value) AS lo, MAX(value) AS hi, COUNT(value) AS valid, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key";
const GLOBAL: &str = "SELECT MIN(value) AS lo, MAX(value) AS hi, COUNT(value) AS valid, SUM(value) AS total, AVG(value) AS mean FROM events";

async fn oracle(actual: &Batch, query: &str, dtype: &DataType, prefix: &[Part], sequence: u64) {
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([("events".into(), input(dtype, prefix, sequence))]),
            Some("empty-prefix-oracle"),
        )
        .await
        .unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(actual.metadata(), expected.metadata());
    if query == GROUPED {
        assert_eq!(rows(actual), rows(&expected));
    } else {
        let cells = |batch: &Batch| {
            batch
                .table_payload()
                .unwrap()
                .batches()
                .iter()
                .flat_map(|record| {
                    (0..record.num_rows()).map(|row| {
                        record
                            .columns()
                            .iter()
                            .map(|array| cell(array, row))
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(cells(actual), cells(&expected));
    }
}

async fn check(query: &str, dtype: &DataType, cold: bool) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "grouped_float", None);
    let mut state = operator_query(dtype, query);
    let mut prefix = Vec::new();
    let mut pools = Vec::new();
    let mut batches = vec![vec![vec![], vec![]], vec![vec![]]];
    batches.extend(arrivals(false));
    for (sequence, parts) in batches.into_iter().enumerate() {
        let batch = input(dtype, &parts, sequence as u64);
        let weak = weak_arrays(&batch);
        let actual = process(&mut state, batch, &context).await;
        prefix.extend(parts);
        oracle(&actual, query, dtype, &prefix, sequence as u64).await;
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        if sequence >= 2 {
            assert!(
                state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
                "a nonempty batch after an empty prefix must activate native aggregates"
            );
            assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
            assert!(weak.iter().all(|array| array.upgrade().is_none()));
        } else {
            assert_eq!(snapshot.inline_metadata["state_layout"], json!(4));
        }
        if cold {
            pools.push(
                state
                    .stream_state
                    .runtime()
                    .unwrap()
                    .incremental_memory_pool(),
            );
            drop(state);
            state = operator_query(dtype, query);
            StreamOperator::restore(&mut state, &snapshot).unwrap();
            same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
        }
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

#[tokio::test]
async fn test_empty_prefix_activates_grouped_aggregates_on_first_rows() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for cold in [false, true] {
            check(GROUPED, &dtype, cold).await;
        }
    }
}

#[tokio::test]
async fn test_empty_prefix_activates_global_aggregates_on_first_rows() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for cold in [false, true] {
            check(GLOBAL, &dtype, cold).await;
        }
    }
}

#[tokio::test]
async fn test_empty_prefix_activation_refusal_preserves_state_and_retries() {
    use crate::CalcFlowError;
    use datafusion::execution::memory_pool::MemoryConsumer;

    let dtype = DataType::Float64;
    for query in [GROUPED, GLOBAL] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = operator_query(&dtype, query);
        drop(process(&mut state, input(&dtype, &[vec![]], 0), &context).await);
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let basis = pool.reserved();
        assert_eq!(job.gather_owner().funding(), (0, 0, 0));
        let next = arrivals(false).remove(0);
        let pressure = MemoryConsumer::new("empty-prefix-pressure").register(&pool);
        pressure.try_grow((1 << 30) - basis - 1).unwrap();
        let held = pool.reserved();
        let rejected = input(&dtype, &next, 1);
        let weak = weak_arrays(&rejected);
        let mut output = EdgeCollector::new(state.output_ports().to_vec());
        assert!(
            state
                .process_data("events", rejected, &context, &mut output)
                .await
                .is_err()
        );
        assert!(output.drain("output").is_empty());
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        assert_eq!(pool.reserved(), held);
        drop(pressure);
        assert_eq!(pool.reserved(), basis);
        same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
        let rejected = input(&dtype, &next, 1);
        let weak = weak_arrays(&rejected);
        assert!(
            matches!(state.process_data("events", rejected, &context, &mut Reject).await,
            Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-grouped-float")
        );
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        let (home, generation, attempt) = job.gather_owner().funding();
        assert_eq!(attempt, 0);
        assert_eq!(pool.reserved(), basis + home + generation);
        same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
        let actual = process(&mut state, input(&dtype, &next, 1), &context).await;
        oracle(&actual, query, &dtype, &next, 1).await;
        assert!(state.incremental.is_some() && state.compact.is_some() && state.retained.is_none());
        drop((actual, state, snapshot));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}
