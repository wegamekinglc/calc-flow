use super::*;

fn filtered_schema(dtype: &DataType) -> SchemaRef {
    let original = schema(dtype);
    let mut fields = original.fields().to_vec();
    fields[3] = Arc::new(Field::new("selected", DataType::Boolean, true));
    fields.push(Arc::new(Field::new("other", DataType::Boolean, true)));
    Arc::new(Schema::new_with_metadata(
        fields,
        original.metadata().clone(),
    ))
}

fn filtered_input(dtype: &DataType, parts: &[Vec<Row>], sequence: u64) -> Batch {
    let original = input(dtype, parts, sequence);
    let records = original
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            let flags = |offset: usize| {
                BooleanArray::from(
                    (0..record.num_rows())
                        .map(|row| {
                            if sequence == 0 {
                                Some(false)
                            } else {
                                [Some(true), Some(false), None][(row + offset) % 3]
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            };
            let mut columns = record.columns().to_vec();
            columns[3] = Arc::new(flags(0));
            columns.push(Arc::new(flags(1)));
            RecordBatch::try_new(filtered_schema(dtype), columns).unwrap()
        })
        .collect();
    Batch::table(records, original.metadata().clone()).unwrap()
}

fn filtered_operator(dtype: &DataType, query: &str) -> SqlOperator {
    SqlOperator::new("string_extrema", query, vec!["events".into()], vec![])
        .unwrap()
        .with_ports(
            vec![
                Port::with_schema_ref(
                    "events",
                    BatchKind::Table,
                    true,
                    Some(filtered_schema(dtype)),
                )
                .unwrap(),
            ],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap()
}

async fn oracle(actual: &Batch, dtype: &DataType, query: &str, history: &[Vec<Vec<Row>>]) {
    let records = history
        .iter()
        .enumerate()
        .flat_map(|(sequence, parts)| {
            filtered_input(dtype, parts, sequence as u64)
                .table_payload()
                .unwrap()
                .batches()
                .to_vec()
        })
        .collect();
    let input = Batch::table(records, actual.metadata().clone()).unwrap();
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([("events".into(), input)]),
            Some("global-filter-oracle"),
        )
        .await
        .unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(actual.metadata(), expected.metadata());
    let actual = rows(actual);
    let expected = rows(&expected);
    assert_eq!(actual, expected);
    for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
        if let (ScalarValue::Float64(actual), ScalarValue::Float64(expected)) = (actual, expected) {
            assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
        }
    }
}

async fn recovery_case(dtype: &DataType, query: &str, native: bool) {
    let job = StreamJobContext::new(944, "texts", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "string_extrema", None);
    let mut state = filtered_operator(dtype, query);
    let mut history = Vec::new();
    let mut pools = Vec::new();
    for (sequence, parts) in arrivals().into_iter().enumerate() {
        let input = filtered_input(dtype, &parts, sequence as u64);
        let weak = input
            .table_payload()
            .unwrap()
            .batches()
            .iter()
            .flat_map(|record| record.columns().iter().map(Arc::downgrade))
            .collect::<Vec<_>>();
        let actual = process(&mut state, input, &context).await;
        history.push(parts);
        oracle(&actual, dtype, query, &history).await;
        assert_eq!(
            state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
            native,
            "global predicates must select the proven incremental path"
        );
        if native {
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
        state = filtered_operator(dtype, query);
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

#[tokio::test]
async fn test_global_filters_release_inputs_and_restore_exact_prefixes() {
    for dtype in [DataType::Utf8, DataType::LargeUtf8] {
        for query in [
            "SELECT COUNT(*) FILTER (WHERE selected) AS hits, COUNT(value) FILTER (WHERE other) AS valid, MIN(value) FILTER (WHERE selected) AS lo, MAX(value) FILTER (WHERE other) AS hi, COUNT(*) AS rows FROM events",
            "SELECT SUM(key) FILTER (WHERE selected) AS total, MIN(key) FILTER (WHERE other) AS lo, MIN(amount) FILTER (WHERE other) AS amount_lo, MAX(amount) FILTER (WHERE selected) AS amount_hi, COUNT(*) FILTER (WHERE other) AS hits FROM events",
        ] {
            recovery_case(&dtype, query, true).await;
        }
    }
}

#[tokio::test]
async fn test_global_where_and_filters_preserve_empty_selected_prefixes() {
    for dtype in [DataType::Utf8, DataType::LargeUtf8] {
        for query in [
            "SELECT MIN(value) AS lo, MAX(value) AS hi, COUNT(*) AS rows FROM events WHERE selected AND (key IS NULL OR key < 3)",
            "SELECT MIN(value) FILTER (WHERE other) AS lo, SUM(key) FILTER (WHERE other) AS total, COUNT(*) FILTER (WHERE other) AS hits, COUNT(*) AS rows FROM events WHERE selected OR key IS NULL",
        ] {
            recovery_case(&dtype, query, true).await;
        }
    }
}

#[tokio::test]
async fn test_global_float_filters_release_inputs_and_restore_exact_prefixes() {
    let query = "SELECT SUM(amount) FILTER (WHERE selected) AS total, AVG(amount) FILTER (WHERE other) AS mean, COUNT(*) AS rows FROM events";
    recovery_case(&DataType::Utf8, query, true).await;
}

#[tokio::test]
async fn test_global_float_where_releases_inputs_and_restores_exact_prefixes() {
    let query = "SELECT SUM(amount) AS total, AVG(amount) AS mean, COUNT(*) AS rows FROM events WHERE selected OR key IS NULL";
    recovery_case(&DataType::Utf8, query, true).await;
}

#[tokio::test]
async fn test_global_predicate_refusal_preserves_state_refunds_and_retries() {
    use datafusion::execution::memory_pool::MemoryConsumer;

    let query = "SELECT MIN(value) FILTER (WHERE selected) AS lo, MAX(value) FILTER (WHERE other) AS hi, SUM(key) FILTER (WHERE other) AS total, COUNT(*) AS rows FROM events WHERE selected OR key IS NULL";
    for dtype in [DataType::Utf8, DataType::LargeUtf8] {
        let job =
            StreamJobContext::new(945, "texts", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "string_extrema", None);
        let mut state = filtered_operator(&dtype, query);
        let prefix = arrivals().remove(0);
        drop(process(&mut state, filtered_input(&dtype, &prefix, 0), &context).await);
        assert!(state.incremental.is_some() && state.retained.is_none());
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let basis = pool.reserved();
        let next = vec![vec![
            (Some(1), Some("高".repeat(65_536))),
            (None, None),
            (Some(4), Some("😀\0".repeat(65_536))),
        ]];
        let pressure = MemoryConsumer::new("global-predicate-pressure").register(&pool);
        pressure.try_grow((1 << 30) - basis - 1).unwrap();
        let held = pool.reserved();
        let mut output = EdgeCollector::new(state.output_ports().to_vec());
        assert!(
            state
                .process_data(
                    "events",
                    filtered_input(&dtype, &next, 1),
                    &context,
                    &mut output,
                )
                .await
                .is_err()
        );
        assert!(output.drain("output").is_empty());
        assert_eq!(pool.reserved(), held);
        drop(pressure);
        assert_eq!(pool.reserved(), basis);
        same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
        assert!(
            state
                .process_data(
                    "events",
                    filtered_input(&dtype, &next, 1),
                    &context,
                    &mut Reject,
                )
                .await
                .is_err()
        );
        assert_eq!(pool.reserved(), basis);
        same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
        let actual = process(&mut state, filtered_input(&dtype, &next, 1), &context).await;
        oracle(&actual, &dtype, query, &[prefix, next]).await;
        drop((actual, snapshot, state));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        assert_eq!(pool.reserved(), 0);
    }
}
