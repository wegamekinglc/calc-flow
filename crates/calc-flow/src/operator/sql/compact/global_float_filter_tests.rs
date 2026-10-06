use super::*;
use datafusion::arrow::array::BooleanArray;

#[path = "global_float_where_tests.rs"]
mod global_float_where_tests;

const FILTERED: &str = "SELECT SUM(value) FILTER (WHERE selected) AS total, AVG(value) FILTER (WHERE selected) AS mean, SUM(alternate) FILTER (WHERE other) AS other_total, AVG(alternate) FILTER (WHERE other) AS other_mean, MIN(value) FILTER (WHERE other) AS lo, MAX(value) FILTER (WHERE selected) AS hi, COUNT(value) FILTER (WHERE selected) AS valid, COUNT(*) FILTER (WHERE other) AS hits, COUNT(*) AS rows, AVG(value) FILTER (WHERE selected) AS again FROM events";

fn filtered_schema(dtype: &DataType) -> SchemaRef {
    let original = schema(dtype);
    let mut fields = original.fields().to_vec();
    fields.push(Arc::new(Field::new("selected", DataType::Boolean, true)));
    fields.push(Arc::new(Field::new("other", DataType::Boolean, true)));
    fields.push(Arc::new(Field::new("alternate", dtype.clone(), true)));
    Arc::new(Schema::new_with_metadata(
        fields,
        original.metadata().clone(),
    ))
}

fn filtered_input(dtype: &DataType, parts: &[Part], sequence: u64) -> Batch {
    let original = input(dtype, parts, sequence);
    let alternate = parts
        .iter()
        .map(|part| {
            part.iter()
                .map(|value| value.map(|bits| (bits.0 ^ (1 << 31), bits.1 ^ (1 << 63))))
                .collect()
        })
        .collect::<Vec<Part>>();
    let alternate = input(dtype, &alternate, sequence);
    let records = original
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .zip(alternate.table_payload().unwrap().batches())
        .map(|(record, alternate)| {
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
            columns.push(Arc::new(flags(0)));
            columns.push(Arc::new(flags(2)));
            columns.push(alternate.column(1).clone());
            RecordBatch::try_new(filtered_schema(dtype), columns).unwrap()
        })
        .collect();
    Batch::table(records, original.metadata().clone()).unwrap()
}

fn filtered_operator(dtype: &DataType, query: &str) -> SqlOperator {
    operator(dtype, query)
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

async fn oracle(actual: &Batch, dtype: &DataType, query: &str, history: &[Vec<Part>]) -> Batch {
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
    let prefix = Batch::table(records, actual.metadata().clone()).unwrap();
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([("events".into(), prefix.clone())]),
            Some("global-float-filter-oracle"),
        )
        .await
        .unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(actual.metadata(), expected.metadata());
    assert_eq!(rows(actual), rows(&expected));
    prefix
}

async fn capture(state: &mut SqlOperator, query: &str, prefix: &Batch) -> OperatorStateSnapshot {
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
    assert!(!snapshot.segments.contains_key("input-retained"));
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .global_record_accumulator_state(query, prefix)
        .await;
    let wire = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
    assert_eq!(
        rows(&wire),
        vec![
            expected
                .iter()
                .map(|value| cell(&value.to_array_of_size(1).unwrap(), 0))
                .collect::<Vec<_>>()
        ]
    );
    let descriptor = state
        .incremental
        .as_ref()
        .unwrap()
        .native_descriptor("float_extrema")
        .unwrap();
    assert_eq!(
        wire.table_payload().unwrap().schema().fields(),
        descriptor.wire_schema.fields()
    );
    for (field, value) in wire
        .table_payload()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .zip(expected)
    {
        assert_eq!(field.data_type(), &value.data_type());
    }
    snapshot
}

async fn prefixes(dtype: &DataType, query: &str, arrivals: Vec<Vec<Part>>) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = filtered_operator(dtype, query);
    let mut history = Vec::new();
    let mut pools = Vec::new();
    for (sequence, parts) in arrivals.into_iter().enumerate() {
        let incoming = filtered_input(dtype, &parts, sequence as u64);
        let weak = weak_arrays(&incoming);
        let actual = process(&mut state, incoming, &context).await;
        history.push(parts);
        let prefix = oracle(&actual, dtype, query, &history).await;
        assert!(
            state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
            "filtered floating aggregates must own chronological native state"
        );
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        if sequence % 2 == 0 {
            state.prepare_compact_capture_async(&context).await.unwrap();
        }
        let snapshot = capture(&mut state, query, &prefix).await;
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
async fn test_global_float_filters_exact_bits_native_state_and_cold_continuation() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for reverse in [false, true] {
            prefixes(&dtype, FILTERED, finite_arrivals(reverse)).await;
        }
        prefixes(&dtype, FILTERED, special_arrivals()).await;
    }
}

#[tokio::test]
async fn test_global_float_filters_refusal_preserves_state_and_once_retry() {
    use datafusion::execution::memory_pool::MemoryConsumer;

    for dtype in [DataType::Float32, DataType::Float64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut state = filtered_operator(&dtype, FILTERED);
        let first = finite_arrivals(false).remove(0);
        let initial = process(&mut state, filtered_input(&dtype, &first, 0), &context).await;
        let prefix = oracle(&initial, &dtype, FILTERED, &[first.clone()]).await;
        let before = capture(&mut state, FILTERED, &prefix).await;
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let basis = pool.reserved();
        let next = finite_arrivals(false).remove(1);
        let pressure = MemoryConsumer::new("global-float-filter-pressure").register(&pool);
        pressure.try_grow((1 << 30) - basis - 1).unwrap();
        let held = pool.reserved();
        let incoming = filtered_input(&dtype, &next, 1);
        let weak = weak_arrays(&incoming);
        let mut collector = EdgeCollector::new(state.output_ports().to_vec());
        assert!(
            state
                .process_data("events", incoming, &context, &mut collector)
                .await
                .is_err()
        );
        assert!(collector.drain("output").is_empty());
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        assert_eq!(pool.reserved(), held);
        drop(pressure);
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
        let incoming = filtered_input(&dtype, &next, 1);
        let weak = weak_arrays(&incoming);
        assert!(matches!(
            state.process_data("events", incoming, &context, &mut Reject).await,
            Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-extrema"
        ));
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        assert_eq!(pool.reserved(), basis);
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
        let actual = process(&mut state, filtered_input(&dtype, &next, 1), &context).await;
        let prefix = oracle(&actual, &dtype, FILTERED, &[first, next]).await;
        let after = capture(&mut state, FILTERED, &prefix).await;
        drop((state, before, after, initial, actual, prefix));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_global_float_computed_filters_and_nondefault_plans() {
    for dtype in [DataType::Float32, DataType::Float64] {
        let parts = vec![vec![Some(ONE), None, Some(TWO)]];
        let computed = "SELECT SUM(value) FILTER (WHERE selected OR other), AVG(value) FILTER (WHERE NOT selected) FROM events";
        prefixes(&dtype, computed, finite_arrivals(false)).await;
        let mut state = filtered_operator(&dtype, FILTERED);
        state.set_stream_resources(
            DataFusionConfig {
                target_partitions: 4,
                ..DataFusionConfig::default()
            },
            UdfRegistrySnapshot::default(),
            vec![],
        );
        global_record_controls::raw_capture(state, filtered_input(&dtype, &parts, 1), FILTERED)
            .await;
    }
}
