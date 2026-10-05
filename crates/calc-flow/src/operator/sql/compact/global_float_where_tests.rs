use super::*;

const QUERIES: [&str; 2] = [
    "SELECT SUM(value) AS total, AVG(value) AS mean, MIN(value) AS lo, MAX(value) AS hi, COUNT(value) AS valid, COUNT(*) AS rows FROM events WHERE selected",
    "SELECT SUM(value) FILTER (WHERE other) AS total, AVG(value) FILTER (WHERE other) AS mean, SUM(alternate) AS other_total, COUNT(*) AS rows FROM events WHERE selected OR key IS NULL",
];

fn string_selected_input(key_type: &DataType, parts: &[Part], sequence: u64) -> Batch {
    use datafusion::arrow::array::{LargeStringArray, StringArray};
    let input = selected_input(&DataType::Float64, parts, sequence);
    let wide = "w".repeat(16384);
    let records = input
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            let values = (0..record.num_rows())
                .map(|row| match row % 17 {
                    0 => Some(wide.as_str()),
                    1 => None,
                    _ => Some("short"),
                })
                .collect::<Vec<_>>();
            let keys: ArrayRef = match key_type {
                DataType::Utf8 => Arc::new(StringArray::from(values)),
                DataType::LargeUtf8 => Arc::new(LargeStringArray::from(values)),
                _ => unreachable!(),
            };
            let mut fields = record.schema().fields().to_vec();
            fields[0] = Arc::new(Field::new("key", key_type.clone(), true));
            let schema = Arc::new(Schema::new_with_metadata(
                fields,
                record.schema().metadata().clone(),
            ));
            let mut columns = record.columns().to_vec();
            columns[0] = keys;
            RecordBatch::try_new(schema, columns).unwrap()
        })
        .collect();
    Batch::table(records, input.metadata().clone()).unwrap()
}

#[tokio::test]
async fn test_global_float_where_variable_payload_cold_boundaries() {
    let query = "SELECT SUM(value) AS total, AVG(value) AS mean, COUNT(key) AS valid FROM events WHERE selected";
    for key_type in [DataType::Utf8, DataType::LargeUtf8] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let fresh =
            || SqlOperator::new("float_extrema", query, vec!["events".into()], vec![]).unwrap();
        let mut state = fresh();
        let mut history = Vec::new();
        let mut pools = Vec::new();
        for (sequence, size) in [4096, 1, 4097, 8193].into_iter().enumerate() {
            let incoming =
                string_selected_input(&key_type, &[vec![Some(ONE); size]], sequence as u64);
            let weak = weak_arrays(&incoming);
            history.extend(incoming.table_payload().unwrap().batches().iter().cloned());
            let actual = process(&mut state, incoming, &context).await;
            let prefix = Batch::table(history.clone(), actual.metadata().clone()).unwrap();
            let expected = DataFusionRuntime::new(DataFusionConfig::default())
                .unwrap()
                .sql(
                    query,
                    &BTreeMap::from([("events".into(), prefix.clone())]),
                    Some("variable-where-oracle"),
                )
                .await
                .unwrap();
            assert_eq!(rows(&actual), rows(&expected));
            assert_eq!(
                actual.table_payload().unwrap().schema(),
                expected.table_payload().unwrap().schema()
            );
            assert_eq!(actual.metadata(), expected.metadata());
            drop((prefix, expected, actual));
            let prefix = Batch::table(history.clone(), metadata(sequence as u64)).unwrap();
            let snapshot = capture(&mut state, query, &prefix).await;
            pools.push(
                state
                    .stream_state
                    .runtime()
                    .unwrap()
                    .incremental_memory_pool(),
            );
            drop(state);
            state = fresh();
            StreamOperator::restore(&mut state, &snapshot).unwrap();
            same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
            drop((prefix, snapshot));
            if sequence == 3 {
                history.clear();
                assert!(weak.iter().all(|array| array.upgrade().is_none()));
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
}

#[tokio::test]
async fn test_global_float_where_large_record_fits_bounded_workspace() {
    use datafusion::execution::memory_pool::MemoryConsumer;
    let dtype = DataType::Float64;
    let query = QUERIES[0];
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = filtered_operator(&dtype, query);
    let first = vec![vec![Some(ONE)]];
    drop(process(&mut state, selected_input(&dtype, &first, 0), &context).await);
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let pressure = MemoryConsumer::new("large-record-where-pressure").register(&pool);
    pressure
        .try_grow((1 << 30) - pool.reserved() - (16 << 20))
        .unwrap();
    let next = vec![vec![Some(ONE); 300_000]];
    let incoming = selected_input(&dtype, &next, 1);
    let weak = weak_arrays(&incoming);
    let actual = process(&mut state, incoming, &context).await;
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    drop(pressure);
    let prefix = selected_oracle(&actual, &dtype, query, &[first, next]).await;
    drop(capture(&mut state, query, &prefix).await);
    drop((state, actual, prefix));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_global_float_where_unproven_payload_shapes_keep_retained_fallback() {
    use datafusion::arrow::{array::ListArray, datatypes::Int64Type};
    let dtype = DataType::Float64;
    let incoming = selected_input(&dtype, &[vec![Some(ONE), None, Some(TWO)]], 0);
    let record = &incoming.table_payload().unwrap().batches()[0];
    let nested = Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>([
        None,
        Some(vec![Some(1)]),
        Some(vec![None]),
    ]));
    let mut fields = record.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "nested",
        nested.data_type().clone(),
        true,
    )));
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        record.schema().metadata().clone(),
    ));
    let mut columns = record.columns().to_vec();
    columns.push(nested);
    let record = RecordBatch::try_new(schema, columns).unwrap();
    let query = "SELECT SUM(value), COUNT(nested) FROM events WHERE selected";
    let state = SqlOperator::new("float_extrema", query, vec!["events".into()], vec![]).unwrap();
    global_record_controls::raw_capture(
        state,
        Batch::table(vec![record], metadata(0)).unwrap(),
        query,
    )
    .await;
}

fn selected_input(dtype: &DataType, parts: &[Part], sequence: u64) -> Batch {
    let batch = filtered_input(dtype, parts, sequence);
    let records = batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            let mut columns = record.columns().to_vec();
            columns[2] = Arc::new(BooleanArray::from(vec![true; record.num_rows()]));
            RecordBatch::try_new(record.schema(), columns).unwrap()
        })
        .collect();
    Batch::table(records, batch.metadata().clone()).unwrap()
}

async fn selected_oracle(
    actual: &Batch,
    dtype: &DataType,
    query: &str,
    history: &[Vec<Part>],
) -> Batch {
    let records = history
        .iter()
        .enumerate()
        .flat_map(|(sequence, parts)| {
            selected_input(dtype, parts, sequence as u64)
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
            Some("where-oracle"),
        )
        .await
        .unwrap();
    assert_eq!(rows(actual), rows(&expected));
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(actual.metadata(), expected.metadata());
    prefix
}

fn boundary_arrivals(special: bool) -> Vec<Vec<Part>> {
    [3, 4096, 4097, 1, 4097, 4095, 8194, 0]
        .into_iter()
        .map(|size| {
            let mut part = vec![Some(ONE); size];
            if size > 2 {
                part[0] = Some(LARGE);
                part[size - 1] = Some(NEG_LARGE);
            }
            if special && size > 4096 {
                part[3] = Some(NAN_A);
                part[4096] = Some(SNAN);
                part[4095] = Some(NEG_NAN);
            }
            vec![part]
        })
        .collect()
}

#[tokio::test]
async fn test_global_float_where_coalesced_native_state_and_cold_continuation() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in QUERIES {
            prefixes(&dtype, query, finite_arrivals(false)).await;
            prefixes(&dtype, query, special_arrivals()).await;
        }
    }
}

#[tokio::test]
async fn test_global_float_where_coalescer_boundaries_and_async_cold_cuts() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in QUERIES {
            for special in [false, true] {
                let job = job();
                let context = StreamOperatorContext::new(&job, "float_extrema", None);
                let mut state = filtered_operator(&dtype, query);
                let mut history = Vec::new();
                let mut pools = Vec::new();
                let mut tails = Vec::new();
                for (sequence, parts) in boundary_arrivals(special).into_iter().enumerate() {
                    let incoming = selected_input(&dtype, &parts, sequence as u64);
                    let weak = weak_arrays(&incoming);
                    let actual = process(&mut state, incoming, &context).await;
                    history.push(parts);
                    let prefix = selected_oracle(&actual, &dtype, query, &history).await;
                    assert!(weak.iter().all(|array| array.upgrade().is_none()));
                    if sequence % 2 == 0 {
                        state.prepare_compact_capture_async(&context).await.unwrap();
                    }
                    let snapshot = capture(&mut state, query, &prefix).await;
                    let tail = decode_sql_state(snapshot.segments["global-tail"].bytes()).unwrap();
                    assert!(tail.num_rows() < 8192);
                    tails.push(tail.num_rows());
                    pools.push(
                        state
                            .stream_state
                            .runtime()
                            .unwrap()
                            .incremental_memory_pool(),
                    );
                    drop(state);
                    state = filtered_operator(&dtype, query);
                    StreamOperator::restore(&mut state, &snapshot).unwrap();
                    same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
                }
                assert_eq!(tails, [3, 4099, 0, 1, 4098, 1, 3, 3]);
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
}

#[tokio::test]
async fn test_global_float_where_refusal_preserves_tail_and_once_retry() {
    use datafusion::execution::memory_pool::MemoryConsumer;
    for dtype in [DataType::Float32, DataType::Float64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let query = QUERIES[1];
        let mut state = filtered_operator(&dtype, query);
        let first = vec![vec![Some(ONE); 4096], vec![Some(ONE)]];
        drop(process(&mut state, selected_input(&dtype, &first, 0), &context).await);
        let before = state.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(
            decode_sql_state(before.segments["global-tail"].bytes())
                .unwrap()
                .num_rows(),
            4097
        );
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let basis = pool.reserved();
        let pressure = MemoryConsumer::new("global-where-pressure").register(&pool);
        pressure.try_grow((1 << 30) - basis - 1).unwrap();
        let held = pool.reserved();
        let next = vec![vec![Some(LARGE); 4095], vec![Some(NEG_LARGE), Some(ONE)]];
        let incoming = selected_input(&dtype, &next, 1);
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
        let incoming = selected_input(&dtype, &next, 1);
        let weak = weak_arrays(&incoming);
        assert!(
            matches!(state.process_data("events", incoming, &context, &mut Reject).await, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-extrema")
        );
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        assert_eq!(pool.reserved(), basis);
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
        let actual = process(&mut state, selected_input(&dtype, &next, 1), &context).await;
        let prefix = selected_oracle(&actual, &dtype, query, &[first, next]).await;
        drop(capture(&mut state, query, &prefix).await);
        drop((state, before, actual, prefix));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}

fn edited_control(snapshot: &OperatorStateSnapshot, control: &Value) -> OperatorStateSnapshot {
    let segment = StateSegment::new(serde_json::to_vec(control).unwrap());
    let mut snapshot = snapshot.clone();
    snapshot
        .inline_metadata
        .insert("control_sha256".into(), json!(segment.sha256()));
    snapshot.segments.insert("control".into(), segment);
    snapshot
}

fn assert_single_complete_record_inventory(
    state: &mut SqlOperator,
    before: &OperatorStateSnapshot,
) {
    let wire = decode_sql_state(before.segments["global-complete"].bytes()).unwrap();
    let record = &wire.table_payload().unwrap().batches()[0];
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    for records in [
        vec![record.slice(0, 0), record.clone()],
        vec![record.clone(), record.slice(0, 0)],
    ] {
        let batch = Batch::table(records, BatchMetadata::default()).unwrap();
        let segment = StateSegment::new(encode_sql_state(&batch).unwrap());
        let mut control: Value =
            serde_json::from_slice(before.segments["control"].bytes()).unwrap();
        control["coalescer"]["filter"]["complete"] = json!(segment.sha256());
        let mut changed = edited_control(before, &control);
        changed.segments.insert("global-complete".into(), segment);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            StreamOperator::restore(state, &changed)
        }));
        same_snapshot(before, &state.checkpoint(Epoch::INITIAL).unwrap());
        assert_eq!(pool.reserved(), basis);
        assert!(
            matches!(result, Ok(Err(_))),
            "restore must reject extra complete batches without panicking"
        );
    }
}

#[tokio::test]
async fn test_global_float_where_corrupt_current_tail_and_complete_state_are_atomic() {
    let dtype = DataType::Float64;
    let query = QUERIES[0];
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = filtered_operator(&dtype, query);
    let parts = vec![vec![Some(ONE); 8193]];
    drop(process(&mut state, selected_input(&dtype, &parts, 0), &context).await);
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    for field in ["coalescer", "group_log"] {
        let mut control: Value =
            serde_json::from_slice(before.segments["control"].bytes()).unwrap();
        control.as_object_mut().unwrap().remove(field);
        assert!(StreamOperator::restore(&mut state, &edited_control(&before, &control)).is_err());
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    }
    assert_single_complete_record_inventory(&mut state, &before);
    for id in ["global-tail", "global-complete"] {
        let mut missing = before.clone();
        missing.segments.remove(id);
        assert!(StreamOperator::restore(&mut state, &missing).is_err());
        let wire = decode_sql_state(before.segments[id].bytes()).unwrap();
        let record = &wire.table_payload().unwrap().batches()[0];
        let column = if id == "global-tail" {
            record.schema().index_of("value").unwrap()
        } else {
            0
        };
        let mut columns = record.columns().to_vec();
        columns[column] = Arc::new(Float64Array::from(vec![12345.0; record.num_rows()]));
        let changed = RecordBatch::try_new(record.schema(), columns).unwrap();
        let batch = Batch::table(vec![changed], BatchMetadata::default()).unwrap();
        let segment = StateSegment::new(encode_sql_state(&batch).unwrap());
        let mut control: Value =
            serde_json::from_slice(before.segments["control"].bytes()).unwrap();
        let field = if id == "global-tail" {
            "tail"
        } else {
            "complete"
        };
        control["coalescer"]["filter"][field] = json!(segment.sha256());
        let mut changed = edited_control(&before, &control);
        changed.segments.insert(id.into(), segment);
        assert!(StreamOperator::restore(&mut state, &changed).is_err());
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    }
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((state, before));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
