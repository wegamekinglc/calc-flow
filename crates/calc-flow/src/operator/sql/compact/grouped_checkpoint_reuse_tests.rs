use super::*;

fn rejected_input(dtype: &DataType, key_type: &DataType, sequence: u64) -> Batch {
    let batch = filtered_input(dtype, key_type, &string_arrivals()[1], sequence);
    let records = batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            let mut columns = record.columns().to_vec();
            columns[2] = Arc::new(BooleanArray::from(vec![false; record.num_rows()]));
            RecordBatch::try_new(record.schema(), columns).unwrap()
        })
        .collect();
    Batch::table(records, batch.metadata().clone()).unwrap()
}

async fn assert_prefix(actual: &Batch, text: &str, history: &[Batch], keys: usize) {
    let batch = Batch::table(
        history
            .iter()
            .flat_map(|batch| batch.table_payload().unwrap().batches().to_vec())
            .collect(),
        history.last().unwrap().metadata().clone(),
    )
    .unwrap();
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            text,
            &BTreeMap::from([("events".into(), batch)]),
            Some("oracle"),
        )
        .await
        .unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(actual.metadata(), expected.metadata());
    assert_eq!(filtered_rows(actual, keys), filtered_rows(&expected, keys));
}

async fn reuse_case(dtype: &DataType, key_type: &DataType, text: &str, keys: usize) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "grouped_float", None);
    let mut state = filtered_operator(dtype, key_type, text);
    let first = filtered_input(dtype, key_type, &string_arrivals()[0], 0);
    let mut rows = first.num_rows();
    let actual = process(&mut state, first.clone(), &context).await;
    let mut history = vec![first];
    assert_prefix(&actual, text, &history, keys).await;
    drop(actual);
    let initial = capture(&mut state);
    let payload = initial.segments["group-state"].bytes_arc();
    drop(initial);
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    let refused = rejected_input(dtype, key_type, 1);
    let weak = weak_arrays(&refused);
    assert!(
        state
            .process_data("events", refused, &context, &mut Reject)
            .await
            .is_err()
    );
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    assert_eq!(pool.reserved(), basis);
    for sequence in 1..5 {
        let batch = rejected_input(dtype, key_type, sequence);
        rows += batch.num_rows();
        let actual = process(&mut state, batch.clone(), &context).await;
        history.push(batch);
        assert_prefix(&actual, text, &history, keys).await;
        drop(actual);
        if sequence % 2 == 0 {
            state.prepare_compact_capture_async(&context).await.unwrap();
        }
        let snapshot = capture(&mut state);
        assert_eq!(snapshot.inline_metadata["rows"], json!(rows));
        assert!(
            Arc::ptr_eq(&payload, &snapshot.segments["group-state"].bytes_arc()),
            "fully rejected input must reuse the unchanged group-state allocation"
        );
    }
    let snapshot = capture(&mut state);
    drop((state, payload));
    let mut recovered = filtered_operator(dtype, key_type, text);
    StreamOperator::restore(&mut recovered, &snapshot).unwrap();
    let batch = filtered_input(dtype, key_type, &string_arrivals()[2], 5);
    let actual = process(&mut recovered, batch.clone(), &context).await;
    history.push(batch);
    assert_prefix(&actual, text, &history, keys).await;
    let updated = capture(&mut recovered);
    assert_ne!(
        updated.segments["control"].sha256(),
        snapshot.segments["control"].sha256()
    );
    drop((recovered, actual, snapshot, updated, history));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_grouped_where_rejected_batches_reuse_state_and_restore_exactly() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for key_type in [DataType::Int64, DataType::Utf8, DataType::LargeUtf8] {
            for composite in [false, true] {
                reuse_case(
                    &dtype,
                    &key_type,
                    &query(composite, "accepted"),
                    if composite { 2 } else { 1 },
                )
                .await;
            }
        }
    }
    reuse_case(
        &DataType::Float64,
        &DataType::Int64,
        &COUNTS.replace(
            "FROM events GROUP BY",
            "FROM events WHERE accepted GROUP BY",
        ),
        1,
    )
    .await;
}
