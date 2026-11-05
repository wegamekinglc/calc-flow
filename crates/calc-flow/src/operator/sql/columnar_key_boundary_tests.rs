use super::*;

fn boundary_input(valid: bool) -> Batch {
    let groups = CHUNK_ROWS + 1;
    let count = 2 * groups;
    let keys = (0..count)
        .map(|row| format!("account-{}", row % groups))
        .collect::<Vec<_>>();
    let values = (0..count)
        .map(|row| (valid && row == 0).then_some(1))
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![
        Field::new("left_key", DataType::Utf8, true),
        Field::new("right_key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(keys)),
            Arc::new(StringArray::from(vec![Some("stock"); count])),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn assert_boundary(records: &[RecordBatch]) {
    let actual = rows(records);
    assert_eq!(actual.len(), CHUNK_ROWS + 1);
    for (group, row) in actual.iter().enumerate() {
        assert_eq!(
            row,
            &vec![
                ScalarValue::Utf8(Some(format!("account-{group}"))),
                ScalarValue::Utf8(Some("stock".into())),
                ScalarValue::Int64((group == 0).then_some(1))
            ]
        );
    }
}

#[tokio::test]
async fn test_columnar_new_group_after_8192_and_active_cancel_refund_null_tail_retry() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let first = boundary_input(false);
    let mut state = configured(
        &runtime,
        &first,
        SQL,
        u64::try_from(CHUNK_ROWS + 1).unwrap(),
        StateBudget::default().max_bytes(),
    )
    .await;
    let initial_job = job(CancellationToken::new());
    let initial_context = StreamOperatorContext::new(&initial_job, "columnar-semantic", None);
    let transaction = state
        .update(&first, &initial_context, "columnar-semantic")
        .await
        .unwrap();
    state.commit(transaction);
    assert_eq!(state.groups.len(), CHUNK_ROWS + 1);
    assert_eq!(state.encoded_rows.load(Ordering::SeqCst), CHUNK_ROWS + 1);
    let saved = state
        .export_native_state("columnar-semantic", || Ok(()))
        .unwrap();
    let paid = runtime.compact_state_bytes();
    let token = CancellationToken::new();
    let cancelled_job = job(token.clone());
    let context = StreamOperatorContext::new(&cancelled_job, "columnar-semantic", None);
    state.encoded_rows.store(0, Ordering::SeqCst);
    let entered = state.encoded_rows.clone();
    let cancel = tokio::spawn(async move {
        while entered.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        token.cancel();
    });
    let incoming = boundary_input(true);
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        state.update(&incoming, &context, "columnar-semantic"),
    )
    .await
    .unwrap();
    cancel.await.unwrap();
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(runtime.compact_state_bytes(), paid);
    let after = state
        .export_native_state("columnar-semantic", || Ok(()))
        .unwrap();
    assert_eq!(saved.records(), after.records());
    let transaction = state
        .update(&incoming, &initial_context, "columnar-semantic")
        .await
        .unwrap();
    assert_boundary(&transaction.records);
    state.commit(transaction);
    drop((state, saved, after));
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}

#[tokio::test]
async fn test_columnar_maintained_dense_sql_shape_uses_10000_distinct_canonical_keys() {
    let groups = 10000;
    let count = 80000;
    let record = RecordBatch::try_from_iter(vec![
        (
            "account",
            Arc::new(StringArray::from_iter_values(
                (0..count).map(|row| format!("account-{}", (row % groups) / 100)),
            )) as ArrayRef,
        ),
        (
            "stock",
            Arc::new(StringArray::from_iter_values(
                (0..count).map(|row| format!("stock-{}", row % 100)),
            )) as ArrayRef,
        ),
        (
            "delta",
            Arc::new(Int64Array::from(vec![1; count])) as ArrayRef,
        ),
    ])
    .unwrap();
    let incoming = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let sql = "SELECT account,stock,SUM(delta) AS position FROM events GROUP BY account,stock";
    let mut state = configured(
        &runtime,
        &incoming,
        sql,
        u64::try_from(groups).unwrap(),
        StateBudget::default().max_bytes(),
    )
    .await;
    let job = job(CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "columnar-semantic", None);
    let transaction = state
        .update(&incoming, &context, "columnar-semantic")
        .await
        .unwrap();
    assert_eq!(state.encoded_rows.load(Ordering::SeqCst), groups);
    assert_eq!(
        state.key_work.columnar_chunks.load(Ordering::SeqCst),
        count.div_ceil(CHUNK_ROWS)
    );
    assert_eq!(rows(&transaction.records).len(), groups);
    for row in rows(&transaction.records) {
        assert_eq!(row[2], ScalarValue::Int64(Some(8)));
    }
    state.commit(transaction);
    drop(state);
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}
