use super::*;

fn unset_null_winner(snapshot: &OperatorStateSnapshot, id: &str) -> OperatorStateSnapshot {
    let mut bad = detached(snapshot);
    let batch = decode_sql_state(bad.segments[id].bytes()).unwrap();
    let record = &batch.table_payload().unwrap().batches()[0];
    let last = record.num_columns() - 1;
    let flags = record
        .column(last)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    let keys = record
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let mut columns = record.columns().to_vec();
    columns[last] = Arc::new(BooleanArray::from(
        keys.iter()
            .enumerate()
            .map(|(row, key)| {
                if key == Some("A") {
                    assert!(flags.value(row));
                    for column in &columns[1..last] {
                        assert!(
                            column.is_null(row),
                            "only an all-null candidate may be changed"
                        );
                    }
                    false
                } else {
                    flags.value(row)
                }
            })
            .collect::<Vec<_>>(),
    ));
    let changed = RecordBatch::try_new(record.schema(), columns).unwrap();
    replace_wire(&mut bad, id, vec![changed]);
    bad
}

fn state_bytes(op: &SqlOperator) -> Vec<u8> {
    let export = op
        .incremental
        .as_ref()
        .unwrap()
        .export_native_state("ordered_last", || Ok(()))
        .unwrap();
    encode_sql_state(&Batch::table(export.records().to_vec(), BatchMetadata::default()).unwrap())
        .unwrap()
}

fn reject_without_install(op: &mut SqlOperator, bad: &OperatorStateSnapshot) {
    let compact = op.compact.as_ref().unwrap();
    let ledger = compact.ledger;
    let metadata = compact.metadata.clone();
    let capture = compact.capture.as_ref().unwrap().clone();
    let native = op.incremental.as_ref().unwrap();
    let owner = std::ptr::from_ref(native.as_ref());
    let groups = native.group_count();
    let dirty = native.checkpoint_changes();
    assert!(dirty > 0, "rejection must also preserve uncaptured updates");
    let bytes = state_bytes(op);
    let runtime = op.retention_runtime().unwrap();
    let pool = runtime.incremental_memory_pool();
    let basis = (pool.reserved(), runtime.compact_state_bytes());
    let error = StreamOperator::restore(op, bad).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("candidate state is inconsistent"),
        "{error}"
    );
    assert_eq!(
        basis,
        (
            pool.reserved(),
            op.retention_runtime().unwrap().compact_state_bytes()
        )
    );
    let compact = op.compact.as_ref().unwrap();
    assert_eq!(compact.ledger, ledger);
    assert_eq!(compact.metadata, metadata);
    assert!(Arc::ptr_eq(&capture, compact.capture.as_ref().unwrap()));
    let native = op.incremental.as_ref().unwrap();
    assert_eq!(owner, std::ptr::from_ref(native.as_ref()));
    assert_eq!(
        (native.group_count(), native.checkpoint_changes()),
        (groups, dirty)
    );
    assert_eq!(state_bytes(op), bytes);
}

async fn respect_case(treatment: &str, delta: bool) {
    let query = format!(
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp ASC NULLS LAST) {treatment} AS price FROM input GROUP BY stock"
    );
    let job = job();
    let ctx = StreamOperatorContext::new(&job, "ordered_last", None);
    let mut live = operator(&query);
    let mut prefix = SEED.to_vec();
    prefix[0] = (Some("A"), None, None, None);
    drop(process(&mut live, input(&prefix), &ctx).await);
    let mut snapshot = live.checkpoint(Epoch::INITIAL).unwrap();
    if delta {
        let row = (Some("A"), None, None, None);
        prefix.push(row);
        drop(process(&mut live, input(&[row]), &ctx).await);
        snapshot = live.checkpoint(Epoch::INITIAL).unwrap();
    }
    let id = if delta {
        delta_ids(&snapshot).remove(0)
    } else {
        "group-state".into()
    };
    let bad = unset_null_winner(&snapshot, &id);
    let mut restored = operator(&query);
    StreamOperator::restore(&mut restored, &detached(&snapshot)).unwrap();
    let update = (Some("new"), Some(9.0), Some(100), Some(0));
    let metadata = BatchMetadata::new("dirty-before-rejection", 10, JsonMap::new()).unwrap();
    let batch = input(&[update]).with_metadata(metadata);
    drop(process(&mut live, batch.clone(), &ctx).await);
    drop(process(&mut restored, batch, &ctx).await);
    prefix.push(update);
    reject_without_install(&mut restored, &bad);
    let earlier = (Some("A"), Some(7.0), Some(1), Some(0));
    prefix.push(earlier);
    let actual = process(&mut restored, input(&[earlier]), &ctx).await;
    let uninterrupted = process(&mut live, input(&[earlier]), &ctx).await;
    let oracle = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            &query,
            &BTreeMap::from([("input".into(), input(&prefix))]),
            None,
        )
        .await
        .unwrap();
    assert_eq!(values(&actual), values(&uninterrupted));
    assert_eq!(values(&actual), values(&oracle));
    assert_eq!(
        values(&actual)["Utf8(\"A\")"][1],
        ScalarValue::Float64(None)
    );
    let recaptured = restored.checkpoint(Epoch::INITIAL).unwrap();
    let mut cold = operator(&query);
    StreamOperator::restore(&mut cold, &detached(&recaptured)).unwrap();
    same_snapshot(&recaptured, &cold.checkpoint(Epoch::INITIAL).unwrap());
}

#[tokio::test]
async fn test_ordered_last_default_nulls_reject_unset_null_winner_in_full_and_delta() {
    for delta in [false, true] {
        respect_case("", delta).await;
    }
}

#[tokio::test]
async fn test_ordered_last_respect_nulls_reject_unset_null_winner_in_full_and_delta() {
    for delta in [false, true] {
        respect_case("RESPECT NULLS", delta).await;
    }
}
