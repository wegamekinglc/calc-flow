use super::*;

async fn raw_capture(mut state: SqlOperator, batch: Batch, query: &str) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([("events".into(), batch.clone())]),
            Some("raw-oracle"),
        )
        .await
        .unwrap();
    let actual = process(&mut state, batch, &context).await;
    assert_eq!(rows(&actual), rows(&expected));
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(actual.metadata(), expected.metadata());
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(4));
    assert!(state.incremental.is_none() && state.compact.is_none() && state.retained.is_some());
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((state, snapshot, actual));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_global_record_nondefault_empty_first_integer_average_stay_raw4() {
    for (dtype, query) in cases() {
        let mut state = operator(&dtype, query);
        state.set_stream_resources(
            DataFusionConfig {
                target_partitions: 4,
                ..DataFusionConfig::default()
            },
            UdfRegistrySnapshot::default(),
            vec![],
        );
        raw_capture(state, input(&dtype, &[vec![Some(ONE), None]], 0), query).await;
        raw_capture(operator(&dtype, query), input(&dtype, &[vec![]], 0), query).await;
    }
    let dtype = DataType::Int64;
    let record = RecordBatch::try_new(
        schema(&dtype),
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 1])),
            Arc::new(Int64Array::from(vec![Some(1_i64 << 54), None, Some(1)])),
        ],
    )
    .unwrap();
    raw_capture(
        operator(&dtype, AVG),
        Batch::table(vec![record], metadata(0)).unwrap(),
        AVG,
    )
    .await;
}

fn policy_snapshot(snapshot: &OperatorStateSnapshot, policy: Value) -> OperatorStateSnapshot {
    let mut control: Value = serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    control["state_policy"] = policy;
    let segment = StateSegment::new(serde_json::to_vec(&control).unwrap());
    let mut snapshot = snapshot.clone();
    snapshot
        .inline_metadata
        .insert("control_sha256".into(), json!(segment.sha256()));
    snapshot.segments.insert("control".into(), segment);
    snapshot
}

fn null_count_snapshot(snapshot: &OperatorStateSnapshot) -> OperatorStateSnapshot {
    state_snapshot(
        snapshot,
        vec![
            Arc::new(datafusion::arrow::array::UInt64Array::from(vec![
                None::<u64>,
            ])),
            Arc::new(Float64Array::from(vec![None::<f64>])),
        ],
        false,
    )
}

fn zero_input_snapshot(snapshot: &OperatorStateSnapshot, query: &str) -> OperatorStateSnapshot {
    let mut values: Vec<ArrayRef> = Vec::new();
    if query == AVG {
        values.push(Arc::new(datafusion::arrow::array::UInt64Array::from(vec![
            Some(0_u64),
        ])));
    }
    values.push(Arc::new(Float64Array::from(vec![None::<f64>])));
    state_snapshot(snapshot, values, true)
}

fn state_snapshot(
    snapshot: &OperatorStateSnapshot,
    values: Vec<ArrayRef>,
    zero_input: bool,
) -> OperatorStateSnapshot {
    let state = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
    let record =
        RecordBatch::try_new(state.table_payload().unwrap().schema().clone(), values).unwrap();
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let segment = StateSegment::new(encode_sql_state(&batch).unwrap());
    let mut control: Value = serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    control["segments"]["group_state"] = json!(segment.sha256());
    if zero_input {
        control["ledger"]["rows"] = json!(0);
        control["ledger"]["bytes"] = json!(0);
    }
    let control = StateSegment::new(serde_json::to_vec(&control).unwrap());
    let mut snapshot = snapshot.clone();
    if zero_input {
        snapshot.inline_metadata.insert("rows".into(), json!(0));
        snapshot.inline_metadata.insert("bytes".into(), json!(0));
    }
    snapshot
        .inline_metadata
        .insert("control_sha256".into(), json!(control.sha256()));
    snapshot.segments.insert("control".into(), control);
    snapshot.segments.insert("group-state".into(), segment);
    snapshot
}

#[tokio::test]
async fn test_global_record_current_policy_required_and_cold_replanned() {
    for (dtype, query) in cases() {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut source = operator(&dtype, query);
        let parts = vec![vec![Some(ONE), None]];
        drop(process(&mut source, input(&dtype, &parts, 0), &context).await);
        let snapshot = capture(&mut source, &dtype, query, &parts, 0).await;
        assert!(
            StreamOperator::restore(
                &mut operator(&dtype, query),
                &zero_input_snapshot(&snapshot, query)
            )
            .is_err()
        );
        if query == AVG {
            assert!(
                StreamOperator::restore(
                    &mut operator(&dtype, query),
                    &null_count_snapshot(&snapshot)
                )
                .is_err()
            );
        }
        let control: Value = serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
        let original = control["state_policy"].clone();
        assert!(original["global_record_float_v1"].is_object());
        for field in ["config", "factory", "model"] {
            let mut policy = original.clone();
            policy["global_record_float_v1"]
                .as_object_mut()
                .unwrap()
                .remove(field);
            assert!(
                StreamOperator::restore(
                    &mut operator(&dtype, query),
                    &policy_snapshot(&snapshot, policy)
                )
                .is_err()
            );
        }
        for (field, value) in [
            ("unknown", json!(1)),
            ("factory", json!("unknown")),
            ("model", json!("unknown")),
        ] {
            let mut policy = original.clone();
            policy["global_record_float_v1"][field] = value;
            assert!(
                StreamOperator::restore(
                    &mut operator(&dtype, query),
                    &policy_snapshot(&snapshot, policy)
                )
                .is_err()
            );
        }
        let mut policy = original.clone();
        policy["global_record_float_v1"]["config"]["target_partitions"] = json!(4);
        assert!(
            StreamOperator::restore(
                &mut operator(&dtype, query),
                &policy_snapshot(&snapshot, policy)
            )
            .is_err()
        );
        let mut policy = original;
        policy["global_record_float_v1"]["config"]
            .as_object_mut()
            .unwrap()
            .remove("batch_size");
        assert!(
            StreamOperator::restore(
                &mut operator(&dtype, query),
                &policy_snapshot(&snapshot, policy)
            )
            .is_err()
        );
        let pool = source
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((source, snapshot));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}
