use super::*;

#[tokio::test]
async fn test_sql_retained_descriptor_rejects_forgery_atomically() {
    let (mut operator, job, mut collector) = setup("SELECT SUM(value) AS total FROM events");
    let context = StreamOperatorContext::new(&job, "totals", None);
    operator
        .process_data("events", wide_input(0).0, &context, &mut collector)
        .await
        .unwrap();
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(before.inline_metadata["state_layout"], json!(3));
    let mutations = [
        ("state_layout", json!(4)),
        ("state_accounting", json!(1)),
        ("rows", json!(4)),
        ("bytes", json!(0)),
        ("unknown", json!(0)),
    ];
    for (field, value) in mutations {
        let mut forged = before.clone();
        forged.inline_metadata.insert(field.into(), value);
        assert!(
            StreamOperator::restore(&mut operator, &forged).is_err(),
            "accepted forged {field}"
        );
        assert_same_snapshot(&before, &operator.checkpoint(Epoch::INITIAL).unwrap());
    }
    for (field, value) in [
        ("retained_ordinals", json!([0, 1])),
        ("logical_schema_sha256", json!("00".repeat(32))),
        ("physical_schema_sha256", json!("00".repeat(32))),
    ] {
        let mut forged = before.clone();
        let mut control: Value =
            serde_json::from_slice(forged.segments["control"].bytes()).unwrap();
        control["identity"][field] = value;
        let segment = StateSegment::new(serde_json::to_vec(&control).unwrap());
        forged
            .inline_metadata
            .insert("control_sha256".into(), json!(segment.sha256()));
        forged.segments.insert("control".into(), segment);
        assert!(
            StreamOperator::restore(&mut operator, &forged).is_err(),
            "accepted forged control {field}"
        );
        assert_same_snapshot(&before, &operator.checkpoint(Epoch::INITIAL).unwrap());
    }
    for name in ["input", "unknown"] {
        let mut forged = before.clone();
        forged
            .segments
            .insert(name.into(), StateSegment::new(vec![]));
        assert!(StreamOperator::restore(&mut operator, &forged).is_err());
        assert_same_snapshot(&before, &operator.checkpoint(Epoch::INITIAL).unwrap());
    }
}

fn assert_same_snapshot(left: &OperatorStateSnapshot, right: &OperatorStateSnapshot) {
    assert_eq!(left.inline_metadata, right.inline_metadata);
    assert_eq!(
        left.segments.keys().collect::<Vec<_>>(),
        right.segments.keys().collect::<Vec<_>>()
    );
    for (name, segment) in &left.segments {
        assert_eq!(segment.bytes(), right.segments[name].bytes());
    }
}

fn change_unused_schema(input: &Batch) -> Batch {
    let table = input.table_payload().unwrap();
    let mut fields = table.schema().fields().to_vec();
    fields[3] = Arc::new(fields[3].as_ref().clone().with_metadata(
        std::collections::HashMap::from([("changed".into(), "unused".into())]),
    ));
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        table.schema().metadata().clone(),
    ));
    let records = table
        .batches()
        .iter()
        .map(|record| RecordBatch::try_new(schema.clone(), record.columns().to_vec()).unwrap())
        .collect();
    Batch::table(records, input.metadata().clone()).unwrap()
}

#[tokio::test]
async fn test_sql_retained_restore_validates_full_logical_schema() {
    let query = "SELECT SUM(value) AS total FROM events";
    let (mut original, job, mut output) = setup(query);
    let context = StreamOperatorContext::new(&job, "totals", None);
    let first = wide_input(0).0;
    original
        .process_data("events", first.clone(), &context, &mut output)
        .await
        .unwrap();
    let checkpoint = original.checkpoint(Epoch::INITIAL).unwrap();
    let changed = change_unused_schema(&first);
    let input_port = Port::with_schema_ref(
        "events",
        BatchKind::Table,
        true,
        Some(changed.table_payload().unwrap().schema().clone()),
    )
    .unwrap();
    let mut exact = SqlOperator::new("totals", query, vec!["events".into()], vec![])
        .unwrap()
        .with_ports(vec![input_port], table_port("output").unwrap())
        .unwrap();
    assert!(StreamOperator::restore(&mut exact, &checkpoint).is_err());
    let (mut dynamic, _, mut continued) = setup(query);
    StreamOperator::restore(&mut dynamic, &checkpoint).unwrap();
    assert!(
        dynamic
            .process_data("events", changed, &context, &mut continued)
            .await
            .is_err()
    );
    assert_same_snapshot(&checkpoint, &dynamic.checkpoint(Epoch::INITIAL).unwrap());
    dynamic
        .process_data("events", wide_input(1).0, &context, &mut continued)
        .await
        .unwrap();
    assert_eq!(
        rows(continued.drain("output")[0].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(24))]]
    );
}

#[tokio::test]
async fn test_sql_retained_row_budget_and_physical_byte_charge() {
    let (mut operator, job, mut output) = setup("SELECT SUM(value) AS total FROM events");
    operator
        .set_state_budget(StateBudget::new(3, 100).unwrap())
        .unwrap();
    let context = StreamOperatorContext::new(&job, "totals", None);
    let input = wide_input(0).0;
    let physical = Batch::table(
        input
            .table_payload()
            .unwrap()
            .batches()
            .iter()
            .map(|record| record.project(&[1]).unwrap())
            .collect(),
        input.metadata().clone(),
    )
    .unwrap();
    assert!(input.estimated_bytes().unwrap() > 100);
    operator
        .process_data("events", input, &context, &mut output)
        .await
        .unwrap();
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(operator.retained.is_none());
    assert_eq!(operator.compact.as_ref().unwrap().ledger.rows, 3);
    assert_eq!(
        before.inline_metadata["bytes"],
        json!(physical.estimated_bytes().unwrap())
    );
    assert!(
        operator
            .process_data("events", wide_input(1).0, &context, &mut output)
            .await
            .is_err()
    );
    assert_same_snapshot(&before, &operator.checkpoint(Epoch::INITIAL).unwrap());
}

struct Reject;

#[async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Err(CalcFlowError::Operator {
            node_id: "sink".into(),
            message: "retained emit rejected".into(),
        })
    }
}

#[tokio::test]
async fn test_sql_retained_rejected_output_does_not_install_projection() {
    let (mut operator, job, mut output) = setup("SELECT SUM(value) AS total FROM events");
    let context = StreamOperatorContext::new(&job, "totals", None);
    let (input, unused, allocation) = wide_input(0);
    assert!(
        operator
            .process_data("events", input, &context, &mut Reject)
            .await
            .is_err()
    );
    assert!(operator.retained.is_none());
    assert!(unused.upgrade().is_none());
    assert_eq!(allocation.strong_count(), 1);
    operator
        .process_data("events", wide_input(1).0, &context, &mut output)
        .await
        .unwrap();
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(
        operator
            .process_data("events", wide_input(2).0, &context, &mut Reject)
            .await
            .is_err()
    );
    assert_same_snapshot(&before, &operator.checkpoint(Epoch::INITIAL).unwrap());
}

#[tokio::test]
async fn test_sql_retained_alias_collision_keeps_source_column() {
    let query =
        "SELECT SUM(value) AS keep FROM events WHERE keep > 0 HAVING SUM(value) > 1 ORDER BY keep";
    let (mut operator, job, mut output) = setup(query);
    let context = StreamOperatorContext::new(&job, "totals", None);
    operator
        .process_data("events", wide_input(0).0, &context, &mut output)
        .await
        .unwrap();
    assert_eq!(retained_names(&operator), vec!["value", "keep"]);
    assert_eq!(
        rows(output.drain("output")[0].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(3))]]
    );
}

#[tokio::test]
async fn test_sql_retained_unproved_lineage_keeps_full_schema() {
    let query =
        "WITH selected AS (SELECT * FROM events) SELECT SUM(abs(value)) AS total FROM selected";
    let (mut operator, job, mut output) = setup(query);
    let context = StreamOperatorContext::new(&job, "totals", None);
    operator
        .process_data("events", wide_input(0).0, &context, &mut output)
        .await
        .unwrap();
    assert_eq!(retained_names(&operator).len(), 8);
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata.len(), 6);
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(4));
    assert_eq!(
        snapshot
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec![
            "batch-metadata",
            "control",
            "input-retained",
            "logical-schema"
        ]
    );
}

#[tokio::test]
async fn test_sql_retained_shared_ipc_backing_detaches_only_selected_payload() {
    let bytes = encode_sql_state(&wide_input(0).0).unwrap();
    let decoded = decode_sql_state(&bytes).unwrap();
    let (mut operator, job, mut output) = setup(RAW_SUM);
    let context = StreamOperatorContext::new(&job, "totals", None);
    let source_capacity = decoded.table_payload().unwrap().batches()[0]
        .column(1)
        .to_data()
        .buffers()[0]
        .capacity();
    assert!(
        source_capacity > 32_000,
        "fixture does not share the wide IPC body"
    );
    operator
        .process_data("events", decoded.clone(), &context, &mut output)
        .await
        .unwrap();
    let emitted = output.drain("output");
    assert_eq!(emitted.len(), 1);
    let observed = emitted[0].as_data().unwrap();
    assert_raw_oracle(&operator, &decoded, observed).await;
    let retained = &operator.retained.as_ref().unwrap().records[0];
    assert_eq!(retained.num_columns(), 1);
    assert!(
        retained.column(0).to_data().buffers()[0].capacity() < 4096,
        "narrow SQL state still pins the wide IPC allocation"
    );
    assert_eq!(
        rows(observed),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(12))]]
    );
}

#[tokio::test]
async fn test_sql_retained_checkpoint_keeps_large_metadata_out_of_inline_manifest() {
    let (mut operator, job, mut output) = setup("SELECT SUM(value) AS total FROM events");
    let context = StreamOperatorContext::new(&job, "totals", None);
    let data = wide_input(0).0;
    let metadata = BatchMetadata::new(
        "metadata",
        19,
        JsonMap::from([("large".into(), json!("x".repeat(1 << 20)))]),
    )
    .unwrap();
    let input = Batch::table(
        data.table_payload().unwrap().batches().to_vec(),
        metadata.clone(),
    )
    .unwrap();
    operator
        .process_data("events", input, &context, &mut output)
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(
        serde_json::to_vec(&snapshot.inline_metadata).unwrap().len() < 4096,
        "large Batch metadata was copied into the inline manifest"
    );
    assert!(snapshot.segments.contains_key("batch-metadata"));
    let (mut restored, _, mut continuation) = setup("SELECT SUM(value) AS total FROM events");
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    assert!(restored.retained.is_none());
    assert_eq!(restored.compact.as_ref().unwrap().metadata, metadata);
    restored
        .process_data("events", wide_input(1).0, &context, &mut continuation)
        .await
        .unwrap();
    assert_eq!(
        rows(continuation.drain("output")[0].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(24))]]
    );
}

#[tokio::test]
async fn test_sql_retained_checkpoint_ipc_body_is_prepaid_and_failure_atomic() {
    let (mut operator, job, mut output) = setup(RAW_SUM);
    let context = StreamOperatorContext::new(&job, "totals", None);
    let values: ArrayRef = Arc::new(Int64Array::from_iter_values(0..100_000));
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )])),
        vec![values],
    )
    .unwrap();
    let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    operator
        .process_data("events", input.clone(), &context, &mut output)
        .await
        .unwrap();
    let emitted = output.drain("output");
    assert_eq!(emitted.len(), 1);
    let observed = emitted[0].as_data().unwrap();
    assert_raw_oracle(&operator, &input, observed).await;
    assert!(operator.retained.as_ref().unwrap().projection.is_none());
    let runtime = operator.stream_state.runtime().unwrap();
    let spare = runtime.incremental_reservation("checkpoint-spare");
    spare.try_grow(64 << 10).unwrap();
    let pressure = runtime.incremental_reservation("checkpoint-pressure");
    let mut size = 1usize << 29;
    while size >= 4096 {
        while pressure.try_grow(size).is_ok() {}
        size /= 2;
    }
    spare.free();
    let result = operator.prepare_checkpoint_async(&context).await;
    assert!(
        result.is_err(),
        "checkpoint IPC body was allocated without available pool credit"
    );
    assert!(operator.retained.as_ref().unwrap().segment.is_none());
    assert_eq!(operator.retained.as_ref().unwrap().rows, 100_000);
    drop(pressure);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(snapshot.segments["input-retained"].bytes().len() > 64 << 10);
}

struct PauseMetadata(tokio::sync::oneshot::Sender<()>);

#[async_trait]
impl StreamCollector for PauseMetadata {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        let (replacement, _) = tokio::sync::oneshot::channel();
        let entered = std::mem::replace(&mut self.0, replacement);
        entered.send(()).unwrap();
        std::future::pending().await
    }
}

fn empty_with_metadata(metadata: BatchMetadata) -> Batch {
    let schema = wide_input(0).0.table_payload().unwrap().schema().clone();
    Batch::table(vec![RecordBatch::new_empty(schema)], metadata).unwrap()
}

#[tokio::test]
async fn test_sql_retained_empty_metadata_changes_and_failed_updates_recover_exactly() {
    let query = "SELECT SUM(value) AS total FROM events";
    let (mut operator, job, mut output) = setup(query);
    let context = StreamOperatorContext::new(&job, "totals", None);
    operator
        .process_data("events", wide_input(0).0, &context, &mut output)
        .await
        .unwrap();
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    let accepted = BatchMetadata::new(
        "accepted-empty",
        11,
        JsonMap::from([("phase".into(), json!("accepted"))]),
    )
    .unwrap();
    operator
        .process_data(
            "events",
            empty_with_metadata(accepted.clone()),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let changed = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(Arc::ptr_eq(
        &before.segments["group-state"].bytes_arc(),
        &changed.segments["group-state"].bytes_arc()
    ));
    assert_ne!(
        before.segments["batch-metadata"].bytes(),
        changed.segments["batch-metadata"].bytes()
    );
    let rejected = BatchMetadata::new("rejected-empty", 12, JsonMap::new()).unwrap();
    assert!(
        operator
            .process_data(
                "events",
                empty_with_metadata(rejected.clone()),
                &context,
                &mut Reject
            )
            .await
            .is_err()
    );
    assert_same_snapshot(&changed, &operator.checkpoint(Epoch::INITIAL).unwrap());
    let (entered, started) = tokio::sync::oneshot::channel();
    let mut paused = PauseMetadata(entered);
    {
        let update = operator.process_data(
            "events",
            empty_with_metadata(rejected),
            &context,
            &mut paused,
        );
        tokio::pin!(update);
        tokio::select! {
            result = &mut update => panic!("metadata update finished before dropping: {result:?}"),
            result = tokio::time::timeout(std::time::Duration::from_secs(10), started) => result.unwrap().unwrap(),
        }
    }
    assert_same_snapshot(&changed, &operator.checkpoint(Epoch::INITIAL).unwrap());
    let (mut restored, _, mut recovered) = setup(query);
    StreamOperator::restore(&mut restored, &changed).unwrap();
    assert!(restored.retained.is_none());
    assert_eq!(restored.compact.as_ref().unwrap().metadata, accepted);
    restored
        .process_data("events", wide_input(1).0, &context, &mut recovered)
        .await
        .unwrap();
    assert_eq!(
        rows(recovered.drain("output")[0].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(24))]]
    );
}

#[tokio::test]
async fn test_sql_retained_snapshot_holds_input_and_metadata_body_credits_after_reset() {
    let mut held = Vec::new();
    for with_metadata in [false, true] {
        let (mut operator, job, mut output) = setup(RAW_SUM);
        let context = StreamOperatorContext::new(&job, "totals", None);
        let input = if with_metadata {
            let data = wide_input(0).0;
            let metadata = BatchMetadata::new(
                "owned-metadata",
                0,
                JsonMap::from([("payload".into(), json!("x".repeat(1 << 20)))]),
            )
            .unwrap();
            Batch::table(data.table_payload().unwrap().batches().to_vec(), metadata).unwrap()
        } else {
            let values: ArrayRef = Arc::new(Int64Array::from_iter_values(0..100_000));
            let record = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "value",
                    DataType::Int64,
                    false,
                )])),
                vec![values],
            )
            .unwrap();
            Batch::table(vec![record], BatchMetadata::default()).unwrap()
        };
        operator
            .process_data("events", input, &context, &mut output)
            .await
            .unwrap();
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        StreamOperator::reset(&mut operator).unwrap();
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        let probe = operator
            .stream_state
            .runtime()
            .unwrap()
            .incremental_reservation("snapshot-liveness-probe");
        let bytes = (1usize << 30) - (16 << 10);
        let alive = probe.try_grow(bytes).is_err();
        probe.free();
        drop(snapshot);
        probe.try_grow(bytes).unwrap();
        probe.free();
        held.push(alive);
    }
    assert_eq!(
        held,
        vec![true, true],
        "captured input/metadata bytes outlived their pool credits after reset"
    );
}

fn with_schema_metadata(
    input: &Batch,
    metadata: std::collections::HashMap<String, String>,
) -> Batch {
    let table = input.table_payload().unwrap();
    let schema = Arc::new(Schema::new_with_metadata(
        table.schema().fields().clone(),
        metadata,
    ));
    let records = table
        .batches()
        .iter()
        .map(|record| RecordBatch::try_new(schema.clone(), record.columns().to_vec()).unwrap())
        .collect();
    Batch::table(records, input.metadata().clone()).unwrap()
}

#[tokio::test]
async fn test_sql_retained_shared_backing_with_large_schema_metadata_remains_valid() {
    let bytes = encode_sql_state(&wide_input(0).0).unwrap();
    let decoded = decode_sql_state(&bytes).unwrap();
    let metadata =
        std::collections::HashMap::from([("schema-payload".into(), "x".repeat(1 << 20))]);
    let input = with_schema_metadata(&decoded, metadata.clone());
    assert!(
        input.table_payload().unwrap().batches()[0]
            .column(1)
            .to_data()
            .buffers()[0]
            .capacity()
            > 32_000
    );
    let (mut operator, job, mut output) = setup(RAW_SUM);
    let context = StreamOperatorContext::new(&job, "totals", None);
    operator
        .process_data("events", input.clone(), &context, &mut output)
        .await
        .unwrap();
    let emitted = output.drain("output");
    assert_eq!(emitted.len(), 1);
    let observed = emitted[0].as_data().unwrap();
    assert_raw_oracle(&operator, &input, observed).await;
    let retained = &operator.retained.as_ref().unwrap().records[0];
    assert_eq!(retained.num_columns(), 1);
    assert!(retained.column(0).to_data().buffers()[0].capacity() < 4096);
    assert_eq!(retained.schema().metadata(), &metadata);
    assert_eq!(
        input.table_payload().unwrap().schema().metadata(),
        &metadata
    );
    assert_eq!(
        rows(observed),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(12))]]
    );
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let physical = decode_sql_state(snapshot.segments["input-retained"].bytes()).unwrap();
    assert_eq!(
        physical.table_payload().unwrap().schema().metadata(),
        &metadata
    );
    let (mut restored, _, mut continued) = setup(RAW_SUM);
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    assert_eq!(
        restored.retained.as_ref().unwrap().records[0]
            .schema()
            .metadata(),
        &metadata
    );
    restored
        .process_data("events", input.clone(), &context, &mut continued)
        .await
        .unwrap();
    assert_eq!(
        rows(continued.drain("output")[0].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(24))]]
    );
}

#[tokio::test]
async fn test_sql_retained_schema_workspace_rejects_before_arrow_allocation() {
    let metadata =
        std::collections::HashMap::from([("schema-payload".into(), "x".repeat(1 << 20))]);
    assert_schema_workspace_funded(metadata, "schema-allocation-boundary").await;
}

#[tokio::test]
async fn test_sql_retained_schema_workspace_accounts_reserved_metadata_capacity() {
    let metadata = std::collections::HashMap::with_capacity(1 << 15);
    assert!(metadata.is_empty());
    assert!(metadata.clone().capacity() >= 1 << 15);
    assert_schema_workspace_funded(metadata, "schema-capacity-boundary").await;
}

async fn assert_schema_workspace_funded(
    metadata: std::collections::HashMap<String, String>,
    source: &str,
) {
    let (mut operator, job, mut output) = setup(RAW_SUM);
    let context = StreamOperatorContext::new(&job, "totals", None);
    let record = RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            vec![Field::new("value", DataType::Int64, false)],
            metadata.clone(),
        )),
        vec![Arc::new(Int64Array::from(vec![7]))],
    )
    .unwrap();
    let input = Batch::table(
        vec![record],
        BatchMetadata::new(source, 0, JsonMap::new()).unwrap(),
    )
    .unwrap();
    operator
        .process_data("events", input.clone(), &context, &mut output)
        .await
        .unwrap();
    let emitted = output.drain("output");
    assert_eq!(emitted.len(), 1);
    assert_raw_oracle(&operator, &input, emitted[0].as_data().unwrap()).await;
    assert!(operator.retained.as_ref().unwrap().projection.is_none());
    let state = operator.retained.as_ref().unwrap();
    let gauges = (state.rows, state.bytes);
    let schema = state.records[0].schema();
    let allocated = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = allocated.clone();
    tests::on_schema_allocation(input.metadata().source(), move || {
        observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    });
    let runtime = operator.retention_runtime().unwrap();
    let spare = runtime.incremental_reservation("schema-workspace-spare");
    spare.try_grow(64 << 10).unwrap();
    let pressure = runtime.incremental_reservation("schema-workspace-pressure");
    let mut size = 1usize << 29;
    while size >= 4096 {
        while pressure.try_grow(size).is_ok() {}
        size /= 2;
    }
    spare.free();
    assert!(operator.prepare_checkpoint_async(&context).await.is_err());
    let state = operator.retained.as_ref().unwrap();
    assert_eq!((state.rows, state.bytes), gauges);
    assert!(state.segment.is_none());
    assert!(Arc::ptr_eq(&state.records[0].schema(), &schema));
    assert_eq!(state.metadata, *input.metadata());
    assert_eq!(
        allocated.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "Arrow schema preparation began before its metadata workspace was funded"
    );
    drop(pressure);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    assert_eq!(allocated.load(std::sync::atomic::Ordering::SeqCst), 1);
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let decoded = decode_sql_state(snapshot.segments["input-retained"].bytes()).unwrap();
    assert_eq!(
        decoded.table_payload().unwrap().schema().metadata(),
        &metadata
    );
}
