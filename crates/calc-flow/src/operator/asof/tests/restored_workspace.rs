use super::*;
use datafusion::arrow::{
    array::{ArrayRef, StringBuilder},
    buffer::Buffer,
};
use datafusion::execution::memory_pool::MemoryConsumer;
use std::collections::HashMap;

const SOURCE_ROWS: usize = 16;
const SOURCE_BATCHES: usize = 2;
const OUTPUT_ROWS: usize = 4;
const PAYLOAD_BYTES: usize = 2_048;

struct RestoredWorkspaceFixture {
    schema: SchemaRef,
    left: Batch,
    right: Batch,
    decoded: Vec<RecordBatch>,
    limit: usize,
}

fn add_owner(buffer: &Buffer, owners: &mut BTreeMap<usize, usize>) {
    if !buffer.is_empty() {
        assert!(buffer.capacity() >= buffer.len());
        let base = buffer.data_ptr().as_ptr() as usize;
        owners
            .entry(base)
            .and_modify(|capacity| *capacity = (*capacity).max(buffer.capacity()))
            .or_insert(buffer.capacity());
    }
}

fn backing_owners(record: &RecordBatch) -> BTreeMap<usize, usize> {
    let mut owners = BTreeMap::new();
    for column in record.columns() {
        let data = column.to_data();
        assert!(data.child_data().is_empty());
        for buffer in data.buffers() {
            add_owner(buffer, &mut owners);
        }
        if let Some(nulls) = data.nulls() {
            add_owner(nulls.buffer(), &mut owners);
        }
    }
    owners
}

fn owned_capacity<'a>(records: impl Iterator<Item = &'a RecordBatch>) -> usize {
    let mut owners = BTreeMap::<usize, usize>::new();
    for record in records {
        for (base, capacity) in backing_owners(record) {
            owners
                .entry(base)
                .and_modify(|current| *current = (*current).max(capacity))
                .or_insert(capacity);
        }
    }
    owners.values().sum()
}

fn workspace_schema() -> SchemaRef {
    let fields = vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
        Field::new("value", DataType::Int64, true),
        Field::new("payload0", DataType::Utf8, true),
        Field::new("payload1", DataType::Utf8, true),
        Field::new("payload2", DataType::Utf8, true),
        Field::new("payload3", DataType::Utf8, true),
    ]
    .into_iter()
    .map(|field| field.with_metadata(HashMap::from([("domain".into(), "workspace-tdd".into())])))
    .collect::<Vec<_>>();
    Arc::new(Schema::new_with_metadata(
        fields,
        HashMap::from([("fixture".into(), "restored-workspace-v1".into())]),
    ))
}

fn workspace_columns(indices: &[Option<usize>], left: bool) -> Vec<ArrayRef> {
    let number = |row: usize| i64::try_from(row).unwrap();
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(
            indices
                .iter()
                .map(|row| row.map(|row| number(row % 2)))
                .collect::<Int64Array>(),
        ),
        Arc::new(
            indices
                .iter()
                .map(|row| row.map(|row| number(row * 2)))
                .collect::<TimestampMicrosecondArray>()
                .with_timezone("UTC"),
        ),
        Arc::new(
            indices
                .iter()
                .map(|row| row.map(number))
                .collect::<Int64Array>(),
        ),
        Arc::new(
            indices
                .iter()
                .map(|row| {
                    row.and_then(|row| {
                        (row % 5 != 0).then(|| number(row * if left { 7 } else { 3 }))
                    })
                })
                .collect::<Int64Array>(),
        ),
    ];
    for field in 0..4 {
        let mut builder =
            StringBuilder::with_capacity(indices.len(), indices.len() * (PAYLOAD_BYTES + 16));
        for row in indices {
            if let Some(row) = row.filter(|row| row % 3 != 0) {
                builder.append_value(format!(
                    "{}:{row}:{field}:{}",
                    if left { "L" } else { "R" },
                    "x".repeat(PAYLOAD_BYTES),
                ));
            } else {
                builder.append_null();
            }
        }
        columns.push(Arc::new(builder.finish()));
    }
    columns
}

fn workspace_input(schema: &SchemaRef, left: bool) -> Batch {
    let records = (0..SOURCE_BATCHES)
        .map(|batch| {
            let indices = (batch * SOURCE_ROWS..(batch + 1) * SOURCE_ROWS)
                .filter(|row| left || row % 7 != 0)
                .map(Some)
                .collect::<Vec<_>>();
            RecordBatch::try_new(schema.clone(), workspace_columns(&indices, left)).unwrap()
        })
        .collect();
    let attributes = JsonMap::from([
        (
            "nested".into(),
            serde_json::json!({"values": [1, null, {"label": "kept"}]}),
        ),
        ("nullable".into(), serde_json::Value::Null),
    ]);
    Batch::table(
        records,
        BatchMetadata::new(
            if left { "left-source" } else { "right-source" },
            41,
            attributes,
        )
        .unwrap(),
    )
    .unwrap()
}

fn restored_workspace_fixture() -> RestoredWorkspaceFixture {
    let schema = workspace_schema();
    let left = workspace_input(&schema, true);
    let right = workspace_input(&schema, false);
    let digest = codec::schema_digest(&schema).unwrap();
    let inputs = left
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .chain(right.table_payload().unwrap().batches());
    let independent = owned_capacity(inputs.clone());
    let encoded = inputs
        .clone()
        .map(|record| codec::encode_batch(record, 256 << 20, &mut Vec::new()).unwrap())
        .collect::<Vec<_>>();
    let decoded = encoded
        .iter()
        .map(|bytes| {
            codec::decode_table_batch(bytes, &digest, &schema, SOURCE_ROWS as u64).unwrap()
        })
        .collect::<Vec<_>>();
    for ((input, record), bytes) in inputs.zip(&decoded).zip(&encoded) {
        assert_eq!(record, input);
        let owners = backing_owners(record);
        assert_eq!(owners.len(), 1);
        let backing = *owners.values().next().unwrap();
        assert!(backing >= usize::try_from(codec::payload_body_bytes(bytes).unwrap()).unwrap());
        assert!(record.get_array_memory_size() >= 17 * backing);
    }
    let restored = owned_capacity(decoded.iter());
    let encoded_capacity = encoded.iter().map(Vec::capacity).sum::<usize>();
    let limit = independent.max(restored) + encoded_capacity + (256 << 10);
    assert!(limit < 256 << 20);
    assert!(decoded[0].get_array_memory_size() > limit);
    RestoredWorkspaceFixture {
        schema,
        left,
        right,
        decoded,
        limit,
    }
}

fn workspace_operator(fixture: &RestoredWorkspaceFixture, narrow: bool) -> StreamAsofJoinOperator {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(1_000, fixture.limit as u64).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamAsofJoinOperator::new("asof", fixture.schema.clone(), fixture.schema.clone(), spec)
            .unwrap();
    if narrow {
        operator.set_output_projection(vec![2, 11]).unwrap();
    }
    operator
}

async fn admit_workspace_input(
    operator: &mut StreamAsofJoinOperator,
    fixture: &RestoredWorkspaceFixture,
    context: &StreamOperatorContext<'_>,
) {
    let configured = operator.runtime.pool.reserved();
    let left_metadata = fixture.left.metadata().clone();
    let right_metadata = fixture.right.metadata().clone();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", fixture.right.clone(), context, &mut output)
        .await
        .unwrap();
    operator
        .process_data("left", fixture.left.clone(), context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    assert_eq!(fixture.left.metadata(), &left_metadata);
    assert_eq!(fixture.right.metadata(), &right_metadata);
    assert_eq!(
        operator.status.pending_left_rows,
        (SOURCE_ROWS * SOURCE_BATCHES) as u64
    );
    tokio::time::timeout(Duration::from_secs(2), async {
        while operator.runtime.pool.reserved()
            != configured + operator.state.right.auxiliary_bytes()
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

fn expected_workspace_output(schema: &SchemaRef, start: usize, narrow: bool) -> RecordBatch {
    let fields = [("left", false), ("right", true)]
        .into_iter()
        .flat_map(|(prefix, nullable)| {
            schema.fields().iter().map(move |field| {
                field
                    .as_ref()
                    .clone()
                    .with_name(format!("{prefix}__{}", field.name()))
                    .with_nullable(nullable || field.is_nullable())
            })
        })
        .collect::<Vec<_>>();
    let left = (start..start + OUTPUT_ROWS).map(Some).collect::<Vec<_>>();
    let right = (start..start + OUTPUT_ROWS)
        .map(|row| (row % 7 != 0).then_some(row))
        .collect::<Vec<_>>();
    let columns = workspace_columns(&left, true)
        .into_iter()
        .chain(workspace_columns(&right, false))
        .collect();
    let expected = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    if narrow {
        expected.project(&[2, 11]).unwrap()
    } else {
        expected
    }
}

fn assert_workspace_output(output: &[Batch], fixture: &RestoredWorkspaceFixture, narrow: bool) {
    assert_eq!(output.len(), SOURCE_ROWS * SOURCE_BATCHES / OUTPUT_ROWS);
    for (index, batch) in output.iter().enumerate() {
        let start = index * OUTPUT_ROWS;
        assert_eq!(
            batch.metadata(),
            &BatchMetadata::new("asof", start as u64, JsonMap::new()).unwrap(),
        );
        assert_eq!(batch.table_payload().unwrap().batches().len(), 1);
        assert_eq!(
            batch.table_payload().unwrap().batches()[0],
            expected_workspace_output(&fixture.schema, start, narrow),
        );
    }
}

fn assert_workspace_failure(error: &CalcFlowError) {
    assert!(
        matches!(
            error,
            CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
                ..
            }
        ),
        "unexpected error: {error:?}"
    );
}

async fn assert_restored_workspace_output(narrow: bool) {
    let fixture = restored_workspace_fixture();
    assert_eq!(fixture.decoded.len(), SOURCE_BATCHES * 2);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None)
        .with_test_output_budget(EdgeBudget::new(OUTPUT_ROWS, 256 << 20).unwrap());
    let mut admitted = workspace_operator(&fixture, narrow);
    admit_workspace_input(&mut admitted, &fixture, &context).await;
    let snapshot = admitted.capture(Epoch::INITIAL).unwrap();
    let mut restored = workspace_operator(&fixture, narrow);
    restored.restore(&snapshot).unwrap();
    assert_eq!(
        restored.capture(Epoch::INITIAL).unwrap().segments,
        snapshot.segments
    );
    for (payload, _) in restored.state.batches.values() {
        assert_eq!(backing_owners(&payload.record).len(), 1);
    }
    let mut independent_output = EdgeCollector::new(admitted.output_ports().to_vec());
    admitted
        .on_watermark(
            EventTime::from_micros(1_000),
            &context,
            &mut independent_output,
        )
        .await
        .unwrap();
    let independent_output = independent_output
        .drain("output")
        .into_iter()
        .map(|message| message.as_data().unwrap().clone())
        .collect::<Vec<_>>();
    assert_workspace_output(&independent_output, &fixture, narrow);
    assert_log_funded(&admitted);
    let before_status = restored.status.clone();
    let before_sequence = restored.next_output_sequence;
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    let result = restored
        .on_watermark(EventTime::from_micros(1_000), &context, &mut output)
        .await;
    let output = output
        .drain("output")
        .into_iter()
        .map(|message| message.as_data().unwrap().clone())
        .collect::<Vec<_>>();
    if let Err(error) = &result {
        assert_workspace_failure(error);
        let mut expected_status = before_status;
        expected_status.workspace_limit_failures += 1;
        assert_eq!(restored.status, expected_status);
        assert_eq!(restored.next_output_sequence, before_sequence);
        assert_eq!(
            restored.capture(Epoch::INITIAL).unwrap().segments,
            snapshot.segments
        );
        assert!(output.is_empty());
        assert_eq!(
            restored.runtime.pool.reserved(),
            restored.state.right.auxiliary_bytes()
        );
    }
    assert!(
        result.is_ok(),
        "valid restored IPC output must fit unique backing budget: {result:?}"
    );
    assert_workspace_output(&output, &fixture, narrow);
    for (actual, expected) in output.iter().zip(&independent_output) {
        assert_eq!(actual.metadata(), expected.metadata());
        assert_eq!(
            actual.table_payload().unwrap().batches(),
            expected.table_payload().unwrap().batches(),
        );
    }
    assert_eq!(
        restored.next_output_sequence,
        (SOURCE_ROWS * SOURCE_BATCHES) as u64
    );
    assert_eq!(restored.status.pending_left_rows, 0);
    assert_log_funded(&restored);
    restored.reset().unwrap();
    admitted.reset().unwrap();
    let restored_pool = restored.runtime.pool.clone();
    let admitted_pool = admitted.runtime.pool.clone();
    drop(snapshot);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    drop(restored);
    drop(admitted);
    assert_eq!(restored_pool.reserved(), 0);
    assert_eq!(admitted_pool.reserved(), 0);
}

#[tokio::test]
async fn restored_ipc_narrow_output_fits_unique_owned_backing_budget() {
    assert_restored_workspace_output(true).await;
}

#[tokio::test]
async fn restored_ipc_full_output_fits_unique_owned_backing_budget() {
    assert_restored_workspace_output(false).await;
}

#[tokio::test]
async fn restored_source_rejects_a_pool_smaller_than_its_full_backing() {
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
    let fixture = restored_workspace_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut admitted = workspace_operator(&fixture, false);
    admit_workspace_input(&mut admitted, &fixture, &context).await;
    let snapshot = admitted.capture(Epoch::INITIAL).unwrap();
    let mut restored = workspace_operator(&fixture, false);
    restored.restore(&snapshot).unwrap();
    let payload = restored
        .state
        .batches
        .values()
        .find(|(payload, _)| payload.key == (0, 0))
        .unwrap()
        .0
        .clone();
    let capacity = owned_capacity(std::iter::once(payload.record.as_ref()));
    let row = state::RowPayload {
        batch: payload,
        row: 0,
    };
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(capacity - 1));
    let mut credit = MemoryConsumer::new("too-small-source-pool").register(&pool);
    let selected = [vec![2], vec![3]];
    let mut builder =
        output_plan::OutputPlanBuilder::new(1, Some(&selected), &mut credit, "asof").unwrap();
    let initial = credit.size();
    let owners = Arc::strong_count(row.batch.record.column(2));
    let error = builder
        .push(row.view(), None, &mut credit, "asof")
        .unwrap_err();
    assert_workspace_failure(&error);
    assert_eq!(credit.size(), initial);
    assert_eq!(Arc::strong_count(row.batch.record.column(2)), owners);
    drop((builder, credit));
    assert_eq!(pool.reserved(), 0);
    restored.reset().unwrap();
    admitted.reset().unwrap();
    let restored_pool = restored.runtime.pool.clone();
    let admitted_pool = admitted.runtime.pool.clone();
    drop(row);
    drop(snapshot);
    drop(restored);
    drop(admitted);
    assert_eq!(restored_pool.reserved(), 0);
    assert_eq!(admitted_pool.reserved(), 0);
}

#[tokio::test]
async fn occupied_workspace_rejects_atomically_and_refunds_only_owned_lease() {
    let fixture = restored_workspace_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None)
        .with_test_output_budget(EdgeBudget::new(OUTPUT_ROWS, 256 << 20).unwrap());
    let mut operator = workspace_operator(&fixture, true);
    admit_workspace_input(&mut operator, &fixture, &context).await;
    let before = operator.capture(Epoch::INITIAL).unwrap();
    let before_status = operator.status.clone();
    let reserved = operator.runtime.pool.reserved();
    let occupied =
        MemoryConsumer::new("owned-workspace-negative-control").register(&operator.runtime.pool);
    occupied.try_grow(fixture.limit - reserved - 1).unwrap();
    let paid = operator.runtime.pool.reserved();
    assert_eq!(paid, fixture.limit - 1);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let error = operator
        .on_watermark(EventTime::from_micros(1_000), &context, &mut output)
        .await
        .unwrap_err();
    assert_workspace_failure(&error);
    let mut expected_status = before_status;
    expected_status.workspace_limit_failures += 1;
    assert_eq!(operator.status, expected_status);
    assert_eq!(operator.next_output_sequence, 0);

    assert!(output.drain("output").is_empty());
    assert_eq!(operator.runtime.pool.reserved(), paid);
    drop(occupied);
    assert_eq!(operator.runtime.pool.reserved(), reserved);
    assert_eq!(
        operator.capture(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
    operator
        .on_watermark(EventTime::from_micros(1_000), &context, &mut output)
        .await
        .unwrap();
    let batches = output
        .drain("output")
        .into_iter()
        .map(|message| message.as_data().unwrap().clone())
        .collect::<Vec<_>>();
    assert_workspace_output(&batches, &fixture, true);
    operator.reset().unwrap();
    let pool = operator.runtime.pool.clone();
    drop(before);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}

struct RejectWorkspaceOutput {
    calls: usize,
}

#[async_trait]
impl StreamCollector for RejectWorkspaceOutput {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        self.calls += 1;
        Err(CalcFlowError::Internal {
            message: "workspace-prefix-rejected".into(),
        })
    }
}

#[tokio::test]
async fn rejected_workspace_prefix_preserves_snapshot_sequence_and_refunds() {
    let fixture = restored_workspace_fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None)
        .with_test_output_budget(EdgeBudget::new(OUTPUT_ROWS, 256 << 20).unwrap());
    let mut operator = workspace_operator(&fixture, true);
    admit_workspace_input(&mut operator, &fixture, &context).await;
    let before = operator.capture(Epoch::INITIAL).unwrap();
    let before_status = operator.status.clone();
    let before_reserved = operator.runtime.pool.reserved();
    let mut output = RejectWorkspaceOutput { calls: 0 };
    let error = operator
        .on_watermark(EventTime::from_micros(1_000), &context, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error, CalcFlowError::Internal { message } if message == "workspace-prefix-rejected"
    ));
    assert_eq!(output.calls, 1);
    assert_eq!(operator.status, before_status);
    assert_eq!(operator.next_output_sequence, 0);
    let funding = job.gather_owner().funding();
    assert_eq!(
        operator.runtime.pool.reserved(),
        before_reserved + funding.0 + funding.1
    );
    assert_eq!(
        operator.capture(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
    operator.reset().unwrap();
    let pool = operator.runtime.pool.clone();
    drop(before);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}
