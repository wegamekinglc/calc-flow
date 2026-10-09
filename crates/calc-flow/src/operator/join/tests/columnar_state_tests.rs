use super::*;
use datafusion::arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Date64Array, Int8Array, LargeStringArray, UInt8Array,
};
use std::{
    collections::BTreeSet,
    future::Future,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Wake, Waker},
};

struct CopyWake(AtomicUsize);
impl Wake for CopyWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn operator_fixture() -> StreamJoinOperator {
    StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap()
}

fn spare_text(value: &str, capacity: usize) -> String {
    let mut text = String::with_capacity(capacity);
    text.push_str(value);
    text
}

fn metadata_schema(capacity: usize) -> SchemaRef {
    let metadata = || HashMap::with_capacity(capacity / 64);
    let fields = left_schema()
        .fields()
        .iter()
        .map(|field| {
            Field::new(
                spare_text(field.name(), capacity),
                field.data_type().clone(),
                field.is_nullable(),
            )
            .with_metadata(metadata())
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new_with_metadata(fields, metadata()))
}

fn historical_metadata() -> HashMap<String, String> {
    let mut values = HashMap::with_capacity(4_096);
    for index in 0..1_024 {
        values.insert(format!("removed-{index}"), String::new());
    }
    values.clear();
    values.insert(spare_text("owner", 65_536), spare_text("same", 65_536));
    values
}

#[tokio::test]
async fn test_nonempty_metadata_history_keeps_exact_legacy_and_caller_readonly() {
    let schema = metadata_history_schema(1);
    let schema_capacity = schema.metadata().capacity();
    let field_capacity = schema.fields()[0].metadata().capacity();
    assert!(schema_capacity >= 4_096);
    assert!(field_capacity >= 4_096);
    assert_exact_legacy_metadata(Arc::clone(&schema)).await;
    assert_eq!(schema.metadata().capacity(), schema_capacity);
    assert_eq!(schema.fields()[0].metadata().capacity(), field_capacity);
    assert_eq!(schema.metadata()["entry-0"], "値-0");
    assert_eq!(schema.fields()[0].metadata()["entry-0"], "値-0");
}

fn metadata_history_schema(entries: usize) -> SchemaRef {
    let metadata = || {
        let mut values = historical_metadata();
        values.clear();
        for index in 0..entries {
            values.insert(format!("entry-{index}"), format!("値-{index}"));
        }
        values
    };
    let fields = left_schema()
        .fields()
        .iter()
        .map(|field| {
            Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                .with_metadata(metadata())
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new_with_metadata(fields, metadata()))
}

fn measure_owned_metadata_history(entries: usize) {
    let schema = metadata_history_schema(entries);
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        left_batch(vec![10]).table_payload().unwrap().batches()[0]
            .columns()
            .to_vec(),
    )
    .unwrap();
    let legacy_ipc = row_ipc::RowIpcEncoder::default()
        .encode(&record, "match", "left")
        .unwrap();
    let mut operator = StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let selected = [columnar::SelectedRow {
        source: 0,
        row_id: 0,
        time: EventTime::from_micros(10),
        retain: true,
    }];
    let mut quantum = columnar::Quantum::default();
    let (chunk, measured) =
        measure_copy_future(operator.owned_payload(&record, 0, &selected, &context, &mut quantum));
    let chunk = chunk.unwrap().unwrap();
    let payload = columnar::RowPayload::at(&record, Some(&chunk), 0);
    let paid = payload.funded_owner().unwrap().1;
    eprintln!("owned-copy poll allocations={measured:?}, actual guard={paid}");
    assert!(measured.bytes_current <= i64::try_from(paid).unwrap());
    assert!(
        measured.bytes_max <= paid as u64,
        "{entries} entries: actual peak={} exceeds guard={paid}",
        measured.bytes_max
    );
    assert_eq!(
        payload.schema_ref().metadata(),
        record.schema_ref().metadata()
    );
    let view = payload.view();
    assert_eq!(
        row_ipc::RowIpcEncoder::default()
            .encode(&view, "match", "left")
            .unwrap(),
        legacy_ipc
    );
    drop(view);
    drop(payload);
    let released = allocation_counter::measure(|| drop(chunk));
    assert_eq!(measured.bytes_current + released.bytes_current, 0);
    assert_eq!(pool.reserved(), 0);
}

fn measure_copy_future<F: Future>(future: F) -> (F::Output, allocation_counter::AllocationInfo) {
    let mut future = std::pin::pin!(future);
    let wake = Arc::new(CopyWake(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&wake));
    let mut context = Context::from_waker(&waker);
    let mut output = None;
    let measured = allocation_counter::measure(|| {
        loop {
            if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
                output = Some(value);
                break;
            }
        }
    });
    // Synchronous polling outside a Tokio task has an unconstrained cooperative budget.
    assert_eq!(wake.0.load(Ordering::Relaxed), 0);
    (output.unwrap(), measured)
}

#[test]
fn test_empty_metadata_history_allocations_are_actually_funded() {
    measure_owned_metadata_history(0);
}

#[tokio::test]
async fn test_unproved_metadata_normalization_keeps_exact_legacy() {
    assert_exact_legacy_metadata(metadata_history_schema(33)).await;
}

#[tokio::test]
async fn test_sparse_schema_or_field_metadata_uses_exact_legacy() {
    let fields = left_schema()
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            let field = Field::new(field.name(), field.data_type().clone(), field.is_nullable());
            if index == 0 {
                field.with_metadata(historical_metadata())
            } else {
                field
            }
        })
        .collect::<Vec<_>>();
    let field_metadata = Arc::new(Schema::new(fields));
    let schema_metadata = Arc::new(Schema::new_with_metadata(
        left_schema().fields().clone(),
        historical_metadata(),
    ));
    for schema in [schema_metadata, field_metadata] {
        assert_exact_legacy_metadata(schema).await;
    }
}

async fn assert_exact_legacy_metadata(schema: SchemaRef) {
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        left_batch(vec![10]).table_payload().unwrap().batches()[0]
            .columns()
            .to_vec(),
    )
    .unwrap();
    let legacy_ipc = row_ipc::RowIpcEncoder::default()
        .encode(&record, "match", "left")
        .unwrap();
    let key = encode_join_key_v1(&record, 0, &[0]).unwrap();
    let charge = state_row_charge_with_key(&record, 0, key.len(), "match").unwrap();
    let mut operator =
        StreamJoinOperator::new("match", Arc::clone(&schema), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let payload = &operator.state.left[0].record;
    assert!(matches!(payload, columnar::RowPayload::Legacy(_)));
    assert!(Arc::ptr_eq(payload.schema_ref(), &schema));
    assert_eq!(
        row_ipc::RowIpcEncoder::default()
            .encode(&payload.view(), "match", "left")
            .unwrap(),
        legacy_ipc
    );
    assert_eq!(operator.state.next_left_row_id, 1);
    assert_eq!(operator.status().left.retained_bytes, charge);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_shared_payload_releases_equal_caller_schema_and_type_owners() {
    let canonical = metadata_schema(0);
    let caller = metadata_schema(1 << 20);
    assert_eq!(caller, canonical);
    let schema_owner = Arc::downgrade(&caller);
    let field_owner = Arc::downgrade(&caller.fields()[1]);
    let timezone: Arc<str> = Arc::from("UTC");
    let type_owner = Arc::downgrade(&timezone);
    let record = RecordBatch::try_new(
        caller.clone(),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![10]).with_timezone(timezone.clone())),
            Arc::new(Int64Array::from(vec![42])),
        ],
    )
    .unwrap();
    let legacy_ipc = row_ipc::RowIpcEncoder::default()
        .encode(&record, "match", "left")
        .unwrap();
    let encoded_key = encode_join_key_v1(&record, 0, &[0]).unwrap();
    let charge = state_row_charge_with_key(&record, 0, encoded_key.len(), "match").unwrap();
    let mut operator =
        StreamJoinOperator::new("match", canonical.clone(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    operator
        .process_data("left", batch.clone(), &context, &mut collector)
        .await
        .unwrap();
    let payload = &operator.state.left[0].record;
    assert!(matches!(payload, columnar::RowPayload::Shared { .. }));
    assert!(payload.funded_owner().unwrap().1 < (1 << 20));
    assert_eq!(payload.schema_ref().clone(), caller);
    assert_eq!(operator.status().left.retained_bytes, charge);
    let captured = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(
        captured.segments["left-delta-1"]
            .bytes()
            .ends_with(&legacy_ipc)
    );
    assert!(
        batch.table_payload().unwrap().schema().fields()[0]
            .name()
            .capacity()
            >= (1 << 20)
    );
    assert!(
        batch
            .table_payload()
            .unwrap()
            .schema()
            .metadata()
            .is_empty()
    );
    drop(batch);
    drop(caller);
    drop(timezone);
    assert!(
        schema_owner.upgrade().is_none(),
        "shared chunk retains the independently allocated caller schema despite its sub-MiB credit"
    );
    assert!(field_owner.upgrade().is_none());
    assert!(type_owner.upgrade().is_none());
    assert!(!Arc::ptr_eq(
        operator.state.left[0].record.schema_ref(),
        &canonical
    ));
    assert_eq!(operator.state.left[0].record.schema_ref(), &canonical);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    assert_eq!(
        pool.reserved(),
        native_lookup_tests::state_funding(&operator)
    );
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}

async fn copied_test_chunk(
    operator: &mut StreamJoinOperator,
    record: &RecordBatch,
    port: usize,
) -> Option<Arc<columnar::PayloadChunk>> {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    copied_test_chunk_with_context(operator, record, port, &context).await
}

async fn copied_test_chunk_with_context(
    operator: &mut StreamJoinOperator,
    record: &RecordBatch,
    port: usize,
    context: &StreamOperatorContext<'_>,
) -> Option<Arc<columnar::PayloadChunk>> {
    let mut quantum = columnar::Quantum::default();
    let plan = operator
        .side_plan(if port == 0 { "left" } else { "right" })
        .unwrap();
    if !operator
        .can_copy_payload(record, &plan, context, &mut quantum)
        .await
        .unwrap()
    {
        return None;
    }
    let mut selected = columnar::CopySelection::reserve(operator, record.num_rows())
        .unwrap()
        .unwrap();
    selected
        .rows
        .extend((0..record.num_rows()).map(|source| columnar::SelectedRow {
            source,
            row_id: source as u64,
            time: EventTime::from_micros(i64::try_from(source).unwrap()),
            retain: true,
        }));
    operator
        .owned_payload(record, port, &selected.rows, context, &mut quantum)
        .await
        .unwrap()
}

async fn shared_preflight_fixture(
    operator: &mut StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
) -> (Vec<AdmittedRow>, Vec<StoredRow>) {
    let left = left_batch((0..1_000).collect());
    let right = right_batch((0..1_000).collect());
    let left = &left.table_payload().unwrap().batches()[0];
    let right = &right.table_payload().unwrap().batches()[0];
    let left_chunk = copied_test_chunk_with_context(operator, left, 0, context)
        .await
        .unwrap();
    let right_chunk = copied_test_chunk_with_context(operator, right, 1, context)
        .await
        .unwrap();
    let admitted = (0..1_000)
        .map(|row| AdmittedRow {
            record: columnar::RowPayload::at(left, Some(&left_chunk), row),
            event_time: EventTime::from_micros(i64::try_from(row).unwrap()),
            row_id: row as u64,
            retain: true,
        })
        .collect();
    let opposite = (0..1_000)
        .map(|row| StoredRow {
            record: columnar::RowPayload::at(right, Some(&right_chunk), row),
            event_time: EventTime::from_micros(i64::try_from(row).unwrap()),
            row_id: row as u64,
            charge: 0,
            encoded_key: Arc::new(vec![].into()),
        })
        .collect();
    (admitted, opposite)
}

#[test]
fn test_shared_flat_preflight_reuses_scratch_under_fanout() {
    checkpoint_compaction_tests::isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(shared_flat_preflight_fanout(service));
    });
}

async fn shared_flat_preflight_fanout(
    service: &crate::runtime::streaming::gather_work::TestService,
) {
    let job = job().with_gather_owner(service.owner("shared-preflight-funding".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut operator = operator_fixture();
    let (admitted, opposite) = shared_preflight_fixture(&mut operator, &context).await;
    let schema = operator.output_ports()[0].schema().unwrap();
    let mut allocations = Vec::new();
    for fanout in [1, 10] {
        let matched = (0..1_000)
            .flat_map(|pos| {
                (0..fanout).map(move |offset| MatchedPair {
                    pos,
                    opposite_index: (pos + offset) % 1_000,
                })
            })
            .collect::<Vec<_>>();
        let output = materialization::JoinOutput {
            schema,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "shared-allocation",
            admitted_charges: None,
        };
        let measured = allocation_counter::measure(|| {
            let ranges = output
                .ranges(EdgeBudget::new(1_000, 1 << 20).unwrap())
                .unwrap();
            assert_eq!(ranges.len(), fanout);
            assert_eq!(
                ranges.iter().map(std::ops::Range::len).sum::<usize>(),
                matched.len()
            );
        });
        allocations.push(measured.count_total);
    }
    assert!(
        allocations[1] <= allocations[0] + 64,
        "shared preflight allocations scale with cached fanout pairs: {allocations:?}"
    );
    assert!(
        allocations[1] < 1_000,
        "per-row Shared scratch remains: {allocations:?}"
    );
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    native_lookup_tests::assert_resident_and_gather_funding(
        pool.as_ref(),
        &job,
        native_lookup_tests::payload_funding(
            admitted
                .iter()
                .map(|row| &row.record)
                .chain(opposite.iter().map(|row| &row.record)),
        ),
    );
    drop(admitted);
    drop(opposite);
    drop(operator);
    native_lookup_tests::assert_resident_and_gather_funding(pool.as_ref(), &job, 0);
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
    assert_eq!(pool.reserved(), home);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn owner_proof_schema(columns: usize) -> SchemaRef {
    let mut fields = left_schema().fields()[..2].to_vec();
    fields.extend((2..columns).map(|index| {
        let data_type = match index % 4 {
            0 => DataType::Int32,
            1 => DataType::UInt8,
            2 => DataType::Utf8,
            _ => DataType::LargeUtf8,
        };
        Arc::new(Field::new(format!("payload_{index}"), data_type, true))
    }));
    Arc::new(Schema::new(fields))
}

fn owner_proof_array(data_type: &DataType) -> ArrayRef {
    match data_type {
        DataType::Int64 => Arc::new(Int64Array::from(vec![7, 7, 7, 7])),
        DataType::Timestamp(_, timezone) => Arc::new(
            TimestampMicrosecondArray::from(vec![0, 1, 2, 3]).with_timezone_opt(timezone.clone()),
        ),
        DataType::Int32 => Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
        DataType::UInt8 => Arc::new(UInt8Array::from(vec![1, 2, 3, 4])),
        DataType::Utf8 => Arc::new(StringArray::from(vec!["é", "", "abc", ""])),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vec!["é", "", "abc", ""])),
        _ => unreachable!("owner proof only uses allowed flat types"),
    }
}

fn measure_shared_owner_controls(columns: usize) {
    let schema = owner_proof_schema(columns);
    let mut operator =
        StreamJoinOperator::new("match", schema.clone(), right_schema(), spec()).unwrap();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let mut source = None;
    let created = allocation_counter::measure(|| {
        let arrays = schema
            .fields()
            .iter()
            .map(|field| owner_proof_array(field.data_type()))
            .collect();
        source = Some(RecordBatch::try_new(schema.clone(), arrays).unwrap());
    });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(tokio::task::yield_now());
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let selected = (0..source.as_ref().unwrap().num_rows())
        .map(|source| columnar::SelectedRow {
            source,
            row_id: u64::try_from(source).unwrap(),
            time: EventTime::from_micros(i64::try_from(source).unwrap()),
            retain: true,
        })
        .collect::<Vec<_>>();
    let record = source.as_ref().unwrap();
    let mut warm_quantum = columnar::Quantum::default();
    let warm = runtime
        .block_on(operator.owned_payload(record, 0, &selected, &context, &mut warm_quantum))
        .unwrap()
        .unwrap();
    drop(warm);
    let warm_funding = job.gather_owner().funding();
    assert_eq!(warm_funding.2, 0);
    assert_eq!(pool.reserved(), warm_funding.0 + warm_funding.1);
    columnar::observe_string_allocations();
    let mut payload = None;
    let reconstruction = allocation_counter::measure(|| {
        let record = source.as_ref().unwrap();
        let mut quantum = columnar::Quantum::default();
        let shared = runtime
            .block_on(operator.owned_payload(record, 0, &selected, &context, &mut quantum))
            .unwrap()
            .unwrap();
        payload = Some(columnar::RowPayload::at(record, Some(&shared), 0));
    });
    let workers = columnar::take_string_allocations();
    let paid = payload.as_ref().unwrap().funded_owner().unwrap().1;
    let source_drop = allocation_counter::measure(|| drop(source.take()));
    let backing_owners = created.bytes_current + source_drop.bytes_current;
    assert!(backing_owners >= 0);
    let peak = u64::try_from(backing_owners).unwrap()
        + reconstruction.bytes_max
        + workers.worker.bytes_max
        + u64::try_from(workers.output_box).unwrap();
    let live = backing_owners + reconstruction.bytes_current - workers.dispatch.bytes_current
        + workers.worker.bytes_current;
    assert!(
        peak <= (paid + workers.max_credit) as u64 && live <= i64::try_from(paid).unwrap(),
        "{columns} columns: peak={peak}, live={live}, payload={paid}, workers={workers:?}, warm={warm_funding:?}"
    );
    eprintln!(
        "{columns} columns: peak={peak}, live={live}, payload={paid}, workers={workers:?}, warm={warm_funding:?}"
    );
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert_eq!(pool.reserved(), paid + home + generation);
    let released = allocation_counter::measure(|| drop(payload.take()));
    assert_eq!(live + released.bytes_current, 0);
    drop(context);
    assert!(
        runtime
            .block_on(job.gather_owner().close_and_drain())
            .is_empty()
    );
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_shared_payload_native_owner_controls_remain_actually_funded() {
    for columns in [3, 16, 128] {
        measure_shared_owner_controls(columns);
    }
}

#[tokio::test]
async fn test_shared_payload_does_not_borrow_positive_capacity_opaque_bytes_owner() {
    use datafusion::arrow::buffer::{Buffer, ScalarBuffer};
    use tokio_util::bytes::Bytes;

    struct OpaqueTimes {
        visible: Buffer,
        _hidden: Vec<u8>,
        _live: Arc<()>,
    }
    impl AsRef<[u8]> for OpaqueTimes {
        fn as_ref(&self) -> &[u8] {
            &self.visible
        }
    }
    let live = Arc::new(());
    let weak = Arc::downgrade(&live);
    let bytes = Bytes::from_owner(OpaqueTimes {
        visible: Buffer::from_vec(vec![10_i64]),
        _hidden: vec![0; 1 << 20],
        _live: live,
    });
    let values = ScalarBuffer::new(Buffer::from(bytes), 0, 1);
    assert_eq!(values.inner().capacity(), size_of::<i64>());
    let record = RecordBatch::try_new(
        left_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::new(values, None).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![42])),
        ],
    )
    .unwrap();
    let mut operator = operator_fixture();
    let shared = copied_test_chunk(&mut operator, &record, 0).await;
    let paid = shared.as_ref().map(|chunk| {
        columnar::RowPayload::at(&record, Some(chunk), 0)
            .funded_owner()
            .unwrap()
            .1
    });
    drop(record);
    assert!(
        weak.upgrade().is_none(),
        "core copy retains opaque owner despite positive capacity8; credit={paid:?}"
    );
    assert!(shared.is_some());
    drop(shared);
    assert!(weak.upgrade().is_none());
    assert_eq!(
        operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool()
            .reserved(),
        0
    );
}

fn preflight_ranges(
    schema: &SchemaRef,
    admitted: &[AdmittedRow],
    opposite: &[StoredRow],
    matched: &[MatchedPair],
    budget: EdgeBudget,
) -> Vec<std::ops::Range<usize>> {
    materialization::JoinOutput {
        schema,
        admitted,
        opposite,
        matched,
        incoming_is_left: true,
        operator_id: "offset-oracle",
        admitted_charges: None,
    }
    .ranges(budget)
    .unwrap()
}

#[tokio::test]
async fn test_shared_flat_preflight_matches_legacy_offsets_and_variable_values() {
    let mut operator = operator_fixture();
    let left = RecordBatch::try_new(
        left_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; 4])),
            Arc::new(TimestampMicrosecondArray::from(vec![0, 1, 2, 3]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![5, 6, 7, 8])),
        ],
    )
    .unwrap()
    .slice(1, 3);
    let right = RecordBatch::try_new(
        right_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; 4])),
            Arc::new(TimestampMicrosecondArray::from(vec![0, 1, 2, 3]).with_timezone("UTC")),
            Arc::new(StringArray::from(vec!["skip", "é", "", "longer-value"])),
        ],
    )
    .unwrap()
    .slice(1, 3);
    let left_chunk = copied_test_chunk(&mut operator, &left, 0).await.unwrap();
    let right_chunk = copied_test_chunk(&mut operator, &right, 1).await.unwrap();
    let mut admitted = (0..3)
        .map(|row| AdmittedRow {
            record: columnar::RowPayload::at(&left, Some(&left_chunk), row),
            event_time: EventTime::from_micros(i64::try_from(row).unwrap()),
            row_id: row as u64,
            retain: true,
        })
        .collect::<Vec<_>>();
    let mut opposite = (0..3)
        .map(|row| StoredRow {
            record: columnar::RowPayload::at(&right, Some(&right_chunk), row),
            event_time: EventTime::from_micros(i64::try_from(row).unwrap()),
            row_id: row as u64,
            charge: 0,
            encoded_key: Arc::new(vec![].into()),
        })
        .collect::<Vec<_>>();
    let matched = (0..9)
        .map(|index| MatchedPair {
            pos: index % 3,
            opposite_index: index / 3,
        })
        .collect::<Vec<_>>();
    let budgets = [
        EdgeBudget::new(1, 64).unwrap(),
        EdgeBudget::new(2, 100).unwrap(),
        EdgeBudget::new(4, 1 << 20).unwrap(),
    ];
    let schema = operator.output_ports()[0].schema().unwrap();
    let shared = budgets
        .iter()
        .map(|&budget| preflight_ranges(schema, &admitted, &opposite, &matched, budget))
        .collect::<Vec<_>>();
    for row in &mut admitted {
        row.record = left.slice(usize::try_from(row.row_id).unwrap(), 1).into();
    }
    for row in &mut opposite {
        row.record = right.slice(usize::try_from(row.row_id).unwrap(), 1).into();
    }
    let legacy = budgets
        .iter()
        .map(|&budget| preflight_ranges(schema, &admitted, &opposite, &matched, budget))
        .collect::<Vec<_>>();
    assert_eq!(shared, legacy);
}

fn left_values_batch(times: Vec<i64>, values: Vec<i64>) -> Batch {
    let record = RecordBatch::try_new(
        left_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; times.len()])),
            Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn expected_record(
    schema: SchemaRef,
    left_times: &[i64],
    left_values: &[i64],
    right_times: &[i64],
) -> RecordBatch {
    let rows = right_times
        .iter()
        .flat_map(|right| {
            left_times
                .iter()
                .zip(left_values)
                .map(move |(left, value)| (*left, *value, *right))
        })
        .collect::<Vec<_>>();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![7; rows.len()])),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(rows.iter().map(|row| row.0))
                    .with_timezone("UTC"),
            ),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.1))),
            Arc::new(Int64Array::from(vec![7; rows.len()])),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(rows.iter().map(|row| row.2))
                    .with_timezone("UTC"),
            ),
            Arc::new(StringArray::from(vec!["paid"; rows.len()])),
        ],
    )
    .unwrap()
}

fn assert_output(
    collector: &mut EdgeCollector,
    operator: &StreamJoinOperator,
    left_times: &[i64],
    left_values: &[i64],
    right_times: &[i64],
    sequence: u64,
) {
    let messages = collector.drain("output");
    assert_eq!(messages.len(), 1);
    let batch = messages[0].as_data().unwrap();
    assert_eq!(batch.metadata().sequence(), sequence);
    let schema = operator.output_ports()[0].schema().unwrap().clone();
    let expected = expected_record(schema, left_times, left_values, right_times);
    assert_eq!(batch.table_payload().unwrap().batches(), &[expected]);
}

#[tokio::test]
async fn test_native_lookup_avoids_legacy_sql_probe_tables() {
    let mut operator = operator_fixture();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            left_values_batch(vec![4, 1, 1, 3], vec![40, 10, 11, 30]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    reset_join_work();
    operator
        .process_data("right", right_batch(vec![2, 0]), &context, &mut collector)
        .await
        .unwrap();
    assert_output(
        &mut collector,
        &operator,
        &[1, 1, 3, 4],
        &[10, 11, 30, 40],
        &[2, 0],
        0,
    );
    assert_eq!(operator.state.next_left_row_id, 4);
    assert_eq!(operator.state.next_right_row_id, 2);
    assert_eq!(operator.status().emitted_match_rows, 8);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert!(operator.retained_key_cache.left.is_none());
}

fn retained_record_containers(operator: &StreamJoinOperator) -> usize {
    let retained = operator
        .state
        .left
        .iter()
        .map(|row| row.record.columns().as_ptr() as usize);
    let dirty = operator
        .state
        .deltas
        .pending
        .iter()
        .filter_map(|op| match op {
            PendingOp::Upsert {
                side: JoinSide::Left,
                record,
                ..
            } => Some(record.columns().as_ptr() as usize),
            PendingOp::Upsert {
                side: JoinSide::Right,
                ..
            }
            | PendingOp::Tombstone { .. } => None,
        });
    retained.chain(dirty).collect::<BTreeSet<_>>().len()
}

#[tokio::test]
async fn test_retained_and_dirty_record_containers_scale_with_batches() {
    let mut operator = operator_fixture();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    for base in [0, 48] {
        operator
            .process_data(
                "left",
                left_batch((base..base + 48).collect()),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
    }
    assert_eq!(operator.status().left.retained_rows, 96);
    assert!(collector.drain("output").is_empty());
    assert!(
        retained_record_containers(&operator) <= 2,
        "two source batches must not leave one payload record per retained/dirty row"
    );
}

fn wire(snapshot: &OperatorStateSnapshot) -> Value {
    serde_json::json!({
        "inline_metadata": snapshot.inline_metadata,
        "segments": snapshot.segments.iter()
            .map(|(name, segment)| (name.clone(), hex::encode(segment.bytes())))
            .collect::<BTreeMap<_, _>>(),
    })
}

async fn resume_capture(snapshot: &OperatorStateSnapshot, left_times: &[i64]) -> usize {
    let mut operator = operator_fixture();
    operator.restore(snapshot).unwrap();
    let bytes = operator.status().left.retained_bytes;
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    reset_join_work();
    operator
        .process_data("right", right_batch(vec![20]), &context, &mut collector)
        .await
        .unwrap();
    assert_output(
        &mut collector,
        &operator,
        left_times,
        &vec![42; left_times.len()],
        &[20],
        0,
    );
    assert_eq!(operator.status().left.retained_bytes, bytes);
    assert_eq!(operator.state.next_right_row_id, 1);
    let epoch = snapshot.inline_metadata["epoch"].as_u64().unwrap() + 1;
    let resumed = operator.checkpoint(Epoch::new(epoch).unwrap()).unwrap();
    assert_eq!(resumed.inline_metadata["layout_version"], 1);
    let mut again = operator_fixture();
    again.restore(&resumed).unwrap();
    assert_eq!(again.status(), operator.status());
    again
        .process_data("right", right_batch(vec![21, 19]), &context, &mut collector)
        .await
        .unwrap();
    assert_output(
        &mut collector,
        &again,
        left_times,
        &vec![42; left_times.len()],
        &[21, 19],
        1,
    );
    assert_eq!(again.state.next_right_row_id, 3);
    assert_eq!(
        again.status().emitted_match_rows,
        3 * left_times.len() as u64
    );
    join_work().sql_probe_table_builds
}

#[tokio::test]
async fn test_all_frozen_v1_captures_restore_and_continue_with_native_lookup() {
    let captures = v1_fixture_captures().await;
    let frozen: Vec<Value> =
        serde_json::from_str(include_str!("../fixtures/checkpoint-v1.json")).unwrap();
    assert_eq!(captures.iter().map(wire).collect::<Vec<_>>(), frozen);
    let times = [
        vec![0, 1, 2, 30, 40, 50],
        vec![1, 2, 30, 40, 50],
        vec![1, 2, 20, 30, 40, 50],
        vec![20, 30, 40, 50],
        vec![20, 30, 40, 50],
    ];
    let mut sql_tables = Vec::new();
    for (capture, expected_times) in captures.iter().zip(&times) {
        sql_tables.push(resume_capture(capture, expected_times).await);
    }
    assert_eq!(sql_tables, vec![0; 5]);
}

#[tokio::test]
async fn test_known_large_slice_bounds_backing_and_keeps_v1_row_ipc() {
    let input = left_batch(vec![100; 8_192]);
    let source = &input.table_payload().unwrap().batches()[0];
    let slice = source.slice(8_191, 1);
    assert!(slice.get_array_memory_size() > 4_096);
    let legacy_ipc = row_ipc::RowIpcEncoder::default()
        .encode(&slice, "match", "left")
        .unwrap();
    let batch = Batch::table(vec![slice], BatchMetadata::default()).unwrap();
    let mut operator = operator_fixture();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", batch, &context, &mut collector)
        .await
        .unwrap();
    let capture = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(
        capture.segments["left-delta-1"]
            .bytes()
            .ends_with(&legacy_ipc)
    );
    assert_eq!(operator.status().left.retained_rows, 1);
    assert!(operator.state.left[0].record.get_array_memory_size() <= 4_096);
}

#[tokio::test]
async fn test_leased_core_buffers_report_actual_backing() {
    let batch = left_batch((0..8_192).collect());
    let record = &batch.table_payload().unwrap().batches()[0];
    let mut operator = operator_fixture();
    let chunk = copied_test_chunk(&mut operator, record, 0).await.unwrap();
    let payload = columnar::RowPayload::at(record, Some(&chunk), 0);
    assert!(matches!(payload, columnar::RowPayload::Shared { .. }));
    let bytes = payload.get_array_memory_size();
    assert!(
        bytes >= 3 * 8_192 * size_of::<i64>(),
        "custom buffer wrappers must not hide actual core-owned backing: {bytes}"
    );
}

#[tokio::test]
async fn test_progress_sparse_chunk_bounds_backing_and_keeps_v1_row_ipc() {
    let input = left_batch((0..8_192).collect());
    let source = &input.table_payload().unwrap().batches()[0];
    let legacy_ipc = row_ipc::RowIpcEncoder::default()
        .encode(&source.slice(8_191, 1), "match", "left")
        .unwrap();
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(8_192, 4_000_000, 1_000).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", input, &context, &mut collector)
        .await
        .unwrap();
    let progress = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_008_191)),
    );
    operator
        .on_ingress_progress("right", &progress)
        .await
        .unwrap();
    assert_eq!(operator.status().left.retained_rows, 1);
    assert_eq!(operator.status().left.evicted_rows, 8_191);
    assert_eq!(operator.state.next_left_row_id, 8_192);
    let captured = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(
        captured.segments["left-delta-1"]
            .bytes()
            .ends_with(&legacy_ipc)
    );
    let bytes = operator.state.left[0].record.get_array_memory_size();
    assert!(
        bytes <= 4_096,
        "one post-eviction row retains {bytes} bytes"
    );
}

fn sparse_left_fixture() -> StreamJoinOperator {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(8_192, 4_000_000, 1_000).unwrap();
    StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap()
}

#[tokio::test]
async fn test_sparse_chunk_refusal_retries_after_actual_credit_release() {
    let mut operator = sparse_left_fixture();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            left_batch((0..8_192).collect()),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let progress = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_008_191)),
    );
    operator
        .evict_progress("right", progress.ingress_progress().get("right").unwrap())
        .unwrap();
    let original = operator.state.left[0].record.clone();
    let owner = original.funded_owner().unwrap();
    let status = operator.status();
    let captured = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let runtime = operator.runtime.runtime().unwrap();
    let pool = runtime.incremental_memory_pool();
    let pressure = runtime.incremental_reservation("sparse-refusal");
    pressure.try_grow((1 << 30) - pool.reserved()).unwrap();
    operator
        .on_ingress_progress("right", &progress)
        .await
        .unwrap();
    assert_eq!(operator.state.left[0].record.funded_owner(), Some(owner));
    assert_eq!(operator.status().left, status.left);
    drop(pressure);
    operator
        .on_ingress_progress("right", &progress)
        .await
        .unwrap();
    let replacement = operator.state.left[0].record.funded_owner().unwrap();
    assert_ne!(
        replacement.0, owner.0,
        "a refused candidate must survive to retry"
    );
    assert!(operator.state.left[0].record.get_array_memory_size() <= 4_096);
    assert_eq!(operator.status().left, status.left);
    assert_eq!(
        operator
            .checkpoint(Epoch::new(2).unwrap())
            .unwrap()
            .segments,
        captured.segments
    );
    assert_eq!(
        pool.reserved(),
        native_lookup_tests::state_funding(&operator) + owner.1
    );
    drop(original);
    assert_eq!(
        pool.reserved(),
        native_lookup_tests::state_funding(&operator)
    );
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_sparse_chunk_variable_width_keeps_only_surviving_value_backing() {
    let mut values = vec!["w".repeat(8_192); 32];
    values[31] = "kept".into();
    let source = RecordBatch::try_new(
        right_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; 32])),
            Arc::new(
                TimestampMicrosecondArray::from((0..32).collect::<Vec<_>>()).with_timezone("UTC"),
            ),
            Arc::new(StringArray::from(values)),
        ],
    )
    .unwrap();
    let ipc = row_ipc::RowIpcEncoder::default()
        .encode(&source.slice(31, 1), "match", "right")
        .unwrap();
    let mut operator = operator_fixture();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            Batch::table(vec![source], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let progress = progress_context(
        &job,
        (IngressState::Active, Some(300_000_031)),
        (IngressState::Active, None),
    );
    operator
        .on_ingress_progress("left", &progress)
        .await
        .unwrap();
    assert_eq!(operator.status().right.retained_rows, 1);
    assert_eq!(operator.status().right.evicted_rows, 31);
    assert_eq!(operator.state.right[0].row_id, 31);
    assert_eq!(
        row_ipc::RowIpcEncoder::default()
            .encode(&operator.state.right[0].record.view(), "match", "right")
            .unwrap(),
        ipc
    );
    let bytes = operator.state.right[0].record.get_array_memory_size();
    assert!(bytes <= 4_096, "surviving short value keeps {bytes} bytes");
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_sparse_mixed_retention_admission_bounds_each_batch_without_new_expiry() {
    let mut operator = operator_fixture();
    let job = job();
    let context = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_010_000)),
    );
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut expected_ipc = Vec::new();
    for batch_id in 0..4 {
        let mut times = vec![0; 512];
        times[511] = 10_000 + batch_id;
        let input = left_batch(times);
        let record = &input.table_payload().unwrap().batches()[0];
        expected_ipc.push(
            row_ipc::RowIpcEncoder::default()
                .encode(&record.slice(511, 1), "match", "left")
                .unwrap(),
        );
        operator
            .process_data("left", input.clone(), &context, &mut collector)
            .await
            .unwrap();
        assert_eq!(input.num_rows(), 512);
    }
    assert_eq!(operator.state.next_left_row_id, 2_048);
    assert_eq!(operator.status().left.retained_rows, 4);
    assert_eq!(operator.status().left.evicted_rows, 0);
    assert_eq!(
        operator
            .state
            .left
            .iter()
            .map(|row| row.row_id)
            .collect::<Vec<_>>(),
        [511, 1_023, 1_535, 2_047]
    );
    for (row, ipc) in operator.state.left.iter().zip(expected_ipc) {
        assert!(row.record.funded_owner().is_some());
        assert_eq!(
            row_ipc::RowIpcEncoder::default()
                .encode(&row.record.view(), "match", "left")
                .unwrap(),
            ipc
        );
        let backing = row.record.get_array_memory_size();
        assert!(
            backing <= 4_096,
            "completed admission retains {backing} bytes without a later progress event"
        );
    }
    let before = operator.status().left;
    operator
        .on_ingress_progress("right", &context)
        .await
        .unwrap();
    assert_eq!(operator.status().left, before);
    assert!(collector.drain("output").is_empty());
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    assert_eq!(
        pool.reserved(),
        native_lookup_tests::state_funding(&operator)
    );
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn sparse_interrupted_copy(cancelled: bool) {
    let mut operator = sparse_left_fixture();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            left_batch((0..8_192).collect()),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let progress = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_008_191)),
    );
    operator
        .evict_progress("right", progress.ingress_progress().get("right").unwrap())
        .unwrap();
    let original = operator.state.left[0].record.funded_owner();
    let status = operator.status();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let paid = native_lookup_tests::state_funding(&operator);
    assert_eq!(pool.reserved(), paid);
    let mut copy = Box::pin(operator.on_ingress_progress("right", &progress));
    assert!(futures::poll!(copy.as_mut()).is_pending());
    assert!(
        pool.reserved() > paid,
        "yielded survivor selection keeps actual scratch credit"
    );
    if cancelled {
        job.cancellation().cancel();
        assert!(matches!(
            copy.as_mut().await,
            Err(CalcFlowError::Cancelled { .. })
        ));
    }
    drop(copy);
    assert_eq!(operator.state.left[0].record.funded_owner(), original);
    assert_eq!(operator.status().left, status.left);
    assert_eq!(pool.reserved(), paid);
    if !cancelled {
        operator
            .on_ingress_progress("right", &progress)
            .await
            .unwrap();
        assert!(operator.state.left[0].record.get_array_memory_size() <= 4_096);
    }
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_sparse_chunk_dropped_future_refunds_scratch_and_keeps_retry_candidate() {
    sparse_interrupted_copy(false).await;
}

#[tokio::test]
async fn test_sparse_chunk_actual_cancel_refunds_scratch_and_keeps_original_owner() {
    sparse_interrupted_copy(true).await;
}

fn schema_with_key(base: &SchemaRef, key: &DataType) -> SchemaRef {
    let mut fields = base
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[0] = Field::new("account_id", key.clone(), false);
    Arc::new(Schema::new(fields))
}

fn right_key_batch(schema: SchemaRef, key: ArrayRef) -> Batch {
    let record = RecordBatch::try_new(
        schema,
        vec![
            key,
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
            Arc::new(StringArray::from(vec!["paid"])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

async fn legacy_key_behavior(key: ArrayRef) {
    let left = schema_with_key(&left_schema(), key.data_type());
    let right = schema_with_key(&right_schema(), key.data_type());
    let mut operator =
        StreamJoinOperator::new("match", left.clone(), right.clone(), spec()).unwrap();
    let job = job();
    let context = progress_context(
        &job,
        (IngressState::Ended, None),
        (IngressState::Active, None),
    );
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            right_key_batch(right.clone(), key.clone()),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert_eq!(operator.state.next_right_row_id, 1);
    assert_eq!(operator.status().right.retained_rows, 0);
    assert!(collector.drain("output").is_empty());
    let mut retained = StreamJoinOperator::new("match", left, right.clone(), spec()).unwrap();
    let active = StreamOperatorContext::new(&job, "match", None);
    let error = retained
        .process_data(
            "right",
            right_key_batch(right, key),
            &active,
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::Internal { .. }));
    assert_eq!(retained.state.next_right_row_id, 0);
    assert_eq!(retained.status().right.retained_rows, 0);
}

#[tokio::test]
async fn test_legacy_int8_and_date_codec_behavior_stays_on_fallback() {
    let keys: Vec<ArrayRef> = vec![
        Arc::new(Int8Array::from(vec![-1])),
        Arc::new(Date32Array::from(vec![0])),
        Arc::new(Date64Array::from(vec![0])),
    ];
    for key in keys {
        legacy_key_behavior(key).await;
    }
}

fn opaque_time_record() -> (RecordBatch, std::sync::Weak<()>) {
    use datafusion::arrow::buffer::{Buffer, ScalarBuffer};
    use tokio_util::bytes::Bytes;
    struct OpaqueTimes {
        visible: Buffer,
        _hidden: Vec<u8>,
        _live: Arc<()>,
    }
    impl AsRef<[u8]> for OpaqueTimes {
        fn as_ref(&self) -> &[u8] {
            &self.visible
        }
    }
    let live = Arc::new(());
    let weak = Arc::downgrade(&live);
    let bytes = Bytes::from_owner(OpaqueTimes {
        visible: Buffer::from_vec(vec![10_i64]),
        _hidden: vec![0; 1 << 20],
        _live: live,
    });
    let values = ScalarBuffer::new(Buffer::from(bytes), 0, 1);
    assert_eq!(values.inner().capacity(), 8);
    let record = RecordBatch::try_new(
        left_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::new(values, None).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![42])),
        ],
    )
    .unwrap();
    (record, weak)
}

#[tokio::test]
async fn test_owned_ingress_copy_releases_opaque_caller_owner() {
    let (record, weak) = opaque_time_record();
    let ipc = row_ipc::RowIpcEncoder::default()
        .encode(&record, "match", "left")
        .unwrap();
    let charge = state_row_charge(&record, 0, &[0], "match").unwrap();
    let mut operator = operator_fixture();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert!(matches!(
        operator.state.left[0].record,
        columnar::RowPayload::Shared { .. }
    ));
    assert!(
        weak.upgrade().is_none(),
        "owned ingress must not retain the caller's hidden 1MiB owner"
    );
    assert_eq!(operator.status().left.retained_bytes, charge);
    assert_eq!(operator.state.next_left_row_id, 1);
    assert!(
        operator.checkpoint(Epoch::INITIAL).unwrap().segments["left-delta-1"]
            .bytes()
            .ends_with(&ipc)
    );
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    assert_eq!(
        pool.reserved(),
        native_lookup_tests::state_funding(&operator)
    );
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}

async fn assert_legacy_copy_adversary(payload: ArrayRef) {
    let mut fields = left_schema().fields().to_vec();
    fields[2] = Arc::new(Field::new("value", payload.data_type().clone(), true));
    let schema = Arc::new(Schema::new(fields));
    let record = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![7, 7])),
            Arc::new(TimestampMicrosecondArray::from(vec![0, 1]).with_timezone("UTC")),
            payload,
        ],
    )
    .unwrap();
    let expected = (0..2)
        .map(|row| {
            row_ipc::RowIpcEncoder::default()
                .encode(&record.slice(row, 1), "match", "left")
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut operator = StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert!(
        operator
            .state
            .left
            .iter()
            .all(|row| row.record.is_unfunded_legacy()),
        "Boolean/null-buffer payload must keep its physical V1 representation on Legacy"
    );
    for (row, ipc) in operator.state.left.iter().zip(expected) {
        assert_eq!(
            row_ipc::RowIpcEncoder::default()
                .encode(&row.record.view(), "match", "left")
                .unwrap(),
            ipc
        );
    }
    reset_join_work();
    operator
        .process_data("right", right_batch(vec![1]), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert_eq!(collector.drain("output").len(), 1);
}

#[tokio::test]
async fn test_owned_ingress_boolean_and_all_valid_masks_use_byte_exact_legacy() {
    use datafusion::arrow::buffer::{NullBuffer, ScalarBuffer};
    assert_legacy_copy_adversary(Arc::new(BooleanArray::from(vec![false, true]))).await;
    let masked = Int64Array::new(
        ScalarBuffer::from(vec![40, 41]),
        Some(NullBuffer::new_valid(2)),
    );
    assert!(masked.nulls().is_some());
    assert_legacy_copy_adversary(Arc::new(masked)).await;
    assert_legacy_copy_adversary(Arc::new(Int64Array::from(vec![Some(40), None]))).await;
}

#[tokio::test]
async fn test_owned_ingress_long_string_cooperates_and_cancels_before_commit() {
    let record = RecordBatch::try_new(
        right_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
            Arc::new(StringArray::from(vec!["é".repeat(262_144)])),
        ],
    )
    .unwrap();
    let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let mut operator = operator_fixture();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let wake = Arc::new(CopyWake(AtomicUsize::new(0)));
    let waker = Waker::from(wake.clone());
    let mut cx = Context::from_waker(&waker);
    let mut future =
        Box::pin(operator.process_data("right", input.clone(), &context, &mut collector));
    let mut owned_allocation = 0_i64;
    for _ in 0..64 {
        let measured = allocation_counter::measure(|| {
            assert!(
                future.as_mut().poll(&mut cx).is_pending(),
                "512KiB string must exhaust Tokio's cooperative budget during funded copying"
            );
        });
        owned_allocation += measured.bytes_current;
        if owned_allocation >= 524_288 {
            break;
        }
    }
    assert!(
        owned_allocation >= 524_288,
        "gate must reach actual prepaid StringBuilder buffers, not only planning credit"
    );
    tokio::task::yield_now().await;
    assert!(
        wake.0.load(Ordering::Relaxed) > 0,
        "copy yield must wake its real observer"
    );
    assert!(i64::try_from(pool.reserved()).unwrap() >= owned_allocation);
    job.cancellation().cancel();
    assert!(matches!(future.await, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(operator.state.next_right_row_id, 0);
    assert_eq!(operator.status().right.retained_rows, 0);
    assert!(collector.drain("output").is_empty());
    assert_eq!(pool.reserved(), 0);
}

async fn close_home_during_string_copy(cancelled: bool) {
    let record = RecordBatch::try_new(
        right_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
            Arc::new(StringArray::from(vec!["é".repeat(262_144)])),
        ],
    )
    .unwrap();
    let expected = row_ipc::RowIpcEncoder::default()
        .encode(&record, "match", "right")
        .unwrap();
    let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let mut operator = operator_fixture();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut future =
        Box::pin(operator.process_data("right", input.clone(), &context, &mut collector));
    let waker = Waker::from(Arc::new(CopyWake(AtomicUsize::new(0))));
    let mut cx = Context::from_waker(&waker);
    let mut resident = 0_i64;
    for _ in 0..64 {
        let measured = allocation_counter::measure(|| {
            assert!(future.as_mut().poll(&mut cx).is_pending());
        });
        resident += measured.bytes_current;
        if resident >= 524_288 {
            break;
        }
    }
    assert!(
        resident >= 524_288,
        "closure must follow actual paid StringBuilder allocation"
    );
    assert!(pool.reserved() >= usize::try_from(resident).unwrap());
    job.gather_owner().close_admission();
    if cancelled {
        job.cancellation().cancel();
        assert!(matches!(future.await, Err(CalcFlowError::Cancelled { .. })));
        assert!(operator.state.right.is_empty());
    } else {
        future.await.unwrap();
        assert!(matches!(
            operator.state.right[0].record,
            columnar::RowPayload::Legacy(_)
        ));
        assert_eq!(operator.state.right[0].row_id, 0);
        assert_eq!(
            row_ipc::RowIpcEncoder::default()
                .encode(&operator.state.right[0].record.view(), "match", "right")
                .unwrap(),
            expected
        );
    }
    assert!(collector.drain("output").is_empty());
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_home_closed_after_string_allocation_falls_back_without_new_error() {
    close_home_during_string_copy(false).await;
}

#[tokio::test]
async fn test_home_closed_after_string_allocation_preserves_real_cancel() {
    close_home_during_string_copy(true).await;
}

#[tokio::test]
async fn test_owned_copy_selected_identity_gaps_and_dirty_only_funding() {
    let source = left_batch(vec![0, 10, 20, 30]);
    let record = &source.table_payload().unwrap().batches()[0];
    let expected = [2, 3].map(|row| {
        row_ipc::RowIpcEncoder::default()
            .encode(&record.slice(row, 1), "match", "left")
            .unwrap()
    });
    let mut operator = operator_fixture();
    let job = job();
    let context = progress_context(
        &job,
        (IngressState::Active, Some(15)),
        (IngressState::Active, None),
    );
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", source.clone(), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(operator.state.next_left_row_id, 4);
    assert_eq!(operator.status().left.late_rows, 2);
    let rows = &operator.state.left;
    assert_eq!(
        rows.iter().map(|row| row.row_id).collect::<Vec<_>>(),
        [2, 3]
    );
    assert_eq!(
        rows.iter()
            .map(|row| row.record.offset())
            .collect::<Vec<_>>(),
        [0, 1]
    );
    for (row, expected) in rows.iter().zip(expected) {
        assert_eq!(
            row_ipc::RowIpcEncoder::default()
                .encode(&row.record.view(), "match", "left")
                .unwrap(),
            expected
        );
    }
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let chunk = operator.state.left[0].record.funded_owner().unwrap().1;
    let keys = operator
        .state
        .deltas
        .pending
        .iter()
        .filter_map(|op| match op {
            PendingOp::Upsert { encoded_key, .. } | PendingOp::Tombstone { encoded_key, .. } => {
                encoded_key.funded_owner()
            }
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(keys.len(), 1);
    let key_credit = keys.values().sum::<usize>();
    assert!(key_credit > 0);
    operator.state.left.1 = None;
    operator.state.left.clear();
    assert_eq!(
        pool.reserved(),
        chunk + key_credit,
        "dirty locators must keep payload and shared key credits after live rows disappear"
    );
    operator.state.deltas.pending.clear();
    assert_eq!(pool.reserved(), 0);
}

fn nullable_admission_schema(unit: TimeUnit) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Int64, true),
        Field::new(
            "authorized_at",
            DataType::Timestamp(unit, Some("UTC".into())),
            true,
        ),
        Field::new("amount", DataType::Int64, true),
    ]))
}

fn nullable_admission_record(
    schema: &SchemaRef,
    keys: Vec<Option<i64>>,
    times: Vec<Option<i64>>,
) -> RecordBatch {
    let rows = times.len();
    let times =
        datafusion::arrow::compute::cast(&Int64Array::from(times), schema.field(1).data_type())
            .unwrap();
    RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(Int64Array::from(keys)),
            times,
            Arc::new(Int64Array::from(vec![42; rows])),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn test_batch_admission_masks_keep_drop_precedence_and_cross_record_identity_gaps() {
    let schema = nullable_admission_schema(TimeUnit::Microsecond);
    let records = vec![
        nullable_admission_record(
            &schema,
            vec![None, None, Some(7), Some(7), None, Some(7)],
            vec![None, Some(9), Some(9), Some(10), None, Some(12)],
        ),
        nullable_admission_record(
            &schema,
            vec![Some(7), None, Some(7)],
            vec![Some(-5), Some(5), Some(11)],
        ),
    ];
    let source = Batch::table(records.clone(), BatchMetadata::default()).unwrap();
    let mut operator = StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
    let job = job();
    let context = progress_context(
        &job,
        (IngressState::Active, Some(10)),
        (IngressState::Active, None),
    );
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", source.clone(), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(operator.state.next_left_row_id, 9);
    assert_eq!(
        operator
            .state
            .left
            .iter()
            .map(|row| (row.row_id, row.event_time.as_micros()))
            .collect::<Vec<_>>(),
        [(3, 10), (5, 12), (8, 11)]
    );
    let metrics = &operator.state.metrics.left;
    assert_eq!(metrics.null_event_time_rows, 2);
    assert_eq!(metrics.null_key_rows, 2);
    assert_eq!(metrics.late_rows, 2);
    assert_eq!(metrics.late_affected_batches, 1);
    assert_eq!(metrics.max_lateness_micros, Some(15));
    assert_eq!(source.table_payload().unwrap().batches(), records);
    assert!(collector.drain("output").is_empty());
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(context);
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_batch_admission_masks_cross_word_boundaries_without_reclassifying_drops() {
    let schema = nullable_admission_schema(TimeUnit::Microsecond);
    let mut keys = vec![Some(7); 130];
    let mut times = vec![Some(9); 130];
    for row in [0, 63, 128] {
        keys[row] = None;
        times[row] = None;
    }
    keys[64] = None;
    times[64] = Some(-5);
    for (row, time) in [(65, 10), (127, 11), (129, 12)] {
        times[row] = Some(time);
    }
    let mut backing_keys = vec![Some(99); 5];
    backing_keys.extend(keys);
    let mut backing_times = vec![Some(i64::MAX); 5];
    backing_times.extend(times);
    let record = nullable_admission_record(&schema, backing_keys, backing_times).slice(5, 130);
    let source = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let mut operator = StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
    let job = job();
    let context = progress_context(
        &job,
        (IngressState::Active, Some(10)),
        (IngressState::Active, None),
    );
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", source, &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(operator.state.next_left_row_id, 130);
    assert_eq!(
        operator
            .state
            .left
            .iter()
            .map(|row| row.row_id)
            .collect::<Vec<_>>(),
        [65, 127, 129]
    );
    let metrics = &operator.state.metrics.left;
    assert_eq!(metrics.null_event_time_rows, 3);
    assert_eq!(metrics.null_key_rows, 1);
    assert_eq!(metrics.late_rows, 123);
    assert_eq!(metrics.late_affected_batches, 1);
    assert_eq!(metrics.max_lateness_micros, Some(1));
    assert!(collector.drain("output").is_empty());
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(context);
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_batch_admission_null_timestamps_skip_masked_overflow_values() {
    use datafusion::arrow::array::TimestampSecondArray;
    use datafusion::arrow::buffer::{NullBuffer, ScalarBuffer};

    let rows = 129;
    let schema = nullable_admission_schema(TimeUnit::Second);
    let times = TimestampSecondArray::new(
        ScalarBuffer::from(vec![i64::MAX; rows]),
        Some(NullBuffer::new_null(rows)),
    )
    .with_timezone("UTC");
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![None; rows])),
            Arc::new(times),
            Arc::new(Int64Array::from(vec![42; rows])),
        ],
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert_eq!(
        operator.state.next_left_row_id,
        u64::try_from(rows).unwrap()
    );
    let metrics = &operator.state.metrics.left;
    assert_eq!(metrics.null_event_time_rows, u64::try_from(rows).unwrap());
    assert_eq!(metrics.null_key_rows, 0);
    assert_eq!(metrics.late_rows, 0);
    assert_eq!(metrics.late_affected_batches, 0);
    assert!(operator.state.left.is_empty());
    assert!(collector.drain("output").is_empty());
    drop(context);
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

#[tokio::test]
async fn test_batch_admission_row_id_overflow_precedes_timestamp_conversion_and_null_key() {
    for preceding in [0, 1, 64, 128] {
        let next_id = u64::MAX - preceding;
        let schema = nullable_admission_schema(TimeUnit::Second);
        let mut times = vec![None; usize::try_from(preceding).unwrap()];
        times.push(Some(i64::MAX));
        let source = Batch::table(
            vec![nullable_admission_record(
                &schema,
                vec![None; times.len()],
                times,
            )],
            BatchMetadata::default(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
        operator.state.next_left_row_id = next_id;
        let before = operator.status();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let error = operator
            .process_data("left", source, &context, &mut collector)
            .await
            .unwrap_err();
        assert_eq!(
            reason_of(&error),
            Some(crate::StreamingFailureReason::JoinCounterOverflow)
        );
        assert!(error.to_string().contains("row_id"));
        assert_eq!(operator.state.next_left_row_id, next_id);
        assert_eq!(operator.status(), before);
        assert!(collector.drain("output").is_empty());
    }
}

#[tokio::test]
async fn test_owned_and_generic_batch_masks_keep_late_and_retention_boundaries() {
    for owned_copy in [true, false] {
        let schema = Arc::new(Schema::new(vec![
            left_schema().field(0).clone(),
            left_schema().field(1).clone(),
            Field::new(
                "amount",
                if owned_copy {
                    DataType::Int64
                } else {
                    DataType::Boolean
                },
                false,
            ),
        ]));
        let records = [vec![0, 9, 10], vec![11, 12]]
            .into_iter()
            .map(|times| {
                let rows = times.len();
                let payload: ArrayRef = if owned_copy {
                    Arc::new(Int64Array::from(vec![42; rows]))
                } else {
                    Arc::new(BooleanArray::from(vec![true; rows]))
                };
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int64Array::from(vec![7; rows])),
                        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
                        payload,
                    ],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let source = Batch::table(records.clone(), BatchMetadata::default()).unwrap();
        let mut operator =
            StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
        let job = job();
        let context = progress_context(
            &job,
            (IngressState::Active, Some(10)),
            (IngressState::Active, Some(60_000_011)),
        );
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", source.clone(), &context, &mut collector)
            .await
            .unwrap();
        assert_eq!(operator.state.next_left_row_id, 5);
        assert_eq!(
            operator
                .state
                .left
                .iter()
                .map(|row| (row.row_id, row.event_time.as_micros()))
                .collect::<Vec<_>>(),
            [(3, 11), (4, 12)]
        );
        assert!(
            operator
                .state
                .left
                .iter()
                .all(|row| row.record.funded_owner().is_some() == owned_copy)
        );
        let metrics = &operator.state.metrics.left;
        assert_eq!(metrics.late_rows, 2);
        assert_eq!(metrics.late_affected_batches, 1);
        assert_eq!(metrics.max_lateness_micros, Some(10));
        assert_eq!(metrics.null_event_time_rows, 0);
        assert_eq!(metrics.null_key_rows, 0);
        assert_eq!(source.table_payload().unwrap().batches(), records);
        assert!(collector.drain("output").is_empty());
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(context);
        drop(operator);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_batch_admission_counter_overflow_refunds_masks_and_keeps_state_uncommitted() {
    for counter in [
        "null_event_time_rows",
        "null_key_rows",
        "late_rows",
        "late_affected_batches",
    ] {
        let schema = nullable_admission_schema(TimeUnit::Microsecond);
        let source = Batch::table(
            vec![nullable_admission_record(
                &schema,
                vec![Some(7), Some(7), None, Some(7)],
                vec![Some(11), None, Some(10), Some(9)],
            )],
            BatchMetadata::default(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
        let job = job();
        let context = progress_context(
            &job,
            (IngressState::Active, Some(10)),
            (IngressState::Active, None),
        );
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("right", right_batch(vec![11]), &context, &mut collector)
            .await
            .unwrap();
        let metrics = &mut operator.state.metrics.left;
        match counter {
            "null_event_time_rows" => metrics.null_event_time_rows = u64::MAX,
            "null_key_rows" => metrics.null_key_rows = u64::MAX,
            "late_rows" => metrics.late_rows = u64::MAX,
            "late_affected_batches" => metrics.late_affected_batches = u64::MAX,
            _ => unreachable!(),
        }
        let before = operator.status();
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let funding = pool.reserved();
        let error = operator
            .process_data("left", source, &context, &mut collector)
            .await
            .unwrap_err();
        assert_eq!(
            reason_of(&error),
            Some(crate::StreamingFailureReason::JoinCounterOverflow)
        );
        assert!(error.to_string().contains(counter));
        assert_eq!(operator.state.next_left_row_id, 0);
        assert_eq!(operator.status(), before);
        assert_eq!(pool.reserved(), funding);
        assert!(collector.drain("output").is_empty());
        drop(context);
        drop(operator);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_owned_copy_private_schema_remains_funded_when_payload_outlives_operator() {
    let mut schema = None;
    let allocated = allocation_counter::measure(|| schema = Some(metadata_schema(4_096)));
    let schema = schema.unwrap();
    let weak = Arc::downgrade(&schema);
    let record = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![42])),
        ],
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let payload = operator.state.left[0].record.clone();
    let paid = payload.funded_owner().unwrap().1;
    assert!(
        i64::try_from(paid).unwrap() < allocated.bytes_current,
        "private schema must not retain the caller's spare allocation {} under chunk {paid}",
        allocated.bytes_current
    );
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(collector);
    drop(operator);
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), paid);
    assert!(payload.schema_ref().metadata().is_empty());
    let view = payload.view();
    assert_eq!(
        pool.reserved(),
        paid,
        "IPC row view keeps schema, buffers and reservation together"
    );
    assert!(weak.upgrade().is_none());
    drop(view);
    drop(payload);
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_owned_copy_headers_exhaust_cooperative_budget_before_native_retention() {
    let schema = owner_proof_schema(1_032);
    let columns = schema
        .fields()
        .iter()
        .map(|field| owner_proof_array(field.data_type()))
        .collect();
    let record = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let mut operator = StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut future = Box::pin(operator.process_data("left", input, &context, &mut collector));
    assert!(futures::poll!(future.as_mut()).is_pending());
    job.cancellation().cancel();
    assert!(matches!(future.await, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(operator.state.next_left_row_id, 0);
    assert!(operator.state.left.is_empty());
    assert!(collector.drain("output").is_empty());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_owned_copy_denied_funding_keeps_legacy_admission_and_pool_zero() {
    let mut operator = operator_fixture();
    let runtime = operator.runtime.runtime().unwrap();
    let pressure = runtime.incremental_reservation("owned-copy-pressure");
    let pool = runtime.incremental_memory_pool();
    pressure.try_grow(1_024 * 1_024 * 1_024).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![0, 1]), &context, &mut collector)
        .await
        .unwrap();
    assert!(
        operator
            .state
            .left
            .iter()
            .all(|row| row.record.is_unfunded_legacy())
    );
    assert_eq!(operator.state.next_left_row_id, 2);
    assert_eq!(pool.reserved(), pressure.size());
    drop(pressure);
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_owned_copy_key_scratch_does_not_borrow_whole_chunk_backing() {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(4_096, 4_000_000, 1_000).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            left_batch((0..4_096).collect()),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let plan = operator.side_plan("right").unwrap();
    operator.state.left = operator.state.left[..1].to_vec().into();
    let keys = operator.owned_state_keys(&plan, &context).await.unwrap();
    assert_eq!(
        keys.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[7]
    );
    assert!(
        keys.column(0).get_array_memory_size() <= 4_096,
        "one SQL key row must not escape the full paid core chunk backing: {} bytes",
        keys.column(0).get_array_memory_size()
    );
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(operator);
    assert!(
        pool.reserved() > 0,
        "escaped paid key buffers must retain their actual funding"
    );
    assert_eq!(
        keys.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[7]
    );
    drop(keys);
    assert_eq!(pool.reserved(), 0);
}
