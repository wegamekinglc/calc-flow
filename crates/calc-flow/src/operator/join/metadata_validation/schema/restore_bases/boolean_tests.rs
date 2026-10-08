use super::*;
use datafusion::arrow::{
    array::{Array, BooleanArray, GenericStringArray, PrimitiveArray},
    buffer::{BooleanBuffer, Buffer},
    datatypes::Int64Type,
    ipc::{self, reader::StreamReader, writer::StreamWriter},
};
use datafusion::execution::memory_pool::MemoryConsumer;

fn boolean_operator(nullable: bool, boolean_key: bool) -> StreamJoinOperator {
    let mut fields = operator().input_schema(0).fields().to_vec();
    if boolean_key {
        fields[0] = Arc::new(Field::new("key", DataType::Boolean, false));
    }
    fields.push(Arc::new(Field::new("flag", DataType::Boolean, nullable)));
    let schema = Arc::new(Schema::new(fields));
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(100, 100_000, 100).unwrap(),
    )
    .unwrap();
    let mut output = StreamJoinOperator::new("match", Arc::clone(&schema), schema, spec).unwrap();
    output.prepare_checkpoint_preload_runtime().unwrap();
    output
}

fn boolean_record(operator: &StreamJoinOperator, flag: BooleanArray) -> RecordBatch {
    let count = flag.len();
    let key: datafusion::arrow::array::ArrayRef =
        if operator.input_schema(0).field(0).data_type() == &DataType::Boolean {
            Arc::new(BooleanArray::from(vec![true; count]))
        } else {
            Arc::new(Int64Array::from(vec![7; count]))
        };
    RecordBatch::try_new(
        Arc::clone(operator.input_schema(0)),
        vec![
            key,
            Arc::new(TimestampMicrosecondArray::from(vec![10; count])),
            Arc::new(flag),
        ],
    )
    .unwrap()
}

fn row_from_record(record: RecordBatch) -> StoredRow {
    let charge = state_row_charge(&record, 0, &[0], "match").unwrap();
    let key = encode_join_key_v1(&record, 0, &[0]).unwrap();
    StoredRow {
        record: record.into(),
        event_time: crate::EventTime::from_micros(10),
        row_id: 0,
        charge,
        encoded_key: Arc::new(key.into()),
    }
}

fn boolean_snapshot(source: &mut StreamJoinOperator, flag: BooleanArray) -> OperatorStateSnapshot {
    let left = row_from_record(boolean_record(source, flag.clone()));
    let right = row_from_record(boolean_record(source, flag));
    source.state.metrics.left.retained_rows = 1;
    source.state.metrics.left.retained_bytes = left.charge;
    source.state.metrics.right.retained_rows = 1;
    source.state.metrics.right.retained_bytes = right.charge;
    source.state.left = vec![left].into();
    source.state.right = vec![right].into();
    source.state.next_left_row_id = 1;
    source.state.next_right_row_id = 1;
    let mut snapshot = source.checkpoint(Epoch::new(7).unwrap()).unwrap();
    for (side, rows) in [("left", &source.state.left), ("right", &source.state.right)] {
        snapshot.segments.insert(
            format!("{side}-base"),
            StateSegment::new(encode_side(rows, "match", side, &|| Ok(())).unwrap()),
        );
    }
    snapshot
}

fn row_ipc(record: &RecordBatch) -> Vec<u8> {
    let mut writer = StreamWriter::try_new(Vec::new(), record.schema_ref()).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    writer.into_inner().unwrap()
}

#[test]
fn test_restored_boolean_padding_allocation_and_last_buffer_refund() {
    assert!(
        size_of::<BooleanArray>()
            <= size_of::<PrimitiveArray<Int64Type>>().max(size_of::<GenericStringArray<i64>>())
    );
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for offset in [8, 9] {
        runtime.block_on(check_boolean_resident(&service, offset));
    }
    drop(runtime);
    service.shutdown();
}

async fn check_boolean_resident(service: &TestService, offset: usize) {
    let target = boolean_operator(false, false);
    let pool = pool(&target);
    let job = job(service, CancellationToken::new());
    let source = Buffer::from(vec![0_u8, 0xfe]);
    assert_eq!(source.as_slice(), [0, 0xfe]);
    let flag = BooleanArray::new(BooleanBuffer::new(source.clone(), offset, 1), None);
    let original = row_from_record(boolean_record(&target, flag));
    let record = original.record.view();
    let paid = columnar::restored::required(
        target.input_schema(0),
        super::super::super::super::inventory::registration_controls().unwrap(),
    )
    .unwrap();
    let credit = MemoryConsumer::new("boolean-resident").register(&pool);
    credit.try_grow(paid).unwrap();
    let lease = columnar::restored::ResidentLease::new(
        credit,
        job.gather_owner().retain_retirement().unwrap(),
    );
    let mut payload = None;
    let measured = allocation_counter::measure(|| {
        payload = Some(columnar::restored::copy_row(
            &record,
            target.input_schema(0),
            0,
            original.event_time,
            lease,
        ));
    });
    assert!(
        measured.bytes_total <= u64::try_from(paid).unwrap(),
        "{measured:?}; paid={paid}"
    );
    println!("Boolean offset {offset} requested allocations {measured:?}; paid={paid}");
    let payload = payload.unwrap();
    let view = payload.view();
    let flag = view
        .column(2)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert_eq!(flag.values().offset(), 0);
    assert_eq!(flag.len(), 1);
    assert_eq!(flag.null_count(), 0);
    assert_eq!(flag.value(0), offset == 9);
    assert_eq!(
        flag.values().values(),
        if offset == 8 { &[0xfe] } else { &[1] }
    );
    assert_eq!(source.as_slice(), [0, 0xfe]);
    assert_eq!(row_ipc(&view), row_ipc(&record));
    assert_eq!(
        state_row_charge(&view, 0, &[0], "match").unwrap(),
        original.charge
    );
    let weak = Arc::downgrade(payload.schema_ref());
    let values = view.column(2).to_data().buffers()[0].clone();
    drop(view);
    drop(payload);
    drop(target);
    assert_eq!(pool.reserved(), paid);
    assert_eq!(values.as_slice(), if offset == 8 { &[0xfe] } else { &[1] });
    {
        let mut drain = std::pin::pin!(job.gather_owner().close_and_drain());
        assert!(std::future::poll_fn(|cx| Poll::Ready(drain.as_mut().poll(cx).is_pending())).await);
        assert!(weak.upgrade().is_some());
        drop(values);
        drain.await;
    }
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
    drop(job);
}

#[test]
fn test_restored_boolean_reader_and_original_boundaries() {
    check_packed_reader();
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let entered = runtime.enter();
    let job = job(&service, CancellationToken::new());
    check_null_original(&job);
    check_boolean_key_original(&job);
    check_malformed_original(&job);
    drop(job);
    drop(entered);
    drop(runtime);
    service.shutdown();
}

fn check_packed_reader() {
    let target = boolean_operator(false, false);
    for count in [8_usize, 9] {
        let flag = BooleanArray::new(
            BooleanBuffer::new(Buffer::from(vec![0xff_u8; count.div_ceil(8)]), 0, count),
            None,
        );
        let record = boolean_record(&target, flag);
        let bytes = row_ipc(&record);
        assert!(inspector::inspect(&bytes, target.input_schema(0)).is_some());
        let decoded = StreamReader::try_new(std::io::Cursor::new(&bytes), None)
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(decoded.num_rows(), count);
        let flag = decoded
            .column(2)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!((0..count).all(|index| flag.value(index)));
        assert_eq!(row_ipc(&decoded), bytes);
    }
    for (byte, expected) in [(0xff_u8, true), (0xfe, false)] {
        let record = boolean_record(
            &target,
            BooleanArray::new(BooleanBuffer::new(Buffer::from(vec![byte]), 0, 1), None),
        );
        let bytes = row_ipc(&record);
        assert!(inspector::inspect(&bytes, target.input_schema(0)).is_some());
        let decoded = StreamReader::try_new(std::io::Cursor::new(&bytes), None)
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(
            decoded
                .column(2)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(0),
            expected
        );
        assert_eq!(row_ipc(&decoded), bytes);
    }
}

fn check_null_original(job: &StreamJobContext) {
    let mut source = boolean_operator(true, false);
    let snapshot = boolean_snapshot(&mut source, BooleanArray::from(vec![None::<bool>]));
    let record = source.state.left[0].record.view();
    assert!(inspector::inspect(&row_ipc(&record), record.schema_ref()).is_none());
    let mut target = boolean_operator(true, false);
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, job);
    assert!(poll_copy(construction.copy(&target, &snapshot, job)).unwrap());
    let decision = work(&mut construction)
        .run(construction.schema.metadata.control.stop.as_ref().unwrap())
        .unwrap();
    assert!(matches!(decision, RestoreBasesDecision::Original(_)));
    target
        .install_bases_decision(decision, &snapshot, job)
        .unwrap();
    let mut original = boolean_operator(true, false);
    original.restore(&snapshot).unwrap();
    assert_eq!(target.status(), original.status());
    for rows in [&target.state.left, &target.state.right] {
        assert!(rows[0].record.funded_owner().is_none());
        assert_eq!(rows[0].record.view().column(2).null_count(), 1);
        assert_eq!(row_ipc(&rows[0].record.view()), row_ipc(&record));
    }
    assert_eq!(
        encode_side(&target.state.left, "match", "left", &|| Ok(())).unwrap(),
        snapshot.segments["left-base"].bytes()
    );
    drop(construction);
    drop(target);
    assert_eq!(pool.reserved(), 0);
}

fn check_boolean_key_original(job: &StreamJobContext) {
    assert!(columnar::restored::key_width(&DataType::Boolean).is_none());
    let mut source = boolean_operator(false, true);
    let snapshot = boolean_snapshot(&mut source, BooleanArray::from(vec![true]));
    let mut target = boolean_operator(false, true);
    assert!(target.bases_construction(&snapshot, job).unwrap().is_none());
    target.restore(&snapshot).unwrap();
    assert_eq!(target.status().left.retained_rows, 1);
    assert!(
        target.state.left[0]
            .record
            .view()
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(0)
    );
    assert!(target.state.left[0].record.funded_owner().is_none());
    assert_eq!(
        encode_side(&target.state.left, "match", "left", &|| Ok(())).unwrap(),
        snapshot.segments["left-base"].bytes()
    );
    let pool = pool(&target);
    drop(target);
    assert_eq!(pool.reserved(), 0);
}

fn invalid_values_ipc(record: &RecordBatch) -> Vec<u8> {
    let mut bytes = row_ipc(record);
    let mut offset = 0;
    let (position, mut buffer) = loop {
        let inspector::StreamItem::Message(metadata) =
            inspector::message_metadata(&bytes, &mut offset).unwrap()
        else {
            panic!("record batch message exists");
        };
        let message = ipc::root_as_message(metadata).unwrap();
        if let Some(record) = message.header_as_record_batch() {
            let buffer = record.buffers().unwrap().get(5);
            break (
                (buffer.0.as_ptr() as usize)
                    .checked_sub(bytes.as_ptr() as usize)
                    .unwrap(),
                *buffer,
            );
        }
        offset += usize::try_from(message.bodyLength()).unwrap();
    };
    buffer.set_length(0);
    bytes[position..position + 16].copy_from_slice(&buffer.0);
    bytes
}

fn check_malformed_original(job: &StreamJobContext) {
    let mut source = boolean_operator(false, false);
    let mut snapshot = boolean_snapshot(&mut source, BooleanArray::from(vec![true]));
    let bad = invalid_values_ipc(&source.state.left[0].record.view());
    assert!(inspector::inspect(&bad, source.input_schema(0)).is_none());
    let original = snapshot.segments["left-base"].bytes();
    let mut base = original[..40].to_vec();
    base.extend_from_slice(&u64::try_from(bad.len()).unwrap().to_le_bytes());
    base.extend_from_slice(&bad);
    snapshot
        .segments
        .insert("left-base".into(), StateSegment::new(base));
    let unchanged = snapshot.clone();
    let mut target = boolean_operator(false, false);
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, job);
    assert!(poll_copy(construction.copy(&target, &snapshot, job)).unwrap());
    let decision = work(&mut construction)
        .run(construction.schema.metadata.control.stop.as_ref().unwrap())
        .unwrap();
    assert!(matches!(decision, RestoreBasesDecision::Original(_)));
    let actual = target
        .install_bases_decision(decision, &snapshot, job)
        .unwrap_err();
    let expected = boolean_operator(false, false)
        .restore(&snapshot)
        .unwrap_err();
    assert_eq!(actual.to_string(), expected.to_string());
    assert_eq!(target.status().left.retained_rows, 0);
    assert_eq!(target.status().right.retained_rows, 0);
    assert_eq!(snapshot.inline_metadata, unchanged.inline_metadata);
    for (id, segment) in &snapshot.segments {
        assert_eq!(segment.bytes(), unchanged.segments[id].bytes());
    }
    drop(construction);
    drop(target);
    assert_eq!(pool.reserved(), 0);
}
