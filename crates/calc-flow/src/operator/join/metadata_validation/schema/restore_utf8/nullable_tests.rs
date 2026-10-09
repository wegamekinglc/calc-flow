use super::*;
use datafusion::arrow::{
    array::{Array, ArrayRef, GenericStringArray, Int64Array, PrimitiveArray},
    buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer},
    datatypes::{Float64Type, Int64Type},
    ipc::{self, reader::StreamReader},
};
use datafusion::execution::memory_pool::{MemoryLimit, MemoryReservation};
use std::sync::atomic::{AtomicUsize, Ordering};

const HIDDEN_BITS: u64 = 0x7ff8_0000_0000_0055;

fn nullable_operator(utf8: bool) -> StreamJoinOperator {
    let key_type = if utf8 {
        DataType::Utf8
    } else {
        DataType::Int64
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("symbol", key_type, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("price", DataType::Float64, true),
    ]));
    let mut target = StreamJoinOperator::new(
        "match",
        Arc::clone(&schema),
        schema,
        operator().spec.clone(),
    )
    .unwrap();
    target.prepare_checkpoint_preload_runtime().unwrap();
    target
}

fn price(offset: usize, packed: u8) -> Float64Array {
    let values = ScalarBuffer::new(Buffer::from(vec![0_u64, HIDDEN_BITS]), 1, 1);
    let nulls = NullBuffer::new(BooleanBuffer::new(
        Buffer::from(vec![0_u8, packed]),
        offset,
        1,
    ));
    assert_eq!(nulls.inner().offset(), offset);
    Float64Array::new(values, Some(nulls))
}

fn record(target: &StreamJoinOperator, price: Float64Array) -> RecordBatch {
    let key: ArrayRef = if target.input_schema(0).field(0).data_type() == &DataType::Utf8 {
        Arc::new(StringArray::from(vec!["猫🙂"]))
    } else {
        Arc::new(Int64Array::from(vec![7]))
    };
    RecordBatch::try_new(
        Arc::clone(target.input_schema(0)),
        vec![
            key,
            Arc::new(TimestampMicrosecondArray::from(vec![10])),
            Arc::new(price),
        ],
    )
    .unwrap()
}

fn decoded(record: &RecordBatch) -> RecordBatch {
    StreamReader::try_new(std::io::Cursor::new(row_ipc(record)), None)
        .unwrap()
        .next()
        .unwrap()
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

fn nullable_snapshot() -> OperatorStateSnapshot {
    let mut source = nullable_operator(true);
    let left = row_from_record(record(&source, price(8, 0xfe)));
    let right = left.clone();
    source.state.metrics.left.retained_rows = 1;
    source.state.metrics.left.retained_bytes = left.charge;
    source.state.metrics.right.retained_rows = 1;
    source.state.metrics.right.retained_bytes = right.charge;
    source.state.left = vec![left].into();
    source.state.right = vec![right].into();
    source.state.next_left_row_id = 1;
    source.state.next_right_row_id = 1;
    let mut snapshot = source.checkpoint_v1(Epoch::new(7).unwrap()).unwrap();
    for (side, rows) in [("left", &source.state.left), ("right", &source.state.right)] {
        snapshot.segments.insert(
            format!("{side}-base"),
            StateSegment::new(encode_side(rows, "match", side, &|| Ok(())).unwrap()),
        );
    }
    snapshot
}

#[test]
fn test_nullable_float_raw_slots_padding_allocation_and_both_last_buffers() {
    assert!(
        size_of::<PrimitiveArray<Float64Type>>()
            <= size_of::<PrimitiveArray<Int64Type>>().max(size_of::<GenericStringArray<i64>>())
    );
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_resident(&service, false, 8, 0xfe));
    runtime.block_on(check_resident(&service, true, 9, 0xfc));
    check_valid_row(&service);
    drop(runtime);
    service.shutdown();
}

fn resident_bytes(target: &StreamJoinOperator, source: &RecordBatch, utf8: bool) -> usize {
    let registration =
        crate::operator::join::metadata_validation::inventory::registration_controls().unwrap();
    if utf8 {
        columnar::restored::utf8::required(source, target.input_schema(0), registration).unwrap()
    } else {
        columnar::restored::required(target.input_schema(0), registration).unwrap()
    }
}

fn copy(
    source: &RecordBatch,
    target: &StreamJoinOperator,
    lease: columnar::restored::ResidentLease,
    utf8: bool,
) -> columnar::RowPayload {
    if utf8 {
        columnar::restored::utf8::copy_row(
            source,
            target.input_schema(0),
            0,
            crate::EventTime::from_micros(10),
            lease,
        )
    } else {
        columnar::restored::copy_row(
            source,
            target.input_schema(0),
            0,
            crate::EventTime::from_micros(10),
            lease,
        )
    }
}

async fn check_resident(service: &TestService, utf8: bool, offset: usize, packed: u8) {
    let target = nullable_operator(utf8);
    let pool = pool(&target);
    let job = job(service, CancellationToken::new());
    let source = record(&target, price(offset, packed));
    let original = decoded(&source);
    assert_eq!(
        original.column(2).to_data().buffers()[0].as_slice(),
        HIDDEN_BITS.to_ne_bytes()
    );
    let paid = resident_bytes(&target, &source, utf8);
    let credit = MemoryConsumer::new("nullable-float-resident").register(&pool);
    credit.try_grow(paid).unwrap();
    let lease = columnar::restored::ResidentLease::new(
        credit,
        job.gather_owner().retain_retirement().unwrap(),
    );
    let mut payload = None;
    let measured =
        allocation_counter::measure(|| payload = Some(copy(&source, &target, lease, utf8)));
    assert!(
        measured.bytes_total <= u64::try_from(paid).unwrap(),
        "{measured:?}; paid={paid}"
    );
    println!("nullable Float64 offset {offset}/Utf8={utf8} requested {measured:?}; paid={paid}");
    let payload = payload.unwrap();
    assert_resident(&payload, &original);
    assert_eq!(
        source.column(2).nulls().unwrap().inner().values(),
        [0, packed]
    );
    let weak = Arc::downgrade(payload.schema_ref());
    let data = payload.columns()[2].to_data();
    let values = data.buffers()[0].clone();
    let bitmap = data.nulls().unwrap().buffer().clone();
    assert_eq!(values.as_slice(), HIDDEN_BITS.to_ne_bytes());
    assert_eq!(bitmap.as_slice(), if offset == 8 { &[0xfe] } else { &[0] });
    drop(data);
    drop(payload);
    drop(target);
    assert_eq!(pool.reserved(), paid);
    let (first, last) = if utf8 {
        (values, bitmap)
    } else {
        (bitmap, values)
    };
    {
        let mut drain = std::pin::pin!(job.gather_owner().close_and_drain());
        assert!(std::future::poll_fn(|cx| Poll::Ready(drain.as_mut().poll(cx).is_pending())).await);
        drop(first);
        assert_eq!(pool.reserved(), paid);
        assert!(weak.upgrade().is_some());
        assert!(!last.is_empty());
        drop(last);
        drain.await;
    }
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
    drop(job);
}

fn assert_resident(payload: &columnar::RowPayload, original: &RecordBatch) {
    let view = payload.view();
    let price = view
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(price.len(), 1);
    assert_eq!(price.null_count(), 1);
    assert!(price.is_null(0));
    assert_eq!(price.value(0).to_bits(), HIDDEN_BITS);
    assert_eq!(price.nulls().unwrap().inner().offset(), 0);
    assert_eq!(row_ipc(&view), row_ipc(original));
    assert_eq!(
        state_row_charge(&view, 0, &[0], "match").unwrap(),
        state_row_charge(original, 0, &[0], "match").unwrap()
    );
}

fn check_valid_row(service: &TestService) {
    let target = nullable_operator(false);
    let source = record(
        &target,
        Float64Array::new(
            vec![-0.0].into(),
            Some(NullBuffer::new(BooleanBuffer::new(
                Buffer::from(vec![0xff_u8]),
                0,
                1,
            ))),
        ),
    );
    let original = decoded(&source);
    assert!(original.column(2).nulls().is_none());
    let pool = pool(&target);
    let paid = resident_bytes(&target, &original, false);
    let job = job(service, CancellationToken::new());
    let credit = MemoryConsumer::new("nullable-valid-resident").register(&pool);
    credit.try_grow(paid).unwrap();
    let lease = columnar::restored::ResidentLease::new(
        credit,
        job.gather_owner().retain_retirement().unwrap(),
    );
    let payload = copy(&original, &target, lease, false);
    assert!(payload.columns()[2].nulls().is_none());
    assert_eq!(row_ipc(&payload.view()), row_ipc(&original));
    drop(payload);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[derive(Clone, Copy)]
enum InvalidBitmap {
    SetBit,
    Empty,
    Outside,
}

fn invalid_bitmap(kind: InvalidBitmap) -> Vec<u8> {
    let source = nullable_operator(true);
    let mut bytes = row_ipc(&record(&source, price(8, 0xfe)));
    let mut offset = 0;
    let (position, mut buffer, body_length) = loop {
        let scalar =
            super::super::super::restore_bases::inspector::message_metadata(&bytes, &mut offset)
                .unwrap();
        let super::super::super::restore_bases::inspector::StreamItem::Message(metadata) = scalar
        else {
            panic!("record batch metadata exists");
        };
        let message = ipc::root_as_message(metadata).unwrap();
        let body_length = usize::try_from(message.bodyLength()).unwrap();
        if let Some(batch) = message.header_as_record_batch() {
            let buffer = batch.buffers().unwrap().get(5);
            break (
                (buffer.0.as_ptr() as usize)
                    .checked_sub(bytes.as_ptr() as usize)
                    .unwrap(),
                *buffer,
                body_length,
            );
        }
        offset += body_length;
    };
    match kind {
        InvalidBitmap::SetBit => bytes[offset + usize::try_from(buffer.offset()).unwrap()] |= 1,
        InvalidBitmap::Empty => buffer.set_length(0),
        InvalidBitmap::Outside => buffer.set_offset(i64::try_from(body_length + 64).unwrap()),
    }
    bytes[position..position + 16].copy_from_slice(&buffer.0);
    bytes
}

fn assert_unchanged(actual: &OperatorStateSnapshot, expected: &OperatorStateSnapshot) {
    assert_eq!(actual.inline_metadata, expected.inline_metadata);
    assert_eq!(actual.segments.len(), expected.segments.len());
    for (id, segment) in &actual.segments {
        assert_eq!(segment.bytes(), expected.segments[id].bytes());
    }
}

fn rejected_outcome(restore: impl FnOnce() -> Result<()>) -> String {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(restore)) {
        Ok(Ok(())) => panic!("malformed IPC must retain Original rejection"),
        Ok(Err(error)) => format!("error: {error}"),
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|text| String::from(*text))
                })
                .expect("Original Arrow panic contains its diagnostic");
            format!("panic: {message}")
        }
    }
}

fn check_original(job: &StreamJobContext, kind: InvalidBitmap) {
    let mut snapshot = nullable_snapshot();
    replace_first_base_ipc(&mut snapshot, &invalid_bitmap(kind));
    let unchanged = snapshot.clone();
    let mut target = nullable_operator(true);
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, job);
    assert!(poll_copy(construction.copy(&target, &snapshot, job)).unwrap());
    let decision = work(&mut construction)
        .run(construction.schema.metadata.control.stop.as_ref().unwrap())
        .unwrap();
    assert!(matches!(decision, RestoreUtf8Decision::Original(_)));
    let actual = rejected_outcome(|| target.install_utf8_fallback(decision, &snapshot, job));
    let expected = rejected_outcome(|| nullable_operator(true).restore(&snapshot));
    assert_eq!(actual, expected);
    assert_eq!(target.status().left.retained_rows, 0);
    assert_eq!(target.state.last_checkpoint_epoch, None);
    assert_unchanged(&snapshot, &unchanged);
    drop(construction);
    drop(target);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_nullable_float_uncertified_bitmaps_and_partial_refusal_keep_original() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let entered = runtime.enter();
    let job = job(&service, CancellationToken::new());
    for kind in [
        InvalidBitmap::SetBit,
        InvalidBitmap::Empty,
        InvalidBitmap::Outside,
    ] {
        check_original(&job, kind);
    }
    check_nonnullable_original(&job);
    check_partial_refusal(&job);
    drop(job);
    drop(entered);
    drop(runtime);
    service.shutdown();
}

fn check_nonnullable_original(job: &StreamJobContext) {
    let snapshot = nullable_snapshot();
    let mut target = operator();
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, job);
    assert!(poll_copy(construction.copy(&target, &snapshot, job)).unwrap());
    let decision = work(&mut construction)
        .run(construction.schema.metadata.control.stop.as_ref().unwrap())
        .unwrap();
    assert!(matches!(decision, RestoreUtf8Decision::Original(_)));
    let actual = target
        .install_utf8_fallback(decision, &snapshot, job)
        .unwrap_err();
    let expected = operator().restore(&snapshot).unwrap_err();
    assert_eq!(actual.to_string(), expected.to_string());
    assert_eq!(target.status().left.retained_rows, 0);
    drop(construction);
    drop(target);
    assert_eq!(pool.reserved(), 0);
}

fn refusal_hook(
    pool: Arc<dyn MemoryPool>,
    pressure: Arc<parking_lot::Mutex<Option<MemoryReservation>>>,
    copies: Arc<AtomicUsize>,
) -> crate::operator::join::DecodedRowTestHook {
    Arc::new(move |workspace, owner, copied, owned| {
        if copied && copies.fetch_add(1, Ordering::Relaxed) == 0 {
            assert!(owned);
            assert!(workspace.unwrap().size() > 0);
            assert!(owner.unwrap().1 > 0);
            let MemoryLimit::Finite(limit) = pool.memory_limit() else {
                panic!("finite configured pool")
            };
            let credit = MemoryConsumer::new("nullable-partial-refusal").register(&pool);
            credit.try_grow(limit - pool.reserved()).unwrap();
            *pressure.lock() = Some(credit);
        }
    })
}

fn check_partial_refusal(job: &StreamJobContext) {
    let snapshot = nullable_snapshot();
    let unchanged = snapshot.clone();
    let mut target = nullable_operator(true);
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, job);
    assert!(poll_copy(construction.copy(&target, &snapshot, job)).unwrap());
    let pressure = Arc::new(parking_lot::Mutex::new(None));
    let copies = Arc::new(AtomicUsize::new(0));
    let mut work = work(&mut construction);
    work.hook = Some(refusal_hook(
        Arc::clone(&pool),
        Arc::clone(&pressure),
        Arc::clone(&copies),
    ));
    let decision = work
        .run(construction.schema.metadata.control.stop.as_ref().unwrap())
        .unwrap();
    assert!(matches!(decision, RestoreUtf8Decision::Original(_)));
    assert_eq!(copies.load(Ordering::Relaxed), 1);
    target
        .install_utf8_fallback(decision, &snapshot, job)
        .unwrap();
    assert!(target.state.left[0].record.funded_owner().is_none());
    assert!(target.state.right[0].record.funded_owner().is_none());
    let mut original = nullable_operator(true);
    original.restore(&snapshot).unwrap();
    assert_eq!(target.status(), original.status());
    assert_unchanged(&snapshot, &unchanged);
    drop(original);
    drop(construction);
    drop(target);
    assert_eq!(pool.reserved(), pressure.lock().as_ref().unwrap().size());
    pressure.lock().take();
    assert_eq!(pool.reserved(), 0);
}
