use super::*;
use datafusion::arrow::{
    array::{Float32Array, Float64Array, PrimitiveArray},
    datatypes::{Float32Type, Float64Type, Int64Type},
    ipc::writer::StreamWriter,
};
use datafusion::execution::memory_pool::MemoryConsumer;

const SINGLE_BITS: u32 = 0x7fc0_1234;
const DOUBLE_BITS: u64 = 0x8000_0000_0000_0000;

fn float_operator(nullable: bool) -> StreamJoinOperator {
    let mut fields = operator().input_schema(0).fields().to_vec();
    fields.push(Arc::new(Field::new("single", DataType::Float32, nullable)));
    fields.push(Arc::new(Field::new("double", DataType::Float64, false)));
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

fn float_row(operator: &StreamJoinOperator, null: bool) -> StoredRow {
    let record = RecordBatch::try_new(
        Arc::clone(operator.input_schema(0)),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![10])),
            Arc::new(Float32Array::from(vec![
                (!null).then(|| f32::from_bits(SINGLE_BITS)),
            ])),
            Arc::new(Float64Array::from(vec![f64::from_bits(DOUBLE_BITS)])),
        ],
    )
    .unwrap();
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

fn float_snapshot(source: &mut StreamJoinOperator, null: bool) -> OperatorStateSnapshot {
    let left = float_row(source, null);
    let right = float_row(source, null);
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

fn row_ipc(record: &RecordBatch) -> Vec<u8> {
    let mut writer = StreamWriter::try_new(Vec::new(), record.schema_ref()).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    writer.into_inner().unwrap()
}

#[test]
fn test_restored_float_resident_allocation_and_last_buffer_refund() {
    assert_eq!(
        size_of::<PrimitiveArray<Float32Type>>(),
        size_of::<PrimitiveArray<Int64Type>>()
    );
    assert_eq!(
        size_of::<PrimitiveArray<Float64Type>>(),
        size_of::<PrimitiveArray<Int64Type>>()
    );
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_float_resident(&service));
    drop(runtime);
    service.shutdown();
}

async fn check_float_resident(service: &TestService) {
    let target = float_operator(false);
    let pool = pool(&target);
    let job = job(service, CancellationToken::new());
    let original = float_row(&target, false);
    let record = original.record.view();
    let paid = columnar::restored::required(
        target.input_schema(0),
        super::super::super::super::inventory::registration_controls().unwrap(),
    )
    .unwrap();
    let credit = MemoryConsumer::new("float-resident").register(&pool);
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
    println!("float resident requested allocations {measured:?}; paid={paid}");
    let payload = payload.unwrap();
    assert_eq!(row_ipc(&payload.view()), row_ipc(&record));
    assert_eq!(
        state_row_charge(&payload.view(), 0, &[0], "match").unwrap(),
        original.charge
    );
    let weak = Arc::downgrade(payload.schema_ref());
    let single = payload.columns()[2].to_data().buffers()[0].clone();
    let double = payload.columns()[3].to_data().buffers()[0].clone();
    drop(payload);
    drop(target);
    assert_eq!(pool.reserved(), paid);
    assert_eq!(
        u32::from_ne_bytes(single.as_slice().try_into().unwrap()),
        SINGLE_BITS
    );
    assert_eq!(
        u64::from_ne_bytes(double.as_slice().try_into().unwrap()),
        DOUBLE_BITS
    );
    {
        let mut drain = std::pin::pin!(job.gather_owner().close_and_drain());
        assert!(std::future::poll_fn(|cx| Poll::Ready(drain.as_mut().poll(cx).is_pending())).await);
        drop(single);
        assert_eq!(pool.reserved(), paid);
        assert!(weak.upgrade().is_some());
        drop(double);
        drain.await;
    }
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
    drop(job);
}

#[test]
fn test_restored_float_null_payload_keeps_original_reader_and_wire() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let entered = runtime.enter();
    let mut source = float_operator(true);
    let snapshot = float_snapshot(&mut source, true);
    let mut target = float_operator(true);
    let pool = pool(&target);
    let job = job(&service, CancellationToken::new());
    let mut construction = construction(&target, &snapshot, &job);
    assert!(poll_copy(construction.copy(&target, &snapshot, &job)).unwrap());
    let record = source.state.left[0].record.view();
    assert!(inspector::inspect(&row_ipc(&record), record.schema_ref()).is_none());
    let decision = work(&mut construction)
        .run(construction.schema.metadata.control.stop.as_ref().unwrap())
        .unwrap();
    assert!(matches!(decision, RestoreBasesDecision::Original(_)));
    target
        .install_bases_decision(decision, &snapshot, &job)
        .unwrap();
    let mut original = float_operator(true);
    original.restore(&snapshot).unwrap();
    assert_eq!(target.status(), original.status());
    for side in [&target.state.left, &target.state.right] {
        assert!(side[0].record.funded_owner().is_none());
        assert_eq!(side[0].record.view().column(2).null_count(), 1);
        assert_eq!(row_ipc(&side[0].record.view()), row_ipc(&record));
    }
    assert_eq!(
        encode_side(&target.state.left, "match", "left", &|| Ok(())).unwrap(),
        snapshot.segments["left-base"].bytes()
    );
    drop(construction);
    assert_eq!(pool.reserved(), 0);
    drop(entered);
    drop(runtime);
    service.shutdown();
}
