use super::*;
use datafusion::arrow::{
    array::{Array, Float64Array},
    buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer},
    ipc::writer::StreamWriter,
};

const HIDDEN_BITS: u64 = 0x7ff8_0000_0000_0055;

fn nullable_operator() -> StreamJoinOperator {
    let mut fields = operator().input_schema(0).fields().to_vec();
    fields.push(Arc::new(Field::new("price", DataType::Float64, true)));
    let schema = Arc::new(Schema::new(fields));
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

fn nullable_row(target: &StreamJoinOperator) -> StoredRow {
    let price = Float64Array::new(
        ScalarBuffer::from(vec![f64::from_bits(HIDDEN_BITS)]),
        Some(NullBuffer::new(BooleanBuffer::new(
            Buffer::from(vec![0xfe_u8]),
            0,
            1,
        ))),
    );
    let record = RecordBatch::try_new(
        Arc::clone(target.input_schema(0)),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![10])),
            Arc::new(price),
        ],
    )
    .unwrap();
    let charge = state_row_charge(&record, 0, &[0], "match").unwrap();
    let encoded_key = encode_join_key_v1(&record, 0, &[0]).unwrap();
    StoredRow {
        record: record.into(),
        event_time: crate::EventTime::from_micros(10),
        row_id: 0,
        charge,
        encoded_key: Arc::new(encoded_key.into()),
    }
}

fn nullable_snapshot() -> OperatorStateSnapshot {
    let mut source = nullable_operator();
    let left = nullable_row(&source);
    let right = left.clone();
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
    assert_eq!(snapshot.segments.len(), 2);
    assert!(
        snapshot
            .segments
            .values()
            .all(|segment| segment.bytes().starts_with(b"CFJOIN1\0"))
    );
    snapshot
}

fn row_ipc(record: &RecordBatch) -> Vec<u8> {
    let mut writer = StreamWriter::try_new(Vec::new(), record.schema_ref()).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    writer.into_inner().unwrap()
}

fn observe_restore(target: &mut StreamJoinOperator) -> [Arc<AtomicUsize>; 3] {
    let counts = std::array::from_fn(|_| Arc::new(AtomicUsize::new(0)));
    let [readers, residents, parses] = counts.clone();
    target.decoded_row_test_hook = Some(Arc::new(move |credit, owner, decoded, owned| {
        assert_eq!(std::thread::current().name(), Some("calc-flow-gather"));
        assert!(owned);
        assert!(credit.unwrap().size() > 0);
        if decoded {
            assert!(owner.unwrap().1 > 0);
            residents.fetch_add(1, Ordering::Relaxed);
        } else {
            readers.fetch_add(1, Ordering::Relaxed);
        }
    }));
    target.metadata_test_hook = Some(Arc::new(move |_, parsing| {
        if parsing {
            assert_eq!(std::thread::current().name(), Some("calc-flow-gather"));
            parses.fetch_add(1, Ordering::Relaxed);
        }
    }));
    counts
}

fn assert_nullable_row(row: &StoredRow, original: &StoredRow) {
    let record = row.record.view();
    assert_eq!(row.row_id, 0);
    assert_eq!(row.event_time.as_micros(), 10);
    assert_eq!(record.num_rows(), 1);
    assert_eq!(
        record
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        7
    );
    assert_eq!(
        record
            .column(1)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        10
    );
    let price = record
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(price.null_count(), 1);
    assert!(price.is_null(0));
    assert_eq!(price.value(0).to_bits(), HIDDEN_BITS);
    assert_eq!(price.values().inner().as_slice(), HIDDEN_BITS.to_ne_bytes());
    assert_eq!(price.nulls().unwrap().inner().offset(), 0);
    assert_eq!(price.nulls().unwrap().buffer().as_slice(), &[0xfe]);
    assert_eq!(record.schema_ref(), original.record.schema_ref());
    assert_eq!(row_ipc(&record), row_ipc(&original.record.view()));
    assert_eq!(row.charge, original.charge);
    assert_eq!(
        state_row_charge(&record, 0, &[0], "match").unwrap(),
        original.charge
    );
    assert!(row.encoded_key == original.encoded_key);
    assert!(row.record.funded_owner().unwrap().1 > 0);
}

#[test]
fn test_scalar_nullable_float_full_bases_use_paid_dispatcher_and_last_buffer() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_restore(&service));
    drop(runtime);
    service.shutdown();
}

async fn check_restore(service: &TestService) {
    let snapshot = nullable_snapshot();
    let unchanged = snapshot.clone();
    let mut original = nullable_operator();
    original.restore(&snapshot).unwrap();
    let facts = inspector::inspect(
        &row_ipc(&original.state.left[0].record.view()),
        original.input_schema(0),
    )
    .unwrap();
    assert_eq!(facts.batches, 1);
    assert!(facts.body_bytes > 0);
    let mut target = nullable_operator();
    let pool = pool(&target);
    let job = job(service, CancellationToken::new());
    let counts = observe_restore(&mut target);
    target
        .restore_managed_metadata(&snapshot, &job, None)
        .await
        .unwrap();
    assert_eq!(counts.map(|count| count.load(Ordering::Relaxed)), [2, 2, 1]);
    assert_eq!(target.state.left.len(), 1);
    assert_eq!(target.state.right.len(), 1);
    assert_eq!(target.state.next_left_row_id, 1);
    assert_eq!(target.state.next_right_row_id, 1);
    assert_eq!(
        target.state.last_checkpoint_epoch,
        Some(Epoch::new(7).unwrap())
    );
    assert_nullable_row(&target.state.left[0], &original.state.left[0]);
    assert_nullable_row(&target.state.right[0], &original.state.right[0]);
    assert_eq!(snapshot.inline_metadata, unchanged.inline_metadata);
    assert_eq!(snapshot.segments, unchanged.segments);
    println!(
        "scalar full-base NULL: 2 native paid readers, 2 paid post-copy rows, 1 parse; hidden bits {HIDDEN_BITS:#x}"
    );
    let payload = &target.state.left[0].record;
    let weak = Arc::downgrade(payload.schema_ref());
    let paid = payload.funded_owner().unwrap().1;
    let data = payload.columns()[2].to_data();
    let values = data.buffers()[0].clone();
    let bitmap = data.nulls().unwrap().buffer().clone();
    drop(data);
    drop(target);
    drop(original);
    assert_eq!(values.as_slice(), HIDDEN_BITS.to_ne_bytes());
    assert_eq!(bitmap.as_slice(), &[0xfe]);
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert_eq!(pool.reserved(), home + generation + paid);
    {
        let mut drain = std::pin::pin!(job.gather_owner().close_and_drain());
        assert!(std::future::poll_fn(|cx| Poll::Ready(drain.as_mut().poll(cx).is_pending())).await);
        drop(bitmap);
        assert!(pool.reserved() >= paid);
        assert!(weak.upgrade().is_some());
        assert_eq!(values.as_slice(), HIDDEN_BITS.to_ne_bytes());
        drop(values);
        drain.await;
    }
    assert!(weak.upgrade().is_none());
    assert_home_credit(&job, &pool);
    drop(job);
    assert_eq!(pool.reserved(), 0);
    println!(
        "scalar full-base NULL: escaped values kept row funding until final Buffer drop; pool refunded to zero"
    );
}
