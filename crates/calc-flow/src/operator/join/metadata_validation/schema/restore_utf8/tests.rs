use super::*;
use crate::{
    CancellationToken, Epoch, JsonMap, OperatorStateSnapshot, StreamJobContext, StreamOperator,
    operator::join::{
        JoinStateLimits, JoinTimeBounds, StreamJoinOperator, StreamJoinSpec, encode_join_key_v1,
        encode_side, state_row_charge,
    },
    runtime::streaming::gather_work::TestService,
};
use datafusion::arrow::{
    array::{Float64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    ipc::writer::StreamWriter,
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool};
use std::{
    future::Future,
    task::{Context, Poll, Waker},
    time::Duration,
};

#[path = "nullable_tests.rs"]
mod nullable_tests;

fn operator() -> StreamJoinOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("symbol", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("price", DataType::Float64, false),
    ]));
    let spec = StreamJoinSpec::inner(
        ["symbol"],
        ["symbol"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(100, 4_194_304, 100).unwrap(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("match", Arc::clone(&schema), schema, spec).unwrap();
    operator.prepare_checkpoint_preload_runtime().unwrap();
    operator
}

fn job(service: &TestService, cancellation: CancellationToken) -> StreamJobContext {
    StreamJobContext::new(1, "fingerprint", JsonMap::new(), None, cancellation)
        .with_gather_owner(service.owner("utf8-restore-test".into()))
}

fn pool(operator: &StreamJoinOperator) -> Arc<dyn MemoryPool> {
    operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool()
}

fn row(operator: &StreamJoinOperator, id: u64, symbol: &str) -> StoredRow {
    let record = RecordBatch::try_new(
        Arc::clone(operator.input_schema(0)),
        vec![
            Arc::new(StringArray::from(vec![symbol])),
            Arc::new(TimestampMicrosecondArray::from(vec![10])),
            Arc::new(Float64Array::from(vec![-0.0])),
        ],
    )
    .unwrap();
    let charge = state_row_charge(&record, 0, &[0], "match").unwrap();
    let key = encode_join_key_v1(&record, 0, &[0]).unwrap();
    StoredRow {
        record: record.into(),
        event_time: crate::EventTime::from_micros(10),
        row_id: id,
        charge,
        encoded_key: Arc::new(key.into()),
    }
}

fn snapshot(source: &mut StreamJoinOperator) -> OperatorStateSnapshot {
    let long = "猫🙂".repeat(256);
    let left = vec![
        row(source, 0, ""),
        row(source, 1, "猫🙂"),
        row(source, 2, &long),
    ];
    let right = vec![row(source, 0, "猫🙂")];
    source.state.metrics.left.retained_rows = left.len() as u64;
    source.state.metrics.left.retained_bytes = left.iter().map(|row| row.charge).sum();
    source.state.metrics.right.retained_rows = 1;
    source.state.metrics.right.retained_bytes = right[0].charge;
    source.state.left = left.into();
    source.state.right = right.into();
    source.state.next_left_row_id = 3;
    source.state.next_right_row_id = 1;
    let mut snapshot = source.checkpoint_v1(Epoch::new(7).unwrap()).unwrap();
    for (side, rows) in [("left", &source.state.left), ("right", &source.state.right)] {
        snapshot.segments.insert(
            format!("{side}-base"),
            StateSegment::new(encode_side(rows, "match", side, &|| Ok(())).unwrap()),
        );
    }
    assert_eq!(snapshot.segments.len(), 2);
    snapshot
}

fn poll_copy<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

fn construction(
    target: &StreamJoinOperator,
    snapshot: &OperatorStateSnapshot,
    job: &StreamJobContext,
) -> RestoreUtf8Construction {
    poll_copy(target.utf8_construction(snapshot, job))
        .unwrap()
        .expect("valid two-base Utf8 restore needs a paid construction")
}

fn work(construction: &mut RestoreUtf8Construction) -> RestoreUtf8Work {
    RestoreUtf8Work {
        schema: Some(construction.schema_work(None, None)),
        input: construction.input.take(),
        workspace: construction.workspace.take(),
        geometry: construction.geometry,
        hook: None,
        schema_hook: None,
    }
}

fn row_ipc(record: &RecordBatch) -> Vec<u8> {
    let mut writer = StreamWriter::try_new(Vec::new(), record.schema_ref()).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    writer.into_inner().unwrap()
}

fn mixed_snapshot() -> OperatorStateSnapshot {
    use crate::operator::join::{JoinSide, PendingOp};
    let base = snapshot(&mut operator());
    let mut source = operator();
    source.restore(&base).unwrap();
    let added = row(&source, 3, "猫🙃🙂");
    let mut left = source.state.left.iter().cloned().collect::<Vec<_>>();
    left.push(added.clone());
    source.state.left = left.into();
    source.state.next_left_row_id = 4;
    source.state.metrics.left.retained_rows = 4;
    source.state.metrics.left.retained_bytes = source.state.left.iter().map(|row| row.charge).sum();
    source.state.deltas.pending.push(PendingOp::Upsert {
        side: JoinSide::Left,
        row_id: added.row_id,
        event_time: added.event_time,
        encoded_key: added.encoded_key,
        record: added.record,
        charge: added.charge,
    });
    let mixed = source.checkpoint_v1(Epoch::new(8).unwrap()).unwrap();
    assert_eq!(mixed.segments.len(), 3);
    for side in ["left", "right"] {
        let segment = &mixed.segments[&format!("{side}-base")];
        assert!(segment.bytes().starts_with(b"CFJOIN1\0"));
        assert_eq!(
            segment.bytes(),
            base.segments[&format!("{side}-base")].bytes()
        );
    }
    let delta = mixed.segments["left-delta-8"].bytes();
    assert!(delta.starts_with(b"CFJDLT1\0"));
    assert_eq!(u64::from_le_bytes(delta[8..16].try_into().unwrap()), 1);
    mixed
}

fn assert_mixed_rows(target: &StreamJoinOperator) {
    let long = "猫🙂".repeat(256);
    for (rows, symbols) in [
        (
            &target.state.left,
            vec!["", "猫🙂", long.as_str(), "猫🙃🙂"],
        ),
        (&target.state.right, vec!["猫🙂"]),
    ] {
        assert_eq!(rows.len(), symbols.len());
        for (id, symbol) in symbols.into_iter().enumerate() {
            let id = u64::try_from(id).unwrap();
            let actual = rows.iter().find(|row| row.row_id == id).unwrap();
            let expected = row(target, id, symbol);
            let view = actual.record.view();
            assert_eq!(
                view.column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .value(0),
                symbol
            );
            assert_eq!(
                view.column(1)
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap()
                    .value(0),
                10
            );
            assert_eq!(
                view.column(2)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(0)
                    .to_bits(),
                (-0.0_f64).to_bits()
            );
            assert_eq!(actual.event_time, expected.event_time);
            assert_eq!(actual.charge, expected.charge);
            assert_eq!(
                actual.encoded_key.as_slice(),
                expected.encoded_key.as_slice()
            );
            assert!(actual.record.funded_owner().unwrap().1 > 0);
        }
    }
    assert_eq!(
        (
            target.state.next_left_row_id,
            target.state.next_right_row_id
        ),
        (4, 1)
    );
}

fn mixed_work(
    construction: &mut RestoreUtf8Construction,
) -> (
    RestoreUtf8Work,
    Arc<std::sync::atomic::AtomicUsize>,
    Arc<std::sync::atomic::AtomicUsize>,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let readers = Arc::new(AtomicUsize::new(0));
    let residents = Arc::new(AtomicUsize::new(0));
    let observed_readers = Arc::clone(&readers);
    let observed_residents = Arc::clone(&residents);
    let mut work = work(construction);
    work.hook = Some(Arc::new(move |credit, owner, copied, owned_work| {
        assert!(owned_work);
        assert!(credit.unwrap().size() > 0);
        if copied {
            assert!(owner.unwrap().1 > 0);
            observed_residents.fetch_add(1, Ordering::Relaxed);
        } else {
            observed_readers.fetch_add(1, Ordering::Relaxed);
        }
    }));
    (work, readers, residents)
}

#[test]
fn test_utf8_mixed_bases_delta_preserve_rows_installation_and_last_buffer_credit() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_mixed_restore(&service));
    drop(runtime);
    service.shutdown();
}

async fn check_mixed_restore(service: &TestService) {
    use std::sync::atomic::Ordering;
    let snapshot = mixed_snapshot();
    let unchanged = snapshot.clone();
    let mut target = operator();
    let job = job(service, CancellationToken::new());
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, &job);
    assert!(construction.copy(&target, &snapshot, &job).await.unwrap());
    let (work, readers, residents) = mixed_work(&mut construction);
    let mut decision = None;
    let measured = allocation_counter::measure(|| {
        decision = Some(
            work.run(construction.schema.metadata.control.stop.as_ref().unwrap())
                .unwrap(),
        );
    });
    let paid = pool.reserved();
    assert!(matches!(decision, Some(RestoreUtf8Decision::Prepared(_))));
    assert_eq!(readers.load(Ordering::Relaxed), 5);
    assert_eq!(residents.load(Ordering::Relaxed), 5);
    let tail = allocation_counter::measure(|| {
        target
            .install_utf8_decision(decision.take().unwrap(), &snapshot, &job)
            .unwrap();
    });
    assert!(
        measured.bytes_total + tail.bytes_total <= paid as u64,
        "{measured:?}/{tail:?}; paid={paid}"
    );
    println!("mixed Utf8 reader/key/fold/install requests {measured:?}/{tail:?}; paid={paid}");
    assert_mixed_rows(&target);
    assert_eq!(target.state.last_checkpoint_epoch, Epoch::new(8));
    assert_eq!(snapshot.inline_metadata, unchanged.inline_metadata);
    for (id, segment) in &snapshot.segments {
        assert_eq!(segment.bytes(), unchanged.segments[id].bytes());
    }
    let mut original = operator();
    original.restore(&snapshot).unwrap();
    assert_eq!(target.status(), original.status());
    let next = target.checkpoint_v1(Epoch::new(9).unwrap()).unwrap();
    let expected = original.checkpoint_v1(Epoch::new(9).unwrap()).unwrap();
    assert_eq!(next.inline_metadata, expected.inline_metadata);
    assert_eq!(next.segments.len(), expected.segments.len());
    for (id, segment) in &next.segments {
        assert_eq!(segment.bytes(), expected.segments[id].bytes());
    }
    drop(next);
    drop(expected);
    drop(original);
    drop(construction);
    check_mixed_last_buffers(target, job).await;
}

async fn check_mixed_last_buffers(target: StreamJoinOperator, job: StreamJobContext) {
    let pool = pool(&target);
    let row = target
        .state
        .left
        .iter()
        .find(|row| row.row_id == 3)
        .unwrap();
    let paid = row.record.funded_owner().unwrap().1;
    let weak = Arc::downgrade(row.record.schema_ref());
    let view = row.record.view();
    let data = view.column(0).to_data();
    let offsets = data.buffers()[0].clone();
    let values = data.buffers()[1].clone();
    assert_eq!(offsets.len(), 8);
    assert_eq!(
        i32::from_ne_bytes(offsets.as_slice()[4..8].try_into().unwrap()),
        i32::try_from("猫🙃🙂".len()).unwrap()
    );
    assert_eq!(values.as_slice(), "猫🙃🙂".as_bytes());
    drop(data);
    drop(view);
    drop(target);
    assert_eq!(pool.reserved(), paid);
    {
        let mut drain = std::pin::pin!(job.gather_owner().close_and_drain());
        assert!(std::future::poll_fn(|cx| Poll::Ready(drain.as_mut().poll(cx).is_pending())).await);
        drop(offsets);
        assert_eq!(pool.reserved(), paid);
        assert!(weak.upgrade().is_some());
        assert_eq!(values.as_slice(), "猫🙃🙂".as_bytes());
        drop(values);
        drain.await;
    }
    assert!(weak.upgrade().is_none());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_utf8_full_bases_fund_variable_keys_and_complete_installation() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let entered = runtime.enter();
    let snapshot = snapshot(&mut operator());
    let mut target = operator();
    let job = job(&service, CancellationToken::new());
    let pool = pool(&target);
    let mut construction = construction(&target, &snapshot, &job);
    assert!(poll_copy(construction.copy(&target, &snapshot, &job)).unwrap());
    let mut decision = None;
    let measured = allocation_counter::measure(|| {
        decision = Some(
            work(&mut construction)
                .run(construction.schema.metadata.control.stop.as_ref().unwrap())
                .unwrap(),
        );
    });
    let paid = pool.reserved();
    let mut original = operator();
    original.restore(&snapshot).unwrap();
    assert!(matches!(decision, Some(RestoreUtf8Decision::Prepared(_))));
    let tail = allocation_counter::measure(|| {
        target
            .install_utf8_decision(decision.take().unwrap(), &snapshot, &job)
            .unwrap();
    });
    assert!(
        measured.bytes_total + tail.bytes_total <= paid as u64,
        "{measured:?}/{tail:?}; paid={paid}"
    );
    println!(
        "variable Utf8 both-side reader/key/fold/install requests {measured:?}/{tail:?}; paid={paid}"
    );
    assert_eq!(target.status(), original.status());
    for side in ["left", "right"] {
        let rows = if side == "left" {
            &target.state.left
        } else {
            &target.state.right
        };
        assert_eq!(
            encode_side(rows, "match", side, &|| Ok(())).unwrap(),
            snapshot.segments[&format!("{side}-base")].bytes()
        );
        assert!(rows.iter().all(|row| row.record.funded_owner().is_some()));
    }
    drop(construction);
    drop(target);
    drop(job);
    assert_eq!(pool.reserved(), 0);
    drop(entered);
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_utf8_fresh_offsets_values_allocation_and_last_owner_refund() {
    use datafusion::arrow::array::GenericStringArray;
    assert!(size_of::<GenericStringArray<i32>>() <= size_of::<GenericStringArray<i64>>());
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_resident_buffers(&service, "猫🙂"));
    runtime.block_on(check_resident_buffers(&service, ""));
    drop(runtime);
    service.shutdown();
}

fn string_record(target: &StreamJoinOperator, strings: StringArray) -> RecordBatch {
    RecordBatch::try_new(
        Arc::clone(target.input_schema(0)),
        vec![
            Arc::new(strings),
            Arc::new(TimestampMicrosecondArray::from(vec![10])),
            Arc::new(Float64Array::from(vec![-0.0])),
        ],
    )
    .unwrap()
}

async fn check_resident_buffers(service: &TestService, selected: &str) {
    let target = operator();
    let pool = pool(&target);
    let job = job(service, CancellationToken::new());
    let source = StringArray::from(vec!["prefix", selected, "suffix"]).slice(1, 1);
    assert_eq!(source.value_offsets()[0], 6);
    let record = string_record(&target, source);
    let paid = columnar::restored::utf8::required(
        &record,
        target.input_schema(0),
        super::super::super::inventory::registration_controls().unwrap(),
    )
    .unwrap();
    let credit = MemoryConsumer::new("utf8-resident").register(&pool);
    credit.try_grow(paid).unwrap();
    let lease = columnar::restored::ResidentLease::new(
        credit,
        job.gather_owner().retain_retirement().unwrap(),
    );
    let mut payload = None;
    let measured = allocation_counter::measure(|| {
        payload = Some(columnar::restored::utf8::copy_row(
            &record,
            target.input_schema(0),
            0,
            crate::EventTime::from_micros(10),
            lease,
        ));
    });
    assert!(
        measured.bytes_total <= paid as u64,
        "{measured:?}; paid={paid}"
    );
    println!(
        "Utf8 selected={} resident requests {measured:?}; paid={paid}",
        selected.len()
    );
    let payload = payload.unwrap();
    assert_eq!(row_ipc(&payload.view()), row_ipc(&record));
    assert_eq!(
        state_row_charge(&payload.view(), 0, &[0], "match").unwrap(),
        state_row_charge(&record, 0, &[0], "match").unwrap()
    );
    let weak = Arc::downgrade(payload.schema_ref());
    let data = payload.columns()[0].to_data();
    let offsets = data.buffers()[0].clone();
    let values = data.buffers()[1].clone();
    assert_eq!(offsets.len(), 8);
    assert_eq!(
        usize::try_from(i32::from_ne_bytes(
            offsets.as_slice()[4..8].try_into().unwrap()
        ))
        .unwrap(),
        selected.len()
    );
    assert_eq!(values.as_slice(), selected.as_bytes());
    drop(data);
    drop(payload);
    drop(target);
    assert_eq!(pool.reserved(), paid);
    {
        let mut drain = std::pin::pin!(job.gather_owner().close_and_drain());
        assert!(std::future::poll_fn(|cx| Poll::Ready(drain.as_mut().poll(cx).is_pending())).await);
        drop(offsets);
        assert_eq!(pool.reserved(), paid);
        assert!(weak.upgrade().is_some());
        drop(values);
        drain.await;
    }
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
    drop(job);
}

fn change_string_ipc(mut bytes: Vec<u8>, reversed: bool) -> Vec<u8> {
    let mut offset = 0;
    let _ = super::super::restore_bases::inspector::message_metadata(&bytes, &mut offset).unwrap();
    let metadata =
        super::super::restore_bases::inspector::message_metadata(&bytes, &mut offset).unwrap();
    let super::super::restore_bases::inspector::StreamItem::Message(metadata) = metadata else {
        panic!("expected record batch metadata")
    };
    let message = datafusion::arrow::ipc::root_as_message(metadata).unwrap();
    let buffers = message.header_as_record_batch().unwrap().buffers().unwrap();
    let offsets = usize::try_from(buffers.get(1).offset()).unwrap() + offset;
    let values = usize::try_from(buffers.get(2).offset()).unwrap() + offset;
    if reversed {
        bytes[offsets..offsets + 4].copy_from_slice(&2_i32.to_le_bytes());
        bytes[offsets + 4..offsets + 8].copy_from_slice(&1_i32.to_le_bytes());
    } else {
        bytes[offsets..offsets + 4].copy_from_slice(&1_i32.to_le_bytes());
        bytes[values] = 0xff;
    }
    bytes
}

fn replace_first_base_ipc(snapshot: &mut OperatorStateSnapshot, bytes: &[u8]) {
    let original = snapshot.segments["left-base"].bytes();
    let mut base = original[..8].to_vec();
    base.extend_from_slice(&1_u64.to_le_bytes());
    base.extend_from_slice(&original[16..40]);
    base.extend_from_slice(&u64::try_from(bytes.len()).unwrap().to_le_bytes());
    base.extend_from_slice(bytes);
    snapshot
        .segments
        .insert("left-base".into(), StateSegment::new(base));
}

#[test]
fn test_utf8_unproved_offsets_keep_original_reader_and_error_priority() {
    use datafusion::arrow::ipc::reader::StreamReader;
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let entered = runtime.enter();
    let target = operator();
    let job = job(&service, CancellationToken::new());
    let record = string_record(&target, StringArray::from(vec!["za"]));
    let unused = change_string_ipc(row_ipc(&record), false);
    let original = StreamReader::try_new(std::io::Cursor::new(&unused), None)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(
        original
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "a"
    );
    let stop = GatherStop::from_job(&job);
    assert!(
        inspector::inspect(&unused, target.input_schema(0), &[0], &stop)
            .unwrap()
            .is_none()
    );
    check_bad_offsets(&job, &change_string_ipc(row_ipc(&record), true));
    drop(job);
    drop(entered);
    drop(runtime);
    service.shutdown();
}

fn check_bad_offsets(job: &StreamJobContext, ipc: &[u8]) {
    let mut snapshot = snapshot(&mut operator());
    replace_first_base_ipc(&mut snapshot, ipc);
    let unchanged = snapshot.clone();
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
    assert_eq!(snapshot.inline_metadata, unchanged.inline_metadata);
    assert_eq!(snapshot.segments.len(), unchanged.segments.len());
    for (id, segment) in &snapshot.segments {
        assert_eq!(segment.bytes(), unchanged.segments[id].bytes());
    }
    drop(construction);
    drop(target);
    assert_eq!(pool.reserved(), 0);
}

async fn submit(
    target: &mut StreamJoinOperator,
    construction: &mut RestoreUtf8Construction,
    job: &StreamJobContext,
) -> ObservedTicket<RestoreUtf8Decision> {
    construction.schema.metadata.control.scope = Some(
        job.gather_owner()
            .client(GatherOperatorId::new(Arc::from("match")))
            .scope()
            .unwrap(),
    );
    construction
        .submit(
            &mut target.compaction_cleanup,
            target.metadata_test_hook.clone(),
            target.schema_test_hook.clone(),
            target.decoded_row_test_hook.clone(),
        )
        .await
        .unwrap()
}

fn assert_home_credit(job: &StreamJobContext, pool: &Arc<dyn MemoryPool>) {
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
    assert!(home > 0);
    assert_eq!(pool.reserved(), home);
}

#[test]
fn test_utf8_dynamic_refusal_and_cancel_keep_state_and_actual_refunds() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_dynamic_refusal(&service));
    runtime.block_on(check_cancel_reader(&service));
    runtime.block_on(check_partial_copy_drop(&service));
    drop(runtime);
    service.shutdown();
}

async fn check_dynamic_refusal(service: &TestService) {
    use datafusion::execution::memory_pool::MemoryLimit;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let snapshot = snapshot(&mut operator());
    let mut target = operator();
    let pool = pool(&target);
    let job = job(service, CancellationToken::new());
    let parses = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&parses);
    target.metadata_test_hook = Some(Arc::new(move |_, parsing| {
        if parsing {
            observed.fetch_add(1, Ordering::Relaxed);
        }
    }));
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let entered = std::sync::Mutex::new(Some(entered));
    let wait = std::sync::Mutex::new(wait);
    target.schema_test_hook = Some(Arc::new(move |_, building| {
        if building {
            entered.lock().unwrap().take().unwrap().send(()).unwrap();
            wait.lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }
    }));
    let readers = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&readers);
    target.decoded_row_test_hook = Some(Arc::new(move |credit, _, copied, native| {
        assert!(!native);
        assert!(credit.is_none());
        if !copied {
            observed.fetch_add(1, Ordering::Relaxed);
        }
    }));
    let mut construction = construction(&target, &snapshot, &job);
    assert!(construction.copy(&target, &snapshot, &job).await.unwrap());
    let ticket = submit(&mut target, &mut construction, &job).await;
    started.await.unwrap();
    let MemoryLimit::Finite(limit) = pool.memory_limit() else {
        panic!("finite pool")
    };
    let pressure = MemoryConsumer::new("utf8-dynamic-refusal").register(&pool);
    pressure.try_grow(limit - pool.reserved()).unwrap();
    release.send(()).unwrap();
    let stop = construction
        .schema
        .metadata
        .control
        .stop
        .as_ref()
        .unwrap()
        .clone();
    assert!(
        target
            .finish_utf8_ticket(ticket, &snapshot, &job, &stop)
            .await
            .unwrap()
    );
    assert_eq!(parses.load(Ordering::Relaxed), 1);
    assert_eq!(readers.load(Ordering::Relaxed), 4);
    let mut original = operator();
    original.restore(&snapshot).unwrap();
    assert_eq!(target.status(), original.status());
    assert!(target.compaction_cleanup.is_none());
    drop(construction);
    drop(target);
    job.gather_owner().close_and_drain().await;
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
    assert_eq!(pool.reserved(), pressure.size() + home);
    drop(job);
    assert_eq!(pool.reserved(), pressure.size());
    drop(pressure);
    assert_eq!(pool.reserved(), 0);
}

async fn check_cancel_reader(service: &TestService) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let snapshot = snapshot(&mut operator());
    let mut target = operator();
    let pool = pool(&target);
    let cancellation = CancellationToken::new();
    let job = job(service, cancellation.clone());
    let readers = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&readers);
    target.decoded_row_test_hook = Some(Arc::new(move |credit, _, copied, native| {
        if !copied && observed.fetch_add(1, Ordering::Relaxed) == 0 {
            assert!(native);
            assert!(credit.unwrap().size() > 0);
            assert_eq!(std::thread::current().name(), Some("calc-flow-gather"));
            cancellation.cancel();
        }
    }));
    let mut construction = construction(&target, &snapshot, &job);
    assert!(construction.copy(&target, &snapshot, &job).await.unwrap());
    let result = submit(&mut target, &mut construction, &job)
        .await
        .finish()
        .await;
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(readers.load(Ordering::Relaxed), 1);
    assert_eq!(target.status().left.retained_rows, 0);
    assert_eq!(target.state.last_checkpoint_epoch, None);
    drop(result);
    drop(construction);
    drop(target);
    job.gather_owner().close_and_drain().await;
    assert_home_credit(&job, &pool);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn check_partial_copy_drop(service: &TestService) {
    let snapshot = snapshot(&mut operator());
    let target = operator();
    let pool = pool(&target);
    let job = job(service, CancellationToken::new());
    let mut construction = construction(&target, &snapshot, &job);
    let paid = pool.reserved();
    {
        let mut copy = std::pin::pin!(construction.copy(&target, &snapshot, &job));
        assert!(std::future::poll_fn(|cx| Poll::Ready(copy.as_mut().poll(cx).is_pending())).await);
        assert_eq!(pool.reserved(), paid);
    }
    drop(construction);
    assert_eq!(pool.reserved(), 0);
    job.gather_owner().close_and_drain().await;
    drop(job);
}
