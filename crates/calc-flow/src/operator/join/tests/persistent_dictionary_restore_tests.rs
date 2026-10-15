use super::*;
use datafusion::execution::memory_pool::{
    MemoryLimit, MemoryPool, MemoryReservation, UnboundedMemoryPool,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[path = "persistent_dictionary_duplicate_id_tests.rs"]
mod duplicate_id_tests;

#[test]
fn test_persistent_dictionary_historical_diagnostic_overlap_is_funded() {
    let observations = [340, 600, 1024].map(|name_bytes| {
        let name = "\u{7f}".repeat(name_bytes);
        let funded =
            metadata_validation::schema::restore_bases::inventory::diagnostic_bytes(&name).unwrap();
        let mut errors = None;
        let measured = allocation_counter::measure(|| {
            let historical =
                checkpoint_error(&name, "left", "row event time or charge is inconsistent");
            let original =
                checkpoint_error(&name, "left", "row event time or charge is inconsistent");
            errors = Some((historical, original));
            std::hint::black_box(&errors);
        });
        assert!(matches!(
            &errors,
            Some((
                CalcFlowError::CheckpointMismatch { .. },
                CalcFlowError::CheckpointMismatch { .. }
            ))
        ));
        drop(errors);
        eprintln!("historical diagnostic name bytes={name_bytes}: {measured:?}; funded={funded}");
        (name_bytes, funded, measured)
    });
    for (name_bytes, funded, measured) in observations {
        assert!(
            measured.bytes_max <= u64::try_from(funded).unwrap(),
            "retained historical error overlaps original diagnostic, name bytes={name_bytes}: {measured:?}; funded={funded}"
        );
    }
}

#[tokio::test]
async fn test_persistent_dictionary_restored_layouts_share_current_identities_and_continue() {
    for (layout, projected) in [(1, false), (2, true), (3, true)] {
        Box::pin(continued_layout(layout, projected)).await;
    }
}

async fn continued_layout(layout: u64, projected: bool) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut source, snapshot) = captured(layout, &context).await;
    let mut restored = operator(projected);
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    restored
        .process_data("right", batch(9, 99), &context, &mut output)
        .await
        .unwrap();
    let old = canonical_right(&restored);
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status(), source.status());
    assert_shared_current(&restored, 2);
    assert!(restored.key_dictionary.lease(&old).is_none());
    assert!(!restored.key_dictionary_needs_rebuild);
    assert_fees(&restored, 1, 1);
    continue_native(&mut restored, &context, &mut output, projected).await;
    let pools = [&mut restored, &mut source]
        .map(|join| join.runtime.runtime().unwrap().incremental_memory_pool());
    drop((source, restored, snapshot, output, old, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

async fn continue_native(
    operator: &mut StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
    output: &mut EdgeCollector,
    projected: bool,
) {
    reset_join_work();
    operator
        .process_data("left", batch(7, 30), context, output)
        .await
        .unwrap();
    assert_eq!(join_work().key_encodings, 0);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert_values(output, projected);
    assert_shared_current(operator, 3);
    assert_fees(operator, 2, 1);
    assert_eq!(
        (
            operator.state.next_left_row_id,
            operator.state.next_right_row_id,
            operator.state.next_output_sequence
        ),
        (2, 1, 2)
    );
}

#[tokio::test]
async fn test_persistent_dictionary_valid_refused_restore_installs_new_sql_state() {
    for (layout, managed) in [(2, false), (1, true)] {
        Box::pin(valid_refused_restore(layout, managed)).await;
    }
}

async fn valid_refused_restore(layout: u64, managed: bool) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut source, snapshot) = captured(layout, &context).await;
    let mut target = operator(false);
    let pool = Arc::new(DictionaryPool::default());
    target
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    let mut output = EdgeCollector::new(target.output_ports().to_vec());
    target
        .process_data("right", batch(9, 99), &context, &mut output)
        .await
        .unwrap();
    let old = canonical_right(&target);
    let previous_writer = target.v2_writer.observation_for_test();
    pool.refuse.store(true, Ordering::SeqCst);
    restore_snapshot(&mut target, &snapshot, &context, managed)
        .await
        .unwrap();
    assert!(
        pool.refusals.load(Ordering::SeqCst) > 0,
        "restore must attempt a real optional dictionary grant"
    );
    assert_eq!(target.status(), source.status());
    assert_ne!(target.v2_writer.observation_for_test(), previous_writer);
    assert_fees(&target, 1, 1);
    assert_eq!(retained_keys(&target), [7, 7]);
    assert_eq!(target.state.last_checkpoint_epoch, Some(Epoch::INITIAL));
    assert!(target.state.deltas.pending.is_empty());
    assert!(target.key_dictionary_needs_rebuild);
    assert_eq!(target.key_dictionary.inventory().0, 0);
    assert!(target.key_dictionary.lease(&old).is_none());
    assert!(target.key_bindings.left.get(0).is_none());
    assert!(target.key_bindings.right.get(0).is_none());
    assert!(output.drain("output").is_empty());
    refused_rebuild_preserves_candidate(&mut target, &pool, &snapshot, &job).await;
    pool.refuse.store(false, Ordering::SeqCst);
    let progress = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, None),
    );
    target.on_ingress_progress("left", &progress).await.unwrap();
    assert!(!target.key_dictionary_needs_rebuild);
    assert_shared_current(&target, 2);
    continue_native(&mut target, &context, &mut output, false).await;
    let source_pool = source.runtime.runtime().unwrap().incremental_memory_pool();
    drop((
        source,
        snapshot,
        target,
        output,
        old,
        previous_writer,
        progress,
        context,
    ));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(source_pool.reserved(), 0);
}

async fn refused_rebuild_preserves_candidate(
    target: &mut StreamJoinOperator,
    pool: &DictionaryPool,
    snapshot: &OperatorStateSnapshot,
    job: &StreamJobContext,
) {
    let installed = state_observation(target);
    let refused = pool.refusals.load(Ordering::SeqCst);
    let progress = progress_context(
        job,
        (IngressState::Active, None),
        (IngressState::Active, None),
    );
    target.on_ingress_progress("left", &progress).await.unwrap();
    assert_eq!(state_observation(target), installed);
    assert!(pool.refusals.load(Ordering::SeqCst) > refused);
    assert!(target.key_dictionary_needs_rebuild);
    let legacy = snapshot.inline_metadata["layout_version"] == 1;
    if legacy {
        assert_legacy_history(target, snapshot);
    }
    target.prepare_checkpoint_async(&progress).await.unwrap();
    let clean = target.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(clean.inline_metadata["layout_version"], 2);
    assert_eq!(
        clean.inline_metadata["metrics"],
        snapshot.inline_metadata["metrics"]
    );
    if !legacy {
        for (name, segment) in &snapshot.segments {
            let retained = clean.segments.get(name).unwrap();
            assert_eq!(retained.bytes(), segment.bytes());
            assert!(Arc::ptr_eq(&retained.bytes_arc(), &segment.bytes_arc()));
        }
    }
}

fn assert_legacy_history(target: &StreamJoinOperator, snapshot: &OperatorStateSnapshot) {
    let bases = carried_base_segments(snapshot);
    let deltas = carried_delta_segments(snapshot, "match").unwrap();
    assert_eq!(bases.len() + deltas.len(), snapshot.segments.len());
    for (side, segment) in bases {
        let retained = &target.state.deltas.base[side];
        assert_eq!(retained.bytes(), segment.bytes());
        assert!(Arc::ptr_eq(&retained.bytes_arc(), &segment.bytes_arc()));
    }
    for (identity, segment) in deltas {
        let retained = &target.state.deltas.segments[&identity];
        assert_eq!(retained.bytes(), segment.bytes());
        assert!(Arc::ptr_eq(&retained.bytes_arc(), &segment.bytes_arc()));
    }
}

#[tokio::test]
async fn test_persistent_dictionary_invalid_or_cancelled_restore_keeps_old_native_state() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut source, snapshot) = captured(2, &context).await;
    let mut target = operator(false);
    let mut output = EdgeCollector::new(target.output_ports().to_vec());
    target
        .process_data("right", batch(9, 99), &context, &mut output)
        .await
        .unwrap();
    let old = canonical_right(&target);
    let before = state_observation(&target);
    let writer = target.v2_writer.observation_for_test();
    let mut invalid = snapshot.clone();
    invalid
        .inline_metadata
        .insert("next_right_row_id".into(), 0.into());
    assert!(matches!(
        target.restore(&invalid),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(state_observation(&target), before);
    assert_eq!(target.v2_writer.observation_for_test(), writer);
    let cancellation = target.restore_v2_checked_for_test(&snapshot, &|| {
        Err(CalcFlowError::Cancelled {
            run_id: "dictionary-restore-control".into(),
        })
    });
    assert!(matches!(cancellation, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(state_observation(&target), before);
    assert_eq!(target.v2_writer.observation_for_test(), writer);
    assert!(Arc::ptr_eq(&old, &canonical_right(&target)));
    assert!(output.drain("output").is_empty());
    reset_join_work();
    target
        .process_data("left", batch(9, 40), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(join_work().key_encodings, 0);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert_eq!(output.drain("output")[0].as_data().unwrap().num_rows(), 1);
    assert_shared_current(&target, 2);
    let source_pool = source.runtime.runtime().unwrap().incremental_memory_pool();
    let target_pool = target.runtime.runtime().unwrap().incremental_memory_pool();
    drop((
        source, snapshot, target, invalid, output, old, before, writer, context,
    ));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(source_pool.reserved(), 0);
    assert_eq!(target_pool.reserved(), 0);
}

#[tokio::test]
async fn test_persistent_dictionary_dead_v1_upsert_fee_is_checked_before_tombstone_fold() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let (mut source, dead) = dead_v1_snapshot(&context).await;
    let mut pools = Vec::new();
    for managed in [false, true] {
        pools.extend(Box::pin(reject_dead_fee(&dead, &context, managed)).await);
    }
    pools.push(source.runtime.runtime().unwrap().incremental_memory_pool());
    drop((source, dead, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

#[test]
fn test_persistent_dictionary_full_v1_derived_worker_cancel_and_drop_preserve_old_state() {
    checkpoint_compaction_tests::isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(derived_worker_boundary(service, false));
        runtime.block_on(derived_worker_boundary(service, true));
    });
}

async fn derived_worker_boundary(
    service: &crate::runtime::streaming::gather_work::TestService,
    cancel: bool,
) {
    let owner = job().with_gather_owner(service.owner("dictionary-restore-worker".into()));
    let context = StreamOperatorContext::new(&owner, "match", None);
    let (mut source, mut snapshot) = captured(1, &context).await;
    let mut target = operator(false);
    let mut output = EdgeCollector::new(target.output_ports().to_vec());
    target
        .process_data("right", batch(9, 99), &context, &mut output)
        .await
        .unwrap();
    let before = state_observation(&target);
    let writer = target.v2_writer.observation_for_test();
    let old = canonical_right(&target);
    let pool = target.runtime.runtime().unwrap().incremental_memory_pool();
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let gate = std::sync::Mutex::new(Some((entered, wait)));
    target.key_restore_test_hook = Some(Arc::new(move |rows| {
        let (entered, wait) = gate.lock().unwrap().take().unwrap();
        entered
            .send((
                Arc::downgrade(rows.at(0).record.column(0)),
                std::thread::current().id(),
            ))
            .unwrap();
        wait.recv_timeout(Duration::from_secs(10)).unwrap();
    }));
    let mut restore = Box::pin(target.restore_managed_snapshot(&mut snapshot, &owner, None));
    let (candidate, worker_thread) = tokio::select! {
        result = started => result.unwrap(),
        result = &mut restore => panic!("restore escaped its actual derived-worker gate: {result:?}"),
    };
    assert_ne!(worker_thread, std::thread::current().id());
    let cancelled = if cancel {
        owner.cancellation().cancel();
        matches!(
            tokio::time::timeout(Duration::from_secs(1), restore.as_mut()).await,
            Ok(Err(CalcFlowError::Cancelled { .. }))
        )
    } else {
        true
    };
    drop(restore);
    assert!(target.legacy_restore_capture.is_none());
    assert_eq!(state_observation(&target), before);
    assert_eq!(target.v2_writer.observation_for_test(), writer);
    assert!(Arc::ptr_eq(&old, &canonical_right(&target)));
    assert!(output.drain("output").is_empty());
    assert!(candidate.upgrade().is_some());
    assert!(owner.gather_owner().funding().2 > 0);
    healthy_after_derived_release(&mut target, &owner, &mut output, release).await;
    assert!(candidate.upgrade().is_none());
    assert_eq!(owner.gather_owner().funding().2, 0);
    let source_pool = source.runtime.runtime().unwrap().incremental_memory_pool();
    drop((
        source, target, snapshot, output, before, writer, old, context,
    ));
    assert!(owner.gather_owner().close_and_drain().await.is_empty());
    drop(owner);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(source_pool.reserved(), 0);
    assert!(
        cancelled,
        "cancellation must return while the actual worker is held"
    );
}

async fn healthy_after_derived_release(
    target: &mut StreamJoinOperator,
    owner: &StreamJobContext,
    output: &mut EdgeCollector,
    release: std::sync::mpsc::Sender<()>,
) {
    let healthy = job().with_gather_owner(owner.gather_owner().clone());
    let context = StreamOperatorContext::new(&healthy, "match", None);
    reset_join_work();
    let mut process = Box::pin(target.process_data("left", batch(9, 40), &context, output));
    assert!(
        futures::poll!(process.as_mut()).is_pending(),
        "healthy mutation must await the abandoned restore worker"
    );
    release.send(()).unwrap();
    process.await.unwrap();
    assert_eq!(join_work().key_encodings, 0);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert_eq!(output.drain("output")[0].as_data().unwrap().num_rows(), 1);
    assert_shared_current(target, 2);
}

async fn dead_v1_snapshot(
    context: &StreamOperatorContext<'_>,
) -> (StreamJoinOperator, OperatorStateSnapshot) {
    let mut source = operator(false);
    source.checkpoint_v1_test_producer = true;
    let mut output = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data("left", batch(7, 20), context, &mut output)
        .await
        .unwrap();
    source.state.deltas.needs_compaction = true;
    source.prepare_checkpoint_async(context).await.unwrap();
    let first = source.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(&first.segments["left-base"].bytes()[..8], JOIN_STATE_MAGIC);
    source
        .evict_progress(
            "right",
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(101))),
            true,
        )
        .unwrap();
    assert!(source.state.left.is_empty());
    assert!(
        source
            .state
            .deltas
            .pending
            .iter()
            .all(|op| matches!(op, PendingOp::Tombstone { .. }))
    );
    let dead = source.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(dead.inline_metadata["layout_version"], 1);
    assert!(dead.segments.contains_key("left-delta-2"));
    assert_eq!(source.status().left.retained_bytes, 0);
    assert!(output.drain("output").is_empty());
    (source, dead)
}

async fn reject_dead_fee(
    dead: &OperatorStateSnapshot,
    context: &StreamOperatorContext<'_>,
    managed: bool,
) -> [Arc<dyn MemoryPool>; 2] {
    let mut valid = operator(false);
    restore_snapshot(&mut valid, dead, context, managed)
        .await
        .unwrap();
    assert!(valid.state.left.is_empty() && valid.state.right.is_empty());
    let mut target = operator(false);
    let mut output = EdgeCollector::new(target.output_ports().to_vec());
    target
        .process_data("right", batch(9, 99), context, &mut output)
        .await
        .unwrap();
    let before = state_observation(&target);
    let writer = target.v2_writer.observation_for_test();
    let damaged = corrupt_dead_fee(dead);
    let result = restore_snapshot(&mut target, &damaged, context, managed).await;
    assert!(
        matches!(result, Err(CalcFlowError::CheckpointMismatch { .. })),
        "tombstoned V1 historical charge must be checked, managed={managed}: {result:?}"
    );
    assert_eq!(state_observation(&target), before);
    assert_eq!(target.v2_writer.observation_for_test(), writer);
    assert!(output.drain("output").is_empty());
    target
        .process_data("left", batch(9, 40), context, &mut output)
        .await
        .unwrap();
    assert_eq!(output.drain("output")[0].as_data().unwrap().num_rows(), 1);
    assert_shared_current(&target, 2);
    let pools = [&mut valid, &mut target]
        .map(|join| join.runtime.runtime().unwrap().incremental_memory_pool());
    drop((valid, target, output, before, writer, damaged));
    pools
}

async fn restore_snapshot(
    target: &mut StreamJoinOperator,
    snapshot: &OperatorStateSnapshot,
    context: &StreamOperatorContext<'_>,
    managed: bool,
) -> Result<()> {
    if managed {
        target
            .restore_managed_snapshot(&mut snapshot.clone(), context.job(), None)
            .await
    } else {
        target.restore(snapshot)
    }
}

fn corrupt_dead_fee(dead: &OperatorStateSnapshot) -> OperatorStateSnapshot {
    let mut bytes = dead.segments["left-base"].bytes().to_vec();
    assert_eq!(&bytes[..8], JOIN_STATE_MAGIC);
    let mut count = 8;
    assert_eq!(read_segment_u64(&bytes, &mut count).unwrap(), 1);
    let mut offset = 32;
    let charge = read_segment_u64(&bytes, &mut offset).unwrap();
    assert_eq!(charge, 133);
    bytes[32..offset].copy_from_slice(&(charge + 1).to_le_bytes());
    let mut damaged = dead.clone();
    damaged
        .segments
        .insert("left-base".into(), StateSegment::new(bytes));
    damaged
}

async fn captured(
    layout: u64,
    context: &StreamOperatorContext<'_>,
) -> (StreamJoinOperator, OperatorStateSnapshot) {
    let mut source = operator(layout == 3);
    source.checkpoint_v1_test_producer = layout == 1;
    let mut output = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data("right", batch(7, 10), context, &mut output)
        .await
        .unwrap();
    source
        .process_data("left", batch(7, 20), context, &mut output)
        .await
        .unwrap();
    assert_eq!(output.drain("output")[0].as_data().unwrap().num_rows(), 1);
    assert_fees(&source, 1, 1);
    source.prepare_checkpoint_async(context).await.unwrap();
    let snapshot = source.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["layout_version"], layout);
    (source, snapshot)
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("unused", DataType::Utf8, false),
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn operator(projected: bool) -> StreamJoinOperator {
    let declaration = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 1000).unwrap(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("match", schema(), schema(), declaration).unwrap();
    if projected {
        operator.set_output_projection(vec![3, 7]).unwrap();
    }
    operator
}

fn batch(key: i64, value: i64) -> Batch {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(vec!["drop"])),
            Arc::new(Int64Array::from(vec![key])),
            Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![value])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn canonical_right(operator: &StreamJoinOperator) -> Arc<columnar::FramedKey> {
    Arc::clone(
        operator
            .key_bindings
            .right
            .canonical(&operator.state.right.at(0)),
    )
}

fn assert_shared_current(operator: &StreamJoinOperator, references: u64) {
    assert_eq!(
        (
            operator.key_dictionary.inventory().0,
            operator.key_dictionary.inventory().1
        ),
        (1, references)
    );
    let first = canonical_right(operator);
    let lease = operator.key_dictionary.lease(&first).unwrap();
    for (bindings, rows) in [
        (&operator.key_bindings.left, &operator.state.left),
        (&operator.key_bindings.right, &operator.state.right),
    ] {
        for row in rows.iter() {
            let current = bindings.canonical(&row);
            assert_eq!(
                operator.key_dictionary.lease(current).unwrap().address(),
                lease.address()
            );
            assert!(Arc::ptr_eq(current, &first));
            assert_eq!(current.as_slice(), row.encoded_key.as_slice());
        }
    }
}

fn assert_fees(operator: &StreamJoinOperator, left: u64, right: u64) {
    for row in operator
        .state
        .left
        .iter()
        .chain(operator.state.right.iter())
    {
        assert_eq!(
            row.charge, 133,
            "full V1 logical fee must include the removed unused payload"
        );
    }
    assert_eq!(operator.status().left.retained_bytes, 133 * left);
    assert_eq!(operator.status().right.retained_bytes, 133 * right);
}

fn retained_keys(operator: &StreamJoinOperator) -> Vec<i64> {
    operator
        .state
        .left
        .iter()
        .chain(operator.state.right.iter())
        .map(|row| {
            let index = row.record.schema_ref().index_of("key").unwrap();
            row.record
                .column_view(index)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0)
        })
        .collect()
}

fn assert_values(output: &mut EdgeCollector, projected: bool) {
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    let record = &messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(record.num_rows(), 1);
    let (columns, left, right) = if projected { (2, 0, 1) } else { (8, 3, 7) };
    assert_eq!(record.num_columns(), columns);
    let value = |index| {
        record
            .column(index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    };
    assert_eq!((value(left), value(right)), (30, 10));
}

fn state_observation(operator: &StreamJoinOperator) -> impl PartialEq + fmt::Debug + use<> {
    (
        operator.status(),
        operator.key_dictionary.inventory(),
        (
            operator.state.next_left_row_id,
            operator.state.next_right_row_id,
            operator.state.next_output_sequence,
            operator.state.last_checkpoint_epoch,
        ),
        rows_observation(&operator.state.left),
        rows_observation(&operator.state.right),
        operator
            .state
            .deltas
            .pending
            .iter()
            .map(|op| match op {
                PendingOp::Upsert {
                    side,
                    row_id,
                    event_time,
                    encoded_key,
                    charge,
                    ..
                } => (
                    *side,
                    *row_id,
                    *event_time,
                    encoded_key.to_vec(),
                    Some(*charge),
                ),
                PendingOp::Tombstone {
                    side,
                    row_id,
                    event_time,
                    encoded_key,
                } => (*side, *row_id, *event_time, encoded_key.to_vec(), None),
            })
            .collect::<Vec<_>>(),
    )
}

fn rows_observation(rows: &dyn Rows) -> impl PartialEq + fmt::Debug + use<> {
    rows.iter()
        .map(|row| {
            (
                row.row_id,
                row.event_time,
                row.charge,
                row.encoded_key.to_vec(),
                Arc::as_ptr(&row.encoded_key) as usize,
                (*row.record.view()).clone(),
            )
        })
        .collect::<Vec<_>>()
}

#[derive(Debug, Default)]
struct DictionaryPool {
    inner: UnboundedMemoryPool,
    refuse: AtomicBool,
    refusals: AtomicUsize,
}

impl fmt::Display for DictionaryPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("restore dictionary refusal control")
    }
}

impl MemoryPool for DictionaryPool {
    fn name(&self) -> &'static str {
        "restore-dictionary-refusal"
    }
    fn grow(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.grow(credit, bytes);
    }
    fn shrink(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.shrink(credit, bytes);
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
    fn try_grow(&self, credit: &MemoryReservation, bytes: usize) -> datafusion::common::Result<()> {
        if self.refuse.load(Ordering::SeqCst)
            && bytes > 0
            && credit.consumer().name() == "sql-incremental:stream-join-dictionary"
        {
            self.refusals.fetch_add(1, Ordering::SeqCst);
            return Err(datafusion::common::DataFusionError::ResourcesExhausted(
                "restore dictionary budget refused".into(),
            ));
        }
        self.inner.try_grow(credit, bytes)
    }
}
