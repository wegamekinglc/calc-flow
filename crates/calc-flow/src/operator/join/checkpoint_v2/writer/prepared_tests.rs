use super::*;
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, JsonMap, OperatorMetadata,
    StreamJobContext, StreamOperator,
    operator::join::{JoinStateLimits, JoinTimeBounds, StreamJoinSpec, with_one_gib_pool},
};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

#[path = "pending_capture_tests.rs"]
mod pending_capture_tests;

#[path = "rebase_tests.rs"]
mod rebase_tests;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("text", DataType::Utf8, true),
    ]))
}

fn operator() -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "at",
        "at",
        JoinTimeBounds::new(Duration::ZERO, Duration::from_micros(10)).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 100).unwrap(),
    )
    .unwrap();
    StreamJoinOperator::new("prepared-reuse", schema(), schema(), spec).unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn isolated_writer_test(
    check: impl FnOnce(&crate::runtime::streaming::gather_work::TestService, &tokio::runtime::Runtime),
) {
    let service = crate::runtime::streaming::gather_work::TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    check(&service, &runtime);
    drop(runtime);
    service.shutdown();
}

fn record(side: &str) -> RecordBatch {
    let (times, values) = if side == "left" {
        (vec![95, 96], vec![None, Some("猫")])
    } else {
        (vec![100], vec![Some("ok")])
    };
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; times.len()])),
            Arc::new(TimestampMicrosecondArray::from(times)),
            Arc::new(StringArray::from(values)),
        ],
    )
    .unwrap()
}

async fn seeded(context: &StreamOperatorContext<'_>) -> StreamJoinOperator {
    seeded_from(context, operator()).await
}

async fn seeded_from(
    context: &StreamOperatorContext<'_>,
    mut join: StreamJoinOperator,
) -> StreamJoinOperator {
    for side in ["left", "right"] {
        let mut output = EdgeCollector::new(join.output_ports().to_vec());
        join.process_data(
            side,
            Batch::table(vec![record(side)], BatchMetadata::default()).unwrap(),
            context,
            &mut output,
        )
        .await
        .unwrap();
    }
    join
}

fn count_writer(join: &mut StreamJoinOperator) -> Arc<AtomicUsize> {
    let count = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&count);
    join.set_checkpoint_writer_test_hook(Arc::new(move |credit| {
        assert!(credit.size() > 0);
        assert_eq!(std::thread::current().name(), Some("calc-flow-gather"));
        observed.fetch_add(1, Ordering::SeqCst);
    }));
    count
}

fn assert_restored(snapshot: &OperatorStateSnapshot, expected: &StreamJoinOperator) {
    let mut restored = operator();
    restored.restore(snapshot).unwrap();
    assert_eq!(
        restored.v2_writer.tracker.identity,
        restored.checkpoint_identity
    );
    assert_eq!(
        restored.v2_writer.history_identity,
        Some(restored.checkpoint_identity)
    );
    assert_eq!(restored.status(), expected.status());
    assert_eq!(restored.state.last_checkpoint_epoch, Some(Epoch::INITIAL));
    assert_eq!(restored.state.next_left_row_id, 2);
    assert_eq!(restored.state.next_right_row_id, 1);
    assert_eq!(restored.state.next_output_sequence, 1);
    for (side, rows, original) in [
        ("left", &restored.state.left, &expected.state.left),
        ("right", &restored.state.right, &expected.state.right),
    ] {
        let input = record(side);
        assert_eq!(rows.len(), input.num_rows());
        for (index, (row, original)) in rows.iter().zip(original.iter()).enumerate() {
            assert_eq!(row.row_id, u64::try_from(index).unwrap());
            assert_eq!(row.event_time, original.event_time);
            assert_eq!(row.encoded_key.as_slice(), original.encoded_key.as_slice());
            assert_eq!(row.charge, original.charge);
            assert_eq!(*row.record.view(), input.slice(index, 1));
        }
    }
}

#[test]
fn test_fresh_preparation_reuses_owners_without_writer_credit() {
    isolated_writer_test(|service, runtime| {
        runtime.block_on(assert_fresh_preparation_reuses_owners(service));
    });
}

async fn assert_fresh_preparation_reuses_owners(
    service: &crate::runtime::streaming::gather_work::TestService,
) {
    let job = job().with_gather_owner(service.owner("prepared-owner-reuse".into()));
    let context = StreamOperatorContext::new(&job, "prepared-reuse", None);
    let mut join = seeded_from(&context, with_one_gib_pool(operator())).await;
    let writers = count_writer(&mut join);
    join.prepare_checkpoint_async(&context).await.unwrap();
    let prepared = join.v2_writer.prepared.as_ref().unwrap();
    let segments = prepared.base.segments.clone();
    let owner = std::ptr::from_ref(prepared.base.credit());
    let owned_bytes = prepared.base.credit().size();
    let previous = join.status();
    let rows = Arc::as_ptr(&join.state.left.0);
    let runtime = join.runtime.runtime().unwrap();
    let pool = runtime.incremental_memory_pool();
    let pressure = runtime.incremental_reservation("prepared-reuse-pressure");
    pressure.try_grow((1 << 30) - pool.reserved()).unwrap();
    join.prepare_checkpoint_async(&context)
        .await
        .expect("a matching fresh preparation needs no second writer credit");
    assert_eq!(writers.load(Ordering::SeqCst), 1);
    assert_eq!(pool.reserved(), 1 << 30);
    assert_eq!(join.status(), previous);
    assert_eq!(Arc::as_ptr(&join.state.left.0), rows);
    let repeated = &join.v2_writer.prepared.as_ref().unwrap().base;
    assert_eq!(std::ptr::from_ref(repeated.credit()), owner);
    assert_eq!(repeated.credit().size(), owned_bytes);
    for (name, segment) in &segments {
        assert!(Arc::ptr_eq(
            &segment.bytes_arc(),
            &repeated.segments[name].bytes_arc()
        ));
    }
    drop(pressure);
    let snapshot = join.checkpoint(Epoch::INITIAL).unwrap();
    assert_restored(&snapshot, &join);
    let mut reference = seeded(&context).await;
    reference.prepare_checkpoint_async(&context).await.unwrap();
    let expected = reference.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata, expected.inline_metadata);
    assert_eq!(snapshot.segments.len(), expected.segments.len());
    for (name, segment) in &snapshot.segments {
        assert_eq!(segment.bytes(), expected.segments[name].bytes());
    }
    drop((snapshot, join, expected, reference));
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    let backing = segments
        .values()
        .map(|segment| segment.bytes_arc().capacity())
        .sum::<usize>();
    assert!(pool.reserved() >= backing);
    for segment in segments.into_values() {
        let before = pool.reserved();
        let capacity = segment.bytes_arc().capacity();
        drop(segment);
        assert!(before - pool.reserved() >= capacity);
    }
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_fresh_preparation_rewrites_each_stale_cut_component() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "prepared-reuse", None);
    let mut pools = Vec::new();
    for field in [
        "generation",
        "captured_epoch",
        "revision",
        "dirty_revision",
        "identity",
    ] {
        let mut join = seeded(&context).await;
        let writers = count_writer(&mut join);
        join.prepare_checkpoint_async(&context).await.unwrap();
        let previous = join.status();
        let rows = Arc::as_ptr(&join.state.left.0);
        let prepared = join.v2_writer.prepared.as_mut().unwrap();
        let segments = prepared.base.segments.clone();
        let owner = std::ptr::from_ref(prepared.base.credit());
        prepared.cut = stale_cut(prepared.cut, field);
        assert!(!join.fresh_v2_preparation_matches(), "stale {field}");
        join.prepare_checkpoint_async(&context).await.unwrap();
        assert_eq!(writers.load(Ordering::SeqCst), 2, "stale {field}");
        assert_eq!(join.status(), previous);
        assert_eq!(Arc::as_ptr(&join.state.left.0), rows);
        assert!(join.fresh_v2_preparation_matches());
        let replacement = &join.v2_writer.prepared.as_ref().unwrap().base;
        assert_ne!(std::ptr::from_ref(replacement.credit()), owner);
        assert_eq!(replacement.segments.len(), segments.len());
        for (name, segment) in &segments {
            assert!(!Arc::ptr_eq(
                &segment.bytes_arc(),
                &replacement.segments[name].bytes_arc()
            ));
            assert_eq!(segment.bytes(), replacement.segments[name].bytes());
        }
        let snapshot = join.checkpoint(Epoch::INITIAL).unwrap();
        assert_restored(&snapshot, &join);
        if field == "identity" {
            assert_reset_identity(&mut join, &context).await;
        }
        pools.push(join.runtime.runtime().unwrap().incremental_memory_pool());
    }
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

fn stale_cut(original: cut::Cut, field: &str) -> cut::Cut {
    match field {
        "generation" => cut::Cut {
            generation: original.generation + 1,
            ..original
        },
        "captured_epoch" => cut::Cut {
            captured_epoch: Some(Epoch::INITIAL),
            ..original
        },
        "revision" => cut::Cut {
            revision: original.revision + 1,
            ..original
        },
        "identity" => cut::Cut {
            identity: different_identity(original.identity),
            ..original
        },
        "dirty_revision" => cut::Cut {
            dirty_revision: original.dirty_revision + 1,
            ..original
        },
        _ => unreachable!(),
    }
}

#[test]
fn test_fresh_preparation_waits_for_retirement_and_rejects_cancellation() {
    isolated_writer_test(|service, runtime| {
        runtime.block_on(assert_fresh_preparation_waits_and_rejects_cancellation(
            service,
        ));
    });
}

async fn assert_fresh_preparation_waits_and_rejects_cancellation(
    service: &crate::runtime::streaming::gather_work::TestService,
) {
    let job = job().with_gather_owner(service.owner("prepared-retirement-cancel".into()));
    let context = StreamOperatorContext::new(&job, "prepared-reuse", None);
    let mut join = seeded(&context).await;
    let writers = count_writer(&mut join);
    join.prepare_checkpoint_async(&context).await.unwrap();
    let previous = join.status();
    let prepared = join.v2_writer.prepared.as_ref().unwrap();
    let owner = std::ptr::from_ref(prepared.base.credit());
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let live = pool.reserved();
    let (released, receiver) = tokio::sync::oneshot::channel();
    join.compaction_release = Some(receiver);
    let mut preparing = Box::pin(join.prepare_checkpoint_async(&context));
    assert!(futures::poll!(preparing.as_mut()).is_pending());
    assert_eq!(writers.load(Ordering::SeqCst), 1);
    released.send(()).unwrap();
    preparing.await.unwrap();
    assert!(join.compaction_release.is_none());
    assert_eq!(writers.load(Ordering::SeqCst), 1);
    assert_eq!(pool.reserved(), live);

    let (released, receiver) = tokio::sync::oneshot::channel();
    join.compaction_release = Some(receiver);
    let mut preparing = Box::pin(join.prepare_checkpoint_async(&context));
    assert!(futures::poll!(preparing.as_mut()).is_pending());
    job.cancellation().cancel();
    assert!(matches!(
        preparing.await,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert!(join.fresh_v2_preparation_matches());
    assert!(matches!(
        join.prepare_checkpoint_async(&context).await,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(join.status(), previous);
    assert_eq!(writers.load(Ordering::SeqCst), 1);
    assert_eq!(
        std::ptr::from_ref(join.v2_writer.prepared.as_ref().unwrap().base.credit()),
        owner
    );
    assert_eq!(pool.reserved(), live);
    drop((released, join));
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn different_identity(identity: HistoryIdentity) -> HistoryIdentity {
    let mut bytes = identity.0;
    bytes[0] ^= 1;
    HistoryIdentity(bytes)
}

async fn assert_reset_identity(join: &mut StreamJoinOperator, context: &StreamOperatorContext<'_>) {
    join.reset().unwrap();
    assert_eq!(join.v2_writer.tracker.identity, join.checkpoint_identity);
    assert!(join.v2_writer.history_identity.is_none());
    join.prepare_checkpoint_async(context).await.unwrap();
    assert!(join.fresh_v2_preparation_matches());
    let snapshot = join.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(
        join.v2_writer.history_identity,
        Some(join.checkpoint_identity)
    );
    let mut restored = operator();
    restored.restore(&snapshot).unwrap();
    assert_same_empty_identity(join, &restored);
    assert_worker_identity_refusal(join, context).await;
}

fn assert_same_empty_identity(expected: &StreamJoinOperator, actual: &StreamJoinOperator) {
    assert_eq!(actual.status(), expected.status());
    assert_eq!(
        actual.v2_writer.tracker.identity,
        actual.checkpoint_identity
    );
    assert_eq!(
        actual.v2_writer.history_identity,
        Some(actual.checkpoint_identity)
    );
}

async fn assert_worker_identity_refusal(
    join: &mut StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
) {
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let baseline = pool.reserved();
    let status = join.status();
    let tracker = join.v2_writer.tracker;
    let history_identity = join.v2_writer.history_identity;
    let workspace = join.v2_writer_workspace().unwrap();
    let mut work = join
        .v2_writer_work(Preparation::Rebase, workspace.new_empty(), context)
        .await
        .unwrap();
    work.inputs.history.as_mut().unwrap().identity = different_identity(work.identity);
    let error = work
        .encode(&GatherStop::from_job(context.job()))
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("rebase history layout differs from worker")
    );
    drop(work);
    join.await_compaction_release(context).await.unwrap();
    drop(workspace);
    assert_eq!(join.status(), status);
    assert_eq!(join.v2_writer.tracker, tracker);
    assert_eq!(join.v2_writer.history_identity, history_identity);
    assert_eq!(pool.reserved(), baseline);
}
