use super::*;
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, JsonMap, OperatorMetadata,
    StreamJobContext, StreamOperator,
    operator::join::{JoinStateLimits, JoinTimeBounds, StreamJoinSpec},
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
    let mut join = operator();
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

#[tokio::test]
async fn test_fresh_preparation_reuses_owners_without_writer_credit() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "prepared-reuse", None);
    let mut join = seeded(&context).await;
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
    for field in ["generation", "captured_epoch", "revision", "dirty_revision"] {
        let mut join = seeded(&context).await;
        let writers = count_writer(&mut join);
        join.prepare_checkpoint_async(&context).await.unwrap();
        let previous = join.status();
        let rows = Arc::as_ptr(&join.state.left.0);
        let prepared = join.v2_writer.prepared.as_mut().unwrap();
        let segments = prepared.base.segments.clone();
        let owner = std::ptr::from_ref(prepared.base.credit());
        prepared.cut = match field {
            "generation" => cut::Cut {
                generation: prepared.cut.generation + 1,
                ..prepared.cut
            },
            "captured_epoch" => cut::Cut {
                captured_epoch: Some(Epoch::INITIAL),
                ..prepared.cut
            },
            "revision" => cut::Cut {
                revision: prepared.cut.revision + 1,
                ..prepared.cut
            },
            "dirty_revision" => cut::Cut {
                dirty_revision: prepared.cut.dirty_revision + 1,
                ..prepared.cut
            },
            _ => unreachable!(),
        };
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
        pools.push(join.runtime.runtime().unwrap().incremental_memory_pool());
    }
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

#[tokio::test]
async fn test_fresh_preparation_waits_for_retirement_and_rejects_cancellation() {
    let job = job();
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
