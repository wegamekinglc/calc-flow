use super::*;
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, IngressProgress, IngressState,
    JoinStateLimits, JoinTimeBounds, JsonMap, OperatorMetadata, StreamJobContext, StreamJoinSpec,
    StreamOperator, runtime::streaming::gather_work::TestService,
};
use datafusion::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::time::Duration;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn operator() -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "at",
        "at",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(2_048, 4_000_000, 4_096).unwrap(),
    )
    .unwrap();
    StreamJoinOperator::new("dense-sparse", schema(), schema(), spec).unwrap()
}

async fn feed(
    join: &mut StreamJoinOperator,
    side: &str,
    times: Vec<i64>,
    context: &StreamOperatorContext<'_>,
) {
    let count = times.len();
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; count])),
            Arc::new(TimestampMicrosecondArray::from(times)),
            Arc::new(Int64Array::from(vec![42; count])),
        ],
    )
    .unwrap();
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    join.process_data(
        side,
        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
        context,
        &mut output,
    )
    .await
    .unwrap();
}

fn shared(payload: &RowPayload) -> (Arc<PayloadChunk>, usize) {
    let RowPayload::Shared { chunk, row } = payload else {
        panic!("fixture needs Shared rows")
    };
    (Arc::clone(chunk), *row)
}

fn check_positions(join: &StreamJoinOperator) {
    for port in 0..2 {
        for (dense, row) in join.sparse_rows(port).indexed() {
            let (chunk, source) = shared(&row.record);
            assert_eq!(chunk.inventory[source].dense.load(Ordering::Relaxed), dense);
            assert_eq!(join.sparse_position(port, &chunk, source), Some(dense));
            assert_eq!(join.sparse_position(1 - port, &chunk, source), None);
        }
    }
}

#[tokio::test]
async fn test_complete_expiry_marks_shared_offsets_dead_and_preserves_alias_funding() {
    let job = StreamJobContext::new(
        1,
        "dense-sparse",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
    .with_checkpointing(false);
    let context = StreamOperatorContext::new(&job, "dense-sparse", None);
    let mut join = operator();
    feed(&mut join, "left", vec![2, 0, 3, 1], &context).await;
    feed(&mut join, "right", vec![2], &context).await;
    let payload = join.state.left.payload_at(0).owner().unwrap();
    let payload_credit = payload.funded_owner().unwrap().1;
    let (old_chunk, _) = shared(payload);
    let buffer = payload.column(2).to_data().buffers()[0].clone();
    let key = Arc::clone(join.state.left.key_at(0));
    let key_credit = key.funded_owner().unwrap().1;
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    join.evict_progress(
        "right",
        IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(4))),
        false,
    )
    .unwrap();
    assert!(join.state.left.is_empty());
    assert!(join.state.left.1.is_none());
    assert_eq!(join.state.left.2.count, 0);
    assert_eq!(old_chunk.live_count.load(Ordering::Relaxed), 0);
    assert_eq!(old_chunk.live_bytes.load(Ordering::Relaxed), 0);
    assert!(!old_chunk.queued.load(Ordering::Relaxed));
    for entry in &old_chunk.inventory {
        assert_eq!(entry.dense.load(Ordering::Relaxed), usize::MAX);
    }
    drop((old_chunk, join, context));
    assert_eq!(pool.reserved(), payload_credit + key_credit);
    let mut drained = Box::pin(job.gather_owner().close_and_drain());
    assert!(
        futures::poll!(drained.as_mut()).is_pending(),
        "payload buffer alias still owns funded retirement"
    );
    drop(key);
    assert_eq!(pool.reserved(), payload_credit);
    drop(buffer);
    assert!(drained.await.is_empty());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_dense_sparse_hints_reject_wrong_offsets_and_retired_aliases() {
    let job = StreamJobContext::new(
        1,
        "dense-sparse",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "dense-sparse", None);
    let mut join = operator();
    for side in ["left", "right"] {
        feed(&mut join, side, vec![2, 0, 3, 1], &context).await;
    }
    check_positions(&join);
    let (old_chunk, _) = shared(&join.state.left.at(0).record);
    join.evict_progress(
        "right",
        IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
        true,
    )
    .unwrap();
    check_positions(&join);
    assert_eq!(join.state.left.at(2).row_id, 3);
    assert_eq!(join.sparse_position(0, &old_chunk, 3), Some(3));
    assert_eq!(join.sparse_position(0, &old_chunk, 1), None);
    let original = join.state.left.at(0).record.clone();
    let reference = join.state.left.indexed().next().unwrap().0;
    let replacement = join.state.left.at(1).record;
    join.state.left.replace_payload(reference, replacement);
    assert_eq!(
        join.sparse_position(0, &old_chunk, 0),
        None,
        "same chunk with wrong source offset is stale"
    );
    join.state.left.replace_payload(reference, original);
    join.reset().unwrap();
    feed(&mut join, "left", vec![2, 0, 3, 1], &context).await;
    for source in 0..old_chunk.inventory.len() {
        assert_eq!(join.sparse_position(0, &old_chunk, source), None);
    }
    check_positions(&join);
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    drop((old_chunk, join, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_dense_sparse_replacement_rebinds_pending_rows_and_v1_restores_each_side() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_replacement_and_restore(&service));
    drop(runtime);
    service.shutdown();
}

async fn check_replacement_and_restore(service: &TestService) {
    let job = StreamJobContext::new(
        1,
        "dense-sparse",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "dense-sparse", None);
    let mut join = operator();
    feed(&mut join, "left", (0..1_024).collect(), &context).await;
    let (old_chunk, _) = shared(&join.state.left.at(0).record);
    join.evict_progress(
        "right",
        IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1_023))),
        true,
    )
    .unwrap();
    assert_eq!(join.sparse_position(0, &old_chunk, 1_023), Some(1_023));
    join.repair_sparse_chunks(&context).await.unwrap();
    let (new_chunk, new_source) = shared(&join.state.left.at(0).record);
    assert!(!Arc::ptr_eq(&old_chunk, &new_chunk));
    assert_eq!(join.sparse_position(0, &old_chunk, 1_023), None);
    assert_eq!(join.sparse_position(0, &new_chunk, new_source), Some(1_023));
    let upsert = join
        .state
        .deltas
        .pending
        .iter()
        .find_map(|op| match op {
            PendingOp::Upsert {
                side: JoinSide::Left,
                record,
                ..
            } => Some(record),
            _ => None,
        })
        .unwrap();
    assert!(upsert.belongs_to(&new_chunk, new_source));
    feed(&mut join, "right", vec![1_025, 1_023], &context).await;
    let snapshot = join.checkpoint_v1(Epoch::INITIAL).unwrap();
    let mut restored = operator();
    restored.prepare_checkpoint_preload_runtime().unwrap();
    let restore_job = StreamJobContext::new(
        1,
        "dense-sparse",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
    .with_gather_owner(service.owner("dense-sparse-restore".into()));
    restored
        .restore_managed_metadata(&snapshot, &restore_job, None)
        .await
        .unwrap();
    check_positions(&restored);
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    drop((old_chunk, new_chunk, restored, snapshot, join, context));
    assert!(
        restore_job
            .gather_owner()
            .close_and_drain()
            .await
            .is_empty()
    );
    drop(restore_job);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
