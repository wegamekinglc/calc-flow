use super::*;
use datafusion::execution::memory_pool::{MemoryLimit, MemoryReservation};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug)]
struct GrowthPool {
    inner: GreedyMemoryPool,
    reject: AtomicBool,
    rejected: AtomicUsize,
}

impl fmt::Display for GrowthPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)
    }
}

impl MemoryPool for GrowthPool {
    fn name(&self) -> &'static str {
        "expiration-growth"
    }
    fn grow(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.grow(credit, bytes);
    }
    fn shrink(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.shrink(credit, bytes);
    }
    fn try_grow(&self, credit: &MemoryReservation, bytes: usize) -> datafusion::common::Result<()> {
        if self.reject.load(Ordering::SeqCst)
            && bytes > 0
            && credit.consumer().name() == "sql-incremental:stream-join-v2-restore"
        {
            self.rejected.fetch_add(1, Ordering::SeqCst);
            return Err(datafusion::common::DataFusionError::ResourcesExhausted(
                "heap growth rejected".into(),
            ));
        }
        self.inner.try_grow(credit, bytes)
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
}

#[tokio::test]
async fn test_restored_heap_growth_refusal_precedes_operator_output_and_allows_retry() {
    for initial_left in [0, 1] {
        let source_job = job();
        let source_context = StreamOperatorContext::new(&source_job, "match", None);
        let mut source =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let mut output = EdgeCollector::new(source.output_ports().to_vec());
        if initial_left != 0 {
            source
                .process_data("left", left_batch(vec![0]), &source_context, &mut output)
                .await
                .unwrap();
        }
        source
            .process_data("right", right_batch(vec![0]), &source_context, &mut output)
            .await
            .unwrap();
        let snapshot = source.checkpoint(Epoch::INITIAL).unwrap();
        let pool = Arc::new(GrowthPool {
            inner: GreedyMemoryPool::new(16 << 20),
            reject: AtomicBool::new(false),
            rejected: AtomicUsize::new(0),
        });
        let mut restored =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        restored
            .runtime
            .initialize_benchmark_pool(pool.clone())
            .unwrap();
        restored.restore(&snapshot).unwrap();
        let target_job = job();
        let context = StreamOperatorContext::new(&target_job, "match", None);
        let mut output = EdgeCollector::new(restored.output_ports().to_vec());
        let before = restored.status();
        let next_id = restored.state.next_left_row_id;
        let sequence = restored.state.next_output_sequence;
        pool.reject.store(true, Ordering::SeqCst);
        let error = restored
            .process_data("left", left_batch(vec![1, 2]), &context, &mut output)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("credit admission"));
        assert_eq!(pool.rejected.load(Ordering::SeqCst), 1);
        assert!(output.drain("output").is_empty());
        assert_eq!(restored.status(), before);
        assert_eq!(restored.state.next_left_row_id, next_id);
        assert_eq!(restored.state.next_output_sequence, sequence);
        assert!(restored.state.deltas.pending.is_empty());
        pool.reject.store(false, Ordering::SeqCst);
        restored
            .process_data("left", left_batch(vec![1, 2]), &context, &mut output)
            .await
            .unwrap();
        assert!(!output.drain("output").is_empty());
        assert_eq!(restored.state.next_left_row_id, next_id + 2);
        assert_eq!(restored.state.left.len(), initial_left + 2);
        drop((restored, context, snapshot, source, source_context, output));
        assert!(source_job.gather_owner().close_and_drain().await.is_empty());
        assert!(target_job.gather_owner().close_and_drain().await.is_empty());
        drop((source_job, target_job));
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_restored_complete_expiry_preserves_funded_growth_refusal_and_retry() {
    let source_job = job();
    let source_context = StreamOperatorContext::new(&source_job, "match", None);
    let mut source =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let mut output = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data(
            "left",
            left_batch(vec![2, 0, 1]),
            &source_context,
            &mut output,
        )
        .await
        .unwrap();
    source
        .process_data("right", right_batch(vec![0]), &source_context, &mut output)
        .await
        .unwrap();
    output.drain("output");
    let snapshot = source.checkpoint(Epoch::INITIAL).unwrap();
    let pool = Arc::new(GrowthPool {
        inner: GreedyMemoryPool::new(16 << 20),
        reject: AtomicBool::new(false),
        rejected: AtomicUsize::new(0),
    });
    let mut restored =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    restored
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    restored.restore(&snapshot).unwrap();
    let target_job = job().with_checkpointing(false);
    let progress = progress_context(
        &target_job,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_000_003)),
    );
    reset_join_work();
    restored
        .on_ingress_progress("right", &progress)
        .await
        .unwrap();
    assert_eq!(
        join_work().retained_visits,
        0,
        "restored maximum selects complete expiry"
    );
    assert!(restored.state.left.is_empty());
    let before = restored.status();
    let next_id = restored.state.next_left_row_id;
    let sequence = restored.state.next_output_sequence;
    pool.reject.store(true, Ordering::SeqCst);
    let error = restored
        .process_data(
            "left",
            left_batch(vec![60_000_004; 4]),
            &progress,
            &mut output,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("credit admission"));
    assert_eq!(pool.rejected.load(Ordering::SeqCst), 1);
    assert_eq!(restored.status(), before);
    assert_eq!(restored.state.next_left_row_id, next_id);
    assert_eq!(restored.state.next_output_sequence, sequence);
    assert!(output.drain("output").is_empty());
    pool.reject.store(false, Ordering::SeqCst);
    restored
        .process_data(
            "left",
            left_batch(vec![60_000_004; 4]),
            &progress,
            &mut output,
        )
        .await
        .unwrap();
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].as_data().unwrap().num_rows(), 4);
    assert_eq!(
        messages[0].as_data().unwrap().metadata().sequence(),
        sequence
    );
    assert_eq!(restored.state.left.len(), 4);
    assert_eq!(restored.state.next_left_row_id, next_id + 4);
    assert!(restored.state.deltas.pending.is_empty());
    drop((messages, restored, progress));
    drop((source, source_context, snapshot, output));
    assert!(source_job.gather_owner().close_and_drain().await.is_empty());
    assert!(target_job.gather_owner().close_and_drain().await.is_empty());
    drop((source_job, target_job));
    assert_eq!(pool.reserved(), 0);
}
