use super::*;
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryLimit, MemoryPool, MemoryReservation,
};
use std::{
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
};

#[path = "column_gather_emit_tests.rs"]
mod emit;

#[derive(Debug)]
struct GatherPool {
    inner: GreedyMemoryPool,
    current: AtomicUsize,
    attempts: AtomicUsize,
    reject_at: usize,
    growth: parking_lot::Mutex<Vec<usize>>,
    refunds: parking_lot::Mutex<Vec<usize>>,
}

impl GatherPool {
    fn new(reject_at: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: GreedyMemoryPool::new(1 << 30),
            current: AtomicUsize::new(0),
            attempts: AtomicUsize::new(0),
            reject_at,
            growth: parking_lot::Mutex::new(Vec::new()),
            refunds: parking_lot::Mutex::new(Vec::new()),
        })
    }

    fn is_gather(credit: &MemoryReservation) -> bool {
        credit.consumer().name() == "sql-incremental:stream-join-gather"
    }

    fn record_growth(&self, credit: &MemoryReservation, bytes: usize) {
        if Self::is_gather(credit) && bytes != 0 {
            self.current.fetch_add(bytes, Ordering::SeqCst);
            self.growth.lock().push(bytes);
        }
    }
}

impl fmt::Display for GatherPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)
    }
}

impl MemoryPool for GatherPool {
    fn name(&self) -> &'static str {
        "join-gather-observation"
    }

    fn grow(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.grow(credit, bytes);
        self.record_growth(credit, bytes);
    }

    fn shrink(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.shrink(credit, bytes);
        if Self::is_gather(credit) && bytes != 0 {
            self.current.fetch_sub(bytes, Ordering::SeqCst);
            self.refunds.lock().push(bytes);
        }
    }

    fn try_grow(&self, credit: &MemoryReservation, bytes: usize) -> datafusion::common::Result<()> {
        if Self::is_gather(credit)
            && bytes != 0
            && self.attempts.fetch_add(1, Ordering::SeqCst) + 1 == self.reject_at
        {
            return Err(datafusion::common::DataFusionError::ResourcesExhausted(
                "Join capture refused".into(),
            ));
        }
        self.inner.try_grow(credit, bytes)?;
        self.record_growth(credit, bytes);
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
}

async fn preloaded(pool: &Arc<GatherPool>) -> (StreamJoinOperator, StreamJobContext) {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    let source_job = job();
    let context = StreamOperatorContext::new(&source_job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut collector)
        .await
        .unwrap();
    assert!(collector.drain("output").is_empty());
    assert_eq!(pool.current.load(Ordering::SeqCst), 0);
    (operator, source_job)
}

#[test]
fn test_join_column_gather_capture_refusal_refunds_before_serial_fallback() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for reject_at in [1, 2] {
        runtime.block_on(assert_capture_refusal(&service, reject_at));
    }
    drop(runtime);
    service.shutdown();
}

async fn assert_capture_refusal(service: &TestService, reject_at: usize) {
    let pool = GatherPool::new(reject_at);
    let (mut operator, source_job) = preloaded(&pool).await;
    let job = job().with_gather_owner(service.owner("join-capture-refusal".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let observed = Arc::clone(&pool);
    let calls = Arc::new(AtomicUsize::new(0));
    let column_calls = Arc::clone(&calls);
    let actor = std::thread::current().id();
    let _configuration = output_gather::test_support::configure(
        true,
        Some(Arc::new(move |_, completed| {
            if !completed {
                assert_eq!(std::thread::current().id(), actor);
                assert_eq!(observed.current.load(Ordering::SeqCst), 0);
                column_calls.fetch_add(1, Ordering::SeqCst);
            }
        })),
    );
    operator
        .process_data("left", left_batch(vec![1, 2]), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 6);
    assert_eq!(pool.attempts.load(Ordering::SeqCst), reject_at);
    assert_eq!(pool.growth.lock().len(), reject_at - 1);
    assert_eq!(*pool.growth.lock(), *pool.refunds.lock());
    assert_eq!(job.gather_owner().funding(), (0, 0, 0));
    assert_eq!(operator.state.next_left_row_id, 2);
    assert_eq!(operator.state.next_output_sequence, 1);
    assert_literal_output(&mut collector, &operator);
    drop((operator, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert!(source_job.gather_owner().close_and_drain().await.is_empty());
    drop((job, source_job));
    assert_eq!(pool.reserved(), 0);
}
