use super::*;
use datafusion::arrow::array::Array;
use std::sync::{Weak, atomic::AtomicBool};

#[derive(Debug)]
struct LeasePool {
    inner: GreedyMemoryPool,
    live: AtomicUsize,
    source: Mutex<Option<Weak<dyn Array>>>,
    refunded_while_owned: AtomicBool,
}

impl LeasePool {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: GreedyMemoryPool::new(1 << 30),
            live: AtomicUsize::new(0),
            source: Mutex::new(None),
            refunded_while_owned: AtomicBool::new(false),
        })
    }

    fn morsel(credit: &MemoryReservation) -> bool {
        credit.consumer().name() == "sql-incremental:stream-join-morsel"
    }
}

impl std::fmt::Display for LeasePool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, formatter)
    }
}

impl MemoryPool for LeasePool {
    fn name(&self) -> &'static str {
        "rowed-lifecycle"
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn grow(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.grow(credit, bytes);
        if Self::morsel(credit) {
            self.live.fetch_add(bytes, Ordering::SeqCst);
        }
    }
    fn shrink(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.shrink(credit, bytes);
        if Self::morsel(credit)
            && bytes > 0
            && self.live.fetch_sub(bytes, Ordering::SeqCst) == bytes
            && self
                .source
                .lock()
                .as_ref()
                .and_then(Weak::upgrade)
                .is_some()
        {
            self.refunded_while_owned.store(true, Ordering::SeqCst);
        }
    }
    fn try_grow(&self, credit: &MemoryReservation, bytes: usize) -> datafusion::common::Result<()> {
        self.inner.try_grow(credit, bytes)?;
        if Self::morsel(credit) {
            self.live.fetch_add(bytes, Ordering::SeqCst);
        }
        Ok(())
    }
}

struct Gate {
    entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    released: Mutex<bool>,
    changed: parking_lot::Condvar,
}

impl Gate {
    fn wait(&self) {
        if let Some(entered) = self.entered.lock().take() {
            entered.send(()).unwrap();
        }
        let mut released = self.released.lock();
        while !*released {
            assert!(
                !self
                    .changed
                    .wait_for(&mut released, Duration::from_secs(10))
                    .timed_out()
            );
        }
    }
    fn release(&self) {
        *self.released.lock() = true;
        self.changed.notify_all();
    }
}

struct Release(Arc<Gate>);
impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}

#[test]
fn batch_parallel_output_sealed_views_keep_leases_until_cancelled_or_abandoned_workers_settle() {
    let _production = ProductionGate::new();
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    for cancel in [true, false] {
        runtime.block_on(stopped_case(&service, cancel));
    }
    drop(runtime);
    service.shutdown();
}

async fn stopped_case(service: &TestService, cancel: bool) {
    let pool = LeasePool::new();
    let mut operator = projected_probe_operator();
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    let job = job(service);
    preload_projected_probe(&mut operator, &job).await;
    let context = probe_only_context(&job);
    let source = input(false);
    let weak = Arc::downgrade(source.table_payload().unwrap().batches()[0].column(2));
    *pool.source.lock() = Some(weak.clone());
    let (entered, entering) = tokio::sync::oneshot::channel();
    let gate = Arc::new(Gate {
        entered: Mutex::new(Some(entered)),
        released: Mutex::new(false),
        changed: parking_lot::Condvar::new(),
    });
    let _release = Release(gate.clone());
    let worker_gate = gate.clone();
    operator.probe_unit_test_hook = Some(Arc::new(move |phase, ordinal, complete| {
        if phase == super::super::super::ProbePhase::Fill && ordinal == 0 && !complete {
            worker_gate.wait();
        }
    }));
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let mut process = Box::pin(operator.process_data("left", source, &context, &mut output));
    tokio::select! {
        entered = entering => entered.unwrap(),
        result = &mut process => panic!("stopped before funded gather: {result:?}"),
    }
    assert!(pool.live.load(Ordering::SeqCst) > 0);
    if cancel {
        job.cancellation().cancel();
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(5), &mut process)
                .await
                .unwrap(),
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
    }
    drop(process);
    assert!(output.drain("output").is_empty());
    assert_eq!(operator.status().left.retained_rows, 0);
    assert_eq!(operator.state.next_output_sequence, 0);
    operator.reset().unwrap();
    drop(operator);
    assert!(pool.live.load(Ordering::SeqCst) > 0);
    assert!(
        weak.upgrade().is_some(),
        "paused worker still owns its source backing"
    );
    gate.release();
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.live.load(Ordering::SeqCst), 0);
    assert_eq!(pool.reserved(), 0);
    assert!(!pool.refunded_while_owned.load(Ordering::SeqCst));
}

#[test]
fn batch_parallel_output_sealed_view_reservation_refusal_uses_serial_fallback() {
    let _production = ProductionGate::new();
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let pool = Arc::new(RefusingPool {
            inner: GreedyMemoryPool::new(1 << 30),
            attempts: AtomicUsize::new(0),
        });
        let mut operator = projected_probe_operator();
        operator
            .runtime
            .initialize_benchmark_pool(pool.clone())
            .unwrap();
        let job = job(&service);
        preload_projected_probe(&mut operator, &job).await;
        let context = probe_only_context(&job);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", input(false), &context, &mut output)
            .await
            .unwrap();
        assert!(pool.attempts.load(Ordering::SeqCst) > 0);
        verify(&mut output, usize::MAX);
        assert_eq!(operator.status().left.retained_rows, 0);
        drop(operator);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}
