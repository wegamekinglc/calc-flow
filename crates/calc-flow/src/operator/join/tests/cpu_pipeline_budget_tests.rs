use super::*;
use crate::runtime::streaming::gather_work::{
    GatherOperatorId, GatherStop, OwnedCpuWork, TestService,
};
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
    UnboundedMemoryPool,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug)]
enum Refusal {
    Never,
    Optional,
    ChunkOnce,
    ChunkAlways,
}

#[derive(Debug)]
struct StatePool {
    inner: GreedyMemoryPool,
    refusal: Refusal,
    rejected: AtomicUsize,
    reports_infinite: bool,
}

impl fmt::Display for StatePool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)
    }
}

impl MemoryPool for StatePool {
    fn name(&self) -> &'static str {
        "retained-cpu-fault"
    }
    fn memory_limit(&self) -> MemoryLimit {
        if self.reports_infinite {
            MemoryLimit::Infinite
        } else {
            self.inner.memory_limit()
        }
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn grow(&self, reservation: &MemoryReservation, bytes: usize) {
        self.inner.grow(reservation, bytes);
    }
    fn shrink(&self, reservation: &MemoryReservation, bytes: usize) {
        self.inner.shrink(reservation, bytes);
    }
    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        bytes: usize,
    ) -> datafusion::common::Result<()> {
        if reservation.consumer().name() == "sql-incremental:stream-join-state-chunks" {
            let matches = match self.refusal {
                Refusal::Never => false,
                Refusal::Optional => bytes > 0 && bytes < 32_768,
                Refusal::ChunkOnce | Refusal::ChunkAlways => (32_768..131_072).contains(&bytes),
            };
            if matches {
                let prior = self.rejected.fetch_add(1, Ordering::SeqCst);
                if !matches!(self.refusal, Refusal::ChunkOnce) || prior == 0 {
                    return Err(datafusion::common::DataFusionError::ResourcesExhausted(
                        "retained test refusal".into(),
                    ));
                }
            }
        }
        self.inner.try_grow(reservation, bytes)
    }
}

fn check_budget(refusal: Refusal) {
    let service = TestService::new(4, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let pool = Arc::new(StatePool {
            inner: GreedyMemoryPool::new(1 << 30),
            refusal,
            rejected: AtomicUsize::new(0),
            reports_infinite: false,
        });
        let mut declaration = spec();
        declaration.limits = JoinStateLimits::new(100_000, 100_000_000, 100_000).unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
        operator
            .runtime
            .initialize_benchmark_pool(pool.clone())
            .unwrap();
        let job = job().with_gather_owner(service.owner("retained-budget".into()));
        let context = StreamOperatorContext::new(&job, "match", None);
        let source = left_batch((0..65_536).collect());
        let warm_pool = warm_service(&context).await;
        pool.rejected.store(0, Ordering::SeqCst);
        let selection_units = Arc::new(AtomicUsize::new(0));
        let selected = Arc::clone(&selection_units);
        let chunk_units = Arc::new(AtomicUsize::new(0));
        let captured = Arc::clone(&chunk_units);
        let _observe = retained::observe_preparation_for_test(Arc::new(move |phase, _, _| {
            if phase == "selection" {
                selected.fetch_add(1, Ordering::SeqCst);
            } else if phase == "chunks" {
                captured.fetch_add(1, Ordering::SeqCst);
            }
        }));
        let result = operator.prepare_batch("left", &source, &context).await;
        assert!(pool.rejected.load(Ordering::SeqCst) > 0);
        assert_eq!(
            selection_units.load(Ordering::SeqCst),
            if matches!(refusal, Refusal::Optional) {
                0
            } else {
                4
            },
            "chunk refusal must follow actual parallel selection",
        );
        assert_eq!(
            chunk_units.load(Ordering::SeqCst),
            0,
            "prepay before dispatch; refusal uses serial fallback"
        );
        if matches!(refusal, Refusal::ChunkAlways) {
            let error = result.err().expect("mandatory refusal");
            assert!(error.to_string().contains("retained test refusal"));
        } else {
            let prepared = result.unwrap();
            assert_eq!(prepared.chunks.len(), 65_536);
            for index in 0..65_536 {
                assert_eq!(
                    prepared.chunks.identity(index),
                    (
                        EventTime::from_micros(i64::try_from(index).unwrap()),
                        u64::try_from(index).unwrap()
                    )
                );
            }
        }
        assert_eq!(operator.state.next_left_row_id, 0);
        assert_eq!(operator.state.left.len(), 0);
        drop(operator);
        drop(context);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
        assert_eq!(warm_pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_cpu_pipeline_optional_state_controls_preserve_serial_admission() {
    check_budget(Refusal::Optional);
}

#[test]
fn test_cpu_pipeline_chunk_refusal_releases_prepaid_units_before_retry() {
    check_budget(Refusal::ChunkOnce);
}

#[test]
fn test_cpu_pipeline_mandatory_chunk_refusal_preserves_uncommitted_state() {
    check_budget(Refusal::ChunkAlways);
}

#[derive(Clone, Copy)]
enum PoolBoundary {
    Warm,
    Cold,
    CustomInfinite,
}

impl PoolBoundary {
    fn pool(self, capacity: usize) -> Arc<dyn MemoryPool> {
        if matches!(self, Self::CustomInfinite) {
            Arc::new(StatePool {
                inner: GreedyMemoryPool::new(capacity),
                refusal: Refusal::Never,
                rejected: AtomicUsize::new(0),
                reports_infinite: true,
            })
        } else {
            Arc::new(GreedyMemoryPool::new(capacity))
        }
    }
}

fn check_real_pool_boundary(boundary: PoolBoundary) {
    let prewarmed = matches!(boundary, PoolBoundary::Warm);
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut declaration = spec();
        declaration.limits = JoinStateLimits::new(100_000, 100_000_000, 100_000).unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
        let job = job().with_gather_owner(service.owner("retained-boundary".into()));
        let context = StreamOperatorContext::new(&job, "match", None);
        let source = left_batch((0..32_768).collect());
        let side = operator.begin_batch("left", &source).unwrap();
        let mut bundle = operator.admission_bundle(&side, &source).unwrap();
        operator
            .admit_batch_records("left", &source, &side, &context, &mut bundle)
            .await
            .unwrap();
        bundle.finish("match").unwrap();
        bundle.admitted.prepare_retained_parents();
        let mut keys = operator.native_probe_keys(&side, &bundle.admitted).unwrap();
        assert!(keys.is_some());
        let state = StateRows::default();

        let warm_pool = if prewarmed {
            Some(warm_service(&context).await)
        } else {
            None
        };
        let (exact_capacity, expected) =
            serial_reference(&state, &bundle.admitted, keys.as_ref().unwrap(), &context).await;

        let pool = boundary.pool(exact_capacity);
        let reuse_only = !pool.is::<UnboundedMemoryPool>();
        let master = MemoryConsumer::new("retained-boundary-candidate").register(&pool);
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&events);
        let _observe = retained::observe_preparation_for_test(Arc::new(move |phase, unit, _| {
            observed.lock().unwrap().push((phase, unit));
        }));
        let plan = retained::ColumnarPlan::prepare_owned(
            &mut bundle.admitted,
            &mut keys,
            &master,
            reuse_only,
            &context,
        )
        .await
        .unwrap();
        if prewarmed {
            assert_eq!(observed_units(&events, "selection"), 2);
        }
        let record = &source.table_payload().unwrap().batches()[0];
        bundle
            .admitted
            .replace_view(0, Arc::new(record.project(&[0]).unwrap()));
        let actual = state
            .prepare_columns_owned(
                &mut bundle.admitted,
                &mut keys,
                plan,
                &master,
                reuse_only,
                &context,
            )
            .await
            .expect("mandatory state must fit wherever the serial plan fits");
        if prewarmed {
            assert_eq!(
                observed_units(&events, "chunks"),
                0,
                "exact serial capacity must refuse optional chunk preparation"
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, expected) in expected.into_iter().enumerate() {
            assert_eq!((actual.identity(index), actual.charge(index)), expected);
        }
        drop(actual);
        if prewarmed {
            assert_eq!(pool.reserved(), 0);
        }
        assert_eq!(operator.state.next_left_row_id, 0);
        assert_eq!(operator.state.left.len(), 0);
        drop(keys);
        drop(bundle);
        drop(operator);
        drop(context);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
        if let Some(warm_pool) = warm_pool {
            assert_eq!(warm_pool.reserved(), 0);
        }
    });
    drop(runtime);
    service.shutdown();
}

struct ServiceWarmup;

impl OwnedCpuWork for ServiceWarmup {
    type Output = ();

    fn run(self, stop: &GatherStop) -> Result<()> {
        stop.check()
    }
}

async fn warm_service(context: &StreamOperatorContext<'_>) -> Arc<dyn MemoryPool> {
    let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
    let master = MemoryConsumer::new("retained-boundary-service").register(&pool);
    let scope = context
        .gather_client(GatherOperatorId::new("retained-budget-warm".into()))
        .scope()
        .unwrap();
    let ticket = scope
        .submit_work(ServiceWarmup, master, GatherStop::from_job(context.job()))
        .await
        .unwrap();
    drop(ticket.finish().await.unwrap());
    // The service keeps its home and worker-generation credit until drain.
    assert!(pool.reserved() > 0);
    pool
}

async fn serial_reference(
    state: &StateRows,
    rows: &AdmittedBatch,
    keys: &native_lookup::NativeKeys,
    context: &StreamOperatorContext<'_>,
) -> (usize, Vec<((EventTime, u64), u64)>) {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
    let master = MemoryConsumer::new("retained-boundary-baseline").register(&pool);
    let plan = retained::ColumnarPlan::prepare(rows, keys, &master, context)
        .await
        .unwrap();
    let baseline = state
        .prepare_columns(rows, keys, &plan, &master, context)
        .await
        .unwrap();
    // Both serial operations retain every reservation until this point.
    let capacity = pool.reserved();
    let expected = (0..baseline.len())
        .map(|index| (baseline.identity(index), baseline.charge(index)))
        .collect();
    drop(baseline);
    drop(plan);
    assert_eq!(pool.reserved(), 0);
    (capacity, expected)
}

fn observed_units(events: &std::sync::Mutex<Vec<(&str, usize)>>, phase: &str) -> usize {
    events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.0 == phase)
        .count()
}

#[test]
fn test_cpu_pipeline_real_pool_boundary_reclaims_optional_selection() {
    check_real_pool_boundary(PoolBoundary::Warm);
}

#[test]
fn test_cpu_pipeline_cold_pool_boundary_preserves_serial_admission() {
    check_real_pool_boundary(PoolBoundary::Cold);
}

#[test]
fn test_cpu_pipeline_custom_infinite_pool_preserves_serial_admission() {
    check_real_pool_boundary(PoolBoundary::CustomInfinite);
}
