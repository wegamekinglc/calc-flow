use super::super::{ArrayRef, GatherPlan, WorkAdapter, launch_registration};
use super::*;
use crate::runtime::streaming::gather_work::{ParallelCpuWork, RowGather};
use datafusion::arrow::array::Int64Array;
use std::{
    ops::Range,
    sync::atomic::{AtomicUsize, Ordering},
};

struct OverlapWork {
    units: usize,
    entered: std::sync::mpsc::Sender<usize>,
    release: Arc<TypedGate>,
}

impl ParallelCpuWork for OverlapWork {
    type Output = usize;

    fn unit_count(&self) -> usize {
        self.units
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> crate::Result<usize> {
        self.entered.send(ordinal).unwrap();
        self.release.wait();
        stop.check()?;
        Ok(ordinal)
    }
}

#[test]
fn cpu_aware_parallel_work_uses_sixteen_workers_and_preserves_order() {
    overlap_case(16, 24);
}

#[test]
fn cpu_aware_parallel_work_respects_smaller_service_capacity() {
    overlap_case(3, 24);
}

fn overlap_case(workers: usize, units: usize) {
    let service = TestService::new(workers, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let job = job(911, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let release = TypedGateRelease(Arc::new(TypedGate::default()));
    let (entered, receiver) = std::sync::mpsc::channel();
    let scope = job
        .gather_owner()
        .client(GatherOperatorId::new("operator:cpu-aware".into()))
        .scope()
        .unwrap();
    let ticket = runtime
        .block_on(scope.submit_parallel_work(
            Arc::new(OverlapWork {
                units,
                entered,
                release: Arc::clone(&release.0),
            }),
            credit(&pool),
            GatherStop::from_job(&job),
        ))
        .unwrap();
    let mut started = Vec::new();
    for _ in 0..workers {
        let Ok(ordinal) = receiver.recv_timeout(Duration::from_secs(2)) else {
            break;
        };
        started.push(ordinal);
    }
    let exceeded = receiver.recv_timeout(Duration::from_millis(20)).is_ok();
    drop(release);
    let output = runtime.block_on(ticket.finish()).unwrap();
    assert_eq!(output.value, (0..units).collect::<Vec<_>>());
    drop(output);
    assert!(
        runtime
            .block_on(job.gather_owner().close_and_drain())
            .is_empty()
    );
    drop((scope, job, runtime));
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
    assert_eq!(
        started.len(),
        workers,
        "CPU capacity was hidden by a fixed worker cap"
    );
    assert!(!exceeded, "work exceeded the service CPU capacity");
}

#[test]
fn cpu_aware_row_gather_covers_rows_above_eight_parts() {
    let shape = RowGather {
        ordinal: 0,
        rows: 129,
        width: 8,
        sources: 1,
    };
    for workers in [1, 3, 16, 64, 256] {
        let parts = shape.parts(workers);
        assert_eq!(parts, workers.min(shape.rows));
        let ranges = (0..parts)
            .map(|ordinal| shape.range(ordinal, parts))
            .collect::<Vec<_>>();
        assert!(ranges.iter().all(|range| !range.is_empty()));
        assert_eq!(
            ranges.iter().cloned().flatten().collect::<Vec<_>>(),
            (0..shape.rows).collect::<Vec<_>>()
        );
    }
}

#[test]
fn cpu_aware_service_capacity_uses_available_cpu_without_a_memory_ceiling() {
    use super::super::process::worker_capacity;

    assert_eq!(worker_capacity(1), 1);
    assert_eq!(worker_capacity(16), 16);
    assert_eq!(worker_capacity(64), 64);
    assert_eq!(worker_capacity(128), 128);
}

#[derive(Default)]
struct ChangingPlan {
    parallelism_calls: AtomicUsize,
    shape_calls: AtomicUsize,
    ranges: parking_lot::Mutex<Vec<Range<usize>>>,
}

impl GatherPlan for ChangingPlan {
    fn column_count(&self) -> usize {
        2
    }

    fn parallelism(&self) -> usize {
        if self.parallelism_calls.fetch_add(1, Ordering::SeqCst) == 0 {
            16
        } else {
            1
        }
    }

    fn row_gather(&self) -> crate::Result<Option<RowGather>> {
        let first = self.shape_calls.fetch_add(1, Ordering::SeqCst) == 0;
        Ok(Some(RowGather {
            ordinal: 1,
            rows: if first { 129 } else { 1 },
            width: 8,
            sources: 1,
        }))
    }

    fn shared_column(&self, ordinal: usize) -> crate::Result<Option<ArrayRef>> {
        assert_eq!(ordinal, 0);
        Ok(Some(Arc::new(Int64Array::from_iter_values(0..129))))
    }

    fn gather(&self, _ordinal: usize, _stop: &GatherStop) -> crate::Result<ArrayRef> {
        Err(super::super::internal("captured row slicing must be used"))
    }

    fn gather_range(
        &self,
        ordinal: usize,
        range: Range<usize>,
        stop: &GatherStop,
    ) -> crate::Result<ArrayRef> {
        stop.check()?;
        assert_eq!(ordinal, 1);
        self.ranges.lock().push(range.clone());
        Ok(Arc::new(Int64Array::from_iter_values(
            range.map(|row| i64::try_from(row).unwrap() + 1_000),
        )))
    }
}

#[test]
fn cpu_aware_column_dispatch_captures_dynamic_parallelism_and_shape_once() {
    let service = TestService::new(16, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(912, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let plan = Arc::new(ChangingPlan::default());
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:captured-columns".into()))
        .scope()
        .unwrap();
    runtime.block_on(async {
        let output = scope
            .submit(plan.clone(), credit(&pool), GatherStop::from_job(&context))
            .await
            .unwrap()
            .finish()
            .await
            .unwrap();
        assert_eq!(output.value.len(), 2);
        for (ordinal, column) in output.value.iter().enumerate() {
            let expected = (0..129)
                .map(|row| row + if ordinal == 0 { 0 } else { 1_000 })
                .collect::<Vec<i64>>();
            assert_eq!(
                column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .as_ref(),
                expected
            );
        }
        drop(output);
        assert_eq!(plan.parallelism_calls.load(Ordering::SeqCst), 1);
        assert_eq!(plan.shape_calls.load(Ordering::SeqCst), 1);
        let mut ranges = plan.ranges.lock().clone();
        ranges.sort_unstable_by_key(|range| range.start);
        assert_eq!(ranges.len(), 16);
        assert_eq!(
            ranges.into_iter().flatten().collect::<Vec<_>>(),
            (0..129).collect::<Vec<_>>()
        );
        assert!(context.gather_owner().close_and_drain().await.is_empty());
    });
    drop((scope, context, runtime));
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
}

struct GrowingWork(Arc<AtomicUsize>);

impl ParallelCpuWork for GrowingWork {
    type Output = usize;

    fn unit_count(&self) -> usize {
        16
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> crate::Result<usize> {
        stop.check()?;
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ordinal)
    }
}

#[test]
fn cpu_aware_partial_pool_growth_failure_refunds_and_allows_same_home_retry() {
    let service = TestService::new(16, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(913, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    runtime.block_on(growth_failure_case(&service, &context, &pool));
    drop((context, runtime));
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
}

async fn growth_failure_case(
    service: &TestService,
    context: &StreamJobContext,
    pool: &Arc<dyn MemoryPool>,
) {
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:growing".into()))
        .scope()
        .unwrap();
    let first = scope
        .submit_work(
            numeric_work(17, pool),
            credit(pool),
            GatherStop::from_job(context),
        )
        .await
        .unwrap();
    let generation = first.generation();
    let output = first.finish().await.unwrap();
    assert_output(&output, 17);
    drop(output);
    assert_eq!(service.joined_workers(), 0);
    assert_eq!(service.available_capacity(), (15, 0, 1));
    service.invalidate_next_worker_name();
    let entered = Arc::new(AtomicUsize::new(0));
    let failed = tokio::time::timeout(
        Duration::from_secs(3),
        scope.submit_parallel_work(
            Arc::new(GrowingWork(entered.clone())),
            credit(pool),
            GatherStop::from_job(context),
        ),
    )
    .await;
    let reported = matches!(
        &failed,
        Ok(Err(super::super::AdmissionFailure::Runtime(
            crate::CalcFlowError::Internal { .. }
        )))
    );
    drop(failed);
    let cleaned = tokio::time::timeout(Duration::from_secs(3), scope.home.wait_closed())
        .await
        .is_ok();
    let funding = context.gather_owner().funding();
    assert!(cleaned, "failed growth must settle all existing workers");
    assert!(reported, "failed expansion must return a domain error");
    assert_eq!(entered.load(Ordering::SeqCst), 0);
    assert_eq!(service.joined_workers(), 1);
    assert_eq!(service.available_capacity(), (16, 1, 0));
    assert_eq!((funding.1, funding.2), (0, 0));
    assert_eq!(
        pool.reserved(),
        funding.0,
        "only the reusable home control remains charged"
    );
    tokio::time::timeout(Duration::from_secs(3), async {
        let next = scope
            .submit_work(
                numeric_work(29, pool),
                credit(pool),
                GatherStop::from_job(context),
            )
            .await
            .unwrap();
        assert!(next.generation() > generation);
        let output = next.finish().await.unwrap();
        assert_output(&output, 29);
        drop(output);
    })
    .await
    .expect("same home must retry after partial-growth cleanup");
    assert!(
        tokio::time::timeout(
            Duration::from_secs(3),
            context.gather_owner().close_and_drain()
        )
        .await
        .unwrap()
        .is_empty()
    );
}

#[test]
fn cpu_aware_closed_home_rejects_registered_launch_and_drains_credit() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(914, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:closed-launch".into()))
        .scope()
        .unwrap();
    runtime.block_on(async {
        let home = scope.home.clone();
        let stop = GatherStop::from_job(&context);
        home.install_attempt(
            &scope,
            WorkAdapter(numeric_work(41, &pool)),
            credit(&pool),
            stop.clone(),
        )
        .await
        .unwrap();
        let execution = home.execution_service().unwrap();
        let registered = execution.register(home.clone(), &stop).await.unwrap();
        home.close();
        let result = launch_registration(registered, 2);
        assert!(matches!(
            result,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        let failures = tokio::time::timeout(
            Duration::from_secs(3),
            context.gather_owner().close_and_drain(),
        )
        .await
        .expect("closed launch must not strand generation retirement");
        assert!(failures.is_empty());
        let funding = context.gather_owner().funding();
        assert!(
            funding.0 > 0,
            "external owners retain the Home control lease"
        );
        assert_eq!((funding.1, funding.2), (0, 0));
        assert_eq!(service.available_capacity(), (2, 1, 0));
        assert_eq!(pool.reserved(), funding.0);
    });
    drop((scope, context, runtime));
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
}
