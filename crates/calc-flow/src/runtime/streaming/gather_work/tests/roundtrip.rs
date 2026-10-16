//! Explicit diagnostic; never a timing assertion in the normal test suite.

use super::*;
use crate::runtime::streaming::gather_work::ParallelCpuWork;

struct Empty(usize);

impl ParallelCpuWork for Empty {
    type Output = usize;

    fn unit_count(&self) -> usize {
        self.0
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> crate::Result<usize> {
        stop.check()?;
        Ok(ordinal)
    }
}

#[test]
#[ignore = "explicit D7 empty-task roundtrip diagnostic"]
fn test_empty_parallel_roundtrip() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let job = job(438, &service);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
        let scope = job
            .gather_owner()
            .client(GatherOperatorId::new("empty-roundtrip".into()))
            .scope()
            .unwrap();
        for units in [1, 4] {
            // Warm the persistent generation before starting clocks.
            empty(&scope, &job, &pool, units).await;
            let clock = crate::benchmark_internals::PhaseClock::start("empty-parallel-roundtrip");
            let mut samples = Vec::new();
            for _ in 0..100 {
                let start = std::time::Instant::now();
                empty(&scope, &job, &pool, units).await;
                samples.push(start.elapsed().as_nanos());
            }
            println!(
                "{}",
                serde_json::json!({"units":units,"samples_ns":samples,"resources":clock.finish()})
            );
        }
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop((scope, job));
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

async fn empty(
    scope: &super::super::GatherScope,
    job: &StreamJobContext,
    pool: &Arc<dyn MemoryPool>,
    units: usize,
) {
    let stop = GatherStop::from_job(job);
    let mut cleanup = None;
    let ticket = scope
        .submit_observed_parallel_work(
            Arc::new(Empty(units)),
            credit(pool),
            stop.clone(),
            job.gather_owner().retain_retirement().unwrap(),
            &mut cleanup,
        )
        .await
        .unwrap();
    ticket
        .finish()
        .await
        .unwrap()
        .install(|values| {
            assert_eq!(values, (0..units).collect::<Vec<_>>());
            Ok(())
        })
        .unwrap();
    cleanup.unwrap().wait(&stop).await.unwrap();
}
