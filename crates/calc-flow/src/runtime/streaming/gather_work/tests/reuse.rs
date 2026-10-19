use std::{sync::Arc, time::Duration};

use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

use super::super::{AdmissionFailure, GatherOperatorId, GatherStop, TestService};
use super::{assert_output, credit, job, numeric_work};

#[test]
fn test_reuse_only_cold_refusal_keeps_no_persistent_credit() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(920, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:reuse".into()))
        .scope()
        .unwrap()
        .reuse_only();
    let mut observer = None;
    let result = runtime.block_on(scope.submit_observed_work(
        numeric_work(11, &pool),
        credit(&pool),
        GatherStop::from_job(&context),
        context.gather_owner().retain_retirement().unwrap(),
        &mut observer,
    ));
    assert!(matches!(result, Err(AdmissionFailure::Budget { .. })));
    drop(result);
    assert!(observer.is_none());
    assert_eq!(context.gather_owner().funding(), (0, 0, 0));
    assert_eq!(pool.reserved(), 0);
    assert_eq!(service.available_capacity(), (1, 1, 0));
    assert!(
        runtime
            .block_on(context.gather_owner().close_and_drain())
            .is_empty()
    );
    drop((scope, context, runtime));
    service.shutdown();
}

#[test]
fn test_reuse_only_warm_work_reuses_funded_generation() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(921, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let client = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:reuse".into()));
    let warm = client.scope().unwrap();
    let ticket = runtime
        .block_on(warm.submit_work(
            numeric_work(11, &pool),
            credit(&pool),
            GatherStop::from_job(&context),
        ))
        .unwrap();
    let generation = ticket.generation();
    drop(runtime.block_on(ticket.finish()).unwrap());
    let resident = pool.reserved();
    assert!(resident > 0);
    let funding = context.gather_owner().funding();
    let reuse = client.scope().unwrap().reuse_only();
    let ticket = runtime
        .block_on(reuse.submit_work(
            numeric_work(22, &pool),
            credit(&pool),
            GatherStop::from_job(&context),
        ))
        .unwrap();
    assert_eq!(ticket.generation(), generation);
    let output = runtime.block_on(ticket.finish()).unwrap();
    assert_output(&output, 22);
    drop(output);
    assert_eq!(context.gather_owner().funding(), funding);
    assert_eq!(pool.reserved(), resident);
    assert_eq!(service.joined_workers(), 0);
    assert!(
        runtime
            .block_on(context.gather_owner().close_and_drain())
            .is_empty()
    );
    drop((warm, reuse, context, runtime));
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_reuse_only_refuses_generation_retired_after_scope_creation() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(922, &service);
    let other = job(923, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let other_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let client = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:reuse".into()));
    let warm = client.scope().unwrap();
    runtime.block_on(async {
        drop(
            warm.submit_work(
                numeric_work(11, &pool),
                credit(&pool),
                GatherStop::from_job(&context),
            )
            .await
            .unwrap()
            .finish()
            .await
            .unwrap(),
        );
    });
    let reuse = client.scope().unwrap().reuse_only();
    let other_scope = other
        .gather_owner()
        .client(GatherOperatorId::new("operator:other".into()))
        .scope()
        .unwrap();
    runtime.block_on(async {
        let ticket = tokio::time::timeout(
            Duration::from_secs(2),
            other_scope.submit_work(
                numeric_work(22, &other_pool),
                credit(&other_pool),
                GatherStop::from_job(&other),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        drop(ticket.finish().await.unwrap());
    });
    assert_eq!(context.gather_owner().funding().1, 0);
    assert!(service.joined_workers() > 0);
    let resident = pool.reserved();
    let result = runtime.block_on(reuse.submit_work(
        numeric_work(33, &pool),
        credit(&pool),
        GatherStop::from_job(&context),
    ));
    assert!(matches!(result, Err(AdmissionFailure::Budget { .. })));
    drop(result);
    assert_eq!(pool.reserved(), resident);
    assert_eq!(context.gather_owner().funding().1, 0);
    runtime.block_on(async {
        assert!(context.gather_owner().close_and_drain().await.is_empty());
        assert!(other.gather_owner().close_and_drain().await.is_empty());
    });
    drop((warm, reuse, other_scope, context, other, runtime));
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
    assert_eq!(other_pool.reserved(), 0);
}
