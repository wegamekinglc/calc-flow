use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryConsumer, MemoryPool, MemoryReservation,
};

use super::{GatherOperatorId, GatherStop, TestService, WorkOutput};
use crate::{CancellationToken, JsonMap, StreamJobContext};

mod cleanup;
mod parallel;

#[test]
fn typed_idle_pool_yields_capacity_with_live_context_and_unobserved_ticket() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let progressed = runtime.block_on(pressure_case(&service));
    drop(runtime);
    service.shutdown();
    assert!(
        progressed,
        "one-worker/one-registry idle typed CPU pool starved another job while the first context and completed ticket stayed alive"
    );
}

async fn pressure_case(service: &TestService) -> bool {
    let job_a = job(1, service);
    let job_b = job(2, service);
    let pool_a: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let pool_b: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let client_a = job_a
        .gather_owner()
        .client(GatherOperatorId::new("operator:asof".into()));
    let client_b = job_b
        .gather_owner()
        .client(GatherOperatorId::new("operator:asof".into()));
    let scope_a = client_a.scope().unwrap();
    let scope_b = client_b.scope().unwrap();
    let mut ticket_a = Some(
        scope_a
            .submit_work(
                numeric_work(11, &pool_a),
                credit(&pool_a),
                GatherStop::from_job(&job_a),
            )
            .await
            .unwrap(),
    );
    let old_generation = ticket_a.as_ref().unwrap().generation();
    ticket_a.as_ref().unwrap().wait_settled().await.unwrap();
    assert_eq!(service.joined_workers(), 0);
    let mut pending_b = Box::pin(scope_b.submit_work(
        numeric_work(22, &pool_b),
        credit(&pool_b),
        GatherStop::from_job(&job_b),
    ));
    let observed_b = tokio::time::timeout(Duration::from_secs(2), async {
        let ticket = pending_b.as_mut().await?;
        ticket.wait_settled().await?;
        Ok::<_, crate::CalcFlowError>(ticket)
    })
    .await;
    let progressed = observed_b.is_ok();
    if let Ok(ticket_b) = observed_b {
        let output_b = ticket_b.unwrap().finish().await.unwrap();
        assert_output(&output_b, 22);
        assert!(service.joined_workers() >= 1);
        assert_eq!(job_a.gather_owner().funding().1, 0);
        drop(output_b);
        let output_a = ticket_a.take().unwrap().finish().await.unwrap();
        assert_output(&output_a, 11);
        drop(output_a);
        let next_a = scope_a
            .submit_work(
                numeric_work(33, &pool_a),
                credit(&pool_a),
                GatherStop::from_job(&job_a),
            )
            .await
            .unwrap();
        assert!(next_a.generation() > old_generation);
        let output_a = next_a.finish().await.unwrap();
        assert_output(&output_a, 33);
        drop(output_a);
    }
    drop(pending_b);
    drop(ticket_a);
    job_a.gather_owner().close_and_drain().await;
    job_b.gather_owner().close_and_drain().await;
    drop(scope_a);
    drop(scope_b);
    drop(job_a);
    drop(job_b);
    assert_eq!(pool_a.reserved(), 0);
    assert_eq!(pool_b.reserved(), 0);
    progressed
}

fn job(id: u64, service: &TestService) -> StreamJobContext {
    StreamJobContext::new(id, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner(id.to_string().into()))
}

fn credit(pool: &Arc<dyn MemoryPool>) -> MemoryReservation {
    let credit = MemoryConsumer::new("asof-pressure").register(pool);
    credit.try_grow(32_768).unwrap();
    credit
}

fn assert_output(output: &WorkOutput<PreparedNumeric>, expected: u64) {
    assert!(output.credit.size() >= 32_768);
    assert_eq!(output.value.sum, expected);
    assert_eq!(output.value.input.values, [expected]);
}

fn numeric_work(value: u64, pool: &Arc<dyn MemoryPool>) -> NumericWork {
    NumericWork {
        input: Arc::new(FundedInput {
            values: vec![value],
            pool: pool.clone(),
            funded_drop: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }),
        gate: None,
        started: None,
        panics: false,
    }
}

#[test]
fn typed_owned_work_keeps_input_funded_after_observer_drop() {
    typed_drop_case(false);
}

#[test]
fn typed_owned_work_panic_keeps_input_funded_until_native_drain() {
    typed_drop_case(true);
}

fn typed_drop_case(panics: bool) {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(3, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let funded_drop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let input = Arc::new(FundedInput {
        values: vec![3, 7, 11],
        pool: pool.clone(),
        funded_drop: funded_drop.clone(),
    });
    let weak = Arc::downgrade(&input);
    let gate = Arc::new(TypedGate::default());
    let (started_tx, started) = std::sync::mpsc::channel();
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:sql".into()))
        .scope()
        .unwrap();
    let ticket = runtime.block_on(scope.submit_work(
        NumericWork {
            input,
            gate: Some(gate.clone()),
            started: Some(started_tx),
            panics,
        },
        credit(&pool),
        GatherStop::from_job(&context),
    ));
    let started = started.recv_timeout(Duration::from_secs(1)).is_ok();
    let alive = weak.upgrade().is_some();
    let original_credit = context.gather_owner().funding().2 >= 32_768;
    drop(ticket);
    let pending = runtime.block_on(async {
        tokio::time::timeout(
            Duration::from_millis(25),
            context.gather_owner().close_and_drain(),
        )
        .await
        .is_err()
    });
    gate.release();
    let failures = runtime.block_on(context.gather_owner().close_and_drain());
    let destroyed = weak.upgrade().is_none();
    drop(scope);
    drop(context);
    drop(runtime);
    service.shutdown();
    assert!(started && alive && original_credit && pending && destroyed);
    assert!(funded_drop.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(failures.len(), usize::from(panics));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn typed_result_retains_transaction_before_output_credit_drop() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(4, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let funded_drop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let input = Arc::new(FundedInput {
        values: vec![3, 7, 11],
        pool: pool.clone(),
        funded_drop: funded_drop.clone(),
    });
    let weak = Arc::downgrade(&input);
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:sql".into()))
        .scope()
        .unwrap();
    let output = runtime.block_on(async {
        scope
            .submit_work(
                NumericWork {
                    input,
                    gate: None,
                    started: None,
                    panics: false,
                },
                credit(&pool),
                GatherStop::from_job(&context),
            )
            .await
            .unwrap()
            .finish()
            .await
            .unwrap()
    });
    runtime.block_on(context.gather_owner().close_and_drain());
    drop(scope);
    drop(context);
    let correct = output.value.sum == 21 && output.value.input.values == [3, 7, 11];
    let credit_alive = output.credit.size() >= 32_768 && pool.reserved() >= 32_768;
    let input_alive = weak.upgrade().is_some();
    drop(output);
    drop(runtime);
    service.shutdown();
    assert!(correct && credit_alive && input_alive);
    assert!(weak.upgrade().is_none());
    assert!(funded_drop.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(pool.reserved(), 0);
}

struct FundedInput {
    values: Vec<u64>,
    pool: Arc<dyn MemoryPool>,
    funded_drop: Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for FundedInput {
    fn drop(&mut self) {
        self.funded_drop.store(
            self.pool.reserved() >= 32_768,
            std::sync::atomic::Ordering::Release,
        );
    }
}

struct PreparedNumeric {
    sum: u64,
    input: Arc<FundedInput>,
}

struct NumericWork {
    input: Arc<FundedInput>,
    gate: Option<Arc<TypedGate>>,
    started: Option<std::sync::mpsc::Sender<()>>,
    panics: bool,
}

impl super::OwnedCpuWork for NumericWork {
    type Output = PreparedNumeric;

    fn run(self, stop: &GatherStop) -> crate::Result<Self::Output> {
        if let Some(started) = &self.started {
            let _ = started.send(());
        }
        if let Some(gate) = &self.gate {
            gate.wait();
        }
        assert!(!self.panics, "typed numeric worker panic");
        stop.check()?;
        Ok(PreparedNumeric {
            sum: self.input.values.iter().sum(),
            input: self.input,
        })
    }
}

#[derive(Default)]
struct TypedGate {
    released: parking_lot::Mutex<bool>,
    changed: parking_lot::Condvar,
}

impl TypedGate {
    fn wait(&self) {
        let mut released = self.released.lock();
        while !*released {
            self.changed.wait(&mut released);
        }
    }

    fn release(&self) {
        *self.released.lock() = true;
        self.changed.notify_all();
    }
}

#[test]
fn typed_settlement_drop_panic_does_not_strand_drain() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(5, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let funded_drop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let input = Arc::new(FundedInput {
        values: vec![3, 7, 11],
        pool: pool.clone(),
        funded_drop: funded_drop.clone(),
    });
    let weak = Arc::downgrade(&input);
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:sql".into()))
        .scope()
        .unwrap();
    let ticket = runtime.block_on(scope.submit_work(
        PanicOutputWork { input },
        credit(&pool),
        GatherStop::from_job(&context),
    ));
    let ticket = ticket.unwrap();
    let attempt = ticket.attempt;
    runtime.block_on(ticket.wait_settled()).unwrap();
    let abandoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(ticket)));
    let first_drain = runtime.block_on(async {
        tokio::time::timeout(
            Duration::from_secs(1),
            context.gather_owner().close_and_drain(),
        )
        .await
    });
    let bounded = first_drain.is_ok();
    let failures = if let Ok(failures) = first_drain {
        failures
    } else {
        // Release only after the failed bounded-drain observation.
        scope.home.release_slot(attempt);
        runtime.block_on(context.gather_owner().close_and_drain())
    };
    drop(scope);
    drop(context);
    drop(runtime);
    service.shutdown();
    assert!(
        abandoned.is_ok() && bounded,
        "settled result destructor escaped ownership and left terminal drain pending"
    );
    assert_eq!(failures.len(), 1);
    assert!(
        failures[0]
            .error
            .to_string()
            .contains("typed result destructor")
    );
    assert!(weak.upgrade().is_none());
    assert!(funded_drop.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn typed_settlement_cancel_rejects_owned_result() {
    settled_stop_case(false);
}

#[test]
fn typed_settlement_deadline_rejects_owned_result() {
    settled_stop_case(true);
}

fn settled_stop_case(deadline: bool) {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let expires = deadline.then(|| chrono::Utc::now() + chrono::Duration::seconds(5));
    let context = StreamJobContext::new(
        6,
        "typed-stop",
        JsonMap::new(),
        expires,
        CancellationToken::new(),
    )
    .with_gather_owner(service.owner("6".into()));
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let funded_drop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let input = Arc::new(FundedInput {
        values: vec![3, 7, 11],
        pool: pool.clone(),
        funded_drop: funded_drop.clone(),
    });
    let weak = Arc::downgrade(&input);
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:sql".into()))
        .scope()
        .unwrap();
    let ticket = runtime
        .block_on(scope.submit_work(
            NumericWork {
                input,
                gate: None,
                started: None,
                panics: false,
            },
            credit(&pool),
            GatherStop::from_job(&context),
        ))
        .unwrap();
    runtime.block_on(ticket.wait_settled()).unwrap();
    let successful_settlement = {
        let state = scope.home.state.lock();
        matches!(&state.slot, super::Slot::Active(record) if matches!(record.outcome, Some(Ok(_))))
    };
    expire_or_cancel(&context, expires);
    let result = runtime.block_on(ticket.finish());
    let rejected = matches!(result, Err(crate::CalcFlowError::Cancelled { .. }));
    let input_destroyed = weak.upgrade().is_none();
    drop(result);
    runtime.block_on(context.gather_owner().close_and_drain());
    drop(scope);
    drop(context);
    drop(runtime);
    service.shutdown();
    assert!(
        successful_settlement,
        "work must settle successfully before stop changes"
    );
    assert!(
        rejected,
        "settled owned result was delivered after its stop expired"
    );
    assert!(input_destroyed);
    assert!(funded_drop.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(pool.reserved(), 0);
}

fn expire_or_cancel(context: &StreamJobContext, expires: Option<chrono::DateTime<chrono::Utc>>) {
    if let Some(expires) = expires {
        while let Ok(remaining) = (expires - chrono::Utc::now()).to_std() {
            std::thread::sleep(remaining + Duration::from_millis(1));
        }
    } else {
        context.cancellation().cancel();
    }
}

struct PanicOutputWork {
    input: Arc<FundedInput>,
}

impl super::OwnedCpuWork for PanicOutputWork {
    type Output = PanicOutput;

    fn run(self, stop: &GatherStop) -> crate::Result<Self::Output> {
        stop.check()?;
        Ok(PanicOutput {
            sum: self.input.values.iter().sum(),
            input: self.input,
        })
    }
}

struct PanicOutput {
    sum: u64,
    input: Arc<FundedInput>,
}

impl Drop for PanicOutput {
    fn drop(&mut self) {
        assert_eq!(self.sum, self.input.values.iter().sum::<u64>());
        panic!("typed result destructor");
    }
}

#[test]
fn typed_managed_kernel_panic_preserves_registered_task_id() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(45, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let (registered, observed, report) = runtime.block_on(async {
        let mut supervisor =
            super::super::supervisor::TaskSupervisor::new(context.cancellation().clone());
        supervisor.spawn("identity-control", async { Ok(()) });
        let callback_job = context.clone();
        let callback_pool = pool.clone();
        let (observed_tx, observed_rx) = tokio::sync::oneshot::channel();
        let registered =
            supervisor.spawn_with_failure_signal("operator:sql", move |signal| async move {
                let callback = crate::StreamOperatorContext::new(&callback_job, "sql", None)
                    .with_task_id(Some(signal.task_id()));
                let scope = callback
                    .gather_client(GatherOperatorId::new("operator:sql".into()))
                    .scope()?;
                let mut work = numeric_work(45, &callback_pool);
                work.panics = true;
                let ticket = scope
                    .submit_work(
                        work,
                        credit(&callback_pool),
                        GatherStop::from_job(&callback_job),
                    )
                    .await?;
                let outcome = ticket.finish().await;
                let observed = match &outcome {
                    Err(crate::CalcFlowError::TaskPanicked { task_id, .. }) => Some(*task_id),
                    _ => None,
                };
                let _ = observed_tx.send(observed);
                outcome.map(|_| ())
            });
        let report = supervisor.join_all().await;
        let observed = observed_rx.await.unwrap();
        context.gather_owner().close_and_drain().await;
        (registered, observed, report)
    });
    drop(context);
    drop(runtime);
    service.shutdown();
    assert_ne!(registered.as_u64(), 0);
    assert_eq!(observed, Some(registered.as_u64()));
    assert_eq!(report.primary_errors()[0].task_id, registered);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn typed_abandoned_panic_diagnostics_report_bounded_overflow() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(46, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let failures = runtime.block_on(async {
        let scope = context
            .gather_owner()
            .client(GatherOperatorId::new("operator:sql".into()))
            .scope()
            .unwrap();
        for value in 0..10 {
            let mut work = numeric_work(value, &pool);
            work.panics = true;
            let ticket = scope
                .submit_work(work, credit(&pool), GatherStop::from_job(&context))
                .await
                .unwrap();
            ticket.wait_settled().await.unwrap();
            drop(ticket);
        }
        context.gather_owner().close_and_drain().await
    });
    drop(context);
    drop(runtime);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
    assert_eq!(
        failures.len(),
        9,
        "eight details plus explicit overflow summary"
    );
    let overflow = failures.last().unwrap();
    assert_eq!(overflow.task_name, "native CPU diagnostics overflow");
    assert!(
        matches!(&overflow.error, crate::CalcFlowError::Internal { message }
        if message == "native CPU diagnostics omitted=2")
    );
}

#[test]
fn typed_external_final_drop_without_tokio_waits_for_native_exit_and_credit() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(47, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let input = Arc::new(FundedInput {
        values: vec![47],
        pool: pool.clone(),
        funded_drop: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    });
    let funded_drop = input.funded_drop.clone();
    let weak = Arc::downgrade(&input);
    let gate = Arc::new(TypedGate::default());
    let _release = TypedGateRelease(gate.clone());
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:sql".into()))
        .scope()
        .unwrap();
    let ticket = runtime.block_on(scope.submit_work(
        NumericWork {
            input,
            gate: Some(gate.clone()),
            started: Some(started_tx),
            panics: false,
        },
        credit(&pool),
        GatherStop::from_job(&context),
    ));
    let started = started_rx.recv_timeout(Duration::from_secs(5)).is_ok();
    drop(ticket);
    drop(scope);
    drop(runtime);
    drop(context);
    let retained = weak.upgrade().is_some() && pool.reserved() >= 32_768;
    gate.release();
    let until = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < until && external_resources_live(&weak, &pool, &service) {
        std::thread::sleep(Duration::from_millis(2));
    }
    let joined_before_cleanup = service.joined_workers() == 1;
    let released_before_cleanup = weak.upgrade().is_none() && pool.reserved() == 0;
    service.shutdown();
    assert!(started && retained);
    assert!(joined_before_cleanup && released_before_cleanup);
    assert!(funded_drop.load(std::sync::atomic::Ordering::Acquire));
}

fn external_resources_live(
    weak: &std::sync::Weak<FundedInput>,
    pool: &Arc<dyn MemoryPool>,
    service: &TestService,
) -> bool {
    weak.upgrade().is_some() || pool.reserved() != 0 || service.joined_workers() == 0
}

struct TypedGateRelease(Arc<TypedGate>);

#[test]
fn typed_native_join_panic_is_published_before_generation_drain() {
    let service = TestService::new(1, 1).unwrap();
    service.panic_next_worker_exit();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(49, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let (registered, report, failures, funding) = runtime.block_on(async {
        let mut supervisor =
            super::super::supervisor::TaskSupervisor::new(context.cancellation().clone());
        supervisor.spawn("identity-control", async { Ok(()) });
        let callback_job = context.clone();
        let callback_pool = pool.clone();
        let registered =
            supervisor.spawn_with_failure_signal("operator:sql", move |signal| async move {
                let callback = crate::StreamOperatorContext::new(&callback_job, "sql", None)
                    .with_task_id(Some(signal.task_id()));
                let scope = callback
                    .gather_client(GatherOperatorId::new("operator:sql".into()))
                    .scope()?;
                let output = scope
                    .submit_work(
                        numeric_work(49, &callback_pool),
                        credit(&callback_pool),
                        GatherStop::from_job(&callback_job),
                    )
                    .await?
                    .finish()
                    .await?;
                assert_eq!(output.value.sum, 49);
                drop(output);
                Ok(())
            });
        let report = supervisor.join_all().await;
        let failures = context.gather_owner().close_and_drain().await;
        let funding = context.gather_owner().funding();
        (registered, report, failures, funding)
    });
    let joined = service.joined_workers();
    drop(context);
    drop(runtime);
    service.shutdown();
    assert_ne!(registered.as_u64(), 0);
    assert!(report.primary_errors().is_empty());
    assert_eq!(joined, 1);
    assert_eq!((funding.1, funding.2), (0, 0));
    assert_eq!(pool.reserved(), 0);
    assert_eq!(
        failures.len(),
        1,
        "actual native join panic must be published"
    );
    assert_eq!(failures[0].task_id, registered);
    assert!(
        matches!(&failures[0].error, crate::CalcFlowError::TaskPanicked { task_id, message }
        if *task_id == registered.as_u64() && message.contains("native worker exit panic"))
    );
}

#[test]
fn typed_reused_native_worker_join_panic_preserves_last_registered_task_id() {
    let service = TestService::new(1, 1).unwrap();
    service.panic_next_worker_exit();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(50, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let operator = GatherOperatorId::new("operator:sql".into());
    let (registered, generations, successful, reused, failures, funding) =
        runtime.block_on(async {
            let mut supervisor =
                super::super::supervisor::TaskSupervisor::new(context.cancellation().clone());
            let mut registered = Vec::new();
            let mut generations = Vec::new();
            let mut successful = true;
            for value in [17, 29] {
                let callback_job = context.clone();
                let callback_pool = pool.clone();
                let callback_operator = operator.clone();
                let (generation_tx, generation_rx) = tokio::sync::oneshot::channel();
                registered.push(supervisor.spawn_with_failure_signal(
                    "operator:sql",
                    move |signal| async move {
                        let callback =
                            crate::StreamOperatorContext::new(&callback_job, "sql", None)
                                .with_task_id(Some(signal.task_id()));
                        let scope = callback.gather_client(callback_operator).scope()?;
                        let ticket = scope
                            .submit_work(
                                numeric_work(value, &callback_pool),
                                credit(&callback_pool),
                                GatherStop::from_job(&callback_job),
                            )
                            .await?;
                        let _ = generation_tx.send(ticket.generation());
                        let output = ticket.finish().await?;
                        assert_output(&output, value);
                        drop(output);
                        Ok(())
                    },
                ));
                successful &= supervisor.join_all().await.primary_errors().is_empty();
                generations.push(generation_rx.await.unwrap());
            }
            let reused = service.joined_workers() == 0;
            let failures = context.gather_owner().close_and_drain().await;
            let funding = context.gather_owner().funding();
            (
                registered,
                generations,
                successful,
                reused,
                failures,
                funding,
            )
        });
    let joined = service.joined_workers();
    drop(operator);
    drop(context);
    drop(runtime);
    service.shutdown();
    assert!(successful && reused);
    assert_ne!(registered[0], registered[1]);
    assert_eq!(generations[0], generations[1]);
    assert_eq!(joined, 1);
    assert_eq!((funding.1, funding.2), (0, 0));
    assert_eq!(pool.reserved(), 0);
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].task_id, registered[1]);
    assert!(
        matches!(&failures[0].error, crate::CalcFlowError::TaskPanicked { task_id, message }
        if *task_id == registered[1].as_u64() && message.contains("native worker exit panic"))
    );
}

#[test]
fn typed_native_launch_panic_cleans_before_error_and_allows_same_home_retry() {
    let service = TestService::new(1, 1).unwrap();
    service.invalidate_next_worker_name();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(51, &service);
    let other_context = job(52, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:sql".into()))
        .scope()
        .unwrap();
    let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.block_on(scope.submit_work(
            numeric_work(51, &pool),
            credit(&pool),
            GatherStop::from_job(&context),
        ))
    }));
    let reported = matches!(
        &first,
        Ok(Err(super::AdmissionFailure::Runtime(
            crate::CalcFlowError::Internal { .. }
        )))
    );
    drop(first);
    let returned_funding = context.gather_owner().funding();
    let returned_capacity = service.available_capacity();
    let home = context.gather_owner().0.home.clone();
    let (cleaned, retried, other_progressed) = runtime.block_on(async {
        let cleaned = tokio::time::timeout(Duration::from_secs(3), home.wait_closed())
            .await
            .is_ok();
        let retried = tokio::time::timeout(Duration::from_secs(3), async {
            let output = scope
                .submit_work(
                    numeric_work(61, &pool),
                    credit(&pool),
                    GatherStop::from_job(&context),
                )
                .await?
                .finish()
                .await?;
            assert_output(&output, 61);
            drop(output);
            Ok::<_, crate::CalcFlowError>(())
        })
        .await;
        let other_progressed = tokio::time::timeout(Duration::from_secs(3), async {
            let other_scope = other_context
                .gather_owner()
                .client(GatherOperatorId::new("operator:sql".into()))
                .scope()?;
            let output = other_scope
                .submit_work(
                    numeric_work(71, &pool),
                    credit(&pool),
                    GatherStop::from_job(&other_context),
                )
                .await?
                .finish()
                .await?;
            assert_output(&output, 71);
            drop(output);
            Ok::<_, crate::CalcFlowError>(())
        })
        .await;
        context.gather_owner().close_and_drain().await;
        other_context.gather_owner().close_and_drain().await;
        (
            cleaned,
            matches!(retried, Ok(Ok(()))),
            matches!(other_progressed, Ok(Ok(()))),
        )
    });
    drop(home);
    drop(scope);
    drop(context);
    drop(other_context);
    drop(runtime);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
    assert!(cleaned && other_progressed);
    assert!(reported, "native launch panic must return a domain error");
    assert_eq!((returned_funding.1, returned_funding.2), (0, 0));
    assert_eq!(returned_capacity, (1, 1, 0));
    assert!(
        retried,
        "failed generation must leave its job admission open"
    );
}

#[test]
fn typed_waiting_job_gets_native_capacity_before_old_job_resubmission() {
    use std::future::poll_fn;
    use std::task::Poll;

    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let job_a = job(53, &service);
    let job_b = job(54, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let scope_a = job_a
        .gather_owner()
        .client(GatherOperatorId::new("operator:sql".into()))
        .scope()
        .unwrap();
    let scope_b = job_b
        .gather_owner()
        .client(GatherOperatorId::new("operator:sql".into()))
        .scope()
        .unwrap();
    let first_gate = Arc::new(TypedGate::default());
    let later_gate = Arc::new(TypedGate::default());
    let _release_first = TypedGateRelease(first_gate.clone());
    let _release_later = TypedGateRelease(later_gate.clone());
    let (first_work, first_rx) = gated_numeric_work(53, &pool, first_gate.clone());
    let first_ticket = runtime
        .block_on(scope_a.submit_work(first_work, credit(&pool), GatherStop::from_job(&job_a)))
        .unwrap();
    let started = first_rx.recv_timeout(Duration::from_secs(3)).is_ok();
    let (old_work, old_rx) = gated_numeric_work(63, &pool, later_gate.clone());
    let (waiting_work, waiting_rx) = gated_numeric_work(73, &pool, later_gate.clone());
    let mut old_request =
        Box::pin(scope_a.submit_work(old_work, credit(&pool), GatherStop::from_job(&job_a)));
    let mut waiting_request =
        Box::pin(scope_b.submit_work(waiting_work, credit(&pool), GatherStop::from_job(&job_b)));
    let (queued, pressured, first_succeeded, winner) = runtime.block_on(async {
        let mut premature = None;
        let queued = poll_fn(|context| match waiting_request.as_mut().poll(context) {
            Poll::Pending => Poll::Ready(true),
            Poll::Ready(result) => {
                premature = Some(result);
                Poll::Ready(false)
            }
        })
        .await;
        let pressured = service.waiting_requests() > 0;
        first_gate.release();
        let output = first_ticket.finish().await.unwrap();
        let first_succeeded = output.value.sum == 53 && output.credit.size() >= 32_768;
        drop(output);
        let winner = if let Some(result) = premature {
            Some((true, result))
        } else {
            tokio::time::timeout(Duration::from_secs(3), async {
                tokio::select! {
                    biased;
                    result = waiting_request.as_mut() => (true, result),
                    result = old_request.as_mut() => (false, result),
                }
            })
            .await
            .ok()
        };
        (queued, pressured, first_succeeded, winner)
    });
    let waiting_won = winner.as_ref().is_some_and(|(waiting, _)| *waiting);
    let winner_started = capacity_winner_started(winner.as_ref(), &waiting_rx, &old_rx);
    later_gate.release();
    let outputs_succeeded = runtime.block_on(finish_capacity_race(
        winner,
        old_request.as_mut(),
        waiting_request.as_mut(),
    ));
    drop(old_request);
    drop(waiting_request);
    runtime.block_on(async {
        job_a.gather_owner().close_and_drain().await;
        job_b.gather_owner().close_and_drain().await;
    });
    drop(scope_a);
    drop(scope_b);
    drop(job_a);
    drop(job_b);
    drop(runtime);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
    assert!(started && queued && pressured && first_succeeded);
    assert!(winner_started && outputs_succeeded);
    assert!(
        waiting_won,
        "old job bypassed the registered capacity waiter"
    );
}

fn gated_numeric_work(
    value: u64,
    pool: &Arc<dyn MemoryPool>,
    gate: Arc<TypedGate>,
) -> (NumericWork, std::sync::mpsc::Receiver<()>) {
    let (started, receiver) = std::sync::mpsc::channel();
    let mut work = numeric_work(value, pool);
    work.gate = Some(gate);
    work.started = Some(started);
    (work, receiver)
}

type NumericTicket = super::AdmissionResult<super::WorkTicket<PreparedNumeric>>;

fn capacity_winner_started(
    winner: Option<&(bool, NumericTicket)>,
    waiting_rx: &std::sync::mpsc::Receiver<()>,
    old_rx: &std::sync::mpsc::Receiver<()>,
) -> bool {
    match winner {
        Some((true, Ok(_))) => waiting_rx.recv_timeout(Duration::from_secs(3)).is_ok(),
        Some((false, Ok(_))) => old_rx.recv_timeout(Duration::from_secs(3)).is_ok(),
        _ => false,
    }
}

async fn finish_numeric_ticket(ticket: NumericTicket, expected: u64) -> crate::Result<bool> {
    let output = ticket?.finish().await?;
    let correct = output.value.sum == expected && output.credit.size() >= 32_768;
    drop(output);
    Ok(correct)
}

async fn finish_capacity_race(
    winner: Option<(bool, NumericTicket)>,
    mut old_request: Pin<&mut impl Future<Output = NumericTicket>>,
    mut waiting_request: Pin<&mut impl Future<Output = NumericTicket>>,
) -> bool {
    let Some((waiting, first)) = winner else {
        return false;
    };
    let first = finish_numeric_ticket(first, if waiting { 73 } else { 63 }).await;
    let second = tokio::time::timeout(Duration::from_secs(3), async {
        let ticket = if waiting {
            old_request.as_mut().await?
        } else {
            waiting_request.as_mut().await?
        };
        finish_numeric_ticket(Ok(ticket), if waiting { 63 } else { 73 }).await
    })
    .await;
    matches!(first, Ok(true)) && matches!(second, Ok(Ok(true)))
}

impl Drop for TypedGateRelease {
    fn drop(&mut self) {
        self.0.release();
    }
}

fn multi_started(probe: &crate::operator::gather_lifecycle_bridge::multi::Probe) -> bool {
    let mut entered = [false; 2];
    for _ in 0..2 {
        if let Ok(ordinal) = probe.entered.recv_timeout(Duration::from_secs(3)) {
            entered[ordinal] = true;
        }
    }
    entered.into_iter().all(std::convert::identity)
}

fn multi_funded(scope: &super::GatherScope, pool: &Arc<dyn MemoryPool>) -> bool {
    let state = scope.home.state.lock();
    let home = state.control.as_ref().map_or(0, MemoryReservation::size);
    let generation = state
        .pool
        .as_ref()
        .and_then(std::sync::Weak::upgrade)
        .map_or(0, |pool| pool.credit_size());
    let attempt = match &state.slot {
        super::Slot::Active(record) => record.credit.size(),
        _ => 0,
    };
    home == 16_384
        && generation == 16_384
        && attempt >= 33_554_432
        && pool.reserved() == home + generation + attempt
}

async fn multi_one_settled(scope: &super::GatherScope) -> bool {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let settled = {
                let state = scope.home.state.lock();
                matches!(&state.slot, super::Slot::Active(record)
                    if record.work.as_ref().is_some_and(|work| work.active() == 1))
            };
            if settled {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .is_ok()
}

fn multi_close(
    runtime: &tokio::runtime::Runtime,
    scope: &super::GatherScope,
    context: &StreamJobContext,
    service: TestService,
    attempt: u64,
) -> (bool, usize) {
    let drained = runtime.block_on(async {
        tokio::time::timeout(
            Duration::from_secs(5),
            context.gather_owner().close_and_drain(),
        )
        .await
    });
    let bounded = drained.is_ok();
    let joined = service.joined_workers();
    context.gather_owner().close_admission();
    service.shutdown();
    if !bounded {
        // Failed drain cleanup runs only after actual native joins.
        let record = {
            let mut state = scope.home.state.lock();
            if matches!(&state.slot, super::Slot::Active(record) | super::Slot::Parked(record)
                if record.id == attempt)
            {
                Some(super::detach(&mut state, attempt))
            } else {
                None
            }
        };
        if let Some(record) = record {
            scope.home.destroy_attempt(record);
        }
        scope.home.release_slot(attempt);
    }
    (bounded, joined)
}

#[test]
fn multi_arrow_abort_keeps_whole_attempt_paid_until_last_native_unit_and_join() {
    multi_arrow_abort(false);
}

#[test]
fn multi_arrow_rows_abort_keeps_whole_attempt_paid_until_last_native_unit_and_join() {
    multi_arrow_abort(true);
}

fn multi_input(
    row_mode: bool,
    pool: &Arc<dyn MemoryPool>,
    first_error: bool,
) -> crate::operator::gather_lifecycle_bridge::multi::Input {
    if row_mode {
        crate::operator::gather_lifecycle_bridge::multi::row_materialization(pool, first_error)
    } else {
        crate::operator::gather_lifecycle_bridge::multi::materialization(pool, first_error)
    }
}

fn multi_sources_paid(
    scope: &super::GatherScope,
    pool: &Arc<dyn MemoryPool>,
    probe: &crate::operator::gather_lifecycle_bridge::multi::Probe,
) -> bool {
    multi_funded(scope, pool)
        && probe.source.upgrade().is_some()
        && probe
            .shared_source
            .as_ref()
            .is_none_or(|source| source.upgrade().is_some())
}

fn assert_multi_cleanup(observed: [bool; 4], joined: usize, pool: &Arc<dyn MemoryPool>) {
    let [bounded, source_gone, shared_gone, row_ranges] = observed;
    assert!(bounded && source_gone && shared_gone && row_ranges);
    assert_eq!(joined, 2);
    assert_eq!(pool.reserved(), 0);
}

fn multi_arrow_abort(row_mode: bool) {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(31, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(67_108_864));
    let input = multi_input(row_mode, &pool, false);
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:asof".into()))
        .scope()
        .unwrap();
    let ticket = runtime
        .block_on(scope.submit(input.plan, input.credit, GatherStop::from_job(&context)))
        .unwrap();
    let attempt = ticket.attempt;
    let started = multi_started(&input.probe);
    drop(ticket);
    let mut drain = Box::pin(context.gather_owner().close_and_drain());
    let pending_both = runtime.block_on(async {
        tokio::time::timeout(Duration::from_millis(100), drain.as_mut())
            .await
            .is_err()
    });
    let paid_both = multi_sources_paid(&scope, &pool, &input.probe);
    input.probe.release(0);
    let first_settled = runtime.block_on(multi_one_settled(&scope));
    let pending_one = pending_both
        && runtime.block_on(async {
            tokio::time::timeout(Duration::from_millis(100), drain.as_mut())
                .await
                .is_err()
        });
    let paid_one = multi_sources_paid(&scope, &pool, &input.probe);
    input.probe.release_all();
    drop(drain);
    let (bounded, joined) = multi_close(&runtime, &scope, &context, service, attempt);
    let source_gone = input.probe.source.upgrade().is_none();
    let shared_gone = input
        .probe
        .shared_source
        .as_ref()
        .is_none_or(|source| source.upgrade().is_none());
    let row_ranges = !row_mode || input.probe.complete_row_ranges();
    drop(input.probe);
    drop(scope);
    drop(context);
    drop(runtime);
    assert!(started && first_settled);
    assert!(pending_both && pending_one && paid_both && paid_one);
    assert_multi_cleanup(
        [bounded, source_gone, shared_gone, row_ranges],
        joined,
        &pool,
    );
}

#[test]
fn multi_arrow_first_error_waits_for_running_sibling_and_preserves_primary() {
    multi_arrow_first_error(false);
}

#[test]
fn multi_arrow_rows_first_error_waits_for_running_sibling_and_preserves_primary() {
    multi_arrow_first_error(true);
}

fn multi_arrow_first_error(row_mode: bool) {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(32, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(67_108_864));
    let input = multi_input(row_mode, &pool, true);
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:asof".into()))
        .scope()
        .unwrap();
    let ticket = runtime
        .block_on(scope.submit(input.plan, input.credit, GatherStop::from_job(&context)))
        .unwrap();
    let attempt = ticket.attempt;
    let started = multi_started(&input.probe);
    input.probe.release(0);
    let first_settled = runtime.block_on(multi_one_settled(&scope));
    let mut finish = Box::pin(ticket.finish());
    let observed = runtime.block_on(async {
        tokio::time::timeout(Duration::from_millis(100), finish.as_mut()).await
    });
    let pending = observed.is_err();
    let paid = multi_sources_paid(&scope, &pool, &input.probe);
    input.probe.release_all();
    let completed = if pending {
        runtime
            .block_on(async { tokio::time::timeout(Duration::from_secs(5), finish.as_mut()).await })
    } else {
        observed
    };
    let expected = if row_mode {
        "multi-row oracle error"
    } else {
        "multi-column oracle error"
    };
    let primary = matches!(&completed,
        Ok(Err(crate::CalcFlowError::Internal { message })) if message == expected);
    drop(completed);
    drop(finish);
    let (bounded, joined) = multi_close(&runtime, &scope, &context, service, attempt);
    let source_gone = input.probe.source.upgrade().is_none();
    let shared_gone = input
        .probe
        .shared_source
        .as_ref()
        .is_none_or(|source| source.upgrade().is_none());
    let row_ranges = !row_mode || input.probe.complete_row_ranges();
    drop(input.probe);
    drop(scope);
    drop(context);
    drop(runtime);
    assert!(started && first_settled && pending && paid);
    assert!(primary);
    assert_multi_cleanup(
        [bounded, source_gone, shared_gone, row_ranges],
        joined,
        &pool,
    );
}
