use super::*;

struct GatedCleanupInput {
    input: Option<Arc<FundedInput>>,
    home: std::sync::Weak<super::super::GatherHome>,
    owner_funded: Arc<std::sync::atomic::AtomicBool>,
    entered: std::sync::mpsc::Sender<()>,
    gate: Arc<TypedGate>,
}

impl Drop for GatedCleanupInput {
    fn drop(&mut self) {
        self.entered.send(()).unwrap();
        self.gate.wait();
        let home = self.home.upgrade().unwrap();
        let input = self.input.take().unwrap();
        let pool = input.pool.clone();
        drop(input);
        self.owner_funded.store(
            original_credit_paid(&home, &pool),
            std::sync::atomic::Ordering::Release,
        );
    }
}

fn original_credit_paid(home: &Arc<super::super::GatherHome>, pool: &Arc<dyn MemoryPool>) -> bool {
    let state = home.state.lock();
    let home_bytes = state.control.as_ref().map_or(0, MemoryReservation::size);
    let generation = state.pool.as_ref().and_then(std::sync::Weak::upgrade);
    let generation_bytes = generation.as_ref().map_or(0, |pool| pool.credit_size());
    pool.reserved() >= home_bytes + generation_bytes + 32_768
}

impl super::super::OwnedCpuWork for GatedCleanupInput {
    type Output = ();

    fn run(self, _: &GatherStop) -> crate::Result<()> {
        unreachable!("a cancelled pre-install work item must never run")
    }
}

#[test]
fn pre_install_submission_keeps_guard_until_owner_and_credit_drop() {
    for polled in [false, true] {
        pre_install_drop(polled);
    }
}

fn pre_install_drop(polled: bool) {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let context = job(26, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:checkpoint".into()))
        .scope()
        .unwrap();
    let first_gate = Arc::new(TypedGate::default());
    let first = polled.then(|| {
        let mut work = numeric_work(26, &pool);
        work.gate = Some(first_gate.clone());
        let ticket = runtime
            .block_on(scope.submit_work(work, credit(&pool), GatherStop::from_job(&context)))
            .unwrap();
        ticket.observe_cleanup(context.gather_owner().retain_retirement().unwrap())
    });
    let input = numeric_work(27, &pool).input;
    let weak = Arc::downgrade(&input);
    let funded = input.funded_drop.clone();
    let owner_funded = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let gate = Arc::new(TypedGate::default());
    let (entered, dropping) = std::sync::mpsc::channel();
    let mut observer = None;
    let mut pending = Box::pin(scope.submit_observed_work(
        GatedCleanupInput {
            input: Some(input),
            home: Arc::downgrade(&context.gather_owner().0.home),
            owner_funded: owner_funded.clone(),
            entered,
            gate: gate.clone(),
        },
        credit(&pool),
        GatherStop::from_job(&context),
        context.gather_owner().retain_retirement().unwrap(),
        &mut observer,
    ));
    if polled {
        runtime.block_on(async {
            assert!(futures::poll!(pending.as_mut()).is_pending());
        });
    }
    std::thread::scope(|threads| {
        let dropper = threads.spawn(move || drop(pending));
        dropping.recv_timeout(Duration::from_secs(2)).unwrap();
        if let Some((ticket, cleanup)) = first {
            drop(ticket);
            first_gate.release();
            runtime
                .block_on(cleanup.wait(&GatherStop::from_job(&context)))
                .unwrap();
        }
        let guarded = context.gather_owner().0.home.state.lock().retiring_owners == 1;
        let alive = weak.upgrade().is_some();
        let (_, _, attempt) = context.gather_owner().funding();
        assert_eq!(
            attempt, 0,
            "the cancelled second work item must remain uninstalled"
        );
        let paid = original_credit_paid(&context.gather_owner().0.home, &pool);
        let waiting = runtime.block_on(async {
            let mut drain = Box::pin(context.gather_owner().close_and_drain());
            futures::poll!(drain.as_mut()).is_pending()
        });
        gate.release();
        dropper.join().unwrap();
        runtime.block_on(context.gather_owner().close_and_drain());
        assert!(
            guarded && alive && paid && waiting,
            "pre-install Drop must retain its funded cleanup guard: {guarded}/{alive}/{paid}/{waiting}"
        );
    });
    assert!(funded.load(std::sync::atomic::Ordering::Acquire));
    assert!(
        owner_funded.load(std::sync::atomic::Ordering::Acquire),
        "the original credit must remain paid beyond home/generation during owner Drop"
    );
    assert!(weak.upgrade().is_none());
    assert!(
        observer.is_none(),
        "pre-install cancellation cannot publish an installed attempt"
    );
    drop(observer);
    drop(scope);
    drop(context);
    drop(runtime);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn cleanup_observer_keeps_escaped_output_funded_until_install() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for fails in [false, true] {
        runtime.block_on(escaped_output(&service, fails));
    }
    drop(runtime);
    service.shutdown();
}

async fn escaped_output(service: &TestService, fails: bool) {
    let context = job(24, service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let work = numeric_work(24, &pool);
    let weak = Arc::downgrade(&work.input);
    let funded = work.input.funded_drop.clone();
    let scope = context
        .gather_owner()
        .client(GatherOperatorId::new("operator:checkpoint".into()))
        .scope()
        .unwrap();
    let ticket = scope
        .submit_work(work, credit(&pool), GatherStop::from_job(&context))
        .await
        .unwrap();
    let (ticket, observer) =
        ticket.observe_cleanup(context.gather_owner().retain_retirement().unwrap());
    let output = ticket.finish().await.unwrap();
    assert_eq!(context.gather_owner().funding().2, 0);
    let (home, generation, _) = context.gather_owner().funding();
    assert!(pool.reserved() >= home + generation + 32_768);
    assert!(weak.upgrade().is_some());
    let stop = GatherStop::from_job(&context);
    let mut waiting = Box::pin(observer.wait(&stop));
    assert!(futures::poll!(waiting.as_mut()).is_pending());
    drop(waiting);
    let mut drain = Box::pin(context.gather_owner().close_and_drain());
    assert!(futures::poll!(drain.as_mut()).is_pending());
    drop(drain);
    let result = output.install(|value| {
        assert_eq!(value.sum, 24);
        assert!(pool.reserved() >= 32_768);
        drop(value);
        if fails {
            Err(crate::CalcFlowError::Internal {
                message: "install failed".into(),
            })
        } else {
            Ok(())
        }
    });
    assert_eq!(result.is_err(), fails);
    observer.wait(&stop).await.unwrap();
    assert!(weak.upgrade().is_none());
    assert!(funded.load(std::sync::atomic::Ordering::Acquire));
    assert!(context.gather_owner().close_and_drain().await.is_empty());
    let (home, generation, attempt) = context.gather_owner().funding();
    assert_eq!(pool.reserved(), home + generation + attempt);
    drop(observer);
    drop(scope);
    drop(context);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn cleanup_observer_preserves_cancel_and_deadline_failure() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let context = job(25, &service);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
        let scope = context
            .gather_owner()
            .client(GatherOperatorId::new("operator:checkpoint".into()))
            .scope()
            .unwrap();
        let ticket = scope
            .submit_work(
                numeric_work(25, &pool),
                credit(&pool),
                GatherStop::from_job(&context),
            )
            .await
            .unwrap();
        let (ticket, observer) =
            ticket.observe_cleanup(context.gather_owner().retain_retirement().unwrap());
        let output = ticket.finish().await.unwrap();
        let mut stop = GatherStop::from_job(&context);
        stop.deadline = Some(chrono::Utc::now() - chrono::Duration::seconds(1));
        assert!(matches!(
            observer.wait(&stop).await,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        stop.deadline = None;
        context.cancellation().cancel();
        assert!(matches!(
            observer.wait(&stop).await,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        drop(output);
        assert!(context.gather_owner().close_and_drain().await.is_empty());
        drop(observer);
        drop(scope);
        drop(context);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}
