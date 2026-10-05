use super::*;

#[tokio::test]
async fn aborted_driver_returns_complete_supervisor_and_settled_primary_error() {
    let cancellation = CancellationToken::new();
    let home = SupervisionHome::default();
    let cpu = JobEntityWorkOwner::new(7, Arc::new(AtomicU64::new(0)));
    let mut supervisor = TaskSupervisor::new(cancellation.clone());
    let failed_id = supervisor.spawn("operator:original", async {
        Err(CalcFlowError::Internal {
            message: "original failure".into(),
        })
    });
    supervisor.spawn("pending", std::future::pending());
    let registry = supervisor.registry();
    let mut loan = home.install(
        supervisor,
        cpu.clone(),
        JobSqlRecoveryOwner::new(),
        super::super::gather_work::JobGatherOwner::new("7".into()),
    );
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let driver = tokio::spawn(async move {
        // The original join keeps its settled error in the supervisor while
        // the other task remains pending.
        let joined = loan.join_all();
        tokio::pin!(joined);
        assert!(futures::poll!(&mut joined).is_pending());
        let _ = entered_tx.send(());
        joined.await
    });
    entered_rx.await.unwrap();
    tokio::task::yield_now().await;
    driver.abort();
    assert!(driver.await.unwrap_err().is_cancelled());
    assert!(!registry.snapshot().is_empty());
    let mut returned = home
        .take(
            cpu,
            JobSqlRecoveryOwner::new(),
            super::super::gather_work::JobGatherOwner::new("7".into()),
        )
        .expect("abort returns the unique loan");
    let report = returned.join_all().await;
    assert_eq!(report.primary_errors().len(), 1);
    assert_eq!(report.primary_errors()[0].task_id, failed_id);
    assert!(
        matches!(&report.primary_errors()[0].error, CalcFlowError::Internal { message } if message == "original failure")
    );
    assert!(registry.snapshot().is_empty());
    drop(returned);
    home.clear_joined();
    assert!(!home.is_loaned());
}

#[tokio::test]
async fn prepared_report_survives_driver_drop_until_lifecycle_takes_it() {
    let home = SupervisionHome::default();
    let cpu = JobEntityWorkOwner::new(7, Arc::new(AtomicU64::new(0)));
    let supervisor = TaskSupervisor::new(CancellationToken::new());
    let mut loan = home.install(
        supervisor,
        cpu,
        JobSqlRecoveryOwner::new(),
        super::super::gather_work::JobGatherOwner::new("7".into()),
    );
    assert!(loan.join_all().await.errors.is_empty());
    let launch_id = LaunchId::new(1);
    home.prepare(DriverReport::aborted(launch_id, "prepared original"));
    drop(loan);
    home.clear_joined();
    let report = home.take_report().expect("one prepared report");
    assert_eq!(report.launch_id, launch_id);
    let DriverCompletion::StartFailed(failure) = report.completion else {
        panic!("keep original completion");
    };
    assert!(
        failure
            .primary
            .error
            .to_string()
            .contains("prepared original")
    );
    assert!(home.take_report().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn gather_report_waits_for_dropped_arrow_observer_and_actual_worker_exit() {
    let job = crate::StreamJobContext::new(
        7,
        "gather",
        crate::JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let (work, probe) = crate::operator::gather_lifecycle_bridge::materialization(job.clone());
    let mut work = Box::pin(work);
    loop {
        assert!(futures::poll!(work.as_mut()).is_pending());
        if probe.started.try_recv().is_ok() {
            break;
        }
        tokio::task::yield_now().await;
    }
    drop(work);
    assert!(probe.source.upgrade().is_some());
    let (home_bytes, generation_bytes, attempt_bytes) = job.gather_owner().funding();
    assert!(attempt_bytes >= 32_768);
    assert_eq!(
        probe.pool.reserved(),
        home_bytes + generation_bytes + attempt_bytes
    );
    let home = SupervisionHome::default();
    let cpu = JobEntityWorkOwner::new(7, Arc::new(AtomicU64::new(0)));
    let supervisor = TaskSupervisor::new(CancellationToken::new());
    let mut loan = home.install(
        supervisor,
        cpu,
        JobSqlRecoveryOwner::new(),
        job.gather_owner().clone(),
    );
    let waited_for_worker;
    {
        let joined = loan.join_all();
        tokio::pin!(joined);
        let first_join = tokio::time::timeout(Duration::from_millis(25), joined.as_mut()).await;
        waited_for_worker = first_join.is_err();
        probe.release.wait();
        if waited_for_worker {
            let _ = joined.await;
        }
    }
    drop(loan);
    home.clear_joined();
    drop(job);
    assert!(
        waited_for_worker,
        "managed report published while real ASOF Arrow gather remained gated"
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while probe.source.upgrade().is_some() || probe.pool.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn gather_columns_only_runtime_report_waits_for_actual_worker_exit() {
    let job = crate::StreamJobContext::new(
        8,
        "gather",
        crate::JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let (work, probe) =
        crate::operator::gather_lifecycle_bridge::runtime_materialization(job.clone());
    let mut work = Box::pin(work);
    let started = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Poll::Ready(result) = futures::poll!(work.as_mut()) {
                return result.and_then(|()| {
                    Err(CalcFlowError::Internal {
                        message: "ASOF work completed before its worker handshake".into(),
                    })
                });
            }
            if probe.started.try_recv().is_ok() {
                return Ok(());
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    drop(work);
    let retained = probe.source.upgrade().is_some();
    let schema_released = probe.output_schema.upgrade().is_none();
    let funding = job.gather_owner().funding();
    let paid = probe.pool.reserved();
    let funded = funding.2 >= 32_768 && paid == funding.0 + funding.1 + funding.2;
    let home = SupervisionHome::default();
    let cpu = JobEntityWorkOwner::new(8, Arc::new(AtomicU64::new(0)));
    let supervisor = TaskSupervisor::new(CancellationToken::new());
    let mut loan = home.install(
        supervisor,
        cpu,
        JobSqlRecoveryOwner::new(),
        job.gather_owner().clone(),
    );
    let waited_for_worker;
    {
        let joined = loan.join_all();
        tokio::pin!(joined);
        waited_for_worker = tokio::time::timeout(Duration::from_millis(25), joined.as_mut())
            .await
            .is_err();
        probe.worker.release();
        if waited_for_worker {
            let _ = joined.await;
        }
    }
    drop(loan);
    home.clear_joined();
    drop(job);
    let cleaned = tokio::time::timeout(Duration::from_secs(5), async {
        while probe.source.upgrade().is_some() || probe.pool.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(
        matches!(started, Ok(Ok(()))),
        "actual Arrow worker never started: {started:?}"
    );
    assert!(
        cleaned.is_ok(),
        "actual Arrow source or credit remained after worker exit"
    );
    assert!(retained, "worker lost source arrays before release");
    assert!(schema_released, "worker retained observer output schema");
    assert!(
        waited_for_worker,
        "managed report published while columns-only ASOF worker was live"
    );
    assert!(
        funded,
        "job attempt did not own complete ASOF source/worker budget: paid={paid} funding={funding:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn gather_aborted_report_loan_keeps_native_join_and_original_failure() {
    let job = crate::StreamJobContext::new(
        48,
        "gather-race",
        crate::JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let (work, probe) =
        crate::operator::gather_lifecycle_bridge::runtime_materialization(job.clone());
    let _release = GatherWorkerRelease(probe.worker.clone());
    let mut work = Box::pin(work);
    let started = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if futures::poll!(work.as_mut()).is_ready() {
                return false;
            }
            if probe.started.try_recv().is_ok() {
                return true;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or(false);
    drop(work);
    let source_retained = probe.source.upgrade().is_some();
    let home = SupervisionHome::default();
    let cpu = JobEntityWorkOwner::new(48, Arc::new(AtomicU64::new(0)));
    let mut supervisor = TaskSupervisor::new(job.cancellation().clone());
    let original = supervisor.spawn("operator:original", async {
        Err(CalcFlowError::Internal {
            message: "original failure".into(),
        })
    });
    let mut loan = home.install(
        supervisor,
        cpu.clone(),
        JobSqlRecoveryOwner::new(),
        job.gather_owner().clone(),
    );
    let (waiting_tx, waiting_rx) = tokio::sync::oneshot::channel();
    let driver = tokio::spawn(async move {
        let joined = loan.join_all();
        tokio::pin!(joined);
        if let Ok(report) = tokio::time::timeout(Duration::from_millis(25), joined.as_mut()).await {
            let _ = waiting_tx.send(false);
            report
        } else {
            let _ = waiting_tx.send(true);
            joined.await
        }
    });
    let waited = tokio::time::timeout(Duration::from_secs(5), waiting_rx)
        .await
        .is_ok_and(|result| result.unwrap_or(false));
    driver.abort();
    let aborted = driver.await.is_err_and(|error| error.is_cancelled());
    probe.worker.release();
    let mut returned = home.take(cpu, JobSqlRecoveryOwner::new(), job.gather_owner().clone());
    let report = if let Some(loan) = &mut returned {
        tokio::time::timeout(Duration::from_secs(5), loan.join_all())
            .await
            .ok()
    } else {
        None
    };
    drop(returned);
    home.clear_joined();
    let joined_funding = job.gather_owner().funding();
    drop(job);
    let cleaned = tokio::time::timeout(Duration::from_secs(5), async {
        while probe.source.upgrade().is_some() || probe.pool.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .is_ok();
    assert_aborted_gather_observations([started, source_retained, waited, aborted, cleaned]);
    assert_eq!((joined_funding.1, joined_funding.2), (0, 0));
    let report = report.expect("returned supervisor completes actual native drain");
    assert_eq!(report.primary_errors().len(), 1);
    assert_eq!(report.primary_errors()[0].task_id, original);
    assert!(matches!(&report.primary_errors()[0].error,
        CalcFlowError::Internal { message } if message == "original failure"));
}

struct GatherWorkerRelease(Arc<crate::operator::gather_lifecycle_bridge::WorkerProbe>);

impl Drop for GatherWorkerRelease {
    fn drop(&mut self) {
        self.0.release();
    }
}

fn assert_aborted_gather_observations(observed: [bool; 5]) {
    let [started, source_retained, waited, aborted, cleaned] = observed;
    assert!(started && source_retained && waited && aborted && cleaned);
}
