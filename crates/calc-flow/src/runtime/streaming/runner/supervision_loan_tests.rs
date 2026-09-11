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
    let mut loan = home.install(supervisor, cpu.clone());
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
    let mut returned = home.take(cpu).expect("abort returns the unique loan");
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
    let mut loan = home.install(supervisor, cpu);
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
