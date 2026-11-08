use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use tokio::sync::{Semaphore, oneshot};

use super::Workers;
use crate::{CalcFlowError, CancellationToken, Result, RunContext};

fn context(cancellation: CancellationToken) -> RunContext {
    RunContext::new(BTreeMap::new(), None, cancellation).unwrap()
}
fn single() -> Arc<Workers> {
    Arc::new(Workers {
        permits: Arc::new(Semaphore::new(1)),
    })
}
fn held_job(
    check: &mut dyn FnMut() -> Result<()>,
    started: oneshot::Sender<()>,
    observed: oneshot::Sender<()>,
    release: &mpsc::Receiver<()>,
) -> Result<()> {
    started.send(()).unwrap();
    let until = Instant::now() + Duration::from_secs(2);
    while Instant::now() < until {
        if let Err(error) = check() {
            observed.send(()).unwrap();
            release.recv_timeout(Duration::from_secs(2)).unwrap();
            return Err(error);
        }
        std::thread::yield_now();
    }
    panic!("worker did not observe cancellation");
}

async fn receipt(receiver: oneshot::Receiver<()>) {
    tokio::time::timeout(Duration::from_secs(3), receiver)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn test_explicit_cancellation_waits_for_worker_and_retains_permit() {
    let pool = single();
    let worker = Arc::clone(&pool);
    let token = CancellationToken::new();
    let run = context(token.clone());
    let (started, start) = oneshot::channel();
    let (observed, stop) = oneshot::channel();
    let (release, held) = mpsc::channel();
    let task = tokio::spawn(async move {
        worker
            .run(run, move |check| held_job(check, started, observed, &held))
            .await
    });
    receipt(start).await;
    token.cancel();
    receipt(stop).await;
    assert!(!task.is_finished());
    assert_eq!(pool.permits.available_permits(), 0);
    release.send(()).unwrap();
    assert!(matches!(
        task.await.unwrap(),
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(pool.permits.available_permits(), 1);
}

#[tokio::test]
async fn test_aborted_waiter_requests_local_cancel_and_supervisor_drains_before_reuse() {
    let pool = single();
    let worker = Arc::clone(&pool);
    let (started, start) = oneshot::channel();
    let (observed, stop) = oneshot::channel();
    let (release, held) = mpsc::channel();
    let task = tokio::spawn(async move {
        worker
            .run(context(CancellationToken::new()), move |check| {
                held_job(check, started, observed, &held)
            })
            .await
    });
    receipt(start).await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    receipt(stop).await;
    assert_eq!(pool.permits.available_permits(), 0);
    release.send(()).unwrap();
    let permit = tokio::time::timeout(
        Duration::from_secs(3),
        Arc::clone(&pool.permits).acquire_owned(),
    )
    .await
    .unwrap()
    .unwrap();
    drop(permit);
    assert_eq!(
        pool.run(context(CancellationToken::new()), |_| Ok(42))
            .await
            .unwrap(),
        42
    );
}

#[tokio::test]
async fn test_cancelled_queued_admission_does_not_launch_or_consume_a_worker() {
    let pool = single();
    let permit = Arc::clone(&pool.permits).acquire_owned().await.unwrap();
    let token = CancellationToken::new();
    let run = context(token.clone());
    let called = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&called);
    let worker = Arc::clone(&pool);
    let task = tokio::spawn(async move {
        worker
            .run(run, move |_| {
                flag.store(true, Ordering::SeqCst);
                Ok(())
            })
            .await
    });
    tokio::task::yield_now().await;
    token.cancel();
    let result = tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert!(!called.load(Ordering::SeqCst));
    assert_eq!(pool.permits.available_permits(), 0);
    drop(permit);
    assert_eq!(pool.permits.available_permits(), 1);
}

#[tokio::test]
async fn test_stream_queued_admission_observes_job_cancellation_before_starting_work() {
    let pool = single();
    let permit = Arc::clone(&pool.permits).acquire_owned().await.unwrap();
    let token = CancellationToken::new();
    let context =
        crate::StreamJobContext::new(8, "native-stream", BTreeMap::new(), None, token.clone());
    let called = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&called);
    let worker = Arc::clone(&pool);
    let task = tokio::spawn(async move {
        worker
            .run_stream(&context, move |_| {
                flag.store(true, Ordering::SeqCst);
                Ok(())
            })
            .await
    });
    tokio::task::yield_now().await;
    token.cancel();
    let result = tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert!(!called.load(Ordering::SeqCst));
    assert_eq!(pool.permits.available_permits(), 0);
    drop(permit);
    assert_eq!(pool.permits.available_permits(), 1);
}

#[test]
fn test_borrowed_stream_control_keeps_actual_job_cancellation_after_context_drop() {
    let token = CancellationToken::new();
    let settings = BTreeMap::from([(
        "evidence".into(),
        serde_json::Value::String("large".repeat(4096)),
    )]);
    let context = crate::StreamJobContext::new(91, "native-control", settings, None, token.clone());
    let scope = super::Scope::stream(&context);
    drop(context);
    token.cancel();
    assert!(
        matches!(scope.check_cancelled(),Err(CalcFlowError::Cancelled{run_id}) if run_id=="91")
    );
}

#[test]
fn test_borrowed_stream_control_keeps_actual_job_deadline() {
    let deadline = chrono::Utc::now() - chrono::Duration::seconds(1);
    let job = crate::StreamJobContext::new(
        92,
        "native-control",
        BTreeMap::new(),
        Some(deadline),
        CancellationToken::new(),
    );
    let scope = super::Scope::stream(&job);
    assert!(
        matches!(scope.check_cancelled(),Err(CalcFlowError::Cancelled{run_id}) if run_id=="92")
    );
}

#[tokio::test]
async fn test_borrowed_stream_control_observes_the_jobs_test_clock() {
    let now = chrono::Utc::now();
    let deadline = now + chrono::Duration::hours(1);
    let (clock, current) = tokio::sync::watch::channel(now);
    let job = crate::StreamJobContext::new(
        93,
        "native-control",
        BTreeMap::new(),
        Some(deadline),
        CancellationToken::new(),
    )
    .with_deadline_clock_for_test(current);
    let scope = super::Scope::stream(&job);
    assert!(scope.check_cancelled().is_ok());
    clock.send(deadline).unwrap();
    assert!(
        matches!(scope.check_cancelled(),Err(CalcFlowError::Cancelled{run_id}) if run_id=="93")
    );
}
