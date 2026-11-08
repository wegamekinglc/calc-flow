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
