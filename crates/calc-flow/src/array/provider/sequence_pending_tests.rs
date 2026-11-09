use std::{
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use tokio::sync::oneshot;

use super::{
    sequence_pending::Pending,
    worker::{Scope, Workers},
};
use crate::{CalcFlowError, CancellationToken, JsonMap, Result, StreamJobContext};

fn context(token: CancellationToken) -> StreamJobContext {
    StreamJobContext::new(77, "sequence-pending", JsonMap::new(), None, token)
}

async fn receipt(receiver: oneshot::Receiver<()>) {
    tokio::time::timeout(Duration::from_secs(3), receiver)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn test_sequence_pending_gate_retains_worker_ownership_after_lost_awaiter() {
    let pending = Arc::new(Pending::default());
    let job = context(CancellationToken::new());
    let lease = pending.enter(&job).await.unwrap();
    let workers = Workers::default();
    let (started, ready) = oneshot::channel();
    let (observed, stopped) = oneshot::channel();
    let (release, held) = mpsc::channel();
    let task = tokio::spawn(async move {
        workers
            .run_scoped(Scope::stream(&job), "array_sequence", move |check| {
                let _lease = lease;
                held_worker(check, started, observed, &held)
            })
            .await
    });
    receipt(ready).await;
    task.abort();
    receipt(stopped).await;
    assert!(pending.try_enter().is_err());
    release.send(()).unwrap();
    let job = context(CancellationToken::new());
    let _lease = tokio::time::timeout(Duration::from_secs(3), pending.enter(&job))
        .await
        .unwrap()
        .unwrap();
}

fn held_worker(
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
    panic!("native worker did not observe waiter loss");
}

#[tokio::test]
async fn test_sequence_pending_result_holds_gate_until_emit_boundary_is_settled() {
    let pending = Pending::default();
    let job = context(CancellationToken::new());
    let lease = pending.enter(&job).await.unwrap();
    let result = Workers::default()
        .run_scoped(Scope::stream(&job), "array_sequence", move |_| {
            Ok((42, lease))
        })
        .await
        .unwrap();
    assert_eq!(result.0, 42);
    assert!(pending.try_enter().is_err());
    drop(result);
    assert!(pending.try_enter().is_ok());
}

#[tokio::test]
async fn test_sequence_pending_admission_checks_actual_job_cancellation() {
    let pending = Pending::default();
    let token = CancellationToken::new();
    let job = context(token.clone());
    let _lease = pending.enter(&job).await.unwrap();
    let next = tokio::spawn(async move { pending.enter(&job).await });
    tokio::task::yield_now().await;
    token.cancel();
    let result = tokio::time::timeout(Duration::from_secs(3), next)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(result, Err(CalcFlowError::Cancelled { run_id }) if run_id == "77"));
}

#[test]
fn test_rolling_pending_gate_reports_builtin_identity_without_changing_sequence_default() {
    let rolling = Pending::rolling("features");
    let _held = rolling.try_enter().unwrap();
    let error = rolling.try_enter().unwrap_err();
    assert!(matches!(error, CalcFlowError::Operator { ref node_id, .. } if node_id == "features"));
    assert!(error.to_string().contains("state.pending"));
    let sequence = Pending::default();
    let _held = sequence.try_enter().unwrap();
    assert!(
        sequence
            .try_enter()
            .unwrap_err()
            .to_string()
            .contains("native:array_sequence@1")
    );
}
