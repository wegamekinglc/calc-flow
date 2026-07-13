mod support;

use std::sync::{Arc, Mutex, atomic::AtomicUsize};

use calc_flow::{CalcFlowError, Checkpoint, CheckpointStore, SinkRouter, StreamingRunner};
use chrono::Utc;
use serde_json::json;
use support::{
    MemoryCheckpointStore, Probe, RecordingSink, int_batch, partially_failing_reset_plan,
    stateful_plan,
};

#[tokio::test]
async fn streaming_recovers_once_and_advances_after_delivery() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("stream", Arc::clone(&probe));
    let checkpoint = Checkpoint::new(
        plan.name(),
        plan.fingerprint(),
        Some(json!("ignored for push input")),
        7,
        plan.snapshot().await.unwrap(),
        Utc::now(),
    )
    .unwrap();
    let store = Arc::new(MemoryCheckpointStore::with_checkpoint(checkpoint));
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut sinks = SinkRouter::new();
    sinks
        .add(
            "output",
            Box::new(RecordingSink::new(
                "sink",
                Arc::clone(&calls),
                Arc::new(AtomicUsize::new(0)),
            )),
        )
        .unwrap();
    let mut runner = StreamingRunner::new(Arc::clone(&plan), clone_store(&store)).unwrap();

    let first = runner.step(int_batch(&[1]), &mut sinks).await.unwrap();
    assert_eq!(first.metadata.pipeline_fingerprint, plan.fingerprint());
    assert_eq!(store.checkpoint().unwrap().sequence, 8);
    let second = runner.step(int_batch(&[2]), &mut sinks).await.unwrap();
    assert_eq!(store.checkpoint().unwrap().sequence, 9);
    assert_eq!(probe.restores(), 1);
    let calls = calls.lock().unwrap();
    assert_eq!(calls[0].1, first.metadata.run_id);
    assert_eq!(calls[1].1, second.metadata.run_id);
}

#[tokio::test]
async fn sink_failure_rolls_back_and_does_not_advance_sequence() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("stream failure", Arc::clone(&probe));
    let initial = plan.snapshot().await.unwrap();
    let store = Arc::new(MemoryCheckpointStore::default());
    let mut sinks = SinkRouter::new();
    sinks
        .add(
            "output",
            Box::new(RecordingSink::new(
                "sink",
                Arc::new(Mutex::new(Vec::new())),
                Arc::new(AtomicUsize::new(1)),
            )),
        )
        .unwrap();
    let mut runner = StreamingRunner::new(Arc::clone(&plan), clone_store(&store)).unwrap();

    assert!(runner.step(int_batch(&[1]), &mut sinks).await.is_err());
    assert_eq!(plan.snapshot().await.unwrap(), initial);
    assert!(store.checkpoint().is_none());
    runner.step(int_batch(&[1]), &mut sinks).await.unwrap();
    assert_eq!(store.checkpoint().unwrap().sequence, 0);
}

#[tokio::test]
async fn checkpoint_failure_restores_state_and_allows_explicit_batch_retry() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("stream save", Arc::clone(&probe));
    let initial = plan.snapshot().await.unwrap();
    let store = Arc::new(MemoryCheckpointStore::default());
    store.fail_next_saves(1);
    let mut runner = StreamingRunner::new(Arc::clone(&plan), clone_store(&store)).unwrap();
    let mut sinks = SinkRouter::new();

    assert!(runner.step(int_batch(&[1]), &mut sinks).await.is_err());
    assert_eq!(plan.snapshot().await.unwrap(), initial);
    runner.step(int_batch(&[1]), &mut sinks).await.unwrap();
    assert_eq!(store.checkpoint().unwrap().sequence, 0);
    assert_eq!(probe.calls(), 2);
}

#[tokio::test]
async fn stale_fingerprint_is_rejected_before_restore() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("stream stale", Arc::clone(&probe));
    let checkpoint = Checkpoint::new(
        plan.name(),
        "stale",
        None,
        4,
        plan.snapshot().await.unwrap(),
        Utc::now(),
    )
    .unwrap();
    let store = Arc::new(MemoryCheckpointStore::with_checkpoint(checkpoint));
    let mut runner = StreamingRunner::new(plan, clone_store(&store)).unwrap();

    assert!(matches!(
        runner.step(int_batch(&[1]), &mut SinkRouter::new()).await,
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(probe.restores(), 0);
}

#[tokio::test]
async fn reset_clears_the_checkpoint_and_keeps_fresh_sequence_zero() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("stream reset", Arc::clone(&probe));
    let store = Arc::new(MemoryCheckpointStore::default());
    let mut runner = StreamingRunner::new(plan, clone_store(&store)).unwrap();
    let mut sinks = SinkRouter::new();

    runner.step(int_batch(&[1]), &mut sinks).await.unwrap();
    assert!(store.checkpoint().is_some());
    runner.reset().await.unwrap();
    assert!(store.checkpoint().is_none());
    assert_eq!(probe.resets(), 1);
    runner.step(int_batch(&[2]), &mut sinks).await.unwrap();
    assert_eq!(store.checkpoint().unwrap().sequence, 0);
}

#[tokio::test]
async fn partially_failing_reset_restores_all_pre_reset_state() {
    let first_probe = Arc::new(Probe::default());
    let second_probe = Arc::new(Probe::default());
    let plan = partially_failing_reset_plan(
        "stream reset rollback",
        Arc::clone(&first_probe),
        Arc::clone(&second_probe),
    );
    let store = Arc::new(MemoryCheckpointStore::default());
    let mut runner = StreamingRunner::new(Arc::clone(&plan), clone_store(&store)).unwrap();
    runner
        .step(int_batch(&[1]), &mut SinkRouter::new())
        .await
        .unwrap();
    let before = plan.snapshot().await.unwrap();

    let error = runner.reset().await.unwrap_err().to_string();

    assert!(error.contains("reset injected"));
    assert_eq!(plan.snapshot().await.unwrap(), before);
    assert!(store.checkpoint().is_some());
    assert_eq!(first_probe.resets(), 1);
    assert_eq!(second_probe.resets(), 1);
}

#[tokio::test]
async fn maximum_next_sequence_is_rejected_before_execution_or_save() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("stream overflow", Arc::clone(&probe));
    let checkpoint = Checkpoint::new(
        plan.name(),
        plan.fingerprint(),
        None,
        u64::MAX - 1,
        plan.snapshot().await.unwrap(),
        Utc::now(),
    )
    .unwrap();
    let store = Arc::new(MemoryCheckpointStore::with_checkpoint(checkpoint));
    let mut runner = StreamingRunner::new(plan, clone_store(&store)).unwrap();

    assert!(matches!(
        runner.step(int_batch(&[1]), &mut SinkRouter::new()).await,
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(probe.calls(), 0);
    assert_eq!(store.saves(), 0);
}

fn clone_store(store: &Arc<MemoryCheckpointStore>) -> Arc<dyn CheckpointStore> {
    Arc::clone(store) as Arc<dyn CheckpointStore>
}
