mod support;

use std::sync::{Arc, Mutex, atomic::AtomicUsize};

use calc_flow::{
    CalcFlowError, Checkpoint, CheckpointStore, MicroBatchRunner, PipelineBuilder, SinkRouter,
    UdfRegistry,
};
use chrono::Utc;
use serde_json::json;
use support::{
    Action, MemoryCheckpointStore, Probe, QueueSource, RecordingSink, TestOperator,
    partially_failing_reset_plan, source_item, stateful_plan,
};

#[tokio::test]
async fn sink_failure_rolls_back_and_retries_the_same_source_item() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("runner", Arc::clone(&probe));
    let initial = plan.snapshot().await.unwrap();
    let (source, opens) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let store = Arc::new(MemoryCheckpointStore::default());
    let calls = Arc::new(Mutex::new(Vec::new()));
    let never = Arc::new(AtomicUsize::new(0));
    let once = Arc::new(AtomicUsize::new(1));
    let mut sinks = SinkRouter::new();
    sinks
        .add(
            "output",
            Box::new(RecordingSink::new(
                "first",
                Arc::clone(&calls),
                Arc::clone(&never),
            )),
        )
        .unwrap();
    sinks
        .add(
            "output",
            Box::new(RecordingSink::new("second", Arc::clone(&calls), once)),
        )
        .unwrap();
    let mut runner = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(source),
        sinks,
        clone_store(&store),
        1,
    )
    .unwrap();

    assert!(runner.next().await.is_err());
    assert!(store.checkpoint().is_none());
    assert_eq!(plan.snapshot().await.unwrap(), initial);
    let result = runner.next().await.unwrap().unwrap();

    assert_eq!(result.outputs["output"].num_rows(), 1);
    assert_eq!(store.checkpoint().unwrap().sequence, 1);
    assert_eq!(probe.calls(), 2);
    assert_eq!(opens.lock().unwrap().len(), 1);
    assert_eq!(
        calls
            .lock()
            .unwrap()
            .iter()
            .map(|call| call.0.as_str())
            .collect::<Vec<_>>(),
        ["first", "second", "first", "second"]
    );
}

#[tokio::test]
async fn checkpoint_cadence_and_eof_flush_use_the_latest_position() {
    let plan = stateful_plan("cadence", Arc::new(Probe::default()));
    let (source, _) = QueueSource::new(vec![
        source_item(&[1], json!("one"), 11),
        source_item(&[2], json!("two"), 12),
        source_item(&[3], json!("three"), 13),
    ]);
    let store = Arc::new(MemoryCheckpointStore::default());
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        clone_store(&store),
        2,
    )
    .unwrap();

    assert!(runner.next().await.unwrap().is_some());
    assert!(store.checkpoint().is_none());
    assert!(runner.next().await.unwrap().is_some());
    assert_eq!(store.checkpoint().unwrap().sequence, 12);
    assert!(runner.next().await.unwrap().is_some());
    assert_eq!(store.checkpoint().unwrap().sequence, 12);
    assert!(runner.next().await.unwrap().is_none());
    let checkpoint = store.checkpoint().unwrap();
    assert_eq!(checkpoint.sequence, 13);
    assert_eq!(checkpoint.source_cursor, Some(json!("three")));
    assert_eq!(store.saves(), 2);
}

#[tokio::test]
async fn checkpoint_failure_rolls_back_and_does_not_skip_the_item() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("save retry", Arc::clone(&probe));
    let initial = plan.snapshot().await.unwrap();
    let (source, opens) = QueueSource::new(vec![source_item(&[1], json!(1), 4)]);
    let store = Arc::new(MemoryCheckpointStore::default());
    store.fail_next_saves(1);
    let mut runner = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(source),
        SinkRouter::new(),
        clone_store(&store),
        1,
    )
    .unwrap();

    assert!(runner.next().await.is_err());
    assert_eq!(plan.snapshot().await.unwrap(), initial);
    assert!(runner.next().await.unwrap().is_some());
    assert_eq!(store.checkpoint().unwrap().sequence, 4);
    assert_eq!(probe.calls(), 2);
    assert_eq!(opens.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn execution_failure_is_rolled_back_and_the_item_remains_retryable() {
    let probe = Arc::new(Probe::default());
    let plan = Arc::new(
        PipelineBuilder::new("execution failure")
            .unwrap()
            .add_node(
                "node",
                Box::new(
                    TestOperator::transform("node", Action::Fail("boom"), Arc::clone(&probe))
                        .stateful(),
                ),
            )
            .unwrap()
            .compile(&UdfRegistry::new().snapshot())
            .unwrap(),
    );
    let initial = plan.snapshot().await.unwrap();
    let (source, opens) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let store = Arc::new(MemoryCheckpointStore::default());
    let mut runner = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(source),
        SinkRouter::new(),
        clone_store(&store),
        1,
    )
    .unwrap();

    assert!(runner.next().await.is_err());
    assert!(runner.next().await.is_err());
    assert_eq!(probe.calls(), 2);
    assert_eq!(plan.snapshot().await.unwrap(), initial);
    assert_eq!(opens.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn stale_checkpoint_is_rejected_before_restore_or_source_open() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("current", Arc::clone(&probe));
    let checkpoint = Checkpoint::new(
        "stale",
        plan.fingerprint(),
        Some(json!(7)),
        7,
        plan.snapshot().await.unwrap(),
        Utc::now(),
    )
    .unwrap();
    let store = Arc::new(MemoryCheckpointStore::with_checkpoint(checkpoint));
    let (source, opens) = QueueSource::new(Vec::new());
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        clone_store(&store),
        1,
    )
    .unwrap();

    assert!(matches!(
        runner.next().await,
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(probe.restores(), 0);
    assert!(opens.lock().unwrap().is_empty());
}

#[tokio::test]
async fn valid_checkpoint_restores_once_and_opens_at_its_cursor() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("recover", Arc::clone(&probe));
    let checkpoint = Checkpoint::new(
        plan.name(),
        plan.fingerprint(),
        Some(json!("cursor")),
        9,
        plan.snapshot().await.unwrap(),
        Utc::now(),
    )
    .unwrap();
    let store = Arc::new(MemoryCheckpointStore::with_checkpoint(checkpoint));
    let (source, opens) = QueueSource::new(Vec::new());
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        clone_store(&store),
        2,
    )
    .unwrap();

    assert!(runner.next().await.unwrap().is_none());
    assert!(runner.next().await.unwrap().is_none());
    assert_eq!(probe.restores(), 1);
    assert_eq!(opens.lock().unwrap().as_slice(), [Some(json!("cursor"))]);
}

#[tokio::test]
async fn eof_checkpoint_failure_retries_the_pending_checkpoint() {
    let plan = stateful_plan("eof retry", Arc::new(Probe::default()));
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!(1), 3)]);
    let store = Arc::new(MemoryCheckpointStore::default());
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        clone_store(&store),
        2,
    )
    .unwrap();

    assert!(runner.next().await.unwrap().is_some());
    store.fail_next_saves(1);
    assert!(runner.next().await.is_err());
    assert!(store.checkpoint().is_none());
    assert!(runner.next().await.unwrap().is_none());
    assert_eq!(store.checkpoint().unwrap().sequence, 3);
}

#[tokio::test]
async fn delivery_and_rollback_failures_keep_both_diagnostics() {
    let plan = Arc::new(
        PipelineBuilder::new("rollback diagnostics")
            .unwrap()
            .add_node(
                "node",
                Box::new(
                    TestOperator::transform("node", Action::Pass, Arc::new(Probe::default()))
                        .stateful()
                        .failing_restore(),
                ),
            )
            .unwrap()
            .compile(&UdfRegistry::new().snapshot())
            .unwrap(),
    );
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let store = Arc::new(MemoryCheckpointStore::default());
    let mut sinks = SinkRouter::new();
    sinks
        .add(
            "output",
            Box::new(RecordingSink::new(
                "delivery",
                Arc::new(Mutex::new(Vec::new())),
                Arc::new(AtomicUsize::new(1)),
            )),
        )
        .unwrap();
    let mut runner =
        MicroBatchRunner::new(plan, Box::new(source), sinks, clone_store(&store), 1).unwrap();

    let error = runner.next().await.unwrap_err().to_string();
    assert!(error.contains("delivery injected"));
    assert!(error.contains("rollback also failed"));
    assert!(error.contains("restore injected"));
}

#[tokio::test]
async fn reset_clears_checkpoint_state_and_starts_a_new_source_lifecycle() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("reset", Arc::clone(&probe));
    let (source, opens) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let store = Arc::new(MemoryCheckpointStore::default());
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        clone_store(&store),
        1,
    )
    .unwrap();

    assert!(runner.next().await.unwrap().is_some());
    assert!(store.checkpoint().is_some());
    runner.reset().await.unwrap();
    assert!(store.checkpoint().is_none());
    assert_eq!(probe.resets(), 1);
    assert!(runner.next().await.unwrap().is_none());
    assert_eq!(opens.lock().unwrap().as_slice(), [None, None]);
}

#[tokio::test]
async fn partially_failing_reset_restores_all_pre_reset_state() {
    let first_probe = Arc::new(Probe::default());
    let second_probe = Arc::new(Probe::default());
    let plan = partially_failing_reset_plan(
        "reset rollback",
        Arc::clone(&first_probe),
        Arc::clone(&second_probe),
    );
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let store = Arc::new(MemoryCheckpointStore::default());
    let mut runner = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(source),
        SinkRouter::new(),
        clone_store(&store),
        1,
    )
    .unwrap();
    runner.next().await.unwrap();
    let before = plan.snapshot().await.unwrap();

    let error = runner.reset().await.unwrap_err().to_string();

    assert!(error.contains("reset injected"));
    assert_eq!(plan.snapshot().await.unwrap(), before);
    assert!(store.checkpoint().is_some());
    assert_eq!(first_probe.resets(), 1);
    assert_eq!(second_probe.resets(), 1);
}

#[test]
fn constructor_requires_positive_cadence_and_exactly_one_external_input() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("limits", probe);
    let (source, _) = QueueSource::new(Vec::new());
    let store = Arc::new(MemoryCheckpointStore::default());
    assert!(matches!(
        MicroBatchRunner::new(plan, Box::new(source), SinkRouter::new(), clone_store(&store), 0),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "checkpoint_every"
    ));

    let two_inputs = Arc::new(
        PipelineBuilder::new("two inputs")
            .unwrap()
            .add_node(
                "one",
                Box::new(TestOperator::transform(
                    "one",
                    Action::Pass,
                    Arc::new(Probe::default()),
                )),
            )
            .unwrap()
            .add_node(
                "two",
                Box::new(TestOperator::transform(
                    "two",
                    Action::Pass,
                    Arc::new(Probe::default()),
                )),
            )
            .unwrap()
            .compile(&UdfRegistry::new().snapshot())
            .unwrap(),
    );
    let (source, _) = QueueSource::new(Vec::new());
    assert!(matches!(
        MicroBatchRunner::new(
            two_inputs,
            Box::new(source),
            SinkRouter::new(),
            clone_store(&store),
            1,
        ),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "plan.external_inputs"
    ));
}

fn clone_store(store: &Arc<MemoryCheckpointStore>) -> Arc<dyn CheckpointStore> {
    Arc::clone(store) as Arc<dyn CheckpointStore>
}
