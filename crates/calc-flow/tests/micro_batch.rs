mod support;

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, atomic::AtomicUsize},
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchOperator, BatchingSource, CalcFlowError, Checkpoint, CheckpointStore,
    ExecutionOptions, MicroBatchRunner, PipelineBuilder, Result, RunContext, Sink, SinkRouter,
    Source, SourceItem, StreamingRunner, UdfRegistry,
};
use chrono::Utc;
use serde_json::json;
use support::{
    Action, CommitThenResetGateStore, GateOnCallSink, GatedCandidateSource, GatedOnceSink,
    GatedSink, MemoryCheckpointStore, Probe, QueueSource, RecordingSink, StoreGate, TestOperator,
    VisibleGateCheckpointStore, partially_failing_reset_plan, source_item, stateful_plan,
};

#[tokio::test]
async fn cancelled_next_at_sink_restores_before_retrying_the_same_item() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("cancelled sink", Arc::clone(&probe));
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!("one"), 1)]);
    let store = Arc::new(MemoryCheckpointStore::default());
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mut sinks = SinkRouter::new();
    sinks
        .add(
            "output",
            Box::new(GatedOnceSink::new(
                Arc::clone(&started),
                Arc::clone(&release),
            )),
        )
        .unwrap();
    let mut runner =
        MicroBatchRunner::new(plan, Box::new(source), sinks, clone_store(&store), 1).unwrap();

    let mut cancelled = Box::pin(runner.next());
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("sink gate did not suspend next: {result:?}"),
    }
    drop(cancelled);

    runner.next().await.unwrap().unwrap();
    let checkpoint = store.checkpoint().unwrap();
    assert_eq!(checkpoint.sequence, 1);
    assert_eq!(checkpoint.source_cursor, Some(json!("one")));
    assert_eq!(checkpoint.state["node"]["state"], json!(1));
    assert_eq!(probe.calls(), 2, "the cancelled delivery is replayed");
}

#[tokio::test]
async fn cancelled_visible_save_restores_the_old_checkpoint_before_retry() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("cancelled save", Arc::clone(&probe));
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!("new"), 8)]);
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let store = Arc::new(VisibleGateCheckpointStore::new(
        None,
        StoreGate::Save,
        Arc::clone(&started),
        release,
    ));
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        Arc::clone(&store) as Arc<dyn CheckpointStore>,
        1,
    )
    .unwrap();

    let mut cancelled = Box::pin(runner.next());
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("visible save did not suspend next: {result:?}"),
    }
    assert_eq!(store.checkpoint().unwrap().sequence, 8);
    drop(cancelled);

    runner.next().await.unwrap().unwrap();
    let checkpoint = store.checkpoint().unwrap();
    assert_eq!(checkpoint.sequence, 8);
    assert_eq!(checkpoint.source_cursor, Some(json!("new")));
    assert_eq!(checkpoint.state["node"]["state"], json!(1));
    assert_eq!(probe.calls(), 2);
}

#[tokio::test]
async fn cancelled_reset_restores_plan_and_checkpoint_before_reuse() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("cancelled reset", Arc::clone(&probe));
    let initial_checkpoint = Checkpoint::new(
        plan.name(),
        plan.fingerprint(),
        Some(json!("old")),
        4,
        BTreeMap::from([("node".into(), json!({"state": 1}))]),
        Utc::now(),
    )
    .unwrap();
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let store = Arc::new(VisibleGateCheckpointStore::new(
        Some(initial_checkpoint.clone()),
        StoreGate::Delete,
        Arc::clone(&started),
        release,
    ));
    let (source, _) = QueueSource::new(Vec::new());
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        Arc::clone(&store) as Arc<dyn CheckpointStore>,
        1,
    )
    .unwrap();
    assert!(runner.next().await.unwrap().is_none());
    assert_eq!(
        runner.plan_snapshot().await.unwrap()["node"]["state"],
        json!(1)
    );

    let mut cancelled = Box::pin(runner.reset());
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("visible delete did not suspend reset: {result:?}"),
    }
    assert!(store.checkpoint().is_none());
    drop(cancelled);

    assert_eq!(
        runner.plan_snapshot().await.unwrap()["node"]["state"],
        json!(1)
    );
    assert_eq!(store.checkpoint(), Some(initial_checkpoint));
    assert!(runner.next().await.unwrap().is_none());
}

#[tokio::test]
async fn dropping_cancelled_runner_requires_store_recovery_before_state_is_visible() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("fresh recovery", Arc::clone(&probe));
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let store = Arc::new(MemoryCheckpointStore::default());
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mut sinks = SinkRouter::new();
    sinks
        .add(
            "output",
            Box::new(GatedOnceSink::new(started.clone(), release)),
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
    let mut cancelled = Box::pin(runner.next());
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("sink gate did not suspend next: {result:?}"),
    }
    drop(cancelled);
    drop(runner);

    assert!(matches!(
        plan.reset().await,
        Err(CalcFlowError::RecoveryRequired { .. })
    ));
    assert!(matches!(
        plan.snapshot().await,
        Err(CalcFlowError::RecoveryRequired { .. })
    ));
    let (replay, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let mut fresh = MicroBatchRunner::new(
        plan,
        Box::new(replay),
        SinkRouter::new(),
        clone_store(&store),
        1,
    )
    .unwrap();
    fresh.next().await.unwrap().unwrap();
    assert_eq!(
        fresh.plan_snapshot().await.unwrap()["node"]["state"],
        json!(1)
    );
}

#[tokio::test]
async fn failed_cancel_compensation_poisons_until_successful_reset() {
    let plan = stateful_plan("cancel poison", Arc::new(Probe::default()));
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let store = Arc::new(VisibleGateCheckpointStore::new(
        None,
        StoreGate::Save,
        Arc::clone(&started),
        release,
    ));
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        Arc::clone(&store) as Arc<dyn CheckpointStore>,
        1,
    )
    .unwrap();
    let mut cancelled = Box::pin(runner.next());
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("save gate did not suspend next: {result:?}"),
    }
    drop(cancelled);
    store.fail_next_compensations(1);

    let recovery = runner.next().await.unwrap_err().to_string();
    assert!(recovery.contains("compensation"));
    assert!(
        runner
            .next()
            .await
            .unwrap_err()
            .to_string()
            .contains("poisoned")
    );
    runner.reset().await.unwrap();
    assert!(store.checkpoint().is_none());
}

#[tokio::test]
async fn cancelled_eof_flush_retries_commit_without_reexecuting_item() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("cancelled eof", Arc::clone(&probe));
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let store = Arc::new(VisibleGateCheckpointStore::new(
        None,
        StoreGate::Save,
        Arc::clone(&started),
        release,
    ));
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        Arc::clone(&store) as Arc<dyn CheckpointStore>,
        2,
    )
    .unwrap();
    runner.next().await.unwrap().unwrap();
    let mut cancelled = Box::pin(runner.next());
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("EOF save gate did not suspend next: {result:?}"),
    }
    drop(cancelled);

    assert_eq!(
        runner.plan_snapshot().await.unwrap()["node"]["state"],
        json!(1)
    );
    assert!(runner.next().await.unwrap().is_none());
    assert_eq!(store.checkpoint().unwrap().sequence, 1);
    assert_eq!(store.saves(), 2, "recovery must not immediately save twice");
    assert_eq!(probe.calls(), 1);
}

#[tokio::test]
async fn cancelled_batching_candidate_retries_the_complete_group() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("cancelled batching candidate", Arc::clone(&probe));
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let source = BatchingSource::new(
        GatedCandidateSource::new(
            vec![
                source_item(&[1], json!(1), 1),
                source_item(&[2], json!(2), 2),
            ],
            2,
            Arc::clone(&started),
            release,
        ),
        10,
        usize::MAX,
    )
    .unwrap();
    let store = Arc::new(MemoryCheckpointStore::default());
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        clone_store(&store),
        1,
    )
    .unwrap();

    let mut cancelled = Box::pin(runner.next());
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("candidate gate did not suspend runner: {result:?}"),
    }
    drop(cancelled);

    let result = runner.next().await.unwrap().unwrap();
    assert_eq!(result.outputs["output"].num_rows(), 2);
    let checkpoint = store.checkpoint().unwrap();
    assert_eq!(checkpoint.sequence, 2);
    assert_eq!(checkpoint.source_cursor, Some(json!(2)));
    assert_eq!(checkpoint.state["node"]["state"], json!(1));
    assert_eq!(probe.calls(), 1);
}

#[tokio::test]
async fn dropping_pending_cadence_restores_the_last_durable_baseline_for_replay() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("drop pending cadence", Arc::clone(&probe));
    let store = Arc::new(MemoryCheckpointStore::default());
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let mut runner = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(source),
        SinkRouter::new(),
        clone_store(&store),
        3,
    )
    .unwrap();
    runner.next().await.unwrap().unwrap();
    assert_eq!(
        runner.plan_snapshot().await.unwrap()["node"]["state"],
        json!(1)
    );
    assert!(store.checkpoint().is_none());
    drop(runner);

    assert!(matches!(
        plan.snapshot().await,
        Err(CalcFlowError::RecoveryRequired { .. })
    ));
    let (replay, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let mut fresh = MicroBatchRunner::new(
        plan,
        Box::new(replay),
        SinkRouter::new(),
        clone_store(&store),
        3,
    )
    .unwrap();
    fresh.next().await.unwrap().unwrap();
    assert_eq!(
        fresh.plan_snapshot().await.unwrap()["node"]["state"],
        json!(1)
    );
    assert_eq!(probe.calls(), 2);
}

#[tokio::test]
async fn dropping_after_later_cancel_replays_from_the_durable_baseline() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("drop after later cancel", Arc::clone(&probe));
    let store = Arc::new(MemoryCheckpointStore::default());
    let (source, _) = QueueSource::new(vec![
        source_item(&[1], json!(1), 1),
        source_item(&[2], json!(2), 2),
    ]);
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mut sinks = SinkRouter::new();
    sinks
        .add(
            "output",
            Box::new(GateOnCallSink::new(2, Arc::clone(&started), release)),
        )
        .unwrap();
    let mut runner = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(source),
        sinks,
        clone_store(&store),
        3,
    )
    .unwrap();
    runner.next().await.unwrap().unwrap();

    let mut cancelled = Box::pin(runner.next());
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("second sink write did not suspend: {result:?}"),
    }
    drop(cancelled);
    drop(runner);

    let (replay, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let mut fresh = MicroBatchRunner::new(
        plan,
        Box::new(replay),
        SinkRouter::new(),
        clone_store(&store),
        3,
    )
    .unwrap();
    fresh.next().await.unwrap().unwrap();
    assert_eq!(
        fresh.plan_snapshot().await.unwrap()["node"]["state"],
        json!(1)
    );
    assert_eq!(probe.calls(), 3);
}

#[tokio::test]
async fn same_runner_cancel_retains_prior_pending_cadence_progress() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("same runner later cancel", Arc::clone(&probe));
    let store = Arc::new(MemoryCheckpointStore::default());
    let (source, _) = QueueSource::new(vec![
        source_item(&[1], json!(1), 1),
        source_item(&[2], json!(2), 2),
    ]);
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mut sinks = SinkRouter::new();
    sinks
        .add(
            "output",
            Box::new(GateOnCallSink::new(2, Arc::clone(&started), release)),
        )
        .unwrap();
    let mut runner =
        MicroBatchRunner::new(plan, Box::new(source), sinks, clone_store(&store), 3).unwrap();
    runner.next().await.unwrap().unwrap();

    let mut cancelled = Box::pin(runner.next());
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("second sink write did not suspend: {result:?}"),
    }
    drop(cancelled);
    runner.next().await.unwrap().unwrap();

    assert_eq!(
        runner.plan_snapshot().await.unwrap()["node"]["state"],
        json!(2)
    );
    assert_eq!(probe.calls(), 3);
}

#[tokio::test]
async fn cancelled_forced_reset_replaces_failed_commit_marker_before_mutation() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("forced reset cancellation", Arc::clone(&probe));
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let save_started = Arc::new(tokio::sync::Notify::new());
    let save_release = Arc::new(tokio::sync::Notify::new());
    let delete_started = Arc::new(tokio::sync::Notify::new());
    let delete_release = Arc::new(tokio::sync::Notify::new());
    let store = Arc::new(CommitThenResetGateStore::new(
        Arc::clone(&save_started),
        save_release,
        Arc::clone(&delete_started),
        delete_release,
    ));
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        Arc::clone(&store) as Arc<dyn CheckpointStore>,
        2,
    )
    .unwrap();
    runner.next().await.unwrap().unwrap();

    let mut cancelled_commit = Box::pin(runner.next());
    tokio::select! {
        () = save_started.notified() => {}
        result = &mut cancelled_commit => panic!("EOF commit did not suspend: {result:?}"),
    }
    drop(cancelled_commit);
    store.fail_next_saves(2);
    assert!(
        runner
            .next()
            .await
            .unwrap_err()
            .to_string()
            .contains("commit recovery")
    );

    let mut cancelled_reset = Box::pin(runner.reset());
    tokio::select! {
        () = delete_started.notified() => {}
        result = &mut cancelled_reset => panic!("forced reset delete did not suspend: {result:?}"),
    }
    assert!(store.checkpoint().is_none());
    drop(cancelled_reset);

    runner.reset().await.unwrap();
    assert!(store.checkpoint().is_none());
    assert_eq!(
        runner.plan_snapshot().await.unwrap()["node"]["state"],
        json!(0)
    );
    assert_eq!(
        probe.calls(),
        1,
        "reset recovery must not execute the item again"
    );
}

#[tokio::test]
async fn runner_holds_the_plan_transaction_through_sink_and_checkpoint() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("atomic runner", Arc::clone(&probe));
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let store = Arc::new(MemoryCheckpointStore::default());
    let sink_started = Arc::new(tokio::sync::Notify::new());
    let sink_release = Arc::new(tokio::sync::Notify::new());
    let direct_attempted = Arc::new(tokio::sync::Notify::new());
    let mut sinks = SinkRouter::new();
    sinks
        .add(
            "output",
            Box::new(GatedSink::new(
                Arc::clone(&sink_started),
                Arc::clone(&sink_release),
            )),
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

    let runner_task = tokio::spawn(async move { runner.next().await });
    sink_started.notified().await;
    let direct_plan = Arc::clone(&plan);
    let direct_attempt = Arc::clone(&direct_attempted);
    let direct_task = tokio::spawn(async move {
        direct_attempt.notify_one();
        direct_plan
            .execute(
                BTreeMap::from([("input".into(), support::int_batch(&[2]))]),
                ExecutionOptions::default(),
            )
            .await
    });
    direct_attempted.notified().await;
    tokio::task::yield_now().await;

    let direct_error = tokio::time::timeout(std::time::Duration::from_millis(100), direct_task)
        .await
        .expect("a leased-plan error must not wait for the runner transaction")
        .unwrap()
        .unwrap_err();
    assert!(matches!(direct_error, CalcFlowError::PlanLeased { .. }));
    assert_eq!(probe.calls(), 1, "direct execute reached the operator");
    sink_release.notify_one();
    runner_task.await.unwrap().unwrap().unwrap();

    assert_eq!(store.checkpoint().unwrap().state["node"]["state"], json!(1));
    assert_eq!(plan.snapshot().await.unwrap()["node"]["state"], json!(1));
}

#[tokio::test]
async fn runner_lease_rejects_direct_calls_and_a_second_runner_until_drop() {
    let plan = stateful_plan("exclusive lease", Arc::new(Probe::default()));
    let store = Arc::new(MemoryCheckpointStore::default());
    let (source, _) = QueueSource::new(vec![
        source_item(&[1], json!(1), 1),
        source_item(&[2], json!(2), 2),
    ]);
    let mut runner = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(source),
        SinkRouter::new(),
        clone_store(&store),
        1,
    )
    .unwrap();

    runner.next().await.unwrap().unwrap();
    let direct = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        plan.execute(
            BTreeMap::from([("input".into(), support::int_batch(&[9]))]),
            ExecutionOptions::default(),
        ),
    )
    .await
    .expect("direct execute must fail fast while leased")
    .unwrap_err();
    assert!(matches!(
        direct,
        CalcFlowError::PlanLeased { pipeline_name } if pipeline_name == "exclusive lease"
    ));
    assert!(matches!(
        tokio::time::timeout(std::time::Duration::from_millis(100), plan.snapshot())
            .await
            .expect("direct snapshot must fail fast while leased"),
        Err(CalcFlowError::PlanLeased { .. })
    ));
    assert!(matches!(
        tokio::time::timeout(std::time::Duration::from_millis(100), plan.reset())
            .await
            .expect("direct reset must fail fast while leased"),
        Err(CalcFlowError::PlanLeased { .. })
    ));
    assert!(matches!(
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            plan.restore(&BTreeMap::new())
        )
        .await
        .expect("direct restore must fail fast while leased"),
        Err(CalcFlowError::PlanLeased { .. })
    ));

    let (other_source, _) = QueueSource::new(Vec::new());
    let second = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(other_source),
        SinkRouter::new(),
        clone_store(&store),
        1,
    );
    assert!(matches!(second, Err(CalcFlowError::PlanLeased { .. })));
    assert!(matches!(
        StreamingRunner::new(Arc::clone(&plan), clone_store(&store)),
        Err(CalcFlowError::PlanLeased { .. })
    ));
    runner.next().await.unwrap().unwrap();

    drop(runner);
    plan.execute(
        BTreeMap::from([("input".into(), support::int_batch(&[3]))]),
        ExecutionOptions::default(),
    )
    .await
    .unwrap();
    let (fresh_source, _) = QueueSource::new(Vec::new());
    MicroBatchRunner::new(
        plan,
        Box::new(fresh_source),
        SinkRouter::new(),
        clone_store(&store),
        1,
    )
    .unwrap();
}

#[tokio::test]
async fn runner_lease_closes_the_queued_direct_call_race() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let plan = Arc::new(
        PipelineBuilder::new("queued direct")
            .unwrap()
            .add_node(
                "node",
                Box::new(TestOperator::transform(
                    "node",
                    Action::GatePass {
                        started: Arc::clone(&started),
                        release: Arc::clone(&release),
                    },
                    Arc::new(Probe::default()),
                )) as Box<dyn BatchOperator>,
            )
            .unwrap()
            .compile_batch(&UdfRegistry::new().snapshot())
            .unwrap(),
    );
    let first = tokio::spawn({
        let plan = Arc::clone(&plan);
        async move {
            plan.execute(
                BTreeMap::from([("input".into(), support::int_batch(&[1]))]),
                ExecutionOptions::default(),
            )
            .await
        }
    });
    started.notified().await;
    let queued = tokio::spawn({
        let plan = Arc::clone(&plan);
        async move {
            plan.execute(
                BTreeMap::from([("input".into(), support::int_batch(&[2]))]),
                ExecutionOptions::default(),
            )
            .await
        }
    });
    tokio::task::yield_now().await;

    let (source, _) = QueueSource::new(Vec::new());
    let runner = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(source),
        SinkRouter::new(),
        Arc::new(MemoryCheckpointStore::default()),
        1,
    )
    .unwrap();
    release.notify_one();
    first.await.unwrap().unwrap();
    let error = tokio::time::timeout(std::time::Duration::from_millis(100), queued)
        .await
        .expect("queued direct call must recheck the lease")
        .unwrap()
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::PlanLeased { .. }));
    drop(runner);
}

#[tokio::test]
async fn reentrant_source_sink_and_store_plan_calls_fail_fast() {
    let plan = stateful_plan("reentrant callbacks", Arc::new(Probe::default()));
    let source_observation = Arc::new(Mutex::new(None));
    let sink_observation = Arc::new(Mutex::new(None));
    let store_observation = Arc::new(Mutex::new(None));
    let store = Arc::new(ReentrantCheckpointStore {
        plan: Arc::clone(&plan),
        checkpoint: Mutex::new(None),
        observation: Arc::clone(&store_observation),
    });
    let mut sinks = SinkRouter::new();
    sinks
        .add(
            "output",
            Box::new(ReentrantSink {
                plan: Arc::clone(&plan),
                observation: Arc::clone(&sink_observation),
            }),
        )
        .unwrap();
    let (source, _) = QueueSource::new(vec![source_item(&[1], json!(1), 1)]);
    let mut runner = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(ReentrantSource {
            plan,
            source,
            observation: Arc::clone(&source_observation),
        }),
        sinks,
        Arc::clone(&store) as Arc<dyn CheckpointStore>,
        1,
    )
    .unwrap();

    tokio::time::timeout(std::time::Duration::from_millis(300), runner.next())
        .await
        .expect("reentrant callbacks must not self-deadlock")
        .unwrap()
        .unwrap();
    assert!(
        source_observation
            .lock()
            .unwrap()
            .as_deref()
            .unwrap()
            .contains("leased")
    );
    assert!(
        sink_observation
            .lock()
            .unwrap()
            .as_deref()
            .unwrap()
            .contains("leased")
    );
    assert!(
        store_observation
            .lock()
            .unwrap()
            .as_deref()
            .unwrap()
            .contains("leased")
    );
}

#[tokio::test]
async fn runner_invalid_input_does_not_snapshot_restore_or_poison() {
    let probe = Arc::new(Probe::default());
    let plan = Arc::new(
        PipelineBuilder::new("invalid runner input")
            .unwrap()
            .add_node(
                "node",
                Box::new(
                    TestOperator::transform("node", Action::Pass, Arc::clone(&probe))
                        .stateful()
                        .failing_restore(),
                ) as Box<dyn BatchOperator>,
            )
            .unwrap()
            .compile_batch(&UdfRegistry::new().snapshot())
            .unwrap(),
    );
    let (source, _) = QueueSource::new(vec![SourceItem {
        batch: support::string_batch(&["wrong schema"]),
        cursor: Some(json!(1)),
        sequence: 1,
    }]);
    let mut runner = MicroBatchRunner::new(
        plan,
        Box::new(source),
        SinkRouter::new(),
        Arc::new(MemoryCheckpointStore::default()),
        1,
    )
    .unwrap();

    for _ in 0..2 {
        assert!(matches!(
            runner.next().await,
            Err(CalcFlowError::Compile { message }) if message.contains("schema mismatch")
        ));
    }
    assert_eq!(probe.calls(), 0);
    assert_eq!(probe.snapshots(), 0);
    assert_eq!(probe.restores(), 0);
}

#[tokio::test]
async fn transient_batching_read_error_is_retryable_through_the_runner() {
    let probe = Arc::new(Probe::default());
    let plan = stateful_plan("batching retry", Arc::clone(&probe));
    let source = BatchingSource::new(
        TransientCandidateSource {
            first: Some(source_item(&[1], json!(1), 1)),
            second: Some(source_item(&[2], json!(2), 2)),
            calls: 0,
        },
        10,
        usize::MAX,
    )
    .unwrap();
    let store = Arc::new(MemoryCheckpointStore::default());
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
        Err(CalcFlowError::Format { message }) if message == "transient candidate read"
    ));
    let retried = runner.next().await.unwrap().unwrap();
    assert_eq!(retried.outputs["output"].num_rows(), 2);
    assert_eq!(probe.calls(), 1);
    let checkpoint = store.checkpoint().unwrap();
    assert_eq!(checkpoint.sequence, 2);
    assert_eq!(checkpoint.source_cursor, Some(json!(2)));
}

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
    assert_eq!(runner.plan_snapshot().await.unwrap(), initial);
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
    assert_eq!(runner.plan_snapshot().await.unwrap(), initial);
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
                ) as Box<dyn BatchOperator>,
            )
            .unwrap()
            .compile_batch(&UdfRegistry::new().snapshot())
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
    assert_eq!(runner.plan_snapshot().await.unwrap(), initial);
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
    let probe = Arc::new(Probe::default());
    let plan = Arc::new(
        PipelineBuilder::new("rollback diagnostics")
            .unwrap()
            .add_node(
                "node",
                Box::new(
                    TestOperator::transform("node", Action::Pass, Arc::clone(&probe))
                        .stateful()
                        .failing_restore(),
                ) as Box<dyn BatchOperator>,
            )
            .unwrap()
            .compile_batch(&UdfRegistry::new().snapshot())
            .unwrap(),
    );
    let (source, _) = QueueSource::new(vec![
        source_item(&[1], json!(1), 1),
        source_item(&[2], json!(2), 2),
    ]);
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

    let poisoned = runner.next().await.unwrap_err().to_string();
    assert!(poisoned.contains("poisoned"));
    assert_eq!(probe.calls(), 1);

    runner.reset().await.unwrap();
    assert!(runner.next().await.unwrap().is_some());
    assert_eq!(probe.calls(), 2);
}

#[tokio::test]
async fn reset_restores_checkpoint_when_delete_errors_after_removal() {
    let plan = stateful_plan("durable reset", Arc::new(Probe::default()));
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
    let checkpoint = store.checkpoint().unwrap();
    let state = runner.plan_snapshot().await.unwrap();
    store.fail_next_deletes_after_remove(1);

    let error = runner.reset().await.unwrap_err().to_string();

    assert!(error.contains("delete after removal injected"));
    assert_eq!(store.checkpoint(), Some(checkpoint));
    assert_eq!(runner.plan_snapshot().await.unwrap(), state);
    assert!(runner.next().await.unwrap().is_none());
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
    let before = runner.plan_snapshot().await.unwrap();

    let error = runner.reset().await.unwrap_err().to_string();

    assert!(error.contains("reset injected"));
    assert_eq!(runner.plan_snapshot().await.unwrap(), before);
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
                )) as Box<dyn BatchOperator>,
            )
            .unwrap()
            .add_node(
                "two",
                Box::new(TestOperator::transform(
                    "two",
                    Action::Pass,
                    Arc::new(Probe::default()),
                )) as Box<dyn BatchOperator>,
            )
            .unwrap()
            .compile_batch(&UdfRegistry::new().snapshot())
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

struct TransientCandidateSource {
    first: Option<SourceItem>,
    second: Option<SourceItem>,
    calls: usize,
}

#[async_trait]
impl Source for TransientCandidateSource {
    async fn open(&mut self, _cursor: Option<serde_json::Value>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceItem>> {
        self.calls += 1;
        match self.calls {
            1 => Ok(self.first.take()),
            2 => Err(CalcFlowError::Format {
                message: "transient candidate read".into(),
            }),
            3 => Ok(self.second.take()),
            _ => Ok(None),
        }
    }
}

struct ReentrantSink {
    plan: Arc<calc_flow::BatchExecutionPlan>,
    observation: Arc<Mutex<Option<String>>>,
}

struct ReentrantSource {
    plan: Arc<calc_flow::BatchExecutionPlan>,
    source: QueueSource,
    observation: Arc<Mutex<Option<String>>>,
}

#[async_trait]
impl Source for ReentrantSource {
    async fn open(&mut self, cursor: Option<serde_json::Value>) -> Result<()> {
        let message =
            match tokio::time::timeout(std::time::Duration::from_millis(100), self.plan.reset())
                .await
            {
                Ok(Err(error)) => error.to_string(),
                Ok(Ok(())) => "unexpected success".into(),
                Err(_) => "timeout".into(),
            };
        *self.observation.lock().unwrap() = Some(message);
        self.source.open(cursor).await
    }

    async fn next(&mut self) -> Result<Option<SourceItem>> {
        self.source.next().await
    }
}

#[async_trait]
impl Sink for ReentrantSink {
    async fn write(&mut self, _batch: &Batch, _context: &RunContext) -> Result<()> {
        let message =
            match tokio::time::timeout(std::time::Duration::from_millis(100), self.plan.snapshot())
                .await
            {
                Ok(Err(error)) => error.to_string(),
                Ok(Ok(_)) => "unexpected success".into(),
                Err(_) => "timeout".into(),
            };
        *self.observation.lock().unwrap() = Some(message);
        Ok(())
    }
}

struct ReentrantCheckpointStore {
    plan: Arc<calc_flow::BatchExecutionPlan>,
    checkpoint: Mutex<Option<Checkpoint>>,
    observation: Arc<Mutex<Option<String>>>,
}

#[async_trait]
impl CheckpointStore for ReentrantCheckpointStore {
    async fn load(&self, _pipeline_name: &str) -> Result<Option<Checkpoint>> {
        Ok(self.checkpoint.lock().unwrap().clone())
    }

    async fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        let message =
            match tokio::time::timeout(std::time::Duration::from_millis(100), self.plan.snapshot())
                .await
            {
                Ok(Err(error)) => error.to_string(),
                Ok(Ok(_)) => "unexpected success".into(),
                Err(_) => "timeout".into(),
            };
        *self.observation.lock().unwrap() = Some(message);
        *self.checkpoint.lock().unwrap() = Some(checkpoint.clone());
        Ok(())
    }

    async fn delete(&self, _pipeline_name: &str) -> Result<()> {
        *self.checkpoint.lock().unwrap() = None;
        Ok(())
    }
}
