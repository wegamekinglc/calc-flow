mod asof_tests;
mod sql_recovery_tests;

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    future::Future as _,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    task::Poll,
    time::{Duration as StdDuration, Instant as StdInstant},
};

use async_trait::async_trait;
use chrono::TimeZone;
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
use parking_lot::Mutex;
use sha2::{Digest as _, Sha256};
use tokio::sync::{Notify, Semaphore, mpsc};

use super::{
    ABANDONED_RUNNER_WARNING, CheckpointCoordinatorHandle, CheckpointFailureCategory,
    CheckpointPhase, CheckpointRuntimeSpec, ContinuousJobState, ContinuousRunner, DriverCompletion,
    DriverOwnership, FailureOrigin as RuntimeFailureOrigin, JobCore, LaunchId,
    OneShotContinuousRunner, OneShotStartObserver, RunnerCore, RunnerDiagnostics,
    RunnerRegistryState, RunnerShutdownObserver, RuntimeFailure, RuntimeTaskProgress,
    TerminalCause, classify_failure_state, finish_running_report,
    maybe_request_terminal_checkpoint, notify_sink_abort, notify_sink_manifest_durable,
    sanitize_managed_preflight_error, settle_durable_manifest, source_cuts_are_terminal,
};
use crate::{
    AggregateFunction, AggregateSpec, Batch, BatchKind, BatchMetadata, CalcFlowError,
    CancellationToken, CheckpointManifestFields, CursorManifestEntry, Edge, EdgeBudget, EventTime,
    ExpressionOperator, JoinStateLimits, JoinTimeBounds, JsonMap, LocalStateBackend,
    ManifestIngressState, OperatorIngressManifestEntry, OperatorManifestEntry, OperatorMetadata,
    PipelineBuilder, Port, PortEndpoint, RecoveryStatus, Result, SinkDeliveryManifest,
    SinkManifestEntry, SourceManifestEntry, SourceWatermarkManifestState, StateBackend,
    StateHandle, StateLineageBackend, StateLineageKey, StreamCollector, StreamJobContext,
    StreamJoinOperator, StreamJoinSpec, StreamOperator, StreamOperatorContext, StreamRequirements,
    StreamRuntimeConfig, UdfRegistry, UnionOperator, WindowAggregateOperator, WindowSpec,
    runtime::streaming::{
        checkpoint::ManagedCheckpointRuntime,
        checkpoint::coordinator::{
            CheckpointAck, CheckpointEvent, CheckpointPhase as CoordinatorPhase, CheckpointRequest,
            ParticipantSet, spawn_checkpoint_coordinator,
        },
        job::{
            ContinuousJobSpec, M2DeliveryMode, NamedSinkBinding, NamedSourceBinding,
            OrdinarySinkBinding, OrdinaryStreamSink, TransactionalStreamSink,
            ValidatedOrdinarySink,
        },
        metrics::MetricsRecorder,
        progress::DurableSourceCut,
        sink_task::SinkCheckpointCommand,
        source_task::{Cursor, SourceBinding, SourceCapabilities, SourceEvent, StreamSource},
        supervisor::{SupervisionReport, TaskId},
    },
};

// Managed checkpoint tests perform real file publication and fsync work;
// allow scheduler and antivirus jitter while still bounding deadlocks.
const FILESYSTEM_SETTLEMENT_TIMEOUT: StdDuration = StdDuration::from_secs(5);

#[test]
fn one_shot_runner_start_has_a_consuming_signature() {
    let start: fn(OneShotContinuousRunner, ContinuousJobSpec) -> OneShotStartObserver =
        OneShotContinuousRunner::start;

    let _ = start;
}

#[test]
fn one_shot_runner_reuse_ui_is_a_move_error() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir.join("tests/ui/one_shot_runner_reuse.rs");
    let output_dir = manifest_dir.join("../../target/ui-tests");
    std::fs::create_dir_all(&output_dir).unwrap();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = std::process::Command::new(rustc)
        .arg("--edition=2024")
        .arg("--crate-name=one_shot_runner_reuse")
        .arg(&fixture)
        .arg("--out-dir")
        .arg(output_dir)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success(), "fixture unexpectedly compiled");
    assert!(stderr.contains("error[E0382]"), "{stderr}");
}

#[test]
fn one_shot_checkpointed_start_has_a_consuming_signature() {
    let start: fn(
        OneShotContinuousRunner,
        ContinuousJobSpec,
        ManagedCheckpointRuntime,
    ) -> OneShotStartObserver = OneShotContinuousRunner::start_checkpointed;

    let _ = start;
}

#[test]
fn managed_manifest_preflight_preserves_cancellation() {
    let error = sanitize_managed_preflight_error(
        CalcFlowError::Cancelled {
            run_id: "credential-secret-run".into(),
        },
        true,
        true,
    );

    assert!(matches!(
        error,
        CalcFlowError::Cancelled { ref run_id } if run_id == "managed-checkpoint-open"
    ));
}

#[tokio::test]
async fn managed_checkpoint_identity_mismatch_is_redacted_before_lifecycle_work() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("credential-secret-checkpoint-root");
    let opened = ManagedCheckpointRuntime::new(&root)
        .unwrap()
        .open(&CancellationToken::new())
        .await
        .unwrap();
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let resets = Arc::new(AtomicUsize::new(0));
    let job_spec = spec(false, Arc::clone(&resets), source.clone(), sink.clone());
    let config = StreamRuntimeConfig::default();
    let manifest = crate::CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: "credential-secret-foreign-job".into(),
        pipeline_fingerprint: job_spec.plan.fingerprint().into(),
        runtime_config_hash: job_spec.plan.runtime_config_hash(&config).unwrap(),
        epoch: crate::Epoch::INITIAL,
        created_at: chrono::Utc.with_ymd_and_hms(2026, 8, 12, 8, 0, 0).unwrap(),
        recovery_status: RecoveryStatus::Final,
        sources: BTreeMap::from([(
            "input".into(),
            SourceManifestEntry {
                history: None,
                cursor: None,
                identity_hash: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                    .into(),
                sequence: 0,
                ended: false,
                watermark_policy: SourceWatermarkManifestState::Disabled { idle: false },
            },
        )]),
        operators: BTreeMap::from([(
            "node".into(),
            OperatorManifestEntry {
                progress: BTreeMap::from([(
                    "input".into(),
                    OperatorIngressManifestEntry {
                        state: ManifestIngressState::Active,
                        watermark: None,
                    },
                )]),
                inline_metadata: BTreeMap::new(),
                segments: Vec::new(),
            },
        )]),
        sinks: BTreeMap::from([(
            "sink".into(),
            SinkManifestEntry {
                delivery: SinkDeliveryManifest::Ordinary,
                pre_commit: None,
                segments: Vec::new(),
            },
        )]),
        static_inputs: BTreeMap::new(),
    })
    .unwrap();
    std::fs::write(
        opened
            .manifest_root_for_test()
            .join("manifest-00000000000000000001.json"),
        manifest.canonical_bytes().unwrap(),
    )
    .unwrap();
    drop(opened);

    let failure = OneShotContinuousRunner::new()
        .start_checkpointed(job_spec, ManagedCheckpointRuntime::new(&root).unwrap())
        .await
        .unwrap_err();

    for rendered in [format!("{failure:?}"), format!("{failure:#?}")] {
        assert!(!rendered.contains("credential-secret"));
        assert!(!rendered.contains(&root.display().to_string()));
    }
    assert!(matches!(
        failure.primary.error,
        CalcFlowError::CheckpointMismatch { .. }
    ));
    assert_eq!(resets.load(Ordering::SeqCst), 0);
    assert_eq!(source.opened.load(Ordering::SeqCst), 0);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the redaction canaries and lifecycle assertions form one recovery scenario"
)]
async fn managed_checkpoint_missing_state_is_redacted_before_lifecycle_work() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("credential-secret-checkpoint-root");
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let resets = Arc::new(AtomicUsize::new(0));
    let job_spec = spec(false, Arc::clone(&resets), source.clone(), sink.clone());
    let config = StreamRuntimeConfig::default();
    let prepared = crate::runtime::streaming::progress::prepare_stream_job(
            job_spec.plan.fingerprint(),
            &[crate::runtime::streaming::progress::SourceBindingSpec {
                descriptor: crate::runtime::streaming::progress::SourceDescriptor::new(
                    crate::runtime::streaming::progress::BindingIdentity::new("input").unwrap(),
                    crate::runtime::streaming::progress::DeclaredSchema::DynamicOrUnknown,
                    crate::runtime::streaming::progress::NativeWatermarkCapability::EmitsNative,
                    crate::runtime::streaming::progress::ReplayPositioningCapability::ExactPauseReportAndSeek,
                    None,
                )
                .with_delivery_and_bounds(true, 1, 1),
                watermark_policy:
                    crate::runtime::streaming::progress::WatermarkPolicy::SourceProvided,
            }],
            crate::runtime::streaming::progress::StreamProgressRuntimeConfig::default(),
        )
        .unwrap();
    let identity_hash = prepared.bindings[0].identity_hash();
    let key = StateLineageKey::new(job_spec.plan.name(), job_spec.plan.fingerprint()).unwrap();
    let missing_segment = local_state_handle(
        &key,
        "node",
        crate::Epoch::INITIAL,
        "credential-secret-segment",
        b"credential-secret-state",
    );
    let checksum = missing_segment.sha256().to_owned();
    let manifest = crate::CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: job_spec.plan.name().into(),
        pipeline_fingerprint: job_spec.plan.fingerprint().into(),
        runtime_config_hash: job_spec.plan.runtime_config_hash(&config).unwrap(),
        epoch: crate::Epoch::INITIAL,
        created_at: chrono::Utc.with_ymd_and_hms(2026, 8, 12, 8, 0, 0).unwrap(),
        recovery_status: RecoveryStatus::Final,
        sources: BTreeMap::from([(
            "input".into(),
            SourceManifestEntry {
                history: None,
                cursor: Some(CursorManifestEntry {
                    order: "09".into(),
                    payload: BTreeMap::from([(
                        "credential-secret-cursor".into(),
                        serde_json::json!("credential-secret-payload"),
                    )]),
                }),
                identity_hash: identity_hash.clone(),
                sequence: 1,
                ended: false,
                watermark_policy: SourceWatermarkManifestState::SourceProvided {
                    last_emitted_micros: None,
                    idle: false,
                },
            },
        )]),
        operators: BTreeMap::from([(
            "node".into(),
            OperatorManifestEntry {
                progress: BTreeMap::from([(
                    "input".into(),
                    OperatorIngressManifestEntry {
                        state: ManifestIngressState::Active,
                        watermark: None,
                    },
                )]),
                inline_metadata:
                    crate::pipeline::OperatorCheckpointCapability::CheckpointedStateful {
                        state_version: 1,
                    }
                    .encode_snapshot("node", crate::OperatorStateSnapshot::default())
                    .unwrap()
                    .inline_metadata,
                segments: vec![missing_segment],
            },
        )]),
        sinks: BTreeMap::from([(
            "sink".into(),
            SinkManifestEntry {
                delivery: SinkDeliveryManifest::Ordinary,
                pre_commit: None,
                segments: Vec::new(),
            },
        )]),
        static_inputs: BTreeMap::new(),
    })
    .unwrap();
    let backend = LocalStateBackend::new(root.join("state")).await.unwrap();
    let lineage = backend.open_lineage(&key).await.unwrap();
    lineage
        .stage_segment(
            &manifest.operators()["node"].segments[0],
            b"credential-secret-state",
        )
        .await
        .unwrap();
    lineage
        .validate_segment(&manifest.operators()["node"].segments[0])
        .await
        .unwrap();
    lineage
        .publish_segment(&manifest.operators()["node"].segments[0])
        .await
        .unwrap();
    let transaction = crate::state::ManifestTransaction::open(
        Arc::from(lineage),
        &key,
        root.join("manifests"),
        config.retained_epochs,
    )
    .await
    .unwrap();
    transaction
        .publish(crate::state::PreparedEpochManifest {
            manifest,
            staged_segments: BTreeMap::new(),
        })
        .await
        .unwrap();
    drop(transaction);

    let failure_path = format!(
        "{}/credential-secret-segment/{checksum}/{identity_hash}/credential-secret-cursor/credential-secret-pre-commit",
        root.display()
    );
    let load_count = Arc::new(AtomicUsize::new(0));
    let checkpoint = CheckpointRuntimeSpec::managed_test_parts(
        Arc::new(FailAfterValidationBackend {
            inner: backend,
            load_count: Arc::clone(&load_count),
            failure_path: failure_path.into(),
        }),
        root.join("manifests"),
        config,
    )
    .unwrap();
    let mut runner = ContinuousRunner::new();
    let failure = runner
        .start_checkpointed(job_spec, checkpoint)
        .await
        .unwrap_err();
    runner.shutdown().await.unwrap();

    assert!(
        matches!(
            failure.primary.error,
            CalcFlowError::Internal { ref message }
                if message == "managed checkpoint recovery failed"
        ),
        "unexpected failure: {failure:#?}"
    );
    let canaries = [
        "credential-secret",
        root.to_str().unwrap(),
        checksum.as_str(),
        identity_hash.as_str(),
    ];
    let mut current: &(dyn std::error::Error + 'static) = &failure.primary.error;
    loop {
        let rendered = format!("{current} {current:?}");
        for canary in canaries {
            assert!(!rendered.contains(canary), "leaked {canary:?}: {rendered}");
        }
        let Some(source) = current.source() else {
            break;
        };
        current = source;
    }
    assert_eq!(resets.load(Ordering::SeqCst), 0);
    assert_eq!(source.opened.load(Ordering::SeqCst), 0);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
    // Recovery loads each referenced segment exactly once: the selection
    // revalidation (AC-08) and the operator state load. The session no
    // longer re-reads the same committed bytes again at retention.
    assert_eq!(load_count.load(Ordering::SeqCst), 2);
}

fn assert_job_status_json_is_allowlisted(encoded: &serde_json::Value) {
    let keys = encoded
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        keys,
        BTreeSet::from([
            "checkpoint",
            "delivery",
            "edges",
            "job_id",
            "metrics_overflowed",
            "operators",
            "sinks",
            "sources",
            "state",
            "task_count",
            "task_errors",
            "terminal_cause",
            "watermark",
        ])
    );
    let encoded = encoded.to_string();
    for forbidden in [
        "cursor",
        "payload",
        "tasks",
        "progress",
        "abandoned_runner_drops",
        "latest_observed_order",
        "durable_order",
    ] {
        assert!(!encoded.contains(forbidden), "leaked field {forbidden:?}");
    }
}

#[tokio::test]
async fn owning_job_status_is_allowlisted_stably_ordered_and_observe_only() {
    let alpha_sink = LifecycleProbe::default();
    let zeta_sink = LifecycleProbe::default();
    let mut job_spec = spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        LifecycleProbe::default(),
        LifecycleProbe::default(),
    );
    job_spec.sinks = vec![
        NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "zeta".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(zeta_sink))),
        },
        NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "alpha".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(alpha_sink))),
        },
    ];
    let job = OneShotContinuousRunner::new()
        .start(job_spec)
        .await
        .unwrap();

    let first = job.status();
    let repeated = job.status();

    assert_eq!(first, repeated);
    assert_eq!(
        first.state,
        crate::runtime::streaming::projection::JobState::Running
    );
    assert_eq!(first.job_id, job.id());
    assert_eq!(
        first.delivery["output"].requested,
        crate::DeliveryGuarantee::AtLeastOnce
    );
    assert_eq!(
        first.delivery["output"].effective,
        crate::DeliveryGuarantee::AtLeastOnce
    );
    assert_eq!(
        first.sources["input"].replay_positioning,
        crate::continuous::ReplayPositioning::ExactPauseReportAndSeek
    );
    assert_eq!(first.sources["input"].max_batch_rows, 1);
    assert_eq!(first.sources["input"].max_batch_bytes, 1);
    assert_eq!(first.edges.values().next().unwrap().envelope_limit, 1);
    assert_eq!(first.edges.values().next().unwrap().row_limit, 1);
    assert_eq!(first.edges.values().next().unwrap().byte_limit, 1);
    assert_eq!(
        first.sinks.keys().cloned().collect::<Vec<_>>(),
        vec!["alpha".to_owned(), "zeta".to_owned()]
    );

    let encoded = serde_json::to_value(&first).unwrap();
    assert_job_status_json_is_allowlisted(&encoded);

    let outcome = job.cancel().await;
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    let terminal = job.status();
    assert_eq!(
        terminal.state,
        crate::runtime::streaming::projection::JobState::Cancelled
    );
    assert_eq!(
        terminal.terminal_cause,
        Some(crate::runtime::streaming::projection::TerminalCause::ExplicitCancel)
    );
}

#[tokio::test]
async fn owning_job_status_projects_the_aggregate_watermark() {
    let mut job_spec = spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        LifecycleProbe::default(),
        LifecycleProbe::default(),
    );
    job_spec.sources[0].binding = SourceBinding::new(
        Box::new(FiniteSource {
            events: VecDeque::from([SourceEvent::Watermark(EventTime::from_micros(17))]),
            closed: Arc::new(AtomicUsize::new(0)),
        }),
        None,
        0,
    )
    .unwrap();
    job_spec.edge_budget.max_bytes = 1 << 20;
    let job = OneShotContinuousRunner::new()
        .start(job_spec)
        .await
        .unwrap();

    let outcome = job.wait().await;

    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    assert_eq!(job.status().watermark, Some(EventTime::from_micros(17)));
}

#[tokio::test]
async fn owning_job_status_remains_safe_during_a_concurrent_lifecycle_transition() {
    const CURSOR_SENTINEL: &str = "private-cursor-order-redaction-sentinel";
    const PAYLOAD_SENTINEL: &str = "private-connector-payload-redaction-sentinel";
    const PATH_SENTINEL: &str = "/srv/private/checkpoints/customer-42";

    let source = LifecycleProbe::default();
    let cursor = Cursor::new(
        "input",
        CURSOR_SENTINEL.as_bytes().to_vec(),
        BTreeMap::from([
            ("connection".into(), serde_json::json!(PAYLOAD_SENTINEL)),
            ("path".into(), serde_json::json!(PATH_SENTINEL)),
        ]),
    )
    .unwrap();
    let mut job_spec = spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        source.clone(),
        LifecycleProbe::default(),
    );
    job_spec.sources[0].binding =
        SourceBinding::new(Box::new(ProbeSource(source)), Some(cursor), 19).unwrap();
    let job = OneShotContinuousRunner::new()
        .start(job_spec)
        .await
        .unwrap();

    let observe = async {
        loop {
            let status = job.status();
            assert!(status.delivery.keys().is_sorted());
            assert!(status.edges.keys().is_sorted());
            assert!(status.sources.keys().is_sorted());
            assert!(status.operators.keys().is_sorted());
            assert!(status.sinks.keys().is_sorted());
            match status.state {
                crate::runtime::streaming::projection::JobState::Running
                | crate::runtime::streaming::projection::JobState::Draining => {
                    assert_eq!(status.terminal_cause, None);
                }
                crate::runtime::streaming::projection::JobState::Completed
                | crate::runtime::streaming::projection::JobState::Cancelled
                | crate::runtime::streaming::projection::JobState::Failed
                | crate::runtime::streaming::projection::JobState::RecoveryRequired => {
                    assert!(status.terminal_cause.is_some());
                    assert_eq!(status.task_count, 0);
                }
            }
            let encoded = serde_json::to_string(&status).unwrap();
            for sentinel in [CURSOR_SENTINEL, PAYLOAD_SENTINEL, PATH_SENTINEL] {
                assert!(!encoded.contains(sentinel), "leaked sentinel {sentinel:?}");
            }
            if status.state == crate::runtime::streaming::projection::JobState::Cancelled {
                break;
            }
            tokio::task::yield_now().await;
        }
    };
    let cancel = job.cancel();

    let ((), outcome) = tokio::join!(observe, cancel);

    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(job.status().sources["input"].next_sequence, Some(19));
}

#[tokio::test]
async fn owning_job_waiters_observe_one_terminal_without_owning_cancellation() {
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let job = OneShotContinuousRunner::new()
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap();
    let runner = job.runner_probe_for_test();

    let mut dropped_waiter = Box::pin(job.wait());
    assert!(matches!(
        futures::poll!(dropped_waiter.as_mut()),
        Poll::Pending
    ));
    drop(dropped_waiter);
    assert_eq!(job.state(), ContinuousJobState::Running);

    let mut dropped_shutdown = Box::pin(job.shutdown());
    assert!(matches!(
        futures::poll!(dropped_shutdown.as_mut()),
        Poll::Pending
    ));
    drop(dropped_shutdown);
    assert_eq!(job.state(), ContinuousJobState::Draining);

    let first = job.cancel();
    let second = job.cancel();
    let (first, second) = tokio::join!(first, second);

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(first.state, ContinuousJobState::Cancelled);
    assert_eq!(first.cause, TerminalCause::ExplicitCancel);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    assert!(job.owner_settled_for_test());
    assert_eq!(runner.registry_counts(), (0, 0));
    assert!(runner.is_finished());
}

#[tokio::test]
async fn one_shot_start_failure_waits_for_begun_resource_cleanup() {
    let source = LifecycleProbe::default();
    source.fail_open.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();

    let failure = OneShotContinuousRunner::new()
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap_err();

    assert!(matches!(
        failure.primary.origin,
        super::FailureOrigin::SourceOpen { .. }
    ));
    assert_eq!(source.opened.load(Ordering::SeqCst), 1);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    assert!(failure.cleanup_failures.is_empty());
}

#[tokio::test]
async fn one_shot_start_failure_surfaces_runner_lifecycle_join_failure() {
    let source = LifecycleProbe::default();
    source.fail_open.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    let runner = OneShotContinuousRunner::new();
    runner.panic_lifecycle_after_shutdown_for_test();

    let failure = runner
        .start(spec(false, Arc::new(AtomicUsize::new(0)), source, sink))
        .await
        .unwrap_err();

    assert!(matches!(
        failure.cleanup_failures.as_slice(),
        [failure]
            if matches!(failure.origin, super::FailureOrigin::RunnerLifecycle)
                && matches!(failure.error, CalcFlowError::Internal { .. })
    ));
}

#[tokio::test]
async fn one_shot_cleanup_observer_does_not_own_or_cancel_job() {
    let source = LifecycleProbe::default();
    source.block_close.store(true, Ordering::SeqCst);
    let runner = OneShotContinuousRunner::new();
    let mut cleanup = Box::pin(runner.cleanup_observer());
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();
    let lifecycle = job.runner_probe_for_test();

    assert!(futures::poll!(cleanup.as_mut()).is_pending());
    assert_eq!(job.state(), ContinuousJobState::Running);
    let close_started = source.close_started.notified();
    drop(job);
    close_started.await;
    assert!(futures::poll!(cleanup.as_mut()).is_pending());
    assert!(!lifecycle.is_finished());

    source.close_release.notify_waiters();
    cleanup.await.unwrap();
    assert_eq!(lifecycle.registry_counts(), (0, 0));
    assert!(lifecycle.is_finished());
}

#[tokio::test]
async fn owning_job_drop_cancels_while_an_existing_waiter_only_observes() {
    let source = LifecycleProbe::default();
    source.block_close.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    let job = OneShotContinuousRunner::new()
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap();
    let runner = job.runner_probe_for_test();
    let mut waiter = Box::pin(job.wait());
    let close_started = source.close_started.notified();

    drop(job);

    close_started.await;
    assert!(futures::poll!(waiter.as_mut()).is_pending());
    assert_eq!(runner.registry_counts().0, 1);
    assert!(!runner.is_finished());

    source.close_release.notify_waiters();
    let outcome = waiter.await;
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    assert_eq!(runner.registry_counts(), (0, 0));
    assert!(runner.is_finished());
}

#[tokio::test]
async fn owning_job_natural_terminal_reaps_runner_without_a_waiter() {
    let source = LifecycleProbe::default();
    source.finite.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    let job = OneShotContinuousRunner::new()
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap();
    let runner = job.runner_probe_for_test();

    runner.join().await.unwrap();

    assert_eq!(job.state(), ContinuousJobState::Completed);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    assert_eq!(runner.registry_counts(), (0, 0));
    assert!(runner.is_finished());
    let outcome = job.wait().await;
    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    assert!(job.owner_settled_for_test());
}

#[tokio::test]
async fn owning_job_contains_task_panic_before_publishing_terminal() {
    let source = LifecycleProbe::default();
    source.panic_next.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    let job = OneShotContinuousRunner::new()
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap();

    let outcome = job.wait().await;

    assert_eq!(outcome.state, ContinuousJobState::Failed);
    assert!(matches!(
        outcome.errors[0].error,
        CalcFlowError::TaskPanicked { ref message, .. }
            if message == "source next panicked"
    ));
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    assert!(job.owner_settled_for_test());
}

#[tokio::test]
async fn owning_job_surfaces_runner_lifecycle_join_failure() {
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let runner = OneShotContinuousRunner::new();
    runner.panic_lifecycle_after_shutdown_for_test();
    let job = runner
        .start(spec(false, Arc::new(AtomicUsize::new(0)), source, sink))
        .await
        .unwrap();
    let runner = job.runner_probe_for_test();

    let outcome = job.cancel().await;

    assert!(matches!(
        outcome.errors.last(),
        Some(failure)
            if matches!(failure.origin, super::FailureOrigin::RunnerLifecycle)
                && matches!(failure.error, CalcFlowError::Internal { .. })
    ));
    assert_eq!(runner.registry_counts(), (0, 0));
    assert!(runner.is_finished());
}

#[tokio::test]
async fn checkpointed_owning_job_releases_transaction_and_lineage_lease() {
    let directory = tempfile::tempdir().unwrap();
    let plan = PipelineBuilder::new("one-shot-checkpoint-lease")
        .unwrap()
        .add_checkpoint_capable_node(
            "node",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_opened = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            9_901,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(0, &source_polls, &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(CheckpointProbeSink {
                opened: Arc::clone(&sink_opened),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let checkpoint = ManagedCheckpointRuntime::new(directory.path()).unwrap();
    let job = OneShotContinuousRunner::new()
        .start_checkpointed(spec, checkpoint)
        .await
        .unwrap();
    wait_for_counter(&source_polls, 1).await;

    let outcome = job.cancel().await;

    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_opened.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    ManagedCheckpointRuntime::new(directory.path())
        .unwrap()
        .open(&CancellationToken::new())
        .await
        .unwrap();
}

struct ResetOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
    resets: Arc<AtomicUsize>,
    fail_reset: bool,
    panic_reset: bool,
}

struct BlockingEntryOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
    entered: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
}

struct EntryDataProbeOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
    resets: Arc<AtomicUsize>,
    processed: Arc<AtomicUsize>,
}

struct OrderedEntryFailureOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
    node_id: &'static str,
    later_node_returned: Arc<AtomicBool>,
}

impl OperatorMetadata for OrderedEntryFailureOperator {
    fn name(&self) -> &'static str {
        "ordered-entry-failure"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl StreamOperator for OrderedEntryFailureOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        _batch: Batch,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        unreachable!("entry failure prevents data execution")
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        if self.node_id == "a" {
            while !self.later_node_returned.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
        } else {
            self.later_node_returned.store(true, Ordering::SeqCst);
        }
        Err(CalcFlowError::Operator {
            node_id: self.node_id.into(),
            message: format!("{} reset failed", self.node_id),
        })
    }
}

struct JobIdentityProbeOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
    observed_job_id: Arc<Mutex<Option<u64>>>,
}

struct ActiveCancelledOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
}

impl OperatorMetadata for ActiveCancelledOperator {
    fn name(&self) -> &'static str {
        "active-cancelled"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl StreamOperator for ActiveCancelledOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        _batch: Batch,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Err(CalcFlowError::Cancelled {
            run_id: "operator-active-cancelled".into(),
        })
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

impl OperatorMetadata for JobIdentityProbeOperator {
    fn name(&self) -> &'static str {
        "job-identity-probe"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl StreamOperator for JobIdentityProbeOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        *self.observed_job_id.lock() = Some(context.job().job_id());
        output.emit("output", batch).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

impl OperatorMetadata for EntryDataProbeOperator {
    fn name(&self) -> &'static str {
        "entry-data-probe"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl StreamOperator for EntryDataProbeOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        batch: Batch,
        _context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.processed.fetch_add(1, Ordering::SeqCst);
        output.emit("output", batch).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.resets.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

impl OperatorMetadata for ResetOperator {
    fn name(&self) -> &'static str {
        "reset-probe"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }
}

#[async_trait]
impl StreamOperator for ResetOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        _batch: Batch,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Err(CalcFlowError::Operator {
            node_id: "node".into(),
            message: "data failure".into(),
        })
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.resets.fetch_add(1, Ordering::SeqCst);
        assert!(!self.panic_reset, "operator reset panicked");
        if self.fail_reset {
            Err(CalcFlowError::Operator {
                node_id: "node".into(),
                message: "reset failed".into(),
            })
        } else {
            Ok(())
        }
    }
}

impl OperatorMetadata for BlockingEntryOperator {
    fn name(&self) -> &'static str {
        "blocking-entry"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl StreamOperator for BlockingEntryOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        batch: Batch,
        _context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        output.emit("output", batch).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.entered.store(true, Ordering::SeqCst);
        while !self.release.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
struct LifecycleProbe {
    opened: Arc<AtomicUsize>,
    open_completed: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
    open_started: Arc<Notify>,
    open_release: Arc<Notify>,
    close_started: Arc<Notify>,
    close_release: Arc<Notify>,
    block_open: Arc<AtomicBool>,
    block_close: Arc<AtomicBool>,
    fail_open: Arc<AtomicBool>,
    panic_open: Arc<AtomicBool>,
    panic_next: Arc<AtomicBool>,
    panic_write: Arc<AtomicBool>,
    fail_close: Arc<AtomicBool>,
    panic_close: Arc<AtomicBool>,
    finite: Arc<AtomicBool>,
}

struct ProbeSource(LifecycleProbe);

#[async_trait]
impl StreamSource for ProbeSource {
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.0.opened.fetch_add(1, Ordering::SeqCst);
        self.0.open_started.notify_waiters();
        if self.0.block_open.load(Ordering::SeqCst) {
            self.0.open_release.notified().await;
        }
        assert!(
            !self.0.panic_open.load(Ordering::SeqCst),
            "source open panicked"
        );
        if self.0.fail_open.load(Ordering::SeqCst) {
            Err(CalcFlowError::Internal {
                message: "source open failed".into(),
            })
        } else {
            self.0.open_completed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        assert!(
            !self.0.panic_next.load(Ordering::SeqCst),
            "source next panicked"
        );
        if self.0.finite.load(Ordering::SeqCst) {
            Ok(None)
        } else {
            std::future::pending().await
        }
    }

    async fn close(&mut self) -> Result<()> {
        self.0.closed.fetch_add(1, Ordering::SeqCst);
        self.0.close_started.notify_waiters();
        if self.0.block_close.load(Ordering::SeqCst) {
            self.0.close_release.notified().await;
        }
        if self.0.fail_close.load(Ordering::SeqCst) {
            Err(CalcFlowError::Internal {
                message: "source close failed".into(),
            })
        } else {
            Ok(())
        }
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1,
        }
    }
}

struct ProbeSink(LifecycleProbe);

#[async_trait]
impl OrdinaryStreamSink for ProbeSink {
    async fn open(&mut self) -> Result<()> {
        self.0.opened.fetch_add(1, Ordering::SeqCst);
        self.0.open_started.notify_waiters();
        if self.0.block_open.load(Ordering::SeqCst) {
            self.0.open_release.notified().await;
        }
        assert!(
            !self.0.panic_open.load(Ordering::SeqCst),
            "sink open panicked"
        );
        if self.0.fail_open.load(Ordering::SeqCst) {
            Err(CalcFlowError::Internal {
                message: "sink open failed".into(),
            })
        } else {
            self.0.open_completed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        assert!(
            !self.0.panic_write.load(Ordering::SeqCst),
            "sink write panicked"
        );
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.0.closed.fetch_add(1, Ordering::SeqCst);
        self.0.close_started.notify_waiters();
        if self.0.block_close.load(Ordering::SeqCst) {
            self.0.close_release.notified().await;
        }
        assert!(
            !self.0.panic_close.load(Ordering::SeqCst),
            "sink close panicked"
        );
        if self.0.fail_close.load(Ordering::SeqCst) {
            Err(CalcFlowError::Internal {
                message: "sink close failed".into(),
            })
        } else {
            Ok(())
        }
    }
}

struct FiniteSource {
    events: VecDeque<SourceEvent>,
    closed: Arc<AtomicUsize>,
}

struct ResumeProbeSource {
    opened_with: Arc<Mutex<Vec<Option<Cursor>>>>,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl StreamSource for ResumeProbeSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.opened_with.lock().push(cursor);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(None)
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

struct GatedDataSource {
    release: Arc<Notify>,
    delivered: bool,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl StreamSource for GatedDataSource {
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.delivered {
            return std::future::pending().await;
        }
        self.release.notified().await;
        self.delivered = true;
        Ok(Some(SourceEvent::Data {
            batch: one_row(1),
            cursor: Cursor::unbound(vec![1], JsonMap::new()).unwrap(),
        }))
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

#[async_trait]
impl StreamSource for FiniteSource {
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(self.events.pop_front())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

struct OrderedRecordingSink {
    id: String,
    writes: Arc<Mutex<Vec<(String, String, u64)>>>,
    closed: Arc<AtomicUsize>,
}

#[derive(Default)]
struct MixedDeliveryTransactionalState {
    committed_epochs: BTreeSet<u64>,
    visible: Vec<(String, u64)>,
}

#[derive(Default)]
struct MixedDeliveryProbes {
    source_closed: Arc<AtomicUsize>,
    transactional_closed: Arc<AtomicUsize>,
    ordinary_closed: Arc<AtomicUsize>,
    transactional: Arc<Mutex<MixedDeliveryTransactionalState>>,
    ordinary_writes: Arc<Mutex<Vec<(String, String, u64)>>>,
}

struct MixedDeliveryTransactionalSink {
    pending: Vec<(String, u64)>,
    state: Arc<Mutex<MixedDeliveryTransactionalState>>,
    closed: Arc<AtomicUsize>,
}

struct CountingPendingSource {
    events: VecDeque<SourceEvent>,
    polls: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
}

struct CheckpointProbeSink {
    opened: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
}

#[derive(Clone)]
struct FailOnceRetentionBackend {
    inner: LocalStateBackend,
    failure_armed: Arc<AtomicBool>,
}

struct FailOnceRetentionLineage {
    inner: Box<dyn StateLineageBackend>,
    failure_armed: Arc<AtomicBool>,
}

#[derive(Clone)]
struct FailAfterValidationBackend {
    inner: LocalStateBackend,
    load_count: Arc<AtomicUsize>,
    failure_path: Arc<str>,
}

struct FailAfterValidationLineage {
    inner: Box<dyn StateLineageBackend>,
    load_count: Arc<AtomicUsize>,
    failure_path: Arc<str>,
}

#[async_trait]
impl StateBackend for FailOnceRetentionBackend {
    async fn open_lineage(&self, key: &StateLineageKey) -> Result<Box<dyn StateLineageBackend>> {
        Ok(Box::new(FailOnceRetentionLineage {
            inner: self.inner.open_lineage(key).await?,
            failure_armed: Arc::clone(&self.failure_armed),
        }))
    }
}

#[async_trait]
impl StateBackend for FailAfterValidationBackend {
    async fn open_lineage(&self, key: &StateLineageKey) -> Result<Box<dyn StateLineageBackend>> {
        Ok(Box::new(FailAfterValidationLineage {
            inner: self.inner.open_lineage(key).await?,
            load_count: Arc::clone(&self.load_count),
            failure_path: Arc::clone(&self.failure_path),
        }))
    }
}

#[async_trait]
impl StateLineageBackend for FailOnceRetentionLineage {
    fn identity_hash(&self) -> &str {
        self.inner.identity_hash()
    }

    async fn stage_segment(&self, handle: &StateHandle, bytes: &[u8]) -> Result<()> {
        self.inner.stage_segment(handle, bytes).await
    }

    async fn validate_segment(&self, handle: &StateHandle) -> Result<()> {
        self.inner.validate_segment(handle).await
    }

    async fn publish_segment(&self, handle: &StateHandle) -> Result<()> {
        self.inner.publish_segment(handle).await
    }

    async fn load_segment(&self, handle: &StateHandle) -> Result<Vec<u8>> {
        self.inner.load_segment(handle).await
    }

    async fn collect_orphans(&self, retained: &[StateHandle]) -> Result<usize> {
        if self.failure_armed.swap(false, Ordering::SeqCst) {
            return Err(CalcFlowError::Internal {
                message: "injected retention failure".into(),
            });
        }
        self.inner.collect_orphans(retained).await
    }
}

#[async_trait]
impl StateLineageBackend for FailAfterValidationLineage {
    fn identity_hash(&self) -> &str {
        self.inner.identity_hash()
    }

    async fn stage_segment(&self, handle: &StateHandle, bytes: &[u8]) -> Result<()> {
        self.inner.stage_segment(handle, bytes).await
    }

    async fn validate_segment(&self, handle: &StateHandle) -> Result<()> {
        self.inner.validate_segment(handle).await
    }

    async fn publish_segment(&self, handle: &StateHandle) -> Result<()> {
        self.inner.publish_segment(handle).await
    }

    async fn load_segment(&self, handle: &StateHandle) -> Result<Vec<u8>> {
        if self.load_count.fetch_add(1, Ordering::SeqCst) >= 1 {
            return Err(CalcFlowError::Io {
                path: self.failure_path.to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "credential-secret-I/O-source",
                ),
            });
        }
        self.inner.load_segment(handle).await
    }

    async fn collect_orphans(&self, retained: &[StateHandle]) -> Result<usize> {
        self.inner.collect_orphans(retained).await
    }
}

struct RecoveryProbeOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
    log: Arc<Mutex<Vec<String>>>,
}

impl OperatorMetadata for RecoveryProbeOperator {
    fn name(&self) -> &'static str {
        "recovery-probe"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl StreamOperator for RecoveryProbeOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        _batch: Batch,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.log.lock().push("operator-reset".into());
        Ok(())
    }

    fn restore(&mut self, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
        assert_eq!(
            snapshot.inline_metadata["restored"],
            serde_json::json!(true)
        );
        self.log.lock().push("operator-restore".into());
        Ok(())
    }
}

struct RecoveryProbeSource {
    log: Arc<Mutex<Vec<String>>>,
    closed: Arc<AtomicUsize>,
}

struct MixedRestoreSource {
    opens: Arc<AtomicUsize>,
    seeks: Arc<AtomicUsize>,
    polls: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl StreamSource for MixedRestoreSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        if cursor.is_some() {
            self.seeks.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        std::future::pending().await
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }

    fn native_watermark_capability(
        &self,
    ) -> crate::runtime::streaming::progress::NativeWatermarkCapability {
        crate::runtime::streaming::progress::NativeWatermarkCapability::NeverEmits
    }
}

#[async_trait]
impl StreamSource for RecoveryProbeSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        let order = cursor
            .as_ref()
            .map_or_else(|| "none".into(), |cursor| hex::encode(cursor.order()));
        self.log.lock().push(format!("source-open:{order}"));
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        std::future::pending().await
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }

    fn native_watermark_capability(
        &self,
    ) -> crate::runtime::streaming::progress::NativeWatermarkCapability {
        crate::runtime::streaming::progress::NativeWatermarkCapability::NeverEmits
    }
}

struct RecoveryProbeSink {
    log: Arc<Mutex<Vec<String>>>,
    closed: Arc<AtomicUsize>,
}

struct PeriodicCheckpointSink {
    log: Arc<Mutex<Vec<String>>>,
    closed: Arc<AtomicUsize>,
}

struct FailOnceCommitSink {
    fail_commit: Arc<AtomicBool>,
    log: Arc<Mutex<Vec<String>>>,
    closed: Arc<AtomicUsize>,
}

struct BlockingCommitSink {
    commit_entered: Arc<AtomicBool>,
    commit_changed: Arc<Notify>,
    commit_release: Arc<Semaphore>,
    log: Arc<Mutex<Vec<String>>>,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl TransactionalStreamSink for PeriodicCheckpointSink {
    async fn open(&mut self) -> Result<()> {
        self.log.lock().push("sink-open".into());
        Ok(())
    }

    async fn begin_epoch(&mut self, epoch: crate::Epoch) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-begin:{}", epoch.as_u64()));
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn pre_commit(&mut self, epoch: crate::Epoch) -> Result<JsonMap> {
        self.log
            .lock()
            .push(format!("sink-precommit:{}", epoch.as_u64()));
        Ok(BTreeMap::from([(
            "epoch".into(),
            serde_json::json!(epoch.as_u64()),
        )]))
    }

    async fn commit(&mut self, epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-commit:{}", epoch.as_u64()));
        Ok(())
    }

    async fn abort(&mut self, epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-abort:{}", epoch.as_u64()));
        Ok(())
    }

    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-recover:{}", manifest.epoch().as_u64()));
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        self.log.lock().push("sink-close".into());
        Ok(())
    }
}

#[async_trait]
impl TransactionalStreamSink for FailOnceCommitSink {
    async fn open(&mut self) -> Result<()> {
        self.log.lock().push("sink-open".into());
        Ok(())
    }

    async fn begin_epoch(&mut self, epoch: crate::Epoch) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-begin:{}", epoch.as_u64()));
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn pre_commit(&mut self, epoch: crate::Epoch) -> Result<JsonMap> {
        self.log
            .lock()
            .push(format!("sink-precommit:{}", epoch.as_u64()));
        Ok(BTreeMap::from([(
            "epoch".into(),
            serde_json::json!(epoch.as_u64()),
        )]))
    }

    async fn commit(&mut self, epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-commit:{}", epoch.as_u64()));
        if self.fail_commit.swap(false, Ordering::SeqCst) {
            return Err(CalcFlowError::Internal {
                message: "injected post-manifest commit failure".into(),
            });
        }
        Ok(())
    }

    async fn abort(&mut self, epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-abort:{}", epoch.as_u64()));
        Ok(())
    }

    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-recover:{}", manifest.epoch().as_u64()));
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        self.log.lock().push("sink-close".into());
        Ok(())
    }
}

#[async_trait]
impl TransactionalStreamSink for BlockingCommitSink {
    async fn open(&mut self) -> Result<()> {
        self.log.lock().push("sink-open".into());
        Ok(())
    }

    async fn begin_epoch(&mut self, epoch: crate::Epoch) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-begin:{}", epoch.as_u64()));
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn pre_commit(&mut self, epoch: crate::Epoch) -> Result<JsonMap> {
        self.log
            .lock()
            .push(format!("sink-precommit:{}", epoch.as_u64()));
        Ok(BTreeMap::from([(
            "epoch".into(),
            serde_json::json!(epoch.as_u64()),
        )]))
    }

    async fn commit(&mut self, epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
        self.commit_entered.store(true, Ordering::Release);
        self.commit_changed.notify_waiters();
        self.commit_release
            .acquire()
            .await
            .expect("test commit gate remains open")
            .forget();
        self.log
            .lock()
            .push(format!("sink-commit:{}", epoch.as_u64()));
        Ok(())
    }

    async fn abort(&mut self, epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-abort:{}", epoch.as_u64()));
        Ok(())
    }

    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-recover:{}", manifest.epoch().as_u64()));
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        self.log.lock().push("sink-close".into());
        Ok(())
    }
}

#[async_trait]
impl TransactionalStreamSink for RecoveryProbeSink {
    async fn open(&mut self) -> Result<()> {
        self.log.lock().push("sink-open".into());
        Ok(())
    }

    async fn begin_epoch(&mut self, epoch: crate::Epoch) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-begin:{}", epoch.as_u64()));
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
        Ok(JsonMap::new())
    }

    async fn commit(&mut self, _epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
        Ok(())
    }

    async fn abort(&mut self, _epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
        Ok(())
    }

    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        self.log
            .lock()
            .push(format!("sink-recover:{}", manifest.epoch().as_u64()));
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[async_trait]
impl TransactionalStreamSink for CheckpointProbeSink {
    async fn open(&mut self) -> Result<()> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
        Ok(JsonMap::new())
    }

    async fn commit(&mut self, _epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
        Ok(())
    }

    async fn abort(&mut self, _epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
        Ok(())
    }

    async fn recover(&mut self, _manifest: &crate::CheckpointManifest) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct ZeroCostLifecycleSource {
    events: VecDeque<SourceEvent>,
    polls: Arc<AtomicUsize>,
    poll_calls: mpsc::UnboundedSender<usize>,
    eof_observed: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
    end: ZeroCostSourceEnd,
}

#[derive(Clone, Copy)]
enum ZeroCostSourceEnd {
    Eof,
    Pending,
    Error,
}

struct ZeroCostLifecycleSink {
    gate: Arc<Semaphore>,
    writes: Arc<Mutex<Vec<u64>>>,
    closed: Arc<AtomicUsize>,
}

#[derive(Clone, Copy)]
enum RunningSourceFailure {
    Next,
    Cursor,
}

struct PrimaryAndCloseFailingSource {
    failure: RunningSourceFailure,
    next_call: usize,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl StreamSource for PrimaryAndCloseFailingSource {
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let call = self.next_call;
        self.next_call += 1;
        match (self.failure, call) {
            (RunningSourceFailure::Next, _) => Err(CalcFlowError::Internal {
                message: "source-next-primary".into(),
            }),
            (RunningSourceFailure::Cursor, 0 | 1) => Ok(Some(SourceEvent::Data {
                batch: one_row(i64::try_from(call).unwrap()),
                cursor: Cursor::unbound(vec![1], JsonMap::new()).unwrap(),
            })),
            (RunningSourceFailure::Cursor, _) => std::future::pending().await,
        }
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Err(CalcFlowError::Internal {
            message: "source-close-secondary".into(),
        })
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

#[async_trait]
impl StreamSource for CountingPendingSource {
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        match self.events.pop_front() {
            Some(event) => Ok(Some(event)),
            None => std::future::pending().await,
        }
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

#[async_trait]
impl StreamSource for ZeroCostLifecycleSource {
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let poll = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(event) = self.events.pop_front() {
            let _ = self.poll_calls.send(poll);
            return Ok(Some(event));
        }
        match std::mem::replace(&mut self.end, ZeroCostSourceEnd::Pending) {
            ZeroCostSourceEnd::Eof => {
                self.eof_observed.fetch_add(1, Ordering::SeqCst);
                let _ = self.poll_calls.send(poll);
                Ok(None)
            }
            ZeroCostSourceEnd::Error => {
                let _ = self.poll_calls.send(poll);
                Err(CalcFlowError::Internal {
                    message: "zero-cost lifecycle source failed".into(),
                })
            }
            ZeroCostSourceEnd::Pending => {
                let _ = self.poll_calls.send(poll);
                std::future::pending().await
            }
        }
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

#[async_trait]
impl OrdinaryStreamSink for ZeroCostLifecycleSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        let permit = self
            .gate
            .acquire()
            .await
            .map_err(|_| CalcFlowError::Internal {
                message: "zero-cost lifecycle sink gate closed".into(),
            })?;
        permit.forget();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.estimated_bytes()?, 0);
        self.writes.lock().push(batch.metadata().sequence());
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct GatedSink {
    started: Arc<AtomicBool>,
    gate: Arc<Notify>,
    writes: Arc<Mutex<Vec<(String, u64)>>>,
    closed: Arc<AtomicUsize>,
}

struct DeadlinePendingOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
    entered: Arc<AtomicBool>,
}

impl OperatorMetadata for DeadlinePendingOperator {
    fn name(&self) -> &'static str {
        "deadline-pending"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl StreamOperator for DeadlinePendingOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        _batch: Batch,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.entered.store(true, Ordering::SeqCst);
        std::future::pending().await
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum StressGate {
    LeftData0,
    LeftData1,
    LeftEof,
    RightData0,
    RightData1,
    RightEof,
    Edge0,
    Edge1,
    Edge2,
    Edge3,
    SinkA0,
    SinkA1,
    SinkA2,
    SinkA3,
    SinkB0,
    SinkB1,
    SinkB2,
    SinkB3,
    Natural,
    Drain,
    Cancel,
}

fn stress_schedule(seed: u64) -> Vec<StressGate> {
    let mut gates = vec![
        StressGate::LeftData0,
        StressGate::LeftData1,
        StressGate::LeftEof,
        StressGate::RightData0,
        StressGate::RightData1,
        StressGate::RightEof,
        StressGate::Edge0,
        StressGate::Edge1,
        StressGate::Edge2,
        StressGate::Edge3,
        StressGate::SinkA0,
        StressGate::SinkA1,
        StressGate::SinkA2,
        StressGate::SinkA3,
        StressGate::SinkB0,
        StressGate::SinkB1,
        StressGate::SinkB2,
        StressGate::SinkB3,
        match seed % 3 {
            0 => StressGate::Natural,
            1 => StressGate::Drain,
            _ => StressGate::Cancel,
        },
    ];
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    for index in (1..gates.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let swap = usize::try_from(state % u64::try_from(index + 1).unwrap()).unwrap();
        gates.swap(index, swap);
    }
    if seed % 3 != 0 {
        let terminal = gates
            .iter()
            .position(|gate| matches!(gate, StressGate::Drain | StressGate::Cancel))
            .unwrap();
        let first_eof = gates
            .iter()
            .position(|gate| matches!(gate, StressGate::LeftEof | StressGate::RightEof))
            .unwrap();
        if terminal > first_eof {
            gates.swap(terminal, first_eof);
        }
    }
    gates
}

struct StressSource {
    events: VecDeque<(Arc<Semaphore>, Option<SourceEvent>)>,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl StreamSource for StressSource {
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let Some((gate, event)) = self.events.pop_front() else {
            return std::future::pending().await;
        };
        let permit = gate.acquire().await.map_err(|_| CalcFlowError::Internal {
            message: "stress source gate closed".into(),
        })?;
        permit.forget();
        Ok(event)
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

struct StressForwardOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
    gates: Option<Arc<Mutex<VecDeque<Arc<Semaphore>>>>>,
    fail_watermark: Option<Arc<AtomicBool>>,
}

impl StressForwardOperator {
    fn new(gates: Option<Vec<Arc<Semaphore>>>) -> Self {
        Self {
            inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
            gates: gates.map(|gates| Arc::new(Mutex::new(gates.into()))),
            fail_watermark: None,
        }
    }

    fn failing_on_watermark(flag: Arc<AtomicBool>) -> Self {
        Self {
            fail_watermark: Some(flag),
            ..Self::new(None)
        }
    }
}

impl OperatorMetadata for StressForwardOperator {
    fn name(&self) -> &'static str {
        "stress-forward"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl StreamOperator for StressForwardOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        batch: Batch,
        _context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let gate = self
            .gates
            .as_ref()
            .and_then(|gates| gates.lock().pop_front());
        if let Some(gate) = gate {
            let permit = gate.acquire().await.map_err(|_| CalcFlowError::Internal {
                message: "stress edge gate closed".into(),
            })?;
            permit.forget();
        }
        output.emit("output", batch).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if self
            .fail_watermark
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst))
        {
            return Err(CalcFlowError::Operator {
                node_id: "node".into(),
                message: "zero-cost lifecycle receiver failed".into(),
            });
        }
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

struct StressSink {
    gates: VecDeque<Arc<Semaphore>>,
    writes: Arc<Mutex<Vec<(String, u64)>>>,
    zero_cost_writes: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl OrdinaryStreamSink for StressSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        let gate = self
            .gates
            .pop_front()
            .ok_or_else(|| CalcFlowError::Internal {
                message: "stress sink exhausted its gates".into(),
            })?;
        let permit = gate.acquire().await.map_err(|_| CalcFlowError::Internal {
            message: "stress sink gate closed".into(),
        })?;
        permit.forget();
        if batch.num_rows() == 0 && batch.estimated_bytes()? == 0 {
            self.zero_cost_writes.fetch_add(1, Ordering::SeqCst);
        }
        self.writes.lock().push((
            batch.metadata().source().into(),
            batch.metadata().sequence(),
        ));
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[async_trait]
impl OrdinaryStreamSink for GatedSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.started.store(true, Ordering::SeqCst);
        self.gate.notified().await;
        self.writes.lock().push((
            batch.metadata().source().into(),
            batch.metadata().sequence(),
        ));
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[async_trait]
impl OrdinaryStreamSink for OrderedRecordingSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.writes.lock().push((
            self.id.clone(),
            batch.metadata().source().into(),
            batch.metadata().sequence(),
        ));
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn mixed_delivery_records(state: &JsonMap) -> Result<Vec<(String, u64)>> {
    serde_json::from_value(state.get("records").cloned().ok_or_else(|| {
        CalcFlowError::CheckpointMismatch {
            message: "mixed-delivery pre-commit records are missing".into(),
        }
    })?)
    .map_err(|error| CalcFlowError::CheckpointMismatch {
        message: format!("mixed-delivery pre-commit records are invalid: {error}"),
    })
}

#[async_trait]
impl TransactionalStreamSink for MixedDeliveryTransactionalSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
        self.pending.clear();
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.pending.push((
            batch.metadata().source().into(),
            batch.metadata().sequence(),
        ));
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
        Ok(BTreeMap::from([(
            "records".into(),
            serde_json::json!(self.pending),
        )]))
    }

    async fn commit(&mut self, epoch: crate::Epoch, state: &JsonMap) -> Result<()> {
        let records = mixed_delivery_records(state)?;
        let mut durable = self.state.lock();
        if durable.committed_epochs.insert(epoch.as_u64()) {
            durable.visible.extend(records);
        }
        Ok(())
    }

    async fn abort(&mut self, _epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
        self.pending.clear();
        Ok(())
    }

    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        let state = manifest
            .sinks()
            .get("transactional")
            .and_then(|entry| entry.pre_commit.clone())
            .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                message: "mixed-delivery transactional recovery state is missing".into(),
            })?;
        self.commit(manifest.epoch(), &state).await
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn one_row(value: i64) -> Batch {
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(vec![value])) as _,
    )])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn zero_row() -> Batch {
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(Vec::<i64>::new())) as _,
    )])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn mixed_delivery_fault_spec(job_id: u64, probes: &MixedDeliveryProbes) -> ContinuousJobSpec {
    let plan = PipelineBuilder::new("checkpoint-mixed-delivery-fault")
        .unwrap()
        .add_checkpoint_capable_node(
            "root",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .add_checkpoint_capable_node(
            "exact",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .add_checkpoint_capable_node(
            "ordinary",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("root", "output").unwrap(),
            PortEndpoint::new("exact", "input").unwrap(),
        ))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("root", "output").unwrap(),
            PortEndpoint::new("ordinary", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "exact.output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    ContinuousJobSpec {
        context: StreamJobContext::new(
            job_id,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: finite_binding(&[1], &probes.source_closed),
        }],
        sinks: vec![
            NamedSinkBinding {
                output_id: "exact.output".into(),
                sink_id: "transactional".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(
                    MixedDeliveryTransactionalSink {
                        pending: Vec::new(),
                        state: Arc::clone(&probes.transactional),
                        closed: Arc::clone(&probes.transactional_closed),
                    },
                )),
            },
            NamedSinkBinding {
                output_id: "ordinary.output".into(),
                sink_id: "ordinary-sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(OrderedRecordingSink {
                    id: "ordinary".into(),
                    writes: Arc::clone(&probes.ordinary_writes),
                    closed: Arc::clone(&probes.ordinary_closed),
                })),
            },
        ],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

fn assert_terminal_checkpoint_resources_released(job: &super::ContinuousJob) {
    let status = job.status();
    assert!(status.tasks.is_empty());
    assert!(status.edges.values().all(|edge| {
        edge.queue_depth == 0 && edge.charged_rows == 0 && edge.charged_bytes == 0
    }));
}

fn finite_binding(values: &[i64], closed: &Arc<AtomicUsize>) -> SourceBinding {
    let events = values
        .iter()
        .enumerate()
        .map(|(index, value)| SourceEvent::Data {
            batch: one_row(*value),
            cursor: Cursor::unbound(vec![u8::try_from(index + 1).unwrap()], JsonMap::new())
                .unwrap(),
        })
        .collect();
    SourceBinding::new(
        Box::new(FiniteSource {
            events,
            closed: Arc::clone(closed),
        }),
        None,
        0,
    )
    .unwrap()
}

fn counting_pending_binding(
    count: usize,
    polls: &Arc<AtomicUsize>,
    closed: &Arc<AtomicUsize>,
) -> SourceBinding {
    let events = (0..count)
        .map(|index| SourceEvent::Data {
            batch: one_row(i64::try_from(index).unwrap()),
            cursor: Cursor::unbound(
                u64::try_from(index + 1).unwrap().to_be_bytes().to_vec(),
                JsonMap::new(),
            )
            .unwrap(),
        })
        .collect();
    SourceBinding::new(
        Box::new(CountingPendingSource {
            events,
            polls: Arc::clone(polls),
            closed: Arc::clone(closed),
        }),
        None,
        0,
    )
    .unwrap()
}

fn pending_checkpoint_spec(
    pipeline_name: &str,
    job_id: u64,
    source_polls: &Arc<AtomicUsize>,
    source_closed: &Arc<AtomicUsize>,
    sink: Box<dyn TransactionalStreamSink>,
) -> ContinuousJobSpec {
    let plan = PipelineBuilder::new(pipeline_name)
        .unwrap()
        .add_checkpoint_capable_node(
            "node",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    ContinuousJobSpec {
        context: StreamJobContext::new(
            job_id,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(0, source_polls, source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(sink),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

fn union_plan() -> crate::StreamExecutionPlan {
    let union = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap();
    PipelineBuilder::new("union-runtime")
        .unwrap()
        .add_node("merge", Box::new(union))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

#[test]
fn checkpoint_admission_covers_all_operator_capability_classes() {
    let operator = || {
        UnionOperator::new(
            "merge",
            vec![
                Port::new("left", BatchKind::Table, true, None).unwrap(),
                Port::new("right", BatchKind::Table, true, None).unwrap(),
            ],
        )
        .unwrap()
    };
    let compile = |builder: PipelineBuilder| {
        builder
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap()
            .into_runtime_parts(EdgeBudget::default())
            .unwrap()
    };
    let stateless = compile(
        PipelineBuilder::new("stateless")
            .unwrap()
            .add_node("merge", Box::new(operator()))
            .unwrap(),
    );
    let checkpointed = compile(
        PipelineBuilder::new("checkpointed")
            .unwrap()
            .add_checkpoint_capable_node("merge", Box::new(operator()) as Box<dyn StreamOperator>)
            .unwrap(),
    );
    let unproven = compile(
        PipelineBuilder::new("unproven")
            .unwrap()
            .add_node("merge", Box::new(operator()) as Box<dyn StreamOperator>)
            .unwrap(),
    );

    assert!(super::validate_checkpoint_operator_capabilities(&stateless).is_ok());
    assert!(super::validate_checkpoint_operator_capabilities(&checkpointed).is_ok());
    let error = super::validate_checkpoint_operator_capabilities(&unproven).unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { ref field, ref message }
            if field == "operators.merge.checkpoint_capability"
                && message.contains("unproven")
    ));
}

fn unary_expression_plan() -> crate::StreamExecutionPlan {
    let expression =
        ExpressionOperator::new("calc", "plus_one = value + 1", Vec::new(), None, Vec::new())
            .unwrap();
    PipelineBuilder::new("unary-runtime")
        .unwrap()
        .add_node("calc", Box::new(expression))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn two_entry_probe_plan(
    resets: &Arc<AtomicUsize>,
    processed: &Arc<AtomicUsize>,
) -> crate::StreamExecutionPlan {
    let probe = |resets: &Arc<AtomicUsize>, processed: &Arc<AtomicUsize>| {
        Box::new(EntryDataProbeOperator {
            inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Table, false, None).unwrap()],
            resets: Arc::clone(resets),
            processed: Arc::clone(processed),
        }) as Box<dyn StreamOperator>
    };
    PipelineBuilder::new("entry-data-gates")
        .unwrap()
        .add_node("first", probe(resets, processed))
        .unwrap()
        .add_node("second", probe(resets, processed))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("first", "output").unwrap(),
            PortEndpoint::new("second", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn deadline_pending_plan(entered: Arc<AtomicBool>) -> crate::StreamExecutionPlan {
    let operator = DeadlinePendingOperator {
        inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
        outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
        entered,
    };
    PipelineBuilder::new("deadline-pending")
        .unwrap()
        .add_node("pending", Box::new(operator) as Box<dyn StreamOperator>)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn spec(
    fail_reset: bool,
    resets: Arc<AtomicUsize>,
    source: LifecycleProbe,
    sink: LifecycleProbe,
) -> ContinuousJobSpec {
    reset_spec(fail_reset, false, resets, source, sink)
}

fn reset_spec(
    fail_reset: bool,
    panic_reset: bool,
    resets: Arc<AtomicUsize>,
    source: LifecycleProbe,
    sink: LifecycleProbe,
) -> ContinuousJobSpec {
    let operator = ResetOperator {
        inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
        outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
        resets,
        fail_reset,
        panic_reset,
    };
    let plan = PipelineBuilder::new("launch")
        .unwrap()
        .add_checkpoint_capable_node("node", Box::new(operator) as Box<dyn StreamOperator>)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    ContinuousJobSpec {
        context: StreamJobContext::new(
            9,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: SourceBinding::new(Box::new(ProbeSource(source)), None, 0).unwrap(),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink))),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

fn blocking_entry_spec(
    entered: &Arc<AtomicBool>,
    release: &Arc<AtomicBool>,
    source: LifecycleProbe,
    sink: LifecycleProbe,
) -> ContinuousJobSpec {
    let operator = BlockingEntryOperator {
        inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
        outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
        entered: Arc::clone(entered),
        release: Arc::clone(release),
    };
    let plan = PipelineBuilder::new("blocking-entry")
        .unwrap()
        .add_node("node", Box::new(operator) as Box<dyn StreamOperator>)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    ContinuousJobSpec {
        context: StreamJobContext::new(
            9,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: SourceBinding::new(Box::new(ProbeSource(source)), None, 0).unwrap(),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink))),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

async fn wait_for_operator_entry(entered: &AtomicBool) -> bool {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !entered.load(Ordering::SeqCst) {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    })
    .await
    .is_ok()
}

async fn wait_for_counter(counter: &AtomicUsize, expected: usize) {
    for _ in 0..100 {
        if counter.load(Ordering::SeqCst) == expected {
            return;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(counter.load(Ordering::SeqCst), expected);
}

fn local_state_handle(
    key: &StateLineageKey,
    operator_id: &str,
    epoch: crate::Epoch,
    segment_id: &str,
    bytes: &[u8],
) -> StateHandle {
    let digest = |value: &[u8]| hex::encode(Sha256::digest(value));
    let lineage_hash =
        digest(format!("{}\0{}", key.pipeline_name(), key.pipeline_fingerprint()).as_bytes());
    let operator_hash = digest(operator_id.as_bytes());
    let segment_hash = digest(segment_id.as_bytes());
    let relative_path = format!(
        "committed/{lineage_hash}/{operator_hash}/{}-{segment_hash}.segment",
        epoch.as_u64()
    );
    StateHandle::new(
        operator_id,
        epoch,
        segment_id,
        &relative_path,
        u64::try_from(bytes.len()).unwrap(),
        &digest(bytes),
    )
    .unwrap()
}

fn forward_spec(
    job_id: u64,
    source: SourceBinding,
    sinks: Vec<NamedSinkBinding>,
) -> ContinuousJobSpec {
    forward_spec_with_operator(
        job_id,
        source,
        sinks,
        Box::new(StressForwardOperator::new(None)),
    )
}

fn forward_spec_with_operator(
    job_id: u64,
    source: SourceBinding,
    sinks: Vec<NamedSinkBinding>,
    operator: Box<dyn StreamOperator>,
) -> ContinuousJobSpec {
    let plan = PipelineBuilder::new("runtime-panic-cleanup")
        .unwrap()
        .add_node("node", operator)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    ContinuousJobSpec {
        context: StreamJobContext::new(
            job_id,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: source,
        }],
        sinks,
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

fn named_probe_sink(sink_id: &str, probe: LifecycleProbe) -> NamedSinkBinding {
    NamedSinkBinding {
        output_id: "output".into(),
        sink_id: sink_id.into(),
        binding: OrdinarySinkBinding::new(Box::new(ProbeSink(probe))),
    }
}

fn stress_plan(
    gates: &BTreeMap<StressGate, Arc<Semaphore>>,
    zero_cost_gate: &Arc<Semaphore>,
    zero_cost_batches: usize,
) -> crate::StreamExecutionPlan {
    let union = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap();
    let unary_gates = std::iter::repeat_with(|| Arc::clone(zero_cost_gate))
        .take(zero_cost_batches)
        .chain(
            [
                StressGate::Edge0,
                StressGate::Edge1,
                StressGate::Edge2,
                StressGate::Edge3,
            ]
            .map(|gate| Arc::clone(&gates[&gate])),
        )
        .collect();
    PipelineBuilder::new("seeded-stress")
        .unwrap()
        .add_node("merge", Box::new(union))
        .unwrap()
        .add_node(
            "unary",
            Box::new(StressForwardOperator::new(Some(unary_gates))) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .add_node(
            "branch_a",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .add_node(
            "branch_b",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("merge", "output").unwrap(),
            PortEndpoint::new("unary", "input").unwrap(),
        ))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("unary", "output").unwrap(),
            PortEndpoint::new("branch_a", "input").unwrap(),
        ))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("unary", "output").unwrap(),
            PortEndpoint::new("branch_b", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn stress_source_binding(
    gates: &BTreeMap<StressGate, Arc<Semaphore>>,
    zero_cost_phase: Option<(&Arc<Semaphore>, usize)>,
    data_gates: [StressGate; 2],
    eof_gate: StressGate,
    values: [i64; 2],
    closed: &Arc<AtomicUsize>,
) -> SourceBinding {
    let zero_cost_count = zero_cost_phase.map_or(0, |(_, count)| count);
    let zero_cost_events = zero_cost_phase.into_iter().flat_map(|(gate, count)| {
        (0..count).map(move |index| {
            (
                Arc::clone(gate),
                Some(SourceEvent::Data {
                    batch: zero_row(),
                    cursor: Cursor::unbound(vec![u8::try_from(index + 1).unwrap()], JsonMap::new())
                        .unwrap(),
                }),
            )
        })
    });
    let events = zero_cost_events
        .chain(
            data_gates
                .into_iter()
                .zip(values)
                .enumerate()
                .map(|(index, (gate, value))| {
                    (
                        Arc::clone(&gates[&gate]),
                        Some(SourceEvent::Data {
                            batch: one_row(value),
                            cursor: Cursor::unbound(
                                vec![u8::try_from(zero_cost_count + index + 1).unwrap()],
                                JsonMap::new(),
                            )
                            .unwrap(),
                        }),
                    )
                }),
        )
        .chain(std::iter::once((Arc::clone(&gates[&eof_gate]), None)))
        .collect();
    SourceBinding::new(
        Box::new(StressSource {
            events,
            closed: Arc::clone(closed),
        }),
        None,
        0,
    )
    .unwrap()
}

fn assert_source_fifo(seed: u64, writes: &[(String, u64)]) {
    let unique = writes.iter().cloned().collect::<BTreeSet<_>>();
    assert_eq!(unique.len(), writes.len(), "duplicate at seed {seed}");
    for source in ["left", "right"] {
        let sequence = writes
            .iter()
            .filter(|(observed_source, _)| observed_source == source)
            .map(|(_, sequence)| *sequence)
            .collect::<Vec<_>>();
        assert_eq!(
            sequence,
            (0..u64::try_from(sequence.len()).unwrap()).collect::<Vec<_>>(),
            "per-source FIFO failed at seed {seed} for {source}"
        );
    }
}

async fn wait_for_stress_writes(
    primary_writes: &Mutex<Vec<(String, u64)>>,
    replica_writes: &Mutex<Vec<(String, u64)>>,
    expected: usize,
) {
    for _ in 0..100 {
        if primary_writes.lock().len() >= expected && replica_writes.lock().len() >= expected {
            return;
        }
        tokio::task::yield_now().await;
    }
}

fn assert_zero_cost_active_edges(job: &super::ContinuousJob) {
    let status = job.status();
    assert!(status.edges.values().all(|edge| {
        edge.queue_depth <= 1 && edge.charged_rows == 0 && edge.charged_bytes == 0
    }));
}

async fn run_zero_cost_stress_phase(
    job: &super::ContinuousJob,
    gates: [&Semaphore; 4],
    writes: [&Mutex<Vec<(String, u64)>>; 2],
    batches: usize,
    seed: u64,
) {
    let [
        input_gate,
        transform_gate,
        primary_sink_gate,
        replica_sink_gate,
    ] = gates;
    let [primary_writes, replica_writes] = writes;
    input_gate.add_permits(batches);
    transform_gate.add_permits(batches);
    for delivered in 1..=batches {
        primary_sink_gate.add_permits(1);
        replica_sink_gate.add_permits(1);
        wait_for_stress_writes(primary_writes, replica_writes, delivered).await;
        assert_eq!(primary_writes.lock().len(), delivered, "seed {seed}");
        assert_eq!(replica_writes.lock().len(), delivered, "seed {seed}");
        assert_zero_cost_active_edges(job);
    }
}

#[tokio::test]
async fn durable_notification_attempts_every_sink_after_one_channel_closes() {
    let (closed_sender, closed_receiver) = mpsc::channel(1);
    drop(closed_receiver);
    let (open_sender, mut open_receiver) = mpsc::channel(1);
    let senders = BTreeMap::from([
        ("a-closed".into(), closed_sender),
        ("b-open".into(), open_sender),
    ]);

    let error = notify_sink_manifest_durable(&senders, crate::Epoch::INITIAL, false)
        .await
        .unwrap_err();

    assert!(matches!(error, CalcFlowError::Internal { .. }));
    assert!(matches!(
        open_receiver.recv().await,
        Some(SinkCheckpointCommand::ManifestDurable(
            crate::Epoch::INITIAL
        ))
    ));
}

#[tokio::test]
async fn unpublished_checkpoint_notifies_prepared_sinks_to_abort() {
    let (sender, mut receiver) = mpsc::channel(1);
    let senders = BTreeMap::from([("output".into(), sender)]);
    notify_sink_abort(&senders, crate::Epoch::INITIAL).await;
    assert!(matches!(
        receiver.recv().await,
        Some(SinkCheckpointCommand::Abort(crate::Epoch::INITIAL))
    ));
}

fn durable_notification_manifest() -> crate::CheckpointManifest {
    crate::CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: "durable-notification-order".into(),
        pipeline_fingerprint: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            .into(),
        runtime_config_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
            .into(),
        epoch: crate::Epoch::INITIAL,
        created_at: chrono::Utc.with_ymd_and_hms(2026, 8, 10, 8, 0, 0).unwrap(),
        recovery_status: RecoveryStatus::Final,
        sources: BTreeMap::new(),
        operators: BTreeMap::new(),
        sinks: BTreeMap::from([(
            "sink".into(),
            SinkManifestEntry {
                delivery: SinkDeliveryManifest::Transactional,
                pre_commit: Some(BTreeMap::from([(
                    "prepared".into(),
                    serde_json::json!(true),
                )])),
                segments: Vec::new(),
            },
        )]),
        static_inputs: BTreeMap::new(),
    })
    .unwrap()
}

async fn install_durable_notification_manifest(root: &Path) -> crate::CheckpointManifest {
    let manifest = durable_notification_manifest();
    let backend = LocalStateBackend::new(root.join("state")).await.unwrap();
    let key =
        StateLineageKey::new(manifest.pipeline_name(), manifest.pipeline_fingerprint()).unwrap();
    let lineage = backend.open_lineage(&key).await.unwrap();
    let transaction = crate::state::ManifestTransaction::open(
        Arc::from(lineage),
        &key,
        root.join("manifests"),
        2,
    )
    .await
    .unwrap();
    transaction
        .publish(crate::state::PreparedEpochManifest {
            manifest: manifest.clone(),
            staged_segments: BTreeMap::new(),
        })
        .await
        .unwrap();
    assert!(
        root.join("manifests/manifest-00000000000000000001.json")
            .exists()
    );
    manifest
}

async fn stopped_manifest_coordinator(timed_out: bool) -> CheckpointCoordinatorHandle {
    let cancellation = CancellationToken::new();
    let timeout = if timed_out {
        StdDuration::from_millis(1)
    } else {
        StdDuration::from_secs(1)
    };
    let (coordinator, mut events, task) = spawn_checkpoint_coordinator(
        ParticipantSet {
            sources: BTreeSet::from(["source".into()]),
            operators: BTreeSet::from(["operator".into()]),
            sinks: BTreeSet::from(["output".into()]),
        },
        crate::Epoch::INITIAL,
        4,
        timeout,
        cancellation.clone(),
    )
    .unwrap();
    if timed_out {
        coordinator
            .request(CheckpointRequest::Periodic)
            .await
            .unwrap();
        assert!(matches!(
            events.recv().await,
            Some(CheckpointEvent::Started(crate::Epoch::INITIAL))
        ));
        task.await.unwrap().unwrap_err();
    } else {
        cancellation.cancel();
        task.await.unwrap().unwrap();
    }
    coordinator
}

async fn assert_durable_notification_precedes_failed_bookkeeping(
    coordinator: &CheckpointCoordinatorHandle,
) {
    let (sink_sender, mut sink_receiver) = mpsc::channel(1);
    let mut phase = super::DurableSettlementPhase::Published;
    let error = settle_durable_manifest(
        coordinator,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::from([("output".into(), sink_sender)]),
        super::DurableSettlementRequest {
            epoch: crate::Epoch::INITIAL,
            terminal: false,
            acknowledgement_timeout: StdDuration::from_secs(1),
            phase: &mut phase,
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Internal { .. }));
    assert_eq!(phase, super::DurableSettlementPhase::SinksCommanded);
    assert!(matches!(
        sink_receiver.try_recv(),
        Ok(SinkCheckpointCommand::ManifestDurable(
            crate::Epoch::INITIAL
        ))
    ));
}

#[test]
fn durable_settlement_rejects_skipped_phase() {
    let mut phase = super::DurableSettlementPhase::Published;
    assert!(
        phase
            .advance(
                super::DurableSettlementPhase::Published,
                super::DurableSettlementPhase::CoordinatorDurable,
                crate::Epoch::INITIAL,
            )
            .is_err()
    );
    assert_eq!(phase, super::DurableSettlementPhase::Published);
}

#[tokio::test]
async fn installed_manifest_notifies_sinks_before_failed_bookkeeping_and_recovers_forward() {
    let directory = tempfile::tempdir().unwrap();
    let manifest = install_durable_notification_manifest(directory.path()).await;
    for timed_out in [false, true] {
        let coordinator = stopped_manifest_coordinator(timed_out).await;
        assert_durable_notification_precedes_failed_bookkeeping(&coordinator).await;
    }

    let log = Arc::new(Mutex::new(Vec::new()));
    let mut sinks = vec![ValidatedOrdinarySink {
        sink_id: "sink".into(),
        binding: OrdinarySinkBinding::new_transactional(Box::new(RecoveryProbeSink {
            log: Arc::clone(&log),
            closed: Arc::new(AtomicUsize::new(0)),
        })),
    }];
    crate::runtime::streaming::sink_task::recover_transactional_sinks(&mut sinks, &manifest)
        .await
        .unwrap();
    assert_eq!(&*log.lock(), &["sink-recover:1"]);
}

#[tokio::test]
async fn checkpoint_connector_open_waits_for_every_source_before_opening_sinks() {
    let source = LifecycleProbe::default();
    source.block_open.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    let mut resources = super::connector_resources(
        BTreeMap::from([(
            "input".into(),
            SourceBinding::new(Box::new(ProbeSource(source.clone())), None, 0).unwrap(),
        )]),
        BTreeMap::from([(
            "output".into(),
            vec![ValidatedOrdinarySink {
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
            }],
        )]),
    );
    let cancellation = CancellationToken::new();
    let source_opened = source.open_started.notified();
    tokio::pin!(source_opened);
    {
        let opening = super::open_checkpoint_connector_resources(&mut resources, &cancellation);
        tokio::pin!(opening);

        tokio::select! {
            failures = &mut opening => panic!("checkpoint opens completed early: {failures:?}"),
            () = &mut source_opened => {}
        }
        assert_eq!(sink.opened.load(Ordering::SeqCst), 0);

        source.open_release.notify_waiters();
        assert!(opening.as_mut().await.is_empty());
    }
    assert_eq!(sink.opened.load(Ordering::SeqCst), 1);
    assert!(super::close_resources(&mut resources).await.is_empty());
}

#[tokio::test(start_paused = true)]
async fn blocked_connector_open_expires_and_keeps_resource_owned() {
    let source = LifecycleProbe::default();
    source.block_open.store(true, Ordering::SeqCst);
    let mut resources = super::connector_resources(
        BTreeMap::from([(
            "input".into(),
            SourceBinding::new(Box::new(ProbeSource(source.clone())), None, 0).unwrap(),
        )]),
        BTreeMap::new(),
    );
    let cancellation = CancellationToken::new();
    let opened = source.open_started.notified();
    tokio::pin!(opened);
    {
        let opening = super::open_connector_resources(&mut resources, &cancellation);
        tokio::pin!(opening);
        tokio::select! {
            failures = &mut opening => panic!("connector open completed early: {failures:?}"),
            () = &mut opened => {}
        }
        tokio::time::advance(StdDuration::from_secs(30)).await;
        let failures = tokio::time::timeout(StdDuration::from_millis(1), &mut opening)
            .await
            .expect("connector open must expire");
        assert_eq!(failures.len(), 1);
    }
    assert_eq!(resources.len(), 1);
    source.open_release.notify_waiters();
    assert!(super::close_resources(&mut resources).await.is_empty());
}

#[tokio::test(start_paused = true)]
#[allow(
    clippy::too_many_lines,
    reason = "one synthetic sink proves both completed and hung open settlement paths"
)]
async fn failed_open_waits_for_native_settlement_before_releasing_resource() {
    struct SettlingSink {
        entered: Arc<Notify>,
        settling: Arc<Notify>,
        release: Arc<Notify>,
    }

    #[async_trait]
    impl TransactionalStreamSink for SettlingSink {
        async fn open(&mut self) -> Result<()> {
            self.entered.notify_one();
            std::future::pending().await
        }

        async fn settle_open(&mut self) -> Result<()> {
            self.settling.notify_one();
            self.release.notified().await;
            Ok(())
        }

        async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
            Ok(JsonMap::new())
        }

        async fn commit(&mut self, _epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
            Ok(())
        }

        async fn abort(&mut self, _epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
            Ok(())
        }

        async fn recover(&mut self, _manifest: &crate::CheckpointManifest) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    let entered = Arc::new(Notify::new());
    let settling = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let sink = SettlingSink {
        entered: Arc::clone(&entered),
        settling: Arc::clone(&settling),
        release: Arc::clone(&release),
    };
    let mut resources = super::connector_resources(
        BTreeMap::new(),
        BTreeMap::from([(
            "output".into(),
            vec![ValidatedOrdinarySink {
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(sink)),
            }],
        )]),
    );
    let cancellation = CancellationToken::new();
    let entered_wait = entered.notified();
    tokio::pin!(entered_wait);
    {
        let opening = super::open_connector_resources(&mut resources, &cancellation);
        tokio::pin!(opening);
        tokio::select! {
            failures = &mut opening => panic!("open completed early: {failures:?}"),
            () = &mut entered_wait => {}
        }
        let settling_wait = settling.notified();
        tokio::pin!(settling_wait);
        tokio::time::advance(StdDuration::from_secs(30)).await;
        tokio::select! {
            failures = &mut opening => panic!("resource released before native settlement: {failures:?}"),
            () = &mut settling_wait => {}
        }
        release.notify_one();
        assert_eq!(opening.await.len(), 1);
    }
    assert_eq!(resources.len(), 1);
    assert!(super::close_resources(&mut resources).await.is_empty());

    let entered = Arc::new(Notify::new());
    let settling = Arc::new(Notify::new());
    let sink = SettlingSink {
        entered: Arc::clone(&entered),
        settling: Arc::clone(&settling),
        release: Arc::new(Notify::new()),
    };
    let mut resources = super::connector_resources(
        BTreeMap::new(),
        BTreeMap::from([(
            "output".into(),
            vec![ValidatedOrdinarySink {
                sink_id: "hung-settlement".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(sink)),
            }],
        )]),
    );
    let cancellation = CancellationToken::new();
    let entered_wait = entered.notified();
    tokio::pin!(entered_wait);
    {
        let opening = super::open_connector_resources(&mut resources, &cancellation);
        tokio::pin!(opening);
        tokio::select! {
            failures = &mut opening => panic!("open completed early: {failures:?}"),
            () = &mut entered_wait => {}
        }
        let settling_wait = settling.notified();
        tokio::pin!(settling_wait);
        tokio::time::advance(StdDuration::from_secs(30)).await;
        tokio::select! {
            failures = &mut opening => panic!("settlement did not begin: {failures:?}"),
            () = &mut settling_wait => {}
        }
        tokio::time::advance(StdDuration::from_secs(35)).await;
        let failures = opening.await;
        assert_eq!(failures.len(), 1);
        assert!(failures[0].error.to_string().contains("settlement"));
    }
    assert_eq!(resources.len(), 1);
    assert!(super::close_resources(&mut resources).await.is_empty());
}

#[test]
fn seeded_gate_generator_is_pure_and_permutates_every_named_gate() {
    for seed in 0..100 {
        let first = stress_schedule(seed);
        let second = stress_schedule(seed);
        assert_eq!(first, second, "non-deterministic schedule at seed {seed}");
        assert_eq!(first.len(), 19, "wrong gate count at seed {seed}");
        assert_eq!(
            first.iter().copied().collect::<BTreeSet<_>>().len(),
            19,
            "duplicate gate at seed {seed}"
        );
        assert_eq!(
            first
                .iter()
                .filter(|gate| {
                    matches!(
                        gate,
                        StressGate::Natural | StressGate::Drain | StressGate::Cancel
                    )
                })
                .count(),
            1,
            "wrong terminal gate count at seed {seed}"
        );
    }
}

#[allow(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the stress scenario keeps paired branch state and all seeded invariants together"
)]
#[tokio::test]
async fn seeded_paused_time_stress_runs_one_hundred_full_graph_schedules() {
    const MESSAGE_SLOT_LIMIT: usize = 1;
    const ZERO_COST_SLOT_MULTIPLIER: usize = 10;
    const ZERO_COST_PHASE_BATCHES: usize = MESSAGE_SLOT_LIMIT * ZERO_COST_SLOT_MULTIPLIER;
    const DATA_GATES: [StressGate; 18] = [
        StressGate::LeftData0,
        StressGate::LeftData1,
        StressGate::LeftEof,
        StressGate::RightData0,
        StressGate::RightData1,
        StressGate::RightEof,
        StressGate::Edge0,
        StressGate::Edge1,
        StressGate::Edge2,
        StressGate::Edge3,
        StressGate::SinkA0,
        StressGate::SinkA1,
        StressGate::SinkA2,
        StressGate::SinkA3,
        StressGate::SinkB0,
        StressGate::SinkB1,
        StressGate::SinkB2,
        StressGate::SinkB3,
    ];
    for seed in 0..100 {
        run_seeded_zero_cost_shape_phase(seed, MESSAGE_SLOT_LIMIT, ZERO_COST_SLOT_MULTIPLIER).await;
        let gates = DATA_GATES
            .into_iter()
            .map(|gate| (gate, Arc::new(Semaphore::new(0))))
            .collect::<BTreeMap<_, _>>();
        let zero_cost_source_gate = Arc::new(Semaphore::new(0));
        let zero_cost_operator_gate = Arc::new(Semaphore::new(0));
        let zero_cost_sink_a_gate = Arc::new(Semaphore::new(0));
        let zero_cost_sink_b_gate = Arc::new(Semaphore::new(0));
        let plan = stress_plan(&gates, &zero_cost_operator_gate, ZERO_COST_PHASE_BATCHES);
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let sink_a_writes = Arc::new(Mutex::new(Vec::new()));
        let sink_b_writes = Arc::new(Mutex::new(Vec::new()));
        let sink_a_zero_cost_writes = Arc::new(AtomicUsize::new(0));
        let sink_b_zero_cost_writes = Arc::new(AtomicUsize::new(0));
        let sink_a_gates = std::iter::repeat_with(|| Arc::clone(&zero_cost_sink_a_gate))
            .take(ZERO_COST_PHASE_BATCHES)
            .chain(
                [
                    StressGate::SinkA0,
                    StressGate::SinkA1,
                    StressGate::SinkA2,
                    StressGate::SinkA3,
                ]
                .map(|gate| Arc::clone(&gates[&gate])),
            );
        let sink_b_gates = std::iter::repeat_with(|| Arc::clone(&zero_cost_sink_b_gate))
            .take(ZERO_COST_PHASE_BATCHES)
            .chain(
                [
                    StressGate::SinkB0,
                    StressGate::SinkB1,
                    StressGate::SinkB2,
                    StressGate::SinkB3,
                ]
                .map(|gate| Arc::clone(&gates[&gate])),
            );
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                10_000 + seed,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![
                NamedSourceBinding {
                    binding_id: "left".into(),
                    binding: stress_source_binding(
                        &gates,
                        Some((&zero_cost_source_gate, ZERO_COST_PHASE_BATCHES)),
                        [StressGate::LeftData0, StressGate::LeftData1],
                        StressGate::LeftEof,
                        [1, 2],
                        &source_closed,
                    ),
                },
                NamedSourceBinding {
                    binding_id: "right".into(),
                    binding: stress_source_binding(
                        &gates,
                        None,
                        [StressGate::RightData0, StressGate::RightData1],
                        StressGate::RightEof,
                        [10, 20],
                        &source_closed,
                    ),
                },
            ],
            sinks: vec![
                NamedSinkBinding {
                    output_id: "branch_a.output".into(),
                    sink_id: "slow-a".into(),
                    binding: OrdinarySinkBinding::new(Box::new(StressSink {
                        gates: sink_a_gates.collect(),
                        writes: Arc::clone(&sink_a_writes),
                        zero_cost_writes: Arc::clone(&sink_a_zero_cost_writes),
                        closed: Arc::clone(&sink_closed),
                    })),
                },
                NamedSinkBinding {
                    output_id: "branch_b.output".into(),
                    sink_id: "slow-b".into(),
                    binding: OrdinarySinkBinding::new(Box::new(StressSink {
                        gates: sink_b_gates.collect(),
                        writes: Arc::clone(&sink_b_writes),
                        zero_cost_writes: Arc::clone(&sink_b_zero_cost_writes),
                        closed: Arc::clone(&sink_closed),
                    })),
                },
            ],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
            static_inputs: crate::static_input::PreparedStaticInputs::default(),
        };
        let mut runner = ContinuousRunner::new();
        let job = runner
            .start(spec)
            .await
            .unwrap_or_else(|failure| panic!("start failed at seed {seed}: {failure:?}"));
        run_zero_cost_stress_phase(
            &job,
            [
                &zero_cost_source_gate,
                &zero_cost_operator_gate,
                &zero_cost_sink_a_gate,
                &zero_cost_sink_b_gate,
            ],
            [&sink_a_writes, &sink_b_writes],
            ZERO_COST_PHASE_BATCHES,
            seed,
        )
        .await;
        assert_eq!(
            sink_a_zero_cost_writes.load(Ordering::SeqCst),
            ZERO_COST_PHASE_BATCHES,
            "seed {seed}"
        );
        assert_eq!(
            sink_b_zero_cost_writes.load(Ordering::SeqCst),
            ZERO_COST_PHASE_BATCHES,
            "seed {seed}"
        );
        let schedule = stress_schedule(seed);
        let mut terminal = None;
        for gate in schedule {
            match gate {
                StressGate::Natural => {}
                StressGate::Drain => terminal = Some(job.shutdown()),
                StressGate::Cancel => terminal = Some(job.cancel()),
                gate => gates[&gate].add_permits(1),
            }
            tokio::task::yield_now().await;
            let status = job.status();
            assert!(status.tasks.len() <= 11, "task growth at seed {seed}");
            assert!(
                status.edges.values().all(|edge| {
                    edge.queue_depth <= 1
                        && edge.charged_rows <= 1
                        && edge.charged_bytes <= (1 << 20)
                }),
                "edge budget breach at seed {seed}: {:?}",
                status.edges
            );
        }
        let outcome = match terminal {
            Some(observer) => observer.await,
            None => job.wait().await,
        };
        let status = job.status();
        assert!(status.tasks.is_empty(), "task leak at seed {seed}");
        assert!(
            status.edges.values().all(|edge| {
                edge.queue_depth == 0 && edge.charged_rows == 0 && edge.charged_bytes == 0
            }),
            "queue leak at seed {seed}: {:?}",
            status.edges
        );
        assert!(
            status.edges.values().all(|edge| {
                edge.high_water_depth <= 1
                    && edge.high_water_rows <= 1
                    && edge.high_water_bytes <= (1 << 20)
            }),
            "high-water budget breach at seed {seed}: {:?}",
            status.edges
        );
        let a = sink_a_writes.lock().clone();
        let b = sink_b_writes.lock().clone();
        assert!(
            a.len() >= ZERO_COST_PHASE_BATCHES && b.len() >= ZERO_COST_PHASE_BATCHES,
            "zero-cost phase did not span ten slot-limit multiples at seed {seed}"
        );
        assert_source_fifo(seed, &a);
        assert_source_fifo(seed, &b);
        match seed % 3 {
            0 => {
                assert_eq!(outcome.cause, TerminalCause::NaturalEnd, "seed {seed}");
                assert_eq!(a.len(), ZERO_COST_PHASE_BATCHES + 4, "loss at seed {seed}");
                assert_eq!(b.len(), ZERO_COST_PHASE_BATCHES + 4, "loss at seed {seed}");
                assert_eq!(a, b, "fan-out divergence at seed {seed}");
                assert!(
                    status
                        .metrics
                        .edges
                        .values()
                        .all(|edge| { edge.input_batches == edge.output_batches }),
                    "edge loss at seed {seed}: {:?}",
                    status.metrics.edges
                );
            }
            1 => {
                assert_eq!(
                    outcome.cause,
                    TerminalCause::GracefulShutdown,
                    "seed {seed}"
                );
                assert_eq!(a, b, "graceful fan-out divergence at seed {seed}");
            }
            _ => assert_eq!(outcome.cause, TerminalCause::ExplicitCancel, "seed {seed}"),
        }
        assert_eq!(source_closed.load(Ordering::SeqCst), 2, "seed {seed}");
        assert_eq!(sink_closed.load(Ordering::SeqCst), 2, "seed {seed}");
        assert_eq!(runner.registry_counts(), (0, 0), "seed {seed}");
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0), "seed {seed}");
    }
}

#[derive(Clone, Copy, Debug)]
enum ZeroCostTermination {
    Graceful,
    ReceiverClose,
    Cancel,
    Error,
}

fn zero_cost_lifecycle_events(cycles: usize) -> VecDeque<SourceEvent> {
    (0..cycles)
        .flat_map(|index| {
            let watermark = i64::try_from(index + 1).unwrap();
            let cursor = u64::try_from(index + 1).unwrap().to_be_bytes().to_vec();
            [
                SourceEvent::Data {
                    batch: zero_row(),
                    cursor: Cursor::unbound(cursor, JsonMap::new()).unwrap(),
                },
                SourceEvent::Idle,
                SourceEvent::Watermark(EventTime::from_micros(watermark)),
            ]
        })
        .collect()
}

fn assert_zero_cost_shape_multiplier(
    events: &VecDeque<SourceEvent>,
    message_slot_limit: usize,
    multiplier: usize,
) {
    let required = message_slot_limit * multiplier;
    let mut idle = 0;
    let mut watermarks = Vec::new();
    let mut empty_data = 0;
    for event in events {
        match event {
            SourceEvent::Idle => idle += 1,
            SourceEvent::Watermark(watermark) => watermarks.push(watermark.as_micros()),
            SourceEvent::Data { batch, .. } => {
                assert_eq!(batch.num_rows(), 0);
                assert_eq!(batch.estimated_bytes().unwrap(), 0);
                empty_data += 1;
            }
        }
    }
    assert_eq!(idle, required);
    assert_eq!(watermarks.len(), required);
    assert!(watermarks.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(empty_data, required);
}

async fn wait_for_single_stress_writes(writes: &Mutex<Vec<(String, u64)>>, expected: usize) {
    for _ in 0..1_000 {
        if writes.lock().len() >= expected {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("seeded zero-cost sink did not reach {expected} writes")
}

async fn run_seeded_zero_cost_shape_phase(seed: u64, message_slot_limit: usize, multiplier: usize) {
    let shape_count = message_slot_limit * multiplier;
    let shape_events = zero_cost_lifecycle_events(shape_count);
    assert_zero_cost_shape_multiplier(&shape_events, message_slot_limit, multiplier);

    let source_gate = Arc::new(Semaphore::new(0));
    let sink_gate = Arc::new(Semaphore::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let writes = Arc::new(Mutex::new(Vec::new()));
    let zero_cost_writes = Arc::new(AtomicUsize::new(0));
    let mut events = shape_events
        .into_iter()
        .map(|event| (Arc::clone(&source_gate), Some(event)))
        .collect::<VecDeque<_>>();
    events.push_back((Arc::clone(&source_gate), None));
    let source = SourceBinding::new(
        Box::new(StressSource {
            events,
            closed: Arc::clone(&source_closed),
        }),
        None,
        0,
    )
    .unwrap();
    let sinks = vec![NamedSinkBinding {
        output_id: "output".into(),
        sink_id: "seeded-zero-cost".into(),
        binding: OrdinarySinkBinding::new(Box::new(StressSink {
            gates: std::iter::repeat_with(|| Arc::clone(&sink_gate))
                .take(shape_count)
                .collect(),
            writes: Arc::clone(&writes),
            zero_cost_writes: Arc::clone(&zero_cost_writes),
            closed: Arc::clone(&sink_closed),
        })),
    }];
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start(forward_spec(30_000 + seed, source, sinks))
        .await
        .unwrap_or_else(|failure| {
            panic!("zero-cost unary start failed at seed {seed}: {failure:?}")
        });

    source_gate.add_permits(shape_count * 3 + 1);
    for delivered in 1..=shape_count {
        sink_gate.add_permits(1);
        wait_for_single_stress_writes(&writes, delivered).await;
        assert_eq!(writes.lock().len(), delivered, "seed {seed}");
        assert_zero_cost_active_edges(&job);
    }
    let outcome = job.wait().await;
    assert_eq!(outcome.state, ContinuousJobState::Completed, "seed {seed}");
    assert_eq!(outcome.cause, TerminalCause::NaturalEnd, "seed {seed}");
    assert_eq!(
        zero_cost_writes.load(Ordering::SeqCst),
        shape_count,
        "seed {seed}"
    );
    let status = job.status();
    assert!(status.tasks.is_empty(), "unary task leak at seed {seed}");
    assert!(status.edges.values().all(|edge| {
        edge.queue_depth == 0
            && edge.charged_rows == 0
            && edge.charged_bytes == 0
            && edge.high_water_depth <= message_slot_limit
            && edge.high_water_rows == 0
            && edge.high_water_bytes == 0
    }));
    assert!(
        status
            .edges
            .values()
            .all(|edge| edge.high_water_depth == message_slot_limit)
    );
    assert!(status.edges.values().any(|edge| edge.blocked_sends > 0));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1, "seed {seed}");
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1, "seed {seed}");
    assert_eq!(runner.registry_counts(), (0, 0), "seed {seed}");
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0), "seed {seed}");
}

async fn expect_poll_calls(
    poll_calls: &mut mpsc::UnboundedReceiver<usize>,
    expected: impl IntoIterator<Item = usize>,
) {
    for expected in expected {
        let observed = tokio::time::timeout(StdDuration::from_secs(5), poll_calls.recv())
            .await
            .expect("zero-cost source did not make the permitted next poll")
            .expect("zero-cost source poll recorder closed early");
        assert_eq!(observed, expected);
    }
}

async fn wait_for_source_boundary_backpressure(job: &super::ContinuousJob) {
    for _ in 0..1_000 {
        let status = job.status();
        if status.edges.iter().any(|(edge_id, edge)| {
            edge_id.starts_with("source/")
                && edge.queue_depth == status.metrics.edges[edge_id].message_slot_limit
                && edge.blocked_sends > 0
        }) {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("unary source boundary did not reach exact slot backpressure")
}

async fn assert_no_additional_poll(poll_calls: &mut mpsc::UnboundedReceiver<usize>) {
    let mut next_poll = Box::pin(poll_calls.recv());
    assert!(matches!(futures::poll!(next_poll.as_mut()), Poll::Pending));
}

fn assert_exact_poll_and_prefetch_bound(
    job: &super::ContinuousJob,
    polls: &AtomicUsize,
    expected: usize,
) {
    assert_eq!(polls.load(Ordering::SeqCst), expected);
    let status = job.status();
    assert_eq!(
        status.metrics.sources["input"].poll_count,
        u64::try_from(expected).unwrap()
    );
    assert!(status.edges.iter().all(|(edge_id, edge)| {
        edge.queue_depth <= status.metrics.edges[edge_id].message_slot_limit
            && edge.charged_rows == 0
            && edge.charged_bytes == 0
    }));
}

async fn wait_for_write_count(writes: &Mutex<Vec<u64>>, expected: usize) {
    for _ in 0..1_000 {
        if writes.lock().len() >= expected {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("zero-cost sink did not reach {expected} writes")
}

async fn finish_zero_cost_lifecycle(
    job: &super::ContinuousJob,
    sink_gate: &Semaphore,
    termination: ZeroCostTermination,
    receiver_failure: &AtomicBool,
    release_permits: usize,
) -> Arc<super::ContinuousJobOutcome> {
    match termination {
        ZeroCostTermination::Graceful => {
            let observer = job.shutdown();
            sink_gate.add_permits(1);
            observer.await
        }
        ZeroCostTermination::ReceiverClose => {
            receiver_failure.store(true, Ordering::SeqCst);
            sink_gate.add_permits(release_permits);
            job.wait().await
        }
        ZeroCostTermination::Cancel => job.cancel().await,
        ZeroCostTermination::Error => {
            sink_gate.add_permits(release_permits);
            job.wait().await
        }
    }
}

fn assert_zero_cost_terminal(
    termination: ZeroCostTermination,
    outcome: &super::ContinuousJobOutcome,
) {
    match termination {
        ZeroCostTermination::Graceful => {
            assert_eq!(outcome.state, ContinuousJobState::Completed);
            assert_eq!(outcome.cause, TerminalCause::GracefulShutdown);
        }
        ZeroCostTermination::Cancel => {
            assert_eq!(outcome.state, ContinuousJobState::Cancelled);
            assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
        }
        ZeroCostTermination::ReceiverClose => {
            assert_eq!(outcome.state, ContinuousJobState::Failed);
            assert!(matches!(outcome.cause, TerminalCause::TaskFailure { .. }));
            assert!(matches!(
                outcome.errors.first().map(|failure| (&failure.origin, &failure.error)),
                Some((
                    super::FailureOrigin::Task { task_name, .. },
                    CalcFlowError::Operator { node_id, message }
                )) if task_name == "operator:node"
                    && node_id == "node"
                    && message == "zero-cost lifecycle receiver failed"
            ));
        }
        ZeroCostTermination::Error => {
            assert_eq!(outcome.state, ContinuousJobState::Failed);
            assert!(matches!(outcome.cause, TerminalCause::TaskFailure { .. }));
            assert!(matches!(
                outcome
                    .errors
                    .first()
                    .map(|failure| (&failure.origin, &failure.error)),
                Some((
                    super::FailureOrigin::Task { task_name, .. },
                    CalcFlowError::Internal { message }
                )) if task_name == "source:input:pump"
                    && message == "zero-cost lifecycle source failed"
            ));
        }
    }
}

fn assert_zero_cost_convergence(job: &super::ContinuousJob, termination: ZeroCostTermination) {
    let status = job.status();
    assert!(status.tasks.is_empty(), "task leak: {termination:?}");
    assert!(status.edges.values().all(|edge| {
        edge.queue_depth == 0
            && edge.charged_rows == 0
            && edge.charged_bytes == 0
            && edge.high_water_depth <= 1
            && edge.high_water_rows == 0
            && edge.high_water_bytes == 0
    }));
    assert!(
        status.edges.values().any(|edge| edge.blocked_sends > 0),
        "zero-cost messages never exercised backpressure: {termination:?}"
    );
}

async fn run_zero_cost_lifecycle_case(termination: ZeroCostTermination) {
    const CYCLES: usize = 10;
    const CREDIT_RETURNS: usize = 3;
    const INITIAL_POLL_BOUND: usize = 6;
    const POLLS_PER_CREDIT: usize = 3;

    let polls = Arc::new(AtomicUsize::new(0));
    let (poll_call_tx, mut poll_calls) = mpsc::unbounded_channel();
    let eof_observed = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_gate = Arc::new(Semaphore::new(0));
    let writes = Arc::new(Mutex::new(Vec::new()));
    let receiver_failure = Arc::new(AtomicBool::new(false));
    let source = SourceBinding::new(
        Box::new(ZeroCostLifecycleSource {
            events: zero_cost_lifecycle_events(CYCLES),
            polls: Arc::clone(&polls),
            poll_calls: poll_call_tx,
            eof_observed: Arc::clone(&eof_observed),
            closed: Arc::clone(&source_closed),
            end: match termination {
                ZeroCostTermination::Graceful => ZeroCostSourceEnd::Eof,
                ZeroCostTermination::Error => ZeroCostSourceEnd::Error,
                ZeroCostTermination::ReceiverClose | ZeroCostTermination::Cancel => {
                    ZeroCostSourceEnd::Pending
                }
            },
        }),
        None,
        0,
    )
    .unwrap();
    let sinks = vec![NamedSinkBinding {
        output_id: "output".into(),
        sink_id: "zero-cost-sink".into(),
        binding: OrdinarySinkBinding::new(Box::new(ZeroCostLifecycleSink {
            gate: Arc::clone(&sink_gate),
            writes: Arc::clone(&writes),
            closed: Arc::clone(&sink_closed),
        })),
    }];
    let mut runner = ContinuousRunner::new();
    let operator: Box<dyn StreamOperator> =
        if matches!(termination, ZeroCostTermination::ReceiverClose) {
            Box::new(StressForwardOperator::failing_on_watermark(Arc::clone(
                &receiver_failure,
            )))
        } else {
            Box::new(StressForwardOperator::new(None))
        };
    let job = runner
        .start(forward_spec_with_operator(20_000, source, sinks, operator))
        .await
        .unwrap();

    expect_poll_calls(&mut poll_calls, 1..=INITIAL_POLL_BOUND).await;
    wait_for_source_boundary_backpressure(&job).await;
    assert_no_additional_poll(&mut poll_calls).await;
    assert_exact_poll_and_prefetch_bound(&job, &polls, INITIAL_POLL_BOUND);
    let mut expected_polls = INITIAL_POLL_BOUND;
    for delivered in 1..=CREDIT_RETURNS {
        sink_gate.add_permits(1);
        let next_expected = expected_polls + POLLS_PER_CREDIT;
        expect_poll_calls(&mut poll_calls, expected_polls + 1..=next_expected).await;
        wait_for_write_count(&writes, delivered).await;
        wait_for_source_boundary_backpressure(&job).await;
        assert_no_additional_poll(&mut poll_calls).await;
        assert_exact_poll_and_prefetch_bound(&job, &polls, next_expected);
        expected_polls = next_expected;
    }

    if matches!(termination, ZeroCostTermination::Graceful) {
        sink_gate.add_permits(CYCLES - CREDIT_RETURNS - 1);
        expect_poll_calls(&mut poll_calls, expected_polls + 1..=CYCLES * 3 + 1).await;
        wait_for_write_count(&writes, CYCLES - 1).await;
        assert_eq!(eof_observed.load(Ordering::SeqCst), 1);
        assert_eq!(job.status().state, ContinuousJobState::Running);
    }

    let outcome =
        finish_zero_cost_lifecycle(&job, &sink_gate, termination, &receiver_failure, CYCLES).await;
    assert_zero_cost_terminal(termination, &outcome);
    assert_zero_cost_convergence(&job, termination);
    let observed = writes.lock().clone();
    assert!(observed.len() >= CREDIT_RETURNS);
    if matches!(termination, ZeroCostTermination::Graceful) {
        assert_eq!(observed.len(), CYCLES);
        assert_eq!(polls.load(Ordering::SeqCst), CYCLES * 3 + 1);
        assert!(job.status().sources["input"].ended);
    }
    assert_eq!(
        observed,
        (0..u64::try_from(observed.len()).unwrap()).collect::<Vec<_>>()
    );
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    assert_eq!(runner.registry_counts(), (0, 0));
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
}

#[tokio::test]
async fn real_unary_zero_cost_lifecycle_matrix_returns_credits_and_converges() {
    for termination in [
        ZeroCostTermination::Graceful,
        ZeroCostTermination::ReceiverClose,
        ZeroCostTermination::Cancel,
        ZeroCostTermination::Error,
    ] {
        run_zero_cost_lifecycle_case(termination).await;
    }
}

#[tokio::test]
async fn operator_entry_failure_joins_before_any_connector_lifecycle() {
    let resets = Arc::new(AtomicUsize::new(0));
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let mut runner = ContinuousRunner::new();

    let failure = runner
        .start(spec(
            true,
            Arc::clone(&resets),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap_err();

    assert!(matches!(
        failure.primary.error,
        CalcFlowError::Operator { .. }
    ));
    assert_eq!(resets.load(Ordering::SeqCst), 1);
    assert_eq!(source.opened.load(Ordering::SeqCst), 0);
    assert_eq!(source.closed.load(Ordering::SeqCst), 0);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 0);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn operator_reset_panic_is_typed_before_any_connector_lifecycle() {
    let resets = Arc::new(AtomicUsize::new(0));
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let mut runner = ContinuousRunner::new();

    let failure = runner
        .start(reset_spec(
            false,
            true,
            Arc::clone(&resets),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap_err();

    assert!(matches!(
        failure.primary.origin,
        super::FailureOrigin::OperatorEntry { ref node_id } if node_id == "node"
    ));
    assert!(matches!(
        failure.primary.error,
        CalcFlowError::TaskPanicked { task_id: 0, ref message }
            if message == "operator reset panicked"
    ));
    assert_eq!(resets.load(Ordering::SeqCst), 1);
    assert_eq!(source.opened.load(Ordering::SeqCst), 0);
    assert_eq!(source.closed.load(Ordering::SeqCst), 0);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 0);
    runner.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn operator_entry_primary_is_sorted_by_node_id_not_ack_arrival() {
    let later_node_returned = Arc::new(AtomicBool::new(false));
    let operator = |node_id: &'static str| {
        Box::new(OrderedEntryFailureOperator {
            inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
            node_id,
            later_node_returned: Arc::clone(&later_node_returned),
        }) as Box<dyn StreamOperator>
    };
    let plan = PipelineBuilder::new("stable-entry-failure")
        .unwrap()
        .add_node("a", operator("a"))
        .unwrap()
        .add_node("z", operator("z"))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("a", "output").unwrap(),
            PortEndpoint::new("z", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let job_spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            93,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: SourceBinding::new(Box::new(ProbeSource(source.clone())), None, 0).unwrap(),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
        }],
        edge_budget: EdgeBudget::default(),
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();

    let failure = runner.start(job_spec).await.unwrap_err();

    assert!(later_node_returned.load(Ordering::SeqCst));
    assert!(matches!(
        failure.primary.origin,
        super::FailureOrigin::OperatorEntry { ref node_id } if node_id == "a"
    ));
    assert!(matches!(
        failure.primary.error,
        CalcFlowError::Operator { ref node_id, ref message }
            if node_id == "a" && message == "a reset failed"
    ));
    assert_eq!(source.opened.load(Ordering::SeqCst), 0);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn all_operators_enter_before_open_and_data_waits_for_handle_claim() {
    let resets = Arc::new(AtomicUsize::new(0));
    let processed = Arc::new(AtomicUsize::new(0));
    let plan = two_entry_probe_plan(&resets, &processed);
    let polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink = LifecycleProbe::default();
    sink.block_open.store(true, Ordering::SeqCst);
    let job_spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            79,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(1, &polls, &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "blocked".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let observer = runner.start(job_spec);
    let core = Arc::clone(observer.core.as_ref().unwrap());
    let mut observer = Box::pin(observer);
    assert!(matches!(futures::poll!(observer.as_mut()), Poll::Pending));

    for _ in 0..100 {
        if resets.load(Ordering::SeqCst) == 2 && sink.opened.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(resets.load(Ordering::SeqCst), 2);
    assert_eq!(core.runtime_status.lock().tasks.snapshot().len(), 2);
    assert_eq!(
        core.state.lock().launch_delivery,
        super::LaunchDeliveryState::Provisional,
        "synchronous reset has no observer suspension state; entry and open remain provisional"
    );
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(processed.load(Ordering::SeqCst), 0);

    sink.open_release.notify_waiters();
    for _ in 0..100 {
        if core.state.lock().launch_delivery == super::LaunchDeliveryState::ReadyUnclaimed {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(
        core.state.lock().launch_delivery,
        super::LaunchDeliveryState::ReadyUnclaimed
    );
    assert_eq!(core.runtime_status.lock().tasks.snapshot().len(), 6);
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(processed.load(Ordering::SeqCst), 0);

    let Poll::Ready(Ok(job)) = futures::poll!(observer.as_mut()) else {
        panic!("ready-unclaimed start must deliver its handle in one poll");
    };
    drop(observer);
    assert_eq!(
        core.state.lock().launch_delivery,
        super::LaunchDeliveryState::Claimed
    );
    assert!(!core.launch_cancel.is_cancelled());
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(processed.load(Ordering::SeqCst), 0);
    for _ in 0..100 {
        if processed.load(Ordering::SeqCst) == 2 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(processed.load(Ordering::SeqCst), 2);
    assert!(polls.load(Ordering::SeqCst) >= 2);

    let cancelled = job.cancel();
    assert!(
        !job.core.launch_cancel.is_cancelled(),
        "claimed jobs submit their cause before the driver cancels running tasks"
    );
    let outcome = cancelled.await;
    assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[test]
fn explicit_cancel_wins_over_deadline_at_the_single_arbiter_commit() {
    let (commands, _commands_rx) = mpsc::unbounded_channel();
    let core = JobCore::new(
        LaunchId::new(0),
        91,
        commands,
        MetricsRecorder::default(),
        super::StatusProjection::default(),
        false,
        "test".into(),
    );
    core.terminal_arbiter.request_deadline();
    core.terminal_arbiter.request_explicit_cancel();
    let cancellation = CancellationToken::new();
    let observation = core.terminal_arbiter.observe_and_commit(&cancellation);

    assert_eq!(
        observation.terminal,
        Some(super::super::supervisor::terminal::TerminalDecision::ExplicitCancel)
    );
    assert!(cancellation.is_cancelled());
}

#[tokio::test]
async fn explicit_recorded_before_linearized_deadline_commit_wins() {
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();
    let (commit_reached, release_commit) = job.core.install_terminal_commit_seam();

    job.core.request_deadline();
    commit_reached.await.unwrap();
    let cancelled = job.cancel();
    release_commit.send(()).unwrap();
    let outcome = cancelled.await;

    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[test]
fn primary_task_failure_wins_over_cancel_and_deadline_same_round() {
    let primary_task_id = TaskId::new(7);
    let report = SupervisionReport {
        primary_error_count: 1,
        errors: vec![super::super::supervisor::TaskFailure {
            task_id: primary_task_id,
            task_name: "operator:node".into(),
            error: CalcFlowError::Operator {
                node_id: "node".into(),
                message: "same-round failure".into(),
            },
        }],
    };
    let progress = RuntimeTaskProgress {
        sources: BTreeMap::new(),
        sinks: BTreeMap::new(),
    };

    let report = finish_running_report(
        LaunchId::new(0),
        None,
        false,
        report,
        &progress,
        &MetricsRecorder::default(),
    );

    let DriverCompletion::Outcome(outcome) = report.completion else {
        panic!("running driver must publish an outcome");
    };
    assert_eq!(
        outcome.cause,
        TerminalCause::TaskFailure { primary_task_id }
    );
    assert_eq!(outcome.state, ContinuousJobState::Failed);
}

#[tokio::test]
async fn active_operator_cancelled_error_is_a_failed_task_failure() {
    let operator = ActiveCancelledOperator {
        inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
        outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
    };
    let plan = PipelineBuilder::new("active-cancelled")
        .unwrap()
        .add_node("node", Box::new(operator) as Box<dyn StreamOperator>)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink = LifecycleProbe::default();
    let job_spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            95,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: finite_binding(&[1], &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let job = runner.start(job_spec).await.unwrap();

    let outcome = job.wait().await;

    assert_eq!(outcome.state, ContinuousJobState::Failed);
    assert_eq!(
        outcome.cause,
        TerminalCause::TaskFailure {
            primary_task_id: TaskId::new(0)
        }
    );
    assert!(matches!(
        outcome.errors[0].error,
        CalcFlowError::Cancelled { ref run_id } if run_id == "operator-active-cancelled"
    ));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[test]
fn recovery_classification_requires_both_allowed_origin_and_recoverable_error() {
    let classify = |origin, error| classify_failure_state(&RuntimeFailure { origin, error });
    let io_error = || CalcFlowError::Io {
        path: "connector".into(),
        source: std::io::Error::other("recoverable"),
    };
    let source_task = || RuntimeFailureOrigin::Task {
        task_id: TaskId::new(2),
        task_name: "source:left:pump".into(),
    };
    let operator_task = || RuntimeFailureOrigin::Task {
        task_id: TaskId::new(0),
        task_name: "operator:node".into(),
    };

    for origin in [
        RuntimeFailureOrigin::SourceOpen {
            binding_id: "left".into(),
        },
        RuntimeFailureOrigin::SourceClose {
            binding_id: "left".into(),
        },
        RuntimeFailureOrigin::SinkOpen {
            output_id: "output".into(),
            sink_id: "sink".into(),
        },
        RuntimeFailureOrigin::SinkClose {
            output_id: "output".into(),
            sink_id: "sink".into(),
        },
        RuntimeFailureOrigin::SinkWrite {
            output_id: "output".into(),
            sink_id: "sink".into(),
        },
        source_task(),
    ] {
        assert_eq!(
            classify(origin, io_error()),
            ContinuousJobState::RecoveryRequired
        );
    }

    for origin in [
        RuntimeFailureOrigin::Preflight,
        RuntimeFailureOrigin::OperatorEntry {
            node_id: "node".into(),
        },
        RuntimeFailureOrigin::SinkIngress {
            output_id: "output".into(),
            edge_id: "edge".into(),
        },
        operator_task(),
        RuntimeFailureOrigin::Metrics {
            component_id: "job".into(),
            counter: "errors",
        },
    ] {
        assert_eq!(classify(origin, io_error()), ContinuousJobState::Failed);
    }
    assert_eq!(
        classify(
            source_task(),
            CalcFlowError::Cancelled {
                run_id: "active-source".into(),
            }
        ),
        ContinuousJobState::Failed
    );
    assert_eq!(
        classify(
            RuntimeFailureOrigin::SourceOpen {
                binding_id: "left".into(),
            },
            CalcFlowError::Internal {
                message: "not recoverable".into(),
            }
        ),
        ContinuousJobState::Failed
    );
}

#[tokio::test]
async fn terminal_observers_are_idempotent_and_dropped_wait_does_not_cancel() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap();
    let mut wait = Box::pin(job.wait());
    futures::future::poll_fn(|context| match wait.as_mut().poll(context) {
        Poll::Pending => Poll::Ready(()),
        Poll::Ready(_) => panic!("running job completed before a terminal cause"),
    })
    .await;
    drop(wait);

    let cancelled = job.cancel().await;
    let observed = job.wait().await;

    assert!(Arc::ptr_eq(&cancelled, &observed));
    assert_eq!(cancelled.state, ContinuousJobState::Cancelled);
    assert_eq!(cancelled.cause, TerminalCause::ExplicitCancel);
    assert_eq!(job.driver_owner(), DriverOwnership::Terminal);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn dropped_cancel_and_shutdown_observers_preserve_driver_and_join_ownership() {
    for (label, cancel, drop_during_close) in [
        ("cancel-driver", true, false),
        ("shutdown-driver", false, false),
        ("cancel-join", true, true),
        ("shutdown-join", false, true),
    ] {
        let mut runner = ContinuousRunner::new();
        let source = LifecycleProbe::default();
        source
            .block_close
            .store(drop_during_close, Ordering::SeqCst);
        let sink = LifecycleProbe::default();
        let job = runner
            .start(spec(
                false,
                Arc::new(AtomicUsize::new(0)),
                source.clone(),
                sink.clone(),
            ))
            .await
            .unwrap();
        let mut observer = Box::pin(if cancel { job.cancel() } else { job.shutdown() });

        assert!(
            matches!(futures::poll!(observer.as_mut()), Poll::Pending),
            "terminal observer completed before the driver linearization point: {label}"
        );
        if drop_during_close {
            for _ in 0..100 {
                if source.closed.load(Ordering::SeqCst) == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(source.closed.load(Ordering::SeqCst), 1, "{label}");
            assert!(
                matches!(futures::poll!(observer.as_mut()), Poll::Pending),
                "terminal observer completed before connector join: {label}"
            );
        }

        drop(observer);
        assert_eq!(runner.registry_counts().0, 1, "registry lost at {label}");
        if drop_during_close {
            source.close_release.notify_waiters();
        }
        let outcome = job.wait().await;
        assert_eq!(
            outcome.cause,
            if cancel {
                TerminalCause::ExplicitCancel
            } else {
                TerminalCause::GracefulShutdown
            },
            "{label}"
        );
        assert!(job.status().tasks.is_empty(), "{label}");
        assert_eq!(source.closed.load(Ordering::SeqCst), 1, "{label}");
        assert_eq!(sink.closed.load(Ordering::SeqCst), 1, "{label}");
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0), "{label}");
    }
}

#[tokio::test]
async fn dropped_runner_shutdown_observer_leaves_join_handle_for_retry() {
    let release = Arc::new(Notify::new());
    let completed = Arc::new(AtomicBool::new(false));
    let release_in_driver = Arc::clone(&release);
    let completed_in_driver = Arc::clone(&completed);
    let driver = tokio::spawn(async move {
        release_in_driver.notified().await;
        completed_in_driver.store(true, Ordering::SeqCst);
    });
    let (commands, _commands_rx) = mpsc::unbounded_channel();
    let core = Arc::new(RunnerCore {
        commands,
        root_cancel: CancellationToken::new(),
        stop_after_first_job: false,
        registry: Mutex::new(RunnerRegistryState {
            provisional: None,
            live_jobs: BTreeMap::new(),
            reaper_jobs: BTreeSet::new(),
            pending_start: None,
            shutting_down: true,
        }),
        driver: Mutex::new(Some(driver)),
        diagnostics: RunnerDiagnostics::default(),
        next_launch_id: AtomicU64::new(0),
        closed: AtomicBool::new(true),
        changed: Notify::new(),
        abandonment_warnings: AtomicU64::new(0),
        next_launch_probe: Mutex::new(None),
        panic_lifecycle_after_shutdown: AtomicBool::new(false),
    });
    let mut first = Box::pin(RunnerShutdownObserver::new(Arc::clone(&core)));

    assert!(matches!(futures::poll!(first.as_mut()), Poll::Pending));
    drop(first);

    assert!(
        core.driver.lock().is_some(),
        "a dropped shutdown observer detached the core-owned lifecycle handle"
    );
    release.notify_one();
    RunnerShutdownObserver::new(Arc::clone(&core))
        .await
        .unwrap();
    assert!(completed.load(Ordering::SeqCst));
    assert!(core.driver.lock().is_none());
}

#[tokio::test]
async fn dropped_real_runner_shutdown_observer_is_joined_by_retry() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    source.block_close.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap();
    let core = Arc::clone(&runner.core);
    let mut shutdown = Box::pin(runner.shutdown());

    assert!(matches!(futures::poll!(shutdown.as_mut()), Poll::Pending));
    drop(shutdown);
    for _ in 0..100 {
        if source.closed.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert!(core.driver.lock().is_some());

    source.close_release.notify_waiters();
    runner.shutdown().await.unwrap();
    assert!(core.driver.lock().is_none());
    assert_eq!(job.wait().await.cause, TerminalCause::ExplicitCancel);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dropping_start_during_open_closes_once_and_next_start_waits_for_reaper() {
    let mut runner = ContinuousRunner::new();
    let blocked_source = LifecycleProbe::default();
    blocked_source.block_open.store(true, Ordering::SeqCst);
    let blocked_sink = LifecycleProbe::default();
    let opened = blocked_source.open_started.notified();
    tokio::pin!(opened);
    let mut start = Box::pin(runner.start(spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        blocked_source.clone(),
        blocked_sink.clone(),
    )));
    tokio::select! {
        result = &mut start => panic!("start completed before the open gate: {result:?}"),
        () = &mut opened => {}
    }
    drop(start);

    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let next = runner
        .start(spec(false, Arc::new(AtomicUsize::new(0)), source, sink))
        .await
        .unwrap();

    assert_eq!(blocked_source.opened.load(Ordering::SeqCst), 1);
    assert_eq!(blocked_source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(blocked_sink.closed.load(Ordering::SeqCst), 1);
    drop(next);
    runner.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn dropped_start_pending_close_expires_and_reaper_allows_next_start() {
    let source = LifecycleProbe::default();
    source.block_open.store(true, Ordering::SeqCst);
    source.block_close.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    let mut runner = ContinuousRunner::new();
    let observer = runner.start(spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        source.clone(),
        sink.clone(),
    ));
    let provisional = Arc::clone(observer.core.as_ref().unwrap());
    wait_for_counter(&source.opened, 1).await;
    wait_for_counter(&sink.open_completed, 1).await;

    drop(observer);

    assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
    wait_for_counter(&source.closed, 1).await;
    let mut next = Box::pin(runner.start(spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        LifecycleProbe::default(),
        LifecycleProbe::default(),
    )));
    assert!(matches!(futures::poll!(next.as_mut()), Poll::Pending));

    tokio::time::advance(StdDuration::from_secs(5)).await;
    let next = tokio::time::timeout(StdDuration::from_secs(1), next)
        .await
        .expect("a bounded cancelled-launch close must release the reaper")
        .unwrap();

    assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    let diagnostic = runner
        .diagnostics()
        .records
        .into_iter()
        .find(|record| record.launch_id == provisional.launch_id)
        .expect("dropped start cleanup must retain its timeout diagnostic");
    assert_eq!(diagnostic.cleanup_failures.len(), 1);
    assert!(matches!(
        &diagnostic.cleanup_failures[0].origin,
        super::FailureOrigin::SourceClose { binding_id } if binding_id == "input"
    ));
    assert!(matches!(
        &diagnostic.cleanup_failures[0].error,
        CalcFlowError::Internal { message }
            if message == "connector close exceeded private teardown bound of 5 seconds"
    ));
    {
        let runtime = provisional.runtime_status.lock();
        assert!(runtime.tasks.snapshot().is_empty());
    }
    assert!(provisional.metrics.snapshot().edges.values().all(|edge| {
        edge.channel.queue_depth == 0
            && edge.channel.charged_rows == 0
            && edge.channel.charged_bytes == 0
    }));

    drop(next);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_start_during_operator_entry_reaps_before_connector_lifecycle() {
    let entered = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let mut runner = ContinuousRunner::new();
    let observer = runner.start(blocking_entry_spec(
        &entered,
        &release,
        source.clone(),
        sink.clone(),
    ));
    let provisional = Arc::clone(observer.core.as_ref().unwrap());
    let entry_observed = wait_for_operator_entry(&entered).await;
    if !entry_observed {
        release.store(true, Ordering::SeqCst);
    }
    assert!(entry_observed, "operator entry did not begin");

    drop(observer);
    assert!(provisional.launch_cancel.is_cancelled());
    assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
    release.store(true, Ordering::SeqCst);

    let next = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();
    assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
    assert_eq!(source.opened.load(Ordering::SeqCst), 0);
    assert_eq!(source.closed.load(Ordering::SeqCst), 0);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 0);
    drop(next);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn dropping_start_immediately_after_provisional_registration_is_reaped() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let observer = runner.start(spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        source.clone(),
        sink.clone(),
    ));
    let provisional = Arc::clone(observer.core.as_ref().unwrap());
    assert_eq!(
        provisional.state.lock().launch_delivery,
        super::LaunchDeliveryState::Provisional
    );

    drop(observer);

    assert!(provisional.launch_cancel.is_cancelled());
    assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
    let next = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();
    assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
    assert_eq!(source.opened.load(Ordering::SeqCst), 0);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 0);
    drop(next);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn dropping_start_after_one_of_two_sources_opens_reaps_every_begun_connector() {
    let plan = union_plan();
    let left = LifecycleProbe::default();
    let right = LifecycleProbe::default();
    right.block_open.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    let job_spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            80,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![
            NamedSourceBinding {
                binding_id: "left".into(),
                binding: SourceBinding::new(Box::new(ProbeSource(left.clone())), None, 0).unwrap(),
            },
            NamedSourceBinding {
                binding_id: "right".into(),
                binding: SourceBinding::new(Box::new(ProbeSource(right.clone())), None, 0).unwrap(),
            },
        ],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
        }],
        edge_budget: EdgeBudget::default(),
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let observer = runner.start(job_spec);
    let provisional = Arc::clone(observer.core.as_ref().unwrap());
    let mut observer = Box::pin(observer);
    assert!(matches!(futures::poll!(observer.as_mut()), Poll::Pending));
    for _ in 0..100 {
        if left.open_completed.load(Ordering::SeqCst) == 1
            && right.opened.load(Ordering::SeqCst) == 1
        {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(left.open_completed.load(Ordering::SeqCst), 1);
    assert_eq!(right.opened.load(Ordering::SeqCst), 1);
    assert_eq!(right.open_completed.load(Ordering::SeqCst), 0);
    assert_eq!(
        provisional.state.lock().launch_delivery,
        super::LaunchDeliveryState::Provisional
    );

    drop(observer);

    let next = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();
    assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
    assert_eq!(left.closed.load(Ordering::SeqCst), 1);
    assert_eq!(right.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    drop(next);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn dropping_ready_unclaimed_start_never_releases_the_data_gate() {
    let plan = unary_expression_plan();
    let polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink = LifecycleProbe::default();
    let job_spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            89,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(1, &polls, &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
        }],
        edge_budget: EdgeBudget::default(),
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let observer = runner.start(job_spec);
    let provisional = Arc::clone(observer.core.as_ref().unwrap());
    for _ in 0..100 {
        if provisional.state.lock().launch_delivery == super::LaunchDeliveryState::ReadyUnclaimed {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(
        provisional.state.lock().launch_delivery,
        super::LaunchDeliveryState::ReadyUnclaimed
    );
    assert_eq!(polls.load(Ordering::SeqCst), 0);

    drop(observer);

    let next = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
    drop(next);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn dropping_start_after_operator_entry_before_live_publication_is_reaped() {
    let resets = Arc::new(AtomicUsize::new(0));
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let probe = Arc::new(super::TestLaunchProbe::new(
        super::TestLaunchCheckpoint::AfterOperatorEntry,
    ));
    let mut runner = ContinuousRunner::new();
    let observer = runner.start_with_test_launch_probe(
        spec(false, Arc::clone(&resets), source.clone(), sink.clone()),
        Arc::clone(&probe),
    );
    let provisional = Arc::clone(observer.core.as_ref().unwrap());

    probe.wait_until_reached().await;
    assert_eq!(resets.load(Ordering::SeqCst), 1);
    assert_eq!(
        provisional.state.lock().launch_delivery,
        super::LaunchDeliveryState::Provisional
    );
    assert_eq!(source.opened.load(Ordering::SeqCst), 0);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 0);

    drop(observer);
    assert!(provisional.launch_cancel.is_cancelled());
    assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
    probe.release();

    let next = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();
    assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
    assert_eq!(source.closed.load(Ordering::SeqCst), 0);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 0);
    drop(next);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
}

#[tokio::test]
async fn dropping_start_after_live_publication_before_handle_delivery_is_reaped() {
    let plan = unary_expression_plan();
    let polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink = LifecycleProbe::default();
    let job_spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            90,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(1, &polls, &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
        }],
        edge_budget: EdgeBudget::default(),
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let probe = Arc::new(super::TestLaunchProbe::new(
        super::TestLaunchCheckpoint::LivePublished,
    ));
    let observer = runner.start_with_test_launch_probe(job_spec, Arc::clone(&probe));
    let provisional = Arc::clone(observer.core.as_ref().unwrap());
    probe.wait_until_reached().await;
    assert_eq!(
        provisional.state.lock().launch_delivery,
        super::LaunchDeliveryState::ReadyUnclaimed
    );
    assert_eq!(sink.open_completed.load(Ordering::SeqCst), 1);

    drop(observer);
    assert!(provisional.launch_cancel.is_cancelled());
    assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    probe.release();

    let next = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();
    assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    drop(next);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn connector_open_failure_closes_every_begun_resource_once() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    source.fail_open.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();

    let failure = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap_err();

    assert!(matches!(
        failure.primary.origin,
        super::FailureOrigin::SourceOpen { .. }
    ));
    assert_eq!(source.opened.load(Ordering::SeqCst), 1);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    runner.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn failed_launch_bounds_close_and_keeps_stable_cleanup_diagnostics() {
    let source = LifecycleProbe::default();
    source.block_open.store(true, Ordering::SeqCst);
    source.fail_open.store(true, Ordering::SeqCst);
    source.block_close.store(true, Ordering::SeqCst);
    let panic_sink = LifecycleProbe::default();
    panic_sink.panic_close.store(true, Ordering::SeqCst);
    let error_sink = LifecycleProbe::default();
    error_sink.fail_close.store(true, Ordering::SeqCst);
    let later_sink = LifecycleProbe::default();
    let job_spec = forward_spec(
        101,
        SourceBinding::new(Box::new(ProbeSource(source.clone())), None, 0).unwrap(),
        vec![
            named_probe_sink("a-panic", panic_sink.clone()),
            named_probe_sink("b-error", error_sink.clone()),
            named_probe_sink("c-later", later_sink.clone()),
        ],
    );
    let mut runner = ContinuousRunner::new();
    let observer = runner.start(job_spec);
    let provisional = Arc::clone(observer.core.as_ref().unwrap());
    wait_for_counter(&source.opened, 1).await;
    wait_for_counter(&panic_sink.open_completed, 1).await;
    wait_for_counter(&error_sink.open_completed, 1).await;
    wait_for_counter(&later_sink.open_completed, 1).await;
    source.open_release.notify_waiters();
    wait_for_counter(&source.closed, 1).await;
    assert_eq!(panic_sink.closed.load(Ordering::SeqCst), 0);
    let shutdown = runner.shutdown();

    tokio::time::advance(StdDuration::from_secs(5)).await;
    let failure = tokio::time::timeout(StdDuration::from_secs(1), observer)
        .await
        .expect("failed launch must outlive a permanently pending close")
        .unwrap_err();
    shutdown.await.unwrap();

    assert!(matches!(
        &failure.primary.origin,
        super::FailureOrigin::SourceOpen { binding_id } if binding_id == "input"
    ));
    assert!(matches!(
        &failure.primary.error,
        CalcFlowError::Internal { message } if message == "source open failed"
    ));
    let diagnostic_id = failure
        .diagnostic_id
        .expect("bounded close failures must be retained as secondaries");
    let diagnostics = runner.diagnostics();
    let cleanup = &diagnostics
        .records
        .iter()
        .find(|record| record.id == diagnostic_id)
        .unwrap()
        .cleanup_failures;
    assert_eq!(cleanup.len(), 3);
    assert!(matches!(
        (&cleanup[0].origin, &cleanup[0].error),
        (
            super::FailureOrigin::SourceClose { binding_id },
            CalcFlowError::Internal { message }
        ) if binding_id == "input"
            && message == "connector close exceeded private teardown bound of 5 seconds"
    ));
    assert!(matches!(
        (&cleanup[1].origin, &cleanup[1].error),
        (
            super::FailureOrigin::SinkClose { output_id, sink_id },
            CalcFlowError::TaskPanicked { task_id: 1, message }
        ) if output_id == "output" && sink_id == "a-panic"
            && message == "sink close panicked"
    ));
    assert!(matches!(
        (&cleanup[2].origin, &cleanup[2].error),
        (
            super::FailureOrigin::SinkClose { output_id, sink_id },
            CalcFlowError::Internal { message }
        ) if output_id == "output" && sink_id == "b-error"
            && message == "sink close failed"
    ));
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(panic_sink.closed.load(Ordering::SeqCst), 1);
    assert_eq!(error_sink.closed.load(Ordering::SeqCst), 1);
    assert_eq!(later_sink.closed.load(Ordering::SeqCst), 1);
    assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
    assert!(
        provisional
            .runtime_status
            .lock()
            .tasks
            .snapshot()
            .is_empty()
    );
    assert!(provisional.metrics.snapshot().edges.values().all(|edge| {
        edge.channel.queue_depth == 0
            && edge.channel.charged_rows == 0
            && edge.channel.charged_bytes == 0
    }));
    assert_eq!(runner.registry_counts(), (0, 0));
}

#[tokio::test]
async fn connector_open_panic_is_typed_and_closes_every_begun_resource_once() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    source.panic_open.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();

    let failure = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap_err();

    assert!(matches!(
        failure.primary.origin,
        super::FailureOrigin::SourceOpen { ref binding_id } if binding_id == "input"
    ));
    assert!(matches!(
        failure.primary.error,
        CalcFlowError::TaskPanicked { task_id: 0, ref message }
            if message == "source open panicked"
    ));
    assert_eq!(source.opened.load(Ordering::SeqCst), 1);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn source_next_panic_is_typed_and_source_and_sink_close_once() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    source.panic_next.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap();

    let outcome = job.wait().await;

    assert_eq!(outcome.state, ContinuousJobState::Failed);
    assert_eq!(
        outcome.cause,
        TerminalCause::TaskFailure {
            primary_task_id: TaskId::new(1)
        }
    );
    assert!(matches!(
        outcome.errors[0].error,
        CalcFlowError::TaskPanicked { task_id: 1, ref message }
            if message == "source next panicked"
    ));
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn sink_write_panic_is_typed_and_all_connectors_close_once() {
    let source_closed = Arc::new(AtomicUsize::new(0));
    let panic_sink = LifecycleProbe::default();
    panic_sink.panic_write.store(true, Ordering::SeqCst);
    let sibling_sink = LifecycleProbe::default();
    let job_spec = forward_spec(
        96,
        finite_binding(&[1], &source_closed),
        vec![
            named_probe_sink("a-panic", panic_sink.clone()),
            named_probe_sink("b-sibling", sibling_sink.clone()),
        ],
    );
    let mut runner = ContinuousRunner::new();
    let job = runner.start(job_spec).await.unwrap();

    let outcome = job.wait().await;

    assert_eq!(outcome.state, ContinuousJobState::Failed);
    assert_eq!(
        outcome.cause,
        TerminalCause::TaskFailure {
            primary_task_id: TaskId::new(3)
        }
    );
    assert!(matches!(
        outcome.errors[0].origin,
        super::FailureOrigin::SinkWrite { ref sink_id, .. } if sink_id == "a-panic"
    ));
    assert!(matches!(
        outcome.errors[0].error,
        CalcFlowError::TaskPanicked { task_id: 3, ref message }
            if message == "sink write panicked"
    ));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(panic_sink.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sibling_sink.closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn sink_close_panic_is_typed_and_does_not_skip_later_sink_close() {
    let source_closed = Arc::new(AtomicUsize::new(0));
    let panic_sink = LifecycleProbe::default();
    panic_sink.panic_close.store(true, Ordering::SeqCst);
    let sibling_sink = LifecycleProbe::default();
    let job_spec = forward_spec(
        97,
        finite_binding(&[1], &source_closed),
        vec![
            named_probe_sink("a-panic", panic_sink.clone()),
            named_probe_sink("b-sibling", sibling_sink.clone()),
        ],
    );
    let mut runner = ContinuousRunner::new();
    let job = runner.start(job_spec).await.unwrap();

    let outcome = job.wait().await;

    assert_eq!(outcome.state, ContinuousJobState::Failed);
    assert_eq!(
        outcome.cause,
        TerminalCause::TaskFailure {
            primary_task_id: TaskId::new(3)
        }
    );
    assert!(matches!(
        outcome.errors[0].origin,
        super::FailureOrigin::SinkClose { ref sink_id, .. } if sink_id == "a-panic"
    ));
    assert!(matches!(
        outcome.errors[0].error,
        CalcFlowError::TaskPanicked { task_id: 3, ref message }
            if message == "sink close panicked"
    ));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(panic_sink.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sibling_sink.closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn sink_open_panic_is_typed_and_closes_every_begun_resource_once() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    sink.panic_open.store(true, Ordering::SeqCst);

    let failure = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap_err();

    assert!(matches!(
        failure.primary.origin,
        super::FailureOrigin::SinkOpen { ref output_id, ref sink_id }
            if output_id == "output" && sink_id == "sink"
    ));
    assert!(matches!(
        failure.primary.error,
        CalcFlowError::TaskPanicked { task_id: 1, ref message }
            if message == "sink open panicked"
    ));
    assert_eq!(source.opened.load(Ordering::SeqCst), 1);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.opened.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn source_open_primary_keeps_close_failure_in_stable_diagnostics() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    source.fail_open.store(true, Ordering::SeqCst);
    source.fail_close.store(true, Ordering::SeqCst);

    let failure = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap_err();

    assert!(matches!(
        failure.primary.origin,
        super::FailureOrigin::SourceOpen { ref binding_id } if binding_id == "input"
    ));
    assert!(matches!(
        failure.primary.error,
        CalcFlowError::Internal { ref message } if message == "source open failed"
    ));
    let diagnostic_id = failure
        .diagnostic_id
        .expect("source close failure must be retained in diagnostics");
    let diagnostics = runner.diagnostics();
    let record = diagnostics
        .records
        .iter()
        .find(|record| record.id == diagnostic_id)
        .unwrap();
    assert_eq!(record.cleanup_failures.len(), 1);
    assert!(matches!(
        record.cleanup_failures[0].origin,
        super::FailureOrigin::SourceClose { ref binding_id } if binding_id == "input"
    ));
    assert!(matches!(
        record.cleanup_failures[0].error,
        CalcFlowError::Internal { ref message } if message == "source close failed"
    ));
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn running_source_primary_errors_keep_close_as_stable_secondary() {
    for (job_id, failure, expected_task, expected_primary) in [
        (
            86,
            RunningSourceFailure::Next,
            "source:input:pump",
            "source-next-primary",
        ),
        (
            87,
            RunningSourceFailure::Cursor,
            "source:input:task",
            "sources.input.cursor",
        ),
    ] {
        let plan = unary_expression_plan();
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                job_id,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: SourceBinding::new(
                    Box::new(PrimaryAndCloseFailingSource {
                        failure,
                        next_call: 0,
                        closed: Arc::clone(&source_closed),
                    }),
                    None,
                    0,
                )
                .unwrap(),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "recording".into(),
                binding: OrdinarySinkBinding::new(Box::new(OrderedRecordingSink {
                    id: "recording".into(),
                    writes: Arc::new(Mutex::new(Vec::new())),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
            static_inputs: crate::static_input::PreparedStaticInputs::default(),
        };
        let mut runner = ContinuousRunner::new();
        let job = runner.start(spec).await.unwrap();
        let outcome = job.wait().await;

        assert_eq!(outcome.state, ContinuousJobState::Failed);
        assert!(matches!(outcome.cause, TerminalCause::TaskFailure { .. }));
        assert_eq!(outcome.errors.len(), 2, "errors: {:?}", outcome.errors);
        assert!(matches!(
            outcome.errors[0].origin,
            super::FailureOrigin::Task { ref task_name, .. } if task_name == expected_task
        ));
        match &outcome.errors[0].error {
            CalcFlowError::Internal { message } => assert_eq!(message, expected_primary),
            CalcFlowError::InvalidArgument { field, .. } => {
                assert_eq!(field, expected_primary);
            }
            error => panic!("unexpected source primary: {error:?}"),
        }
        assert!(matches!(
            outcome.errors[1].origin,
            super::FailureOrigin::SourceClose { ref binding_id } if binding_id == "input"
        ));
        assert!(matches!(
            outcome.errors[1].error,
            CalcFlowError::Internal { ref message } if message == "source-close-secondary"
        ));
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn graceful_shutdown_and_deadline_have_explicit_distinct_causes() {
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();

    let drained = job.shutdown().await;

    assert_eq!(drained.state, ContinuousJobState::Completed);
    assert_eq!(drained.cause, TerminalCause::GracefulShutdown);
    drop(job);

    let mut deadline_spec = spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        LifecycleProbe::default(),
        LifecycleProbe::default(),
    );
    deadline_spec.context = StreamJobContext::new(
        10,
        deadline_spec.plan.fingerprint(),
        JsonMap::new(),
        Some(chrono::Utc::now() - chrono::Duration::milliseconds(1)),
        CancellationToken::new(),
    );
    let deadline_job = runner.start(deadline_spec).await.unwrap();
    let deadline = deadline_job.wait().await;

    assert_eq!(deadline.state, ContinuousJobState::Cancelled);
    assert_eq!(deadline.cause, TerminalCause::DeadlineExceeded);
    assert!(deadline.errors.is_empty());
    drop(deadline_job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn explicit_cancel_keeps_source_and_sink_close_errors_secondary() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    source.fail_close.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    sink.fail_close.store(true, Ordering::SeqCst);
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap();

    let outcome = job.cancel().await;

    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
    assert_eq!(outcome.errors.len(), 2);
    assert!(matches!(
        outcome.errors[0].origin,
        super::FailureOrigin::SourceClose { ref binding_id } if binding_id == "input"
    ));
    assert!(matches!(
        outcome.errors[1].origin,
        super::FailureOrigin::SinkClose { ref output_id, ref sink_id }
            if output_id == "output" && sink_id == "sink"
    ));
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn deadline_keeps_source_and_sink_close_errors_secondary() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    source.fail_close.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    sink.fail_close.store(true, Ordering::SeqCst);
    let mut deadline_spec = spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        source.clone(),
        sink.clone(),
    );
    deadline_spec.context = StreamJobContext::new(
        10,
        deadline_spec.plan.fingerprint(),
        JsonMap::new(),
        Some(chrono::Utc::now() - chrono::Duration::milliseconds(1)),
        CancellationToken::new(),
    );
    let job = runner.start(deadline_spec).await.unwrap();

    let outcome = job.wait().await;

    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(outcome.cause, TerminalCause::DeadlineExceeded);
    assert_eq!(outcome.errors.len(), 2);
    assert!(matches!(
        outcome.errors[0].origin,
        super::FailureOrigin::SourceClose { ref binding_id } if binding_id == "input"
    ));
    assert!(matches!(
        outcome.errors[1].origin,
        super::FailureOrigin::SinkClose { ref output_id, ref sink_id }
            if output_id == "output" && sink_id == "sink"
    ));
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn committed_deadline_is_immutable_while_connector_close_is_blocked() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    source.block_close.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    sink.block_close.store(true, Ordering::SeqCst);
    let mut deadline_spec = spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        source.clone(),
        sink.clone(),
    );
    deadline_spec.context = StreamJobContext::new(
        94,
        deadline_spec.plan.fingerprint(),
        JsonMap::new(),
        Some(chrono::Utc::now() - chrono::Duration::milliseconds(1)),
        CancellationToken::new(),
    );
    let job = runner.start(deadline_spec).await.unwrap();

    while job.status().terminal_cause != Some(TerminalCause::DeadlineExceeded)
        || source.closed.load(Ordering::SeqCst) != 1
        || sink.closed.load(Ordering::SeqCst) != 1
    {
        tokio::task::yield_now().await;
    }
    let cancelled = job.cancel();
    assert_eq!(
        job.status().terminal_cause,
        Some(TerminalCause::DeadlineExceeded)
    );

    source.close_release.notify_one();
    sink.close_release.notify_one();
    let outcome = cancelled.await;

    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(outcome.cause, TerminalCause::DeadlineExceeded);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn context_task_handle_status_and_diagnostics_share_the_context_job_id() {
    let expected_job_id = 42_424;
    let observed_job_id = Arc::new(Mutex::new(None));
    let operator = JobIdentityProbeOperator {
        inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
        outputs: [Port::new("output", BatchKind::Table, false, None).unwrap()],
        observed_job_id: Arc::clone(&observed_job_id),
    };
    let plan = PipelineBuilder::new("job-identity")
        .unwrap()
        .add_node("node", Box::new(operator) as Box<dyn StreamOperator>)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink = LifecycleProbe::default();
    let identity_spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            expected_job_id,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: finite_binding(&[1], &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let job = runner.start(identity_spec).await.unwrap();

    let outcome = job.wait().await;

    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(job.id(), expected_job_id);
    assert_eq!(job.status().job_id, expected_job_id);
    assert_eq!(*observed_job_id.lock(), Some(expected_job_id));
    drop(job);

    let failed_job_id = expected_job_id + 1;
    let source = LifecycleProbe::default();
    source.fail_open.store(true, Ordering::SeqCst);
    source.fail_close.store(true, Ordering::SeqCst);
    let mut failed_spec = spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        source,
        LifecycleProbe::default(),
    );
    failed_spec.context = StreamJobContext::new(
        failed_job_id,
        failed_spec.plan.fingerprint(),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let failed = runner.start(failed_spec);
    let failed_launch_id = failed.core.as_ref().unwrap().launch_id;
    let failure = failed.await.unwrap_err();
    let diagnostic_id = failure.diagnostic_id.unwrap();
    let diagnostics = runner.diagnostics();
    let diagnostic = diagnostics
        .records
        .iter()
        .find(|record| record.id == diagnostic_id)
        .unwrap();
    assert_eq!(diagnostic.launch_id, failed_launch_id);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn reused_context_job_id_has_distinct_private_diagnostic_launch_ids() {
    let context_job_id = 42_425;
    let mut runner = ContinuousRunner::new();
    let mut observed = Vec::new();
    for _ in 0..2 {
        let source = LifecycleProbe::default();
        source.fail_open.store(true, Ordering::SeqCst);
        source.fail_close.store(true, Ordering::SeqCst);
        let mut failed_spec = spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source,
            LifecycleProbe::default(),
        );
        failed_spec.context = StreamJobContext::new(
            context_job_id,
            failed_spec.plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let start = runner.start(failed_spec);
        let launch_id = start.core.as_ref().unwrap().launch_id;
        let failure = start.await.unwrap_err();
        observed.push((launch_id, failure.diagnostic_id.unwrap()));
    }

    assert_ne!(observed[0].0, observed[1].0);
    let diagnostics = runner.diagnostics();
    let diagnostic_launch_ids = observed
        .iter()
        .map(|(launch_id, diagnostic_id)| {
            let record = diagnostics
                .records
                .iter()
                .find(|record| record.id == *diagnostic_id)
                .unwrap();
            assert_eq!(record.launch_id, *launch_id);
            record.launch_id
        })
        .collect::<Vec<_>>();
    assert_eq!(diagnostic_launch_ids, [observed[0].0, observed[1].0]);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn dropped_job_transfers_driver_and_next_start_reaps_before_launch() {
    let mut runner = ContinuousRunner::new();
    let first_source = LifecycleProbe::default();
    let first = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            first_source.clone(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();
    assert_eq!(first.driver_owner(), DriverOwnership::Driving);

    drop(first);
    let second = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();

    assert_eq!(first_source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(second.state(), ContinuousJobState::Running);
    drop(second);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
}

#[tokio::test]
async fn dropped_job_transfers_driver_and_runner_shutdown_reaps() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap();
    let core = Arc::clone(&job.core);

    drop(job);

    assert_eq!(core.state.lock().owner, DriverOwnership::ReaperOwned);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(core.state.lock().owner, DriverOwnership::Terminal);
    assert_eq!(core.metrics.snapshot().job.reaper_joins, 1);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn runner_drop_records_abandonment_and_requests_reaper_cancellation() {
    let runner = ContinuousRunner::new();
    let runner_core = Arc::clone(&runner.core);
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();
    assert_eq!(job.status().metrics.job.abandoned_runner_drops, 0);
    assert!(!job.core.launch_cancel.is_cancelled());

    drop(runner);

    assert_eq!(
        ABANDONED_RUNNER_WARNING,
        "continuous runner dropped before shutdown completed; cancellation requested"
    );
    assert_eq!(runner_core.abandonment_warnings.load(Ordering::SeqCst), 1);
    assert_eq!(job.status().metrics.job.abandoned_runner_drops, 1);
    assert!(job.core.terminal_arbiter.explicit_cancel_requested());
    assert_eq!(job.core.state.lock().owner, DriverOwnership::ReaperOwned);
    assert!(
        !job.core.launch_cancel.is_cancelled(),
        "claimed jobs are cancelled only after the driver arbitrates the terminal cause"
    );
}

#[tokio::test]
async fn runner_drop_reaper_publishes_completion_for_a_live_job() {
    let runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap();

    drop(runner);

    let mut wait = Box::pin(job.wait());
    let mut completed = None;
    for _ in 0..100 {
        if let Poll::Ready(outcome) = futures::poll!(wait.as_mut()) {
            completed = Some(outcome);
            break;
        }
        tokio::task::yield_now().await;
    }
    let outcome = completed.expect("runner Drop reaper did not publish the live job outcome");
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
    assert_eq!(job.driver_owner(), DriverOwnership::Terminal);
    assert!(job.status().tasks.is_empty());
    assert_eq!(job.status().metrics.job.abandoned_runner_drops, 1);
    assert_eq!(job.status().metrics.job.reaper_joins, 1);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn runner_drop_cancels_and_reaps_a_provisional_open() {
    let runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    source.block_open.store(true, Ordering::SeqCst);
    let sink = LifecycleProbe::default();
    let observer = runner.start(spec(
        false,
        Arc::new(AtomicUsize::new(0)),
        source.clone(),
        sink.clone(),
    ));
    let provisional = Arc::clone(observer.core.as_ref().unwrap());
    let mut observer = Box::pin(observer);
    assert!(matches!(futures::poll!(observer.as_mut()), Poll::Pending));
    for _ in 0..100 {
        if source.opened.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(source.opened.load(Ordering::SeqCst), 1);

    drop(runner);

    assert!(provisional.launch_cancel.is_cancelled());
    assert_eq!(provisional.state.lock().owner, DriverOwnership::ReaperOwned);
    assert_eq!(provisional.metrics.snapshot().job.abandoned_runner_drops, 1);
    let mut completed = None;
    for _ in 0..100 {
        if let Poll::Ready(result) = futures::poll!(observer.as_mut()) {
            completed = Some(result);
            break;
        }
        tokio::task::yield_now().await;
    }
    let failure = completed
        .expect("runner Drop reaper did not publish provisional launch cancellation")
        .unwrap_err();
    assert!(matches!(
        failure.primary.error,
        CalcFlowError::Cancelled { .. }
    ));
    assert_eq!(provisional.state.lock().owner, DriverOwnership::Terminal);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn job_drop_then_runner_drop_transfers_and_publishes_once() {
    let runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    let sink = LifecycleProbe::default();
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source.clone(),
            sink.clone(),
        ))
        .await
        .unwrap();
    let core = Arc::clone(&job.core);
    let mut wait = Box::pin(job.wait());

    drop(job);
    assert_eq!(core.state.lock().owner, DriverOwnership::ReaperOwned);
    drop(runner);

    let mut completed = None;
    for _ in 0..100 {
        if let Poll::Ready(outcome) = futures::poll!(wait.as_mut()) {
            completed = Some(outcome);
            break;
        }
        tokio::task::yield_now().await;
    }
    let outcome = completed.expect("combined Drop reaper did not publish the job outcome");
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
    assert_eq!(core.state.lock().owner, DriverOwnership::Terminal);
    assert_eq!(core.metrics.snapshot().job.abandoned_runner_drops, 1);
    assert_eq!(core.metrics.snapshot().job.reaper_joins, 1);
    assert_eq!(source.closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn explicit_cancel_wins_over_an_unobserved_graceful_request() {
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            LifecycleProbe::default(),
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();

    let graceful = job.shutdown();
    let cancelled = job.cancel().await;
    drop(graceful);

    assert_eq!(cancelled.state, ContinuousJobState::Cancelled);
    assert_eq!(cancelled.cause, TerminalCause::ExplicitCancel);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn deadline_preempts_graceful_drain_with_pending_operator_and_source() {
    let entered = Arc::new(AtomicBool::new(false));
    let plan = deadline_pending_plan(Arc::clone(&entered));
    let polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let writes = Arc::new(Mutex::new(Vec::new()));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            84,
            plan.fingerprint(),
            JsonMap::new(),
            Some(chrono::Utc::now() + chrono::Duration::seconds(5)),
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(1, &polls, &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "recording".into(),
            binding: OrdinarySinkBinding::new(Box::new(OrderedRecordingSink {
                id: "recording".into(),
                writes: Arc::clone(&writes),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let job = runner.start(spec).await.unwrap();
    for _ in 0..100 {
        if entered.load(Ordering::SeqCst) && polls.load(Ordering::SeqCst) >= 2 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(entered.load(Ordering::SeqCst));
    assert!(polls.load(Ordering::SeqCst) >= 2);

    let shutdown = job.shutdown();
    assert_eq!(job.state(), ContinuousJobState::Draining);
    tokio::time::advance(StdDuration::from_secs(6)).await;
    let outcome = tokio::time::timeout(StdDuration::from_secs(1), shutdown)
        .await
        .expect("deadline must interrupt a graceful drain blocked in an operator handler");

    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(outcome.cause, TerminalCause::DeadlineExceeded);
    assert!(writes.lock().is_empty());
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn deadline_preempts_graceful_drain_with_pending_sink_and_source() {
    let plan = unary_expression_plan();
    let polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_started = Arc::new(AtomicBool::new(false));
    let sink_gate = Arc::new(Notify::new());
    let writes = Arc::new(Mutex::new(Vec::new()));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            85,
            plan.fingerprint(),
            JsonMap::new(),
            Some(chrono::Utc::now() + chrono::Duration::seconds(5)),
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(1, &polls, &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "gated".into(),
            binding: OrdinarySinkBinding::new(Box::new(GatedSink {
                started: Arc::clone(&sink_started),
                gate: sink_gate,
                writes: Arc::clone(&writes),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let job = runner.start(spec).await.unwrap();
    for _ in 0..100 {
        if sink_started.load(Ordering::SeqCst) && polls.load(Ordering::SeqCst) >= 2 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(sink_started.load(Ordering::SeqCst));
    assert!(polls.load(Ordering::SeqCst) >= 2);

    let shutdown = job.shutdown();
    assert_eq!(job.state(), ContinuousJobState::Draining);
    tokio::time::advance(StdDuration::from_secs(6)).await;
    let outcome = tokio::time::timeout(StdDuration::from_secs(1), shutdown)
        .await
        .expect("deadline must interrupt a graceful drain blocked in a sink write");

    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(outcome.cause, TerminalCause::DeadlineExceeded);
    assert!(writes.lock().is_empty());
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn finite_source_eof_drives_the_registered_graph_to_natural_completion() {
    let mut runner = ContinuousRunner::new();
    let source = LifecycleProbe::default();
    source.finite.store(true, Ordering::SeqCst);
    let job = runner
        .start(spec(
            false,
            Arc::new(AtomicUsize::new(0)),
            source,
            LifecycleProbe::default(),
        ))
        .await
        .unwrap();

    let outcome = tokio::time::timeout(std::time::Duration::from_millis(50), job.wait())
        .await
        .expect("finite source graph did not converge after explicit EOF");

    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[allow(
    clippy::too_many_lines,
    reason = "the end-to-end scenario keeps setup, lifecycle, and status assertions together"
)]
#[tokio::test]
async fn two_sources_union_expression_and_ordered_sinks_complete_end_to_end() {
    let union = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap();
    let expression =
        ExpressionOperator::new("calc", "plus_one = value + 1", Vec::new(), None, Vec::new())
            .unwrap();
    let plan = PipelineBuilder::new("e2e")
        .unwrap()
        .add_node("merge", Box::new(union))
        .unwrap()
        .add_node("calc", Box::new(expression))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("merge", "output").unwrap(),
            PortEndpoint::new("calc", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let left_closed = Arc::new(AtomicUsize::new(0));
    let right_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let writes = Arc::new(Mutex::new(Vec::new()));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            77,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![
            NamedSourceBinding {
                binding_id: "left".into(),
                binding: finite_binding(&[1, 2], &left_closed),
            },
            NamedSourceBinding {
                binding_id: "right".into(),
                binding: finite_binding(&[10, 20], &right_closed),
            },
        ],
        sinks: ["first", "second"]
            .into_iter()
            .map(|sink_id| NamedSinkBinding {
                output_id: "output".into(),
                sink_id: sink_id.into(),
                binding: OrdinarySinkBinding::new(Box::new(OrderedRecordingSink {
                    id: sink_id.into(),
                    writes: Arc::clone(&writes),
                    closed: Arc::clone(&sink_closed),
                })),
            })
            .collect(),
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let job = runner.start(spec).await.unwrap();

    let outcome = job.wait().await;

    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    assert!(outcome.errors.is_empty());
    let status = job.status();
    assert_eq!(status.job_id, job.id());
    assert_eq!(status.state, ContinuousJobState::Completed);
    assert_eq!(status.terminal_cause, Some(TerminalCause::NaturalEnd));
    assert!(status.tasks.is_empty());
    assert_eq!(status.edges.len(), 4);
    assert!(status.edges.values().all(|edge| {
        edge.queue_depth == 0 && edge.charged_rows == 0 && edge.charged_bytes == 0
    }));
    assert!(
        status
            .edges
            .values()
            .all(|edge| { edge.high_water_rows <= 1 && edge.high_water_bytes <= (1 << 20) })
    );
    for (edge_id, batches) in [
        ("source/6c656674/6d65726765/6c656674", 2),
        ("source/7269676874/6d65726765/7269676874", 2),
        ("merge.output->calc.input", 4),
        ("sink/63616c63/6f7574707574/6f7574707574", 4),
    ] {
        let edge = &status.metrics.edges[edge_id];
        assert_eq!(edge.input_batches, batches, "enqueue boundary {edge_id}");
        assert_eq!(edge.output_batches, batches, "dequeue boundary {edge_id}");
    }
    for source_id in ["left", "right"] {
        assert_eq!(
            status.sources[source_id].latest_observed_order,
            Some(vec![2])
        );
        assert_eq!(status.sources[source_id].durable_order, None);
        assert_eq!(status.sources[source_id].next_sequence, Some(2));
        assert!(status.sources[source_id].ended);
        assert_eq!(status.metrics.sources[source_id].poll_count, 3);
        assert_eq!(status.metrics.sources[source_id].data_batches, 2);
        assert_eq!(
            status.metrics.sources[source_id].fully_fanned_out_batches,
            2
        );
    }
    for node_id in ["merge", "calc"] {
        assert_eq!(status.nodes[node_id].input_batches, 4);
        assert_eq!(status.nodes[node_id].fully_fanned_out_batches, 4);
        assert!(status.nodes[node_id].ended);
        assert_eq!(status.metrics.nodes[node_id].input_batches, 4);
        assert_eq!(status.metrics.nodes[node_id].fully_fanned_out_batches, 4);
    }
    for sink_id in ["first", "second"] {
        let metric_id = super::sink_metric_id("output", sink_id);
        assert_eq!(status.sinks[&metric_id].delivered_batches, 4);
        assert!(status.sinks[&metric_id].ended);
        assert_eq!(status.metrics.sinks[&metric_id].delivered_batches, 4);
    }
    assert_eq!(
        status.metrics.job.terminal_state,
        Some(ContinuousJobState::Completed)
    );
    assert_eq!(
        status.metrics.job.terminal_cause,
        Some(TerminalCause::NaturalEnd)
    );
    assert!(!format!("{status:?}").contains("secret-canary"));
    let writes = writes.lock().clone();
    assert_eq!(writes.len(), 8);
    for pair in writes.chunks_exact(2) {
        assert_eq!(pair[0].0, "first");
        assert_eq!(pair[1].0, "second");
        assert_eq!((&pair[0].1, pair[0].2), (&pair[1].1, pair[1].2));
    }
    for source in ["left", "right"] {
        let sequence = writes
            .iter()
            .filter(|(sink, observed_source, _)| sink == "first" && observed_source == source)
            .map(|(_, _, sequence)| *sequence)
            .collect::<Vec<_>>();
        assert_eq!(sequence, [0, 1]);
    }
    assert_eq!(left_closed.load(Ordering::SeqCst), 1);
    assert_eq!(right_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 2);
    assert_eq!(runner.registry_counts(), (0, 0));
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn two_sources_reopen_once_at_their_distinct_resume_cursors() {
    let plan = union_plan();
    let left_resume = Cursor::new("left", vec![1], JsonMap::new()).unwrap();
    let right_resume = Cursor::new("right", vec![2], JsonMap::new()).unwrap();
    let left_opened = Arc::new(Mutex::new(Vec::new()));
    let right_opened = Arc::new(Mutex::new(Vec::new()));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink = LifecycleProbe::default();
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            92,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![
            NamedSourceBinding {
                binding_id: "left".into(),
                binding: SourceBinding::new(
                    Box::new(ResumeProbeSource {
                        opened_with: Arc::clone(&left_opened),
                        closed: Arc::clone(&source_closed),
                    }),
                    Some(left_resume.clone()),
                    3,
                )
                .unwrap(),
            },
            NamedSourceBinding {
                binding_id: "right".into(),
                binding: SourceBinding::new(
                    Box::new(ResumeProbeSource {
                        opened_with: Arc::clone(&right_opened),
                        closed: Arc::clone(&source_closed),
                    }),
                    Some(right_resume.clone()),
                    7,
                )
                .unwrap(),
            },
        ],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(sink.clone()))),
        }],
        edge_budget: EdgeBudget::default(),
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let job = runner.start(spec).await.unwrap();

    let outcome = job.wait().await;

    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    assert_eq!(&*left_opened.lock(), &[Some(left_resume)]);
    assert_eq!(&*right_opened.lock(), &[Some(right_resume)]);
    assert_eq!(source_closed.load(Ordering::SeqCst), 2);
    assert_eq!(sink.closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn slow_sink_backpressures_both_sources_after_bounded_prefetch() {
    let plan = union_plan();
    let left_polls = Arc::new(AtomicUsize::new(0));
    let right_polls = Arc::new(AtomicUsize::new(0));
    let left_closed = Arc::new(AtomicUsize::new(0));
    let right_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_started = Arc::new(AtomicBool::new(false));
    let sink_gate = Arc::new(Notify::new());
    let writes = Arc::new(Mutex::new(Vec::new()));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            81,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![
            NamedSourceBinding {
                binding_id: "left".into(),
                binding: counting_pending_binding(100, &left_polls, &left_closed),
            },
            NamedSourceBinding {
                binding_id: "right".into(),
                binding: counting_pending_binding(100, &right_polls, &right_closed),
            },
        ],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "slow".into(),
            binding: OrdinarySinkBinding::new(Box::new(GatedSink {
                started: Arc::clone(&sink_started),
                gate: Arc::clone(&sink_gate),
                writes,
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let job = runner.start(spec).await.unwrap();
    while !sink_started.load(Ordering::SeqCst) {
        tokio::task::yield_now().await;
    }
    for _ in 0..200 {
        tokio::task::yield_now().await;
    }
    let stopped = (
        left_polls.load(Ordering::SeqCst),
        right_polls.load(Ordering::SeqCst),
    );
    for _ in 0..200 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        stopped,
        (
            left_polls.load(Ordering::SeqCst),
            right_polls.load(Ordering::SeqCst),
        )
    );
    assert!(
        stopped.0 < 100 && stopped.1 < 100,
        "polls did not stop: {stopped:?}"
    );

    let outcome = job.cancel().await;
    assert_eq!(outcome.cause, TerminalCause::ExplicitCancel);
    assert_eq!(left_closed.load(Ordering::SeqCst), 1);
    assert_eq!(right_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn terminal_metrics_overflow_keeps_operator_primary_and_publishes_once() {
    let resets = Arc::new(AtomicUsize::new(0));
    let operator = ResetOperator {
        inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
        outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
        resets,
        fail_reset: false,
        panic_reset: false,
    };
    let plan = PipelineBuilder::new("metrics-overflow")
        .unwrap()
        .add_node("node", Box::new(operator) as Box<dyn StreamOperator>)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let release = Arc::new(Notify::new());
    let source_closed = Arc::new(AtomicUsize::new(0));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            91,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: SourceBinding::new(
                Box::new(GatedDataSource {
                    release: Arc::clone(&release),
                    delivered: false,
                    closed: Arc::clone(&source_closed),
                }),
                None,
                0,
            )
            .unwrap(),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink(LifecycleProbe::default()))),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let job = runner.start(spec).await.unwrap();
    job.core.metrics.preset_job_task_errors_for_test(u64::MAX);
    release.notify_one();

    let outcome = job.wait().await;

    assert_eq!(outcome.state, ContinuousJobState::Failed);
    assert_eq!(
        outcome
            .errors
            .iter()
            .filter(|failure| matches!(failure.origin, super::FailureOrigin::Metrics { .. }))
            .count(),
        1,
        "errors: {:?}",
        outcome.errors
    );
    assert!(matches!(
        outcome.errors[0].error,
        CalcFlowError::Operator { ref message, .. } if message == "data failure"
    ));
    assert!(matches!(
        outcome.errors.last().unwrap().origin,
        super::FailureOrigin::Metrics {
            counter: "task_errors",
            ..
        }
    ));
    let status = job.status();
    assert_eq!(status.metrics.job.task_errors, u64::MAX);
    assert!(status.metrics.job.metrics_overflowed);
    assert!(status.tasks.is_empty());
    assert!(status.edges.values().all(|edge| edge.queue_depth == 0));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn graceful_drains_the_accepted_prefix_while_cancel_makes_no_drain_promise() {
    async fn run_case(graceful: bool) -> (Arc<super::ContinuousJobOutcome>, Vec<(String, u64)>) {
        let plan = unary_expression_plan();
        let polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let sink_started = Arc::new(AtomicBool::new(false));
        let sink_gate = Arc::new(Notify::new());
        let writes = Arc::new(Mutex::new(Vec::new()));
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                if graceful { 82 } else { 83 },
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(1, &polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "gated".into(),
                binding: OrdinarySinkBinding::new(Box::new(GatedSink {
                    started: Arc::clone(&sink_started),
                    gate: Arc::clone(&sink_gate),
                    writes: Arc::clone(&writes),
                    closed: Arc::clone(&sink_closed),
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
            static_inputs: crate::static_input::PreparedStaticInputs::default(),
        };
        let mut runner = ContinuousRunner::new();
        let job = runner.start(spec).await.unwrap();
        while !sink_started.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
        let outcome = if graceful {
            let observer = job.shutdown();
            sink_gate.notify_one();
            observer.await
        } else {
            job.cancel().await
        };
        assert_eq!(source_closed.load(Ordering::SeqCst), 1);
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
        drop(job);
        runner.shutdown().await.unwrap();
        let observed = writes.lock().clone();
        (outcome, observed)
    }

    let (graceful, drained) = run_case(true).await;
    assert_eq!(graceful.state, ContinuousJobState::Completed);
    assert_eq!(graceful.cause, TerminalCause::GracefulShutdown);
    assert_eq!(drained, [("input".into(), 0)]);

    let (cancelled, not_drained) = run_case(false).await;
    assert_eq!(cancelled.state, ContinuousJobState::Cancelled);
    assert_eq!(cancelled.cause, TerminalCause::ExplicitCancel);
    assert!(not_drained.is_empty());
}

#[tokio::test]
async fn checkpointed_runner_opens_an_empty_lineage_only_after_preflight() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let plan = PipelineBuilder::new("checkpoint-empty")
        .unwrap()
        .add_checkpoint_capable_node(
            "node",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_opened = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            901,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(0, &source_polls, &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(CheckpointProbeSink {
                opened: Arc::clone(&sink_opened),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let checkpoint = CheckpointRuntimeSpec::new(
        backend.clone(),
        directory.path().join("manifests"),
        StreamRuntimeConfig::default(),
    )
    .unwrap();
    let mut runner = ContinuousRunner::new();

    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();

    assert_eq!(sink_opened.load(Ordering::SeqCst), 1);
    wait_for_counter(&source_polls, 1).await;
    let outcome = job.cancel().await;
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
    assert!(directory.path().join("manifests").is_dir());
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the no-manifest recovery oracle owns the orphan, retry, and lifecycle assertions"
)]
async fn checkpointed_runner_cleans_pre_manifest_state_before_retry() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let plan = PipelineBuilder::new("checkpoint-pre-manifest-cleanup")
        .unwrap()
        .add_checkpoint_capable_node(
            "node",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    let lineage_key = StateLineageKey::new(plan.name(), plan.fingerprint()).unwrap();
    let orphan_bytes = b"published-before-manifest";
    let abandoned_bytes = b"staged-before-manifest";
    let retry_bytes = b"retry-at-same-coordinate";
    let orphan = local_state_handle(
        &lineage_key,
        "node",
        crate::Epoch::INITIAL,
        "published",
        orphan_bytes,
    );
    let orphan_retry = local_state_handle(
        &lineage_key,
        "node",
        crate::Epoch::INITIAL,
        "published",
        retry_bytes,
    );
    let abandoned = local_state_handle(
        &lineage_key,
        "node",
        crate::Epoch::INITIAL,
        "staged",
        abandoned_bytes,
    );
    let abandoned_retry = local_state_handle(
        &lineage_key,
        "node",
        crate::Epoch::INITIAL,
        "staged",
        retry_bytes,
    );
    {
        let lineage = backend.open_lineage(&lineage_key).await.unwrap();
        lineage.stage_segment(&orphan, orphan_bytes).await.unwrap();
        lineage.validate_segment(&orphan).await.unwrap();
        lineage.publish_segment(&orphan).await.unwrap();
        lineage
            .stage_segment(&abandoned, abandoned_bytes)
            .await
            .unwrap();
    }
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_opened = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            902,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(0, &source_polls, &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(CheckpointProbeSink {
                opened: Arc::clone(&sink_opened),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let checkpoint = CheckpointRuntimeSpec::new(
        backend.clone(),
        directory.path().join("manifests"),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap();
    let mut runner = ContinuousRunner::new();

    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();

    wait_for_counter(&source_polls, 1).await;
    assert_eq!(job.status().metrics.checkpoints.orphan_segments_removed, 1);
    assert_eq!(job.status().checkpoint.unwrap().last_completed_epoch, None);
    let outcome = job.cancel().await;
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_opened.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);

    let lineage = backend.open_lineage(&lineage_key).await.unwrap();
    assert!(matches!(
        lineage.load_segment(&orphan).await,
        Err(CalcFlowError::NotFound { .. })
    ));
    lineage
        .stage_segment(&orphan_retry, retry_bytes)
        .await
        .unwrap();
    lineage.validate_segment(&orphan_retry).await.unwrap();
    lineage.publish_segment(&orphan_retry).await.unwrap();
    lineage
        .stage_segment(&abandoned_retry, retry_bytes)
        .await
        .unwrap();
    lineage.validate_segment(&abandoned_retry).await.unwrap();
    lineage.publish_segment(&abandoned_retry).await.unwrap();
    assert_eq!(
        lineage.load_segment(&orphan_retry).await.unwrap(),
        retry_bytes
    );
    assert_eq!(
        lineage.load_segment(&abandoned_retry).await.unwrap(),
        retry_bytes
    );
}

#[tokio::test]
async fn exactly_once_start_without_checkpoint_fails_before_lifecycle_work() {
    let plan = PipelineBuilder::new("checkpoint-required")
        .unwrap()
        .add_checkpoint_capable_node(
            "node",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_opened = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            906,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(0, &source_polls, &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(CheckpointProbeSink {
                opened: Arc::clone(&sink_opened),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();

    let failure = match runner.start(spec).await {
        Ok(_) => panic!("exactly-once start unexpectedly ran without checkpoint ownership"),
        Err(failure) => failure,
    };

    assert!(matches!(
        &failure.primary.error,
        CalcFlowError::InvalidArgument { field, message }
            if field == "requirements.delivery.output"
                && message.contains("checkpoint runtime")
    ));
    assert_eq!(source_polls.load(Ordering::SeqCst), 0);
    assert_eq!(source_closed.load(Ordering::SeqCst), 0);
    assert_eq!(sink_opened.load(Ordering::SeqCst), 0);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 0);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the collision preflight assertion owns both output bindings and lifecycle probes"
)]
async fn checkpointed_start_rejects_cross_output_sink_id_collisions_before_open() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let plan = PipelineBuilder::new("checkpoint-sink-collision")
        .unwrap()
        .add_checkpoint_capable_node(
            "root",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .add_checkpoint_capable_node(
            "branch_a",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .add_checkpoint_capable_node(
            "branch_b",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("root", "output").unwrap(),
            PortEndpoint::new("branch_a", "input").unwrap(),
        ))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("root", "output").unwrap(),
            PortEndpoint::new("branch_b", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([
                    (
                        "branch_a.output".into(),
                        crate::DeliveryGuarantee::ExactlyOnce,
                    ),
                    (
                        "branch_b.output".into(),
                        crate::DeliveryGuarantee::ExactlyOnce,
                    ),
                ]),
            },
        )
        .unwrap();
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_opened = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            907,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(0, &source_polls, &source_closed),
        }],
        sinks: ["branch_a.output", "branch_b.output"]
            .into_iter()
            .map(|output_id| NamedSinkBinding {
                output_id: output_id.into(),
                sink_id: "duplicate".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(CheckpointProbeSink {
                    opened: Arc::clone(&sink_opened),
                    closed: Arc::clone(&sink_closed),
                })),
            })
            .collect(),
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let checkpoint = CheckpointRuntimeSpec::new(
        backend,
        directory.path().join("manifests"),
        StreamRuntimeConfig::default(),
    )
    .unwrap();
    let mut runner = ContinuousRunner::new();

    let failure = match runner.start_checkpointed(spec, checkpoint).await {
        Ok(_) => panic!("duplicate sink IDs unexpectedly entered lifecycle work"),
        Err(failure) => failure,
    };

    assert!(
        matches!(
            &failure.primary.error,
            CalcFlowError::InvalidArgument { field, message }
                if field == "sinks.duplicate" && message.contains("more than one output")
        ),
        "unexpected preflight error: {:?}",
        failure.primary.error
    );
    assert_eq!(source_polls.load(Ordering::SeqCst), 0);
    assert_eq!(source_closed.load(Ordering::SeqCst), 0);
    assert_eq!(sink_opened.load(Ordering::SeqCst), 0);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 0);
    assert!(!directory.path().join("manifests").exists());
    runner.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
#[allow(
    clippy::too_many_lines,
    reason = "the periodic lifecycle and bounded status snapshot are one integration contract"
)]
async fn checkpointed_runner_periodically_publishes_before_sink_commit() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let plan = PipelineBuilder::new("checkpoint-periodic")
        .unwrap()
        .add_checkpoint_capable_node(
            "node",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let log = Arc::new(Mutex::new(Vec::new()));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            903,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(0, &source_polls, &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&log),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let checkpoint = CheckpointRuntimeSpec::new(
        backend,
        directory.path().join("manifests"),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_millis(10),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap();
    let mut runner = ContinuousRunner::new();
    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
    wait_for_counter(&source_polls, 1).await;

    tokio::time::timeout(StdDuration::from_secs(5), async {
        loop {
            if log.lock().iter().any(|entry| entry == "sink-begin:2") {
                break;
            }
            tokio::time::sleep(StdDuration::from_millis(1)).await;
        }
    })
    .await
    .expect("periodic checkpoint should complete");

    assert_eq!(
        &*log.lock(),
        &[
            "sink-open",
            "sink-begin:1",
            "sink-precommit:1",
            "sink-commit:1",
            "sink-begin:2",
        ]
    );
    let checkpoint_status = job
        .status()
        .checkpoint
        .expect("checkpointed jobs expose bounded checkpoint status");
    assert_eq!(checkpoint_status.current_epoch, None);
    assert_eq!(
        checkpoint_status.last_completed_epoch,
        Some(crate::Epoch::INITIAL)
    );
    assert_eq!(checkpoint_status.failure_category, None);
    assert!(!checkpoint_status.runtime_config_changed);
    assert_eq!(checkpoint_status.expected_sources, 1);
    assert_eq!(checkpoint_status.expected_operators, 1);
    assert_eq!(checkpoint_status.expected_sinks, 1);
    let checkpoint_metrics = job.status().metrics.checkpoints;
    assert_eq!(checkpoint_metrics.requested, 1);
    assert_eq!(checkpoint_metrics.completed, 1);
    assert_eq!(checkpoint_metrics.failed, 0);
    assert_eq!(checkpoint_metrics.terminal_completed, 0);
    assert!(checkpoint_metrics.manifest_bytes > 0);
    assert!(
        checkpoint_metrics
            .phase_duration
            .contains_key(&CheckpointPhase::ManifestDurable)
    );
    let outcome = job.cancel().await;
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn manual_checkpoint_returns_only_after_durable_manifest_and_sink_commit() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let plan = PipelineBuilder::new("checkpoint-manual")
        .unwrap()
        .add_checkpoint_capable_node(
            "node",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let log = Arc::new(Mutex::new(Vec::new()));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            913,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(0, &source_polls, &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&log),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let checkpoint = CheckpointRuntimeSpec::new(
        backend,
        directory.path().join("manifests"),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap();
    let mut runner = ContinuousRunner::new();
    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
    wait_for_counter(&source_polls, 1).await;

    let epoch = tokio::time::timeout(StdDuration::from_secs(5), job.trigger_checkpoint())
        .await
        .expect("manual checkpoint should not hang")
        .unwrap();

    assert_eq!(epoch, crate::Epoch::INITIAL);
    assert!(
        directory
            .path()
            .join("manifests/manifest-00000000000000000001.json")
            .is_file()
    );
    assert_eq!(
        &*log.lock(),
        &[
            "sink-open",
            "sink-begin:1",
            "sink-precommit:1",
            "sink-commit:1",
            "sink-begin:2",
        ]
    );

    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "post-manifest recovery and the next manual epoch are one restart contract"
)]
async fn post_manifest_manual_commit_failure_requires_recovery_and_continues_epoch() {
    let directory = tempfile::tempdir().unwrap();
    let manifest_root = directory.path().join("manifests");
    let fail_commit = Arc::new(AtomicBool::new(true));
    let log = Arc::new(Mutex::new(Vec::new()));
    let plan = || {
        PipelineBuilder::new("checkpoint-manual-recovery")
            .unwrap()
            .add_checkpoint_capable_node(
                "node",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements {
                    delivery: BTreeMap::from([(
                        "output".into(),
                        crate::DeliveryGuarantee::ExactlyOnce,
                    )]),
                },
            )
            .unwrap()
    };
    let checkpoint = || {
        CheckpointRuntimeSpec::managed(
            ManagedCheckpointRuntime::new(directory.path()).unwrap(),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
    };
    let spec = |job_id, source_polls: &Arc<AtomicUsize>, source_closed, sink_closed| {
        let plan = plan();
        ContinuousJobSpec {
            context: StreamJobContext::new(
                job_id,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: counting_pending_binding(0, source_polls, &source_closed),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(FailOnceCommitSink {
                    fail_commit: Arc::clone(&fail_commit),
                    log: Arc::clone(&log),
                    closed: sink_closed,
                })),
            }],
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1 << 20,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
            static_inputs: crate::static_input::PreparedStaticInputs::default(),
        }
    };
    let first_source_polls = Arc::new(AtomicUsize::new(0));
    let first_source_closed = Arc::new(AtomicUsize::new(0));
    let first_sink_closed = Arc::new(AtomicUsize::new(0));
    let mut first_runner = ContinuousRunner::new();
    let first_job = first_runner
        .start_checkpointed(
            spec(
                914,
                &first_source_polls,
                Arc::clone(&first_source_closed),
                Arc::clone(&first_sink_closed),
            ),
            checkpoint(),
        )
        .await
        .unwrap();
    wait_for_counter(&first_source_polls, 1).await;

    let (manual, failed) = tokio::time::timeout(StdDuration::from_secs(5), async {
        tokio::join!(first_job.trigger_checkpoint(), first_job.wait())
    })
    .await
    .expect("post-manifest commit failure should converge");

    let CalcFlowError::Streaming(manual_error) = manual.unwrap_err() else {
        panic!("manual commit failure must use the safe streaming boundary");
    };
    assert_eq!(
        manual_error.category(),
        crate::runtime::streaming::projection::StreamingErrorCategory::Connector
    );
    assert_eq!(manual_error.epoch(), Some(crate::Epoch::INITIAL));
    assert_eq!(
        manual_error.checkpoint_phase(),
        Some(crate::runtime::streaming::projection::CheckpointPhase::ManifestDurable)
    );
    assert_eq!(
        manual_error.component_kind(),
        Some(crate::runtime::streaming::projection::ComponentKind::Sink)
    );
    assert_eq!(manual_error.component_id(), Some("sink"));
    assert_eq!(failed.state, ContinuousJobState::RecoveryRequired);
    assert!(
        manifest_root
            .join("manifest-00000000000000000001.json")
            .is_file()
    );
    assert!(!log.lock().iter().any(|entry| entry == "sink-abort:1"));
    drop(first_job);
    first_runner.shutdown().await.unwrap();
    assert_eq!(first_runner.registry_counts(), (0, 0));
    assert_eq!(first_source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(first_sink_closed.load(Ordering::SeqCst), 1);

    let restart_source_polls = Arc::new(AtomicUsize::new(0));
    let restart_source_closed = Arc::new(AtomicUsize::new(0));
    let restart_sink_closed = Arc::new(AtomicUsize::new(0));
    let mut restart_runner = ContinuousRunner::new();
    let restart_job = restart_runner
        .start_checkpointed(
            spec(
                915,
                &restart_source_polls,
                Arc::clone(&restart_source_closed),
                Arc::clone(&restart_sink_closed),
            ),
            checkpoint(),
        )
        .await
        .unwrap();
    wait_for_counter(&restart_source_polls, 1).await;

    let restarted_epoch =
        tokio::time::timeout(StdDuration::from_secs(5), restart_job.trigger_checkpoint())
            .await
            .expect("restart manual checkpoint should not hang")
            .unwrap();

    assert_eq!(restarted_epoch, crate::Epoch::new(2).unwrap());
    assert!(log.lock().iter().any(|entry| entry == "sink-recover:1"));
    assert!(log.lock().iter().any(|entry| entry == "sink-commit:2"));
    assert_eq!(
        restart_job.cancel().await.state,
        ContinuousJobState::Cancelled
    );
    drop(restart_job);
    restart_runner.shutdown().await.unwrap();
    assert_eq!(restart_runner.registry_counts(), (0, 0));
    assert_eq!(restart_source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(restart_sink_closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancel_fails_inflight_and_queued_manual_requests_without_leaks() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let gate = super::CheckpointStartedTestGate::default();
    let checkpoint = CheckpointRuntimeSpec::new(
        backend,
        directory.path().join("manifests"),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
    .with_started_gate(gate.clone());
    let spec = pending_checkpoint_spec(
        "checkpoint-manual-cancel",
        916,
        &source_polls,
        &source_closed,
        Box::new(CheckpointProbeSink {
            opened: Arc::new(AtomicUsize::new(0)),
            closed: Arc::clone(&sink_closed),
        }),
    );
    let mut runner = ContinuousRunner::new();
    let job = Arc::new(runner.start_checkpointed(spec, checkpoint).await.unwrap());
    wait_for_counter(&source_polls, 1).await;
    let inflight = tokio::spawn({
        let job = Arc::clone(&job);
        async move { job.trigger_checkpoint().await }
    });
    gate.wait_until_entered().await;
    let coordinator = job
        .core
        .manual_checkpoint
        .lock()
        .clone()
        .expect("checkpoint task registered its manual queue");
    let queued = coordinator.request_manual().await.unwrap();

    let cancelled = job.cancel();
    gate.release();
    let outcome = cancelled.await;
    let inflight = inflight.await.unwrap();

    assert!(matches!(inflight, Err(CalcFlowError::Cancelled { .. })));
    assert!(matches!(queued.await, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_terminal_checkpoint_resources_released(&job);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dropped_manual_future_keeps_accepted_request_in_fifo() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let log = Arc::new(Mutex::new(Vec::new()));
    let gate = super::CheckpointStartedTestGate::default();
    let spec = pending_checkpoint_spec(
        "checkpoint-manual-drop",
        920,
        &source_polls,
        &source_closed,
        Box::new(PeriodicCheckpointSink {
            log: Arc::clone(&log),
            closed: Arc::clone(&sink_closed),
        }),
    );
    let checkpoint = CheckpointRuntimeSpec::new(
        backend,
        directory.path().join("manifests"),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
    .with_started_gate(gate.clone());
    let mut runner = ContinuousRunner::new();
    let job = Arc::new(runner.start_checkpointed(spec, checkpoint).await.unwrap());
    wait_for_counter(&source_polls, 1).await;
    let dropped = tokio::spawn({
        let job = Arc::clone(&job);
        async move { job.trigger_checkpoint().await }
    });
    gate.wait_until_entered().await;

    dropped.abort();
    assert!(dropped.await.unwrap_err().is_cancelled());
    gate.release();
    tokio::time::timeout(StdDuration::from_secs(5), async {
        loop {
            if log.lock().iter().any(|entry| entry == "sink-begin:2") {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropped waiter must not cancel its accepted checkpoint");

    assert_eq!(
        job.trigger_checkpoint().await.unwrap(),
        crate::Epoch::new(2).unwrap()
    );
    assert!(log.lock().iter().any(|entry| entry == "sink-commit:1"));
    assert!(log.lock().iter().any(|entry| entry == "sink-commit:2"));
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    assert_terminal_checkpoint_resources_released(&job);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancel_after_manifest_durability_finishes_commit_before_manual_success() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let commit_entered = Arc::new(AtomicBool::new(false));
    let commit_changed = Arc::new(Notify::new());
    let commit_release = Arc::new(Semaphore::new(0));
    let log = Arc::new(Mutex::new(Vec::new()));
    let spec = pending_checkpoint_spec(
        "checkpoint-manual-durable-cancel",
        917,
        &source_polls,
        &source_closed,
        Box::new(BlockingCommitSink {
            commit_entered: Arc::clone(&commit_entered),
            commit_changed: Arc::clone(&commit_changed),
            commit_release: Arc::clone(&commit_release),
            log: Arc::clone(&log),
            closed: Arc::clone(&sink_closed),
        }),
    );
    let checkpoint = CheckpointRuntimeSpec::new(
        backend,
        directory.path().join("manifests"),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap();
    let mut runner = ContinuousRunner::new();
    let job = Arc::new(runner.start_checkpointed(spec, checkpoint).await.unwrap());
    wait_for_counter(&source_polls, 1).await;
    let manual = tokio::spawn({
        let job = Arc::clone(&job);
        async move { job.trigger_checkpoint().await }
    });
    while !commit_entered.load(Ordering::Acquire) {
        let changed = commit_changed.notified();
        if commit_entered.load(Ordering::Acquire) {
            break;
        }
        changed.await;
    }

    let cancelled = job.cancel();
    loop {
        if job.core.state.lock().selected_cause == Some(TerminalCause::ExplicitCancel) {
            break;
        }
        tokio::task::yield_now().await;
    }
    commit_release.add_permits(1);
    let (manual, outcome) = tokio::join!(manual, cancelled);

    assert_eq!(manual.unwrap().unwrap(), crate::Epoch::INITIAL);
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert!(log.lock().iter().any(|entry| entry == "sink-commit:1"));
    assert!(!log.lock().iter().any(|entry| entry == "sink-abort:1"));
    assert_terminal_checkpoint_resources_released(&job);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn shutdown_drains_inflight_and_queued_manual_requests_before_completion() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let gate = super::CheckpointStartedTestGate::default();
    let spec = pending_checkpoint_spec(
        "checkpoint-manual-shutdown",
        918,
        &source_polls,
        &source_closed,
        Box::new(CheckpointProbeSink {
            opened: Arc::new(AtomicUsize::new(0)),
            closed: Arc::clone(&sink_closed),
        }),
    );
    let checkpoint = CheckpointRuntimeSpec::new(
        backend,
        directory.path().join("manifests"),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
    .with_started_gate(gate.clone());
    let mut runner = ContinuousRunner::new();
    let job = Arc::new(runner.start_checkpointed(spec, checkpoint).await.unwrap());
    wait_for_counter(&source_polls, 1).await;
    let inflight = tokio::spawn({
        let job = Arc::clone(&job);
        async move { job.trigger_checkpoint().await }
    });
    gate.wait_until_entered().await;
    let coordinator = job
        .core
        .manual_checkpoint
        .lock()
        .clone()
        .expect("checkpoint task registered its manual queue");
    let queued = coordinator.request_manual().await.unwrap();

    let shutdown = job.shutdown();
    gate.release();
    let (inflight, queued, outcome) = tokio::join!(inflight, queued, shutdown);

    assert_eq!(inflight.unwrap().unwrap(), crate::Epoch::INITIAL);
    assert_eq!(queued.unwrap(), crate::Epoch::new(2).unwrap());
    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(outcome.cause, TerminalCause::GracefulShutdown);
    assert!(
        directory
            .path()
            .join("manifests/manifest-00000000000000000001.json")
            .is_file()
    );
    assert!(
        directory
            .path()
            .join("manifests/manifest-00000000000000000002.json")
            .is_file()
    );
    assert_terminal_checkpoint_resources_released(&job);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn manual_request_queued_during_periodic_checkpoint_receives_next_epoch() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let log = Arc::new(Mutex::new(Vec::new()));
    let gate = super::CheckpointStartedTestGate::default();
    let spec = pending_checkpoint_spec(
        "checkpoint-periodic-manual-race",
        919,
        &source_polls,
        &source_closed,
        Box::new(PeriodicCheckpointSink {
            log: Arc::clone(&log),
            closed: Arc::clone(&sink_closed),
        }),
    );
    let checkpoint = CheckpointRuntimeSpec::new(
        backend,
        directory.path().join("manifests"),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_millis(10),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
    .with_started_gate(gate.clone());
    let mut runner = ContinuousRunner::new();
    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
    wait_for_counter(&source_polls, 1).await;

    tokio::time::advance(StdDuration::from_millis(10)).await;
    gate.wait_until_entered().await;
    let coordinator = job
        .core
        .manual_checkpoint
        .lock()
        .clone()
        .expect("checkpoint task registered its manual queue");
    let manual = coordinator.request_manual().await.unwrap();
    gate.release();

    assert_eq!(manual.await.unwrap(), crate::Epoch::new(2).unwrap());
    assert!(log.lock().iter().any(|entry| entry == "sink-commit:1"));
    assert!(log.lock().iter().any(|entry| entry == "sink-commit:2"));
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(runner.registry_counts(), (0, 0));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn manual_checkpoint_faults_before_durability_fail_without_manifest_or_leaks() {
    for (name, point) in [
        ("barrier", super::CheckpointFaultPoint::SourceCut),
        ("precommit", super::CheckpointFaultPoint::SinkPreCommit),
        ("manifest", super::CheckpointFaultPoint::ManifestWrite),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            LocalStateBackend::new(directory.path().join("state"))
                .await
                .unwrap(),
        );
        let source_polls = Arc::new(AtomicUsize::new(0));
        let source_closed = Arc::new(AtomicUsize::new(0));
        let sink_closed = Arc::new(AtomicUsize::new(0));
        let log = Arc::new(Mutex::new(Vec::new()));
        let spec = pending_checkpoint_spec(
            &format!("checkpoint-manual-{name}-fault"),
            930,
            &source_polls,
            &source_closed,
            Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&log),
                closed: Arc::clone(&sink_closed),
            }),
        );
        let checkpoint = CheckpointRuntimeSpec::new(
            backend,
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
        .with_fault(point, super::CheckpointFaultMode::Io);
        let mut runner = ContinuousRunner::new();
        let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
        wait_for_counter(&source_polls, 1).await;

        let (manual, outcome) = tokio::time::timeout(StdDuration::from_secs(5), async {
            tokio::join!(job.trigger_checkpoint(), job.wait())
        })
        .await
        .unwrap_or_else(|error| panic!("{name} fault did not converge: {error}"));

        let CalcFlowError::Streaming(error) = manual.unwrap_err() else {
            panic!("{name} manual failure must use the streaming boundary");
        };
        assert_eq!(
            error.category(),
            crate::runtime::streaming::projection::StreamingErrorCategory::Io,
            "{name}"
        );
        assert_eq!(error.epoch(), Some(crate::Epoch::INITIAL), "{name}");
        assert_eq!(outcome.state, ContinuousJobState::Failed, "{name}");
        assert!(
            !directory
                .path()
                .join("manifests/manifest-00000000000000000001.json")
                .exists(),
            "{name} fault published a manifest"
        );
        if name != "barrier" {
            assert!(
                log.lock().iter().any(|entry| entry == "sink-abort:1"),
                "{name} fault left a prepared sink without a manifest"
            );
        }
        drop(job);
        runner.shutdown().await.unwrap();
        assert_eq!(runner.registry_counts(), (0, 0), "{name}");
        assert_eq!(source_closed.load(Ordering::SeqCst), 1, "{name}");
        assert_eq!(sink_closed.load(Ordering::SeqCst), 1, "{name}");
    }
}

#[tokio::test]
async fn installed_manifest_with_unknown_durability_requires_recovery() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_log = Arc::new(Mutex::new(Vec::new()));
    let spec = pending_checkpoint_spec(
        "checkpoint-manifest-installed-unknown",
        931,
        &source_polls,
        &source_closed,
        Box::new(PeriodicCheckpointSink {
            log: Arc::clone(&sink_log),
            closed: Arc::clone(&sink_closed),
        }),
    );
    let checkpoint = CheckpointRuntimeSpec::new(
        backend,
        directory.path().join("manifests"),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
    .with_fault(
        super::CheckpointFaultPoint::ManifestRename,
        super::CheckpointFaultMode::Io,
    );
    let mut runner = ContinuousRunner::new();
    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
    wait_for_counter(&source_polls, 1).await;

    let (manual, outcome) = tokio::time::timeout(StdDuration::from_secs(5), async {
        tokio::join!(job.trigger_checkpoint(), job.wait())
    })
    .await
    .expect("indeterminate publication should converge");

    assert!(matches!(
        manual,
        Err(CalcFlowError::RecoveryRequired { .. })
    ));
    assert_eq!(outcome.state, ContinuousJobState::RecoveryRequired);
    let status = job.public_status();
    assert_eq!(
        status.state,
        crate::runtime::streaming::projection::JobState::RecoveryRequired
    );
    assert_eq!(
        status.checkpoint.installed_unknown_epoch,
        Some(crate::Epoch::INITIAL)
    );
    assert_eq!(status.checkpoint.last_completed_epoch, None);
    assert_eq!(
        status.checkpoint.phase,
        Some(crate::runtime::streaming::projection::CheckpointPhase::ManifestInstalled)
    );
    assert_eq!(
            status.checkpoint.failure_category,
            Some(
                crate::runtime::streaming::projection::StreamingErrorCategory::CheckpointPublicationUnknown
            )
        );
    let internal_status = job.status();
    let public_outcome = crate::runtime::streaming::projection::project_job_outcome(
        job.id(),
        &outcome,
        internal_status.checkpoint.as_ref(),
        None,
    );
    assert_eq!(public_outcome.errors.len(), 1);
    assert_eq!(
        public_outcome.errors[0].category(),
        crate::runtime::streaming::projection::StreamingErrorCategory::CheckpointPublicationUnknown
    );
    assert_eq!(
        public_outcome.errors[0].epoch(),
        Some(crate::Epoch::INITIAL)
    );
    assert!(!sink_log.lock().iter().any(|entry| entry == "sink-commit:1"));
    assert!(!sink_log.lock().iter().any(|entry| entry == "sink-abort:1"));
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn installed_unknown_publication_wins_over_concurrent_cancellation() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_log = Arc::new(Mutex::new(Vec::new()));
    let spec = pending_checkpoint_spec(
        "checkpoint-manifest-installed-cancelled",
        932,
        &source_polls,
        &source_closed,
        Box::new(PeriodicCheckpointSink {
            log: Arc::clone(&sink_log),
            closed: Arc::clone(&sink_closed),
        }),
    );
    let checkpoint = CheckpointRuntimeSpec::new(
        backend,
        directory.path().join("manifests"),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
    .with_fault(
        super::CheckpointFaultPoint::ManifestRename,
        super::CheckpointFaultMode::Cancel,
    );
    let mut runner = ContinuousRunner::new();
    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
    wait_for_counter(&source_polls, 1).await;

    let (manual, outcome) = tokio::time::timeout(StdDuration::from_secs(5), async {
        tokio::join!(job.trigger_checkpoint(), job.wait())
    })
    .await
    .expect("publication/cancellation overlap should converge");

    assert!(matches!(
        manual,
        Err(CalcFlowError::RecoveryRequired { .. })
    ));
    assert_eq!(outcome.state, ContinuousJobState::RecoveryRequired);
    let status = job.public_status();
    assert_eq!(
        status.checkpoint.installed_unknown_epoch,
        Some(crate::Epoch::INITIAL)
    );
    assert_eq!(status.checkpoint.last_completed_epoch, None);
    assert!(!sink_log.lock().iter().any(|entry| entry == "sink-commit:1"));
    assert!(!sink_log.lock().iter().any(|entry| entry == "sink-abort:1"));
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the mixed-delivery runtime proof owns both disjoint output lifecycles"
)]
async fn checkpointed_runner_supports_disjoint_exactly_once_and_ordinary_outputs() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let plan = PipelineBuilder::new("checkpoint-mixed-delivery")
        .unwrap()
        .add_checkpoint_capable_node(
            "root",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .add_checkpoint_capable_node(
            "exact",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .add_checkpoint_capable_node(
            "ordinary",
            Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
        )
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("root", "output").unwrap(),
            PortEndpoint::new("exact", "input").unwrap(),
        ))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("root", "output").unwrap(),
            PortEndpoint::new("ordinary", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "exact.output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    let source_closed = Arc::new(AtomicUsize::new(0));
    let transactional_closed = Arc::new(AtomicUsize::new(0));
    let transactional_log = Arc::new(Mutex::new(Vec::new()));
    let ordinary_closed = Arc::new(AtomicUsize::new(0));
    let ordinary_writes = Arc::new(Mutex::new(Vec::new()));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            908,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: finite_binding(&[1], &source_closed),
        }],
        sinks: vec![
            NamedSinkBinding {
                output_id: "exact.output".into(),
                sink_id: "transactional".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                    log: Arc::clone(&transactional_log),
                    closed: Arc::clone(&transactional_closed),
                })),
            },
            NamedSinkBinding {
                output_id: "ordinary.output".into(),
                sink_id: "ordinary-sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(OrderedRecordingSink {
                    id: "ordinary".into(),
                    writes: Arc::clone(&ordinary_writes),
                    closed: Arc::clone(&ordinary_closed),
                })),
            },
        ],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let manifest_root = directory.path().join("manifests");
    let checkpoint = CheckpointRuntimeSpec::new(
        backend,
        &manifest_root,
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap();
    let mut runner = ContinuousRunner::new();

    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
    let outcome = tokio::time::timeout(StdDuration::from_secs(5), job.wait())
        .await
        .expect("mixed-delivery checkpointed job hung");

    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    assert_eq!(ordinary_writes.lock().len(), 1);
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(transactional_closed.load(Ordering::SeqCst), 1);
    assert_eq!(ordinary_closed.load(Ordering::SeqCst), 1);
    assert!(
        transactional_log
            .lock()
            .iter()
            .any(|entry| entry == "sink-commit:1")
    );
    let manifest = crate::CheckpointManifest::from_bytes(
        &tokio::fs::read(manifest_root.join("manifest-00000000000000000001.json"))
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        manifest.sinks()["transactional"].delivery,
        SinkDeliveryManifest::Transactional
    );
    assert_eq!(
        manifest.sinks()["ordinary-sink"].delivery,
        SinkDeliveryManifest::Ordinary
    );
    assert!(manifest.sinks()["ordinary-sink"].pre_commit.is_none());
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the mixed-delivery restart proof owns both job generations and their lifecycle assertions"
)]
async fn mixed_delivery_restart_preserves_exactly_once_and_exposes_ordinary_replay() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let manifest_root = directory.path().join("manifests");
    let probes = MixedDeliveryProbes::default();
    let checkpoint = || {
        CheckpointRuntimeSpec::new(
            backend.clone(),
            &manifest_root,
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
    };

    let mut first_runner = ContinuousRunner::new();
    let first_job = first_runner
        .start_checkpointed(
            mixed_delivery_fault_spec(910, &probes),
            checkpoint().with_fault(
                super::CheckpointFaultPoint::ManifestWrite,
                super::CheckpointFaultMode::Restart,
            ),
        )
        .await
        .unwrap();
    let first_outcome = tokio::time::timeout(StdDuration::from_secs(5), first_job.wait())
        .await
        .expect("mixed-delivery manifest-write fault hung");

    assert_eq!(first_outcome.state, ContinuousJobState::Failed);
    assert!(first_outcome.errors.iter().any(|failure| {
        matches!(
            &failure.error,
            CalcFlowError::Internal { message }
                if message == "injected checkpoint restart at ManifestWrite"
        )
    }));
    assert_eq!(
        &*probes.ordinary_writes.lock(),
        &[("ordinary".into(), "input".into(), 0)]
    );
    assert!(probes.transactional.lock().visible.is_empty());
    assert!(
        !manifest_root
            .join("manifest-00000000000000000001.json")
            .exists()
    );
    assert_terminal_checkpoint_resources_released(&first_job);
    drop(first_job);
    first_runner.shutdown().await.unwrap();
    assert_eq!(first_runner.registry_counts(), (0, 0));
    assert_eq!(probes.source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(probes.transactional_closed.load(Ordering::SeqCst), 1);
    assert_eq!(probes.ordinary_closed.load(Ordering::SeqCst), 1);

    let mut restart_runner = ContinuousRunner::new();
    let restart_job = restart_runner
        .start_checkpointed(mixed_delivery_fault_spec(911, &probes), checkpoint())
        .await
        .unwrap();
    let restart_outcome = tokio::time::timeout(StdDuration::from_secs(5), restart_job.wait())
        .await
        .expect("mixed-delivery restart hung");

    assert_eq!(
        restart_outcome.state,
        ContinuousJobState::Completed,
        "{restart_outcome:?}"
    );
    let ordinary_at_least_once_boundary = ("ordinary".into(), "input".into(), 0);
    assert_eq!(
        &*probes.ordinary_writes.lock(),
        &[
            ordinary_at_least_once_boundary.clone(),
            ordinary_at_least_once_boundary,
        ]
    );
    assert_eq!(&probes.transactional.lock().visible, &[("input".into(), 0)]);
    assert_eq!(probes.source_closed.load(Ordering::SeqCst), 2);
    assert_eq!(probes.transactional_closed.load(Ordering::SeqCst), 2);
    assert_eq!(probes.ordinary_closed.load(Ordering::SeqCst), 2);
    assert_terminal_checkpoint_resources_released(&restart_job);
    drop(restart_job);
    restart_runner.shutdown().await.unwrap();
    assert_eq!(restart_runner.registry_counts(), (0, 0));
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the checkpoint ordering scenario is clearest as one end-to-end test"
)]
async fn terminal_checkpoint_waits_until_the_periodic_epoch_completes() {
    let participants = ParticipantSet {
        sources: BTreeSet::from(["source".into()]),
        operators: BTreeSet::from(["operator".into()]),
        sinks: BTreeSet::from(["sink".into()]),
    };
    let cancellation = CancellationToken::new();
    let (coordinator, mut events, task) = spawn_checkpoint_coordinator(
        participants.clone(),
        crate::Epoch::INITIAL,
        4,
        StdDuration::from_secs(30),
        cancellation.clone(),
    )
    .unwrap();
    coordinator
        .request(CheckpointRequest::Periodic)
        .await
        .unwrap();
    let mut request_active = true;
    let mut terminal_request_active = false;
    assert_eq!(
        events.recv().await.unwrap(),
        CheckpointEvent::Started(crate::Epoch::INITIAL)
    );

    maybe_request_terminal_checkpoint(
        &coordinator,
        &participants.operators,
        &participants.sinks,
        &participants.operators,
        &participants.sinks,
        true,
        &mut request_active,
        &mut terminal_request_active,
    )
    .await
    .unwrap();
    assert!(request_active);
    assert!(!terminal_request_active);

    coordinator
        .ack(CheckpointAck::source(
            "source",
            crate::Epoch::INITIAL,
            "source-state",
        ))
        .await
        .unwrap();
    assert_eq!(
        events.recv().await.unwrap(),
        CheckpointEvent::PhaseAdvanced(crate::Epoch::INITIAL, CoordinatorPhase::SourcesCut)
    );
    coordinator
        .ack(CheckpointAck::operator(
            "operator",
            crate::Epoch::INITIAL,
            "operator-state",
        ))
        .await
        .unwrap();
    assert_eq!(
        events.recv().await.unwrap(),
        CheckpointEvent::PhaseAdvanced(
            crate::Epoch::INITIAL,
            CoordinatorPhase::OperatorsSnapshotted,
        )
    );
    coordinator
        .ack(CheckpointAck::sink_precommit(
            "sink",
            crate::Epoch::INITIAL,
            "sink-state",
        ))
        .await
        .unwrap();
    assert_eq!(
        events.recv().await.unwrap(),
        CheckpointEvent::ReadyToPublish(crate::Epoch::INITIAL)
    );
    coordinator
        .manifest_durable(crate::Epoch::INITIAL)
        .await
        .unwrap();
    assert_eq!(
        events.recv().await.unwrap(),
        CheckpointEvent::PhaseAdvanced(crate::Epoch::INITIAL, CoordinatorPhase::ManifestDurable,)
    );
    coordinator
        .ack(CheckpointAck::sink_commit("sink", crate::Epoch::INITIAL))
        .await
        .unwrap();
    assert_eq!(
        events.recv().await.unwrap(),
        CheckpointEvent::Completed(crate::Epoch::INITIAL)
    );
    tokio::task::yield_now().await;
    assert!(matches!(
        events.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));

    request_active = false;
    maybe_request_terminal_checkpoint(
        &coordinator,
        &participants.operators,
        &participants.sinks,
        &participants.operators,
        &participants.sinks,
        true,
        &mut request_active,
        &mut terminal_request_active,
    )
    .await
    .unwrap();
    assert!(request_active);
    assert!(terminal_request_active);
    assert_eq!(
        events.recv().await.unwrap(),
        CheckpointEvent::Started(crate::Epoch::INITIAL.next().unwrap())
    );

    cancellation.cancel();
    task.await.unwrap().unwrap();
}

#[test]
fn periodic_cut_after_all_sources_end_is_terminal() {
    let cut = |ended| DurableSourceCut {
        history: None,
        cursor: None,
        next_sequence: 1,
        ended,
    };
    let all_ended = BTreeMap::from([
        (
            crate::runtime::streaming::progress::BindingIdentity::new("left").unwrap(),
            cut(true),
        ),
        (
            crate::runtime::streaming::progress::BindingIdentity::new("right").unwrap(),
            cut(true),
        ),
    ]);
    let one_live = BTreeMap::from([
        (
            crate::runtime::streaming::progress::BindingIdentity::new("left").unwrap(),
            cut(true),
        ),
        (
            crate::runtime::streaming::progress::BindingIdentity::new("right").unwrap(),
            cut(false),
        ),
    ]);

    assert!(source_cuts_are_terminal(&all_ended));
    assert!(!source_cuts_are_terminal(&one_live));
    assert!(!source_cuts_are_terminal(&BTreeMap::new()));
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "terminal publication and restart short-circuit are one end-to-end recovery contract"
)]
async fn checkpointed_runner_commits_a_terminal_epoch_without_a_post_end_barrier() {
    let directory = tempfile::tempdir().unwrap();
    let terminal_plan = || {
        PipelineBuilder::new("checkpoint-terminal")
            .unwrap()
            .add_checkpoint_capable_node(
                "node",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements {
                    delivery: BTreeMap::from([(
                        "output".into(),
                        crate::DeliveryGuarantee::ExactlyOnce,
                    )]),
                },
            )
            .unwrap()
    };
    let plan = terminal_plan();
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let log = Arc::new(Mutex::new(Vec::new()));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            904,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: finite_binding(&[7], &source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&log),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let checkpoint = CheckpointRuntimeSpec::managed(
        ManagedCheckpointRuntime::new(directory.path()).unwrap(),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap();
    let mut runner = ContinuousRunner::new();
    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();

    let outcome = match tokio::time::timeout(FILESYSTEM_SETTLEMENT_TIMEOUT, job.wait()).await {
        Ok(outcome) => outcome,
        Err(error) => {
            let _ = job.cancel().await;
            drop(job);
            runner.shutdown().await.unwrap();
            panic!("terminal checkpoint did not complete: {error}");
        }
    };

    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(
        &*log.lock(),
        &[
            "sink-open",
            "sink-begin:1",
            "sink-precommit:1",
            "sink-commit:1",
            "sink-close",
        ]
    );
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);

    log.lock().clear();
    let source_polls = Arc::new(AtomicUsize::new(0));
    let restored_source_closed = Arc::new(AtomicUsize::new(0));
    let restored_sink_closed = Arc::new(AtomicUsize::new(0));
    let plan = terminal_plan();
    let restored_spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            905,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(0, &source_polls, &restored_source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&log),
                closed: Arc::clone(&restored_sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let restored_checkpoint = CheckpointRuntimeSpec::managed(
        ManagedCheckpointRuntime::new(directory.path()).unwrap(),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap();
    let mut restored_runner = ContinuousRunner::new();
    let restored_job = restored_runner
        .start_checkpointed(restored_spec, restored_checkpoint)
        .await
        .unwrap();
    let restored = tokio::time::timeout(FILESYSTEM_SETTLEMENT_TIMEOUT, restored_job.wait())
        .await
        .expect("terminal manifest recovery should short-circuit");
    assert_eq!(restored.state, ContinuousJobState::Completed);
    drop(restored_job);
    restored_runner.shutdown().await.unwrap();
    assert_eq!(source_polls.load(Ordering::SeqCst), 0);
    assert_eq!(restored_source_closed.load(Ordering::SeqCst), 0);
    assert_eq!(restored_sink_closed.load(Ordering::SeqCst), 1);
    assert_eq!(&*log.lock(), &["sink-open", "sink-recover:1", "sink-close"]);
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the retention failure and restart are one durable-manifest recovery contract"
)]
async fn retention_failure_after_terminal_commit_recovers_the_durable_epoch() {
    let directory = tempfile::tempdir().unwrap();
    let retention_failure_armed = Arc::new(AtomicBool::new(false));
    let backend = Arc::new(FailOnceRetentionBackend {
        inner: LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
        failure_armed: Arc::clone(&retention_failure_armed),
    });
    let terminal_plan = || {
        PipelineBuilder::new("checkpoint-retention-fault")
            .unwrap()
            .add_checkpoint_capable_node(
                "node",
                Box::new(StressForwardOperator::new(None)) as Box<dyn StreamOperator>,
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements {
                    delivery: BTreeMap::from([(
                        "output".into(),
                        crate::DeliveryGuarantee::ExactlyOnce,
                    )]),
                },
            )
            .unwrap()
    };
    let checkpoint = || {
        CheckpointRuntimeSpec::new(
            backend.clone(),
            directory.path().join("manifests"),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
    };
    let log = Arc::new(Mutex::new(Vec::new()));
    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let source_gate = Arc::new(Semaphore::new(0));
    let plan = terminal_plan();
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            906,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: SourceBinding::new(
                Box::new(StressSource {
                    events: VecDeque::from([
                        (
                            Arc::clone(&source_gate),
                            Some(SourceEvent::Data {
                                batch: one_row(7),
                                cursor: Cursor::new("input", vec![1], JsonMap::new()).unwrap(),
                            }),
                        ),
                        (Arc::clone(&source_gate), None),
                    ]),
                    closed: Arc::clone(&source_closed),
                }),
                None,
                0,
            )
            .unwrap(),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&log),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let job = runner.start_checkpointed(spec, checkpoint()).await.unwrap();
    retention_failure_armed.store(true, Ordering::SeqCst);
    source_gate.add_permits(2);

    let failed = tokio::time::timeout(StdDuration::from_secs(1), job.wait())
        .await
        .expect("retention fault should terminate the job");

    assert_eq!(failed.state, ContinuousJobState::Failed);
    assert!(failed.errors.iter().any(|failure| {
        matches!(
            &failure.error,
            CalcFlowError::Internal { message } if message == "injected retention failure"
        )
    }));
    let failed_checkpoint = job.status().checkpoint.unwrap();
    assert_eq!(failed_checkpoint.current_epoch, Some(crate::Epoch::INITIAL));
    assert_eq!(
        failed_checkpoint.phase,
        Some(CheckpointPhase::SinksCommitted)
    );
    assert_eq!(
        failed_checkpoint.last_completed_epoch,
        Some(crate::Epoch::INITIAL)
    );
    assert_eq!(
        failed_checkpoint.failure_category,
        Some(CheckpointFailureCategory::Maintenance)
    );
    assert!(failed_checkpoint.elapsed.is_some());
    assert_eq!(failed_checkpoint.source_acks, 1);
    assert_eq!(failed_checkpoint.operator_acks, 1);
    assert_eq!(failed_checkpoint.sink_precommit_acks, 1);
    assert_eq!(failed_checkpoint.sink_commit_acks, 1);
    let failed_metrics = job.status().metrics.checkpoints;
    assert_eq!(failed_metrics.requested, 1);
    assert_eq!(failed_metrics.completed, 1);
    assert_eq!(failed_metrics.failed, 1);
    assert_eq!(failed_metrics.terminal_requested, 1);
    assert_eq!(failed_metrics.terminal_completed, 1);
    assert_eq!(failed_metrics.terminal_failed, 1);
    drop(job);
    runner.shutdown().await.unwrap();
    assert!(log.lock().contains(&"sink-commit:1".into()));
    assert!(!log.lock().iter().any(|entry| entry.contains("abort")));
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);

    log.lock().clear();
    let source_polls = Arc::new(AtomicUsize::new(0));
    let restored_source_closed = Arc::new(AtomicUsize::new(0));
    let restored_sink_closed = Arc::new(AtomicUsize::new(0));
    let plan = terminal_plan();
    let restored_spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            907,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: counting_pending_binding(0, &source_polls, &restored_source_closed),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&log),
                closed: Arc::clone(&restored_sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut restored_runner = ContinuousRunner::new();
    let restored_job = restored_runner
        .start_checkpointed(restored_spec, checkpoint())
        .await
        .unwrap();
    let restored = tokio::time::timeout(FILESYSTEM_SETTLEMENT_TIMEOUT, restored_job.wait())
        .await
        .expect("durable terminal epoch should recover after retention failure");
    assert_eq!(restored.state, ContinuousJobState::Completed);
    let restored_metrics = restored_job.status().metrics.checkpoints;
    assert_eq!(restored_metrics.requested, 0);
    assert_eq!(restored_metrics.sink_commit_retries, 1);
    drop(restored_job);
    restored_runner.shutdown().await.unwrap();
    assert_eq!(source_polls.load(Ordering::SeqCst), 0);
    assert_eq!(restored_source_closed.load(Ordering::SeqCst), 0);
    assert_eq!(restored_sink_closed.load(Ordering::SeqCst), 1);
    assert_eq!(&*log.lock(), &["sink-open", "sink-recover:1", "sink-close"]);
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the recovery lifecycle is asserted as one ordered integration scenario"
)]
async fn checkpointed_runner_restores_operator_source_and_sink_before_polling() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let log = Arc::new(Mutex::new(Vec::new()));
    let operator = RecoveryProbeOperator {
        inputs: [Port::new("input", BatchKind::Table, true, None).unwrap()],
        outputs: [Port::new("output", BatchKind::Table, true, None).unwrap()],
        log: Arc::clone(&log),
    };
    let requirements = StreamRequirements {
        delivery: BTreeMap::from([("output".into(), crate::DeliveryGuarantee::ExactlyOnce)]),
    };
    let plan = PipelineBuilder::new("checkpoint-recovery")
        .unwrap()
        .add_checkpoint_capable_node("node", Box::new(operator) as Box<dyn StreamOperator>)
        .unwrap()
        .compile_stream(&UdfRegistry::new().snapshot(), &requirements)
        .unwrap();
    let config = StreamRuntimeConfig::default();
    let prepared = crate::runtime::streaming::progress::prepare_stream_job(
            plan.fingerprint(),
            &[crate::runtime::streaming::progress::SourceBindingSpec {
                descriptor: crate::runtime::streaming::progress::SourceDescriptor::new(
                    crate::runtime::streaming::progress::BindingIdentity::new("input").unwrap(),
                    crate::runtime::streaming::progress::DeclaredSchema::DynamicOrUnknown,
                    crate::runtime::streaming::progress::NativeWatermarkCapability::NeverEmits,
                    crate::runtime::streaming::progress::ReplayPositioningCapability::ExactPauseReportAndSeek,
                    None,
                )
                .with_delivery_and_bounds(true, 1, 1 << 20),
                watermark_policy: crate::runtime::streaming::progress::WatermarkPolicy::Disabled {
                    idle_timeout: None,
                },
            }],
            crate::runtime::streaming::progress::StreamProgressRuntimeConfig::default(),
        )
        .unwrap();
    let manifest = crate::CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: plan.name().into(),
        pipeline_fingerprint: plan.fingerprint().into(),
        runtime_config_hash: plan.runtime_config_hash(&config).unwrap(),
        epoch: crate::Epoch::INITIAL,
        created_at: chrono::Utc.with_ymd_and_hms(2026, 8, 9, 9, 0, 0).unwrap(),
        recovery_status: RecoveryStatus::Final,
        sources: BTreeMap::from([(
            "input".into(),
            SourceManifestEntry {
                history: None,
                cursor: Some(CursorManifestEntry {
                    order: "09".into(),
                    payload: BTreeMap::from([("offset".into(), serde_json::json!(9))]),
                }),
                identity_hash: prepared.bindings[0].identity_hash(),
                sequence: 12,
                ended: false,
                watermark_policy: SourceWatermarkManifestState::Disabled { idle: true },
            },
        )]),
        operators: BTreeMap::from([(
            "node".into(),
            OperatorManifestEntry {
                progress: BTreeMap::from([(
                    "input".into(),
                    OperatorIngressManifestEntry {
                        state: ManifestIngressState::Active,
                        watermark: None,
                    },
                )]),
                inline_metadata:
                    crate::pipeline::OperatorCheckpointCapability::CheckpointedStateful {
                        state_version: 1,
                    }
                    .encode_snapshot(
                        "node",
                        crate::OperatorStateSnapshot {
                            inline_metadata: BTreeMap::from([(
                                "restored".into(),
                                serde_json::json!(true),
                            )]),
                            segments: BTreeMap::new(),
                        },
                    )
                    .unwrap()
                    .inline_metadata,
                segments: Vec::new(),
            },
        )]),
        sinks: BTreeMap::from([(
            "sink".into(),
            SinkManifestEntry {
                delivery: SinkDeliveryManifest::Transactional,
                pre_commit: Some(JsonMap::new()),
                segments: Vec::new(),
            },
        )]),
        static_inputs: BTreeMap::new(),
    })
    .unwrap();
    let key = StateLineageKey::new(plan.name(), plan.fingerprint()).unwrap();
    let lineage = backend.open_lineage(&key).await.unwrap();
    let transaction = crate::state::ManifestTransaction::open(
        Arc::from(lineage),
        &key,
        directory.path().join("manifests"),
        config.retained_epochs,
    )
    .await
    .unwrap();
    transaction
        .publish(crate::state::PreparedEpochManifest {
            manifest,
            staged_segments: BTreeMap::new(),
        })
        .await
        .unwrap();
    drop(transaction);

    let source_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            902,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: SourceBinding::new(
                Box::new(RecoveryProbeSource {
                    log: Arc::clone(&log),
                    closed: Arc::clone(&source_closed),
                }),
                None,
                0,
            )
            .unwrap()
            .with_watermark_policy(
                crate::runtime::streaming::progress::WatermarkPolicy::Disabled {
                    idle_timeout: None,
                },
            ),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(RecoveryProbeSink {
                log: Arc::clone(&log),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let checkpoint = CheckpointRuntimeSpec::managed(
        ManagedCheckpointRuntime::new(directory.path()).unwrap(),
        config,
    )
    .unwrap();
    let mut runner = ContinuousRunner::new();

    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();

    assert_eq!(
        &*log.lock(),
        &[
            "operator-reset",
            "operator-restore",
            "source-open:09",
            "sink-open",
            "sink-recover:1",
        ]
    );
    assert_eq!(job.status().sources["input"].next_sequence, Some(12));
    let progress = job.status().progress.unwrap();
    assert!(matches!(
        progress.current.bindings
            [&crate::runtime::streaming::progress::BindingIdentity::new("input").unwrap()]
            .activity,
        crate::runtime::streaming::progress::aggregate::IngressActivity::Idle { watermark: None }
    ));
    assert_eq!(progress.current.counters.trace_records, 0);
    let outcome = job.cancel().await;
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(source_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
#[allow(
    clippy::too_many_lines,
    reason = "mixed ended/live restore owns the manifest, connector probes, and periodic cut"
)]
async fn restored_ended_source_participates_without_open_seek_poll_or_barrier() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let union = UnionOperator::new(
        "merge",
        vec![
            Port::new("ended", BatchKind::Table, true, None).unwrap(),
            Port::new("live", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap();
    let plan = PipelineBuilder::new("checkpoint-mixed-source-restore")
        .unwrap()
        .add_node("merge", Box::new(union))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    let config = StreamRuntimeConfig {
        checkpoint_interval: StdDuration::from_millis(10),
        checkpoint_timeout: StdDuration::from_secs(10),
        ..StreamRuntimeConfig::default()
    };
    let source_spec = |binding_id: &str| {
        crate::runtime::streaming::progress::SourceBindingSpec {
                descriptor: crate::runtime::streaming::progress::SourceDescriptor::new(
                    crate::runtime::streaming::progress::BindingIdentity::new(binding_id).unwrap(),
                    crate::runtime::streaming::progress::DeclaredSchema::DynamicOrUnknown,
                    crate::runtime::streaming::progress::NativeWatermarkCapability::NeverEmits,
                    crate::runtime::streaming::progress::ReplayPositioningCapability::ExactPauseReportAndSeek,
                    None,
                )
                .with_delivery_and_bounds(true, 1, 1 << 20),
                watermark_policy:
                    crate::runtime::streaming::progress::WatermarkPolicy::Disabled {
                        idle_timeout: None,
                    },
            }
    };
    let prepared = crate::runtime::streaming::progress::prepare_stream_job(
        plan.fingerprint(),
        &[source_spec("ended"), source_spec("live")],
        crate::runtime::streaming::progress::StreamProgressRuntimeConfig::default(),
    )
    .unwrap();
    let identity_hash = |binding_id: &str| {
        prepared
            .bindings
            .iter()
            .find(|binding| binding.identity.as_str() == binding_id)
            .unwrap()
            .identity_hash()
    };
    let manifest = crate::CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: plan.name().into(),
        pipeline_fingerprint: plan.fingerprint().into(),
        runtime_config_hash: plan.runtime_config_hash(&config).unwrap(),
        epoch: crate::Epoch::INITIAL,
        created_at: chrono::Utc.with_ymd_and_hms(2026, 8, 9, 9, 30, 0).unwrap(),
        recovery_status: RecoveryStatus::Final,
        sources: BTreeMap::from([
            (
                "ended".into(),
                SourceManifestEntry {
                    history: None,
                    cursor: Some(CursorManifestEntry {
                        order: "01".into(),
                        payload: BTreeMap::new(),
                    }),
                    identity_hash: identity_hash("ended"),
                    sequence: 1,
                    ended: true,
                    watermark_policy: SourceWatermarkManifestState::Disabled { idle: false },
                },
            ),
            (
                "live".into(),
                SourceManifestEntry {
                    history: None,
                    cursor: Some(CursorManifestEntry {
                        order: "09".into(),
                        payload: BTreeMap::new(),
                    }),
                    identity_hash: identity_hash("live"),
                    sequence: 12,
                    ended: false,
                    watermark_policy: SourceWatermarkManifestState::Disabled { idle: false },
                },
            ),
        ]),
        operators: BTreeMap::from([(
            "merge".into(),
            OperatorManifestEntry {
                progress: BTreeMap::from([
                    (
                        "ended".into(),
                        OperatorIngressManifestEntry {
                            state: ManifestIngressState::Ended,
                            watermark: None,
                        },
                    ),
                    (
                        "live".into(),
                        OperatorIngressManifestEntry {
                            state: ManifestIngressState::Active,
                            watermark: None,
                        },
                    ),
                ]),
                inline_metadata: BTreeMap::new(),
                segments: Vec::new(),
            },
        )]),
        sinks: BTreeMap::from([(
            "sink".into(),
            SinkManifestEntry {
                delivery: SinkDeliveryManifest::Transactional,
                pre_commit: Some(JsonMap::new()),
                segments: Vec::new(),
            },
        )]),
        static_inputs: BTreeMap::new(),
    })
    .unwrap();
    let key = StateLineageKey::new(plan.name(), plan.fingerprint()).unwrap();
    let lineage = backend.open_lineage(&key).await.unwrap();
    let transaction = crate::state::ManifestTransaction::open(
        Arc::from(lineage),
        &key,
        directory.path().join("manifests"),
        config.retained_epochs,
    )
    .await
    .unwrap();
    transaction
        .publish(crate::state::PreparedEpochManifest {
            manifest,
            staged_segments: BTreeMap::new(),
        })
        .await
        .unwrap();
    drop(transaction);

    let ended_opens = Arc::new(AtomicUsize::new(0));
    let ended_seeks = Arc::new(AtomicUsize::new(0));
    let ended_polls = Arc::new(AtomicUsize::new(0));
    let ended_closed = Arc::new(AtomicUsize::new(0));
    let live_opens = Arc::new(AtomicUsize::new(0));
    let live_seeks = Arc::new(AtomicUsize::new(0));
    let live_polls = Arc::new(AtomicUsize::new(0));
    let live_closed = Arc::new(AtomicUsize::new(0));
    let sink_closed = Arc::new(AtomicUsize::new(0));
    let sink_log = Arc::new(Mutex::new(Vec::new()));
    let source_binding = |opens: &Arc<AtomicUsize>,
                          seeks: &Arc<AtomicUsize>,
                          polls: &Arc<AtomicUsize>,
                          closed: &Arc<AtomicUsize>| {
        SourceBinding::new(
            Box::new(MixedRestoreSource {
                opens: Arc::clone(opens),
                seeks: Arc::clone(seeks),
                polls: Arc::clone(polls),
                closed: Arc::clone(closed),
            }),
            None,
            0,
        )
        .unwrap()
        .with_watermark_policy(
            crate::runtime::streaming::progress::WatermarkPolicy::Disabled { idle_timeout: None },
        )
    };
    let spec = ContinuousJobSpec {
        context: StreamJobContext::new(
            909,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![
            NamedSourceBinding {
                binding_id: "ended".into(),
                binding: source_binding(&ended_opens, &ended_seeks, &ended_polls, &ended_closed),
            },
            NamedSourceBinding {
                binding_id: "live".into(),
                binding: source_binding(&live_opens, &live_seeks, &live_polls, &live_closed),
            },
        ],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(PeriodicCheckpointSink {
                log: Arc::clone(&sink_log),
                closed: Arc::clone(&sink_closed),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let checkpoint =
        CheckpointRuntimeSpec::new(backend, directory.path().join("manifests"), config).unwrap();
    let mut runner = ContinuousRunner::new();

    let job = runner.start_checkpointed(spec, checkpoint).await.unwrap();
    wait_for_counter(&live_polls, 1).await;
    tokio::time::advance(StdDuration::from_millis(20)).await;
    let deadline = StdInstant::now() + StdDuration::from_secs(30);
    while !sink_log.lock().iter().any(|entry| entry == "sink-begin:3") {
        assert!(
            StdInstant::now() < deadline,
            "mixed-source restored checkpoint did not complete: {:?}",
            job.status()
        );
        tokio::task::yield_now().await;
    }

    assert_eq!(ended_opens.load(Ordering::SeqCst), 0);
    assert_eq!(ended_seeks.load(Ordering::SeqCst), 0);
    assert_eq!(ended_polls.load(Ordering::SeqCst), 0);
    assert_eq!(ended_closed.load(Ordering::SeqCst), 0);
    assert_eq!(live_opens.load(Ordering::SeqCst), 1);
    assert_eq!(live_seeks.load(Ordering::SeqCst), 1);
    assert_eq!(job.status().sources["live"].next_sequence, Some(12));
    assert_eq!(
        job.status().checkpoint.unwrap().last_completed_epoch,
        Some(crate::Epoch::new(2).unwrap())
    );
    let outcome = job.cancel().await;
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(live_closed.load(Ordering::SeqCst), 1);
    assert_eq!(sink_closed.load(Ordering::SeqCst), 1);
}

#[test]
fn checkpoint_fault_matrix_enumerates_every_durable_boundary_and_mode() {
    use super::{CheckpointFaultInjector, CheckpointFaultMode, CheckpointFaultPoint};

    assert_eq!(
        CheckpointFaultPoint::ALL,
        [
            CheckpointFaultPoint::SourceAdmission,
            CheckpointFaultPoint::SourceCut,
            CheckpointFaultPoint::PartialAlignment,
            CheckpointFaultPoint::StateStage,
            CheckpointFaultPoint::SinkPreCommit,
            CheckpointFaultPoint::ManifestWrite,
            CheckpointFaultPoint::ManifestRename,
            CheckpointFaultPoint::ManifestParentSync,
            CheckpointFaultPoint::PartialSinkCommit,
            CheckpointFaultPoint::CompletedCommit,
            CheckpointFaultPoint::Retention,
            CheckpointFaultPoint::Compaction,
        ]
    );
    assert_eq!(
        CheckpointFaultMode::ALL,
        [
            CheckpointFaultMode::Io,
            CheckpointFaultMode::Panic,
            CheckpointFaultMode::Cancel,
            CheckpointFaultMode::Restart,
        ]
    );

    let cancellation = CancellationToken::new();
    let injector =
        CheckpointFaultInjector::armed(CheckpointFaultPoint::SourceCut, CheckpointFaultMode::Io);
    assert!(
        injector
            .trigger(CheckpointFaultPoint::SourceAdmission, &cancellation)
            .is_ok()
    );
    assert!(
        injector
            .trigger(CheckpointFaultPoint::SourceCut, &cancellation)
            .is_err()
    );
    assert!(
        injector
            .trigger(CheckpointFaultPoint::SourceCut, &cancellation)
            .is_ok()
    );
    assert_eq!(injector.trigger_count(), 1);

    let cancellation = CancellationToken::new();
    let injector = CheckpointFaultInjector::armed(
        CheckpointFaultPoint::SourceCut,
        CheckpointFaultMode::Cancel,
    );
    assert!(
        injector
            .trigger(CheckpointFaultPoint::SourceCut, &cancellation)
            .is_ok()
    );
    assert!(cancellation.is_cancelled());
    assert_eq!(injector.trigger_count(), 1);
}

/// One keyed event-time row for the Join->Window AC5 scenarios.
fn join_ts_row(key: &str, ts: i64) -> Batch {
    use datafusion::arrow::array::{StringArray, TimestampMicrosecondArray};
    let schema = Arc::new(datafusion::arrow::datatypes::Schema::new(vec![
        datafusion::arrow::datatypes::Field::new(
            "key",
            datafusion::arrow::datatypes::DataType::Utf8,
            false,
        ),
        datafusion::arrow::datatypes::Field::new(
            "ts",
            datafusion::arrow::datatypes::DataType::Timestamp(
                datafusion::arrow::datatypes::TimeUnit::Microsecond,
                None,
            ),
            false,
        ),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![key])),
            Arc::new(TimestampMicrosecondArray::from(vec![ts])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn ac5_scripted_events(rows: &[(&str, i64)], watermark: i64) -> VecDeque<SourceEvent> {
    let mut events = VecDeque::new();
    for (index, (key, ts)) in rows.iter().enumerate() {
        events.push_back(SourceEvent::Data {
            batch: join_ts_row(key, *ts),
            cursor: Cursor::unbound(
                u64::try_from(index + 1).unwrap().to_be_bytes().to_vec(),
                JsonMap::new(),
            )
            .unwrap(),
        });
    }
    events.push_back(SourceEvent::Watermark(EventTime::from_micros(watermark)));
    events
}

struct PairCountSink {
    rows: Arc<Mutex<Vec<i64>>>,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl OrdinaryStreamSink for PairCountSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        for record in batch.table_payload()?.batches() {
            let column = record
                .column_by_name("pairs")
                .expect("the window emits the pairs aggregate");
            let counts = column
                .as_any()
                .downcast_ref::<datafusion::arrow::array::UInt64Array>()
                .expect("the pairs aggregate is a UInt64 column");
            self.rows
                .lock()
                .extend((0..counts.len()).map(|index| i64::try_from(counts.value(index)).unwrap()));
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn ac5_job_spec(
    plan: crate::StreamExecutionPlan,
    rows: &Arc<Mutex<Vec<i64>>>,
) -> ContinuousJobSpec {
    ContinuousJobSpec {
        context: StreamJobContext::new(
            91,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: Vec::new(),
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "pairs".into(),
            binding: OrdinarySinkBinding::new(Box::new(PairCountSink {
                rows: Arc::clone(rows),
                closed: Arc::new(AtomicUsize::new(0)),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

/// Builds the AC5 `Join`->Window graph on one prefixed event-time column.
fn ac5_join_window_plan(bounds: JoinTimeBounds, window_column: &str) -> crate::StreamExecutionPlan {
    let schema = Arc::new(datafusion::arrow::datatypes::Schema::new(vec![
        datafusion::arrow::datatypes::Field::new(
            "key",
            datafusion::arrow::datatypes::DataType::Utf8,
            false,
        ),
        datafusion::arrow::datatypes::Field::new(
            "ts",
            datafusion::arrow::datatypes::DataType::Timestamp(
                datafusion::arrow::datatypes::TimeUnit::Microsecond,
                None,
            ),
            false,
        ),
    ]));
    let join = StreamJoinOperator::new(
        "match",
        Arc::clone(&schema),
        schema,
        StreamJoinSpec::inner(
            ["key"],
            ["key"],
            "ts",
            "ts",
            bounds,
            JoinStateLimits::new(100_000, 134_217_728, 1_000_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let join_output = Arc::clone(join.output_ports()[0].schema().expect("derived schema"));
    let mut window_spec =
        WindowSpec::tumbling(window_column, StdDuration::from_micros(10)).unwrap();
    window_spec.aggregates = vec![AggregateSpec {
        function: AggregateFunction::Count,
        column: "left__ts".into(),
        output: "pairs".into(),
    }];
    let window = WindowAggregateOperator::new("agg", join_output, window_spec).unwrap();
    PipelineBuilder::new("ac5")
        .unwrap()
        .add_node("match", Box::new(join))
        .unwrap()
        .add_node("agg", Box::new(window))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("match", "output").unwrap(),
            PortEndpoint::new("agg", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

/// Runs one AC5 `Join`->Window scenario through a real `StreamingRunner`
/// with the given source bindings and returns the window's emitted pair
/// counts (spec AC5).
async fn run_ac5_join_window_from_bindings(
    plan: crate::StreamExecutionPlan,
    left: SourceBinding,
    right: SourceBinding,
) -> Vec<i64> {
    let rows = Arc::new(Mutex::new(Vec::new()));
    let mut spec = ac5_job_spec(plan, &rows);
    spec.sources = vec![
        NamedSourceBinding {
            binding_id: "left".into(),
            binding: left,
        },
        NamedSourceBinding {
            binding_id: "right".into(),
            binding: right,
        },
    ];
    let runner = ContinuousRunner::new();
    let job = runner.start(spec).await.unwrap();
    let outcome = job.wait().await;
    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    drop(job);
    rows.lock().clone()
}

fn ac5_finite_binding(events: VecDeque<SourceEvent>) -> SourceBinding {
    SourceBinding::new(
        Box::new(FiniteSource {
            events,
            closed: Arc::new(AtomicUsize::new(0)),
        }),
        None,
        0,
    )
    .unwrap()
}

/// Runs one AC5 `Join`->Window scenario through a real `StreamingRunner` and
/// returns the window's emitted pair counts (spec AC5).
async fn run_ac5_join_window(
    bounds: JoinTimeBounds,
    window_column: &str,
    left_rows: &[(&str, i64)],
    left_watermark: i64,
    right_rows: &[(&str, i64)],
    right_watermark: i64,
) -> Vec<i64> {
    run_ac5_join_window_from_bindings(
        ac5_join_window_plan(bounds, window_column),
        ac5_finite_binding(ac5_scripted_events(left_rows, left_watermark)),
        ac5_finite_binding(ac5_scripted_events(right_rows, right_watermark)),
    )
    .await
}

#[tokio::test]
async fn ac5_retained_left_after_counterexample_reaches_the_window() {
    // after=10, left=95, WL=WR=100: the decided frontier is
    // min(100-0, 100-10)=90, so the [90,100) tumbling window on the
    // prefixed left event-time column never closes ahead of the pair.
    let counts = run_ac5_join_window(
        JoinTimeBounds::new(StdDuration::from_micros(0), StdDuration::from_micros(10)).unwrap(),
        "left__ts",
        &[("a", 95)],
        100,
        &[("a", 100)],
        100,
    )
    .await;
    assert_eq!(counts, vec![1]);
}

#[tokio::test]
async fn ac5_symmetric_before_counterexample_reaches_the_window() {
    // before=10, right=95, WL=WR=100: min(100-10, 100-0)=90 keeps the
    // [90,100) window on the prefixed right event-time column open.
    let counts = run_ac5_join_window(
        JoinTimeBounds::new(StdDuration::from_micros(10), StdDuration::from_micros(0)).unwrap(),
        "right__ts",
        &[("a", 100)],
        100,
        &[("a", 95)],
        100,
    )
    .await;
    assert_eq!(counts, vec![1]);
}

#[tokio::test]
async fn ac5_right_end_uses_left_watermark_minus_before() {
    // After the right ingress ends, the frontier is WL-before =
    // 120-10=110, closing the [100,110) window exactly after the pair is
    // accepted.
    let counts = run_ac5_join_window(
        JoinTimeBounds::new(StdDuration::from_micros(10), StdDuration::from_micros(0)).unwrap(),
        "left__ts",
        &[("a", 105)],
        120,
        &[("a", 104)],
        100,
    )
    .await;
    assert_eq!(counts, vec![1]);
}

#[tokio::test]
async fn ac5_left_end_uses_right_watermark_minus_after() {
    // After the left ingress ends, the frontier is WR-after =
    // 120-10=110 for the [100,110) window on the right column.
    let counts = run_ac5_join_window(
        JoinTimeBounds::new(StdDuration::from_micros(0), StdDuration::from_micros(10)).unwrap(),
        "right__ts",
        &[("a", 104)],
        100,
        &[("a", 105)],
        120,
    )
    .await;
    assert_eq!(counts, vec![1]);
}

#[tokio::test]
async fn ac5_idle_reactivation_keeps_retained_rows_matchable() {
    // Left goes Idle after its first watermark; Idle must not evict the
    // retained left row and must not remove left from the output-frontier
    // calculation. Reactivated left data still matches later right rows,
    // and both windows close exactly once (spec AC5/AC6).
    let mut left = ac5_scripted_events(&[("a", 95)], 100);
    left.push_back(SourceEvent::Idle);
    let reactivation = join_ts_row("b", 105);
    left.push_back(SourceEvent::Data {
        batch: reactivation,
        cursor: Cursor::unbound(2_u64.to_be_bytes().to_vec(), JsonMap::new()).unwrap(),
    });
    left.push_back(SourceEvent::Watermark(EventTime::from_micros(110)));
    let mut right = ac5_scripted_events(&[("a", 100)], 100);
    let late_right = join_ts_row("b", 106);
    right.push_back(SourceEvent::Data {
        batch: late_right,
        cursor: Cursor::unbound(2_u64.to_be_bytes().to_vec(), JsonMap::new()).unwrap(),
    });
    right.push_back(SourceEvent::Watermark(EventTime::from_micros(120)));

    let counts = run_ac5_join_window_from_bindings(
        ac5_join_window_plan(
            JoinTimeBounds::new(StdDuration::from_micros(0), StdDuration::from_micros(10)).unwrap(),
            "left__ts",
        ),
        ac5_finite_binding(left),
        ac5_finite_binding(right),
    )
    .await;
    // [90,100) holds the pre-Idle pair and [100,110) the post-reactivation
    // pair; both close because the final frontier min(110-0, 120-10)=110.
    assert_eq!(counts, vec![1, 1], "idle reactivation windows");
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the restore scenario is clearest as one end-to-end test"
)]
async fn ac5_checkpoint_restore_preserves_the_join_window_result() {
    // Durable cut after both data rows but before either watermark: the
    // restored job resumes its source cursors, restores Join state, and
    // still delivers every legal pair to the Window exactly once instead
    // of re-forwarding or losing it (spec AC5/AC12).
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let manifest_root = directory.path().join("manifests");
    let checkpoint = || {
        CheckpointRuntimeSpec::new(
            backend.clone(),
            &manifest_root,
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
    };
    let left_release = Arc::new(AtomicBool::new(false));
    let right_release = Arc::new(AtomicBool::new(false));
    let rows = Arc::new(Mutex::new(Vec::new()));
    let spec =
        |left_release: Arc<AtomicBool>, right_release: Arc<AtomicBool>| -> ContinuousJobSpec {
            let mut spec = ac5_job_spec(
                ac5_join_window_plan(
                    JoinTimeBounds::new(StdDuration::from_micros(0), StdDuration::from_micros(10))
                        .unwrap(),
                    "left__ts",
                ),
                &rows,
            );
            spec.sources = vec![
                NamedSourceBinding {
                    binding_id: "left".into(),
                    binding: SourceBinding::new(
                        Box::new(PausedWatermarkSource {
                            key: "a",
                            ts: 95,
                            order: 1,
                            watermark: 110,
                            release: left_release,
                            data_delivered: false,
                            watermark_delivered: false,
                        }),
                        None,
                        0,
                    )
                    .unwrap(),
                },
                NamedSourceBinding {
                    binding_id: "right".into(),
                    binding: SourceBinding::new(
                        Box::new(PausedWatermarkSource {
                            key: "a",
                            ts: 100,
                            order: 1,
                            watermark: 120,
                            release: right_release,
                            data_delivered: false,
                            watermark_delivered: false,
                        }),
                        None,
                        0,
                    )
                    .unwrap(),
                },
            ];
            spec
        };

    let mut first_runner = ContinuousRunner::new();
    let first_job = first_runner
        .start_checkpointed(
            spec(Arc::clone(&left_release), Arc::clone(&right_release)),
            checkpoint(),
        )
        .await
        .unwrap();
    wait_for_join_emission(&first_job, 1).await;
    let epoch = tokio::time::timeout(StdDuration::from_secs(30), first_job.trigger_checkpoint())
        .await
        .expect("ac5 restore checkpoint should not hang")
        .unwrap();
    assert_eq!(epoch, crate::Epoch::INITIAL);
    assert_eq!(
        first_job.cancel().await.state,
        ContinuousJobState::Cancelled
    );
    drop(first_job);
    first_runner.shutdown().await.unwrap();

    left_release.store(true, Ordering::SeqCst);
    right_release.store(true, Ordering::SeqCst);
    let mut restart_runner = ContinuousRunner::new();
    let restart_job = restart_runner
        .start_checkpointed(spec(left_release, right_release), checkpoint())
        .await
        .unwrap();
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), restart_job.wait())
        .await
        .expect("ac5 restart hung");
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    drop(restart_job);
    restart_runner.shutdown().await.unwrap();
    // Exactly one window emission: restore resumed past the committed
    // cursors, so the retained pair is neither lost nor replayed.
    assert_eq!(*rows.lock(), vec![1]);
}

/// Waits until the `match` Join of one running job has emitted `pairs`
/// match rows, bounding the wait for the restore scenario.
async fn wait_for_join_emission(job: &super::ContinuousJob, pairs: u64) {
    for _ in 0..3_000 {
        let joins = job.stream_join_status();
        if joins
            .get("match")
            .is_some_and(|status| status.emitted_match_rows >= pairs)
        {
            return;
        }
        tokio::time::sleep(StdDuration::from_millis(10)).await;
    }
    panic!("join never emitted {pairs} pairs before the restore cut");
}

/// A replayable scripted source that delivers one keyed event-time row,
/// then holds its watermark back until released, then ends. Restarts skip
/// the data row when the durable cursor covers it.
struct PausedWatermarkSource {
    key: &'static str,
    ts: i64,
    order: u64,
    watermark: i64,
    release: Arc<AtomicBool>,
    data_delivered: bool,
    watermark_delivered: bool,
}

#[async_trait]
impl StreamSource for PausedWatermarkSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if cursor.is_some_and(|cursor| cursor.order() >= self.order.to_be_bytes().as_slice()) {
            self.data_delivered = true;
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if !self.data_delivered {
            self.data_delivered = true;
            return Ok(Some(SourceEvent::Data {
                batch: join_ts_row(self.key, self.ts),
                cursor: Cursor::unbound(self.order.to_be_bytes().to_vec(), JsonMap::new()).unwrap(),
            }));
        }
        if !self.watermark_delivered {
            while !self.release.load(Ordering::SeqCst) {
                tokio::time::sleep(StdDuration::from_millis(5)).await;
            }
            self.watermark_delivered = true;
            return Ok(Some(SourceEvent::Watermark(EventTime::from_micros(
                self.watermark,
            ))));
        }
        Ok(None)
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

/// One committed Window emission of the AC14 fault matrix: the operator
/// output sequence and the tumbling window's start micros.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
struct Ac14WindowRecord {
    sequence: u64,
    window_start: i64,
    pairs: u64,
}

#[derive(Default)]
struct Ac14TransactionalState {
    committed_epochs: BTreeSet<u64>,
    visible: Vec<Ac14WindowRecord>,
}

struct Ac14TransactionalSink {
    pending: Vec<Ac14WindowRecord>,
    state: Arc<Mutex<Ac14TransactionalState>>,
}

fn ac14_records(state: &JsonMap) -> Result<Vec<Ac14WindowRecord>> {
    serde_json::from_value(state.get("records").cloned().ok_or_else(|| {
        CalcFlowError::CheckpointMismatch {
            message: "AC14 pre-commit records are missing".into(),
        }
    })?)
    .map_err(|error| CalcFlowError::CheckpointMismatch {
        message: format!("AC14 pre-commit records are invalid: {error}"),
    })
}

#[async_trait]
impl TransactionalStreamSink for Ac14TransactionalSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
        self.pending.clear();
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        for record in batch.table_payload()?.batches() {
            let starts = record
                .column_by_name("window_start")
                .expect("the window emits its start column")
                .as_any()
                .downcast_ref::<datafusion::arrow::array::TimestampMicrosecondArray>()
                .expect("microsecond window starts");
            let counts = record
                .column_by_name("pairs")
                .expect("the window emits the pairs aggregate")
                .as_any()
                .downcast_ref::<datafusion::arrow::array::UInt64Array>()
                .expect("the pairs aggregate is a UInt64 column");
            self.pending
                .extend((0..starts.len()).map(|index| Ac14WindowRecord {
                    sequence: batch.metadata().sequence() + u64::try_from(index).unwrap(),
                    window_start: starts.value(index),
                    pairs: counts.value(index),
                }));
        }
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
        Ok(BTreeMap::from([(
            "records".into(),
            serde_json::to_value(&self.pending).unwrap(),
        )]))
    }

    async fn commit(&mut self, epoch: crate::Epoch, state: &JsonMap) -> Result<()> {
        let records = ac14_records(state)?;
        let mut durable = self.state.lock();
        if durable.committed_epochs.insert(epoch.as_u64()) {
            durable.visible.extend(records);
        }
        Ok(())
    }

    async fn abort(&mut self, _epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
        self.pending.clear();
        Ok(())
    }

    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        let state = manifest
            .sinks()
            .get("window")
            .and_then(|entry| entry.pre_commit.clone())
            .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                message: "AC14 transactional recovery state is missing".into(),
            })?;
        self.commit(manifest.epoch(), &state).await
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Transactional mirror of the window output: records the committed
/// `(window_start, pairs)` multiset so the partial sink-commit crash can
/// fire between two transactional sinks on one output (spec AC14).
struct Ac14MirrorTransactionalSink {
    pending: Vec<(i64, i64)>,
    state: Arc<Mutex<Ac14TapState>>,
}

#[derive(Default)]
struct Ac14TapState {
    committed_epochs: BTreeSet<u64>,
    visible: Vec<(i64, i64)>,
}

fn ac14_pair_records(state: &JsonMap) -> Result<Vec<(i64, i64)>> {
    serde_json::from_value(state.get("windows").cloned().ok_or_else(|| {
        CalcFlowError::CheckpointMismatch {
            message: "AC14 mirror pre-commit windows are missing".into(),
        }
    })?)
    .map_err(|error| CalcFlowError::CheckpointMismatch {
        message: format!("AC14 mirror pre-commit windows are invalid: {error}"),
    })
}

#[async_trait]
impl TransactionalStreamSink for Ac14MirrorTransactionalSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
        self.pending.clear();
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        for record in batch.table_payload()?.batches() {
            let starts = record
                .column_by_name("window_start")
                .expect("the window emits its start column")
                .as_any()
                .downcast_ref::<datafusion::arrow::array::TimestampMicrosecondArray>()
                .expect("microsecond window starts");
            let counts = record
                .column_by_name("pairs")
                .expect("the window emits the pairs aggregate")
                .as_any()
                .downcast_ref::<datafusion::arrow::array::UInt64Array>()
                .expect("the pairs aggregate is a UInt64 column");
            self.pending.extend((0..starts.len()).map(|index| {
                (
                    starts.value(index),
                    i64::try_from(counts.value(index)).unwrap(),
                )
            }));
        }
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
        Ok(BTreeMap::from([(
            "windows".into(),
            serde_json::to_value(&self.pending).unwrap(),
        )]))
    }

    async fn commit(&mut self, epoch: crate::Epoch, state: &JsonMap) -> Result<()> {
        let windows = ac14_pair_records(state)?;
        let mut durable = self.state.lock();
        if durable.committed_epochs.insert(epoch.as_u64()) {
            durable.visible.extend(windows);
        }
        Ok(())
    }

    async fn abort(&mut self, _epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
        self.pending.clear();
        Ok(())
    }

    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        let state = manifest
            .sinks()
            .get("mirror")
            .and_then(|entry| entry.pre_commit.clone())
            .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                message: "AC14 mirror recovery state is missing".into(),
            })?;
        self.commit(manifest.epoch(), &state).await
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A replayable multi-row scripted source: data rows carry monotonic
/// cursor orders and the trailing watermark always replays; restarts skip
/// the data rows the durable cursor already covers. `hold_open` keeps the
/// source alive after its watermark so a manual checkpoint runs while the
/// ingresses are still active.
struct ScriptedSeekingSource {
    rows: Vec<(&'static str, i64)>,
    watermark: i64,
    next: usize,
    watermark_sent: bool,
    hold_open: bool,
}

#[async_trait]
impl StreamSource for ScriptedSeekingSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if let Some(cursor) = cursor {
            let bytes: [u8; 8] =
                cursor
                    .order()
                    .try_into()
                    .map_err(|_| CalcFlowError::CheckpointMismatch {
                        message: "AC14 cursor order is not a u64".into(),
                    })?;
            self.next =
                usize::try_from(u64::from_be_bytes(bytes)).expect("cursor order fits usize");
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.next < self.rows.len() {
            let (key, ts) = self.rows[self.next];
            let order = u64::try_from(self.next + 1).unwrap();
            self.next += 1;
            return Ok(Some(SourceEvent::Data {
                batch: join_ts_row(key, ts),
                cursor: Cursor::unbound(order.to_be_bytes().to_vec(), JsonMap::new()).unwrap(),
            }));
        }
        if !self.watermark_sent {
            self.watermark_sent = true;
            return Ok(Some(SourceEvent::Watermark(EventTime::from_micros(
                self.watermark,
            ))));
        }
        if self.hold_open {
            std::future::pending::<()>().await;
        }
        Ok(None)
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

/// Builds the AC14 Join->Window job spec with exactly-once delivery on
/// the window output (spec AC14). When `tap` is set, a second
/// transactional sink taps the raw Join output so the partial sink-commit
/// crash point can fire between the two commits.
#[allow(
    clippy::too_many_lines,
    reason = "the fault-matrix job spec is clearest as one inline fixture"
)]
fn ac14_job_spec(
    job_id: u64,
    transactional: Arc<Mutex<Ac14TransactionalState>>,
    hold_open: bool,
    tap: Option<Arc<Mutex<Ac14TapState>>>,
) -> ContinuousJobSpec {
    let schema = Arc::new(datafusion::arrow::datatypes::Schema::new(vec![
        datafusion::arrow::datatypes::Field::new(
            "key",
            datafusion::arrow::datatypes::DataType::Utf8,
            false,
        ),
        datafusion::arrow::datatypes::Field::new(
            "ts",
            datafusion::arrow::datatypes::DataType::Timestamp(
                datafusion::arrow::datatypes::TimeUnit::Microsecond,
                None,
            ),
            false,
        ),
    ]));
    let join = StreamJoinOperator::new(
        "match",
        Arc::clone(&schema),
        schema,
        StreamJoinSpec::inner(
            ["key"],
            ["key"],
            "ts",
            "ts",
            JoinTimeBounds::new(StdDuration::from_micros(0), StdDuration::from_micros(10)).unwrap(),
            JoinStateLimits::new(100_000, 134_217_728, 1_000_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let join_output = Arc::clone(join.output_ports()[0].schema().expect("derived schema"));
    let mut window_spec = WindowSpec::tumbling("left__ts", StdDuration::from_micros(10)).unwrap();
    window_spec.aggregates = vec![AggregateSpec {
        function: AggregateFunction::Count,
        column: "left__ts".into(),
        output: "pairs".into(),
    }];
    let window = WindowAggregateOperator::new("agg", join_output, window_spec).unwrap();
    let builder = PipelineBuilder::new("ac14-join-window-fault")
        .unwrap()
        .add_node("match", Box::new(join))
        .unwrap()
        .add_node("agg", Box::new(window))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("match", "output").unwrap(),
            PortEndpoint::new("agg", "input").unwrap(),
        ))
        .unwrap();
    let plan = builder
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    "output".into(),
                    crate::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .unwrap();
    ContinuousJobSpec {
        context: StreamJobContext::new(
            job_id,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![
            NamedSourceBinding {
                binding_id: "left".into(),
                binding: SourceBinding::new(
                    Box::new(ScriptedSeekingSource {
                        rows: vec![("a", 95), ("b", 40)],
                        watermark: 110,
                        next: 0,
                        watermark_sent: false,
                        hold_open,
                    }),
                    None,
                    0,
                )
                .unwrap(),
            },
            NamedSourceBinding {
                binding_id: "right".into(),
                binding: SourceBinding::new(
                    Box::new(ScriptedSeekingSource {
                        rows: vec![("a", 100), ("b", 50)],
                        watermark: 120,
                        next: 0,
                        watermark_sent: false,
                        hold_open,
                    }),
                    None,
                    0,
                )
                .unwrap(),
            },
        ],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "window".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(Ac14TransactionalSink {
                pending: Vec::new(),
                state: transactional,
            })),
        }]
        .into_iter()
        .chain(tap.map(|state| NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "mirror".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(
                Ac14MirrorTransactionalSink {
                    pending: Vec::new(),
                    state,
                },
            )),
        }))
        .collect(),
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

/// One AC14 fault-point case: crash during the terminal checkpoint,
/// recover with a clean runner, and prove the transactional Join->Window
/// output is exactly the expected pair set with non-retreating window
/// starts and exactly-restored output sequences (spec AC14).
async fn ac14_run_one_fault_point(point: super::CheckpointFaultPoint) {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let manifest_root = directory.path().join("manifests");
    let state = Arc::new(Mutex::new(Ac14TransactionalState::default()));
    let checkpoint = |faulted: bool| {
        let spec = CheckpointRuntimeSpec::new(
            backend.clone(),
            &manifest_root,
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap();
        if faulted {
            spec.with_fault(point, super::CheckpointFaultMode::Restart)
        } else {
            spec
        }
    };

    let mut first_runner = ContinuousRunner::new();
    let first_job = first_runner
        .start_checkpointed(
            ac14_job_spec(920, Arc::clone(&state), false, None),
            checkpoint(true),
        )
        .await
        .unwrap();
    let first_outcome = tokio::time::timeout(StdDuration::from_secs(10), first_job.wait())
        .await
        .unwrap_or_else(|_| panic!("AC14 fault at {point:?} hung"));
    // The injected checkpoint fault must terminate the faulted run before
    // it can report success.
    assert_ne!(
        first_outcome.state,
        ContinuousJobState::Completed,
        "AC14 fault at {point:?} must fail the run"
    );
    drop(first_job);
    first_runner.shutdown().await.unwrap();

    let mut restart_runner = ContinuousRunner::new();
    let restart_job = restart_runner
        .start_checkpointed(
            ac14_job_spec(921, Arc::clone(&state), false, None),
            checkpoint(false),
        )
        .await
        .unwrap();
    let restart_outcome = tokio::time::timeout(StdDuration::from_secs(10), restart_job.wait())
        .await
        .unwrap_or_else(|_| panic!("AC14 recovery after {point:?} hung"));
    assert_eq!(
        restart_outcome.state,
        ContinuousJobState::Completed,
        "AC14 recovery after {point:?}: {restart_outcome:?}"
    );
    drop(restart_job);
    restart_runner.shutdown().await.unwrap();

    // Zero missing and zero duplicate pairs: exactly the two expected
    // window emissions, in window-start order, with the exact restored
    // output sequences.
    let visible = state.lock().visible.clone();
    assert_eq!(
        visible,
        vec![
            Ac14WindowRecord {
                sequence: 0,
                window_start: 40,
                pairs: 1,
            },
            Ac14WindowRecord {
                sequence: 1,
                window_start: 90,
                pairs: 1,
            },
        ],
        "AC14 transactional output after {point:?}"
    );
}

/// Partial alignment only fires while an ingress is still active, so its
/// crash point runs a manual checkpoint against held-open sources: the
/// faulted epoch fails before the operator barrier completes, and the
/// recovered run still delivers the exact window set exactly once (spec
/// AC14).
#[tokio::test]
async fn ac14_partial_alignment_mid_flight_checkpoint_recovers_exactly_once() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let manifest_root = directory.path().join("manifests");
    let state = Arc::new(Mutex::new(Ac14TransactionalState::default()));
    let spec = |job_id: u64| ac14_job_spec(job_id, Arc::clone(&state), true, None);
    let checkpoint = |faulted: bool| {
        let spec = CheckpointRuntimeSpec::new(
            backend.clone(),
            &manifest_root,
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap();
        if faulted {
            spec.with_fault(
                super::CheckpointFaultPoint::PartialAlignment,
                super::CheckpointFaultMode::Restart,
            )
        } else {
            spec
        }
    };

    let mut first_runner = ContinuousRunner::new();
    let first_job = first_runner
        .start_checkpointed(spec(924), checkpoint(true))
        .await
        .unwrap();
    wait_for_join_emission(&first_job, 2).await;
    let (manual, first_failed) = tokio::time::timeout(StdDuration::from_secs(10), async {
        tokio::join!(first_job.trigger_checkpoint(), first_job.wait())
    })
    .await
    .expect("AC14 partial-alignment fault hung");
    assert!(manual.is_err(), "the faulted checkpoint must fail");
    assert_ne!(
        first_failed.state,
        ContinuousJobState::Completed,
        "the faulted run must not complete"
    );
    drop(first_job);
    first_runner.shutdown().await.unwrap();

    let mut restart_runner = ContinuousRunner::new();
    let restart_job = restart_runner
        .start_checkpointed(spec(925), checkpoint(false))
        .await
        .unwrap();
    wait_for_join_emission(&restart_job, 2).await;
    let epoch = tokio::time::timeout(StdDuration::from_secs(10), restart_job.trigger_checkpoint())
        .await
        .expect("AC14 partial-alignment restart checkpoint hung");
    assert!(epoch.is_ok(), "recovered checkpoint failed: {epoch:?}");
    assert_eq!(
        restart_job.cancel().await.state,
        ContinuousJobState::Cancelled
    );
    drop(restart_job);
    restart_runner.shutdown().await.unwrap();

    let visible = state.lock().visible.clone();
    assert_eq!(
        visible,
        vec![
            Ac14WindowRecord {
                sequence: 0,
                window_start: 40,
                pairs: 1,
            },
            Ac14WindowRecord {
                sequence: 1,
                window_start: 90,
                pairs: 1,
            },
        ],
        "AC14 partial-alignment transactional output"
    );
}

#[tokio::test]
async fn ac14_join_window_fault_matrix_recovers_exactly_once() {
    for point in super::CheckpointFaultPoint::ALL
        .into_iter()
        .filter(|point| {
            !matches!(
                point,
                super::CheckpointFaultPoint::PartialAlignment
                    | super::CheckpointFaultPoint::PartialSinkCommit
            )
        })
    {
        ac14_run_one_fault_point(point).await;
    }
}

/// The partial sink-commit crash point fires between transactional
/// sinks, so it runs a two-sink Join graph: the window sink commits, the
/// tap sink crashes after it, and recovery completes both commits forward
/// with the exact expected multisets (spec AC14).
#[tokio::test]
async fn ac14_partial_sink_commit_between_two_transactional_sinks_recovers() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let manifest_root = directory.path().join("manifests");
    let window_state = Arc::new(Mutex::new(Ac14TransactionalState::default()));
    let tap_state = Arc::new(Mutex::new(Ac14TapState::default()));
    let checkpoint = |faulted: bool| {
        let spec = CheckpointRuntimeSpec::new(
            backend.clone(),
            &manifest_root,
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap();
        if faulted {
            spec.with_fault(
                super::CheckpointFaultPoint::PartialSinkCommit,
                super::CheckpointFaultMode::Restart,
            )
        } else {
            spec
        }
    };

    let mut first_runner = ContinuousRunner::new();
    let first_job = first_runner
        .start_checkpointed(
            ac14_job_spec(
                926,
                Arc::clone(&window_state),
                false,
                Some(Arc::clone(&tap_state)),
            ),
            checkpoint(true),
        )
        .await
        .unwrap();
    let first_outcome = tokio::time::timeout(StdDuration::from_secs(10), first_job.wait())
        .await
        .expect("AC14 partial sink-commit fault hung");
    assert_ne!(
        first_outcome.state,
        ContinuousJobState::Completed,
        "the faulted two-sink run must not complete: {first_outcome:?}"
    );
    drop(first_job);
    first_runner.shutdown().await.unwrap();

    let mut restart_runner = ContinuousRunner::new();
    let restart_job = restart_runner
        .start_checkpointed(
            ac14_job_spec(
                927,
                Arc::clone(&window_state),
                false,
                Some(Arc::clone(&tap_state)),
            ),
            checkpoint(false),
        )
        .await
        .unwrap();
    let restart_outcome = tokio::time::timeout(StdDuration::from_secs(10), restart_job.wait())
        .await
        .expect("AC14 partial sink-commit recovery hung");
    assert_eq!(
        restart_outcome.state,
        ContinuousJobState::Completed,
        "AC14 partial sink-commit recovery: {restart_outcome:?}"
    );
    drop(restart_job);
    restart_runner.shutdown().await.unwrap();

    // The window sink committed before the crash; recovery re-installs
    // that epoch exactly once and the tap completes forward, so both
    // multisets are exact with no duplicates.
    let windows = window_state.lock().visible.clone();
    assert_eq!(
        windows,
        vec![
            Ac14WindowRecord {
                sequence: 0,
                window_start: 40,
                pairs: 1,
            },
            Ac14WindowRecord {
                sequence: 1,
                window_start: 90,
                pairs: 1,
            },
        ],
        "AC14 partial sink-commit window output"
    );
    let mut windows = tap_state.lock().visible.clone();
    windows.sort_unstable();
    assert_eq!(
        windows,
        vec![(40, 1), (90, 1)],
        "AC14 partial sink-commit mirror windows"
    );
}

/// The same crash point with an ordinary sink stays explicitly
/// at-least-once: every window still arrives at least once and no pair is
/// lost, while the replayed epoch may re-observe whole window batches
/// (spec AC14).
#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the at-least-once boundary is clearest as one end-to-end test"
)]
async fn ac14_ordinary_sink_stays_at_least_once_across_restart() {
    let directory = tempfile::tempdir().unwrap();
    let backend = Arc::new(
        LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap(),
    );
    let manifest_root = directory.path().join("manifests");
    let writes = Arc::new(Mutex::new(Vec::new()));
    let spec = || {
        let mut spec = ac5_job_spec(
            ac5_join_window_plan(
                JoinTimeBounds::new(StdDuration::from_micros(0), StdDuration::from_micros(10))
                    .unwrap(),
                "left__ts",
            ),
            &writes,
        );
        spec.sources = vec![
            NamedSourceBinding {
                binding_id: "left".into(),
                binding: SourceBinding::new(
                    Box::new(ScriptedSeekingSource {
                        rows: vec![("a", 95), ("b", 40)],
                        watermark: 110,
                        next: 0,
                        watermark_sent: false,
                        hold_open: false,
                    }),
                    None,
                    0,
                )
                .unwrap(),
            },
            NamedSourceBinding {
                binding_id: "right".into(),
                binding: SourceBinding::new(
                    Box::new(ScriptedSeekingSource {
                        rows: vec![("a", 100), ("b", 50)],
                        watermark: 120,
                        next: 0,
                        watermark_sent: false,
                        hold_open: false,
                    }),
                    None,
                    0,
                )
                .unwrap(),
            },
        ];
        spec
    };
    let checkpoint = |faulted: bool| {
        let spec = CheckpointRuntimeSpec::new(
            backend.clone(),
            &manifest_root,
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap();
        if faulted {
            spec.with_fault(
                super::CheckpointFaultPoint::ManifestWrite,
                super::CheckpointFaultMode::Restart,
            )
        } else {
            spec
        }
    };

    let mut first_runner = ContinuousRunner::new();
    let first_job = first_runner
        .start_checkpointed(spec(), checkpoint(true))
        .await
        .unwrap();
    let first_outcome = tokio::time::timeout(StdDuration::from_secs(10), first_job.wait())
        .await
        .expect("AC14 ordinary fault run hung");
    assert_ne!(first_outcome.state, ContinuousJobState::Completed);
    drop(first_job);
    first_runner.shutdown().await.unwrap();

    let mut restart_runner = ContinuousRunner::new();
    let restart_job = restart_runner
        .start_checkpointed(spec(), checkpoint(false))
        .await
        .unwrap();
    let restart_outcome = tokio::time::timeout(StdDuration::from_secs(10), restart_job.wait())
        .await
        .expect("AC14 ordinary restart hung");
    assert_eq!(restart_outcome.state, ContinuousJobState::Completed);
    drop(restart_job);
    restart_runner.shutdown().await.unwrap();

    // Zero missing: both tumbling windows arrived, each carrying exactly
    // one pair. At-least-once replay may add whole-batch duplicates, so
    // only the per-window pair counts and the minimum coverage are exact.
    let observed = writes.lock().clone();
    assert!(
        observed.len() >= 2,
        "ordinary sink lost window emissions: {observed:?}"
    );
    assert!(
        observed.iter().all(|count| *count == 1),
        "window pair counts changed across replay: {observed:?}"
    );
}
