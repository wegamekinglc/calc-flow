use super::*;
use crate::{
    AsofJoinSide, AsofStateLimits, Epoch, LocalStateBackend, StateBackend, StateHandle,
    StateLineageBackend, StateLineageKey, StateSegment, StreamAsofJoinOperator, StreamAsofJoinSpec,
    StreamingFailureReason,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use std::time::Duration;

const BUDGET: u64 = 16_384;

fn operator() -> CompiledStreamOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["sequence".into()],
            prefix.into(),
        )
        .unwrap()
    };
    CompiledStreamOperator::StreamAsofJoin(Box::new(
        StreamAsofJoinOperator::new(
            "asof",
            schema.clone(),
            schema,
            StreamAsofJoinSpec::new(
                side("left"),
                side("right"),
                Duration::from_micros(1),
                AsofStateLimits::new(100, BUDGET).unwrap(),
            )
            .unwrap(),
        )
        .unwrap(),
    ))
}

fn entry(segments: Vec<StateHandle>) -> OperatorManifestEntry {
    OperatorManifestEntry {
        progress: BTreeMap::new(),
        inline_metadata: BTreeMap::new(),
        segments,
    }
}

fn handle(owner: &str, length: u64) -> StateHandle {
    StateHandle::new(
        owner,
        Epoch::INITIAL,
        "index",
        "committed/missing.segment",
        length,
        &"0".repeat(64),
    )
    .unwrap()
}

fn assert_workspace(error: &CalcFlowError) {
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        }
    ));
}

async fn transaction(
    directory: &std::path::Path,
) -> (Arc<ManifestTransaction>, Arc<dyn StateLineageBackend>) {
    let backend = LocalStateBackend::new(directory.join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new("preload", &"1".repeat(64)).unwrap();
    let lineage: Arc<dyn StateLineageBackend> =
        Arc::from(backend.open_lineage(&key).await.unwrap());
    let transaction =
        ManifestTransaction::open(lineage.clone(), &key, directory.join("manifests"), 2)
            .await
            .unwrap();
    (Arc::new(transaction), lineage)
}

#[tokio::test]
async fn test_loaded_asof_segments_keep_prepaid_wire_budget_until_last_owner_drops() {
    let directory = tempfile::tempdir().unwrap();
    let (transaction, lineage) = transaction(directory.path()).await;
    let staged = transaction
        .stage_operator_state(
            "asof",
            Epoch::INITIAL,
            OperatorStateSnapshot {
                inline_metadata: BTreeMap::new(),
                segments: BTreeMap::from([("index".into(), StateSegment::new(vec![7; 1024]))]),
            },
        )
        .await
        .unwrap();
    for handle in &staged.segments {
        lineage.publish_segment(handle).await.unwrap();
    }
    let operator = operator();
    let snapshot = load_snapshot(
        &transaction,
        &LoadOwner::default(),
        &operator,
        "asof",
        &entry(staged.segments),
        &CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(snapshot.segments["index"].bytes(), &[7; 1024]);
    let clone = snapshot.clone();
    let CompiledStreamOperator::StreamAsofJoin(asof) = &operator else {
        unreachable!();
    };
    assert_workspace(&asof.reserve_checkpoint_preload([BUDGET - 256]).unwrap_err());
    drop(snapshot);
    assert_workspace(&asof.reserve_checkpoint_preload([BUDGET - 256]).unwrap_err());
    drop(clone);
    let reservation = asof.reserve_checkpoint_preload([BUDGET - 256]).unwrap();
    assert_eq!(reservation.size(), usize::try_from(BUDGET).unwrap());
}

#[test]
fn test_asof_preload_validates_all_handle_identities_before_reserving() {
    let operator = operator();
    for handles in [
        vec![handle("asof", BUDGET * 2), handle("other", 1)],
        vec![handle("asof", BUDGET * 2), handle("asof", BUDGET * 2)],
    ] {
        let error = preload_owner(&operator, "asof", &entry(handles)).unwrap_err();
        assert!(matches!(error, CalcFlowError::CheckpointMismatch { .. }));
    }
    let owner = preload_owner(&operator, "asof", &entry(vec![handle("asof", 3840)]))
        .unwrap()
        .unwrap();
    assert!(owner.size() >= 4096);
}

#[tokio::test]
async fn test_asof_preload_refuses_before_storage_and_refunds_failed_loads() {
    let directory = tempfile::tempdir().unwrap();
    let (transaction, _) = transaction(directory.path()).await;
    let operator = operator();
    let error = load_snapshot(
        &transaction,
        &LoadOwner::default(),
        &operator,
        "asof",
        &entry(vec![handle("asof", BUDGET * 2)]),
        &CancellationToken::new(),
    )
    .await
    .unwrap_err();
    assert_workspace(&error);
    let error = load_snapshot(
        &transaction,
        &LoadOwner::default(),
        &operator,
        "asof",
        &entry(vec![handle("asof", 3840)]),
        &CancellationToken::new(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::InvalidArgument { .. }));
    let owner = preload_owner(&operator, "asof", &entry(vec![handle("asof", 3840)]))
        .unwrap()
        .unwrap();
    assert!(owner.size() >= 4096);
}

#[tokio::test]
async fn test_cancelled_asof_load_does_not_reserve_or_read_state() {
    let directory = tempfile::tempdir().unwrap();
    let (transaction, _) = transaction(directory.path()).await;
    let operator = operator();
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let error = load_snapshot(
        &transaction,
        &LoadOwner::default(),
        &operator,
        "asof",
        &entry(vec![handle("other", u64::MAX)]),
        &cancellation,
    )
    .await
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    let owner = preload_owner(&operator, "asof", &entry(vec![handle("asof", 3840)]))
        .unwrap()
        .unwrap();
    assert!(owner.size() >= 4096);
}

fn core() -> Arc<super::super::JobCore> {
    let (commands, _) = tokio::sync::mpsc::unbounded_channel();
    Arc::new(super::super::JobCore::new(
        super::super::LaunchId::new(1),
        91,
        commands,
        super::super::MetricsRecorder::default(),
        super::super::StatusProjection::default(),
        true,
        "preload".into(),
    ))
}

fn assert_load_report(report: &super::super::DriverReport, fail: bool) {
    assert_eq!(report.cleanup_failures.len(), usize::from(fail));
    if fail {
        assert!(
            report.cleanup_failures[0]
                .error
                .to_string()
                .contains("late ASOF load failure")
        );
    }
}

async fn abandoned_load_report(prepared: bool, fail: bool) {
    let core = core();
    let operator = operator();
    let credit = preload_owner(&operator, "asof", &entry(vec![handle("asof", 1024)]))
        .unwrap()
        .unwrap();
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, gate) = std::sync::mpsc::channel();
    let observer_core = core.clone();
    let observer = tokio::spawn(async move {
        observer_core
            .asof_loads
            .load(async move {
                let bytes = tokio::task::spawn_blocking(move || {
                    let bytes = vec![7; 1024];
                    entered.send(()).unwrap();
                    gate.recv().unwrap();
                    bytes
                })
                .await
                .unwrap();
                if fail {
                    return Err(CalcFlowError::Internal {
                        message: "late ASOF load failure".into(),
                    });
                }
                Ok(OperatorStateSnapshot {
                    inline_metadata: BTreeMap::new(),
                    segments: BTreeMap::from([(
                        "index".into(),
                        StateSegment::new(bytes).with_owner(credit),
                    )]),
                })
            })
            .await
    });
    tokio::time::timeout(Duration::from_secs(5), started)
        .await
        .unwrap()
        .unwrap();
    observer.abort();
    assert!(observer.await.unwrap_err().is_cancelled());
    let CompiledStreamOperator::StreamAsofJoin(asof) = &operator else {
        unreachable!();
    };
    let paid_after_abort = matches!(
        asof.reserve_checkpoint_preload([BUDGET - 256]),
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    );
    if prepared {
        core.supervision
            .prepare(super::super::DriverReport::aborted(
                core.launch_id,
                "prepared before load settlement",
            ));
    }
    let mut report = Box::pin(super::super::settle_driver_report(
        &core,
        Some("aborted load"),
    ));
    let early = match futures::poll!(report.as_mut()) {
        std::task::Poll::Pending => None,
        std::task::Poll::Ready(report) => Some(report),
    };
    let waited = early.is_none();
    let paid_before_release = matches!(
        asof.reserve_checkpoint_preload([BUDGET - 256]),
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    );
    release.send(()).unwrap();
    let report = match early {
        Some(report) => {
            let _ = core.asof_loads.close_and_drain().await;
            report
        }
        None => tokio::time::timeout(Duration::from_secs(5), report)
            .await
            .unwrap(),
    };
    let reservation = asof.reserve_checkpoint_preload([BUDGET - 256]).unwrap();
    assert_eq!(reservation.size(), usize::try_from(BUDGET).unwrap());
    assert!(paid_after_abort && paid_before_release);
    assert!(
        waited,
        "report published while the actual wire read was gated"
    );
    assert_load_report(&report, fail);
    assert!(core.asof_loads.close_and_drain().await.is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn test_aborted_asof_load_without_supervisor_delays_report_and_refund_until_real_exit() {
    abandoned_load_report(false, false).await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_prepared_report_keeps_abandoned_asof_load_and_appends_late_error_once() {
    abandoned_load_report(true, true).await;
}
