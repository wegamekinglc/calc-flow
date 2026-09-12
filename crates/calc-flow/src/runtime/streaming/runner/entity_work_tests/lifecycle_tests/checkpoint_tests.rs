use super::*;
use std::path::Path;

use crate::{
    CheckpointManifest, StreamRuntimeConfig,
    runtime::streaming::{
        checkpoint::ManagedCheckpointRuntime, checkpoint_runtime::CheckpointRuntimeSpec,
    },
};

fn checkpoint(root: &Path) -> CheckpointRuntimeSpec {
    CheckpointRuntimeSpec::managed(
        ManagedCheckpointRuntime::new(root).unwrap(),
        StreamRuntimeConfig {
            checkpoint_interval: Duration::from_secs(3_600),
            checkpoint_timeout: Duration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
}

fn continuation_input() -> RecordBatch {
    let end = TOTAL + 2 * ROWS;
    let symbols = (0..ENTITIES)
        .map(|entity| format!("S{entity:03}"))
        .collect::<Vec<_>>();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                (TOTAL..end).map(|row| i64::try_from(row / ENTITIES).unwrap()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(UInt64Array::from_iter_values(
            (TOTAL..end).map(|row| u64::try_from(row).unwrap()),
        )),
        Arc::new(StringArray::from_iter_values(
            (TOTAL..end).map(|row| symbols[row % ENTITIES].as_str()),
        )),
        Arc::new(Float64Array::from_iter_values((TOTAL..end).map(price))),
    ];
    RecordBatch::try_new(schema(), columns).unwrap()
}

struct ResumedSource {
    source: GatedSource,
    second_batch: Option<Arc<Notify>>,
    restored_cursor: Arc<Mutex<Option<Vec<u8>>>>,
}

#[async_trait]
impl StreamSource for ResumedSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        *self.restored_cursor.lock() = cursor.as_ref().map(|cursor| cursor.order().to_vec());
        self.source.open(cursor).await
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.source.source.0.front().is_some_and(
            |event| matches!(event, SourceEvent::Data { cursor, .. } if cursor.order() == [4]),
        ) {
            if let Some(gate) = self.second_batch.take() {
                gate.notified().await;
            }
        }
        self.source.next().await
    }

    async fn close(&mut self) -> Result<()> {
        self.source.close().await
    }

    fn capabilities(&self) -> SourceCapabilities {
        self.source.capabilities()
    }
}

fn resumed_fixture(
    input: &RecordBatch,
    second_batch: Arc<Notify>,
    restored_cursor: Arc<Mutex<Option<Vec<u8>>>>,
) -> Fixture {
    let mut f = fixture(false);
    let events = (0..2)
        .flat_map(|index| {
            [
                SourceEvent::Data {
                    batch: Batch::table(
                        vec![input.slice(index * ROWS, ROWS)],
                        BatchMetadata::default(),
                    )
                    .unwrap(),
                    cursor: Cursor::unbound(vec![u8::try_from(index + 3).unwrap()], JsonMap::new())
                        .unwrap(),
                },
                SourceEvent::Watermark(EventTime::from_micros(
                    i64::try_from((TOTAL + (index + 1) * ROWS - 1) / ENTITIES).unwrap(),
                )),
            ]
        })
        .collect();
    f.spec.sources = vec![NamedSourceBinding {
        binding_id: "input".into(),
        binding: SourceBinding::new(
            Box::new(ResumedSource {
                source: GatedSource {
                    source: Source(events),
                    start: Some(f.start.clone()),
                    hold_open: false,
                },
                second_batch: Some(second_batch),
                restored_cursor,
            }),
            None,
            0,
        )
        .unwrap(),
    }];
    f
}

async fn checkpoint_a_real_pair(root: &Path) {
    let f = fixture(true);
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(f.spec, checkpoint(root))
        .await
        .unwrap();
    let weak = Arc::downgrade(&job.core);
    let gate = Arc::new(WorkerGate::default());
    let _release = Release(gate.clone());
    block_after_numeric(&job, &gate);
    f.start.notify_one();
    wait_until(|| gate.entered.load(Ordering::SeqCst) == 2).await;
    let epoch = {
        let cut = job.trigger_checkpoint();
        tokio::pin!(cut);
        assert!(futures::poll!(&mut cut).is_pending());
        assert_eq!(rows(&f.records), PRELOAD);
        assert!(job.core.entity_work.active_bytes() > 0);
        gate.release();
        tokio::time::timeout(Duration::from_secs(10), &mut cut)
            .await
            .unwrap()
            .unwrap()
    };
    let manifest = root
        .join("manifests")
        .join(format!("manifest-{:020}.json", epoch.as_u64()));
    let manifest =
        CheckpointManifest::from_bytes(&tokio::fs::read(manifest).await.unwrap()).unwrap();
    assert_eq!(
        manifest.sources()["input"].cursor.as_ref().unwrap().order,
        "02"
    );
    assert!(!manifest.operators()["rolling"].segments.is_empty());
    assert_eq!(rows(&f.records), TOTAL);
    assert_eq!(job.core.owned_lane_launches.load(Ordering::SeqCst), 2);
    assert_eq!(job.core.entity_work.active_bytes(), 0);
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    assert!(job.core.runtime_status.lock().tasks.snapshot().is_empty());
    drop(job);
    runner.shutdown().await.unwrap();
    assert!(
        weak.upgrade().is_none(),
        "checkpoint participant ownership must drain"
    );
}

fn assert_continuation(records: &Arc<Mutex<Vec<RecordBatch>>>, input: &RecordBatch) {
    let actual = {
        let records = records.lock();
        concat_batches(&records[0].schema(), records.iter()).unwrap()
    };
    assert_eq!(actual.num_rows(), 2 * ROWS);
    assert_eq!(&actual.columns()[..4], input.columns());
    let spread = actual
        .column(4)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(spread.null_count(), 0);
    for index in 0..actual.num_rows() {
        let row = TOTAL + index;
        let mean = |window| {
            (0..window)
                .map(|offset| price(row - offset * ENTITIES))
                .sum::<f64>()
                / f64::from(u32::try_from(window).unwrap())
        };
        assert!((spread.value(index) - (mean(5) - mean(20))).abs() < 1e-10);
    }
}

#[tokio::test]
async fn real_checkpoint_drains_lanes_and_restores_serial_before_the_next_warm_pair() {
    let directory = tempfile::tempdir().unwrap();
    checkpoint_a_real_pair(directory.path()).await;
    let input = continuation_input();
    let second_batch = Arc::new(Notify::new());
    let restored_cursor = Arc::new(Mutex::new(None));
    let f = resumed_fixture(&input, second_batch.clone(), restored_cursor.clone());
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(f.spec, checkpoint(directory.path()))
        .await
        .unwrap();
    let weak = Arc::downgrade(&job.core);
    assert_eq!(*restored_cursor.lock(), Some(vec![2]));
    f.start.notify_one();
    wait_until(|| rows(&f.records) == ROWS).await;
    assert_eq!(job.core.owned_lane_launches.load(Ordering::SeqCst), 0);
    second_batch.notify_one();
    let outcome = tokio::time::timeout(Duration::from_secs(10), job.wait())
        .await
        .unwrap();
    assert_eq!(
        outcome.state,
        ContinuousJobState::Completed,
        "{:?}",
        outcome.errors
    );
    assert!(outcome.errors.is_empty());
    assert_eq!(job.core.owned_lane_launches.load(Ordering::SeqCst), 2);
    assert_continuation(&f.records, &input);
    assert!(job.core.runtime_status.lock().tasks.snapshot().is_empty());
    assert_eq!(job.core.entity_work.active_bytes(), 0);
    drop(job);
    runner.shutdown().await.unwrap();
    assert!(weak.upgrade().is_none());
}

#[tokio::test]
async fn claimed_terminal_manifest_driver_abort_resolves_without_a_supervisor_or_fake_task() {
    let directory = tempfile::tempdir().unwrap();
    let f = fixture(false);
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(f.spec, checkpoint(directory.path()))
        .await
        .unwrap();
    f.start.notify_one();
    let outcome = tokio::time::timeout(Duration::from_secs(10), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert_eq!(rows(&f.records), TOTAL);
    drop(job);
    runner.shutdown().await.unwrap();

    let f = fixture(false);
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(f.spec, checkpoint(directory.path()))
        .await
        .unwrap();
    job.core.abort_driver_for_test();
    let result = tokio::time::timeout(Duration::from_secs(2), job.wait()).await;
    assert!(
        result.is_ok(),
        "claimed terminal recovery must publish an outcome after driver abort"
    );
    let outcome = result.unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Failed);
    assert_eq!(outcome.cause, TerminalCause::RunnerFailure);
    assert!(
        matches!(outcome.errors.as_slice(), [failure] if failure.origin == FailureOrigin::RunnerLifecycle)
    );
    assert_eq!(job.core.owned_lane_launches.load(Ordering::SeqCst), 0);
    assert!(job.core.runtime_status.lock().tasks.snapshot().is_empty());
    let weak = Arc::downgrade(&job.core);
    drop(job);
    runner.shutdown().await.unwrap();
    assert!(weak.upgrade().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn classified_terminal_recovery_failure_closes_claim_before_report_publication() {
    use crate::runtime::streaming::failure::LaunchDeliveryState;

    struct ReleaseReport(Arc<super::super::super::DriverReportGate>);
    impl Drop for ReleaseReport {
        fn drop(&mut self) {
            self.0.release.notify_one();
        }
    }

    let directory = tempfile::tempdir().unwrap();
    let f = fixture(false);
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(f.spec, checkpoint(directory.path()))
        .await
        .unwrap();
    f.start.notify_one();
    assert_eq!(job.wait().await.state, ContinuousJobState::Completed);
    drop(job);
    runner.shutdown().await.unwrap();

    let f = fixture(false);
    let mut runner = ContinuousRunner::new();
    let mut start = Box::pin(runner.start_checkpointed(f.spec, checkpoint(directory.path())));
    let core = start.core.as_ref().unwrap().clone();
    let gate = core.pause_report_publication_for_test();
    let _release = ReleaseReport(gate.clone());
    wait_until(|| core.state.lock().launch_delivery == LaunchDeliveryState::ReadyUnclaimed).await;
    core.abort_driver_for_test();
    tokio::time::timeout(Duration::from_secs(5), gate.entered.notified())
        .await
        .unwrap();
    assert!(
        futures::poll!(&mut start).is_pending(),
        "an already classified undelivered failure must close the claim gate before publication"
    );
    gate.release.notify_one();
    let failure = tokio::time::timeout(Duration::from_secs(5), &mut start)
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(failure.primary.origin, FailureOrigin::Preflight);
    assert!(matches!(
        &failure.primary.error,
        crate::CalcFlowError::Internal { message } if message.starts_with("job driver join failed:")
    ));
    assert!(core.state.lock().outcome.is_none());
    assert!(core.state.lock().start_failure.is_some());
    assert!(core.runtime_status.lock().tasks.snapshot().is_empty());
    drop(start);
    let weak = Arc::downgrade(&core);
    drop(core);
    runner.shutdown().await.unwrap();
    assert!(weak.upgrade().is_none());
}
