mod checkpoint_tests;

use super::*;
use crate::runtime::streaming::{
    entity_work::TestHooks,
    failure::{FailureOrigin, TerminalCause},
};
use std::{
    sync::{Condvar, atomic::AtomicU64},
    time::Duration,
};
use tokio::sync::Notify;

struct GatedSource {
    source: Source,
    start: Option<Arc<Notify>>,
    hold_open: bool,
}

#[async_trait]
impl StreamSource for GatedSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if let Some(cursor) = cursor {
            while self.source.0.front().is_some_and(|event| match event {
                SourceEvent::Data {
                    cursor: candidate, ..
                } => candidate.order() <= cursor.order(),
                _ => true,
            }) {
                self.source.0.pop_front();
            }
        }
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if let Some(start) = self.start.take() {
            start.notified().await;
        }
        if self.source.0.is_empty() && self.hold_open {
            std::future::pending().await
        } else {
            self.source.next().await
        }
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
    fn capabilities(&self) -> SourceCapabilities {
        self.source.capabilities()
    }
}

struct Fixture {
    spec: ContinuousJobSpec,
    start: Arc<Notify>,
    records: Arc<Mutex<Vec<RecordBatch>>>,
    cancellation: CancellationToken,
}

fn fixture(hold_open: bool) -> Fixture {
    let input = input();
    let plan = plan();
    let records = Arc::new(Mutex::new(Vec::new()));
    let start = Arc::new(Notify::new());
    let source = GatedSource {
        source: Source(VecDeque::from([
            SourceEvent::Data {
                batch: Batch::table(vec![input.slice(0, PRELOAD)], BatchMetadata::default())
                    .unwrap(),
                cursor: Cursor::unbound(vec![1], JsonMap::new()).unwrap(),
            },
            SourceEvent::Watermark(EventTime::from_micros(19)),
            SourceEvent::Data {
                batch: Batch::table(vec![input.slice(PRELOAD, ROWS)], BatchMetadata::default())
                    .unwrap(),
                cursor: Cursor::unbound(vec![2], JsonMap::new()).unwrap(),
            },
            SourceEvent::Watermark(EventTime::from_micros(1019)),
        ])),
        start: Some(start.clone()),
        hold_open,
    };
    let cancellation = CancellationToken::new();
    Fixture {
        spec: ContinuousJobSpec {
            context: StreamJobContext::new(
                7,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                cancellation.clone(),
            ),
            plan,
            sources: vec![NamedSourceBinding {
                binding_id: "input".into(),
                binding: SourceBinding::new(Box::new(source), None, 0).unwrap(),
            }],
            sinks: vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "sink".into(),
                binding: OrdinarySinkBinding::new(Box::new(Sink(records.clone()))),
            }],
            edge_budget: EdgeBudget::new(ROWS, 8 << 20).unwrap(),
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
            static_inputs: crate::static_input::PreparedStaticInputs::default(),
        },
        start,
        records,
        cancellation,
    }
}

#[derive(Default)]
struct WorkerGate {
    entered: AtomicU64,
    released: std::sync::Mutex<bool>,
    changed: Condvar,
}
impl WorkerGate {
    fn block(&self) {
        self.entered.fetch_add(1, Ordering::SeqCst);
        let mut released = self.released.lock().unwrap();
        while !*released {
            released = self.changed.wait(released).unwrap();
        }
    }
    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.changed.notify_all();
    }
}
struct Release(Arc<WorkerGate>);
impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}

async fn wait_until(condition: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !condition() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("test condition must become observable");
}

fn block_after_numeric(job: &super::super::ContinuousJob, gate: &Arc<WorkerGate>) {
    job.core.entity_work.set_hooks(TestHooks {
        after_run: Some({
            let gate = gate.clone();
            Arc::new(move |_| gate.block())
        }),
        ..TestHooks::default()
    });
}

fn rows(records: &Arc<Mutex<Vec<RecordBatch>>>) -> usize {
    records.lock().iter().map(RecordBatch::num_rows).sum()
}

#[tokio::test]
async fn cancelled_managed_pair_joins_before_terminal_and_never_emits_stale_candidate() {
    let f = fixture(true);
    let mut runner = ContinuousRunner::new();
    let job = runner.start(f.spec).await.unwrap();
    let gate = Arc::new(WorkerGate::default());
    let _release = Release(gate.clone());
    block_after_numeric(&job, &gate);
    f.start.notify_one();
    wait_until(|| gate.entered.load(Ordering::SeqCst) == 2).await;
    let outcome = job.cancel();
    tokio::pin!(outcome);
    wait_until(|| f.cancellation.is_cancelled()).await;
    assert!(futures::poll!(&mut outcome).is_pending());
    assert_eq!(job.rolling_metrics()["rolling"].watermark.cancelled, 0);
    assert_eq!(rows(&f.records), PRELOAD);
    gate.release();
    let outcome = tokio::time::timeout(Duration::from_secs(5), &mut outcome)
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Cancelled);
    assert_eq!(job.rolling_metrics()["rolling"].watermark.cancelled, 1);
    assert_eq!(
        job.rolling_metrics()["rolling"].watermark.numeric_rows,
        TOTAL as u64
    );
    assert_eq!(rows(&f.records), PRELOAD);
    assert!(job.core.runtime_status.lock().tasks.snapshot().is_empty());
    assert_eq!(job.core.entity_work.active_bytes(), 0);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn driver_abort_drains_real_lanes_then_resolves_the_claimed_job_waiter() {
    let f = fixture(true);
    let mut runner = ContinuousRunner::new();
    let job = runner.start(f.spec).await.unwrap();
    let gate = Arc::new(WorkerGate::default());
    let _release = Release(gate.clone());
    block_after_numeric(&job, &gate);
    f.start.notify_one();
    wait_until(|| gate.entered.load(Ordering::SeqCst) == 2).await;
    job.core.abort_driver_for_test();
    wait_until(|| f.cancellation.is_cancelled()).await;
    let mut outcome = Box::pin(job.wait());
    assert!(futures::poll!(&mut outcome).is_pending());
    gate.release();
    let result = tokio::time::timeout(Duration::from_secs(2), &mut outcome).await;
    assert!(
        result.is_ok(),
        "a claimed job must receive an outcome after driver abort"
    );
    let outcome = result.unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Failed);
    assert!(
        outcome
            .errors
            .iter()
            .any(|failure| failure.origin == FailureOrigin::RunnerLifecycle)
    );
    assert_eq!(rows(&f.records), PRELOAD);
    assert!(job.core.runtime_status.lock().tasks.snapshot().is_empty());
    assert_eq!(job.core.entity_work.active_bytes(), 0);
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn prepared_completed_report_survives_a_real_driver_panic_before_publication() {
    let f = fixture(false);
    let mut runner = ContinuousRunner::new();
    let job = runner.start(f.spec).await.unwrap();
    job.core.panic_after_prepared_report_for_test();
    f.start.notify_one();
    let outcome = tokio::time::timeout(Duration::from_secs(5), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed);
    assert!(outcome.errors.is_empty());
    assert_eq!(rows(&f.records), TOTAL);
    assert_eq!(job.core.owned_lane_launches.load(Ordering::SeqCst), 2);
    assert!(job.core.runtime_status.lock().tasks.snapshot().is_empty());
    let weak = Arc::downgrade(&job.core);
    drop(job);
    runner.shutdown().await.unwrap();
    assert!(weak.upgrade().is_none());
}

#[tokio::test]
async fn worker_panic_preserves_original_operator_id_and_interrupted_p2_work() {
    let f = fixture(true);
    let mut runner = ContinuousRunner::new();
    let job = runner.start(f.spec).await.unwrap();
    job.core.entity_work.set_hooks(TestHooks {
        after_run: Some(Arc::new(|lane| {
            assert!(lane != 1, "real numeric worker panic");
        })),
        ..TestHooks::default()
    });
    f.start.notify_one();
    let outcome = tokio::time::timeout(Duration::from_secs(5), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Failed);
    let TerminalCause::TaskFailure { primary_task_id } = outcome.cause else {
        panic!("worker failure keeps its logical task");
    };
    assert_eq!(primary_task_id.as_u64(), 0);
    assert!(
        matches!(&outcome.errors[0].origin, FailureOrigin::Task { task_id, task_name } if task_id == &primary_task_id && task_name == "operator:rolling")
    );
    assert!(
        matches!(&outcome.errors[0].error, crate::CalcFlowError::TaskPanicked { task_id: 0, message } if message == "real numeric worker panic")
    );
    let metrics = &job.rolling_metrics()["rolling"].watermark;
    assert_eq!(metrics.interrupted, 1);
    assert_eq!(metrics.failed, 0);
    assert_eq!(metrics.numeric_rows, TOTAL as u64);
    assert_eq!(rows(&f.records), PRELOAD);
    assert!(job.core.runtime_status.lock().tasks.snapshot().is_empty());
    drop(job);
    runner.shutdown().await.unwrap();
}

#[tokio::test]
async fn last_job_owner_drop_keeps_the_reaper_alive_until_the_real_workers_exit() {
    let f = fixture(true);
    let mut runner = ContinuousRunner::new();
    let job = runner.start(f.spec).await.unwrap();
    let weak = Arc::downgrade(&job.core);
    let gate = Arc::new(WorkerGate::default());
    let _release = Release(gate.clone());
    block_after_numeric(&job, &gate);
    f.start.notify_one();
    wait_until(|| gate.entered.load(Ordering::SeqCst) == 2).await;
    drop(job);
    wait_until(|| f.cancellation.is_cancelled()).await;
    let mut shutdown = Box::pin(runner.shutdown());
    assert!(futures::poll!(&mut shutdown).is_pending());
    assert!(weak.upgrade().is_some());
    gate.release();
    tokio::time::timeout(Duration::from_secs(5), &mut shutdown)
        .await
        .unwrap()
        .unwrap();
    assert!(weak.upgrade().is_none());
    assert_eq!(rows(&f.records), PRELOAD);
}

#[tokio::test]
async fn driver_takeover_preserves_settled_tasks_while_only_cpu_work_remains() {
    let f = fixture(true);
    let mut runner = ContinuousRunner::new();
    let job = runner.start(f.spec).await.unwrap();
    let gate = Arc::new(WorkerGate::default());
    let _release = Release(gate.clone());
    block_after_numeric(&job, &gate);
    f.start.notify_one();
    wait_until(|| gate.entered.load(Ordering::SeqCst) == 2).await;
    let tasks = job.core.runtime_status.lock().tasks.clone();
    tasks.abort_all_for_test();
    wait_until(|| tasks.snapshot().is_empty()).await;
    assert!(job.core.entity_work.active_bytes() > 0);
    assert_eq!(
        job.rolling_metrics()["rolling"].watermark.numeric_rows,
        PRELOAD as u64
    );
    job.core.abort_driver_for_test();
    let mut outcome = Box::pin(job.wait());
    assert!(futures::poll!(&mut outcome).is_pending());
    gate.release();
    let outcome = tokio::time::timeout(Duration::from_secs(5), &mut outcome)
        .await
        .unwrap();
    assert!(matches!(outcome.cause, TerminalCause::TaskFailure { .. }));
    assert!(
        outcome
            .errors
            .iter()
            .any(|failure| matches!(failure.origin, FailureOrigin::Task { .. }))
    );
    assert_eq!(
        job.rolling_metrics()["rolling"].watermark.numeric_rows,
        TOTAL as u64
    );
    assert_eq!(job.core.entity_work.active_bytes(), 0);
    assert_eq!(rows(&f.records), PRELOAD);
    assert!(tasks.snapshot().is_empty());
    drop(job);
    runner.shutdown().await.unwrap();
}
