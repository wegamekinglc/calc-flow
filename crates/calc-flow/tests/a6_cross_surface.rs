#[path = "support/restart_vector.rs"]
mod restart_vector;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    future::Future,
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CheckpointManifest, Cursor, DeliveryGuarantee, ExpressionOperator,
    JobState, JsonMap, ManagedCheckpointRuntime, NativeWatermarkCapability, PipelineBuilder,
    ReplayPositioning, Result, SinkBinding, SinkRecovery, SourceBinding, SourceCapabilities,
    SourceDeliveryCapability, SourceEvent, SourceSchema, StreamRequirements, StreamSource,
    StreamingJob, StreamingRunner, TransactionalStreamSink, UdfRegistry, WatermarkPolicy,
};
use command_group::{CommandGroup, GroupChild};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
use restart_vector::{RestartRecordVector, RestartVector, restart_vector};
use serde::{Deserialize, Serialize};
use tokio::sync::{oneshot, watch};

const RUN_GATE: &str = "CALC_FLOW_A6_CROSS_SURFACE_E2E";
const WORKER_MODE: &str = "CALC_FLOW_A6_CROSS_SURFACE_MODE";
const WORKER_MANAGED_ROOT: &str = "CALC_FLOW_A6_CROSS_SURFACE_MANAGED_ROOT";
const WORKER_SINK_ROOT: &str = "CALC_FLOW_A6_CROSS_SURFACE_SINK_ROOT";
const WORKER_REPORT: &str = "CALC_FLOW_A6_CROSS_SURFACE_REPORT";
const WORKER_OPERATION_TIMEOUT: Duration = Duration::from_secs(30);
const WORKER_PROCESS_TIMEOUT: Duration = Duration::from_secs(120);
const WORKER_PROCESS_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);
const WORKER_LOG_EXCERPT_BYTES: usize = 8 * 1_024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct WorkerReport {
    surface: String,
    mode: String,
    plan_fingerprint: String,
    opened_offset: usize,
    outcome_state: String,
    completed_epoch: Option<u64>,
    task_count: usize,
    charged_edges: usize,
    source_closes: usize,
    sink_closes: usize,
    visible_values: Vec<i64>,
    temporary_artifacts: usize,
}

#[derive(Clone)]
struct WorkerProbe {
    opened_offsets: Arc<Mutex<Vec<usize>>>,
    source_closes: Arc<AtomicUsize>,
    sink_closes: Arc<AtomicUsize>,
    writes: watch::Sender<usize>,
}

impl Default for WorkerProbe {
    fn default() -> Self {
        let (writes, _) = watch::channel(0);
        Self {
            opened_offsets: Arc::default(),
            source_closes: Arc::default(),
            sink_closes: Arc::default(),
            writes,
        }
    }
}

impl WorkerProbe {
    async fn wait_for_writes(&self, expected: usize) {
        self.wait_for_writes_with_interlock(expected, None).await;
    }

    async fn wait_for_writes_with_interlock(
        &self,
        expected: usize,
        interlock: Option<(oneshot::Sender<()>, oneshot::Receiver<()>)>,
    ) {
        let mut writes = self.writes.subscribe();
        let mut interlock = interlock;
        loop {
            if *writes.borrow_and_update() >= expected {
                break;
            }
            if let Some((checked, resume)) = interlock.take() {
                checked.send(()).expect("write-wait test remains active");
                resume.await.expect("write-wait test resumes waiter");
            }
            writes
                .changed()
                .await
                .expect("cross-surface write counter remains open");
        }
    }
}

#[tokio::test]
async fn worker_write_wait_preserves_update_between_check_and_await() {
    let probe = WorkerProbe::default();
    let waiting_probe = probe.clone();
    let (checked_sender, checked_receiver) = oneshot::channel();
    let (resume_sender, resume_receiver) = oneshot::channel();
    let waiter = tokio::spawn(async move {
        waiting_probe
            .wait_for_writes_with_interlock(1, Some((checked_sender, resume_receiver)))
            .await;
    });

    checked_receiver.await.unwrap();
    probe.writes.send_modify(|writes| *writes += 1);
    resume_sender.send(()).unwrap();
    tokio::time::timeout(Duration::from_millis(500), waiter)
        .await
        .expect("write wait lost an update before registering its await")
        .unwrap();
}

struct VectorSource {
    records: Arc<[RestartRecordVector]>,
    pause_at: Option<usize>,
    offset: usize,
    probe: WorkerProbe,
}

#[async_trait]
impl StreamSource for VectorSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 1,
            max_batch_bytes: 1024,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.offset = match cursor {
            Some(cursor) => usize::try_from(
                cursor
                    .payload()
                    .get("offset")
                    .and_then(serde_json::Value::as_u64)
                    .expect("cross-surface cursor carries an integer offset"),
            )
            .unwrap(),
            None => 0,
        };
        self.probe.opened_offsets.lock().unwrap().push(self.offset);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.pause_at == Some(self.offset) {
            tokio::task::yield_now().await;
            return Ok(Some(SourceEvent::Idle));
        }
        let Some(record) = self.records.get(self.offset).copied() else {
            return Ok(None);
        };
        assert_eq!(record.offset, self.offset);
        self.offset += 1;
        let record_batch = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![record.value])) as _,
        )])
        .unwrap();
        Ok(Some(SourceEvent::Data {
            batch: Batch::table(
                vec![record_batch],
                BatchMetadata::new("a6-vector", record.offset as u64, BTreeMap::new())?,
            )?,
            cursor: Cursor::unbound(
                u64::try_from(self.offset).unwrap().to_be_bytes().to_vec(),
                BTreeMap::from([("offset".into(), self.offset.into())]),
            )?,
        }))
    }

    async fn close(&mut self) -> Result<()> {
        self.probe.source_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct VectorSink {
    root: PathBuf,
    pending: Vec<i64>,
    probe: WorkerProbe,
}

impl VectorSink {
    fn values(pre_commit: &JsonMap) -> Vec<i64> {
        pre_commit["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap())
            .collect()
    }

    async fn commit_values(&self, epoch: calc_flow::Epoch, values: &[i64]) -> Result<()> {
        tokio::fs::create_dir_all(&self.root)
            .await
            .map_err(|source| calc_flow::CalcFlowError::Io {
                path: self.root.to_string_lossy().into_owned(),
                source,
            })?;
        let target = self
            .root
            .join(format!("visible-{:020}.json", epoch.as_u64()));
        if target.exists() {
            let observed: Vec<i64> =
                serde_json::from_slice(&tokio::fs::read(&target).await.map_err(|source| {
                    calc_flow::CalcFlowError::Io {
                        path: target.to_string_lossy().into_owned(),
                        source,
                    }
                })?)
                .unwrap();
            assert_eq!(observed, values);
            return Ok(());
        }
        let temporary = self.root.join(format!(".tmp-{:020}.json", epoch.as_u64()));
        tokio::fs::write(&temporary, serde_json::to_vec(values).unwrap())
            .await
            .map_err(|source| calc_flow::CalcFlowError::Io {
                path: temporary.to_string_lossy().into_owned(),
                source,
            })?;
        tokio::fs::rename(&temporary, &target)
            .await
            .map_err(|source| calc_flow::CalcFlowError::Io {
                path: target.to_string_lossy().into_owned(),
                source,
            })
    }
}

#[async_trait]
impl TransactionalStreamSink for VectorSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: calc_flow::Epoch) -> Result<()> {
        self.pending.clear();
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        for record in batch.table_payload()?.batches() {
            let values = record
                .column_by_name("doubled")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            self.pending.extend(values.values());
            self.probe
                .writes
                .send_modify(|writes| *writes += values.len());
        }
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: calc_flow::Epoch) -> Result<JsonMap> {
        Ok(BTreeMap::from([(
            "values".into(),
            serde_json::json!(self.pending),
        )]))
    }

    async fn commit(&mut self, epoch: calc_flow::Epoch, pre_commit: &JsonMap) -> Result<()> {
        self.commit_values(epoch, &Self::values(pre_commit)).await
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        self.pending.clear();
        Ok(())
    }

    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        self.commit_values(recovery.epoch(), &Self::values(recovery.pre_commit()))
            .await
    }

    async fn close(&mut self) -> Result<()> {
        self.probe.sink_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn vector_plan(vector: &RestartVector) -> calc_flow::StreamExecutionPlan {
    let requirements = StreamRequirements {
        delivery: BTreeMap::from([(
            vector.plan.output_id.clone(),
            DeliveryGuarantee::ExactlyOnce,
        )]),
    };
    PipelineBuilder::new(&vector.plan.name)
        .unwrap()
        .add_node(
            &vector.plan.operator_id,
            Box::new(
                ExpressionOperator::new(
                    &vector.plan.operator_id,
                    &vector.plan.expression,
                    Vec::new(),
                    None,
                    Vec::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(&UdfRegistry::new().snapshot(), &requirements)
        .unwrap()
}

fn vector_runner(
    vector: &RestartVector,
    managed_root: &Path,
    sink_root: &Path,
    pause_at: Option<usize>,
    probe: WorkerProbe,
) -> (String, StreamingRunner) {
    let plan = vector_plan(vector);
    let fingerprint = plan.fingerprint().to_owned();
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::from([(
            vector.plan.source_id.clone(),
            SourceBinding::new(VectorSource {
                records: vector.records.clone().into(),
                pause_at,
                offset: 0,
                probe: probe.clone(),
            })
            .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        BTreeMap::from([(
            vector.plan.output_id.clone(),
            vec![
                SinkBinding::transactional(
                    &vector.plan.sink_id,
                    VectorSink {
                        root: sink_root.to_owned(),
                        pending: Vec::new(),
                        probe,
                    },
                )
                .unwrap(),
            ],
        )]),
        ManagedCheckpointRuntime::new(managed_root).unwrap(),
    )
    .unwrap();
    (fingerprint, runner)
}

async fn visible_values(root: &Path) -> Vec<i64> {
    let mut entries = tokio::fs::read_dir(root).await.unwrap();
    let mut paths = Vec::new();
    while let Some(entry) = entries.next_entry().await.unwrap() {
        if entry.file_name().to_str().is_some_and(|name| {
            name.starts_with("visible-")
                && Path::new(name)
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
        }) {
            paths.push(entry.path());
        }
    }
    paths.sort();
    let mut values = Vec::new();
    for path in paths {
        values.extend(
            serde_json::from_slice::<Vec<i64>>(&tokio::fs::read(path).await.unwrap()).unwrap(),
        );
    }
    values
}

fn temporary_artifacts(root: &Path) -> usize {
    let mut pending = vec![root.to_owned()];
    let mut count = 0;
    while let Some(directory) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.map(std::result::Result::unwrap) {
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else if entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.contains(".tmp") || name.starts_with("tmp"))
            {
                count += 1;
            }
        }
    }
    count
}

fn worker_path(name: &str) -> PathBuf {
    std::env::var_os(name).map_or_else(
        || panic!("missing worker environment variable {name}"),
        PathBuf::from,
    )
}

async fn wait_for_rust_worker_operation<T>(
    job: &StreamingJob,
    mode: &str,
    operation: &str,
    future: impl Future<Output = T>,
) -> T {
    let Ok(value) = tokio::time::timeout(WORKER_OPERATION_TIMEOUT, future).await else {
        let status = job.status();
        let cleanup = tokio::time::timeout(WORKER_PROCESS_CLEANUP_TIMEOUT, job.cancel()).await;
        panic!(
            "Rust cross-surface {mode} worker {operation} exceeded \
             {WORKER_OPERATION_TIMEOUT:?}; status before cleanup: {status:?}; \
             cancellation cleanup: {cleanup:?}"
        );
    };
    value
}

#[tokio::test]
#[ignore = "invoked by the cross-surface parent test"]
async fn cross_surface_rust_worker() {
    let mode = std::env::var(WORKER_MODE).expect("missing cross-surface worker mode");
    assert!(matches!(mode.as_str(), "stage" | "resume"));
    let managed_root = worker_path(WORKER_MANAGED_ROOT);
    let sink_root = worker_path(WORKER_SINK_ROOT);
    let report_path = worker_path(WORKER_REPORT);
    let vector = restart_vector().await;
    let probe = WorkerProbe::default();
    let (fingerprint, runner) = vector_runner(
        &vector,
        &managed_root,
        &sink_root,
        (mode == "stage").then_some(vector.checkpoint_after),
        probe.clone(),
    );
    let job = runner.start().await.unwrap();
    let outcome = if mode == "stage" {
        wait_for_rust_worker_operation(
            &job,
            &mode,
            "wait for staged sink writes",
            probe.wait_for_writes(vector.checkpoint_after),
        )
        .await;
        let epoch = wait_for_rust_worker_operation(
            &job,
            &mode,
            "manual checkpoint",
            job.trigger_checkpoint(),
        )
        .await
        .unwrap();
        assert_eq!(epoch.as_u64(), vector.expected.checkpoint_epoch);
        wait_for_rust_worker_operation(&job, &mode, "cancellation cleanup", job.cancel()).await
    } else {
        wait_for_rust_worker_operation(&job, &mode, "terminal completion", job.wait()).await
    };
    let status = job.status();
    let opened_offset = probe.opened_offsets.lock().unwrap()[0];
    let final_visible_values = visible_values(&sink_root).await;
    let report = WorkerReport {
        surface: "rust".into(),
        mode,
        plan_fingerprint: fingerprint,
        opened_offset,
        outcome_state: match outcome.state {
            JobState::Completed => "completed",
            JobState::Cancelled => "cancelled",
            state => panic!("cross-surface Rust worker failed with {state:?}: {outcome:?}"),
        }
        .into(),
        completed_epoch: outcome.completed_epoch.map(calc_flow::Epoch::as_u64),
        task_count: status.task_count,
        charged_edges: status
            .edges
            .values()
            .filter(|edge| {
                edge.current_envelopes != 0 || edge.current_rows != 0 || edge.current_bytes != 0
            })
            .count(),
        source_closes: probe.source_closes.load(Ordering::SeqCst),
        sink_closes: probe.sink_closes.load(Ordering::SeqCst),
        visible_values: final_visible_values,
        temporary_artifacts: temporary_artifacts(
            managed_root
                .parent()
                .expect("managed root has a scenario parent"),
        ),
    };
    tokio::fs::write(report_path, serde_json::to_vec(&report).unwrap())
        .await
        .unwrap();
}

#[derive(Debug)]
struct WorkerTimeout {
    pid: u32,
    timeout: Duration,
    kill_error: Option<std::io::Error>,
    reap_status: Option<ExitStatus>,
    reap_error: Option<std::io::Error>,
    cleanup_timed_out: bool,
}

fn wait_for_worker_blocking(
    mut child: GroupChild,
    timeout: Duration,
) -> std::result::Result<ExitStatus, WorkerTimeout> {
    let pid = child.id();
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {}
            Err(error) => panic!("failed to wait for worker process {pid}: {error}"),
        }
        if started.elapsed() >= timeout {
            let kill_error = match child.kill() {
                Ok(()) => None,
                Err(error) => match child.try_wait() {
                    Ok(Some(status)) => return Ok(status),
                    Ok(None) | Err(_) => Some(error),
                },
            };
            let cleanup_started = Instant::now();
            let (reap_status, reap_error, cleanup_timed_out) = loop {
                match child.try_wait() {
                    Ok(Some(status)) => break (Some(status), None, false),
                    Ok(None) if cleanup_started.elapsed() < WORKER_PROCESS_CLEANUP_TIMEOUT => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Ok(None) => break (None, None, true),
                    Err(error) => break (None, Some(error), false),
                }
            };
            return Err(WorkerTimeout {
                pid,
                timeout,
                kill_error,
                reap_status,
                reap_error,
                cleanup_timed_out,
            });
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
#[ignore = "spawned by worker_timeout_terminates_and_reaps_process"]
fn cross_surface_timeout_probe_worker() {
    thread::sleep(Duration::from_secs(60));
}

fn current_test_executable() -> PathBuf {
    // This path only relaunches the current test harness; it is not a security boundary.
    std::env::current_exe().unwrap() // nosemgrep: rust.lang.security.current-exe.current-exe
}

#[test]
fn worker_timeout_terminates_and_reaps_process() {
    let child = Command::new(current_test_executable())
        .args([
            "cross_surface_timeout_probe_worker",
            "--ignored",
            "--exact",
            "--test-threads=1",
        ])
        .stdin(Stdio::null())
        .group_spawn()
        .unwrap();
    let pid = child.id();
    let timeout = wait_for_worker_blocking(child, Duration::from_millis(100)).unwrap_err();
    assert_eq!(timeout.pid, pid);
    assert_eq!(timeout.timeout, Duration::from_millis(100));
    assert!(timeout.kill_error.is_none(), "{timeout:?}");
    assert!(timeout.reap_status.is_some(), "{timeout:?}");
    assert!(timeout.reap_error.is_none(), "{timeout:?}");
    assert!(!timeout.cleanup_timed_out, "{timeout:?}");
}

fn wait_for_worker_pid(path: &Path, timeout: Duration) -> u32 {
    let started = Instant::now();
    loop {
        if let Ok(raw_pid) = std::fs::read_to_string(path)
            && let Ok(pid) = raw_pid.trim().parse()
        {
            return pid;
        }
        assert!(
            started.elapsed() < timeout,
            "worker did not write its PID to {} within {timeout:?}",
            path.display()
        );
        thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(unix)]
fn worker_process_is_alive(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(windows)]
fn worker_process_is_alive(pid: u32) -> bool {
    let output = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let expected = pid.to_string();
    String::from_utf8_lossy(&output.stdout).lines().any(|line| {
        line.split(',')
            .nth(1)
            .is_some_and(|field| field.trim_matches('"') == expected)
    })
}

fn wait_for_worker_process_to_stop(pid: u32, timeout: Duration) -> bool {
    let started = Instant::now();
    while worker_process_is_alive(pid) {
        if started.elapsed() >= timeout {
            return false;
        }
        thread::sleep(Duration::from_millis(10));
    }
    true
}

#[cfg(unix)]
fn terminate_surviving_test_process(pid: u32) {
    let _ = Command::new("kill")
        .args(["-KILL", &pid.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(windows)]
fn terminate_surviving_test_process(pid: u32) {
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn uv_python_command() -> Command {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut command = Command::new("uv");
    command
        .args(["run", "--no-sync", "python"])
        .env("UV_CACHE_DIR", workspace_root.join("target/uv-cache"));
    command
}

fn uv_is_available() -> bool {
    Command::new("uv")
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn worker_log_paths(report: &Path) -> (PathBuf, PathBuf) {
    (
        report.with_extension("stdout.log"),
        report.with_extension("stderr.log"),
    )
}

fn worker_log_excerpt(path: &Path) -> String {
    std::fs::read(path).map_or_else(
        |error| format!("unavailable at {}: {error}", path.display()),
        |bytes| {
            let start = bytes.len().saturating_sub(WORKER_LOG_EXCERPT_BYTES);
            let excerpt = String::from_utf8_lossy(&bytes[start..]);
            if excerpt.is_empty() {
                format!("empty ({})", path.display())
            } else {
                format!("{} ({})", excerpt.trim_end(), path.display())
            }
        },
    )
}

fn worker_process_failure(
    surface: &str,
    mode: &str,
    status: std::result::Result<ExitStatus, WorkerTimeout>,
    stdout_path: &Path,
    stderr_path: &Path,
) {
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => panic!(
            "{surface} cross-surface {mode} worker exited {status}; stdout tail: {}; \
             stderr tail: {}",
            worker_log_excerpt(stdout_path),
            worker_log_excerpt(stderr_path)
        ),
        Err(timeout) => panic!(
            "{surface} cross-surface {mode} worker timed out: {timeout:?}; stdout tail: {}; \
             stderr tail: {}",
            worker_log_excerpt(stdout_path),
            worker_log_excerpt(stderr_path)
        ),
    }
}

#[test]
fn worker_timeout_terminates_uv_python_process_tree() {
    if !uv_is_available() {
        eprintln!("skipping process-tree cleanup test because uv is unavailable");
        return;
    }

    let temporary = tempfile::tempdir().unwrap();
    let pid_path = temporary.path().join("python-worker.pid");
    let script = "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8'); time.sleep(60)";
    let child = uv_python_command()
        .args(["-c", script])
        .arg(&pid_path)
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .group_spawn()
        .unwrap();
    let launcher_pid = child.id();
    let python_pid = wait_for_worker_pid(&pid_path, Duration::from_secs(10));
    assert_ne!(python_pid, launcher_pid);

    let timeout = wait_for_worker_blocking(child, Duration::from_millis(100)).unwrap_err();
    let python_stopped = wait_for_worker_process_to_stop(python_pid, Duration::from_secs(1));
    if !python_stopped {
        terminate_surviving_test_process(python_pid);
    }

    assert_eq!(timeout.pid, launcher_pid);
    assert!(
        python_stopped,
        "Python worker {python_pid} survived launcher timeout {timeout:?}"
    );
}

async fn run_rust_worker(mode: &str, managed_root: &Path, sink_root: &Path, report: &Path) {
    let executable = current_test_executable();
    let mode = mode.to_owned();
    let managed_root = managed_root.to_owned();
    let sink_root = sink_root.to_owned();
    let report = report.to_owned();
    tokio::task::spawn_blocking(move || {
        let (stdout_path, stderr_path) = worker_log_paths(&report);
        let child = Command::new(executable)
            .args([
                "cross_surface_rust_worker",
                "--ignored",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(WORKER_MODE, &mode)
            .env(WORKER_MANAGED_ROOT, managed_root)
            .env(WORKER_SINK_ROOT, sink_root)
            .env(WORKER_REPORT, &report)
            .stdin(Stdio::null())
            .stdout(Stdio::from(File::create(&stdout_path).unwrap()))
            .stderr(Stdio::from(File::create(&stderr_path).unwrap()))
            .group_spawn()
            .unwrap();
        worker_process_failure(
            "Rust",
            &mode,
            wait_for_worker_blocking(child, WORKER_PROCESS_TIMEOUT),
            &stdout_path,
            &stderr_path,
        );
    })
    .await
    .unwrap();
}

async fn run_python_worker(mode: &str, managed_root: &Path, sink_root: &Path, report: &Path) {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let worker = repository.join("python/tests/a6_cross_surface_worker.py");
    let mode = mode.to_owned();
    let managed_root = managed_root.to_owned();
    let sink_root = sink_root.to_owned();
    let report = report.to_owned();
    tokio::task::spawn_blocking(move || {
        let (stdout_path, stderr_path) = worker_log_paths(&report);
        let child = uv_python_command()
            .arg(worker)
            .arg(&mode)
            .arg(managed_root)
            .arg(sink_root)
            .arg(&report)
            .current_dir(repository)
            .stdin(Stdio::null())
            .stdout(Stdio::from(File::create(&stdout_path).unwrap()))
            .stderr(Stdio::from(File::create(&stderr_path).unwrap()))
            .group_spawn()
            .unwrap();
        worker_process_failure(
            "Python",
            &mode,
            wait_for_worker_blocking(child, WORKER_PROCESS_TIMEOUT),
            &stdout_path,
            &stderr_path,
        );
    })
    .await
    .unwrap();
}

async fn read_report(path: &Path) -> WorkerReport {
    serde_json::from_slice(&tokio::fs::read(path).await.unwrap()).unwrap()
}

async fn manifest_epochs(managed_root: &Path) -> Vec<u64> {
    let mut entries = tokio::fs::read_dir(managed_root.join("manifests"))
        .await
        .unwrap();
    let mut paths = Vec::new();
    while let Some(entry) = entries.next_entry().await.unwrap() {
        if entry.file_name().to_str().is_some_and(|name| {
            name.starts_with("manifest-")
                && Path::new(name)
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
        }) {
            paths.push(entry.path());
        }
    }
    paths.sort();
    let mut epochs = Vec::new();
    for path in paths {
        let manifest =
            CheckpointManifest::from_bytes(&tokio::fs::read(path).await.unwrap()).unwrap();
        epochs.push(manifest.epoch().as_u64());
    }
    epochs
}

fn assert_worker_report(report: &WorkerReport, surface: &str, mode: &str, vector: &RestartVector) {
    assert_eq!(report.surface, surface);
    assert_eq!(report.mode, mode);
    assert_eq!(
        report.opened_offset,
        if mode == "stage" {
            vector.expected.opened_offsets[0]
        } else {
            vector.expected.opened_offsets[1]
        }
    );
    assert_eq!(
        report.outcome_state,
        if mode == "stage" {
            "cancelled"
        } else {
            "completed"
        }
    );
    assert_eq!(
        report.completed_epoch,
        Some(if mode == "stage" {
            vector.expected.checkpoint_epoch
        } else {
            vector.expected.terminal_epoch
        })
    );
    assert_eq!(report.task_count, vector.expected.terminal_tasks);
    assert_eq!(report.charged_edges, vector.expected.terminal_charged_edges);
    assert_eq!(report.source_closes, 1);
    assert_eq!(report.sink_closes, 1);
    assert_eq!(
        report.temporary_artifacts,
        vector.expected.temporary_artifacts
    );
}

async fn assert_scenario(
    scenario: &Path,
    stage: &WorkerReport,
    resume: &WorkerReport,
    vector: &RestartVector,
) {
    assert_eq!(stage.plan_fingerprint, resume.plan_fingerprint);
    assert_eq!(
        resume.visible_values, vector.expected.values,
        "cross-surface recovery changed transactional output"
    );
    let unique = resume
        .visible_values
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let expected = vector
        .expected
        .values
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        resume.visible_values.len() - unique.len(),
        vector.expected.duplicates
    );
    assert_eq!(
        expected.difference(&unique).count(),
        vector.expected.missing
    );
    assert_eq!(
        manifest_epochs(&scenario.join("managed")).await,
        vec![
            vector.expected.checkpoint_epoch,
            vector.expected.terminal_epoch
        ]
    );
    assert_eq!(
        temporary_artifacts(scenario),
        vector.expected.temporary_artifacts
    );
}

#[tokio::test]
#[ignore = "requires the developed Python extension; set CALC_FLOW_A6_CROSS_SURFACE_E2E=1"]
async fn cross_surface_restart_is_compatible_in_both_directions() {
    assert_eq!(
        std::env::var(RUN_GATE).as_deref(),
        Ok("1"),
        "set {RUN_GATE}=1 to run the cross-surface process E2E"
    );
    let directory = tempfile::tempdir().unwrap();
    let vector = restart_vector().await;

    let rust_to_python = directory.path().join("rust-to-python");
    let rust_report_path = directory.path().join("rust-stage.json");
    let python_report_path = directory.path().join("python-resume.json");
    run_rust_worker(
        "stage",
        &rust_to_python.join("managed"),
        &rust_to_python.join("sink"),
        &rust_report_path,
    )
    .await;
    run_python_worker(
        "resume",
        &rust_to_python.join("managed"),
        &rust_to_python.join("sink"),
        &python_report_path,
    )
    .await;
    let rust_stage = read_report(&rust_report_path).await;
    let python_resume = read_report(&python_report_path).await;
    assert_worker_report(&rust_stage, "rust", "stage", &vector);
    assert_worker_report(&python_resume, "python", "resume", &vector);
    assert_scenario(&rust_to_python, &rust_stage, &python_resume, &vector).await;

    let python_to_rust = directory.path().join("python-to-rust");
    let python_report_path = directory.path().join("python-stage.json");
    let rust_report_path = directory.path().join("rust-resume.json");
    run_python_worker(
        "stage",
        &python_to_rust.join("managed"),
        &python_to_rust.join("sink"),
        &python_report_path,
    )
    .await;
    run_rust_worker(
        "resume",
        &python_to_rust.join("managed"),
        &python_to_rust.join("sink"),
        &rust_report_path,
    )
    .await;
    let python_stage = read_report(&python_report_path).await;
    let rust_resume = read_report(&rust_report_path).await;
    assert_worker_report(&python_stage, "python", "stage", &vector);
    assert_worker_report(&rust_resume, "rust", "resume", &vector);
    assert_scenario(&python_to_rust, &python_stage, &rust_resume, &vector).await;
}
