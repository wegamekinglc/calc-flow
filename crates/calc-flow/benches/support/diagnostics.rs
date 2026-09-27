use std::{
    collections::{BTreeSet, VecDeque},
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Instant,
};

use parking_lot::Mutex;
use serde_json::{Value, json};

use super::{
    CheckpointSoakProcessMode, CheckpointSoakProcessPlan, checkpoint_soak_report_path,
    write_checkpoint_soak_document,
};
use crate::runtime::streaming::{
    checkpoint::coordinator::CheckpointEvent, checkpoint_status::CheckpointStatus,
};

const MAX_EVENTS: usize = 128;
static TRACE: OnceLock<Mutex<SmokeTrace>> = OnceLock::new();

struct SmokeTrace {
    identity: Value,
    started: Instant,
    restored_epoch: Option<u64>,
    wait: Value,
    events: VecDeque<Value>,
    discarded: usize,
}

impl SmokeTrace {
    fn new(plan: &CheckpointSoakProcessPlan) -> Self {
        Self {
            identity: json!({
                "schema": "calc-flow.checkpoint-smoke-diagnostics.v1",
                "generation": plan.generation,
                "pid": std::process::id(),
                "parent_pid": plan.parent_pid,
                "commit": plan.commit,
                "executable_sha256": plan.executable_sha256,
                "config_hash": plan.config_hash,
                "checkpoint_timeout_millis": plan.checkpoint_timeout_millis,
                "parent_launch_offset_micros": plan.parent_launch_offset_micros,
                "clock": "process-local monotonic microseconds since child diagnostic start",
            }),
            started: Instant::now(),
            restored_epoch: None,
            wait: Value::Null,
            events: VecDeque::with_capacity(MAX_EVENTS),
            discarded: 0,
        }
    }

    fn wait_started(&mut self, baseline: u64, target: u64) {
        self.wait = json!({
            "baseline_epoch": baseline,
            "target_epoch": target,
            "elapsed_micros": self.started.elapsed().as_micros(),
        });
    }

    fn record(&mut self, detail: Value) {
        if self.events.len() == MAX_EVENTS {
            self.events.pop_front();
            self.discarded += 1;
        }
        let mut event = json!({
            "elapsed_micros": self.started.elapsed().as_micros(),
        });
        event["detail"] = detail;
        self.events.push_back(event);
    }

    fn document(&self, unwinding: bool) -> Value {
        let mut document = self.identity.clone();
        document["restored_epoch"] = json!(self.restored_epoch);
        document["wait"] = self.wait.clone();
        document["unwinding"] = json!(unwinding);
        document["events"] = json!(self.events);
        document["discarded_events"] = json!(self.discarded);
        document
    }
}

fn with_trace(update: impl FnOnce(&mut SmokeTrace)) {
    if let Some(trace) = TRACE.get() {
        update(&mut trace.lock());
    }
}

pub(super) struct Session(PathBuf);

impl Session {
    pub(super) fn start(plan: &CheckpointSoakProcessPlan) -> Option<Self> {
        if plan.mode != CheckpointSoakProcessMode::Smoke {
            return None;
        }
        assert!(TRACE.set(Mutex::new(SmokeTrace::new(plan))).is_ok());
        Some(Self(diagnostic_path(plan, "diagnostics")))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let document = TRACE
            .get()
            .unwrap()
            .lock()
            .document(std::thread::panicking());
        publish(&self.0, &document);
    }
}

pub(super) fn restored(epoch: Option<u64>) {
    with_trace(|trace| trace.restored_epoch = epoch);
}

pub(super) fn wait_started(baseline: u64, target: u64) {
    with_trace(|trace| trace.wait_started(baseline, target));
}

pub(super) fn observation(kind: &str, status: &crate::JobStatus) {
    with_trace(|trace| trace.record(json!({"kind": kind, "status": status})));
}

pub(crate) fn checkpoint_event(event: &CheckpointEvent) {
    with_trace(|trace| {
        trace.record(json!({"kind": "coordinator_event_received", "event": format!("{event:?}")}));
    });
}

pub(crate) fn operator_ack_received(node_id: &str, epoch: crate::Epoch) {
    with_trace(|trace| {
        trace.record(json!({"kind": "operator_ack_received", "node_id": node_id, "epoch": epoch}));
    });
}

fn operator_ack_evidence<'a>(
    expected: &BTreeSet<String>,
    received: impl Iterator<Item = &'a String>,
) -> Value {
    let received = received.cloned().collect::<BTreeSet<_>>();
    json!({
        "received_operator_snapshot_ids": received,
        "pending_operator_snapshot_ids": expected.difference(&received).collect::<Vec<_>>(),
    })
}

pub(crate) fn driver_finished<'a>(
    failed: bool,
    status: &CheckpointStatus,
    expected: &BTreeSet<String>,
    received: impl Iterator<Item = &'a String>,
) {
    if !failed {
        return;
    }
    with_trace(|trace| {
        trace.record(json!({
            "kind": "checkpoint_driver_failed",
            "epoch": status.current_epoch,
            "phase": format!("{:?}", status.phase),
            "elapsed": status.elapsed,
            "source_acks": status.source_acks,
            "expected_sources": status.expected_sources,
            "operator_acks": status.operator_acks,
            "expected_operators": status.expected_operators,
            "operators": operator_ack_evidence(expected, received),
            "failure_category": format!("{:?}", status.failure_category),
        }));
    });
}

fn diagnostic_path(plan: &CheckpointSoakProcessPlan, kind: &str) -> PathBuf {
    plan.run_root
        .join("evidence")
        .join(format!("generation-{}.{kind}.json", plan.generation))
}

fn publish(path: &Path, document: &Value) {
    if let Err(error) = write_checkpoint_soak_document(path, document) {
        eprintln!(
            "checkpoint smoke diagnostic write failed at {}: {error}",
            path.display()
        );
    }
}

pub(super) fn process_exit(plan: &CheckpointSoakProcessPlan, result: &crate::Result<i32>) {
    if plan.mode == CheckpointSoakProcessMode::Smoke {
        publish(
            &diagnostic_path(plan, "exit"),
            &json!({
                "generation": plan.generation,
                "exit_code": result.as_ref().ok(),
                "error": result.as_ref().err().map(ToString::to_string),
                "report_present": checkpoint_soak_report_path(plan).is_file(),
            }),
        );
    }
}

fn retained_run_directory(root: &Path, retained_epochs: usize) -> PathBuf {
    std::fs::create_dir_all(root).unwrap();
    tempfile::Builder::new()
        .prefix(&format!("retention-{retained_epochs}-"))
        .tempdir_in(root)
        .unwrap()
        .keep()
}

pub(super) fn run_directory(retained_epochs: usize) -> PathBuf {
    let root = std::env::var_os("CALC_FLOW_CHECKPOINT_SMOKE_ARTIFACTS").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/checkpoint-smoke-evidence"),
        PathBuf::from,
    );
    retained_run_directory(&root, retained_epochs)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::json;

    use super::*;
    use crate::runtime::streaming::soak::{
        CheckpointSoakProcessMode, checkpoint_soak_plan_path, checkpoint_soak_process_plans,
        checkpoint_soak_report_path, checkpoint_soak_test_executable,
        spawn_checkpoint_soak_process,
    };

    #[test]
    fn smoke_trace_bounds_events_without_losing_recovery_baseline() {
        let directory = tempfile::tempdir().unwrap();
        let plan = checkpoint_soak_process_plans(
            directory.path(),
            "commit",
            "binary",
            CheckpointSoakProcessMode::Smoke,
        )
        .remove(1);
        let mut trace = SmokeTrace::new(&plan);
        trace.restored_epoch = Some(12);
        trace.wait_started(12, 24);
        for index in 0..MAX_EVENTS + 3 {
            trace.record(json!({"index": index}));
        }
        let report = trace.document(true);
        assert_eq!(report["restored_epoch"], 12);
        assert_eq!(report["wait"]["baseline_epoch"], 12);
        assert_eq!(report["wait"]["target_epoch"], 24);
        assert_eq!(report["generation"], 1);
        assert_eq!(report["pid"], std::process::id());
        assert_eq!(report["unwinding"], true);
        let events = report["events"].as_array().unwrap();
        assert_eq!(events.len(), MAX_EVENTS);
        assert_eq!(events.first().unwrap()["detail"]["index"], 3);
        assert_eq!(report["discarded_events"], 3);
        assert!(events.windows(2).all(|pair| {
            pair[0]["elapsed_micros"].as_u64() <= pair[1]["elapsed_micros"].as_u64()
        }));
    }

    #[test]
    fn smoke_failure_names_only_missing_operator_snapshots() {
        let expected = BTreeSet::from(["merge".into(), "window".into(), "branch_a".into()]);
        let received = BTreeSet::from(["merge".into()]);
        let detail = operator_ack_evidence(&expected, received.iter());
        assert_eq!(
            detail["pending_operator_snapshot_ids"],
            json!(["branch_a", "window"])
        );
        assert_eq!(detail["received_operator_snapshot_ids"], json!(["merge"]));
    }

    #[tokio::test]
    async fn smoke_failed_child_preserves_plan_logs_and_exit_evidence() {
        let root = tempfile::tempdir().unwrap();
        let run_root = retained_run_directory(root.path(), 2);
        let mut plan = checkpoint_soak_process_plans(
            &run_root,
            "commit",
            "binary",
            CheckpointSoakProcessMode::Smoke,
        )
        .remove(0);
        plan.config_hash = "invalid-plan-for-diagnostic-probe".into();
        let error =
            spawn_checkpoint_soak_process(&checkpoint_soak_test_executable().unwrap(), &plan)
                .await
                .unwrap_err();
        assert!(error.to_string().contains("exited 101"), "{error}");
        assert!(checkpoint_soak_plan_path(&plan).exists());
        assert!(run_root.join("evidence/logs/generation-0.stdout").exists());
        let stderr =
            std::fs::read_to_string(run_root.join("evidence/logs/generation-0.stderr")).unwrap();
        assert!(stderr.contains("panicked"), "{stderr}");
        assert!(!checkpoint_soak_report_path(&plan).exists());
        let exit: Value = serde_json::from_slice(
            &std::fs::read(run_root.join("evidence/generation-0.exit.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(exit["exit_code"], 101);
        assert_eq!(exit["report_present"], false);
        assert!(
            run_root
                .join("evidence/generation-0.diagnostics.json")
                .exists()
        );
    }
}
