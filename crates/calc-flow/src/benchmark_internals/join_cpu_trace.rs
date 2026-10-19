use std::{collections::HashMap, sync::Arc, thread::ThreadId, time::Instant};

use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{Value, json};

use crate::Result;

const CAPACITY: usize = 16_384;

#[derive(Serialize)]
struct Event {
    operator: String,
    stage: &'static str,
    reason: Option<&'static str>,
    rows: usize,
    units: usize,
    bytes: usize,
    worker: bool,
    start_ns: u64,
    end_ns: Option<u64>,
    thread_cpu_ns: Option<u64>,
}

struct Trace {
    job_id: u64,
    fingerprint: String,
    started: Instant,
    events: Vec<Event>,
    active: usize,
    active_workers: HashMap<ThreadId, usize>,
    markers: u8,
    max_active_workers: usize,
    overflowed: bool,
}

#[derive(Default)]
enum State {
    #[default]
    Disabled,
    Active(Trace),
    Sealed,
}

/// Opt-in job-owned phase records, excluded from default release builds.
#[derive(Clone, Default)]
pub struct JoinCpuTraceOwner(Arc<Mutex<State>>);

impl JoinCpuTraceOwner {
    /// Starts a bounded capture before input feeding begins.
    ///
    /// # Errors
    /// Rejects empty fingerprints and reused captures.
    pub fn start(&self, job_id: u64, fingerprint: &str) -> Result<()> {
        let mut capture = self.0.lock();
        if fingerprint.is_empty() || !matches!(*capture, State::Disabled) {
            return Err(super::invalid("join CPU trace cannot be started"));
        }
        *capture = State::Active(Trace {
            job_id,
            fingerprint: fingerprint.into(),
            started: Instant::now(),
            events: Vec::with_capacity(CAPACITY),
            active: 0,
            active_workers: HashMap::new(),
            markers: 0,
            max_active_workers: 0,
            overflowed: false,
        });
        Ok(())
    }

    pub(crate) fn stage(&self, operator: &str, stage: &'static str, rows: usize) -> Stage {
        self.begin(operator, stage, rows, false)
    }

    pub(crate) fn worker_stage(&self, operator: &str, stage: &'static str, rows: usize) -> Stage {
        self.begin(operator, stage, rows, true)
    }

    fn begin(&self, operator: &str, stage: &'static str, rows: usize, worker: bool) -> Stage {
        let mut capture = self.0.lock();
        let State::Active(trace) = &mut *capture else {
            return Stage::disabled();
        };
        if trace.events.len() == CAPACITY {
            trace.overflowed = true;
            return Stage::disabled();
        }
        let index = trace.events.len();
        trace.events.push(Event {
            operator: operator.into(),
            stage,
            reason: None,
            rows,
            units: 0,
            bytes: 0,
            worker,
            start_ns: elapsed(trace.started),
            end_ns: None,
            thread_cpu_ns: None,
        });
        trace.active += 1;
        if worker {
            *trace
                .active_workers
                .entry(std::thread::current().id())
                .or_default() += 1;
            trace.max_active_workers = trace.max_active_workers.max(trace.active_workers.len());
        }
        Stage {
            owner: Some(self.clone()),
            index,
            thread: std::thread::current().id(),
            cpu: worker.then(thread_cpu).flatten(),
        }
    }

    pub(crate) fn note(
        &self,
        operator: &str,
        stage: &'static str,
        reason: &'static str,
        rows: usize,
        units: usize,
        bytes: usize,
    ) {
        let mut capture = self.0.lock();
        let State::Active(trace) = &mut *capture else {
            return;
        };
        if trace.events.len() == CAPACITY {
            trace.overflowed = true;
            return;
        }
        let at = elapsed(trace.started);
        trace.events.push(Event {
            operator: operator.into(),
            stage,
            reason: Some(reason),
            rows,
            units,
            bytes,
            worker: false,
            start_ns: at,
            end_ns: Some(at),
            thread_cpu_ns: None,
        });
    }

    pub(crate) fn mark(&self, label: &str) -> Result<()> {
        let (stage, expected) = match label {
            "ready_timer_start" => ("ready_timer_start", 0),
            "ready_timer_end" => ("ready_timer_end", 1),
            _ => return Err(super::invalid("unknown join CPU marker")),
        };
        let mut capture = self.0.lock();
        let State::Active(trace) = &mut *capture else {
            return Err(super::invalid("join CPU trace is not active"));
        };
        if trace.markers != expected {
            return Err(super::invalid("join CPU timer markers are out of order"));
        }
        if trace.events.len() == CAPACITY {
            trace.overflowed = true;
            return Err(super::invalid("join CPU trace capacity exceeded"));
        }
        let at = elapsed(trace.started);
        trace.events.push(Event {
            operator: "runtime".into(),
            stage,
            reason: Some("marker"),
            rows: 0,
            units: 0,
            bytes: 0,
            worker: false,
            start_ns: at,
            end_ns: Some(at),
            thread_cpu_ns: None,
        });
        trace.markers += 1;
        Ok(())
    }

    /// Seals a complete capture after all instrumented operations have settled.
    ///
    /// # Errors
    /// Rejects disabled, active, repeated, or truncated captures.
    pub fn take(&self) -> Result<Value> {
        let mut capture = self.0.lock();
        let State::Active(trace) = &*capture else {
            return Err(super::invalid("join CPU trace is not active"));
        };
        if trace.markers != 2 {
            return Err(super::invalid(
                "join CPU trace has an incomplete timer window",
            ));
        }
        if trace.active != 0 {
            return Err(super::invalid("join CPU trace has unfinished stages"));
        }
        let State::Active(trace) = std::mem::replace(&mut *capture, State::Sealed) else {
            unreachable!()
        };
        drop(capture);
        if trace.overflowed {
            return Err(super::invalid("join CPU trace capacity exceeded"));
        }
        Ok(json!({
            "schema": "calc-flow.join-cpu-trace.v1", "job_id": trace.job_id,
            "compiled_plan_fingerprint": trace.fingerprint,
            "capacity": CAPACITY, "overflowed": false,
            "max_active_workers": trace.max_active_workers,
            "time_origin": "capture-start",
            "thread_cpu_clock": cfg!(target_os = "linux").then_some("CLOCK_THREAD_CPUTIME_ID"),
            "events": trace.events,
        }))
    }
}

pub(crate) struct Stage {
    owner: Option<JoinCpuTraceOwner>,
    index: usize,
    thread: ThreadId,
    cpu: Option<u64>,
}

impl Stage {
    fn disabled() -> Self {
        Self {
            owner: None,
            index: 0,
            thread: std::thread::current().id(),
            cpu: None,
        }
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        let Some(owner) = &self.owner else {
            return;
        };
        let cpu = self.cpu.and_then(|start| {
            (self.thread == std::thread::current().id())
                .then(thread_cpu)
                .flatten()?
                .checked_sub(start)
        });
        let mut capture = owner.0.lock();
        if let State::Active(trace) = &mut *capture {
            let event = &mut trace.events[self.index];
            event.end_ns = Some(elapsed(trace.started));
            event.thread_cpu_ns = cpu;
            trace.active -= 1;
            if event.worker {
                let count = trace
                    .active_workers
                    .get_mut(&self.thread)
                    .expect("active worker thread");
                *count -= 1;
                if *count == 0 {
                    trace.active_workers.remove(&self.thread);
                }
            }
        }
    }
}

fn elapsed(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(target_os = "linux")]
fn thread_cpu() -> Option<u64> {
    let value = nix::time::clock_gettime(nix::time::ClockId::CLOCK_THREAD_CPUTIME_ID).ok()?;
    let seconds = u64::try_from(value.tv_sec()).ok()?;
    let nanos = u64::try_from(value.tv_nsec()).ok()?;
    if nanos >= 1_000_000_000 {
        return None;
    }
    seconds.checked_mul(1_000_000_000)?.checked_add(nanos)
}

#[cfg(not(target_os = "linux"))]
fn thread_cpu() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::JoinCpuTraceOwner;

    #[test]
    fn test_join_cpu_trace_is_job_owned_and_seals_only_settled_stages() {
        let trace = JoinCpuTraceOwner::default();
        drop(trace.stage("join", "disabled", 10));
        assert!(trace.take().is_err());
        trace.start(7, "fingerprint").unwrap();
        let stage = trace.stage("join", "prepare", 17);
        assert!(trace.take().is_err());
        trace.note("join", "probe", "budget-refusal", 17, 4, 128);
        drop(stage);
        drop(trace.worker_stage("join", "count", 17));
        trace.mark("ready_timer_start").unwrap();
        trace.mark("ready_timer_end").unwrap();
        let unsettled = trace.stage("join", "settling", 0);
        assert!(trace.take().is_err());
        drop(unsettled);
        let captured = trace.take().unwrap();
        assert_eq!(captured["job_id"], 7);
        assert_eq!(captured["compiled_plan_fingerprint"], "fingerprint");
        assert_eq!(captured["events"].as_array().unwrap().len(), 6);
        assert_eq!(captured["events"][0]["stage"], "prepare");
        assert_eq!(captured["events"][0]["rows"], 17);
        assert!(captured["events"][0]["thread_cpu_ns"].is_null());
        assert_eq!(captured["events"][1]["reason"], "budget-refusal");
        assert_eq!(captured["events"][1]["units"], 4);
        assert_eq!(captured["events"][1]["bytes"], 128);
        assert_eq!(captured["max_active_workers"], 1);
        assert!(trace.take().is_err());
        assert!(trace.start(8, "again").is_err());
    }

    #[test]
    fn test_join_cpu_trace_clones_share_only_their_job_and_record_overlap() {
        let trace = JoinCpuTraceOwner::default();
        let other = JoinCpuTraceOwner::default();
        trace.start(1, "one").unwrap();
        other.start(2, "two").unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let trace = trace.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    let _stage = trace.worker_stage("asof", "match", 11);
                    barrier.wait();
                });
            }
        });
        trace.mark("ready_timer_start").unwrap();
        trace.mark("ready_timer_end").unwrap();
        other.mark("ready_timer_start").unwrap();
        other.mark("ready_timer_end").unwrap();
        let captured = trace.take().unwrap();
        assert_eq!(captured["max_active_workers"], 2);
        assert_eq!(captured["events"].as_array().unwrap().len(), 4);
        assert_eq!(other.take().unwrap()["events"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_join_cpu_trace_rejects_truncated_evidence() {
        let trace = JoinCpuTraceOwner::default();
        trace.start(1, "one").unwrap();
        trace.mark("ready_timer_start").unwrap();
        trace.mark("ready_timer_end").unwrap();
        for _ in 0..=super::CAPACITY {
            trace.note("join", "count", "selected", 1, 1, 0);
        }
        assert!(trace.take().is_err());
    }
    #[test]
    fn test_join_cpu_trace_requires_one_complete_marker_pair() {
        let trace = JoinCpuTraceOwner::default();
        assert!(trace.mark("ready_timer_start").is_err());
        trace.start(1, "one").unwrap();
        assert!(trace.take().is_err());
        assert!(trace.mark("ready_timer_end").is_err());
        trace.mark("ready_timer_start").unwrap();
        assert!(trace.mark("ready_timer_start").is_err());
        drop(trace.stage("join", "between", 10));
        assert!(trace.take().is_err());
        trace.mark("ready_timer_end").unwrap();
        assert!(trace.mark("ready_timer_end").is_err());
        assert!(trace.take().is_ok());
        assert!(trace.mark("ready_timer_start").is_err());
    }

    #[test]
    fn test_join_cpu_trace_nested_worker_spans_count_one_thread() {
        let trace = JoinCpuTraceOwner::default();
        trace.start(1, "one").unwrap();
        trace.mark("ready_timer_start").unwrap();
        let outer = trace.worker_stage("join", "outer", 10);
        let inner = trace.worker_stage("join", "inner", 10);
        drop(inner);
        drop(outer);
        trace.mark("ready_timer_end").unwrap();
        assert_eq!(trace.take().unwrap()["max_active_workers"], 1);
    }
}
