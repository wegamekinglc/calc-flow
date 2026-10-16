//! Linux process CPU and per-thread scheduler runtime for opt-in measurements.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use serde_json::{Value, json};

use super::{Result, invalid};

struct ThreadRuntime {
    runtime_ns: u64,
    name: String,
    tid: u32,
}

/// Owns one benchmark interval. Thread snapshots bracket the process clock.
pub struct PhaseClock {
    scope: &'static str,
    started: Instant,
    process_cpu: Option<Duration>,
    threads: Option<BTreeMap<String, ThreadRuntime>>,
    resolution: Option<Duration>,
}

impl PhaseClock {
    /// Starts clocks without changing the operator or caller's inputs.
    pub fn start(scope: &'static str) -> Self {
        let resolution = process_resolution().ok();
        let threads = thread_runtimes().ok();
        let process_cpu = process_cpu().ok();
        Self {
            scope,
            started: Instant::now(),
            process_cpu,
            threads,
            resolution,
        }
    }

    /// Retains process totals even if a worker exits before the final snapshot.
    pub fn finish(self) -> Value {
        let wall = self.started.elapsed().as_secs_f64();
        let process_end = process_cpu().ok();
        let current = thread_runtimes().ok();
        self.evidence(wall, process_end, current)
    }

    fn evidence(
        self,
        wall: f64,
        process_end: Option<Duration>,
        current: Option<BTreeMap<String, ThreadRuntime>>,
    ) -> Value {
        let cpu = process_interval(self.process_cpu, process_end, self.resolution);
        let measured = cpu.is_some() && self.threads.is_some() && current.is_some();
        let before = self.threads.unwrap_or_default();
        let after = current.unwrap_or_default();
        let threads = after
            .iter()
            .map(|(identity, thread)| {
                let start = before.get(identity).map_or(0, |entry| entry.runtime_ns);
                json!({"tid":thread.tid,"identity":identity,"name":thread.name,
                "runtime_seconds":Duration::from_nanos(thread.runtime_ns.saturating_sub(start)).as_secs_f64(),
                "new_thread":!before.contains_key(identity)})
            })
            .collect::<Vec<_>>();
        let departed = before
            .keys()
            .filter(|key| !after.contains_key(*key))
            .collect::<Vec<_>>();
        json!({
            "schema":"calc-flow.phase-resources.v2", "scope":self.scope,
            "status":if measured {"measured"} else {"unmeasured"},
            "reason":if measured {None} else {Some("Linux process CPU clock/resolution or procfs unavailable or invalid")},
            "wall_seconds":wall,"process_cpu_seconds":cpu,
            "process_cpu_clock":"CLOCK_PROCESS_CPUTIME_ID",
            "process_cpu_resolution_seconds":self.resolution.map(|value|value.as_secs_f64()),
            "thread_clock":"linux-task-schedstat-runtime-ns",
            "thread_boundary":"thread snapshots bracket the process CPU interval",
            "threads":threads,"departed_threads":departed,
            "thread_coverage":"surviving-and-new-threads; exited CPU retained in process total",
        })
    }
}

fn stat_fields(raw: &str) -> Result<Vec<&str>> {
    raw.rsplit_once(')')
        .map(|(_, fields)| fields.split_whitespace().collect())
        .ok_or_else(|| invalid("invalid proc stat"))
}

fn field(fields: &[&str], index: usize) -> Result<u64> {
    fields
        .get(index)
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| invalid("missing proc stat counter"))
}

fn process_interval(
    start: Option<Duration>,
    end: Option<Duration>,
    resolution: Option<Duration>,
) -> Option<f64> {
    resolution.filter(|value| !value.is_zero())?;
    end?.checked_sub(start?).map(|value| value.as_secs_f64())
}

#[cfg(target_os = "linux")]
fn timespec_duration(value: nix::sys::time::TimeSpec) -> Result<Duration> {
    let seconds = u64::try_from(value.tv_sec()).map_err(|_| invalid("negative CPU clock"))?;
    let nanos = u32::try_from(value.tv_nsec()).map_err(|_| invalid("invalid CPU nanoseconds"))?;
    if nanos >= 1_000_000_000 {
        return Err(invalid("invalid CPU nanoseconds"));
    }
    Ok(Duration::new(seconds, nanos))
}

#[cfg(target_os = "linux")]
fn process_cpu() -> Result<Duration> {
    nix::time::clock_gettime(nix::time::ClockId::CLOCK_PROCESS_CPUTIME_ID)
        .map_err(|error| invalid(error.to_string()))
        .and_then(timespec_duration)
}

#[cfg(target_os = "linux")]
fn process_resolution() -> Result<Duration> {
    let resolution = nix::time::clock_getres(nix::time::ClockId::CLOCK_PROCESS_CPUTIME_ID)
        .map_err(|error| invalid(error.to_string()))
        .and_then(timespec_duration)?;
    if resolution.is_zero() {
        return Err(invalid("zero process CPU resolution"));
    }
    Ok(resolution)
}

#[cfg(not(target_os = "linux"))]
fn process_cpu() -> Result<Duration> {
    Err(invalid("Linux process CPU clock required"))
}

#[cfg(not(target_os = "linux"))]
fn process_resolution() -> Result<Duration> {
    Err(invalid("Linux process CPU resolution required"))
}

fn thread_runtimes() -> Result<BTreeMap<String, ThreadRuntime>> {
    let entries =
        std::fs::read_dir("/proc/self/task").map_err(|error| invalid(error.to_string()))?;
    let mut result = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|error| invalid(error.to_string()))?;
        if let Some((identity, runtime)) = read_thread_runtime(&entry)? {
            result.insert(identity, runtime);
        }
    }
    Ok(result)
}

fn read_thread_files(path: &std::path::Path) -> std::io::Result<(String, String, String)> {
    Ok((
        std::fs::read_to_string(path.join("stat"))?,
        std::fs::read_to_string(path.join("schedstat"))?,
        std::fs::read_to_string(path.join("comm"))?,
    ))
}

fn read_thread_runtime(entry: &std::fs::DirEntry) -> Result<Option<(String, ThreadRuntime)>> {
    match read_thread_files(&entry.path()) {
        Ok((stat, schedstat, name)) => {
            parse_thread_runtime(entry, &stat, &schedstat, &name).map(Some)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(invalid(error.to_string())),
    }
}

fn parse_thread_runtime(
    entry: &std::fs::DirEntry,
    stat: &str,
    schedstat: &str,
    name: &str,
) -> Result<(String, ThreadRuntime)> {
    let birth_ticks = field(&stat_fields(stat)?, 19)?;
    let tid = entry
        .file_name()
        .to_string_lossy()
        .parse::<u32>()
        .map_err(|error| invalid(error.to_string()))?;
    let runtime_ns = schedstat
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| invalid("invalid thread schedstat"))?;
    Ok((
        format!("{tid}/{birth_ticks}"),
        ThreadRuntime {
            runtime_ns,
            name: name.trim().to_owned(),
            tid,
        },
    ))
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, time::Duration};

    #[test]
    fn test_invalid_or_unavailable_process_clock_cannot_be_measured() {
        for (start, end, resolution) in [
            (None, Some(2), Some(1)),
            (Some(1), None, Some(1)),
            (Some(2), Some(1), Some(1)),
            (Some(1), Some(2), None),
            (Some(1), Some(2), Some(0)),
        ] {
            let clock = super::PhaseClock {
                scope: "handler",
                started: std::time::Instant::now(),
                process_cpu: start.map(Duration::from_nanos),
                resolution: resolution.map(Duration::from_nanos),
                threads: Some(BTreeMap::new()),
            };
            let evidence =
                clock.evidence(0.001, end.map(Duration::from_nanos), Some(BTreeMap::new()));
            assert_eq!(evidence["status"], "unmeasured");
            assert!(evidence["process_cpu_seconds"].is_null());
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_process_clock_retains_cpu_after_worker_exit() {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            go_rx.recv().unwrap();
            let before = super::process_cpu().unwrap();
            let until = std::time::Instant::now() + Duration::from_millis(2);
            while std::time::Instant::now() < until {
                std::hint::spin_loop();
            }
            super::process_cpu().unwrap() - before
        });
        ready_rx.recv().unwrap();
        let clock = super::PhaseClock::start("handler");
        go_tx.send(()).unwrap();
        let worker_cpu = worker.join().unwrap();
        let evidence = clock.finish();
        assert_eq!(evidence["status"], "measured");
        assert!(evidence["process_cpu_seconds"].as_f64().unwrap() >= worker_cpu.as_secs_f64());
        assert!(!evidence["departed_threads"].as_array().unwrap().is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_timespec_rejects_abnormal_values() {
        use nix::sys::time::TimeSpec;
        assert!(super::timespec_duration(TimeSpec::new(-1, 0)).is_err());
        assert!(super::timespec_duration(TimeSpec::new(0, -1)).is_err());
        assert!(super::timespec_duration(TimeSpec::new(0, 1_000_000_000)).is_err());
        assert_eq!(
            super::timespec_duration(TimeSpec::new(1, 2)).unwrap(),
            Duration::new(1, 2)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_process_clock_resolves_sub_tick_phase() {
        let clock = super::PhaseClock::start("handler");
        let deadline = std::time::Instant::now() + Duration::from_millis(1);
        while std::time::Instant::now() < deadline {
            std::hint::spin_loop();
        }
        let evidence = clock.finish();
        assert_eq!(evidence["status"], "measured");
        assert_eq!(evidence["process_cpu_clock"], "CLOCK_PROCESS_CPUTIME_ID");
        assert!(evidence["process_cpu_seconds"].as_f64().unwrap() > 0.0);
        assert!(evidence["process_cpu_resolution_seconds"].as_f64().unwrap() < 0.00001);
    }

    #[test]
    fn test_proc_stat_uses_the_last_parenthesis_and_excludes_children() {
        let raw = "7 (worker ) name) S 1 2 3 4 5 6 7 8 9 10 11 12 1000 2000 15 16 17 18 19";
        let fields = super::stat_fields(raw).unwrap();
        assert_eq!(super::field(&fields, 11).unwrap(), 11);
        assert_eq!(super::field(&fields, 12).unwrap(), 12);
        assert_eq!(super::field(&fields, 19).unwrap(), 19);
    }
}
