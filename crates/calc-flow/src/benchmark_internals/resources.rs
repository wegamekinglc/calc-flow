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
    process_ticks: Option<u64>,
    threads: Option<BTreeMap<String, ThreadRuntime>>,
    ticks_per_second: Option<u32>,
}

impl PhaseClock {
    /// Starts clocks without changing the operator or caller's inputs.
    pub fn start(scope: &'static str) -> Self {
        let ticks_per_second = ticks_per_second();
        let threads = thread_runtimes().ok();
        let process_ticks = process_ticks().ok();
        Self {
            scope,
            started: Instant::now(),
            process_ticks,
            threads,
            ticks_per_second,
        }
    }

    /// Retains process totals even if a worker exits before the final snapshot.
    pub fn finish(self) -> Value {
        let wall = self.started.elapsed().as_secs_f64();
        let process_end = process_ticks().ok();
        let current = thread_runtimes().ok();
        let cpu = self
            .process_ticks
            .zip(process_end)
            .zip(self.ticks_per_second)
            .and_then(|((start, end), hz)| {
                end.checked_sub(start)
                    .map(|ticks| Duration::from_secs(ticks).as_secs_f64() / f64::from(hz))
            });
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
            "schema":"calc-flow.phase-resources.v1", "scope":self.scope,
            "status":if measured {"measured"} else {"unmeasured"},
            "reason":if measured {None} else {Some("Linux procfs/getconf clocks unavailable")},
            "wall_seconds":wall,"process_cpu_seconds":cpu,
            "process_cpu_clock":"linux-proc-stat-utime-stime",
            "process_cpu_resolution_seconds":self.ticks_per_second.map(|hz|1.0 / f64::from(hz)),
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

fn process_ticks() -> Result<u64> {
    let raw =
        std::fs::read_to_string("/proc/self/stat").map_err(|error| invalid(error.to_string()))?;
    let fields = stat_fields(&raw)?;
    field(&fields, 11)?
        .checked_add(field(&fields, 12)?)
        .ok_or_else(|| invalid("process CPU overflow"))
}

fn thread_runtimes() -> Result<BTreeMap<String, ThreadRuntime>> {
    let entries =
        std::fs::read_dir("/proc/self/task").map_err(|error| invalid(error.to_string()))?;
    let mut result = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|error| invalid(error.to_string()))?;
        let path = entry.path();
        let read = || -> std::io::Result<_> {
            Ok((
                std::fs::read_to_string(path.join("stat"))?,
                std::fs::read_to_string(path.join("schedstat"))?,
                std::fs::read_to_string(path.join("comm"))?,
            ))
        };
        let (stat, schedstat, name) = match read() {
            Ok(values) => values,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(invalid(error.to_string())),
        };
        let birth_ticks = field(&stat_fields(&stat)?, 19)?;
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
        result.insert(
            format!("{tid}/{birth_ticks}"),
            ThreadRuntime {
                runtime_ns,
                name: name.trim().to_owned(),
                tid,
            },
        );
    }
    Ok(result)
}

fn ticks_per_second() -> Option<u32> {
    static TICKS: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *TICKS.get_or_init(|| {
        let result = std::process::Command::new("getconf")
            .arg("CLK_TCK")
            .output()
            .ok()?;
        result.status.success().then_some(())?;
        String::from_utf8(result.stdout)
            .ok()?
            .trim()
            .parse::<u32>()
            .ok()
            .filter(|hz| *hz > 0)
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_proc_stat_uses_the_last_parenthesis_and_excludes_children() {
        let raw = "7 (worker ) name) S 1 2 3 4 5 6 7 8 9 10 11 12 1000 2000 15 16 17 18 19";
        let fields = super::stat_fields(raw).unwrap();
        assert_eq!(super::field(&fields, 11).unwrap(), 11);
        assert_eq!(super::field(&fields, 12).unwrap(), 12);
        assert_eq!(super::field(&fields, 19).unwrap(), 19);
    }
}
