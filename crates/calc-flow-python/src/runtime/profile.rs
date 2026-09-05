//! Opt-in, bounded wall-clock diagnostics. Never enabled by normal execution.

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use parking_lot::Mutex;

const CAPACITY: usize = 1024;

#[derive(Default)]
pub(super) struct CallbackProfile {
    buffer: Mutex<ProfileBuffer>,
}

#[derive(Default)]
struct ProfileBuffer {
    records: VecDeque<serde_json::Value>,
    dropped: u64,
}

pub(super) struct CallbackTrace {
    owner: Arc<CallbackProfile>,
    callback: String,
    started: Instant,
    attached: AtomicU64,
    queued: AtomicU64,
    dispatched: AtomicU64,
    completed: AtomicU64,
}

impl CallbackProfile {
    pub(super) fn take_json(&self) -> serde_json::Result<String> {
        // Serialize after releasing the lock; diagnostics must not hold it
        // while Python callbacks publish their completion records.
        let buffer = std::mem::take(&mut *self.buffer.lock());
        serde_json::to_string(&serde_json::json!({
            "records": buffer.records,
            "dropped": buffer.dropped,
        }))
    }
}

impl CallbackTrace {
    fn elapsed(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_nanos())
            .unwrap_or(u64::MAX)
            .max(1)
    }

    pub(super) fn attached(&self) {
        self.attached.store(self.elapsed(), Ordering::Release);
    }

    pub(super) fn queued(&self) {
        self.queued.store(self.elapsed(), Ordering::Release);
    }

    pub(super) fn dispatched(&self) {
        self.dispatched.store(self.elapsed(), Ordering::Release);
    }

    pub(super) fn completed(&self) {
        self.completed.store(self.elapsed(), Ordering::Release);
    }

    fn record(&self, outcome: &'static str) -> serde_json::Value {
        fn value(stage: &AtomicU64) -> Option<u64> {
            let nanos = stage.load(Ordering::Acquire);
            (nanos != 0).then_some(nanos)
        }
        serde_json::json!({
            "callback": self.callback,
            "outcome": outcome,
            "attached_ns": value(&self.attached),
            "queued_ns": value(&self.queued),
            "dispatched_ns": value(&self.dispatched),
            "completed_ns": value(&self.completed),
            "elapsed_ns": self.elapsed(),
        })
    }
}

pub(super) struct CallbackProbe {
    pub(super) trace: Option<Arc<CallbackTrace>>,
    outcome: &'static str,
}

impl CallbackProbe {
    pub(super) fn new(owner: Option<&Arc<CallbackProfile>>, name: &str) -> Self {
        Self {
            trace: owner.map(|owner| {
                Arc::new(CallbackTrace {
                    owner: Arc::clone(owner),
                    callback: name.into(),
                    started: Instant::now(),
                    attached: AtomicU64::new(0),
                    queued: AtomicU64::new(0),
                    dispatched: AtomicU64::new(0),
                    completed: AtomicU64::new(0),
                })
            }),
            outcome: "cancelled",
        }
    }

    pub(super) fn finish(&mut self, outcome: &'static str) {
        self.outcome = outcome;
    }
}

impl Drop for CallbackProbe {
    fn drop(&mut self) {
        if let Some(trace) = &self.trace {
            let record = trace.record(self.outcome);
            let mut buffer = trace.owner.buffer.lock();
            if buffer.records.len() == CAPACITY {
                buffer.records.pop_front();
                buffer.dropped = buffer.dropped.saturating_add(1);
            }
            buffer.records.push_back(record);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiling_is_opt_in_bounded_and_drainable() {
        assert!(CallbackProbe::new(None, "disabled").trace.is_none());
        let profile = Arc::new(CallbackProfile::default());
        for _ in 0..CAPACITY + 3 {
            let mut probe = CallbackProbe::new(Some(&profile), "ready");
            probe.trace.as_ref().unwrap().attached();
            probe.finish("ready");
        }
        let result: serde_json::Value =
            serde_json::from_str(&profile.take_json().unwrap()).unwrap();
        assert_eq!(result["records"].as_array().unwrap().len(), CAPACITY);
        assert_eq!(result["dropped"], 3);
        assert_eq!(result["records"][0]["outcome"], "ready");
        assert!(result["records"][0]["attached_ns"].as_u64().unwrap() > 0);
        assert!(result["records"][0]["dispatched_ns"].is_null());
        let empty: serde_json::Value = serde_json::from_str(&profile.take_json().unwrap()).unwrap();
        assert_eq!(empty, serde_json::json!({"records": [], "dropped": 0}));
    }

    #[test]
    fn cancelled_probe_keeps_missing_boundaries_explicit() {
        let profile = Arc::new(CallbackProfile::default());
        let trace = {
            let probe = CallbackProbe::new(Some(&profile), "pending");
            Arc::clone(probe.trace.as_ref().unwrap())
        };
        trace.completed();
        let result: serde_json::Value =
            serde_json::from_str(&profile.take_json().unwrap()).unwrap();
        assert_eq!(result["records"][0]["outcome"], "cancelled");
        assert!(result["records"][0]["completed_ns"].is_null());
    }
}
