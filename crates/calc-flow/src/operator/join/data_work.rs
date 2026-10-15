#[cfg(test)]
use std::sync::Arc;

mod control;
mod inputs;
mod morsel;
#[cfg(test)]
pub(super) use morsel::configure_for_test;
pub(super) use morsel::{DeferredProbe, GatheredOutput};
mod pairs;
mod partition;
mod process;
mod worker;

pub(super) enum ProbeAttempt {
    Serial(Option<super::native_lookup::NativeKeys>),
    Complete(crate::Result<super::PreparedMatches>),
}

const MIN_PROBE_ROWS: usize = 8_192;

/// Minimum estimated key-run visits for parallel work. Empty observed dispatch
/// medians were 45/88 us at one/four units on the D7 development runner. Keep
/// thin one-to-one batches serial until release crossover evidence warrants it.
pub(in crate::operator::join) const PARALLEL_PROBE_MIN_VISITS: usize = 320_000;

#[cfg(test)]
thread_local! {
    static PROBE_COST_GATE: std::cell::Cell<usize> =
        const { std::cell::Cell::new(PARALLEL_PROBE_MIN_VISITS) };
}

/// Probe-cost gate for this thread; tests may relax it to exercise the
/// dispatch machinery below the production threshold.
pub(in crate::operator::join) fn probe_min_visits() -> usize {
    #[cfg(test)]
    {
        PROBE_COST_GATE.get()
    }
    #[cfg(not(test))]
    {
        PARALLEL_PROBE_MIN_VISITS
    }
}

#[cfg(test)]
pub(in crate::operator::join) fn relax_probe_cost_gate_for_test() {
    PROBE_COST_GATE.with(|gate| gate.set(0));
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProbePhase {
    Count,
    Fill,
}

#[cfg(test)]
pub(super) type TestHook = Arc<dyn Fn(ProbePhase, usize, usize) + Send + Sync>;

#[cfg(test)]
pub(super) type UnitTestHook = Arc<dyn Fn(ProbePhase, usize, bool) + Send + Sync>;

#[cfg(test)]
impl inputs::ProbeInputs {
    fn observe_unit(&self, phase: ProbePhase, ordinal: usize, completed: bool) {
        if let Some(hook) = &self.unit_hook {
            hook(phase, ordinal, completed);
        }
    }

    fn observe(&self, phase: ProbePhase) {
        if let Some(hook) = &self.hook {
            hook(
                phase,
                self.opposite
                    .as_ref()
                    .expect("owned opposite rows")
                    .as_ptr() as usize,
                Arc::as_ptr(self.index.as_ref().expect("owned native index")) as usize,
            );
        }
    }
}

#[cfg(test)]
#[derive(Default)]
pub(super) struct ResultTestGate {
    pub(super) entered: tokio::sync::Notify,
    pub(super) release: tokio::sync::Notify,
}
