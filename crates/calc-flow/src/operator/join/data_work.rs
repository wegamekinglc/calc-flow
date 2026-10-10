#[cfg(test)]
use std::sync::Arc;

mod control;
mod inputs;
mod pairs;
mod partition;
mod process;
mod worker;

const MIN_PROBE_ROWS: usize = 8_192;

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
