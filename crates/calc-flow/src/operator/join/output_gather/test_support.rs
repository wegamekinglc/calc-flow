use std::{cell::RefCell, sync::Arc};

pub(in crate::operator::join) type Hook = Arc<dyn Fn(usize, bool) + Send + Sync>;

#[derive(Default)]
struct Configuration {
    force: bool,
    hook: Option<Hook>,
}

thread_local! {
    static CONFIGURATION: RefCell<Configuration> = RefCell::default();
}

pub(in crate::operator::join) struct Restore(Configuration);

impl Drop for Restore {
    fn drop(&mut self) {
        CONFIGURATION.with(|config| *config.borrow_mut() = std::mem::take(&mut self.0));
    }
}

pub(in crate::operator::join) fn configure(force: bool, hook: Option<Hook>) -> Restore {
    CONFIGURATION.with(|config| Restore(config.replace(Configuration { force, hook })))
}

pub(in crate::operator::join) fn force() -> bool {
    CONFIGURATION.with(|config| config.borrow().force)
}

pub(in crate::operator::join) fn hook() -> Option<Hook> {
    CONFIGURATION.with(|config| config.borrow().hook.clone())
}

pub(in crate::operator::join) struct Observation {
    hook: Option<Hook>,
    ordinal: usize,
}

pub(in crate::operator::join) fn observe(hook: Option<Hook>, ordinal: usize) -> Observation {
    if let Some(hook) = &hook {
        hook(ordinal, false);
    }
    Observation { hook, ordinal }
}

impl Drop for Observation {
    fn drop(&mut self) {
        if let Some(hook) = &self.hook {
            hook(self.ordinal, true);
        }
    }
}
