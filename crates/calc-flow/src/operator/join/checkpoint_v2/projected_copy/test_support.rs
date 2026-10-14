use std::{cell::RefCell, sync::Arc};

pub(in crate::operator::join) type Hook = Arc<dyn Fn(bool) + Send + Sync>;

thread_local! {
    static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

pub(in crate::operator::join) struct Restore(Option<Hook>);

impl Drop for Restore {
    fn drop(&mut self) {
        HOOK.with(|hook| *hook.borrow_mut() = self.0.take());
    }
}

pub(in crate::operator::join) fn configure(hook: Hook) -> Restore {
    HOOK.with(|current| Restore(current.replace(Some(hook))))
}

pub(super) fn hook() -> Option<Hook> {
    HOOK.with(|hook| hook.borrow().clone())
}

pub(super) fn note(hook: Option<&Hook>, copied: bool) {
    if let Some(hook) = hook {
        hook(copied);
    }
}
