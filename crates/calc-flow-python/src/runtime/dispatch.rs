//! One reusable loop wakeup for a job's concurrently ready Python callbacks.
//!
//! The managed runtime bounds outstanding calls by its source/operator/sink
//! tasks. This queue does not prefetch source events or bypass edge backpressure.

use std::collections::VecDeque;

use parking_lot::Mutex;
use pyo3::{PyTraverseError, PyVisit, exceptions::PyRuntimeError, prelude::*};

use super::PythonAwaitScheduler;

const DISPATCH_BUDGET: usize = 64;

#[derive(Default)]
struct DispatchQueue {
    pending: VecDeque<Py<PythonAwaitScheduler>>,
    scheduled: bool,
}

#[pyclass]
pub(super) struct PythonAwaitDispatcher {
    event_loop: Option<Py<PyAny>>,
    queue: Mutex<DispatchQueue>,
}

impl PythonAwaitDispatcher {
    pub(super) fn new(event_loop: Py<PyAny>) -> Self {
        Self {
            event_loop: Some(event_loop),
            queue: Mutex::new(DispatchQueue::default()),
        }
    }

    pub(super) fn submit(
        dispatcher: &Bound<'_, Self>,
        scheduler: Py<PythonAwaitScheduler>,
    ) -> PyResult<()> {
        let owner = dispatcher.borrow();
        let py = dispatcher.py();
        let event_loop = owner.event_loop(py)?;
        {
            let mut queue = owner.queue.lock();
            queue.pending.push_back(scheduler);
            if queue.scheduled {
                return Ok(());
            }
            queue.scheduled = true;
        }
        if let Err(error) =
            event_loop.call_method1(pyo3::intern!(py, "call_soon_threadsafe"), (dispatcher,))
        {
            owner.fail_pending(py, &error);
            return Err(error);
        }
        Ok(())
    }

    fn event_loop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.event_loop
            .as_ref()
            .map(|event_loop| event_loop.bind(py).clone())
            .ok_or_else(|| PyRuntimeError::new_err("Python await dispatcher was cleared"))
    }

    fn fail_pending(&self, py: Python<'_>, error: &PyErr) {
        let pending = {
            let mut queue = self.queue.lock();
            queue.scheduled = false;
            std::mem::take(&mut queue.pending)
        };
        for scheduler in pending {
            let mut scheduler = scheduler.borrow_mut(py);
            scheduler.completion.send(Err(error.clone_ref(py)));
            scheduler.__clear__();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::Ordering};

    use pyo3::types::PyDict;

    use super::*;
    use crate::runtime::{
        PythonAwaitCompletion, PythonAwaitReceiver, PythonAwaitRegistry, PythonAwaitState,
    };

    fn fixture(py: Python<'_>) -> Bound<'_, PyDict> {
        let locals = PyDict::new(py);
        py.run(
            cr#"
import asyncio
class LoopProxy:
    def __init__(self):
        self.error = RuntimeError("dispatch unavailable")
        self.fail_submit = False
        self.fail_continue = False
        self.callback = None
    def call_soon_threadsafe(self, callback):
        if self.fail_submit:
            raise self.error
        self.callback = callback
    def call_soon(self, callback):
        if self.fail_continue:
            raise self.error
        self.callback = callback
class BrokenContext:
    def run(self, callback):
        raise RuntimeError("context unavailable")
event_loop = asyncio.new_event_loop()
proxy = LoopProxy()
"#,
            Some(&locals),
            None,
        )
        .unwrap();
        locals
    }

    fn ready_scheduler(
        py: Python<'_>,
        event_loop: &Bound<'_, PyAny>,
        registry: &Arc<PythonAwaitRegistry>,
    ) -> (Py<PythonAwaitScheduler>, PythonAwaitReceiver) {
        let future = event_loop.call_method0("create_future").unwrap();
        future.call_method1("set_result", (7,)).unwrap();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let scheduler = Py::new(
            py,
            PythonAwaitScheduler {
                awaitable: Some(future.unbind()),
                context: None,
                state: Some(Arc::new(PythonAwaitState::new(event_loop.clone().unbind()))),
                completion: Arc::new(PythonAwaitCompletion {
                    sender: Mutex::new(Some(sender)),
                    lease: Mutex::new(Some(registry.retain())),
                }),
            },
        )
        .unwrap();
        (scheduler, receiver)
    }

    #[test]
    fn failed_initial_wakeup_releases_pending_calls_and_can_be_retried() {
        Python::initialize();
        Python::attach(|py| {
            let locals = fixture(py);
            let event_loop = locals.get_item("event_loop").unwrap().unwrap();
            let proxy = locals.get_item("proxy").unwrap().unwrap();
            proxy.setattr("fail_submit", true).unwrap();
            let dispatcher =
                Bound::new(py, PythonAwaitDispatcher::new(proxy.clone().unbind())).unwrap();
            let registry = Arc::new(PythonAwaitRegistry::new());
            let (scheduler, mut receiver) = ready_scheduler(py, &event_loop, &registry);
            let error = PythonAwaitDispatcher::submit(&dispatcher, scheduler).unwrap_err();
            assert!(error.value(py).is(proxy.getattr("error").unwrap()));
            let delivered = receiver.try_recv().unwrap().unwrap_err();
            assert!(delivered.value(py).is(error.value(py)));
            assert_eq!(registry.pending.load(Ordering::Acquire), 0);
            proxy.setattr("fail_submit", false).unwrap();
            let (scheduler, mut receiver) = ready_scheduler(py, &event_loop, &registry);
            PythonAwaitDispatcher::submit(&dispatcher, scheduler).unwrap();
            dispatcher.call0().unwrap();
            assert_eq!(
                receiver
                    .try_recv()
                    .unwrap()
                    .unwrap()
                    .extract::<u64>(py)
                    .unwrap(),
                7
            );
            assert_eq!(registry.pending.load(Ordering::Acquire), 0);
            proxy.setattr("callback", py.None()).unwrap();
            event_loop.call_method0("close").unwrap();
        });
    }

    #[test]
    fn failed_continuation_preserves_completed_results_and_fails_only_pending_calls() {
        Python::initialize();
        Python::attach(|py| {
            let locals = fixture(py);
            let event_loop = locals.get_item("event_loop").unwrap().unwrap();
            let proxy = locals.get_item("proxy").unwrap().unwrap();
            proxy.setattr("fail_continue", true).unwrap();
            let dispatcher =
                Bound::new(py, PythonAwaitDispatcher::new(proxy.clone().unbind())).unwrap();
            let registry = Arc::new(PythonAwaitRegistry::new());
            let mut receivers = Vec::new();
            for _ in 0..=DISPATCH_BUDGET {
                let (scheduler, receiver) = ready_scheduler(py, &event_loop, &registry);
                PythonAwaitDispatcher::submit(&dispatcher, scheduler).unwrap();
                receivers.push(receiver);
            }
            let error = dispatcher.call0().unwrap_err();
            assert!(error.value(py).is(proxy.getattr("error").unwrap()));
            for receiver in &mut receivers[..DISPATCH_BUDGET] {
                assert_eq!(
                    receiver
                        .try_recv()
                        .unwrap()
                        .unwrap()
                        .extract::<u64>(py)
                        .unwrap(),
                    7
                );
            }
            let failed = receivers[DISPATCH_BUDGET].try_recv().unwrap().unwrap_err();
            assert!(failed.value(py).is(error.value(py)));
            assert_eq!(registry.pending.load(Ordering::Acquire), 0);
            assert!(!dispatcher.borrow().queue.lock().scheduled);
            proxy.setattr("callback", py.None()).unwrap();
            event_loop.call_method0("close").unwrap();
        });
    }

    #[test]
    fn context_failure_completes_the_request_and_releases_its_lease() {
        Python::initialize();
        Python::attach(|py| {
            let locals = fixture(py);
            let event_loop = locals.get_item("event_loop").unwrap().unwrap();
            let proxy = locals.get_item("proxy").unwrap().unwrap();
            let dispatcher =
                Bound::new(py, PythonAwaitDispatcher::new(proxy.clone().unbind())).unwrap();
            let registry = Arc::new(PythonAwaitRegistry::new());
            let (scheduler, mut receiver) = ready_scheduler(py, &event_loop, &registry);
            scheduler.borrow_mut(py).context = Some(
                locals
                    .get_item("BrokenContext")
                    .unwrap()
                    .unwrap()
                    .call0()
                    .unwrap()
                    .unbind(),
            );
            PythonAwaitDispatcher::submit(&dispatcher, scheduler).unwrap();
            dispatcher.call0().unwrap();
            let error = receiver.try_recv().unwrap().unwrap_err();
            assert_eq!(
                error.value(py).str().unwrap().to_str().unwrap(),
                "context unavailable"
            );
            assert_eq!(registry.pending.load(Ordering::Acquire), 0);
            proxy.setattr("callback", py.None()).unwrap();
            event_loop.call_method0("close").unwrap();
        });
    }

    #[test]
    fn clearing_dispatcher_releases_pending_requests_and_rejects_later_submissions() {
        Python::initialize();
        Python::attach(|py| {
            let locals = fixture(py);
            let event_loop = locals.get_item("event_loop").unwrap().unwrap();
            let proxy = locals.get_item("proxy").unwrap().unwrap();
            let dispatcher =
                Bound::new(py, PythonAwaitDispatcher::new(proxy.clone().unbind())).unwrap();
            let registry = Arc::new(PythonAwaitRegistry::new());
            let (scheduler, mut receiver) = ready_scheduler(py, &event_loop, &registry);
            PythonAwaitDispatcher::submit(&dispatcher, scheduler).unwrap();
            dispatcher.borrow_mut().__clear__();
            assert_eq!(registry.pending.load(Ordering::Acquire), 0);
            assert!(matches!(
                receiver.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Closed)
            ));
            let (scheduler, mut receiver) = ready_scheduler(py, &event_loop, &registry);
            let error = PythonAwaitDispatcher::submit(&dispatcher, scheduler).unwrap_err();
            assert!(error.to_string().contains("dispatcher was cleared"));
            assert_eq!(registry.pending.load(Ordering::Acquire), 0);
            assert!(matches!(
                receiver.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Closed)
            ));
            proxy.setattr("callback", py.None()).unwrap();
            event_loop.call_method0("close").unwrap();
        });
    }
}

#[pymethods]
impl PythonAwaitDispatcher {
    fn __call__(dispatcher: &Bound<'_, Self>) -> PyResult<()> {
        let py = dispatcher.py();
        let owner = dispatcher.borrow();
        // No queue lock is held while Python user code or finalizers can run.
        for _ in 0..DISPATCH_BUDGET {
            let scheduler = owner.queue.lock().pending.pop_front();
            let Some(scheduler) = scheduler else {
                break;
            };
            let context = scheduler
                .borrow(py)
                .context
                .as_ref()
                .map(|context| context.clone_ref(py));
            let result = if let Some(context) = context {
                context
                    .bind(py)
                    .call_method1(pyo3::intern!(py, "run"), (&scheduler,))
            } else {
                scheduler.bind(py).call0()
            };
            if let Err(error) = result {
                let mut scheduler = scheduler.borrow_mut(py);
                scheduler.completion.send(Err(error));
                scheduler.__clear__();
            }
        }
        {
            let mut queue = owner.queue.lock();
            if queue.pending.is_empty() {
                queue.scheduled = false;
                return Ok(());
            }
        }
        // Yield after a bounded group so unrelated loop tasks remain runnable.
        if let Err(error) = owner.event_loop(py).and_then(|event_loop| {
            event_loop.call_method1(pyo3::intern!(py, "call_soon"), (dispatcher,))
        }) {
            owner.fail_pending(py, &error);
            return Err(error);
        }
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(event_loop) = &self.event_loop {
            visit.call(event_loop)?;
        }
        for scheduler in &self.queue.lock().pending {
            visit.call(scheduler)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        // Drop Python references after releasing the queue lock.
        let pending = {
            let mut queue = self.queue.lock();
            queue.scheduled = false;
            std::mem::take(&mut queue.pending)
        };
        drop(pending);
        self.event_loop.take();
    }
}
