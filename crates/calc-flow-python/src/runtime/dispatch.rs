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
