use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use parking_lot::Mutex;
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyAny, PyDict},
};

use crate::config::PythonRoot;

fn provider_error(name: &str, error: impl std::fmt::Display) -> calc_flow::CalcFlowError {
    calc_flow::CalcFlowError::ExternalProvider {
        provider: "python".into(),
        name: name.into(),
        version: "1".into(),
        message: error.to_string(),
    }
}

pub(crate) struct PythonAwaitRegistry {
    pending: AtomicU64,
    idle: tokio::sync::Notify,
}

pub(crate) struct PythonAsyncContext {
    event_loop: Arc<PythonRoot>,
    context: Arc<PythonRoot>,
}

impl PythonAsyncContext {
    pub(crate) fn capture(py: Python<'_>) -> PyResult<Self> {
        let event_loop = py
            .import(pyo3::intern!(py, "asyncio"))?
            .call_method0(pyo3::intern!(py, "get_running_loop"))?;
        let context = py
            .import(pyo3::intern!(py, "contextvars"))?
            .call_method0(pyo3::intern!(py, "copy_context"))?;
        Ok(Self {
            event_loop: Arc::new(PythonRoot::new(event_loop.unbind())),
            context: Arc::new(PythonRoot::new(context.unbind())),
        })
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(self.event_loop.object())?;
        visit.call(self.context.object())
    }
}

impl PythonAwaitRegistry {
    pub(crate) fn new() -> Self {
        Self {
            pending: AtomicU64::new(0),
            idle: tokio::sync::Notify::new(),
        }
    }

    fn retain(self: &Arc<Self>) -> PythonAwaitLease {
        self.pending.fetch_add(1, Ordering::AcqRel);
        PythonAwaitLease {
            registry: Arc::clone(self),
        }
    }

    pub(crate) async fn wait_idle(&self) {
        loop {
            let notified = self.idle.notified();
            if self.pending.load(Ordering::Acquire) == 0 {
                return;
            }
            notified.await;
        }
    }
}

struct PythonAwaitLease {
    registry: Arc<PythonAwaitRegistry>,
}

impl Drop for PythonAwaitLease {
    fn drop(&mut self) {
        let previous = self.registry.pending.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        if previous == 1 {
            self.registry.idle.notify_waiters();
        }
    }
}

struct PythonAwaitState {
    cancel_requested: AtomicBool,
    event_loop: Arc<PythonRoot>,
    task: Mutex<Option<Arc<PythonRoot>>>,
}

impl PythonAwaitState {
    fn new(event_loop: Py<PyAny>) -> Self {
        Self {
            cancel_requested: AtomicBool::new(false),
            event_loop: Arc::new(PythonRoot::new(event_loop)),
            task: Mutex::new(None),
        }
    }

    fn register_task(&self, task: Py<PyAny>) {
        let previous = self.task.lock().replace(Arc::new(PythonRoot::new(task)));
        drop(previous);
        if self.cancel_requested.load(Ordering::Acquire) {
            self.schedule_cancel();
        }
    }

    fn clear_task(&self) {
        let task = self.task.lock().take();
        drop(task);
    }

    fn request_cancel(&self) {
        self.cancel_requested.store(true, Ordering::Release);
        self.schedule_cancel();
    }

    fn schedule_cancel(&self) {
        let task = self.task.lock().clone();
        let event_loop = Arc::clone(&self.event_loop);
        let Some(task) = task else {
            return;
        };
        let _ = Python::attach(|py| {
            let cancel = task
                .object()
                .bind(py)
                .getattr(pyo3::intern!(py, "cancel"))?;
            event_loop
                .object()
                .bind(py)
                .call_method1(pyo3::intern!(py, "call_soon_threadsafe"), (cancel,))?;
            Ok::<(), PyErr>(())
        });
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let task = self.task.lock().clone();
        visit.call(self.event_loop.object())?;
        if let Some(task) = task {
            visit.call(task.object())?;
        }
        Ok(())
    }
}

struct PythonAwaitCompletion {
    sender: Mutex<Option<tokio::sync::oneshot::Sender<PyResult<Py<PyAny>>>>>,
    lease: Mutex<Option<PythonAwaitLease>>,
}

impl PythonAwaitCompletion {
    fn send(&self, result: PyResult<Py<PyAny>>) {
        let sender = self.sender.lock().take();
        let lease = self.lease.lock().take();
        drop(lease);
        if let Some(sender) = sender {
            let _ = sender.send(result);
        }
    }
}

#[pyclass]
struct PythonAwaitCompleter {
    state: Option<Arc<PythonAwaitState>>,
    completion: Arc<PythonAwaitCompletion>,
}

#[pymethods]
impl PythonAwaitCompleter {
    fn __call__(&mut self, task: &Bound<'_, PyAny>) {
        let result = task
            .call_method0(pyo3::intern!(task.py(), "result"))
            .map(Bound::unbind);
        if let Some(state) = self.state.take() {
            state.clear_task();
        }
        self.completion.send(result);
    }

    #[allow(clippy::needless_pass_by_value)]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(state) = &self.state {
            state.traverse(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        let state = self.state.take();
        if let Some(state) = state {
            state.request_cancel();
        }
    }
}

#[pyclass]
struct PythonAwaitScheduler {
    awaitable: Option<Py<PyAny>>,
    context: Option<Py<PyAny>>,
    state: Option<Arc<PythonAwaitState>>,
    completion: Arc<PythonAwaitCompletion>,
}

impl PythonAwaitScheduler {
    fn schedule(&mut self, py: Python<'_>) -> PyResult<()> {
        let awaitable = self
            .awaitable
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Python awaitable was already scheduled"))?;
        let state = self
            .state
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Python await state was already scheduled"))?;
        let task = py
            .import(pyo3::intern!(py, "asyncio"))?
            .getattr(pyo3::intern!(py, "ensure_future"))?
            .call1((awaitable,))?;
        state.register_task(task.clone().unbind());
        let completer = match Py::new(
            py,
            PythonAwaitCompleter {
                state: Some(Arc::clone(&state)),
                completion: Arc::clone(&self.completion),
            },
        ) {
            Ok(completer) => completer,
            Err(error) => {
                state.request_cancel();
                return Err(error);
            }
        };
        if let Err(error) = task.call_method1(pyo3::intern!(py, "add_done_callback"), (completer,))
        {
            state.request_cancel();
            return Err(error);
        }
        Ok(())
    }
}

#[pymethods]
impl PythonAwaitScheduler {
    fn __call__(&mut self, py: Python<'_>) {
        let result = self.schedule(py);
        let context = self.context.take();
        drop(context);
        if let Err(error) = result {
            if let Some(state) = self.state.take() {
                state.request_cancel();
            }
            self.completion.send(Err(error));
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(awaitable) = &self.awaitable {
            visit.call(awaitable)?;
        }
        if let Some(context) = &self.context {
            visit.call(context)?;
        }
        if let Some(state) = &self.state {
            state.traverse(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        let awaitable = self.awaitable.take();
        let context = self.context.take();
        let state = self.state.take();
        drop(awaitable);
        drop(context);
        if let Some(state) = state {
            state.request_cancel();
        }
    }
}

struct PythonAwaitCancelGuard {
    state: Option<Arc<PythonAwaitState>>,
}

impl PythonAwaitCancelGuard {
    fn disarm(&mut self) {
        let state = self.state.take();
        drop(state);
    }

    fn cancel(&mut self) {
        if let Some(state) = self.state.take() {
            state.request_cancel();
        }
    }
}

impl Drop for PythonAwaitCancelGuard {
    fn drop(&mut self) {
        self.cancel();
    }
}

pub(crate) async fn resolve_python_in_context(
    value: Py<PyAny>,
    callback_name: &str,
    awaits: &Arc<PythonAwaitRegistry>,
    context: &Arc<PythonAsyncContext>,
) -> calc_flow::Result<Py<PyAny>> {
    resolve_python_with_context(value, callback_name, awaits, Some(context)).await
}

async fn resolve_python_with_context(
    value: Py<PyAny>,
    callback_name: &str,
    awaits: &Arc<PythonAwaitRegistry>,
    context: Option<&Arc<PythonAsyncContext>>,
) -> calc_flow::Result<Py<PyAny>> {
    let is_awaitable = Python::attach(|py| {
        py.import(pyo3::intern!(py, "inspect"))?
            .getattr(pyo3::intern!(py, "isawaitable"))?
            .call1((value.bind(py),))?
            .is_truthy()
    })
    .map_err(|error| provider_error(callback_name, error))?;
    if !is_awaitable {
        return Ok(value);
    }
    let scheduled = Python::attach(|py| -> PyResult<_> {
        let (event_loop, python_context) = if let Some(context) = context {
            (
                context.event_loop.object().clone_ref(py),
                context
                    .context
                    .object()
                    .bind(py)
                    .call_method0(pyo3::intern!(py, "copy"))?
                    .unbind(),
            )
        } else {
            let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
            (locals.event_loop(py).unbind(), locals.context(py).unbind())
        };
        let state = Arc::new(PythonAwaitState::new(event_loop.clone_ref(py)));
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let completion = Arc::new(PythonAwaitCompletion {
            sender: Mutex::new(Some(sender)),
            lease: Mutex::new(Some(awaits.retain())),
        });
        let scheduler = Py::new(
            py,
            PythonAwaitScheduler {
                awaitable: Some(value),
                context: Some(python_context.clone_ref(py)),
                state: Some(Arc::clone(&state)),
                completion,
            },
        )?;
        let kwargs = PyDict::new(py);
        kwargs.set_item(pyo3::intern!(py, "context"), python_context)?;
        event_loop.bind(py).call_method(
            pyo3::intern!(py, "call_soon_threadsafe"),
            (scheduler,),
            Some(&kwargs),
        )?;
        Ok((receiver, PythonAwaitCancelGuard { state: Some(state) }))
    })
    .map_err(|error| provider_error(callback_name, error))?;
    let (receiver, mut cancellation) = scheduled;
    let result = if let Ok(result) = receiver.await {
        result
    } else {
        cancellation.cancel();
        Err(PyRuntimeError::new_err(
            "Python awaitable completion channel closed unexpectedly",
        ))
    };
    cancellation.disarm();
    result.map_err(|error| provider_error(callback_name, error))
}

pub(crate) fn python_json(
    value: &Bound<'_, PyAny>,
    label: &str,
) -> calc_flow::Result<serde_json::Value> {
    let py = value.py();
    let kwargs = PyDict::new(py);
    kwargs
        .set_item(pyo3::intern!(py, "allow_nan"), false)
        .map_err(|error| provider_error(label, error))?;
    kwargs
        .set_item(pyo3::intern!(py, "sort_keys"), true)
        .map_err(|error| provider_error(label, error))?;
    let encoded: String = py
        .import(pyo3::intern!(py, "json"))
        .and_then(|module| module.getattr(pyo3::intern!(py, "dumps")))
        .and_then(|dumps| dumps.call((value,), Some(&kwargs)))
        .and_then(|encoded| encoded.extract())
        .map_err(|error| provider_error(label, error))?;
    let parsed = serde_json::from_str(&encoded).map_err(|error| provider_error(label, error))?;
    calc_flow::canonical_json(&parsed).map_err(|error| provider_error(label, error))?;
    Ok(parsed)
}
