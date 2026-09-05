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

mod dispatch;
use dispatch::PythonAwaitDispatcher;

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
    dispatcher: Arc<PythonRoot>,
    is_awaitable: Arc<PythonRoot>,
}

impl PythonAsyncContext {
    pub(crate) fn capture(py: Python<'_>) -> PyResult<Self> {
        let event_loop = py
            .import(pyo3::intern!(py, "asyncio"))?
            .call_method0(pyo3::intern!(py, "get_running_loop"))?;
        let context = py
            .import(pyo3::intern!(py, "contextvars"))?
            .call_method0(pyo3::intern!(py, "copy_context"))?;
        let dispatcher = Py::new(py, PythonAwaitDispatcher::new(event_loop.clone().unbind()))?;
        let is_awaitable = py
            .import(pyo3::intern!(py, "inspect"))?
            .getattr(pyo3::intern!(py, "isawaitable"))?;
        Ok(Self {
            event_loop: Arc::new(PythonRoot::new(event_loop.unbind())),
            context: Arc::new(PythonRoot::new(context.unbind())),
            dispatcher: Arc::new(PythonRoot::new(dispatcher.into_any())),
            is_awaitable: Arc::new(PythonRoot::new(is_awaitable.unbind())),
        })
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(self.event_loop.object())?;
        visit.call(self.context.object())?;
        visit.call(self.dispatcher.object())?;
        visit.call(self.is_awaitable.object())
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
    /// Called only by the captured Python loop, never on a Tokio worker.
    fn create_task<'py>(
        py: Python<'py>,
        awaitable: Py<PyAny>,
        state: &PythonAwaitState,
    ) -> PyResult<Bound<'py, PyAny>> {
        let asyncio = py.import(pyo3::intern!(py, "asyncio"))?;
        let event_loop = state.event_loop.object().bind(py);
        if Self::uses_default_coroutine_task(py, &awaitable, event_loop)? {
            Self::eager_task(&asyncio, awaitable, state)
        } else {
            // Futures, custom awaitables, and configured task factories keep
            // their existing asyncio scheduling policy.
            asyncio
                .getattr(pyo3::intern!(py, "ensure_future"))?
                .call1((awaitable,))
        }
    }

    fn uses_default_coroutine_task(
        py: Python<'_>,
        awaitable: &Py<PyAny>,
        event_loop: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let is_coroutine = py
            .import(pyo3::intern!(py, "inspect"))?
            .call_method1(pyo3::intern!(py, "iscoroutine"), (awaitable,))?
            .is_truthy()?;
        Ok(is_coroutine
            && event_loop
                .call_method0(pyo3::intern!(py, "get_task_factory"))?
                .is_none())
    }

    fn eager_task<'py>(
        asyncio: &Bound<'py, PyModule>,
        awaitable: Py<PyAny>,
        state: &PythonAwaitState,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = asyncio.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(pyo3::intern!(py, "loop"), state.event_loop.object())?;
        // Task copies the dispatch context instead of re-entering it. A
        // queued cancellation cannot start an unstarted coroutine eagerly.
        kwargs.set_item(
            pyo3::intern!(py, "eager_start"),
            !state.cancel_requested.load(Ordering::Acquire),
        )?;
        asyncio
            .getattr(pyo3::intern!(py, "Task"))?
            .call((awaitable,), Some(&kwargs))
    }

    fn take_request(&mut self) -> PyResult<(Py<PyAny>, Arc<PythonAwaitState>)> {
        let awaitable = self
            .awaitable
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Python awaitable was already scheduled"))?;
        let state = self
            .state
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("Python await state was already scheduled"))?;
        Ok((awaitable, state))
    }

    fn finish_if_ready(&mut self, task: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = task.py();
        if !task.call_method0(pyo3::intern!(py, "done"))?.is_truthy()? {
            return Ok(false);
        }
        let result = task
            .call_method0(pyo3::intern!(py, "result"))
            .map(Bound::unbind);
        self.state.take();
        self.completion.send(result);
        Ok(true)
    }

    fn schedule(&mut self, py: Python<'_>) -> PyResult<()> {
        let (awaitable, state) = self.take_request()?;
        let task = Self::create_task(py, awaitable, &state)?;
        if state.cancel_requested.load(Ordering::Acquire) {
            task.call_method0(pyo3::intern!(py, "cancel"))?;
        }
        if self.finish_if_ready(&task)? {
            return Ok(());
        }
        state.register_task(task.clone().unbind());
        self.install_completer(py, &task, &state)?;
        self.state.take();
        Ok(())
    }

    fn install_completer(
        &self,
        py: Python<'_>,
        task: &Bound<'_, PyAny>,
        state: &Arc<PythonAwaitState>,
    ) -> PyResult<()> {
        let completer = match Py::new(
            py,
            PythonAwaitCompleter {
                state: Some(Arc::clone(state)),
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

type PythonAwaitReceiver = tokio::sync::oneshot::Receiver<PyResult<Py<PyAny>>>;

fn is_python_awaitable(
    value: &Py<PyAny>,
    context: Option<&Arc<PythonAsyncContext>>,
) -> PyResult<bool> {
    Python::attach(|py| {
        if let Some(context) = context {
            return context
                .is_awaitable
                .object()
                .bind(py)
                .call1((value.bind(py),))?
                .is_truthy();
        }
        py.import(pyo3::intern!(py, "inspect"))?
            .getattr(pyo3::intern!(py, "isawaitable"))?
            .call1((value.bind(py),))?
            .is_truthy()
    })
}

fn python_async_context(
    py: Python<'_>,
    context: Option<&Arc<PythonAsyncContext>>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    if let Some(context) = context {
        return Ok((
            context.event_loop.object().clone_ref(py),
            context
                .context
                .object()
                .bind(py)
                .call_method0(pyo3::intern!(py, "copy"))?
                .unbind(),
        ));
    }
    let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
    Ok((locals.event_loop(py).unbind(), locals.context(py).unbind()))
}

fn schedule_python_await(
    value: Py<PyAny>,
    awaits: &Arc<PythonAwaitRegistry>,
    context: Option<&Arc<PythonAsyncContext>>,
) -> PyResult<(PythonAwaitReceiver, PythonAwaitCancelGuard)> {
    Python::attach(|py| {
        let (event_loop, python_context) = python_async_context(py, context)?;
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
        if let Some(context) = context {
            PythonAwaitDispatcher::submit(
                context
                    .dispatcher
                    .object()
                    .bind(py)
                    .cast::<PythonAwaitDispatcher>()?,
                scheduler,
            )?;
        } else {
            let kwargs = PyDict::new(py);
            kwargs.set_item(pyo3::intern!(py, "context"), python_context)?;
            event_loop.bind(py).call_method(
                pyo3::intern!(py, "call_soon_threadsafe"),
                (scheduler,),
                Some(&kwargs),
            )?;
        }
        Ok((receiver, PythonAwaitCancelGuard { state: Some(state) }))
    })
}

async fn await_python_result(
    receiver: PythonAwaitReceiver,
    cancellation: &mut PythonAwaitCancelGuard,
) -> PyResult<Py<PyAny>> {
    if let Ok(result) = receiver.await {
        return result;
    }
    cancellation.cancel();
    Err(PyRuntimeError::new_err(
        "Python awaitable completion channel closed unexpectedly",
    ))
}

async fn resolve_python_with_context(
    value: Py<PyAny>,
    callback_name: &str,
    awaits: &Arc<PythonAwaitRegistry>,
    context: Option<&Arc<PythonAsyncContext>>,
) -> calc_flow::Result<Py<PyAny>> {
    if !is_python_awaitable(&value, context)
        .map_err(|error| provider_error(callback_name, error))?
    {
        return Ok(value);
    }
    let (receiver, mut cancellation) = schedule_python_await(value, awaits, context)
        .map_err(|error| provider_error(callback_name, error))?;
    let result = await_python_result(receiver, &mut cancellation).await;
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

#[cfg(test)]
mod tests {
    use pyo3::types::PyDict;

    use super::*;

    #[pyfunction]
    fn make_test_scheduler(
        py: Python<'_>,
        awaitable: Py<PyAny>,
        event_loop: Py<PyAny>,
    ) -> PyResult<Py<PythonAwaitScheduler>> {
        let context = py
            .import("contextvars")?
            .call_method0("copy_context")?
            .unbind();
        Py::new(
            py,
            PythonAwaitScheduler {
                awaitable: Some(awaitable),
                context: Some(context),
                state: Some(Arc::new(PythonAwaitState::new(event_loop))),
                completion: completion(),
            },
        )
    }

    #[test]
    fn ready_coroutine_finishes_in_the_bridge_dispatch_turn() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "make_scheduler",
                    wrap_pyfunction!(make_test_scheduler, py).unwrap(),
                )
                .unwrap();
            py.run(
                cr#"
import asyncio
events = []
async def ready():
    events.append("ready")
    return 42
async def exercise():
    loop = asyncio.get_running_loop()
    loop.call_soon(make_scheduler(ready(), loop))
    await asyncio.sleep(0)
    assert events == ["ready"], events
asyncio.run(exercise())
"#,
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[test]
    fn completed_future_is_delivered_without_an_extra_loop_turn() {
        Python::initialize();
        Python::attach(|py| {
            let event_loop = py
                .import("asyncio")
                .unwrap()
                .call_method0("new_event_loop")
                .unwrap();
            let future = event_loop.call_method0("create_future").unwrap();
            future.call_method1("set_result", (42,)).unwrap();
            let (sender, mut receiver) = tokio::sync::oneshot::channel();
            let mut scheduler = PythonAwaitScheduler {
                awaitable: Some(future.unbind()),
                context: None,
                state: Some(Arc::new(PythonAwaitState::new(event_loop.clone().unbind()))),
                completion: Arc::new(PythonAwaitCompletion {
                    sender: Mutex::new(Some(sender)),
                    lease: Mutex::new(None),
                }),
            };
            scheduler.schedule(py).unwrap();
            let result = receiver.try_recv();
            event_loop.call_method0("close").unwrap();
            assert_eq!(
                result.unwrap().unwrap().bind(py).extract::<i64>().unwrap(),
                42
            );
        });
    }

    #[test]
    fn configured_task_factory_keeps_its_scheduling_policy() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "make_scheduler",
                    wrap_pyfunction!(make_test_scheduler, py).unwrap(),
                )
                .unwrap();
            py.run(
                cr#"
import asyncio
events = []
calls = []
async def ready():
    events.append("ready")
def factory(loop, coro, **kwargs):
    calls.append("factory")
    return asyncio.Task(coro, loop=loop, **kwargs)
async def exercise():
    loop = asyncio.get_running_loop()
    loop.set_task_factory(factory)
    try:
        make_scheduler(ready(), loop)()
        assert calls == ["factory"], calls
        assert events == [], events
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert events == ["ready"], events
    finally:
        loop.set_task_factory(None)
asyncio.run(exercise())
"#,
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[pyfunction]
    fn make_cancelled_scheduler(
        py: Python<'_>,
        awaitable: Py<PyAny>,
        event_loop: Py<PyAny>,
    ) -> PyResult<Py<PythonAwaitScheduler>> {
        let scheduler = make_test_scheduler(py, awaitable, event_loop)?;
        scheduler
            .borrow(py)
            .state
            .as_ref()
            .unwrap()
            .request_cancel();
        Ok(scheduler)
    }

    #[test]
    fn custom_awaitables_keep_asyncio_suspension_and_task_factory_behavior() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "make_scheduler",
                    wrap_pyfunction!(make_test_scheduler, py).unwrap(),
                )
                .unwrap();
            py.run(
                cr#"
import asyncio
events = []
calls = []
class CustomAwaitable:
    def __init__(self, gate):
        self.gate = gate
    def __await__(self):
        events.append("started")
        yield from self.gate.__await__()
        events.append("finished")
        return 11
def factory(loop, coro, **kwargs):
    calls.append("factory")
    return asyncio.Task(coro, loop=loop, **kwargs)
async def exercise():
    loop = asyncio.get_running_loop()
    gate = loop.create_future()
    loop.set_task_factory(factory)
    try:
        make_scheduler(CustomAwaitable(gate), loop)()
        assert calls == ["factory"], calls
        assert events == [], events
        await asyncio.sleep(0)
        assert events == ["started"], events
        gate.set_result(None)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert events == ["started", "finished"], events
    finally:
        loop.set_task_factory(None)
asyncio.run(exercise())
"#,
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[test]
    fn queued_cancellation_never_starts_the_coroutine() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "make_scheduler",
                    wrap_pyfunction!(make_cancelled_scheduler, py).unwrap(),
                )
                .unwrap();
            py.run(
                cr#"
import asyncio
events = []
async def ready():
    events.append("must not start")
def factory(loop, coro, **kwargs):
    return asyncio.Task(coro, loop=loop, **kwargs)
async def exercise():
    loop = asyncio.get_running_loop()
    for policy in (None, factory):
        loop.set_task_factory(policy)
        make_scheduler(ready(), loop)()
        for _ in range(3):
            await asyncio.sleep(0)
        assert events == [], events
    loop.set_task_factory(None)
asyncio.run(exercise())
"#,
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[test]
    fn completed_future_errors_keep_the_original_python_exception() {
        Python::initialize();
        Python::attach(|py| {
            let event_loop = py
                .import("asyncio")
                .unwrap()
                .call_method0("new_event_loop")
                .unwrap();
            let future = event_loop.call_method0("create_future").unwrap();
            let error = pyo3::exceptions::PyValueError::new_err("callback failed");
            future
                .call_method1("set_exception", (error.value(py),))
                .unwrap();
            let (sender, mut receiver) = tokio::sync::oneshot::channel();
            let mut scheduler = PythonAwaitScheduler {
                awaitable: Some(future.unbind()),
                context: None,
                state: Some(Arc::new(PythonAwaitState::new(event_loop.clone().unbind()))),
                completion: Arc::new(PythonAwaitCompletion {
                    sender: Mutex::new(Some(sender)),
                    lease: Mutex::new(None),
                }),
            };
            scheduler.schedule(py).unwrap();
            let actual = receiver.try_recv().unwrap().unwrap_err();
            assert!(actual.value(py).is(error.value(py)));
            event_loop.call_method0("close").unwrap();
        });
    }

    fn completion() -> Arc<PythonAwaitCompletion> {
        let (sender, _receiver) = tokio::sync::oneshot::channel();
        Arc::new(PythonAwaitCompletion {
            sender: Mutex::new(Some(sender)),
            lease: Mutex::new(None),
        })
    }

    #[pyfunction]
    fn submit_test_callbacks(callback: &Bound<'_, PyAny>, count: usize) -> PyResult<()> {
        let py = callback.py();
        let context = Arc::new(PythonAsyncContext::capture(py)?);
        let awaits = Arc::new(PythonAwaitRegistry::new());
        for _ in 0..count {
            let value = callback.call0()?.unbind();
            let (_receiver, mut cancellation) =
                schedule_python_await(value, &awaits, Some(&context))?;
            cancellation.disarm();
        }
        Ok(())
    }

    #[test]
    fn ready_dispatches_share_one_wakeup_and_yield_between_bounded_groups() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "submit",
                    wrap_pyfunction!(submit_test_callbacks, py).unwrap(),
                )
                .unwrap();
            py.run(
                cr#"
import asyncio
events = []
wakeups = []
async def ready():
    events.append("ready")
async def exercise():
    loop = asyncio.get_running_loop()
    original = loop.call_soon_threadsafe
    def counted(*args, **kwargs):
        wakeups.append(1)
        return original(*args, **kwargs)
    loop.call_soon_threadsafe = counted
    try:
        submit(ready, 129)
        assert len(wakeups) == 1, len(wakeups)
        await asyncio.sleep(0)
        assert len(events) == 64, len(events)
        await asyncio.sleep(0)
        assert len(events) == 128, len(events)
        await asyncio.sleep(0)
        assert len(events) == 129, len(events)
    finally:
        loop.call_soon_threadsafe = original
asyncio.run(exercise())
"#,
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[test]
    fn dispatch_keeps_independent_contexts_across_suspension_and_gc() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "submit",
                    wrap_pyfunction!(submit_test_callbacks, py).unwrap(),
                )
                .unwrap();
            py.run(
                cr#"
import asyncio
import contextvars
import gc
value = contextvars.ContextVar("value", default="unset")
events = []
async def exercise():
    gate = asyncio.Event()
    async def ready():
        assert value.get() == "captured", value.get()
        value.set("private")
        await gate.wait()
        events.append(value.get())
    value.set("captured")
    submit(ready, 3)
    value.set("caller")
    await asyncio.sleep(0)
    gc.collect()
    gate.set()
    for _ in range(3):
        await asyncio.sleep(0)
    assert events == ["private"] * 3, events
    assert value.get() == "caller", value.get()
asyncio.run(exercise())
"#,
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[test]
    fn gc_referents_keep_pending_await_resources_reachable() {
        Python::initialize();
        Python::attach(|py| {
            let event_loop = PyDict::new(py).into_any().unbind();
            let task = PyDict::new(py).into_any().unbind();
            let awaitable = PyDict::new(py).into_any().unbind();
            let context = PyDict::new(py).into_any().unbind();
            let state = Arc::new(PythonAwaitState::new(event_loop.clone_ref(py)));
            state.register_task(task.clone_ref(py));
            let scheduler = Py::new(
                py,
                PythonAwaitScheduler {
                    awaitable: Some(awaitable.clone_ref(py)),
                    context: Some(context.clone_ref(py)),
                    state: Some(Arc::clone(&state)),
                    completion: completion(),
                },
            )
            .unwrap();
            let completer = Py::new(
                py,
                PythonAwaitCompleter {
                    state: Some(state),
                    completion: completion(),
                },
            )
            .unwrap();
            let locals = PyDict::new(py);
            locals.set_item("scheduler", scheduler).unwrap();
            locals.set_item("completer", completer).unwrap();
            locals.set_item("event_loop", event_loop).unwrap();
            locals.set_item("task", task).unwrap();
            locals.set_item("awaitable", awaitable).unwrap();
            locals.set_item("context", context).unwrap();
            py.run(
                c"import gc\ngc.collect()\nscheduler_referents = gc.get_referents(scheduler)\nassert any(value is awaitable for value in scheduler_referents)\nassert any(value is context for value in scheduler_referents)\nassert any(value is event_loop for value in scheduler_referents)\nassert any(value is task for value in scheduler_referents)\ncompleter_referents = gc.get_referents(completer)\nassert any(value is event_loop for value in completer_referents)\nassert any(value is task for value in completer_referents)",
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[test]
    fn closed_completion_channel_cancels_the_await_and_reports_the_error() {
        Python::initialize();
        let state = Python::attach(|py| Arc::new(PythonAwaitState::new(py.None())));
        let (sender, receiver) = tokio::sync::oneshot::channel();
        drop(sender);
        let mut cancellation = PythonAwaitCancelGuard {
            state: Some(Arc::clone(&state)),
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let error = runtime
            .block_on(await_python_result(receiver, &mut cancellation))
            .unwrap_err();

        assert!(state.cancel_requested.load(Ordering::Acquire));
        assert!(cancellation.state.is_none());
        Python::attach(|py| {
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(
                error.value(py).str().unwrap().to_str().unwrap(),
                "Python awaitable completion channel closed unexpectedly"
            );
        });
    }

    #[test]
    fn provider_error_preserves_callback_identity_and_message() {
        let error = provider_error("load_prices", "callback failed");

        assert!(matches!(
            error,
            calc_flow::CalcFlowError::ExternalProvider {
                provider,
                name,
                version,
                message,
            } if provider == "python"
                && name == "load_prices"
                && version == "1"
                && message == "callback failed"
        ));
    }
}
