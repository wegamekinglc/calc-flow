use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyAny, PyBool, PyDict, PyList, PyTuple},
};

use crate::{
    batch::PyBatch,
    config::PythonRoot,
    pipeline::{PyExecutionPlan, PyRunResult},
    store::PyFileCheckpointStore,
};

const CLEARED_MICRO_MESSAGE: &str = "MicroBatchRunner has been cleared by garbage collection";
const CLEARED_STREAMING_MESSAGE: &str = "StreamingRunner has been cleared by garbage collection";

fn provider_error(name: &str, error: impl std::fmt::Display) -> calc_flow::CalcFlowError {
    calc_flow::CalcFlowError::ExternalProvider {
        provider: "python".into(),
        name: name.into(),
        version: "1".into(),
        message: error.to_string(),
    }
}

async fn resolve_python(value: Py<PyAny>, callback_name: &str) -> calc_flow::Result<Py<PyAny>> {
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
    let future = Python::attach(|py| pyo3_async_runtimes::tokio::into_future(value.into_bound(py)))
        .map_err(|error| provider_error(callback_name, error))?;
    future
        .await
        .map_err(|error| provider_error(callback_name, error))
}

fn python_json(value: &Bound<'_, PyAny>, label: &str) -> calc_flow::Result<serde_json::Value> {
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

struct PythonSource {
    callback: Arc<PythonRoot>,
    roots: Arc<RootRegistry>,
}

impl PythonSource {
    fn call_method(
        &self,
        method: &'static str,
        argument: Option<Py<PyAny>>,
    ) -> calc_flow::Result<Py<PyAny>> {
        Python::attach(|py| match argument {
            Some(argument) => self.callback.object().call_method1(py, method, (argument,)),
            None => self.callback.object().call_method0(py, method),
        })
        .map_err(|error| provider_error(&format!("source.{method}"), error))
    }
}

#[async_trait]
impl calc_flow::Source for PythonSource {
    async fn open(&mut self, cursor: Option<serde_json::Value>) -> calc_flow::Result<()> {
        let cursor =
            serde_json::to_string(&cursor).map_err(|error| provider_error("source.open", error))?;
        let argument =
            Python::attach(|py| crate::config::json_to_python(py, &cursor).map(Bound::unbind))
                .map_err(|error| provider_error("source.open", error))?;
        let value = self.call_method("open", Some(argument))?;
        resolve_python(value, "source.open").await?;
        Ok(())
    }

    async fn next(&mut self) -> calc_flow::Result<Option<calc_flow::SourceItem>> {
        let value = self.call_method("next", None)?;
        let value = resolve_python(value, "source.next").await?;
        Python::attach(|py| {
            let value = value.bind(py);
            if value.is_none() {
                self.roots.replace_current_source_payload(None);
                return Ok(None);
            }
            let tuple = value.cast::<PyTuple>().map_err(|_| {
                provider_error(
                    "source.next",
                    "must return None or exactly (Batch, cursor, sequence)",
                )
            })?;
            if tuple.len() != 3 {
                return Err(provider_error(
                    "source.next",
                    "must return None or exactly (Batch, cursor, sequence)",
                ));
            }
            let batch = tuple
                .get_item(0)
                .map_err(|error| provider_error("source.next", error))?
                .extract::<PyRef<'_, PyBatch>>()
                .map_err(|_| provider_error("source.next", "item 0 must be a calc_flow.Batch"))?;
            let batch = crate::batch::rehome_python_payload(
                py,
                batch
                    .clone_inner()
                    .map_err(|error| provider_error("source.next", error))?,
            )
            .map_err(|error| provider_error("source.next", error))?;
            let cursor_value = tuple
                .get_item(1)
                .map_err(|error| provider_error("source.next", error))?;
            let cursor = if cursor_value.is_none() {
                None
            } else {
                Some(python_json(&cursor_value, "source.next cursor")?)
            };
            let sequence_value = tuple
                .get_item(2)
                .map_err(|error| provider_error("source.next", error))?;
            if sequence_value.is_instance_of::<PyBool>() {
                return Err(provider_error(
                    "source.next",
                    "sequence must be a non-negative u64 integer",
                ));
            }
            let sequence = sequence_value.extract::<u64>().map_err(|_| {
                provider_error("source.next", "sequence must be a non-negative u64 integer")
            })?;
            let item = calc_flow::SourceItem {
                batch,
                cursor,
                sequence,
            };
            let payload =
                crate::batch::python_payload_root(&item.batch).map(|_| item.batch.clone());
            self.roots.replace_current_source_payload(payload);
            Ok(Some(item))
        })
    }
}

struct PythonSink {
    callback: Arc<PythonRoot>,
    output: String,
}

#[async_trait]
impl calc_flow::Sink for PythonSink {
    async fn write(
        &mut self,
        batch: &calc_flow::Batch,
        _context: &calc_flow::RunContext,
    ) -> calc_flow::Result<()> {
        let value = Python::attach(|py| {
            let batch = Py::new(py, PyBatch::from_inner_python(py, batch.clone())?)?;
            self.callback.object().call1(py, (batch,))
        })
        .map_err(|error| provider_error(&format!("sink.{}", self.output), error))?;
        resolve_python(value, &format!("sink.{}", self.output)).await?;
        Ok(())
    }
}

fn validate_source(py: Python<'_>, source: &Py<PyAny>) -> PyResult<()> {
    for method in ["open", "next"] {
        let callback = source.bind(py).getattr(method).map_err(|_| {
            PyTypeError::new_err(format!("source must provide callable {method}()"))
        })?;
        if !callback.is_callable() {
            return Err(PyTypeError::new_err(format!(
                "source must provide callable {method}()"
            )));
        }
    }
    Ok(())
}

fn build_sinks(
    py: Python<'_>,
    sinks: Option<&Bound<'_, PyDict>>,
    external_outputs: &BTreeSet<String>,
) -> PyResult<(calc_flow::SinkRouter, Vec<Arc<PythonRoot>>)> {
    let mut router = calc_flow::SinkRouter::new();
    let mut roots = Vec::new();
    let Some(sinks) = sinks else {
        return Ok((router, roots));
    };
    for output in sinks.keys() {
        let output = output
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("sink output names must be strings"))?;
        if output.is_empty() {
            return Err(crate::error::to_py_err(
                calc_flow::CalcFlowError::InvalidArgument {
                    field: "sink.output".into(),
                    message: "must not be empty".into(),
                },
            ));
        }
        if !external_outputs.contains(&output) {
            return Err(crate::error::to_py_err(
                calc_flow::CalcFlowError::InvalidArgument {
                    field: "sinks".into(),
                    message: format!("sink configured for unknown graph output {output:?}"),
                },
            ));
        }
    }
    for (output, callbacks) in sinks {
        let output = output
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("sink output names must be strings"))?;
        let callbacks: Vec<Bound<'_, PyAny>> = if let Ok(items) = callbacks.cast::<PyList>() {
            items.iter().collect()
        } else if let Ok(items) = callbacks.cast::<PyTuple>() {
            items.iter().collect()
        } else {
            return Err(PyTypeError::new_err(format!(
                "sinks[{output:?}] must be a list or tuple of callables"
            )));
        };
        for callback in callbacks {
            if !callback.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "sinks[{output:?}] must contain only callables"
                )));
            }
            let root = Arc::new(PythonRoot::new(callback.unbind()));
            router
                .add(
                    &output,
                    Box::new(PythonSink {
                        callback: Arc::clone(&root),
                        output: output.clone(),
                    }),
                )
                .map_err(crate::error::to_py_err)?;
            roots.push(root);
        }
    }
    let _ = py;
    Ok((router, roots))
}

struct RunnerRoots {
    plan: Py<PyAny>,
    source: Option<Arc<PythonRoot>>,
    current_source_payload: Option<calc_flow::Batch>,
    persistent_sinks: Vec<Arc<PythonRoot>>,
    active_sinks: BTreeMap<u64, Vec<Arc<PythonRoot>>>,
}

struct RootRegistry {
    roots: RwLock<RunnerRoots>,
    next_id: AtomicU64,
}

impl RootRegistry {
    fn new(
        plan: Py<PyAny>,
        source: Option<Arc<PythonRoot>>,
        persistent_sinks: Vec<Arc<PythonRoot>>,
    ) -> Self {
        Self {
            roots: RwLock::new(RunnerRoots {
                plan,
                source,
                current_source_payload: None,
                persistent_sinks,
                active_sinks: BTreeMap::new(),
            }),
            next_id: AtomicU64::new(0),
        }
    }

    fn retain(self: &Arc<Self>, roots: Vec<Arc<PythonRoot>>) -> RootLease {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.roots.write().active_sinks.insert(id, roots);
        RootLease {
            registry: Arc::clone(self),
            id,
        }
    }

    fn replace_current_source_payload(&self, payload: Option<calc_flow::Batch>) {
        let previous = {
            let mut roots = self.roots.write();
            std::mem::replace(&mut roots.current_source_payload, payload)
        };
        drop(previous);
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let roots = self.roots.read();
        visit.call(&roots.plan)?;
        if let Some(source) = &roots.source {
            visit.call(source.object())?;
        }
        if let Some(payload) = &roots.current_source_payload {
            if let Some(payload) = crate::batch::python_payload_root(payload) {
                visit.call(payload)?;
            }
        }
        for root in &roots.persistent_sinks {
            visit.call(root.object())?;
        }
        for active in roots.active_sinks.values() {
            for root in active {
                visit.call(root.object())?;
            }
        }
        Ok(())
    }
}

struct RootLease {
    registry: Arc<RootRegistry>,
    id: u64,
}

impl Drop for RootLease {
    fn drop(&mut self) {
        let roots = self.registry.roots.write().active_sinks.remove(&self.id);
        drop(roots);
    }
}

struct RunnerSlot<T> {
    inner: Mutex<Option<T>>,
}

impl<T> RunnerSlot<T> {
    fn new(runner: T) -> Self {
        Self {
            inner: Mutex::new(Some(runner)),
        }
    }

    fn checkout(self: &Arc<Self>, message: &'static str) -> PyResult<RunnerCheckout<T>> {
        let runner = self
            .inner
            .lock()
            .take()
            .ok_or_else(|| PyRuntimeError::new_err(message))?;
        Ok(RunnerCheckout {
            slot: Arc::clone(self),
            runner: Some(runner),
        })
    }
}

struct RunnerCheckout<T> {
    slot: Arc<RunnerSlot<T>>,
    runner: Option<T>,
}

impl<T> RunnerCheckout<T> {
    fn get_mut(&mut self) -> &mut T {
        self.runner
            .as_mut()
            .expect("checked-out runner remains owned until guard drop")
    }
}

impl<T> Drop for RunnerCheckout<T> {
    fn drop(&mut self) {
        let runner = self.runner.take();
        let previous = self
            .slot
            .inner
            .lock()
            .replace(runner.expect("checked-out runner remains owned until guard drop"));
        debug_assert!(previous.is_none());
        drop(previous);
    }
}

struct MicroShared {
    runner: Arc<RunnerSlot<calc_flow::MicroBatchRunner>>,
    roots: Arc<RootRegistry>,
}

#[pyclass(name = "_MicroBatchRunner", frozen, module = "calc_flow._native")]
pub(crate) struct PyMicroBatchRunner {
    state: RwLock<Option<Arc<MicroShared>>>,
}

impl PyMicroBatchRunner {
    fn shared(&self) -> PyResult<Arc<MicroShared>> {
        self.state
            .read()
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_MICRO_MESSAGE))
    }
}

#[pymethods]
impl PyMicroBatchRunner {
    #[new]
    #[pyo3(signature = (plan, source, checkpoints, sinks=None, checkpoint_every=100))]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts native class arguments as owned PyRef guards"
    )]
    fn new(
        py: Python<'_>,
        plan: PyRef<'_, PyExecutionPlan>,
        source: Py<PyAny>,
        checkpoints: PyRef<'_, PyFileCheckpointStore>,
        sinks: Option<&Bound<'_, PyDict>>,
        checkpoint_every: u64,
    ) -> PyResult<Self> {
        validate_source(py, &source)?;
        let source_root = Arc::new(PythonRoot::new(source));
        let owned = PyExecutionPlan::owned(plan, py)?;
        let external_outputs = owned.inner().external_outputs().keys().cloned().collect();
        let (sinks, sink_roots) = build_sinks(py, sinks, &external_outputs)?;
        let roots = Arc::new(RootRegistry::new(
            owned.owner().clone_ref(py),
            Some(Arc::clone(&source_root)),
            sink_roots,
        ));
        let runner = calc_flow::MicroBatchRunner::new(
            Arc::clone(owned.inner()),
            Box::new(PythonSource {
                callback: source_root,
                roots: Arc::clone(&roots),
            }),
            sinks,
            checkpoints.clone_store(),
            checkpoint_every,
        )
        .map_err(crate::error::to_py_err)?;
        Ok(Self {
            state: RwLock::new(Some(Arc::new(MicroShared {
                runner: Arc::new(RunnerSlot::new(runner)),
                roots,
            }))),
        })
    }

    fn next_async<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shared = slf.shared()?;
        let roots = Arc::clone(&shared.roots);
        let owner = slf.into_pyobject(py)?.into_any().unbind();
        let mut runner = shared
            .runner
            .checkout("MicroBatchRunner already has an operation in progress")?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = runner
                .get_mut()
                .next()
                .await
                .map_err(crate::error::to_py_err)?;
            roots.replace_current_source_payload(None);
            let result = Python::attach(|py| {
                result
                    .map(|result| Py::new(py, PyRunResult::from_inner(py, result)?))
                    .transpose()
            });
            drop(runner);
            drop(owner);
            result
        })
    }

    fn reset_async<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shared = slf.shared()?;
        let roots = Arc::clone(&shared.roots);
        let owner = slf.into_pyobject(py)?.into_any().unbind();
        let mut runner = shared
            .runner
            .checkout("MicroBatchRunner already has an operation in progress")?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = runner
                .get_mut()
                .reset()
                .await
                .map_err(crate::error::to_py_err);
            if result.is_ok() {
                roots.replace_current_source_payload(None);
            }
            drop(runner);
            drop(owner);
            result
        })
    }

    fn plan_snapshot_async<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shared = slf.shared()?;
        let owner = slf.into_pyobject(py)?.into_any().unbind();
        let mut runner = shared
            .runner
            .checkout("MicroBatchRunner already has an operation in progress")?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let state = runner
                .get_mut()
                .plan_snapshot()
                .await
                .map_err(crate::error::to_py_err)?;
            let encoded = serde_json::to_string(&state)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            let result =
                Python::attach(|py| crate::config::json_to_python(py, &encoded).map(Bound::unbind));
            drop(runner);
            drop(owner);
            result
        })
    }

    #[allow(clippy::needless_pass_by_value)]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(shared) = self.state.read().as_ref() {
            shared.roots.traverse(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        let state = self.state.write().take();
        drop(state);
    }
}

struct StreamingShared {
    runner: Arc<RunnerSlot<calc_flow::StreamingRunner>>,
    roots: Arc<RootRegistry>,
    external_outputs: BTreeSet<String>,
}

#[pyclass(name = "_StreamingRunner", frozen, module = "calc_flow._native")]
pub(crate) struct PyStreamingRunner {
    state: RwLock<Option<Arc<StreamingShared>>>,
}

impl PyStreamingRunner {
    fn shared(&self) -> PyResult<Arc<StreamingShared>> {
        self.state
            .read()
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_STREAMING_MESSAGE))
    }
}

#[pymethods]
impl PyStreamingRunner {
    #[new]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts native class arguments as owned PyRef guards"
    )]
    fn new(
        py: Python<'_>,
        plan: PyRef<'_, PyExecutionPlan>,
        checkpoints: PyRef<'_, PyFileCheckpointStore>,
    ) -> PyResult<Self> {
        let owned = PyExecutionPlan::owned(plan, py)?;
        let external_outputs = owned.inner().external_outputs().keys().cloned().collect();
        let runner =
            calc_flow::StreamingRunner::new(Arc::clone(owned.inner()), checkpoints.clone_store())
                .map_err(crate::error::to_py_err)?;
        let roots = Arc::new(RootRegistry::new(
            owned.owner().clone_ref(py),
            None,
            Vec::new(),
        ));
        Ok(Self {
            state: RwLock::new(Some(Arc::new(StreamingShared {
                runner: Arc::new(RunnerSlot::new(runner)),
                roots,
                external_outputs,
            }))),
        })
    }

    #[pyo3(signature = (batch, sinks=None))]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts native class arguments as owned PyRef guards"
    )]
    fn step_async<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        batch: PyRef<'_, PyBatch>,
        sinks: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shared = slf.shared()?;
        let batch = crate::batch::rehome_python_payload(py, batch.clone_inner()?)?;
        let (mut sinks, roots) = build_sinks(py, sinks, &shared.external_outputs)?;
        let root_lease = shared.roots.retain(roots);
        let owner = slf.into_pyobject(py)?.into_any().unbind();
        let mut runner = shared
            .runner
            .checkout("StreamingRunner already has an operation in progress")?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = runner
                .get_mut()
                .step(batch, &mut sinks)
                .await
                .map_err(crate::error::to_py_err)?;
            let result = Python::attach(|py| PyRunResult::from_inner(py, result));
            drop(runner);
            drop(root_lease);
            drop(owner);
            result
        })
    }

    fn reset_async<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shared = slf.shared()?;
        let owner = slf.into_pyobject(py)?.into_any().unbind();
        let mut runner = shared
            .runner
            .checkout("StreamingRunner already has an operation in progress")?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = runner
                .get_mut()
                .reset()
                .await
                .map_err(crate::error::to_py_err);
            drop(runner);
            drop(owner);
            result
        })
    }

    fn plan_snapshot_async<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shared = slf.shared()?;
        let owner = slf.into_pyobject(py)?.into_any().unbind();
        let mut runner = shared
            .runner
            .checkout("StreamingRunner already has an operation in progress")?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let state = runner
                .get_mut()
                .plan_snapshot()
                .await
                .map_err(crate::error::to_py_err)?;
            let encoded = serde_json::to_string(&state)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            let result =
                Python::attach(|py| crate::config::json_to_python(py, &encoded).map(Bound::unbind));
            drop(runner);
            drop(owner);
            result
        })
    }

    #[allow(clippy::needless_pass_by_value)]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(shared) = self.state.read().as_ref() {
            shared.roots.traverse(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        let state = self.state.write().take();
        drop(state);
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMicroBatchRunner>()?;
    module.add_class::<PyStreamingRunner>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, sync::atomic::AtomicU64};

    use calc_flow::{Sink as _, Source as _};
    use datafusion::arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use pyo3::types::{PyList, PyString};

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    fn directory(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "calc-flow-python-runtime-{label}-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn plan(name: &str) -> PyExecutionPlan {
        let core = calc_flow::PipelineBuilder::new(name)
            .unwrap()
            .add_node(
                "calc",
                Box::new(
                    calc_flow::ExpressionOperator::new(
                        "calc",
                        "result = value + 1",
                        Vec::new(),
                        None,
                        Vec::new(),
                    )
                    .unwrap(),
                ),
            )
            .unwrap()
            .compile(&calc_flow::UdfRegistry::new().snapshot())
            .unwrap();
        PyExecutionPlan::new(
            Arc::new(core),
            Arc::new(tokio::runtime::Runtime::new().unwrap()),
            Vec::new(),
        )
    }

    fn batch() -> PyBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let values = Arc::new(Int64Array::from(vec![1_i64]));
        let record = RecordBatch::try_new(schema, vec![values]).unwrap();
        PyBatch::from_inner(
            calc_flow::Batch::table(vec![record], calc_flow::BatchMetadata::default()).unwrap(),
        )
    }

    #[test]
    fn native_micro_and_streaming_methods_use_python_callbacks_and_core_runners() {
        Python::initialize();
        Python::attach(|py| {
            let micro_directory = directory("micro");
            let stream_directory = directory("stream");
            let locals = PyDict::new(py);
            locals
                .set_item("batch", Py::new(py, batch()).unwrap())
                .unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    "import asyncio\nclass Source:\n    def __init__(self):\n        self.emitted = False\n        self.opened = []\n    async def open(self, cursor):\n        await asyncio.sleep(0)\n        self.opened.append(cursor)\n    def next(self):\n        if self.emitted:\n            return None\n        self.emitted = True\n        return batch, {'offset': 1}, 1\nsource = Source()\ncalls = []\ndef sync_sink(value):\n    calls.append(('sync', value.num_rows))\nasync def async_sink(value):\n    await asyncio.sleep(0)\n    calls.append(('async', value.num_rows))"
                ),
                Some(&locals),
                None,
            )
            .unwrap();
            let source = locals.get_item("source").unwrap().unwrap().unbind();
            let sync_sink = locals.get_item("sync_sink").unwrap().unwrap();
            let async_sink = locals.get_item("async_sink").unwrap().unwrap();
            let sinks = PyDict::new(py);
            sinks
                .set_item(
                    "output",
                    PyList::new(py, [&sync_sink, &async_sink]).unwrap(),
                )
                .unwrap();

            let micro_plan = Py::new(py, plan("micro-native")).unwrap();
            let micro_store = Py::new(
                py,
                PyFileCheckpointStore::from_directory(micro_directory.clone()),
            )
            .unwrap();
            let micro = Py::new(
                py,
                PyMicroBatchRunner::new(
                    py,
                    micro_plan.borrow(py),
                    source,
                    micro_store.borrow(py),
                    Some(&sinks),
                    1,
                )
                .unwrap(),
            )
            .unwrap();

            let stream_plan = Py::new(py, plan("stream-native")).unwrap();
            let stream_store = Py::new(
                py,
                PyFileCheckpointStore::from_directory(stream_directory.clone()),
            )
            .unwrap();
            let streaming = Py::new(
                py,
                PyStreamingRunner::new(py, stream_plan.borrow(py), stream_store.borrow(py))
                    .unwrap(),
            )
            .unwrap();
            locals.set_item("micro", &micro).unwrap();
            locals.set_item("streaming", &streaming).unwrap();
            py.run(
                &CString::new(
                    "async def exercise():\n    first = await micro.next_async()\n    assert first.outputs['output'].num_rows == 1\n    assert await micro.next_async() is None\n    assert (await micro.plan_snapshot_async()) == {'calc': None}\n    await micro.reset_async()\n    streamed = await streaming.step_async(batch, {'output': [async_sink, sync_sink]})\n    assert streamed.outputs['output'].num_rows == 1\n    assert (await streaming.plan_snapshot_async()) == {'calc': None}\n    try:\n        await streaming.step_async(batch, {'missing': [sync_sink]})\n    except Exception as error:\n        assert 'unknown graph output' in str(error)\n    await streaming.reset_async()\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
            let calls = locals.get_item("calls").unwrap().unwrap();
            assert_eq!(calls.len().unwrap(), 4);

            micro.borrow(py).__clear__();
            assert!(micro.bind(py).call_method0("next_async").is_err());
            micro.borrow(py).__clear__();
            streaming.borrow(py).__clear__();
            assert!(streaming.bind(py).call_method0("reset_async").is_err());
            streaming.borrow(py).__clear__();

            std::fs::remove_dir_all(micro_directory).unwrap();
            std::fs::remove_dir_all(stream_directory).unwrap();
        });
    }

    #[test]
    fn adapters_validate_callbacks_json_and_in_flight_ownership() {
        Python::initialize();
        Python::attach(|py| {
            let invalid = py.eval(pyo3::ffi::c_str!("object()"), None, None).unwrap();
            let external_outputs = BTreeSet::from(["output".to_owned()]);
            assert!(validate_source(py, &invalid.clone().unbind()).is_err());

            let invalid_sinks = PyDict::new(py);
            invalid_sinks.set_item(1, PyList::empty(py)).unwrap();
            assert!(build_sinks(py, Some(&invalid_sinks), &external_outputs).is_err());
            invalid_sinks.clear();
            invalid_sinks.set_item("output", "not-a-list").unwrap();
            assert!(build_sinks(py, Some(&invalid_sinks), &external_outputs).is_err());
            invalid_sinks.clear();
            invalid_sinks
                .set_item("output", PyList::new(py, [invalid]).unwrap())
                .unwrap();
            assert!(build_sinks(py, Some(&invalid_sinks), &external_outputs).is_err());
            assert_eq!(build_sinks(py, None, &external_outputs).unwrap().1.len(), 0);

            let valid = PyDict::new(py);
            valid
                .set_item("nested", PyList::new(py, [1, 2]).unwrap())
                .unwrap();
            assert_eq!(python_json(valid.as_any(), "json").unwrap()["nested"][0], 1);
            let invalid_json = py
                .eval(pyo3::ffi::c_str!("float('nan')"), None, None)
                .unwrap();
            assert!(python_json(&invalid_json, "json").is_err());

            let registry = Arc::new(RootRegistry::new(
                PyString::new(py, "plan").unbind().into_any(),
                None,
                Vec::new(),
            ));
            let root = Arc::new(PythonRoot::new(
                PyString::new(py, "sink").unbind().into_any(),
            ));
            let lease = registry.retain(vec![root]);
            assert_eq!(registry.roots.read().active_sinks.len(), 1);
            drop(lease);
            assert!(registry.roots.read().active_sinks.is_empty());

            let slot = Arc::new(RunnerSlot::new(41));
            let mut checkout = slot.checkout("busy").unwrap();
            *checkout.get_mut() += 1;
            assert!(slot.checkout("busy").is_err());
            drop(checkout);
            assert_eq!(*slot.checkout("busy").unwrap().get_mut(), 42);
        });
    }

    #[test]
    fn source_and_sink_adapters_cover_sync_results_and_provider_failures() {
        Python::initialize();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item("batch", Py::new(py, batch()).unwrap())
                .unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    "class Valid:\n    def open(self, cursor):\n        self.cursor = cursor\n    def next(self):\n        return batch, {'offset': 1}, 1\nclass InvalidTuple(Valid):\n    def next(self): return (batch,)\nclass InvalidSequence(Valid):\n    def next(self): return batch, None, True\nclass Broken(Valid):\n    def open(self, cursor): raise RuntimeError('open failed')\ndef sink(value): return None\ndef broken_sink(value): raise RuntimeError('sink failed')\nvalid = Valid()\ninvalid_tuple = InvalidTuple()\ninvalid_sequence = InvalidSequence()\nbroken = Broken()"
                ),
                Some(&locals),
                None,
            )
            .unwrap();

            let source_roots = Arc::new(RootRegistry::new(
                PyString::new(py, "plan").unbind().into_any(),
                None,
                Vec::new(),
            ));
            let make_source = |name: &str| PythonSource {
                callback: Arc::new(PythonRoot::new(
                    locals.get_item(name).unwrap().unwrap().unbind(),
                )),
                roots: Arc::clone(&source_roots),
            };
            let mut valid = make_source("valid");
            py.detach(|| {
                runtime.block_on(async {
                    valid.open(None).await.unwrap();
                    assert!(valid.next().await.unwrap().is_some());
                });
            });
            let mut invalid_tuple = make_source("invalid_tuple");
            let mut invalid_sequence = make_source("invalid_sequence");
            let mut broken = make_source("broken");
            py.detach(|| {
                runtime.block_on(async {
                    assert!(invalid_tuple.next().await.is_err());
                    assert!(invalid_sequence.next().await.is_err());
                    assert!(broken.open(None).await.is_err());
                });
            });

            let inner = batch().clone_inner().unwrap();
            let mut sink = PythonSink {
                callback: Arc::new(PythonRoot::new(
                    locals.get_item("sink").unwrap().unwrap().unbind(),
                )),
                output: "output".into(),
            };
            let mut broken_sink = PythonSink {
                callback: Arc::new(PythonRoot::new(
                    locals.get_item("broken_sink").unwrap().unwrap().unbind(),
                )),
                output: "output".into(),
            };
            let context = calc_flow::RunContext::new(
                BTreeMap::new(),
                None,
                calc_flow::CancellationToken::new(),
            )
            .unwrap();
            py.detach(|| {
                runtime.block_on(async {
                    sink.write(&inner, &context).await.unwrap();
                    assert!(broken_sink.write(&inner, &context).await.is_err());
                });
            });
        });
    }

    #[test]
    fn native_runtime_registration_exposes_both_runner_classes() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_native").unwrap();
            register(&module).unwrap();
            assert!(module.getattr("_MicroBatchRunner").is_ok());
            assert!(module.getattr("_StreamingRunner").is_ok());
        });
    }
}
