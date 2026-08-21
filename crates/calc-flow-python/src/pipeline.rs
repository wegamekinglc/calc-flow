use std::{collections::BTreeMap, sync::Arc};

use parking_lot::{Mutex, RwLock};
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyAny, PyDict, PyString, PyTuple},
};

use crate::batch::PyBatch;
use crate::execution_options::{PyExecutionCancellation, PyExecutionOptions};

const CLEARED_PLAN_MESSAGE: &str = "ExecutionPlan has been cleared by garbage collection";
const CLEARED_STREAM_PLAN_MESSAGE: &str =
    "StreamExecutionPlan has been consumed or cleared by garbage collection";
const CLEARED_RESULT_MESSAGE: &str = "RunResult has been cleared by garbage collection";

struct PlanState {
    inner: Arc<calc_flow::BatchExecutionPlan>,
    tokio: Arc<tokio::runtime::Runtime>,
    owners: Vec<Py<PyAny>>,
    roots: Vec<Arc<crate::config::PythonRoot>>,
}

/// A core execution plan paired with the Python wrapper that exposes its roots.
///
/// Tasks that retain or execute the core plan must retain this whole value. A
/// Python class storing it must report [`Self::owner`] from `__traverse__`.
pub(crate) struct ExecutionPlanOwner {
    inner: Arc<calc_flow::BatchExecutionPlan>,
    tokio: Arc<tokio::runtime::Runtime>,
    owner: Py<PyAny>,
}

impl ExecutionPlanOwner {
    #[allow(
        dead_code,
        reason = "Task 20 passes the owned core plan to native runner construction"
    )]
    pub(crate) const fn inner(&self) -> &Arc<calc_flow::BatchExecutionPlan> {
        &self.inner
    }

    #[allow(
        dead_code,
        reason = "Task 20 uses the plan's Tokio runtime for guarded blocking facades"
    )]
    pub(crate) const fn tokio(&self) -> &Arc<tokio::runtime::Runtime> {
        &self.tokio
    }

    #[allow(
        dead_code,
        reason = "Task 20 reports the retained plan wrapper from runner __traverse__"
    )]
    pub(crate) const fn owner(&self) -> &Py<PyAny> {
        &self.owner
    }
}

#[pyclass(name = "ExecutionPlan", frozen, module = "calc_flow._native")]
pub(crate) struct PyExecutionPlan {
    state: RwLock<Option<PlanState>>,
}

struct StreamPlanState {
    inner: calc_flow::StreamExecutionPlan,
    owner: Py<PyAny>,
}

#[pyclass(name = "StreamExecutionPlan", frozen, module = "calc_flow._native")]
pub(crate) struct PyStreamExecutionPlan {
    state: Mutex<Option<StreamPlanState>>,
}

impl PyStreamExecutionPlan {
    pub(crate) fn new(inner: calc_flow::StreamExecutionPlan, owner: Py<PyAny>) -> Self {
        Self {
            state: Mutex::new(Some(StreamPlanState { inner, owner })),
        }
    }

    pub(crate) fn take(&self) -> PyResult<(calc_flow::StreamExecutionPlan, Py<PyAny>)> {
        let state = self
            .state
            .lock()
            .take()
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_STREAM_PLAN_MESSAGE))?;
        Ok((state.inner, state.owner))
    }

    fn with_plan<T>(&self, read: impl FnOnce(&calc_flow::StreamExecutionPlan) -> T) -> PyResult<T> {
        self.state
            .lock()
            .as_ref()
            .map(|state| read(&state.inner))
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_STREAM_PLAN_MESSAGE))
    }
}

#[pymethods]
impl PyStreamExecutionPlan {
    #[getter]
    fn name(&self) -> PyResult<String> {
        self.with_plan(|plan| plan.name().to_owned())
    }

    #[getter]
    fn fingerprint(&self) -> PyResult<String> {
        self.with_plan(|plan| plan.fingerprint().to_owned())
    }

    #[getter]
    fn source_binding_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self.with_plan(|plan| {
            plan.source_binding_ids()
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })?;
        PyTuple::new(py, values)
    }

    #[getter]
    fn sink_binding_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self.with_plan(|plan| {
            plan.sink_binding_ids()
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })?;
        PyTuple::new(py, values)
    }

    #[getter]
    fn requirements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let requirements = self.with_plan(|plan| {
            plan.requirements()
                .delivery
                .iter()
                .map(|(output, guarantee)| {
                    let guarantee = match guarantee {
                        calc_flow::DeliveryGuarantee::BestEffort => "best_effort",
                        calc_flow::DeliveryGuarantee::AtLeastOnce => "at_least_once",
                        calc_flow::DeliveryGuarantee::ExactlyOnce => "exactly_once",
                    };
                    (output.clone(), guarantee)
                })
                .collect::<Vec<_>>()
        })?;
        let delivery = PyDict::new(py);
        for (output, guarantee) in requirements {
            delivery.set_item(output, guarantee)?;
        }
        Ok(delivery)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(state) = self.state.lock().as_ref() {
            visit.call(&state.owner)?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        drop(self.state.lock().take());
    }
}

impl PyExecutionPlan {
    #[allow(
        dead_code,
        reason = "Task 20 can wrap a core plan while preserving explicit Python GC roots"
    )]
    pub(crate) fn new(
        inner: Arc<calc_flow::BatchExecutionPlan>,
        tokio: Arc<tokio::runtime::Runtime>,
        roots: Vec<Arc<crate::config::PythonRoot>>,
    ) -> Self {
        Self {
            state: RwLock::new(Some(PlanState {
                inner,
                tokio,
                owners: Vec::new(),
                roots: crate::config::deduplicate_python_roots(roots),
            })),
        }
    }

    pub(crate) fn new_with_owner(
        inner: Arc<calc_flow::BatchExecutionPlan>,
        tokio: Arc<tokio::runtime::Runtime>,
        owner: Py<PyAny>,
    ) -> Self {
        Self {
            state: RwLock::new(Some(PlanState {
                inner,
                tokio,
                owners: vec![owner],
                roots: Vec::new(),
            })),
        }
    }

    fn execution_handles(
        &self,
    ) -> PyResult<(
        Arc<calc_flow::BatchExecutionPlan>,
        Arc<tokio::runtime::Runtime>,
    )> {
        let state = self.state.read();
        let state = state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_PLAN_MESSAGE))?;
        Ok((Arc::clone(&state.inner), Arc::clone(&state.tokio)))
    }

    fn identity(&self) -> PyResult<(String, String)> {
        let state = self.state.read();
        let state = state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_PLAN_MESSAGE))?;
        Ok((state.inner.name().into(), state.inner.fingerprint().into()))
    }

    /// Clones native handles together with a real Python owner.
    ///
    /// The owner is what keeps callback roots traversable while native state is
    /// retained by a future or a Task 20 runner.
    pub(crate) fn owned(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<ExecutionPlanOwner> {
        let (inner, tokio) = slf.execution_handles()?;
        let owner = slf.into_pyobject(py)?.into_any().unbind();
        Ok(ExecutionPlanOwner {
            inner,
            tokio,
            owner,
        })
    }

    fn execute_async_future<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
        options: Option<PyRef<'py, PyExecutionOptions>>,
    ) -> PyResult<(Bound<'py, PyAny>, calc_flow::CancellationToken)> {
        let inputs = extract_inputs(inputs)?;
        let options = options.map_or_else(calc_flow::ExecutionOptions::default, |value| {
            value.to_core()
        });
        let cancellation = options.cancellation.clone();
        let ExecutionPlanOwner { inner, owner, .. } = Self::owned(slf, py)?;
        let future = pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .execute(inputs, options)
                .await
                .map_err(crate::error::to_py_err)?;
            let result = Python::attach(|py| PyRunResult::from_inner(py, result));
            drop(owner);
            result
        })?;
        Ok((future, cancellation))
    }
}

#[pymethods]
impl PyExecutionPlan {
    #[getter]
    fn name(&self) -> PyResult<String> {
        self.identity().map(|(name, _)| name)
    }

    #[getter]
    fn fingerprint(&self) -> PyResult<String> {
        self.identity().map(|(_, fingerprint)| fingerprint)
    }

    #[pyo3(signature = (inputs, *, options = None))]
    fn execute(
        &self,
        py: Python<'_>,
        inputs: &Bound<'_, PyDict>,
        options: Option<PyRef<'_, PyExecutionOptions>>,
    ) -> PyResult<PyRunResult> {
        let inputs = extract_inputs(inputs)?;
        let options = options.map_or_else(calc_flow::ExecutionOptions::default, |value| {
            value.to_core()
        });
        let (plan, runtime) = self.execution_handles()?;
        let result = py.detach(move || {
            runtime
                .block_on(plan.execute(inputs, options))
                .map_err(crate::error::to_py_err)
        })?;
        PyRunResult::from_inner(py, result)
    }

    #[pyo3(signature = (inputs, *, options = None))]
    fn execute_async<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
        options: Option<PyRef<'py, PyExecutionOptions>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::execute_async_future(slf, py, inputs, options).map(|(future, _)| future)
    }

    #[pyo3(signature = (inputs, *, options = None))]
    fn _execute_async_cancellable<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
        options: Option<PyRef<'py, PyExecutionOptions>>,
    ) -> PyResult<(Bound<'py, PyAny>, Py<PyExecutionCancellation>)> {
        let (future, cancellation) = Self::execute_async_future(slf, py, inputs, options)?;
        Ok((
            future,
            Py::new(py, PyExecutionCancellation::new(cancellation))?,
        ))
    }

    fn snapshot_async<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ExecutionPlanOwner { inner, owner, .. } = Self::owned(slf, py)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let state = inner.snapshot().await.map_err(crate::error::to_py_err)?;
            let encoded = serde_json::to_string(&state)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            let result =
                Python::attach(|py| crate::config::json_to_python(py, &encoded).map(Bound::unbind));
            drop(owner);
            result
        })
    }

    fn restore_async<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        state_json: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = serde_json::from_str::<BTreeMap<String, serde_json::Value>>(state_json)
            .map_err(|error| PyTypeError::new_err(format!("invalid plan state: {error}")))?;
        let ExecutionPlanOwner { inner, owner, .. } = Self::owned(slf, py)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.restore(&state).await.map_err(crate::error::to_py_err);
            drop(owner);
            result
        })
    }

    fn reset_async<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ExecutionPlanOwner { inner, owner, .. } = Self::owned(slf, py)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.reset().await.map_err(crate::error::to_py_err);
            drop(owner);
            result
        })
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3's garbage-collector protocol requires PyVisit by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let state = self.state.read();
        let Some(state) = state.as_ref() else {
            return Ok(());
        };
        for owner in &state.owners {
            visit.call(owner)?;
        }
        for root in &state.roots {
            visit.call(root.object())?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        let state = self.state.write().take();
        drop(state);
    }
}

struct ResultState {
    inner: calc_flow::RunResult,
}

/// A cloned core result paired with the Python wrapper that traverses payloads.
///
/// Long-lived Python-owned native state must retain this whole value and visit
/// [`Self::owner`]. Cloning only the core result would share `PythonPayload`
/// references without giving cyclic GC a corresponding visible owner.
pub(crate) struct RunResultOwner {
    inner: calc_flow::RunResult,
    owner: Py<PyAny>,
}

impl RunResultOwner {
    #[allow(
        dead_code,
        reason = "Task 20 consumes runner results while retaining their Python owner"
    )]
    pub(crate) const fn inner(&self) -> &calc_flow::RunResult {
        &self.inner
    }

    #[allow(
        dead_code,
        reason = "Task 20 reports retained result wrappers from __traverse__"
    )]
    pub(crate) const fn owner(&self) -> &Py<PyAny> {
        &self.owner
    }
}

#[pyclass(name = "RunResult", frozen, module = "calc_flow._native")]
pub(crate) struct PyRunResult {
    state: RwLock<Option<ResultState>>,
}

impl PyRunResult {
    pub(crate) fn from_inner(py: Python<'_>, mut inner: calc_flow::RunResult) -> PyResult<Self> {
        for output in inner.outputs.values_mut() {
            *output = crate::batch::rehome_python_payload(py, output.clone())?;
        }
        Ok(Self {
            state: RwLock::new(Some(ResultState { inner })),
        })
    }

    fn snapshot_inner(&self) -> PyResult<calc_flow::RunResult> {
        let state = self.state.read();
        state
            .as_ref()
            .map(|state| state.inner.clone())
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_RESULT_MESSAGE))
    }

    #[allow(
        dead_code,
        reason = "Task 20 retains Python result wrappers around cloned native runner results"
    )]
    pub(crate) fn owned(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<RunResultOwner> {
        let inner = slf.snapshot_inner()?;
        let owner = slf.into_pyobject(py)?.into_any().unbind();
        Ok(RunResultOwner { inner, owner })
    }
}

#[pymethods]
impl PyRunResult {
    #[getter]
    fn outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.snapshot_inner()?;
        let outputs = PyDict::new(py);
        for (name, batch) in inner.outputs {
            outputs.set_item(name, Py::new(py, PyBatch::from_inner_python(py, batch)?)?)?;
        }
        Ok(outputs)
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.snapshot_inner()?;
        let encoded = serde_json::to_string(&inner.metadata)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        crate::config::json_to_python(py, &encoded)
    }

    #[getter]
    fn node_timings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.snapshot_inner()?;
        let encoded = serde_json::to_string(&inner.node_timings)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        crate::config::json_to_python(py, &encoded)
    }

    #[getter]
    fn datafusion_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.snapshot_inner()?;
        let encoded = serde_json::to_string(&inner.datafusion_metrics)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        crate::config::json_to_python(py, &encoded)
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3's garbage-collector protocol requires PyVisit by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let state = self.state.read();
        let Some(state) = state.as_ref() else {
            return Ok(());
        };
        for output in state.inner.outputs.values() {
            if let Some(root) = crate::batch::python_payload_root(output) {
                visit.call(root)?;
            }
        }
        Ok(())
    }

    fn __clear__(&self) {
        let state = self.state.write().take();
        drop(state);
    }
}

fn extract_inputs(inputs: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, calc_flow::Batch>> {
    let py = inputs.py();
    inputs
        .iter()
        .map(|(name, value)| {
            if !name.is_instance_of::<PyString>() {
                return Err(PyTypeError::new_err("input names must be strings"));
            }
            let name = name.extract::<String>()?;
            let batch = value.extract::<PyRef<'_, PyBatch>>().map_err(|_| {
                PyTypeError::new_err(format!("input {name:?} must contain a calc_flow.Batch"))
            })?;
            let inner = crate::batch::rehome_python_payload(py, batch.clone_inner()?)?;
            Ok((name, inner))
        })
        .collect()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExecutionPlan>()?;
    module.add_class::<PyStreamExecutionPlan>()?;
    module.add_class::<PyRunResult>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        ffi::CString,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use async_trait::async_trait;
    use datafusion::arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use parking_lot::Mutex;
    use pyo3::types::{PyDict, PyList, PyTuple};

    use super::*;

    struct RootedPassthrough {
        _callback: Arc<crate::config::PythonRoot>,
        inputs: Vec<calc_flow::Port>,
        outputs: Vec<calc_flow::Port>,
    }

    impl calc_flow::OperatorMetadata for RootedPassthrough {
        fn name(&self) -> &'static str {
            "passthrough"
        }

        fn input_ports(&self) -> &[calc_flow::Port] {
            &self.inputs
        }

        fn output_ports(&self) -> &[calc_flow::Port] {
            &self.outputs
        }

        fn configuration(&self) -> calc_flow::JsonMap {
            BTreeMap::new()
        }
    }

    #[async_trait]
    impl calc_flow::BatchOperator for RootedPassthrough {
        async fn process(
            &mut self,
            inputs: &BTreeMap<String, calc_flow::Batch>,
            _context: &calc_flow::BatchOperatorContext<'_>,
        ) -> calc_flow::Result<BTreeMap<String, calc_flow::Batch>> {
            Ok(BTreeMap::from([("output".into(), inputs["input"].clone())]))
        }
    }

    struct GatedPassthrough {
        started: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
        input: Vec<calc_flow::Port>,
        output: Vec<calc_flow::Port>,
    }

    struct GatedPythonPassthrough {
        callback: Arc<crate::config::PythonRoot>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
        validations: Arc<AtomicUsize>,
        input: Vec<calc_flow::Port>,
        output: Vec<calc_flow::Port>,
    }

    struct StatefulGatedPassthrough {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
        baselines: Arc<Mutex<Vec<usize>>>,
        state: usize,
        input: Vec<calc_flow::Port>,
        output: Vec<calc_flow::Port>,
    }

    struct CountingPassthrough {
        calls: Arc<AtomicUsize>,
        input: Vec<calc_flow::Port>,
        output: Vec<calc_flow::Port>,
    }

    impl calc_flow::OperatorMetadata for GatedPythonPassthrough {
        fn name(&self) -> &'static str {
            "python_gate"
        }

        fn input_ports(&self) -> &[calc_flow::Port] {
            &self.input
        }

        fn output_ports(&self) -> &[calc_flow::Port] {
            &self.output
        }

        fn configuration(&self) -> calc_flow::JsonMap {
            BTreeMap::new()
        }
    }

    #[async_trait]
    impl calc_flow::BatchOperator for GatedPythonPassthrough {
        async fn process(
            &mut self,
            inputs: &BTreeMap<String, calc_flow::Batch>,
            _context: &calc_flow::BatchOperatorContext<'_>,
        ) -> calc_flow::Result<BTreeMap<String, calc_flow::Batch>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            self.release.notified().await;
            Python::attach(|py| -> PyResult<()> {
                let marker: String = self
                    .callback
                    .object()
                    .call_method0(py, "touch")?
                    .extract(py)?;
                assert_eq!(marker, "callback-alive");
                let payload = crate::batch::python_payload_root(&inputs["input"])
                    .expect("the gated input must remain a Python payload");
                let value: usize = payload.getattr(py, "value")?.extract(py)?;
                assert_eq!(value, 41);
                Ok(())
            })
            .map_err(|error| calc_flow::CalcFlowError::Operator {
                node_id: "python_gate".into(),
                message: error.to_string(),
            })?;
            self.validations.fetch_add(1, Ordering::SeqCst);
            Ok(BTreeMap::from([("output".into(), inputs["input"].clone())]))
        }
    }

    impl calc_flow::OperatorMetadata for GatedPassthrough {
        fn name(&self) -> &'static str {
            "gate"
        }

        fn input_ports(&self) -> &[calc_flow::Port] {
            &self.input
        }

        fn output_ports(&self) -> &[calc_flow::Port] {
            &self.output
        }

        fn configuration(&self) -> calc_flow::JsonMap {
            BTreeMap::new()
        }
    }

    #[async_trait]
    impl calc_flow::BatchOperator for GatedPassthrough {
        async fn process(
            &mut self,
            inputs: &BTreeMap<String, calc_flow::Batch>,
            _context: &calc_flow::BatchOperatorContext<'_>,
        ) -> calc_flow::Result<BTreeMap<String, calc_flow::Batch>> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.started.notify_one();
                std::future::pending::<()>().await;
            }
            Ok(BTreeMap::from([("output".into(), inputs["input"].clone())]))
        }
    }

    impl calc_flow::OperatorMetadata for StatefulGatedPassthrough {
        fn name(&self) -> &'static str {
            "stateful_gate"
        }

        fn input_ports(&self) -> &[calc_flow::Port] {
            &self.input
        }

        fn output_ports(&self) -> &[calc_flow::Port] {
            &self.output
        }

        fn configuration(&self) -> calc_flow::JsonMap {
            BTreeMap::new()
        }
    }

    #[async_trait]
    impl calc_flow::BatchOperator for StatefulGatedPassthrough {
        async fn process(
            &mut self,
            inputs: &BTreeMap<String, calc_flow::Batch>,
            _context: &calc_flow::BatchOperatorContext<'_>,
        ) -> calc_flow::Result<BTreeMap<String, calc_flow::Batch>> {
            self.baselines.lock().push(self.state);
            self.state += 1;
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.started.notify_one();
                self.release.notified().await;
            }
            Ok(BTreeMap::from([("output".into(), inputs["input"].clone())]))
        }

        fn snapshot(&self) -> calc_flow::Result<serde_json::Value> {
            Ok(serde_json::json!(self.state))
        }

        fn restore(&mut self, state: &serde_json::Value) -> calc_flow::Result<()> {
            let state = state
                .as_u64()
                .and_then(|state| usize::try_from(state).ok())
                .ok_or_else(|| calc_flow::CalcFlowError::Format {
                    message: "stateful gate snapshot must be an unsigned integer".into(),
                })?;
            self.state = state;
            Ok(())
        }
    }

    impl calc_flow::OperatorMetadata for CountingPassthrough {
        fn name(&self) -> &'static str {
            "downstream"
        }

        fn input_ports(&self) -> &[calc_flow::Port] {
            &self.input
        }

        fn output_ports(&self) -> &[calc_flow::Port] {
            &self.output
        }

        fn configuration(&self) -> calc_flow::JsonMap {
            BTreeMap::new()
        }
    }

    #[async_trait]
    impl calc_flow::BatchOperator for CountingPassthrough {
        async fn process(
            &mut self,
            inputs: &BTreeMap<String, calc_flow::Batch>,
            _context: &calc_flow::BatchOperatorContext<'_>,
        ) -> calc_flow::Result<BTreeMap<String, calc_flow::Batch>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(BTreeMap::from([("output".into(), inputs["input"].clone())]))
        }
    }

    #[pyclass]
    struct StartSignal {
        started: Arc<tokio::sync::Notify>,
    }

    #[pyclass]
    struct ReleaseSignal {
        release: Arc<tokio::sync::Notify>,
    }

    #[pyclass]
    struct DeadlineSignal {
        deadline: chrono::DateTime<chrono::Utc>,
    }

    #[pymethods]
    impl ReleaseSignal {
        fn fire(&self) {
            self.release.notify_one();
        }
    }

    #[pymethods]
    impl StartSignal {
        fn wait<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let started = Arc::clone(&self.started);
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                started.notified().await;
                Ok(())
            })
        }
    }

    #[pymethods]
    impl DeadlineSignal {
        fn wait_until_crossed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let remaining = self
                .deadline
                .signed_duration_since(chrono::Utc::now())
                .to_std()
                .unwrap_or_default();
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                tokio::time::sleep(remaining + std::time::Duration::from_millis(10)).await;
                Ok(())
            })
        }
    }

    fn plan() -> PyExecutionPlan {
        let udfs = calc_flow::UdfRegistry::new().snapshot();
        let plan = calc_flow::PipelineBuilder::new("pipeline")
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
            .compile_batch(&udfs)
            .unwrap();
        PyExecutionPlan::new(
            Arc::new(plan),
            Arc::new(tokio::runtime::Runtime::new().unwrap()),
            Vec::new(),
        )
    }

    fn rooted_plan(
        root: Arc<crate::config::PythonRoot>,
        kind: calc_flow::BatchKind,
    ) -> Arc<calc_flow::BatchExecutionPlan> {
        let input = calc_flow::Port::new("input", kind, true, None).unwrap();
        let output = calc_flow::Port::new("output", kind, true, None).unwrap();
        Arc::new(
            calc_flow::PipelineBuilder::new("rooted")
                .unwrap()
                .add_node(
                    "passthrough",
                    Box::new(RootedPassthrough {
                        _callback: root,
                        inputs: vec![input],
                        outputs: vec![output],
                    }) as Box<dyn calc_flow::BatchOperator>,
                )
                .unwrap()
                .compile_batch(&calc_flow::UdfRegistry::new().snapshot())
                .unwrap(),
        )
    }

    fn gated_python_plan(
        root: Arc<crate::config::PythonRoot>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
        validations: Arc<AtomicUsize>,
    ) -> Arc<calc_flow::BatchExecutionPlan> {
        let input = calc_flow::Port::new("input", calc_flow::BatchKind::Array, true, None).unwrap();
        let output =
            calc_flow::Port::new("output", calc_flow::BatchKind::Array, true, None).unwrap();
        Arc::new(
            calc_flow::PipelineBuilder::new("python_gate")
                .unwrap()
                .add_node(
                    "python_gate",
                    Box::new(GatedPythonPassthrough {
                        callback: root,
                        started,
                        release,
                        calls,
                        validations,
                        input: vec![input],
                        output: vec![output],
                    }) as Box<dyn calc_flow::BatchOperator>,
                )
                .unwrap()
                .compile_batch(&calc_flow::UdfRegistry::new().snapshot())
                .unwrap(),
        )
    }

    fn stateful_cancellation_plan(
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
        baselines: Arc<Mutex<Vec<usize>>>,
        downstream_calls: Arc<AtomicUsize>,
    ) -> Arc<calc_flow::BatchExecutionPlan> {
        let gate_input =
            calc_flow::Port::new("input", calc_flow::BatchKind::Table, true, None).unwrap();
        let gate_output =
            calc_flow::Port::new("output", calc_flow::BatchKind::Table, true, None).unwrap();
        let downstream_input =
            calc_flow::Port::new("input", calc_flow::BatchKind::Table, true, None).unwrap();
        let downstream_output =
            calc_flow::Port::new("output", calc_flow::BatchKind::Table, true, None).unwrap();
        Arc::new(
            calc_flow::PipelineBuilder::new("stateful_cancellation")
                .unwrap()
                .add_node(
                    "gate",
                    Box::new(StatefulGatedPassthrough {
                        started,
                        release,
                        calls,
                        baselines,
                        state: 0,
                        input: vec![gate_input],
                        output: vec![gate_output],
                    }) as Box<dyn calc_flow::BatchOperator>,
                )
                .unwrap()
                .add_node(
                    "downstream",
                    Box::new(CountingPassthrough {
                        calls: downstream_calls,
                        input: vec![downstream_input],
                        output: vec![downstream_output],
                    }) as Box<dyn calc_flow::BatchOperator>,
                )
                .unwrap()
                .connect(calc_flow::Edge::new(
                    calc_flow::PortEndpoint::new("gate", "output").unwrap(),
                    calc_flow::PortEndpoint::new("downstream", "input").unwrap(),
                ))
                .unwrap()
                .compile_batch(&calc_flow::UdfRegistry::new().snapshot())
                .unwrap(),
        )
    }

    fn batch() -> PyBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let values = Arc::new(Int64Array::from(vec![1_i64]));
        let record_batch = RecordBatch::try_new(schema, vec![values]).unwrap();
        PyBatch::from_inner(
            calc_flow::Batch::table(vec![record_batch], calc_flow::BatchMetadata::default())
                .unwrap(),
        )
    }

    #[test]
    fn input_extraction_rejects_non_string_names_and_non_batches() {
        Python::initialize();
        Python::attach(|py| {
            let inputs = PyDict::new(py);
            inputs.set_item(1, "batch").unwrap();
            assert!(
                extract_inputs(&inputs)
                    .unwrap_err()
                    .is_instance_of::<PyTypeError>(py)
            );

            inputs.clear();
            inputs.set_item("input", "batch").unwrap();
            assert!(
                extract_inputs(&inputs)
                    .unwrap_err()
                    .is_instance_of::<PyTypeError>(py)
            );
        });
    }

    #[test]
    fn input_extraction_rehomes_python_payloads_for_opaque_execution() {
        Python::initialize();
        Python::attach(|py| {
            let metadata = PyDict::new(py);
            let payload = PyDict::new(py);
            let batch = py
                .get_type::<PyBatch>()
                .call_method1("_from_external", (&payload, "test", 1, &metadata))
                .unwrap();
            let original = batch
                .extract::<PyRef<'_, PyBatch>>()
                .unwrap()
                .clone_inner()
                .unwrap();
            let inputs = PyDict::new(py);
            inputs.set_item("input", &batch).unwrap();

            let extracted = extract_inputs(&inputs).unwrap();

            assert!(!Arc::ptr_eq(
                original.external_payload().unwrap(),
                extracted["input"].external_payload().unwrap(),
            ));
        });
    }

    #[test]
    fn execution_plan_deduplicates_explicit_python_roots() {
        Python::initialize();
        Python::attach(|py| {
            let root = Arc::new(crate::config::PythonRoot::new(py.None()));
            let plan = PyExecutionPlan::new(
                rooted_plan(Arc::clone(&root), calc_flow::BatchKind::Array),
                Arc::new(tokio::runtime::Runtime::new().unwrap()),
                vec![Arc::clone(&root), root],
            );

            let state = plan.state.read();
            assert_eq!(state.as_ref().unwrap().roots.len(), 1);
        });
    }

    #[test]
    fn sync_execution_and_result_getters_expose_defensive_values() {
        Python::initialize();
        Python::attach(|py| {
            let inputs = PyDict::new(py);
            inputs
                .set_item("input", Py::new(py, batch()).unwrap())
                .unwrap();
            let result = plan().execute(py, &inputs, None).unwrap();

            let outputs = result.outputs(py).unwrap();
            assert_eq!(outputs.len(), 1);
            outputs.clear();
            assert_eq!(result.outputs(py).unwrap().len(), 1);

            let metadata = result.metadata(py).unwrap();
            assert_eq!(
                metadata
                    .get_item("pipeline_name")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "pipeline"
            );
            let timings = result.node_timings(py).unwrap();
            assert!(timings.get_item("calc").is_ok());
            let metrics = result.datafusion_metrics(py).unwrap();
            assert_eq!(metrics.cast::<PyList>().unwrap().len(), 1);
        });
    }

    #[test]
    fn async_execution_uses_the_python_event_loop() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item("plan", Py::new(py, plan()).unwrap())
                .unwrap();
            locals
                .set_item("batch", Py::new(py, batch()).unwrap())
                .unwrap();
            py.run(
                &CString::new(
                    "import asyncio\nasync def run():\n    return await plan.execute_async({'input': batch})\nresult = asyncio.run(run())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
            let result = locals.get_item("result").unwrap().unwrap();
            let metadata = result.getattr("metadata").unwrap();
            assert_eq!(
                metadata
                    .get_item("pipeline_name")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "pipeline"
            );
        });
    }

    #[test]
    fn execution_plan_exposes_core_name_and_fingerprint() {
        let plan = plan();
        assert_eq!(plan.name().unwrap(), "pipeline");
        assert!(!plan.fingerprint().unwrap().is_empty());
    }

    #[test]
    fn active_async_execution_owns_python_state_through_cancel_and_recovery() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    "class Callback:\n    def __init__(self):\n        self.calls = 0\n    def touch(self):\n        self.calls += 1\n        return 'callback-alive'\nclass Payload:\n    def __init__(self):\n        self.value = 41\ncallback = Callback()\npayload = Payload()"
                ),
                Some(&locals),
                None,
            )
            .unwrap();
            let callback = locals.get_item("callback").unwrap().unwrap();
            let payload = locals.get_item("payload").unwrap().unwrap();
            let root = Arc::new(crate::config::PythonRoot::new(callback.clone().unbind()));
            let started = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(tokio::sync::Notify::new());
            let calls = Arc::new(AtomicUsize::new(0));
            let validations = Arc::new(AtomicUsize::new(0));
            let plan = PyExecutionPlan::new(
                gated_python_plan(
                    Arc::clone(&root),
                    Arc::clone(&started),
                    Arc::clone(&release),
                    Arc::clone(&calls),
                    Arc::clone(&validations),
                ),
                Arc::new(tokio::runtime::Runtime::new().unwrap()),
                vec![root],
            );
            let metadata = PyDict::new(py);
            let batch = py
                .get_type::<PyBatch>()
                .call_method1("_from_external", (&payload, "test", 1, &metadata))
                .unwrap();
            locals.set_item("plan", Py::new(py, plan).unwrap()).unwrap();
            locals.set_item("batch", &batch).unwrap();
            locals
                .set_item("started", Py::new(py, StartSignal { started }).unwrap())
                .unwrap();
            locals
                .set_item("release", Py::new(py, ReleaseSignal { release }).unwrap())
                .unwrap();
            drop(callback);
            drop(payload);
            drop(batch);

            py.run(
                &CString::new(
                    "import asyncio, gc, weakref\nasync def run():\n    owned_plan = plan\n    owned_batch = batch\n    owned_callback = callback\n    owned_payload = payload\n    owned_callback.owner = owned_plan\n    owned_payload.owner = owned_batch\n    callback_ref = weakref.ref(owned_callback)\n    payload_ref = weakref.ref(owned_payload)\n    task = asyncio.ensure_future(owned_plan.execute_async({'input': owned_batch}))\n    globals().pop('plan')\n    globals().pop('batch')\n    globals().pop('callback')\n    globals().pop('payload')\n    del owned_plan, owned_batch, owned_callback, owned_payload\n    await started.wait()\n    gc.collect()\n    assert callback_ref() is not None\n    assert payload_ref() is not None\n    task.cancel()\n    try:\n        await task\n    except asyncio.CancelledError:\n        pass\n    recovered_callback = callback_ref()\n    recovered_payload = payload_ref()\n    assert recovered_callback is not None\n    assert recovered_payload is not None\n    recovered_plan = recovered_callback.owner\n    recovered_batch = recovered_payload.owner\n    recovered = asyncio.ensure_future(recovered_plan.execute_async({'input': recovered_batch}))\n    del recovered_plan, recovered_batch, recovered_callback, recovered_payload\n    await started.wait()\n    gc.collect()\n    assert callback_ref() is not None\n    assert payload_ref() is not None\n    release.fire()\n    return await recovered\nresult = asyncio.run(run())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();

            assert_eq!(calls.load(Ordering::SeqCst), 2);
            assert_eq!(validations.load(Ordering::SeqCst), 1);
            assert_eq!(
                locals
                    .get_item("result")
                    .unwrap()
                    .unwrap()
                    .getattr("outputs")
                    .unwrap()
                    .len()
                    .unwrap(),
                1
            );
        });
    }

    #[test]
    fn private_async_cancellation_rolls_back_state_and_skips_downstream() {
        Python::initialize();
        Python::attach(|py| {
            let started = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(tokio::sync::Notify::new());
            let calls = Arc::new(AtomicUsize::new(0));
            let baselines = Arc::new(Mutex::new(Vec::new()));
            let downstream_calls = Arc::new(AtomicUsize::new(0));
            let plan = PyExecutionPlan::new(
                stateful_cancellation_plan(
                    Arc::clone(&started),
                    Arc::clone(&release),
                    Arc::clone(&calls),
                    Arc::clone(&baselines),
                    Arc::clone(&downstream_calls),
                ),
                Arc::new(tokio::runtime::Runtime::new().unwrap()),
                Vec::new(),
            );
            let locals = PyDict::new(py);
            locals.set_item("plan", Py::new(py, plan).unwrap()).unwrap();
            locals
                .set_item("batch", Py::new(py, batch()).unwrap())
                .unwrap();
            locals
                .set_item("started", Py::new(py, StartSignal { started }).unwrap())
                .unwrap();
            locals
                .set_item("release", Py::new(py, ReleaseSignal { release }).unwrap())
                .unwrap();

            py.run(
                pyo3::ffi::c_str!(
                    "import asyncio\nasync def run():\n    execution, cancellation = plan._execute_async_cancellable({'input': batch})\n    await started.wait()\n    cancellation.cancel()\n    release.fire()\n    try:\n        await execution\n    except Exception as error:\n        assert type(error).__name__ == 'CancelledError'\n    else:\n        raise AssertionError('cancelled native execution unexpectedly succeeded')\n    rolled_back = await plan.snapshot_async()\n    recovered = await plan.execute_async({'input': batch})\n    after_recovery = await plan.snapshot_async()\n    return rolled_back, after_recovery, recovered\nstates = asyncio.run(run())"
                ),
                Some(&locals),
                None,
            )
            .unwrap();

            let states = locals
                .get_item("states")
                .unwrap()
                .unwrap()
                .cast_into::<PyTuple>()
                .unwrap();
            let rolled_back = states.get_item(0).unwrap().cast_into::<PyDict>().unwrap();
            assert_eq!(
                rolled_back
                    .get_item("gate")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                0
            );
            assert!(
                rolled_back
                    .get_item("downstream")
                    .unwrap()
                    .unwrap()
                    .is_none()
            );
            let after_recovery = states.get_item(1).unwrap().cast_into::<PyDict>().unwrap();
            assert_eq!(
                after_recovery
                    .get_item("gate")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
            assert_eq!(
                states
                    .get_item(2)
                    .unwrap()
                    .getattr("outputs")
                    .unwrap()
                    .len()
                    .unwrap(),
                1
            );
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            assert_eq!(*baselines.lock(), vec![0, 0]);
            assert_eq!(downstream_calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn async_deadline_crossing_rolls_back_state_and_skips_downstream() {
        Python::initialize();
        Python::attach(|py| {
            let started = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(tokio::sync::Notify::new());
            let calls = Arc::new(AtomicUsize::new(0));
            let baselines = Arc::new(Mutex::new(Vec::new()));
            let downstream_calls = Arc::new(AtomicUsize::new(0));
            let deadline = chrono::Utc::now() + chrono::Duration::seconds(1);
            let plan = PyExecutionPlan::new(
                stateful_cancellation_plan(
                    Arc::clone(&started),
                    Arc::clone(&release),
                    Arc::clone(&calls),
                    Arc::clone(&baselines),
                    Arc::clone(&downstream_calls),
                ),
                Arc::new(tokio::runtime::Runtime::new().unwrap()),
                Vec::new(),
            );
            let locals = PyDict::new(py);
            locals.set_item("plan", Py::new(py, plan).unwrap()).unwrap();
            locals
                .set_item("batch", Py::new(py, batch()).unwrap())
                .unwrap();
            locals
                .set_item(
                    "options",
                    Py::new(
                        py,
                        PyExecutionOptions::for_test(BTreeMap::new(), Some(deadline)),
                    )
                    .unwrap(),
                )
                .unwrap();
            locals
                .set_item("started", Py::new(py, StartSignal { started }).unwrap())
                .unwrap();
            locals
                .set_item("release", Py::new(py, ReleaseSignal { release }).unwrap())
                .unwrap();
            let deadline_signal = Py::new(py, DeadlineSignal { deadline }).unwrap();
            locals.set_item("deadline", deadline_signal).unwrap();

            py.run(
                pyo3::ffi::c_str!(
                    "import asyncio\nasync def run():\n    execution = plan.execute_async({'input': batch}, options=options)\n    await started.wait()\n    await deadline.wait_until_crossed()\n    release.fire()\n    try:\n        await execution\n    except Exception as error:\n        assert type(error).__name__ == 'CancelledError'\n    else:\n        raise AssertionError('expired execution unexpectedly succeeded')\n    rolled_back = await plan.snapshot_async()\n    recovered = await plan.execute_async({'input': batch})\n    after_recovery = await plan.snapshot_async()\n    return rolled_back, after_recovery, recovered\nstates = asyncio.run(run())"
                ),
                Some(&locals),
                None,
            )
            .unwrap();

            let states = locals
                .get_item("states")
                .unwrap()
                .unwrap()
                .cast_into::<PyTuple>()
                .unwrap();
            let rolled_back = states.get_item(0).unwrap().cast_into::<PyDict>().unwrap();
            assert_eq!(
                rolled_back
                    .get_item("gate")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                0
            );
            assert!(
                rolled_back
                    .get_item("downstream")
                    .unwrap()
                    .unwrap()
                    .is_none()
            );
            let after_recovery = states.get_item(1).unwrap().cast_into::<PyDict>().unwrap();
            assert_eq!(
                after_recovery
                    .get_item("gate")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
            assert_eq!(
                states
                    .get_item(2)
                    .unwrap()
                    .getattr("outputs")
                    .unwrap()
                    .len()
                    .unwrap(),
                1
            );
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            assert_eq!(*baselines.lock(), vec![0, 0]);
            assert_eq!(downstream_calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn execution_plan_gc_collects_cycles_hidden_in_core_operators() {
        Python::initialize();
        Python::attach(|py| {
            let weakref = py.import("weakref").unwrap().getattr("ref").unwrap();
            let runtime = Arc::new(tokio::runtime::Runtime::new().unwrap());
            let mut references = Vec::new();

            for _ in 0..1_000 {
                let holder = py
                    .eval(pyo3::ffi::c_str!("type('Holder', (), {})()"), None, None)
                    .unwrap();
                let root = Arc::new(crate::config::PythonRoot::new(holder.clone().unbind()));
                let plan = Py::new(
                    py,
                    PyExecutionPlan::new(
                        rooted_plan(Arc::clone(&root), calc_flow::BatchKind::Array),
                        Arc::clone(&runtime),
                        vec![root],
                    ),
                )
                .unwrap();
                holder.setattr("owner", plan.bind(py)).unwrap();
                references.push(weakref.call1((&holder,)).unwrap().unbind());
            }

            py.import("gc")
                .unwrap()
                .getattr("collect")
                .unwrap()
                .call0()
                .unwrap();
            let alive = references
                .iter()
                .filter(|reference| !reference.call0(py).unwrap().is_none(py))
                .count();
            assert_eq!(alive, 0);
        });
    }

    #[test]
    fn run_result_gc_collects_cycles_hidden_in_external_outputs() {
        Python::initialize();
        Python::attach(|py| {
            let hidden = Arc::new(crate::config::PythonRoot::new(
                PyList::empty(py).unbind().into_any(),
            ));
            let native_plan = PyExecutionPlan::new(
                rooted_plan(Arc::clone(&hidden), calc_flow::BatchKind::Array),
                Arc::new(tokio::runtime::Runtime::new().unwrap()),
                vec![hidden],
            );
            let weakref = py.import("weakref").unwrap().getattr("ref").unwrap();
            let metadata = PyDict::new(py);
            let mut references = Vec::new();
            let mut retained_inputs = Vec::new();

            for _ in 0..100 {
                let holder = py
                    .eval(pyo3::ffi::c_str!("type('Holder', (), {})()"), None, None)
                    .unwrap();
                let batch = py
                    .get_type::<PyBatch>()
                    .call_method1("_from_external", (&holder, "test", 1, &metadata))
                    .unwrap();
                let inputs = PyDict::new(py);
                inputs.set_item("input", &batch).unwrap();
                let result = Py::new(py, native_plan.execute(py, &inputs, None).unwrap()).unwrap();
                let outputs = result.bind(py).getattr("outputs").unwrap();
                let output = outputs.get_item("output").unwrap();
                holder.setattr("input", &batch).unwrap();
                holder.setattr("owner", result.bind(py)).unwrap();
                holder.setattr("output", output).unwrap();
                references.push(weakref.call1((&holder,)).unwrap().unbind());
                retained_inputs.push(batch.unbind());
            }

            py.import("gc")
                .unwrap()
                .getattr("collect")
                .unwrap()
                .call0()
                .unwrap();
            assert!(
                references
                    .iter()
                    .all(|reference| !reference.call0(py).unwrap().is_none(py))
            );
            assert!(retained_inputs.iter().all(|batch| {
                batch
                    .bind(py)
                    .getattr("kind")
                    .unwrap()
                    .extract::<String>()
                    .unwrap()
                    == "array"
            }));

            drop(retained_inputs);
            py.import("gc")
                .unwrap()
                .getattr("collect")
                .unwrap()
                .call0()
                .unwrap();
            let alive = references
                .iter()
                .filter(|reference| !reference.call0(py).unwrap().is_none(py))
                .count();
            assert_eq!(alive, 0);
        });
    }

    #[test]
    fn cleared_plan_and_result_reject_access() {
        Python::initialize();
        Python::attach(|py| {
            let plan = Py::new(py, plan()).unwrap();
            let inputs = PyDict::new(py);
            inputs
                .set_item("input", Py::new(py, batch()).unwrap())
                .unwrap();
            let result = plan.borrow(py).execute(py, &inputs, None).unwrap();

            result.__clear__();
            let result_errors = [
                result.outputs(py).unwrap_err(),
                result.metadata(py).unwrap_err(),
                result.node_timings(py).unwrap_err(),
                result.datafusion_metrics(py).unwrap_err(),
                match result.snapshot_inner() {
                    Ok(_) => panic!("a cleared result must not expose its core value"),
                    Err(error) => error,
                },
            ];
            assert!(
                result_errors
                    .iter()
                    .all(|error| { error.value(py).to_string() == CLEARED_RESULT_MESSAGE })
            );
            result.__clear__();

            plan.borrow(py).__clear__();
            let plan_errors = [
                match plan.borrow(py).execute(py, &inputs, None) {
                    Ok(_) => panic!("a cleared plan must reject blocking execution"),
                    Err(error) => error,
                },
                plan.bind(py)
                    .call_method1("execute_async", (&inputs,))
                    .unwrap_err(),
                match plan.borrow(py).execution_handles() {
                    Ok(_) => panic!("a cleared plan must not expose execution handles"),
                    Err(error) => error,
                },
            ];
            assert!(
                plan_errors
                    .iter()
                    .all(|error| error.value(py).to_string() == CLEARED_PLAN_MESSAGE)
            );
            plan.borrow(py).__clear__();
        });
    }

    #[test]
    fn async_cancellation_occurs_after_the_authenticated_marker() {
        Python::initialize();
        Python::attach(|py| {
            let started = Arc::new(tokio::sync::Notify::new());
            let calls = Arc::new(AtomicUsize::new(0));
            let input =
                calc_flow::Port::new("input", calc_flow::BatchKind::Table, true, None).unwrap();
            let output =
                calc_flow::Port::new("output", calc_flow::BatchKind::Table, true, None).unwrap();
            let core = calc_flow::PipelineBuilder::new("cancel")
                .unwrap()
                .add_node(
                    "gate",
                    Box::new(GatedPassthrough {
                        started: Arc::clone(&started),
                        calls: Arc::clone(&calls),
                        input: vec![input],
                        output: vec![output],
                    }) as Box<dyn calc_flow::BatchOperator>,
                )
                .unwrap()
                .compile_batch(&calc_flow::UdfRegistry::new().snapshot())
                .unwrap();
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "plan",
                    Py::new(
                        py,
                        PyExecutionPlan::new(
                            Arc::new(core),
                            Arc::new(tokio::runtime::Runtime::new().unwrap()),
                            Vec::new(),
                        ),
                    )
                    .unwrap(),
                )
                .unwrap();
            locals
                .set_item("batch", Py::new(py, batch()).unwrap())
                .unwrap();
            locals
                .set_item("started", Py::new(py, StartSignal { started }).unwrap())
                .unwrap();
            py.run(
                &CString::new(
                    "import asyncio\nasync def run():\n    task = asyncio.ensure_future(plan.execute_async({'input': batch}))\n    await started.wait()\n    task.cancel()\n    try:\n        await task\n    except asyncio.CancelledError:\n        pass\n    return await plan.execute_async({'input': batch})\nresult = asyncio.run(run())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();

            assert_eq!(calls.load(Ordering::SeqCst), 2);
            let result = locals.get_item("result").unwrap().unwrap();
            assert_eq!(result.getattr("outputs").unwrap().len().unwrap(), 1);
        });
    }
}
