use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyAny, PyDict, PyString},
};

use crate::batch::PyBatch;

const CLEARED_PLAN_MESSAGE: &str = "ExecutionPlan has been cleared by garbage collection";
const CLEARED_RESULT_MESSAGE: &str = "RunResult has been cleared by garbage collection";

struct PlanState {
    inner: Arc<calc_flow::ExecutionPlan>,
    tokio: Arc<tokio::runtime::Runtime>,
    owners: Vec<Py<PyAny>>,
    roots: Vec<Arc<crate::config::PythonRoot>>,
}

#[pyclass(name = "ExecutionPlan", frozen, module = "calc_flow._native")]
pub(crate) struct PyExecutionPlan {
    state: RwLock<Option<PlanState>>,
}

impl PyExecutionPlan {
    #[allow(
        dead_code,
        reason = "Task 20 can wrap a core plan while preserving explicit Python GC roots"
    )]
    pub(crate) fn new(
        inner: Arc<calc_flow::ExecutionPlan>,
        tokio: Arc<tokio::runtime::Runtime>,
        roots: Vec<Arc<crate::config::PythonRoot>>,
    ) -> Self {
        Self {
            state: RwLock::new(Some(PlanState {
                inner,
                tokio,
                owners: Vec::new(),
                roots,
            })),
        }
    }

    pub(crate) fn new_with_owner(
        inner: Arc<calc_flow::ExecutionPlan>,
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
    ) -> PyResult<(Arc<calc_flow::ExecutionPlan>, Arc<tokio::runtime::Runtime>)> {
        let state = self.state.read();
        let state = state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_PLAN_MESSAGE))?;
        Ok((Arc::clone(&state.inner), Arc::clone(&state.tokio)))
    }

    pub(crate) fn clone_inner(&self) -> PyResult<Arc<calc_flow::ExecutionPlan>> {
        let state = self.state.read();
        state
            .as_ref()
            .map(|state| Arc::clone(&state.inner))
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_PLAN_MESSAGE))
    }

    #[allow(
        dead_code,
        reason = "Task 20 reuses the plan runtime for blocking runner facades"
    )]
    pub(crate) fn clone_tokio(&self) -> PyResult<Arc<tokio::runtime::Runtime>> {
        let state = self.state.read();
        state
            .as_ref()
            .map(|state| Arc::clone(&state.tokio))
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_PLAN_MESSAGE))
    }
}

#[pymethods]
impl PyExecutionPlan {
    fn execute(&self, py: Python<'_>, inputs: &Bound<'_, PyDict>) -> PyResult<PyRunResult> {
        let inputs = extract_inputs(inputs)?;
        let (plan, runtime) = self.execution_handles()?;
        let result = py.detach(move || {
            runtime
                .block_on(plan.execute(inputs, calc_flow::ExecutionOptions::default()))
                .map_err(crate::error::to_py_err)
        })?;
        PyRunResult::from_inner(py, result)
    }

    fn execute_async<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inputs = extract_inputs(inputs)?;
        let plan = self.clone_inner()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = plan
                .execute(inputs, calc_flow::ExecutionOptions::default())
                .await
                .map_err(crate::error::to_py_err)?;
            Python::attach(|py| PyRunResult::from_inner(py, result))
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

#[pyclass(name = "RunResult", frozen, module = "calc_flow._native")]
pub(crate) struct PyRunResult {
    state: RwLock<Option<ResultState>>,
}

impl PyRunResult {
    fn from_inner(py: Python<'_>, mut inner: calc_flow::RunResult) -> PyResult<Self> {
        for output in inner.outputs.values_mut() {
            *output = crate::batch::rehome_python_payload(py, output.clone())?;
        }
        Ok(Self {
            state: RwLock::new(Some(ResultState { inner })),
        })
    }

    pub(crate) fn clone_inner(&self) -> PyResult<calc_flow::RunResult> {
        let state = self.state.read();
        state
            .as_ref()
            .map(|state| state.inner.clone())
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_RESULT_MESSAGE))
    }
}

#[pymethods]
impl PyRunResult {
    #[getter]
    fn outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.clone_inner()?;
        let outputs = PyDict::new(py);
        for (name, batch) in inner.outputs {
            outputs.set_item(name, Py::new(py, PyBatch::from_inner_python(py, batch)?)?)?;
        }
        Ok(outputs)
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.clone_inner()?;
        let encoded = serde_json::to_string(&inner.metadata)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        crate::config::json_to_python(py, &encoded)
    }

    #[getter]
    fn node_timings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.clone_inner()?;
        let encoded = serde_json::to_string(&inner.node_timings)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        crate::config::json_to_python(py, &encoded)
    }

    #[getter]
    fn datafusion_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.clone_inner()?;
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
            Ok((name, batch.clone_inner()?))
        })
        .collect()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExecutionPlan>()?;
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
    use pyo3::types::{PyDict, PyList};

    use super::*;

    struct RootedPassthrough {
        _callback: Arc<crate::config::PythonRoot>,
        inputs: Vec<calc_flow::Port>,
        outputs: Vec<calc_flow::Port>,
    }

    #[async_trait]
    impl calc_flow::Operator for RootedPassthrough {
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

        async fn process(
            &mut self,
            inputs: &BTreeMap<String, calc_flow::Batch>,
            _context: &calc_flow::OperatorContext<'_>,
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

    #[async_trait]
    impl calc_flow::Operator for GatedPassthrough {
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

        async fn process(
            &mut self,
            inputs: &BTreeMap<String, calc_flow::Batch>,
            _context: &calc_flow::OperatorContext<'_>,
        ) -> calc_flow::Result<BTreeMap<String, calc_flow::Batch>> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.started.notify_one();
                std::future::pending::<()>().await;
            }
            Ok(BTreeMap::from([("output".into(), inputs["input"].clone())]))
        }
    }

    #[pyclass]
    struct StartSignal {
        started: Arc<tokio::sync::Notify>,
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
            .compile(&udfs)
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
    ) -> Arc<calc_flow::ExecutionPlan> {
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
                    }),
                )
                .unwrap()
                .compile(&calc_flow::UdfRegistry::new().snapshot())
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
    fn sync_execution_and_result_getters_expose_defensive_values() {
        Python::initialize();
        Python::attach(|py| {
            let inputs = PyDict::new(py);
            inputs
                .set_item("input", Py::new(py, batch()).unwrap())
                .unwrap();
            let result = plan().execute(py, &inputs).unwrap();

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
                let result = Py::new(py, native_plan.execute(py, &inputs).unwrap()).unwrap();
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
            let plan = plan();
            let inputs = PyDict::new(py);
            inputs
                .set_item("input", Py::new(py, batch()).unwrap())
                .unwrap();
            let result = plan.execute(py, &inputs).unwrap();

            result.__clear__();
            let result_errors = [
                result.outputs(py).unwrap_err(),
                result.metadata(py).unwrap_err(),
                result.node_timings(py).unwrap_err(),
                result.datafusion_metrics(py).unwrap_err(),
                match result.clone_inner() {
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

            plan.__clear__();
            let plan_errors = [
                match plan.execute(py, &inputs) {
                    Ok(_) => panic!("a cleared plan must reject blocking execution"),
                    Err(error) => error,
                },
                plan.execute_async(py, &inputs).unwrap_err(),
                match plan.clone_inner() {
                    Ok(_) => panic!("a cleared plan must not expose its core plan"),
                    Err(error) => error,
                },
                match plan.clone_tokio() {
                    Ok(_) => panic!("a cleared plan must not expose its Tokio runtime"),
                    Err(error) => error,
                },
            ];
            assert!(
                plan_errors
                    .iter()
                    .all(|error| error.value(py).to_string() == CLEARED_PLAN_MESSAGE)
            );
            plan.__clear__();
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
                    }),
                )
                .unwrap()
                .compile(&calc_flow::UdfRegistry::new().snapshot())
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
