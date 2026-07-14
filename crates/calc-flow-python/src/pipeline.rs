use std::{collections::BTreeMap, sync::Arc};

use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyDict, PyString},
};

use crate::batch::PyBatch;

#[pyclass(name = "ExecutionPlan", frozen, module = "calc_flow._native")]
pub(crate) struct PyExecutionPlan {
    pub(crate) inner: Arc<calc_flow::ExecutionPlan>,
    pub(crate) tokio: Arc<tokio::runtime::Runtime>,
}

impl PyExecutionPlan {
    pub(crate) const fn new(
        inner: Arc<calc_flow::ExecutionPlan>,
        tokio: Arc<tokio::runtime::Runtime>,
    ) -> Self {
        Self { inner, tokio }
    }
}

#[pymethods]
impl PyExecutionPlan {
    fn execute(&self, py: Python<'_>, inputs: &Bound<'_, PyDict>) -> PyResult<PyRunResult> {
        let inputs = extract_inputs(inputs)?;
        let plan = Arc::clone(&self.inner);
        let runtime = Arc::clone(&self.tokio);
        py.detach(move || {
            runtime
                .block_on(plan.execute(inputs, calc_flow::ExecutionOptions::default()))
                .map(PyRunResult::from)
                .map_err(crate::error::to_py_err)
        })
    }

    fn execute_async<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inputs = extract_inputs(inputs)?;
        let plan = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            plan.execute(inputs, calc_flow::ExecutionOptions::default())
                .await
                .map(PyRunResult::from)
                .map_err(crate::error::to_py_err)
        })
    }
}

#[pyclass(name = "RunResult", frozen, module = "calc_flow._native")]
pub(crate) struct PyRunResult {
    inner: calc_flow::RunResult,
}

impl From<calc_flow::RunResult> for PyRunResult {
    fn from(inner: calc_flow::RunResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRunResult {
    #[getter]
    fn outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let outputs = PyDict::new(py);
        for (name, batch) in &self.inner.outputs {
            outputs.set_item(name, Py::new(py, PyBatch::from_inner(batch.clone()))?)?;
        }
        Ok(outputs)
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let encoded = serde_json::to_string(&self.inner.metadata)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        crate::config::json_to_python(py, &encoded)
    }

    #[getter]
    fn node_timings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let encoded = serde_json::to_string(&self.inner.node_timings)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        crate::config::json_to_python(py, &encoded)
    }

    #[getter]
    fn datafusion_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let encoded = serde_json::to_string(&self.inner.datafusion_metrics)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        crate::config::json_to_python(py, &encoded)
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
    use std::{ffi::CString, sync::Arc};

    use datafusion::arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use pyo3::types::{PyDict, PyList};

    use super::*;

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
}
