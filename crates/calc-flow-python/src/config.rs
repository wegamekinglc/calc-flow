use std::sync::Arc;

use parking_lot::RwLock;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::pipeline::PyExecutionPlan;

#[pyclass(name = "Runtime", frozen, module = "calc_flow._native")]
pub(crate) struct PyRuntime {
    pub(crate) providers: Arc<calc_flow::ProviderRegistry>,
    pub(crate) udfs: Arc<RwLock<calc_flow::UdfRegistry>>,
    pub(crate) tokio: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> PyResult<Self> {
        let tokio = tokio::runtime::Runtime::new()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            providers: Arc::new(calc_flow::ProviderRegistry::default()),
            udfs: Arc::new(RwLock::new(calc_flow::UdfRegistry::new())),
            tokio: Arc::new(tokio),
        })
    }

    fn compile_project(&self, project_json: &str) -> PyResult<PyExecutionPlan> {
        let project = calc_flow::import_project_json(project_json.as_bytes())
            .map_err(crate::error::to_py_err)?;
        let udfs = self.udfs.read().snapshot();
        let plan = calc_flow::compile_project(&project, &self.providers, &udfs)
            .map_err(crate::error::to_py_err)?;
        Ok(PyExecutionPlan::new(
            Arc::new(plan),
            Arc::clone(&self.tokio),
        ))
    }
}

#[pyfunction]
pub(crate) fn project_json_schema() -> PyResult<String> {
    let schema = calc_flow::project_json_schema().map_err(crate::error::to_py_err)?;
    calc_flow::canonical_json(&schema).map_err(crate::error::to_py_err)
}

#[pyfunction]
pub(crate) fn validate_project_json(project_json: &str) -> PyResult<String> {
    let project =
        calc_flow::import_project_json(project_json.as_bytes()).map_err(crate::error::to_py_err)?;
    let value = serde_json::to_value(project).map_err(|error| {
        crate::error::to_py_err(calc_flow::CalcFlowError::Format {
            message: error.to_string(),
        })
    })?;
    calc_flow::canonical_json(&value).map_err(crate::error::to_py_err)
}

pub(crate) fn json_to_python<'py>(py: Python<'py>, encoded: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import(pyo3::intern!(py, "json"))?
        .getattr(pyo3::intern!(py, "loads"))?
        .call1((encoded,))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRuntime>()?;
    module.add_function(wrap_pyfunction!(project_json_schema, module)?)?;
    module.add_function(wrap_pyfunction!(validate_project_json, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyDict;

    use super::*;

    const PROJECT: &str = r#"{
        "format_version": 2,
        "id": "demo",
        "name": "Demo",
        "pipeline": {
            "name": "demo",
            "nodes": [{
                "id": "calc",
                "operator": {"kind": "expression", "expression": "b = a + 1"}
            }]
        },
        "data_sources": [{
            "id": "source_1", "input": "input", "format": "inline_json", "data": []
        }]
    }"#;

    #[test]
    fn runtime_compiles_strict_projects_and_helpers_are_canonical() {
        let runtime = PyRuntime::new().unwrap();
        let plan = runtime.compile_project(PROJECT).unwrap();
        assert_eq!(plan.inner.name(), "demo");
        assert!(runtime.compile_project("not JSON").is_err());

        let canonical = validate_project_json(PROJECT).unwrap();
        let value: serde_json::Value = serde_json::from_str(&canonical).unwrap();
        assert_eq!(value["description"], "");
        assert_eq!(value["run_options"]["timeout_seconds"], 30);
        assert_eq!(canonical, calc_flow::canonical_json(&value).unwrap());

        let schema: serde_json::Value =
            serde_json::from_str(&project_json_schema().unwrap()).unwrap();
        assert_eq!(schema["title"], "Calc Flow Project V2");
        assert_eq!(schema["properties"]["format_version"]["const"], 2);
    }

    #[test]
    fn json_conversion_returns_independent_python_values() {
        Python::initialize();
        Python::attach(|py| {
            let first = json_to_python(py, r#"{"nested":{"value":1}}"#).unwrap();
            first
                .cast::<PyDict>()
                .unwrap()
                .get_item("nested")
                .unwrap()
                .unwrap()
                .cast::<PyDict>()
                .unwrap()
                .set_item("value", 2)
                .unwrap();
            let second = json_to_python(py, r#"{"nested":{"value":1}}"#).unwrap();
            let value: usize = second
                .cast::<PyDict>()
                .unwrap()
                .get_item("nested")
                .unwrap()
                .unwrap()
                .cast::<PyDict>()
                .unwrap()
                .get_item("value")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(value, 1);
        });
    }
}
