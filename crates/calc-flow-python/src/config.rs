use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
use pyo3::{
    Borrowed, PyTraverseError, PyVisit,
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyAny, PyBool},
};

use crate::pipeline::{PyExecutionPlan, PyStreamExecutionPlan};

const CLEARED_RUNTIME_MESSAGE: &str = "Runtime has been cleared by garbage collection";

#[derive(Clone, Copy)]
struct ExactBool(bool);

impl<'a, 'py> FromPyObject<'a, 'py> for ExactBool {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if !value.is_exact_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "accepts_context must be an exact bool",
            ));
        }
        value.extract::<bool>().map(Self)
    }
}

/// The single Python reference shared by an opaque Rust implementation and GC.
///
/// Provider factories and UDF closures must retain this same `Arc` instead of
/// cloning the contained `Py`. That keeps the number of references reported by
/// `__traverse__` equal to the number actually owned by the native object.
pub(crate) struct PythonRoot {
    object: Py<PyAny>,
}

impl PythonRoot {
    #[allow(
        dead_code,
        reason = "Tasks 18 and 19 construct shared roots before registering Python callbacks"
    )]
    pub(crate) const fn new(object: Py<PyAny>) -> Self {
        Self { object }
    }

    pub(crate) const fn object(&self) -> &Py<PyAny> {
        &self.object
    }
}

pub(crate) fn deduplicate_python_roots(
    roots: impl IntoIterator<Item = Arc<PythonRoot>>,
) -> Vec<Arc<PythonRoot>> {
    roots.into_iter().fold(Vec::new(), |mut unique, root| {
        if !unique.iter().any(|existing| Arc::ptr_eq(existing, &root)) {
            unique.push(root);
        }
        unique
    })
}

fn mapping_port_contract(name: &str, kind: &str) -> PyResult<crate::provider::PortContract> {
    let kind = match kind {
        "table" => calc_flow::BatchKind::Table,
        "array" => calc_flow::BatchKind::Array,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mapping provider port kind must be 'table' or 'array'",
            ));
        }
    };
    Ok(crate::provider::PortContract::new(name, kind))
}

struct RuntimeState {
    providers: Arc<calc_flow::ProviderRegistry>,
    udfs: Arc<RwLock<calc_flow::UdfRegistry>>,
    tokio: Arc<tokio::runtime::Runtime>,
    roots: Vec<Arc<PythonRoot>>,
    udf_catalog: BTreeMap<(String, String, String), UdfCatalogMetadata>,
}

#[derive(Clone)]
pub(crate) struct UdfCatalogMetadata {
    pub(crate) provider: String,
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) input_types: Vec<String>,
    pub(crate) return_type: String,
    pub(crate) volatility: String,
}

struct RuntimeSnapshot {
    providers: Arc<calc_flow::ProviderRegistry>,
    udfs: calc_flow::UdfRegistrySnapshot,
    tokio: Arc<tokio::runtime::Runtime>,
}

#[pyclass(name = "Runtime", frozen, module = "calc_flow._native")]
pub(crate) struct PyRuntime {
    state: RwLock<Option<RuntimeState>>,
}

impl PyRuntime {
    fn from_tokio(tokio: Arc<tokio::runtime::Runtime>) -> Self {
        Self {
            state: RwLock::new(Some(RuntimeState {
                providers: Arc::new(calc_flow::ProviderRegistry::default()),
                udfs: Arc::new(RwLock::new(calc_flow::UdfRegistry::new())),
                tokio,
                roots: Vec::new(),
                udf_catalog: BTreeMap::new(),
            })),
        }
    }

    fn snapshot(&self) -> PyResult<RuntimeSnapshot> {
        let state = self.state.read();
        let state = state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_RUNTIME_MESSAGE))?;
        let udfs = state.udfs.read().snapshot();
        Ok(RuntimeSnapshot {
            providers: Arc::clone(&state.providers),
            udfs,
            tokio: Arc::clone(&state.tokio),
        })
    }

    #[allow(
        dead_code,
        reason = "Task 18 registers Python-hosted provider factories through this GC-safe seam"
    )]
    /// Registers a factory that retains the same `root` allocation.
    pub(crate) fn register_provider_factory(
        &self,
        provider: &str,
        name: &str,
        version: &str,
        factory: &Arc<dyn calc_flow::BatchOperatorFactory>,
        root: Arc<PythonRoot>,
    ) -> PyResult<()> {
        let providers = {
            let state = self.state.read();
            let state = state
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err(CLEARED_RUNTIME_MESSAGE))?;
            Arc::clone(&state.providers)
        };
        providers
            .register_batch(provider, name, version, Arc::clone(factory))
            .map_err(crate::error::to_py_err)?;
        let mut state = self.state.write();
        let state = state
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_RUNTIME_MESSAGE))?;
        if !state
            .roots
            .iter()
            .any(|existing| Arc::ptr_eq(existing, &root))
        {
            state.roots.push(root);
        }
        Ok(())
    }

    #[allow(
        dead_code,
        reason = "Task 19 registers Python scalar UDFs through this GC-safe seam"
    )]
    /// Registers a UDF whose implementation retains the same `root` allocation.
    pub(crate) fn register_datafusion_udf(
        &self,
        reference: calc_flow::UdfReference,
        udf: &Arc<datafusion::logical_expr::ScalarUDF>,
        argument_count: usize,
        root: Arc<PythonRoot>,
    ) -> PyResult<()> {
        let udfs = {
            let state = self.state.read();
            let state = state
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err(CLEARED_RUNTIME_MESSAGE))?;
            Arc::clone(&state.udfs)
        };
        let registration = {
            let mut registry = udfs.write();
            registry.register_datafusion(reference, Arc::clone(udf), argument_count)
        };
        registration.map_err(crate::error::to_py_err)?;
        let mut state = self.state.write();
        let state = state
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_RUNTIME_MESSAGE))?;
        if !state
            .roots
            .iter()
            .any(|existing| Arc::ptr_eq(existing, &root))
        {
            state.roots.push(root);
        }
        Ok(())
    }

    fn record_udf_metadata(&self, metadata: UdfCatalogMetadata) -> PyResult<()> {
        let mut state = self.state.write();
        let state = state
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_RUNTIME_MESSAGE))?;
        state.udf_catalog.insert(
            (
                metadata.provider.clone(),
                metadata.name.clone(),
                metadata.version.clone(),
            ),
            metadata,
        );
        Ok(())
    }

    #[allow(
        dead_code,
        reason = "Task 19 reads the data-only UDF catalog through this guarded snapshot"
    )]
    pub(crate) fn udf_snapshot(&self) -> PyResult<calc_flow::UdfRegistrySnapshot> {
        let state = self.state.read();
        let state = state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_RUNTIME_MESSAGE))?;
        Ok(state.udfs.read().snapshot())
    }

    #[allow(
        dead_code,
        reason = "Task 20 reuses the runtime for store and runner blocking facades"
    )]
    pub(crate) fn clone_tokio(&self) -> PyResult<Arc<tokio::runtime::Runtime>> {
        let state = self.state.read();
        state
            .as_ref()
            .map(|state| Arc::clone(&state.tokio))
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_RUNTIME_MESSAGE))
    }
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> PyResult<Self> {
        let tokio = tokio::runtime::Runtime::new()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self::from_tokio(Arc::new(tokio)))
    }

    #[pyo3(name = "register_provider")]
    #[pyo3(
        signature = (provider, name, version, callback, *, accepts_context = ExactBool(false)),
        text_signature = "($self, provider, name, version, callback, *, accepts_context=False)"
    )]
    fn register_python_provider(
        &self,
        py: Python<'_>,
        provider: &str,
        name: &str,
        version: &str,
        callback: Py<PyAny>,
        accepts_context: ExactBool,
    ) -> PyResult<()> {
        if !callback.bind(py).is_callable() {
            return Err(PyTypeError::new_err("provider callback must be callable"));
        }
        let root = Arc::new(PythonRoot::new(callback));
        let factory: Arc<dyn calc_flow::BatchOperatorFactory> =
            Arc::new(crate::provider::PythonOperatorFactory::new_with_context(
                Arc::clone(&root),
                provider,
                name,
                version,
                accepts_context.0,
            ));
        self.register_provider_factory(provider, name, version, &factory, root)
    }

    #[pyo3(
        signature = (provider, name, version, callback, *, input_ports, output_ports, accepts_context = ExactBool(false)),
        text_signature = "($self, provider, name, version, callback, *, input_ports, output_ports, accepts_context=False)"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "the private binding preserves the explicit mapping provider contract"
    )]
    fn _register_mapping_provider(
        &self,
        py: Python<'_>,
        provider: &str,
        name: &str,
        version: &str,
        callback: Py<PyAny>,
        input_ports: Vec<(String, String)>,
        output_ports: Vec<(String, String)>,
        accepts_context: ExactBool,
    ) -> PyResult<()> {
        if !callback.bind(py).is_callable() {
            return Err(PyTypeError::new_err("provider callback must be callable"));
        }
        let inputs = input_ports
            .into_iter()
            .map(|(port, kind)| mapping_port_contract(&port, &kind))
            .collect::<PyResult<Vec<_>>>()?;
        let outputs = output_ports
            .into_iter()
            .map(|(port, kind)| mapping_port_contract(&port, &kind))
            .collect::<PyResult<Vec<_>>>()?;
        let root = Arc::new(PythonRoot::new(callback));
        let factory: Arc<dyn calc_flow::BatchOperatorFactory> = Arc::new(
            crate::provider::PythonOperatorFactory::new_mapping_with_context(
                Arc::clone(&root),
                provider,
                name,
                version,
                inputs,
                outputs,
                accepts_context.0,
            ),
        );
        self.register_provider_factory(provider, name, version, &factory, root)
    }

    #[pyo3(signature = (*, provider, name, version, input_types, return_type, volatility, function))]
    #[allow(
        clippy::too_many_arguments,
        reason = "the public binding preserves the explicit keyword-only UDF contract"
    )]
    fn register_scalar_udf(
        &self,
        py: Python<'_>,
        provider: &str,
        name: &str,
        version: &str,
        input_types: Vec<String>,
        return_type: String,
        volatility: String,
        function: Py<PyAny>,
    ) -> PyResult<()> {
        let prepared = crate::udf::prepare_python_scalar_udf(
            py,
            provider,
            name,
            version,
            input_types,
            return_type,
            volatility,
            function,
        )?;
        self.register_datafusion_udf(
            prepared.reference,
            &prepared.udf,
            prepared.metadata.input_types.len(),
            prepared.root,
        )?;
        self.record_udf_metadata(prepared.metadata)
    }

    fn catalog<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let entries = {
            let state = self.state.read();
            let state = state
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err(CLEARED_RUNTIME_MESSAGE))?;
            state.udf_catalog.values().cloned().collect::<Vec<_>>()
        };
        let value = serde_json::Value::Array(
            entries
                .into_iter()
                .map(|entry| {
                    serde_json::json!({
                        "provider": entry.provider,
                        "name": entry.name,
                        "version": entry.version,
                        "kind": "data_fusion_scalar",
                        "signature": {
                            "input_types": entry.input_types,
                            "return_type": entry.return_type,
                        },
                        "volatility": entry.volatility,
                    })
                })
                .collect(),
        );
        let encoded = calc_flow::canonical_json(&value).map_err(crate::error::to_py_err)?;
        json_to_python(py, &encoded)
    }

    fn validation_report<'py>(
        &self,
        py: Python<'py>,
        project_json: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let runtime = self.snapshot()?;
        let project = calc_flow::import_project_json(project_json.as_bytes())
            .map_err(crate::error::to_py_err)?;
        let report = calc_flow::validate_project(&project, &runtime.providers, &runtime.udfs);
        let value = serde_json::to_value(report).map_err(|error| {
            crate::error::to_py_err(calc_flow::CalcFlowError::Format {
                message: error.to_string(),
            })
        })?;
        let encoded = calc_flow::canonical_json(&value).map_err(crate::error::to_py_err)?;
        json_to_python(py, &encoded)
    }

    fn compile_project(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        project_json: &str,
    ) -> PyResult<PyExecutionPlan> {
        let runtime = slf.snapshot()?;
        let project = calc_flow::import_project_json(project_json.as_bytes())
            .map_err(crate::error::to_py_err)?;
        let plan = calc_flow::compile_project(&project, &runtime.providers, &runtime.udfs)
            .map_err(crate::error::to_py_err)?;
        let owner = slf.into_pyobject(py)?.into_any().unbind();
        Ok(PyExecutionPlan::new_with_owner(
            Arc::new(plan),
            runtime.tokio,
            owner,
        ))
    }

    fn compile_stream_project(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        project_json: &str,
        delivery: BTreeMap<String, String>,
    ) -> PyResult<PyStreamExecutionPlan> {
        let runtime = slf.snapshot()?;
        let project = calc_flow::import_project_json(project_json.as_bytes())
            .map_err(crate::error::to_py_err)?;
        let delivery = delivery
            .into_iter()
            .map(|(output, guarantee)| {
                let guarantee = match guarantee.as_str() {
                    "at_least_once" => calc_flow::DeliveryGuarantee::AtLeastOnce,
                    "exactly_once" => calc_flow::DeliveryGuarantee::ExactlyOnce,
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "delivery requirement for {output:?} must be 'at_least_once' or 'exactly_once'"
                        )));
                    }
                };
                Ok((output, guarantee))
            })
            .collect::<PyResult<BTreeMap<_, _>>>()?;
        let requirements = calc_flow::StreamRequirements { delivery };
        let plan = calc_flow::compile_stream_project(
            &project,
            &runtime.providers,
            &runtime.udfs,
            &requirements,
        )
        .map_err(crate::error::to_py_err)?;
        let owner = slf.into_pyobject(py)?.into_any().unbind();
        Ok(PyStreamExecutionPlan::new(plan, owner))
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
    use std::sync::Arc;

    use datafusion::{
        common::ScalarValue,
        logical_expr::{ColumnarValue, Volatility, create_udf},
    };
    use pyo3::types::PyDict;

    use super::*;

    struct RootedFactory {
        _callback: Arc<PythonRoot>,
    }

    impl calc_flow::BatchOperatorFactory for RootedFactory {
        fn create(
            &self,
            _spec: &calc_flow::ExternalOperatorSpec,
            _inputs: Vec<calc_flow::Port>,
            _outputs: Vec<calc_flow::Port>,
        ) -> calc_flow::Result<Box<dyn calc_flow::BatchOperator>> {
            unreachable!("the GC ownership test never compiles this provider")
        }
    }

    fn rooted_udf(name: &str, root: Arc<PythonRoot>) -> Arc<datafusion::logical_expr::ScalarUDF> {
        Arc::new(create_udf(
            name,
            vec![],
            datafusion::arrow::datatypes::DataType::Int64,
            Volatility::Immutable,
            Arc::new(move |_: &[ColumnarValue]| {
                let _keep_hidden_callback_alive = &root;
                Ok(ColumnarValue::Scalar(ScalarValue::Int64(Some(1))))
            }),
        ))
    }

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
        Python::initialize();
        Python::attach(|py| {
            let runtime = Py::new(py, PyRuntime::new().unwrap()).unwrap();
            let plan = runtime
                .bind(py)
                .call_method1("compile_project", (PROJECT,))
                .unwrap();
            let plan = plan.extract::<PyRef<'_, PyExecutionPlan>>().unwrap();
            let owned = PyExecutionPlan::owned(plan, py).unwrap();
            assert_eq!(owned.inner().name(), "demo");
            assert!(Arc::strong_count(owned.tokio()) >= 2);
            assert!(runtime.borrow(py).clone_tokio().is_ok());
            assert!(
                runtime
                    .borrow(py)
                    .udf_snapshot()
                    .unwrap()
                    .catalog()
                    .is_empty()
            );
            assert!(
                runtime
                    .bind(py)
                    .call_method1("compile_project", ("not JSON",))
                    .is_err()
            );

            let canonical = validate_project_json(PROJECT).unwrap();
            let value: serde_json::Value = serde_json::from_str(&canonical).unwrap();
            assert_eq!(value["description"], "");
            assert_eq!(value["run_options"]["timeout_seconds"], 30);
            assert_eq!(canonical, calc_flow::canonical_json(&value).unwrap());

            let schema: serde_json::Value =
                serde_json::from_str(&project_json_schema().unwrap()).unwrap();
            assert_eq!(schema["title"], "Calc Flow Project V2");
            assert_eq!(schema["properties"]["format_version"]["const"], 2);
        });
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

    #[test]
    fn cleared_runtime_rejects_public_and_extension_access() {
        Python::initialize();
        Python::attach(|py| {
            let runtime = Py::new(py, PyRuntime::new().unwrap()).unwrap();
            runtime.borrow(py).__clear__();

            let compile = runtime
                .bind(py)
                .call_method1("compile_project", (PROJECT,))
                .unwrap_err();
            assert_eq!(compile.value(py).to_string(), CLEARED_RUNTIME_MESSAGE);
            let tokio_error = match runtime.borrow(py).clone_tokio() {
                Ok(_) => panic!("a cleared runtime must not expose its Tokio runtime"),
                Err(error) => error,
            };
            let udf_error = match runtime.borrow(py).udf_snapshot() {
                Ok(_) => panic!("a cleared runtime must not expose its UDF registry"),
                Err(error) => error,
            };
            for error in [tokio_error, udf_error] {
                assert_eq!(error.value(py).to_string(), CLEARED_RUNTIME_MESSAGE);
            }
            runtime.borrow(py).__clear__();
        });
    }

    #[test]
    fn runtime_gc_collects_cycles_hidden_in_provider_and_udf_registries() {
        Python::initialize();
        Python::attach(|py| {
            let weakref = py.import("weakref").unwrap().getattr("ref").unwrap();
            let tokio = Arc::new(tokio::runtime::Runtime::new().unwrap());
            let mut references = Vec::new();

            for _ in 0..1_000 {
                let runtime = Py::new(py, PyRuntime::from_tokio(Arc::clone(&tokio))).unwrap();
                let provider = py
                    .eval(pyo3::ffi::c_str!("type('Holder', (), {})()"), None, None)
                    .unwrap();
                let udf = py
                    .eval(pyo3::ffi::c_str!("type('Holder', (), {})()"), None, None)
                    .unwrap();

                let provider_root = Arc::new(PythonRoot::new(provider.clone().unbind()));
                let factory: Arc<dyn calc_flow::BatchOperatorFactory> = Arc::new(RootedFactory {
                    _callback: Arc::clone(&provider_root),
                });
                runtime
                    .borrow(py)
                    .register_provider_factory("python", "array", "1", &factory, provider_root)
                    .unwrap();

                let udf_root = Arc::new(PythonRoot::new(udf.clone().unbind()));
                let native = rooted_udf("rooted", Arc::clone(&udf_root));
                runtime
                    .borrow(py)
                    .register_datafusion_udf(
                        calc_flow::UdfReference::new(
                            "python",
                            "rooted",
                            "1",
                            calc_flow::UdfKind::DataFusionScalar,
                        )
                        .unwrap(),
                        &native,
                        0,
                        udf_root,
                    )
                    .unwrap();

                provider.setattr("owner", runtime.bind(py)).unwrap();
                udf.setattr("owner", runtime.bind(py)).unwrap();
                references.push(weakref.call1((&provider,)).unwrap().unbind());
                references.push(weakref.call1((&udf,)).unwrap().unbind());
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
    fn runtime_reports_each_shared_python_root_once() {
        Python::initialize();
        Python::attach(|py| {
            let runtime = PyRuntime::new().unwrap();
            let root = Arc::new(PythonRoot::new(py.None()));
            let factory: Arc<dyn calc_flow::BatchOperatorFactory> = Arc::new(RootedFactory {
                _callback: Arc::clone(&root),
            });
            runtime
                .register_provider_factory("python", "array", "1", &factory, Arc::clone(&root))
                .unwrap();
            let native = rooted_udf("rooted", Arc::clone(&root));
            runtime
                .register_datafusion_udf(
                    calc_flow::UdfReference::new(
                        "python",
                        "rooted",
                        "1",
                        calc_flow::UdfKind::DataFusionScalar,
                    )
                    .unwrap(),
                    &native,
                    0,
                    root,
                )
                .unwrap();

            let state = runtime.state.read();
            assert_eq!(state.as_ref().unwrap().roots.len(), 1);
        });
    }

    #[test]
    fn rejected_registrations_drop_reentrant_callbacks_after_unlocking() {
        Python::initialize();
        Python::attach(|py| {
            let runtime = Py::new(py, PyRuntime::new().unwrap()).unwrap();
            let inert_provider = Arc::new(PythonRoot::new(py.None()));
            let inert_factory: Arc<dyn calc_flow::BatchOperatorFactory> = Arc::new(RootedFactory {
                _callback: Arc::clone(&inert_provider),
            });
            runtime
                .borrow(py)
                .register_provider_factory("python", "array", "1", &inert_factory, inert_provider)
                .unwrap();
            let inert_udf = Arc::new(PythonRoot::new(py.None()));
            let inert_native = rooted_udf("rooted", Arc::clone(&inert_udf));
            runtime
                .borrow(py)
                .register_datafusion_udf(
                    calc_flow::UdfReference::new(
                        "python",
                        "rooted",
                        "1",
                        calc_flow::UdfKind::DataFusionScalar,
                    )
                    .unwrap(),
                    &inert_native,
                    0,
                    inert_udf,
                )
                .unwrap();

            let locals = PyDict::new(py);
            locals.set_item("runtime", runtime.bind(py)).unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    "events = []\nclass Callback:\n    def __init__(self, owner):\n        self.owner = owner\n    def __del__(self):\n        try:\n            self.owner.compile_project('not JSON')\n        except Exception:\n            events.append('reentered')"
                ),
                Some(&locals),
                None,
            )
            .unwrap();
            let callback_type = locals.get_item("Callback").unwrap().unwrap();

            let provider_callback = callback_type.call1((runtime.bind(py),)).unwrap();
            let provider_root = Arc::new(PythonRoot::new(provider_callback.unbind()));
            let rejected_factory: Arc<dyn calc_flow::BatchOperatorFactory> =
                Arc::new(RootedFactory {
                    _callback: Arc::clone(&provider_root),
                });
            assert!(
                runtime
                    .borrow(py)
                    .register_provider_factory(
                        "python",
                        "array",
                        "1",
                        &rejected_factory,
                        provider_root,
                    )
                    .is_err()
            );
            drop(rejected_factory);

            let udf_callback = callback_type.call1((runtime.bind(py),)).unwrap();
            let udf_root = Arc::new(PythonRoot::new(udf_callback.unbind()));
            let rejected_udf = rooted_udf("rooted", Arc::clone(&udf_root));
            assert!(
                runtime
                    .borrow(py)
                    .register_datafusion_udf(
                        calc_flow::UdfReference::new(
                            "python",
                            "rooted",
                            "1",
                            calc_flow::UdfKind::DataFusionScalar,
                        )
                        .unwrap(),
                        &rejected_udf,
                        0,
                        udf_root,
                    )
                    .is_err()
            );
            drop(rejected_udf);

            let events = locals
                .get_item("events")
                .unwrap()
                .unwrap()
                .extract::<Vec<String>>()
                .unwrap();
            assert_eq!(events, ["reentered", "reentered"]);
        });
    }

    #[test]
    fn duplicate_python_udf_registration_leaves_roots_and_catalog_unchanged() {
        Python::initialize();
        Python::attach(|py| {
            let runtime = PyRuntime::new().unwrap();
            let callback = || {
                py.eval(pyo3::ffi::c_str!("lambda value: value"), None, None)
                    .unwrap()
                    .unbind()
            };
            runtime
                .register_scalar_udf(
                    py,
                    "python",
                    "identity",
                    "1",
                    vec!["int64".into()],
                    "int64".into(),
                    "immutable".into(),
                    callback(),
                )
                .unwrap();
            let before = {
                let state = runtime.state.read();
                let state = state.as_ref().unwrap();
                (state.roots.len(), state.udf_catalog.len())
            };
            assert!(
                runtime
                    .register_scalar_udf(
                        py,
                        "python",
                        "identity",
                        "1",
                        vec!["int64".into()],
                        "int64".into(),
                        "immutable".into(),
                        callback(),
                    )
                    .is_err()
            );
            let state = runtime.state.read();
            let state = state.as_ref().unwrap();
            assert_eq!((state.roots.len(), state.udf_catalog.len()), before);
        });
    }

    #[test]
    fn public_catalog_and_validation_report_are_sorted_redacted_and_defensive() {
        Python::initialize();
        Python::attach(|py| {
            let runtime = PyRuntime::new().unwrap();
            let callback = py
                .eval(pyo3::ffi::c_str!("lambda value: value"), None, None)
                .unwrap()
                .unbind();
            runtime
                .register_scalar_udf(
                    py,
                    "python",
                    "identity",
                    "1",
                    vec!["int64".into()],
                    "int64".into(),
                    "stable".into(),
                    callback,
                )
                .unwrap();

            let catalog = runtime.catalog(py).unwrap();
            let encoded: String = py
                .import("json")
                .unwrap()
                .getattr("dumps")
                .unwrap()
                .call1((catalog,))
                .unwrap()
                .extract()
                .unwrap();
            assert!(encoded.contains("identity"));
            assert!(encoded.contains("stable"));
            assert!(!encoded.contains("function"));

            let report = runtime.validation_report(py, PROJECT).unwrap();
            let valid: bool = report
                .cast::<PyDict>()
                .unwrap()
                .get_item("valid")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(valid);
            assert!(runtime.validation_report(py, "not JSON").is_err());
        });
    }
}
