//! Python bindings for Calc Flow's Rust-native v2 engine.

mod batch;
mod config;
mod connector;
mod continuous;
mod error;
mod execution_options;
mod pipeline;
mod provider;
mod runtime;
mod store;
mod udf;

use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    calc_flow::VERSION
}

#[pymodule(gil_used = true)]
#[pyo3(name = "_native")]
fn calc_flow_python(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    error::register(module)?;
    batch::register(module)?;
    execution_options::register(module)?;
    config::register(module)?;
    connector::register(module)?;
    continuous::register(module)?;
    pipeline::register(module)?;
    store::register(module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyType;

    use super::*;

    #[test]
    fn initializes_the_native_module_surface() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_native").unwrap();
            calc_flow_python(&module).unwrap();

            let reported_version: String = module
                .getattr("version")
                .unwrap()
                .call0()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(reported_version, calc_flow::VERSION);

            for name in [
                "CalcFlowError",
                "ConfigError",
                "CompileError",
                "ExecutionError",
                "ProviderError",
                "CheckpointError",
                "CancelledError",
                "Batch",
                "Runtime",
                "ExecutionPlan",
                "RunResult",
            ] {
                let exception = module.getattr(name).unwrap();
                assert!(exception.is_instance_of::<PyType>());
                assert_eq!(
                    exception
                        .getattr("__module__")
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    "calc_flow._native"
                );
            }
        });
    }

    #[test]
    fn registers_the_continuous_owning_job_bridge() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_native").unwrap();
            calc_flow_python(&module).unwrap();

            for name in [
                "_StreamingRunner",
                "_StreamingJob",
                "StreamingRuntimeError",
                "CheckpointPublicationUnknownError",
            ] {
                assert!(module.getattr(name).is_ok(), "missing native {name}");
            }
            for name in [
                "_ContinuousStreamingRunner",
                "_FileCheckpointStore",
                "_MicroBatchRunner",
            ] {
                assert!(module.getattr(name).is_err(), "unexpected native {name}");
            }
        });
    }
}
