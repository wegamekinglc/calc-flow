use pyo3::{
    create_exception,
    exceptions::{PyException, PyOSError},
    prelude::*,
};

create_exception!(calc_flow._native, CalcFlowError, PyException);
create_exception!(calc_flow._native, ConfigError, CalcFlowError);
create_exception!(calc_flow._native, CompileError, CalcFlowError);
create_exception!(calc_flow._native, ExecutionError, CalcFlowError);
create_exception!(calc_flow._native, ProviderError, ExecutionError);
create_exception!(calc_flow._native, CheckpointError, CalcFlowError);
create_exception!(calc_flow._native, CancelledError, ExecutionError);

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("CalcFlowError", py.get_type::<CalcFlowError>())?;
    module.add("ConfigError", py.get_type::<ConfigError>())?;
    module.add("CompileError", py.get_type::<CompileError>())?;
    module.add("ExecutionError", py.get_type::<ExecutionError>())?;
    module.add("ProviderError", py.get_type::<ProviderError>())?;
    module.add("CheckpointError", py.get_type::<CheckpointError>())?;
    module.add("CancelledError", py.get_type::<CancelledError>())?;
    Ok(())
}

/// Convert the core error surface at the Python boundary.
///
/// A direct `From<calc_flow::CalcFlowError> for PyErr` implementation is not
/// legal under Rust's orphan rules. Binding modules call this local conversion
/// function with `map_err` instead.
#[allow(
    dead_code,
    reason = "consumed by the binding modules introduced in Tasks 16-21"
)]
pub(crate) fn to_py_err(error: calc_flow::CalcFlowError) -> PyErr {
    let message = error.to_string();
    match error {
        calc_flow::CalcFlowError::InvalidArgument { .. }
        | calc_flow::CalcFlowError::UnsupportedVersion { .. }
        | calc_flow::CalcFlowError::Format { .. }
        | calc_flow::CalcFlowError::Conflict { .. }
        | calc_flow::CalcFlowError::NotFound { .. } => ConfigError::new_err(message),
        calc_flow::CalcFlowError::Compile { .. } => CompileError::new_err(message),
        calc_flow::CalcFlowError::Operator { .. }
        | calc_flow::CalcFlowError::DataFusion { .. }
        | calc_flow::CalcFlowError::PlanLeased { .. }
        | calc_flow::CalcFlowError::Internal { .. } => ExecutionError::new_err(message),
        calc_flow::CalcFlowError::ExternalProvider { .. } => ProviderError::new_err(message),
        calc_flow::CalcFlowError::Cancelled { .. } => CancelledError::new_err(message),
        calc_flow::CalcFlowError::CheckpointMismatch { .. }
        | calc_flow::CalcFlowError::RecoveryRequired { .. } => CheckpointError::new_err(message),
        calc_flow::CalcFlowError::Io { source, .. } => {
            let translated = CheckpointError::new_err(message);
            Python::attach(|py| {
                translated.set_cause(py, Some(PyOSError::new_err(source.to_string())));
            });
            translated
        }
        // The core error is deliberately non-exhaustive during alpha. New
        // categories remain catchable as execution failures until this mapping
        // gives them a more precise Python exception.
        _ => future_core_error(message),
    }
}

fn future_core_error(message: String) -> PyErr {
    ExecutionError::new_err(message)
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    #[derive(Clone, Copy)]
    enum ExpectedException {
        Config,
        Compile,
        Execution,
        Provider,
        Checkpoint,
        Cancelled,
    }

    fn assert_mapping(
        error: calc_flow::CalcFlowError,
        expected: ExpectedException,
        expected_message: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let translated = to_py_err(error);
            let matches = match expected {
                ExpectedException::Config => translated.is_instance_of::<ConfigError>(py),
                ExpectedException::Compile => translated.is_instance_of::<CompileError>(py),
                ExpectedException::Execution => translated.is_instance_of::<ExecutionError>(py),
                ExpectedException::Provider => translated.is_instance_of::<ProviderError>(py),
                ExpectedException::Checkpoint => translated.is_instance_of::<CheckpointError>(py),
                ExpectedException::Cancelled => translated.is_instance_of::<CancelledError>(py),
            };
            assert!(matches);
            assert_eq!(
                translated.value(py).str().unwrap().to_str().unwrap(),
                expected_message
            );
        });
    }

    #[test]
    fn maps_configuration_and_compilation_error_categories() {
        assert_mapping(
            calc_flow::CalcFlowError::InvalidArgument {
                field: "batch".into(),
                message: "is invalid".into(),
            },
            ExpectedException::Config,
            "invalid batch: is invalid",
        );
        assert_mapping(
            calc_flow::CalcFlowError::UnsupportedVersion {
                expected: 2,
                found: 1,
            },
            ExpectedException::Config,
            "project format version 1 is unsupported; expected 2",
        );
        assert_mapping(
            calc_flow::CalcFlowError::Compile {
                message: "cycle".into(),
            },
            ExpectedException::Compile,
            "graph compilation failed: cycle",
        );
        assert_mapping(
            calc_flow::CalcFlowError::Format {
                message: "bad JSON".into(),
            },
            ExpectedException::Config,
            "stored document is invalid: bad JSON",
        );
        assert_mapping(
            calc_flow::CalcFlowError::Conflict {
                resource: "project".into(),
                key: "alpha".into(),
            },
            ExpectedException::Config,
            "project \"alpha\" already exists",
        );
        assert_mapping(
            calc_flow::CalcFlowError::NotFound {
                resource: "project".into(),
                key: "alpha".into(),
            },
            ExpectedException::Config,
            "project \"alpha\" was not found",
        );
    }

    #[test]
    fn maps_execution_and_checkpoint_error_categories() {
        assert_mapping(
            calc_flow::CalcFlowError::Operator {
                node_id: "sum".into(),
                message: "failed".into(),
            },
            ExpectedException::Execution,
            "node sum failed: failed",
        );
        assert_mapping(
            calc_flow::CalcFlowError::DataFusion {
                node_id: Some("query".into()),
                message: "failed".into(),
            },
            ExpectedException::Execution,
            "DataFusion failed for node Some(\"query\"): failed",
        );
        assert_mapping(
            calc_flow::CalcFlowError::ExternalProvider {
                provider: "python".into(),
                name: "custom".into(),
                version: "1".into(),
                message: "failed".into(),
            },
            ExpectedException::Provider,
            "external provider python:custom@1 failed: failed",
        );
        assert_mapping(
            calc_flow::CalcFlowError::Cancelled {
                run_id: "run-1".into(),
            },
            ExpectedException::Cancelled,
            "run run-1 was cancelled",
        );
        assert_mapping(
            calc_flow::CalcFlowError::CheckpointMismatch {
                message: "fingerprint".into(),
            },
            ExpectedException::Checkpoint,
            "checkpoint mismatch: fingerprint",
        );
        assert_mapping(
            calc_flow::CalcFlowError::PlanLeased {
                pipeline_name: "daily".into(),
            },
            ExpectedException::Execution,
            "execution plan \"daily\" is exclusively leased by a runner",
        );
        assert_mapping(
            calc_flow::CalcFlowError::RecoveryRequired {
                pipeline_name: "daily".into(),
                message: "restore first".into(),
            },
            ExpectedException::Checkpoint,
            "execution plan \"daily\" requires recovery: restore first",
        );
        assert_mapping(
            calc_flow::CalcFlowError::Internal {
                message: "unreachable".into(),
            },
            ExpectedException::Execution,
            "internal invariant failed: unreachable",
        );
    }

    #[test]
    fn preserves_io_error_as_python_cause() {
        Python::initialize();
        Python::attach(|py| {
            let translated = to_py_err(calc_flow::CalcFlowError::Io {
                path: "/tmp/checkpoint".into(),
                source: io::Error::other("disk full"),
            });
            assert!(translated.is_instance_of::<CheckpointError>(py));
            assert_eq!(
                translated.value(py).str().unwrap().to_str().unwrap(),
                "I/O failed for /tmp/checkpoint: disk full"
            );
            let cause = translated.value(py).getattr("__cause__").unwrap();
            assert!(cause.is_instance_of::<PyOSError>());
            assert_eq!(cause.str().unwrap().to_str().unwrap(), "disk full");
        });
    }
}
