//! Connector capability enumeration for the Python surface.
//!
//! Exposes the connectors compiled into this wheel as data-only
//! descriptors; the Python layer never constructs factory objects.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use crate::error::to_py_err;

/// Returns every connector registered in the built-in trusted registry
/// as a list of data-only dictionaries.
#[pyfunction]
fn registered_connectors(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    let mut registry = calc_flow::ConnectorRegistry::new();
    register_builtin_connectors(&mut registry).map_err(to_py_err)?;
    let snapshot = registry.snapshot();
    let mut result = Vec::new();
    for identity in snapshot.identities() {
        let descriptor = snapshot
            .resolve_source(&identity)
            .map_err(to_py_err)?
            .descriptor()
            .clone();
        let dict = PyDict::new(py);
        dict.set_item("provider", identity.provider.to_string())?;
        dict.set_item("name", identity.name.to_string())?;
        dict.set_item("version", identity.version.to_string())?;
        dict.set_item("kind", format!("{:?}", descriptor.kind).to_lowercase())?;
        let caps = PyDict::new(py);
        caps.set_item("delivery", delivery_str(descriptor.capabilities.delivery))?;
        caps.set_item("replay", replay_str(descriptor.capabilities.replay))?;
        caps.set_item(
            "watermark",
            watermark_str(descriptor.capabilities.watermark),
        )?;
        caps.set_item(
            "transaction",
            transaction_str(descriptor.capabilities.transaction),
        )?;
        caps.set_item("snapshot", descriptor.capabilities.snapshot)?;
        caps.set_item("polling", descriptor.capabilities.polling)?;
        caps.set_item("cdc", descriptor.capabilities.cdc)?;
        caps.set_item("lookup", descriptor.capabilities.lookup)?;
        dict.set_item("capabilities", caps)?;
        let formats: Vec<String> = descriptor
            .formats
            .iter()
            .map(|f| f.name.to_string())
            .collect();
        dict.set_item("formats", formats)?;
        let options: Value = serde_json::to_value(&descriptor.config_schema)
            .map_err(|e| calc_flow::CalcFlowError::Internal {
                message: e.to_string(),
            })
            .map_err(to_py_err)?;
        dict.set_item("options_schema", options.to_string())?;
        result.push(dict);
    }
    Ok(result)
}

fn register_builtin_connectors(
    registry: &mut calc_flow::ConnectorRegistry,
) -> calc_flow::Result<()> {
    // register_file_connectors already registers the format codecs.
    calc_flow_connectors::register_file_connectors(registry)
}

fn delivery_str(value: calc_flow::DeliveryCapability) -> String {
    match value {
        calc_flow::DeliveryCapability::BestEffort => "best_effort".into(),
        calc_flow::DeliveryCapability::AtLeastOnce => "at_least_once".into(),
        calc_flow::DeliveryCapability::ExactlyOnce => "exactly_once".into(),
    }
}

fn replay_str(value: calc_flow::ReplayCapability) -> String {
    match value {
        calc_flow::ReplayCapability::ReplayableExact => "replayable_exact".into(),
        calc_flow::ReplayCapability::Unreplayable => "unreplayable".into(),
    }
}

fn watermark_str(value: calc_flow::WatermarkSupport) -> String {
    match value {
        calc_flow::WatermarkSupport::Native => "native".into(),
        calc_flow::WatermarkSupport::GeneratedOnly => "generated_only".into(),
    }
}

fn transaction_str(value: calc_flow::TransactionSupport) -> String {
    match value {
        calc_flow::TransactionSupport::None => "none".into(),
        calc_flow::TransactionSupport::PreCommitCommit => "pre_commit_commit".into(),
        calc_flow::TransactionSupport::LedgerIdempotent => "ledger_idempotent".into(),
    }
}

/// Registers the connector module members.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(registered_connectors, module)?)?;
    Ok(())
}
