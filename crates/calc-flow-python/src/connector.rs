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
        result.push(connector_to_dict(py, &identity, &descriptor)?);
    }
    Ok(result)
}

fn connector_to_dict<'py>(
    py: Python<'py>,
    identity: &calc_flow::ConnectorIdentity,
    descriptor: &calc_flow::ConnectorDescriptor,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    set_identity_fields(&dict, identity)?;
    dict.set_item("kind", kind_str(descriptor.kind))?;
    dict.set_item(
        "capabilities",
        capabilities_to_dict(py, descriptor.capabilities)?,
    )?;
    set_payload_fields(&dict, descriptor)?;
    Ok(dict)
}

fn set_identity_fields(
    dict: &Bound<'_, PyDict>,
    identity: &calc_flow::ConnectorIdentity,
) -> PyResult<()> {
    dict.set_item("provider", identity.provider.to_string())?;
    dict.set_item("name", identity.name.to_string())?;
    dict.set_item("version", identity.version.to_string())
}

fn set_payload_fields(
    dict: &Bound<'_, PyDict>,
    descriptor: &calc_flow::ConnectorDescriptor,
) -> PyResult<()> {
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
    dict.set_item("options_schema", options.to_string())
}

fn kind_str(kind: calc_flow::ConnectorKind) -> String {
    format!("{kind:?}").to_lowercase()
}

fn capabilities_to_dict(
    py: Python<'_>,
    caps: calc_flow::ConnectorCapabilities,
) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    set_axis_fields(&dict, caps)?;
    dict.set_item("snapshot", caps.snapshot)?;
    dict.set_item("polling", caps.polling)?;
    dict.set_item("cdc", caps.cdc)?;
    dict.set_item("lookup", caps.lookup)?;
    Ok(dict)
}

fn set_axis_fields(
    dict: &Bound<'_, PyDict>,
    caps: calc_flow::ConnectorCapabilities,
) -> PyResult<()> {
    dict.set_item("delivery", delivery_str(caps.delivery))?;
    dict.set_item("replay", replay_str(caps.replay))?;
    dict.set_item("watermark", watermark_str(caps.watermark))?;
    dict.set_item("transaction", transaction_str(caps.transaction))?;
    Ok(())
}

pub(crate) fn register_builtin_connectors(
    registry: &mut calc_flow::ConnectorRegistry,
) -> calc_flow::Result<()> {
    let _ = &registry;
    #[cfg(feature = "connector-file")]
    calc_flow_connectors::register_file_connectors(registry)?;
    #[cfg(all(
        not(feature = "connector-file"),
        any(
            feature = "connector-kafka",
            feature = "connector-postgresql",
            feature = "connector-clickhouse",
            feature = "connector-http",
            feature = "connector-websocket"
        )
    ))]
    calc_flow_connectors::register_format_codecs(registry)?;

    #[cfg(feature = "connector-kafka")]
    calc_flow_connectors::register_kafka_connectors(registry)?;
    #[cfg(feature = "connector-clickhouse")]
    calc_flow_connectors::register_clickhouse_connectors(registry)?;
    #[cfg(feature = "connector-http")]
    calc_flow_connectors::register_http_connectors(registry)?;
    #[cfg(feature = "connector-websocket")]
    calc_flow_connectors::register_websocket_connectors(registry)?;
    #[cfg(feature = "connector-postgresql")]
    calc_flow_connectors::register_postgresql_connectors(registry)?;

    Ok(())
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
        calc_flow::TransactionSupport::RetryDeduplicated => "retry_deduplicated".into(),
    }
}

/// Registers the connector module members.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(registered_connectors, module)?)?;
    Ok(())
}
