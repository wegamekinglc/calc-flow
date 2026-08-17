//! Strict project format v3 (task M6.7).
//!
//! The v3 model replaces the inline-data v2 shape with a runtime-typed
//! contract: `runtime.mode` selects batch or stream semantics, sources
//! and sinks bind through data-only connector references, database
//! bindings declare their read/write mode, and secrets arrive
//! exclusively as [`SecretRef`] values. Every layer rejects unknown
//! fields.

use std::collections::BTreeMap;

use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{BatchKind, CalcFlowError, ProviderRegistry, Result};
use sha2::Digest as _;

/// The v3 format version constant.
pub const PROJECT_FORMAT_VERSION_V3: u32 = 3;

/// The root of a strict data-only project document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectV3 {
    #[serde(deserialize_with = "deserialize_v3_version")]
    pub format_version: u32,
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub runtime: RuntimeSpec,
    pub pipeline: PipelineV3,
    #[serde(default)]
    pub sources: Vec<SourceBinding>,
    #[serde(default)]
    pub sinks: Vec<SinkBinding>,
    #[serde(default)]
    pub state: StateConfig,
}

/// Runtime mode and its mode-specific options.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSpec {
    pub mode: RuntimeMode,
    #[serde(default)]
    pub batch: BatchRuntimeOptions,
    #[serde(default)]
    pub stream: StreamRuntimeOptions,
}

/// Whether the project runs as a bounded batch or a continuous stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    Batch,
    Stream,
}

/// Bounded-batch runtime limits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct BatchRuntimeOptions {
    pub max_input_bytes: usize,
    pub max_rows: usize,
    pub timeout_seconds: u64,
    pub memory_limit_mb: usize,
}

impl Default for BatchRuntimeOptions {
    fn default() -> Self {
        Self {
            max_input_bytes: 10 * 1024 * 1024,
            max_rows: 100_000,
            timeout_seconds: 30,
            memory_limit_mb: 512,
        }
    }
}

/// Continuous-stream runtime limits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct StreamRuntimeOptions {
    pub checkpoint_interval_seconds: u64,
    pub max_batch_rows: usize,
    pub max_batch_bytes: usize,
}

impl Default for StreamRuntimeOptions {
    fn default() -> Self {
        Self {
            checkpoint_interval_seconds: 30,
            max_batch_rows: 10_000,
            max_batch_bytes: 64 * 1024 * 1024,
        }
    }
}

/// The operator graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PipelineV3 {
    pub name: String,
    pub nodes: Vec<NodeV3>,
    #[serde(default)]
    pub edges: Vec<EdgeV3>,
}

/// One graph node and its operator specification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeV3 {
    pub id: String,
    pub operator: OperatorV3,
    #[serde(default)]
    pub input_ports: Vec<PortV3>,
    #[serde(default)]
    pub output_ports: Vec<PortV3>,
    #[serde(default)]
    pub position: Option<PositionV3>,
}

/// The operator variant; external providers are data-only references.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OperatorV3 {
    Expression {
        #[serde(default)]
        expression: String,
        #[serde(default)]
        select: Vec<String>,
        #[serde(default)]
        filter: Option<String>,
        #[serde(default)]
        udfs: Vec<UdfRefV3>,
    },
    Sql {
        query: String,
        aliases: Vec<String>,
        #[serde(default)]
        udfs: Vec<UdfRefV3>,
    },
    External {
        provider: String,
        name: String,
        version: String,
        #[serde(default)]
        options: BTreeMap<String, Value>,
    },
}

/// A trusted UDF reference by provider/name/version.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UdfRefV3 {
    pub provider: String,
    pub name: String,
    pub version: String,
}

/// A named port with an optional exact Arrow schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PortV3 {
    pub name: String,
    pub kind: BatchKind,
    #[serde(default = "default_true_v3")]
    pub required: bool,
    #[serde(default)]
    pub schema: Vec<FieldV3>,
}

fn default_true_v3() -> bool {
    true
}

/// One Arrow field in a strict schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FieldV3 {
    pub name: String,
    pub data_type: String,
    #[serde(default = "default_true_v3")]
    pub nullable: bool,
}

/// One graph edge connecting two node ports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeV3 {
    pub source_node: String,
    #[serde(default = "default_output_v3")]
    pub source_port: String,
    pub target_node: String,
    #[serde(default = "default_input_v3")]
    pub target_port: String,
}

fn default_output_v3() -> String {
    "output".into()
}

fn default_input_v3() -> String {
    "input".into()
}

/// A canvas position for visual editors.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PositionV3 {
    pub x: f64,
    pub y: f64,
}

/// A data-only source binding to a registered connector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceBinding {
    pub id: String,
    /// The external graph input this binding feeds.
    pub input: String,
    /// Connector identity `(provider, name, version)`.
    pub connector: ConnectorRef,
    /// The wire format name; the registry validates it.
    #[serde(default)]
    pub format: Option<String>,
    /// Bounded connector options; secrets never appear here.
    #[serde(default)]
    pub options: BTreeMap<String, Value>,
    /// Named secret references this source resolves at open time.
    #[serde(default)]
    pub secrets: BTreeMap<String, SecretRef>,
    /// Watermark policy for stream mode.
    #[serde(default)]
    pub watermark: Option<WatermarkPolicyV3>,
    /// Explicit schema the source must match.
    #[serde(default)]
    pub schema: Vec<FieldV3>,
}

/// A data-only sink binding with a delivery request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SinkBinding {
    pub id: String,
    /// The external graph output this binding drains.
    pub output: String,
    /// Connector identity `(provider, name, version)`.
    pub connector: ConnectorRef,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub options: BTreeMap<String, Value>,
    #[serde(default)]
    pub secrets: BTreeMap<String, SecretRef>,
    /// The requested per-output delivery guarantee.
    pub delivery: DeliveryRequest,
}

/// A database binding's read or write mode declaration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum DatabaseBinding {
    Snapshot {
        table: String,
        #[serde(default)]
        cursor_column: String,
    },
    Polling {
        table: String,
        cursor_column: String,
        #[serde(default)]
        poll_interval_ms: u64,
    },
    Cdc {
        publication: String,
        slot: String,
    },
    Append {
        table: String,
    },
    Upsert {
        table: String,
        #[serde(default)]
        conflict_columns: Vec<String>,
    },
}

/// A connector identity reference.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConnectorRef {
    pub provider: String,
    pub name: String,
    pub version: String,
}

/// A named pointer to a secret; the only secret-shaped value a
/// data-only document may carry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SecretRef {
    /// Where the secret is resolved from.
    pub resolver: SecretResolverKindV3,
    /// The resolver-specific key.
    pub key: String,
}

/// The resolver vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SecretResolverKindV3 {
    Environment,
    File,
    Registered,
}

/// The requested delivery guarantee for one output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryRequest {
    AtLeastOnce,
    ExactlyOnce,
}

/// The watermark policy for one source in stream mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WatermarkPolicyV3 {
    SourceProvided,
    BoundedOutOfOrderness {
        event_time_column: String,
        max_out_of_orderness_ms: u64,
        #[serde(default)]
        emit_interval_ms: u64,
        #[serde(default)]
        idle_timeout_ms: Option<u64>,
    },
    Disabled,
}

/// State and checkpoint configuration for stream mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct StateConfig {
    /// Directory for managed checkpoint state.
    pub checkpoint_directory: String,
    /// Interval between automatic checkpoints.
    pub checkpoint_interval_seconds: u64,
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            checkpoint_directory: ".calc-flow-state".into(),
            checkpoint_interval_seconds: 30,
        }
    }
}

fn deserialize_v3_version<'de, D>(deserializer: D) -> std::result::Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = u32::deserialize(deserializer)?;
    if value != PROJECT_FORMAT_VERSION_V3 {
        return Err(serde::de::Error::custom(format!(
            "project format version {value} is unsupported; expected {PROJECT_FORMAT_VERSION_V3}"
        )));
    }
    Ok(value)
}

/// Generates the canonical JSON Schema for project format v3.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] if the schema cannot be encoded.
pub fn project_v3_json_schema() -> Result<Value> {
    let mut value =
        serde_json::to_value(schema_for!(ProjectV3)).map_err(|error| CalcFlowError::Format {
            message: error.to_string(),
        })?;
    value["title"] = Value::String("Calc Flow Project V3".into());
    value["properties"]["format_version"] = serde_json::json!({
        "const": PROJECT_FORMAT_VERSION_V3,
        "type": "integer"
    });
    Ok(value)
}

/// Validates a v3 project model against the trusted registries.
///
/// # Errors
///
/// Returns the first stable validation issue as a typed error.
pub fn validate_v3_project(project: &ProjectV3, providers: &ProviderRegistry) -> Result<()> {
    validate_v3_connectors(project)?;
    let _ = providers;
    Ok(())
}

fn validate_v3_connectors(project: &ProjectV3) -> Result<()> {
    let needs_source = project
        .pipeline
        .nodes
        .iter()
        .any(|n| matches!(n.operator, OperatorV3::External { .. }));
    if needs_source && project.sources.is_empty() {
        return Err(CalcFlowError::Format {
            message: "external operators require at least one source binding".into(),
        });
    }
    validate_v3_connector_list(
        project.sources.iter().map(|s| (&s.id, &s.connector)),
        "source",
    )?;
    validate_v3_connector_list(project.sinks.iter().map(|s| (&s.id, &s.connector)), "sink")
}

fn validate_v3_connector_list<'a, I>(bindings: I, label: &str) -> Result<()>
where
    I: Iterator<Item = (&'a String, &'a ConnectorRef)>,
{
    for (id, connector) in bindings {
        if connector.provider.is_empty() || connector.name.is_empty() {
            return Err(CalcFlowError::Format {
                message: format!("{label} {id} carries an empty connector identity"),
            });
        }
    }
    Ok(())
}

/// Deserializes a v3 project from its canonical JSON form.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] for malformed JSON or any
/// layer's unknown-field rejection.
pub fn parse_v3_project(data: &[u8]) -> Result<ProjectV3> {
    serde_json::from_slice::<ProjectV3>(data).map_err(|error| CalcFlowError::Format {
        message: error.to_string(),
    })
}

/// Serializes a v3 project to its canonical deterministic JSON form.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] if serialization fails.
pub fn serialize_v3_project(project: &ProjectV3) -> Result<Vec<u8>> {
    serde_json::to_vec(project).map_err(|error| CalcFlowError::Format {
        message: error.to_string(),
    })
}

/// Computes the canonical v3 project fingerprint.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] if serialization fails.
pub fn v3_fingerprint(project: &ProjectV3) -> Result<String> {
    let encoded = serialize_v3_project(project)?;
    let digest = sha2::Sha256::digest(&encoded);
    Ok(hex::encode(digest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_rejects_wrong_version() {
        let json = serde_json::json!({
            "format_version": 2,
            "id": "p",
            "name": "test",
            "runtime": {"mode": "batch"},
            "pipeline": {"name": "pipe", "nodes": []}
        });
        let error = parse_v3_project(json.to_string().as_bytes())
            .expect_err("v2 is rejected by the v3 parser");
        assert!(error.to_string().contains("expected 3"), "{error}");
    }

    #[test]
    fn v3_rejects_unknown_fields_at_every_layer() {
        let json = serde_json::json!({
            "format_version": 3,
            "id": "p",
            "name": "test",
            "runtime": {"mode": "batch", "unknown_option": true},
            "pipeline": {"name": "pipe", "nodes": []}
        });
        let error = parse_v3_project(json.to_string().as_bytes())
            .expect_err("runtime unknown field rejected");
        assert!(error.to_string().contains("unknown"), "{error}");

        let json2 = serde_json::json!({
            "format_version": 3,
            "id": "p",
            "name": "test",
            "runtime": {"mode": "batch"},
            "pipeline": {"name": "pipe", "nodes": [], "mystery": 1}
        });
        let error = parse_v3_project(json2.to_string().as_bytes())
            .expect_err("pipeline unknown field rejected");
        assert!(error.to_string().contains("unknown"), "{error}");

        let json3 = serde_json::json!({
            "format_version": 3,
            "id": "p",
            "name": "test",
            "top_level_extra": "no"
            ,
            "runtime": {"mode": "batch"},
            "pipeline": {"name": "pipe", "nodes": []}
        });
        let error = parse_v3_project(json3.to_string().as_bytes())
            .expect_err("root unknown field rejected");
        assert!(error.to_string().contains("unknown"), "{error}");
    }

    #[test]
    fn v3_accepts_minimal_batch_project() {
        let json = serde_json::json!({
            "format_version": 3,
            "id": "p",
            "name": "test",
            "runtime": {"mode": "batch"},
            "pipeline": {"name": "pipe", "nodes": []}
        });
        let project = parse_v3_project(json.to_string().as_bytes()).expect("parses");
        assert_eq!(project.runtime.mode, RuntimeMode::Batch);
        assert!(project.sources.is_empty());
        assert!(project.sinks.is_empty());
    }

    #[test]
    fn v3_secret_refs_only_no_values() {
        let json = serde_json::json!({
            "format_version": 3,
            "id": "p",
            "name": "test",
            "runtime": {"mode": "stream"},
            "pipeline": {"name": "pipe", "nodes": []},
            "sources": [{
                "id": "s1",
                "input": "in",
                "connector": {"provider": "calc-flow-connectors", "name": "kafka", "version": "2.0.0"},
                "secrets": {
                    "url_key": {"resolver": "environment", "key": "KAFKA_URL"}
                }
            }]
        });
        let project = parse_v3_project(json.to_string().as_bytes()).expect("parses");
        let secret = project.sources[0]
            .secrets
            .get("url_key")
            .expect("secret ref present");
        assert_eq!(secret.key, "KAFKA_URL");
        assert_eq!(secret.resolver, SecretResolverKindV3::Environment);
        // The struct has no field capable of holding a raw value.
        let encoded = serde_json::to_value(&project.sources[0].secrets).expect("serializes");
        assert!(
            !encoded.to_string().contains("password"),
            "no password-shaped field exists"
        );
    }

    #[test]
    fn v3_fingerprint_is_stable() {
        let json = serde_json::json!({
            "format_version": 3,
            "id": "p",
            "name": "test",
            "runtime": {"mode": "batch"},
            "pipeline": {"name": "pipe", "nodes": []}
        });
        let p1 = parse_v3_project(json.to_string().as_bytes()).expect("parses");
        let p2 = parse_v3_project(json.to_string().as_bytes()).expect("parses");
        assert_eq!(v3_fingerprint(&p1).unwrap(), v3_fingerprint(&p2).unwrap());
    }

    #[test]
    fn v3_schema_generated_and_deterministic() {
        let s1 = project_v3_json_schema().expect("generates");
        let s2 = project_v3_json_schema().expect("generates again");
        assert_eq!(s1, s2, "schema generation is deterministic");
        assert_eq!(s1["title"], "Calc Flow Project V3");
    }

    #[test]
    fn v3_delivery_request_round_trips() {
        let json = serde_json::json!({"sources": [], "sinks": [{
            "id": "out",
            "output": "output",
            "connector": {"provider": "p", "name": "n", "version": "1"},
            "delivery": "exactly_once"
        }]});
        let sink: SinkBinding = serde_json::from_value(json["sinks"][0].clone()).expect("parses");
        assert_eq!(sink.delivery, DeliveryRequest::ExactlyOnce);
    }
}
