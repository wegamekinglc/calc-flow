use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use datafusion::arrow::datatypes::{DataType, Field, TimeUnit};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::{Value, json};

use crate::operator::expression_query;
use crate::{
    Batch, BatchExecutionPlan, BatchKind, CalcFlowError, ConnectorIdentity,
    ConnectorRegistrySnapshot, ConnectorSinkFactory, ConnectorSourceFactory, Cursor,
    DataFusionConfig, DeliveryGuarantee, DeliveryParticipant, Edge, Epoch, ExpressionOperator,
    ExternalOperatorSpec, FormatIdentity, JsonMap, NodeOperator, ParticipantRole, PipelineBuilder,
    Port, PortEndpoint, ProviderRegistry, Result, RetentionClass, SecretHandle, SecretReference,
    SecretResolver, SecretResolverKind, SinkBinding as RuntimeSinkBinding, SinkRecovery,
    SourceBinding as RuntimeSourceBinding, SourceCapabilities, SourceEvent, SourceSchema,
    SqlOperator, StreamExecutionPlan, StreamRequirements, StreamSink, StreamSource,
    TransactionSupport, TransactionalStreamSink, UdfKind, UdfReference, UdfRegistrySnapshot,
    UnionOperator, WatermarkPolicy, WindowAggregateOperator, WindowSpec,
    validate_delivery_guarantee, validate_selected_udfs,
};

pub const PROJECT_FORMAT_VERSION: u32 = 3;

const MAX_INPUT_BYTES: usize = 10 * 1024 * 1024;
const MAX_ROWS: usize = 100_000;
const MAX_TIMEOUT_SECONDS: u64 = 30;
const MIN_MEMORY_MB: usize = 64;
const MAX_MEMORY_MB: usize = 2_048;
const MAX_OUTPUT_ROWS: usize = 10_000;
const MAX_BATCH_SIZE: usize = 1_000_000;
const MAX_TARGET_PARTITIONS: usize = 256;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectSpec {
    #[serde(deserialize_with = "deserialize_format_version")]
    pub format_version: u32,
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub runtime: RuntimeSpec,
    pub graph: PipelineSpec,
    #[serde(default)]
    pub data_sources: Vec<DataSourceSpec>,
    #[serde(default)]
    pub sources: Vec<ProjectSourceBinding>,
    #[serde(default)]
    pub sinks: Vec<ProjectSinkBinding>,
    #[serde(default)]
    pub state: StateConfig,
}

/// Runtime mode and mode-specific limits for a project-v3 document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(
    tag = "mode",
    content = "options",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum RuntimeSpec {
    /// A bounded batch run using inline data fixtures.
    Batch(#[serde(default)] RunOptions),
    /// A continuous run using registered connector bindings.
    Stream(#[serde(default)] StreamRunOptions),
}

impl Default for RuntimeSpec {
    fn default() -> Self {
        Self::Batch(RunOptions::default())
    }
}

/// Continuous runtime limits carried by project v3.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct StreamRunOptions {
    pub checkpoint_interval_ms: u64,
    pub max_batch_rows: usize,
    pub max_batch_bytes: usize,
}

impl Default for StreamRunOptions {
    fn default() -> Self {
        Self {
            checkpoint_interval_ms: 30_000,
            max_batch_rows: 10_000,
            max_batch_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PipelineSpec {
    pub name: String,
    pub nodes: Vec<NodeSpec>,
    #[serde(default)]
    pub edges: Vec<EdgeSpec>,
    #[serde(default)]
    pub datafusion: DataFusionConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeSpec {
    pub id: String,
    pub operator: OperatorSpec,
    #[serde(default)]
    pub input_ports: Vec<PortSpec>,
    #[serde(default)]
    pub output_ports: Vec<PortSpec>,
    #[serde(default)]
    pub position: Option<PositionSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OperatorSpec {
    Expression {
        #[serde(default)]
        expression: String,
        #[serde(default)]
        select: Vec<String>,
        #[serde(default)]
        filter: Option<String>,
        #[serde(default)]
        udfs: Vec<UdfReference>,
    },
    Sql {
        query: String,
        aliases: Vec<String>,
        #[serde(default)]
        udfs: Vec<UdfReference>,
    },
    /// Same-schema multi-input forwarding for stream graphs.
    Union,
    /// Stateful event-time aggregation for stream graphs.
    Window { spec: WindowSpec },
    External {
        provider: String,
        name: String,
        version: String,
        #[serde(default)]
        options: JsonMap,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PortSpec {
    pub name: String,
    pub kind: BatchKind,
    #[serde(default = "default_true")]
    pub required: bool,
    #[serde(default)]
    pub schema: Vec<ArrowFieldSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArrowFieldSpec {
    pub name: String,
    pub data_type: String,
    #[serde(default = "default_true")]
    pub nullable: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeSpec {
    pub source_node: String,
    #[serde(default = "default_output")]
    pub source_port: String,
    pub target_node: String,
    #[serde(default = "default_input")]
    pub target_port: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PositionSpec {
    #[serde(deserialize_with = "deserialize_finite")]
    pub x: f64,
    #[serde(deserialize_with = "deserialize_finite")]
    pub y: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DataSourceSpec {
    pub id: String,
    pub input: String,
    pub format: String,
    pub data: Value,
}

/// A data-only source binding to one registered connector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectSourceBinding {
    /// External graph input this binding feeds.
    pub binding: String,
    /// Exact registered connector identity.
    pub connector: ConnectorRef,
    /// Optional exact wire-format identity.
    #[serde(default)]
    pub format: Option<FormatRef>,
    /// Bounded non-secret connector options.
    #[serde(default)]
    pub options: JsonMap,
    /// Named secret references resolved only when the connector opens.
    #[serde(default)]
    pub secrets: BTreeMap<String, SecretReference>,
    /// Optional event-time watermark policy.
    #[serde(default)]
    pub watermark: Option<ProjectWatermarkPolicy>,
    /// Optional exact Arrow schema.
    #[serde(default)]
    pub schema: Vec<ArrowFieldSpec>,
}

/// A data-only sink binding to one registered connector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectSinkBinding {
    /// External graph output this binding drains.
    pub binding: String,
    /// Exact registered connector identity.
    pub connector: ConnectorRef,
    /// Optional exact wire-format identity.
    #[serde(default)]
    pub format: Option<FormatRef>,
    /// Bounded non-secret connector options.
    #[serde(default)]
    pub options: JsonMap,
    /// Named secret references resolved only when the connector opens.
    #[serde(default)]
    pub secrets: BTreeMap<String, SecretReference>,
    /// Requested delivery guarantee for this output.
    pub delivery: DeliveryRequest,
}

/// Exact identity of a connector implementation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConnectorRef {
    pub provider: String,
    pub name: String,
    pub version: String,
}

/// Exact identity of a registered format codec.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FormatRef {
    pub name: String,
    pub version: String,
}

/// Requested delivery guarantee for one project output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryRequest {
    BestEffort,
    AtLeastOnce,
    ExactlyOnce,
}

/// Event-time policy carried by one project source binding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "policy", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProjectWatermarkPolicy {
    SourceProvided,
    BoundedOutOfOrderness {
        column: String,
        delay_ms: u64,
        #[serde(default)]
        emit_interval_ms: u64,
        #[serde(default)]
        idle_timeout_ms: Option<u64>,
    },
    Disabled,
}

/// Managed state location and checkpoint retention for stream projects.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct StateConfig {
    pub root: String,
    pub retention: usize,
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            root: ".calc-flow-state".into(),
            retention: 3,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct RunOptions {
    pub max_input_bytes: usize,
    pub max_rows: usize,
    pub timeout_seconds: u64,
    pub memory_limit_mb: usize,
    pub output_rows: usize,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            max_input_bytes: MAX_INPUT_BYTES,
            max_rows: MAX_ROWS,
            timeout_seconds: MAX_TIMEOUT_SECONDS,
            memory_limit_mb: 512,
            output_rows: 1_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidationIssue {
    pub path: String,
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidationReport {
    pub valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub fingerprint: Option<String>,
}

/// Generates the single canonical JSON Schema for project format v3.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] if the generated schema cannot be encoded.
pub fn project_json_schema() -> Result<Value> {
    let mut value =
        serde_json::to_value(schema_for!(ProjectSpec)).map_err(|error| CalcFlowError::Format {
            message: error.to_string(),
        })?;
    value["title"] = Value::String("Calc Flow Project V3".into());
    value["properties"]["format_version"] = json!({
        "const": PROJECT_FORMAT_VERSION,
        "type": "integer"
    });
    Ok(value)
}

/// Validates both deserialized and directly constructed project models.
pub fn validate_project(
    project: &ProjectSpec,
    providers: &ProviderRegistry,
    udfs: &UdfRegistrySnapshot,
) -> ValidationReport {
    let issues = semantic_issues(project, providers, udfs, CompileMode::Batch);
    if !issues.is_empty() {
        return ValidationReport {
            valid: false,
            issues,
            fingerprint: None,
        };
    }
    let mut issues = Vec::new();
    let mut fingerprint = None;
    match build_project(project, providers, udfs) {
        Ok(plan) => {
            validate_source_coverage(
                project,
                plan.external_inputs().keys().map(String::as_str),
                CompileMode::Batch,
                &mut issues,
            );
            fingerprint = issues.is_empty().then(|| plan.fingerprint().into());
        }
        Err(error) => issues.push(issue(
            graph_error_path(project, &error),
            "graph_compile",
            error.to_string(),
        )),
    }
    ValidationReport {
        valid: issues.is_empty(),
        issues,
        fingerprint,
    }
}

/// Compiles a strict, data-only project through the public pipeline builder.
///
/// # Errors
///
/// Returns a stable validation error or the underlying graph compilation
/// failure. No executable value is read from project configuration.
pub fn compile_project(
    project: &ProjectSpec,
    providers: &ProviderRegistry,
    udfs: &UdfRegistrySnapshot,
) -> Result<BatchExecutionPlan> {
    let issues = semantic_issues(project, providers, udfs, CompileMode::Batch);
    if let Some(first) = issues.first() {
        return Err(validation_error(first));
    }
    let plan =
        build_project_builder(project, providers, CompileMode::Batch)?.compile_batch(udfs)?;
    let mut coverage = Vec::new();
    validate_source_coverage(
        project,
        plan.external_inputs().keys().map(String::as_str),
        CompileMode::Batch,
        &mut coverage,
    );
    if let Some(first) = coverage.first() {
        return Err(validation_error(first));
    }
    Ok(plan)
}

/// Compiles a strict, data-only project into a continuous stream plan.
///
/// # Errors
///
/// Returns a stable validation error or the underlying stream graph
/// compilation failure. External operators must provide a stream factory.
pub fn compile_stream_project(
    project: &ProjectSpec,
    providers: &ProviderRegistry,
    udfs: &UdfRegistrySnapshot,
    connectors: &ConnectorRegistrySnapshot,
    requirements: &StreamRequirements,
) -> Result<StreamExecutionPlan> {
    let issues = semantic_issues(project, providers, udfs, CompileMode::Stream);
    if let Some(first) = issues.first() {
        return Err(validation_error(first));
    }
    let requirements = project_stream_requirements(project, requirements)?;
    let plan = build_project_builder(project, providers, CompileMode::Stream)?
        .compile_stream(udfs, &requirements)?;
    let mut coverage = Vec::new();
    validate_source_coverage(
        project,
        plan.source_binding_ids(),
        CompileMode::Stream,
        &mut coverage,
    );
    validate_sink_coverage(project, plan.sink_binding_ids(), &mut coverage);
    if let Some(first) = coverage.first() {
        return Err(validation_error(first));
    }
    let connector_issues = validate_project_connectors(project, connectors, &requirements, &plan);
    if let Some(first) = connector_issues.first() {
        return Err(validation_error(first));
    }
    let project_json = serde_json::to_value(project).map_err(|error| CalcFlowError::Format {
        message: error.to_string(),
    })?;
    let project_fingerprint = crate::canonical_json(&project_json)?;
    let (sources, sinks) = build_project_runtime_bindings(project, connectors)?;
    Ok(plan
        .with_project_fingerprint(&project_fingerprint)
        .with_project_bindings(sources, sinks))
}

fn project_stream_requirements(
    project: &ProjectSpec,
    supplied: &StreamRequirements,
) -> Result<StreamRequirements> {
    let mut delivery = BTreeMap::new();
    for (index, sink) in project.sinks.iter().enumerate() {
        let guarantee = match sink.delivery {
            DeliveryRequest::BestEffort => DeliveryGuarantee::BestEffort,
            DeliveryRequest::AtLeastOnce => DeliveryGuarantee::AtLeastOnce,
            DeliveryRequest::ExactlyOnce => DeliveryGuarantee::ExactlyOnce,
        };
        if delivery.insert(sink.binding.clone(), guarantee).is_some() {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sinks[{index}].binding"),
                message: "sink bindings must be unique".into(),
            });
        }
    }
    for (output, requested) in &supplied.delivery {
        match delivery.get(output) {
            Some(project_request) if project_request == requested => {}
            Some(_) => {
                return Err(CalcFlowError::InvalidArgument {
                    field: format!("sinks.{output}.delivery"),
                    message: "project sink delivery disagrees with the compile requirement".into(),
                });
            }
            None => {
                return Err(CalcFlowError::InvalidArgument {
                    field: format!("requirements.delivery.{output}"),
                    message: "delivery requirement names an unbound project output".into(),
                });
            }
        }
    }
    Ok(StreamRequirements { delivery })
}

/// Compiles only the graph portion of a stream-mode project.
///
/// This preserves the public A6 host-provided source/sink workflow. Project-v3
/// connector documents must use [`compile_stream_project`] so connector,
/// secret, format, and delivery preflight cannot be bypassed.
///
/// # Errors
///
/// Returns a stable validation or graph compilation error. Connector bindings
/// are rejected because this entry point leaves source and sink ownership to
/// the host runtime.
#[doc(hidden)]
pub fn compile_stream_project_graph(
    project: &ProjectSpec,
    providers: &ProviderRegistry,
    udfs: &UdfRegistrySnapshot,
    requirements: &StreamRequirements,
) -> Result<StreamExecutionPlan> {
    let issues = semantic_issues(project, providers, udfs, CompileMode::Stream);
    if let Some(first) = issues.first() {
        return Err(validation_error(first));
    }
    if !project.sources.is_empty() || !project.sinks.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: "sources".into(),
            message: "host-provided stream compilation rejects connector bindings".into(),
        });
    }
    build_project_builder(project, providers, CompileMode::Stream)?
        .compile_stream(udfs, requirements)
}

fn validate_project_connectors(
    project: &ProjectSpec,
    connectors: &ConnectorRegistrySnapshot,
    requirements: &StreamRequirements,
    plan: &StreamExecutionPlan,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    for (index, binding) in project.sources.iter().enumerate() {
        let base = format!("sources[{index}]");
        let Some(identity) = connector_identity(&binding.connector, &base, &mut issues) else {
            continue;
        };
        let factory = match connectors.resolve_source(&identity) {
            Ok(factory) => factory,
            Err(error) => {
                issues.push(issue(
                    format!("{base}.connector"),
                    "missing_connector",
                    error.to_string(),
                ));
                continue;
            }
        };
        validate_project_connector_config(
            &base,
            factory.descriptor(),
            factory.validate(&binding.options),
            binding.format.as_ref(),
            &binding.secrets,
            connectors,
            &mut issues,
        );
    }
    for (index, binding) in project.sinks.iter().enumerate() {
        let base = format!("sinks[{index}]");
        let Some(identity) = connector_identity(&binding.connector, &base, &mut issues) else {
            continue;
        };
        let factory = match connectors.resolve_sink(&identity) {
            Ok(factory) => factory,
            Err(error) => {
                issues.push(issue(
                    format!("{base}.connector"),
                    "missing_connector",
                    error.to_string(),
                ));
                continue;
            }
        };
        validate_project_connector_config(
            &base,
            factory.descriptor(),
            factory.validate(&binding.options),
            binding.format.as_ref(),
            &binding.secrets,
            connectors,
            &mut issues,
        );
    }
    validate_project_delivery(project, connectors, requirements, plan, &mut issues);
    issues
}

// Delivery validation is an exhaustive capability cross-product whose issues
// must be accumulated deterministically in sink order.
// #lizard forgives
fn validate_project_delivery(
    project: &ProjectSpec,
    connectors: &ConnectorRegistrySnapshot,
    requirements: &StreamRequirements,
    plan: &StreamExecutionPlan,
    issues: &mut Vec<ValidationIssue>,
) {
    for (sink_index, binding) in project.sinks.iter().enumerate() {
        let requested = match binding.delivery {
            DeliveryRequest::BestEffort => DeliveryGuarantee::BestEffort,
            DeliveryRequest::AtLeastOnce => DeliveryGuarantee::AtLeastOnce,
            DeliveryRequest::ExactlyOnce => DeliveryGuarantee::ExactlyOnce,
        };
        if let Some(requirement) = requirements.delivery.get(&binding.binding) {
            if *requirement != requested {
                issues.push(issue(
                    format!("sinks[{sink_index}].delivery"),
                    "delivery_mismatch",
                    "project sink delivery disagrees with the compile requirement",
                ));
                continue;
            }
        }
        let mut participants = Vec::new();
        let reachable_sources = plan.reachable_source_binding_ids(&binding.binding);
        for (source_index, source) in project.sources.iter().enumerate() {
            if !reachable_sources.contains(source.binding.as_str()) {
                continue;
            }
            let Ok(identity) = ConnectorIdentity::new(
                &source.connector.provider,
                &source.connector.name,
                &source.connector.version,
            ) else {
                continue;
            };
            if let Ok(factory) = connectors.resolve_source(&identity) {
                if let Ok(capabilities) = factory.capabilities(&source.options) {
                    participants.push(DeliveryParticipant {
                        path: format!("sources[{source_index}]"),
                        role: ParticipantRole::Source,
                        capabilities,
                    });
                }
            }
        }
        let Ok(identity) = ConnectorIdentity::new(
            &binding.connector.provider,
            &binding.connector.name,
            &binding.connector.version,
        ) else {
            continue;
        };
        if let Ok(factory) = connectors.resolve_sink(&identity) {
            if let Ok(capabilities) = factory.capabilities(&binding.options) {
                participants.push(DeliveryParticipant {
                    path: format!("sinks[{sink_index}]"),
                    role: ParticipantRole::Sink,
                    capabilities,
                });
            }
        }
        if let Err(error) = validate_delivery_guarantee(requested, &participants) {
            issues.push(issue(
                format!("sinks[{sink_index}].delivery"),
                "unsupported_delivery",
                error.to_string(),
            ));
        }
    }
}

fn connector_identity(
    reference: &ConnectorRef,
    base: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Option<ConnectorIdentity> {
    match ConnectorIdentity::new(&reference.provider, &reference.name, &reference.version) {
        Ok(identity) => Some(identity),
        Err(error) => {
            issues.push(issue(
                format!("{base}.connector"),
                "invalid_connector",
                error.to_string(),
            ));
            None
        }
    }
}

// Connector config validation intentionally accumulates independent option,
// format, and secret errors rather than returning only the first failure.
// #lizard forgives
fn validate_project_connector_config(
    base: &str,
    descriptor: &crate::ConnectorDescriptor,
    option_validation: Result<()>,
    format: Option<&FormatRef>,
    secrets: &BTreeMap<String, SecretReference>,
    connectors: &ConnectorRegistrySnapshot,
    issues: &mut Vec<ValidationIssue>,
) {
    if let Err(error) = option_validation {
        issues.push(issue(
            format!("{base}.options"),
            "invalid_connector_options",
            error.to_string(),
        ));
    }
    for slot in secrets.keys() {
        if !descriptor.secret_slots.contains(slot) {
            issues.push(issue(
                format!("{base}.secrets.{slot}"),
                "unknown_secret_slot",
                "connector does not declare this secret slot",
            ));
        }
    }
    for slot in &descriptor.required_secret_slots {
        if !secrets.contains_key(slot) {
            issues.push(issue(
                format!("{base}.secrets.{slot}"),
                "missing_secret_slot",
                "connector requires this secret reference",
            ));
        }
    }
    let Some(format) = format else {
        return;
    };
    let identity = match FormatIdentity::new(&format.name, &format.version) {
        Ok(identity) => identity,
        Err(error) => {
            issues.push(issue(
                format!("{base}.format"),
                "invalid_format",
                error.to_string(),
            ));
            return;
        }
    };
    if let Err(error) = connectors.resolve_format(&identity) {
        issues.push(issue(
            format!("{base}.format"),
            "missing_format",
            error.to_string(),
        ));
    } else if !descriptor.formats.contains(&identity) {
        issues.push(issue(
            format!("{base}.format"),
            "unsupported_format",
            "connector does not support this registered format",
        ));
    }
}

type RuntimeProjectBindings = (
    BTreeMap<String, RuntimeSourceBinding>,
    BTreeMap<String, Vec<RuntimeSinkBinding>>,
);

fn build_project_runtime_bindings(
    project: &ProjectSpec,
    connectors: &ConnectorRegistrySnapshot,
) -> Result<RuntimeProjectBindings> {
    let RuntimeSpec::Stream(options) = &project.runtime else {
        return Err(CalcFlowError::InvalidArgument {
            field: "runtime.mode".into(),
            message: "connector bindings require stream mode".into(),
        });
    };
    Ok((
        build_project_runtime_sources(project, connectors, options)?,
        build_project_runtime_sinks(project, connectors)?,
    ))
}

fn build_project_runtime_sources(
    project: &ProjectSpec,
    connectors: &ConnectorRegistrySnapshot,
    options: &StreamRunOptions,
) -> Result<BTreeMap<String, RuntimeSourceBinding>> {
    let mut sources = BTreeMap::new();
    for binding in &project.sources {
        let identity = ConnectorIdentity::new(
            &binding.connector.provider,
            &binding.connector.name,
            &binding.connector.version,
        )?;
        let factory = connectors.resolve_source(&identity)?;
        let capabilities = source_capabilities(factory.as_ref(), binding, options)?;
        let source = DeferredProjectSource {
            factory,
            options: binding.options.clone(),
            secrets: ProjectBindingSecretResolver::new(
                binding.secrets.clone(),
                connectors.registered_secret_resolver(),
            ),
            capabilities,
            inner: None,
        };
        let mut runtime_binding = RuntimeSourceBinding::new(source);
        if let Some(policy) = &binding.watermark {
            runtime_binding =
                runtime_binding.with_watermark_policy(runtime_watermark_policy(policy));
        }
        sources.insert(binding.binding.clone(), runtime_binding);
    }
    Ok(sources)
}

// Runtime sink construction keeps identity, factory, delivery, format, and
// secret binding checks in one fail-closed conversion boundary.
// #lizard forgives
fn build_project_runtime_sinks(
    project: &ProjectSpec,
    connectors: &ConnectorRegistrySnapshot,
) -> Result<BTreeMap<String, Vec<RuntimeSinkBinding>>> {
    let mut sinks = BTreeMap::<String, Vec<RuntimeSinkBinding>>::new();
    for (index, binding) in project.sinks.iter().enumerate() {
        let identity = ConnectorIdentity::new(
            &binding.connector.provider,
            &binding.connector.name,
            &binding.connector.version,
        )?;
        let factory = connectors.resolve_sink(&identity)?;
        let sink_id = format!("{}-{index}", binding.binding);
        let secrets = ProjectBindingSecretResolver::new(
            binding.secrets.clone(),
            connectors.registered_secret_resolver(),
        );
        let transaction = factory.capabilities(&binding.options)?.transaction;
        let runtime_binding = match (binding.delivery, transaction) {
            (
                DeliveryRequest::BestEffort | DeliveryRequest::AtLeastOnce,
                TransactionSupport::None,
            ) => RuntimeSinkBinding::ordinary(
                &sink_id,
                DeferredProjectSink {
                    factory,
                    options: binding.options.clone(),
                    secrets,
                    inner: None,
                },
            )?,
            (_, TransactionSupport::PreCommitCommit) => RuntimeSinkBinding::transactional(
                &sink_id,
                DeferredProjectTransactionalSink {
                    factory,
                    options: binding.options.clone(),
                    secrets,
                    inner: None,
                },
            )?,
            (_, TransactionSupport::LedgerIdempotent) => RuntimeSinkBinding::epoch_idempotent(
                &sink_id,
                DeferredProjectTransactionalSink {
                    factory,
                    options: binding.options.clone(),
                    secrets,
                    inner: None,
                },
                "connector-ledger",
                RetentionClass::Unbounded,
            )?,
            (_, TransactionSupport::RetryDeduplicated) => RuntimeSinkBinding::epoch_idempotent(
                &sink_id,
                DeferredProjectTransactionalSink {
                    factory,
                    options: binding.options.clone(),
                    secrets,
                    inner: None,
                },
                "connector-retry-deduplicated",
                RetentionClass::Bounded,
            )?,
            (DeliveryRequest::ExactlyOnce, TransactionSupport::None) => {
                return Err(CalcFlowError::Compile {
                    message: format!(
                        "sink {:?} requests exactly-once but its connector has no transaction support",
                        binding.binding
                    ),
                });
            }
        };
        sinks
            .entry(binding.binding.clone())
            .or_default()
            .push(runtime_binding);
    }
    Ok(sinks)
}

fn source_capabilities(
    factory: &dyn ConnectorSourceFactory,
    binding: &ProjectSourceBinding,
    options: &StreamRunOptions,
) -> Result<SourceCapabilities> {
    let schema = if binding.schema.is_empty() {
        SourceSchema::DynamicOrUnknown
    } else {
        let fields = binding
            .schema
            .iter()
            .map(|field| {
                arrow_data_type(&field.data_type)
                    .map(|data_type| Field::new(&field.name, data_type, field.nullable))
                    .ok_or_else(|| CalcFlowError::InvalidArgument {
                        field: "sources.schema.data_type".into(),
                        message: format!("unsupported Arrow type {:?}", field.data_type),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        SourceSchema::Exact(Arc::new(datafusion::arrow::datatypes::Schema::new(fields)))
    };
    let capabilities = factory.capabilities(&binding.options)?;
    Ok(SourceCapabilities {
        replay_positioning: capabilities.replay_positioning(),
        delivery: capabilities.source_delivery(),
        max_batch_rows: options.max_batch_rows,
        max_batch_bytes: options.max_batch_bytes,
        schema,
        native_watermarks: capabilities.native_watermarks(),
    })
}

fn runtime_watermark_policy(policy: &ProjectWatermarkPolicy) -> WatermarkPolicy {
    match policy {
        ProjectWatermarkPolicy::SourceProvided => WatermarkPolicy::SourceProvided,
        ProjectWatermarkPolicy::BoundedOutOfOrderness {
            column,
            delay_ms,
            emit_interval_ms,
            idle_timeout_ms,
        } => WatermarkPolicy::BoundedOutOfOrderness {
            event_time_column: column.clone(),
            max_out_of_orderness: Duration::from_millis(*delay_ms),
            emit_interval: Duration::from_millis(*emit_interval_ms),
            idle_timeout: idle_timeout_ms.map(Duration::from_millis),
        },
        ProjectWatermarkPolicy::Disabled => WatermarkPolicy::Disabled { idle_timeout: None },
    }
}

struct ProjectBindingSecretResolver {
    references: BTreeMap<String, SecretReference>,
    registered: Option<Arc<dyn SecretResolver>>,
}

impl ProjectBindingSecretResolver {
    fn new(
        references: BTreeMap<String, SecretReference>,
        registered: Option<Arc<dyn SecretResolver>>,
    ) -> Self {
        Self {
            references,
            registered,
        }
    }
}

impl SecretResolver for ProjectBindingSecretResolver {
    fn resolve(&self, requested: &SecretReference) -> Result<SecretHandle> {
        let reference =
            self.references
                .get(&requested.key)
                .ok_or_else(|| CalcFlowError::NotFound {
                    resource: "connector secret slot".into(),
                    key: requested.key.clone(),
                })?;
        match reference.resolver {
            SecretResolverKind::Environment => std::env::var(&reference.key)
                .map(|value| SecretHandle::from_bytes(value.as_bytes()))
                .map_err(|_| CalcFlowError::NotFound {
                    resource: "secret".into(),
                    key: reference.key.clone(),
                }),
            SecretResolverKind::File => std::fs::read(&reference.key)
                .map(|value| SecretHandle::from_bytes(&value))
                .map_err(|source| CalcFlowError::Io {
                    path: reference.key.clone(),
                    source,
                }),
            SecretResolverKind::Registered => self
                .registered
                .as_deref()
                .ok_or_else(|| CalcFlowError::NotFound {
                    resource: "registered secret resolver".into(),
                    key: reference.key.clone(),
                })?
                .resolve(reference),
        }
    }
}

struct DeferredProjectSource {
    factory: Arc<dyn ConnectorSourceFactory>,
    options: JsonMap,
    secrets: ProjectBindingSecretResolver,
    capabilities: SourceCapabilities,
    inner: Option<Box<dyn StreamSource>>,
}

#[async_trait]
impl StreamSource for DeferredProjectSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        let mut source = self.factory.open(&self.options, &self.secrets).await?;
        if let Err(error) = source.open(cursor).await {
            let _cleanup = source.close().await;
            return Err(error);
        }
        self.inner = Some(source);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.inner
            .as_mut()
            .ok_or_else(|| CalcFlowError::Internal {
                message: "project source was polled before open".into(),
            })?
            .next()
            .await
    }

    async fn close(&mut self) -> Result<()> {
        match self.inner.as_mut() {
            Some(source) => source.close().await,
            None => Ok(()),
        }
    }
}

struct DeferredProjectSink {
    factory: Arc<dyn ConnectorSinkFactory>,
    options: JsonMap,
    secrets: ProjectBindingSecretResolver,
    inner: Option<Box<dyn StreamSink>>,
}

#[async_trait]
impl StreamSink for DeferredProjectSink {
    async fn open(&mut self) -> Result<()> {
        let mut sink = self.factory.open(&self.options, &self.secrets).await?;
        if let Err(error) = sink.open().await {
            let _cleanup = sink.close().await;
            return Err(error);
        }
        self.inner = Some(sink);
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.inner
            .as_mut()
            .ok_or_else(|| CalcFlowError::Internal {
                message: "project sink was written before open".into(),
            })?
            .write(batch)
            .await
    }

    async fn close(&mut self) -> Result<()> {
        match self.inner.as_mut() {
            Some(sink) => sink.close().await,
            None => Ok(()),
        }
    }
}

struct DeferredProjectTransactionalSink {
    factory: Arc<dyn ConnectorSinkFactory>,
    options: JsonMap,
    secrets: ProjectBindingSecretResolver,
    inner: Option<Box<dyn TransactionalStreamSink>>,
}

impl DeferredProjectTransactionalSink {
    fn inner(&mut self) -> Result<&mut Box<dyn TransactionalStreamSink>> {
        self.inner.as_mut().ok_or_else(|| CalcFlowError::Internal {
            message: "project transactional sink was used before open".into(),
        })
    }
}

#[async_trait]
impl TransactionalStreamSink for DeferredProjectTransactionalSink {
    async fn open(&mut self) -> Result<()> {
        let mut sink = self
            .factory
            .open_transactional(&self.options, &self.secrets)
            .await?
            .ok_or_else(|| CalcFlowError::Compile {
                message: "connector declared transaction support without a transactional factory"
                    .into(),
            })?;
        if let Err(error) = sink.open().await {
            let _cleanup = sink.close().await;
            return Err(error);
        }
        self.inner = Some(sink);
        Ok(())
    }

    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()> {
        self.inner()?.begin_epoch(epoch).await
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.inner()?.write(batch).await
    }

    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap> {
        self.inner()?.pre_commit(epoch).await
    }

    async fn commit(&mut self, epoch: Epoch, pre_commit: &JsonMap) -> Result<()> {
        self.inner()?.commit(epoch, pre_commit).await
    }

    async fn abort(&mut self, epoch: Epoch, pre_commit: Option<&JsonMap>) -> Result<()> {
        self.inner()?.abort(epoch, pre_commit).await
    }

    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        self.inner()?.recover(recovery).await
    }

    async fn close(&mut self) -> Result<()> {
        match self.inner.as_mut() {
            Some(sink) => sink.close().await,
            None => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompileMode {
    Batch,
    Stream,
}

fn build_project(
    project: &ProjectSpec,
    providers: &ProviderRegistry,
    udfs: &UdfRegistrySnapshot,
) -> Result<BatchExecutionPlan> {
    build_project_builder(project, providers, CompileMode::Batch)?.compile_batch(udfs)
}

fn build_project_builder(
    project: &ProjectSpec,
    providers: &ProviderRegistry,
    mode: CompileMode,
) -> Result<PipelineBuilder> {
    let builder =
        PipelineBuilder::new(&project.graph.name)?.with_datafusion_config(project.graph.datafusion);
    let builder = add_project_nodes(builder, project, providers, mode)?;
    add_project_edges(builder, project)
}

fn add_project_nodes(
    mut builder: PipelineBuilder,
    project: &ProjectSpec,
    providers: &ProviderRegistry,
    mode: CompileMode,
) -> Result<PipelineBuilder> {
    for node in &project.graph.nodes {
        let operator = project_node_operator(node, providers, mode)?;
        builder = builder.add_node(&node.id, operator)?;
    }
    Ok(builder)
}

fn add_project_edges(
    mut builder: PipelineBuilder,
    project: &ProjectSpec,
) -> Result<PipelineBuilder> {
    for edge in &project.graph.edges {
        builder = builder.connect(Edge::new(
            PortEndpoint::new(&edge.source_node, &edge.source_port)?,
            PortEndpoint::new(&edge.target_node, &edge.target_port)?,
        ))?;
    }
    Ok(builder)
}

fn project_node_operator(
    node: &NodeSpec,
    providers: &ProviderRegistry,
    mode: CompileMode,
) -> Result<NodeOperator> {
    let inputs = configured_ports(node, true)?;
    let outputs = configured_ports(node, false)?;
    match &node.operator {
        OperatorSpec::Expression {
            expression,
            select,
            filter,
            udfs: references,
        } => expression_node(
            node,
            inputs,
            outputs,
            expression,
            select,
            filter.as_deref(),
            references,
        ),
        OperatorSpec::Sql {
            query,
            aliases,
            udfs: references,
        } => sql_node(node, inputs, outputs, query, aliases, references),
        OperatorSpec::Union => union_node(node, inputs, &outputs, mode),
        OperatorSpec::Window { spec } => window_node(node, &inputs, &outputs, spec, mode),
        OperatorSpec::External {
            provider,
            name,
            version,
            options,
        } => external_node(
            providers,
            mode,
            inputs,
            outputs,
            (provider, name, version),
            options,
        ),
    }
}

fn union_node(
    node: &NodeSpec,
    inputs: Vec<Port>,
    outputs: &[Port],
    mode: CompileMode,
) -> Result<NodeOperator> {
    if mode != CompileMode::Stream {
        return Err(CalcFlowError::Compile {
            message: format!("node {:?} uses stream-only union", node.id),
        });
    }
    let operator = UnionOperator::new(&node.id, inputs)?;
    validate_derived_outputs(outputs, crate::OperatorMetadata::output_ports(&operator))?;
    Ok(NodeOperator::Union(operator))
}

// Window construction validates the complete stream-only operator contract
// before exposing the node to the compiled plan.
// #lizard forgives
fn window_node(
    node: &NodeSpec,
    inputs: &[Port],
    outputs: &[Port],
    spec: &WindowSpec,
    mode: CompileMode,
) -> Result<NodeOperator> {
    if mode != CompileMode::Stream {
        return Err(CalcFlowError::Compile {
            message: format!("node {:?} uses a stream-only window", node.id),
        });
    }
    let [input] = inputs else {
        return Err(CalcFlowError::InvalidArgument {
            field: "node.input_ports".into(),
            message: "window operators require one explicit input port".into(),
        });
    };
    if input.name() != "input" || input.kind() != BatchKind::Table || !input.required() {
        return Err(CalcFlowError::InvalidArgument {
            field: "node.input_ports".into(),
            message: "window operators require one required table input named `input`".into(),
        });
    }
    let schema = input
        .schema()
        .cloned()
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: "node.input_ports[0].schema".into(),
            message: "window operators require an exact input schema".into(),
        })?;
    let operator = WindowAggregateOperator::new(&node.id, schema, spec.clone())?;
    validate_derived_outputs(outputs, crate::OperatorMetadata::output_ports(&operator))?;
    Ok(NodeOperator::Window(operator))
}

fn validate_derived_outputs(configured: &[Port], actual: &[Port]) -> Result<()> {
    if configured.is_empty() {
        return Ok(());
    }
    let matches = configured.len() == actual.len()
        && configured.iter().zip(actual).all(|(configured, actual)| {
            configured.name() == actual.name()
                && configured.kind() == actual.kind()
                && configured.required() == actual.required()
                && configured.schema() == actual.schema()
        });
    if matches {
        Ok(())
    } else {
        Err(CalcFlowError::InvalidArgument {
            field: "node.output_ports".into(),
            message: "configured output ports do not match the derived operator outputs".into(),
        })
    }
}

fn expression_node(
    node: &NodeSpec,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    expression: &str,
    select: &[String],
    filter: Option<&str>,
    references: &[UdfReference],
) -> Result<NodeOperator> {
    let (inputs, outputs) =
        builtin_ports(inputs, outputs, &["input"], &["output"], BatchKind::Table)?;
    Ok(NodeOperator::Expression(
        ExpressionOperator::new(
            &node.id,
            expression,
            select.to_vec(),
            filter.map(str::to_owned),
            references.to_vec(),
        )?
        .with_ports(
            inputs.into_iter().next().unwrap(),
            outputs.into_iter().next().unwrap(),
        )?,
    ))
}

fn sql_node(
    node: &NodeSpec,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    query: &str,
    aliases: &[String],
    references: &[UdfReference],
) -> Result<NodeOperator> {
    let expected = aliases.iter().map(String::as_str).collect::<Vec<_>>();
    let (inputs, outputs) =
        builtin_ports(inputs, outputs, &expected, &["output"], BatchKind::Table)?;
    Ok(NodeOperator::Sql(
        SqlOperator::new(&node.id, query, aliases.to_vec(), references.to_vec())?
            .with_ports(inputs, outputs.into_iter().next().unwrap())?,
    ))
}

fn external_node(
    providers: &ProviderRegistry,
    mode: CompileMode,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    identity: (&str, &str, &str),
    options: &JsonMap,
) -> Result<NodeOperator> {
    let (provider, name, version) = identity;
    let spec = ExternalOperatorSpec::new(provider, name, version, options.clone())?;
    match mode {
        CompileMode::Batch => Ok(NodeOperator::Batch(
            providers
                .resolve_batch(provider, name, version)?
                .create(&spec, inputs, outputs)?,
        )),
        CompileMode::Stream => Ok(NodeOperator::Stream(
            providers
                .resolve_stream(provider, name, version)?
                .create(&spec, inputs, outputs)?,
        )),
    }
}

fn configured_ports(node: &NodeSpec, inputs: bool) -> Result<Vec<Port>> {
    let ports = if inputs {
        &node.input_ports
    } else {
        &node.output_ports
    };
    ports.iter().map(port_from_spec).collect()
}

fn port_from_spec(spec: &PortSpec) -> Result<Port> {
    let fields = (!spec.schema.is_empty())
        .then(|| {
            spec.schema
                .iter()
                .map(field_from_spec)
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    Port::new(&spec.name, spec.kind, spec.required, fields)
}

fn field_from_spec(spec: &ArrowFieldSpec) -> Result<Field> {
    let data_type =
        arrow_data_type(&spec.data_type).ok_or_else(|| CalcFlowError::InvalidArgument {
            field: "port.schema.data_type".into(),
            message: format!("unsupported Arrow type {:?}", spec.data_type),
        })?;
    Ok(Field::new(&spec.name, data_type, spec.nullable))
}

fn builtin_ports(
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    input_names: &[&str],
    output_names: &[&str],
    kind: BatchKind,
) -> Result<(Vec<Port>, Vec<Port>)> {
    let inputs = if inputs.is_empty() {
        input_names
            .iter()
            .map(|name| Port::new(name, kind, true, None))
            .collect::<Result<Vec<_>>>()?
    } else {
        inputs
    };
    let outputs = if outputs.is_empty() {
        output_names
            .iter()
            .map(|name| Port::new(name, kind, true, None))
            .collect::<Result<Vec<_>>>()?
    } else {
        outputs
    };
    Ok((inputs, outputs))
}

fn semantic_issues(
    project: &ProjectSpec,
    providers: &ProviderRegistry,
    udfs: &UdfRegistrySnapshot,
    mode: CompileMode,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    if project.format_version != PROJECT_FORMAT_VERSION {
        issues.push(issue(
            "format_version",
            "unsupported_version",
            format!(
                "project format version {} is unsupported; expected {PROJECT_FORMAT_VERSION}",
                project.format_version
            ),
        ));
    }
    validate_id(&project.id, "id", &mut issues);
    validate_required(&project.name, "name", 120, &mut issues);
    if project.description.len() > 2_000 {
        issues.push(issue(
            "description",
            "out_of_range",
            "must contain at most 2000 bytes",
        ));
    }
    validate_required(&project.graph.name, "graph.name", 120, &mut issues);
    if project.graph.nodes.is_empty() {
        issues.push(issue(
            "graph.nodes",
            "required",
            "graph requires at least one node",
        ));
    }
    if project_requires_datafusion(project) {
        if let Err(CalcFlowError::InvalidArgument { field, message }) =
            project.graph.datafusion.validate()
        {
            issues.push(issue(format!("graph.{field}"), "out_of_range", message));
        }
        validate_maximum(
            project.graph.datafusion.batch_size,
            MAX_BATCH_SIZE,
            "graph.datafusion.batch_size",
            &mut issues,
        );
        validate_maximum(
            project.graph.datafusion.target_partitions,
            MAX_TARGET_PARTITIONS,
            "graph.datafusion.target_partitions",
            &mut issues,
        );
    }
    validate_runtime(project, mode, &mut issues);
    validate_nodes(project, providers, udfs, mode, &mut issues);
    validate_edges(project, &mut issues);
    if mode == CompileMode::Batch {
        validate_sources(project, &mut issues);
    }
    issues
}

fn validate_runtime(project: &ProjectSpec, mode: CompileMode, issues: &mut Vec<ValidationIssue>) {
    match (&project.runtime, mode) {
        (RuntimeSpec::Batch(options), CompileMode::Batch) => {
            if !project.sources.is_empty() || !project.sinks.is_empty() {
                issues.push(issue(
                    "runtime.mode",
                    "mixed_runtime_inputs",
                    "batch projects reject connector source and sink bindings",
                ));
            }
            validate_run_options(options, issues);
        }
        (RuntimeSpec::Stream(options), CompileMode::Stream) => {
            if !project.data_sources.is_empty() {
                issues.push(issue(
                    "data_sources",
                    "mixed_runtime_inputs",
                    "stream projects reject inline data fixtures",
                ));
            }
            if options.checkpoint_interval_ms == 0 {
                issues.push(issue(
                    "runtime.options.checkpoint_interval_ms",
                    "out_of_range",
                    "must be greater than zero",
                ));
            }
            if options.max_batch_rows == 0 {
                issues.push(issue(
                    "runtime.options.max_batch_rows",
                    "out_of_range",
                    "must be greater than zero",
                ));
            }
            if options.max_batch_bytes == 0 {
                issues.push(issue(
                    "runtime.options.max_batch_bytes",
                    "out_of_range",
                    "must be greater than zero",
                ));
            }
        }
        (RuntimeSpec::Batch(_), CompileMode::Stream) => issues.push(issue(
            "runtime.mode",
            "runtime_mode_mismatch",
            "stream compilation requires runtime.mode to be stream",
        )),
        (RuntimeSpec::Stream(_), CompileMode::Batch) => issues.push(issue(
            "runtime.mode",
            "runtime_mode_mismatch",
            "batch compilation requires runtime.mode to be batch",
        )),
    }
}

fn project_requires_datafusion(project: &ProjectSpec) -> bool {
    project.graph.nodes.iter().any(|node| {
        matches!(
            node.operator,
            OperatorSpec::Expression { .. } | OperatorSpec::Sql { .. }
        )
    })
}

fn validate_nodes(
    project: &ProjectSpec,
    providers: &ProviderRegistry,
    udfs: &UdfRegistrySnapshot,
    mode: CompileMode,
    issues: &mut Vec<ValidationIssue>,
) {
    let mut node_ids = BTreeSet::new();
    let mut selected_udfs = Vec::new();
    for (node_index, node) in project.graph.nodes.iter().enumerate() {
        let base = format!("graph.nodes[{node_index}]");
        validate_id(&node.id, &format!("{base}.id"), issues);
        if !node_ids.insert(node.id.as_str()) {
            issues.push(issue(
                format!("{base}.id"),
                "duplicate_id",
                format!("duplicate node ID {:?}", node.id),
            ));
        }
        if let Some(position) = node.position {
            for (coordinate, value) in [("x", position.x), ("y", position.y)] {
                if !value.is_finite() {
                    issues.push(issue(
                        format!("{base}.position.{coordinate}"),
                        "not_finite",
                        "position coordinates must be finite",
                    ));
                }
            }
        }
        validate_ports(&node.input_ports, &format!("{base}.input_ports"), issues);
        validate_ports(&node.output_ports, &format!("{base}.output_ports"), issues);
        validate_operator(node, node_index, providers, udfs, mode, issues);
        selected_udfs.extend(operator_udfs(&node.operator).iter().cloned());
    }
    if validate_selected_udfs(&selected_udfs).is_err() {
        let path = project
            .graph
            .nodes
            .iter()
            .enumerate()
            .find(|(_, node)| !operator_udfs(&node.operator).is_empty())
            .map_or_else(
                || "graph.nodes".into(),
                |(index, _)| format!("graph.nodes[{index}].operator.udfs"),
            );
        issues.push(issue(
            path,
            "conflicting_udf",
            "selected native UDFs contain conflicting DataFusion SQL names",
        ));
    }
}

fn validate_operator(
    node: &NodeSpec,
    index: usize,
    providers: &ProviderRegistry,
    udfs: &UdfRegistrySnapshot,
    mode: CompileMode,
    issues: &mut Vec<ValidationIssue>,
) {
    let base = format!("graph.nodes[{index}].operator");
    let (expected_inputs, expected_outputs) = match &node.operator {
        OperatorSpec::Expression {
            expression,
            select,
            filter,
            udfs: references,
        } => {
            let invalid_mode = expression.trim().is_empty() == select.is_empty()
                || select.iter().any(|value| value.trim().is_empty());
            if invalid_mode {
                issues.push(issue(
                    &base,
                    "invalid_operator",
                    "expression requires exactly one expression or non-empty select list",
                ));
            } else if let Err(error) = expression_query(
                (!expression.trim().is_empty()).then_some(expression.as_str()),
                select,
                filter.as_deref(),
            ) {
                issues.push(issue(&base, "invalid_operator", error.to_string()));
            }
            validate_udfs(references, &base, udfs, issues);
            (Some(vec!["input"]), Some(vec!["output"]))
        }
        OperatorSpec::Sql {
            query,
            aliases,
            udfs: references,
        } => {
            if query.trim().is_empty() || aliases.is_empty() {
                issues.push(issue(
                    &base,
                    "invalid_operator",
                    "SQL requires a query and at least one alias",
                ));
            } else if let Err(error) = crate::expression::validate_select_query(query) {
                issues.push(issue(&base, "invalid_operator", error.to_string()));
            }
            let unique = aliases.iter().collect::<BTreeSet<_>>();
            if unique.len() != aliases.len() {
                issues.push(issue(
                    format!("{base}.aliases"),
                    "duplicate_alias",
                    "SQL aliases must be unique",
                ));
            }
            validate_udfs(references, &base, udfs, issues);
            (
                Some(aliases.iter().map(String::as_str).collect()),
                Some(vec!["output"]),
            )
        }
        OperatorSpec::Union => {
            validate_union_operator(node, index, mode, &base, issues);
            (None, Some(vec!["output"]))
        }
        OperatorSpec::Window { spec } => {
            validate_window_operator(node, index, spec, mode, &base, issues);
            (Some(vec!["input"]), Some(vec!["output"]))
        }
        OperatorSpec::External {
            provider,
            name,
            version,
            ..
        } => {
            validate_external_operator(providers, mode, provider, name, version, &base, issues);
            (None, None)
        }
    };
    if let Some(names) = expected_inputs {
        validate_builtin_port_specs(
            &node.input_ports,
            &names,
            &format!("graph.nodes[{index}].input_ports"),
            true,
            issues,
        );
    }
    if let Some(names) = expected_outputs {
        validate_builtin_port_specs(
            &node.output_ports,
            &names,
            &format!("graph.nodes[{index}].output_ports"),
            false,
            issues,
        );
    }
}

fn validate_union_operator(
    node: &NodeSpec,
    index: usize,
    mode: CompileMode,
    base: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    if mode != CompileMode::Stream {
        issues.push(issue(
            base,
            "incompatible_runtime",
            "union is available only in stream runtime mode",
        ));
    }
    if node.input_ports.len() < 2 {
        issues.push(issue(
            format!("graph.nodes[{index}].input_ports"),
            "invalid_ports",
            "union requires at least two explicit input ports",
        ));
    } else if let Ok(inputs) = configured_ports(node, true)
        && let Err(error) = UnionOperator::new(&node.id, inputs)
    {
        issues.push(issue(base, "invalid_operator", error.to_string()));
    }
}

// Validation mirrors window construction while accumulating stable field-level
// diagnostics, so the operator matrix remains intentionally centralized.
// #lizard forgives
fn validate_window_operator(
    node: &NodeSpec,
    index: usize,
    spec: &WindowSpec,
    mode: CompileMode,
    base: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    if mode != CompileMode::Stream {
        issues.push(issue(
            base,
            "incompatible_runtime",
            "window is available only in stream runtime mode",
        ));
    }
    let valid_input = matches!(
        node.input_ports.as_slice(),
        [input]
            if input.name == "input"
                && input.kind == BatchKind::Table
                && input.required
                && !input.schema.is_empty()
    );
    if !valid_input {
        issues.push(issue(
            format!("graph.nodes[{index}].input_ports"),
            "invalid_ports",
            "window requires one required table input named `input` with an exact schema",
        ));
    } else if let Ok(input) = port_from_spec(&node.input_ports[0])
        && let Some(schema) = input.schema().cloned()
        && let Err(error) = WindowAggregateOperator::new(&node.id, schema, spec.clone())
    {
        issues.push(issue(base, "invalid_operator", error.to_string()));
    }
}

fn validate_external_operator(
    providers: &ProviderRegistry,
    mode: CompileMode,
    provider: &str,
    name: &str,
    version: &str,
    base: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    match ExternalOperatorSpec::new(provider, name, version, BTreeMap::new()) {
        Ok(_) if !provider_available(providers, mode, provider, name, version) => {
            issues.push(issue(
                base,
                "missing_provider",
                format!("provider {provider}:{name}@{version} is unavailable"),
            ));
        }
        Err(error) => issues.push(issue(base, "invalid_operator", error.to_string())),
        Ok(_) => {}
    }
}

fn provider_available(
    providers: &ProviderRegistry,
    mode: CompileMode,
    provider: &str,
    name: &str,
    version: &str,
) -> bool {
    match mode {
        CompileMode::Batch => providers.resolve_batch(provider, name, version).is_ok(),
        CompileMode::Stream => providers.resolve_stream(provider, name, version).is_ok(),
    }
}

fn validate_udfs(
    references: &[UdfReference],
    base: &str,
    udfs: &UdfRegistrySnapshot,
    issues: &mut Vec<ValidationIssue>,
) {
    for (index, reference) in references.iter().enumerate() {
        let exists = udfs.catalog().iter().any(|entry| {
            entry.provider == reference.provider()
                && entry.name == reference.name()
                && entry.version == reference.version()
                && entry.kind == reference.kind()
        });
        let native_exists =
            reference.kind() != UdfKind::DataFusionScalar || udfs.resolve_native(reference).is_ok();
        if !exists || !native_exists {
            issues.push(issue(
                format!("{base}.udfs[{index}]"),
                "missing_udf",
                format!(
                    "UDF {}:{}@{} is unavailable",
                    reference.provider(),
                    reference.name(),
                    reference.version()
                ),
            ));
        }
    }
}

fn validate_builtin_port_specs(
    ports: &[PortSpec],
    expected_names: &[&str],
    path: &str,
    inputs_must_be_required: bool,
    issues: &mut Vec<ValidationIssue>,
) {
    if ports.is_empty() {
        return;
    }
    let actual = ports
        .iter()
        .map(|port| port.name.as_str())
        .collect::<Vec<_>>();
    if actual != expected_names
        || ports.iter().any(|port| port.kind != BatchKind::Table)
        || (inputs_must_be_required && ports.iter().any(|port| !port.required))
    {
        issues.push(issue(
            path,
            "invalid_ports",
            format!("must be table ports {expected_names:?} in order"),
        ));
    }
}

fn validate_ports(ports: &[PortSpec], path: &str, issues: &mut Vec<ValidationIssue>) {
    let mut port_names = BTreeSet::new();
    for (port_index, port) in ports.iter().enumerate() {
        let base = format!("{path}[{port_index}]");
        if !port_names.insert(port.name.as_str()) {
            issues.push(issue(
                format!("{base}.name"),
                "duplicate_port",
                format!("duplicate port {:?}", port.name),
            ));
        }
        if Port::new(&port.name, port.kind, port.required, None).is_err() {
            issues.push(issue(
                format!("{base}.name"),
                "invalid_port",
                "port name must be a portable SQL identifier",
            ));
        }
        if port.kind != BatchKind::Table && !port.schema.is_empty() {
            issues.push(issue(
                format!("{base}.schema"),
                "array_schema",
                "only table ports may declare an Arrow schema",
            ));
        }
        let mut field_names = BTreeSet::new();
        for (field_index, field) in port.schema.iter().enumerate() {
            let field_base = format!("{base}.schema[{field_index}]");
            if !field_names.insert(field.name.as_str()) {
                issues.push(issue(
                    format!("{field_base}.name"),
                    "duplicate_field",
                    format!("duplicate Arrow field {:?}", field.name),
                ));
            }
            if !is_port_identifier(&field.name) {
                issues.push(issue(
                    format!("{field_base}.name"),
                    "invalid_field",
                    "Arrow field name must be a portable SQL identifier",
                ));
            }
            if arrow_data_type(&field.data_type).is_none() {
                issues.push(issue(
                    format!("{field_base}.data_type"),
                    "unsupported_arrow_type",
                    format!("unsupported Arrow type {:?}", field.data_type),
                ));
            }
        }
    }
}

fn validate_edges(project: &ProjectSpec, issues: &mut Vec<ValidationIssue>) {
    let nodes = project
        .graph
        .nodes
        .iter()
        .map(|node| node.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut unique = BTreeSet::new();
    let mut writers = BTreeMap::new();
    for (index, edge) in project.graph.edges.iter().enumerate() {
        let edge_key = (
            edge.source_node.as_str(),
            edge.source_port.as_str(),
            edge.target_node.as_str(),
            edge.target_port.as_str(),
        );
        let duplicate = !unique.insert(edge_key);
        if duplicate {
            issues.push(issue(
                format!("graph.edges[{index}]"),
                "duplicate_edge",
                "duplicates an earlier edge",
            ));
        } else if let Some(previous) = writers.insert(
            (edge.target_node.as_str(), edge.target_port.as_str()),
            index,
        ) {
            issues.push(issue(
                format!("graph.edges[{index}]"),
                "multiple_writers",
                format!("target input already has a writer at graph.edges[{previous}]"),
            ));
        }
        if !nodes.contains(edge.source_node.as_str()) || !nodes.contains(edge.target_node.as_str())
        {
            issues.push(issue(
                format!("graph.edges[{index}]"),
                "graph_compile",
                "edge references an unknown node",
            ));
        }
    }
}

fn validate_sources(project: &ProjectSpec, issues: &mut Vec<ValidationIssue>) {
    let mut ids = BTreeSet::new();
    let mut inputs = BTreeSet::new();
    for (index, source) in project.data_sources.iter().enumerate() {
        let base = format!("data_sources[{index}]");
        validate_id(&source.id, &format!("{base}.id"), issues);
        if !ids.insert(source.id.as_str()) {
            issues.push(issue(
                format!("{base}.id"),
                "duplicate_id",
                format!("duplicate source ID {:?}", source.id),
            ));
        }
        if source.input.is_empty() || source.input.len() > 160 {
            issues.push(issue(
                format!("{base}.input"),
                "invalid_input",
                "source input must contain 1 to 160 bytes",
            ));
        }
        if !inputs.insert(source.input.as_str()) {
            issues.push(issue(
                format!("{base}.input"),
                "duplicate_input",
                format!("duplicate source input {:?}", source.input),
            ));
        }
        if !matches!(
            source.format.as_str(),
            "inline_json" | "csv" | "json" | "arrow_ipc"
        ) {
            issues.push(issue(
                format!("{base}.format"),
                "unsupported_source_format",
                format!("unsupported source format {:?}", source.format),
            ));
        }
    }
}

fn validate_source_coverage<'a>(
    project: &ProjectSpec,
    expected: impl IntoIterator<Item = &'a str>,
    mode: CompileMode,
    issues: &mut Vec<ValidationIssue>,
) {
    let expected = expected.into_iter().collect::<BTreeSet<_>>();
    let (configured, configured_len, path) = match mode {
        CompileMode::Batch => (
            project
                .data_sources
                .iter()
                .map(|source| source.input.as_str())
                .collect::<BTreeSet<_>>(),
            project.data_sources.len(),
            "data_sources",
        ),
        CompileMode::Stream => (
            project
                .sources
                .iter()
                .map(|source| source.binding.as_str())
                .collect::<BTreeSet<_>>(),
            project.sources.len(),
            "sources",
        ),
    };
    if expected != configured || configured_len != expected.len() {
        issues.push(issue(
            path,
            "source_input_mismatch",
            format!("configured source inputs must be {expected:?}; configured {configured:?}"),
        ));
    }
}

fn validate_sink_coverage<'a>(
    project: &ProjectSpec,
    expected: impl IntoIterator<Item = &'a str>,
    issues: &mut Vec<ValidationIssue>,
) {
    let expected = expected.into_iter().collect::<BTreeSet<_>>();
    let configured = project
        .sinks
        .iter()
        .map(|sink| sink.binding.as_str())
        .collect::<BTreeSet<_>>();
    if expected != configured || project.sinks.len() != expected.len() {
        issues.push(issue(
            "sinks",
            "sink_output_mismatch",
            format!("configured sink outputs must be {expected:?}; configured {configured:?}"),
        ));
    }
}

fn validate_run_options(options: &RunOptions, issues: &mut Vec<ValidationIssue>) {
    validate_range(
        options.max_input_bytes,
        1,
        MAX_INPUT_BYTES,
        "runtime.options.max_input_bytes",
        issues,
    );
    validate_range(
        options.max_rows,
        1,
        MAX_ROWS,
        "runtime.options.max_rows",
        issues,
    );
    validate_range(
        options.timeout_seconds,
        1,
        MAX_TIMEOUT_SECONDS,
        "runtime.options.timeout_seconds",
        issues,
    );
    validate_range(
        options.memory_limit_mb,
        MIN_MEMORY_MB,
        MAX_MEMORY_MB,
        "runtime.options.memory_limit_mb",
        issues,
    );
    validate_range(
        options.output_rows,
        1,
        MAX_OUTPUT_ROWS,
        "runtime.options.output_rows",
        issues,
    );
}

fn validate_range<T>(
    value: T,
    minimum: T,
    maximum: T,
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) where
    T: Copy + Ord + std::fmt::Display,
{
    if value < minimum || value > maximum {
        issues.push(issue(
            path,
            "out_of_range",
            format!("must be between {minimum} and {maximum}; received {value}"),
        ));
    }
}

fn validate_maximum<T>(value: T, maximum: T, path: &str, issues: &mut Vec<ValidationIssue>)
where
    T: Copy + Ord + std::fmt::Display,
{
    if value > maximum {
        issues.push(issue(
            path,
            "out_of_range",
            format!("must be at most {maximum}; received {value}"),
        ));
    }
}

fn validate_id(value: &str, path: &str, issues: &mut Vec<ValidationIssue>) {
    if !is_id(value) {
        issues.push(issue(
            path,
            "invalid_id",
            "must start with an ASCII letter and contain at most 64 letters, digits, '_' or '-'",
        ));
    }
}

fn validate_required(value: &str, path: &str, maximum: usize, issues: &mut Vec<ValidationIssue>) {
    if value.trim().is_empty() {
        issues.push(issue(path, "required", "must not be empty"));
    } else if value.len() > maximum {
        issues.push(issue(
            path,
            "out_of_range",
            format!("must contain at most {maximum} bytes"),
        ));
    }
}

fn is_id(value: &str) -> bool {
    value.len() <= 64
        && value
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn is_port_identifier(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|first| first == b'_' || first.is_ascii_alphabetic())
        && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
}

fn arrow_data_type(value: &str) -> Option<DataType> {
    Some(match value {
        "bool" => DataType::Boolean,
        "date32" => DataType::Date32,
        "date64" => DataType::Date64,
        "float32" => DataType::Float32,
        "float64" => DataType::Float64,
        "int8" => DataType::Int8,
        "int16" => DataType::Int16,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "large_string" => DataType::LargeUtf8,
        "string" => DataType::Utf8,
        "time32[s]" => DataType::Time32(TimeUnit::Second),
        "time64[us]" => DataType::Time64(TimeUnit::Microsecond),
        "timestamp[ms]" => DataType::Timestamp(TimeUnit::Millisecond, None),
        "timestamp[us]" => DataType::Timestamp(TimeUnit::Microsecond, None),
        "uint8" => DataType::UInt8,
        "uint16" => DataType::UInt16,
        "uint32" => DataType::UInt32,
        "uint64" => DataType::UInt64,
        _ => return None,
    })
}

fn operator_udfs(operator: &OperatorSpec) -> &[UdfReference] {
    match operator {
        OperatorSpec::Expression { udfs, .. } | OperatorSpec::Sql { udfs, .. } => udfs,
        OperatorSpec::Union | OperatorSpec::Window { .. } | OperatorSpec::External { .. } => &[],
    }
}

fn issue(
    path: impl Into<String>,
    code: impl Into<String>,
    message: impl Into<String>,
) -> ValidationIssue {
    ValidationIssue {
        path: path.into(),
        code: code.into(),
        message: message.into(),
    }
}

fn validation_error(issue: &ValidationIssue) -> CalcFlowError {
    CalcFlowError::Format {
        message: format!("{} [{}]: {}", issue.path, issue.code, issue.message),
    }
}

fn graph_error_path(project: &ProjectSpec, error: &CalcFlowError) -> String {
    let message = error.to_string();
    project
        .graph
        .edges
        .iter()
        .enumerate()
        .find(|(_, edge)| {
            message.contains(&edge.source_node)
                || message.contains(&edge.target_node)
                || message.contains(&edge.source_port)
                || message.contains(&edge.target_port)
        })
        .map_or_else(
            || "graph".into(),
            |(index, _)| format!("graph.edges[{index}]"),
        )
}

fn deserialize_format_version<'de, D>(deserializer: D) -> std::result::Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    let version = u32::deserialize(deserializer)?;
    if version == PROJECT_FORMAT_VERSION {
        Ok(version)
    } else {
        Err(D::Error::custom(format!(
            "project format version {version} is unsupported; expected {PROJECT_FORMAT_VERSION}"
        )))
    }
}

fn deserialize_finite<'de, D>(deserializer: D) -> std::result::Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let value = f64::deserialize(deserializer)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(D::Error::custom("position coordinate must be finite"))
    }
}

const fn default_true() -> bool {
    true
}

fn default_input() -> String {
    "input".into()
}

fn default_output() -> String {
    "output".into()
}
