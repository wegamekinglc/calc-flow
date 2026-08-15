use std::collections::{BTreeMap, BTreeSet};

use datafusion::arrow::datatypes::{DataType, Field, TimeUnit};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::{Value, json};

use crate::operator::expression_query;
use crate::{
    BatchExecutionPlan, BatchKind, CalcFlowError, DataFusionConfig, Edge, ExpressionOperator,
    ExternalOperatorSpec, JsonMap, NodeOperator, PipelineBuilder, Port, PortEndpoint,
    ProviderRegistry, Result, SqlOperator, StreamExecutionPlan, StreamRequirements, UdfKind,
    UdfReference, UdfRegistrySnapshot, validate_selected_udfs,
};

pub const PROJECT_FORMAT_VERSION: u32 = 2;

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
    pub pipeline: PipelineSpec,
    #[serde(default)]
    pub data_sources: Vec<DataSourceSpec>,
    #[serde(default)]
    pub run_options: RunOptions,
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

/// Generates the single canonical JSON Schema for project format v2.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] if the generated schema cannot be encoded.
pub fn project_json_schema() -> Result<Value> {
    let mut value =
        serde_json::to_value(schema_for!(ProjectSpec)).map_err(|error| CalcFlowError::Format {
            message: error.to_string(),
        })?;
    value["title"] = Value::String("Calc Flow Project V2".into());
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
            validate_source_coverage(project, &plan, &mut issues);
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
    validate_source_coverage(project, &plan, &mut coverage);
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
    requirements: &StreamRequirements,
) -> Result<StreamExecutionPlan> {
    let issues = semantic_issues(project, providers, udfs, CompileMode::Stream);
    if let Some(first) = issues.first() {
        return Err(validation_error(first));
    }
    build_project_builder(project, providers, CompileMode::Stream)?
        .compile_stream(udfs, requirements)
}

#[derive(Clone, Copy)]
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
    let builder = PipelineBuilder::new(&project.pipeline.name)?
        .with_datafusion_config(project.pipeline.datafusion);
    let builder = add_project_nodes(builder, project, providers, mode)?;
    add_project_edges(builder, project)
}

fn add_project_nodes(
    mut builder: PipelineBuilder,
    project: &ProjectSpec,
    providers: &ProviderRegistry,
    mode: CompileMode,
) -> Result<PipelineBuilder> {
    for node in &project.pipeline.nodes {
        let operator = project_node_operator(node, providers, mode)?;
        builder = builder.add_node(&node.id, operator)?;
    }
    Ok(builder)
}

fn add_project_edges(
    mut builder: PipelineBuilder,
    project: &ProjectSpec,
) -> Result<PipelineBuilder> {
    for edge in &project.pipeline.edges {
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
    validate_required(&project.pipeline.name, "pipeline.name", 120, &mut issues);
    if project.pipeline.nodes.is_empty() {
        issues.push(issue(
            "pipeline.nodes",
            "required",
            "pipeline requires at least one node",
        ));
    }
    if project_requires_datafusion(project) {
        if let Err(CalcFlowError::InvalidArgument { field, message }) =
            project.pipeline.datafusion.validate()
        {
            issues.push(issue(format!("pipeline.{field}"), "out_of_range", message));
        }
        validate_maximum(
            project.pipeline.datafusion.batch_size,
            MAX_BATCH_SIZE,
            "pipeline.datafusion.batch_size",
            &mut issues,
        );
        validate_maximum(
            project.pipeline.datafusion.target_partitions,
            MAX_TARGET_PARTITIONS,
            "pipeline.datafusion.target_partitions",
            &mut issues,
        );
    }
    validate_run_options(&project.run_options, &mut issues);
    validate_nodes(project, providers, udfs, mode, &mut issues);
    validate_edges(project, &mut issues);
    validate_sources(project, &mut issues);
    issues
}

fn project_requires_datafusion(project: &ProjectSpec) -> bool {
    project.pipeline.nodes.iter().any(|node| {
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
    for (node_index, node) in project.pipeline.nodes.iter().enumerate() {
        let base = format!("pipeline.nodes[{node_index}]");
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
            .pipeline
            .nodes
            .iter()
            .enumerate()
            .find(|(_, node)| !operator_udfs(&node.operator).is_empty())
            .map_or_else(
                || "pipeline.nodes".into(),
                |(index, _)| format!("pipeline.nodes[{index}].operator.udfs"),
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
    let base = format!("pipeline.nodes[{index}].operator");
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
        OperatorSpec::External {
            provider,
            name,
            version,
            ..
        } => {
            match ExternalOperatorSpec::new(provider, name, version, BTreeMap::new()) {
                Ok(_) => {
                    if !provider_available(providers, mode, provider, name, version) {
                        issues.push(issue(
                            &base,
                            "missing_provider",
                            format!("provider {provider}:{name}@{version} is unavailable"),
                        ));
                    }
                }
                Err(error) => issues.push(issue(&base, "invalid_operator", error.to_string())),
            }
            (None, None)
        }
    };
    if let Some(names) = expected_inputs {
        validate_builtin_port_specs(
            &node.input_ports,
            &names,
            &format!("pipeline.nodes[{index}].input_ports"),
            true,
            issues,
        );
    }
    if let Some(names) = expected_outputs {
        validate_builtin_port_specs(
            &node.output_ports,
            &names,
            &format!("pipeline.nodes[{index}].output_ports"),
            false,
            issues,
        );
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
        .pipeline
        .nodes
        .iter()
        .map(|node| node.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut unique = BTreeSet::new();
    let mut writers = BTreeMap::new();
    for (index, edge) in project.pipeline.edges.iter().enumerate() {
        let edge_key = (
            edge.source_node.as_str(),
            edge.source_port.as_str(),
            edge.target_node.as_str(),
            edge.target_port.as_str(),
        );
        let duplicate = !unique.insert(edge_key);
        if duplicate {
            issues.push(issue(
                format!("pipeline.edges[{index}]"),
                "duplicate_edge",
                "duplicates an earlier edge",
            ));
        } else if let Some(previous) = writers.insert(
            (edge.target_node.as_str(), edge.target_port.as_str()),
            index,
        ) {
            issues.push(issue(
                format!("pipeline.edges[{index}]"),
                "multiple_writers",
                format!("target input already has a writer at pipeline.edges[{previous}]"),
            ));
        }
        if !nodes.contains(edge.source_node.as_str()) || !nodes.contains(edge.target_node.as_str())
        {
            issues.push(issue(
                format!("pipeline.edges[{index}]"),
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

fn validate_source_coverage(
    project: &ProjectSpec,
    plan: &BatchExecutionPlan,
    issues: &mut Vec<ValidationIssue>,
) {
    let expected = plan.external_inputs().keys().collect::<BTreeSet<_>>();
    let configured = project
        .data_sources
        .iter()
        .map(|source| &source.input)
        .collect::<BTreeSet<_>>();
    if expected != configured || project.data_sources.len() != expected.len() {
        issues.push(issue(
            "data_sources",
            "source_input_mismatch",
            format!("saved source inputs must be {expected:?}; configured {configured:?}"),
        ));
    }
}

fn validate_run_options(options: &RunOptions, issues: &mut Vec<ValidationIssue>) {
    validate_range(
        options.max_input_bytes,
        1,
        MAX_INPUT_BYTES,
        "run_options.max_input_bytes",
        issues,
    );
    validate_range(
        options.max_rows,
        1,
        MAX_ROWS,
        "run_options.max_rows",
        issues,
    );
    validate_range(
        options.timeout_seconds,
        1,
        MAX_TIMEOUT_SECONDS,
        "run_options.timeout_seconds",
        issues,
    );
    validate_range(
        options.memory_limit_mb,
        MIN_MEMORY_MB,
        MAX_MEMORY_MB,
        "run_options.memory_limit_mb",
        issues,
    );
    validate_range(
        options.output_rows,
        1,
        MAX_OUTPUT_ROWS,
        "run_options.output_rows",
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
        OperatorSpec::External { .. } => &[],
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
        .pipeline
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
            || "pipeline".into(),
            |(index, _)| format!("pipeline.edges[{index}]"),
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
