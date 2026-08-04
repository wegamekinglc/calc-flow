//! Shared graph-validation passes used by both `compile_batch` and
//! `compile_stream` (plan M1.1: one implementation, no behavior drift).

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use datafusion::arrow::{
    datatypes::SchemaRef,
    ipc::{convert::IpcSchemaEncoder, writer::DictionaryTracker},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    CalcFlowError, DataFusionConfig, Edge, NodeOperator, PipelineBuilder, Port, PortEndpoint,
    Result, UdfCatalogEntry, UdfKind, UdfReference, UdfRegistrySnapshot, canonical_json,
    validate_selected_udfs,
};

pub(crate) struct NodeDefinition {
    pub(crate) node_id: String,
    pub(crate) operator: NodeOperator,
}

pub(crate) struct CompiledNode<O> {
    pub(crate) node_id: String,
    pub(crate) operator: Arc<tokio::sync::Mutex<O>>,
    pub(crate) input_ports: Vec<Port>,
    pub(crate) output_ports: Vec<Port>,
    pub(crate) inbound: BTreeMap<String, PortEndpoint>,
    #[allow(
        dead_code,
        reason = "M2.3 operator tasks fan out through the compiled outbound targets"
    )]
    pub(crate) outbound: BTreeMap<String, Vec<PortEndpoint>>,
}

pub(crate) struct TablePlanResources {
    pub(crate) config: DataFusionConfig,
    pub(crate) udfs: UdfRegistrySnapshot,
    pub(crate) selected_udfs: Vec<UdfReference>,
}

/// The mode-independent products of graph compilation.
pub(crate) struct CompiledGraph {
    pub(crate) order: Vec<String>,
    pub(crate) external_inputs: BTreeMap<String, PortEndpoint>,
    pub(crate) external_outputs: BTreeMap<String, PortEndpoint>,
    pub(crate) fingerprint: String,
    pub(crate) table: Option<TablePlanResources>,
}

/// Runs every validation pass shared by the batch and stream compilers.
pub(crate) fn compile_graph(
    builder: &PipelineBuilder,
    execution_mode: &str,
    udfs: &UdfRegistrySnapshot,
) -> Result<CompiledGraph> {
    let requires_datafusion = builder
        .nodes
        .values()
        .any(|node| node.operator.requires_datafusion());
    let order = validate_and_order(builder, requires_datafusion)?;
    let selected_catalog = selected_udf_catalog(&builder.nodes, udfs)?;
    let selected_udfs = selected_catalog
        .iter()
        .map(|(reference, _)| reference.clone())
        .collect();
    let (external_inputs, external_outputs) = external_ports(&builder.nodes, &builder.edges)?;
    let fingerprint = graph_fingerprint(
        execution_mode,
        &builder.name,
        requires_datafusion.then_some(builder.datafusion_config),
        &builder.nodes,
        &builder.edges,
        &selected_catalog,
    )?;
    let table = requires_datafusion.then(|| TablePlanResources {
        config: builder.datafusion_config,
        udfs: udfs.clone(),
        selected_udfs,
    });
    Ok(CompiledGraph {
        order,
        external_inputs,
        external_outputs,
        fingerprint,
        table,
    })
}

/// Runs the validation passes that precede compilation and returns the
/// deterministic topological order of the graph.
fn validate_and_order(builder: &PipelineBuilder, requires_datafusion: bool) -> Result<Vec<String>> {
    if requires_datafusion {
        builder.datafusion_config.validate()?;
    }
    validate_nodes(&builder.nodes)?;
    validate_edges(&builder.nodes, &builder.edges)?;
    topological_order(&builder.nodes, &builder.edges)
}

/// Moves builder nodes into compiled nodes in topological order, converting
/// each operator with the mode-specific converter.
pub(crate) fn build_nodes<O>(
    mut builder: PipelineBuilder,
    order: Vec<String>,
    mut convert: impl FnMut(NodeDefinition) -> O,
) -> Vec<CompiledNode<O>> {
    let inbound = builder
        .edges
        .iter()
        .map(|edge| {
            (
                (edge.target.node_id.clone(), edge.target.port.clone()),
                edge.source.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut outbound = BTreeMap::<(String, String), Vec<PortEndpoint>>::new();
    for edge in &builder.edges {
        outbound
            .entry((edge.source.node_id.clone(), edge.source.port.clone()))
            .or_default()
            .push(edge.target.clone());
    }
    for targets in outbound.values_mut() {
        targets.sort();
        targets.dedup();
    }
    order
        .into_iter()
        .map(|node_id| {
            let definition = builder
                .nodes
                .remove(&node_id)
                .expect("topology contains every validated node exactly once");
            let input_ports = definition.operator.input_ports().to_vec();
            let output_ports = definition.operator.output_ports().to_vec();
            let node_inbound = input_ports
                .iter()
                .filter_map(|port| {
                    inbound
                        .get(&(node_id.clone(), port.name().into()))
                        .cloned()
                        .map(|source| (port.name().into(), source))
                })
                .collect();
            let node_outbound = output_ports
                .iter()
                .filter_map(|port| {
                    outbound
                        .get(&(node_id.clone(), port.name().into()))
                        .cloned()
                        .map(|targets| (port.name().into(), targets))
                })
                .collect();
            CompiledNode {
                node_id: node_id.clone(),
                operator: Arc::new(tokio::sync::Mutex::new(convert(definition))),
                input_ports,
                output_ports,
                inbound: node_inbound,
                outbound: node_outbound,
            }
        })
        .collect()
}

pub(crate) fn validate_nodes(nodes: &BTreeMap<String, NodeDefinition>) -> Result<()> {
    for (node_id, node) in nodes {
        validate_unique_ports(node_id, "input", node.operator.input_ports())?;
        validate_unique_ports(node_id, "output", node.operator.output_ports())?;
    }
    Ok(())
}

fn validate_unique_ports(node_id: &str, direction: &str, ports: &[Port]) -> Result<()> {
    let mut names = BTreeSet::new();
    for port in ports {
        if !names.insert(port.name()) {
            return Err(CalcFlowError::Compile {
                message: format!(
                    "node {node_id} has duplicate {direction} port {}",
                    port.name()
                ),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_edges(
    nodes: &BTreeMap<String, NodeDefinition>,
    edges: &[Edge],
) -> Result<()> {
    validate_edge_uniqueness(edges)?;
    let mut writers = BTreeMap::new();
    for edge in edges {
        let source = endpoint_port(nodes, &edge.source, EndpointDirection::Source)?;
        let target = endpoint_port(nodes, &edge.target, EndpointDirection::Target)?;
        if source.kind() != target.kind() {
            return Err(CalcFlowError::Compile {
                message: format!(
                    "edge {}.{} -> {}.{} has incompatible batch kinds",
                    edge.source.node_id, edge.source.port, edge.target.node_id, edge.target.port
                ),
            });
        }
        if source.schema() != target.schema() {
            return Err(CalcFlowError::Compile {
                message: format!(
                    "edge {}.{} -> {}.{} has incompatible Arrow schemas",
                    edge.source.node_id, edge.source.port, edge.target.node_id, edge.target.port
                ),
            });
        }
        if let Some(previous) = writers.insert(&edge.target, &edge.source) {
            return Err(CalcFlowError::Compile {
                message: format!(
                    "input {}.{} has multiple writers: {}.{} and {}.{}",
                    edge.target.node_id,
                    edge.target.port,
                    previous.node_id,
                    previous.port,
                    edge.source.node_id,
                    edge.source.port
                ),
            });
        }
    }
    Ok(())
}

/// Rejects duplicate edges and stable edge ID collisions before endpoint
/// checks run. Node IDs are only validated non-empty, so `.` or `->` inside
/// them can make two distinct edges format to the same stable ID; collecting
/// those edges into an ID-keyed map would silently drop one.
fn validate_edge_uniqueness(edges: &[Edge]) -> Result<()> {
    let mut unique_edges = BTreeSet::new();
    let mut stable_ids = BTreeMap::new();
    for edge in edges {
        if !unique_edges.insert(edge) {
            return Err(CalcFlowError::Compile {
                message: format!(
                    "duplicate edge {}.{} -> {}.{}",
                    edge.source.node_id, edge.source.port, edge.target.node_id, edge.target.port
                ),
            });
        }
        let stable_id = edge.stable_id();
        if let Some(previous) = stable_ids.insert(stable_id.clone(), edge) {
            return Err(CalcFlowError::Compile {
                message: format!(
                    "edges {}.{} -> {}.{} and {}.{} -> {}.{} collide on stable edge ID {stable_id:?}",
                    previous.source.node_id,
                    previous.source.port,
                    previous.target.node_id,
                    previous.target.port,
                    edge.source.node_id,
                    edge.source.port,
                    edge.target.node_id,
                    edge.target.port,
                ),
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum EndpointDirection {
    Source,
    Target,
}

fn endpoint_port<'a>(
    nodes: &'a BTreeMap<String, NodeDefinition>,
    endpoint: &PortEndpoint,
    direction: EndpointDirection,
) -> Result<&'a Port> {
    let node = nodes
        .get(&endpoint.node_id)
        .ok_or_else(|| CalcFlowError::Compile {
            message: format!("edge references unknown node {}", endpoint.node_id),
        })?;
    let (expected, opposite, label) = match direction {
        EndpointDirection::Source => (
            node.operator.output_ports(),
            node.operator.input_ports(),
            "source",
        ),
        EndpointDirection::Target => (
            node.operator.input_ports(),
            node.operator.output_ports(),
            "target",
        ),
    };
    if let Some(port) = expected.iter().find(|port| port.name() == endpoint.port) {
        return Ok(port);
    }
    let message = if opposite.iter().any(|port| port.name() == endpoint.port) {
        format!(
            "{label} endpoint {}.{} has the wrong port direction",
            endpoint.node_id, endpoint.port
        )
    } else {
        format!(
            "{label} endpoint {}.{} names a missing port",
            endpoint.node_id, endpoint.port
        )
    };
    Err(CalcFlowError::Compile { message })
}

pub(crate) fn topological_order(
    nodes: &BTreeMap<String, NodeDefinition>,
    edges: &[Edge],
) -> Result<Vec<String>> {
    let mut indegree = nodes
        .keys()
        .map(|node_id| (node_id.clone(), 0_usize))
        .collect::<BTreeMap<_, _>>();
    let mut outgoing = nodes
        .keys()
        .map(|node_id| (node_id.clone(), BTreeMap::new()))
        .collect::<BTreeMap<_, _>>();
    for edge in edges {
        *indegree
            .get_mut(&edge.target.node_id)
            .expect("edges were validated before sorting") += 1;
        outgoing
            .get_mut(&edge.source.node_id)
            .expect("edges were validated before sorting")
            .entry(edge.target.node_id.clone())
            .and_modify(|count| *count += 1)
            .or_insert(1_usize);
    }

    let mut ready = indegree
        .iter()
        .filter_map(|(node_id, degree)| (*degree == 0).then_some(node_id.clone()))
        .collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(node_id) = ready.pop_first() {
        for (target, edge_count) in &outgoing[&node_id] {
            let degree = indegree
                .get_mut(target)
                .expect("edges were validated before sorting");
            *degree -= edge_count;
            if *degree == 0 {
                ready.insert(target.clone());
            }
        }
        order.push(node_id);
    }
    if order.len() != nodes.len() {
        return Err(CalcFlowError::Compile {
            message: "pipeline graph contains a cycle".into(),
        });
    }
    Ok(order)
}

pub(crate) fn external_ports(
    nodes: &BTreeMap<String, NodeDefinition>,
    edges: &[Edge],
) -> Result<(
    BTreeMap<String, PortEndpoint>,
    BTreeMap<String, PortEndpoint>,
)> {
    let connected_inputs = edges
        .iter()
        .map(|edge| edge.target.clone())
        .collect::<BTreeSet<_>>();
    let connected_outputs = edges
        .iter()
        .map(|edge| edge.source.clone())
        .collect::<BTreeSet<_>>();
    let inputs = nodes
        .iter()
        .flat_map(|(node_id, node)| {
            node.operator.input_ports().iter().map(|port| PortEndpoint {
                node_id: node_id.clone(),
                port: port.name().into(),
            })
        })
        .filter(|endpoint| !connected_inputs.contains(endpoint))
        .collect::<BTreeSet<_>>();
    let outputs = nodes
        .iter()
        .flat_map(|(node_id, node)| {
            node.operator
                .output_ports()
                .iter()
                .map(|port| PortEndpoint {
                    node_id: node_id.clone(),
                    port: port.name().into(),
                })
        })
        .filter(|endpoint| !connected_outputs.contains(endpoint))
        .collect::<BTreeSet<_>>();
    if outputs.is_empty() {
        return Err(CalcFlowError::Compile {
            message: "pipeline requires at least one external output".into(),
        });
    }
    Ok((external_names(inputs), external_names(outputs)))
}

/// Assigns a bare port name when it is unique in one external direction.
/// Every endpoint sharing a port name is instead qualified as `node_id.port`.
/// Port names cannot contain `.`, so qualification is unambiguous and cannot
/// collide with a bare name. Sorted endpoints make the assignment independent
/// of graph insertion order.
fn external_names(endpoints: BTreeSet<PortEndpoint>) -> BTreeMap<String, PortEndpoint> {
    let counts = endpoints
        .iter()
        .fold(BTreeMap::new(), |mut counts, endpoint| {
            *counts.entry(endpoint.port.clone()).or_insert(0_usize) += 1;
            counts
        });
    endpoints
        .into_iter()
        .map(|endpoint| {
            let name = if counts[&endpoint.port] == 1 {
                endpoint.port.clone()
            } else {
                format!("{}.{}", endpoint.node_id, endpoint.port)
            };
            (name, endpoint)
        })
        .collect()
}

pub(crate) fn selected_udf_catalog(
    nodes: &BTreeMap<String, NodeDefinition>,
    udfs: &UdfRegistrySnapshot,
) -> Result<Vec<(UdfReference, UdfCatalogEntry)>> {
    let references = nodes
        .values()
        .flat_map(|node| node.operator.udf_references())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    validate_selected_udfs(&references)?;

    references
        .into_iter()
        .map(|reference| {
            let entry = udfs
                .catalog()
                .iter()
                .find(|entry| catalog_matches(entry, &reference))
                .cloned()
                .ok_or_else(|| CalcFlowError::Compile {
                    message: format!(
                        "unknown UDF {}:{}@{}",
                        reference.provider(),
                        reference.name(),
                        reference.version()
                    ),
                })?;
            if reference.kind() == UdfKind::DataFusionScalar {
                udfs.resolve_native(&reference)?;
            }
            Ok((reference, entry))
        })
        .collect()
}

fn catalog_matches(entry: &UdfCatalogEntry, reference: &UdfReference) -> bool {
    entry.provider == reference.provider()
        && entry.name == reference.name()
        && entry.version == reference.version()
        && entry.kind == reference.kind()
}

/// Computes the semantic fingerprint (spec NFR-5): execution mode, graph
/// structure, operator configurations, and the UDF catalog. Runtime-tunable
/// values (channel capacities, checkpoint intervals) are never inputs; they
/// feed the separate runtime-config hash on `StreamExecutionPlan`.
pub(crate) fn graph_fingerprint(
    execution_mode: &str,
    name: &str,
    datafusion_config: Option<DataFusionConfig>,
    nodes: &BTreeMap<String, NodeDefinition>,
    edges: &[Edge],
    selected_catalog: &[(UdfReference, UdfCatalogEntry)],
) -> Result<String> {
    let node_values = nodes
        .iter()
        .map(|(node_id, node)| {
            let declared_udfs = node.operator.udf_references();
            let canonical_udfs = canonical_udf_references(&declared_udfs);
            let configuration = fingerprint_configuration(
                node.operator.configuration(),
                &declared_udfs,
                &canonical_udfs,
            );
            Ok(json!({
                "configuration": configuration,
                "input_ports": port_values(node.operator.input_ports()),
                "node_id": node_id,
                "output_ports": port_values(node.operator.output_ports()),
                "udf_references": canonical_udfs,
            }))
        })
        .collect::<Result<Vec<Value>>>()?;
    let mut sorted_edges = edges.to_vec();
    sorted_edges.sort();
    let catalog_values = selected_catalog
        .iter()
        .map(|(reference, entry)| json!({"reference": reference, "catalog": entry}))
        .collect::<Vec<_>>();
    let mut value = json!({
        "edges": sorted_edges,
        "execution_mode": execution_mode,
        "name": name,
        "nodes": node_values,
        "selected_udfs": catalog_values,
    });
    if let Some(datafusion_config) = datafusion_config {
        value
            .as_object_mut()
            .expect("fingerprint root is an object")
            .insert("datafusion".into(), json!(datafusion_config));
    }
    let canonical = canonical_json(&value)?;
    Ok(hex::encode(Sha256::digest(canonical.as_bytes())))
}

fn canonical_udf_references(references: &[UdfReference]) -> Vec<UdfReference> {
    references
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Canonicalizes only the conventional projection of declared UDF references.
/// An arbitrary configuration array retains its order unless `configuration.udfs`
/// exactly mirrors the operator's declared references in their original order.
fn fingerprint_configuration(
    mut configuration: BTreeMap<String, Value>,
    declared_udfs: &[UdfReference],
    canonical_udfs: &[UdfReference],
) -> BTreeMap<String, Value> {
    let declared_projection = Value::Array(declared_udfs.iter().map(|udf| json!(udf)).collect());
    if configuration.get("udfs") == Some(&declared_projection) {
        configuration.insert(
            "udfs".into(),
            Value::Array(canonical_udfs.iter().map(|udf| json!(udf)).collect()),
        );
    }
    configuration
}

fn port_values(ports: &[Port]) -> Vec<Value> {
    let mut ports = ports.iter().collect::<Vec<_>>();
    ports.sort_by_key(|port| port.name());
    ports
        .into_iter()
        .map(|port| {
            json!({
                "kind": port.kind(),
                "name": port.name(),
                "required": port.required(),
                "schema": port.schema().map(schema_value),
            })
        })
        .collect()
}

fn schema_value(schema: &SchemaRef) -> Value {
    let mut dictionary_tracker = DictionaryTracker::new(true);
    let bytes = IpcSchemaEncoder::new()
        .with_dictionary_tracker(&mut dictionary_tracker)
        .schema_to_fb(schema)
        .finished_data()
        .to_vec();
    Value::String(hex::encode(bytes))
}
