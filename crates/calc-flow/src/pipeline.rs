use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::{Duration, Instant},
};

use chrono::{DateTime, Utc};
use datafusion::arrow::{
    datatypes::SchemaRef,
    ipc::{convert::IpcSchemaEncoder, writer::DictionaryTracker},
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    Batch, CalcFlowError, CancellationToken, DataFusionConfig, DataFusionQueryMetric,
    DataFusionRuntime, JsonMap, Operator, OperatorContext, Port, Result, RunContext,
    UdfCatalogEntry, UdfReference, UdfRegistrySnapshot, canonical_json, validate_selected_udfs,
};

#[derive(Clone, Debug, Default)]
pub struct ExecutionOptions {
    pub settings: JsonMap,
    pub deadline: Option<DateTime<Utc>>,
    pub cancellation: CancellationToken,
}

#[derive(Clone, Debug, Serialize)]
pub struct NodeTiming {
    pub duration_ns: u64,
    pub input_rows: BTreeMap<String, usize>,
    pub output_rows: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunMetadata {
    pub run_id: String,
    pub pipeline_name: String,
    pub pipeline_fingerprint: String,
}

#[derive(Clone, Debug)]
pub struct RunResult {
    pub outputs: BTreeMap<String, Batch>,
    pub node_timings: BTreeMap<String, NodeTiming>,
    pub datafusion_metrics: Vec<DataFusionQueryMetric>,
    pub metadata: RunMetadata,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct PortEndpoint {
    pub node_id: String,
    pub port: String,
}

impl PortEndpoint {
    /// Creates an endpoint naming one port on one pipeline node.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when either component is
    /// empty.
    pub fn new(node_id: &str, port: &str) -> Result<Self> {
        if node_id.is_empty() || port.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "endpoint".into(),
                message: "node and port must not be empty".into(),
            });
        }
        Ok(Self {
            node_id: node_id.into(),
            port: port.into(),
        })
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Edge {
    pub source: PortEndpoint,
    pub target: PortEndpoint,
}

impl Edge {
    pub const fn new(source: PortEndpoint, target: PortEndpoint) -> Self {
        Self { source, target }
    }
}

struct NodeDefinition {
    node_id: String,
    operator: Box<dyn Operator>,
}

pub struct PipelineBuilder {
    name: String,
    nodes: BTreeMap<String, NodeDefinition>,
    edges: Vec<Edge>,
}

#[allow(dead_code)]
pub(crate) struct CompiledNode {
    pub(crate) node_id: String,
    pub(crate) operator: Arc<tokio::sync::Mutex<Box<dyn Operator>>>,
    pub(crate) input_ports: Vec<Port>,
    pub(crate) output_ports: Vec<Port>,
    pub(crate) inbound: BTreeMap<String, PortEndpoint>,
}

pub struct ExecutionPlan {
    pub(crate) name: String,
    pub(crate) nodes: Vec<CompiledNode>,
    pub(crate) external_inputs: BTreeMap<String, PortEndpoint>,
    pub(crate) external_outputs: BTreeMap<String, PortEndpoint>,
    pub(crate) fingerprint: String,
    pub(crate) run_lock: tokio::sync::Mutex<()>,
    pub(crate) udfs: UdfRegistrySnapshot,
    pub(crate) selected_udfs: Vec<UdfReference>,
}

impl ExecutionPlan {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn topological_order(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .map(|node| node.node_id.as_str())
            .collect()
    }

    pub const fn external_inputs(&self) -> &BTreeMap<String, PortEndpoint> {
        &self.external_inputs
    }

    pub const fn external_outputs(&self) -> &BTreeMap<String, PortEndpoint> {
        &self.external_outputs
    }

    /// Executes one run while owning the plan's state lifecycle lock.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid external inputs, cancellation, operator or
    /// runtime failures, invalid operator outputs, or a failed rollback.
    pub async fn execute(
        &self,
        inputs: BTreeMap<String, Batch>,
        options: ExecutionOptions,
    ) -> Result<RunResult> {
        let _run_guard = self.run_lock.lock().await;
        self.validate_external_inputs(&inputs)?;
        let before = self.snapshot_unlocked().await?;
        let result = self.execute_unlocked(inputs, options).await;
        match result {
            Ok(result) => Ok(result),
            Err(original) => match self.restore_unlocked(&before).await {
                Ok(()) => Err(original),
                Err(rollback) => Err(CalcFlowError::Internal {
                    message: format!(
                        "run failed with {original}; rollback also failed with {rollback}"
                    ),
                }),
            },
        }
    }

    /// Captures every node's JSON state under the plan's run lock.
    ///
    /// # Errors
    ///
    /// Returns an error when any operator cannot capture its state.
    pub async fn snapshot(&self) -> Result<BTreeMap<String, Value>> {
        let _run_guard = self.run_lock.lock().await;
        self.snapshot_unlocked().await
    }

    /// Restores an exact node-keyed state map under the plan's run lock.
    ///
    /// Node IDs are validated before any operator is mutated. Once validated,
    /// every node is given its state even if another node rejects restoration.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::CheckpointMismatch`] for missing or extra node
    /// IDs, or an error summarizing operator restore failures.
    pub async fn restore(&self, state: &BTreeMap<String, Value>) -> Result<()> {
        let _run_guard = self.run_lock.lock().await;
        self.restore_unlocked(state).await
    }

    /// Resets every node under the plan's run lock.
    ///
    /// All nodes are attempted even if one reset fails.
    ///
    /// # Errors
    ///
    /// Returns an error summarizing operator reset failures.
    pub async fn reset(&self) -> Result<()> {
        let _run_guard = self.run_lock.lock().await;
        let mut failures = Vec::new();
        for node in &self.nodes {
            if let Err(error) = node.operator.lock().await.reset() {
                failures.push(format!("{}: {error}", node.node_id));
            }
        }
        lifecycle_result("reset", &failures)
    }

    async fn execute_unlocked(
        &self,
        inputs: BTreeMap<String, Batch>,
        options: ExecutionOptions,
    ) -> Result<RunResult> {
        let context = RunContext::new(options.settings, options.deadline, options.cancellation)?;
        let mut runtime = DataFusionRuntime::new(DataFusionConfig::default())?;
        runtime.register_udfs(&self.udfs, &self.selected_udfs)?;
        let execution = self.execute_nodes(&inputs, &context, &runtime).await;
        runtime.close();
        let (outputs, node_timings) = execution?;
        Ok(RunResult {
            outputs,
            node_timings,
            datafusion_metrics: runtime.metrics(),
            metadata: RunMetadata {
                run_id: context.run_id().into(),
                pipeline_name: self.name.clone(),
                pipeline_fingerprint: self.fingerprint.clone(),
            },
        })
    }

    fn validate_external_inputs(&self, inputs: &BTreeMap<String, Batch>) -> Result<()> {
        let unknown = inputs
            .keys()
            .filter(|name| !self.external_inputs.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        if !unknown.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "inputs".into(),
                message: format!("unknown graph inputs: {unknown:?}"),
            });
        }
        for (name, endpoint) in &self.external_inputs {
            let node = self.node(&endpoint.node_id)?;
            let port = node
                .input_ports
                .iter()
                .find(|port| port.name() == endpoint.port)
                .ok_or_else(|| CalcFlowError::Internal {
                    message: format!("compiled external input {name} has no matching port"),
                })?;
            match inputs.get(name) {
                Some(batch) => port.validate(
                    batch,
                    &format!(
                        "graph input {name:?} ({}.{})",
                        endpoint.node_id, endpoint.port
                    ),
                )?,
                None if port.required() => {
                    return Err(CalcFlowError::InvalidArgument {
                        field: "inputs".into(),
                        message: format!("missing required graph input {name:?}"),
                    });
                }
                None => {}
            }
        }
        Ok(())
    }

    async fn execute_nodes(
        &self,
        inputs: &BTreeMap<String, Batch>,
        context: &RunContext,
        runtime: &DataFusionRuntime,
    ) -> Result<(BTreeMap<String, Batch>, BTreeMap<String, NodeTiming>)> {
        let external_names = self
            .external_inputs
            .iter()
            .map(|(name, endpoint)| (endpoint.clone(), name.clone()))
            .collect::<BTreeMap<_, _>>();
        let external_values = self
            .external_inputs
            .iter()
            .filter_map(|(name, endpoint)| {
                inputs
                    .get(name)
                    .cloned()
                    .map(|batch| (endpoint.clone(), batch))
            })
            .collect::<BTreeMap<_, _>>();
        let mut produced_values = BTreeMap::new();
        let mut timings = BTreeMap::new();

        context.check_cancelled()?;
        for node in &self.nodes {
            let node_context = context.for_node(&node.node_id)?;
            let mut operator = node.operator.lock().await;
            let operator_inputs =
                gather_node_inputs(node, &produced_values, &external_values, &external_names)?;

            node_context.check_cancelled()?;
            let started = Instant::now();
            let process_result = operator
                .process(
                    &operator_inputs,
                    &OperatorContext {
                        run: &node_context,
                        datafusion: runtime,
                    },
                )
                .await;
            let duration_ns = nanos(started.elapsed());
            node_context.check_cancelled()?;
            let operator_outputs = process_result?;
            validate_and_store_outputs(node, &operator_outputs, &mut produced_values)?;
            timings.insert(
                node.node_id.clone(),
                NodeTiming {
                    duration_ns,
                    input_rows: row_counts(&operator_inputs),
                    output_rows: row_counts(&operator_outputs),
                },
            );
        }

        let outputs = self
            .external_outputs
            .iter()
            .filter_map(|(name, endpoint)| {
                produced_values
                    .get(endpoint)
                    .cloned()
                    .map(|batch| (name.clone(), batch))
            })
            .collect();
        Ok((outputs, timings))
    }

    async fn snapshot_unlocked(&self) -> Result<BTreeMap<String, Value>> {
        let mut state = BTreeMap::new();
        for node in &self.nodes {
            state.insert(node.node_id.clone(), node.operator.lock().await.snapshot()?);
        }
        Ok(state)
    }

    async fn restore_unlocked(&self, state: &BTreeMap<String, Value>) -> Result<()> {
        let expected = self
            .nodes
            .iter()
            .map(|node| node.node_id.as_str())
            .collect::<BTreeSet<_>>();
        let actual = state.keys().map(String::as_str).collect::<BTreeSet<_>>();
        if actual != expected {
            let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
            let extra = actual.difference(&expected).copied().collect::<Vec<_>>();
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "state node IDs do not match the plan; missing={missing:?}, extra={extra:?}"
                ),
            });
        }
        let mut failures = Vec::new();
        for node in &self.nodes {
            if let Err(error) = node.operator.lock().await.restore(&state[&node.node_id]) {
                failures.push(format!("{}: {error}", node.node_id));
            }
        }
        lifecycle_result("restore", &failures)
    }

    fn node(&self, node_id: &str) -> Result<&CompiledNode> {
        self.nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .ok_or_else(|| CalcFlowError::Internal {
                message: format!("compiled plan has no node {node_id}"),
            })
    }
}

fn row_counts(batches: &BTreeMap<String, Batch>) -> BTreeMap<String, usize> {
    batches
        .iter()
        .map(|(name, batch)| (name.clone(), batch.num_rows()))
        .collect()
}

fn gather_node_inputs(
    node: &CompiledNode,
    produced_values: &BTreeMap<PortEndpoint, Batch>,
    external_values: &BTreeMap<PortEndpoint, Batch>,
    external_names: &BTreeMap<PortEndpoint, String>,
) -> Result<BTreeMap<String, Batch>> {
    let mut inputs = BTreeMap::new();
    for port in &node.input_ports {
        let target = PortEndpoint {
            node_id: node.node_id.clone(),
            port: port.name().into(),
        };
        let source = node.inbound.get(port.name());
        let batch = source
            .and_then(|endpoint| produced_values.get(endpoint))
            .or_else(|| {
                source
                    .is_none()
                    .then(|| external_values.get(&target))
                    .flatten()
            });
        match batch {
            Some(batch) => {
                port.validate(batch, &format!("input {}.{}", node.node_id, port.name()))?;
                inputs.insert(port.name().into(), batch.clone());
            }
            None if port.required() => {
                return Err(missing_node_input(
                    node,
                    port,
                    source,
                    external_names,
                    &target,
                ));
            }
            None => {}
        }
    }
    Ok(inputs)
}

fn missing_node_input(
    node: &CompiledNode,
    port: &Port,
    source: Option<&PortEndpoint>,
    external_names: &BTreeMap<PortEndpoint, String>,
    target: &PortEndpoint,
) -> CalcFlowError {
    let label = source.map_or_else(
        || {
            external_names.get(target).map_or_else(
                || format!("{target:?}"),
                |name| format!("graph input {name:?}"),
            )
        },
        |source| {
            format!(
                "required input {}.{} from optional output {}.{}",
                node.node_id,
                port.name(),
                source.node_id,
                source.port
            )
        },
    );
    CalcFlowError::Operator {
        node_id: node.node_id.clone(),
        message: format!("{label} is missing"),
    }
}

fn validate_and_store_outputs(
    node: &CompiledNode,
    outputs: &BTreeMap<String, Batch>,
    values: &mut BTreeMap<PortEndpoint, Batch>,
) -> Result<()> {
    let output_ports = node
        .output_ports
        .iter()
        .map(|port| (port.name(), port))
        .collect::<BTreeMap<_, _>>();
    let unknown = outputs
        .keys()
        .filter(|name| !output_ports.contains_key(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(CalcFlowError::Operator {
            node_id: node.node_id.clone(),
            message: format!("returned unknown outputs: {unknown:?}"),
        });
    }
    let missing = output_ports
        .values()
        .filter(|port| port.required() && !outputs.contains_key(port.name()))
        .map(|port| port.name())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(CalcFlowError::Operator {
            node_id: node.node_id.clone(),
            message: format!("omitted required outputs: {missing:?}"),
        });
    }
    for (name, batch) in outputs {
        output_ports[name.as_str()].validate(batch, &format!("output {}.{name}", node.node_id))?;
        values.insert(
            PortEndpoint {
                node_id: node.node_id.clone(),
                port: name.clone(),
            },
            batch.clone(),
        );
    }
    Ok(())
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn lifecycle_result(action: &str, failures: &[String]) -> Result<()> {
    if failures.is_empty() {
        Ok(())
    } else {
        Err(CalcFlowError::Internal {
            message: format!("operator {action} failed: {}", failures.join("; ")),
        })
    }
}

impl PipelineBuilder {
    /// Creates an empty owned graph builder.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when `name` is empty.
    pub fn new(name: &str) -> Result<Self> {
        if name.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "pipeline.name".into(),
                message: "must not be empty".into(),
            });
        }
        Ok(Self {
            name: name.into(),
            nodes: BTreeMap::new(),
            edges: Vec::new(),
        })
    }

    /// Returns a new builder that owns the added operator.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Compile`] when `node_id` is empty or already
    /// exists.
    pub fn add_node(mut self, node_id: &str, operator: Box<dyn Operator>) -> Result<Self> {
        if node_id.is_empty() {
            return Err(CalcFlowError::Compile {
                message: "node ID must not be empty".into(),
            });
        }
        if self.nodes.contains_key(node_id) {
            return Err(CalcFlowError::Compile {
                message: format!("duplicate node {node_id}"),
            });
        }
        self.nodes.insert(
            node_id.into(),
            NodeDefinition {
                node_id: node_id.into(),
                operator,
            },
        );
        Ok(self)
    }

    /// Returns a new builder containing the directed edge.
    ///
    /// Port validation is deferred until [`Self::compile`] so every operator
    /// remains owned by the builder while the complete graph is checked.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Compile`] when either endpoint names an unknown
    /// node.
    pub fn connect(mut self, edge: Edge) -> Result<Self> {
        if !self.nodes.contains_key(&edge.source.node_id)
            || !self.nodes.contains_key(&edge.target.node_id)
        {
            return Err(CalcFlowError::Compile {
                message: "edge references an unknown node".into(),
            });
        }
        self.edges.push(edge);
        Ok(self)
    }

    /// Validates and consumes this graph into an immutable execution topology.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Compile`] for an invalid graph or selected UDF
    /// catalog.
    pub fn compile(self, udfs: &UdfRegistrySnapshot) -> Result<ExecutionPlan> {
        validate_nodes(&self.nodes)?;
        validate_edges(&self.nodes, &self.edges)?;
        let order = topological_order(&self.nodes, &self.edges)?;
        let selected_catalog = selected_udf_catalog(&self.nodes, udfs)?;
        let selected_udfs = selected_catalog
            .iter()
            .map(|(reference, _)| reference.clone())
            .collect();
        let (external_inputs, external_outputs) = external_ports(&self.nodes, &self.edges)?;
        let fingerprint =
            graph_fingerprint(&self.name, &self.nodes, &self.edges, &selected_catalog)?;
        Ok(build_plan(
            self,
            order,
            external_inputs,
            external_outputs,
            fingerprint,
            udfs.clone(),
            selected_udfs,
        ))
    }
}

fn validate_nodes(nodes: &BTreeMap<String, NodeDefinition>) -> Result<()> {
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

fn validate_edges(nodes: &BTreeMap<String, NodeDefinition>, edges: &[Edge]) -> Result<()> {
    let mut unique_edges = BTreeSet::new();
    let mut writers = BTreeMap::new();
    for edge in edges {
        if !unique_edges.insert(edge) {
            return Err(CalcFlowError::Compile {
                message: format!(
                    "duplicate edge {}.{} -> {}.{}",
                    edge.source.node_id, edge.source.port, edge.target.node_id, edge.target.port
                ),
            });
        }
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

fn topological_order(
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

fn external_ports(
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

fn selected_udf_catalog(
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
            if reference.kind() == crate::UdfKind::DataFusionScalar {
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

fn graph_fingerprint(
    name: &str,
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
    let value = json!({
        "edges": sorted_edges,
        "name": name,
        "nodes": node_values,
        "selected_udfs": catalog_values,
    });
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

fn build_plan(
    mut builder: PipelineBuilder,
    order: Vec<String>,
    external_inputs: BTreeMap<String, PortEndpoint>,
    external_outputs: BTreeMap<String, PortEndpoint>,
    fingerprint: String,
    udfs: UdfRegistrySnapshot,
    selected_udfs: Vec<UdfReference>,
) -> ExecutionPlan {
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
    let nodes = order
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
            CompiledNode {
                node_id: definition.node_id,
                operator: Arc::new(tokio::sync::Mutex::new(definition.operator)),
                input_ports,
                output_ports,
                inbound: node_inbound,
            }
        })
        .collect();
    ExecutionPlan {
        name: builder.name,
        nodes,
        external_inputs,
        external_outputs,
        fingerprint,
        run_lock: tokio::sync::Mutex::new(()),
        udfs,
        selected_udfs,
    }
}
