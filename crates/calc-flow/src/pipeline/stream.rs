//! The continuously running stream execution plan (plan task M1.1).
//!
//! A `StreamExecutionPlan` is a compiled artifact: graph topology, stable
//! edge IDs, source/sink binding slots, the semantic fingerprint, and the
//! delivery requirements. It never executes directly; the crate-private M2
//! runtime consumes it. Runtime-tunable values feed only the runtime-config hash
//! (spec NFR-5), never the semantic fingerprint.

use std::{collections::BTreeMap, time::Duration};

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::{
    CalcFlowError, DataFusionConfig, Edge, NodeOperator, OperatorMetadata, PipelineBuilder, Port,
    PortEndpoint, Result, StreamOperator, UdfKind, UdfRegistrySnapshot, UnionOperator,
    canonical_json,
};

use super::{NodeDefinition, TablePlanResources, compile_graph};

/// Per graph-output delivery requests recorded into the compiled stream plan
/// (API note A1.2).
#[derive(Clone, Debug, Default)]
pub struct StreamRequirements {
    /// Outputs absent from the map default to
    /// [`DeliveryGuarantee::AtLeastOnce`].
    pub delivery: BTreeMap<String, DeliveryGuarantee>,
}

/// The per-sink delivery contract requested for one graph output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryGuarantee {
    AtLeastOnce,
    ExactlyOnce,
}

/// Dual row and byte hard limits for one edge channel (spec S10.1).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EdgeBudget {
    pub max_rows: usize,
    pub max_bytes: usize,
}

impl EdgeBudget {
    /// Creates a budget; both limits must be positive (S10.1).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the zero field.
    pub fn new(max_rows: usize, max_bytes: usize) -> Result<Self> {
        if max_rows == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "max_rows".into(),
                message: "must be greater than zero".into(),
            });
        }
        if max_bytes == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "max_bytes".into(),
                message: "must be greater than zero".into(),
            });
        }
        Ok(Self {
            max_rows,
            max_bytes,
        })
    }
}

impl Default for EdgeBudget {
    fn default() -> Self {
        Self {
            max_rows: 10_000,
            max_bytes: 64 << 20,
        }
    }
}

/// Runtime-tunable stream configuration (API note A6).
///
/// Every value here is runtime-tunable (spec NFR-5): it feeds the
/// runtime-config hash for observability and diagnostics, never the semantic
/// fingerprint, so tuning it cannot invalidate checkpoints.
#[derive(Clone, Copy, Debug)]
pub struct StreamRuntimeConfig {
    pub checkpoint_interval: Duration,
    pub checkpoint_timeout: Duration,
    pub edge_budget: EdgeBudget,
    pub retained_epochs: usize,
}

impl Default for StreamRuntimeConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: Duration::from_secs(60),
            checkpoint_timeout: Duration::from_secs(600),
            edge_budget: EdgeBudget::default(),
            retained_epochs: 2,
        }
    }
}

impl StreamRuntimeConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when a duration is not an
    /// exact multiple of one microsecond or exceeds the microsecond range.
    pub fn validate(&self) -> Result<()> {
        exact_micros(self.checkpoint_interval, "checkpoint_interval")?;
        exact_micros(self.checkpoint_timeout, "checkpoint_timeout")?;
        EdgeBudget::new(self.edge_budget.max_rows, self.edge_budget.max_bytes)?;
        Ok(())
    }
}

fn exact_micros(duration: Duration, field: &str) -> Result<u64> {
    let nanos = duration.as_nanos();
    if nanos % 1_000 != 0 {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "must be an exact multiple of one microsecond".into(),
        });
    }
    u64::try_from(nanos / 1_000).map_err(|_| CalcFlowError::InvalidArgument {
        field: field.into(),
        message: "exceeds the microsecond range".into(),
    })
}

#[allow(
    dead_code,
    reason = "M2.3 operator tasks drive the compiled stream operators"
)]
pub(crate) enum CompiledStreamOperator {
    External(Box<dyn StreamOperator>),
    Expression(crate::ExpressionOperator),
    Sql(crate::SqlOperator),
    Union(UnionOperator),
}

/// A directly owned compiled node ready to move into one runtime task.
pub(crate) struct RuntimeStreamNode {
    pub(crate) node_id: String,
    pub(crate) operator: CompiledStreamOperator,
    pub(crate) input_ports: BTreeMap<String, Port>,
    pub(crate) output_ports: BTreeMap<String, Port>,
    pub(crate) ingress_edges: BTreeMap<String, String>,
    pub(crate) output_edges: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeEdgeKind {
    SourceBoundary,
    Internal,
    SinkBoundary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeProducer {
    Source { binding_id: String },
    Node { node_id: String, port: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeConsumer {
    Node { node_id: String, ingress: String },
    Sink { output_id: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeSourceRoute {
    pub(crate) binding_id: String,
    pub(crate) target: PortEndpoint,
    pub(crate) edge_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeSinkRoute {
    pub(crate) output_id: String,
    pub(crate) source: PortEndpoint,
    pub(crate) edge_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeEdge {
    pub(crate) stable_id: String,
    pub(crate) kind: RuntimeEdgeKind,
    pub(crate) producer: RuntimeProducer,
    pub(crate) consumer: RuntimeConsumer,
    pub(crate) budget: EdgeBudget,
}

pub(crate) struct StreamRuntimePlanParts {
    pub(crate) name: String,
    pub(crate) fingerprint: String,
    pub(crate) requirements: StreamRequirements,
    pub(crate) nodes: Vec<RuntimeStreamNode>,
    pub(crate) edges: BTreeMap<String, RuntimeEdge>,
    pub(crate) source_routes: BTreeMap<String, RuntimeSourceRoute>,
    pub(crate) sink_routes: BTreeMap<String, RuntimeSinkRoute>,
}

impl CompiledStreamOperator {
    fn convert(definition: NodeDefinition, table: Option<&TablePlanResources>) -> Result<Self> {
        match definition.operator {
            NodeOperator::Expression(mut operator) => {
                if let Some(table) = table {
                    operator.set_stream_resources(
                        table.config,
                        table.udfs.clone(),
                        table.selected_udfs.clone(),
                    );
                }
                Ok(Self::Expression(operator))
            }
            NodeOperator::Sql(mut operator) => {
                if let Some(table) = table {
                    operator.set_stream_resources(
                        table.config,
                        table.udfs.clone(),
                        table.selected_udfs.clone(),
                    );
                }
                Ok(Self::Sql(operator))
            }
            NodeOperator::Union(operator) => Ok(Self::Union(operator)),
            NodeOperator::Stream(operator) => Ok(Self::External(operator)),
            NodeOperator::Batch(_) => Err(CalcFlowError::Compile {
                message: format!(
                    "node {:?} offers only a batch operator; stream graphs require stream-capable operators",
                    definition.node_id
                ),
            }),
        }
    }

    pub(crate) fn reset(&mut self) -> Result<()> {
        match self {
            Self::External(operator) => operator.reset(),
            Self::Expression(operator) => operator.reset(),
            Self::Sql(operator) => operator.reset(),
            Self::Union(operator) => operator.reset(),
        }
    }

    pub(crate) async fn process_data(
        &mut self,
        ingress: &str,
        batch: crate::Batch,
        context: &crate::StreamOperatorContext<'_>,
        output: &mut dyn crate::StreamCollector,
    ) -> Result<()> {
        match self {
            Self::External(operator) => {
                operator.process_data(ingress, batch, context, output).await
            }
            Self::Expression(operator) => {
                operator.process_data(ingress, batch, context, output).await
            }
            Self::Sql(operator) => operator.process_data(ingress, batch, context, output).await,
            Self::Union(operator) => operator.process_data(ingress, batch, context, output).await,
        }
    }

    pub(crate) async fn on_watermark(
        &mut self,
        watermark: crate::EventTime,
        context: &crate::StreamOperatorContext<'_>,
        output: &mut dyn crate::StreamCollector,
    ) -> Result<()> {
        match self {
            Self::External(operator) => operator.on_watermark(watermark, context, output).await,
            Self::Expression(operator) => operator.on_watermark(watermark, context, output).await,
            Self::Sql(operator) => operator.on_watermark(watermark, context, output).await,
            Self::Union(operator) => operator.on_watermark(watermark, context, output).await,
        }
    }

    pub(crate) async fn on_end(
        &mut self,
        context: &crate::StreamOperatorContext<'_>,
        output: &mut dyn crate::StreamCollector,
    ) -> Result<()> {
        match self {
            Self::External(operator) => operator.on_end(context, output).await,
            Self::Expression(operator) => operator.on_end(context, output).await,
            Self::Sql(operator) => operator.on_end(context, output).await,
            Self::Union(operator) => operator.on_end(context, output).await,
        }
    }

    pub(crate) const fn datafusion_runtime_initialized(&self) -> bool {
        match self {
            Self::Expression(operator) => operator.stream_runtime_initialized(),
            Self::Sql(operator) => operator.stream_runtime_initialized(),
            Self::External(_) | Self::Union(_) => false,
        }
    }
}

impl std::fmt::Debug for StreamExecutionPlan {
    /// Diagnostics show the pipeline identity only; operator state never
    /// appears (invariant I4).
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StreamExecutionPlan")
            .field("name", &self.name)
            .field("fingerprint", &self.fingerprint)
            .finish_non_exhaustive()
    }
}

/// The compiled continuously running plan (plan task M1.1).
pub struct StreamExecutionPlan {
    name: String,
    nodes: Vec<RuntimeStreamNode>,
    external_inputs: BTreeMap<String, PortEndpoint>,
    external_outputs: BTreeMap<String, PortEndpoint>,
    edges: BTreeMap<String, Edge>,
    fingerprint: String,
    requirements: StreamRequirements,
    table: Option<TablePlanResources>,
}

impl PipelineBuilder {
    /// Validates and consumes this graph into an immutable stream execution
    /// topology.
    ///
    /// Every stream-rule violation is reported here, before any source opens
    /// (plan M1.1 acceptance gate): multi-alias SQL, batch-only operators,
    /// unknown delivery outputs, and volatile UDFs in an exactly-once plan.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Compile`] for an invalid graph or a stream
    /// rule violation.
    ///
    /// # Panics
    ///
    /// Panics when a node that passed stream-capability validation still
    /// fails conversion; validation runs before conversion, so this is
    /// unreachable and guards the internal invariant only.
    pub fn compile_stream(
        self,
        udfs: &UdfRegistrySnapshot,
        requirements: &StreamRequirements,
    ) -> Result<StreamExecutionPlan> {
        for (node_id, node) in &self.nodes {
            validate_stream_node(node_id, &node.operator)?;
        }
        let graph = compile_graph(&self, "stream", udfs)?;
        for output in requirements.delivery.keys() {
            if !graph.external_outputs.contains_key(output) {
                return Err(CalcFlowError::Compile {
                    message: format!("delivery requirement names unknown graph output {output:?}"),
                });
            }
        }
        validate_deterministic_udfs(&self.nodes, requirements, udfs)?;
        let edges = self
            .edges
            .iter()
            .map(|edge| (edge.stable_id(), edge.clone()))
            .collect::<BTreeMap<_, _>>();
        let name = self.name.clone();
        let table = graph.table;
        let nodes = build_runtime_nodes(self, &graph.order, table.as_ref());
        Ok(StreamExecutionPlan {
            name,
            nodes,
            external_inputs: graph.external_inputs,
            external_outputs: graph.external_outputs,
            edges,
            fingerprint: graph.fingerprint,
            requirements: requirements.clone(),
            table,
        })
    }
}

fn build_runtime_nodes(
    mut builder: PipelineBuilder,
    order: &[String],
    table: Option<&TablePlanResources>,
) -> Vec<RuntimeStreamNode> {
    let inbound = builder
        .edges
        .iter()
        .map(|edge| {
            (
                (edge.target.node_id.clone(), edge.target.port.clone()),
                edge.stable_id(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut outbound = BTreeMap::<(String, String), Vec<String>>::new();
    for edge in &builder.edges {
        outbound
            .entry((edge.source.node_id.clone(), edge.source.port.clone()))
            .or_default()
            .push(edge.stable_id());
    }
    for edge_ids in outbound.values_mut() {
        edge_ids.sort();
        edge_ids.dedup();
    }
    order
        .iter()
        .map(|node_id| {
            let definition = builder
                .nodes
                .remove(node_id)
                .expect("topology contains every validated node exactly once");
            let input_ports = definition
                .operator
                .input_ports()
                .iter()
                .cloned()
                .map(|port| (port.name().to_owned(), port))
                .collect::<BTreeMap<_, _>>();
            let output_ports = definition
                .operator
                .output_ports()
                .iter()
                .cloned()
                .map(|port| (port.name().to_owned(), port))
                .collect::<BTreeMap<_, _>>();
            let ingress_edges = input_ports
                .keys()
                .filter_map(|ingress| {
                    inbound
                        .get(&(node_id.clone(), ingress.clone()))
                        .cloned()
                        .map(|edge_id| (ingress.clone(), edge_id))
                })
                .collect();
            let output_edges = output_ports
                .keys()
                .filter_map(|port| {
                    outbound
                        .get(&(node_id.clone(), port.clone()))
                        .cloned()
                        .map(|edge_ids| (port.clone(), edge_ids))
                })
                .collect();
            RuntimeStreamNode {
                node_id: node_id.clone(),
                operator: CompiledStreamOperator::convert(definition, table)
                    .expect("stream-capable nodes were validated before conversion"),
                input_ports,
                output_ports,
                ingress_edges,
                output_edges,
            }
        })
        .collect()
}

fn validate_stream_node(node_id: &str, operator: &NodeOperator) -> Result<()> {
    match operator {
        NodeOperator::Expression(_) | NodeOperator::Union(_) | NodeOperator::Stream(_) => Ok(()),
        NodeOperator::Sql(operator) if operator.input_ports().len() == 1 => Ok(()),
        NodeOperator::Sql(_) => Err(CalcFlowError::Compile {
            message: format!(
                "stream node {node_id:?} uses multi-input SQL; incremental multi-input joins are unsupported"
            ),
        }),
        NodeOperator::Batch(_) => Err(CalcFlowError::Compile {
            message: format!(
                "node {node_id:?} offers only a batch operator; stream graphs require stream-capable operators"
            ),
        }),
    }
}

fn validate_deterministic_udfs(
    nodes: &BTreeMap<String, NodeDefinition>,
    requirements: &StreamRequirements,
    udfs: &UdfRegistrySnapshot,
) -> Result<()> {
    let exactly_once = requirements
        .delivery
        .values()
        .any(|guarantee| *guarantee == DeliveryGuarantee::ExactlyOnce);
    if !exactly_once {
        return Ok(());
    }
    for (node_id, node) in nodes {
        for reference in node.operator.udf_references() {
            if reference.kind() != UdfKind::DataFusionScalar {
                continue;
            }
            let udf = udfs.resolve_native(&reference)?;
            if matches!(
                udf.signature().volatility,
                datafusion::logical_expr::Volatility::Volatile
            ) {
                return Err(CalcFlowError::Compile {
                    message: format!(
                        "node {node_id:?} selects volatile UDF {}:{}@{}; exactly-once requires deterministic operators",
                        reference.provider(),
                        reference.name(),
                        reference.version()
                    ),
                });
            }
        }
    }
    Ok(())
}

impl StreamExecutionPlan {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The semantic fingerprint (spec NFR-5): execution mode, graph
    /// structure, operator configurations, and the UDF catalog. It decides
    /// checkpoint compatibility.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Hashes runtime-tunable configuration for observability and
    /// diagnostics (spec NFR-5); the value is a pure function of `config`
    /// and never affects the semantic fingerprint. It takes `&self` to pair
    /// with [`Self::fingerprint`] at call sites in the checkpoint manifest.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the configuration is
    /// invalid (see [`StreamRuntimeConfig::validate`]).
    pub fn runtime_config_hash(&self, config: &StreamRuntimeConfig) -> Result<String> {
        config.validate()?;
        let value = json!({
            "checkpoint_interval_micros": exact_micros(config.checkpoint_interval, "checkpoint_interval")?,
            "checkpoint_timeout_micros": exact_micros(config.checkpoint_timeout, "checkpoint_timeout")?,
            "edge_budget": {
                "max_rows": config.edge_budget.max_rows,
                "max_bytes": config.edge_budget.max_bytes,
            },
            "retained_epochs": config.retained_epochs,
        });
        let canonical = canonical_json(&value)?;
        Ok(hex::encode(Sha256::digest(canonical.as_bytes())))
    }

    pub const fn requirements(&self) -> &StreamRequirements {
        &self.requirements
    }

    /// The stable source binding slots: external graph input names in
    /// deterministic order (plan M1.1).
    pub fn source_binding_ids(&self) -> Vec<&str> {
        self.external_inputs.keys().map(String::as_str).collect()
    }

    /// The stable sink binding slots: external graph output names in
    /// deterministic order (plan M1.1).
    pub fn sink_binding_ids(&self) -> Vec<&str> {
        self.external_outputs.keys().map(String::as_str).collect()
    }

    /// The stable edge identifiers in deterministic order (plan M1.1).
    pub fn edge_ids(&self) -> Vec<&str> {
        self.edges.keys().map(String::as_str).collect()
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

    pub const fn datafusion_config(&self) -> Option<DataFusionConfig> {
        match &self.table {
            Some(table) => Some(table.config),
            None => None,
        }
    }

    /// Returns whether any node needs a `DataFusion` session. Pure array
    /// graphs never initialize one (plan section 2.2).
    pub const fn requires_datafusion(&self) -> bool {
        self.table.is_some()
    }

    /// Consumes the compiled plan into directly owned runtime wiring.
    pub(crate) fn into_runtime_parts(
        self,
        default_budget: EdgeBudget,
    ) -> Result<StreamRuntimePlanParts> {
        let StreamExecutionPlan {
            name,
            mut nodes,
            external_inputs,
            external_outputs,
            edges: compiled_edges,
            fingerprint,
            requirements,
            table: _,
        } = self;
        let mut edges = project_internal_edges(compiled_edges, default_budget)?;
        let node_indexes = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.node_id.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let source_routes = project_source_routes(
            &mut nodes,
            &node_indexes,
            external_inputs,
            &mut edges,
            default_budget,
        )?;
        let sink_routes = project_sink_routes(
            &mut nodes,
            &node_indexes,
            external_outputs,
            &mut edges,
            default_budget,
        )?;

        Ok(StreamRuntimePlanParts {
            name,
            fingerprint,
            requirements,
            nodes,
            edges,
            source_routes,
            sink_routes,
        })
    }
}

fn project_internal_edges(
    compiled_edges: BTreeMap<String, Edge>,
    budget: EdgeBudget,
) -> Result<BTreeMap<String, RuntimeEdge>> {
    let mut edges = BTreeMap::new();
    for (stable_id, edge) in compiled_edges {
        insert_runtime_edge(
            &mut edges,
            RuntimeEdge {
                stable_id,
                kind: RuntimeEdgeKind::Internal,
                producer: RuntimeProducer::Node {
                    node_id: edge.source.node_id,
                    port: edge.source.port,
                },
                consumer: RuntimeConsumer::Node {
                    node_id: edge.target.node_id,
                    ingress: edge.target.port,
                },
                budget,
            },
        )?;
    }
    Ok(edges)
}

fn project_source_routes(
    nodes: &mut [RuntimeStreamNode],
    node_indexes: &BTreeMap<String, usize>,
    external_inputs: BTreeMap<String, PortEndpoint>,
    edges: &mut BTreeMap<String, RuntimeEdge>,
    budget: EdgeBudget,
) -> Result<BTreeMap<String, RuntimeSourceRoute>> {
    let mut routes = BTreeMap::new();
    for (binding_id, target) in external_inputs {
        let edge_id = format!(
            "source/{}/{}/{}",
            hex::encode(binding_id.as_bytes()),
            hex::encode(target.node_id.as_bytes()),
            hex::encode(target.port.as_bytes())
        );
        insert_runtime_edge(
            edges,
            RuntimeEdge {
                stable_id: edge_id.clone(),
                kind: RuntimeEdgeKind::SourceBoundary,
                producer: RuntimeProducer::Source {
                    binding_id: binding_id.clone(),
                },
                consumer: RuntimeConsumer::Node {
                    node_id: target.node_id.clone(),
                    ingress: target.port.clone(),
                },
                budget,
            },
        )?;
        let previous = nodes[node_indexes[&target.node_id]]
            .ingress_edges
            .insert(target.port.clone(), edge_id.clone());
        debug_assert!(
            previous.is_none(),
            "external inputs have no internal ingress edge"
        );
        routes.insert(
            binding_id.clone(),
            RuntimeSourceRoute {
                binding_id,
                target,
                edge_id,
            },
        );
    }
    Ok(routes)
}

fn project_sink_routes(
    nodes: &mut [RuntimeStreamNode],
    node_indexes: &BTreeMap<String, usize>,
    external_outputs: BTreeMap<String, PortEndpoint>,
    edges: &mut BTreeMap<String, RuntimeEdge>,
    budget: EdgeBudget,
) -> Result<BTreeMap<String, RuntimeSinkRoute>> {
    let mut routes = BTreeMap::new();
    for (output_id, source) in external_outputs {
        let edge_id = format!(
            "sink/{}/{}/{}",
            hex::encode(source.node_id.as_bytes()),
            hex::encode(source.port.as_bytes()),
            hex::encode(output_id.as_bytes())
        );
        insert_runtime_edge(
            edges,
            RuntimeEdge {
                stable_id: edge_id.clone(),
                kind: RuntimeEdgeKind::SinkBoundary,
                producer: RuntimeProducer::Node {
                    node_id: source.node_id.clone(),
                    port: source.port.clone(),
                },
                consumer: RuntimeConsumer::Sink {
                    output_id: output_id.clone(),
                },
                budget,
            },
        )?;
        nodes[node_indexes[&source.node_id]]
            .output_edges
            .entry(source.port.clone())
            .or_default()
            .push(edge_id.clone());
        routes.insert(
            output_id.clone(),
            RuntimeSinkRoute {
                output_id,
                source,
                edge_id,
            },
        );
    }
    Ok(routes)
}

fn insert_runtime_edge(edges: &mut BTreeMap<String, RuntimeEdge>, edge: RuntimeEdge) -> Result<()> {
    let stable_id = edge.stable_id.clone();
    if edges.insert(stable_id.clone(), edge).is_some() {
        return Err(CalcFlowError::Compile {
            message: format!("runtime edge ID {stable_id:?} is not unique"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod runtime_projection_tests {
    use crate::{
        BatchKind, Edge, PipelineBuilder, Port, PortEndpoint, StreamRequirements, UdfRegistry,
        UnionOperator,
    };

    use super::{EdgeBudget, RuntimeEdgeKind, RuntimeProducer};

    fn endpoint(node_id: &str, port: &str) -> PortEndpoint {
        PortEndpoint::new(node_id, port).unwrap()
    }

    fn union(name: &str, ingresses: &[&str]) -> UnionOperator {
        UnionOperator::new(
            name,
            ingresses
                .iter()
                .map(|ingress| Port::new(ingress, BatchKind::Table, true, None).unwrap())
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn runtime_projection_owns_operators_and_synthesizes_exact_boundary_ids() {
        let plan = PipelineBuilder::new("runtime projection")
            .unwrap()
            .add_node("first", Box::new(union("first", &["left", "right"])))
            .unwrap()
            .add_node("tail", Box::new(union("tail", &["main", "side"])))
            .unwrap()
            .connect(Edge::new(
                endpoint("first", "output"),
                endpoint("tail", "main"),
            ))
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();

        let parts = plan
            .into_runtime_parts(EdgeBudget {
                max_rows: 7,
                max_bytes: 11,
            })
            .unwrap();

        assert_eq!(parts.name, "runtime projection");
        assert!(!parts.fingerprint.is_empty());
        assert_eq!(parts.nodes.len(), 2);
        assert_eq!(parts.nodes[0].node_id, "first");
        assert_eq!(parts.nodes[1].node_id, "tail");
        assert_eq!(parts.nodes[0].input_ports.len(), 2);
        assert_eq!(parts.nodes[1].output_ports.len(), 1);
        assert_eq!(
            parts.source_routes["left"].edge_id,
            "source/6c656674/6669727374/6c656674"
        );
        assert_eq!(
            parts.source_routes["right"].edge_id,
            "source/7269676874/6669727374/7269676874"
        );
        assert_eq!(
            parts.source_routes["side"].edge_id,
            "source/73696465/7461696c/73696465"
        );
        assert_eq!(
            parts.sink_routes["output"].edge_id,
            "sink/7461696c/6f7574707574/6f7574707574"
        );
        assert_eq!(
            parts.edges["first.output->tail.main"].kind,
            RuntimeEdgeKind::Internal
        );
        assert!(matches!(
            parts.edges["source/6c656674/6669727374/6c656674"].producer,
            RuntimeProducer::Source { ref binding_id } if binding_id == "left"
        ));
        assert!(
            parts
                .edges
                .values()
                .all(|edge| edge.budget.max_rows == 7 && edge.budget.max_bytes == 11)
        );
        assert_eq!(parts.edges.len(), 5);
    }
}
