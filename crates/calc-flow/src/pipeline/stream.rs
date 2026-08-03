//! The continuously running stream execution plan (plan task M1.1).
//!
//! A `StreamExecutionPlan` is a compiled artifact: graph topology, stable
//! edge IDs, source/sink binding slots, the semantic fingerprint, and the
//! delivery requirements. It never executes directly; the M2 `StreamingRunner`
//! owns execution. Runtime-tunable values feed only the runtime-config hash
//! (spec NFR-5), never the semantic fingerprint.

use std::{collections::BTreeMap, time::Duration};

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::{
    CalcFlowError, DataFusionConfig, Edge, NodeOperator, OperatorMetadata, PipelineBuilder,
    PortEndpoint, Result, StreamOperator, UdfKind, UdfRegistrySnapshot, UnionOperator,
    canonical_json,
};

use super::{CompiledNode, NodeDefinition, TablePlanResources, build_nodes, compile_graph};

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
    nodes: Vec<CompiledNode<CompiledStreamOperator>>,
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
        let nodes = build_nodes(self, graph.order, |definition| {
            CompiledStreamOperator::convert(definition, table.as_ref())
                .expect("stream-capable nodes were validated before conversion")
        });
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
}
