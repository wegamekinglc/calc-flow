//! Graph building and the batch/stream execution-plan split.
//!
//! [`PipelineBuilder`] collects an immutable graph description and compiles
//! it through one of two entry points: [`PipelineBuilder::compile_batch`]
//! produces a finite one-shot [`BatchExecutionPlan`];
//! [`PipelineBuilder::compile_stream`] produces a continuously running
//! [`StreamExecutionPlan`]. Both entry points share every graph-validation
//! pass in `compile` so the two paths cannot drift (plan task M1.1).

mod batch;
mod compile;
mod stream;

pub use batch::BatchExecutionPlan;
pub use stream::{
    DeliveryGuarantee, EdgeBudget, StreamExecutionPlan, StreamRequirements, StreamRuntimeConfig,
};

pub(crate) use compile::{
    CompiledNode, NodeDefinition, TablePlanResources, build_nodes, compile_graph,
};
pub(crate) use stream::{
    CompiledStreamOperator, OUTPUT_FRONTIER_METADATA_KEY_V1, OperatorCheckpointCapability,
    RuntimeProducer, RuntimeSinkRoute, RuntimeSourceRoute, RuntimeStreamNode,
    StreamRuntimePlanParts,
};

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::{
    Batch, CalcFlowError, CancellationToken, DataFusionQueryMetric, JsonMap, NodeOperator, Result,
    RunContext,
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
#[non_exhaustive]
pub struct RunResult {
    pub outputs: BTreeMap<String, Batch>,
    pub node_timings: BTreeMap<String, NodeTiming>,
    pub datafusion_metrics: Vec<DataFusionQueryMetric>,
    pub metadata: RunMetadata,
    context: RunContext,
}

impl RunResult {
    /// Returns the exact context used to execute this run.
    ///
    /// Sinks use this accessor so delivery observes the same run identity,
    /// settings, deadline, and cancellation token as operators.
    pub const fn context(&self) -> &RunContext {
        &self.context
    }
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

    /// Returns the stable edge identifier assigned at compile time (plan
    /// M1.1): `source.node.port->target.node.port`.
    ///
    /// Node IDs may contain `.` or `->`, so the formatted ID is not
    /// self-delimiting; graph compilation rejects any pair of edges whose
    /// formatted IDs collide (see `compile::validate_edges`).
    pub(crate) fn stable_id(&self) -> String {
        format!(
            "{}.{}->{}.{}",
            self.source.node_id, self.source.port, self.target.node_id, self.target.port
        )
    }
}

pub struct PipelineBuilder {
    pub(crate) name: String,
    pub(crate) datafusion_config: crate::DataFusionConfig,
    pub(crate) nodes: BTreeMap<String, NodeDefinition>,
    pub(crate) edges: Vec<Edge>,
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
            datafusion_config: crate::DataFusionConfig::default(),
            nodes: BTreeMap::new(),
            edges: Vec::new(),
        })
    }

    /// Returns this builder with the run-scoped `DataFusion` configuration.
    #[must_use]
    pub const fn with_datafusion_config(mut self, config: crate::DataFusionConfig) -> Self {
        self.datafusion_config = config;
        self
    }

    /// Returns a new builder that owns the added operator.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Compile`] when `node_id` is empty or already
    /// exists.
    pub fn add_node<O>(self, node_id: &str, operator: O) -> Result<Self>
    where
        O: Into<NodeOperator>,
    {
        let operator = operator.into();
        let checkpoint_capability = match &operator {
            NodeOperator::Expression(_) | NodeOperator::Sql(_) | NodeOperator::Union(_) => {
                OperatorCheckpointCapability::Stateless
            }
            NodeOperator::Window(_) => OperatorCheckpointCapability::CheckpointedStateful {
                state_version: crate::operator::WINDOW_STATE_LAYOUT_VERSION,
            },
            NodeOperator::Rolling(_) => OperatorCheckpointCapability::CheckpointedStateful {
                state_version: crate::operator::ROLLING_STATE_LAYOUT_VERSION,
            },
            NodeOperator::CrossSection(_) => OperatorCheckpointCapability::CheckpointedStateful {
                state_version: crate::operator::CROSS_SECTION_STATE_LAYOUT_VERSION,
            },
            NodeOperator::StreamJoin(_) => {
                OperatorCheckpointCapability::CheckpointedStateful { state_version: 1 }
            }
            NodeOperator::Stream(operator) if operator.lifecycle().is_proven_stateless() => {
                OperatorCheckpointCapability::Stateless
            }
            NodeOperator::Batch(_) | NodeOperator::Stream(_) => {
                OperatorCheckpointCapability::Unproven
            }
        };
        self.add_node_with_capability(node_id, operator, checkpoint_capability)
    }

    #[cfg(test)]
    pub(crate) fn add_checkpoint_capable_node<O>(self, node_id: &str, operator: O) -> Result<Self>
    where
        O: Into<NodeOperator>,
    {
        self.add_checkpoint_capable_node_with_version(node_id, operator, 1)
    }

    #[cfg(test)]
    pub(crate) fn add_checkpoint_capable_node_with_version<O>(
        self,
        node_id: &str,
        operator: O,
        state_version: u32,
    ) -> Result<Self>
    where
        O: Into<NodeOperator>,
    {
        self.add_node_with_capability(
            node_id,
            operator.into(),
            OperatorCheckpointCapability::CheckpointedStateful { state_version },
        )
    }

    fn add_node_with_capability(
        mut self,
        node_id: &str,
        operator: NodeOperator,
        checkpoint_capability: OperatorCheckpointCapability,
    ) -> Result<Self> {
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
                checkpoint_capability,
            },
        );
        Ok(self)
    }

    /// Returns a new builder containing the directed edge.
    ///
    /// Port validation is deferred until compilation so every operator
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
}

pub(crate) fn row_counts(batches: &BTreeMap<String, Batch>) -> BTreeMap<String, usize> {
    batches
        .iter()
        .map(|(name, batch)| (name.clone(), batch.num_rows()))
        .collect()
}

pub(crate) fn nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

pub(crate) fn lifecycle_result(action: &str, failures: &[String]) -> Result<()> {
    if failures.is_empty() {
        Ok(())
    } else {
        Err(CalcFlowError::Internal {
            message: format!("operator {action} failed: {}", failures.join("; ")),
        })
    }
}
