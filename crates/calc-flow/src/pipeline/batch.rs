//! The finite one-shot execution plan and its state-lifecycle machinery.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Mutex as StdMutex,
    time::Instant,
};

use serde_json::Value;

use crate::{
    Batch, BatchOperator, BatchOperatorContext, CalcFlowError, DataFusionConfig, DataFusionRuntime,
    ExecutionOptions, ExpressionOperator, NodeOperator, OperatorMetadata, PipelineBuilder, Port,
    PortEndpoint, Result, RunContext, RunMetadata, RunResult, SqlOperator, UdfRegistrySnapshot,
};

use super::{
    CompiledNode, NodeDefinition, NodeTiming, TablePlanResources, build_nodes, compile_graph,
    lifecycle_result, nanos, row_counts,
};

pub(crate) enum CompiledBatchOperator {
    External(Box<dyn BatchOperator>),
    Expression(ExpressionOperator),
    Sql(SqlOperator),
    Rolling(crate::RollingOperator),
    CrossSection(crate::CrossSectionOperator),
}

impl CompiledBatchOperator {
    fn try_convert(definition: NodeDefinition) -> Result<Self> {
        match definition.operator {
            NodeOperator::Expression(operator) => Ok(Self::Expression(operator)),
            NodeOperator::Sql(operator) => Ok(Self::Sql(operator)),
            NodeOperator::Rolling(operator) => Ok(Self::Rolling(operator)),
            NodeOperator::CrossSection(operator) => Ok(Self::CrossSection(operator)),
            NodeOperator::Batch(operator) => Ok(Self::External(operator)),
            NodeOperator::Union(_)
            | NodeOperator::Window(_)
            | NodeOperator::StreamJoin(_)
            | NodeOperator::Stream(_) => Err(CalcFlowError::Compile {
                message: format!(
                    "node {:?} is stream-only; batch graphs compose multi-input logic through SQL aliases",
                    definition.node_id
                ),
            }),
        }
    }

    pub(crate) fn snapshot(&self) -> Result<Value> {
        match self {
            Self::External(operator) => operator.snapshot(),
            Self::Expression(_) | Self::Sql(_) | Self::Rolling(_) | Self::CrossSection(_) => {
                Ok(Value::Null)
            }
        }
    }

    pub(crate) fn restore(&mut self, state: &Value) -> Result<()> {
        match self {
            Self::External(operator) => operator.restore(state),
            Self::Expression(_) | Self::Sql(_) | Self::Rolling(_) | Self::CrossSection(_)
                if state.is_null() =>
            {
                Ok(())
            }
            Self::Expression(_) | Self::Sql(_) | Self::Rolling(_) | Self::CrossSection(_) => {
                Err(CalcFlowError::Format {
                    message: "stateless operator state must be null".into(),
                })
            }
        }
    }

    pub(crate) fn reset(&mut self) -> Result<()> {
        match self {
            Self::External(operator) => operator.reset(),
            Self::Expression(_) | Self::Sql(_) | Self::Rolling(_) | Self::CrossSection(_) => Ok(()),
        }
    }

    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        run: &RunContext,
        datafusion: Option<&DataFusionRuntime>,
    ) -> Result<BTreeMap<String, Batch>> {
        match self {
            Self::External(operator) => {
                operator
                    .process(inputs, &BatchOperatorContext { run })
                    .await
            }
            Self::Rolling(operator) => {
                operator
                    .process(inputs, &BatchOperatorContext { run })
                    .await
            }
            Self::CrossSection(operator) => {
                operator
                    .process(inputs, &BatchOperatorContext { run })
                    .await
            }
            Self::Expression(operator) => {
                let datafusion = required_datafusion(datafusion, operator.name())?;
                operator.process_table(inputs, run, datafusion).await
            }
            Self::Sql(operator) => {
                let datafusion = required_datafusion(datafusion, operator.name())?;
                operator.process_table(inputs, run, datafusion).await
            }
        }
    }
}

fn required_datafusion<'a>(
    datafusion: Option<&'a DataFusionRuntime>,
    operator: &str,
) -> Result<&'a DataFusionRuntime> {
    datafusion.ok_or_else(|| CalcFlowError::Internal {
        message: format!("table operator {operator:?} has no run-scoped DataFusion runtime"),
    })
}

impl std::fmt::Debug for BatchExecutionPlan {
    /// Diagnostics show the pipeline identity only; operator state never
    /// appears (invariant I4).
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BatchExecutionPlan")
            .field("name", &self.name)
            .field("fingerprint", &self.fingerprint)
            .finish_non_exhaustive()
    }
}

pub struct BatchExecutionPlan {
    pub(crate) name: String,
    pub(crate) nodes: Vec<CompiledNode<CompiledBatchOperator>>,
    pub(crate) external_inputs: BTreeMap<String, PortEndpoint>,
    pub(crate) external_outputs: BTreeMap<String, PortEndpoint>,
    pub(crate) fingerprint: String,
    table: Option<TablePlanResources>,
    pub(crate) run_lock: tokio::sync::Mutex<()>,
    operation_state: StdMutex<OperationState>,
}

#[derive(Default)]
struct OperationState {
    generation: u64,
    in_flight: Option<InFlightOperation>,
}

#[derive(Clone)]
struct InFlightOperation {
    token: u64,
    state: BTreeMap<String, Value>,
    clear_after_recovery: bool,
}

#[derive(Clone, Copy)]
struct OperationToken(u64);

/// Crate-internal state transaction that owns a plan's lifecycle lock.
///
/// The guard prevents concurrent callers from observing intermediate state.
struct PlanTransaction<'a> {
    plan: &'a BatchExecutionPlan,
    _guard: tokio::sync::MutexGuard<'a, ()>,
}

impl PipelineBuilder {
    /// Validates and consumes this graph into an immutable batch execution
    /// topology (the v2 finite one-shot contract).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Compile`] for an invalid graph, a stream-only
    /// node, or a selected UDF catalog problem.
    ///
    /// # Panics
    ///
    /// Panics when a node that passed the stream-only rejection above still
    /// fails conversion; validation runs before conversion, so this is
    /// unreachable and guards the internal invariant only.
    pub fn compile_batch(self, udfs: &UdfRegistrySnapshot) -> Result<BatchExecutionPlan> {
        for (node_id, node) in &self.nodes {
            if matches!(
                node.operator,
                NodeOperator::Union(_)
                    | NodeOperator::Window(_)
                    | NodeOperator::StreamJoin(_)
                    | NodeOperator::Stream(_)
            ) {
                return Err(CalcFlowError::Compile {
                    message: format!(
                        "node {node_id:?} is stream-only; batch graphs compose multi-input logic through SQL aliases"
                    ),
                });
            }
        }
        let graph = compile_graph(&self, "batch", udfs)?;
        let name = self.name.clone();
        let nodes = build_nodes(self, graph.order, |definition| {
            CompiledBatchOperator::try_convert(definition)
                .expect("batch-only nodes were validated before conversion")
        });
        Ok(BatchExecutionPlan {
            name,
            nodes,
            external_inputs: graph.external_inputs,
            external_outputs: graph.external_outputs,
            fingerprint: graph.fingerprint,
            table: graph.table,
            run_lock: tokio::sync::Mutex::new(()),
            operation_state: StdMutex::new(OperationState::default()),
        })
    }
}

impl BatchExecutionPlan {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub const fn datafusion_config(&self) -> Option<DataFusionConfig> {
        match &self.table {
            Some(table) => Some(table.config),
            None => None,
        }
    }

    pub const fn requires_datafusion(&self) -> bool {
        self.table.is_some()
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

    /// Returns the only external input name accepted by a runner.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] unless the compiled graph
    /// exposes exactly one external input.
    pub fn single_external_input(&self) -> Result<&str> {
        if self.external_inputs.len() != 1 {
            return Err(CalcFlowError::InvalidArgument {
                field: "plan.external_inputs".into(),
                message: "runners require exactly one external input".into(),
            });
        }
        self.external_inputs
            .keys()
            .next()
            .map(String::as_str)
            .ok_or_else(|| CalcFlowError::Internal {
                message: "single external input disappeared after validation".into(),
            })
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
        let transaction = self.public_transaction().await?;
        transaction.validate_inputs(&inputs)?;
        let before = transaction.snapshot().await?;
        let operation = transaction.begin_rollback(before)?;
        let result = transaction.execute_validated(inputs, options).await;
        match result {
            Ok(result) => {
                transaction.commit_operation(operation)?;
                Ok(result)
            }
            Err(original) => Err(transaction.rollback_error(operation, original).await),
        }
    }

    /// Captures every node's JSON state under the plan's run lock.
    ///
    /// # Errors
    ///
    /// Returns an error when any operator cannot capture its state.
    pub async fn snapshot(&self) -> Result<BTreeMap<String, Value>> {
        self.public_transaction().await?.snapshot().await
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
        let transaction = self.public_transaction().await?;
        transaction.validate_state(state)?;
        let before = transaction.snapshot().await?;
        let operation = transaction.begin_rollback(before)?;
        match transaction.restore(state).await {
            Ok(()) => transaction.commit_operation(operation),
            Err(original) => Err(transaction.rollback_error(operation, original).await),
        }
    }

    /// Resets every node under the plan's run lock.
    ///
    /// All nodes are attempted even if one reset fails.
    ///
    /// # Errors
    ///
    /// Returns an error summarizing operator reset failures.
    pub async fn reset(&self) -> Result<()> {
        let transaction = self.raw_public_transaction().await;
        if transaction.recover_in_flight().await.is_err() {
            let before = transaction.snapshot().await?;
            let operation = transaction.replace_for_direct_forced_reset(before)?;
            return match transaction.reset().await {
                Ok(()) => transaction.commit_operation(operation),
                Err(original) => Err(transaction.rollback_error(operation, original).await),
            };
        }
        let before = transaction.snapshot().await?;
        let operation = transaction.begin_rollback(before)?;
        match transaction.reset().await {
            Ok(()) => transaction.commit_operation(operation),
            Err(original) => Err(transaction.rollback_error(operation, original).await),
        }
    }

    async fn public_transaction(&self) -> Result<PlanTransaction<'_>> {
        let transaction = self.raw_public_transaction().await;
        transaction.recover_in_flight().await?;
        Ok(transaction)
    }

    async fn raw_public_transaction(&self) -> PlanTransaction<'_> {
        let guard = self.run_lock.lock().await;
        PlanTransaction {
            plan: self,
            _guard: guard,
        }
    }

    async fn reset_unlocked(&self) -> Result<()> {
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
        let mut runtime = self
            .table
            .as_ref()
            .map(|table| {
                let mut runtime = DataFusionRuntime::new(table.config)?;
                runtime.register_udfs(&table.udfs, &table.selected_udfs)?;
                Ok::<DataFusionRuntime, CalcFlowError>(runtime)
            })
            .transpose()?;
        let execution = self
            .execute_nodes(&inputs, &context, runtime.as_ref())
            .await;
        if let Some(runtime) = &mut runtime {
            runtime.close();
        }
        let mut datafusion_metrics = runtime
            .as_ref()
            .map_or_else(Vec::new, DataFusionRuntime::metrics);
        let (outputs, node_timings) = execution?;
        let run_result_start = Instant::now();
        let metadata = RunMetadata {
            run_id: context.run_id().into(),
            pipeline_name: self.name.clone(),
            pipeline_fingerprint: self.fingerprint.clone(),
        };
        let run_result_ns = nanos(run_result_start.elapsed());
        if let Some(metric) = datafusion_metrics.last_mut() {
            metric.run_result_ns = run_result_ns;
        }
        Ok(RunResult {
            outputs,
            node_timings,
            datafusion_metrics,
            metadata,
            context,
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
        runtime: Option<&DataFusionRuntime>,
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
                .process(&operator_inputs, &node_context, runtime)
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
            .collect::<BTreeMap<_, _>>();
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
        self.validate_state_map(state)?;
        let mut failures = Vec::new();
        for node in &self.nodes {
            if let Err(error) = node.operator.lock().await.restore(&state[&node.node_id]) {
                failures.push(format!("{}: {error}", node.node_id));
            }
        }
        lifecycle_result("restore", &failures)
    }

    fn validate_state_map(&self, state: &BTreeMap<String, Value>) -> Result<()> {
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
        Ok(())
    }

    fn node(&self, node_id: &str) -> Result<&CompiledNode<CompiledBatchOperator>> {
        self.nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .ok_or_else(|| CalcFlowError::Internal {
                message: format!("compiled plan has no node {node_id}"),
            })
    }
}

impl PlanTransaction<'_> {
    fn validate_inputs(&self, inputs: &BTreeMap<String, Batch>) -> Result<()> {
        self.plan.validate_external_inputs(inputs)
    }

    fn validate_state(&self, state: &BTreeMap<String, Value>) -> Result<()> {
        self.plan.validate_state_map(state)
    }

    async fn execute_validated(
        &self,
        inputs: BTreeMap<String, Batch>,
        options: ExecutionOptions,
    ) -> Result<RunResult> {
        self.plan.execute_unlocked(inputs, options).await
    }

    async fn snapshot(&self) -> Result<BTreeMap<String, Value>> {
        self.plan.snapshot_unlocked().await
    }

    async fn restore(&self, state: &BTreeMap<String, Value>) -> Result<()> {
        self.plan.restore_unlocked(state).await
    }

    async fn reset(&self) -> Result<()> {
        self.plan.reset_unlocked().await
    }

    fn begin_rollback(&self, state: BTreeMap<String, Value>) -> Result<OperationToken> {
        self.begin_operation(state, true)
    }

    fn replace_for_direct_forced_reset(
        &self,
        state_before_reset: BTreeMap<String, Value>,
    ) -> Result<OperationToken> {
        let mut state = self
            .plan
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.in_flight.is_none() {
            return Err(self
                .recovery_error("only a direct rollback marker can be replaced by public reset"));
        }
        let token = state
            .generation
            .checked_add(1)
            .ok_or_else(|| CalcFlowError::Internal {
                message: format!(
                    "execution plan {:?} exhausted operation generations",
                    self.plan.name
                ),
            })?;
        state.generation = token;
        state.in_flight = Some(InFlightOperation {
            token,
            state: state_before_reset,
            clear_after_recovery: false,
        });
        Ok(OperationToken(token))
    }

    fn begin_operation(
        &self,
        rollback_state: BTreeMap<String, Value>,
        clear_after_recovery: bool,
    ) -> Result<OperationToken> {
        let mut state = self
            .plan
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.in_flight.is_some() {
            return Err(self.recovery_error("another operation is still marked in flight"));
        }
        let token = state
            .generation
            .checked_add(1)
            .ok_or_else(|| CalcFlowError::Internal {
                message: format!(
                    "execution plan {:?} exhausted operation generations",
                    self.plan.name
                ),
            })?;
        state.generation = token;
        state.in_flight = Some(InFlightOperation {
            token,
            state: rollback_state,
            clear_after_recovery,
        });
        Ok(OperationToken(token))
    }

    fn commit_operation(&self, token: OperationToken) -> Result<()> {
        let mut state = self
            .plan
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &state.in_flight {
            Some(operation) if operation.token == token.0 => {
                state.in_flight = None;
                Ok(())
            }
            _ => Err(self.stale_operation_error(token)),
        }
    }

    async fn recover_in_flight(&self) -> Result<()> {
        let operation = self
            .plan
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .in_flight
            .clone();
        let Some(operation) = operation else {
            return Ok(());
        };

        self.restore(&operation.state)
            .await
            .map_err(|error| self.recovery_error(&format!("plan restoration failed: {error}")))?;
        if !operation.clear_after_recovery {
            return Err(
                self.recovery_error("an interrupted forced reset must be explicitly reset again")
            );
        }
        self.commit_operation(OperationToken(operation.token))
    }

    async fn rollback_error(
        &self,
        token: OperationToken,
        original: CalcFlowError,
    ) -> CalcFlowError {
        if let Err(ownership) = self.ensure_operation_owner(token) {
            return CalcFlowError::Internal {
                message: format!(
                    "operation failed with {original}; rollback ownership also failed with {ownership}"
                ),
            };
        }
        match self.recover_in_flight().await {
            Ok(()) => original,
            Err(rollback) => CalcFlowError::Internal {
                message: format!(
                    "operation failed with {original}; rollback also failed with {rollback}"
                ),
            },
        }
    }

    fn ensure_operation_owner(&self, token: OperationToken) -> Result<()> {
        let state = self
            .plan
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match &state.in_flight {
            Some(operation) if operation.token == token.0 => Ok(()),
            _ => Err(self.stale_operation_error(token)),
        }
    }

    fn stale_operation_error(&self, token: OperationToken) -> CalcFlowError {
        CalcFlowError::Internal {
            message: format!(
                "operation {} no longer owns execution plan {:?}'s recovery marker",
                token.0, self.plan.name
            ),
        }
    }

    fn recovery_error(&self, message: &str) -> CalcFlowError {
        CalcFlowError::RecoveryRequired {
            pipeline_name: self.plan.name.clone(),
            message: message.into(),
        }
    }
}

fn gather_node_inputs(
    node: &CompiledNode<CompiledBatchOperator>,
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
    node: &CompiledNode<CompiledBatchOperator>,
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
    node: &CompiledNode<CompiledBatchOperator>,
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

#[cfg(test)]
mod lifecycle_tests {
    use std::{
        future::Future,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        task::Poll,
    };

    use async_trait::async_trait;
    use chrono::Utc;
    use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};
    use serde_json::json;

    use super::*;
    use crate::{
        BatchKind, BatchMetadata, BatchOperator, CancellationToken, JsonMap, OperatorMetadata,
        UdfRegistry,
    };

    struct LifecycleGate {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
        restores: Arc<AtomicUsize>,
        fail_snapshot: bool,
        fail_restore: bool,
        state: usize,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    }

    impl OperatorMetadata for LifecycleGate {
        fn name(&self) -> &'static str {
            "lifecycle_gate"
        }

        fn input_ports(&self) -> &[Port] {
            &self.inputs
        }

        fn output_ports(&self) -> &[Port] {
            &self.outputs
        }

        fn configuration(&self) -> JsonMap {
            BTreeMap::new()
        }
    }

    #[async_trait]
    impl BatchOperator for LifecycleGate {
        async fn process(
            &mut self,
            inputs: &BTreeMap<String, Batch>,
            _context: &BatchOperatorContext<'_>,
        ) -> Result<BTreeMap<String, Batch>> {
            self.state += 1;
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.started.notify_one();
                self.release.notified().await;
            }
            Ok(BTreeMap::from([("output".into(), inputs["input"].clone())]))
        }

        fn snapshot(&self) -> Result<Value> {
            if self.fail_snapshot {
                return Err(CalcFlowError::Format {
                    message: "snapshot failure injected".into(),
                });
            }
            Ok(json!(self.state))
        }

        fn restore(&mut self, state: &Value) -> Result<()> {
            self.restores.fetch_add(1, Ordering::SeqCst);
            if self.fail_restore {
                return Err(CalcFlowError::Format {
                    message: "restore failure injected".into(),
                });
            }
            self.state = state
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| CalcFlowError::Format {
                    message: "lifecycle gate state is invalid".into(),
                })?;
            Ok(())
        }
    }

    fn lifecycle_plan(
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
        restores: Arc<AtomicUsize>,
        fail_snapshot: bool,
        fail_restore: bool,
    ) -> Arc<BatchExecutionPlan> {
        let input = Port::new("input", BatchKind::Table, true, None).unwrap();
        let output = Port::new("output", BatchKind::Table, true, None).unwrap();
        Arc::new(
            PipelineBuilder::new("queued lifecycle")
                .unwrap()
                .add_node(
                    "gate",
                    Box::new(LifecycleGate {
                        started,
                        release,
                        calls,
                        restores,
                        fail_snapshot,
                        fail_restore,
                        state: 0,
                        inputs: vec![input],
                        outputs: vec![output],
                    }) as Box<dyn BatchOperator>,
                )
                .unwrap()
                .compile_batch(&UdfRegistry::new().snapshot())
                .unwrap(),
        )
    }

    fn inputs() -> BTreeMap<String, Batch> {
        let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
        BTreeMap::from([(
            "input".into(),
            Batch::table(vec![batch], BatchMetadata::default()).unwrap(),
        )])
    }

    async fn assert_pending_once(future: &mut (impl Future + Unpin)) {
        std::future::poll_fn(
            |context| match Future::poll(std::pin::Pin::new(future), context) {
                Poll::Pending => Poll::Ready(()),
                Poll::Ready(_) => panic!("queued execution unexpectedly completed"),
            },
        )
        .await;
    }

    fn marker(plan: &BatchExecutionPlan) -> Option<u64> {
        plan.operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .in_flight
            .as_ref()
            .map(|operation| operation.token)
    }

    #[tokio::test]
    async fn queued_cancellation_is_pending_before_any_second_marker_and_plan_recovers() {
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let restores = Arc::new(AtomicUsize::new(0));
        let plan = lifecycle_plan(
            Arc::clone(&started),
            Arc::clone(&release),
            Arc::clone(&calls),
            Arc::clone(&restores),
            false,
            false,
        );

        let mut first = Box::pin(plan.execute(inputs(), ExecutionOptions::default()));
        tokio::select! {
            () = started.notified() => {}
            result = &mut first => panic!("first execution did not hold the gate: {result:?}"),
        }
        let first_marker = marker(&plan).expect("the active run must own a rollback marker");

        let cancellation = CancellationToken::new();
        let mut second = Box::pin(plan.execute(
            inputs(),
            ExecutionOptions {
                cancellation: cancellation.clone(),
                ..ExecutionOptions::default()
            },
        ));
        assert_pending_once(&mut second).await;
        assert_eq!(marker(&plan), Some(first_marker));

        cancellation.cancel();
        assert_eq!(marker(&plan), Some(first_marker));
        release.notify_one();
        first.await.unwrap();
        assert!(matches!(
            second.await.unwrap_err(),
            CalcFlowError::Cancelled { .. }
        ));

        plan.execute(inputs(), ExecutionOptions::default())
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(restores.load(Ordering::SeqCst), 1);
        assert_eq!(marker(&plan), None);
    }

    #[tokio::test]
    async fn queued_deadline_is_absolute_and_expires_before_second_provider_entry() {
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let restores = Arc::new(AtomicUsize::new(0));
        let plan = lifecycle_plan(
            Arc::clone(&started),
            Arc::clone(&release),
            Arc::clone(&calls),
            Arc::clone(&restores),
            false,
            false,
        );

        let mut first = Box::pin(plan.execute(inputs(), ExecutionOptions::default()));
        tokio::select! {
            () = started.notified() => {}
            result = &mut first => panic!("first execution did not hold the gate: {result:?}"),
        }

        let deadline = Utc::now() + chrono::Duration::milliseconds(50);
        let mut second = Box::pin(plan.execute(
            inputs(),
            ExecutionOptions {
                deadline: Some(deadline),
                ..ExecutionOptions::default()
            },
        ));
        assert_pending_once(&mut second).await;
        let wait = deadline
            .signed_duration_since(Utc::now())
            .to_std()
            .unwrap_or_default();
        tokio::time::sleep(wait + std::time::Duration::from_millis(10)).await;

        release.notify_one();
        first.await.unwrap();
        assert!(matches!(
            second.await.unwrap_err(),
            CalcFlowError::Cancelled { .. }
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        plan.execute(inputs(), ExecutionOptions::default())
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(restores.load(Ordering::SeqCst), 1);
        assert_eq!(marker(&plan), None);
    }

    #[tokio::test]
    async fn pre_checkpoint_input_snapshot_and_marker_failures_beat_expired_deadline() {
        let expired = Utc::now() - chrono::Duration::seconds(1);

        let input_calls = Arc::new(AtomicUsize::new(0));
        let input_restores = Arc::new(AtomicUsize::new(0));
        let input_plan = lifecycle_plan(
            Arc::new(tokio::sync::Notify::new()),
            Arc::new(tokio::sync::Notify::new()),
            Arc::clone(&input_calls),
            Arc::clone(&input_restores),
            false,
            false,
        );
        let input_error = input_plan
            .execute(
                BTreeMap::from([("wrong".into(), inputs().remove("input").unwrap())]),
                ExecutionOptions {
                    deadline: Some(expired),
                    ..ExecutionOptions::default()
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(
            input_error,
            CalcFlowError::InvalidArgument { field, .. } if field == "inputs"
        ));
        assert_eq!(input_calls.load(Ordering::SeqCst), 0);
        assert_eq!(input_restores.load(Ordering::SeqCst), 0);

        let snapshot_calls = Arc::new(AtomicUsize::new(0));
        let snapshot_restores = Arc::new(AtomicUsize::new(0));
        let snapshot_plan = lifecycle_plan(
            Arc::new(tokio::sync::Notify::new()),
            Arc::new(tokio::sync::Notify::new()),
            Arc::clone(&snapshot_calls),
            Arc::clone(&snapshot_restores),
            true,
            false,
        );
        let snapshot_error = snapshot_plan
            .execute(
                inputs(),
                ExecutionOptions {
                    deadline: Some(expired),
                    ..ExecutionOptions::default()
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(
            snapshot_error,
            CalcFlowError::Format { message } if message == "snapshot failure injected"
        ));
        assert_eq!(snapshot_calls.load(Ordering::SeqCst), 0);
        assert_eq!(snapshot_restores.load(Ordering::SeqCst), 0);

        let marker_calls = Arc::new(AtomicUsize::new(0));
        let marker_restores = Arc::new(AtomicUsize::new(0));
        let marker_plan = lifecycle_plan(
            Arc::new(tokio::sync::Notify::new()),
            Arc::new(tokio::sync::Notify::new()),
            Arc::clone(&marker_calls),
            Arc::clone(&marker_restores),
            false,
            false,
        );
        marker_plan
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .generation = u64::MAX;
        let marker_error = marker_plan
            .execute(
                inputs(),
                ExecutionOptions {
                    deadline: Some(expired),
                    ..ExecutionOptions::default()
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(
            marker_error,
            CalcFlowError::Internal { message }
                if message.contains("exhausted operation generations")
        ));
        assert_eq!(marker_calls.load(Ordering::SeqCst), 0);
        assert_eq!(marker_restores.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn abandoned_recovery_failure_beats_expired_deadline_and_provider_entry() {
        let started = Arc::new(tokio::sync::Notify::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let restores = Arc::new(AtomicUsize::new(0));
        let plan = lifecycle_plan(
            Arc::clone(&started),
            Arc::new(tokio::sync::Notify::new()),
            Arc::clone(&calls),
            Arc::clone(&restores),
            false,
            true,
        );
        let mut abandoned = Box::pin(plan.execute(inputs(), ExecutionOptions::default()));
        tokio::select! {
            () = started.notified() => {}
            result = &mut abandoned => panic!("execution did not reach the gate: {result:?}"),
        }
        drop(abandoned);

        let error = plan
            .execute(
                inputs(),
                ExecutionOptions {
                    deadline: Some(Utc::now() - chrono::Duration::seconds(1)),
                    ..ExecutionOptions::default()
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            CalcFlowError::RecoveryRequired { message, .. }
                if message.contains("plan restoration failed")
                    && message.contains("restore failure injected")
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(restores.load(Ordering::SeqCst), 1);
    }
}

#[cfg(test)]
mod data_tests {
    use std::{collections::BTreeMap, sync::Arc};

    use datafusion::arrow::{
        array::{Array, Int64Array},
        record_batch::RecordBatch,
    };

    use super::{gather_node_inputs, validate_and_store_outputs};
    use crate::{
        Batch, BatchMetadata, ExpressionOperator, PipelineBuilder, PortEndpoint, UdfRegistry,
    };

    #[test]
    fn endpoint_slots_preserve_batch_payload_identity() {
        let plan = PipelineBuilder::new("data slots")
            .unwrap()
            .add_node(
                "expression",
                Box::new(
                    ExpressionOperator::new(
                        "expression",
                        "plus_one = value + 1",
                        Vec::new(),
                        None,
                        Vec::new(),
                    )
                    .unwrap(),
                ),
            )
            .unwrap()
            .compile_batch(&UdfRegistry::new().snapshot())
            .unwrap();
        let node = &plan.nodes[0];
        let input_endpoint = plan.external_inputs["input"].clone();
        let record_batch = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as Arc<dyn Array>,
        )])
        .unwrap();
        let batch = Batch::table(vec![record_batch], BatchMetadata::default()).unwrap();
        let schema = Arc::clone(batch.table_payload().unwrap().schema());
        let external_names = BTreeMap::from([(input_endpoint.clone(), "input".to_owned())]);
        let external_values = BTreeMap::from([(input_endpoint, batch.clone())]);

        let inputs =
            gather_node_inputs(node, &BTreeMap::new(), &external_values, &external_names).unwrap();

        assert!(Arc::ptr_eq(
            inputs["input"].table_payload().unwrap().schema(),
            &schema
        ));

        let mut produced_values = BTreeMap::new();
        validate_and_store_outputs(
            node,
            &BTreeMap::from([("output".to_owned(), batch)]),
            &mut produced_values,
        )
        .unwrap();
        let output_endpoint = PortEndpoint {
            node_id: "expression".to_owned(),
            port: "output".to_owned(),
        };

        assert!(Arc::ptr_eq(
            produced_values[&output_endpoint]
                .table_payload()
                .unwrap()
                .schema(),
            &schema
        ));
    }
}
