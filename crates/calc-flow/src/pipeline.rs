use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex as StdMutex},
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
    Batch, CalcFlowError, CancellationToken, Checkpoint, CheckpointStore, DataFusionConfig,
    DataFusionQueryMetric, DataFusionRuntime, JsonMap, OperatorDefinition, Port, Result,
    RunContext, UdfCatalogEntry, UdfReference, UdfRegistrySnapshot, canonical_json,
    validate_selected_udfs,
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
}

struct NodeDefinition {
    node_id: String,
    operator: OperatorDefinition,
}

pub struct PipelineBuilder {
    name: String,
    datafusion_config: DataFusionConfig,
    nodes: BTreeMap<String, NodeDefinition>,
    edges: Vec<Edge>,
}

#[allow(dead_code)]
pub(crate) struct CompiledNode {
    pub(crate) node_id: String,
    pub(crate) operator: Arc<tokio::sync::Mutex<OperatorDefinition>>,
    pub(crate) input_ports: Vec<Port>,
    pub(crate) output_ports: Vec<Port>,
    pub(crate) inbound: BTreeMap<String, PortEndpoint>,
}

struct TablePlanResources {
    config: DataFusionConfig,
    udfs: UdfRegistrySnapshot,
    selected_udfs: Vec<UdfReference>,
}

pub struct ExecutionPlan {
    pub(crate) name: String,
    pub(crate) nodes: Vec<CompiledNode>,
    pub(crate) external_inputs: BTreeMap<String, PortEndpoint>,
    pub(crate) external_outputs: BTreeMap<String, PortEndpoint>,
    pub(crate) fingerprint: String,
    table: Option<TablePlanResources>,
    pub(crate) run_lock: tokio::sync::Mutex<()>,
    lease_state: StdMutex<LeaseState>,
    operation_state: StdMutex<OperationState>,
}

#[derive(Default)]
struct LeaseState {
    owner: Option<u64>,
    generation: u64,
}

#[derive(Default)]
struct OperationState {
    generation: u64,
    in_flight: Option<InFlightOperation>,
}

#[derive(Clone)]
struct InFlightOperation {
    token: u64,
    runner_owner: Option<u64>,
    action: RecoveryAction,
}

#[derive(Clone)]
enum RecoveryAction {
    Rollback {
        state: BTreeMap<String, Value>,
        checkpoint: CheckpointRecovery,
        store_mutation_started: bool,
        clear_after_recovery: bool,
    },
    CommitCheckpoint {
        checkpoint: Checkpoint,
    },
}

#[derive(Clone)]
enum CheckpointRecovery {
    NotRequired,
    Restore(Option<Checkpoint>),
}

#[derive(Clone, Copy)]
pub(crate) struct OperationToken(u64);

pub(crate) struct RollbackOutcome {
    pub(crate) error: CalcFlowError,
    pub(crate) recovery_failed: bool,
}

pub(crate) enum RecoveryOutcome {
    None,
    RolledBack,
    CommittedCheckpoint(Checkpoint),
}

impl RecoveryOutcome {
    pub(crate) const fn committed_checkpoint(&self) -> Option<&Checkpoint> {
        match self {
            Self::CommittedCheckpoint(checkpoint) => Some(checkpoint),
            Self::None | Self::RolledBack => None,
        }
    }
}

/// Exclusive, non-cloneable runner ownership of an execution plan.
///
/// The synchronous lease flag makes reentrant public calls fail before they
/// can wait on the async run lock. The lock is still rechecked after every
/// acquisition so calls queued before this lease was created cannot slip
/// through the ownership transition.
pub(crate) struct PlanLease {
    plan: Arc<ExecutionPlan>,
    token: u64,
}

/// Crate-internal state transaction that owns a plan's lifecycle lock.
///
/// Runners keep this guard alive while awaiting sinks and checkpoint stores so
/// another owner of the same [`ExecutionPlan`] cannot observe or mutate an
/// intermediate state.
pub(crate) struct PlanTransaction<'a> {
    plan: &'a ExecutionPlan,
    lease_token: Option<u64>,
    _guard: tokio::sync::MutexGuard<'a, ()>,
}

impl ExecutionPlan {
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
            Err(original) => Err(transaction
                .rollback_error(operation, original, None)
                .await
                .error),
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
            Err(original) => Err(transaction
                .rollback_error(operation, original, None)
                .await
                .error),
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
        let transaction = self.raw_public_transaction().await?;
        if let Err(recovery) = transaction.recover_in_flight(None).await {
            if transaction.has_runner_owned_operation() {
                return Err(recovery);
            }
            if !transaction.has_direct_operation() {
                return Err(recovery);
            }
            let before = transaction.snapshot().await?;
            let operation = transaction.replace_for_direct_forced_reset(before)?;
            return match transaction.reset().await {
                Ok(()) => transaction.commit_operation(operation),
                Err(original) => Err(transaction
                    .rollback_error(operation, original, None)
                    .await
                    .error),
            };
        }
        let before = transaction.snapshot().await?;
        let operation = transaction.begin_rollback(before)?;
        match transaction.reset().await {
            Ok(()) => transaction.commit_operation(operation),
            Err(original) => Err(transaction
                .rollback_error(operation, original, None)
                .await
                .error),
        }
    }

    pub(crate) fn acquire_lease(self: &Arc<Self>) -> Result<PlanLease> {
        let mut state = self
            .lease_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.owner.is_some() {
            return Err(self.leased_error());
        }
        let token = state
            .generation
            .checked_add(1)
            .ok_or_else(|| CalcFlowError::Internal {
                message: format!(
                    "execution plan {:?} exhausted runner lease generations",
                    self.name
                ),
            })?;
        state.generation = token;
        state.owner = Some(token);
        drop(state);
        Ok(PlanLease {
            plan: Arc::clone(self),
            token,
        })
    }

    async fn public_transaction(&self) -> Result<PlanTransaction<'_>> {
        let transaction = self.raw_public_transaction().await?;
        transaction.recover_in_flight(None).await?;
        Ok(transaction)
    }

    async fn raw_public_transaction(&self) -> Result<PlanTransaction<'_>> {
        self.ensure_unleased()?;
        let guard = self.run_lock.lock().await;
        self.ensure_unleased()?;
        Ok(PlanTransaction {
            plan: self,
            lease_token: None,
            _guard: guard,
        })
    }

    pub(crate) async fn leased_transaction<'plan>(
        &'plan self,
        lease: &PlanLease,
    ) -> Result<PlanTransaction<'plan>> {
        self.ensure_lease_owner(lease)?;
        let guard = self.run_lock.lock().await;
        self.ensure_lease_owner(lease)?;
        Ok(PlanTransaction {
            plan: self,
            lease_token: Some(lease.token),
            _guard: guard,
        })
    }

    pub(crate) fn handoff_runner_drop(
        &self,
        lease: &PlanLease,
        durable_state: BTreeMap<String, Value>,
        durable_checkpoint: Option<Checkpoint>,
    ) {
        let lease_state = self
            .lease_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !std::ptr::eq(self, Arc::as_ptr(&lease.plan)) || lease_state.owner != Some(lease.token) {
            return;
        }
        let mut state = self
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match state.in_flight.as_mut() {
            Some(operation) if operation.runner_owner == Some(lease.token) => {
                if let RecoveryAction::Rollback {
                    state,
                    checkpoint: CheckpointRecovery::Restore(checkpoint),
                    ..
                } = &mut operation.action
                {
                    *state = durable_state;
                    *checkpoint = durable_checkpoint;
                }
            }
            Some(_) => {}
            None => {
                let Some(token) = state.generation.checked_add(1) else {
                    return;
                };
                state.generation = token;
                state.in_flight = Some(InFlightOperation {
                    token,
                    runner_owner: Some(lease.token),
                    action: RecoveryAction::Rollback {
                        state: durable_state,
                        checkpoint: CheckpointRecovery::Restore(durable_checkpoint),
                        store_mutation_started: false,
                        clear_after_recovery: true,
                    },
                });
            }
        }
    }

    fn ensure_unleased(&self) -> Result<()> {
        let state = self
            .lease_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match state.owner {
            Some(_) => Err(self.leased_error()),
            None => Ok(()),
        }
    }

    fn ensure_lease_owner(&self, lease: &PlanLease) -> Result<()> {
        let state = self
            .lease_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if std::ptr::eq(self, Arc::as_ptr(&lease.plan)) && state.owner == Some(lease.token) {
            Ok(())
        } else {
            Err(CalcFlowError::Internal {
                message: format!(
                    "runner lease for execution plan {:?} is no longer active",
                    self.name
                ),
            })
        }
    }

    fn leased_error(&self) -> CalcFlowError {
        CalcFlowError::PlanLeased {
            pipeline_name: self.name.clone(),
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
                Ok(runtime)
            })
            .transpose()?;
        let execution = self
            .execute_nodes(&inputs, &context, runtime.as_ref())
            .await;
        if let Some(runtime) = &mut runtime {
            runtime.close();
        }
        let datafusion_metrics = runtime
            .as_ref()
            .map_or_else(Vec::new, DataFusionRuntime::metrics);
        let (outputs, node_timings) = execution?;
        Ok(RunResult {
            outputs,
            node_timings,
            datafusion_metrics,
            metadata: RunMetadata {
                run_id: context.run_id().into(),
                pipeline_name: self.name.clone(),
                pipeline_fingerprint: self.fingerprint.clone(),
            },
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

    fn node(&self, node_id: &str) -> Result<&CompiledNode> {
        self.nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .ok_or_else(|| CalcFlowError::Internal {
                message: format!("compiled plan has no node {node_id}"),
            })
    }
}

impl PlanTransaction<'_> {
    pub(crate) fn validate_inputs(&self, inputs: &BTreeMap<String, Batch>) -> Result<()> {
        self.plan.validate_external_inputs(inputs)
    }

    pub(crate) fn validate_state(&self, state: &BTreeMap<String, Value>) -> Result<()> {
        self.plan.validate_state_map(state)
    }

    pub(crate) async fn execute_validated(
        &self,
        inputs: BTreeMap<String, Batch>,
        options: ExecutionOptions,
    ) -> Result<RunResult> {
        self.plan.execute_unlocked(inputs, options).await
    }

    pub(crate) async fn snapshot(&self) -> Result<BTreeMap<String, Value>> {
        self.plan.snapshot_unlocked().await
    }

    pub(crate) async fn restore(&self, state: &BTreeMap<String, Value>) -> Result<()> {
        self.plan.restore_unlocked(state).await
    }

    pub(crate) async fn reset(&self) -> Result<()> {
        self.plan.reset_unlocked().await
    }

    /// Records direct-plan rollback data synchronously before the first
    /// cancellable state mutation.
    fn begin_rollback(&self, state: BTreeMap<String, Value>) -> Result<OperationToken> {
        self.begin_operation(
            RecoveryAction::Rollback {
                state,
                checkpoint: CheckpointRecovery::NotRequired,
                store_mutation_started: false,
                clear_after_recovery: true,
            },
            None,
        )
    }

    /// Records runner rollback data, including an absent durable checkpoint.
    pub(crate) fn begin_runner_rollback(
        &self,
        state: BTreeMap<String, Value>,
        checkpoint_before: Option<Checkpoint>,
    ) -> Result<OperationToken> {
        self.begin_operation(
            RecoveryAction::Rollback {
                state,
                checkpoint: CheckpointRecovery::Restore(checkpoint_before),
                store_mutation_started: false,
                clear_after_recovery: true,
            },
            Some(self.runner_token()?),
        )
    }

    /// Replaces an unrecoverable marker before a forced reset. Recovery of
    /// this marker repairs partial reset work but deliberately keeps the marker
    /// until reset and durable deletion both commit.
    pub(crate) fn replace_for_forced_reset(
        &self,
        state: BTreeMap<String, Value>,
        checkpoint_before: Option<Checkpoint>,
    ) -> Result<OperationToken> {
        let action = RecoveryAction::Rollback {
            state,
            checkpoint: CheckpointRecovery::Restore(checkpoint_before),
            store_mutation_started: false,
            clear_after_recovery: false,
        };
        let mut state = self
            .plan
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
            runner_owner: Some(self.runner_token()?),
            action,
        });
        Ok(OperationToken(token))
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
        match state.in_flight.as_ref() {
            Some(InFlightOperation {
                runner_owner: None,
                action:
                    RecoveryAction::Rollback {
                        checkpoint: CheckpointRecovery::NotRequired,
                        ..
                    },
                ..
            }) => {}
            _ => {
                return Err(self.recovery_error(
                    "only a direct rollback marker can be replaced by public reset",
                ));
            }
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
            runner_owner: None,
            action: RecoveryAction::Rollback {
                state: state_before_reset,
                checkpoint: CheckpointRecovery::NotRequired,
                store_mutation_started: false,
                clear_after_recovery: false,
            },
        });
        Ok(OperationToken(token))
    }

    /// Records an idempotent durable commit, used by the micro-batch EOF flush.
    pub(crate) fn begin_checkpoint_commit(&self, checkpoint: Checkpoint) -> Result<OperationToken> {
        self.begin_operation(
            RecoveryAction::CommitCheckpoint { checkpoint },
            Some(self.runner_token()?),
        )
    }

    fn begin_operation(
        &self,
        action: RecoveryAction,
        runner_owner: Option<u64>,
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
            runner_owner,
            action,
        });
        Ok(OperationToken(token))
    }

    /// Marks the exact point immediately before a checkpoint store mutation.
    pub(crate) fn mark_store_mutation(&self, token: OperationToken) -> Result<()> {
        let mut state = self
            .plan
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let operation = state
            .in_flight
            .as_mut()
            .filter(|operation| operation.token == token.0)
            .ok_or_else(|| self.stale_operation_error(token))?;
        match &mut operation.action {
            RecoveryAction::Rollback {
                checkpoint,
                store_mutation_started,
                ..
            } => {
                if matches!(checkpoint, CheckpointRecovery::NotRequired) {
                    return Err(CalcFlowError::Internal {
                        message: "direct plan operation cannot mutate a checkpoint store".into(),
                    });
                }
                *store_mutation_started = true;
                Ok(())
            }
            RecoveryAction::CommitCheckpoint { .. } => Ok(()),
        }
    }

    /// Clears a fully committed operation, authenticating its generation.
    pub(crate) fn commit_operation(&self, token: OperationToken) -> Result<()> {
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

    /// Repairs an operation abandoned by cancellation. The marker remains
    /// present until every required plan and durable-store action succeeds.
    pub(crate) async fn recover_in_flight(
        &self,
        checkpoints: Option<&dyn CheckpointStore>,
    ) -> Result<RecoveryOutcome> {
        let operation = self
            .plan
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .in_flight
            .clone();
        let Some(operation) = operation else {
            return Ok(RecoveryOutcome::None);
        };

        let outcome = match &operation.action {
            RecoveryAction::Rollback {
                state,
                checkpoint,
                store_mutation_started,
                clear_after_recovery,
            } => {
                if matches!(checkpoint, CheckpointRecovery::Restore(_)) && checkpoints.is_none() {
                    return Err(
                        self.recovery_error("runner-owned recovery requires its checkpoint store")
                    );
                }
                self.restore(state).await.map_err(|error| {
                    self.recovery_error(&format!("plan restoration failed: {error}"))
                })?;
                if *store_mutation_started {
                    let store = checkpoints.ok_or_else(|| {
                        self.recovery_error("checkpoint compensation requires its store")
                    })?;
                    match checkpoint {
                        CheckpointRecovery::Restore(Some(checkpoint)) => {
                            store.save(checkpoint).await
                        }
                        CheckpointRecovery::Restore(None) => store.delete(self.plan.name()).await,
                        CheckpointRecovery::NotRequired => {
                            unreachable!("store phase requires runner checkpoint metadata")
                        }
                    }
                    .map_err(|error| {
                        self.recovery_error(&format!(
                            "checkpoint compensation also failed: {error}"
                        ))
                    })?;
                }
                if !clear_after_recovery {
                    return Err(self.recovery_error(
                        "an interrupted forced reset must be explicitly reset again",
                    ));
                }
                RecoveryOutcome::RolledBack
            }
            RecoveryAction::CommitCheckpoint { checkpoint } => {
                let store = checkpoints.ok_or_else(|| {
                    self.recovery_error("pending checkpoint commit requires its store")
                })?;
                store.save(checkpoint).await.map_err(|error| {
                    self.recovery_error(&format!("pending checkpoint commit failed: {error}"))
                })?;
                RecoveryOutcome::CommittedCheckpoint(checkpoint.clone())
            }
        };
        self.commit_operation(OperationToken(operation.token))?;
        Ok(outcome)
    }

    pub(crate) async fn rollback_error(
        &self,
        token: OperationToken,
        original: CalcFlowError,
        checkpoints: Option<&dyn CheckpointStore>,
    ) -> RollbackOutcome {
        if let Err(ownership) = self.ensure_operation_owner(token) {
            return RollbackOutcome {
                error: CalcFlowError::Internal {
                    message: format!(
                        "operation failed with {original}; rollback ownership also failed with {ownership}"
                    ),
                },
                recovery_failed: true,
            };
        }
        match self.recover_in_flight(checkpoints).await {
            Ok(_) => RollbackOutcome {
                error: original,
                recovery_failed: false,
            },
            Err(rollback) => RollbackOutcome {
                error: CalcFlowError::Internal {
                    message: format!(
                        "operation failed with {original}; rollback also failed with {rollback}"
                    ),
                },
                recovery_failed: true,
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

    fn runner_token(&self) -> Result<u64> {
        self.lease_token.ok_or_else(|| CalcFlowError::Internal {
            message: "runner-owned operation requires a leased transaction".into(),
        })
    }

    fn has_runner_owned_operation(&self) -> bool {
        self.plan
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .in_flight
            .as_ref()
            .is_some_and(|operation| operation.runner_owner.is_some())
    }

    fn has_direct_operation(&self) -> bool {
        self.plan
            .operation_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .in_flight
            .as_ref()
            .is_some_and(|operation| operation.runner_owner.is_none())
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

impl Drop for PlanLease {
    fn drop(&mut self) {
        let mut state = self
            .plan
            .lease_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.owner == Some(self.token) {
            state.owner = None;
        }
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
            datafusion_config: DataFusionConfig::default(),
            nodes: BTreeMap::new(),
            edges: Vec::new(),
        })
    }

    /// Returns this builder with the run-scoped `DataFusion` configuration.
    #[must_use]
    pub const fn with_datafusion_config(mut self, config: DataFusionConfig) -> Self {
        self.datafusion_config = config;
        self
    }

    /// Returns a new builder that owns the added operator.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Compile`] when `node_id` is empty or already
    /// exists.
    pub fn add_node<O>(mut self, node_id: &str, operator: O) -> Result<Self>
    where
        O: Into<OperatorDefinition>,
    {
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
                operator: operator.into(),
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
        let requires_datafusion = self
            .nodes
            .values()
            .any(|node| node.operator.requires_datafusion());
        if requires_datafusion {
            self.datafusion_config.validate()?;
        }
        validate_nodes(&self.nodes)?;
        validate_edges(&self.nodes, &self.edges)?;
        let order = topological_order(&self.nodes, &self.edges)?;
        let selected_catalog = selected_udf_catalog(&self.nodes, udfs)?;
        let selected_udfs = selected_catalog
            .iter()
            .map(|(reference, _)| reference.clone())
            .collect();
        let (external_inputs, external_outputs) = external_ports(&self.nodes, &self.edges)?;
        let fingerprint = graph_fingerprint(
            &self.name,
            requires_datafusion.then_some(self.datafusion_config),
            &self.nodes,
            &self.edges,
            &selected_catalog,
        )?;
        let table = requires_datafusion.then(|| TablePlanResources {
            config: self.datafusion_config,
            udfs: udfs.clone(),
            selected_udfs,
        });
        Ok(build_plan(
            self,
            order,
            external_inputs,
            external_outputs,
            fingerprint,
            table,
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

fn build_plan(
    mut builder: PipelineBuilder,
    order: Vec<String>,
    external_inputs: BTreeMap<String, PortEndpoint>,
    external_outputs: BTreeMap<String, PortEndpoint>,
    fingerprint: String,
    table: Option<TablePlanResources>,
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
        table,
        nodes,
        external_inputs,
        external_outputs,
        fingerprint,
        run_lock: tokio::sync::Mutex::new(()),
        lease_state: StdMutex::new(LeaseState::default()),
        operation_state: StdMutex::new(OperationState::default()),
    }
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
    use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};

    use super::*;
    use crate::{BatchKind, BatchMetadata, Operator, OperatorContext, UdfRegistry};

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

    #[async_trait]
    impl Operator for LifecycleGate {
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

        async fn process(
            &mut self,
            inputs: &BTreeMap<String, Batch>,
            _context: &OperatorContext<'_>,
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
    ) -> Arc<ExecutionPlan> {
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
                    }),
                )
                .unwrap()
                .compile(&UdfRegistry::new().snapshot())
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

    fn marker(plan: &ExecutionPlan) -> Option<u64> {
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
        tokio::time::sleep(wait + Duration::from_millis(10)).await;

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
