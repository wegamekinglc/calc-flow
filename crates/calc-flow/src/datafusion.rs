use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use datafusion::{
    arrow::record_batch::RecordBatch,
    datasource::MemTable,
    execution::{
        context::{SessionConfig, SessionContext},
        session_state::SessionStateBuilder,
    },
    logical_expr::ScalarUDF,
    physical_plan::{ExecutionPlanProperties, displayable, execute_stream},
};
use futures::{StreamExt, TryStreamExt};
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;

use crate::{
    Batch, BatchMetadata, CalcFlowError, Result, UdfKind, UdfReference, UdfRegistrySnapshot,
    datafusion_rolling::{CalcFlowQueryPlanner, RollingRewriteAudit},
    expression::{sql_projection, validate_select_query},
    validate_selected_udfs,
};

/// Selects the requested `DataFusion` partition policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DataFusionParallelismMode {
    /// Preserve the explicit `target_partitions` request, subject only to the
    /// configured minimum rows per partition.
    #[default]
    Fixed,
    /// Choose a conservative partition count from host capacity, row count,
    /// and trusted input entity statistics.
    Auto,
}

/// Batch metadata key carrying a trusted positive active-entity count.
///
/// Auto parallelism never scans the input to derive this value. A missing,
/// malformed, zero, or row-count-exceeding value safely falls back to one
/// partition and records the reason in [`DataFusionQueryMetric`].
pub const DATAFUSION_ACTIVE_ENTITIES_METADATA_KEY: &str = "calc_flow.datafusion.active_entities";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct DataFusionConfig {
    /// Target Arrow batch size used by the `DataFusion` session.
    pub batch_size: usize,
    /// Requested partitions in [`DataFusionParallelismMode::Fixed`] mode.
    pub target_partitions: usize,
    /// Fixed or conservative automatic partition selection.
    pub parallelism_mode: DataFusionParallelismMode,
    /// Maximum requested partitions in automatic mode.
    pub max_partitions: usize,
    /// Minimum input rows assigned to each useful partition.
    pub min_rows_per_partition: usize,
    /// Inputs below this row count remain single-partition in automatic mode.
    pub small_rows_threshold: usize,
    /// Enables the fail-closed bounded SQL `AVG` rolling rewrite.
    pub enable_rolling_rewrite: bool,
    /// Collects plan strings and recursively traversed physical metrics.
    pub collect_diagnostics: bool,
}

impl Default for DataFusionConfig {
    fn default() -> Self {
        Self {
            batch_size: 8_192,
            target_partitions: 1,
            parallelism_mode: DataFusionParallelismMode::Fixed,
            max_partitions: 32,
            min_rows_per_partition: 65_536,
            small_rows_threshold: 10_001,
            enable_rolling_rewrite: true,
            collect_diagnostics: true,
        }
    }
}

impl DataFusionConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "datafusion.batch_size".into(),
                message: "must be positive".into(),
            });
        }
        if self.target_partitions == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "datafusion.target_partitions".into(),
                message: "must be positive".into(),
            });
        }
        for (field, value) in [
            ("datafusion.max_partitions", self.max_partitions),
            (
                "datafusion.min_rows_per_partition",
                self.min_rows_per_partition,
            ),
            ("datafusion.small_rows_threshold", self.small_rows_threshold),
        ] {
            if value == 0 {
                return Err(CalcFlowError::InvalidArgument {
                    field: field.into(),
                    message: "must be positive".into(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataFusionQueryMetric {
    pub query_id: u64,
    pub node_id: Option<String>,
    pub runtime_acquire_ns: u64,
    pub session_state_create_ns: u64,
    pub input_adapter_ns: u64,
    pub table_register_ns: u64,
    pub sql_parse_ns: u64,
    pub logical_planning_ns: u64,
    pub physical_planning_ns: u64,
    pub physical_planning_count: u32,
    pub planning_ns: u64,
    pub stream_open_ns: u64,
    pub execution_to_first_batch_ns: u64,
    pub execution_remaining_ns: u64,
    pub execution_ns: u64,
    pub collect_ns: u64,
    pub output_arrow_wrap_ns: u64,
    pub audit_ns: u64,
    pub metrics_traversal_ns: u64,
    pub logical_plan_string_ns: u64,
    pub physical_plan_string_ns: u64,
    pub batch_envelope_ns: u64,
    pub run_result_ns: u64,
    pub physical_metric_count: usize,
    pub output_partition_count: usize,
    pub output_partition_rows: Vec<usize>,
    pub window_partition_count: usize,
    pub window_partition_rows: Vec<usize>,
    pub spill_bytes: usize,
    pub elapsed_compute_ns: usize,
    pub window_compute_ns: usize,
    pub repartition_sort_compute_ns: usize,
    pub window_operator_count: usize,
    pub repartition_operator_count: usize,
    pub sort_operator_count: usize,
    pub coalesce_operator_count: usize,
    pub output_rows: usize,
    pub configured_batch_size: usize,
    pub parallelism_mode: DataFusionParallelismMode,
    pub configured_target_partitions: usize,
    pub requested_target_partitions: usize,
    pub effective_target_partitions: usize,
    pub available_parallelism: usize,
    pub max_partitions: usize,
    pub min_rows_per_partition: usize,
    pub small_rows_threshold: usize,
    pub parallelism_decision_reused: bool,
    pub decision_input_rows: usize,
    pub decision_active_entities: Option<usize>,
    pub decision_active_entities_source: String,
    pub input_rows: usize,
    pub active_entities: Option<usize>,
    pub active_entities_source: String,
    pub partition_limit_reason: String,
    pub rolling_rewrite_enabled: bool,
    pub diagnostics_collected: bool,
    pub rolling_candidate_windows: usize,
    pub rolling_rewritten_windows: usize,
    pub rolling_fallback_reasons: Vec<String>,
    pub logical_plan: String,
    pub physical_plan: String,
}

pub struct DataFusionRuntime {
    config: DataFusionConfig,
    context: OnceLock<SessionContext>,
    selected_udfs: Vec<(UdfReference, Arc<ScalarUDF>)>,
    query_lock: AsyncMutex<()>,
    metrics: Mutex<Vec<DataFusionQueryMetric>>,
    next_query: AtomicU64,
    effective_target_partitions: AtomicUsize,
    parallelism_decision: OnceLock<DataFusionParallelismDecision>,
    rolling_rewrite_audit: Arc<RollingRewriteAudit>,
    runtime_acquire_ns: u64,
    closed: AtomicBool,
}

impl DataFusionRuntime {
    /// Creates a run-scoped runtime that owns a lazily initialized `DataFusion` session.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when either configuration
    /// value is zero.
    pub fn new(config: DataFusionConfig) -> Result<Self> {
        let started = Instant::now();
        config.validate()?;
        let runtime_acquire_ns = nanos(started.elapsed());
        Ok(Self {
            config,
            context: OnceLock::new(),
            selected_udfs: Vec::new(),
            query_lock: AsyncMutex::new(()),
            metrics: Mutex::new(Vec::new()),
            next_query: AtomicU64::new(1),
            effective_target_partitions: AtomicUsize::new(0),
            parallelism_decision: OnceLock::new(),
            rolling_rewrite_audit: Arc::new(RollingRewriteAudit::default()),
            runtime_acquire_ns,
            closed: AtomicBool::new(false),
        })
    }

    /// Registers the selected native UDFs during runtime setup.
    ///
    /// This method intentionally requires exclusive runtime access. Call it
    /// before the run-scoped runtime is shared or any queries are started, so
    /// the session's function catalog remains stable throughout query awaits.
    /// External UDF references remain catalog metadata and are not registered
    /// with `DataFusion`.
    ///
    /// # Errors
    ///
    /// Returns an error when the runtime is closed, selected SQL names collide,
    /// or a selected native reference is absent from the snapshot. Resolution
    /// and namespace validation complete before the shared session is mutated.
    pub fn register_udfs(
        &mut self,
        snapshot: &UdfRegistrySnapshot,
        references: &[UdfReference],
    ) -> Result<()> {
        self.ensure_open()?;
        validate_selected_udfs(references)?;
        let selected = references
            .iter()
            .filter(|reference| reference.kind() == UdfKind::DataFusionScalar)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|reference| {
                snapshot
                    .resolve_native(reference)
                    .map(|udf| (reference.clone(), udf))
            })
            .collect::<Result<Vec<_>>>()?;
        validate_udf_sql_namespace(
            self.selected_udfs
                .iter()
                .chain(&selected)
                .map(|(reference, udf)| (reference, udf.as_ref())),
        )?;
        let selected = selected
            .into_iter()
            .filter(|(reference, _)| {
                !self
                    .selected_udfs
                    .iter()
                    .any(|(registered, _)| registered == reference)
            })
            .collect::<Vec<_>>();
        if let Some(context) = self.context.get() {
            for (_, udf) in &selected {
                context.register_udf(udf.as_ref().clone());
            }
        }
        self.selected_udfs.extend(selected);
        Ok(())
    }

    /// Evaluates an expression or assignment over one table batch.
    ///
    /// # Errors
    ///
    /// Returns an error when the expression is invalid, the input is not a
    /// table batch, the runtime is closed, or `DataFusion` cannot execute it.
    pub async fn evaluate(
        &self,
        expression: &str,
        input: &Batch,
        node_id: Option<&str>,
    ) -> Result<Batch> {
        let query = sql_projection(expression, "input")?;
        let tables = BTreeMap::from([("input".to_owned(), input.clone())]);
        self.sql(&query, &tables, node_id).await
    }

    /// Executes one read-only SQL query over run-scoped table aliases.
    ///
    /// # Errors
    ///
    /// Returns an error when the runtime is closed, the input map is empty, an
    /// alias or query is invalid, an input is not a table batch, or `DataFusion`
    /// cannot plan or execute the query.
    #[allow(
        clippy::too_many_lines,
        reason = "the ordered phase boundaries are kept together so benchmark attribution cannot drift"
    )]
    pub async fn sql(
        &self,
        query: &str,
        tables: &BTreeMap<String, Batch>,
        node_id: Option<&str>,
    ) -> Result<Batch> {
        self.ensure_open()?;
        if tables.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "tables".into(),
                message: "must not be empty".into(),
            });
        }
        let parse_start = Instant::now();
        let query = validate_select_query(query)?;
        let sql_parse_ns = nanos(parse_start.elapsed());
        // Declared before registrations so alias cleanup runs before unlock.
        let _query_guard = self.query_lock.lock().await;
        self.ensure_open()?;
        let input_rows = tables.values().fold(0_usize, |total, batch| {
            total.saturating_add(batch.num_rows())
        });
        let (active_entities, active_entities_source) = active_entities(tables, input_rows);
        let context_preexisting = self.context.get().is_some();
        let session_state_create_start = Instant::now();
        let context = self.context_for_rows(input_rows, active_entities, active_entities_source);
        let session_state_create_ns = if context_preexisting {
            0
        } else {
            nanos(session_state_create_start.elapsed())
        };
        let mut registrations = TableRegistrations::new(context);
        let mut input_adapter_ns = 0_u64;
        let mut table_register_ns = 0_u64;
        for (alias, batch) in tables {
            let timing = registrations.register(alias, batch, node_id)?;
            input_adapter_ns = input_adapter_ns.saturating_add(timing.input_adapter_ns);
            table_register_ns = table_register_ns.saturating_add(timing.table_register_ns);
        }

        let logical_planning_start = Instant::now();
        let dataframe = context
            .sql(&query)
            .await
            .map_err(|error| datafusion_error(node_id, error))?;
        let logical_plan_string_start = Instant::now();
        let logical_plan = if self.config.collect_diagnostics {
            dataframe.logical_plan().display_indent_schema().to_string()
        } else {
            String::new()
        };
        let logical_plan_string_ns = if self.config.collect_diagnostics {
            nanos(logical_plan_string_start.elapsed())
        } else {
            0
        };
        let logical_planning_ns = nanos(logical_planning_start.elapsed());

        let physical_planning_start = Instant::now();
        let physical_plan = dataframe
            .create_physical_plan()
            .await
            .map_err(|error| datafusion_error(node_id, error))?;
        let physical_plan_string_start = Instant::now();
        let physical_plan_text = if self.config.collect_diagnostics {
            displayable(physical_plan.as_ref()).indent(true).to_string()
        } else {
            String::new()
        };
        let physical_plan_string_ns = if self.config.collect_diagnostics {
            nanos(physical_plan_string_start.elapsed())
        } else {
            0
        };
        let audit_start = Instant::now();
        let rolling_audit = self.rolling_rewrite_audit.snapshot();
        let audit_ns = nanos(audit_start.elapsed());
        let physical_planning_ns = nanos(physical_planning_start.elapsed());
        let planning_ns = sql_parse_ns
            .saturating_add(logical_planning_ns)
            .saturating_add(physical_planning_ns);

        let execution_start = Instant::now();
        let result_schema = physical_plan.schema();
        let metrics_plan = Arc::clone(&physical_plan);
        let mut stream = execute_stream(physical_plan, Arc::new(dataframe.task_ctx()))
            .map_err(|error| datafusion_error(node_id, error))?;
        let stream_open_ns = nanos(execution_start.elapsed());

        let collect_start = Instant::now();
        let first_batch = stream
            .next()
            .await
            .transpose()
            .map_err(|error| datafusion_error(node_id, error))?;
        let execution_to_first_batch_ns = nanos(execution_start.elapsed());
        let remaining_start = Instant::now();
        let remaining = stream
            .try_collect::<Vec<_>>()
            .await
            .map_err(|error| datafusion_error(node_id, error))?;
        let execution_remaining_ns = nanos(remaining_start.elapsed());
        let mut batches = Vec::with_capacity(remaining.len() + usize::from(first_batch.is_some()));
        batches.extend(first_batch);
        batches.extend(remaining);
        let collect_ns = nanos(collect_start.elapsed());
        let execution_ns = stream_open_ns.saturating_add(collect_ns);
        // A zero-row result (for example an INNER JOIN with no key-equal
        // pairs) collects to zero RecordBatches; represent it as one
        // zero-row batch, exactly as the Batch::table contract prescribes.
        let output_arrow_wrap_start = Instant::now();
        let batches = if batches.is_empty() {
            vec![RecordBatch::new_empty(result_schema)]
        } else {
            batches
        };
        let output_arrow_wrap_ns = nanos(output_arrow_wrap_start.elapsed());
        let batch_envelope_start = Instant::now();
        let output = Batch::table(batches, merged_metadata(tables))?;
        let batch_envelope_ns = nanos(batch_envelope_start.elapsed());
        let output_rows = output.num_rows();
        let metrics_traversal_start = Instant::now();
        let plan_statistics = if self.config.collect_diagnostics {
            physical_plan_statistics(metrics_plan.as_ref(), output_rows)
        } else {
            PhysicalPlanStatistics::default()
        };
        let metrics_traversal_ns = if self.config.collect_diagnostics {
            nanos(metrics_traversal_start.elapsed())
        } else {
            0
        };
        let Some(decision) = self.parallelism_decision.get() else {
            return Err(CalcFlowError::Internal {
                message: "DataFusion context initialized without a parallelism decision".into(),
            });
        };
        self.metrics.lock().push(DataFusionQueryMetric {
            query_id: self.next_query.fetch_add(1, Ordering::Relaxed),
            node_id: node_id.map(str::to_owned),
            runtime_acquire_ns: self.runtime_acquire_ns,
            session_state_create_ns,
            input_adapter_ns,
            table_register_ns,
            sql_parse_ns,
            logical_planning_ns,
            physical_planning_ns,
            physical_planning_count: 1,
            planning_ns,
            stream_open_ns,
            execution_to_first_batch_ns,
            execution_remaining_ns,
            execution_ns,
            collect_ns,
            output_arrow_wrap_ns,
            audit_ns,
            metrics_traversal_ns,
            logical_plan_string_ns,
            physical_plan_string_ns,
            batch_envelope_ns,
            run_result_ns: 0,
            physical_metric_count: plan_statistics.metric_count,
            output_partition_count: plan_statistics.partition_rows.len(),
            output_partition_rows: plan_statistics.partition_rows,
            window_partition_count: plan_statistics.window_partition_rows.len(),
            window_partition_rows: plan_statistics.window_partition_rows,
            spill_bytes: plan_statistics.spill_bytes,
            elapsed_compute_ns: plan_statistics.elapsed_compute_ns,
            window_compute_ns: plan_statistics.window_compute_ns,
            repartition_sort_compute_ns: plan_statistics.repartition_sort_compute_ns,
            window_operator_count: plan_statistics.window_operator_count,
            repartition_operator_count: plan_statistics.repartition_operator_count,
            sort_operator_count: plan_statistics.sort_operator_count,
            coalesce_operator_count: plan_statistics.coalesce_operator_count,
            output_rows,
            configured_batch_size: self.config.batch_size,
            parallelism_mode: self.config.parallelism_mode,
            configured_target_partitions: self.config.target_partitions,
            requested_target_partitions: decision.requested_partitions,
            effective_target_partitions: self.effective_target_partitions.load(Ordering::Acquire),
            available_parallelism: decision.available_parallelism,
            max_partitions: self.config.max_partitions,
            min_rows_per_partition: self.config.min_rows_per_partition,
            small_rows_threshold: self.config.small_rows_threshold,
            parallelism_decision_reused: context_preexisting,
            decision_input_rows: decision.input_rows,
            decision_active_entities: decision.active_entities,
            decision_active_entities_source: decision.active_entities_source.into(),
            input_rows,
            active_entities,
            active_entities_source: active_entities_source.into(),
            partition_limit_reason: decision.limit_reason.into(),
            rolling_rewrite_enabled: self.config.enable_rolling_rewrite,
            diagnostics_collected: self.config.collect_diagnostics,
            rolling_candidate_windows: rolling_audit.candidate_windows,
            rolling_rewritten_windows: rolling_audit.rewritten_windows,
            rolling_fallback_reasons: rolling_audit.fallback_reasons,
            logical_plan,
            physical_plan: physical_plan_text,
        });
        Ok(output)
    }

    pub fn metrics(&self) -> Vec<DataFusionQueryMetric> {
        self.metrics.lock().clone()
    }

    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }

    #[cfg(test)]
    fn context(&self) -> &SessionContext {
        self.context_for_rows(0, None, "not_evaluated")
    }

    fn context_for_rows(
        &self,
        input_rows: usize,
        active_entities: Option<usize>,
        active_entities_source: &'static str,
    ) -> &SessionContext {
        self.context.get_or_init(|| {
            let available_parallelism = std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1);
            let mut decision = parallelism_decision(
                &self.config,
                input_rows,
                active_entities,
                available_parallelism,
            );
            decision.active_entities_source = active_entities_source;
            let target_partitions = decision.effective_partitions;
            self.parallelism_decision
                .set(decision)
                .expect("parallelism decision initializes with the session");
            self.effective_target_partitions
                .store(target_partitions, Ordering::Release);
            let session = SessionConfig::new()
                .with_batch_size(self.config.batch_size)
                .with_target_partitions(target_partitions);
            let state = SessionStateBuilder::new()
                .with_config(session)
                .with_default_features();
            let state = if self.config.enable_rolling_rewrite {
                state
                    .with_query_planner(Arc::new(CalcFlowQueryPlanner::new(Arc::clone(
                        &self.rolling_rewrite_audit,
                    ))))
                    .build()
            } else {
                state.build()
            };
            let context = SessionContext::new_with_state(state);
            for (_, udf) in &self.selected_udfs {
                context.register_udf(udf.as_ref().clone());
            }
            context
        })
    }

    fn ensure_open(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            Err(CalcFlowError::InvalidArgument {
                field: "runtime".into(),
                message: "is closed".into(),
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Default)]
struct PhysicalPlanStatistics {
    metric_count: usize,
    partition_rows: Vec<usize>,
    window_partition_rows: Vec<usize>,
    spill_bytes: usize,
    elapsed_compute_ns: usize,
    window_compute_ns: usize,
    repartition_sort_compute_ns: usize,
    window_operator_count: usize,
    repartition_operator_count: usize,
    sort_operator_count: usize,
    coalesce_operator_count: usize,
}

fn physical_plan_statistics(
    plan: &dyn datafusion::physical_plan::ExecutionPlan,
    output_rows: usize,
) -> PhysicalPlanStatistics {
    // The recursive accumulator keeps every physical metric on the same plan
    // traversal so operator and timing totals cannot observe different trees.
    // #lizard forgives
    use datafusion::physical_plan::metrics::MetricValue;

    let partition_count = plan.output_partitioning().partition_count().max(1);
    let mut statistics = PhysicalPlanStatistics {
        partition_rows: vec![0; partition_count],
        ..PhysicalPlanStatistics::default()
    };
    let name = plan.name();
    let is_window = name.contains("WindowAggExec") || name.contains("CalcFlowRollingExec");
    let is_repartition = name.contains("RepartitionExec");
    let is_sort = name.contains("SortExec");
    statistics.window_operator_count = usize::from(is_window);
    statistics.repartition_operator_count = usize::from(is_repartition);
    statistics.sort_operator_count = usize::from(is_sort);
    statistics.coalesce_operator_count = usize::from(name.contains("Coalesce"));
    if let Some(metrics) = plan.metrics() {
        statistics.metric_count = metrics.iter().count();
        for metric in metrics.iter() {
            if matches!(metric.value(), MetricValue::OutputRows(_)) {
                if let Some(partition) = metric.partition() {
                    if let Some(rows) = statistics.partition_rows.get_mut(partition) {
                        *rows = rows.saturating_add(metric.value().as_usize());
                    }
                } else if partition_count == 1 {
                    statistics.partition_rows[0] =
                        statistics.partition_rows[0].saturating_add(metric.value().as_usize());
                }
            }
        }
    }
    if partition_count == 1 && statistics.partition_rows[0] == 0 {
        statistics.partition_rows[0] = output_rows;
    }
    if is_window {
        statistics
            .window_partition_rows
            .clone_from(&statistics.partition_rows);
    }
    for child in plan.children() {
        let child = physical_plan_statistics(child.as_ref(), 0);
        statistics.metric_count = statistics.metric_count.saturating_add(child.metric_count);
        statistics.spill_bytes = statistics.spill_bytes.saturating_add(child.spill_bytes);
        statistics.elapsed_compute_ns = statistics
            .elapsed_compute_ns
            .saturating_add(child.elapsed_compute_ns);
        statistics.window_compute_ns = statistics
            .window_compute_ns
            .saturating_add(child.window_compute_ns);
        statistics.repartition_sort_compute_ns = statistics
            .repartition_sort_compute_ns
            .saturating_add(child.repartition_sort_compute_ns);
        statistics.window_operator_count = statistics
            .window_operator_count
            .saturating_add(child.window_operator_count);
        statistics.repartition_operator_count = statistics
            .repartition_operator_count
            .saturating_add(child.repartition_operator_count);
        statistics.sort_operator_count = statistics
            .sort_operator_count
            .saturating_add(child.sort_operator_count);
        statistics.coalesce_operator_count = statistics
            .coalesce_operator_count
            .saturating_add(child.coalesce_operator_count);
        if statistics.window_partition_rows.is_empty() {
            statistics.window_partition_rows = child.window_partition_rows;
        }
    }
    if let Some(metrics) = plan.metrics() {
        let elapsed_compute_ns = metrics.elapsed_compute().unwrap_or(0);
        statistics.spill_bytes = statistics
            .spill_bytes
            .saturating_add(metrics.spilled_bytes().unwrap_or(0));
        statistics.elapsed_compute_ns = statistics
            .elapsed_compute_ns
            .saturating_add(elapsed_compute_ns);
        if is_window {
            statistics.window_compute_ns = statistics
                .window_compute_ns
                .saturating_add(elapsed_compute_ns);
        }
        if is_repartition || is_sort {
            statistics.repartition_sort_compute_ns = statistics
                .repartition_sort_compute_ns
                .saturating_add(elapsed_compute_ns);
        }
    }
    statistics
}

#[derive(Clone, Copy, Debug)]
struct DataFusionParallelismDecision {
    requested_partitions: usize,
    effective_partitions: usize,
    available_parallelism: usize,
    limit_reason: &'static str,
    input_rows: usize,
    active_entities: Option<usize>,
    active_entities_source: &'static str,
}

fn parallelism_decision(
    config: &DataFusionConfig,
    input_rows: usize,
    active_entities: Option<usize>,
    available_parallelism: usize,
) -> DataFusionParallelismDecision {
    // The ordered early returns are the audit trail for every conservative
    // partition cap and intentionally remain in one decision function.
    // #lizard forgives
    let available_parallelism = available_parallelism.max(1);
    let requested_partitions = match config.parallelism_mode {
        DataFusionParallelismMode::Fixed => config.target_partitions,
        DataFusionParallelismMode::Auto => available_parallelism.min(config.max_partitions),
    };
    let decision = |effective_partitions, limit_reason| DataFusionParallelismDecision {
        requested_partitions,
        effective_partitions,
        available_parallelism,
        limit_reason,
        input_rows,
        active_entities,
        active_entities_source: if active_entities.is_some() {
            "provided"
        } else {
            "missing"
        },
    };

    if input_rows == 0 {
        return decision(1, "empty_input");
    }
    if config.parallelism_mode == DataFusionParallelismMode::Auto {
        if input_rows < config.small_rows_threshold {
            return decision(1, "small_input");
        }
        let Some(active_entities) = active_entities else {
            return decision(1, "missing_active_entities");
        };
        if active_entities < 2 {
            return decision(1, "single_entity");
        }
        let work_cap = input_rows.div_ceil(config.min_rows_per_partition).max(1);
        let effective_partitions = requested_partitions.min(work_cap).min(active_entities);
        let reason = if effective_partitions == active_entities
            && active_entities < requested_partitions.min(work_cap)
        {
            "active_entities"
        } else if effective_partitions == work_cap && work_cap < requested_partitions {
            "minimum_rows_per_partition"
        } else if config.max_partitions <= available_parallelism {
            "configured_max_partitions"
        } else {
            "available_parallelism"
        };
        return decision(effective_partitions, reason);
    }

    let work_cap = input_rows.div_ceil(config.min_rows_per_partition).max(1);
    let effective_partitions = requested_partitions.min(work_cap);
    decision(
        effective_partitions,
        if effective_partitions < requested_partitions {
            "minimum_rows_per_partition"
        } else {
            "configured_target_partitions"
        },
    )
}

fn active_entities(
    tables: &BTreeMap<String, Batch>,
    input_rows: usize,
) -> (Option<usize>, &'static str) {
    let mut values = tables.values();
    let Some(batch) = values.next() else {
        return (None, "missing");
    };
    if values.next().is_some() {
        return (None, "multiple_inputs");
    }
    let Some(value) = batch
        .metadata()
        .attributes()
        .get(DATAFUSION_ACTIVE_ENTITIES_METADATA_KEY)
    else {
        return (None, "missing");
    };
    let Some(active_entities) = value.as_u64().and_then(|value| usize::try_from(value).ok()) else {
        return (None, "invalid");
    };
    if active_entities == 0 || active_entities > input_rows {
        return (None, "invalid");
    }
    (Some(active_entities), "batch_metadata")
}

fn validate_udf_sql_namespace<'a>(
    selected: impl IntoIterator<Item = (&'a UdfReference, &'a ScalarUDF)>,
) -> Result<()> {
    let mut owners: BTreeMap<&str, &UdfReference> = BTreeMap::new();
    for (reference, udf) in selected {
        for sql_name in std::iter::once(udf.name()).chain(udf.aliases().iter().map(String::as_str))
        {
            if let Some(&owner) = owners.get(sql_name) {
                if owner != reference {
                    return Err(CalcFlowError::Compile {
                        message: format!(
                            "DataFusion SQL name '{sql_name}' collides between {}:{}@{} ({:?}) and {}:{}@{} ({:?})",
                            owner.provider(),
                            owner.name(),
                            owner.version(),
                            owner.kind(),
                            reference.provider(),
                            reference.name(),
                            reference.version(),
                            reference.kind()
                        ),
                    });
                }
            } else {
                owners.insert(sql_name, reference);
            }
        }
    }
    Ok(())
}

struct TableRegistrations<'a> {
    context: &'a SessionContext,
    aliases: Vec<String>,
}

struct TableRegistrationTiming {
    input_adapter_ns: u64,
    table_register_ns: u64,
}

impl<'a> TableRegistrations<'a> {
    fn new(context: &'a SessionContext) -> Self {
        Self {
            context,
            aliases: Vec::new(),
        }
    }

    fn register(
        &mut self,
        alias: &str,
        batch: &Batch,
        node_id: Option<&str>,
    ) -> Result<TableRegistrationTiming> {
        if !is_identifier(alias) {
            return Err(registration_error(
                alias,
                node_id,
                "alias must be a SQL identifier",
            ));
        }
        let input_adapter_start = Instant::now();
        let table = batch
            .table_payload()
            .map_err(|error| registration_error(alias, node_id, error))?;
        let provider =
            MemTable::try_new(Arc::clone(table.schema()), vec![table.batches().to_vec()])
                .map_err(|error| registration_error(alias, node_id, error))?;
        let input_adapter_ns = nanos(input_adapter_start.elapsed());
        let table_register_start = Instant::now();
        self.context
            .register_table(alias, Arc::new(provider))
            .map_err(|error| registration_error(alias, node_id, error))?;
        let table_register_ns = nanos(table_register_start.elapsed());
        self.aliases.push(alias.to_owned());
        Ok(TableRegistrationTiming {
            input_adapter_ns,
            table_register_ns,
        })
    }
}

impl Drop for TableRegistrations<'_> {
    fn drop(&mut self) {
        for alias in self.aliases.iter().rev() {
            let _ = self.context.deregister_table(alias.as_str());
        }
    }
}

fn datafusion_error(node_id: Option<&str>, error: impl std::fmt::Display) -> CalcFlowError {
    CalcFlowError::DataFusion {
        node_id: node_id.map(str::to_owned),
        message: error.to_string(),
    }
}

fn registration_error(
    alias: &str,
    node_id: Option<&str>,
    error: impl std::fmt::Display,
) -> CalcFlowError {
    datafusion_error(
        node_id,
        format!("failed to register table alias {alias:?}: {error}"),
    )
}

fn merged_metadata(tables: &BTreeMap<String, Batch>) -> BatchMetadata {
    if tables.len() == 1 {
        tables
            .values()
            .next()
            .expect("length checked")
            .metadata()
            .clone()
    } else {
        BatchMetadata::default()
    }
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn is_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    chars
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UdfRegistry;
    use datafusion::{
        arrow::datatypes::DataType,
        common::ScalarValue,
        logical_expr::{ColumnarValue, Volatility, create_udf},
    };

    fn constant_udf(name: &str, value: i64) -> Arc<ScalarUDF> {
        Arc::new(create_udf(
            name,
            vec![],
            DataType::Int64,
            Volatility::Immutable,
            Arc::new(move |_| Ok(ColumnarValue::Scalar(ScalarValue::Int64(Some(value))))),
        ))
    }

    #[test]
    fn runtime_preparation_and_close_do_not_initialize_a_session() {
        let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        assert!(runtime.context.get().is_none());

        runtime
            .register_udfs(&UdfRegistrySnapshot::default(), &[])
            .unwrap();
        assert!(runtime.context.get().is_none());

        runtime.close();
        assert!(runtime.context.get().is_none());
    }

    #[test]
    fn context_initializes_once_and_reuses_the_same_session() {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        assert!(runtime.context.get().is_none());

        let first = std::ptr::from_ref(runtime.context());
        assert!(runtime.context.get().is_some());
        let second = std::ptr::from_ref(runtime.context());

        assert_eq!(first, second);
    }

    #[test]
    fn successful_udf_preparation_queues_native_udf_without_initializing_context() {
        let selected =
            UdfReference::new("rust", "prepared_value", "1", UdfKind::DataFusionScalar).unwrap();
        let udf = constant_udf("prepared_value", 11);
        let mut registry = UdfRegistry::new();
        registry
            .register_datafusion(selected.clone(), Arc::clone(&udf), 0)
            .unwrap();
        let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

        runtime
            .register_udfs(&registry.snapshot(), &[selected])
            .unwrap();

        assert!(runtime.context.get().is_none());
        assert_eq!(runtime.selected_udfs.len(), 1);
        assert!(Arc::ptr_eq(&runtime.selected_udfs[0].1, &udf));
    }

    #[test]
    fn failed_udf_preparation_preserves_uninitialized_context_and_queue() {
        let queued =
            UdfReference::new("rust", "queued_value", "1", UdfKind::DataFusionScalar).unwrap();
        let conflicting =
            UdfReference::new("rust", "queued_value", "2", UdfKind::DataFusionScalar).unwrap();
        let missing =
            UdfReference::new("rust", "missing_value", "1", UdfKind::DataFusionScalar).unwrap();
        let queued_udf = constant_udf("queued_value", 11);
        let mut registry = UdfRegistry::new();
        registry
            .register_datafusion(queued.clone(), Arc::clone(&queued_udf), 0)
            .unwrap();
        registry
            .register_datafusion(conflicting.clone(), constant_udf("queued_value", 22), 0)
            .unwrap();
        let snapshot = registry.snapshot();
        let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        runtime
            .register_udfs(&snapshot, std::slice::from_ref(&queued))
            .unwrap();
        assert!(runtime.context.get().is_none());
        assert_eq!(runtime.selected_udfs.len(), 1);
        assert!(Arc::ptr_eq(&runtime.selected_udfs[0].1, &queued_udf));

        assert!(
            runtime
                .register_udfs(&snapshot, &[queued, conflicting])
                .is_err()
        );
        assert!(runtime.context.get().is_none());
        assert_eq!(runtime.selected_udfs.len(), 1);
        assert!(Arc::ptr_eq(&runtime.selected_udfs[0].1, &queued_udf));

        assert!(runtime.register_udfs(&snapshot, &[missing]).is_err());
        assert!(runtime.context.get().is_none());
        assert_eq!(runtime.selected_udfs.len(), 1);
        assert!(Arc::ptr_eq(&runtime.selected_udfs[0].1, &queued_udf));
    }

    #[test]
    fn auto_parallelism_is_conservative_and_explainable() {
        let config = DataFusionConfig {
            parallelism_mode: DataFusionParallelismMode::Auto,
            max_partitions: 32,
            min_rows_per_partition: 65_536,
            small_rows_threshold: 10_001,
            ..DataFusionConfig::default()
        };

        let small = parallelism_decision(&config, 10_000, Some(64), 32);
        assert_eq!(small.effective_partitions, 1);
        assert_eq!(small.limit_reason, "small_input");

        let one_entity = parallelism_decision(&config, 1_000_000, Some(1), 32);
        assert_eq!(one_entity.effective_partitions, 1);
        assert_eq!(one_entity.limit_reason, "single_entity");

        let missing = parallelism_decision(&config, 1_000_000, None, 32);
        assert_eq!(missing.effective_partitions, 1);
        assert_eq!(missing.limit_reason, "missing_active_entities");

        let saturated = parallelism_decision(&config, 1_000_000, Some(64), 32);
        assert_eq!(saturated.requested_partitions, 32);
        assert_eq!(saturated.effective_partitions, 16);
        assert_eq!(saturated.limit_reason, "minimum_rows_per_partition");
    }

    #[test]
    fn fixed_parallelism_preserves_p1_and_reports_its_cap() {
        let config = DataFusionConfig {
            target_partitions: 32,
            ..DataFusionConfig::default()
        };

        let decision = parallelism_decision(&config, 1_000_000, None, 64);

        assert_eq!(decision.requested_partitions, 32);
        assert_eq!(decision.effective_partitions, 16);
        assert_eq!(decision.limit_reason, "minimum_rows_per_partition");
    }
}
