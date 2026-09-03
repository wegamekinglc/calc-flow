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
    physical_plan::{common::collect as collect_stream, displayable, execute_stream},
};
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct DataFusionConfig {
    pub batch_size: usize,
    pub target_partitions: usize,
}

impl Default for DataFusionConfig {
    fn default() -> Self {
        Self {
            batch_size: 8_192,
            target_partitions: 1,
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
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataFusionQueryMetric {
    pub query_id: u64,
    pub node_id: Option<String>,
    pub sql_parse_ns: u64,
    pub logical_planning_ns: u64,
    pub physical_planning_ns: u64,
    pub physical_planning_count: u32,
    pub planning_ns: u64,
    pub stream_open_ns: u64,
    pub execution_ns: u64,
    pub collect_ns: u64,
    pub output_rows: usize,
    pub configured_target_partitions: usize,
    pub effective_target_partitions: usize,
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
    rolling_rewrite_audit: Arc<RollingRewriteAudit>,
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
        config.validate()?;
        Ok(Self {
            config,
            context: OnceLock::new(),
            selected_udfs: Vec::new(),
            query_lock: AsyncMutex::new(()),
            metrics: Mutex::new(Vec::new()),
            next_query: AtomicU64::new(1),
            effective_target_partitions: AtomicUsize::new(0),
            rolling_rewrite_audit: Arc::new(RollingRewriteAudit::default()),
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
        let context = self.context_for_rows(input_rows);
        let mut registrations = TableRegistrations::new(context);
        for (alias, batch) in tables {
            registrations.register(alias, batch, node_id)?;
        }

        let logical_planning_start = Instant::now();
        let dataframe = context
            .sql(&query)
            .await
            .map_err(|error| datafusion_error(node_id, error))?;
        let logical_plan = dataframe.logical_plan().display_indent_schema().to_string();
        let logical_planning_ns = nanos(logical_planning_start.elapsed());

        let physical_planning_start = Instant::now();
        let physical_plan = dataframe
            .create_physical_plan()
            .await
            .map_err(|error| datafusion_error(node_id, error))?;
        let physical_plan_text = displayable(physical_plan.as_ref()).indent(true).to_string();
        let rolling_audit = self.rolling_rewrite_audit.snapshot();
        let physical_planning_ns = nanos(physical_planning_start.elapsed());
        let planning_ns = sql_parse_ns
            .saturating_add(logical_planning_ns)
            .saturating_add(physical_planning_ns);

        let execution_start = Instant::now();
        let result_schema = physical_plan.schema();
        let stream = execute_stream(physical_plan, Arc::new(dataframe.task_ctx()))
            .map_err(|error| datafusion_error(node_id, error))?;
        let stream_open_ns = nanos(execution_start.elapsed());

        let collect_start = Instant::now();
        let batches = collect_stream(stream)
            .await
            .map_err(|error| datafusion_error(node_id, error))?;
        let collect_ns = nanos(collect_start.elapsed());
        let execution_ns = stream_open_ns.saturating_add(collect_ns);
        // A zero-row result (for example an INNER JOIN with no key-equal
        // pairs) collects to zero RecordBatches; represent it as one
        // zero-row batch, exactly as the Batch::table contract prescribes.
        let batches = if batches.is_empty() {
            vec![RecordBatch::new_empty(result_schema)]
        } else {
            batches
        };
        let output = Batch::table(batches, merged_metadata(tables))?;
        let output_rows = output.num_rows();
        self.metrics.lock().push(DataFusionQueryMetric {
            query_id: self.next_query.fetch_add(1, Ordering::Relaxed),
            node_id: node_id.map(str::to_owned),
            sql_parse_ns,
            logical_planning_ns,
            physical_planning_ns,
            physical_planning_count: 1,
            planning_ns,
            stream_open_ns,
            execution_ns,
            collect_ns,
            output_rows,
            configured_target_partitions: self.config.target_partitions,
            effective_target_partitions: self.effective_target_partitions.load(Ordering::Acquire),
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
        self.context_for_rows(0)
    }

    fn context_for_rows(&self, input_rows: usize) -> &SessionContext {
        self.context.get_or_init(|| {
            let target_partitions =
                adaptive_target_partitions(self.config.target_partitions, input_rows);
            self.effective_target_partitions
                .store(target_partitions, Ordering::Release);
            let session = SessionConfig::new()
                .with_batch_size(self.config.batch_size)
                .with_target_partitions(target_partitions);
            let state = SessionStateBuilder::new()
                .with_config(session)
                .with_default_features()
                .with_query_planner(Arc::new(CalcFlowQueryPlanner::new(Arc::clone(
                    &self.rolling_rewrite_audit,
                ))))
                .build();
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

const MIN_ROWS_PER_DATAFUSION_PARTITION: usize = 65_536;

fn adaptive_target_partitions(configured: usize, input_rows: usize) -> usize {
    if configured == 1 || input_rows == 0 {
        return 1;
    }
    configured.min(
        input_rows
            .div_ceil(MIN_ROWS_PER_DATAFUSION_PARTITION)
            .max(1),
    )
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

impl<'a> TableRegistrations<'a> {
    fn new(context: &'a SessionContext) -> Self {
        Self {
            context,
            aliases: Vec::new(),
        }
    }

    fn register(&mut self, alias: &str, batch: &Batch, node_id: Option<&str>) -> Result<()> {
        if !is_identifier(alias) {
            return Err(registration_error(
                alias,
                node_id,
                "alias must be a SQL identifier",
            ));
        }
        let table = batch
            .table_payload()
            .map_err(|error| registration_error(alias, node_id, error))?;
        let provider =
            MemTable::try_new(Arc::clone(table.schema()), vec![table.batches().to_vec()])
                .map_err(|error| registration_error(alias, node_id, error))?;
        self.context
            .register_table(alias, Arc::new(provider))
            .map_err(|error| registration_error(alias, node_id, error))?;
        self.aliases.push(alias.to_owned());
        Ok(())
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
}
