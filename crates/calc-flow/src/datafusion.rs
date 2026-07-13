use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use datafusion::{
    arrow::record_batch::RecordBatch,
    datasource::MemTable,
    execution::context::{SessionConfig, SessionContext},
    physical_plan::displayable,
};
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;

use crate::{
    Batch, BatchMetadata, CalcFlowError, Result, UdfKind, UdfReference, UdfRegistrySnapshot,
    sql_projection, validate_select_query, validate_selected_udfs,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataFusionQueryMetric {
    pub query_id: u64,
    pub node_id: Option<String>,
    pub planning_ns: u64,
    pub execution_ns: u64,
    pub output_rows: usize,
    pub logical_plan: String,
    pub physical_plan: String,
}

pub struct DataFusionRuntime {
    context: SessionContext,
    query_lock: AsyncMutex<()>,
    metrics: Mutex<Vec<DataFusionQueryMetric>>,
    next_query: AtomicU64,
    closed: AtomicBool,
}

impl DataFusionRuntime {
    /// Creates a run-scoped `DataFusion` session.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when either configuration
    /// value is zero.
    pub fn new(config: DataFusionConfig) -> Result<Self> {
        if config.batch_size == 0 || config.target_partitions == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "datafusion".into(),
                message: "batch_size and target_partitions must be positive".into(),
            });
        }
        let session = SessionConfig::new()
            .with_batch_size(config.batch_size)
            .with_target_partitions(config.target_partitions);
        Ok(Self {
            context: SessionContext::new_with_config(session),
            query_lock: AsyncMutex::new(()),
            metrics: Mutex::new(Vec::new()),
            next_query: AtomicU64::new(1),
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
    /// Returns an error when the runtime is closed, selected versions conflict,
    /// or a selected native reference is absent from the snapshot. Resolution
    /// completes before the shared session is mutated.
    pub fn register_udfs(
        &mut self,
        snapshot: &UdfRegistrySnapshot,
        references: &[UdfReference],
    ) -> Result<()> {
        self.ensure_open()?;
        validate_selected_udfs(references)?;
        let selected = references
            .iter()
            .filter(|reference| reference.kind == UdfKind::DataFusionScalar)
            .map(|reference| snapshot.resolve_native(reference))
            .collect::<Result<Vec<_>>>()?;
        for udf in selected {
            self.context.register_udf(udf.as_ref().clone());
        }
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
        let query = validate_select_query(query)?;
        // Declared before registrations so alias cleanup runs before unlock.
        let _query_guard = self.query_lock.lock().await;
        self.ensure_open()?;
        let mut registrations = TableRegistrations::new(&self.context);
        for (alias, batch) in tables {
            registrations.register(alias, batch, node_id)?;
        }

        let planning_start = Instant::now();
        let dataframe = self
            .context
            .sql(&query)
            .await
            .map_err(|error| datafusion_error(node_id, error))?;
        let logical_plan = dataframe.logical_plan().display_indent_schema().to_string();
        let physical_plan = dataframe
            .create_physical_plan()
            .await
            .map_err(|error| datafusion_error(node_id, error))?;
        let physical_plan = displayable(physical_plan.as_ref()).indent(true).to_string();
        let planning_ns = nanos(planning_start.elapsed());

        let execution_start = Instant::now();
        let batches = dataframe
            .collect()
            .await
            .map_err(|error| datafusion_error(node_id, error))?;
        let execution_ns = nanos(execution_start.elapsed());
        let output_rows = batches.iter().map(RecordBatch::num_rows).sum();
        let output = Batch::table(batches, merged_metadata(tables))?;
        self.metrics.lock().push(DataFusionQueryMetric {
            query_id: self.next_query.fetch_add(1, Ordering::Relaxed),
            node_id: node_id.map(str::to_owned),
            planning_ns,
            execution_ns,
            output_rows,
            logical_plan,
            physical_plan,
        });
        Ok(output)
    }

    pub fn metrics(&self) -> Vec<DataFusionQueryMetric> {
        self.metrics.lock().clone()
    }

    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
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
