//! Final-only relations with mandatory owned input and output normalization.
use std::{collections::BTreeMap, sync::Arc, time::Instant};

use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::SchemaRef,
    execution::memory_pool::{MemoryPool, UnboundedMemoryPool},
};

use super::{BatchOperator, BatchOperatorContext, OperatorMetadata};
use crate::{
    Batch, BatchKind, BatchMetadata, DataFusionConfig, DataFusionRuntime, EventTime, JsonMap, Port,
    Result, StreamCollector, StreamOperator, StreamOperatorContext, StreamOperatorLifecycle,
    dataframe::{invalid, quote},
    state_memory_pool::OwnedStatePool,
};

mod budget;
mod checkpoint;
mod late;
mod owned;
mod spec;
pub use spec::{DistinctHistory, DistinctSpec};

/// Native final DISTINCT. Raw ingress is borrowed synchronously and copied.
///
/// `max_bytes` bounds this node's owned participating peak, including tracked
/// `DataFusion` workspace. Unknown external ingress backing during the temporary
/// borrow, other graph owners, sources, callers and process RSS are excluded.
pub struct DistinctOperator {
    name: String,
    columns: Vec<String>,
    spec: DistinctSpec,
    input: [Port; 1],
    output: [Port; 1],
    stream: Option<StreamState>,
    config: DataFusionConfig,
    pool: Arc<OwnedStatePool>,
    _declaration: datafusion::execution::memory_pool::MemoryReservation,
}

struct StreamState {
    pool: Arc<OwnedStatePool>,
    history: Vec<Arc<owned::OwnedBatch>>,
    accepted: usize,
    closed: bool,
    watermark: Option<EventTime>,
    _control: datafusion::execution::memory_pool::MemoryReservation,
    capture: Option<crate::OperatorStateSnapshot>,
    metrics: super::stream::LateMetricDelta,
}

fn qualify(name: &str, error: crate::CalcFlowError) -> crate::CalcFlowError {
    if let Some(message) = budget::detail(&error, name) {
        return crate::CalcFlowError::OperatorReason {
            node_id: name.into(),
            reason_code: crate::StreamingFailureReason::StateBudgetExceeded,
            message,
        };
    }
    match error {
        crate::CalcFlowError::OperatorReason {
            reason_code,
            message,
            ..
        } => crate::CalcFlowError::OperatorReason {
            node_id: name.into(),
            reason_code,
            message,
        },
        crate::CalcFlowError::InvalidArgument { field, message } => {
            crate::CalcFlowError::InvalidArgument {
                field: format!("operators.{name}.{field}"),
                message,
            }
        }
        crate::CalcFlowError::DataFusion { message, .. } => crate::CalcFlowError::DataFusion {
            node_id: Some(name.into()),
            message,
        },
        other => other,
    }
}

impl DistinctOperator {
    /// Declares exact-schema final DISTINCT over ordered keys.
    ///
    /// All admitted values are copied under `copy_visible_v2`. The budget covers
    /// node-owned peak storage; temporarily borrowed external ingress is excluded.
    ///
    /// # Errors
    /// Rejects empty/duplicate/unknown keys and unsupported key data types.
    pub fn new(
        name: &str,
        schema: SchemaRef,
        columns: Vec<String>,
        spec: DistinctSpec,
    ) -> Result<Self> {
        super::validate_operator_name(name)?;
        let (pool, declaration) = declaration_pool(name, &schema, &columns, &spec)?;
        let fresh_schema = owned::fresh_schema(&schema);
        drop(schema);
        let schema = fresh_schema;
        let fresh_columns = columns
            .iter()
            .map(|s| s.as_str().to_owned())
            .collect::<Vec<_>>();
        drop(columns);
        let columns = fresh_columns;
        validate_window_type(&schema, &spec)?;
        // The validated time name is covered by the declaration reservation;
        // cloning releases any excess capacity carried by a deserialized spec.
        let fresh_spec = spec.clone();
        drop(spec);
        let spec = fresh_spec;
        let output = owned::projected_schema(&schema, &columns)?;
        Ok(Self {
            name: name.into(),
            columns,
            spec,
            stream: None,
            config: DataFusionConfig::default(),
            pool,
            _declaration: declaration,
            input: [Port::with_schema_ref(
                "input",
                BatchKind::Table,
                true,
                Some(schema),
            )?],
            output: [Port::with_schema_ref(
                "output",
                BatchKind::Table,
                true,
                Some(output),
            )?],
        })
    }

    pub(crate) fn set_datafusion_config(&mut self, config: DataFusionConfig) {
        self.config = config;
    }
    pub(crate) fn spec(&self) -> &DistinctSpec {
        &self.spec
    }

    async fn finite(&self, input: &Batch, context: &BatchOperatorContext<'_>) -> Result<Batch> {
        let check = || context.run.check_cancelled();
        check()?;
        self.input[0].validate(input, "distinct.input")?;
        let rows = owned::count(input, Some(self.spec.history()), &check)?;
        if rows as u64 > self.spec.state_budget().max_rows() {
            return Err(invalid(
                "distinct.spec.state_budget",
                "state_budget_exceeded",
                format!(
                    "history required={} rows limit={}; copy_visible_v2",
                    rows,
                    self.spec.state_budget().max_rows()
                ),
            ));
        }
        let pool: Arc<dyn MemoryPool> = self.pool.clone();
        // The borrowed input never enters this function's asynchronous query scope.
        let history = owned::normalize(
            input,
            &self.columns,
            &pool,
            Some(self.spec.history()),
            &check,
        )?;
        Ok(self.finalize(&history, pool, &check).await?.batch)
    }

    async fn finalize(
        &self,
        history: &owned::OwnedBatch,
        pool: Arc<dyn MemoryPool>,
        check: &(dyn Fn() -> Result<()> + Sync),
    ) -> Result<owned::OwnedBatch> {
        let (runtime, lease) = self.prepare_query(history, &pool)?;
        check()?;
        let result = runtime
            .sql(
                &self.query_text(),
                &BTreeMap::from([("input".into(), history.batch.clone())]),
                Some(&self.name),
            )
            .await?;
        check()?;
        let compact = self.compact_result(&result, lease.size(), &pool, check)?;
        check()?;
        Ok(compact)
    }
    fn query_text(&self) -> String {
        format!(
            "SELECT DISTINCT {} FROM input",
            self.columns
                .iter()
                .map(|n| quote(n))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
    fn prepare_query(
        &self,
        history: &owned::OwnedBatch,
        pool: &Arc<dyn MemoryPool>,
    ) -> Result<(
        DataFusionRuntime,
        datafusion::execution::memory_pool::MemoryReservation,
    )> {
        let partitions = match self.config.parallelism_mode {
            crate::DataFusionParallelismMode::Fixed => self.config.target_partitions,
            _ => self.config.max_partitions,
        };
        let bound = prepared_bound(history.charge(), self.columns.len(), partitions)?;
        let lease = owned::reserve(pool, bound, "prepared query result")?;
        let runtime = DataFusionRuntime::new_with_pool(self.config, pool.clone(), Instant::now())?;
        Ok((runtime, lease))
    }
    fn compact_result(
        &self,
        result: &Batch,
        bound: usize,
        pool: &Arc<dyn MemoryPool>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<owned::OwnedBatch> {
        self.output[0].validate(result, "distinct.output")?;
        if owned::backing(result)? > bound {
            return Err(owned::error("query result exceeded prepaid bound"));
        }
        let mut compact = owned::normalize(result, &self.columns, pool, None, check)?;
        compact.batch =
            compact
                .batch
                .with_metadata(BatchMetadata::new(&self.name, 0, JsonMap::new())?);
        Ok(compact)
    }

    fn ensure_stream(&mut self) -> Result<()> {
        if self.stream.is_none() {
            let pool = self.pool.clone();
            let erased: Arc<dyn MemoryPool> = pool.clone();
            let control = owned::reserve(&erased, 4096 + self.metric_storage(), "stream control")
                .map_err(|e| qualify(&self.name, e))?;
            self.stream = Some(StreamState {
                pool,
                history: vec![],
                accepted: 0,
                closed: false,
                watermark: None,
                _control: control,
                capture: None,
                metrics: super::stream::LateMetricDelta::default(),
            });
        }
        Ok(())
    }

    async fn close(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let Some((combined, pool)) = self.close_input(context)? else {
            return Ok(());
        };
        let result = self
            .finalize(&combined, pool, &|| context.check_cancelled())
            .await?;
        validate_final(&result.batch, context.output_budget())?;
        context.check_cancelled()?;
        output.emit("output", result.batch.clone()).await?;
        let state = self.stream.as_mut().unwrap();
        state.closed = true;
        state.history = Vec::new();
        state.capture = None;
        Ok(())
    }
    fn close_input(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<(owned::OwnedBatch, Arc<dyn MemoryPool>)>> {
        context.check_cancelled()?;
        self.ensure_stream()?;
        let state = self.stream.as_ref().unwrap();
        if state.closed {
            return Ok(None);
        }
        let pool: Arc<dyn MemoryPool> = state.pool.clone();
        let combined = owned::combine(
            &state.history,
            self.output[0].schema().unwrap(),
            &self.columns,
            &pool,
            &|| context.check_cancelled(),
        )?;
        Ok(Some((combined, pool)))
    }
    fn accepted_rows(&self, raw: &Batch, context: &StreamOperatorContext<'_>) -> Result<usize> {
        let old = self.stream.as_ref().map_or(0, |s| s.accepted);
        let rows = owned::count(raw, Some(self.spec.history()), &|| {
            context.check_cancelled()
        })?;
        let accepted = owned::add(old, rows)?;
        if accepted as u64 > self.spec.state_budget().max_rows() {
            return Err(invalid(
                "distinct.spec.state_budget",
                "state_budget_exceeded",
                format!(
                    "copy_visible_v2 history required={accepted} rows limit={}",
                    self.spec.state_budget().max_rows()
                ),
            ));
        }
        Ok(accepted)
    }
    fn admit_data(&mut self, raw: Batch, context: &StreamOperatorContext<'_>) -> Result<()> {
        let accepted = self.accepted_rows(&raw, context)?;
        self.ensure_stream()?;
        let state = self.stream.as_ref().unwrap();
        let pool: Arc<dyn MemoryPool> = state.pool.clone();
        let copy = owned::normalize(
            &raw,
            &self.columns,
            &pool,
            Some(self.spec.history()),
            &|| context.check_cancelled(),
        )?;
        drop(raw);
        let _candidate = history_candidate_lease(state, &pool)?;
        let mut history = Vec::with_capacity(state.history.len() + 1);
        history.extend(state.history.iter().cloned());
        history.push(Arc::new(copy));
        context.check_cancelled()?;
        let state = self.stream.as_mut().unwrap();
        state.history = history;
        state.accepted = accepted;
        state.capture = None;
        Ok(())
    }
}

impl OperatorMetadata for DistinctOperator {
    fn name(&self) -> &str {
        &self.name
    }
    fn input_ports(&self) -> &[Port] {
        &self.input
    }
    fn output_ports(&self) -> &[Port] {
        &self.output
    }
    fn configuration(&self) -> JsonMap {
        BTreeMap::from([
            ("kind".into(), serde_json::json!("distinct_v2")),
            ("columns".into(), serde_json::json!(self.columns)),
            ("spec".into(), serde_json::json!(self.spec)),
        ])
    }
}

#[async_trait]
impl BatchOperator for DistinctOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let input = inputs
            .get("input")
            .ok_or_else(|| invalid("distinct.input", "invalid_literal", "missing input"))?;
        self.finite(input, context)
            .await
            .map(|output| BTreeMap::from([("output".into(), output)]))
            .map_err(|e| qualify(&self.name, e))
    }
}

#[async_trait]
impl StreamOperator for DistinctOperator {
    fn lifecycle(&self) -> StreamOperatorLifecycle {
        StreamOperatorLifecycle::CheckpointedStateful {
            state_version: 1,
            microbatch_invariant: true,
            deterministic: true,
            replay_safe: true,
        }
    }
    async fn process_data(
        &mut self,
        ingress: &str,
        raw: Batch,
        context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        if ingress != "input" {
            return Err(owned::error("unknown ingress"));
        }
        self.input[0].validate(&raw, "distinct.input")?;
        if self.stream.as_ref().is_some_and(|state| state.closed) {
            return self
                .closed_data(raw, context)
                .map_err(|e| qualify(&self.name, e));
        }
        self.admit_data(raw, context)
            .map_err(|e| qualify(&self.name, e))
    }

    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        if matches!(self.spec.history(),DistinctHistory::ClosedWindow {end_micros,..} if watermark.as_micros()>=*end_micros)
        {
            self.close(context, output)
                .await
                .map_err(|e| qualify(&self.name, e))?;
        }
        self.ensure_stream()?;
        self.stream.as_mut().unwrap().watermark = Some(watermark);
        self.stream.as_mut().unwrap().capture = None;
        Ok(())
    }
    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.close(context, output)
            .await
            .map_err(|e| qualify(&self.name, e))
    }
    fn reset(&mut self) -> Result<()> {
        self.stream = None;
        Ok(())
    }
    async fn prepare_checkpoint_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.capture_checked(&|| context.check_cancelled())
            .map(|_| ())
            .map_err(|e| qualify(&self.name, e))
    }
    async fn restore_async(
        &mut self,
        snapshot: &crate::OperatorStateSnapshot,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.restore_with_context(snapshot, context)
            .map_err(|e| qualify(&self.name, e))
    }
    fn checkpoint(&mut self, _epoch: crate::Epoch) -> Result<crate::OperatorStateSnapshot> {
        self.capture().map_err(|e| qualify(&self.name, e))
    }
    fn restore(&mut self, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
        self.restore_capture(snapshot)
            .map_err(|e| qualify(&self.name, e))
    }
}

fn declaration_pool(
    name: &str,
    schema: &SchemaRef,
    columns: &[String],
    spec: &DistinctSpec,
) -> Result<(
    Arc<OwnedStatePool>,
    datafusion::execution::memory_pool::MemoryReservation,
)> {
    let pool = Arc::new(OwnedStatePool::new(
        Arc::new(UnboundedMemoryPool::default()),
        "distinct copy_visible_v2 owned peak",
        "distinct max_bytes",
    ));
    pool.set_limit(Some(spec.state_budget().max_bytes()))
        .map_err(owned::error)?;
    let erased: Arc<dyn MemoryPool> = pool.clone();
    let policy = if spec.state_layout() == 3 { 64 } else { 0 };
    let bound = owned::add(declaration_bound(name, schema, columns)?, policy)?;
    let lease = owned::reserve(&erased, bound, "declaration schema/metadata")
        .map_err(|e| qualify(name, e))?;
    Ok((pool, lease))
}
fn declaration_bound(name: &str, schema: &SchemaRef, columns: &[String]) -> Result<usize> {
    let parts = [
        owned::mul(owned::schema_quote(schema)?, 4)?,
        column_names_bound(columns)?,
        owned::mul(name.len(), 4)?,
    ];
    parts.into_iter().try_fold(0, owned::add)
}
fn column_names_bound(columns: &[String]) -> Result<usize> {
    columns.iter().try_fold(0, |n, c| {
        owned::add(n, owned::add(256, owned::mul(c.len(), 4)?)?)
    })
}
fn validate_window_type(schema: &SchemaRef, spec: &DistinctSpec) -> Result<()> {
    if let DistinctHistory::ClosedWindow {
        event_time_column, ..
    } = spec.history()
    {
        let time = schema.field_with_name(event_time_column).map_err(|_| {
            invalid(
                "distinct.spec.history.event_time_column",
                "unresolved_type",
                event_time_column,
            )
        })?;
        if !window_time_type(time.data_type()) {
            return Err(invalid(
                "distinct.spec.history.event_time_column",
                "unsupported_type",
                "requires timestamp[ms] naive or timestamp[us] naive/UTC",
            ));
        }
    }
    Ok(())
}
fn window_time_type(dtype: &datafusion::arrow::datatypes::DataType) -> bool {
    use datafusion::arrow::datatypes::{DataType, TimeUnit};
    matches!(dtype, DataType::Timestamp(TimeUnit::Millisecond, None))
        || matches!(dtype,DataType::Timestamp(TimeUnit::Microsecond,tz) if tz.as_deref().is_none_or(|tz|tz=="UTC"))
}
fn prepared_bound(history: usize, columns: usize, partitions: usize) -> Result<usize> {
    owned::add(
        owned::mul(history, 16)?,
        owned::mul(owned::mul(columns, partitions)?, 131_072)?,
    )
}
fn validate_final(batch: &Batch, budget: crate::EdgeBudget) -> Result<()> {
    if batch.num_rows() > budget.max_rows || batch.estimated_bytes()? > budget.max_bytes {
        return Err(owned::error(
            "final relation exceeds whole output edge budget",
        ));
    }
    Ok(())
}
fn history_candidate_lease(
    state: &StreamState,
    pool: &Arc<dyn MemoryPool>,
) -> Result<datafusion::execution::memory_pool::MemoryReservation> {
    let history = state
        .history
        .iter()
        .try_fold(0, |n, h| owned::add(n, h.charge()))?;
    let bound = owned::add(owned::mul(state.history.len() + 1, 64)?, history)?;
    owned::reserve(pool, bound, "candidate history")
}
