//! Final ordering over a complete accepted history.
use super::distinct::owned;
use crate::{
    Batch, BatchKind, BatchOperator, BatchOperatorContext, JsonMap, OperatorMetadata, Port, Result,
    dataframe::invalid, state_memory_pool::OwnedStatePool,
};
use crate::{
    EventTime, StreamCollector, StreamOperator, StreamOperatorContext, StreamOperatorLifecycle,
};
use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::SchemaRef,
    execution::memory_pool::{MemoryPool, MemoryReservation, UnboundedMemoryPool},
};
use std::{collections::BTreeMap, sync::Arc};
mod checkpoint;
mod late;
mod sequence;
mod spec;
#[cfg(test)]
mod tests;
mod work;
pub(crate) use spec::validate_fetch;
pub use spec::{SortHistory, SortKey, SortSpec, SortTies};

/// Complete-history ordering with copied inputs and a node-owned peak budget.
pub struct BoundedOrderOperator {
    name: String,
    by: Vec<SortKey>,
    fetch: Option<u64>,
    spec: SortSpec,
    domain: crate::DistinctHistory,
    columns: Vec<String>,
    input: [Port; 1],
    output: [Port; 1],
    pool: Arc<OwnedStatePool>,
    _declaration: MemoryReservation,
    stream: Option<StreamState>,
    config: crate::DataFusionConfig,
}
struct StreamState {
    pool: Arc<OwnedStatePool>,
    history: Vec<Arc<owned::OwnedBatch>>,
    history_lease: Option<MemoryReservation>,
    accepted: usize,
    closed: bool,
    watermark: Option<EventTime>,
    _control: MemoryReservation,
    capture: Option<crate::OperatorStateSnapshot>,
    metrics: super::stream::LateMetricDelta,
}

impl BoundedOrderOperator {
    /// Declares a final sort (fetch=None) or top-k (fetch=Some(k)).
    /// # Errors
    /// Rejects empty/duplicate/unknown keys and unsupported data types.
    pub fn new(
        name: &str,
        schema: SchemaRef,
        by: Vec<SortKey>,
        fetch: Option<u64>,
        spec: SortSpec,
    ) -> Result<Self> {
        super::validate_operator_name(name)?;
        validate_fetch(fetch, &spec)?;
        if by.is_empty() {
            return Err(invalid(
                "sort.by",
                "invalid_literal",
                "at least one sort key required",
            ));
        }
        for (i, key) in by.iter().enumerate() {
            if by[..i].iter().any(|prior| prior.column() == key.column()) {
                return Err(invalid(
                    &format!("sort.by[{i}].column"),
                    "duplicate_name",
                    "unique sort columns required",
                ));
            }
            schema.field_with_name(key.column()).map_err(|_| {
                invalid(
                    &format!("sort.by[{i}].column"),
                    "unresolved_type",
                    "unknown sort column",
                )
            })?;
        }
        sequence::validate_schema(&schema, &by, &spec)?;
        if let SortHistory::ClosedWindow {
            event_time_column, ..
        } = spec.history()
        {
            let field = schema.field_with_name(event_time_column).map_err(|_| {
                invalid(
                    "sort.spec.history.event_time_column",
                    "unresolved_type",
                    "requires an existing original timestamp column",
                )
            })?;
            if !super::distinct::window_time_type(field.data_type()) {
                return Err(invalid(
                    "sort.spec.history.event_time_column",
                    "unsupported_type",
                    "requires timestamp[ms] naive or timestamp[us] naive/UTC",
                ));
            }
        }
        let pool = Arc::new(OwnedStatePool::new(
            Arc::new(UnboundedMemoryPool::default()),
            "bounded_order copy_visible_v2 owned peak",
            "bounded_order max_bytes",
        ));
        pool.set_limit(Some(spec.state_budget().max_bytes()))
            .map_err(owned::error)?;
        let erased: Arc<dyn MemoryPool> = pool.clone();
        let bound = owned::add(owned::mul(owned::schema_quote(&schema)?, 4)?, 4096)?;
        let declaration = owned::reserve(&erased, bound, "declaration schema/metadata")?;
        let fresh_by = by.clone();
        drop(by);
        let by = fresh_by;
        let fresh_spec = spec.clone();
        drop(spec);
        let spec = fresh_spec;
        let columns = schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>();
        let fresh_schema = owned::order_schema(&schema, &columns).map_err(schema_error)?;
        drop(schema);
        let schema = fresh_schema;
        let domain = spec.domain_spec()?.history().clone();
        Ok(Self {
            name: name.into(),
            by,
            fetch,
            spec,
            domain,
            columns,
            input: [Port::with_schema_ref(
                "input",
                BatchKind::Table,
                true,
                Some(schema.clone()),
            )?],
            output: [Port::with_schema_ref(
                "output",
                BatchKind::Table,
                true,
                Some(schema),
            )?],
            pool,
            _declaration: declaration,
            stream: None,
            config: crate::DataFusionConfig::default(),
        })
    }
    pub(crate) fn declaration(&self) -> crate::query_analysis::order::OrderDeclaration {
        crate::query_analysis::order::OrderDeclaration {
            by: self.by.clone(),
            fetch: self.fetch,
            spec: self.spec.clone(),
        }
    }
    pub(crate) fn spec(&self) -> &SortSpec {
        &self.spec
    }
    pub(crate) fn set_datafusion_config(&mut self, config: crate::DataFusionConfig) {
        self.config = config;
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
                history_lease: None,
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
        let Some((combined, _pool)) = self.close_input(context)? else {
            return Ok(());
        };
        let result = self.finalize_stream(combined, context).await?;
        validate_final(&result.batch, context.output_budget())?;
        context.check_cancelled()?;
        output.emit("output", result.batch.clone()).await?;
        let state = self.stream.as_mut().unwrap();
        state.closed = true;
        state.history = Vec::new();
        state.history_lease = None;
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
        let rows = owned::count(raw, Some(&self.domain), &|| context.check_cancelled())?;
        let accepted = owned::add(old, rows)?;
        if accepted as u64 > self.spec.state_budget().max_rows() {
            return Err(history_error(
                &self.name,
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
        let copy = owned::normalize(&raw, &self.columns, &pool, Some(&self.domain), &|| {
            context.check_cancelled()
        })?;
        drop(raw);
        let candidate_lease = history_candidate_lease(state, &pool)?;
        let mut history = Vec::with_capacity(state.history.len() + 1);
        history.extend(state.history.iter().cloned());
        history.push(Arc::new(copy));
        self.validate_candidate(&history, &pool, &|| context.check_cancelled())?;
        context.check_cancelled()?;
        let state = self.stream.as_mut().unwrap();
        state.history = history;
        state.history_lease = Some(candidate_lease);
        state.accepted = accepted;
        state.capture = None;
        Ok(())
    }
    fn finite_input(
        &self,
        raw: &Batch,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<owned::OwnedBatch> {
        self.input[0].validate(raw, "sort.input")?;
        let count = owned::count(raw, Some(&self.domain), check)?;
        if count as u64 > self.spec.state_budget().max_rows() {
            return Err(history_error(
                &self.name,
                "state_budget_exceeded",
                format!(
                    "copy_visible_v2 history required={count} rows limit={}",
                    self.spec.state_budget().max_rows()
                ),
            ));
        }
        let pool: Arc<dyn MemoryPool> = self.pool.clone();
        owned::normalize(raw, &self.columns, &pool, Some(&self.domain), check)
    }
}
fn workspace_bound(history: &owned::OwnedBatch, keys: usize) -> Result<usize> {
    work::validate_concat(history)?;
    // Two full owned table layouts cover concat and take; sorted indices and
    // the largest admitted single-column (u32, slice) scratch coexist with H.
    let arrays = owned::mul(owned::backing(&history.batch)?, 2)?;
    let indices = owned::mul(history.batch.num_rows(), 32)?;
    owned::add(
        owned::add(arrays, indices)?,
        owned::add(8192, owned::mul(keys, 4096)?)?,
    )
}
impl OperatorMetadata for BoundedOrderOperator {
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
            ("kind".into(), serde_json::json!("bounded_order_v1")),
            ("by".into(), serde_json::json!(self.by)),
            ("fetch".into(), serde_json::json!(self.fetch)),
            ("spec".into(), serde_json::json!(self.spec)),
        ])
    }
}
#[async_trait]
impl BatchOperator for BoundedOrderOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let check = || context.run.check_cancelled();
        let input = inputs
            .get("input")
            .ok_or_else(|| invalid("sort.input", "invalid_literal", "missing input"))?;
        let history = self.finite_input(input, &check)?;
        let pool: Arc<dyn MemoryPool> = self.pool.clone();
        self.validate_sequence(&history, &pool, &check)?;
        let result = self.finalize_finite(history, context.run).await?;
        Ok(BTreeMap::from([("output".into(), result.batch)]))
    }
}

#[async_trait]
impl StreamOperator for BoundedOrderOperator {
    fn lifecycle(&self) -> StreamOperatorLifecycle {
        StreamOperatorLifecycle::CheckpointedStateful {
            state_version: 1,
            microbatch_invariant: matches!(self.spec.ties(), SortTies::Sequence { .. }),
            deterministic: matches!(self.spec.ties(), SortTies::Sequence { .. }),
            replay_safe: matches!(self.spec.ties(), SortTies::Sequence { .. }),
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
        self.input[0].validate(&raw, "sort.input")?;
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
        if matches!(self.spec.history(),SortHistory::ClosedWindow {end_micros,..} if watermark.as_micros()>=*end_micros)
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
) -> Result<MemoryReservation> {
    let history = state
        .history
        .iter()
        .try_fold(0, |n, h| owned::add(n, h.charge()))?;
    let bound = owned::add(owned::mul(state.history.len() + 1, 64)?, history)?;
    owned::reserve(pool, bound, "candidate history")
}

fn qualify(name: &str, error: crate::CalcFlowError) -> crate::CalcFlowError {
    match error {
        crate::CalcFlowError::InvalidArgument { field, message } => {
            crate::CalcFlowError::InvalidArgument {
                field: format!("operators.{name}.{}", field.replace("distinct", "sort")),
                message,
            }
        }
        crate::CalcFlowError::OperatorReason {
            reason_code,
            message,
            ..
        } => crate::CalcFlowError::OperatorReason {
            node_id: name.into(),
            reason_code,
            message,
        },
        other => other,
    }
}

fn history_error(name: &str, _code: &str, message: impl std::fmt::Display) -> crate::CalcFlowError {
    crate::CalcFlowError::OperatorReason {
        node_id: name.into(),
        reason_code: crate::StreamingFailureReason::StateBudgetExceeded,
        message: format!("state_budget_exceeded: {message}"),
    }
}

fn schema_error(error: crate::CalcFlowError) -> crate::CalcFlowError {
    match error {
        crate::CalcFlowError::InvalidArgument { field, message } => {
            crate::CalcFlowError::InvalidArgument {
                field: field
                    .replace("distinct.columns", "sort.input.schema")
                    .replace("distinct", "sort"),
                message,
            }
        }
        other => other,
    }
}
