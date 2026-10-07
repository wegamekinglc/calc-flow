use super::{
    Arc, BTreeMap, Batch, CalcFlowError, CollectedOutput, DataFusionQueryMetric, DataFusionRuntime,
    ExecutionPlanProperties, Instant, MAX_SQL_RESULT_BYTES, MAX_SQL_RESULT_ROWS, Ordering,
    PlannedQuery, RecordBatch, Result, SchemaRef, SendableRecordBatchStream, SessionContext,
    StreamExt, TableRegistrations, ValidatedQuery, active_entities, datafusion_error, diagnostic,
    displayable, execute_stream, merged_metadata, nanos, plan_statement, push_bounded_sql_batch,
    register_tables, require_tables,
};
use datafusion::{
    common::DataFusionError,
    datasource::memory::{DataSourceExec, MemorySourceConfig},
    physical_plan::{
        ExecutionPlan,
        coalesce_partitions::CoalescePartitionsExec,
        coop::CooperativeExec,
        empty::EmptyExec,
        filter::FilterExec,
        joins::{CrossJoinExec, HashJoinExec, PartitionMode},
        projection::ProjectionExec,
    },
};

#[cfg(test)]
thread_local! {
    static LEGACY_INPUTS: std::cell::RefCell<Option<Vec<RecordBatch>>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(test)]
pub(crate) fn observe_legacy_inputs() {
    LEGACY_INPUTS.with(|inputs| *inputs.borrow_mut() = Some(Vec::new()));
}

#[cfg(test)]
pub(crate) fn take_legacy_inputs() -> Vec<RecordBatch> {
    LEGACY_INPUTS.with(|inputs| {
        inputs
            .borrow_mut()
            .take()
            .expect("enabled legacy observation")
    })
}

#[cfg(test)]
pub(crate) fn note_legacy_inputs(tables: &BTreeMap<String, Batch>) {
    LEGACY_INPUTS.with(|inputs| {
        if let Some(inputs) = inputs.borrow_mut().as_mut() {
            for table in tables.values() {
                inputs.extend_from_slice(table.table_payload().unwrap().batches());
            }
        }
    });
}

pub(crate) struct Input<O> {
    pub(crate) tables: BTreeMap<String, Batch>,
    owner: O,
}

impl<O> Input<O> {
    pub(crate) const fn new(tables: BTreeMap<String, Batch>, owner: O) -> Self {
        Self { tables, owner }
    }

    pub(crate) fn finish(self) -> O {
        let Self { tables, owner } = self;
        drop(tables);
        owner
    }

    pub(crate) const fn owner(&self) -> &O {
        &self.owner
    }
}

pub(crate) struct Output<O> {
    batch: Batch,
    _input: Input<O>,
}

impl<O> Output<O> {
    pub(crate) const fn batch(&self) -> &Batch {
        &self.batch
    }

    pub(crate) fn finish(self) -> O {
        let Self {
            batch,
            _input: input,
        } = self;
        drop(batch);
        input.finish()
    }
}

#[derive(Debug)]
pub(crate) enum Error {
    Unproved,
    Runtime(CalcFlowError),
    DataFusion(DataFusionError),
}

impl From<CalcFlowError> for Error {
    fn from(error: CalcFlowError) -> Self {
        Self::Runtime(error)
    }
}

impl From<DataFusionError> for Error {
    fn from(error: DataFusionError) -> Self {
        Self::DataFusion(error)
    }
}

impl Error {
    pub(crate) fn retry_legacy(&self, optional_paid: bool) -> bool {
        match self {
            Self::Unproved => true,
            Self::DataFusion(error) => {
                optional_paid && matches!(error.find_root(), DataFusionError::ResourcesExhausted(_))
            }
            Self::Runtime(_) => false,
        }
    }
}

pub(crate) struct Failure<O> {
    pub(crate) error: Error,
    pub(crate) input: Input<O>,
}

impl<O> Failure<O> {
    pub(crate) fn into_parts(self, node_id: Option<&str>) -> (Option<CalcFlowError>, Input<O>) {
        let Self { error, input } = self;
        let error = match error {
            Error::Unproved => None,
            Error::Runtime(error) => Some(error),
            Error::DataFusion(error) => Some(datafusion_error(node_id, error)),
        };
        (error, input)
    }
}

struct Query<'a, O> {
    stream: Option<SendableRecordBatchStream>,
    planned: Option<PlannedQuery<'a>>,
    registrations: Option<TableRegistrations<'a>>,
    input: Input<O>,
}

impl<O> Query<'_, O> {
    fn release(self) -> Input<O> {
        let Self {
            stream,
            planned,
            registrations,
            input,
        } = self;
        drop(stream);
        drop(planned);
        drop(registrations);
        input
    }
}

pub(crate) fn serial_plan(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let mut stack = [None; 32];
    stack[0] = Some(plan);
    let mut count = 1;
    let mut visited = 0;
    while count != 0 {
        count -= 1;
        let node = stack[count].take().expect("occupied traversal slot");
        visited += 1;
        if visited > 64 || node.output_partitioning().partition_count() != 1 {
            return false;
        }
        let Some(children) = serial_children(node) else {
            return false;
        };
        for child in children.into_iter().flatten() {
            if count == stack.len() {
                return false;
            }
            stack[count] = Some(child);
            count += 1;
        }
    }
    true
}

fn serial_children(plan: &Arc<dyn ExecutionPlan>) -> Option<[Option<&Arc<dyn ExecutionPlan>>; 2]> {
    let value = plan.as_ref() as &dyn std::any::Any;
    if let Some(scan) = value.downcast_ref::<DataSourceExec>() {
        return (scan.data_source().as_ref() as &dyn std::any::Any)
            .is::<MemorySourceConfig>()
            .then_some([None, None]);
    }
    if value.is::<EmptyExec>() {
        return Some([None, None]);
    }
    unary_children(plan).or_else(|| binary_children(plan))
}

fn unary_children(plan: &Arc<dyn ExecutionPlan>) -> Option<[Option<&Arc<dyn ExecutionPlan>>; 2]> {
    let value = plan.as_ref() as &dyn std::any::Any;
    let input = if let Some(node) = value.downcast_ref::<ProjectionExec>() {
        node.input()
    } else if let Some(node) = value.downcast_ref::<FilterExec>() {
        node.input()
    } else if let Some(node) = value.downcast_ref::<CoalescePartitionsExec>() {
        node.input()
    } else if let Some(node) = value.downcast_ref::<CooperativeExec>() {
        node.input()
    } else {
        return None;
    };
    Some([Some(input), None])
}

fn binary_children(plan: &Arc<dyn ExecutionPlan>) -> Option<[Option<&Arc<dyn ExecutionPlan>>; 2]> {
    let value = plan.as_ref() as &dyn std::any::Any;
    if let Some(node) = value.downcast_ref::<HashJoinExec>() {
        if *node.partition_mode() == PartitionMode::Auto {
            return None;
        }
        return Some([Some(node.left()), Some(node.right())]);
    }
    value
        .downcast_ref::<CrossJoinExec>()
        .map(|node| [Some(node.left()), Some(node.right())])
}

impl DataFusionRuntime {
    #[cfg(test)]
    pub(crate) async fn lock_owned_test_query(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.query_lock.lock().await
    }

    pub(crate) async fn sql_equality_owned<O: Send + Sync>(
        &self,
        query: &ValidatedQuery,
        input: Input<O>,
        node_id: Option<&str>,
    ) -> std::result::Result<Output<O>, Failure<O>> {
        let _query_guard = self.query_lock.lock().await;
        let mut scope = Query {
            stream: None,
            planned: None,
            registrations: None,
            input,
        };
        let result = self.execute_owned(query, &mut scope, node_id).await;
        let input = scope.release();
        match result {
            Ok(batch) => Ok(Output {
                batch,
                _input: input,
            }),
            Err(error) => Err(Failure { error, input }),
        }
    }

    async fn execute_owned<'a, O: Send + Sync>(
        &'a self,
        query: &ValidatedQuery,
        scope: &mut Query<'a, O>,
        node_id: Option<&str>,
    ) -> std::result::Result<Batch, Error> {
        let (input_rows, entities, source) = self.validate_owned_input(&scope.input)?;
        let context_preexisting = self.context.get().is_some();
        let start = Instant::now();
        let context = self.context_for_rows(input_rows, entities, source);
        let context_ns = u64::from(!context_preexisting) * nanos(start.elapsed());
        self.prepare_owned_plan(query, scope, context, node_id)
            .await?;
        let planned = scope.planned.as_ref().expect("planned query installed");
        let start = Instant::now();
        scope.stream = Some(execute_stream(
            Arc::clone(&planned.physical_plan),
            Arc::clone(&planned.task_ctx),
        )?);
        let stream_ns = nanos(start.elapsed());
        let mut collected = collect_owned(
            scope.stream.as_mut().expect("stream installed"),
            planned.physical_plan.schema(),
            start,
        )
        .await?;
        let envelope_start = Instant::now();
        let output = Batch::table(
            std::mem::take(&mut collected.batches),
            merged_metadata(&scope.input.tables),
        )?;
        let envelope_ns = nanos(envelope_start.elapsed());
        let metric = MetricInput {
            planned: scope.planned.take().expect("planned query installed"),
            collected,
            context_ns,
            context_preexisting,
            stream_ns,
            envelope_ns,
            input_rows,
            entities,
            source,
            output_rows: output.num_rows(),
        };
        record_metric(self, metric, node_id)?;
        Ok(output)
    }

    fn validate_owned_input<O>(
        &self,
        input: &Input<O>,
    ) -> std::result::Result<(usize, Option<usize>, &'static str), Error> {
        self.ensure_open()?;
        require_tables(&input.tables)?;
        if !self.serial_owned_sql() {
            return Err(Error::Unproved);
        }
        let input_rows = input.tables.values().fold(0_usize, |total, batch| {
            total.saturating_add(batch.num_rows())
        });
        let (entities, source) = active_entities(&input.tables, input_rows);
        Ok((input_rows, entities, source))
    }

    async fn prepare_owned_plan<'a, O>(
        &self,
        query: &ValidatedQuery,
        scope: &mut Query<'a, O>,
        context: &'a SessionContext,
        node_id: Option<&str>,
    ) -> std::result::Result<(), Error> {
        let (registrations, adapter_ns, register_ns) =
            register_tables(context, &scope.input.tables, node_id)?;
        scope.registrations = Some(registrations);
        let mut planned = plan_uncached(self, context, query).await?;
        planned.input_adapter_ns = adapter_ns;
        planned.table_register_ns = register_ns;
        scope.planned = Some(planned);
        let planned = scope.planned.as_ref().expect("planned query installed");
        if !serial_plan(&planned.physical_plan) {
            return Err(Error::Unproved);
        }
        Ok(())
    }
}

async fn plan_uncached<'a>(
    runtime: &DataFusionRuntime,
    context: &'a SessionContext,
    query: &ValidatedQuery,
) -> std::result::Result<PlannedQuery<'a>, Error> {
    let start = Instant::now();
    let dataframe = plan_statement(context, query).await?;
    let (logical_plan, logical_plan_string_ns) =
        diagnostic(runtime.config.collect_diagnostics, || {
            dataframe.logical_plan().display_indent_schema().to_string()
        });
    let logical_planning_ns = nanos(start.elapsed());
    let start = Instant::now();
    let physical_plan = dataframe.create_physical_plan().await?;
    let (physical_plan_text, physical_plan_string_ns) =
        diagnostic(runtime.config.collect_diagnostics, || {
            displayable(physical_plan.as_ref()).indent(true).to_string()
        });
    let audit_start = Instant::now();
    let rolling_audit = runtime.rolling_rewrite_audit.snapshot();
    let audit_ns = nanos(audit_start.elapsed());
    Ok(PlannedQuery {
        task_ctx: Arc::new(dataframe.task_ctx()),
        physical_plan,
        logical_plan,
        logical_plan_string_ns,
        logical_planning_ns,
        physical_plan_text,
        physical_plan_string_ns,
        physical_planning_ns: nanos(start.elapsed()),
        physical_planning_count: 1,
        rolling_audit,
        audit_ns,
        input_adapter_ns: 0,
        table_register_ns: 0,
        _registrations: None,
    })
}

async fn collect_owned(
    stream: &mut SendableRecordBatchStream,
    schema: SchemaRef,
    execution_start: Instant,
) -> std::result::Result<CollectedOutput, Error> {
    let start = Instant::now();
    let first = stream.next().await.transpose()?;
    let first_ns = nanos(execution_start.elapsed());
    let mut batches = Vec::new();
    let mut rows = 0;
    let mut bytes = 0;
    if let Some(batch) = first {
        append_batch(&mut batches, &mut rows, &mut bytes, batch)?;
    }
    let remaining = Instant::now();
    while let Some(batch) = stream.next().await {
        append_batch(&mut batches, &mut rows, &mut bytes, batch?)?;
    }
    let remaining_ns = nanos(remaining.elapsed());
    let collect_ns = nanos(start.elapsed());
    let wrap_start = Instant::now();
    if batches.is_empty() {
        batches.push(RecordBatch::new_empty(schema));
    }
    Ok(CollectedOutput {
        batches,
        execution_to_first_batch_ns: first_ns,
        execution_remaining_ns: remaining_ns,
        collect_ns,
        output_arrow_wrap_ns: nanos(wrap_start.elapsed()),
    })
}

fn append_batch(
    batches: &mut Vec<RecordBatch>,
    rows: &mut usize,
    bytes: &mut usize,
    batch: RecordBatch,
) -> Result<()> {
    push_bounded_sql_batch(
        batches,
        rows,
        bytes,
        batch,
        MAX_SQL_RESULT_ROWS,
        MAX_SQL_RESULT_BYTES,
    )
}

struct MetricInput<'a> {
    planned: PlannedQuery<'a>,
    collected: CollectedOutput,
    context_ns: u64,
    context_preexisting: bool,
    stream_ns: u64,
    envelope_ns: u64,
    input_rows: usize,
    entities: Option<usize>,
    source: &'static str,
    output_rows: usize,
}

fn record_metric(
    runtime: &DataFusionRuntime,
    metric: MetricInput<'_>,
    node_id: Option<&str>,
) -> Result<()> {
    let (stats, traverse_ns) =
        runtime.plan_statistics(metric.planned.physical_plan.as_ref(), metric.output_rows);
    let decision = runtime.decision()?;
    let PlannedQuery {
        logical_plan,
        logical_plan_string_ns,
        logical_planning_ns,
        physical_plan_text,
        physical_plan_string_ns,
        physical_planning_ns,
        physical_planning_count,
        rolling_audit,
        audit_ns,
        input_adapter_ns,
        table_register_ns,
        ..
    } = metric.planned;
    let CollectedOutput {
        execution_to_first_batch_ns,
        execution_remaining_ns,
        collect_ns,
        output_arrow_wrap_ns,
        ..
    } = metric.collected;
    runtime.metrics.lock().push(DataFusionQueryMetric {
        query_id: runtime.next_query.fetch_add(1, Ordering::Relaxed),
        node_id: node_id.map(str::to_owned),
        runtime_acquire_ns: runtime.runtime_acquire_ns,
        session_state_create_ns: metric.context_ns,
        input_adapter_ns,
        table_register_ns,
        sql_parse_ns: 0,
        logical_planning_ns,
        physical_planning_ns,
        physical_planning_count,
        planning_ns: logical_planning_ns.saturating_add(physical_planning_ns),
        stream_open_ns: metric.stream_ns,
        execution_to_first_batch_ns,
        execution_remaining_ns,
        execution_ns: metric.stream_ns.saturating_add(collect_ns),
        collect_ns,
        output_arrow_wrap_ns,
        audit_ns,
        metrics_traversal_ns: traverse_ns,
        logical_plan_string_ns,
        physical_plan_string_ns,
        batch_envelope_ns: metric.envelope_ns,
        run_result_ns: 0,
        physical_metric_count: stats.metric_count,
        output_partition_count: stats.partition_rows.len(),
        output_partition_rows: stats.partition_rows,
        window_partition_count: stats.window_partition_rows.len(),
        window_partition_rows: stats.window_partition_rows,
        spill_bytes: stats.spill_bytes,
        elapsed_compute_ns: stats.elapsed_compute_ns,
        window_compute_ns: stats.window_compute_ns,
        repartition_sort_compute_ns: stats.repartition_sort_compute_ns,
        window_operator_count: stats.window_operator_count,
        repartition_operator_count: stats.repartition_operator_count,
        sort_operator_count: stats.sort_operator_count,
        coalesce_operator_count: stats.coalesce_operator_count,
        output_rows: metric.output_rows,
        configured_batch_size: runtime.config.batch_size,
        parallelism_mode: runtime.config.parallelism_mode,
        configured_target_partitions: runtime.config.target_partitions,
        requested_target_partitions: decision.requested_partitions,
        effective_target_partitions: runtime.effective_target_partitions.load(Ordering::Acquire),
        available_parallelism: decision.available_parallelism,
        max_partitions: runtime.config.max_partitions,
        min_rows_per_partition: runtime.config.min_rows_per_partition,
        small_rows_threshold: runtime.config.small_rows_threshold,
        parallelism_decision_reused: metric.context_preexisting,
        decision_input_rows: decision.input_rows,
        decision_active_entities: decision.active_entities,
        decision_active_entities_source: decision.active_entities_source.into(),
        input_rows: metric.input_rows,
        active_entities: metric.entities,
        active_entities_source: metric.source.into(),
        partition_limit_reason: decision.limit_reason.into(),
        rolling_rewrite_enabled: runtime.config.enable_rolling_rewrite,
        diagnostics_collected: runtime.config.collect_diagnostics,
        rolling_candidate_windows: rolling_audit.candidate_windows,
        rolling_rewritten_windows: rolling_audit.rewritten_windows,
        rolling_fallback_reasons: rolling_audit.fallback_reasons,
        logical_plan,
        physical_plan: physical_plan_text,
    });
    Ok(())
}
