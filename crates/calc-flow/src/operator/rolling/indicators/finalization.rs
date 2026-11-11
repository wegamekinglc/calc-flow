//! One owned preparation for a complete watermark or EOF callback.

use std::sync::Arc;

use datafusion::arrow::{array::ArrayRef, record_batch::RecordBatch};
use tokio::sync::OwnedSemaphorePermit;

use super::{
    StateOwner, chunk_requirements, controlled,
    footprint::Footprint,
    footprint_state, math_requirements, prefix,
    resources::{Commit, WorkCredit},
    scalar_output_plan,
};
use crate::operator::rolling::{
    BufferedRow, CompiledRollingSpec, EntityRollingState, EventTime, KeyValue, Result,
    RollingMetricsRecorder, RollingOperator, RollingStreamState, ScalarValue, SchemaRef,
    StateBudget, StreamCollector, StreamOperatorContext, closing_coordinate,
    compute_output_cells_checked, operator_error,
};
use crate::{Batch, CalcFlowError, CancellationToken, EdgeBudget, array::RollingScope};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

struct Work {
    source: Arc<RollingStreamState>,
    compiled: Arc<CompiledRollingSpec>,
    schema: SchemaRef,
    node: Arc<crate::array::RollingWorkerIdentity>,
    fingerprint: String,
    observer: Option<RollingMetricsRecorder>,
    watermark: Option<EventTime>,
    lateness: u64,
    budget: StateBudget,
    edge: EdgeBudget,
    generation: CancellationToken,
    job_id: u64,
    quote: Footprint,
    scratch: u64,
    credit: WorkCredit,
    guard: OwnedSemaphorePermit,
}
struct Evaluated {
    state: RollingStreamState,
    rows: Vec<BufferedRow>,
    columns: Vec<Vec<Option<ScalarValue>>>,
}
struct Output {
    record: RecordBatch,
    footprint: Footprint,
    residual: u64,
}
struct Prepared {
    _identity: Arc<crate::array::RollingWorkerIdentity>,
    state: Arc<RollingStreamState>,
    batches: Vec<Batch>,
    commit: Commit,
    generation: CancellationToken,
    job_id: u64,
    credit: WorkCredit,
    _guard: OwnedSemaphorePermit,
}

impl RollingOperator {
    pub(in super::super) async fn finalize_indicator_worker(
        &mut self,
        watermark: Option<EventTime>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let guard = self
            .native_checkpoint_guard(context.operator_id(), context.job())
            .await?;
        let work = self.finalization_work(watermark, context, guard)?;
        let prepared = self
            .native_checkpoint_workers()
            .run_rolling_credited(
                RollingScope::stream(context.job()),
                Arc::clone(&work.node),
                move |check| work.run(check),
            )
            .await?;
        self.publish_indicator_finalization(prepared, context, output)
            .await
    }
    async fn publish_indicator_finalization(
        &mut self,
        prepared: Prepared,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        generation(&prepared.generation, prepared.job_id)?;
        for batch in prepared.batches {
            output.emit("output", batch).await?;
        }
        self.indicator_capture = None;
        self.state = StateOwner::Shared(prepared.state);
        prepared.credit.commit(prepared.commit);
        Ok(())
    }
    fn finalization_work(
        &self,
        watermark: Option<EventTime>,
        context: &StreamOperatorContext<'_>,
        guard: OwnedSemaphorePermit,
    ) -> Result<Work> {
        let resources = self
            .indicator_resources
            .as_ref()
            .expect("indicator resources");
        let node = super::work_identity::reserve(
            resources,
            context.operator_id(),
            context.job().fingerprint().len(),
            self.state_budget,
        )?;
        Ok(Work {
            source: self.indicator_state_packet().expect("indicator packet"),
            compiled: Arc::clone(&self.compiled),
            schema: Arc::clone(self.output_ports[0].schema().expect("compiled output")),
            node,
            fingerprint: context.job().fingerprint().into(),
            observer: context.rolling_metrics().cloned(),
            watermark,
            lateness: self.spec.allowed_lateness_micros,
            budget: self.state_budget,
            edge: context.output_budget(),
            generation: resources.work_control(),
            job_id: context.job().job_id(),
            quote: Footprint::default(),
            scratch: 0,
            credit: resources.begin_work(),
            guard,
        })
    }
}
impl Work {
    fn run(self, check: Check<'_>) -> Result<Prepared> {
        let token = self.generation.clone();
        let id = self.job_id;
        self.run_checked(&mut || {
            check()?;
            generation(&token, id)
        })
    }
    fn run_checked(mut self, check: Check<'_>) -> Result<Prepared> {
        let evaluated = self.evaluate(check)?;
        let output = self.materialize(&evaluated, check)?;
        let (batches, chunk_bytes) =
            self.chunk(&output.record, evaluated.state.next_output_sequence, check)?;
        self.finish_preparation(evaluated, output, batches, chunk_bytes, check)
    }
    fn finish_preparation(
        mut self,
        evaluated: Evaluated,
        output: Output,
        batches: Vec<Batch>,
        chunk_bytes: u64,
        check: Check<'_>,
    ) -> Result<Prepared> {
        let (state, footprint) = self.finish_state(evaluated, batches.len(), check)?;
        drop(output.record);
        let reserved = math_requirements::bytes::add(chunk_bytes, output.residual)?;
        let commit = self.finish_credit(&state, footprint, &output.footprint, reserved, check)?;
        check()?;
        Ok(Prepared {
            _identity: self.node,
            state,
            batches,
            commit,
            generation: self.generation,
            job_id: self.job_id,
            credit: self.credit,
            _guard: self.guard,
        })
    }
    fn evaluate(&mut self, check: Check<'_>) -> Result<Evaluated> {
        let count = self.preflight(check)?;
        let mut state = controlled::clone_state(&self.source, check)?;
        let rows = self.take_rows(&mut state, count, check)?;
        prefix::stream(
            &rows,
            &self.source,
            &self.compiled,
            self.budget,
            &self.node,
            check,
        )?;
        let computed = compute_output_cells_checked(
            &rows,
            &state.histories,
            &self.compiled,
            &self.node,
            self.observer.as_ref(),
            None,
            check,
        )?;
        state.histories.apply(computed.touched);
        self.promote_cells(&computed.columns, check)?;
        Ok(Evaluated {
            state,
            rows,
            columns: computed.columns,
        })
    }
    fn preflight(&mut self, check: Check<'_>) -> Result<usize> {
        validate_watermark(self.source.last_input_watermark, self.watermark, &self.node)?;
        let count = self.closing_count(check)?;
        self.quote = self.credit.source().clone_bound(&self.node)?;
        self.scratch = self.preparation_scratch(count, check)?;
        self.reserve(0)?;
        Ok(count)
    }
    fn preparation_scratch(&self, count: usize, check: Check<'_>) -> Result<u64> {
        use math_requirements::bytes::{slots, total, tree_nodes};
        total([
            math_requirements::plan(&self.source, &self.compiled, count, check),
            prefix::stream_scratch(
                &self.source,
                &self.compiled,
                count,
                self.budget,
                &self.node,
                check,
            ),
            slots::<BufferedRow>(count),
            tree_nodes::<Vec<Option<KeyValue>>, EntityRollingState>(count),
            slots::<ArrayRef>(self.schema.fields().len()),
            footprint_state::identity_bytes(&self.fingerprint, &self.node),
        ])
    }
    fn closing_count(&self, check: Check<'_>) -> Result<usize> {
        let mut count = 0_usize;
        for identity in self.source.buffer.keys() {
            check()?;
            if closes(
                identity.event_time,
                self.watermark,
                self.lateness,
                &self.node,
            )? {
                count = count
                    .checked_add(1)
                    .ok_or_else(|| operator_error(&self.node, "closing row count overflowed"))?;
            }
        }
        Ok(count)
    }
    fn take_rows(
        &self,
        state: &mut RollingStreamState,
        count: usize,
        check: Check<'_>,
    ) -> Result<Vec<BufferedRow>> {
        let mut rows = Vec::with_capacity(count);
        while let Some(entry) = state.buffer.first_entry() {
            check()?;
            if !closes(
                entry.key().event_time,
                self.watermark,
                self.lateness,
                &self.node,
            )? {
                break;
            }
            rows.push(entry.remove_entry().1);
        }
        check()?;
        Ok(rows)
    }
    fn promote_cells(
        &mut self,
        columns: &[Vec<Option<ScalarValue>>],
        check: Check<'_>,
    ) -> Result<()> {
        let mut footprint = Footprint::default();
        for value in columns.iter().flatten().flatten() {
            check()?;
            footprint.scalar(value, &self.node)?;
        }
        let extra = footprint.incremental_bytes(&self.quote, &self.node)?;
        self.scratch = self.scratch.checked_sub(extra).ok_or_else(|| {
            operator_error(
                &self.node,
                "state.math_bytes exceeded the admitted component",
            )
        })?;
        self.quote.merge(&footprint, &self.node)?;
        self.reserve(0)
    }
    fn reserve(&mut self, extra: u64) -> Result<()> {
        let bytes = math_requirements::bytes::add(self.scratch, extra)?;
        self.credit
            .adjust(self.quote.clone(), bytes, self.budget, &self.node)
    }
    fn materialize(&mut self, evaluated: &Evaluated, check: Check<'_>) -> Result<Output> {
        let plan = self.output_plan(evaluated, check)?;
        let (columns, footprint) = self.output_columns(evaluated, plan.output_bytes, check)?;
        let residual = plan
            .output_bytes
            .checked_sub(footprint.incremental_bytes(&self.quote, &self.node)?)
            .ok_or_else(|| {
                operator_error(
                    &self.node,
                    "state.output_bytes exceeded its admitted component",
                )
            })?;
        let record = RecordBatch::try_new(Arc::clone(&self.schema), columns)
            .map_err(|_| operator_error(&self.node, "prepared rolling output schema is invalid"))?;
        self.adopt_output(&footprint, residual)?;
        Ok(Output {
            record,
            footprint,
            residual,
        })
    }
    fn output_plan(
        &mut self,
        evaluated: &Evaluated,
        check: Check<'_>,
    ) -> Result<scalar_output_plan::OutputPlan> {
        let count = evaluated.rows.len();
        let mut cell = |column, row| cell(evaluated, column, row);
        let prefix = scalar_output_plan::planning_workspace(&self.schema, count, &mut cell, check)?;
        self.reserve(prefix)?;
        let plan = scalar_output_plan::plan(&self.schema, count, &mut cell, check)?;
        self.reserve(math_requirements::bytes::add(
            plan.output_bytes,
            plan.scratch_bytes,
        )?)?;
        Ok(plan)
    }
    fn adopt_output(&mut self, footprint: &Footprint, residual: u64) -> Result<()> {
        self.quote.merge(footprint, &self.node)?;
        self.scratch = math_requirements::bytes::add(self.scratch, residual)?;
        self.reserve(0)
    }
    fn output_columns(
        &self,
        evaluated: &Evaluated,
        quota: u64,
        check: Check<'_>,
    ) -> Result<(Vec<ArrayRef>, Footprint)> {
        let mut columns = Vec::with_capacity(self.schema.fields().len());
        let mut footprint = Footprint::default();
        for (ordinal, field) in self.schema.fields().iter().enumerate() {
            check()?;
            let column = scalar_output_plan::encode_column(
                field.data_type(),
                evaluated.rows.len(),
                &mut |row| cell(evaluated, ordinal, row),
                check,
            )?;
            footprint.array_checked(&column, &self.node, check)?;
            self.verify_output_component(&footprint, quota)?;
            columns.push(column);
        }
        Ok((columns, footprint))
    }
    fn verify_output_component(&self, footprint: &Footprint, quota: u64) -> Result<()> {
        if footprint.incremental_bytes(&self.quote, &self.node)? > quota {
            return Err(operator_error(
                &self.node,
                "state.output_bytes exceeded its admitted component",
            ));
        }
        Ok(())
    }
    fn chunk(
        &mut self,
        record: &RecordBatch,
        sequence: u64,
        check: Check<'_>,
    ) -> Result<(Vec<Batch>, u64)> {
        use crate::operator::output_chunk::{OutputChunkErrors, chunk_output_record_checked};
        let plan = chunk_requirements::plan(record, self.edge, &self.node, check)?;
        self.reserve(math_requirements::bytes::add(
            plan.output_bytes,
            plan.scratch_bytes,
        )?)?;
        let batches = chunk_output_record_checked(
            record,
            &self.node,
            sequence,
            self.edge,
            OutputChunkErrors::ROLLING,
            check,
        )?;
        self.verify_chunk_component(&batches, plan.output_bytes, check)?;
        self.reserve(plan.output_bytes)?;
        Ok((batches, plan.output_bytes))
    }
    fn verify_chunk_component(
        &self,
        batches: &Vec<Batch>,
        quota: u64,
        check: Check<'_>,
    ) -> Result<()> {
        let footprint = self.chunk_footprint(batches, check)?;
        if footprint.incremental_bytes(&self.quote, &self.node)? > quota {
            return Err(operator_error(
                &self.node,
                "state.chunk_bytes exceeded its admitted component",
            ));
        }
        Ok(())
    }
    fn chunk_footprint(&self, batches: &Vec<Batch>, check: Check<'_>) -> Result<Footprint> {
        let mut footprint = Footprint::default();
        footprint.add_owned(
            math_requirements::bytes::slots::<Batch>(batches.capacity())?,
            &self.node,
        )?;
        for batch in batches {
            check()?;
            self.accumulate_batch(&mut footprint, batch, check)?;
        }
        Ok(footprint)
    }
    fn accumulate_batch(
        &self,
        footprint: &mut Footprint,
        batch: &Batch,
        check: Check<'_>,
    ) -> Result<()> {
        for record in batch.table_payload()?.batches() {
            for column in record.columns() {
                footprint.array_checked(column, &self.node, check)?;
            }
        }
        Ok(())
    }
    fn finish_state(
        &self,
        evaluated: Evaluated,
        chunks: usize,
        check: Check<'_>,
    ) -> Result<(Arc<RollingStreamState>, Footprint)> {
        let Evaluated {
            mut state,
            rows,
            columns,
        } = evaluated;
        drop(rows);
        drop(columns);
        let chunks = u64::try_from(chunks)
            .map_err(|_| operator_error(&self.node, "output chunk count overflowed"))?;
        state.next_output_sequence = state
            .next_output_sequence
            .checked_add(chunks)
            .ok_or_else(|| operator_error(&self.node, "output sequence overflowed"))?;
        state.last_input_watermark = self.watermark.or(state.last_input_watermark);
        state.ended = self.watermark.is_none();
        state.pipeline_fingerprint = Some(self.fingerprint.clone());
        state.operator_id = Some(self.node.as_str().to_owned());
        footprint_state::prepare_checked(state, &self.node, check)
    }
    fn finish_credit(
        &mut self,
        state: &Arc<RollingStreamState>,
        footprint: Footprint,
        output: &Footprint,
        chunk_bytes: u64,
        check: Check<'_>,
    ) -> Result<Commit> {
        if !self.budget.allows(state.charge.rows, state.charge.bytes) {
            return Err(operator_error(&self.node, "rolling state budget exceeded"));
        }
        let commit = Commit::new(footprint.clone());
        let mut live = footprint;
        live.merge(output, &self.node)?;
        drop(std::mem::take(&mut self.quote));
        self.credit
            .adjust(live, chunk_bytes, self.budget, &self.node)?;
        check()?;
        Ok(commit)
    }
}
fn cell(evaluated: &Evaluated, column: usize, row: usize) -> Option<&ScalarValue> {
    let inputs = evaluated.rows.first().map_or(0, |row| row.values.len());
    if column < inputs {
        evaluated.rows[row].values.get(column)
    } else {
        evaluated.columns.get(column - inputs)?.get(row)?.as_ref()
    }
}
fn closes(time: i64, watermark: Option<EventTime>, lateness: u64, node: &str) -> Result<bool> {
    match watermark {
        Some(watermark) => Ok(closing_coordinate(time, lateness, node)? <= watermark.as_micros()),
        None => Ok(true),
    }
}
fn generation(token: &CancellationToken, id: u64) -> Result<()> {
    if token.is_cancelled() {
        Err(CalcFlowError::Cancelled {
            run_id: id.to_string(),
        })
    } else {
        Ok(())
    }
}
fn validate_watermark(
    previous: Option<EventTime>,
    next: Option<EventTime>,
    node: &str,
) -> Result<()> {
    if previous
        .zip(next)
        .is_some_and(|(previous, next)| next <= previous)
    {
        return Err(operator_error(
            node,
            "input watermark did not advance strictly",
        ));
    }
    Ok(())
}
