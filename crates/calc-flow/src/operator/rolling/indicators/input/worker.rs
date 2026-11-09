use std::sync::Arc;
use tokio::sync::OwnedSemaphorePermit;

use super::super::{
    StateOwner, controlled, footprint_state,
    resources::{Commit, WorkCredit},
};
use super::classify::Classification;
use super::{Check, selection::Selection};
mod late;
mod selection;
use crate::operator::rolling::{
    BTreeMap, Batch, BufferedRow, CompiledRollingSpec, EventTime, LateMetricDelta, LatePolicySpec,
    Result, RollingMetricsRecorder, RollingOperator, RollingStreamState, RowIdentity, SchemaRef,
    StateBudget, StreamOperatorContext, accumulate_late_metrics, operator_error,
};
use crate::{CalcFlowError, CancellationToken, array::RollingScope};

struct Work {
    source: Arc<RollingStreamState>,
    input: Batch,
    compiled: Arc<CompiledRollingSpec>,
    schema: SchemaRef,
    node: Arc<crate::array::RollingWorkerIdentity>,
    fingerprint: String,
    observer: Option<RollingMetricsRecorder>,
    budget: StateBudget,
    generation: CancellationToken,
    job_id: u64,
    watermark: Option<EventTime>,
    policy: LatePolicySpec,
    lateness: u64,
    late_schema: Option<SchemaRef>,
    edge: crate::EdgeBudget,
    credit: WorkCredit,
    guard: OwnedSemaphorePermit,
}
struct Prepared {
    _identity: Arc<crate::array::RollingWorkerIdentity>,
    state: Arc<RollingStreamState>,
    commit: Commit,
    delta: LateMetricDelta,
    late: Vec<Batch>,
    generation: CancellationToken,
    job_id: u64,
    credit: WorkCredit,
    _guard: OwnedSemaphorePermit,
}
impl RollingOperator {
    pub(in super::super::super) async fn process_indicator_input_worker(
        &mut self,
        batch: Batch,
        watermark: Option<EventTime>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn crate::StreamCollector,
        guard: OwnedSemaphorePermit,
    ) -> Result<()> {
        let work = self.indicator_input_work(batch, watermark, context, guard)?;
        let prepared = self
            .native_checkpoint_workers()
            .run_rolling_credited(
                RollingScope::stream(context.job()),
                Arc::clone(&work.node),
                move |check| work.run(check),
            )
            .await?;
        let metrics = context.prepare_window_metrics(prepared.delta)?;
        context.check_cancelled()?;
        generation_check(&prepared.generation, prepared.job_id)?;
        self.publish_indicator_input(prepared, metrics, output)
            .await
    }
    async fn publish_indicator_input(
        &mut self,
        prepared: Prepared,
        metrics: crate::operator::PreparedLateMetrics,
        output: &mut dyn crate::StreamCollector,
    ) -> Result<()> {
        if !prepared.late.is_empty() {
            self.indicator_late_failed = true;
        }
        for batch in prepared.late {
            output.emit("late", batch).await?;
        }
        self.indicator_late_failed = false;
        self.indicator_capture = None;
        self.state = StateOwner::Shared(prepared.state);
        metrics.commit();
        prepared.credit.commit(prepared.commit);
        Ok(())
    }
    fn indicator_input_work(
        &self,
        input: Batch,
        watermark: Option<EventTime>,
        context: &StreamOperatorContext<'_>,
        guard: OwnedSemaphorePermit,
    ) -> Result<Work> {
        let resources = self
            .indicator_resources
            .as_ref()
            .expect("indicator resources")
            .clone();
        let node = super::super::work_identity::reserve(
            &resources,
            context.operator_id(),
            context.job().fingerprint().len(),
            self.state_budget,
        )?;
        let credit = resources.begin_work();
        Ok(Work {
            source: self.indicator_state_packet().expect("indicator packet"),
            input,
            compiled: Arc::clone(&self.compiled),
            schema: Arc::clone(self.input_ports[0].schema().expect("compiled input schema")),
            node,
            fingerprint: context.job().fingerprint().into(),
            observer: context.rolling_metrics().cloned(),
            budget: self.state_budget,
            generation: resources.work_control(),
            job_id: context.job().job_id(),
            watermark,
            policy: self.spec.late_policy,
            lateness: self.spec.allowed_lateness_micros,
            late_schema: self
                .output_ports
                .get(1)
                .and_then(|port| port.schema())
                .cloned(),
            edge: context.output_budget(),
            credit,
            guard,
        })
    }
}
impl Work {
    fn run(self, check: Check<'_>) -> Result<Prepared> {
        let generation = self.generation.clone();
        let job = self.job_id;
        let mut controlled = || {
            check()?;
            generation_check(&generation, job)
        };
        self.run_checked(&mut controlled)
    }
    fn run_checked(mut self, check: Check<'_>) -> Result<Prepared> {
        let selection = self.preflight(check)?;
        let (mut state, delta) = self.read_candidate(&selection, check)?;
        drop(selection);
        let late = self.prepare_late(&mut state, check)?;
        let (state, commit) = self.adopt_candidate(state, &late, check)?;
        check()?;
        Ok(Prepared {
            _identity: self.node,
            state,
            commit,
            delta,
            late: late.batches,
            generation: self.generation,
            job_id: self.job_id,
            credit: self.credit,
            _guard: self.guard,
        })
    }
    fn read_candidate(
        &self,
        selection: &Selection,
        check: Check<'_>,
    ) -> Result<(RollingStreamState, LateMetricDelta)> {
        let (accepted, delta) = self.classify(selection, check)?;
        let state = self.candidate(accepted, delta, check)?;
        Ok((state, delta))
    }
    fn adopt_candidate(
        &mut self,
        state: RollingStreamState,
        late: &late::Prepared,
        check: Check<'_>,
    ) -> Result<(Arc<RollingStreamState>, Commit)> {
        let (state, footprint) = footprint_state::prepare_checked(state, &self.node, check)?;
        if !self.budget.allows(state.charge.rows, state.charge.bytes) {
            return Err(operator_error(&self.node, "rolling state budget exceeded"));
        }
        let commit = Commit::new(footprint.clone());
        let mut live = footprint;
        live.merge(&late.footprint, &self.node)?;
        self.credit
            .adjust(live, late.reserved, self.budget, &self.node)?;
        Ok((state, commit))
    }
    fn classify(
        &self,
        selection: &Selection,
        check: Check<'_>,
    ) -> Result<(BTreeMap<RowIdentity, BufferedRow>, LateMetricDelta)> {
        Classification {
            state: &self.source,
            compiled: &self.compiled,
            node: &self.node,
            observer: self.observer.as_ref(),
        }
        .run(self.input.table_payload()?, selection, &self.schema, check)
    }
    fn candidate(
        &self,
        accepted: BTreeMap<RowIdentity, BufferedRow>,
        delta: LateMetricDelta,
        check: Check<'_>,
    ) -> Result<RollingStreamState> {
        let mut state = controlled::clone_state(&self.source, check)?;
        state.buffer.extend(accepted);
        state.metrics = accumulate_late_metrics(state.metrics, delta)?;
        state.pipeline_fingerprint = Some(self.fingerprint.clone());
        state.operator_id = Some(self.node.as_str().to_owned());
        Ok(state)
    }
}
fn generation_check(generation: &CancellationToken, job_id: u64) -> Result<()> {
    if generation.is_cancelled() {
        return Err(CalcFlowError::Cancelled {
            run_id: job_id.to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BatchMetadata, JsonMap, StreamJobContext};
    use datafusion::arrow::{
        array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };

    #[tokio::test]
    async fn test_selection_allocation_drops_before_actual_worker_refunds_its_credit() {
        let (prototype, input) = large_string_case();
        let mut spec = prototype.spec.clone();
        spec.late_policy = LatePolicySpec::Drop { metrics_version: 1 };
        let schema = Arc::clone(prototype.input_ports[0].schema().unwrap());
        let operator = RollingOperator::new("features", schema, spec).unwrap();
        let job = StreamJobContext::new(
            17,
            "a".repeat(64),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let watermark = Some(EventTime::from_micros(2));
        let context = StreamOperatorContext::new(&job, "features", watermark);
        let guard = operator
            .native_checkpoint_guard("features", &job)
            .await
            .unwrap();
        let base = operator
            .state
            .charge
            .bytes
            .checked_mul(2)
            .unwrap()
            .checked_add(footprint_state::identity_bytes(job.fingerprint(), "features").unwrap())
            .unwrap()
            .checked_add(
                super::super::super::work_identity::allocation_bytes(
                    "features",
                    job.fingerprint().len(),
                )
                .unwrap(),
            )
            .unwrap();
        let work = operator
            .indicator_input_work(input, watermark, &context, guard)
            .unwrap();
        let resources = operator.indicator_resources.as_ref().unwrap().clone();
        let (observed, early_refund, late_rows, empty, refunded) = operator
            .native_checkpoint_workers()
            .run_rolling_scoped(RollingScope::stream(&job), "features", move |check| {
                let tiny = StateBudget::new(1, base).unwrap();
                let mut observed = false;
                let mut early_refund = None;
                let prepared = work.run(&mut || {
                    check()?;
                    let live = Selection::live_test_bytes();
                    observed |= live > 0;
                    if live > 0 && resources.check(0, tiny, "features").is_ok() {
                        early_refund = Some(live);
                    }
                    Ok(())
                })?;
                Ok((
                    observed,
                    early_refund,
                    prepared.delta.late_rows,
                    prepared.state.buffer.is_empty(),
                    resources.check(0, tiny, "features").is_ok(),
                ))
            })
            .await
            .unwrap();
        assert!(observed, "the real selection must own nonzero capacity");
        assert_eq!(late_rows, 1);
        assert!(empty);
        assert!(
            refunded,
            "settled preparation must release the selection quota"
        );
        assert_eq!(early_refund, None, "live selection was refunded early");
    }

    #[tokio::test]
    async fn test_large_string_guard_is_measured_inside_the_actual_worker() {
        let (mut operator, input) = large_string_case();
        let keep_input = input.clone();
        operator
            .set_state_budget(StateBudget::new(8, 64 << 10).unwrap())
            .unwrap();
        let job = StreamJobContext::new(
            17,
            "a".repeat(64),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "features", None);
        let guard = operator
            .native_checkpoint_guard("features", &job)
            .await
            .unwrap();
        let work = operator
            .indicator_input_work(input, None, &context, guard)
            .unwrap();
        let (result, measured) = operator
            .native_checkpoint_workers()
            .run_rolling_scoped(RollingScope::stream(&job), "features", move |check| {
                let mut result = None;
                let measured = allocation_counter::measure(|| {
                    result = Some(work.run(check).map(|_| ()));
                });
                Ok((result.unwrap(), measured))
            })
            .await
            .unwrap();
        assert!(
            matches!(result, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "features")
        );
        assert!(
            measured.bytes_total < 64 << 10,
            "worker copied the 4MiB input before admission: {measured:?}"
        );
        drop(keep_input);
    }
    pub(super) fn large_string_case() -> (RollingOperator, Batch) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("entity", DataType::Utf8, false),
            Field::new("seq", DataType::UInt64, false),
            Field::new("price", DataType::Float64, false),
            Field::new("context", DataType::Utf8, false),
        ]));
        let spec = serde_json::from_value(serde_json::json!({
            "configuration_version":1, "state_layout_version":5, "partition_by":["entity"], "event_time":"ts", "sequence_by":["seq"],
            "outputs":[{"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":1}],
            "allowed_lateness_micros":0, "late_policy":{"kind":"error","scope":"envelope"}, "value_policy":"stateful_numeric_v1"
        })).unwrap();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(TimestampMicrosecondArray::from(vec![1]).with_timezone("UTC")),
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(UInt64Array::from(vec![1])),
            Arc::new(Float64Array::from(vec![1.0])),
            Arc::new(StringArray::from(vec!["x".repeat(4 << 20)])),
        ];
        let record = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
        (
            RollingOperator::new("features", schema, spec).unwrap(),
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
        )
    }
}

#[cfg(test)]
mod late_tests;
