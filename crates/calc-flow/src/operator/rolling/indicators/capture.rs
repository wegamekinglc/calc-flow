//! Prepared immutable payloads, independently of checkpoint epochs.

use std::sync::{Arc, Weak};

use tokio::sync::OwnedSemaphorePermit;

use super::{
    Resources,
    footprint::Footprint,
    ipc_encode_plan,
    resources::WorkCredit,
    state_record::{self, Inputs, PreparedRecord},
};
use crate::operator::rolling::{
    CompiledRollingSpec, RollingOperator, RollingStreamState, SchemaRef, StateBudget,
    StreamOperatorContext, operator_error,
};
use crate::{CancellationToken, Result, StateSegment, array::RollingScope};

mod encoding;
mod snapshot;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(in super::super) struct Capture {
    state: Weak<RollingStreamState>,
    segment: Option<StateSegment>,
    graph: String,
    node: Arc<crate::array::RollingWorkerIdentity>,
}

struct Work {
    source: Arc<RollingStreamState>,
    compiled: Arc<CompiledRollingSpec>,
    schema: SchemaRef,
    graph: String,
    node: Arc<crate::array::RollingWorkerIdentity>,
    budget: StateBudget,
    generation: CancellationToken,
    job_id: u64,
    resources: Resources,
    credit: WorkCredit,
    guard: OwnedSemaphorePermit,
}

struct Prepared {
    source: Arc<RollingStreamState>,
    segment: Option<StateSegment>,
    graph: String,
    node: Arc<crate::array::RollingWorkerIdentity>,
    generation: CancellationToken,
    job_id: u64,
    _credit: WorkCredit,
    _guard: OwnedSemaphorePermit,
}

impl Capture {
    fn is_current(&self, state: &Arc<RollingStreamState>, graph: &str, node: &str) -> bool {
        self.state.ptr_eq(&Arc::downgrade(state))
            && self.graph == graph
            && self.node.as_str() == node
    }
}

impl RollingOperator {
    pub(in super::super) async fn prepare_indicator_capture(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let guard = self
            .native_checkpoint_guard(context.operator_id(), context.job())
            .await?;
        self.observe_context(context)?;
        let source = self.indicator_state_packet().expect("indicator packet");
        if self.indicator_capture.as_ref().is_some_and(|capture| {
            capture.is_current(&source, context.job().fingerprint(), context.operator_id())
        }) {
            return context.check_cancelled();
        }
        let work = self.capture_work(source, guard, context)?;
        let prepared = self
            .native_checkpoint_workers()
            .run_rolling_credited(
                RollingScope::stream(context.job()),
                Arc::clone(&work.node),
                move |check| work.run(check),
            )
            .await?;
        self.publish_indicator_capture(prepared, context)
    }

    fn publish_indicator_capture(
        &mut self,
        prepared: Prepared,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()?;
        generation(&prepared.generation, prepared.job_id)?;
        self.indicator_capture = Some(Capture {
            state: Arc::downgrade(&prepared.source),
            segment: prepared.segment,
            graph: prepared.graph,
            node: prepared.node,
        });
        Ok(())
    }

    fn capture_work(
        &self,
        source: Arc<RollingStreamState>,
        guard: OwnedSemaphorePermit,
        context: &StreamOperatorContext<'_>,
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
            source,
            compiled: Arc::clone(&self.compiled),
            schema: Arc::clone(self.input_ports[0].schema().expect("compiled input")),
            graph: context.job().fingerprint().into(),
            node,
            budget: self.state_budget,
            generation: resources.work_control(),
            job_id: context.job().job_id(),
            resources: resources.clone(),
            credit: resources.begin_work(),
            guard,
        })
    }
}

impl Work {
    fn run(mut self, check: Check<'_>) -> Result<Prepared> {
        let token = self.generation.clone();
        let job = self.job_id;
        let segment = self.segment(&mut || {
            check()?;
            generation(&token, job)
        })?;
        Ok(Prepared {
            source: self.source,
            segment,
            graph: self.graph,
            node: self.node,
            generation: self.generation,
            job_id: self.job_id,
            _credit: self.credit,
            _guard: self.guard,
        })
    }

    fn segment(&mut self, check: Check<'_>) -> Result<Option<StateSegment>> {
        check()?;
        if self.source.buffer.is_empty() && self.source.histories.by_entity.is_empty() {
            return Ok(None);
        }
        let prepared = self.record(check)?;
        let bytes = self.encode(&prepared, check)?;
        drop(prepared);
        self.capture(bytes, check).map(Some)
    }

    fn encode(&mut self, prepared: &PreparedRecord, check: Check<'_>) -> Result<Vec<u8>> {
        let prefix =
            ipc_encode_plan::planning_workspace(&prepared.schema, &prepared.record, check)?;
        self.admit_record(prepared, prefix)?;
        let plan = ipc_encode_plan::plan(&prepared.schema, &prepared.record, check)?;
        self.admit_record(
            prepared,
            add(plan.encoded_quota, plan.scratch_bytes, &self.node)?,
        )?;
        encoding::write(prepared, plan.encoded_quota, &self.node, check)
    }

    fn admit_record(&mut self, prepared: &PreparedRecord, scratch: u64) -> Result<()> {
        self.admit(
            prepared.footprint.clone(),
            add(prepared.scratch_bytes, scratch, &self.node)?,
        )
    }

    fn record(&mut self, check: Check<'_>) -> Result<PreparedRecord> {
        let source = self.credit.source().clone();
        let inputs = Inputs {
            state: &self.source,
            source_footprint: &source,
            input_schema: &self.schema,
            compiled: &self.compiled,
            fingerprint: &self.graph,
            node: &self.node,
        };
        let credit = &mut self.credit;
        state_record::prepare(
            inputs,
            &mut |quote, scratch| credit.adjust(quote, scratch, self.budget, &self.node),
            check,
        )
    }

    fn capture(&mut self, bytes: Vec<u8>, check: Check<'_>) -> Result<StateSegment> {
        self.admit_hash(bytes.capacity())?;
        let segment = StateSegment::new_controlled(bytes, check)?;
        let mut quote = Footprint::default();
        quote.segment(&segment, &self.node)?;
        self.admit(quote, 0)?;
        let segment = self.resources.retain(segment, 0, self.budget, &self.node)?;
        check()?;
        Ok(segment)
    }

    fn admit_hash(&mut self, capacity: usize) -> Result<()> {
        let payload = u64::try_from(capacity).map_err(|_| invalid(&self.node))?;
        let scratch = u64::try_from(StateSegment::controlled_scratch_bytes())
            .map_err(|_| invalid(&self.node))?;
        self.admit(
            Footprint::default(),
            add(payload, add(64, scratch, &self.node)?, &self.node)?,
        )
    }

    fn admit(&mut self, quote: Footprint, scratch: u64) -> Result<()> {
        self.credit.adjust(quote, scratch, self.budget, &self.node)
    }
}

fn generation(token: &CancellationToken, job: u64) -> Result<()> {
    if token.is_cancelled() {
        return Err(crate::CalcFlowError::Cancelled {
            run_id: job.to_string(),
        });
    }
    Ok(())
}
fn add(left: u64, right: u64, node: &str) -> Result<u64> {
    left.checked_add(right).ok_or_else(|| invalid(node))
}
fn invalid(node: &str) -> crate::CalcFlowError {
    operator_error(node, "state.capture_bytes overflowed")
}
