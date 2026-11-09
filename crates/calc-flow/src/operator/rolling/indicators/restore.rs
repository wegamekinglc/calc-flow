//! Transactional preparation and publication of indicator checkpoint restoration.

use super::{
    Resources, StateOwner,
    footprint::Footprint,
    resources::{Commit, WorkCredit},
};
use crate::operator::rolling::{
    CompiledRollingSpec, RollingOperator, RollingSnapshotMetadata, RollingStreamState, SchemaRef,
    StateBudget, StreamOperatorContext,
};
use crate::{
    CalcFlowError, CancellationToken, OperatorStateSnapshot, Result, StateSegment,
    array::{RollingScope, RollingWorkerIdentity},
};
use std::{alloc::Layout, sync::Arc};
use tokio::sync::OwnedSemaphorePermit;

mod legacy;
pub(super) mod metadata;
mod outer;
mod work;
type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
struct Work {
    source: Arc<RollingStreamState>,
    schema: SchemaRef,
    compiled: Arc<CompiledRollingSpec>,
    metadata: RollingSnapshotMetadata,
    metadata_bytes: u64,
    node: Arc<RollingWorkerIdentity>,
    segment: Option<StateSegment>,
    budget: StateBudget,
    late_policy: crate::LatePolicySpec,
    generation: CancellationToken,
    job_id: u64,
    credit: WorkCredit,
    guard: OwnedSemaphorePermit,
}
struct Prepared {
    source: Arc<RollingStreamState>,
    candidate: Arc<RollingStreamState>,
    commit: Commit,
    epoch: crate::Epoch,
    generation: CancellationToken,
    job_id: u64,
    node: Arc<RollingWorkerIdentity>,
    credit: WorkCredit,
    _guard: OwnedSemaphorePermit,
}
impl RollingOperator {
    pub(in super::super) fn restore_indicators(
        &mut self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<()> {
        if empty(snapshot) {
            return crate::StreamOperator::reset(self);
        }
        let guard = self
            .try_native_checkpoint_guard(self.checkpoint_identity())?
            .expect("indicator gate");
        let work = self.restore_work(snapshot, guard, None)?;
        let prepared = work.run(&mut || Ok(()))?;
        generation(&prepared.generation, prepared.job_id)?;
        self.publish_restore(prepared);
        Ok(())
    }
    pub(in super::super) async fn restore_indicators_async(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()?;
        if empty(snapshot) {
            return crate::StreamOperator::reset(self);
        }
        let guard = self
            .native_checkpoint_guard(context.operator_id(), context.job())
            .await?;
        let work = self.restore_work(snapshot, guard, Some(context))?;
        let identity = Arc::clone(&work.node);
        let prepared = self
            .native_checkpoint_workers()
            .run_rolling_credited(
                RollingScope::stream(context.job()),
                identity,
                move |check| work.run(check),
            )
            .await?;
        ready(&prepared, context)?;
        self.publish_restore(prepared);
        Ok(())
    }
    fn restore_work(
        &self,
        snapshot: &OperatorStateSnapshot,
        guard: OwnedSemaphorePermit,
        context: Option<&StreamOperatorContext<'_>>,
    ) -> Result<Work> {
        Preparation::new(self, snapshot, context, guard).prepare()
    }
    fn publish_restore(&mut self, prepared: Prepared) {
        let Prepared {
            source,
            candidate,
            commit,
            epoch,
            generation: _,
            job_id: _,
            node,
            credit,
            _guard,
        } = prepared;
        self.indicator_capture = None;
        self.indicator_late_failed = false;
        self.indicator_checkpoint_epoch = Some(epoch);
        self.state = StateOwner::Shared(candidate);
        drop(source);
        drop(node);
        credit.commit(commit);
    }
}
struct Preparation<'a, 'job> {
    operator: &'a RollingOperator,
    snapshot: &'a OperatorStateSnapshot,
    context: Option<&'a StreamOperatorContext<'job>>,
    resources: &'a Resources,
    credit: WorkCredit,
    guard: OwnedSemaphorePermit,
    generation: CancellationToken,
    job_id: u64,
    node: &'a str,
}
impl<'a, 'job> Preparation<'a, 'job> {
    fn new(
        operator: &'a RollingOperator,
        snapshot: &'a OperatorStateSnapshot,
        context: Option<&'a StreamOperatorContext<'job>>,
        guard: OwnedSemaphorePermit,
    ) -> Self {
        let resources = operator
            .indicator_resources
            .as_ref()
            .expect("indicator resources");
        Self {
            operator,
            snapshot,
            context,
            resources,
            credit: resources.begin_work(),
            guard,
            generation: resources.work_control(),
            job_id: context.map_or(0, |context| context.job().job_id()),
            node: context.map_or_else(
                || operator.checkpoint_identity(),
                StreamOperatorContext::operator_id,
            ),
        }
    }
    fn prepare(mut self) -> Result<Work> {
        let token = self.generation.clone();
        let job = self.job_id;
        let context = self.context;
        let mut check = || capture_check(context, &token, job);
        let quote = metadata::quote(self.snapshot, &mut check)?;
        let (metadata, segment) = self.capture(quote.peak)?;
        check()?;
        self.reserve(quote.retained)?;
        let identity =
            super::work_identity::reserve(self.resources, self.node, 0, self.operator.state_budget)
                .map_err(|error| budget_error(error, self.node))?;
        Ok(self.into_work(metadata, segment, quote.retained, identity))
    }
    fn capture(&mut self, peak: u64) -> Result<(RollingSnapshotMetadata, Option<StateSegment>)> {
        self.reserve(peak)?;
        let metadata = self.metadata()?;
        let segment = retain_segment(
            self.snapshot,
            self.resources,
            self.operator.state_budget,
            self.node,
        )?;
        Ok((metadata, segment))
    }
    fn metadata(&self) -> Result<RollingSnapshotMetadata> {
        let metadata = metadata::parse(&self.snapshot.inline_metadata)?;
        metadata::validate(&metadata, &self.operator.compiled, self.snapshot)?;
        identity(&metadata, self.context)?;
        Ok(metadata)
    }
    fn reserve(&mut self, bytes: u64) -> Result<()> {
        reserve(
            &mut self.credit,
            bytes,
            self.operator.state_budget,
            self.node,
        )
    }
    fn into_work(
        self,
        metadata: RollingSnapshotMetadata,
        segment: Option<StateSegment>,
        metadata_bytes: u64,
        identity: Arc<RollingWorkerIdentity>,
    ) -> Work {
        Work {
            source: self
                .operator
                .indicator_state_packet()
                .expect("indicator packet"),
            schema: Arc::clone(
                self.operator.input_ports[0]
                    .schema()
                    .expect("compiled schema"),
            ),
            compiled: Arc::clone(&self.operator.compiled),
            metadata,
            metadata_bytes,
            node: identity,
            segment,
            budget: self.operator.state_budget,
            late_policy: self.operator.spec.late_policy,
            generation: self.generation,
            job_id: self.job_id,
            credit: self.credit,
            guard: self.guard,
        }
    }
}
fn capture_check(
    context: Option<&StreamOperatorContext<'_>>,
    token: &CancellationToken,
    job: u64,
) -> Result<()> {
    if let Some(context) = context {
        context.check_cancelled()?;
    }
    generation_check(token, job)
}

fn retain_segment(
    snapshot: &OperatorStateSnapshot,
    resources: &Resources,
    budget: StateBudget,
    node: &str,
) -> Result<Option<StateSegment>> {
    snapshot
        .segments
        .values()
        .next()
        .map(|segment| resources.retain(segment.clone(), 0, budget, node))
        .transpose()
}
fn identity(
    metadata: &RollingSnapshotMetadata,
    context: Option<&StreamOperatorContext<'_>>,
) -> Result<()> {
    if let Some(context) = context
        && (metadata.operator_id.as_deref() != Some(context.operator_id())
            || metadata.pipeline_fingerprint.as_deref() != Some(context.job().fingerprint()))
    {
        return Err(mismatch("identity", "graph or operator differs"));
    }
    Ok(())
}
fn reserve(credit: &mut WorkCredit, bytes: u64, budget: StateBudget, node: &str) -> Result<()> {
    credit
        .adjust(Footprint::default(), bytes, budget, node)
        .map_err(|error| budget_error(error, node))
}
fn ready(prepared: &Prepared, context: &StreamOperatorContext<'_>) -> Result<()> {
    context.check_cancelled()?;
    generation_check(&prepared.generation, prepared.job_id)
}

fn generation(token: &CancellationToken, job_id: u64) -> Result<()> {
    generation_check(token, job_id)
}
fn generation_check(token: &CancellationToken, job_id: u64) -> Result<()> {
    if token.is_cancelled() {
        Err(CalcFlowError::Cancelled {
            run_id: job_id.to_string(),
        })
    } else {
        Ok(())
    }
}
fn budget_error(error: CalcFlowError, node: &str) -> CalcFlowError {
    match error {
        CalcFlowError::Operator { .. } => CalcFlowError::CheckpointMismatch {
            message: format!(
                "rolling operator {node:?} state.max_bytes exceeds the configured budget"
            ),
        },
        error => error,
    }
}
fn mismatch(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("rolling state.{field}: {message}"),
    }
}
fn empty(snapshot: &OperatorStateSnapshot) -> bool {
    snapshot.inline_metadata.is_empty() && snapshot.segments.is_empty()
}
fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| mismatch("bytes", "overflowed u64"))
}
fn length(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| mismatch("bytes", "exceeds u64"))
}
fn layout<T>(count: usize) -> Result<u64> {
    length(
        Layout::array::<T>(count)
            .map_err(|_| mismatch("bytes", "allocation is not representable"))?
            .size(),
    )
}
#[cfg(test)]
mod admission_tests;
#[cfg(test)]
mod extrema_tests;
#[cfg(test)]
mod metadata_tests;
#[cfg(test)]
mod precopy_tests;
#[cfg(test)]
mod preload_tests;
#[cfg(test)]
mod state_tests;
