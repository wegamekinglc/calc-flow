use std::sync::Arc;
use tokio::sync::OwnedSemaphorePermit;

use super::{
    super::{RollingOperator, StateBudget},
    resources::{self, Credit},
};
use crate::{
    CalcFlowError, Result, StateSegment, StreamJobContext, array::RollingScope,
    state::NativeSegmentReader,
};

struct Read<L> {
    reader: NativeSegmentReader,
    leases: L,
    reservation: Option<Credit>,
    budget: StateBudget,
    committed: u64,
    node: String,
    identity: Option<Arc<crate::array::RollingWorkerIdentity>>,
    generation: Option<crate::CancellationToken>,
    job_id: u64,
}

impl RollingOperator {
    pub(crate) async fn native_checkpoint_guard(
        &self,
        node: &str,
        job: &StreamJobContext,
    ) -> Result<OwnedSemaphorePermit> {
        self.checkpoint_pending
            .as_ref()
            .expect("layout5 pending gate")
            .enter(job)
            .await
            .map_err(|error| pending_error(error, node))
    }

    pub(crate) async fn read_native_segment<L: Send + 'static>(
        &self,
        node: &str,
        reader: NativeSegmentReader,
        leases: L,
        job: &StreamJobContext,
    ) -> Result<(StateSegment, L)> {
        job.check_cancelled()?;
        let reservation = self.reserve_native_read(reader.byte_len(), node)?;
        let identity = self.native_read_identity(node)?;
        let read = Read {
            reader,
            leases,
            reservation,
            budget: self.state_budget,
            committed: self.state.charge.bytes,
            node: node.into(),
            identity,
            generation: self
                .indicator_resources
                .as_ref()
                .map(super::Resources::work_control),
            job_id: job.job_id(),
        };
        read.dispatch(self.native_checkpoint_workers().as_ref(), node, job)
            .await
    }

    fn native_read_identity(
        &self,
        node: &str,
    ) -> Result<Option<Arc<crate::array::RollingWorkerIdentity>>> {
        self.indicator_resources
            .as_ref()
            .map(|resources| {
                super::work_identity::reserve(resources, node, node.len(), self.state_budget)
                    .map_err(|error| admission_error(error, node))
            })
            .transpose()
    }

    pub(in super::super) fn try_native_checkpoint_guard(
        &self,
        node: &str,
    ) -> Result<Option<OwnedSemaphorePermit>> {
        self.checkpoint_pending
            .as_ref()
            .map(|pending| {
                pending
                    .try_enter()
                    .map_err(|error| pending_error(error, node))
            })
            .transpose()
    }

    pub(in super::super) fn checkpoint_identity(&self) -> &str {
        self.state.operator_id.as_deref().unwrap_or(&self.name)
    }

    fn reserve_native_read(&self, length: usize, node: &str) -> Result<Option<Credit>> {
        let bytes = read_charge(length, read_scratch(length), node)?;
        self.indicator_resources
            .as_ref()
            .map(|resources| {
                resources
                    .reserve(self.state.charge.bytes, bytes, self.state_budget, node)
                    .map_err(|error| admission_error(error, node))
            })
            .transpose()
    }
}

impl<L: Send + 'static> Read<L> {
    async fn dispatch(
        self,
        workers: &crate::array::RollingWorkers,
        node: &str,
        job: &StreamJobContext,
    ) -> Result<(StateSegment, L)> {
        let scope = RollingScope::stream(job);
        match self.identity.clone() {
            Some(identity) => {
                workers
                    .run_rolling_credited(scope, identity, move |check| self.run(check))
                    .await
            }
            None => {
                workers
                    .run_rolling_scoped(scope, node, move |check| self.run(check))
                    .await
            }
        }
    }
}

impl<L> Read<L> {
    fn run(mut self, check: &mut dyn FnMut() -> Result<()>) -> Result<(StateSegment, L)> {
        let generation = self.generation.take();
        let job_id = self.job_id;
        let mut checked = || {
            check()?;
            check_generation(generation.as_ref(), job_id)
        };
        self.run_checked(&mut checked)
    }

    fn run_checked(mut self, check: &mut dyn FnMut() -> Result<()>) -> Result<(StateSegment, L)> {
        let (bytes, scratch) = self.allocate(check)?;
        let segment = self.reader.read(bytes, scratch, check)?;
        check()?;
        let segment = match self.reservation {
            Some(credit) => credit
                .capture_segment(segment, self.committed, self.budget, &self.node)
                .map_err(|error| admission_error(error, &self.node))?,
            None => segment,
        };
        Ok((segment, self.leases))
    }

    fn allocate(&mut self, check: &mut dyn FnMut() -> Result<()>) -> Result<(Vec<u8>, Vec<u8>)> {
        check()?;
        let scratch = self.scratch(check)?;
        let bytes = self.encoded(scratch.capacity(), check)?;
        Ok((bytes, scratch))
    }

    fn scratch(&mut self, check: &mut dyn FnMut() -> Result<()>) -> Result<Vec<u8>> {
        let length = self.reader.byte_len().min(4096);
        let mut scratch = allocate(length, &self.node)?;
        self.reconcile(self.reader.byte_len(), scratch.capacity())?;
        scratch.resize(length, 0);
        check()?;
        Ok(scratch)
    }

    fn encoded(
        &mut self,
        scratch: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<u8>> {
        let bytes = allocate(self.reader.byte_len(), &self.node)?;
        self.reconcile(bytes.capacity(), scratch)?;
        check()?;
        Ok(bytes)
    }

    fn reconcile(&mut self, encoded: usize, scratch: usize) -> Result<()> {
        if let Some(credit) = &mut self.reservation {
            let bytes = read_charge(
                encoded,
                scratch.max(StateSegment::controlled_scratch_bytes()),
                &self.node,
            )?;
            credit
                .adjust(bytes, self.committed, self.budget, &self.node)
                .map_err(|error| admission_error(error, &self.node))?;
        }
        Ok(())
    }
}

fn check_generation(token: Option<&crate::CancellationToken>, job_id: u64) -> Result<()> {
    if token.is_some_and(crate::CancellationToken::is_cancelled) {
        return Err(CalcFlowError::Cancelled {
            run_id: job_id.to_string(),
        });
    }
    Ok(())
}

fn allocate(length: usize, node: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|error| mismatch(node, "encoded_bytes", &error.to_string()))?;
    Ok(bytes)
}
fn read_charge(encoded: usize, scratch: usize, node: &str) -> Result<u64> {
    let encoded = resources::add(resources::size(encoded, node)?, 64, node)?;
    resources::add(encoded, resources::size(scratch, node)?, node)
}
fn read_scratch(length: usize) -> usize {
    length
        .min(4096)
        .max(StateSegment::controlled_scratch_bytes())
}
fn pending_error(error: CalcFlowError, node: &str) -> CalcFlowError {
    match error {
        CalcFlowError::Operator { message, .. } => CalcFlowError::Operator {
            node_id: node.into(),
            message,
        },
        other => other,
    }
}
fn admission_error(error: CalcFlowError, node: &str) -> CalcFlowError {
    match error {
        CalcFlowError::Operator { message, .. } => mismatch(node, "max_state_bytes", &message),
        other => other,
    }
}
fn mismatch(node: &str, field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("rolling operator {node:?} state.{field}: {message}"),
    }
}
