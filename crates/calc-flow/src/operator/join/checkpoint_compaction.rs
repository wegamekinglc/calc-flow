#[cfg(test)]
pub(super) struct TestGate {
    pub entered: tokio::sync::oneshot::Sender<(usize, std::thread::ThreadId)>,
    pub wait: std::sync::mpsc::Receiver<()>,
    pub failed: bool,
}

#[cfg(test)]
pub(super) struct TestRetirementGate {
    pub entered: tokio::sync::oneshot::Sender<()>,
    pub wait: std::sync::mpsc::Receiver<()>,
}
use super::{StoredRow, StreamJoinOperator, encode_side};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, OwnedCpuWork, cleanup_control_bytes,
};
use crate::{CalcFlowError, Result, StateSegment, StreamOperatorContext};
use datafusion::execution::memory_pool::MemoryReservation;
use std::{collections::BTreeMap, sync::Arc};

#[cfg(test)]
impl TestGate {
    fn wait_for_release(self, input: &[StoredRow], name: &str, stop: &GatherStop) -> Result<()> {
        self.entered
            .send((input.as_ptr() as usize, std::thread::current().id()))
            .map_err(|_| memory_error(name, "checkpoint test observer dropped"))?;
        self.wait
            .recv_timeout(std::time::Duration::from_secs(10))
            .map_err(|error| memory_error(name, &error.to_string()))?;
        stop.check()?;
        if self.failed {
            return Err(memory_error(name, "injected checkpoint failure"));
        }
        Ok(())
    }
}

struct InputOwners {
    left: Option<Arc<Vec<StoredRow>>>,
    right: Option<Arc<Vec<StoredRow>>>,
    released: Option<tokio::sync::oneshot::Sender<()>>,
    #[cfg(test)]
    retirement_gate: Option<TestRetirementGate>,
}

impl Drop for InputOwners {
    fn drop(&mut self) {
        drop(self.left.take());
        drop(self.right.take());
        if let Some(released) = self.released.take() {
            let _ = released.send(());
        }
        #[cfg(test)]
        if let Some(gate) = self.retirement_gate.take() {
            gate.entered.send(()).unwrap();
            gate.wait
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
        }
    }
}

struct BaseWork {
    inputs: InputOwners,
    name: String,
    #[cfg(test)]
    gate: Option<TestGate>,
}

#[cfg(test)]
pub(super) fn expected_attempt_control_bytes(name: &str) -> usize {
    2 * size_of::<BaseWork>()
        + name.len()
        + cleanup_control_bytes::<BTreeMap<&'static str, StateSegment>>()
        + size_of::<BTreeMap<&'static str, StateSegment>>()
        + 8_192
        + name.len()
        + 64
}

impl OwnedCpuWork for BaseWork {
    type Output = BTreeMap<&'static str, StateSegment>;

    fn control_bytes(&self) -> Result<usize> {
        size_of::<Self>()
            .checked_add(self.name.capacity())
            .and_then(|bytes| bytes.checked_add(cleanup_control_bytes::<Self::Output>()))
            .ok_or_else(|| memory_error(&self.name, "checkpoint control size overflow"))
    }

    fn run(self, stop: &GatherStop) -> Result<Self::Output> {
        stop.check()?;
        let left = self.inputs.left.as_ref().expect("left compaction input");
        let right = self.inputs.right.as_ref().expect("right compaction input");
        #[cfg(test)]
        if let Some(gate) = self.gate {
            gate.wait_for_release(left, &self.name, stop)?;
        }
        let left = encode_base_side(left, &self.name, "left", stop)?;
        let right = encode_base_side(right, &self.name, "right", stop)?;
        Ok(BTreeMap::from([("left", left), ("right", right)]))
    }
}

fn encode_base_side(
    rows: &[StoredRow],
    name: &str,
    side: &str,
    stop: &GatherStop,
) -> Result<StateSegment> {
    let encoded = encode_side(rows, name, side, &|| stop.check())?;
    let segment = StateSegment::new(encoded);
    stop.check()?;
    Ok(segment)
}

impl StreamJoinOperator {
    pub(super) async fn await_compaction_release(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.await_snapshot_release(context).await?;
        if let Some(cleanup) = &self.compaction_cleanup {
            cleanup.wait(&GatherStop::from_job(context.job())).await?;
        }
        self.compaction_cleanup = None;
        context.check_cancelled()
    }

    async fn await_snapshot_release(&mut self, context: &StreamOperatorContext<'_>) -> Result<()> {
        let Some(released) = self.compaction_release.as_mut() else {
            return Ok(());
        };
        loop {
            context.check_cancelled()?;
            let deadline = async {
                match context.job().deadline() {
                    Some(deadline) => {
                        let delay = (*deadline - chrono::Utc::now())
                            .to_std()
                            .unwrap_or_default();
                        tokio::time::sleep(delay).await;
                    }
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                biased;
                () = context.job().cancellation().cancelled() => context.check_cancelled()?,
                () = deadline => context.check_cancelled()?,
                _ = &mut *released => break,
            }
        }
        self.compaction_release = None;
        context.check_cancelled()
    }

    pub(super) async fn prepare_compaction(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()?;
        self.await_compaction_release(context).await?;
        if !self.state.deltas.needs_compaction {
            return Ok(());
        }
        self.rebuild_compaction_base(context).await
    }

    async fn rebuild_compaction_base(&mut self, context: &StreamOperatorContext<'_>) -> Result<()> {
        let credit = self.reserve_compaction_workspace()?;
        let scope = context
            .gather_client(GatherOperatorId::new(Arc::from(self.name.as_str())))
            .scope()?;
        let retirement = context.job().gather_owner().retain_retirement()?;
        let work = self.compaction_work();
        let ticket = scope
            .submit_observed_work(
                work,
                credit,
                GatherStop::from_job(context.job()),
                retirement,
                &mut self.compaction_cleanup,
            )
            .await
            .map_err(|failure| admission_error(&self.name, failure))?;
        let prepared = ticket.finish().await?;
        self.await_snapshot_release(context).await?;
        prepared.install(|base| self.install_compaction_base(base, context))?;
        self.compaction_cleanup = None;
        Ok(())
    }

    fn reserve_compaction_workspace(&mut self) -> Result<MemoryReservation> {
        let workspace = self.compaction_workspace()?;
        let credit = self.runtime.runtime()?.incremental_reservation(&self.name);
        credit
            .try_grow(workspace)
            .map_err(|error| memory_error(&self.name, &error.to_string()))?;
        Ok(credit)
    }

    fn compaction_work(&mut self) -> BaseWork {
        let (released, receiver) = tokio::sync::oneshot::channel();
        self.compaction_release = Some(receiver);
        BaseWork {
            inputs: InputOwners {
                left: Some(Arc::clone(&self.state.left.0)),
                right: Some(Arc::clone(&self.state.right.0)),
                released: Some(released),
                #[cfg(test)]
                retirement_gate: self.checkpoint_retirement_gate.take().map(|gate| {
                    gate.into_inner()
                        .expect("exclusive checkpoint retirement gate")
                }),
            },
            name: self.name.clone(),
            #[cfg(test)]
            gate: self
                .checkpoint_gate
                .take()
                .map(|gate| gate.into_inner().expect("exclusive checkpoint test gate")),
        }
    }

    fn install_compaction_base(
        &mut self,
        base: BTreeMap<&'static str, StateSegment>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()?;
        self.state.deltas.base = base;
        self.state.deltas.segments.clear();
        self.state.deltas.pending.clear();
        self.state.deltas.segments_since_base = 0;
        self.state.deltas.needs_compaction = false;
        Ok(())
    }

    fn compaction_workspace(&self) -> Result<usize> {
        let logical = self
            .state
            .metrics
            .left
            .retained_bytes
            .checked_add(self.state.metrics.right.retained_bytes)
            .and_then(|bytes| usize::try_from(bytes).ok());
        let descriptors = self
            .state
            .left
            .len()
            .checked_add(self.state.right.len())
            .and_then(|rows| rows.checked_mul(2 * size_of::<&StoredRow>()));
        logical
            .zip(descriptors)
            .and_then(|(logical, descriptors)| logical.checked_add(descriptors))
            .ok_or_else(|| memory_error(&self.name, "checkpoint workspace size overflow"))
    }
}

fn admission_error(name: &str, failure: AdmissionFailure) -> CalcFlowError {
    match failure {
        AdmissionFailure::Budget { stage, source } => {
            memory_error(name, &format!("checkpoint {stage} admission: {source}"))
        }
        AdmissionFailure::Runtime(error) => error,
    }
}

fn memory_error(name: &str, message: &str) -> CalcFlowError {
    CalcFlowError::DataFusion {
        node_id: Some(name.to_owned()),
        message: message.to_owned(),
    }
}
