use super::{
    JoinCheckpointMetadata, JoinMetrics, StreamJoinOperator, StreamJoinSpec,
    checkpoint_metadata_compatible, decode_join_metadata,
};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, AttemptCleanup, GatherOperatorId, GatherScope, GatherStop, ObservedTicket,
    OwnedCpuWork, RetirementGuard, cleanup_control_bytes,
};
use crate::{OperatorStateSnapshot, Result, runtime::streaming::StreamJobContext};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

mod inventory;
mod profile;
pub(super) mod schema;

#[cfg(test)]
mod tests;

pub(super) struct ValidatedMetadata {
    pub next_left_row_id: u64,
    pub next_right_row_id: u64,
    pub next_output_sequence: u64,
    pub metrics: JoinMetrics,
    pub ended: bool,
    pub epoch: u64,
}

impl From<JoinCheckpointMetadata> for ValidatedMetadata {
    fn from(metadata: JoinCheckpointMetadata) -> Self {
        Self {
            next_left_row_id: metadata.next_left_row_id,
            next_right_row_id: metadata.next_right_row_id,
            next_output_sequence: metadata.next_output_sequence,
            metrics: metadata.metrics,
            ended: metadata.ended,
            epoch: metadata.epoch,
        }
    }
}

type Decision = Option<ValidatedMetadata>;

struct MetadataWork {
    snapshot: OperatorStateSnapshot,
    expected: StreamJoinSpec,
    name: String,
    #[cfg(test)]
    hook: Option<super::MetadataTestHook>,
}

impl OwnedCpuWork for MetadataWork {
    type Output = Decision;

    fn control_bytes(&self) -> Result<usize> {
        inventory::caller_controls(&self.name)
            .and_then(|bytes| bytes.checked_add(cleanup_control_bytes::<Decision>()))
            .ok_or_else(|| crate::CalcFlowError::Internal {
                message: "metadata control overflow".into(),
            })
    }

    fn run(self, stop: &GatherStop) -> Result<Decision> {
        stop.check()?;
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(None, true);
        }
        let metadata = decode_join_metadata(&self.snapshot, &self.name);
        stop.check()?;
        Ok(match metadata {
            Ok(metadata) if checkpoint_metadata_compatible(&metadata, &self.expected) => {
                Some(metadata.into())
            }
            _ => None,
        })
    }
}

// The partial copied data is destroyed before its actual credit and retirement guard.
struct Construction {
    snapshot: OperatorStateSnapshot,
    expected: Option<StreamJoinSpec>,
    name: String,
    control: SubmissionControl,
    credit: Option<MemoryReservation>,
    retirement: Option<RetirementGuard>,
}

struct SubmissionControl {
    scope: Option<GatherScope>,
    stop: Option<GatherStop>,
    _credit: MemoryReservation,
}

impl Construction {
    async fn copy(
        &mut self,
        operator: &StreamJoinOperator,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
    ) -> Result<()> {
        copy_boundary(job).await?;
        self.name = String::from(operator.name.as_str());
        copy_boundary(job).await?;
        self.expected = Some(operator.spec.clone());
        for (key, value) in &snapshot.inline_metadata {
            copy_boundary(job).await?;
            self.snapshot
                .inline_metadata
                .insert(key.clone(), value.clone());
        }
        Ok(())
    }

    fn submit<'a>(
        &'a mut self,
        observer: &'a mut Option<AttemptCleanup>,
        #[cfg(test)] hook: Option<super::MetadataTestHook>,
    ) -> impl Future<Output = std::result::Result<ObservedTicket<Decision>, AdmissionFailure>> + 'a
    {
        let work = MetadataWork {
            snapshot: std::mem::take(&mut self.snapshot),
            expected: self
                .expected
                .take()
                .expect("completed metadata construction"),
            name: std::mem::take(&mut self.name),
            #[cfg(test)]
            hook,
        };
        self.control
            .scope
            .as_ref()
            .expect("prepared metadata scope")
            .submit_observed_work(
                work,
                self.credit.take().expect("paid metadata input"),
                self.control
                    .stop
                    .as_ref()
                    .expect("paid metadata stop")
                    .clone(),
                self.retirement.take().expect("registered metadata work"),
                observer,
            )
    }
}

async fn copy_boundary(job: &StreamJobContext) -> Result<()> {
    job.check_cancelled()?;
    tokio::task::yield_now().await;
    job.check_cancelled()
}

impl StreamJoinOperator {
    pub(crate) async fn restore_managed_metadata(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
        task: Option<crate::runtime::streaming::gather_work::TaskId>,
    ) -> Result<()> {
        job.check_cancelled()?;
        if self.try_restore_owned_utf8(snapshot, job, task).await? {
            return Ok(());
        }
        self.restore_managed_non_utf8(snapshot, job, task).await
    }

    async fn restore_managed_non_utf8(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
        task: Option<crate::runtime::streaming::gather_work::TaskId>,
    ) -> Result<()> {
        if self.try_restore_owned_segments(snapshot, job, task).await? {
            return Ok(());
        }
        if self.try_restore_owned_bases(snapshot, job, task).await? {
            return Ok(());
        }
        if self.try_restore_owned_schema(snapshot, job, task).await? {
            return Ok(());
        }
        self.restore_managed_metadata_only(snapshot, job, task)
            .await
    }

    async fn restore_managed_metadata_only(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
        task: Option<crate::runtime::streaming::gather_work::TaskId>,
    ) -> Result<()> {
        job.check_cancelled()?;
        let Some(mut construction) = self.metadata_construction(snapshot, job)? else {
            return self.restore_metadata_legacy(snapshot, job);
        };
        #[cfg(test)]
        if let Some(hook) = &self.metadata_test_hook {
            hook(construction.credit.as_ref(), false);
        }
        construction.copy(self, snapshot, job).await?;
        construction.control.scope = match job
            .gather_owner()
            .client(GatherOperatorId::new(Arc::from(self.name.as_str())).with_task(task))
            .scope()
        {
            Ok(scope) => Some(scope),
            Err(crate::CalcFlowError::Cancelled { .. }) => {
                job.check_cancelled()?;
                drop(construction);
                return self.restore_metadata_legacy(snapshot, job);
            }
            Err(error) => return Err(error),
        };
        let admission = construction
            .submit(
                &mut self.compaction_cleanup,
                #[cfg(test)]
                self.metadata_test_hook.clone(),
            )
            .await;
        self.finish_metadata_admission(
            admission,
            snapshot,
            job,
            construction
                .control
                .stop
                .as_ref()
                .expect("paid metadata stop"),
        )
        .await
    }

    fn metadata_construction(
        &self,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
    ) -> Result<Option<Construction>> {
        if self.compaction_release.is_some() || self.compaction_cleanup.is_some() {
            return Ok(None);
        }
        let Some(runtime) = self.runtime.runtime.as_ref() else {
            return Ok(None);
        };
        let Some(bytes) = profile::eligible(snapshot, &self.spec, &self.name)
            .and_then(|()| inventory::required(snapshot, &self.spec, &self.name))
        else {
            return Ok(None);
        };
        let credit = runtime.incremental_reservation("stream-join-metadata");
        if credit.try_grow(bytes).is_err() {
            job.check_cancelled()?;
            return Ok(None);
        }
        let retirement = match job.gather_owner().retain_retirement() {
            Ok(retirement) => retirement,
            Err(crate::CalcFlowError::Cancelled { .. }) => {
                job.check_cancelled()?;
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let controls = inventory::caller_controls(&self.name).expect("checked metadata controls");
        let mut construction = Construction {
            snapshot: OperatorStateSnapshot::default(),
            expected: None,
            name: String::new(),
            control: SubmissionControl {
                scope: None,
                stop: None,
                _credit: credit.split(controls),
            },
            credit: Some(credit),
            retirement: Some(retirement),
        };
        construction.control.stop = Some(GatherStop::from_job(job));
        Ok(Some(construction))
    }

    async fn finish_metadata_admission(
        &mut self,
        admission: std::result::Result<ObservedTicket<Decision>, AdmissionFailure>,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
        stop: &GatherStop,
    ) -> Result<()> {
        match admission {
            Ok(ticket) => {
                let output = ticket.finish().await?;
                output.install(|decision| match decision {
                    Some(metadata) => self
                        .install_restored_metadata(snapshot, metadata, &|| job.check_cancelled()),
                    None => self.restore_metadata_legacy(snapshot, job),
                })?;
                self.wait_metadata_cleanup(stop, job).await
            }
            Err(
                AdmissionFailure::Budget { .. }
                | AdmissionFailure::Runtime(crate::CalcFlowError::Cancelled { .. }),
            ) => {
                job.check_cancelled()?;
                self.wait_metadata_cleanup(stop, job).await?;
                self.restore_metadata_legacy(snapshot, job)
            }
            Err(AdmissionFailure::Runtime(error)) => Err(error),
        }
    }

    async fn wait_metadata_cleanup(
        &mut self,
        stop: &GatherStop,
        job: &StreamJobContext,
    ) -> Result<()> {
        if let Some(cleanup) = &self.compaction_cleanup {
            cleanup.wait_job(stop, job).await?;
        }
        self.compaction_cleanup = None;
        Ok(())
    }

    fn restore_metadata_legacy(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
    ) -> Result<()> {
        let metadata = self.parse_restore_metadata(snapshot)?;
        self.install_restored_metadata(snapshot, metadata, &|| job.check_cancelled())
    }
}
