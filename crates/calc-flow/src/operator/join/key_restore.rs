use std::sync::Arc;

use datafusion::execution::memory_pool::MemoryReservation;

use super::{
    OperatorStateSnapshot, RetainedKeyCache, StreamJoinOperator, StreamJoinState,
    checkpoint_v2::WriterState,
    key_dictionary::{Dictionary, Rebuilt, grant},
};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, OwnedCpuWork, TaskId, cleanup_control_bytes,
};
use crate::{Result, StreamJobContext};

struct Candidate {
    state: StreamJoinState,
    writer: WriterState,
}

struct FundedCandidate {
    candidate: Box<Candidate>,
    _credit: MemoryReservation,
}

struct DerivedKeys {
    keys: Option<Rebuilt>,
    grant_refused: bool,
}

pub(super) struct Capture {
    candidate: Option<FundedCandidate>,
    credit: Option<MemoryReservation>,
}

pub(super) const fn capture_control_bytes() -> usize {
    size_of::<Capture>() + size_of::<Candidate>() + 1024
}

struct CaptureGuard<'a> {
    operator: &'a mut StreamJoinOperator,
}

impl Drop for CaptureGuard<'_> {
    fn drop(&mut self) {
        self.operator.legacy_restore_capture = None;
    }
}

#[cfg(test)]
pub(super) type TestHook = Arc<dyn Fn(&super::StateRows) + Send + Sync>;

struct DerivedWork {
    rows: [Arc<super::StateRows>; 2],
    name: String,
    #[cfg(test)]
    hook: Option<TestHook>,
    credit: MemoryReservation,
}

impl OwnedCpuWork for DerivedWork {
    type Output = Option<Rebuilt>;

    fn control_bytes(&self) -> Result<usize> {
        Ok(size_of::<Self>() + cleanup_control_bytes::<Option<Rebuilt>>() + 1024)
    }

    fn run(self, stop: &GatherStop) -> Result<Self::Output> {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(&self.rows[0]);
        }
        Dictionary::default().rebuild_checked(
            [&self.rows[0], &self.rows[1]],
            &self.credit,
            &self.name,
            &|| stop.check(),
        )
    }
}

impl StreamJoinOperator {
    pub(crate) async fn restore_managed_metadata(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<()> {
        self.restore_managed_legacy_keys(snapshot, job, task).await
    }

    pub(super) async fn restore_managed_legacy_keys(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<()> {
        job.check_cancelled()?;
        if !snapshot_has_retained_rows(snapshot) {
            return self
                .restore_managed_metadata_inner(snapshot, job, task)
                .await;
        }
        self.prepare_legacy_capture()?;
        let guard = CaptureGuard { operator: self };
        guard
            .operator
            .restore_managed_metadata_inner(snapshot, job, task)
            .await?;
        let candidate = guard
            .operator
            .legacy_restore_capture
            .as_mut()
            .unwrap()
            .candidate
            .take();
        if let Some(candidate) = candidate {
            let keys = guard
                .operator
                .derived_legacy_keys(&candidate.candidate, job, task)
                .await?;
            guard.operator.publish_funded_legacy(candidate, keys, job)?;
        }
        Ok(())
    }

    fn prepare_legacy_capture(&mut self) -> Result<()> {
        let master = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-dictionary");
        let credit = grant(&master, capture_control_bytes(), &self.name)?;
        let refusals = self.restored_key_refusals(credit.is_none())?;
        self.key_work.dictionary_grant_refusals = refusals;
        self.legacy_restore_capture = Some(Capture {
            candidate: None,
            credit,
        });
        Ok(())
    }

    fn publish_funded_legacy(
        &mut self,
        candidate: FundedCandidate,
        keys: DerivedKeys,
        job: &StreamJobContext,
    ) -> Result<()> {
        job.check_cancelled()?;
        let refusals = self.restored_key_refusals(keys.grant_refused)?;
        let FundedCandidate { candidate, _credit } = candidate;
        let Candidate { state, writer } = *candidate;
        self.publish_legacy_keys(state, writer, keys.keys);
        self.key_work.dictionary_grant_refusals = refusals;
        Ok(())
    }

    pub(super) fn install_legacy_keys(
        &mut self,
        state: StreamJoinState,
        writer: WriterState,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        if let Some(capture) = &mut self.legacy_restore_capture {
            if let Some(credit) = capture.credit.take() {
                capture.candidate = Some(FundedCandidate {
                    candidate: Box::new(Candidate { state, writer }),
                    _credit: credit,
                });
                return Ok(());
            }
            self.publish_legacy_keys(state, writer, None);
            return Ok(());
        }
        let master = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-dictionary");
        let keys = Dictionary::default().rebuild_checked(
            [&state.left, &state.right],
            &master,
            &self.name,
            check,
        )?;
        check()?;
        let refusals = self.restored_key_refusals(keys.is_none())?;
        self.publish_legacy_keys(state, writer, keys);
        self.key_work.dictionary_grant_refusals = refusals;
        Ok(())
    }

    fn publish_legacy_keys(
        &mut self,
        state: StreamJoinState,
        writer: WriterState,
        keys: Option<Rebuilt>,
    ) {
        self.install_restored_keys(keys, !state.left.is_empty() || !state.right.is_empty());
        self.state = state;
        self.retained_key_cache = RetainedKeyCache::default();
        self.ingress_progress = super::IngressProgressSnapshot::default();
        self.v2_containers = None;
        self.v2_writer = writer;
    }

    async fn derived_legacy_keys(
        &mut self,
        candidate: &Candidate,
        job: &StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<DerivedKeys> {
        let Some(work) = self.capture_derived_work(candidate)? else {
            return Ok(DerivedKeys {
                keys: None,
                grant_refused: true,
            });
        };
        let controls = work.credit.new_empty();
        let scope = job
            .gather_owner()
            .client(GatherOperatorId::new(Arc::from(self.name.as_str())).with_task(task))
            .scope();
        let scope = match scope {
            Ok(scope) => scope,
            Err(crate::CalcFlowError::Cancelled { .. }) => return self.closed_derived_keys(job),
            Err(error) => return Err(error),
        };
        let stop = GatherStop::from_job(job);
        let retirement = job.gather_owner().retain_retirement()?;
        let admission = scope
            .submit_observed_work(
                work,
                controls,
                stop.clone(),
                retirement,
                &mut self.compaction_cleanup,
            )
            .await;
        let result = self.finish_derived_keys(admission, job).await;
        job.check_cancelled()?;
        self.finish_derived_cleanup(&stop, job).await?;
        result
    }

    fn capture_derived_work(&mut self, candidate: &Candidate) -> Result<Option<DerivedWork>> {
        let master = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-dictionary");
        let bytes = size_of::<DerivedWork>()
            .checked_add(self.name.len())
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| super::counter_overflow(&self.name, "key_restore"))?;
        let Some(credit) = grant(&master, bytes, &self.name)? else {
            return Ok(None);
        };
        Ok(Some(DerivedWork {
            rows: [
                Arc::clone(&candidate.state.left.0),
                Arc::clone(&candidate.state.right.0),
            ],
            name: self.name.clone(),
            #[cfg(test)]
            hook: self.key_restore_test_hook.take(),
            credit,
        }))
    }

    async fn finish_derived_keys(
        &mut self,
        admission: std::result::Result<
            crate::runtime::streaming::gather_work::ObservedTicket<Option<Rebuilt>>,
            AdmissionFailure,
        >,
        job: &StreamJobContext,
    ) -> Result<DerivedKeys> {
        let ticket = match admission {
            Ok(ticket) => ticket,
            Err(AdmissionFailure::Budget { .. }) => {
                return Ok(DerivedKeys {
                    keys: None,
                    grant_refused: true,
                });
            }
            Err(AdmissionFailure::Runtime(crate::CalcFlowError::Cancelled { .. })) => {
                return self.closed_derived_keys(job);
            }
            Err(AdmissionFailure::Runtime(error)) => return Err(error),
        };
        let output = tokio::select! {
            biased;
            () = super::output_gather::cancelled(job) => {
                job.check_cancelled()?;
                unreachable!("derived restore stop is terminal")
            }
            output = ticket.finish() => output?,
        };
        let mut result = None;
        output.install(|keys| {
            result = Some(keys);
            Ok(())
        })?;
        let keys = result.expect("observed derived key candidate");
        Ok(DerivedKeys {
            grant_refused: keys.is_none(),
            keys,
        })
    }

    fn closed_derived_keys(&mut self, job: &StreamJobContext) -> Result<DerivedKeys> {
        job.check_cancelled()?;
        self.key_work.closed_restore_routes = super::checked_metric(
            self.key_work.closed_restore_routes,
            1,
            &self.name,
            "closed_restore_routes",
        )?;
        Ok(DerivedKeys {
            keys: None,
            grant_refused: false,
        })
    }

    async fn finish_derived_cleanup(
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
}

fn snapshot_has_retained_rows(snapshot: &OperatorStateSnapshot) -> bool {
    ["left", "right"].into_iter().any(|side| {
        snapshot
            .inline_metadata
            .get("metrics")
            .and_then(|metrics| metrics.get(side))
            .and_then(|side| side.get("retained_rows"))
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|rows| rows > 0)
    })
}
