use super::{Construction, MetadataWork, StreamJoinOperator, ValidatedMetadata};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, AttemptCleanup, GatherOperatorId, GatherStop, ObservedTicket, OwnedCpuWork,
    RetirementGuard, TaskId, cleanup_control_bytes,
};
use crate::{OperatorStateSnapshot, Result, StreamJobContext};
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

mod inventory;
mod plan;
mod restore_bases;
mod restore_segments;

#[cfg(test)]
mod tests;

type SchemaDecision = Option<SchemaSuccess>;

struct SchemaSuccess {
    metadata: ValidatedMetadata,
    schemas: OwnedExpectedSchemas,
}

#[derive(Clone)]
pub(in crate::operator::join) struct OwnedExpectedSchemas {
    left: SchemaRef,
    right: SchemaRef,
    _funding: Arc<DescriptorFunding>,
}

struct DescriptorFunding {
    _credit: MemoryReservation,
    _retirement: RetirementGuard,
}

impl OwnedExpectedSchemas {
    pub(in crate::operator::join) fn schema(&self, side: usize) -> &Schema {
        match side {
            0 => &self.left,
            1 => &self.right,
            _ => unreachable!("two Join schemas"),
        }
    }

    #[cfg(test)]
    pub(in crate::operator::join) fn credit(&self) -> &MemoryReservation {
        let OwnedExpectedSchemas {
            _funding: funding, ..
        } = self;
        let DescriptorFunding {
            _credit: credit, ..
        } = funding.as_ref();
        credit
    }
}

struct SchemaWork {
    metadata: MetadataWork,
    plans: [Vec<plan::FieldPlan>; 2],
    _input_credit: MemoryReservation,
    funding: Arc<DescriptorFunding>,
    #[cfg(test)]
    hook: Option<super::super::SchemaTestHook>,
}

impl OwnedCpuWork for SchemaWork {
    type Output = SchemaDecision;

    fn control_bytes(&self) -> Result<usize> {
        super::inventory::caller_controls(&self.metadata.name)
            .and_then(|bytes| bytes.checked_add(cleanup_control_bytes::<SchemaDecision>()))
            .ok_or_else(|| crate::CalcFlowError::Internal {
                message: "schema metadata control overflow".into(),
            })
    }

    fn run(mut self, stop: &GatherStop) -> Result<SchemaDecision> {
        let Some(metadata) = self.metadata.run(stop)? else {
            return Ok(None);
        };
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            let DescriptorFunding {
                _credit: credit, ..
            } = self.funding.as_ref();
            hook(Some(credit), true);
        }
        let left = plan::build(std::mem::take(&mut self.plans[0]), stop)?;
        let right = plan::build(std::mem::take(&mut self.plans[1]), stop)?;
        Ok(Some(SchemaSuccess {
            metadata,
            schemas: OwnedExpectedSchemas {
                left,
                right,
                _funding: self.funding,
            },
        }))
    }
}

// Every partial plan precedes its input credit and final descriptor lease.
struct SchemaConstruction {
    metadata: Construction,
    plans: [Vec<plan::FieldPlan>; 2],
    input_credit: Option<MemoryReservation>,
    funding: Option<Arc<DescriptorFunding>>,
}

impl SchemaConstruction {
    fn new(
        operator: &StreamJoinOperator,
        metadata: Construction,
        job: &StreamJobContext,
    ) -> Result<Option<Self>> {
        let Some((input, output)) =
            inventory::required([operator.input_schema(0), operator.input_schema(1)])
        else {
            return Ok(None);
        };
        let original = metadata
            .credit
            .as_ref()
            .expect("paid metadata construction");
        let input_credit = original.new_empty();
        let output_credit = original.new_empty();
        if input_credit.try_grow(input).is_err() || output_credit.try_grow(output).is_err() {
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
        Ok(Some(Self {
            metadata,
            plans: [Vec::new(), Vec::new()],
            input_credit: Some(input_credit),
            funding: Some(Arc::new(DescriptorFunding {
                _credit: output_credit,
                _retirement: retirement,
            })),
        }))
    }

    async fn copy(
        &mut self,
        operator: &StreamJoinOperator,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
    ) -> Result<()> {
        self.metadata.copy(operator, snapshot, job).await?;
        plan::copy(
            &mut self.plans,
            [operator.input_schema(0), operator.input_schema(1)],
            job,
        )
        .await
    }

    fn submit<'a>(
        &'a mut self,
        observer: &'a mut Option<AttemptCleanup>,
        #[cfg(test)] metadata_hook: Option<super::super::MetadataTestHook>,
        #[cfg(test)] schema_hook: Option<super::super::SchemaTestHook>,
    ) -> impl Future<Output = std::result::Result<ObservedTicket<SchemaDecision>, AdmissionFailure>> + 'a
    {
        let work = SchemaWork {
            metadata: MetadataWork {
                snapshot: std::mem::take(&mut self.metadata.snapshot),
                expected: self
                    .metadata
                    .expected
                    .take()
                    .expect("completed metadata construction"),
                name: std::mem::take(&mut self.metadata.name),
                #[cfg(test)]
                hook: metadata_hook,
            },
            plans: std::mem::take(&mut self.plans),
            _input_credit: self.input_credit.take().expect("paid schema input"),
            funding: self.funding.take().expect("paid descriptor output"),
            #[cfg(test)]
            hook: schema_hook,
        };
        self.metadata
            .control
            .scope
            .as_ref()
            .expect("prepared metadata scope")
            .submit_observed_work(
                work,
                self.metadata.credit.take().expect("paid metadata input"),
                self.metadata
                    .control
                    .stop
                    .as_ref()
                    .expect("paid metadata stop")
                    .clone(),
                self.metadata
                    .retirement
                    .take()
                    .expect("registered metadata work"),
                observer,
            )
    }
}

impl StreamJoinOperator {
    pub(super) async fn try_restore_owned_schema(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<bool> {
        let Some(metadata) = self.metadata_construction(snapshot, job)? else {
            return Ok(false);
        };
        let Some(mut construction) = SchemaConstruction::new(self, metadata, job)? else {
            return Ok(false);
        };
        #[cfg(test)]
        if let Some(hook) = &self.metadata_test_hook {
            hook(construction.metadata.credit.as_ref(), false);
        }
        construction.copy(self, snapshot, job).await?;
        construction.metadata.control.scope = match job
            .gather_owner()
            .client(GatherOperatorId::new(Arc::from(self.name.as_str())).with_task(task))
            .scope()
        {
            Ok(scope) => Some(scope),
            Err(crate::CalcFlowError::Cancelled { .. }) => {
                job.check_cancelled()?;
                return Ok(false);
            }
            Err(error) => return Err(error),
        };
        let admission = construction
            .submit(
                &mut self.compaction_cleanup,
                #[cfg(test)]
                self.metadata_test_hook.clone(),
                #[cfg(test)]
                self.schema_test_hook.clone(),
            )
            .await;
        self.finish_schema_admission(
            admission,
            snapshot,
            job,
            construction
                .metadata
                .control
                .stop
                .as_ref()
                .expect("paid metadata stop"),
        )
        .await
    }

    async fn finish_schema_admission(
        &mut self,
        admission: std::result::Result<ObservedTicket<SchemaDecision>, AdmissionFailure>,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
        stop: &GatherStop,
    ) -> Result<bool> {
        match admission {
            Ok(ticket) => self.finish_schema_ticket(ticket, snapshot, job, stop).await,
            Err(AdmissionFailure::Budget {
                stage: "attempt", ..
            }) => {
                job.check_cancelled()?;
                Ok(false)
            }
            Err(
                AdmissionFailure::Budget { .. }
                | AdmissionFailure::Runtime(crate::CalcFlowError::Cancelled { .. }),
            ) => self.finish_schema_refusal(snapshot, job, stop).await,
            Err(AdmissionFailure::Runtime(error)) => Err(error),
        }
    }

    async fn finish_schema_ticket(
        &mut self,
        ticket: ObservedTicket<SchemaDecision>,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
        stop: &GatherStop,
    ) -> Result<bool> {
        let output = ticket.finish().await?;
        output.install(|decision| self.install_schema_decision(decision, snapshot, job))?;
        self.wait_metadata_cleanup(stop, job).await?;
        Ok(true)
    }

    fn install_schema_decision(
        &mut self,
        decision: SchemaDecision,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
    ) -> Result<()> {
        match decision {
            Some(success) => self.install_restored_metadata_with_schemas(
                snapshot,
                success.metadata,
                Some(&success.schemas),
                &|| job.check_cancelled(),
            ),
            None => self.restore_metadata_legacy(snapshot, job),
        }
    }

    async fn finish_schema_refusal(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
        stop: &GatherStop,
    ) -> Result<bool> {
        job.check_cancelled()?;
        self.wait_metadata_cleanup(stop, job).await?;
        self.restore_metadata_legacy(snapshot, job)?;
        Ok(true)
    }
}
