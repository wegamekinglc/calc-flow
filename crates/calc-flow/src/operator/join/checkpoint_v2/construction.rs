use std::sync::Arc;

use datafusion::execution::memory_pool::MemoryReservation;

use super::super::{OperatorStateSnapshot, StreamJoinOperator, metadata_validation::inventory};
use super::ipc::accounting::{add, product, sum};
use super::work::{OwnedConfig, RestoreWork, initial_input_bytes, retirement_bytes};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherClient, GatherOperatorId, GatherScope, GatherStop, RetirementGuard,
    TaskId,
};
use crate::{CalcFlowError, Result, StreamJobContext};

const INPUT_LABEL: &str = "stream-join-v2-restore-input";
const V1_CONTROL_LABEL: &str = "stream-join-metadata";

pub(super) struct Construction {
    pub(super) work: Option<RestoreWork>,
    pub(super) scope: GatherScope,
    pub(super) stop: GatherStop,
    pub(super) control: MemoryReservation,
    pub(super) retirement: Option<RetirementGuard>,
}

// Partial copies also die before their credit and the observed retirement.
struct Inputs {
    config: Option<OwnedConfig>,
    credit: Option<MemoryReservation>,
    retirements: Vec<RetirementGuard>,
    retirement_credit: Option<MemoryReservation>,
    control: Option<MemoryReservation>,
    retirement: Option<RetirementGuard>,
}

pub(super) async fn prepare(
    operator: &StreamJoinOperator,
    snapshot: &mut OperatorStateSnapshot,
    job: &StreamJobContext,
    task: Option<TaskId>,
) -> std::result::Result<Construction, AdmissionFailure> {
    let mut inputs = admitted_inputs(operator, snapshot, job)?;
    boundary(job).await?;
    validate_owners(snapshot, job).await?;
    inputs.config = Some(copy_config(operator));
    boundary(job).await?;
    let scope = job
        .gather_owner()
        .client(GatherOperatorId::new(Arc::from(operator.name.as_str())).with_task(task))
        .scope()?;
    let stop = GatherStop::from_job(job);
    inputs
        .retain_residents(job, snapshot.segments.len())
        .await?;
    boundary(job).await?;
    Ok(inputs.finish(snapshot, scope, stop))
}

fn admitted_inputs(
    operator: &StreamJoinOperator,
    snapshot: &OperatorStateSnapshot,
    job: &StreamJobContext,
) -> std::result::Result<Inputs, AdmissionFailure> {
    job.check_cancelled()?;
    fund(operator, snapshot.segments.len(), job)
}

async fn validate_owners(
    snapshot: &OperatorStateSnapshot,
    job: &StreamJobContext,
) -> std::result::Result<(), AdmissionFailure> {
    for segment in snapshot.segments.values() {
        boundary(job).await?;
        if !segment.has_owner() {
            return Err(internal("V2 managed restore requires owned segment input").into());
        }
    }
    Ok(())
}

fn fund(
    operator: &StreamJoinOperator,
    segments: usize,
    job: &StreamJobContext,
) -> std::result::Result<Inputs, AdmissionFailure> {
    let (bytes, controls) = required(operator, segments)?;
    let retirement_size = retirement_bytes(segments)?;
    let runtime = operator
        .runtime
        .runtime
        .as_ref()
        .ok_or_else(|| internal("V2 restore requires the configured runtime"))?;
    let credit = runtime.incremental_reservation(INPUT_LABEL);
    admit_input(&credit, bytes)?;
    let retirement = job.gather_owner().retain_retirement()?;
    let control = credit.split(controls);
    let retirement_credit = credit.split(retirement_size);
    Ok(Inputs {
        config: None,
        credit: Some(credit),
        retirements: Vec::new(),
        retirement_credit: Some(retirement_credit),
        control: Some(control),
        retirement: Some(retirement),
    })
}

fn required(operator: &StreamJoinOperator, segments: usize) -> Result<(usize, usize)> {
    let controls = caller_controls(&operator.name)?;
    let input = initial_input_bytes(
        &operator.name,
        &operator.spec,
        [
            &operator.compiled.left_key_indices,
            &operator.compiled.right_key_indices,
        ],
        segments,
    )?;
    Ok((add(input, controls)?, controls))
}

fn admit_input(
    credit: &MemoryReservation,
    bytes: usize,
) -> std::result::Result<(), AdmissionFailure> {
    credit
        .try_grow(bytes)
        .map_err(|source| AdmissionFailure::Budget {
            stage: "V2 restore input",
            source,
        })
}

fn copy_config(operator: &StreamJoinOperator) -> OwnedConfig {
    OwnedConfig {
        name: String::from(operator.name.as_str()),
        spec: operator.spec.clone(),
        schemas: [
            Arc::clone(operator.input_schema(0)),
            Arc::clone(operator.input_schema(1)),
        ],
        keys: [
            operator.compiled.left_key_indices.clone(),
            operator.compiled.right_key_indices.clone(),
        ],
        time_indices: [
            operator.compiled.left_event_time_index,
            operator.compiled.right_event_time_index,
        ],
        #[cfg(test)]
        metadata_hook: operator.metadata_test_hook.clone(),
        #[cfg(test)]
        decoded_hook: operator.decoded_row_test_hook.clone(),
    }
}

impl Inputs {
    async fn retain_residents(&mut self, job: &StreamJobContext, segments: usize) -> Result<()> {
        for _ in 0..add(segments, 1)? {
            boundary(job).await?;
            self.retirements
                .push(job.gather_owner().retain_retirement()?);
        }
        Ok(())
    }

    fn finish(
        mut self,
        snapshot: &mut OperatorStateSnapshot,
        scope: GatherScope,
        stop: GatherStop,
    ) -> Construction {
        let work = RestoreWork {
            snapshot: std::mem::take(snapshot),
            config: self.config.take().expect("paid V2 config was copied"),
            input_credit: self.credit.take().expect("paid V2 input"),
            retirements: std::mem::take(&mut self.retirements),
            _retirement_credit: self
                .retirement_credit
                .take()
                .expect("paid V2 retirement Vec"),
        };
        Construction {
            work: Some(work),
            scope,
            stop,
            control: self.control.take().expect("paid V2 caller controls"),
            retirement: self.retirement.take(),
        }
    }
}

fn caller_controls(name: &str) -> Result<usize> {
    let legacy = inventory::caller_controls(name)
        .ok_or_else(|| internal("V2 restore caller control charge overflow"))?;
    let label_extra = INPUT_LABEL.len().saturating_sub(V1_CONTROL_LABEL.len());
    sum(&[
        legacy,
        product(label_extra, 3)?,
        size_of::<Construction>(),
        size_of::<Inputs>(),
        size_of::<GatherOperatorId>(),
        size_of::<GatherClient>(),
        "V2 managed restore requires owned segment input".len(),
    ])
}

async fn boundary(job: &StreamJobContext) -> Result<()> {
    job.check_cancelled()?;
    tokio::task::yield_now().await;
    job.check_cancelled()
}

fn internal(message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: message.into(),
    }
}
