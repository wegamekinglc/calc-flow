use std::sync::Arc;

use tokio::sync::OwnedMutexGuard;

use super::{NativeCheckpointRecord, NativeSequenceRecord};
use crate::{
    CalcFlowError, CancellationToken, Result, StateHandle, StateSegment,
    array::{RollingWorkerIdentity, provider::worker::Scope},
    operator::rolling::{IndicatorRollingRecord, RollingIoReservation},
    runtime::streaming::StreamJobControl,
    state::{LocalStateLineageBackend, NativeSegmentIo},
};

pub(crate) type NativeOperationGuard = Arc<OwnedMutexGuard<()>>;

pub(crate) struct NativeIoCompletion {
    handle: StateHandle,
    control: Arc<StreamJobControl>,
    cancellation: CancellationToken,
    _operation: NativeOperationGuard,
    _publication: OwnedMutexGuard<()>,
    _lineage: Arc<LocalStateLineageBackend>,
    segment: Option<StateSegment>,
    _rolling_reservation: Option<RollingIoReservation>,
}
impl NativeIoCompletion {
    pub(crate) fn handle(&self) -> &StateHandle {
        &self.handle
    }
    pub(crate) fn check(&self) -> Result<()> {
        check(&self.control, &self.cancellation)
    }
    fn into_checked(self) -> Result<Self> {
        self.check()?;
        Ok(self)
    }
}
enum Action {
    Verify,
    Stage(StateSegment),
}
struct Work {
    io: NativeSegmentIo,
    completion: NativeIoCompletion,
    stage: bool,
}
impl Work {
    fn run(self, check_worker: &mut dyn FnMut() -> Result<()>) -> Result<NativeIoCompletion> {
        let mut check = || {
            check_worker()?;
            self.completion.check()
        };
        check()?;
        if self.stage {
            self.io.stage_and_publish(
                self.completion
                    .segment
                    .as_ref()
                    .expect("stage owns captured segment"),
                &mut check,
            )?;
        } else {
            self.io.verify(&mut check)?;
        }
        check()?;
        Ok(self.completion)
    }
}
impl NativeSequenceRecord {
    pub(crate) async fn verify(
        &self,
        lineage: Arc<LocalStateLineageBackend>,
        operation: NativeOperationGuard,
        handle: &StateHandle,
        cancellation: &CancellationToken,
    ) -> Result<NativeIoCompletion> {
        self.run_io(
            lineage,
            operation,
            handle.clone(),
            Action::Verify,
            cancellation.clone(),
        )
        .await
    }
    pub(crate) async fn stage_publish(
        &self,
        lineage: Arc<LocalStateLineageBackend>,
        operation: NativeOperationGuard,
        handle: &StateHandle,
        segment: &StateSegment,
        cancellation: &CancellationToken,
    ) -> Result<NativeIoCompletion> {
        self.run_io(
            lineage,
            operation,
            handle.clone(),
            Action::Stage(segment.clone()),
            cancellation.clone(),
        )
        .await
    }
    async fn run_io(
        &self,
        lineage: Arc<LocalStateLineageBackend>,
        operation: NativeOperationGuard,
        handle: StateHandle,
        action: Action,
        cancellation: CancellationToken,
    ) -> Result<NativeIoCompletion> {
        check(&self.control, &cancellation)?;
        let publication = lineage
            .native_publication_guard(|| check(&self.control, &cancellation))
            .await?;
        let io = lineage.native_io(&handle)?;
        let (stage, segment) = match action {
            Action::Verify => (false, None),
            Action::Stage(segment) => (true, Some(segment)),
        };
        let work = Work {
            io,
            stage,
            completion: NativeIoCompletion {
                handle,
                control: Arc::clone(&self.control),
                cancellation,
                _operation: operation,
                _publication: publication,
                _lineage: lineage,
                segment,
                _rolling_reservation: None,
            },
        };
        let scope = Scope::Stream(self.control.as_ref().clone());
        let complete = self
            .workers
            .run_scoped(scope, "array_sequence", move |check| work.run(check))
            .await?;
        complete.check()?;
        Ok(complete)
    }
}
impl NativeCheckpointRecord {
    pub(crate) async fn verify(
        &self,
        lineage: Arc<LocalStateLineageBackend>,
        operation: NativeOperationGuard,
        handle: &StateHandle,
        cancellation: &CancellationToken,
    ) -> Result<NativeIoCompletion> {
        match self {
            Self::Sequence(record) => {
                record
                    .verify(lineage, operation, handle, cancellation)
                    .await
            }
            Self::IndicatorRolling(record) => {
                rolling_io(
                    record,
                    lineage,
                    operation,
                    handle.clone(),
                    Action::Verify,
                    cancellation.clone(),
                )
                .await
            }
        }
    }
    pub(crate) async fn stage_publish(
        &self,
        lineage: Arc<LocalStateLineageBackend>,
        operation: NativeOperationGuard,
        handle: &StateHandle,
        segment: &StateSegment,
        cancellation: &CancellationToken,
    ) -> Result<NativeIoCompletion> {
        match self {
            Self::Sequence(record) => {
                record
                    .stage_publish(lineage, operation, handle, segment, cancellation)
                    .await
            }
            Self::IndicatorRolling(record) => {
                let segment = record.retain_for_io(segment)?;
                rolling_io(
                    record,
                    lineage,
                    operation,
                    handle.clone(),
                    Action::Stage(segment),
                    cancellation.clone(),
                )
                .await
            }
        }
    }
}

async fn rolling_io(
    record: &IndicatorRollingRecord,
    lineage: Arc<LocalStateLineageBackend>,
    operation: NativeOperationGuard,
    handle: StateHandle,
    action: Action,
    cancellation: CancellationToken,
) -> Result<NativeIoCompletion> {
    check(record.control(), &cancellation)?;
    let identity = record.worker_identity()?;
    let reservation = record.reserve_io()?;
    let publication = lineage
        .native_publication_guard(|| check(record.control(), &cancellation))
        .await?;
    let io = lineage.native_io(&handle)?;
    let (stage, segment) = rolling_action(action);
    let work = Work {
        io,
        stage,
        completion: NativeIoCompletion {
            handle,
            control: Arc::clone(record.control()),
            cancellation,
            _operation: operation,
            _publication: publication,
            _lineage: lineage,
            segment,
            _rolling_reservation: Some(reservation),
        },
    };
    run_rolling_io(record, identity, work).await
}

async fn run_rolling_io(
    record: &IndicatorRollingRecord,
    identity: Option<Arc<RollingWorkerIdentity>>,
    work: Work,
) -> Result<NativeIoCompletion> {
    let workers = record.workers();
    let scope = Scope::Stream(record.control().as_ref().clone());
    let completion = match identity {
        Some(identity) => {
            workers
                .run_rolling_credited(scope, identity, move |check| work.run(check))
                .await?
        }
        None => {
            workers
                .run_rolling_scoped(scope, record.node(), move |check| work.run(check))
                .await?
        }
    };
    completion.into_checked()
}

fn rolling_action(action: Action) -> (bool, Option<StateSegment>) {
    match action {
        Action::Verify => (false, None),
        Action::Stage(segment) => (true, Some(segment)),
    }
}
fn check(control: &StreamJobControl, cancellation: &CancellationToken) -> Result<()> {
    control.check_cancelled()?;
    if cancellation.is_cancelled() {
        return Err(CalcFlowError::Cancelled {
            run_id: control.job_id().to_string(),
        });
    }
    Ok(())
}
