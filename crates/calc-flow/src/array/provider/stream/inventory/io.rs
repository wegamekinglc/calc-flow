use std::sync::Arc;

use tokio::sync::OwnedMutexGuard;

use super::NativeSequenceRecord;
use crate::{
    CalcFlowError, CancellationToken, Result, StateHandle, StateSegment,
    array::provider::worker::Scope,
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
}
impl NativeIoCompletion {
    pub(crate) fn handle(&self) -> &StateHandle {
        &self.handle
    }
    pub(crate) fn check(&self) -> Result<()> {
        check(&self.control, &self.cancellation)
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
fn check(control: &StreamJobControl, cancellation: &CancellationToken) -> Result<()> {
    control.check_cancelled()?;
    if cancellation.is_cancelled() {
        return Err(CalcFlowError::Cancelled {
            run_id: control.job_id().to_string(),
        });
    }
    Ok(())
}
