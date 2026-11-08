use std::{sync::Arc, time::Duration};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{CalcFlowError, Result, StreamJobContext};

pub(super) struct Pending {
    permit: Arc<Semaphore>,
}

impl Default for Pending {
    fn default() -> Self {
        Self {
            permit: Arc::new(Semaphore::new(1)),
        }
    }
}

impl Pending {
    pub(super) async fn enter(&self, job: &StreamJobContext) -> Result<OwnedSemaphorePermit> {
        let permit = self.permit.clone().acquire_owned();
        tokio::pin!(permit);
        loop {
            job.check_cancelled()?;
            tokio::select! {
                permit = &mut permit => return permit.map_err(|_| busy()),
                () = tokio::time::sleep(Duration::from_millis(10)) => {},
            }
        }
    }

    pub(super) fn try_enter(&self) -> Result<OwnedSemaphorePermit> {
        self.permit.clone().try_acquire_owned().map_err(|_| busy())
    }
}

fn busy() -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: "native:array_sequence@1 state.pending: previous native work has not settled"
            .into(),
    }
}
