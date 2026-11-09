use std::{sync::Arc, time::Duration};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{CalcFlowError, Result, StreamJobContext};

pub(crate) struct Pending {
    permit: Arc<Semaphore>,
    rolling: Option<String>,
}

impl Default for Pending {
    fn default() -> Self {
        Self {
            permit: Arc::new(Semaphore::new(1)),
            rolling: None,
        }
    }
}

impl Pending {
    pub(crate) fn rolling(node: &str) -> Self {
        Self {
            rolling: Some(node.to_owned()),
            ..Self::default()
        }
    }
    fn busy(&self) -> CalcFlowError {
        match &self.rolling {
            Some(node) => CalcFlowError::Operator {
                node_id: node.clone(),
                message: "rolling state.pending: previous work has not settled".into(),
            },
            None => busy(),
        }
    }
    pub(crate) async fn enter(&self, job: &StreamJobContext) -> Result<OwnedSemaphorePermit> {
        let permit = self.permit.clone().acquire_owned();
        tokio::pin!(permit);
        loop {
            job.check_cancelled()?;
            tokio::select! {
                permit = &mut permit => return permit.map_err(|_| self.busy()),
                () = tokio::time::sleep(Duration::from_millis(10)) => {},
            }
        }
    }

    pub(crate) fn try_enter(&self) -> Result<OwnedSemaphorePermit> {
        self.permit
            .clone()
            .try_acquire_owned()
            .map_err(|_| self.busy())
    }
}

fn busy() -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: "native:array_sequence@1 state.pending: previous native work has not settled"
            .into(),
    }
}
