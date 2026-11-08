use std::{sync::Arc, time::Duration};

use tokio::{
    sync::{Semaphore, oneshot},
    task::JoinHandle,
};

use crate::{CalcFlowError, CancellationToken, Result, RunContext};

const WORKERS: usize = 2;
const ADMISSION_CHECK: Duration = Duration::from_millis(10);

pub(super) struct Workers {
    permits: Arc<Semaphore>,
}
impl Default for Workers {
    fn default() -> Self {
        Self {
            permits: Arc::new(Semaphore::new(WORKERS)),
        }
    }
}

struct CancelOnDrop(CancellationToken);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

struct OwnedWorker<T: Send + 'static> {
    handle: Option<JoinHandle<Result<T>>>,
    stop: CancellationToken,
}
impl<T: Send + 'static> OwnedWorker<T> {
    async fn settle(mut self) -> Result<T> {
        let result = self
            .handle
            .as_mut()
            .expect("worker handle remains owned")
            .await;
        self.handle.take();
        result.map_err(|_| worker_error())?
    }
}
impl<T: Send + 'static> Drop for OwnedWorker<T> {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.stop.cancel();
        }
    }
}

fn worker_error() -> CalcFlowError {
    CalcFlowError::ExternalProvider {
        provider: "native".into(),
        name: "array_expr".into(),
        version: "1".into(),
        message: "native computation worker terminated".into(),
    }
}
fn check(context: &RunContext, stop: &CancellationToken) -> Result<()> {
    context.check_cancelled()?;
    if stop.is_cancelled() {
        return Err(CalcFlowError::Cancelled {
            run_id: context.run_id().to_owned(),
        });
    }
    Ok(())
}

impl Workers {
    pub(super) async fn run<T: Send + 'static>(
        &self,
        context: RunContext,
        job: impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let permit = self.admit(&context).await?;
        context.check_cancelled()?;
        let stop = CancellationToken::new();
        let _cancel = CancelOnDrop(stop.clone());
        let worker_stop = stop.clone();
        let worker = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut checker = || check(&context, &worker_stop);
            checker()?;
            let result = job(&mut checker);
            checker()?;
            result
        });
        let owned = OwnedWorker {
            handle: Some(worker),
            stop,
        };
        let (sender, receiver) = oneshot::channel();
        drop(tokio::spawn(async move {
            let _ = sender.send(owned.settle().await);
        }));
        receiver.await.map_err(|_| worker_error())?
    }

    async fn admit(&self, context: &RunContext) -> Result<tokio::sync::OwnedSemaphorePermit> {
        let permit = Arc::clone(&self.permits).acquire_owned();
        tokio::pin!(permit);
        loop {
            context.check_cancelled()?;
            tokio::select! {
                permit=&mut permit=>return permit.map_err(|_|worker_error()),
                ()=tokio::time::sleep(ADMISSION_CHECK)=>{},
            }
        }
    }
}

#[cfg(test)]
mod tests;
