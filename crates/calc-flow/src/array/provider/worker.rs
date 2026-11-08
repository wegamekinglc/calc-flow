use std::{sync::Arc, time::Duration};

use tokio::{
    sync::{Semaphore, oneshot},
    task::JoinHandle,
};

use crate::runtime::streaming::StreamJobControl;
use crate::{CalcFlowError, CancellationToken, Result, RunContext, StreamJobContext};

use super::preparation::settlement::Preparation;

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
    name: &'static str,
}
impl<T: Send + 'static> OwnedWorker<T> {
    async fn settle(mut self) -> Result<T> {
        let result = self
            .handle
            .as_mut()
            .expect("worker handle remains owned")
            .await;
        self.handle.take();
        result.map_err(|_| worker_error(self.name))?
    }
}
impl<T: Send + 'static> Drop for OwnedWorker<T> {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.stop.cancel();
        }
    }
}

fn worker_error(name: &str) -> CalcFlowError {
    CalcFlowError::ExternalProvider {
        provider: "native".into(),
        name: name.into(),
        version: "1".into(),
        message: "native computation worker terminated".into(),
    }
}
pub(super) enum Scope {
    Batch(RunContext),
    Stream(StreamJobControl),
}
impl Scope {
    pub(super) fn stream(job: &StreamJobContext) -> Self {
        Self::Stream(job.native_control())
    }
    fn check_cancelled(&self) -> Result<()> {
        match self {
            Self::Batch(context) => context.check_cancelled(),
            Self::Stream(context) => context.check_cancelled(),
        }
    }
    fn cancelled(&self) -> CalcFlowError {
        let run_id = match self {
            Self::Batch(context) => context.run_id().to_owned(),
            Self::Stream(context) => context.job_id().to_string(),
        };
        CalcFlowError::Cancelled { run_id }
    }
}
fn check(context: &Scope, stop: &CancellationToken) -> Result<()> {
    context.check_cancelled()?;
    if stop.is_cancelled() {
        return Err(context.cancelled());
    }
    Ok(())
}

impl Workers {
    #[cfg(test)]
    pub(super) fn test_permits(&self) -> Arc<Semaphore> {
        Arc::clone(&self.permits)
    }
    pub(super) async fn run<T: Send + 'static>(
        &self,
        context: RunContext,
        job: impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        self.run_scoped(Scope::Batch(context), "array_expr", job)
            .await
    }
    pub(super) async fn run_table<T: Send + 'static>(
        &self,
        context: RunContext,
        job: impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        self.run_scoped(Scope::Batch(context), "table_array", job)
            .await
    }
    pub(super) async fn run_stream<T: Send + 'static>(
        &self,
        context: &StreamJobContext,
        job: impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        self.run_scoped(Scope::stream(context), "table_array", job)
            .await
    }
    pub(super) async fn run_scoped<T: Send + 'static>(
        &self,
        context: Scope,
        name: &'static str,
        job: impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let mut result = self
            .run_work(context, name, Preparation::from_parts(job, None))
            .await?;
        Ok(result.take_value())
    }
    pub(super) async fn run_preparation<T: Send + 'static>(
        &self,
        context: Scope,
        name: &'static str,
        pending: Preparation<
            impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<T> + Send + 'static,
        >,
    ) -> Result<Preparation<T>> {
        self.run_work(context, name, pending).await
    }
    async fn run_work<T: Send + 'static>(
        &self,
        context: Scope,
        name: &'static str,
        pending: Preparation<
            impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<T> + Send + 'static,
        >,
    ) -> Result<Preparation<T>> {
        let permit = self.admit(&context, name).await?;
        context.check_cancelled()?;
        let (job, lease) = pending.into_parts();
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
            name,
        };
        let (sender, receiver) = oneshot::channel();
        drop(tokio::spawn(async move {
            let packet = Preparation::from_parts(owned.settle().await, lease);
            let _ = sender.send(packet);
        }));
        receiver.await.map_err(|_| worker_error(name))?.transpose()
    }

    async fn admit(
        &self,
        context: &Scope,
        name: &str,
    ) -> Result<tokio::sync::OwnedSemaphorePermit> {
        let permit = Arc::clone(&self.permits).acquire_owned();
        tokio::pin!(permit);
        loop {
            context.check_cancelled()?;
            tokio::select! {
                permit=&mut permit=>return permit.map_err(|_|worker_error(name)),
                ()=tokio::time::sleep(ADMISSION_CHECK)=>{},
            }
        }
    }
}

#[cfg(test)]
mod tests;
