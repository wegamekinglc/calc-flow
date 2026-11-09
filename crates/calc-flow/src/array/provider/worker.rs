use std::{sync::Arc, time::Duration};

use tokio::{
    sync::{Semaphore, oneshot},
    task::JoinHandle,
};

use crate::runtime::streaming::StreamJobControl;
use crate::{CalcFlowError, CancellationToken, Result, RunContext, StreamJobContext};

use super::preparation::settlement::Preparation;

mod rolling_identity;
pub(crate) use rolling_identity::RollingWorkerIdentity;

const WORKERS: usize = 2;
const ADMISSION_CHECK: Duration = Duration::from_millis(10);

pub(crate) struct Workers {
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
    identity: WorkerIdentity,
}
impl<T: Send + 'static> OwnedWorker<T> {
    async fn settle(mut self) -> Result<T> {
        let result = self
            .handle
            .as_mut()
            .expect("worker handle remains owned")
            .await;
        self.handle.take();
        result.map_err(|_| self.identity.error())?
    }
}
impl<T: Send + 'static> Drop for OwnedWorker<T> {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.stop.cancel();
        }
    }
}

#[derive(Clone)]
enum WorkerIdentity {
    External(&'static str),
    Rolling(String),
    CreditedRolling(Arc<RollingWorkerIdentity>),
}
impl WorkerIdentity {
    fn error(&self) -> CalcFlowError {
        match self {
            Self::External(name) => CalcFlowError::ExternalProvider {
                provider: "native".into(),
                name: (*name).into(),
                version: "1".into(),
                message: "native computation worker terminated".into(),
            },
            Self::Rolling(node) => CalcFlowError::Operator {
                node_id: node.clone(),
                message: "rolling computation worker terminated".into(),
            },
            Self::CreditedRolling(identity) => CalcFlowError::Operator {
                node_id: identity.as_str().into(),
                message: "rolling computation worker terminated".into(),
            },
        }
    }
}
pub(crate) enum Scope {
    Batch(RunContext),
    Stream(StreamJobControl),
}
impl Scope {
    pub(crate) fn stream(job: &StreamJobContext) -> Self {
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
    pub(crate) async fn run_scoped<T: Send + 'static>(
        &self,
        context: Scope,
        name: &'static str,
        job: impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let mut result = self
            .run_work(
                context,
                WorkerIdentity::External(name),
                Preparation::from_parts(job, None),
            )
            .await?;
        Ok(result.take_value())
    }
    pub(crate) async fn run_rolling_scoped<T: Send + 'static>(
        &self,
        context: Scope,
        node: &str,
        job: impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let mut result = self
            .run_work(
                context,
                WorkerIdentity::Rolling(node.to_owned()),
                Preparation::from_parts(job, None),
            )
            .await?;
        Ok(result.take_value())
    }
    pub(crate) async fn run_rolling_credited<T: Send + 'static>(
        &self,
        context: Scope,
        identity: Arc<RollingWorkerIdentity>,
        job: impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let mut result = self
            .run_work(
                context,
                WorkerIdentity::CreditedRolling(identity),
                Preparation::from_parts(job, None),
            )
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
        self.run_work(context, WorkerIdentity::External(name), pending)
            .await
    }
    async fn run_work<T: Send + 'static>(
        &self,
        context: Scope,
        identity: WorkerIdentity,
        pending: Preparation<
            impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<T> + Send + 'static,
        >,
    ) -> Result<Preparation<T>> {
        let permit = self.admit(&context, &identity).await?;
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
            identity: identity.clone(),
        };
        let (sender, receiver) = oneshot::channel();
        drop(tokio::spawn(async move {
            let packet = Preparation::from_parts(owned.settle().await, lease);
            let _ = sender.send(packet);
        }));
        receiver.await.map_err(|_| identity.error())?.transpose()
    }

    async fn admit(
        &self,
        context: &Scope,
        identity: &WorkerIdentity,
    ) -> Result<tokio::sync::OwnedSemaphorePermit> {
        let permit = Arc::clone(&self.permits).acquire_owned();
        tokio::pin!(permit);
        loop {
            context.check_cancelled()?;
            tokio::select! {
                permit=&mut permit=>return permit.map_err(|_|identity.error()),
                ()=tokio::time::sleep(ADMISSION_CHECK)=>{},
            }
        }
    }
}

#[cfg(test)]
mod tests;
