use crate::{CalcFlowError, Result};
use futures::FutureExt;
use parking_lot::Mutex;
use std::{
    future::{Future, poll_fn},
    panic::AssertUnwindSafe,
    pin::Pin,
    sync::Arc,
    task::Poll,
};
use tokio::{sync::oneshot, task::JoinHandle};

#[derive(Default)]
pub(super) struct WorkOwner {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    closed: bool,
    task: Option<JoinHandle<()>>,
    failures: Vec<CalcFlowError>,
}

impl WorkOwner {
    pub(super) async fn run<T: Send + 'static>(
        self: &Arc<Self>,
        future: impl Future<Output = Result<T>> + Send + 'static,
    ) -> Result<T> {
        let (sender, receiver) = oneshot::channel();
        {
            let mut state = self.state.lock();
            if state.closed || state.task.is_some() {
                return Err(super::budget::limit_error(
                    "operation",
                    "history is closed or busy",
                ));
            }
            let owner = Arc::downgrade(self);
            state.task = Some(tokio::spawn(async move {
                let result = AssertUnwindSafe(future)
                    .catch_unwind()
                    .await
                    .unwrap_or_else(|_| {
                        Err(CalcFlowError::Internal {
                            message: "source-history task panicked".into(),
                        })
                    });
                if let Err(Err(error)) = sender.send(result)
                    && let Some(owner) = owner.upgrade()
                {
                    owner.state.lock().failures.push(error);
                }
            }));
        }
        self.join().await?;
        receiver.await.map_err(|_| CalcFlowError::Internal {
            message: "source-history task ended without a result".into(),
        })?
    }

    pub(super) fn when_idle<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        let state = self.state.lock();
        if state.closed || state.task.is_some() {
            return Err(super::budget::limit_error(
                "operation",
                "history is closed or busy",
            ));
        }
        operation()
    }

    async fn join(&self) -> Result<()> {
        poll_fn(|cx| {
            let mut state = self.state.lock();
            let Some(task) = state.task.as_mut() else {
                return Poll::Ready(Ok(()));
            };
            match Pin::new(task).poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(result) => {
                    state.task = None;
                    Poll::Ready(result.map_err(|error| CalcFlowError::Internal {
                        message: format!("source-history task join failed: {error}"),
                    }))
                }
            }
        })
        .await
    }

    pub(super) async fn drain(&self) -> Vec<CalcFlowError> {
        self.state.lock().closed = true;
        if let Err(error) = self.join().await {
            self.state.lock().failures.push(error);
        }
        std::mem::take(&mut self.state.lock().failures)
    }
}
