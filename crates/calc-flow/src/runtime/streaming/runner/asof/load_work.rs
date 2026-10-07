use std::{future::Future, sync::Arc};

use parking_lot::Mutex;
use tokio::{sync::Notify, task::JoinHandle};

use crate::{CalcFlowError, OperatorStateSnapshot, Result};

type LoadHandle = JoinHandle<Result<OperatorStateSnapshot>>;

#[derive(Clone, Default)]
pub(crate) struct LoadOwner(Arc<Home>);

#[derive(Default)]
struct Home {
    state: Mutex<State>,
    changed: Notify,
}

#[derive(Default)]
struct State {
    closed: bool,
    loaned: bool,
    handle: Option<LoadHandle>,
}

struct Loan {
    owner: LoadOwner,
    handle: Option<LoadHandle>,
}

impl LoadOwner {
    pub(crate) async fn load(
        &self,
        future: impl Future<Output = Result<OperatorStateSnapshot>> + Send + 'static,
    ) -> Result<OperatorStateSnapshot> {
        let mut loan = {
            let mut state = self.0.state.lock();
            if state.closed {
                return Err(CalcFlowError::Cancelled {
                    run_id: "checkpoint:state-load".into(),
                });
            }
            if state.loaned || state.handle.is_some() {
                return Err(CalcFlowError::Internal {
                    message: "operator checkpoint load is already active".into(),
                });
            }
            let handle = tokio::spawn(future);
            state.loaned = true;
            Loan {
                owner: self.clone(),
                handle: Some(handle),
            }
        };
        loan.join().await
    }

    pub(crate) async fn close_and_drain(&self) -> Option<CalcFlowError> {
        self.0.state.lock().closed = true;
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let loan = {
                let mut state = self.0.state.lock();
                if state.loaned {
                    None
                } else if let Some(handle) = state.handle.take() {
                    state.loaned = true;
                    Some(Loan {
                        owner: self.clone(),
                        handle: Some(handle),
                    })
                } else {
                    return None;
                }
            };
            if let Some(mut loan) = loan {
                return loan
                    .join()
                    .await
                    .err()
                    .filter(|error| !matches!(error, CalcFlowError::Cancelled { .. }));
            }
            changed.await;
        }
    }
}

impl Loan {
    async fn join(&mut self) -> Result<OperatorStateSnapshot> {
        let result = self.handle.as_mut().expect("owned load handle").await;
        self.handle = None;
        result.map_err(|error| {
            if error.is_panic() {
                CalcFlowError::Internal {
                    message: format!(
                        "operator checkpoint load panicked: {}",
                        crate::runtime::streaming::failure::panic_message(
                            error.into_panic().as_ref(),
                        ),
                    ),
                }
            } else {
                CalcFlowError::Cancelled {
                    run_id: "checkpoint:state-load".into(),
                }
            }
        })?
    }
}

impl Drop for Loan {
    fn drop(&mut self) {
        let mut state = self.owner.0.state.lock();
        debug_assert!(state.loaned && state.handle.is_none());
        state.handle = self.handle.take();
        state.loaned = false;
        self.owner.0.changed.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_shared_operator_load_busy_diagnostic_has_no_asof_label() {
        let owner = LoadOwner::default();
        let (entered, entrance) = tokio::sync::oneshot::channel();
        let (released, release) = tokio::sync::oneshot::channel();
        let mut loading = Box::pin(owner.load(async move {
            entered.send(()).unwrap();
            let _ = release.await;
            Ok(OperatorStateSnapshot::default())
        }));
        tokio::select! {
            _ = entrance => {},
            result = &mut loading => panic!("load missed the actual gate: {result:?}"),
        }

        let failure = owner
            .load(std::future::ready(Ok(OperatorStateSnapshot::default())))
            .await
            .unwrap_err();
        released.send(()).unwrap();
        loading.await.unwrap();
        assert!(owner.close_and_drain().await.is_none());
        assert!(matches!(failure, CalcFlowError::Internal { message }
            if message == "operator checkpoint load is already active"));
    }

    #[tokio::test]
    async fn test_shared_operator_load_panic_diagnostic_preserves_original_message() {
        let owner = LoadOwner::default();
        let failure = owner
            .load(async { panic!("Join loader panic sentinel") })
            .await
            .unwrap_err();
        assert!(owner.close_and_drain().await.is_none());
        assert!(matches!(failure, CalcFlowError::Internal { message }
            if message == "operator checkpoint load panicked: Join loader panic sentinel"));
    }
}
