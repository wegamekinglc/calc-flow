//! Keep retired-owner refunds observable across dropped handler futures.
use crate::operator::asof::kernel::KernelContext;

use crate::Result;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::Notify;

#[derive(Default)]
struct State {
    pending: AtomicUsize,
    released: Notify,
}

#[derive(Default)]
pub(super) struct Owner(Arc<State>);

pub(super) struct Ticket {
    state: Arc<State>,
    _job: Option<crate::runtime::streaming::gather_work::RetirementGuard>,
}

impl Owner {
    pub fn register(&self, context: &dyn KernelContext) -> Result<Ticket> {
        let job = context
            .stream()
            .map(|c| c.job().gather_owner().retain_retirement())
            .transpose()?;
        self.0.pending.fetch_add(1, Ordering::AcqRel);
        Ok(Ticket {
            state: self.0.clone(),
            _job: job,
        })
    }

    pub async fn wait(&self, context: &dyn KernelContext) -> Result<()> {
        let stop = async {
            let deadline = async {
                match context.deadline() {
                    Some(deadline) => {
                        let delay = (*deadline - chrono::Utc::now())
                            .to_std()
                            .unwrap_or_default();
                        tokio::time::sleep(delay).await;
                    }
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                () = context.cancellation().cancelled() => {},
                () = deadline => {},
            }
        };
        tokio::pin!(stop);
        loop {
            context.check_cancelled()?;
            let released = self.0.released.notified();
            tokio::pin!(released);
            released.as_mut().enable();
            if self.0.pending.load(Ordering::Acquire) == 0 {
                return Ok(());
            }
            tokio::select! {
                biased;
                () = &mut stop => context.check_cancelled()?,
                () = released => {},
            }
        }
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if self.state.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.state.released.notify_waiters();
        }
    }
}
