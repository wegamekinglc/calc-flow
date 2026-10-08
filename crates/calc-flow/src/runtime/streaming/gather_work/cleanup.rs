use std::{
    marker::PhantomData,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

use datafusion::execution::memory_pool::MemoryReservation;
use tokio::sync::Notify;

use super::{
    AdmissionResult, GatherHome, GatherScope, GatherStop, OwnedCpuWork, RetirementGuard, Slot,
    WorkAdapter, WorkTicket, cancelled,
};
use crate::{Result, StreamJobContext};

pub(crate) struct AttemptCleanup {
    home: Weak<GatherHome>,
    attempt: u64,
    released: Arc<ReleaseSignal>,
}

struct ReleaseSignal {
    refunded: AtomicBool,
    changed: Notify,
}

struct CreditRelease {
    home: Weak<GatherHome>,
    released: Arc<ReleaseSignal>,
    _retirement: RetirementGuard,
}

impl Drop for CreditRelease {
    fn drop(&mut self) {
        self.released.refunded.store(true, Ordering::Release);
        self.released.changed.notify_waiters();
        if let Some(home) = self.home.upgrade() {
            home.changed.notify_waiters();
        }
    }
}

pub(crate) struct ObservedTicket<T: Send + 'static> {
    ticket: WorkTicket<T>,
    release: CreditRelease,
}

struct TrackedCredit {
    _credit: MemoryReservation,
    _release: CreditRelease,
}

struct ObservedSubmission<W> {
    work: Option<WorkAdapter<W>>,
    credit: MemoryReservation,
    stop: GatherStop,
    retirement: RetirementGuard,
}

pub(crate) struct ObservedOutput<T> {
    value: T,
    _credit: TrackedCredit,
}

impl<T> ObservedOutput<T> {
    pub(crate) fn install(self, install: impl FnOnce(T) -> Result<()>) -> Result<()> {
        install(self.value)
    }
}

pub(crate) const fn cleanup_control_bytes<T: Send + 'static>() -> usize {
    size_of::<AttemptCleanup>()
        + size_of::<ObservedTicket<T>>()
        + size_of::<ObservedOutput<T>>()
        + size_of::<CreditRelease>()
        + size_of::<ReleaseSignal>()
        + size_of::<MemoryReservation>()
        + size_of::<GatherStop>()
        + size_of::<RetirementGuard>()
        + 2 * size_of::<usize>()
        + 64
}

impl<T: Send + 'static> WorkTicket<T> {
    pub(crate) fn observe_cleanup(
        self,
        retirement: RetirementGuard,
    ) -> (ObservedTicket<T>, AttemptCleanup) {
        let home = Arc::downgrade(&self.home);
        let released = Arc::new(ReleaseSignal {
            refunded: AtomicBool::new(false),
            changed: Notify::new(),
        });
        let observer = AttemptCleanup {
            home: home.clone(),
            attempt: self.attempt,
            released: released.clone(),
        };
        (
            ObservedTicket {
                ticket: self,
                release: CreditRelease {
                    home,
                    released,
                    _retirement: retirement,
                },
            },
            observer,
        )
    }
}

impl GatherScope {
    pub(crate) fn submit_observed_work<'a, W: OwnedCpuWork>(
        &'a self,
        work: W,
        credit: MemoryReservation,
        stop: GatherStop,
        retirement: RetirementGuard,
        observer: &'a mut Option<AttemptCleanup>,
    ) -> impl Future<Output = AdmissionResult<ObservedTicket<W::Output>>> + 'a {
        let submission = ObservedSubmission {
            work: Some(WorkAdapter(work)),
            credit,
            stop,
            retirement,
        };
        async move {
            let mut submission = submission;
            let attempt = loop {
                let changed = self.home.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                submission.stop.check()?;
                if let Some(attempt) = self.home.try_install(
                    self,
                    &mut submission.work,
                    &submission.credit,
                    submission.stop.clone(),
                )? {
                    break attempt;
                }
                tokio::select! {
                    () = &mut changed => {},
                    () = submission.stop.wait() => return Err(cancelled(&self.home.run_id).into()),
                }
            };
            let ticket = WorkTicket {
                home: self.home.clone(),
                generation: 0,
                attempt,
                active: true,
                output: PhantomData,
            };
            let (mut ticket, cleanup) = ticket.observe_cleanup(submission.retirement);
            *observer = Some(cleanup);
            let workers = self.home.attempt_workers(attempt)?;
            let generation = self.home.ensure_pool(&submission.stop, workers).await?;
            self.home.dispatch(attempt, generation)?;
            ticket.ticket.generation = generation;
            Ok(ticket)
        }
    }
}

impl<T: Send + 'static> ObservedTicket<T> {
    pub(crate) async fn finish(self) -> Result<ObservedOutput<T>> {
        let output = self.ticket.finish().await?;
        Ok(ObservedOutput {
            value: output.value,
            _credit: TrackedCredit {
                _credit: output.credit,
                _release: self.release,
            },
        })
    }
}

impl AttemptCleanup {
    pub(crate) async fn wait(&self, stop: &GatherStop) -> Result<()> {
        self.wait_checked(stop, || stop.check()).await
    }

    pub(crate) async fn wait_job(&self, stop: &GatherStop, job: &StreamJobContext) -> Result<()> {
        self.wait_checked(stop, || job.check_cancelled()).await
    }

    async fn wait_checked(&self, stop: &GatherStop, check: impl Fn() -> Result<()>) -> Result<()> {
        check()?;
        let home = self.home.upgrade();
        loop {
            let refunded = self.released.changed.notified();
            tokio::pin!(refunded);
            refunded.as_mut().enable();
            let notification = home.as_ref().map(|home| home.changed.notified());
            tokio::pin!(notification);
            if let Some(notification) = notification.as_mut().as_pin_mut() {
                notification.enable();
            }
            check()?;
            let active = self.is_active(home.as_ref());
            if !active && self.released.refunded.load(Ordering::Acquire) {
                return Ok(());
            }
            let slot_changed = async {
                match notification.as_mut().as_pin_mut() {
                    Some(notification) => notification.await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                () = slot_changed => {},
                () = &mut refunded => {},
                () = stop.wait() => {},
            }
        }
    }

    fn is_active(&self, home: Option<&Arc<GatherHome>>) -> bool {
        home.is_some_and(|home| match &home.state.lock().slot {
            Slot::Active(record) | Slot::Parked(record) => record.id == self.attempt,
            Slot::Dropping(id) => *id == self.attempt,
            Slot::Empty => false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CancellationToken;
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

    #[derive(Default)]
    struct WakeCount(std::sync::atomic::AtomicUsize);

    impl std::task::Wake for WakeCount {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn expired_home_does_not_complete_before_output_refund() {
        let released = Arc::new(ReleaseSignal {
            refunded: AtomicBool::new(false),
            changed: Notify::new(),
        });
        let observer = AttemptCleanup {
            home: Weak::new(),
            attempt: 1,
            released: released.clone(),
        };
        let job = StreamJobContext::new(
            1,
            "expired-home",
            crate::JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let stop = GatherStop::from_job(&job);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4_096));
        let credit = MemoryConsumer::new("expired-home-output").register(&pool);
        credit.try_grow(4_096).unwrap();
        let funded = TrackedCredit {
            _credit: credit,
            _release: CreditRelease {
                home: Weak::new(),
                released,
                _retirement: job.gather_owner().retain_retirement().unwrap(),
            },
        };
        let woke = Arc::new(WakeCount::default());
        let waker = std::task::Waker::from(woke.clone());
        let mut context = std::task::Context::from_waker(&waker);
        let mut wait = Box::pin(observer.wait(&stop));
        assert!(
            Future::poll(wait.as_mut(), &mut context).is_pending(),
            "an expired home does not prove credit was refunded"
        );
        assert_eq!(pool.reserved(), 4_096);
        drop(funded);
        assert_eq!(pool.reserved(), 0);
        assert!(
            woke.0.load(Ordering::Relaxed) > 0,
            "actual refund must wake an expired-home observer"
        );
        assert!(matches!(
            Future::poll(wait.as_mut(), &mut context),
            std::task::Poll::Ready(Ok(()))
        ));
    }

    #[test]
    fn job_only_wait_preserves_credit_and_actual_cancel_or_deadline() {
        for deadline in [
            None,
            Some(chrono::Utc::now() - chrono::Duration::seconds(1)),
        ] {
            let job = StreamJobContext::new(
                2,
                "job-only-cleanup",
                crate::JsonMap::new(),
                deadline,
                CancellationToken::new(),
            );
            let stop = GatherStop::from_job(&job);
            stop.abandon();
            let released = Arc::new(ReleaseSignal {
                refunded: AtomicBool::new(false),
                changed: Notify::new(),
            });
            let observer = AttemptCleanup {
                home: Weak::new(),
                attempt: 2,
                released: released.clone(),
            };
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4_096));
            let credit = MemoryConsumer::new("job-only-cleanup-output").register(&pool);
            credit.try_grow(4_096).unwrap();
            let funded = TrackedCredit {
                _credit: credit,
                _release: CreditRelease {
                    home: Weak::new(),
                    released,
                    _retirement: job.gather_owner().retain_retirement().unwrap(),
                },
            };
            let waker = std::task::Waker::from(Arc::new(WakeCount::default()));
            let mut context = std::task::Context::from_waker(&waker);
            let mut ordinary = Box::pin(observer.wait(&stop));
            assert!(matches!(
                ordinary.as_mut().poll(&mut context),
                std::task::Poll::Ready(Err(crate::CalcFlowError::Cancelled { .. }))
            ));
            let mut wait = Box::pin(observer.wait_job(&stop, &job));
            if deadline.is_none() {
                assert!(wait.as_mut().poll(&mut context).is_pending());
                assert_eq!(pool.reserved(), 4_096);
                job.cancellation().cancel();
            }
            assert!(matches!(
                wait.as_mut().poll(&mut context),
                std::task::Poll::Ready(Err(crate::CalcFlowError::Cancelled { run_id })) if run_id == "2"
            ));
            assert_eq!(
                pool.reserved(),
                4_096,
                "stop must not release an escaping output"
            );
            drop(funded);
            assert_eq!(pool.reserved(), 0);
        }
    }
}
