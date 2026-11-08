use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::sync::oneshot;

pub(crate) struct PreparationLease(Option<oneshot::Sender<()>>);
pub(crate) struct PreparationReceipt {
    receiver: oneshot::Receiver<()>,
}
pub(crate) fn channel() -> (PreparationLease, PreparationReceipt) {
    let (sender, receiver) = oneshot::channel();
    (
        PreparationLease(Some(sender)),
        PreparationReceipt { receiver },
    )
}
impl Drop for PreparationLease {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}
impl Future for PreparationReceipt {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        Pin::new(&mut self.receiver).poll(context).map(|_| ())
    }
}
pub(crate) struct Preparation<T> {
    value: Option<T>,
    lease: Option<PreparationLease>,
}
impl<T> Preparation<T> {
    pub(crate) fn new(value: T, lease: PreparationLease) -> Self {
        Self {
            value: Some(value),
            lease: Some(lease),
        }
    }
    pub(crate) fn from_parts(value: T, lease: Option<PreparationLease>) -> Self {
        Self {
            value: Some(value),
            lease,
        }
    }
    pub(crate) fn into_parts(mut self) -> (T, Option<PreparationLease>) {
        let value = self.take_value();
        (value, self.lease.take())
    }
    pub(crate) fn value(&self) -> &T {
        self.value
            .as_ref()
            .expect("preparation retains its candidate")
    }
    pub(crate) fn map<U>(mut self, operation: impl FnOnce(T) -> U) -> Preparation<U> {
        let value = operation(self.take_value());
        Preparation::from_parts(value, self.lease.take())
    }
    pub(crate) fn take_value(&mut self) -> T {
        self.value.take().expect("preparation value is moved once")
    }
}
impl<T, E> Preparation<Result<T, E>> {
    pub(crate) fn transpose(mut self) -> Result<Preparation<T>, E> {
        let value = self.take_value()?;
        Ok(Preparation::from_parts(value, self.lease.take()))
    }
}
impl<T> Drop for Preparation<T> {
    fn drop(&mut self) {
        drop(self.value.take());
        drop(self.lease.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    use tokio::sync::{Semaphore, oneshot};
    struct Input(Arc<AtomicBool>);
    impl Drop for Input {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn test_running_startup_packet_cleanup_waits_for_joined_work_and_input_release() {
        let pool = Arc::new(Semaphore::new(1));
        let permit = Arc::clone(&pool).acquire_owned().await.unwrap();
        let released = Arc::new(AtomicBool::new(false));
        let input = Input(Arc::clone(&released));
        let (guard, mut cleanup) = channel();
        let (entered, ready) = oneshot::channel();
        let (release, wait) = mpsc::channel();
        let worker = tokio::task::spawn_blocking(move || {
            entered.send(()).unwrap();
            wait.recv_timeout(std::time::Duration::from_secs(3))
                .unwrap();
            drop(permit);
            Preparation::new(input, guard)
        });
        let (result, receive) = oneshot::channel();
        let supervisor = tokio::spawn(async move {
            let packet = worker.await.unwrap();
            let _ = result.send(packet);
        });
        let waiter = tokio::spawn(receive);
        ready.await.unwrap();
        waiter.abort();
        let _ = waiter.await;
        assert!(matches!(
            cleanup.receiver.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert!(!released.load(Ordering::SeqCst));
        assert_eq!(pool.available_permits(), 0);
        release.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(3), cleanup)
            .await
            .unwrap();
        supervisor.await.unwrap();
        assert!(released.load(Ordering::SeqCst));
        assert_eq!(pool.available_permits(), 1);
    }
    #[tokio::test]
    async fn test_unstarted_queued_packet_drops_its_input_before_cleanup_receipt_completes() {
        let pool = Arc::new(Semaphore::new(1));
        let held = Arc::clone(&pool).acquire_owned().await.unwrap();
        let released = Arc::new(AtomicBool::new(false));
        let input = Input(Arc::clone(&released));
        let (guard, cleanup) = channel();
        let queued = Preparation::new(move || drop(input), guard);
        let waiting = Arc::clone(&pool);
        let task = tokio::spawn(async move {
            let mut queued = queued;
            let _permit = waiting.acquire_owned().await.unwrap();
            queued.take_value()();
        });
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;
        tokio::time::timeout(std::time::Duration::from_secs(3), cleanup)
            .await
            .unwrap();
        assert!(released.load(Ordering::SeqCst));
        assert_eq!(pool.available_permits(), 0);
        drop(held);
        assert_eq!(pool.available_permits(), 1);
    }
    #[tokio::test]
    async fn test_worker_error_transpose_settles_the_receipt_and_preserves_the_error() {
        let (lease, receipt) = channel();
        let packet = Preparation::new(Err::<Input, _>("worker failed"), lease);
        assert!(matches!(packet.transpose(), Err("worker failed")));
        tokio::time::timeout(std::time::Duration::from_secs(3), receipt)
            .await
            .unwrap();
    }
    #[tokio::test]
    async fn test_successful_candidate_mapping_keeps_the_same_cleanup_obligation_until_handoff() {
        let released = Arc::new(AtomicBool::new(false));
        let (lease, mut receipt) = channel();
        let packet = Preparation::new(Ok::<_, &str>(Input(Arc::clone(&released))), lease);
        let mut candidate = packet.transpose().unwrap().map(|input| vec![input]);
        assert!(matches!(
            receipt.receiver.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert_eq!(candidate.value().len(), 1);
        let transferred = candidate.take_value();
        assert!(matches!(
            receipt.receiver.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        drop(transferred);
        drop(candidate);
        tokio::time::timeout(std::time::Duration::from_secs(3), receipt)
            .await
            .unwrap();
        assert!(released.load(Ordering::SeqCst));
    }
    #[tokio::test]
    async fn test_moving_candidate_parts_does_not_signal_cleanup_during_transfer() {
        let released = Arc::new(AtomicBool::new(false));
        let (lease, mut receipt) = channel();
        let candidate = Preparation::new(Input(Arc::clone(&released)), lease);
        let (value, lease) = candidate.into_parts();
        assert!(matches!(
            receipt.receiver.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        let candidate = Preparation::from_parts(value, lease);
        drop(candidate);
        tokio::time::timeout(std::time::Duration::from_secs(3), receipt)
            .await
            .unwrap();
        assert!(released.load(Ordering::SeqCst));
    }
}
