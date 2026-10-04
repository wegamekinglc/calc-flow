use std::{
    future::{Future, poll_fn},
    pin::Pin,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering::SeqCst},
    },
    task::{Context, Poll, Wake, Waker},
};

use futures::task::AtomicWaker;

use super::TaskId;

#[cfg(test)]
mod tests;

#[cfg(test)]
use super::ready_pair::Phase;

#[cfg(test)]
type Observer = Arc<dyn Fn(Phase, &Arc<Shared>, &[Waker]) + Send + Sync>;

#[cfg(test)]
fn observe(observer: Option<&Observer>, phase: Phase, shared: &Arc<Shared>, wakers: &[Waker]) {
    if let Some(observer) = observer {
        observer(phase, shared, wakers);
    }
}

pub(super) struct Shared {
    parent: AtomicWaker,
    polling: AtomicBool,
    closed: AtomicBool,
    ready: Vec<AtomicBool>,
    done: Vec<AtomicBool>,
}

impl Shared {
    fn has_ready(&self) -> bool {
        self.ready
            .iter()
            .zip(&self.done)
            .any(|(ready, done)| !done.load(SeqCst) && ready.load(SeqCst))
    }
}

struct Owner(Arc<Shared>);

impl Drop for Owner {
    fn drop(&mut self) {
        self.0.closed.store(true, SeqCst);
        drop(self.0.parent.take());
    }
}

struct ChildWake {
    shared: Weak<Shared>,
    index: usize,
}

impl Wake for ChildWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        if shared.closed.load(SeqCst) || shared.done[self.index].load(SeqCst) {
            return;
        }
        shared.ready[self.index].store(true, SeqCst);
        if !shared.polling.load(SeqCst) {
            shared.parent.wake();
        }
    }
}

pub(super) async fn run(
    mut children: Vec<Pin<Box<dyn Future<Output = TaskId> + Send + 'static>>>,
    #[cfg(test)] observer: Option<Observer>,
) -> Vec<TaskId> {
    let count = children.len();
    let owner = Owner(Arc::new(Shared {
        parent: AtomicWaker::new(),
        polling: AtomicBool::new(false),
        closed: AtomicBool::new(false),
        ready: (0..count).map(|_| AtomicBool::new(true)).collect(),
        done: (0..count).map(|_| AtomicBool::new(false)).collect(),
    }));
    let wakers: Vec<_> = (0..count)
        .map(|index| {
            Waker::from(Arc::new(ChildWake {
                shared: Arc::downgrade(&owner.0),
                index,
            }))
        })
        .collect();
    let mut results = vec![None; count];
    let mut cursor = 0;
    poll_fn(|context| {
        #[cfg(test)]
        observe(observer.as_ref(), Phase::BeforeRegister, &owner.0, &wakers);
        owner.0.parent.register(context.waker());
        #[cfg(test)]
        observe(observer.as_ref(), Phase::Registered, &owner.0, &wakers);
        owner.0.polling.store(true, SeqCst);
        #[cfg(test)]
        observe(observer.as_ref(), Phase::Polling, &owner.0, &wakers);
        for offset in 0..count {
            let index = (cursor + offset) % count;
            if results[index].is_some() || !owner.0.ready[index].swap(false, SeqCst) {
                continue;
            }
            #[cfg(test)]
            observe(
                observer.as_ref(),
                Phase::BeforeChild(index),
                &owner.0,
                &wakers,
            );
            if let Poll::Ready(id) = children[index]
                .as_mut()
                .poll(&mut Context::from_waker(&wakers[index]))
            {
                owner.0.done[index].store(true, SeqCst);
                results[index] = Some(id);
            }
            #[cfg(test)]
            observe(
                observer.as_ref(),
                Phase::AfterChild(index),
                &owner.0,
                &wakers,
            );
        }
        cursor = (cursor + 1) % count;
        #[cfg(test)]
        observe(observer.as_ref(), Phase::BeforeIdle, &owner.0, &wakers);
        owner.0.polling.store(false, SeqCst);
        #[cfg(test)]
        observe(observer.as_ref(), Phase::Idle, &owner.0, &wakers);
        if results.iter().all(Option::is_some) {
            Poll::Ready(
                results
                    .iter()
                    .map(|id| id.expect("completed member owns its original ID"))
                    .collect(),
            )
        } else {
            let ready = owner.0.has_ready();
            #[cfg(test)]
            observe(observer.as_ref(), Phase::Checked, &owner.0, &wakers);
            if ready {
                context.waker().wake_by_ref();
            }
            Poll::Pending
        }
    })
    .await
}
