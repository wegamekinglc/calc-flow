use std::{
    future::{Future, poll_fn},
    pin::{Pin, pin},
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering::SeqCst},
    },
    task::{Context, Poll, Wake, Waker},
};

use futures::task::AtomicWaker;

use super::TaskId;

#[cfg(test)]
mod native_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Phase {
    BeforeRegister,
    Registered,
    Polling,
    BeforeChild(usize),
    AfterChild(usize),
    BeforeIdle,
    Idle,
    Checked,
}

#[cfg(test)]
pub(super) type Observer = Arc<dyn Fn(Phase, &Arc<Shared>, &[Waker; 2]) + Send + Sync>;

#[cfg(test)]
fn observe(observer: Option<&Observer>, phase: Phase, shared: &Arc<Shared>, wakers: &[Waker; 2]) {
    if let Some(observer) = observer {
        observer(phase, shared, wakers);
    }
}

pub(super) struct Shared {
    parent: AtomicWaker,
    polling: AtomicBool,
    closed: AtomicBool,
    ready: [AtomicBool; 2],
    done: [AtomicBool; 2],
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

fn poll_child<F: Future<Output = TaskId>>(
    future: Pin<&mut F>,
    index: usize,
    shared: &Arc<Shared>,
    wakers: &[Waker; 2],
    polled: &mut bool,
    result: &mut Option<TaskId>,
    #[cfg(test)] observer: Option<&Observer>,
) {
    if *polled || result.is_some() || !shared.ready[index].swap(false, SeqCst) {
        return;
    }
    *polled = true;
    #[cfg(test)]
    if let Some(observer) = observer {
        observer(Phase::BeforeChild(index), shared, wakers);
    }
    if let Poll::Ready(id) = future.poll(&mut Context::from_waker(&wakers[index])) {
        shared.done[index].store(true, SeqCst);
        *result = Some(id);
    }
    #[cfg(test)]
    if let Some(observer) = observer {
        observer(Phase::AfterChild(index), shared, wakers);
    }
}

pub(super) async fn run<F, G>(
    first: F,
    second: G,
    #[cfg(test)] observer: Option<Observer>,
) -> [TaskId; 2]
where
    F: Future<Output = TaskId>,
    G: Future<Output = TaskId>,
{
    let owner = Owner(Arc::new(Shared {
        parent: AtomicWaker::new(),
        polling: AtomicBool::new(false),
        closed: AtomicBool::new(false),
        ready: [AtomicBool::new(true), AtomicBool::new(true)],
        done: [AtomicBool::new(false), AtomicBool::new(false)],
    }));
    let wakers = [0, 1].map(|index| {
        Waker::from(Arc::new(ChildWake {
            shared: Arc::downgrade(&owner.0),
            index,
        }))
    });
    let mut first = pin!(first);
    let mut second = pin!(second);
    let mut results = [None, None];
    poll_fn(|context| {
        #[cfg(test)]
        observe(observer.as_ref(), Phase::BeforeRegister, &owner.0, &wakers);
        owner.0.parent.register(context.waker());
        #[cfg(test)]
        observe(observer.as_ref(), Phase::Registered, &owner.0, &wakers);
        owner.0.polling.store(true, SeqCst);
        #[cfg(test)]
        observe(observer.as_ref(), Phase::Polling, &owner.0, &wakers);
        let mut polled = [false, false];
        poll_child(
            second.as_mut(),
            1,
            &owner.0,
            &wakers,
            &mut polled[1],
            &mut results[1],
            #[cfg(test)]
            observer.as_ref(),
        );
        poll_child(
            first.as_mut(),
            0,
            &owner.0,
            &wakers,
            &mut polled[0],
            &mut results[0],
            #[cfg(test)]
            observer.as_ref(),
        );
        poll_child(
            second.as_mut(),
            1,
            &owner.0,
            &wakers,
            &mut polled[1],
            &mut results[1],
            #[cfg(test)]
            observer.as_ref(),
        );
        #[cfg(test)]
        observe(observer.as_ref(), Phase::BeforeIdle, &owner.0, &wakers);
        owner.0.polling.store(false, SeqCst);
        #[cfg(test)]
        observe(observer.as_ref(), Phase::Idle, &owner.0, &wakers);
        if let [Some(first), Some(second)] = results {
            Poll::Ready([first, second])
        } else {
            if owner.0.has_ready() {
                owner.0.parent.wake();
            }
            #[cfg(test)]
            observe(observer.as_ref(), Phase::Checked, &owner.0, &wakers);
            Poll::Pending
        }
    })
    .await
}
