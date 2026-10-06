use std::{cell::RefCell, collections::BTreeMap, sync::Arc};

use parking_lot::Mutex;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Counts {
    pub(crate) constructions: usize,
    pub(crate) enqueues: usize,
    pub(crate) dequeues: usize,
}

type SharedCounts = Arc<Mutex<BTreeMap<String, Counts>>>;

thread_local! {
    static ACTIVE: RefCell<Option<SharedCounts>> = const { RefCell::new(None) };
}

pub(crate) struct Observer(SharedCounts);

impl Observer {
    pub(crate) fn start() -> Self {
        let counts = Arc::new(Mutex::new(BTreeMap::new()));
        ACTIVE.with(|active| {
            assert!(active.borrow().is_none());
            *active.borrow_mut() = Some(counts.clone());
        });
        Self(counts)
    }

    pub(crate) fn snapshot(&self) -> BTreeMap<String, Counts> {
        self.0.lock().clone()
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            active.borrow_mut().take();
        });
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Event {
    Construct,
    Enqueue,
    Dequeue,
}

pub(crate) fn record(edge: &str, event: Event) {
    ACTIVE.with(|active| {
        if let Some(counts) = active.borrow().as_ref() {
            let mut counts = counts.lock();
            let counts = counts.entry(edge.into()).or_default();
            match event {
                Event::Construct => counts.constructions += 1,
                Event::Enqueue => counts.enqueues += 1,
                Event::Dequeue => counts.dequeues += 1,
            }
        }
    });
}
