use std::{
    any::Any,
    collections::BTreeMap,
    sync::{
        Arc, LazyLock, Mutex, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use datafusion::execution::memory_pool::{MemoryPool, MemoryReservation};

use super::{BatchMetadata, Result, SqlOperator, compact::CompactCapture};

type Key = (String, String, String);
type Hook = Box<dyn FnOnce(Candidate) -> Result<()> + Send>;

pub(crate) struct Candidate {
    pub(crate) metadata: BatchMetadata,
    pub(crate) thread: std::thread::ThreadId,
    pub(crate) layout: u64,
    pub(crate) segments: Vec<String>,
    pub(crate) operator: Weak<()>,
    pub(crate) capture: Weak<dyn Any + Send + Sync>,
    pub(crate) fee: Weak<MemoryReservation>,
    pub(crate) reserved: usize,
    pub(crate) pool: Arc<dyn MemoryPool>,
}

struct Entry {
    prepared: Option<Hook>,
    installs: Arc<AtomicUsize>,
    acks: Arc<AtomicUsize>,
    exits: Arc<AtomicUsize>,
    thread: Option<std::thread::ThreadId>,
}

static HOOKS: LazyLock<Mutex<BTreeMap<Key, Entry>>> = LazyLock::new(|| Mutex::new(BTreeMap::new()));

pub(crate) struct Registration {
    key: Key,
    installs: Arc<AtomicUsize>,
    acks: Arc<AtomicUsize>,
    exits: Arc<AtomicUsize>,
}

impl Registration {
    pub(crate) fn installs(&self) -> usize {
        self.installs.load(Ordering::SeqCst)
    }

    pub(crate) fn acks(&self) -> usize {
        self.acks.load(Ordering::SeqCst)
    }

    pub(crate) fn exits(&self) -> usize {
        self.exits.load(Ordering::SeqCst)
    }

    pub(crate) fn counts(&self) -> [Arc<AtomicUsize>; 3] {
        [self.installs.clone(), self.acks.clone(), self.exits.clone()]
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        HOOKS.lock().unwrap().remove(&self.key);
    }
}

impl SqlOperator {
    pub(crate) fn on_prepared_checkpoint_for_test(
        &self,
        metadata: &BatchMetadata,
        hook: impl FnOnce(Candidate) -> Result<()> + Send + 'static,
    ) -> Registration {
        let key = key(self, metadata);
        let installs = Arc::new(AtomicUsize::new(0));
        let acks = Arc::new(AtomicUsize::new(0));
        let exits = Arc::new(AtomicUsize::new(0));
        assert!(
            HOOKS
                .lock()
                .unwrap()
                .insert(
                    key.clone(),
                    Entry {
                        prepared: Some(Box::new(hook)),
                        installs: installs.clone(),
                        acks: acks.clone(),
                        exits: exits.clone(),
                        thread: None,
                    },
                )
                .is_none()
        );
        Registration {
            key,
            installs,
            acks,
            exits,
        }
    }

    pub(crate) fn checkpoint_acknowledged_for_test(&self) {
        if let Some(state) = &self.compact {
            let registry = HOOKS.lock().unwrap();
            if let Some(entry) = registry.get(&key(self, &state.metadata)) {
                entry.acks.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}

fn key(operator: &SqlOperator, metadata: &BatchMetadata) -> Key {
    (
        operator.name.clone(),
        crate::json::canonical_json(&serde_json::to_value(metadata).unwrap()).unwrap(),
        operator.query.clone(),
    )
}

pub(super) fn prepared(operator: &SqlOperator, capture: &Arc<CompactCapture>) -> Result<()> {
    let fee = capture.checkpoint_fee_for_test();
    let metadata = &operator.compact.as_ref().unwrap().metadata;
    let hook = HOOKS
        .lock()
        .unwrap()
        .get_mut(&key(operator, metadata))
        .and_then(|entry| {
            entry
                .prepared
                .take()
                .map(|hook| (hook, entry.thread.unwrap()))
        });
    if let Some((hook, thread)) = hook {
        let opaque: Arc<dyn Any + Send + Sync> = capture.clone();
        let weak = Arc::downgrade(&opaque);
        drop(opaque);
        hook(Candidate {
            metadata: metadata.clone(),
            thread,
            layout: capture.snapshot.inline_metadata["state_layout"]
                .as_u64()
                .unwrap(),
            segments: capture.snapshot.segments.keys().cloned().collect(),
            operator: Arc::downgrade(operator.checkpoint_test_owner.get_or_init(|| Arc::new(()))),
            capture: weak,
            fee: Arc::downgrade(fee),
            reserved: fee.size(),
            pool: operator.retention_runtime()?.incremental_memory_pool(),
        })?;
    }
    Ok(())
}

pub(super) fn installed(operator: &SqlOperator) {
    if let Some(state) = &operator.compact {
        let registry = HOOKS.lock().unwrap();
        if let Some(entry) = registry.get(&key(operator, &state.metadata)) {
            entry.installs.fetch_add(1, Ordering::SeqCst);
        }
    }
}

pub(super) struct WorkExit(Option<Key>);

impl WorkExit {
    pub(super) fn new(operator: &SqlOperator) -> Self {
        let key = operator
            .compact
            .as_ref()
            .map(|state| key(operator, &state.metadata));
        if let Some(key) = &key
            && let Some(entry) = HOOKS.lock().unwrap().get_mut(key)
        {
            entry.thread = Some(std::thread::current().id());
        }
        Self(key)
    }
}

impl Drop for WorkExit {
    fn drop(&mut self) {
        if let Some(key) = &self.0 {
            let registry = HOOKS.lock().unwrap();
            if let Some(entry) = registry.get(key) {
                entry.exits.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}
