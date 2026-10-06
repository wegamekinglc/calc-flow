use std::{
    collections::BTreeMap,
    sync::{
        Arc, LazyLock, Mutex, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use datafusion::execution::memory_pool::MemoryReservation;

use super::{BatchMetadata, RecordBatch, Result, RetainedSqlInput, SqlOperator};

type Key = (String, String, String);
type Hook = Box<dyn FnOnce(Candidate) -> Result<()> + Send>;

pub(crate) struct Candidate {
    pub(crate) metadata: BatchMetadata,
    pub(crate) records: Vec<RecordBatch>,
    pub(crate) rows: u64,
    pub(crate) bytes: u64,
    pub(crate) reserved: usize,
    pub(crate) backing: Vec<Weak<MemoryReservation>>,
}

struct Entry {
    prepared: Option<Hook>,
    installs: Arc<AtomicUsize>,
}

static HOOKS: LazyLock<Mutex<BTreeMap<Key, Entry>>> = LazyLock::new(|| Mutex::new(BTreeMap::new()));

pub(crate) struct Registration {
    key: Key,
    installs: Arc<AtomicUsize>,
}

impl Registration {
    pub(crate) fn installs(&self) -> usize {
        self.installs.load(Ordering::SeqCst)
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        HOOKS.lock().unwrap().remove(&self.key);
    }
}

impl SqlOperator {
    pub(crate) fn on_prepared_restore_for_test(
        &self,
        metadata: &BatchMetadata,
        hook: impl FnOnce(Candidate) -> Result<()> + Send + 'static,
    ) -> Registration {
        let key = key(self, metadata);
        let installs = Arc::new(AtomicUsize::new(0));
        assert!(
            HOOKS
                .lock()
                .unwrap()
                .insert(
                    key.clone(),
                    Entry {
                        prepared: Some(Box::new(hook)),
                        installs: installs.clone(),
                    }
                )
                .is_none()
        );
        Registration { key, installs }
    }
}

fn key(operator: &SqlOperator, metadata: &BatchMetadata) -> Key {
    (
        operator.name.clone(),
        crate::json::canonical_json(&serde_json::to_value(metadata).unwrap()).unwrap(),
        operator.query.clone(),
    )
}

pub(super) fn prepared(operator: &SqlOperator, retained: &RetainedSqlInput) -> Result<()> {
    let key = key(operator, &retained.metadata);
    let hook = HOOKS
        .lock()
        .unwrap()
        .get_mut(&key)
        .and_then(|entry| entry.prepared.take());
    if let Some(hook) = hook {
        let reserved = retained
            .backing_reservations
            .iter()
            .map(|lease| lease.size())
            .sum::<usize>()
            + retained
                .reservation
                .as_ref()
                .map_or(0, MemoryReservation::size);
        hook(Candidate {
            metadata: retained.metadata.clone(),
            records: retained.records.clone(),
            rows: retained.rows,
            bytes: retained.bytes,
            reserved,
            backing: retained
                .backing_reservations
                .iter()
                .map(Arc::downgrade)
                .collect(),
        })?;
    }
    Ok(())
}

pub(super) fn installing(operator: &SqlOperator, retained: &RetainedSqlInput) {
    installing_metadata(operator, &retained.metadata);
}

pub(super) fn installing_metadata(operator: &SqlOperator, metadata: &BatchMetadata) {
    let key = key(operator, metadata);
    let counter = HOOKS
        .lock()
        .unwrap()
        .get(&key)
        .map(|entry| entry.installs.clone());
    if let Some(counter) = counter {
        counter.fetch_add(1, Ordering::SeqCst);
    }
}

pub(super) fn prepared_compact(
    operator: &SqlOperator,
    state: &super::compact::CompactSqlState,
    native: &super::incremental::IncrementalSql,
) -> Result<()> {
    let key = key(operator, &state.metadata);
    let hook = HOOKS
        .lock()
        .unwrap()
        .get_mut(&key)
        .and_then(|entry| entry.prepared.take());
    if let Some(hook) = hook {
        let exported = native.export_native_state(&operator.name, || Ok(()))?;
        let copies = super::record_copy_reservation(
            operator.retention_runtime()?,
            &operator.name,
            exported.records().len(),
            exported.descriptor.wire_schema.fields().len(),
            1,
        )?;
        let (reserved, backing) = state.recovery_credit();
        hook(Candidate {
            metadata: state.metadata.clone(),
            records: exported.records().to_vec(),
            rows: state.ledger.rows,
            bytes: state.ledger.bytes,
            reserved: reserved + exported.reserved_bytes() + copies.size(),
            backing,
        })?;
    }
    Ok(())
}
