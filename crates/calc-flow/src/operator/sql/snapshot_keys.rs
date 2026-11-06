//! Shared key columns for repeated full snapshots.
//!
//! Group slots are append-only and a group's key never changes, so the key
//! columns of a sealed output chunk are identical in every later snapshot.
//! Freshly built key columns are staged while a transaction prepares and are
//! installed only when that transaction commits; a refused or failed output
//! leaves nothing behind. Installed bytes are reserved from the incremental
//! memory pool, and a refused reservation simply leaves the chunk uncached.

use std::sync::Arc;

use datafusion::{
    arrow::array::{Array, ArrayRef},
    execution::memory_pool::MemoryReservation,
};
use parking_lot::Mutex;

pub(super) struct SnapshotKeys {
    inner: Mutex<State>,
}

struct State {
    staged: Vec<Chunk>,
    chunks: Vec<Option<Chunk>>,
    reservation: MemoryReservation,
}

struct Chunk {
    ordinal: usize,
    rows: usize,
    first_key: Arc<[u8]>,
    bytes: usize,
    columns: Arc<[ArrayRef]>,
}

impl SnapshotKeys {
    pub(super) fn new(reservation: MemoryReservation) -> Self {
        Self {
            inner: Mutex::new(State {
                staged: Vec::new(),
                chunks: Vec::new(),
                reservation,
            }),
        }
    }

    /// Drops entries staged by a transaction that never committed.
    pub(super) fn begin(&self) {
        self.inner.lock().staged.clear();
    }

    /// Key columns installed for `chunk` when its row count and first group
    /// are unchanged since the committing snapshot that built them.
    pub(super) fn get(
        &self,
        chunk: usize,
        rows: usize,
        first_key: &Arc<[u8]>,
    ) -> Option<Arc<[ArrayRef]>> {
        let inner = self.inner.lock();
        let cached = inner.chunks.get(chunk)?.as_ref()?;
        (cached.rows == rows && Arc::ptr_eq(&cached.first_key, first_key))
            .then(|| Arc::clone(&cached.columns))
    }

    /// Stages freshly built key columns for the running transaction.
    pub(super) fn stage(
        &self,
        chunk: usize,
        rows: usize,
        first_key: Arc<[u8]>,
        columns: Arc<[ArrayRef]>,
    ) {
        let bytes = columns
            .iter()
            .fold(0usize, |total, column: &Arc<dyn Array>| {
                total.saturating_add(column.get_array_memory_size())
            });
        self.inner.lock().staged.push(Chunk {
            ordinal: chunk,
            rows,
            first_key,
            bytes,
            columns,
        });
    }

    /// Installs every staged chunk, replacing its previous entry; a chunk
    /// whose bytes the pool refuses stays uncached.
    pub(super) fn install_staged(&self) {
        let staged = std::mem::take(&mut self.inner.lock().staged);
        if staged.is_empty() {
            return;
        }
        let mut inner = self.inner.lock();
        for staged in &staged {
            if inner.chunks.len() <= staged.ordinal {
                inner.chunks.resize_with(staged.ordinal + 1, || None);
            }
            if let Some(previous) = inner.chunks[staged.ordinal].take() {
                inner.reservation.shrink(previous.bytes);
            }
        }
        for staged in staged {
            let ordinal = staged.ordinal;
            if inner.reservation.try_grow(staged.bytes).is_ok() {
                inner.chunks[ordinal] = Some(staged);
            }
        }
    }
}
