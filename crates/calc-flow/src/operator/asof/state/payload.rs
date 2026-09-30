use super::{BatchKey, PayloadBatch, RowPayload};
use ahash::RandomState;
use std::{collections::HashMap, num::NonZeroU32, ops::Index, sync::Arc};

/// A retained row owns no Arrow buffers or reference counts. The nonzero
/// batch handle also keeps an optional row reference eight bytes wide.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) struct RowRef {
    batch: NonZeroU32,
    pub row: u32,
}

#[cfg(test)]
impl RowRef {
    pub fn fixture(row: u32) -> Self {
        Self {
            batch: NonZeroU32::MIN,
            row,
        }
    }
}

const _: () = assert!(size_of::<RowRef>() == 8 && size_of::<Option<RowRef>>() == 8);

#[derive(Clone, Copy)]
pub(in super::super) struct PayloadView<'a> {
    pub batch: &'a PayloadBatch,
    pub row: usize,
}

impl RowPayload {
    #[cfg(test)]
    pub fn view(&self) -> PayloadView<'_> {
        PayloadView {
            batch: &self.batch,
            row: self.row,
        }
    }
}

/// Own each payload batch once. Handles remain stable while any row is live;
/// expired entries release both indexes and their Arrow owner immediately.
#[derive(Clone, Default)]
pub(in super::super) struct PayloadPool {
    by_key: HashMap<BatchKey, NonZeroU32, RandomState>,
    by_id: HashMap<NonZeroU32, (Arc<PayloadBatch>, usize), RandomState>,
    next_id: u32,
}

impl PayloadPool {
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn values(&self) -> impl Iterator<Item = &(Arc<PayloadBatch>, usize)> {
        self.by_id.values()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&BatchKey, &(Arc<PayloadBatch>, usize))> {
        self.by_key.iter().map(|(key, id)| (key, &self.by_id[id]))
    }

    pub fn attach(&mut self, row: &RowPayload) -> RowRef {
        let row_index = u32::try_from(row.row).expect("preflighted ASOF payload row index");
        let id = if let Some(id) = self.by_key.get(&row.batch.key) {
            *id
        } else {
            let id = self.allocate_id();
            self.by_key.insert(row.batch.key, id);
            self.by_id.insert(id, (row.batch.clone(), 0));
            id
        };
        self.by_id.get_mut(&id).expect("indexed ASOF batch").1 += 1;
        RowRef {
            batch: id,
            row: row_index,
        }
    }

    fn allocate_id(&mut self) -> NonZeroU32 {
        loop {
            self.next_id = self.next_id.wrapping_add(1).max(1);
            let id = NonZeroU32::new(self.next_id).expect("nonzero ASOF batch handle");
            if !self.by_id.contains_key(&id) {
                return id;
            }
        }
    }

    pub fn view(&self, row: RowRef) -> PayloadView<'_> {
        PayloadView {
            batch: &self.by_id[&row.batch].0,
            row: row.row as usize,
        }
    }

    pub fn key(&self, row: RowRef) -> BatchKey {
        self.by_id[&row.batch].0.key
    }

    pub fn detach(&mut self, row: RowRef) {
        self.detach_id(row.batch, 1);
    }

    pub fn detach_count(&mut self, key: BatchKey, removed: usize) {
        let id = self.by_key[&key];
        self.detach_id(id, removed);
    }

    fn detach_id(&mut self, id: NonZeroU32, removed: usize) {
        let references = &mut self.by_id.get_mut(&id).expect("indexed ASOF batch").1;
        *references -= removed;
        if *references == 0 {
            let key = self.by_id[&id].0.key;
            self.by_key.remove(&key);
            self.by_id.remove(&id);
            if self.by_id.is_empty() {
                *self = Self::default();
            } else if self.by_id.capacity() > 4 * self.by_id.len() {
                self.by_key.shrink_to_fit();
                self.by_id.shrink_to_fit();
            }
        }
    }
}

impl Index<&BatchKey> for PayloadPool {
    type Output = (Arc<PayloadBatch>, usize);

    fn index(&self, key: &BatchKey) -> &Self::Output {
        &self.by_id[&self.by_key[key]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};
    use std::sync::OnceLock;

    fn row(key: u64) -> RowPayload {
        RowPayload {
            batch: Arc::new(PayloadBatch {
                key: (1, key),
                record: Arc::new(RecordBatch::new_empty(Arc::new(Schema::empty()))),
                encoded: OnceLock::new(),
                encoded_charge_bytes: 0,
                body_bytes: 0,
            }),
            row: 0,
        }
    }

    #[test]
    fn live_batch_survives_handle_wrap_and_other_batch_release() {
        let mut pool = PayloadPool::default();
        let first = pool.attach(&row(10));
        pool.next_id = u32::MAX;
        let second = pool.attach(&row(20));
        assert_ne!(first.batch, second.batch);
        pool.detach(first);
        assert_eq!(pool.view(second).batch.key, (1, 20));
        assert_eq!(pool.len(), 1);
        pool.detach(second);
        assert_eq!(pool.by_key.capacity(), 0);
        assert_eq!(pool.by_id.capacity(), 0);
    }

    #[test]
    fn sparse_pool_allocations_fit_existing_batch_header_charge() {
        for count in [1, 2, 3, 4, 16, 64, 1_000] {
            let rows = (0..count).map(row).collect::<Vec<_>>();
            let mut pool = PayloadPool::default();
            let measured = allocation_counter::measure(|| {
                for row in &rows {
                    pool.attach(row);
                }
            });
            assert!(
                measured.bytes_current <= i64::try_from(256 * count).unwrap(),
                "batch pool allocations exceed legacy header charge: {count}, {measured:?}"
            );
            for row in pool.iter().map(|(key, _)| *key).collect::<Vec<_>>() {
                pool.detach_count(row, 1);
            }
            assert!(pool.is_empty());
        }
    }
}
