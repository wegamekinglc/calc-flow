use super::{BatchKey, PayloadBatch, RowPayload};
use ahash::RandomState;
use hashbrown::HashTable;
use std::{num::NonZeroU32, ops::Index, sync::Arc};

#[cfg(test)]
thread_local! {
    static COMPACTION_MOVES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// A retained row owns no Arrow buffers or reference counts. The nonzero
/// batch handle also keeps an optional row reference eight bytes wide.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) struct RowRef {
    batch: NonZeroU32,
    pub row: u32,
}

impl RowRef {
    pub fn batch_handle(self) -> u32 {
        self.batch.get()
    }
    pub fn with_row(self, row: u32) -> Self {
        Self { row, ..self }
    }

    #[cfg(test)]
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
#[derive(Clone)]
pub(in super::super) struct PayloadPool {
    by_key: HashTable<KeyEntry>,
    by_id: HashTable<IdEntry>,
    hasher: RandomState,
    next_id: u32,
    defer_compaction: bool,
}

#[derive(Clone)]
struct KeyEntry {
    key: BatchKey,
    id: NonZeroU32,
}

#[derive(Clone)]
struct IdEntry {
    id: NonZeroU32,
    value: (Arc<PayloadBatch>, usize),
}

impl Default for PayloadPool {
    fn default() -> Self {
        Self {
            by_key: HashTable::new(),
            by_id: HashTable::new(),
            hasher: RandomState::new(),
            next_id: 0,
            defer_compaction: false,
        }
    }
}

pub(in super::super) fn backing_buckets<T>(table: &HashTable<T>) -> usize {
    let estimate = table.allocation_size() / (size_of::<T>() + 1);
    if estimate == 0 {
        0
    } else {
        1 << (usize::BITS - 1 - estimate.leading_zeros())
    }
}

pub(in super::super) fn bucket_capacity(buckets: usize) -> usize {
    if buckets <= 8 {
        buckets.saturating_sub(1)
    } else {
        buckets / 8 * 7
    }
}

pub(in super::super) fn required_backing(entries: usize) -> usize {
    if entries == 0 {
        0
    } else if entries <= 3 {
        4
    } else {
        (entries * 8).div_ceil(7).next_power_of_two()
    }
}

fn charge(buckets: usize, slot: usize) -> u64 {
    if buckets == 0 {
        0
    } else {
        (buckets * (slot + 1) + 64) as u64
    }
}

fn projected_backing<T>(table: &HashTable<T>, additional: usize) -> usize {
    let buckets = backing_buckets(table);
    if additional <= table.capacity() - table.len() {
        return buckets;
    }
    let required = table.len() + additional;
    let capacity = bucket_capacity(buckets);
    if required <= capacity / 2 {
        buckets
    } else {
        required_backing(required.max(capacity + 1))
    }
}

pub(in super::super) struct PayloadAdmission {
    pub metadata_bytes: u64,
    pub payload_bytes: u64,
    pub workspace: u64,
    pub new_batches: usize,
}

pub(in super::super) struct PayloadRemoval {
    pub metadata_bytes: u64,
    pub released_bytes: u64,
    pub workspace: u64,
    buckets: [usize; 2],
    pub replace: bool,
    pub remaining: usize,
}

pub(in super::super) struct PreparedPayloadRemoval {
    replacement: Option<PayloadPool>,
    inputs: Vec<(u32, Arc<PayloadBatch>, usize)>,
    workspace: Option<datafusion::execution::memory_pool::MemoryReservation>,
    retirement: Option<tokio::sync::oneshot::Sender<RetiredPool>>,
}

struct RetiredPool {
    _pool: Option<PayloadPool>,
    _inputs: Vec<(u32, Arc<PayloadBatch>, usize)>,
    _workspace: Option<datafusion::execution::memory_pool::MemoryReservation>,
}

impl PreparedPayloadRemoval {
    pub fn empty() -> Self {
        Self {
            replacement: None,
            inputs: Vec::new(),
            workspace: None,
            retirement: None,
        }
    }

    pub fn capture(
        pool: &PayloadPool,
        layout: &PayloadRemoval,
        workspace: datafusion::execution::memory_pool::MemoryReservation,
    ) -> Self {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        // Create the retirement task before delivery. Sending the old pool at
        // commit only transfers pointers; all table/batch destruction follows
        // on a blocking worker, with the same retained-input reservation.
        tokio::spawn(async move {
            if let Ok(retired) = receiver.await {
                tokio::task::spawn_blocking(move || drop(retired))
                    .await
                    .ok();
            }
        });
        Self {
            replacement: Some(PayloadPool {
                hasher: pool.hasher.clone(),
                next_id: pool.next_id,
                ..PayloadPool::default()
            }),
            inputs: Vec::with_capacity(layout.remaining),
            workspace: Some(workspace),
            retirement: Some(sender),
        }
    }

    pub fn retain(&mut self, id: u32, batch: &Arc<PayloadBatch>, references: usize) {
        self.inputs.push((id, batch.clone(), references));
    }

    pub fn populate(&mut self, layout: &PayloadRemoval) {
        let pool = self.replacement.as_mut().expect("prepared replacement");
        pool.by_key = HashTable::with_capacity(bucket_capacity(layout.buckets[0]));
        pool.by_id = HashTable::with_capacity(bucket_capacity(layout.buckets[1]));
        for (id, batch, references) in self.inputs.drain(..) {
            #[cfg(test)]
            COMPACTION_MOVES.with(|moves| moves.set(moves.get() + 2));
            let id = NonZeroU32::new(id).expect("live batch handle");
            pool.by_key.insert_unique(
                pool.hasher.hash_one(batch.key),
                KeyEntry { key: batch.key, id },
                |entry| pool.hasher.hash_one(entry.key),
            );
            pool.by_id.insert_unique(
                u64::from(id.get()),
                IdEntry {
                    id,
                    value: (batch, references),
                },
                |entry| u64::from(entry.id.get()),
            );
        }
    }
}

impl Drop for PreparedPayloadRemoval {
    fn drop(&mut self) {
        let retired = RetiredPool {
            _pool: self.replacement.take(),
            _inputs: std::mem::take(&mut self.inputs),
            _workspace: self.workspace.take(),
        };
        if let Some(sender) = self.retirement.take() {
            sender.send(retired).ok();
        }
    }
}

impl PayloadPool {
    pub fn references(&self, key: &BatchKey) -> usize {
        self.by_key
            .find(self.hasher.hash_one(key), |entry| &entry.key == key)
            .map_or(0, |entry| self.by_id_entry(entry.id).value.1)
    }

    pub fn checkpoint_right_handles(&self) -> impl Iterator<Item = (u32, u64)> + '_ {
        self.by_id
            .iter()
            .filter(|entry| entry.value.0.key.0 == 1)
            .map(|entry| (entry.id.get(), entry.value.0.key.1))
    }
    pub fn with_backing_buckets(key_buckets: usize, id_buckets: usize) -> Self {
        Self {
            by_key: HashTable::with_capacity(bucket_capacity(key_buckets)),
            by_id: HashTable::with_capacity(bucket_capacity(id_buckets)),
            ..Self::default()
        }
    }

    pub fn backing_buckets(&self) -> (usize, usize) {
        (backing_buckets(&self.by_key), backing_buckets(&self.by_id))
    }

    pub fn metadata_bytes(&self) -> u64 {
        // Fixed control/alignment headroom keeps accounting portable while the
        // bucket count comes from the real allocation, including tombstones.
        let (keys, ids) = self.backing_buckets();
        charge(keys, size_of::<KeyEntry>()) + charge(ids, size_of::<IdEntry>())
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn project_admission(
        &self,
        batches: &[Arc<PayloadBatch>],
        name: &str,
    ) -> crate::Result<PayloadAdmission> {
        let mut new_batches = 0;
        let mut payload_bytes = 0;
        for batch in batches {
            if self
                .by_key
                .find(self.hasher.hash_one(batch.key), |entry| {
                    entry.key == batch.key
                })
                .is_some()
            {
                continue;
            }
            new_batches += 1;
            payload_bytes = super::super::checked(
                name,
                payload_bytes,
                super::capacity_batch_allocation(batch, name)?,
            )?;
        }
        let (old_keys, old_ids) = self.backing_buckets();
        let keys = projected_backing(&self.by_key, new_batches);
        let ids = projected_backing(&self.by_id, new_batches);
        let metadata_bytes =
            charge(keys, size_of::<KeyEntry>()) + charge(ids, size_of::<IdEntry>());
        let workspace = if keys == old_keys {
            0
        } else {
            charge(keys, size_of::<KeyEntry>())
        } + if ids == old_ids {
            0
        } else {
            charge(ids, size_of::<IdEntry>())
        };
        Ok(PayloadAdmission {
            metadata_bytes,
            payload_bytes,
            workspace,
            new_batches,
        })
    }

    pub fn reserve_admission(&mut self, batches: usize) {
        let hasher = &self.hasher;
        self.by_key
            .reserve(batches, |entry| hasher.hash_one(entry.key));
        self.by_id
            .reserve(batches, |entry| u64::from(entry.id.get()));
    }

    pub fn project_remove(
        &self,
        removals: &std::collections::BTreeMap<BatchKey, usize>,
        name: &str,
    ) -> crate::Result<PayloadRemoval> {
        let mut remaining = self.len();
        let mut released_bytes = 0;
        let (mut keys, mut ids) = self.backing_buckets();
        let mut workspace = 0;
        for (key, removed) in removals {
            let (batch, references) = &self[key];
            assert!(removed <= references, "preflighted ASOF batch removals");
            if removed != references {
                continue;
            }
            released_bytes = super::super::checked(
                name,
                released_bytes,
                super::capacity_batch_allocation(batch, name)?,
            )?;
            remaining -= 1;
            if remaining == 0 {
                keys = 0;
                ids = 0;
            } else if ids > 4 * remaining {
                let next = required_backing(remaining);
                keys = next;
                ids = next;
                workspace = workspace
                    .max(charge(keys, size_of::<KeyEntry>()) + charge(ids, size_of::<IdEntry>()));
            }
        }
        Ok(PayloadRemoval {
            metadata_bytes: charge(keys, size_of::<KeyEntry>()) + charge(ids, size_of::<IdEntry>()),
            released_bytes,
            workspace,
            buckets: [keys, ids],
            replace: remaining != 0 && (keys, ids) != self.backing_buckets(),
            remaining,
        })
    }

    pub fn defer_compaction(&mut self) {
        self.defer_compaction = true;
    }

    pub fn install_compaction(&mut self, mut prepared: PreparedPayloadRemoval) {
        self.defer_compaction = false;
        let Some(replacement) = prepared.replacement.take() else {
            return;
        };
        prepared.replacement = Some(std::mem::replace(self, replacement));
    }

    pub fn compaction_entries<'a>(
        &'a self,
        removals: &'a std::collections::BTreeMap<BatchKey, usize>,
    ) -> impl Iterator<Item = (u32, &'a Arc<PayloadBatch>, usize)> + 'a {
        self.by_id.iter().filter_map(move |entry| {
            let (batch, references) = &entry.value;
            let references = references - removals.get(&batch.key).copied().unwrap_or(0);
            (references != 0).then_some((entry.id.get(), batch, references))
        })
    }

    #[cfg(test)]
    pub fn prepare_removal(
        &self,
        removals: &std::collections::BTreeMap<BatchKey, usize>,
    ) -> PreparedPayloadRemoval {
        let layout = self.project_remove(removals, "asof").unwrap();
        if !layout.replace {
            return PreparedPayloadRemoval::empty();
        }
        let mut prepared = PreparedPayloadRemoval {
            replacement: Some(PayloadPool {
                hasher: self.hasher.clone(),
                next_id: self.next_id,
                ..PayloadPool::default()
            }),
            inputs: Vec::with_capacity(layout.remaining),
            workspace: None,
            retirement: None,
        };
        for (id, batch, references) in self.compaction_entries(removals) {
            prepared.retain(id, batch, references);
        }
        prepared.populate(&layout);
        prepared
    }

    pub fn values(&self) -> impl Iterator<Item = &(Arc<PayloadBatch>, usize)> {
        self.by_id.iter().map(|entry| &entry.value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&BatchKey, &(Arc<PayloadBatch>, usize))> {
        self.by_key
            .iter()
            .map(|entry| (&entry.key, &self.by_id_entry(entry.id).value))
    }

    fn by_id_entry(&self, id: NonZeroU32) -> &IdEntry {
        self.by_id
            .find(u64::from(id.get()), |entry| entry.id == id)
            .expect("indexed ASOF batch")
    }

    pub fn attach(&mut self, row: &RowPayload) -> RowRef {
        let row_index = u32::try_from(row.row).expect("preflighted ASOF payload row index");
        self.attach_batch(&row.batch, 1).with_row(row_index)
    }

    pub fn attach_batch(&mut self, batch: &Arc<PayloadBatch>, count: usize) -> RowRef {
        let hash = self.hasher.hash_one(batch.key);
        let id = if let Some(entry) = self.by_key.find(hash, |entry| entry.key == batch.key) {
            entry.id
        } else {
            let id = self.allocate_id();
            let hasher = &self.hasher;
            self.by_key
                .insert_unique(hash, KeyEntry { key: batch.key, id }, |entry| {
                    hasher.hash_one(entry.key)
                });
            self.by_id.insert_unique(
                u64::from(id.get()),
                IdEntry {
                    id,
                    value: (batch.clone(), 0),
                },
                |entry| u64::from(entry.id.get()),
            );
            id
        };
        self.by_id
            .find_mut(u64::from(id.get()), |entry| entry.id == id)
            .expect("indexed ASOF batch")
            .value
            .1 += count;
        RowRef { batch: id, row: 0 }
    }

    fn allocate_id(&mut self) -> NonZeroU32 {
        loop {
            self.next_id = self.next_id.wrapping_add(1).max(1);
            let id = NonZeroU32::new(self.next_id).expect("nonzero ASOF batch handle");
            if self
                .by_id
                .find(u64::from(id.get()), |entry| entry.id == id)
                .is_none()
            {
                return id;
            }
        }
    }

    pub fn view(&self, row: RowRef) -> PayloadView<'_> {
        PayloadView {
            batch: &self.by_id_entry(row.batch).value.0,
            row: row.row as usize,
        }
    }

    pub fn key(&self, row: RowRef) -> BatchKey {
        self.by_id_entry(row.batch).value.0.key
    }
    #[cfg(test)]
    pub fn detach(&mut self, row: RowRef) {
        self.detach_id(row.batch, 1);
    }

    pub fn detach_count(&mut self, key: BatchKey, removed: usize) {
        let id = self
            .by_key
            .find(self.hasher.hash_one(key), |entry| entry.key == key)
            .expect("indexed ASOF batch")
            .id;
        self.detach_id(id, removed);
    }

    fn detach_id(&mut self, id: NonZeroU32, removed: usize) {
        let entry = self
            .by_id
            .find_mut(u64::from(id.get()), |entry| entry.id == id)
            .expect("indexed ASOF batch");
        entry.value.1 -= removed;
        if entry.value.1 != 0 {
            return;
        }
        let key = entry.value.0.key;
        self.by_key
            .find_entry(self.hasher.hash_one(key), |entry| entry.key == key)
            .ok()
            .expect("indexed ASOF batch")
            .remove();
        self.by_id
            .find_entry(u64::from(id.get()), |entry| entry.id == id)
            .ok()
            .expect("indexed ASOF batch")
            .remove();
        if self.by_id.is_empty() {
            *self = Self::default();
        } else if !self.defer_compaction && backing_buckets(&self.by_id) > 4 * self.by_id.len() {
            let hasher = &self.hasher;
            self.by_key
                .shrink_to_fit(|entry| hasher.hash_one(entry.key));
            self.by_id.shrink_to_fit(|entry| u64::from(entry.id.get()));
        }
    }
}

impl Index<&BatchKey> for PayloadPool {
    type Output = (Arc<PayloadBatch>, usize);
    fn index(&self, key: &BatchKey) -> &Self::Output {
        let entry = self
            .by_key
            .find(self.hasher.hash_one(key), |entry| &entry.key == key)
            .expect("indexed ASOF batch");
        &self.by_id_entry(entry.id).value
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
    fn prepared_pool_commit_does_not_rehash_surviving_entries() {
        let mut pool = PayloadPool::default();
        for key in 0..4_096 {
            pool.attach(&row(key));
        }
        let surviving = pool
            .by_key
            .iter()
            .find(|entry| entry.key == (1, 0))
            .unwrap()
            .id;
        let removals = (1_000..4_096).map(|key| ((1, key), 1)).collect();
        let layout = pool.project_remove(&removals, "asof").unwrap();
        assert!(layout.replace);
        let prepared = pool.prepare_removal(&removals);
        pool.defer_compaction();
        for (&key, &count) in &removals {
            pool.detach_count(key, count);
        }
        COMPACTION_MOVES.with(|moves| moves.set(0));
        let allocations = allocation_counter::measure(|| pool.install_compaction(prepared));
        assert_eq!(
            COMPACTION_MOVES.with(std::cell::Cell::get),
            0,
            "commit must swap a populated pool without visiting its survivors"
        );
        assert_eq!(allocations.count_total, 0);
        assert_eq!(pool.len(), 1_000);
        assert_eq!(
            pool.view(RowRef {
                batch: surviving,
                row: 0
            })
            .batch
            .key,
            (1, 0)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn detached_pool_population_preserves_payloads_and_lease_until_release() {
        use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
        let mut pool = PayloadPool::default();
        for key in 0..4_096 {
            pool.attach(&row(key));
        }
        let removals = (1_000..4_096).map(|key| ((1, key), 1)).collect();
        let layout = pool.project_remove(&removals, "asof").unwrap();
        let bytes = pool.metadata_bytes()
            + layout.metadata_bytes
            + layout.remaining as u64 * 24
            + 512
            + pool
                .compaction_entries(&removals)
                .map(|(_, batch, _)| {
                    super::super::capacity_batch_allocation(batch, "asof").unwrap()
                })
                .sum::<u64>();
        let memory: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(usize::try_from(bytes).unwrap()));
        let lease = MemoryConsumer::new("pool-population").register(&memory);
        lease.try_grow(usize::try_from(bytes).unwrap()).unwrap();
        let weak = pool
            .compaction_entries(&removals)
            .map(|(_, batch, _)| Arc::downgrade(batch))
            .collect::<Vec<_>>();
        let mut prepared = PreparedPayloadRemoval::capture(&pool, &layout, lease);
        for (id, batch, count) in pool.compaction_entries(&removals) {
            prepared.retain(id, batch, count);
        }
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, blocked) = std::sync::mpsc::channel();
        let worker = tokio::task::spawn_blocking(move || {
            started.send(()).unwrap();
            blocked.recv().unwrap();
            let measured = allocation_counter::measure(|| prepared.populate(&layout));
            assert!(
                measured.bytes_max <= bytes,
                "{measured:?}, reserved={bytes}"
            );
            drop(prepared);
        });
        ready.await.unwrap();
        drop(worker);
        drop(pool);
        assert!(weak.iter().all(|batch| batch.upgrade().is_some()));
        assert_eq!(memory.reserved() as u64, bytes);
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            tokio::time::sleep(std::time::Duration::from_millis(1)),
        )
        .await
        .unwrap();
        release.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while memory.reserved() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(weak.iter().all(|batch| batch.upgrade().is_none()));
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

    #[test]
    fn sparse_churn_charges_the_backing_hash_allocations() {
        let rows = (0..4_096).map(row).collect::<Vec<_>>();
        let mut pool = PayloadPool::default();
        let measured = allocation_counter::measure(|| {
            for row in &rows {
                pool.attach(row);
            }
            for row in &rows[1..] {
                pool.detach_count(row.batch.key, 1);
            }
        });
        assert_eq!(pool.len(), 1);
        assert!(
            pool.metadata_bytes() >= u64::try_from(measured.bytes_current).unwrap(),
            "usable slots hid retained hash allocations: {}, {measured:?}",
            pool.metadata_bytes(),
        );
    }

    #[test]
    fn admission_projects_hash_growth_and_funds_each_rehash() {
        for initial in [0, 1, 3, 7, 14, 28, 56, 112, 1_024] {
            for additional in [1, 2, 4, 35] {
                let mut pool = PayloadPool::default();
                for id in 0..initial {
                    pool.attach(&row(id));
                }
                let batches = (initial..initial + additional)
                    .map(|id| row(id).batch)
                    .collect::<Vec<_>>();
                let projection = pool.project_admission(&batches, "asof").unwrap();
                assert_eq!(projection.new_batches, usize::try_from(additional).unwrap());
                let allocation = allocation_counter::measure(|| {
                    pool.reserve_admission(projection.new_batches);
                    for batch in &batches {
                        pool.attach_batch(batch, 1);
                    }
                });
                assert_eq!(
                    pool.metadata_bytes(),
                    projection.metadata_bytes,
                    "initial={initial}, added={additional}"
                );
                assert!(
                    allocation.bytes_max <= projection.workspace,
                    "rehash exceeded workspace: initial={initial}, added={additional}, {allocation:?}, budget={}",
                    projection.workspace
                );
            }
        }
    }

    #[test]
    fn removals_project_threshold_shrinks_without_cloning_the_pool() {
        for initial in [1, 7, 14, 28, 56, 112, 1_024] {
            for retained in [0, 1, initial / 4, initial / 2, initial] {
                let mut pool = PayloadPool::default();
                for id in 0..initial {
                    pool.attach(&row(id));
                }
                let removals = (retained..initial).map(|id| ((1, id), 1)).collect();
                let projection = pool.project_remove(&removals, "asof").unwrap();
                let allocation = allocation_counter::measure(|| {
                    for (&key, &count) in &removals {
                        pool.detach_count(key, count);
                    }
                });
                assert_eq!(
                    pool.metadata_bytes(),
                    projection.metadata_bytes,
                    "initial={initial}, retained={retained}"
                );
                assert!(
                    allocation.bytes_max <= projection.workspace,
                    "shrink exceeded workspace: initial={initial}, retained={retained}, {allocation:?}"
                );
            }
        }
    }
}
