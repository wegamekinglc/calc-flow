use super::funding::{Growth, allocation, failure, total};
use super::{Entry, Heap, Kind, RightBucket, RightState, hash_allocation, required_backing};
use crate::Result;
use datafusion::execution::memory_pool::{MemoryPool, MemoryReservation};
use hashbrown::HashTable;
use std::sync::Arc;

pub(super) struct Capacities {
    entries: usize,
    backing: usize,
}

impl Capacities {
    pub fn container_bytes(&self, keys: usize) -> u64 {
        (self.entries * (size_of::<Entry>() + 48)
            + hash_allocation(self.backing)
            + keys * (size_of::<RightBucket>() + 2 * size_of::<usize>())) as u64
    }

    pub fn workspace_bytes(&self, state: &RightState, name: &str) -> Result<usize> {
        total(
            &[
                allocation(state.entries.capacity(), size_of::<Entry>() - 16, name)?,
                hash_allocation(super::super::payload::backing_buckets(&state.buckets)),
                allocation(self.entries, size_of::<Entry>() + 48, name)?,
                hash_allocation(self.backing),
            ],
            name,
        )
    }
}

pub(in super::super::super) struct PreparedCompaction {
    entries: Vec<Entry>,
    buckets: HashTable<u32>,
    payloads: Heap,
    identities: Heap,
    dominance: Heap,
    growth: Growth,
    _workspace: MemoryReservation,
}

impl PreparedCompaction {
    pub fn install(mut self, state: &mut RightState) {
        self.entries.append(&mut state.entries);
        for (id, entry) in self.entries.iter().enumerate() {
            #[cfg(test)]
            super::super::expiration_cost_tests::record_compaction();
            self.buckets.insert_unique(
                entry.hash,
                u32::try_from(id).expect("preflighted ASOF dictionary handle"),
                |id| self.entries[*id as usize].hash,
            );
        }
        self.payloads.copy_records_from(&state.payloads);
        self.identities.copy_records_from(&state.identities);
        self.dominance.copy_records_from(&state.dominance);
        state.entries = self.entries;
        state.buckets = self.buckets;
        state.payloads = self.payloads;
        state.identities = self.identities;
        state.dominance = self.dominance;
        let reservation = self.growth.commit();
        reservation.shrink(reservation.size() - state.auxiliary_bytes());
        state.lease = Some(reservation);
    }
}

impl RightState {
    pub(super) fn compaction_capacities(&self, keys: usize) -> Option<Capacities> {
        if keys == 0 {
            return None;
        }
        let capacity = self
            .entries
            .capacity()
            .max(self.payloads.capacity())
            .max(self.identities.capacity())
            .max(self.dominance.capacity())
            .max(super::super::payload::bucket_capacity(
                super::super::payload::backing_buckets(&self.buckets),
            ));
        (capacity > keys * 4).then(|| Capacities {
            entries: keys,
            backing: required_backing(keys * 2),
        })
    }

    pub fn prepare_compaction(
        &self,
        keys: usize,
        workspace: &MemoryReservation,
        pool: &Arc<dyn MemoryPool>,
        name: &str,
    ) -> Result<Option<PreparedCompaction>> {
        let Some(capacities) = self.compaction_capacities(keys) else {
            return Ok(None);
        };
        let growth = Growth::new(self.lease.as_ref(), self.auxiliary_bytes(), pool, name)?;
        let bytes = capacities.workspace_bytes(self, name)?;
        if bytes > workspace.size() {
            return Err(failure(name));
        }
        let workspace = workspace.split(bytes);
        Ok(Some(PreparedCompaction {
            entries: Vec::with_capacity(capacities.entries),
            buckets: HashTable::with_capacity(super::super::payload::bucket_capacity(
                capacities.backing,
            )),
            payloads: Heap::new(Kind::Payload, capacities.entries),
            identities: Heap::new(Kind::Identity, capacities.entries),
            dominance: Heap::new(Kind::Dominance, capacities.entries),
            growth,
            _workspace: workspace,
        }))
    }

    #[cfg(test)]
    pub fn prepare_fixture_compaction(&self, keys: usize) -> Option<PreparedCompaction> {
        use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer};
        let capacities = self.compaction_capacities(keys)?;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(usize::MAX));
        let workspace = MemoryConsumer::new("asof-fixture-compaction").register(&pool);
        workspace
            .try_grow(capacities.workspace_bytes(self, "asof").unwrap())
            .unwrap();
        self.prepare_compaction(keys, &workspace, &pool, "asof")
            .unwrap()
    }
}
