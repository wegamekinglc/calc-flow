use super::{Encoding, Heap, RightState, entries, hash_allocation};
use crate::{Result, StreamingFailureReason};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use hashbrown::HashTable;
use std::sync::Arc;

pub(super) struct Growth {
    reservation: Arc<MemoryReservation>,
    added: usize,
    committed: bool,
}

impl Growth {
    pub(super) fn new(
        previous: Option<&Arc<MemoryReservation>>,
        size: usize,
        pool: &Arc<dyn MemoryPool>,
        name: &str,
    ) -> Result<Self> {
        let reservation = previous.cloned().unwrap_or_else(|| {
            Arc::new(MemoryConsumer::new("asof-expiration-index").register(pool))
        });
        let added = size
            .checked_sub(reservation.size())
            .ok_or_else(|| failure(name))?;
        reservation.try_grow(added).map_err(|_| failure(name))?;
        Ok(Self {
            reservation,
            added,
            committed: false,
        })
    }

    pub(super) fn commit(mut self) -> Arc<MemoryReservation> {
        self.committed = true;
        self.reservation.clone()
    }
}

impl Drop for Growth {
    fn drop(&mut self) {
        if !self.committed {
            self.reservation.shrink(self.added);
        }
    }
}

pub(in super::super::super) struct PreparedStorage {
    entries: entries::PreparedEntries,
    buckets: Option<HashTable<u32>>,
    payloads: Option<Heap>,
    identities: Option<Heap>,
    dominance: Option<Heap>,
    growth: Growth,
    _workspace: MemoryReservation,
}

impl PreparedStorage {
    pub fn install(self, state: &mut RightState) {
        self.entries.install(&mut state.entries);
        if let Some(buckets) = self.buckets {
            state.buckets = buckets;
        }
        if let Some(payloads) = self.payloads {
            state.payloads = payloads;
        }
        if let Some(identities) = self.identities {
            state.identities = identities;
        }
        if let Some(dominance) = self.dominance {
            state.dominance = dominance;
        }
        state.lease = Some(self.growth.commit());
    }
}

pub(super) fn failure(name: &str) -> crate::CalcFlowError {
    super::super::super::reason(
        name,
        StreamingFailureReason::AsofWorkspaceLimitExceeded,
        "ASOF expiration index and dictionary preparation exceed max_state_bytes workspace",
    )
}

pub(super) fn allocation(capacity: usize, width: usize, name: &str) -> Result<usize> {
    capacity.checked_mul(width).ok_or_else(|| failure(name))
}

pub(super) fn total(values: &[usize], name: &str) -> Result<usize> {
    values.iter().try_fold(0_usize, |bytes, value| {
        bytes.checked_add(*value).ok_or_else(|| failure(name))
    })
}

impl RightState {
    pub fn install_recovery_lease(&mut self, reservation: MemoryReservation) {
        self.lease = (reservation.size() != 0).then(|| Arc::new(reservation));
    }

    pub fn auxiliary_bytes(&self) -> usize {
        self.entries.capacities().directory * 8
            + self.entries.capacities().shards.into_iter().sum::<usize>() * 24
            + self.payloads.allocation_bytes()
            + self.identities.allocation_bytes()
            + self.dominance.allocation_bytes()
    }

    pub fn prepare_storage(
        &self,
        additions: &[(Encoding, usize)],
        pool: &Arc<dyn MemoryPool>,
        name: &str,
    ) -> Result<PreparedStorage> {
        let (capacity, backing) = self.admission_capacities(additions);
        for value in std::iter::once(capacity.directory).chain(capacity.shards) {
            u32::try_from(value).map_err(|_| failure(name))?;
        }
        let auxiliary = self.auxiliary_capacity_bytes(capacity, name)?;
        let growth = Growth::new(self.lease.as_ref(), auxiliary, pool, name)?;
        let bytes = self.storage_workspace(capacity, backing, name)?;
        let workspace = MemoryConsumer::new("asof-dictionary-preparation").register(pool);
        workspace.try_grow(bytes).map_err(|_| failure(name))?;
        Ok(PreparedStorage {
            entries: self.entries.prepare(capacity),
            buckets: self.copy_hash(backing),
            payloads: Self::copy_heap(&self.payloads, capacity.directory),
            identities: Self::copy_heap(&self.identities, capacity.directory),
            dominance: Self::copy_heap(&self.dominance, capacity.directory),
            growth,
            _workspace: workspace,
        })
    }

    fn auxiliary_capacity_bytes(&self, capacity: entries::Capacities, name: &str) -> Result<usize> {
        total(
            &[
                capacity.auxiliary_bytes(name)?,
                allocation(capacity.directory.max(self.payloads.capacity()), 16, name)?,
                allocation(capacity.directory.max(self.identities.capacity()), 16, name)?,
                allocation(capacity.directory.max(self.dominance.capacity()), 16, name)?,
            ],
            name,
        )
    }

    fn storage_workspace(
        &self,
        capacity: entries::Capacities,
        backing: Option<usize>,
        name: &str,
    ) -> Result<usize> {
        let entries = self.entries.copy_workspace(capacity, name)?;
        let heaps = [&self.payloads, &self.identities, &self.dominance]
            .into_iter()
            .filter(|heap| capacity.directory > heap.capacity())
            .map(Heap::allocation_bytes)
            .sum::<usize>();
        let buckets = backing.map_or(0, |backing| {
            hash_allocation(super::super::payload::backing_buckets(&self.buckets))
                + hash_allocation(backing)
        });
        total(&[entries, heaps, buckets], name)
    }

    fn copy_hash(&self, backing: Option<usize>) -> Option<HashTable<u32>> {
        let backing = backing?;
        let mut table = HashTable::with_capacity(super::super::payload::bucket_capacity(backing));
        for (id, entry) in self.entries.iter().enumerate() {
            #[cfg(test)]
            super::super::expiration_cost_tests::record_compaction();
            table.insert_unique(
                entry.hash,
                u32::try_from(id).expect("preflighted ASOF dictionary handle"),
                |id| self.entries[*id as usize].hash,
            );
        }
        Some(table)
    }

    fn copy_heap(heap: &Heap, capacity: usize) -> Option<Heap> {
        (capacity > heap.capacity()).then(|| heap.copy_with_capacity(capacity))
    }
}
