//! Canonical key bytes owned once by the retained right dictionary. Hash table
//! slots contain only u32 handles; arrival IDs never determine output order.

use super::{Encoding, RightBucket, SequenceKind};
use crate::Result;
use datafusion::common::hash_utils::RandomState;
use hashbrown::HashTable;
#[cfg(test)]
use std::ops::{Deref, DerefMut};
use std::{hash::BuildHasher, sync::Arc};

mod compaction;
mod expiration;
mod funding;
pub(in super::super) use compaction::PreparedCompaction;
use expiration::{ABSENT, Heap, Kind};

#[derive(Clone)]
struct Entry {
    key: Encoding,
    hash: u64,
    bucket: Arc<RightBucket>,
    payload_position: u32,
    identity_position: u32,
    dominance_position: u32,
}

const _: () = assert!(size_of::<Entry>() == 48);

pub(in super::super) struct RightState {
    pub(in super::super) buckets: HashTable<u32>,
    entries: Vec<Entry>,
    hasher: RandomState,
    payloads: Heap,
    identities: Heap,
    dominance: Heap,
    bucket_bytes: u64,
    lease: Option<Arc<datafusion::execution::memory_pool::MemoryReservation>>,
}

impl Default for RightState {
    fn default() -> Self {
        Self {
            buckets: HashTable::new(),
            entries: Vec::new(),
            hasher: RandomState::with_seed(ahash::RandomState::new().hash_one(0_u64)),
            payloads: Heap::new(Kind::Payload, 0),
            identities: Heap::new(Kind::Identity, 0),
            dominance: Heap::new(Kind::Dominance, 0),
            bucket_bytes: 0,
            lease: None,
        }
    }
}

impl RightState {
    pub fn checkpoint_capacities(&self) -> [usize; 2] {
        [
            self.entries.capacity(),
            super::payload::backing_buckets(&self.buckets),
        ]
    }

    pub fn with_index_capacities(entries: usize, buckets: usize, heaps: [usize; 3]) -> Self {
        Self {
            entries: Vec::with_capacity(entries),
            payloads: Heap::new(Kind::Payload, heaps[0]),
            identities: Heap::new(Kind::Identity, heaps[1]),
            dominance: Heap::new(Kind::Dominance, heaps[2]),
            buckets: HashTable::with_capacity(super::payload::bucket_capacity(buckets)),
            ..Self::default()
        }
    }
    pub fn metadata_bytes(&self) -> u64 {
        self.container_bytes() + self.bucket_bytes
    }

    pub fn bucket_bytes(&self) -> u64 {
        self.bucket_bytes
    }

    pub fn heap_capacities(&self) -> [usize; 3] {
        [
            self.payloads.capacity(),
            self.identities.capacity(),
            self.dominance.capacity(),
        ]
    }

    pub fn minima(&self) -> [Option<i64>; 3] {
        [
            self.payloads.minimum(),
            self.identities.minimum(),
            self.dominance.minimum(),
        ]
    }

    pub fn due_count(&self, cutoffs: [i128; 3]) -> usize {
        self.payloads.due_count(cutoffs[0])
            + self.identities.due_count(cutoffs[1])
            + self.dominance.due_count(cutoffs[2])
    }

    pub fn due_keys(&self, cutoffs: [i128; 3]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(self.due_count(cutoffs));
        self.payloads.collect_due(cutoffs[0], &mut ids);
        self.identities.collect_due(cutoffs[1], &mut ids);
        self.dominance.collect_due(cutoffs[2], &mut ids);
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    pub fn indexed_bucket(&self, id: u32) -> (&Encoding, &RightBucket) {
        let entry = &self.entries[id as usize];
        (&entry.key, &entry.bucket)
    }

    pub fn container_bytes(&self) -> u64 {
        (self.entries.capacity() * size_of::<Entry>()
            + self.payloads.allocation_bytes()
            + self.identities.allocation_bytes()
            + self.dominance.allocation_bytes()
            + hash_allocation(super::payload::backing_buckets(&self.buckets))
            + self.entries.len() * (size_of::<RightBucket>() + 2 * size_of::<usize>()))
            as u64
    }

    pub fn shared_admission_buckets<'a>(
        &'a self,
        additions: &'a [(Encoding, usize)],
    ) -> impl Iterator<Item = (u32, &'a Arc<RightBucket>)> + Clone {
        additions
            .iter()
            .filter_map(|(key, _)| self.find(self.hasher.hash_one(key.as_slice()), key.as_slice()))
            .filter_map(|id| {
                let bucket = &self.entries[id].bucket;
                (Arc::strong_count(bucket) > 1).then_some((
                    u32::try_from(id).expect("preflighted bucket domain"),
                    bucket,
                ))
            })
    }

    pub fn shared_eviction_buckets<'a>(
        &'a self,
        selected: &'a [u32],
    ) -> impl Iterator<Item = (u32, &'a Arc<RightBucket>)> + Clone {
        selected.iter().filter_map(|&id| {
            let bucket = &self.entries[id as usize].bucket;
            (Arc::strong_count(bucket) > 1).then_some((id, bucket))
        })
    }

    pub fn install_prepared_bucket(&mut self, id: u32, bucket: Arc<RightBucket>) {
        self.entries[id as usize].bucket = bucket;
    }

    pub fn projected_admission_growth(
        &self,
        additions: &[(Encoding, usize)],
        kind: SequenceKind,
    ) -> u64 {
        let new = additions
            .iter()
            .filter(|(key, _)| !self.contains_key(key))
            .count();
        let (capacity, replacement) = self.admission_capacities(additions);
        let previous_buckets = super::payload::backing_buckets(&self.buckets);
        let buckets = replacement.unwrap_or(previous_buckets);
        let mut bytes = ((capacity - self.entries.capacity()) * size_of::<Entry>()
            + capacity.saturating_sub(self.payloads.capacity()) * 16
            + capacity.saturating_sub(self.identities.capacity()) * 16
            + capacity.saturating_sub(self.dominance.capacity()) * 16
            + hash_allocation(buckets)
            - hash_allocation(previous_buckets)
            + new * (size_of::<RightBucket>() + 2 * size_of::<usize>()))
            as u64;
        for (key, count) in additions {
            let empty = RightBucket::with_sequence_kind(kind);
            let bucket = self.get(key).unwrap_or(&empty);
            bytes += bucket.projected_admission_bytes(*count) - bucket.metadata_bytes();
        }
        bytes
    }

    fn admission_capacities(&self, additions: &[(Encoding, usize)]) -> (usize, Option<usize>) {
        let new = additions
            .iter()
            .filter(|(key, _)| !self.contains_key(key))
            .count();
        let required = self.entries.len() + new;
        let mut capacity = self.entries.capacity();
        if capacity == 0 && new > 0 {
            capacity = 1;
        }
        while capacity < required {
            capacity = (capacity * 2).max(4);
        }
        let backing =
            (self.buckets.capacity() < required).then(|| self.replacement_backing(required));
        (capacity, backing)
    }

    fn replacement_backing(&self, required: usize) -> usize {
        let previous = super::payload::backing_buckets(&self.buckets);
        let required = required_backing(required);
        if required > previous {
            required
        } else {
            previous * 2
        }
    }

    pub fn projected_eviction_bytes(
        &self,
        keys: usize,
        rows_bytes: u64,
        mut workspace: u64,
        name: &str,
    ) -> Result<(usize, u64, u64)> {
        if keys == 0 {
            return Ok((self.entries.len(), 0, workspace));
        }
        let container = if let Some(capacities) = self.compaction_capacities(keys) {
            workspace = super::super::checked(
                name,
                workspace,
                capacities.workspace_bytes(self, name)? as u64,
            )?;
            capacities.container_bytes(keys)
        } else {
            self.container_bytes()
                - (self.entries.len() - keys) as u64
                    * (size_of::<RightBucket>() + 2 * size_of::<usize>()) as u64
        };
        Ok((self.entries.len() - keys, container + rows_bytes, workspace))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn hasher(&self) -> &RandomState {
        &self.hasher
    }

    fn find(&self, hash: u64, bytes: &[u8]) -> Option<usize> {
        self.buckets
            .find(hash, |id| {
                self.entries[*id as usize].key.as_slice() == bytes
            })
            .map(|id| *id as usize)
    }

    pub fn encoding_hashed(&self, hash: u64, bytes: &[u8]) -> Option<&Encoding> {
        self.find(hash, bytes).map(|id| &self.entries[id].key)
    }

    pub fn values(&self) -> impl Iterator<Item = &RightBucket> {
        self.entries.iter().map(|entry| entry.bucket.as_ref())
    }

    pub fn owned_buckets(
        &self,
    ) -> impl ExactSizeIterator<Item = (Encoding, Arc<RightBucket>)> + '_ {
        self.entries
            .iter()
            .map(|entry| (entry.key.clone(), entry.bucket.clone()))
    }

    pub fn contains_key(&self, key: &Encoding) -> bool {
        self.find(self.hasher.hash_one(key.as_slice()), key.as_slice())
            .is_some()
    }

    pub fn get(&self, key: &Encoding) -> Option<&RightBucket> {
        self.get_hashed(self.hasher.hash_one(key.as_slice()), key.as_slice())
    }

    pub fn owned_bucket(&self, key: &Encoding) -> Option<Arc<RightBucket>> {
        self.find(self.hasher.hash_one(key.as_slice()), key.as_slice())
            .map(|id| self.entries[id].bucket.clone())
    }

    pub fn get_hashed(&self, hash: u64, bytes: &[u8]) -> Option<&RightBucket> {
        self.find(hash, bytes)
            .map(|id| self.entries[id].bucket.as_ref())
    }

    pub fn indexed_keys(&self) -> impl ExactSizeIterator<Item = (u32, &Encoding)> {
        self.entries.iter().enumerate().map(|(id, entry)| {
            (
                u32::try_from(id).expect("preflighted ASOF key domain"),
                &entry.key,
            )
        })
    }

    pub fn ordered_iter(&self) -> impl Iterator<Item = (&Encoding, &RightBucket)> {
        let mut ids = self.indexed_keys().map(|(id, _)| id).collect::<Vec<_>>();
        ids.sort_unstable_by(|left, right| {
            self.entries[*left as usize]
                .key
                .cmp(&self.entries[*right as usize].key)
        });
        ids.into_iter().map(|id| {
            let entry = &self.entries[id as usize];
            (&entry.key, entry.bucket.as_ref())
        })
    }

    #[cfg(test)]
    pub fn bucket_mut_or_default(&mut self, key: Encoding) -> BucketMut<'_> {
        self.bucket_mut_or_kind(key, SequenceKind::Canonical)
    }

    #[cfg(test)]
    pub fn bucket_mut_or_kind(&mut self, key: Encoding, kind: SequenceKind) -> BucketMut<'_> {
        let id = self.ensure_key(key, kind);
        let previous_bytes = self.entries[id].bucket.metadata_bytes();
        BucketMut {
            state: self,
            id,
            previous_bytes,
        }
    }

    fn ensure_key(&mut self, key: Encoding, kind: SequenceKind) -> usize {
        let hash = self.hasher.hash_one(key.as_slice());
        self.find(hash, key.as_slice())
            .unwrap_or_else(|| self.insert_unique(hash, key, RightBucket::with_sequence_kind(kind)))
    }

    pub fn update_unindexed(
        &mut self,
        key: Encoding,
        kind: SequenceKind,
        update: impl FnOnce(&mut RightBucket),
    ) {
        let id = self.ensure_key(key, kind);
        let bucket = &mut self.entries[id].bucket;
        let previous = bucket.metadata_bytes();
        update(Arc::make_mut(bucket));
        self.bucket_bytes = self.bucket_bytes - previous + bucket.metadata_bytes();
    }

    pub fn refresh_key(&mut self, key: &Encoding) {
        let id = self
            .find(self.hasher.hash_one(key.as_slice()), key.as_slice())
            .expect("admitted key");
        self.refresh(u32::try_from(id).expect("preflighted ASOF dictionary handle"));
    }

    fn refresh(&mut self, id: u32) {
        #[cfg(test)]
        super::expiration_cost_tests::record_minima();
        let bucket = &self.entries[id as usize].bucket;
        let payload = bucket.payload_min();
        let identity = bucket.identity_min();
        let dominance = bucket.dominance_min();
        self.payloads.replace(&mut self.entries, id, payload);
        self.identities.replace(&mut self.entries, id, identity);
        if self.dominance.capacity() != 0 {
            self.dominance.replace(&mut self.entries, id, dominance);
        }
    }

    pub fn evict_bucket(
        &mut self,
        id: u32,
        status: &super::super::StreamAsofJoinStatus,
        tolerance: u64,
        threshold: i128,
    ) -> u64 {
        let bucket = &mut self.entries[id as usize].bucket;
        let previous = bucket.metadata_bytes();
        let evicted = Arc::make_mut(bucket).evict(status, tolerance, threshold);
        self.bucket_bytes = self.bucket_bytes - previous + bucket.metadata_bytes();
        self.refresh(id);
        evicted
    }

    pub fn remove_empty(&mut self, selected: &[u32]) {
        for &id in selected.iter().rev() {
            if self.entries[id as usize].bucket.is_empty() {
                self.remove(id);
            }
        }
        if self.entries.is_empty() {
            *self = Self::default();
        }
    }

    fn remove(&mut self, id: u32) {
        #[cfg(test)]
        super::expiration_cost_tests::record_dictionary();
        self.payloads.replace(&mut self.entries, id, None);
        self.identities.replace(&mut self.entries, id, None);
        self.dominance.replace(&mut self.entries, id, None);
        let hash = self.entries[id as usize].hash;
        self.buckets
            .find_entry(hash, |entry| *entry == id)
            .expect("retained ASOF hash handle")
            .remove();
        let removed = self.entries.swap_remove(id as usize);
        self.bucket_bytes -= removed.bucket.metadata_bytes();
        self.repair_moved(id);
    }

    fn repair_moved(&mut self, id: u32) {
        if id as usize == self.entries.len() {
            return;
        }
        #[cfg(test)]
        super::expiration_cost_tests::record_dictionary();
        let previous = u32::try_from(self.entries.len()).expect("retained key domain");
        let hash = self.entries[id as usize].hash;
        *self
            .buckets
            .find_mut(hash, |entry| *entry == previous)
            .expect("moved ASOF hash handle") = id;
        self.payloads.rename(&mut self.entries, id);
        self.identities.rename(&mut self.entries, id);
        self.dominance.rename(&mut self.entries, id);
    }

    #[cfg(test)]
    pub fn insert(&mut self, key: Encoding, bucket: RightBucket) {
        let hash = self.hasher.hash_one(key.as_slice());
        if let Some(id) = self.find(hash, key.as_slice()) {
            self.bucket_bytes -= self.entries[id].bucket.metadata_bytes();
            self.bucket_bytes += bucket.metadata_bytes();
            self.entries[id].bucket = Arc::new(bucket);
            self.refresh(u32::try_from(id).expect("preflighted ASOF dictionary handle"));
        } else {
            self.insert_unique(hash, key, bucket);
        }
    }

    fn insert_unique(&mut self, hash: u64, key: Encoding, bucket: RightBucket) -> usize {
        let id = self.insert_unindexed(hash, key, bucket);
        if self.payloads.capacity() < self.entries.len() {
            self.payloads.reserve_exact(self.entries.capacity());
        }
        if self.identities.capacity() < self.entries.len() {
            self.identities.reserve_exact(self.entries.capacity());
        }
        if self.dominance.capacity() < self.entries.len() {
            self.dominance.reserve_exact(self.entries.capacity());
        }
        self.refresh(u32::try_from(id).expect("preflighted ASOF dictionary handle"));
        id
    }

    pub fn insert_restored(&mut self, key: Encoding, bucket: RightBucket) {
        let hash = self.hasher.hash_one(key.as_slice());
        let id = self.insert_unindexed(hash, key, bucket);
        self.refresh(u32::try_from(id).expect("preflighted ASOF dictionary handle"));
    }

    fn insert_unindexed(&mut self, hash: u64, key: Encoding, bucket: RightBucket) -> usize {
        let id = u32::try_from(self.entries.len()).expect("preflighted ASOF key domain");
        if self.entries.capacity() == 0 {
            // Vec's four-entry minimum is unnecessary for a sparse dictionary.
            self.entries = Vec::with_capacity(1);
        }
        self.bucket_bytes += bucket.metadata_bytes();
        self.entries.push(Entry {
            key,
            hash,
            bucket: Arc::new(bucket),
            payload_position: ABSENT,
            identity_position: ABSENT,
            dominance_position: ABSENT,
        });
        let entries = &self.entries;
        self.buckets
            .insert_unique(hash, id, |id| entries[*id as usize].hash);
        id as usize
    }

    #[cfg(test)]
    pub fn retain(&mut self, mut keep: impl FnMut(&Encoding, &mut Arc<RightBucket>) -> bool) {
        for id in (0..self.entries.len()).rev() {
            let entry = &mut self.entries[id];
            if !keep(&entry.key, &mut entry.bucket) {
                self.remove(u32::try_from(id).expect("preflighted ASOF dictionary handle"));
            }
        }
        if self.entries.is_empty() {
            *self = Self::default();
        } else if let Some(compaction) = self.prepare_fixture_compaction(self.entries.len()) {
            compaction.install(self);
        }
    }
}

#[cfg(test)]
pub(in super::super) struct BucketMut<'a> {
    state: &'a mut RightState,
    id: usize,
    previous_bytes: u64,
}

#[cfg(test)]
impl Deref for BucketMut<'_> {
    type Target = RightBucket;
    fn deref(&self) -> &Self::Target {
        &self.state.entries[self.id].bucket
    }
}

#[cfg(test)]
impl DerefMut for BucketMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.state.entries[self.id].bucket)
    }
}

#[cfg(test)]
impl Drop for BucketMut<'_> {
    fn drop(&mut self) {
        self.state.bucket_bytes = self.state.bucket_bytes - self.previous_bytes
            + self.state.entries[self.id].bucket.metadata_bytes();
        self.state
            .refresh(u32::try_from(self.id).expect("preflighted ASOF dictionary handle"));
    }
}

fn hash_allocation(buckets: usize) -> usize {
    if buckets == 0 { 0 } else { buckets * 5 + 64 }
}

fn required_backing(entries: usize) -> usize {
    if entries <= 3 {
        4
    } else {
        (entries * 8).div_ceil(7).next_power_of_two()
    }
}

pub(in super::super) fn validate_key_count(count: u64, name: &str) -> Result<()> {
    if count > u64::from(u32::MAX) {
        return Err(super::super::reason(
            name,
            crate::StreamingFailureReason::AsofCounterOverflow,
            "ASOF key dictionary exceeds its u32 handle domain",
        ));
    }
    Ok(())
}

pub(in super::super) struct RightStateIter<'a>(std::slice::Iter<'a, Entry>);

impl<'a> Iterator for RightStateIter<'a> {
    type Item = (&'a Encoding, &'a RightBucket);

    fn next(&mut self) -> Option<Self::Item> {
        self.0
            .next()
            .map(|entry| (&entry.key, entry.bucket.as_ref()))
    }
}

impl<'a> IntoIterator for &'a RightState {
    type Item = (&'a Encoding, &'a RightBucket);
    type IntoIter = RightStateIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        RightStateIter(self.entries.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a03_tombstone_rehash_cost_is_amortized_under_churn() {
        use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let mut state = RightState::default();
        for key in 0_u32..28 {
            state.insert_unique(
                0,
                Encoding::from_slice(&key.to_le_bytes()),
                RightBucket::default(),
            );
        }
        assert_eq!(state.buckets.capacity(), 28);
        let initial = state.prepare_storage(&[], &pool, "asof").unwrap();
        initial.install(&mut state);
        super::super::expiration_cost_tests::take_compaction_visits();
        for key in 28_u32..92 {
            state.remove(0);
            let key = Encoding::from_slice(&key.to_le_bytes());
            let growth = state
                .prepare_storage(&[(key.clone(), 1)], &pool, "asof")
                .unwrap();
            growth.install(&mut state);
            state.insert_unique(0, key, RightBucket::default());
            assert_eq!(state.len(), 28);
        }
        let visits = super::super::expiration_cost_tests::take_compaction_visits();
        assert!(
            visits <= 2 * (28 + 64),
            "cold rehash must amortize across churn: {visits}"
        );
        assert_eq!(pool.reserved(), state.auxiliary_bytes());
        drop(state);
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn collided_handles_compare_complete_key_bytes() {
        let mut state = RightState::default();
        for key in [b"alpha".as_slice(), b"beta", b"gamma"] {
            state.insert_unique(0, Encoding::from_slice(key), RightBucket::default());
        }
        assert_eq!(state.find(0, b"beta"), Some(1));
        assert_eq!(state.find(0, b"absent"), None);
        assert_eq!(
            state.encoding_hashed(0, b"gamma").unwrap().as_slice(),
            b"gamma"
        );
        state.retain(|key, _| key.as_slice() != b"alpha");
        assert_eq!(state.find(0, b"beta"), Some(1));
        assert_eq!(state.find(0, b"gamma"), Some(0));
    }

    #[test]
    fn shared_right_admission_preserves_preflighted_capacities() {
        use super::super::RowRef;

        let kind = SequenceKind::Signed(1);
        let key = Encoding::from_slice(b"key");
        let sequence = kind.decode_integer(&[0]);
        let mut bucket = RightBucket::with_sequence_kind(kind);
        bucket.reserve_payloads(7);
        for time in 0..7 {
            bucket.insert_admitted(
                (time, sequence.clone()),
                RowRef::fixture(u32::try_from(time).unwrap()),
            );
        }
        bucket.reserve_payloads(1);
        bucket.insert_admitted((7, sequence.clone()), RowRef::fixture(7));
        let mut state = RightState::default();
        state.insert(key.clone(), bucket);
        let frozen = state.owned_buckets().next().unwrap().1;
        let projected =
            state.metadata_bytes() + state.projected_admission_growth(&[(key.clone(), 1)], kind);
        let mut live = state.bucket_mut_or_kind(key, kind);
        live.reserve_payloads(1);
        live.insert_admitted((8, sequence), RowRef::fixture(8));
        drop(live);
        assert_eq!(
            state.metadata_bytes(),
            projected,
            "copying shared columns must preserve charged capacities"
        );
        assert_eq!(frozen.len(), 8);
    }

    #[test]
    fn eviction_keeps_untouched_shared_buckets_on_the_same_allocation() {
        use super::super::State;
        use crate::{EventTime, StreamAsofJoinStatus};

        let mut state = State::default();
        for (key, range) in [(b"expired".as_slice(), 0..64), (b"retained", 100..4_196)] {
            let mut bucket = RightBucket::new();
            for time in range {
                bucket.insert((time, Encoding::from_slice(&[1])), None);
            }
            state.right.insert(Encoding::from_slice(key), bucket);
        }
        let key = Encoding::from_slice(b"retained");
        let frozen = state
            .right
            .owned_buckets()
            .find(|(entry, _)| entry == &key)
            .unwrap()
            .1;
        let mut status = StreamAsofJoinStatus::default();
        status.left.watermark_micros = Some(EventTime::from_micros(100));
        status.right.watermark_micros = status.left.watermark_micros;
        state.evict(&status, 0);
        let live = state.right.owned_buckets().next().unwrap().1;
        assert!(
            Arc::ptr_eq(&frozen, &live),
            "a sweep must borrow every unaffected shared bucket"
        );
        assert_eq!(live.len(), 4_096);
    }

    #[test]
    fn dictionary_handle_overflow_is_a_typed_preflight_failure() {
        validate_key_count(u64::from(u32::MAX), "asof").unwrap();
        assert!(matches!(
            validate_key_count(u64::from(u32::MAX) + 1, "asof"),
            Err(crate::CalcFlowError::OperatorReason {
                reason_code: crate::StreamingFailureReason::AsofCounterOverflow,
                ..
            })
        ));
    }
}
