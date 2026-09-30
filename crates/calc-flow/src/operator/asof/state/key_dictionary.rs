//! Canonical key bytes owned once by the retained right dictionary. Hash table
//! slots contain only u32 handles; arrival IDs never determine output order.

use super::{Encoding, RightBucket, SequenceKind};
use crate::Result;
use datafusion::common::hash_utils::RandomState;
use hashbrown::HashTable;
use std::{hash::BuildHasher, sync::Arc};

#[derive(Clone)]
struct Entry {
    key: Encoding,
    hash: u64,
    bucket: Arc<RightBucket>,
}

#[derive(Clone)]
pub(in super::super) struct RightState {
    pub(in super::super) buckets: HashTable<u32>,
    entries: Vec<Entry>,
    hasher: RandomState,
    greatest_key: Option<u32>,
}

impl Default for RightState {
    fn default() -> Self {
        Self {
            buckets: HashTable::new(),
            entries: Vec::new(),
            hasher: RandomState::with_seed(ahash::RandomState::new().hash_one(0_u64)),
            greatest_key: None,
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

    pub fn with_capacities(entries: usize, buckets: usize) -> Self {
        Self {
            entries: Vec::with_capacity(entries),
            buckets: HashTable::with_capacity(super::payload::bucket_capacity(buckets)),
            ..Self::default()
        }
    }
    pub fn metadata_bytes(&self) -> u64 {
        let containers = self.entries.capacity() * size_of::<Entry>()
            + hash_allocation(super::payload::backing_buckets(&self.buckets))
            + self.entries.len() * (size_of::<RightBucket>() + 2 * size_of::<usize>());
        containers as u64
            + self
                .entries
                .iter()
                .map(|entry| entry.bucket.metadata_bytes())
                .sum::<u64>()
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
        status: &'a super::super::StreamAsofJoinStatus,
        tolerance: u64,
        threshold: i128,
    ) -> impl Iterator<Item = (u32, &'a Arc<RightBucket>)> + Clone {
        self.entries
            .iter()
            .enumerate()
            .filter_map(move |(id, entry)| {
                (Arc::strong_count(&entry.bucket) > 1
                    && entry.bucket.eviction_pending(status, tolerance, threshold))
                .then_some((
                    u32::try_from(id).expect("preflighted bucket domain"),
                    &entry.bucket,
                ))
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
        let required = self.entries.len() + new;
        let mut capacity = self.entries.capacity();
        if capacity == 0 && new > 0 {
            capacity = 1;
        }
        while capacity < required {
            capacity = (capacity * 2).max(4);
        }
        let previous_buckets = super::payload::backing_buckets(&self.buckets);
        let buckets = if self.buckets.capacity() >= required {
            previous_buckets
        } else {
            required_backing(required).max(previous_buckets)
        };
        let mut bytes =
            ((capacity - self.entries.capacity()) * size_of::<Entry>() + hash_allocation(buckets)
                - hash_allocation(previous_buckets)
                + new * (size_of::<RightBucket>() + 2 * size_of::<usize>())) as u64;
        for (key, count) in additions {
            let empty = RightBucket::with_sequence_kind(kind);
            let bucket = self.get(key).unwrap_or(&empty);
            bytes += bucket.projected_admission_bytes(*count) - bucket.metadata_bytes();
        }
        bytes
    }

    pub fn projected_eviction_bytes(
        &self,
        status: &super::super::StreamAsofJoinStatus,
        tolerance: u64,
        threshold: i128,
    ) -> (usize, u64, u64) {
        let mut keys = 0;
        let mut rows_bytes = 0;
        let mut workspace = 0;
        for entry in &self.entries {
            let (rows, bytes, scratch) = entry
                .bucket
                .projected_eviction(status, tolerance, threshold);
            if rows != 0 {
                keys += 1;
                rows_bytes += bytes;
            }
            workspace += scratch;
        }
        if keys == 0 {
            return (self.entries.len(), 0, workspace);
        }
        let capacity = if self.entries.capacity() > keys * 2 {
            keys
        } else {
            self.entries.capacity()
        };
        let current_backing = super::payload::backing_buckets(&self.buckets);
        let buckets = if super::payload::bucket_capacity(current_backing) > (keys * 2).max(4) {
            required_backing(keys)
        } else {
            current_backing
        };
        let metadata = (capacity * size_of::<Entry>()
            + hash_allocation(buckets)
            + keys * (size_of::<RightBucket>() + 2 * size_of::<usize>()))
            as u64
            + rows_bytes;
        if keys != self.entries.len() {
            workspace += (capacity * size_of::<Entry>() + hash_allocation(buckets)) as u64;
        }
        (self.entries.len() - keys, metadata, workspace)
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

    pub fn last_key_value(&self) -> Option<(&Encoding, &RightBucket)> {
        let entry = &self.entries[self.greatest_key? as usize];
        Some((&entry.key, entry.bucket.as_ref()))
    }

    pub fn bucket_mut_or_default(&mut self, key: Encoding) -> &mut RightBucket {
        self.bucket_mut_or_kind(key, SequenceKind::Canonical)
    }

    pub fn bucket_mut_or_kind(&mut self, key: Encoding, kind: SequenceKind) -> &mut RightBucket {
        let hash = self.hasher.hash_one(key.as_slice());
        let id = self.find(hash, key.as_slice()).unwrap_or_else(|| {
            self.insert_unique(hash, key, RightBucket::with_sequence_kind(kind))
        });
        Arc::make_mut(&mut self.entries[id].bucket)
    }

    pub fn insert(&mut self, key: Encoding, bucket: RightBucket) {
        let hash = self.hasher.hash_one(key.as_slice());
        if let Some(id) = self.find(hash, key.as_slice()) {
            self.entries[id].bucket = Arc::new(bucket);
        } else {
            self.insert_unique(hash, key, bucket);
        }
    }

    fn insert_unique(&mut self, hash: u64, key: Encoding, bucket: RightBucket) -> usize {
        let id = u32::try_from(self.entries.len()).expect("preflighted ASOF key domain");
        if self.entries.capacity() == 0 {
            // Vec's four-entry minimum is unnecessary for a sparse dictionary.
            self.entries = Vec::with_capacity(1);
        }
        if self
            .greatest_key
            .is_none_or(|previous| self.entries[previous as usize].key < key)
        {
            self.greatest_key = Some(id);
        }
        self.entries.push(Entry {
            key,
            hash,
            bucket: Arc::new(bucket),
        });
        let entries = &self.entries;
        self.buckets
            .insert_unique(hash, id, |id| entries[*id as usize].hash);
        id as usize
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&Encoding, &mut Arc<RightBucket>) -> bool) {
        let previous = self.entries.len();
        self.entries
            .retain_mut(|entry| keep(&entry.key, &mut entry.bucket));
        if self.entries.len() == previous {
            return;
        }
        if self.entries.is_empty() {
            self.entries = Vec::new();
            self.buckets = HashTable::new();
            self.greatest_key = None;
            return;
        }
        if self.entries.capacity() > self.entries.len().saturating_mul(2) {
            self.entries = std::mem::take(&mut self.entries)
                .into_boxed_slice()
                .into_vec();
        }
        self.buckets.clear();
        self.greatest_key = None;
        for (id, entry) in self.entries.iter().enumerate() {
            let id = u32::try_from(id).expect("retained ASOF key domain");
            if self
                .greatest_key
                .is_none_or(|previous| self.entries[previous as usize].key < entry.key)
            {
                self.greatest_key = Some(id);
            }
            self.buckets
                .insert_unique(entry.hash, id, |id| self.entries[*id as usize].hash);
        }
        if self.buckets.capacity() > self.entries.len().saturating_mul(2).max(4) {
            self.buckets
                .shrink_to_fit(|id| self.entries[*id as usize].hash);
        }
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
        assert_eq!(state.find(0, b"beta"), Some(0));
        assert_eq!(state.find(0, b"gamma"), Some(1));
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
            bucket.insert_admitted((time, sequence.clone()), RowRef::fixture(time as u32));
        }
        bucket.reserve_payloads(1);
        bucket.insert_admitted((7, sequence.clone()), RowRef::fixture(7));
        let mut state = RightState::default();
        state.insert(key.clone(), bucket);
        let frozen = state.owned_buckets().next().unwrap().1;
        let projected =
            state.metadata_bytes() + state.projected_admission_growth(&[(key.clone(), 1)], kind);
        let live = state.bucket_mut_or_kind(key, kind);
        live.reserve_payloads(1);
        live.insert_admitted((8, sequence), RowRef::fixture(8));
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
