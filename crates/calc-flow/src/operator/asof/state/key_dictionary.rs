//! Canonical key bytes owned once by the retained right dictionary. Hash table
//! slots contain only u32 handles; arrival IDs never determine output order.

use super::{Encoding, RightBucket};
use crate::Result;
use datafusion::common::hash_utils::RandomState;
use hashbrown::HashTable;
use std::hash::BuildHasher;

#[derive(Clone)]
struct Entry {
    key: Encoding,
    hash: u64,
    bucket: Box<RightBucket>,
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

    pub fn bucket_by_id(&self, id: u32) -> &RightBucket {
        &self.entries[id as usize].bucket
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
        let hash = self.hasher.hash_one(key.as_slice());
        let id = self
            .find(hash, key.as_slice())
            .unwrap_or_else(|| self.insert_unique(hash, key, RightBucket::default()));
        &mut self.entries[id].bucket
    }

    pub fn insert(&mut self, key: Encoding, bucket: RightBucket) {
        let hash = self.hasher.hash_one(key.as_slice());
        if let Some(id) = self.find(hash, key.as_slice()) {
            *self.entries[id].bucket = bucket;
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
            bucket: Box::new(bucket),
        });
        let entries = &self.entries;
        self.buckets
            .insert_unique(hash, id, |id| entries[*id as usize].hash);
        id as usize
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&Encoding, &mut RightBucket) -> bool) {
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
