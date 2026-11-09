use std::collections::BTreeMap;

use super::{Key, add};
use crate::Result;

#[derive(Clone, Default)]
pub(in super::super) struct Charges {
    owned: u64,
    allocations: BTreeMap<Key, u64>,
}

impl Charges {
    pub(in super::super) fn buffer(&mut self, identity: usize, bytes: u64) {
        self.allocations
            .entry((0, identity))
            .and_modify(|current| *current = (*current).max(bytes))
            .or_insert(bytes);
    }

    pub(super) fn new(owned: u64, allocations: impl Iterator<Item = (Key, u64)>) -> Self {
        Self {
            owned,
            allocations: allocations.collect(),
        }
    }

    pub(in super::super) fn add_owned(&mut self, bytes: u64, node: &str) -> Result<()> {
        self.owned = add(self.owned, bytes, node)?;
        Ok(())
    }

    pub(in super::super) fn merge(&mut self, other: &Self, node: &str) -> Result<()> {
        self.add_owned(other.owned, node)?;
        self.merge_owners(other);
        Ok(())
    }

    pub(in super::super) fn merge_owners(&mut self, other: &Self) {
        for (key, bytes) in &other.allocations {
            self.allocations
                .entry(*key)
                .and_modify(|current| *current = (*current).max(*bytes))
                .or_insert(*bytes);
        }
    }

    pub(in super::super) fn bytes(&self, node: &str) -> Result<u64> {
        self.allocations
            .values()
            .try_fold(self.owned, |total, bytes| add(total, *bytes, node))
    }
}
