use std::sync::Arc;

use datafusion::execution::memory_pool::MemoryReservation;
use hashbrown::HashTable;

use super::{Lease, counter_overflow, grant};
use crate::Result;

struct Binding {
    row_id: u64,
    lease: Lease,
}

#[derive(Default)]
pub(in crate::operator::join) struct SideBindings {
    entries: HashTable<Binding>,
    _credit: Option<MemoryReservation>,
}

#[derive(Default)]
pub(in crate::operator::join) struct Bindings {
    pub(in crate::operator::join) left: SideBindings,
    pub(in crate::operator::join) right: SideBindings,
}

impl SideBindings {
    pub(super) fn prepare(
        count: usize,
        master: &MemoryReservation,
        name: &str,
    ) -> Result<Option<Self>> {
        if count == 0 {
            return Ok(Some(Self::default()));
        }
        let bytes = count
            .checked_mul(size_of::<Binding>() + 64)
            .and_then(|bytes| bytes.checked_mul(4))
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| counter_overflow(name, "key_bindings"))?;
        Ok(grant(master, bytes, name)?.map(|credit| Self {
            entries: HashTable::with_capacity(count),
            _credit: Some(credit),
        }))
    }

    pub(super) fn insert(&mut self, row_id: u64, lease: Lease) {
        assert!(self.get(row_id).is_none(), "unique live row identity");
        self.entries
            .insert_unique(hash(row_id), Binding { row_id, lease }, |binding| {
                hash(binding.row_id)
            });
    }

    pub(in crate::operator::join) fn get(&self, row_id: u64) -> Option<&Lease> {
        self.entries
            .find(hash(row_id), |binding| binding.row_id == row_id)
            .map(|binding| &binding.lease)
    }

    pub(in crate::operator::join) fn canonical<'a>(
        &'a self,
        row: &'a super::super::StoredRow,
    ) -> &'a Arc<super::FramedKey> {
        self.get(row.row_id).map_or(&row.encoded_key, Lease::owner)
    }

    pub(in crate::operator::join) fn canonical_key<'a>(
        &'a self,
        row_id: u64,
        key: &'a Arc<super::FramedKey>,
    ) -> &'a Arc<super::FramedKey> {
        self.get(row_id).map_or(key, Lease::owner)
    }

    pub(in crate::operator::join) fn remove(
        &mut self,
        row_id: u64,
    ) -> Option<Arc<super::FramedKey>> {
        let entry = self
            .entries
            .find_entry(hash(row_id), |binding| binding.row_id == row_id)
            .ok()?;
        let (binding, _) = entry.remove();
        let canonical = Arc::clone(binding.lease.owner());
        drop(binding);
        if self.entries.is_empty() {
            *self = Self::default();
        }
        Some(canonical)
    }

    #[cfg(test)]
    pub(in crate::operator::join) fn bytes(&self) -> usize {
        let Self {
            _credit: credit, ..
        } = self;
        credit.as_ref().map_or(0, MemoryReservation::size)
    }
}

impl Bindings {
    pub(in crate::operator::join) fn side(&self, left: bool) -> &SideBindings {
        if left { &self.left } else { &self.right }
    }
}

fn hash(row_id: u64) -> u64 {
    row_id.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(27)
}
