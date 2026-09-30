//! Count shared encoding allocations independently of their live row handles.

use super::Encoding;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct Allocation {
    references: usize,
    bytes: u64,
    encoded_length: u64,
}

#[derive(Clone, Default)]
pub(in super::super) struct EncodingOwners {
    allocations: BTreeMap<usize, Allocation>,
    bytes: u64,
    encoded_length: u64,
}

pub(in super::super) type OwnerRemovals = BTreeMap<usize, usize>;

pub(in super::super) struct OwnerUpdates {
    allocations: BTreeMap<usize, Allocation>,
    bytes: u64,
    encoded_length: u64,
    new_owners: usize,
}

fn metadata_bytes(count: usize) -> u64 {
    if count == 0 {
        0
    } else {
        384 + count as u64 * 128
    }
}

impl EncodingOwners {
    pub fn project_add_counts<'a>(
        &self,
        values: impl Iterator<Item = (&'a Encoding, usize)>,
    ) -> OwnerUpdates {
        let mut updates = OwnerUpdates {
            allocations: BTreeMap::new(),
            bytes: self.bytes,
            encoded_length: self.encoded_length,
            new_owners: 0,
        };
        for (encoding, count) in values {
            let Some((id, bytes)) = encoding.allocation() else {
                continue;
            };
            let allocation = updates.allocations.entry(id).or_insert_with(|| {
                if let Some(previous) = self.allocations.get(&id) {
                    return *previous;
                }
                let encoded_length = encoding.owner_wire_length();
                updates.bytes += bytes;
                updates.encoded_length += encoded_length;
                updates.new_owners += 1;
                Allocation {
                    references: 0,
                    bytes,
                    encoded_length,
                }
            });
            allocation.references += count;
        }
        updates
    }

    pub fn projected_allocation_bytes(&self, updates: &OwnerUpdates) -> u64 {
        updates.bytes + metadata_bytes(self.allocations.len() + updates.new_owners)
    }

    pub fn projected_encoded_length(updates: &OwnerUpdates) -> u64 {
        updates.encoded_length
    }

    pub fn commit_add(&mut self, updates: OwnerUpdates) {
        self.bytes = updates.bytes;
        self.encoded_length = updates.encoded_length;
        for (id, allocation) in updates.allocations {
            self.allocations.insert(id, allocation);
        }
    }

    pub fn attach(&mut self, encoding: &Encoding) {
        let Some((id, bytes)) = encoding.allocation() else {
            return;
        };
        let allocation = self.allocations.entry(id).or_insert_with(|| {
            self.bytes += bytes;
            let encoded_length = encoding.owner_wire_length();
            self.encoded_length += encoded_length;
            Allocation {
                references: 0,
                bytes,
                encoded_length,
            }
        });
        allocation.references += 1;
    }

    pub fn detach(&mut self, encoding: &Encoding) {
        if let Some((id, _)) = encoding.allocation() {
            self.detach_count(id, 1);
        }
    }

    pub fn detach_count(&mut self, id: usize, count: usize) {
        let allocation = self.allocations.get_mut(&id).expect("owned ASOF encoding");
        allocation.references -= count;
        if allocation.references != 0 {
            return;
        }
        self.bytes -= allocation.bytes;
        self.encoded_length -= allocation.encoded_length;
        self.allocations.remove(&id);
        if self.allocations.is_empty() {
            self.allocations = BTreeMap::new();
        }
    }

    pub fn remove(&mut self, removals: &OwnerRemovals) {
        for (&id, &count) in removals {
            self.detach_count(id, count);
        }
    }

    pub fn record_remove(removals: &mut OwnerRemovals, encoding: &Encoding, count: usize) {
        if let Some((id, _)) = encoding.allocation() {
            *removals.entry(id).or_default() += count;
        }
    }

    pub fn projected_remove(&self, removals: &OwnerRemovals) -> (u64, u64) {
        let mut bytes = self.bytes;
        let mut encoded_length = self.encoded_length;
        let mut count = self.allocations.len();
        for (&id, &removed) in removals {
            let allocation = &self.allocations[&id];
            assert!(
                removed <= allocation.references,
                "owned ASOF encoding count"
            );
            if removed == allocation.references {
                bytes -= allocation.bytes;
                encoded_length -= allocation.encoded_length;
                count -= 1;
            }
        }
        (bytes + metadata_bytes(count), encoded_length)
    }

    pub fn projected_metadata_bytes(&self, removals: &OwnerRemovals) -> u64 {
        let released = removals
            .iter()
            .filter(|(id, count)| self.allocations[id].references == **count)
            .count();
        metadata_bytes(self.allocations.len() - released)
    }

    pub fn encoded_length(&self) -> u64 {
        self.encoded_length
    }
    pub fn buffers_bytes(&self) -> u64 {
        self.bytes
    }
    pub fn metadata_bytes(&self) -> u64 {
        metadata_bytes(self.allocations.len())
    }
    pub fn allocation_bytes(&self) -> u64 {
        self.buffers_bytes() + self.metadata_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregated_owner_additions_keep_one_allocation_until_the_last_reference() {
        let encoding = Encoding::from_slice(&[7; 64]);
        let (id, _) = encoding.allocation().unwrap();
        let mut owners = EncodingOwners::default();
        let updates = owners.project_add_counts([(&encoding, 3), (&encoding, 4)].into_iter());
        owners.commit_add(updates);
        assert_eq!(owners.allocations.len(), 1);
        assert_eq!(owners.allocations[&id].references, 7);
        let bytes = owners.buffers_bytes();
        let encoded = owners.encoded_length();
        owners.detach_count(id, 6);
        assert_eq!(owners.buffers_bytes(), bytes);
        assert_eq!(owners.encoded_length(), encoded);
        owners.detach_count(id, 1);
        assert_eq!(owners.buffers_bytes(), 0);
        assert_eq!(owners.encoded_length(), 0);
        assert_eq!(owners.metadata_bytes(), 0);
    }
}
