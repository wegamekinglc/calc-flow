//! Project native state changes from the committed ledger and changed inputs.
//! Retained payload and owner dictionaries are never cloned for preflight.

use super::{
    Encoding, EncodingOwners, Inventory, LeftOrder, LeftPrefix, OwnerUpdates, PreparedLeftChunk,
    PreparedLeftDrain, State,
};
use crate::{
    Result,
    operator::asof::{StreamAsofJoinStatus, checked},
};

#[derive(Clone, Copy)]
pub(in super::super) struct CapacitySnapshot {
    pub inventory: Inventory,
    pub index_length: u64,
    pub index_bytes: u64,
}

impl CapacitySnapshot {
    fn inventory_without_index(self) -> Inventory {
        Inventory {
            bytes: self.inventory.bytes - self.index_bytes,
            ..self.inventory
        }
    }
}

fn indexed_inventory_bytes(bytes: u64, length: u64, name: &str) -> Result<u64> {
    checked(name, bytes, checked(name, length, 256)?)
}

impl State {
    pub fn admission_staging_bytes<R>(
        &self,
        rows: &[(LeftOrder, R)],
        chunks: Option<&[PreparedLeftChunk]>,
        right_counts: &[(Encoding, usize)],
        batches: &[std::sync::Arc<super::PayloadBatch>],
        name: &str,
    ) -> Result<u64> {
        let owners = self
            .admission_owned_encodings(rows, chunks, right_counts)
            .filter(|(encoding, _)| encoding.allocation().is_some())
            .count() as u64;
        let owner_workspace = owners
            .checked_mul(128)
            .and_then(|bytes| bytes.checked_add(384))
            .ok_or_else(|| {
                super::super::reason(
                    name,
                    crate::StreamingFailureReason::AsofCounterOverflow,
                    "ASOF owner workspace overflowed",
                )
            })?;
        checked(
            name,
            checked(
                name,
                owner_workspace,
                (batches.len() * size_of::<super::RowRef>()) as u64,
            )?,
            self.batches.project_admission(batches, name)?.workspace,
        )
    }

    pub fn project_capacity_admission<R>(
        &self,
        snapshot: CapacitySnapshot,
        rows: &[(LeftOrder, R)],
        chunks: Option<&[PreparedLeftChunk]>,
        right_counts: &[(Encoding, usize)],
        batches: &[std::sync::Arc<super::PayloadBatch>],
        name: &str,
    ) -> Result<(u64, Inventory, OwnerUpdates)> {
        let owners = self.encoding_owners.as_ref().expect("tracked native state");
        let updates =
            owners.project_add_counts(self.admission_owned_encodings(rows, chunks, right_counts));
        let pool = self.batches.project_admission(batches, name)?;
        let mut inventory = snapshot.inventory_without_index();
        inventory =
            self.admission_column_inventory(inventory, rows.len(), chunks, right_counts, name)?;
        inventory.bytes = checked(
            name,
            inventory.bytes,
            owners.projected_allocation_bytes(&updates) - owners.allocation_bytes(),
        )?;
        inventory.bytes = inventory.bytes - self.batches.metadata_bytes() + pool.metadata_bytes;
        inventory.bytes = checked(name, inventory.bytes, pool.payload_bytes)?;
        let length = self.project_admission_length(
            snapshot.index_length,
            &updates,
            chunks,
            right_counts,
            name,
        )?;
        super::validate_key_count((self.batches.len() + pool.new_batches) as u64, name)?;
        inventory.bytes = indexed_inventory_bytes(inventory.bytes, length, name)?;
        Ok((length, inventory, updates))
    }

    /// Integer sequence columns own no canonical encoding allocations. Reuse
    /// admission's per-key row counts instead of visiting every row again.
    fn admission_owned_encodings<'a, R: 'a>(
        &self,
        rows: &'a [(LeftOrder, R)],
        chunks: Option<&'a [PreparedLeftChunk]>,
        right_counts: &'a [(Encoding, usize)],
    ) -> impl Iterator<Item = (&'a Encoding, usize)> {
        let left = chunks.filter(|_| self.sequence_kinds[0].width().is_some());
        let right = chunks.is_none()
            && !right_counts.is_empty()
            && self.sequence_kinds[1].width().is_some();
        let generic = if left.is_some() || right {
            &[][..]
        } else {
            rows
        };
        let right_counts = if right { right_counts } else { &[][..] };
        generic
            .iter()
            .flat_map(|(order, _)| [(&order.1, 1), (&order.2, 1)])
            .chain(
                left.unwrap_or(&[])
                    .iter()
                    .flat_map(PreparedLeftChunk::key_counts),
            )
            .chain(right_counts.iter().map(|(key, count)| (key, *count)))
    }

    fn project_admission_length(
        &self,
        length: u64,
        updates: &OwnerUpdates,
        chunks: Option<&[PreparedLeftChunk]>,
        right_counts: &[(Encoding, usize)],
        name: &str,
    ) -> Result<u64> {
        let owners = self.encoding_owners.as_ref().expect("tracked native state");
        let length = checked(
            name,
            length.max(80),
            EncodingOwners::projected_encoded_length(updates) - owners.encoded_length(),
        )?;
        self.admission_index_length(length, chunks, right_counts, name)
    }

    fn admission_column_inventory(
        &self,
        mut inventory: Inventory,
        count: usize,
        chunks: Option<&[PreparedLeftChunk]>,
        right_counts: &[(Encoding, usize)],
        name: &str,
    ) -> Result<Inventory> {
        inventory.identities = checked(name, inventory.identities, count as u64)?;
        let growth = if let Some(chunks) = chunks {
            self.left.projected_admission_bytes(chunks, name)? - self.left.capacity_bytes(name)?
        } else {
            inventory.right_payloads = checked(name, inventory.right_payloads, count as u64)?;
            self.right
                .projected_admission_growth(right_counts, self.sequence_kinds[1])
        };
        inventory.bytes = checked(name, inventory.bytes, growth)?;
        Ok(inventory)
    }

    fn admission_index_length(
        &self,
        mut length: u64,
        chunks: Option<&[PreparedLeftChunk]>,
        right_counts: &[(Encoding, usize)],
        name: &str,
    ) -> Result<u64> {
        if let Some(chunks) = chunks {
            for chunk in chunks {
                length = checked(name, length, chunk.v3_length(self.sequence_kinds[0]))?;
            }
        } else {
            length = self.right_admission_index_length(length, right_counts, name)?;
        }
        Ok(length)
    }

    fn right_admission_index_length(
        &self,
        mut length: u64,
        counts: &[(Encoding, usize)],
        name: &str,
    ) -> Result<u64> {
        let mut new_keys = 0;
        for (key, count) in counts {
            if !self.right.contains_key(key) {
                new_keys += 1;
                length = checked(name, length, 65)?;
            }
            length = checked(
                name,
                length,
                *count as u64 * (21 + self.sequence_kinds[1].reference_bytes()),
            )?;
        }
        super::validate_key_count((self.right.len() + new_keys) as u64, name)?;
        Ok(length)
    }

    pub fn project_capacity_prefix(
        &self,
        snapshot: CapacitySnapshot,
        prefix: &LeftPrefix,
        drain: &PreparedLeftDrain,
        name: &str,
    ) -> Result<(u64, Inventory, u64)> {
        let pool = self.batches.project_remove(&prefix.batches, name)?;
        let owners = self.encoding_owners.as_ref().expect("tracked native state");
        let (owner_bytes, owner_length) = owners.projected_remove(&prefix.owners);
        let (left_bytes, removed_index) = self.left.projected_drain(
            prefix,
            drain,
            &self.batches,
            self.sequence_kinds[0],
            name,
        )?;
        let mut length =
            snapshot.index_length - removed_index - (owners.encoded_length() - owner_length);
        let mut inventory = snapshot.inventory_without_index();
        inventory.identities -= prefix.count as u64;
        if inventory.identities == 0 {
            length = 0;
        }
        inventory.bytes = inventory.bytes - self.left.capacity_bytes(name)? + left_bytes;
        inventory.bytes = inventory.bytes - owners.allocation_bytes() + owner_bytes;
        inventory.bytes = inventory.bytes - self.batches.metadata_bytes() + pool.metadata_bytes
            - pool.released_bytes;
        if length != 0 {
            inventory.bytes = checked(name, inventory.bytes, length + 256)?;
        }
        Ok((length, inventory, pool.workspace))
    }

    pub fn project_capacity_eviction(
        &self,
        snapshot: CapacitySnapshot,
        preview: &super::EvictionPreview,
        status: &StreamAsofJoinStatus,
        tolerance: u64,
        name: &str,
    ) -> Result<(u64, Inventory, u64)> {
        let pool = self.batches.project_remove(&preview.batches, name)?;
        let owners = self.encoding_owners.as_ref().expect("tracked native state");
        let (owner_bytes, owner_length) = owners.projected_remove(&preview.owners);
        let (removed_keys, right_bytes, workspace) = self.right.projected_eviction_bytes(
            status,
            tolerance,
            super::retention_threshold(self, status),
        );
        let mut inventory = snapshot.inventory_without_index();
        inventory.identities -= preview.removed_identities;
        inventory.right_payloads -= preview.evicted_payloads;
        inventory.identity_only =
            inventory.identity_only + preview.added_identity_only - preview.removed_identity_only;
        let removed_rows = preview.removed_identities
            * (9 + self.sequence_kinds[1].reference_bytes())
            + preview.evicted_payloads * 12
            + removed_keys as u64 * 65;
        let mut length =
            snapshot.index_length - removed_rows - (owners.encoded_length() - owner_length);
        if inventory.identities == 0 {
            length = 0;
        }
        inventory.bytes = inventory.bytes - self.right.metadata_bytes() + right_bytes;
        inventory.bytes = inventory.bytes - owners.allocation_bytes() + owner_bytes;
        inventory.bytes = inventory.bytes - self.batches.metadata_bytes() + pool.metadata_bytes
            - pool.released_bytes;
        if length != 0 {
            inventory.bytes = checked(name, inventory.bytes, length + 256)?;
        }
        Ok((length, inventory, workspace))
    }

    #[cfg(test)]
    pub fn capacity_snapshot(&self, name: &str) -> CapacitySnapshot {
        let length = super::super::checkpoint::v3_encoded_length(self, name).unwrap();
        let index_bytes = if length == 0 { 0 } else { length + 256 };
        let mut inventory = self.capacity_inventory(None, name).unwrap();
        inventory.bytes += index_bytes;
        CapacitySnapshot {
            inventory,
            index_length: length,
            index_bytes,
        }
    }
}
