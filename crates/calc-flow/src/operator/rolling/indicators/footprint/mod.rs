//! Checked ownership accounting for indicator-containing Rolling nodes.

use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::{array::ArrayRef, buffer::Buffer};

use crate::{CalcFlowError, Result};

mod arrays;
mod charges;
mod leaf;
mod scalars;
pub(super) use charges::Charges;

#[cfg(test)]
mod tests;

type Key = (u8, usize);

#[derive(Clone)]
enum Owner {
    Buffer {
        _value: Buffer,
    },
    Array {
        _value: ArrayRef,
    },
    Timezone {
        _value: Arc<str>,
    },
    Encoded {
        _value: Arc<Vec<u8>>,
    },
    Checksum {
        _value: Arc<String>,
    },
    State {
        _value: Arc<dyn std::any::Any + Send + Sync>,
    },
}

#[derive(Clone)]
struct Allocation {
    bytes: u64,
    owner: Owner,
}

#[derive(Clone, Default)]
pub(super) struct Footprint {
    owned: u64,
    allocations: BTreeMap<Key, Allocation>,
}

impl Footprint {
    pub(super) fn pin_state<T: Send + Sync + 'static>(&mut self, state: Arc<T>) {
        let key = (4, Arc::as_ptr(&state) as usize);
        let bytes = std::mem::take(&mut self.owned);
        self.allocation(key, bytes, Owner::State { _value: state });
    }

    pub(super) fn clone_bound(&self, node: &str) -> Result<Self> {
        let mut copied = Self {
            owned: self.owned,
            ..Self::default()
        };
        for (key, allocation) in &self.allocations {
            if key.0 == 4 {
                copied.add_owned(allocation.bytes, node)?;
            } else {
                copied.allocation(*key, allocation.bytes, allocation.owner.clone());
            }
        }
        Ok(copied)
    }

    pub(super) fn charges(&self) -> Charges {
        Charges::new(
            self.owned,
            self.allocations
                .iter()
                .map(|(key, value)| (*key, value.bytes)),
        )
    }

    pub(super) fn segment(&mut self, segment: &crate::StateSegment, node: &str) -> Result<()> {
        let (bytes, checksum) = segment.shared_allocations();
        self.allocation(
            (0, bytes.as_ptr() as usize),
            size(bytes.capacity(), node)?,
            Owner::Encoded { _value: bytes },
        );
        self.allocation(
            (0, checksum.as_ptr() as usize),
            size(checksum.capacity(), node)?,
            Owner::Checksum { _value: checksum },
        );
        Ok(())
    }

    pub(super) fn buffer(&mut self, buffer: &Buffer, node: &str) -> Result<()> {
        let extent = buffer
            .ptr_offset()
            .checked_add(buffer.len())
            .ok_or_else(|| overflow(node))?;
        let bytes = size(buffer.capacity().max(extent), node)?;
        let key = (0, buffer.data_ptr().as_ptr() as usize);
        self.allocation(
            key,
            bytes,
            Owner::Buffer {
                _value: buffer.clone(),
            },
        );
        Ok(())
    }

    pub(super) fn bytes(&self, node: &str) -> Result<u64> {
        self.allocations
            .values()
            .try_fold(self.owned, |total, allocation| {
                add(total, allocation.bytes, node)
            })
    }

    pub(super) fn incremental_bytes(&self, baseline: &Self, node: &str) -> Result<u64> {
        self.allocations
            .iter()
            .try_fold(self.owned, |total, (key, allocation)| {
                let existing = baseline.allocations.get(key).map_or(0, |value| value.bytes);
                add(total, allocation.bytes.saturating_sub(existing), node)
            })
    }

    pub(super) fn merge(&mut self, other: &Self, node: &str) -> Result<()> {
        self.owned = add(self.owned, other.owned, node)?;
        for (key, allocation) in &other.allocations {
            self.allocation(*key, allocation.bytes, allocation.owner.clone());
        }
        Ok(())
    }

    pub(super) fn add_owned(&mut self, bytes: u64, node: &str) -> Result<()> {
        self.owned = add(self.owned, bytes, node)?;
        Ok(())
    }

    fn allocation(&mut self, key: Key, bytes: u64, owner: Owner) {
        self.allocations
            .entry(key)
            .and_modify(|entry| entry.bytes = entry.bytes.max(bytes))
            .or_insert(Allocation { bytes, owner });
    }
}

fn overflow(node: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node.to_owned(),
        message: "rolling participating byte charge overflowed".into(),
    }
}

fn size(value: usize, node: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| overflow(node))
}

fn add(left: u64, right: u64, node: &str) -> Result<u64> {
    left.checked_add(right).ok_or_else(|| overflow(node))
}
