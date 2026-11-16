use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use super::internal;
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct KeyGroupId(pub u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct PageId {
    pub(crate) key_group: KeyGroupId,
    pub(crate) slot: u32,
    pub(crate) generation: u64,
}

#[derive(Clone, Debug)]
pub(super) struct Page<T> {
    pub(super) id: PageId,
    pub(super) value: Arc<T>,
}

impl<T> Page<T> {
    pub(super) fn new(id: PageId, value: T) -> Self {
        Self {
            id,
            value: Arc::new(value),
        }
    }
}

impl<T: Clone> Page<T> {
    pub(super) fn cow(&self, generation: u64) -> Self {
        Self::new(
            PageId {
                generation,
                ..self.id
            },
            (*self.value).clone(),
        )
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct DirtySet {
    pub(super) bits: BTreeSet<u32>,
    pub(super) first_dirty: Vec<u32>,
}

impl DirtySet {
    pub(super) fn mark(&mut self, slot: u32) {
        if self.bits.insert(slot) {
            self.first_dirty.push(slot);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct SlotAllocator {
    pub(super) next_generations: BTreeMap<u32, u64>,
    pub(super) free_slots: BTreeSet<u32>,
    pub(super) next_slot: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AllocatorUndo {
    pub(super) next_generation: Option<u64>,
    pub(super) next_slot: u32,
    pub(super) was_free: bool,
}

impl SlotAllocator {
    pub(super) fn allocate(&mut self, key_group: KeyGroupId) -> Result<(PageId, AllocatorUndo)> {
        let next_slot = self.next_slot;
        let reused = self.free_slots.first().copied();
        let slot = reused.unwrap_or(next_slot);
        let previous = self.next_generations.get(&slot).copied();
        let generation = previous.unwrap_or(0);
        let next_generation = generation
            .checked_add(1)
            .ok_or_else(|| internal("state-table page generation exhausted"))?;
        let following_slot = if reused.is_none() {
            next_slot
                .checked_add(1)
                .ok_or_else(|| internal("state-table slot space exhausted"))?
        } else {
            next_slot
        };
        if reused.is_some() {
            self.free_slots.pop_first();
        } else {
            self.next_slot = following_slot;
        }
        self.next_generations.insert(slot, next_generation);
        Ok((
            PageId {
                key_group,
                slot,
                generation,
            },
            AllocatorUndo {
                next_generation: previous,
                next_slot,
                was_free: reused.is_some(),
            },
        ))
    }

    pub(super) fn replace(&mut self, slot: u32) -> Result<(u64, AllocatorUndo)> {
        let previous = self.next_generations.get(&slot).copied();
        let generation = previous.unwrap_or(0);
        self.next_generations.insert(
            slot,
            generation
                .checked_add(1)
                .ok_or_else(|| internal("state-table page generation exhausted"))?,
        );
        Ok((
            generation,
            AllocatorUndo {
                next_generation: previous,
                next_slot: self.next_slot,
                was_free: false,
            },
        ))
    }

    pub(super) fn remove(&mut self, slot: u32) -> Result<(u64, AllocatorUndo)> {
        let (generation, undo) = self.replace(slot)?;
        self.free_slots.insert(slot);
        Ok((generation, undo))
    }

    pub(super) fn restore(&mut self, slot: u32, undo: AllocatorUndo) {
        match undo.next_generation {
            Some(generation) => self.next_generations.insert(slot, generation),
            None => self.next_generations.remove(&slot),
        };
        self.next_slot = undo.next_slot;
        if undo.was_free {
            self.free_slots.insert(slot);
        } else {
            self.free_slots.remove(&slot);
        }
    }
}
