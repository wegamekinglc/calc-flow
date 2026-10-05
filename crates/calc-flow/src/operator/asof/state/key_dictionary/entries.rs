use super::{Encoding, Entry, funding::allocation, funding::total};
use crate::Result;
use std::ops::{Index, IndexMut};

const SHARDS: usize = super::super::KEY_SHARDS;

#[derive(Clone, Copy)]
struct Location {
    shard: u32,
    position: u32,
}

#[derive(Clone)]
struct ShardEntry {
    id: u32,
    entry: Entry,
}

const _: () = assert!(size_of::<Location>() == 8);
const _: () = assert!(size_of::<ShardEntry>() == 56);

#[derive(Clone, Copy)]
pub(super) struct Capacities {
    pub directory: usize,
    pub shards: [usize; SHARDS],
}

impl Capacities {
    pub fn metadata_bytes(self) -> usize {
        self.directory * size_of::<Location>()
            + self.shards.into_iter().sum::<usize>() * size_of::<ShardEntry>()
    }

    pub fn auxiliary_bytes(self, name: &str) -> Result<usize> {
        total(
            &[
                allocation(self.directory, 8, name)?,
                self.shards
                    .into_iter()
                    .try_fold(0_usize, |bytes, capacity| {
                        total(&[bytes, allocation(capacity, 24, name)?], name)
                    })?,
            ],
            name,
        )
    }
}

#[derive(Default)]
pub(super) struct Entries {
    directory: Vec<Location>,
    shards: [Vec<ShardEntry>; SHARDS],
}

pub(super) struct PreparedEntries {
    directory: Option<Vec<Location>>,
    shards: [Option<Vec<ShardEntry>>; SHARDS],
}

impl PreparedEntries {
    pub fn install(self, entries: &mut Entries) {
        if let Some(directory) = self.directory {
            entries.directory = directory;
        }
        for (target, replacement) in entries.shards.iter_mut().zip(self.shards) {
            if let Some(replacement) = replacement {
                *target = replacement;
            }
        }
    }
}

impl Entries {
    pub fn with_capacities(capacities: Capacities) -> Self {
        Self {
            directory: Vec::with_capacity(capacities.directory),
            shards: capacities.shards.map(Vec::with_capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.directory.len()
    }

    pub fn is_empty(&self) -> bool {
        self.directory.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.directory.capacity()
    }

    pub fn capacities(&self) -> Capacities {
        Capacities {
            directory: self.capacity(),
            shards: std::array::from_fn(|shard| self.shards[shard].capacity()),
        }
    }

    #[cfg(test)]
    pub fn counts(&self) -> [usize; SHARDS] {
        std::array::from_fn(|shard| self.shards[shard].len())
    }

    pub fn projected(&self, additions: [usize; SHARDS]) -> Capacities {
        Capacities {
            directory: growth_capacity(
                self.capacity(),
                self.len() + additions.iter().sum::<usize>(),
            ),
            shards: std::array::from_fn(|shard| {
                growth_capacity(
                    self.shards[shard].capacity(),
                    self.shards[shard].len() + additions[shard],
                )
            }),
        }
    }

    pub fn compacted(&self, keys: usize) -> Capacities {
        Capacities {
            directory: keys,
            shards: std::array::from_fn(|shard| self.shards[shard].len().min(keys)),
        }
    }

    pub fn copy_workspace(&self, capacities: Capacities, name: &str) -> Result<usize> {
        let directory = if capacities.directory > self.capacity() {
            allocation(self.capacity() + capacities.directory, 8, name)?
        } else {
            0
        };
        self.shards
            .iter()
            .zip(capacities.shards)
            .try_fold(directory, |bytes, (shard, capacity)| {
                let copied = if capacity > shard.capacity() {
                    allocation(shard.capacity() + capacity, 56, name)?
                } else {
                    0
                };
                total(&[bytes, copied], name)
            })
    }

    pub fn prepare(&self, capacities: Capacities) -> PreparedEntries {
        PreparedEntries {
            directory: (capacities.directory > self.capacity()).then(|| {
                let mut directory = Vec::with_capacity(capacities.directory);
                directory.extend_from_slice(&self.directory);
                directory
            }),
            shards: std::array::from_fn(|shard| {
                (capacities.shards[shard] > self.shards[shard].capacity()).then(|| {
                    let mut entries = Vec::with_capacity(capacities.shards[shard]);
                    entries.extend_from_slice(&self.shards[shard]);
                    entries
                })
            }),
        }
    }

    pub fn can_insert(&self, key: &Encoding) -> bool {
        let shard = &self.shards[super::super::key_shard(key)];
        self.len() < self.capacity() && shard.len() < shard.capacity()
    }

    pub fn push(&mut self, entry: Entry) {
        let shard = super::super::key_shard(&entry.key);
        let target = &mut self.shards[shard];
        if target.capacity() == 0 {
            *target = Vec::with_capacity(1);
        }
        if self.directory.capacity() == 0 {
            self.directory = Vec::with_capacity(1);
        }
        let id = u32::try_from(self.directory.len()).expect("preflighted ASOF key domain");
        let position = u32::try_from(target.len()).expect("preflighted ASOF shard key domain");
        target.push(ShardEntry { id, entry });
        self.directory.push(Location {
            shard: u32::try_from(shard).expect("bounded ASOF shard"),
            position,
        });
    }

    pub fn swap_remove(&mut self, id: usize) -> Entry {
        let location = self.directory[id];
        let shard = &mut self.shards[location.shard as usize];
        let removed = shard.swap_remove(location.position as usize);
        if let Some(moved) = shard.get(location.position as usize) {
            self.directory[moved.id as usize].position = location.position;
        }
        self.directory.swap_remove(id);
        if let Some(moved) = self.directory.get(id) {
            self.shards[moved.shard as usize][moved.position as usize].id =
                u32::try_from(id).expect("retained ASOF key domain");
        }
        removed.entry
    }

    pub fn extend_from_entries(&mut self, source: &Self) {
        for entry in source.iter() {
            self.push(entry.clone());
        }
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter {
            entries: self,
            position: 0,
        }
    }
}

impl Index<usize> for Entries {
    type Output = Entry;
    fn index(&self, id: usize) -> &Entry {
        let location = self.directory[id];
        &self.shards[location.shard as usize][location.position as usize].entry
    }
}

impl IndexMut<usize> for Entries {
    fn index_mut(&mut self, id: usize) -> &mut Entry {
        let location = self.directory[id];
        &mut self.shards[location.shard as usize][location.position as usize].entry
    }
}

pub(super) struct Iter<'a> {
    entries: &'a Entries,
    position: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a Entry;
    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.entries.len() {
            return None;
        }
        let id = self.position;
        self.position += 1;
        Some(&self.entries[id])
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.entries.len() - self.position;
        (count, Some(count))
    }
}

impl ExactSizeIterator for Iter<'_> {}

fn growth_capacity(mut capacity: usize, required: usize) -> usize {
    if capacity == 0 && required != 0 {
        capacity = 1;
    }
    while capacity < required {
        capacity = (capacity * 2).max(4);
    }
    capacity
}
