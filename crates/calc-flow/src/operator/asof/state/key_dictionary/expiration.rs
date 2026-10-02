use super::Entry;

pub(super) const ABSENT: u32 = u32::MAX;

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct Deadline {
    time: i64,
    id: u32,
}

const _: () = assert!(size_of::<Deadline>() == 16);

#[derive(Clone, Copy)]
pub(super) enum Kind {
    Payload,
    Identity,
}

impl Kind {
    fn position(self, entry: &Entry) -> u32 {
        match self {
            Self::Payload => entry.payload_position,
            Self::Identity => entry.identity_position,
        }
    }

    fn set_position(self, entry: &mut Entry, position: u32) {
        match self {
            Self::Payload => entry.payload_position = position,
            Self::Identity => entry.identity_position = position,
        }
    }
}

pub(super) struct Heap {
    values: Vec<Deadline>,
    kind: Kind,
}

impl Heap {
    pub fn new(kind: Kind, capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            kind,
        }
    }

    pub fn copy_with_capacity(&self, capacity: usize) -> Self {
        let mut values = Vec::with_capacity(capacity);
        values.extend_from_slice(&self.values);
        Self {
            values,
            kind: self.kind,
        }
    }

    pub fn copy_records_from(&mut self, source: &Self) {
        debug_assert!(self.capacity() >= source.values.len());
        self.values.extend_from_slice(&source.values);
    }

    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }

    pub fn allocation_bytes(&self) -> usize {
        self.capacity() * size_of::<Deadline>()
    }

    pub fn reserve_exact(&mut self, capacity: usize) {
        if self.capacity() < capacity {
            self.values.reserve_exact(capacity - self.values.len());
        }
    }

    pub fn minimum(&self) -> Option<i64> {
        self.values.first().map(|value| value.time)
    }

    pub fn due_count(&self, cutoff: i128) -> usize {
        self.count_subtree(0, cutoff)
    }

    fn count_subtree(&self, position: usize, cutoff: i128) -> usize {
        if !self.is_due(position, cutoff) {
            return 0;
        }
        1 + self.count_subtree(position * 2 + 1, cutoff)
            + self.count_subtree(position * 2 + 2, cutoff)
    }

    fn is_due(&self, position: usize, cutoff: i128) -> bool {
        self.values
            .get(position)
            .is_some_and(|value| i128::from(value.time) < cutoff)
    }

    pub fn collect_due(&self, cutoff: i128, ids: &mut Vec<u32>) {
        self.collect_subtree(0, cutoff, ids);
    }

    fn collect_subtree(&self, position: usize, cutoff: i128, ids: &mut Vec<u32>) {
        if !self.is_due(position, cutoff) {
            return;
        }
        ids.push(self.values[position].id);
        self.collect_subtree(position * 2 + 1, cutoff, ids);
        self.collect_subtree(position * 2 + 2, cutoff, ids);
    }

    pub fn replace(&mut self, entries: &mut [Entry], id: u32, time: Option<i64>) {
        let position = self.kind.position(&entries[id as usize]);
        match (position, time) {
            (ABSENT, None) => {}
            (ABSENT, Some(time)) => self.insert(entries, Deadline { time, id }),
            (_, None) => self.remove(entries, position as usize),
            (_, Some(time)) => {
                self.values[position as usize].time = time;
                self.repair(entries, position as usize);
            }
        }
    }

    fn insert(&mut self, entries: &mut [Entry], deadline: Deadline) {
        let position = self.values.len();
        self.values.push(deadline);
        self.set_position(entries, position);
        self.sift_up(entries, position);
    }

    fn remove(&mut self, entries: &mut [Entry], position: usize) {
        let removed = self.values.swap_remove(position);
        self.kind
            .set_position(&mut entries[removed.id as usize], ABSENT);
        if position < self.values.len() {
            self.set_position(entries, position);
            self.repair(entries, position);
        }
    }

    pub fn rename(&mut self, entries: &mut [Entry], id: u32) {
        let position = self.kind.position(&entries[id as usize]);
        if position != ABSENT {
            self.values[position as usize].id = id;
            self.repair(entries, position as usize);
        }
    }

    fn repair(&mut self, entries: &mut [Entry], position: usize) {
        if position > 0 && self.values[position] < self.values[(position - 1) / 2] {
            self.sift_up(entries, position);
        } else {
            self.sift_down(entries, position);
        }
    }

    fn sift_up(&mut self, entries: &mut [Entry], mut position: usize) {
        while position > 0 {
            let parent = (position - 1) / 2;
            if self.values[parent] <= self.values[position] {
                break;
            }
            self.swap(entries, position, parent);
            position = parent;
        }
    }

    fn sift_down(&mut self, entries: &mut [Entry], mut position: usize) {
        while let Some(child) = self.smaller_child(position) {
            if self.values[position] <= self.values[child] {
                break;
            }
            self.swap(entries, position, child);
            position = child;
        }
    }

    fn smaller_child(&self, position: usize) -> Option<usize> {
        let left = position * 2 + 1;
        if left >= self.values.len() {
            return None;
        }
        let right = left + 1;
        Some(
            if right < self.values.len() && self.values[right] < self.values[left] {
                right
            } else {
                left
            },
        )
    }

    fn swap(&mut self, entries: &mut [Entry], left: usize, right: usize) {
        self.values.swap(left, right);
        self.set_position(entries, left);
        self.set_position(entries, right);
    }

    fn set_position(&self, entries: &mut [Entry], position: usize) {
        let id = self.values[position].id;
        self.kind.set_position(
            &mut entries[id as usize],
            u32::try_from(position).expect("preflighted ASOF heap domain"),
        );
    }
}
