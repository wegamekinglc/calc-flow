#[cfg(test)]
use super::borrowed_key::framed_hash;
use std::sync::Arc;

use datafusion::execution::memory_pool::MemoryReservation;
use hashbrown::HashTable;

use super::super::key_identity::{IdentityLease, NamespaceLease};
use super::{
    AdmittedRows,
    borrowed_key::KeyHashState,
    columnar::FramedKey,
    counter_overflow,
    key_arena::KeyWriter,
    typed_key::{KeyFlow, TypedKey},
};
use crate::Result;

mod bindings;
mod framed;
mod rebuild;
pub(super) use bindings::{Bindings, SideBindings};
pub(super) use rebuild::Rebuilt;

pub(super) type Lease = IdentityLease<Arc<FramedKey>>;

struct Entry {
    lease: Lease,
    hash: u64,
    retained: u64,
    next: Option<u32>,
    previous: Option<u32>,
    provisional: u64,
    touched_next: Option<u32>,
}

struct Head {
    hash: u64,
    id: u32,
}

#[derive(Default)]
pub(super) struct Dictionary {
    namespace: Option<NamespaceLease>,
    entries: HashTable<Head>,
    slots: Vec<Option<Entry>>,
    free: Vec<u32>,
    retired_head: Option<u32>,
    retired_tail: Option<u32>,
    retired_len: usize,
    hasher: KeyHashState,
    _credit: Option<MemoryReservation>,
    #[cfg(test)]
    force_collisions: bool,
    #[cfg(test)]
    limits: TestLimits,
}

struct Change {
    lease: Lease,
    hash: u64,
    base: u64,
    retained: u64,
    birth: bool,
    next: Option<u32>,
}

pub(super) struct Pending {
    namespace: Option<NamespaceLease>,
    changes: Vec<Change>,
    heads: HashTable<Head>,
    births: usize,
    replacement: Option<Dictionary>,
    keys: Vec<Option<Arc<FramedKey>>>,
    ids: Vec<u32>,
    _credit: Arc<MemoryReservation>,
    #[cfg(test)]
    limits: TestLimits,
}

pub(super) struct PreparedKeys {
    pub(super) dictionary: Pending,
    pub(super) keys: Vec<Option<Arc<FramedKey>>>,
    pub(super) ids: Vec<u32>,
    pub(super) scratch: Arc<MemoryReservation>,
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) struct TestLimits {
    pub(super) max_slot: u32,
    pub(super) max_references: u64,
    pub(super) max_key_bytes: usize,
    pub(super) reject_new_namespace: bool,
}

#[cfg(test)]
impl Default for TestLimits {
    fn default() -> Self {
        Self {
            max_slot: u32::MAX - 1,
            max_references: u64::MAX,
            max_key_bytes: usize::MAX,
            reject_new_namespace: false,
        }
    }
}

struct ReferenceGuard<'a> {
    dictionary: &'a mut Dictionary,
    head: Option<u32>,
}

impl ReferenceGuard<'_> {
    fn add(&mut self, key: &Arc<FramedKey>, name: &str) -> Result<()> {
        #[cfg(test)]
        let max_references = self.dictionary.limits.max_references;
        let Some((id, entry)) = self.dictionary.matched_entry(key) else {
            return Ok(());
        };
        let next = entry
            .provisional
            .checked_add(1)
            .ok_or_else(|| counter_overflow(name, "key_references"))?;
        let total = entry
            .retained
            .checked_add(next)
            .ok_or_else(|| counter_overflow(name, "key_references"))?;
        #[cfg(test)]
        if total > max_references {
            return Err(counter_overflow(name, "key_references"));
        }
        if entry.provisional == 0 {
            entry.touched_next = self.head;
            self.head = Some(id);
        }
        let _ = total;
        entry.provisional = next;
        Ok(())
    }
}

impl Drop for ReferenceGuard<'_> {
    fn drop(&mut self) {
        while let Some(id) = self.head {
            #[cfg(test)]
            super::note_join_work(|work| work.dictionary_reference_visits += 1);
            let entry = self.dictionary.slots[id as usize].as_mut().unwrap();
            self.head = entry.touched_next.take();
            entry.provisional = 0;
        }
    }
}

struct BirthCredits {
    canonical: Arc<MemoryReservation>,
    lease: MemoryReservation,
}

struct BirthBudget<'a> {
    master: &'a MemoryReservation,
    name: &'a str,
}

impl Dictionary {
    #[cfg(test)]
    pub(super) fn set_test_limits(&mut self, limits: TestLimits) {
        self.limits = limits;
    }

    #[cfg(test)]
    pub(super) fn force_collisions(&mut self) {
        self.force_collisions = true;
    }
    #[cfg(test)]
    pub(super) fn last_collision_key(&self) -> Arc<FramedKey> {
        let mut id = self.entries.find(0, |head| head.hash == 0).unwrap().id;
        while let Some(next) = self.slots[id as usize].as_ref().unwrap().next {
            id = next;
        }
        Arc::clone(self.slots[id as usize].as_ref().unwrap().lease.owner())
    }
    #[cfg(test)]
    pub(super) fn inventory(&self) -> (usize, u64, usize, bool) {
        (
            self.slots
                .iter()
                .flatten()
                .filter(|entry| entry.retained > 0)
                .count(),
            self.slots
                .iter()
                .flatten()
                .map(|entry| entry.retained)
                .sum(),
            self.slots.capacity(),
            self.namespace.is_some(),
        )
    }

    #[cfg(test)]
    pub(super) fn funded_controls(&self) -> usize {
        let Self {
            _credit: credit, ..
        } = self;
        credit.as_ref().map_or(0, MemoryReservation::size)
            + self
                .namespace
                .as_ref()
                .map_or(0, NamespaceLease::funded_bytes)
            + self
                .slots
                .iter()
                .flatten()
                .map(|entry| entry.lease.funded_bytes())
                .sum::<usize>()
    }

    #[cfg(test)]
    pub(super) fn canonical_owners(&self) -> impl Iterator<Item = &Arc<FramedKey>> {
        self.slots.iter().flatten().map(|entry| entry.lease.owner())
    }

    pub(super) async fn prepare(
        &mut self,
        rows: &dyn AdmittedRows,
        indices: &[usize],
        master: MemoryReservation,
        context: Option<&crate::StreamOperatorContext<'_>>,
        name: &str,
    ) -> Result<Option<PreparedKeys>> {
        let bytes = scratch_bytes(rows.len(), indices.len())
            .ok_or_else(|| counter_overflow(name, "key_capacity"))?;
        let Some(credit) = grant(&master, bytes, name)? else {
            return Ok(None);
        };
        let scratch = Arc::new(credit);
        let mut pending = Pending::new(rows.len(), self.namespace.clone(), Arc::clone(&scratch));
        #[cfg(test)]
        {
            pending.limits = self.limits;
        }
        let mut flow = KeyFlow::new(context);
        let budget = BirthBudget {
            master: &master,
            name,
        };
        if !pending
            .prepare_rows(self, rows, indices, &budget, &mut flow)
            .await?
        {
            return Ok(None);
        }
        Ok(Some(PreparedKeys {
            keys: std::mem::take(&mut pending.keys),
            ids: std::mem::take(&mut pending.ids),
            dictionary: pending,
            scratch,
        }))
    }

    async fn copy_for_capacity<C: Fn() -> Result<()>>(
        &self,
        capacity: usize,
        credit: MemoryReservation,
        flow: &mut KeyFlow<'_, '_, C>,
        growth: bool,
    ) -> Result<Self> {
        #[cfg(test)]
        if growth {
            super::note_join_work(|work| work.dictionary_capacity_replacements += 1);
        }
        #[cfg(not(test))]
        let _ = growth;
        let mut slots = Vec::with_capacity(capacity);
        for entry in &self.slots {
            #[cfg(test)]
            if growth {
                super::note_join_work(|work| work.dictionary_snapshot_visits += 1);
            }
            flow.step(1, 0).await?;
            slots.push(entry.as_ref().map(|entry| Entry {
                lease: entry.lease.clone(),
                hash: entry.hash,
                retained: entry.retained,
                next: if entry.retained == 0 {
                    entry.next
                } else {
                    None
                },
                previous: None,
                provisional: entry.provisional,
                touched_next: None,
            }));
        }
        let mut replacement = Self {
            namespace: self.namespace.clone(),
            entries: HashTable::with_capacity(capacity),
            slots,
            free: Vec::with_capacity(capacity),
            retired_head: self.retired_head,
            retired_tail: self.retired_tail,
            retired_len: self.retired_len,
            hasher: self.hasher.clone(),
            _credit: Some(credit),
            #[cfg(test)]
            force_collisions: self.force_collisions,
            #[cfg(test)]
            limits: self.limits,
        };
        replacement.copy_free_and_link(self, flow).await?;
        Ok(replacement)
    }

    async fn copy_free_and_link<C: Fn() -> Result<()>>(
        &mut self,
        source: &Self,
        flow: &mut KeyFlow<'_, '_, C>,
    ) -> Result<()> {
        for &id in &source.free {
            flow.step(1, 0).await?;
            self.free.push(id);
        }
        for id in 0..self.slots.len() {
            flow.step(1, 0).await?;
            if self.slots[id]
                .as_ref()
                .is_some_and(|entry| entry.retained > 0)
            {
                self.link(u32::try_from(id).expect("existing key ID"));
            }
        }
        Ok(())
    }

    async fn find(
        &self,
        hash: u64,
        key: TypedKey<'_, '_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<Option<u32>> {
        let mut candidate = self
            .entries
            .find(hash, |head| head.hash == hash)
            .map(|head| head.id);
        // why: a small key's single row charge covers this walk's first
        // comparison; the same row can reach both the pending and dictionary
        // walks, so at most two uncharged first comparisons share that charge
        let mut charged = key.small().is_some();
        while let Some(id) = candidate {
            if !charged {
                flow.step(1, 0).await?;
            }
            charged = false;
            #[cfg(test)]
            super::typed_key::note(super::typed_key::KeyPhase::Comparison, 0);
            let entry = self.slots[id as usize].as_ref().unwrap();
            if key.equals(entry.lease.owner(), flow).await? {
                return Ok(Some(id));
            }
            candidate = entry.next;
        }
        Ok(None)
    }

    fn next_slot(&self, births: usize, name: &str) -> Result<u32> {
        let ordinal = births
            .checked_add(1)
            .ok_or_else(|| counter_overflow(name, "key_id"))?;
        let slot = self
            .free
            .len()
            .checked_sub(ordinal)
            .map_or_else(
                || {
                    self.slots
                        .len()
                        .checked_add(births.saturating_sub(self.free.len()))
                },
                |index| Some(self.free[index] as usize),
            )
            .ok_or_else(|| counter_overflow(name, "key_id"))?;
        u32::try_from(slot)
            .ok()
            .filter(|id| *id != u32::MAX)
            .filter(|id| {
                #[cfg(test)]
                {
                    *id <= self.limits.max_slot
                }
                #[cfg(not(test))]
                {
                    let _ = id;
                    true
                }
            })
            .ok_or_else(|| counter_overflow(name, "key_id"))
    }

    pub(super) fn commit(&mut self, pending: Pending) {
        let Pending {
            namespace,
            changes,
            replacement,
            _credit,
            ..
        } = pending;
        if let Some(replacement) = replacement {
            *self = replacement;
        }
        for change in changes {
            let id = change.lease.address().slot();
            if change.birth {
                self.namespace.clone_from(&namespace);
                let entry = Some(Entry {
                    lease: change.lease,
                    hash: change.hash,
                    retained: change.retained,
                    next: None,
                    previous: None,
                    provisional: 0,
                    touched_next: None,
                });
                if id as usize == self.slots.len() {
                    self.slots.push(entry);
                } else {
                    assert_eq!(self.free.pop(), Some(id), "preflighted free slot");
                    self.slots[id as usize] = entry;
                }
                self.link(id);
            } else {
                let entry = self.slots[id as usize]
                    .as_mut()
                    .expect("leased committed identity");
                entry.retained = entry
                    .retained
                    .checked_add(change.retained)
                    .expect("preflighted retained references");
            }
        }
    }

    fn link(&mut self, id: u32) {
        let hash = self.slots[id as usize].as_ref().unwrap().hash;
        let previous_head = self
            .entries
            .find(hash, |head| head.hash == hash)
            .map(|head| head.id);
        self.slots[id as usize].as_mut().unwrap().next = previous_head;
        if let Some(head) = previous_head {
            self.slots[head as usize].as_mut().unwrap().previous = Some(id);
            self.entries
                .find_mut(hash, |head| head.hash == hash)
                .unwrap()
                .id = id;
        } else {
            self.entries
                .insert_unique(hash, Head { hash, id }, |head| head.hash);
        }
    }

    #[cfg(test)]
    pub(super) fn canonical(&self, bytes: &[u8]) -> Option<Arc<FramedKey>> {
        let hash = framed_hash(&self.hasher, bytes);
        #[cfg(test)]
        let hash = if self.force_collisions { 0 } else { hash };
        let mut id = self.entries.find(hash, |head| head.hash == hash)?.id;
        loop {
            let entry = self.slots[id as usize].as_ref().unwrap();
            if entry.lease.owner().as_slice() == bytes {
                return Some(Arc::clone(entry.lease.owner()));
            }
            id = entry.next?;
        }
    }

    pub(super) async fn validate_known_references(
        &mut self,
        rows: &[super::StoredRow],
        context: &crate::StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<()> {
        let mut flow = KeyFlow::new(Some(context));
        let mut guard = ReferenceGuard {
            dictionary: self,
            head: None,
        };
        for row in rows {
            flow.step(1, 0).await?;
            guard.add(&row.encoded_key, name)?;
        }
        Ok(())
    }

    pub(super) fn commit_known_references(&mut self, rows: &[super::StoredRow]) {
        for row in rows {
            if let Some(lease) = self.lease(&row.encoded_key) {
                let entry = self.slots[lease.address().slot() as usize]
                    .as_mut()
                    .unwrap();
                entry.retained = entry
                    .retained
                    .checked_add(1)
                    .expect("preflighted known retained references");
            }
        }
    }

    fn matched_entry(&mut self, key: &Arc<FramedKey>) -> Option<(u32, &mut Entry)> {
        let address = key.address()?;
        if self.namespace.as_ref()?.address(address.slot())? != address {
            return None;
        }
        #[cfg(test)]
        super::note_join_work(|work| work.dictionary_reference_visits += 1);
        let entry = self.slots.get_mut(address.slot() as usize)?.as_mut()?;
        (entry.retained > 0 && Arc::ptr_eq(entry.lease.owner(), key))
            .then_some((address.slot(), entry))
    }

    pub(super) fn lease(&self, key: &Arc<FramedKey>) -> Option<Lease> {
        let entry = self.entry(key.address()?)?;
        (entry.retained > 0 && Arc::ptr_eq(entry.lease.owner(), key)).then(|| entry.lease.clone())
    }

    fn entry(&self, address: super::super::key_identity::KeyAddress) -> Option<&Entry> {
        if self.namespace.as_ref()?.address(address.slot())? != address {
            return None;
        }
        self.slots.get(address.slot() as usize)?.as_ref()
    }

    pub(super) fn retire(&mut self, key: &Arc<FramedKey>, name: &str) -> Result<()> {
        let Some(lease) = self.lease(key) else {
            return Ok(());
        };
        let id = lease.address().slot();
        drop(lease);
        self.retire_slot(id, 1, name)
    }

    /// Aggregated retirement (spec FR13): decrements one key's retained
    /// count once for a whole evicted prefix with the same zero-crossing
    /// behavior and `counter_overflow` identity as per-row retirement.
    pub(super) fn retire_n(&mut self, key: &Arc<FramedKey>, count: u64, name: &str) -> Result<()> {
        let Some(lease) = self.lease(key) else {
            return Ok(());
        };
        let id = lease.address().slot();
        drop(lease);
        self.retire_slot(id, count, name)
    }

    fn retire_slot(&mut self, id: u32, count: u64, name: &str) -> Result<()> {
        let entry = self.slots[id as usize].as_mut().unwrap();
        entry.retained = entry
            .retained
            .checked_sub(count)
            .ok_or_else(|| counter_overflow(name, "key_references"))?;
        if entry.retained > 0 {
            return Ok(());
        }
        let hash = entry.hash;
        self.unlink(hash, id);
        self.release_slot(id);
        if self.entries.is_empty() {
            *self = Self::default();
        }
        Ok(())
    }

    fn release_slot(&mut self, id: u32) {
        if self.slots[id as usize]
            .as_ref()
            .unwrap()
            .lease
            .strong_count()
            == 1
        {
            self.slots[id as usize] = None;
            self.free.push(id);
        } else {
            self.queue_retired(id);
        }
    }

    fn queue_retired(&mut self, id: u32) {
        self.slots[id as usize].as_mut().unwrap().next = None;
        if let Some(tail) = self.retired_tail {
            self.slots[tail as usize].as_mut().unwrap().next = Some(id);
        } else {
            self.retired_head = Some(id);
        }
        self.retired_tail = Some(id);
        self.retired_len = self
            .retired_len
            .checked_add(1)
            .expect("checked directory slot domain");
    }

    fn pop_retired(&mut self) -> Option<u32> {
        let id = self.retired_head?;
        self.retired_head = self.slots[id as usize].as_mut().unwrap().next.take();
        if self.retired_head.is_none() {
            self.retired_tail = None;
        }
        self.retired_len -= 1;
        Some(id)
    }

    pub(super) async fn reclaim_retired(
        &mut self,
        context: Option<&crate::StreamOperatorContext<'_>>,
    ) -> Result<()> {
        let mut flow = KeyFlow::new(context);
        let count = self.retired_len.min(64);
        for _ in 0..count {
            flow.step(1, 0).await?;
            let id = self.pop_retired().expect("retired queue count");
            self.release_slot(id);
        }
        Ok(())
    }

    fn unlink(&mut self, hash: u64, removed: u32) {
        #[cfg(test)]
        super::note_join_work(|work| work.dictionary_unlink_visits += 1);
        let entry = self.slots[removed as usize].as_mut().unwrap();
        let previous = entry.previous.take();
        let next = entry.next.take();
        if let Some(previous) = previous {
            #[cfg(test)]
            super::note_join_work(|work| work.dictionary_unlink_visits += 1);
            self.slots[previous as usize].as_mut().unwrap().next = next;
        } else {
            self.unlink_head(hash, next);
        }
        if let Some(next) = next {
            #[cfg(test)]
            super::note_join_work(|work| work.dictionary_unlink_visits += 1);
            self.slots[next as usize].as_mut().unwrap().previous = previous;
        }
    }

    fn unlink_head(&mut self, hash: u64, next: Option<u32>) {
        if let Some(next) = next {
            self.entries
                .find_mut(hash, |head| head.hash == hash)
                .unwrap()
                .id = next;
        } else {
            self.entries
                .find_entry(hash, |head| head.hash == hash)
                .unwrap_or_else(|_| unreachable!())
                .remove();
        }
    }
}

impl Pending {
    fn new(
        capacity: usize,
        namespace: Option<NamespaceLease>,
        credit: Arc<MemoryReservation>,
    ) -> Self {
        Self {
            namespace,
            changes: Vec::with_capacity(capacity),
            heads: HashTable::with_capacity(capacity),
            births: 0,
            replacement: None,
            keys: Vec::with_capacity(capacity),
            ids: Vec::with_capacity(capacity),
            _credit: credit,
            #[cfg(test)]
            limits: TestLimits::default(),
        }
    }

    async fn prepare_rows(
        &mut self,
        dictionary: &mut Dictionary,
        rows: &dyn AdmittedRows,
        indices: &[usize],
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<bool> {
        let mut columns = Vec::with_capacity(indices.len());
        for group in rows.groups() {
            columns.clear();
            for &index in indices {
                flow.step(1, 0).await?;
                columns.push(KeyWriter::bind(group.row(0).record.column(index).as_ref())?);
            }
            for (offset, retain) in group.key_offsets() {
                let key = TypedKey {
                    columns: &columns,
                    row: offset,
                };
                let Some(id) = self.resolve(dictionary, key, retain, budget, flow).await? else {
                    return Ok(false);
                };
                self.ids.push(id);
            }
        }
        Ok(true)
    }

    async fn resolve(
        &mut self,
        dictionary: &Dictionary,
        key: TypedKey<'_, '_>,
        retain: bool,
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<Option<u32>> {
        let small = key.small();
        let hash = match small.as_ref() {
            Some(small) => {
                let (visits, bytes) = small.charge();
                flow.step(visits, bytes).await?;
                #[cfg(test)]
                super::typed_key::note(super::typed_key::KeyPhase::Hash, bytes);
                small.hash(&dictionary.hasher)?
            }
            None => key.hash(&dictionary.hasher, flow).await?,
        };
        #[cfg(test)]
        let hash = if dictionary.force_collisions { 0 } else { hash };
        let Some(id) = self
            .resolve_hashed(dictionary, key, hash, retain, budget, flow)
            .await?
        else {
            return Ok(None);
        };
        if retain {
            self.retain(id, budget.name)?;
        }
        Ok(Some(id))
    }

    async fn resolve_hashed(
        &mut self,
        dictionary: &Dictionary,
        key: TypedKey<'_, '_>,
        hash: u64,
        retain: bool,
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<Option<u32>> {
        if let Some(id) = self.find(hash, key, flow).await? {
            return Ok(Some(id));
        }
        if let Some(id) = dictionary.find(hash, key, flow).await? {
            let entry = dictionary.slots[id as usize].as_ref().unwrap();
            return self
                .add(
                    entry.lease.clone(),
                    hash,
                    entry.retained,
                    false,
                    budget.name,
                )
                .map(Some);
        }
        if !retain {
            return Ok(Some(u32::MAX));
        }
        self.birth(dictionary, key, hash, budget, flow).await
    }

    async fn find(
        &self,
        hash: u64,
        key: TypedKey<'_, '_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<Option<u32>> {
        let mut candidate = self
            .heads
            .find(hash, |head| head.hash == hash)
            .map(|head| head.id);
        // why: a small key's single row charge covers this walk's first
        // comparison; the same row can reach both the pending and dictionary
        // walks, so at most two uncharged first comparisons share that charge
        let mut charged = key.small().is_some();
        while let Some(id) = candidate {
            if !charged {
                flow.step(1, 0).await?;
            }
            charged = false;
            #[cfg(test)]
            super::typed_key::note(super::typed_key::KeyPhase::Comparison, 0);
            let change = &self.changes[id as usize];
            if key.equals(change.lease.owner(), flow).await? {
                return Ok(Some(id));
            }
            candidate = change.next;
        }
        Ok(None)
    }

    fn add(&mut self, lease: Lease, hash: u64, base: u64, birth: bool, name: &str) -> Result<u32> {
        let id = u32::try_from(self.changes.len())
            .ok()
            .filter(|id| *id != u32::MAX)
            .ok_or_else(|| counter_overflow(name, "key_id"))?;
        let next = self
            .heads
            .find(hash, |head| head.hash == hash)
            .map(|head| head.id);
        self.keys.push(Some(Arc::clone(lease.owner())));
        self.changes.push(Change {
            lease,
            hash,
            base,
            retained: 0,
            birth,
            next,
        });
        if let Some(head) = self.heads.find_mut(hash, |head| head.hash == hash) {
            head.id = id;
        } else {
            self.heads
                .insert_unique(hash, Head { hash, id }, |head| head.hash);
        }
        Ok(id)
    }

    async fn birth(
        &mut self,
        dictionary: &Dictionary,
        key: TypedKey<'_, '_>,
        hash: u64,
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<Option<u32>> {
        let Some(slot) = self.birth_slot(dictionary, budget, flow).await? else {
            return Ok(None);
        };
        let Some(lease) = self.allocate_lease(key, slot, budget, flow).await? else {
            return Ok(None);
        };
        self.add_birth(lease, hash, budget.name).map(Some)
    }

    async fn birth_slot(
        &mut self,
        dictionary: &Dictionary,
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<Option<u32>> {
        let slot = dictionary.next_slot(self.births, budget.name)?;
        if !self
            .ensure_capacity(dictionary, slot as usize + 1, budget, flow)
            .await?
        {
            return Ok(None);
        }
        Ok(self.ensure_namespace(budget)?.then_some(slot))
    }

    async fn ensure_capacity(
        &mut self,
        dictionary: &Dictionary,
        count: usize,
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<bool> {
        let backing = self.replacement.as_ref().unwrap_or(dictionary);
        if count <= backing.slots.capacity() {
            return Ok(true);
        }
        let capacity = count
            .checked_next_power_of_two()
            .map(|value| value.max(4))
            .ok_or_else(|| counter_overflow(budget.name, "key_capacity"))?;
        let bytes =
            table_bytes(capacity).ok_or_else(|| counter_overflow(budget.name, "key_capacity"))?;
        let Some(credit) = grant(budget.master, bytes, budget.name)? else {
            return Ok(false);
        };
        let replacement = backing
            .copy_for_capacity(capacity, credit, flow, true)
            .await?;
        self.replacement = Some(replacement);
        Ok(true)
    }

    async fn allocate_lease(
        &self,
        key: TypedKey<'_, '_>,
        slot: u32,
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<Option<Lease>> {
        let frame_bytes = key.length(flow).await?;
        #[cfg(test)]
        if frame_bytes > self.limits.max_key_bytes {
            return Err(counter_overflow(budget.name, "key_length"));
        }
        let Some(credits) = canonical_credit(frame_bytes, budget)? else {
            return Ok(None);
        };
        let bytes = key.encode(frame_bytes, flow).await?;
        self.identified_lease(slot, bytes, credits, budget.name)
            .map(Some)
    }

    fn identified_lease(
        &self,
        slot: u32,
        bytes: Vec<u8>,
        credits: BirthCredits,
        name: &str,
    ) -> Result<Lease> {
        let namespace = self.namespace.as_ref().expect("prepared namespace");
        let address = namespace
            .address(slot)
            .ok_or_else(|| counter_overflow(name, "key_id"))?;
        let owner = super::columnar::identified_key(bytes, credits.canonical, address);
        Lease::new(namespace, slot, owner, credits.lease)
            .ok_or_else(|| counter_overflow(name, "key_id"))
    }

    fn add_birth(&mut self, lease: Lease, hash: u64, name: &str) -> Result<u32> {
        let births = self
            .births
            .checked_add(1)
            .ok_or_else(|| counter_overflow(name, "key_id"))?;
        let id = self.add(lease, hash, 0, true, name)?;
        self.births = births;
        Ok(id)
    }

    fn ensure_namespace(&mut self, budget: &BirthBudget<'_>) -> Result<bool> {
        if self.namespace.is_some() {
            return Ok(true);
        }
        #[cfg(test)]
        if self.limits.reject_new_namespace {
            return Err(counter_overflow(budget.name, "key_namespace"));
        }
        let bytes = NamespaceLease::control_bytes()
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| counter_overflow(budget.name, "key_namespace"))?;
        let Some(credit) = grant(budget.master, bytes, budget.name)? else {
            return Ok(false);
        };
        self.namespace = Some(
            NamespaceLease::new(credit)
                .ok_or_else(|| counter_overflow(budget.name, "key_namespace"))?,
        );
        Ok(true)
    }

    fn retain(&mut self, id: u32, name: &str) -> Result<()> {
        let change = &mut self.changes[id as usize];
        let next = change
            .retained
            .checked_add(1)
            .ok_or_else(|| counter_overflow(name, "key_references"))?;
        let total = change
            .base
            .checked_add(next)
            .ok_or_else(|| counter_overflow(name, "key_references"))?;
        #[cfg(test)]
        if total > self.limits.max_references {
            return Err(counter_overflow(name, "key_references"));
        }
        let _ = total;
        change.retained = next;
        Ok(())
    }
}

fn canonical_credit(frame_bytes: usize, budget: &BirthBudget<'_>) -> Result<Option<BirthCredits>> {
    let lease_bytes = Lease::control_bytes()
        .and_then(|bytes| bytes.checked_add(1024))
        .ok_or_else(|| counter_overflow(budget.name, "key_bytes"))?;
    let bytes = frame_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(1024))
        .and_then(|bytes| bytes.checked_add(lease_bytes))
        .ok_or_else(|| counter_overflow(budget.name, "key_bytes"))?;
    let Some(credit) = grant(budget.master, bytes, budget.name)? else {
        return Ok(None);
    };
    let lease = credit.split(lease_bytes);
    Ok(Some(BirthCredits {
        canonical: Arc::new(credit),
        lease,
    }))
}

pub(super) fn grant(
    master: &MemoryReservation,
    bytes: usize,
    name: &str,
) -> Result<Option<MemoryReservation>> {
    let credit = master.new_empty();
    match credit.try_grow(bytes) {
        Ok(()) => Ok(Some(credit)),
        Err(datafusion::common::DataFusionError::ResourcesExhausted(_)) => Ok(None),
        Err(error) => Err(crate::CalcFlowError::DataFusion {
            node_id: Some(name.to_owned()),
            message: error.to_string(),
        }),
    }
}

fn table_bytes(count: usize) -> Option<usize> {
    count
        .checked_mul(size_of::<Entry>() + size_of::<Head>() + size_of::<u32>() + 64)?
        .checked_mul(4)?
        .checked_add(1024)
}

fn scratch_bytes(rows: usize, columns: usize) -> Option<usize> {
    rows.checked_mul(
        size_of::<Change>()
            + size_of::<Head>() * 4
            + size_of::<Option<Arc<FramedKey>>>()
            + size_of::<u32>()
            + size_of::<Option<u32>>()
            + 64,
    )?
    .checked_add(columns.checked_mul(size_of::<KeyWriter<'_>>())?)?
    .checked_add(1024)
}
