#[cfg(test)]
use super::borrowed_key::framed_hash;
use std::sync::Arc;

use datafusion::execution::memory_pool::MemoryReservation;
use hashbrown::HashTable;

use super::super::key_identity::{IdentityLease, NamespaceLease};
use super::{
    AdmittedRows,
    admitted::AdmittedGroup,
    borrowed_key::KeyHashState,
    columnar::FramedKey,
    counter_overflow,
    key_arena::KeyWriter,
    typed_key::{KeyFlow, SmallKey, TypedKey},
};
use crate::Result;

mod bindings;
mod framed;
mod local;
mod parallel;
mod rebuild;
#[cfg(test)]
mod tests;
pub(super) use bindings::{Bindings, SideBindings};
pub(super) use rebuild::Rebuilt;

pub(super) type Lease = IdentityLease<Arc<FramedKey>>;

#[cfg(test)]
thread_local! {
    static PER_ROW_KEYS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test control that pins key-identity preparation to the per-row path even
/// when the batch-local interning table is fundable; drops to release.
#[cfg(test)]
pub(in crate::operator::join) struct PerRowKeys;

#[cfg(test)]
impl Drop for PerRowKeys {
    fn drop(&mut self) {
        PER_ROW_KEYS.with(|forced| forced.set(false));
    }
}

#[cfg(test)]
pub(in crate::operator::join) fn force_per_row_keys() -> PerRowKeys {
    PER_ROW_KEYS.with(|forced| forced.set(true));
    PerRowKeys
}

#[cfg(test)]
fn intern_locally() -> bool {
    !PER_ROW_KEYS.with(std::cell::Cell::get)
}

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
    frequencies: Vec<usize>,
    _credit: Arc<MemoryReservation>,
    #[cfg(test)]
    limits: TestLimits,
}

pub(super) struct PreparedKeys {
    pub(super) dictionary: Pending,
    pub(super) keys: Vec<Option<Arc<FramedKey>>>,
    pub(super) ids: Vec<u32>,
    pub(super) frequencies: Vec<usize>,
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
        let base = base_scratch_bytes(rows.len(), indices.len())
            .ok_or_else(|| counter_overflow(name, "key_capacity"))?;
        let local_bytes =
            local_table_bytes(rows.len()).ok_or_else(|| counter_overflow(name, "key_capacity"))?;
        let interned = intern_locally();
        let wanted = if interned {
            base.checked_add(local_bytes)
                .ok_or_else(|| counter_overflow(name, "key_capacity"))?
        } else {
            base
        };
        let (credit, local_credit) = match grant(&master, wanted, name)? {
            Some(credit) => {
                let local_credit = interned.then(|| credit.split(local_bytes));
                (credit, local_credit)
            }
            // A local-table refusal only sheds the deduplication pass; the
            // per-row identity path still prepares when the base scratch fits.
            None if interned => {
                let Some(credit) = grant(&master, base, name)? else {
                    return Ok(None);
                };
                (credit, None)
            }
            None => return Ok(None),
        };
        let scratch = Arc::new(credit);
        let local = local_credit.map(|credit| local::LocalIntern::new(rows.len(), credit));
        let mut pending = Pending::new(rows.len(), self.namespace.clone(), Arc::clone(&scratch));
        #[cfg(test)]
        {
            pending.limits = self.limits;
        }
        let budget = BirthBudget {
            master: &master,
            name,
        };
        if !pending
            .prepare_rows(self, rows, indices, &budget, context, local)
            .await?
        {
            return Ok(None);
        }
        Ok(Some(PreparedKeys {
            keys: std::mem::take(&mut pending.keys),
            ids: std::mem::take(&mut pending.ids),
            frequencies: std::mem::take(&mut pending.frequencies),
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
        small: Option<&SmallKey<'_, '_>>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<Option<u32>> {
        #[cfg(test)]
        super::note_join_work(|work| work.dictionary_probes += 1);
        let mut candidate = self
            .entries
            .find(hash, |head| head.hash == hash)
            .map(|head| head.id);
        // why: a small key's single row charge covers this walk's first
        // comparison; the same row can reach both the pending and dictionary
        // walks, so at most two uncharged first comparisons share that charge
        let mut charged = small.is_some();
        while let Some(id) = candidate {
            if !charged {
                flow.step(1, 0).await?;
            }
            charged = false;
            #[cfg(test)]
            super::typed_key::note(super::typed_key::KeyPhase::Comparison, 0);
            let entry = self.slots[id as usize].as_ref().unwrap();
            let matched = match small {
                Some(small) => small.equals(entry.lease.owner())?,
                None => key.equals(entry.lease.owner(), flow).await?,
            };
            if matched {
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
        rows: &dyn super::retained::Rows,
        context: &crate::StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<()> {
        let mut flow = KeyFlow::new(Some(context));
        let mut guard = ReferenceGuard {
            dictionary: self,
            head: None,
        };
        for index in rows.references() {
            flow.step(1, 0).await?;
            guard.add(rows.key(index), name)?;
        }
        Ok(())
    }

    pub(super) fn commit_known_references(&mut self, rows: &dyn super::retained::Rows) {
        for index in rows.references() {
            if let Some(lease) = self.lease(rows.key(index)) {
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
            frequencies: Vec::with_capacity(capacity),
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
        context: Option<&crate::StreamOperatorContext<'_>>,
        mut local: Option<local::LocalIntern>,
    ) -> Result<bool> {
        let mut flow = KeyFlow::new(context);
        let mut columns = Vec::with_capacity(indices.len());
        for group in rows.groups() {
            columns.clear();
            for &index in indices {
                flow.step(1, 0).await?;
                columns.push(KeyWriter::bind(group.row(0).record.column(index).as_ref())?);
            }
            let bound = if u32::try_from(group.len()).is_ok() {
                local::bind_columns(group.row(0).record, indices)
            } else {
                None
            };
            let interned = match (local.as_mut(), bound) {
                (Some(intern), Some(bound)) => Some((intern, bound)),
                _ => None,
            };
            let prepared = match interned {
                Some((intern, bound)) => {
                    intern.begin_group();
                    let parallel = parallel::intern_group(
                        intern, &bound, group, indices, budget, &mut flow, context,
                    )
                    .await?;
                    if parallel.is_none() {
                        intern
                            .intern_group(&bound, group, budget.name, &mut flow)
                            .await?;
                    }
                    self.prepare_group_local(
                        &columns,
                        dictionary,
                        intern,
                        budget,
                        &mut flow,
                        parallel.zip(context),
                    )
                    .await?
                }
                None => {
                    self.prepare_group_rows(&columns, dictionary, group, budget, &mut flow)
                        .await?
                }
            };
            if !prepared {
                return Ok(false);
            }
        }
        // The batch-local table drops here: nothing after preparation reads
        // it, and dropping releases its split credit before matching, output
        // preflight and checkpoint-writer grants contend for the pool.
        drop(local);
        Ok(true)
    }

    /// Per-row identity path: the pre-deduplication behavior, kept for
    /// scratch refusal, exotic key layouts, and test equivalence.
    async fn prepare_group_rows(
        &mut self,
        columns: &[KeyWriter<'_>],
        dictionary: &Dictionary,
        group: AdmittedGroup<'_>,
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<bool> {
        for (offset, retain) in group.key_offsets() {
            let key = TypedKey {
                columns,
                row: offset,
            };
            let Some(id) = self.resolve(dictionary, key, retain, budget, flow).await? else {
                return Ok(false);
            };
            self.ids.push(id);
            self.note_frequency(id, 1);
        }
        Ok(true)
    }

    /// Vectorized path: after `intern_group` deduplicated the rows, resolve
    /// each distinct key once and broadcast the identity to every row.
    async fn prepare_group_local(
        &mut self,
        columns: &[KeyWriter<'_>],
        dictionary: &Dictionary,
        local: &mut local::LocalIntern,
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
        parallel: Option<(parallel::Group, &crate::StreamOperatorContext<'_>)>,
    ) -> Result<bool> {
        self.classify_distinct(columns, dictionary, local, budget.name, flow)
            .await?;
        // Changes stage in the exact row order the per-row path would have
        // staged them: dictionary hits at first appearance, births at the
        // first retained row of an unknown key.
        local.sort_changes();
        if !self
            .stage_changes(columns, dictionary, local, budget, flow)
            .await?
        {
            return Ok(false);
        }
        if let Some((group, context)) = parallel {
            group.finish(self, local, flow, context).await?;
        } else {
            #[cfg(test)]
            parallel::note_owner_remap(local.locals.len());
            let Self {
                ids, frequencies, ..
            } = self;
            ids.extend(local.broadcast().inspect(|&id| {
                if let Some(count) = frequencies.get_mut(id as usize) {
                    *count = count.saturating_add(1);
                }
            }));
        }
        Ok(true)
    }

    fn note_frequency(&mut self, id: u32, count: usize) {
        if let Some(frequency) = self.frequencies.get_mut(id as usize) {
            *frequency = frequency.saturating_add(count);
        }
    }

    /// Classification runs per distinct key in first-appearance order; the
    /// plain finds are pure, so their outcome matches the per-row path's
    /// first resolution of the key.
    async fn classify_distinct(
        &mut self,
        columns: &[KeyWriter<'_>],
        dictionary: &Dictionary,
        local: &mut local::LocalIntern,
        name: &str,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<()> {
        for distinct in 0..local.distinct {
            self.classify_one(columns, dictionary, local, distinct, name, flow)
                .await?;
        }
        Ok(())
    }

    /// Classifies one distinct key: a pending hit keeps its slot, a
    /// dictionary hit schedules a staged add, and an unknown key schedules a
    /// birth at its first retained row.
    async fn classify_one(
        &mut self,
        columns: &[KeyWriter<'_>],
        dictionary: &Dictionary,
        local: &mut local::LocalIntern,
        distinct: usize,
        name: &str,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<()> {
        let key = TypedKey {
            columns,
            row: local.rows[distinct],
        };
        let small = key.small();
        let hash = canonical_hash(dictionary, key, small.as_ref(), flow).await?;
        local.canonicals.push(hash);
        let retains = local.retains[distinct];
        if let Some(id) = self.find(hash, key, small.as_ref(), flow).await? {
            local.plans.push(local::Plan::Pending);
            if retains > 0 {
                self.retain(id, retains, name)?;
            }
            local.posts.push(id);
            local.splits.push(u32::MAX);
            return Ok(());
        }
        let found = dictionary.find(hash, key, small.as_ref(), flow).await?;
        local.stage_classification(distinct, found, retains);
        Ok(())
    }

    /// Stages one classified change per ordered entry, matching the per-row
    /// staging sequence exactly.
    async fn stage_changes(
        &mut self,
        columns: &[KeyWriter<'_>],
        dictionary: &Dictionary,
        local: &mut local::LocalIntern,
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<bool> {
        for index in 0..local.order.len() {
            let (_, distinct) = local.order[index];
            let retains = local.retains[distinct as usize];
            match local.plans[distinct as usize] {
                local::Plan::Pending => {
                    debug_assert!(false, "pending identities never stage changes");
                }
                local::Plan::Dictionary(entry) => {
                    self.stage_dictionary_hit(
                        dictionary,
                        local,
                        distinct as usize,
                        entry,
                        retains,
                        budget.name,
                    )?;
                }
                local::Plan::Unknown => {
                    if !self
                        .stage_birth(columns, dictionary, local, distinct as usize, budget, flow)
                        .await?
                    {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }

    fn stage_dictionary_hit(
        &mut self,
        dictionary: &Dictionary,
        local: &mut local::LocalIntern,
        distinct: usize,
        entry: u32,
        retains: u32,
        name: &str,
    ) -> Result<()> {
        let slot = dictionary.slots[entry as usize]
            .as_ref()
            .expect("classified dictionary identity");
        let id = self.add(
            slot.lease.clone(),
            local.canonicals[distinct],
            slot.retained,
            false,
            name,
        )?;
        local.posts[distinct] = id;
        if retains > 0 {
            self.retain(id, retains, name)?;
        }
        Ok(())
    }

    async fn stage_birth(
        &mut self,
        columns: &[KeyWriter<'_>],
        dictionary: &Dictionary,
        local: &mut local::LocalIntern,
        distinct: usize,
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<bool> {
        let key = TypedKey {
            columns,
            row: local.rows[distinct],
        };
        let Some(id) = self
            .birth(dictionary, key, local.canonicals[distinct], budget, flow)
            .await?
        else {
            return Ok(false);
        };
        local.posts[distinct] = id;
        local.splits[distinct] = local.first_retained[distinct];
        let retains = local.retains[distinct];
        if retains > 0 {
            self.retain(id, retains, budget.name)?;
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
        let hash = canonical_hash(dictionary, key, small.as_ref(), flow).await?;
        let Some(id) = self
            .resolve_hashed(
                dictionary,
                (key, small.as_ref()),
                hash,
                retain,
                budget,
                flow,
            )
            .await?
        else {
            return Ok(None);
        };
        if retain {
            self.retain(id, 1, budget.name)?;
        }
        Ok(Some(id))
    }

    async fn resolve_hashed(
        &mut self,
        dictionary: &Dictionary,
        identity: (TypedKey<'_, '_>, Option<&SmallKey<'_, '_>>),
        hash: u64,
        retain: bool,
        budget: &BirthBudget<'_>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<Option<u32>> {
        let (key, small) = identity;
        if let Some(id) = self.find(hash, key, small, flow).await? {
            return Ok(Some(id));
        }
        if let Some(id) = dictionary.find(hash, key, small, flow).await? {
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
        small: Option<&SmallKey<'_, '_>>,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<Option<u32>> {
        let mut candidate = self
            .heads
            .find(hash, |head| head.hash == hash)
            .map(|head| head.id);
        // why: a small key's single row charge covers this walk's first
        // comparison; the same row can reach both the pending and dictionary
        // walks, so at most two uncharged first comparisons share that charge
        let mut charged = small.is_some();
        while let Some(id) = candidate {
            if !charged {
                flow.step(1, 0).await?;
            }
            charged = false;
            #[cfg(test)]
            super::typed_key::note(super::typed_key::KeyPhase::Comparison, 0);
            let change = &self.changes[id as usize];
            let matched = match small {
                Some(small) => small.equals(change.lease.owner())?,
                None => key.equals(change.lease.owner(), flow).await?,
            };
            if matched {
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
        self.frequencies.push(0);
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

    /// Applies the aggregated references of one distinct key; per-row callers
    /// pass one at a time.
    fn retain(&mut self, id: u32, count: u32, name: &str) -> Result<()> {
        let change = &mut self.changes[id as usize];
        let next = change
            .retained
            .checked_add(u64::from(count))
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

/// Canonical framed hash for one key, charged exactly like the per-row path.
/// The caller computes the row's `SmallKey` once and shares it here so the
/// byte-summing walk is not repeated per hash, find and comparison.
async fn canonical_hash(
    dictionary: &Dictionary,
    key: TypedKey<'_, '_>,
    small: Option<&SmallKey<'_, '_>>,
    flow: &mut KeyFlow<'_, '_>,
) -> Result<u64> {
    let hash = match small {
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
    Ok(hash)
}

#[cfg(not(test))]
fn intern_locally() -> bool {
    true
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

fn base_scratch_bytes(rows: usize, columns: usize) -> Option<usize> {
    rows.checked_mul(
        size_of::<Change>()
            + size_of::<Head>() * 4
            + size_of::<Option<Arc<FramedKey>>>()
            + size_of::<u32>()
            + size_of::<usize>()
            + size_of::<Option<u32>>()
            + 64,
    )?
    .checked_add(columns.checked_mul(size_of::<KeyWriter<'_>>())?)?
    .checked_add(1024)
}

/// Batch-local interning budget: per row, a plain and a canonical hash, a
/// representative offset, six u32 lanes, the change order and the local
/// plan, plus the interning table's slots. Four-byte slots round up to a power
/// of two at least 8/7 of the rows. Keep the existing conservative 12-byte
/// per-row table allowance and fixed header, including on optional refusal.
pub(super) fn local_table_bytes(rows: usize) -> Option<usize> {
    local::slot_count(rows)?;
    rows.checked_mul(
        2 * size_of::<u64>()
            + size_of::<usize>()
            + 6 * size_of::<u32>()
            + size_of::<(u32, u32)>()
            + size_of::<local::Plan>()
            + 12,
    )?
    .checked_add(64)
}
