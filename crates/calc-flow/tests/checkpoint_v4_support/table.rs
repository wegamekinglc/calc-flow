use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    sync::Arc,
    thread,
};

use sha2::{Digest, Sha256};

pub type Slot = (u32, u32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PageId {
    pub key_group: u32,
    pub slot: u32,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Page {
    pub id: PageId,
    bytes: Arc<Vec<u8>>,
    len: usize,
}

impl Page {
    fn new(id: PageId, bytes: Vec<u8>) -> Self {
        let len = bytes.len();
        Self {
            id,
            bytes: Arc::new(bytes),
            len,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    pub const fn storage(&self) -> &Arc<Vec<u8>> {
        &self.bytes
    }

    const fn descriptor(&self) -> PageDescriptor {
        PageDescriptor {
            id: self.id,
            len: self.len,
        }
    }

    fn appended(&self, suffix: &[u8], generation: u64) -> Self {
        let mut bytes = self.bytes().to_vec();
        bytes.extend_from_slice(suffix);
        Self::new(
            PageId {
                generation,
                ..self.id
            },
            bytes,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct PageDescriptor {
    id: PageId,
    len: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PageChange {
    Write(Page),
    Delete(PageId),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct DirtySet {
    bits: BTreeSet<u32>,
    first_dirty: Vec<u32>,
}

impl DirtySet {
    fn mark(&mut self, slot: u32) {
        if self.bits.insert(slot) {
            self.first_dirty.push(slot);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartitionImage {
    pages: BTreeMap<Slot, Page>,
    page_digests: BTreeMap<Slot, String>,
    next_generations: BTreeMap<Slot, u64>,
    partition_digests: BTreeMap<u32, String>,
    lineage_digest: String,
}

impl PartitionImage {
    fn from_parts(
        pages: BTreeMap<Slot, Page>,
        page_digests: BTreeMap<Slot, String>,
        next_generations: BTreeMap<Slot, u64>,
    ) -> Self {
        let groups = pages
            .keys()
            .chain(next_generations.keys())
            .map(|(group, _)| *group)
            .collect::<BTreeSet<_>>();
        let mut image = Self {
            pages,
            page_digests,
            next_generations,
            partition_digests: BTreeMap::new(),
            lineage_digest: String::new(),
        };
        for group in groups {
            let digest = image.partition_digest(group);
            image.partition_digests.insert(group, digest);
        }
        image.refresh_lineage_digest();
        image
    }

    fn partition_bytes(&self, group: u32) -> Vec<u8> {
        let mut bytes = format!("partition:{group}\n").into_bytes();
        for ((page_group, slot), page) in &self.pages {
            if *page_group != group {
                continue;
            }
            let digest = &self.page_digests[&(*page_group, *slot)];
            bytes.extend_from_slice(
                format!(
                    "page:{slot}:{}:{}:{digest}\n",
                    page.id.generation,
                    page.bytes().len()
                )
                .as_bytes(),
            );
        }
        for ((page_group, slot), generation) in &self.next_generations {
            if *page_group == group {
                bytes.extend_from_slice(format!("next:{slot}:{generation}\n").as_bytes());
            }
        }
        bytes
    }

    fn partition_digest(&self, group: u32) -> String {
        hex::encode(Sha256::digest(self.partition_bytes(group)))
    }

    fn refresh_lineage_digest(&mut self) {
        self.lineage_digest =
            self.partition_digests
                .iter()
                .fold(String::new(), |mut lineage, (group, digest)| {
                    write!(lineage, "{group}:{digest};").expect("formatting a String cannot fail");
                    lineage
                });
    }

    pub fn state_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for ((group, slot), page) in &self.pages {
            bytes.extend_from_slice(
                format!("page:{group}:{slot}:{}:{}:", page.id.generation, page.len).as_bytes(),
            );
            bytes.extend_from_slice(page.bytes());
            bytes.push(b'\n');
        }
        for ((group, slot), generation) in &self.next_generations {
            bytes.extend_from_slice(format!("next:{group}:{slot}:{generation}\n").as_bytes());
        }
        bytes
    }

    pub fn page(&self, slot: Slot) -> Option<&Page> {
        self.pages.get(&slot)
    }
}

#[derive(Clone, Debug)]
pub struct TableFreeze {
    pub epoch: u64,
    previous: Arc<PartitionImage>,
    changes: Vec<PageChange>,
    next_generations: BTreeMap<Slot, u64>,
    digest: String,
    reservation_bytes: u64,
}

impl TableFreeze {
    pub fn bytes_for(&self, id: PageId) -> Option<&[u8]> {
        self.changes.iter().find_map(|change| match change {
            PageChange::Write(page) if page.id == id => Some(page.bytes()),
            _ => None,
        })
    }

    pub const fn reservation_bytes(&self) -> u64 {
        self.reservation_bytes
    }

    fn metadata_digest(epoch: u64, previous: &PartitionImage, changes: &[PageChange]) -> String {
        let mut bytes = format!("freeze:{epoch}:{}\n", previous.lineage_digest).into_bytes();
        for change in changes {
            match change {
                PageChange::Write(page) => bytes.extend_from_slice(
                    format!(
                        "write:{}:{}:{}:{}\n",
                        page.id.key_group,
                        page.id.slot,
                        page.id.generation,
                        page.bytes().len()
                    )
                    .as_bytes(),
                ),
                PageChange::Delete(id) => bytes.extend_from_slice(
                    format!("delete:{}:{}:{}\n", id.key_group, id.slot, id.generation).as_bytes(),
                ),
            }
        }
        hex::encode(Sha256::digest(bytes))
    }
}

#[derive(Clone, Copy, Debug)]
struct UndoDirty {
    was_dirty: bool,
    first_dirty_len: usize,
}

#[derive(Debug)]
struct UndoRecord {
    slot: Slot,
    previous: Option<PageDescriptor>,
    previous_owned: Option<Page>,
    next_generation: u64,
    dirty: UndoDirty,
}

#[derive(Debug)]
pub struct FakeTable {
    checkpointing: bool,
    current: BTreeMap<Slot, Page>,
    durable: Arc<PartitionImage>,
    next_generations: BTreeMap<Slot, u64>,
    dirty: Option<BTreeMap<u32, DirtySet>>,
    frozen_pages: BTreeMap<PageId, Page>,
    active_epoch: Option<u64>,
    undo_open: bool,
    undo: Vec<UndoRecord>,
    undo_initial_bytes: u64,
    logical_bytes: u64,
}

impl FakeTable {
    pub fn seed(checkpointing: bool, entries: &[(u32, u32, &[u8])]) -> Self {
        let mut pages = BTreeMap::new();
        let mut digests = BTreeMap::new();
        let mut next_generations = BTreeMap::new();
        for (group, slot, bytes) in entries {
            let id = PageId {
                key_group: *group,
                slot: *slot,
                generation: 0,
            };
            let page = Page::new(id, bytes.to_vec());
            digests.insert((*group, *slot), hex::encode(Sha256::digest(page.bytes())));
            pages.insert((*group, *slot), page);
            next_generations.insert((*group, *slot), 1);
        }
        let logical_bytes = pages.values().map(|page| page.len as u64).sum::<u64>();
        let durable = PartitionImage::from_parts(pages.clone(), digests, next_generations.clone());
        Self {
            checkpointing,
            current: pages,
            durable: Arc::new(durable),
            next_generations,
            dirty: checkpointing.then(BTreeMap::new),
            frozen_pages: BTreeMap::new(),
            active_epoch: None,
            undo_open: false,
            undo: Vec::new(),
            undo_initial_bytes: 0,
            logical_bytes,
        }
    }

    pub fn restore(image: &PartitionImage, checkpointing: bool) -> Self {
        let logical_bytes = image
            .pages
            .values()
            .map(|page| page.len as u64)
            .sum::<u64>();
        Self {
            checkpointing,
            current: image.pages.clone(),
            durable: Arc::new(image.clone()),
            next_generations: image.next_generations.clone(),
            dirty: checkpointing.then(BTreeMap::new),
            frozen_pages: BTreeMap::new(),
            active_epoch: None,
            undo_open: false,
            undo: Vec::new(),
            undo_initial_bytes: 0,
            logical_bytes,
        }
    }

    pub fn begin(&mut self) {
        assert!(!self.undo_open, "an undo transaction is already open");
        self.undo_open = true;
        self.undo_initial_bytes = self.logical_bytes;
        self.undo = Vec::new();
    }

    pub fn commit(&mut self) {
        assert!(self.undo_open, "no undo transaction is open");
        self.undo_open = false;
        self.undo.clear();
        self.undo_initial_bytes = 0;
    }

    fn record_undo(&mut self, slot: Slot) {
        if self.undo.iter().any(|record| record.slot == slot) {
            return;
        }
        let record = UndoRecord {
            slot,
            previous: self.current.get(&slot).map(Page::descriptor),
            previous_owned: None,
            next_generation: self.next_generations.get(&slot).copied().unwrap_or(0),
            dirty: self
                .dirty
                .as_ref()
                .and_then(|dirty| dirty.get(&slot.0))
                .map_or(
                    UndoDirty {
                        was_dirty: false,
                        first_dirty_len: 0,
                    },
                    |dirty| UndoDirty {
                        was_dirty: dirty.bits.contains(&slot.1),
                        first_dirty_len: dirty.first_dirty.len(),
                    },
                ),
        };
        self.undo.push(record);
    }

    fn retained(&self, page: &Page) -> bool {
        self.frozen_pages.contains_key(&page.id)
            || self
                .durable
                .pages
                .get(&(page.id.key_group, page.id.slot))
                .is_some_and(|durable| durable.id == page.id)
    }

    fn retained_page(&self, descriptor: PageDescriptor) -> Option<Page> {
        self.frozen_pages
            .get(&descriptor.id)
            .or_else(|| {
                self.durable
                    .pages
                    .get(&(descriptor.id.key_group, descriptor.id.slot))
            })
            .filter(|page| page.id == descriptor.id && page.len == descriptor.len)
            .cloned()
    }

    fn mark_dirty(&mut self, slot: Slot) {
        if let Some(dirty) = &mut self.dirty {
            dirty.entry(slot.0).or_default().mark(slot.1);
        }
    }

    fn next_generation(&self, slot: Slot) -> u64 {
        self.next_generations.get(&slot).copied().unwrap_or(0)
    }

    fn install_fresh(&mut self, slot: Slot, old: Option<&Page>, suffix: &[u8]) {
        let generation = self.next_generation(slot);
        let next = generation.checked_add(1).expect("page generation overflow");
        let page = match old {
            Some(old) => old.appended(suffix, generation),
            None => Page::new(
                PageId {
                    key_group: slot.0,
                    slot: slot.1,
                    generation,
                },
                suffix.to_vec(),
            ),
        };
        self.logical_bytes = self
            .logical_bytes
            .checked_sub(old.map_or(0, |page| page.len as u64))
            .and_then(|bytes| bytes.checked_add(page.len as u64))
            .expect("logical byte accounting overflow");
        self.next_generations.insert(slot, next);
        self.current.insert(slot, page);
    }

    pub fn append(&mut self, group: u32, slot: u32, suffix: &[u8]) {
        assert!(self.undo_open, "writes require an undo transaction");
        let slot = (group, slot);
        self.record_undo(slot);
        let retained = self
            .current
            .get(&slot)
            .is_some_and(|page| self.retained(page));
        if self.current.contains_key(&slot) && !retained {
            let page = self.current.get_mut(&slot).expect("current page exists");
            Arc::get_mut(&mut page.bytes)
                .expect("an unfrozen page has unique storage")
                .extend_from_slice(suffix);
            page.len = page
                .len
                .checked_add(suffix.len())
                .expect("page length overflow");
            self.logical_bytes = self
                .logical_bytes
                .checked_add(suffix.len() as u64)
                .expect("logical byte accounting overflow");
        } else {
            let old = self.current.get(&slot).cloned();
            self.install_fresh(slot, old.as_ref(), suffix);
        }
        self.mark_dirty(slot);
    }

    pub fn delete(&mut self, group: u32, slot: u32) {
        assert!(self.undo_open, "deletes require an undo transaction");
        let slot = (group, slot);
        self.record_undo(slot);
        let old = self.current.remove(&slot).expect("deleted page must exist");
        self.logical_bytes = self
            .logical_bytes
            .checked_sub(old.len as u64)
            .expect("logical byte accounting underflow");
        let generation = self.next_generation(slot);
        self.next_generations.insert(
            slot,
            generation.checked_add(1).expect("page generation overflow"),
        );
        self.undo
            .last_mut()
            .expect("undo record exists")
            .previous_owned = Some(old);
        self.mark_dirty(slot);
    }

    pub fn rollback(&mut self) {
        assert!(self.undo_open, "no undo transaction is open");
        self.undo_open = false;
        let initial_bytes = self.undo_initial_bytes;
        let records = std::mem::take(&mut self.undo);
        for mut record in records.into_iter().rev() {
            self.restore_undo_page(&mut record);
            self.next_generations
                .insert(record.slot, record.next_generation);
            self.restore_dirty(&record);
        }
        self.logical_bytes = initial_bytes;
        self.undo_initial_bytes = 0;
    }

    fn restore_dirty(&mut self, record: &UndoRecord) {
        let Some(dirty) = &mut self.dirty else {
            return;
        };
        let set = dirty.entry(record.slot.0).or_default();
        if record.dirty.was_dirty {
            set.mark(record.slot.1);
        } else {
            set.bits.remove(&record.slot.1);
        }
        set.first_dirty.truncate(record.dirty.first_dirty_len);
        if set.bits.is_empty() {
            dirty.remove(&record.slot.0);
        }
    }

    fn restore_undo_page(&mut self, record: &mut UndoRecord) {
        let Some(previous) = record.previous else {
            self.current.remove(&record.slot);
            return;
        };
        if self
            .current
            .get(&record.slot)
            .is_some_and(|page| page.id == previous.id)
        {
            let page = self
                .current
                .get_mut(&record.slot)
                .expect("current page exists");
            Arc::get_mut(&mut page.bytes)
                .expect("an unfrozen rollback page has unique storage")
                .truncate(previous.len);
            page.len = previous.len;
            return;
        }
        let restored = self
            .retained_page(previous)
            .or_else(|| owned_undo_page(record, previous))
            .expect("the prior page remains retained by undo or a freeze");
        self.current.insert(record.slot, restored);
    }

    pub fn freeze(&mut self, epoch: u64) -> Result<TableFreeze, String> {
        if !self.checkpointing {
            return Err("checkpointing is disabled".into());
        }
        if self.undo_open {
            return Err("freeze during an open undo transaction".into());
        }
        if let Some(active) = self.active_epoch {
            return Err(format!("epoch {active} still owns frozen pages"));
        }
        let dirty = self.dirty.take().unwrap_or_default();
        let mut changes = Vec::new();
        let mut reservation_bytes = 0_u64;
        for (group, dirty) in dirty {
            for slot in dirty.bits {
                let slot = (group, slot);
                let change = match self.current.get(&slot) {
                    Some(page) => {
                        reservation_bytes = reservation_bytes
                            .checked_add(page.len as u64 + 64)
                            .expect("freeze reservation overflow");
                        self.frozen_pages.insert(page.id, page.clone());
                        PageChange::Write(page.clone())
                    }
                    None => PageChange::Delete(PageId {
                        key_group: slot.0,
                        slot: slot.1,
                        generation: self.next_generation(slot) - 1,
                    }),
                };
                changes.push(change);
            }
        }
        self.dirty = Some(BTreeMap::new());
        self.active_epoch = Some(epoch);
        let digest = TableFreeze::metadata_digest(epoch, &self.durable, &changes);
        Ok(TableFreeze {
            epoch,
            previous: self.durable.clone(),
            changes,
            next_generations: self.next_generations.clone(),
            digest,
            reservation_bytes,
        })
    }

    pub fn publish(&mut self, image: PartitionImage) {
        self.durable = Arc::new(image);
    }

    pub fn settle(&mut self) {
        self.active_epoch = None;
        self.frozen_pages.clear();
    }

    pub fn page(&self, group: u32, slot: u32) -> Option<&Page> {
        self.current.get(&(group, slot))
    }

    pub fn dirty_slots(&self, group: u32) -> Vec<u32> {
        self.dirty
            .as_ref()
            .and_then(|dirty| dirty.get(&group))
            .map_or(Vec::new(), |dirty| dirty.bits.iter().copied().collect())
    }

    pub fn first_dirty_slots(&self, group: u32) -> Vec<u32> {
        self.dirty
            .as_ref()
            .and_then(|dirty| dirty.get(&group))
            .map_or(Vec::new(), |dirty| dirty.first_dirty.clone())
    }

    pub const fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    pub fn durable_image(&self) -> PartitionImage {
        (*self.durable).clone()
    }

    pub fn state_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for ((group, slot), page) in &self.current {
            bytes.extend_from_slice(
                format!("page:{group}:{slot}:{}:{}:", page.id.generation, page.len).as_bytes(),
            );
            bytes.extend_from_slice(page.bytes());
            bytes.push(b'\n');
        }
        for ((group, slot), generation) in &self.next_generations {
            bytes.extend_from_slice(format!("next:{group}:{slot}:{generation}\n").as_bytes());
        }
        bytes
    }
}

fn owned_undo_page(record: &mut UndoRecord, previous: PageDescriptor) -> Option<Page> {
    let mut page = record.previous_owned.take()?;
    if page.id != previous.id || page.len < previous.len {
        return None;
    }
    if page.len > previous.len {
        Arc::get_mut(&mut page.bytes)
            .expect("an owned unfrozen undo page has unique storage")
            .truncate(previous.len);
        page.len = previous.len;
    }
    Some(page)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistenceFault {
    Write,
    Checksum,
    DiskFull,
    WorkerPanic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistenceFailure {
    Write,
    Checksum,
    DiskFull,
    WorkerPanic,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PersistenceCounters {
    pub content_hashes: u64,
    pub file_syncs: u64,
    pub staging_directory_syncs: u64,
    pub committed_directory_syncs: u64,
    pub carried_pages: u64,
    pub carried_roots: u64,
}

#[derive(Clone, Debug)]
pub struct PersistedTable {
    pub image: PartitionImage,
    pub counters: PersistenceCounters,
    evidence_digest: String,
}

impl PersistedTable {
    pub fn evidence_digest(&self) -> &str {
        &self.evidence_digest
    }
}

#[derive(Debug, Default)]
pub struct FakePersistence {
    objects: BTreeMap<String, Vec<u8>>,
    page_hash_calls: BTreeMap<PageId, u64>,
    root_hash_calls: u64,
    read_back_hashes: u64,
}

impl FakePersistence {
    pub fn persist(
        &mut self,
        freeze: &TableFreeze,
        fault: Option<PersistenceFault>,
    ) -> Result<PersistedTable, PersistenceFailure> {
        match fault {
            Some(PersistenceFault::Write) => return Err(PersistenceFailure::Write),
            Some(PersistenceFault::Checksum) => return Err(PersistenceFailure::Checksum),
            Some(PersistenceFault::DiskFull) => return Err(PersistenceFailure::DiskFull),
            Some(PersistenceFault::WorkerPanic) => return Err(join_worker_panic()),
            None => {}
        }
        self.persist_frozen(freeze)
    }

    fn persist_frozen(
        &mut self,
        freeze: &TableFreeze,
    ) -> Result<PersistedTable, PersistenceFailure> {
        let previous = freeze.previous.as_ref();
        let mut pages = previous.pages.clone();
        let mut page_digests = previous.page_digests.clone();
        let mut counters = PersistenceCounters::default();
        let mut affected = BTreeSet::new();
        let mut evidence = freeze.digest.clone();
        for change in &freeze.changes {
            let slot = match change {
                PageChange::Write(page) => (page.id.key_group, page.id.slot),
                PageChange::Delete(id) => (id.key_group, id.slot),
            };
            affected.insert(slot.0);
            match change {
                PageChange::Write(page) => {
                    let digest = self.persist_page(previous, page, &mut counters)?;
                    evidence.push_str(&digest);
                    page_digests.insert(slot, digest);
                    pages.insert(slot, page.clone());
                }
                PageChange::Delete(_) => {
                    pages.remove(&slot);
                    page_digests.remove(&slot);
                }
            }
        }
        let mut image = PartitionImage {
            pages,
            page_digests,
            next_generations: freeze.next_generations.clone(),
            partition_digests: previous.partition_digests.clone(),
            lineage_digest: String::new(),
        };
        for group in &affected {
            let bytes = image.partition_bytes(*group);
            let digest = hex::encode(Sha256::digest(&bytes));
            self.insert_object(format!("root/{group}/{digest}"), bytes)?;
            self.root_hash_calls += 1;
            counters.content_hashes += 1;
            evidence.push_str(&digest);
            image.partition_digests.insert(*group, digest);
        }
        image.refresh_lineage_digest();
        counters.file_syncs = counters.content_hashes;
        counters.staging_directory_syncs = affected.len() as u64;
        counters.committed_directory_syncs = affected.len() as u64;
        counters.carried_pages = previous
            .pages
            .iter()
            .filter(|(slot, page)| image.pages.get(*slot).is_some_and(|new| new.id == page.id))
            .count() as u64;
        counters.carried_roots = previous
            .partition_digests
            .keys()
            .filter(|group| !affected.contains(group))
            .count() as u64;
        Ok(PersistedTable {
            image,
            counters,
            evidence_digest: hex::encode(Sha256::digest(evidence)),
        })
    }

    fn persist_page(
        &mut self,
        previous: &PartitionImage,
        page: &Page,
        counters: &mut PersistenceCounters,
    ) -> Result<String, PersistenceFailure> {
        let slot = (page.id.key_group, page.id.slot);
        if previous
            .pages
            .get(&slot)
            .is_some_and(|old| old.id == page.id)
        {
            if previous.pages[&slot].bytes() != page.bytes() {
                return Err(PersistenceFailure::Checksum);
            }
            return Ok(previous.page_digests[&slot].clone());
        }
        let digest = hex::encode(Sha256::digest(page.bytes()));
        *self.page_hash_calls.entry(page.id).or_default() += 1;
        counters.content_hashes += 1;
        self.insert_object(
            format!(
                "page/{}/{}/{}/{digest}",
                page.id.key_group, page.id.slot, page.id.generation
            ),
            page.bytes().to_vec(),
        )?;
        Ok(digest)
    }

    fn insert_object(&mut self, path: String, bytes: Vec<u8>) -> Result<(), PersistenceFailure> {
        if self.objects.get(&path).is_some_and(|old| old != &bytes) {
            return Err(PersistenceFailure::Checksum);
        }
        self.objects.insert(path, bytes);
        Ok(())
    }

    pub fn page_hash_calls(&self, id: PageId) -> u64 {
        self.page_hash_calls.get(&id).copied().unwrap_or(0)
    }

    pub const fn root_hash_calls(&self) -> u64 {
        self.root_hash_calls
    }

    pub const fn read_back_hashes(&self) -> u64 {
        self.read_back_hashes
    }
}

fn join_worker_panic() -> PersistenceFailure {
    let joined = thread::spawn(|| panic!("injected checkpoint worker panic")).join();
    assert!(joined.is_err(), "the injected worker must panic");
    PersistenceFailure::WorkerPanic
}
