use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    mem::size_of,
    ops::Range,
    sync::Arc,
};

use datafusion::arrow::array::{Array, ArrayRef, Float64Array, StringArray, UInt64Array};

use super::{
    internal, keyed_array,
    pages::{AllocatorUndo, DirtySet, Page, SlotAllocator},
    ring_array,
};
use crate::{Result, accumulate::NumericAccumulator};

pub(crate) use super::pages::{KeyGroupId, PageId};

const KEYED_PAGE_ROWS: usize = 1_024;
const VALUE_LOG_PAGE_BYTES: usize = 64 * 1_024;
const PAGE_DESCRIPTOR_BYTES: usize = 64;

type RestoredPartition = (StatePartition, Vec<(Vec<u8>, EntityId)>);

#[derive(Clone, Copy)]
struct ValueLogUndo {
    chunks: usize,
    len: u32,
    logical_bytes: usize,
    total_len: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EntityId {
    index: u32,
    key_group: KeyGroupId,
    row: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct EntityKey {
    pub(super) bytes: Vec<u8>,
}

impl EntityKey {
    pub(crate) fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    pub(in crate::state::table) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ValueKind {
    UInt64,
    Int64,
    Float64,
    Boolean,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RingKind {
    Float64,
    Int64,
    UInt64,
    PairFloat64,
}

impl RingKind {
    fn width(self) -> usize {
        match self {
            Self::Float64 | Self::Int64 | Self::UInt64 => 8,
            Self::PairFloat64 => 16,
        }
    }

    fn validity_bits(self) -> usize {
        usize::from(self != Self::PairFloat64) + usize::from(self == Self::PairFloat64) * 2
    }

    pub(super) fn empty(self) -> RingValue {
        match self {
            Self::Float64 => RingValue::Float64(None),
            Self::Int64 => RingValue::Int64(None),
            Self::UInt64 => RingValue::UInt64(None),
            Self::PairFloat64 => RingValue::PairFloat64(None),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Retention {
    Rows(usize),
    Duration,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum RingValue {
    Float64(Option<f64>),
    Int64(Option<i64>),
    UInt64(Option<u64>),
    PairFloat64(Option<(f64, f64)>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum KeyedValue {
    UInt64(u64),
    Int64(i64),
    Float64(f64),
    Boolean(bool),
}

impl KeyedValue {
    fn default(kind: ValueKind) -> Self {
        match kind {
            ValueKind::UInt64 => Self::UInt64(0),
            ValueKind::Int64 => Self::Int64(0),
            ValueKind::Float64 => Self::Float64(0.0),
            ValueKind::Boolean => Self::Boolean(false),
        }
    }

    fn width(self) -> usize {
        match self {
            Self::UInt64(_) | Self::Int64(_) | Self::Float64(_) => 8,
            Self::Boolean(_) => 1,
        }
    }
}

#[derive(Clone, Debug)]
struct KeyedColumn {
    name: String,
    kind: ValueKind,
}

#[derive(Clone, Debug)]
pub(super) struct Ring {
    pub(super) head: usize,
    pub(super) len: usize,
    pub(super) slots: Vec<RingValue>,
}

impl Ring {
    fn new(kind: RingKind, capacity: usize) -> Self {
        Self {
            head: 0,
            len: 0,
            slots: vec![kind.empty(); capacity],
        }
    }

    pub(super) fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub(super) fn position(&self, offset: usize) -> usize {
        (self.head + offset) % self.capacity()
    }

    pub(super) fn history(&self) -> Vec<RingValue> {
        (0..self.len)
            .map(|offset| self.slots[self.position(offset)])
            .collect()
    }

    fn value_bytes(&self, kind: RingKind) -> usize {
        self.capacity() * kind.width() + (self.capacity() * kind.validity_bits()).div_ceil(8)
    }
}

#[derive(Clone)]
struct NumericHistory<'a> {
    ring: &'a Ring,
    offset: usize,
}

impl Iterator for NumericHistory<'_> {
    type Item = Option<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.ring.len {
            return None;
        }
        let RingValue::Float64(value) = self.ring.slots[self.ring.position(self.offset)] else {
            unreachable!()
        };
        self.offset += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ring.len - self.offset;
        (remaining, Some(remaining))
    }

    fn count(self) -> usize {
        self.ring.len - self.offset
    }
}

#[derive(Clone, Copy)]
struct NumericRingUpdate {
    previous_head: usize,
    previous_len: usize,
    evicted: Option<f64>,
}

#[derive(Clone, Debug)]
struct RingColumn {
    name: String,
    kind: RingKind,
    retention: Retention,
    numeric: bool,
    numeric_windows: Vec<usize>,
    numeric_dispersion: Vec<bool>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct PartitionMetadata {
    pub(super) entities: Vec<PartitionEntity>,
    pub(super) dictionary_bytes: usize,
    pub(super) value_log_lengths: Vec<u32>,
}

#[derive(Clone, Debug)]
pub(super) struct PartitionEntity {
    pub(super) index: u32,
    pub(super) key: EntityKey,
}

#[derive(Clone, Debug)]
pub(super) struct KeyedPage {
    pub(super) kind: ValueKind,
    pub(super) values: Vec<KeyedValue>,
}

#[derive(Clone, Debug)]
pub(super) struct RingPage {
    pub(super) kind: RingKind,
    pub(super) ring: Ring,
    pub(super) slot_stamps: Vec<u64>,
    pub(super) numeric: Option<Vec<NumericAccumulator>>,
}

#[derive(Clone, Debug)]
pub(super) struct ValueLogPage {
    pub(super) chunks: Vec<ArrayRef>,
    pub(super) start: u32,
    pub(super) len: u32,
    pub(super) logical_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct ValueLogAppendPlan {
    start: u32,
    end: u32,
    len: u32,
    bytes: usize,
    pack: bool,
    page_index: usize,
}

#[derive(Clone, Debug, Default)]
struct ValueLog {
    pages: Vec<Option<Page<ValueLogPage>>>,
    len: u32,
}

#[derive(Clone, Debug)]
pub(super) enum PageRef {
    Metadata(Page<PartitionMetadata>),
    Keyed {
        column: usize,
        page_index: usize,
        page: Page<KeyedPage>,
    },
    Ring {
        column: usize,
        row: usize,
        page: Page<RingPage>,
    },
    ValueLog {
        column: usize,
        page_index: usize,
        page: Page<ValueLogPage>,
    },
}

impl PageRef {
    pub(super) fn id(&self) -> PageId {
        match self {
            Self::Metadata(page) => page.id,
            Self::Keyed { page, .. } => page.id,
            Self::Ring { page, .. } => page.id,
            Self::ValueLog { page, .. } => page.id,
        }
    }

    fn location(&self) -> PageLocation {
        match self {
            Self::Metadata(_) => PageLocation::Metadata,
            Self::Keyed {
                column, page_index, ..
            } => PageLocation::Keyed {
                column: *column,
                page_index: *page_index,
            },
            Self::Ring { column, row, .. } => PageLocation::Ring {
                column: *column,
                row: *row,
            },
            Self::ValueLog {
                column, page_index, ..
            } => PageLocation::ValueLog {
                column: *column,
                page_index: *page_index,
            },
        }
    }

    fn cow(&self, generation: u64) -> Self {
        match self {
            Self::Metadata(page) => Self::Metadata(page.cow(generation)),
            Self::Keyed {
                column,
                page_index,
                page,
            } => Self::Keyed {
                column: *column,
                page_index: *page_index,
                page: page.cow(generation),
            },
            Self::Ring { column, row, page } => Self::Ring {
                column: *column,
                row: *row,
                page: page.cow(generation),
            },
            Self::ValueLog {
                column,
                page_index,
                page,
            } => Self::ValueLog {
                column: *column,
                page_index: *page_index,
                page: page.cow(generation),
            },
        }
    }

    pub(super) fn logical_bytes(&self) -> usize {
        match self {
            Self::Metadata(page) => page.value.dictionary_bytes,
            Self::Keyed { page, .. } => {
                page.value.values.capacity() * KeyedValue::default(page.value.kind).width()
            }
            Self::Ring { page, .. } => {
                page.value.ring.value_bytes(page.value.kind)
                    + size_of::<Ring>()
                    + page.value.numeric.as_ref().map_or(0, |values| {
                        values.capacity() * size_of::<NumericAccumulator>()
                    })
            }
            Self::ValueLog { page, .. } => page.value.logical_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PageLocation {
    Metadata,
    Keyed { column: usize, page_index: usize },
    Ring { column: usize, row: usize },
    ValueLog { column: usize, page_index: usize },
}

#[derive(Debug)]
struct StatePartition {
    id: KeyGroupId,
    metadata: Page<PartitionMetadata>,
    keyed: Vec<Vec<Option<Page<KeyedPage>>>>,
    rings: Vec<Vec<Option<Page<RingPage>>>>,
    value_logs: Vec<ValueLog>,
    locations: BTreeMap<u32, PageLocation>,
    allocator: SlotAllocator,
    dirty: DirtySet,
    stamps: BTreeMap<StampKey, u64>,
}

impl StatePartition {
    fn new(id: KeyGroupId) -> Self {
        let metadata = Page::new(
            PageId {
                key_group: id,
                slot: 0,
                generation: 0,
            },
            PartitionMetadata::default(),
        );
        let mut allocator = SlotAllocator::default();
        allocator.next_generations.insert(0, 1);
        allocator.next_slot = 1;
        Self {
            id,
            metadata,
            keyed: Vec::new(),
            rings: Vec::new(),
            value_logs: Vec::new(),
            locations: BTreeMap::from([(0, PageLocation::Metadata)]),
            allocator,
            dirty: DirtySet::default(),
            stamps: BTreeMap::new(),
        }
    }

    fn page(&self, slot: u32) -> Option<PageRef> {
        match self.locations.get(&slot).copied()? {
            PageLocation::Metadata => Some(PageRef::Metadata(self.metadata.clone())),
            PageLocation::Keyed { column, page_index } => Some(PageRef::Keyed {
                column,
                page_index,
                page: self.keyed[column][page_index].clone()?,
            }),
            PageLocation::Ring { column, row } => Some(PageRef::Ring {
                column,
                row,
                page: self.rings[column][row].clone()?,
            }),
            PageLocation::ValueLog { column, page_index } => Some(PageRef::ValueLog {
                column,
                page_index,
                page: self.value_logs[column].pages[page_index].clone()?,
            }),
        }
    }

    fn insert(&mut self, page: PageRef) {
        let id = page.id();
        self.locations.insert(id.slot, page.location());
        match page {
            PageRef::Metadata(page) => self.metadata = page,
            PageRef::Keyed {
                column,
                page_index,
                page,
            } => set_page(&mut self.keyed[column], page_index, page),
            PageRef::Ring { column, row, page } => set_page(&mut self.rings[column], row, page),
            PageRef::ValueLog {
                column,
                page_index,
                page,
            } => set_page(&mut self.value_logs[column].pages, page_index, page),
        }
    }

    fn remove_location(&mut self, slot: u32) -> Result<PageLocation> {
        let location = self
            .locations
            .get(&slot)
            .copied()
            .ok_or_else(|| internal("state-table page location is absent"))?;
        if matches!(location, PageLocation::Metadata) {
            return Err(internal("partition metadata page cannot be deleted"));
        }
        self.locations.remove(&slot);
        Ok(location)
    }

    fn remove(&mut self, slot: u32) -> Result<PageRef> {
        match self.remove_location(slot)? {
            PageLocation::Metadata => unreachable!(),
            PageLocation::Keyed { column, page_index } => self.take_keyed(column, page_index),
            PageLocation::Ring { column, row } => self.take_ring(column, row),
            PageLocation::ValueLog { column, page_index } => {
                self.take_value_log(column, page_index)
            }
        }
    }

    fn take_keyed(&mut self, column: usize, page_index: usize) -> Result<PageRef> {
        Ok(PageRef::Keyed {
            column,
            page_index,
            page: self.keyed[column][page_index]
                .take()
                .ok_or_else(|| internal("keyed page is absent"))?,
        })
    }

    fn take_ring(&mut self, column: usize, row: usize) -> Result<PageRef> {
        Ok(PageRef::Ring {
            column,
            row,
            page: self.rings[column][row]
                .take()
                .ok_or_else(|| internal("ring page is absent"))?,
        })
    }

    fn take_value_log(&mut self, column: usize, page_index: usize) -> Result<PageRef> {
        Ok(PageRef::ValueLog {
            column,
            page_index,
            page: self.value_logs[column].pages[page_index]
                .take()
                .ok_or_else(|| internal("value-log page is absent"))?,
        })
    }

    fn logical_bytes(&self) -> usize {
        self.locations
            .keys()
            .filter_map(|slot| self.page(*slot))
            .map(|page| page.logical_bytes())
            .sum()
    }

    fn deep_copy(&self) -> Self {
        let mut copied = StatePartition::new(self.id);
        copied.keyed = vec![Vec::new(); self.keyed.len()];
        copied.rings = vec![Vec::new(); self.rings.len()];
        copied.value_logs = vec![ValueLog::default(); self.value_logs.len()];
        for slot in self.locations.keys() {
            let page = self.page(*slot).expect("located page");
            copied.insert(page.cow(page.id().generation));
        }
        let value_log_lengths = copied.metadata.value.value_log_lengths.clone();
        for (column, log) in copied.value_logs.iter_mut().enumerate() {
            log.len = value_log_lengths.get(column).copied().unwrap_or(0);
        }
        copied.allocator = self.allocator.clone();
        copied.dirty = self.dirty.clone();
        copied.stamps = self.stamps.clone();
        copied
    }
}

fn set_page<T>(pages: &mut Vec<Option<Page<T>>>, index: usize, page: Page<T>) {
    if pages.len() <= index {
        pages.resize_with(index + 1, || None);
    }
    pages[index] = Some(page);
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TableLineage {
    pub(super) epoch: u64,
    pub(super) partitions: BTreeMap<KeyGroupId, PartitionLineage>,
    pub(super) retained_bytes: usize,
}

#[derive(Clone, Debug, Default)]
pub(super) struct PartitionLineage {
    pub(super) pages: BTreeMap<u32, PageRef>,
    pub(super) allocator: SlotAllocator,
}

impl TableLineage {
    fn deep_copy(&self) -> Self {
        Self {
            epoch: self.epoch,
            partitions: self
                .partitions
                .iter()
                .map(|(group, partition)| {
                    let pages = partition
                        .pages
                        .iter()
                        .map(|(slot, page)| (*slot, page.cow(page.id().generation)))
                        .collect();
                    (
                        *group,
                        PartitionLineage {
                            pages,
                            allocator: partition.allocator.clone(),
                        },
                    )
                })
                .collect(),
            retained_bytes: self.retained_bytes,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum FrozenChange {
    Write(PageRef),
    Delete(PageId),
}

#[derive(Clone, Debug)]
pub(super) struct PartitionFreeze {
    pub(super) id: KeyGroupId,
    pub(super) metadata: Page<PartitionMetadata>,
    pub(super) changes: Vec<FrozenChange>,
    pub(super) allocator: SlotAllocator,
    pub(super) dirty_slots: Vec<u32>,
}

#[derive(Debug)]
pub(crate) struct TableFreeze {
    pub(super) epoch: u64,
    pub(super) previous: Arc<TableLineage>,
    pub(super) partitions: Vec<PartitionFreeze>,
    reservation_bytes: usize,
}

impl TableFreeze {
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    pub(crate) fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    pub(crate) fn captured_pages(&self) -> usize {
        self.partitions
            .iter()
            .flat_map(|partition| &partition.changes)
            .filter(|change| matches!(change, FrozenChange::Write(_)))
            .count()
    }

    pub(crate) fn reservation_bytes(&self) -> usize {
        self.reservation_bytes
    }

    pub(crate) fn ring_history(&self, column: usize, entity: EntityId) -> Result<Vec<Option<f64>>> {
        let page = self
            .find_frozen_ring(column, entity)
            .or_else(|| self.find_previous_ring(column, entity))
            .ok_or_else(|| internal("frozen ring page is absent"))?;
        Ok(f64_history(&page))
    }

    fn find_frozen_ring(&self, column: usize, entity: EntityId) -> Option<Page<RingPage>> {
        let row = entity.row as usize;
        self.partitions
            .iter()
            .filter(|partition| partition.id == entity.key_group)
            .flat_map(|partition| &partition.changes)
            .find_map(|change| match change {
                FrozenChange::Write(PageRef::Ring {
                    column: found,
                    row: found_row,
                    page,
                }) if *found == column && *found_row == row => Some(page.clone()),
                _ => None,
            })
    }

    fn find_previous_ring(&self, column: usize, entity: EntityId) -> Option<Page<RingPage>> {
        let row = entity.row as usize;
        self.previous
            .partitions
            .get(&entity.key_group)
            .and_then(|partition| {
                partition.pages.values().find_map(|page| match page {
                    PageRef::Ring {
                        column: found,
                        row: found_row,
                        page,
                    } if *found == column && *found_row == row => Some(page.clone()),
                    _ => None,
                })
            })
    }

    pub(crate) fn complete(self) -> TableLineage {
        let mut lineage = (*self.previous).clone();
        lineage.epoch = self.epoch;
        for mut frozen in self.partitions {
            let partition = lineage
                .partitions
                .get_mut(&frozen.id)
                .expect("prepared group");
            Self::apply_frozen_changes(partition, std::mem::take(&mut frozen.changes));
            Self::apply_frozen_allocator(partition, &frozen);
            Self::ensure_frozen_metadata(partition, frozen.metadata);
        }
        lineage.retained_bytes = lineage
            .partitions
            .values()
            .flat_map(|partition| partition.pages.values())
            .map(PageRef::logical_bytes)
            .sum();
        lineage
    }

    fn apply_frozen_changes(partition: &mut PartitionLineage, changes: Vec<FrozenChange>) {
        for change in changes {
            match change {
                FrozenChange::Write(page) => {
                    partition.pages.insert(page.id().slot, page);
                }
                FrozenChange::Delete(id) => {
                    partition.pages.remove(&id.slot);
                }
            }
        }
    }

    fn apply_frozen_allocator(partition: &mut PartitionLineage, frozen: &PartitionFreeze) {
        for slot in &frozen.dirty_slots {
            if let Some(generation) = frozen.allocator.next_generations.get(slot) {
                partition
                    .allocator
                    .next_generations
                    .insert(*slot, *generation);
            }
            if frozen.allocator.free_slots.contains(slot) {
                partition.allocator.free_slots.insert(*slot);
            } else {
                partition.allocator.free_slots.remove(slot);
            }
        }
        partition.allocator.next_slot = frozen.allocator.next_slot;
        let metadata_slot = frozen.metadata.id.slot;
        if let Some(generation) = frozen.allocator.next_generations.get(&metadata_slot) {
            partition
                .allocator
                .next_generations
                .insert(metadata_slot, *generation);
        }
    }

    fn ensure_frozen_metadata(partition: &mut PartitionLineage, metadata: Page<PartitionMetadata>) {
        if !partition
            .pages
            .values()
            .any(|page| matches!(page, PageRef::Metadata(_)))
        {
            partition
                .pages
                .insert(metadata.id.slot, PageRef::Metadata(metadata));
        }
    }
}

fn f64_history(page: &Page<RingPage>) -> Vec<Option<f64>> {
    page.value
        .ring
        .history()
        .into_iter()
        .map(|value| match value {
            RingValue::Float64(value) => value,
            _ => unreachable!(),
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum StampKind {
    Dirty,
    Keyed,
    RingMetadata,
    RingSlot,
    Numeric,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StampKey {
    slot: u32,
    kind: StampKind,
    sub: u32,
}

#[derive(Debug)]
enum Undo {
    Entity {
        group: KeyGroupId,
        key: EntityKey,
        next_entity: u32,
    },
    Dirty {
        group: KeyGroupId,
        slot: u32,
        was_dirty: bool,
        first_dirty_len: usize,
    },
    Stamp {
        group: KeyGroupId,
        key: StampKey,
        previous: Option<u64>,
    },
    PageCreate {
        group: KeyGroupId,
        slot: u32,
        allocator: AllocatorUndo,
    },
    PageReplace {
        group: KeyGroupId,
        page: PageRef,
        allocator: AllocatorUndo,
    },
    PageRemove {
        group: KeyGroupId,
        page: PageRef,
        allocator: AllocatorUndo,
    },
    Keyed {
        group: KeyGroupId,
        slot: u32,
        values: Vec<KeyedValue>,
        capacity: usize,
    },
    RingMetadata {
        group: KeyGroupId,
        slot: u32,
        head: usize,
        len: usize,
        capacity: usize,
    },
    RingSlot {
        group: KeyGroupId,
        slot: u32,
        index: usize,
        value: RingValue,
    },
    Numeric {
        group: KeyGroupId,
        slot: u32,
        value: Vec<NumericAccumulator>,
    },
    ValueLog {
        group: KeyGroupId,
        column: usize,
        slot: u32,
        chunks: usize,
        len: u32,
        logical_bytes: usize,
        total_len: u32,
    },
}

#[derive(Debug)]
struct Transaction {
    id: u64,
    undo: Vec<Undo>,
}

#[derive(Debug)]
struct ActiveFreeze {
    epoch: u64,
    frozen: BTreeSet<PageId>,
    reservation_bytes: usize,
    dirty: BTreeMap<KeyGroupId, DirtySet>,
}

#[derive(Clone)]
pub(super) struct KeyedSnapshot {
    pub(super) name: String,
    pub(super) kind: ValueKind,
    pub(super) values: ArrayRef,
}

#[derive(Clone)]
pub(super) struct RingSnapshot {
    pub(super) name: String,
    pub(super) kind: RingKind,
    pub(super) retention: Retention,
    pub(super) values: ArrayRef,
    pub(super) offsets: Vec<usize>,
    pub(super) capacities: Vec<usize>,
    pub(super) numeric: Option<Vec<Vec<NumericAccumulator>>>,
}

#[derive(Clone)]
struct ValueLogSnapshot {
    group: KeyGroupId,
    name: String,
    chunks: Vec<ArrayRef>,
    len: u32,
}

#[derive(Clone)]
pub(crate) struct TableSnapshot {
    pub(super) keys: Vec<EntityKey>,
    pub(super) keyed: Vec<KeyedSnapshot>,
    pub(super) rings: Vec<RingSnapshot>,
    value_logs: Vec<ValueLogSnapshot>,
    pub(super) logical_bytes: usize,
    lineage: TableLineage,
    next_entity: u32,
}

impl TableSnapshot {
    pub(crate) fn logical_bytes(&self) -> usize {
        self.logical_bytes
    }
}

pub(super) struct Layout7HistorySchema {
    pub(super) counter_name: String,
    pub(super) keyed_names: Vec<String>,
    pub(super) history_name: String,
    pub(super) kind: RingKind,
    pub(super) retention: Retention,
}

pub(super) struct Layout7Schema {
    pub(super) counter_name: String,
    pub(super) keyed_names: Vec<String>,
    pub(super) numeric_name: String,
    pub(super) retention: Retention,
    pub(super) numeric_windows: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct StateTable {
    keyed: Vec<KeyedColumn>,
    rings: Vec<RingColumn>,
    value_logs: Vec<String>,
    partitions: BTreeMap<KeyGroupId, StatePartition>,
    entities: HashMap<Vec<u8>, EntityId>,
    next_entity: u32,
    transaction: Option<Transaction>,
    next_transaction: u64,
    checkpointing: bool,
    durable: Arc<TableLineage>,
    active: Option<ActiveFreeze>,
}

impl Clone for StateTable {
    fn clone(&self) -> Self {
        assert!(
            self.transaction.is_none() && self.active.is_none(),
            "state-table clones occur between transactions and freezes"
        );
        Self {
            keyed: self.keyed.clone(),
            rings: self.rings.clone(),
            value_logs: self.value_logs.clone(),
            partitions: self
                .partitions
                .iter()
                .map(|(group, partition)| (*group, partition.deep_copy()))
                .collect(),
            entities: self.entities.clone(),
            next_entity: self.next_entity,
            transaction: None,
            next_transaction: self.next_transaction,
            checkpointing: self.checkpointing,
            durable: Arc::new(self.durable.deep_copy()),
            active: None,
        }
    }
}

impl Default for StateTable {
    fn default() -> Self {
        Self::new()
    }
}

impl StateTable {
    pub(crate) fn new() -> Self {
        Self::with_checkpointing(false)
    }

    pub(crate) fn with_checkpointing(checkpointing: bool) -> Self {
        Self::with_key_groups(checkpointing, &[KeyGroupId(0)]).expect("default key group is valid")
    }

    pub(crate) fn with_key_groups(checkpointing: bool, groups: &[KeyGroupId]) -> Result<Self> {
        if groups.is_empty() {
            return Err(internal("state table requires at least one key group"));
        }
        let unique = groups.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != groups.len() {
            return Err(internal("state-table key groups must be unique"));
        }
        let partitions = unique
            .into_iter()
            .map(|group| (group, StatePartition::new(group)))
            .collect();
        let durable = TableLineage {
            partitions: groups
                .iter()
                .map(|group| (*group, PartitionLineage::default()))
                .collect(),
            ..TableLineage::default()
        };
        Ok(Self {
            keyed: Vec::new(),
            rings: Vec::new(),
            value_logs: Vec::new(),
            partitions,
            entities: HashMap::new(),
            next_entity: 0,
            transaction: None,
            next_transaction: 1,
            checkpointing,
            durable: Arc::new(durable),
            active: None,
        })
    }

    pub(crate) fn add_keyed_column(&mut self, name: &str, kind: ValueKind) -> usize {
        assert!(
            self.transaction.is_none() && self.next_entity == 0,
            "keyed schema must precede entities"
        );
        self.keyed.push(KeyedColumn {
            name: name.to_owned(),
            kind,
        });
        for partition in self.partitions.values_mut() {
            partition.keyed.push(Vec::new());
        }
        self.keyed.len() - 1
    }

    pub(crate) fn add_ring_column(
        &mut self,
        name: &str,
        kind: RingKind,
        retention: Retention,
    ) -> usize {
        assert!(
            self.transaction.is_none() && self.next_entity == 0,
            "ring schema must precede entities"
        );
        self.rings.push(RingColumn {
            name: name.to_owned(),
            kind,
            retention,
            numeric: false,
            numeric_windows: Vec::new(),
            numeric_dispersion: Vec::new(),
        });
        for partition in self.partitions.values_mut() {
            partition.rings.push(Vec::new());
        }
        self.rings.len() - 1
    }

    pub(crate) fn add_f64_rows_column(&mut self, name: &str, window: usize) -> usize {
        assert!(window > 0, "rolling history windows must be non-empty");
        self.add_ring_column(name, RingKind::Float64, Retention::Rows(window))
    }

    pub(crate) fn add_i64_rows_column(&mut self, name: &str, window: usize) -> usize {
        assert!(window > 0, "rolling history windows must be non-empty");
        self.add_ring_column(name, RingKind::Int64, Retention::Rows(window))
    }

    pub(crate) fn add_u64_rows_column(&mut self, name: &str, window: usize) -> usize {
        assert!(window > 0, "rolling history windows must be non-empty");
        self.add_ring_column(name, RingKind::UInt64, Retention::Rows(window))
    }

    pub(crate) fn add_numeric_rows_column(&mut self, name: &str, window: usize) -> usize {
        assert!(window > 0, "rolling Rows windows must be non-empty");
        let column = self.add_ring_column(name, RingKind::Float64, Retention::Rows(window));
        self.rings[column].numeric = true;
        self.rings[column].numeric_windows = vec![window];
        self.rings[column].numeric_dispersion = vec![true];
        column
    }

    pub(crate) fn add_numeric_window(&mut self, column: usize, window: usize) -> usize {
        assert!(window > 0, "rolling Rows windows must be non-empty");
        assert!(
            self.transaction.is_none() && self.next_entity == 0,
            "numeric windows must precede entities"
        );
        assert!(self.rings[column].numeric, "numeric Rows group is required");
        let Retention::Rows(retained) = self.rings[column].retention else {
            panic!("shared numeric windows require Rows retention");
        };
        self.rings[column].retention = Retention::Rows(retained.max(window));
        self.rings[column].numeric_windows.push(window);
        self.rings[column].numeric_dispersion.push(true);
        self.rings[column].numeric_windows.len() - 1
    }

    pub(crate) fn set_numeric_dispersion(&mut self, column: usize, group: usize, enabled: bool) {
        assert!(
            self.transaction.is_none() && self.next_entity == 0,
            "numeric families must precede entities"
        );
        self.rings[column].numeric_dispersion[group] = enabled;
    }

    pub(crate) fn add_value_log(&mut self, name: &str) -> usize {
        assert!(
            self.transaction.is_none(),
            "cannot alter a state table inside a transaction"
        );
        self.value_logs.push(name.to_owned());
        for partition in self.partitions.values_mut() {
            partition.value_logs.push(ValueLog::default());
        }
        self.value_logs.len() - 1
    }

    fn layout7_columns(&self) -> Option<(&KeyedColumn, &RingColumn)> {
        if !self.value_logs.is_empty()
            || !self.partitions.contains_key(&KeyGroupId(0))
            || self.keyed.is_empty()
            || self
                .keyed
                .iter()
                .any(|column| column.kind != ValueKind::UInt64)
        {
            return None;
        }
        let counter = self.keyed.first()?;
        let [numeric] = self.rings.as_slice() else {
            return None;
        };
        Self::layout7_column_kinds_match(counter, numeric).then_some((counter, numeric))
    }

    fn layout7_column_kinds_match(counter: &KeyedColumn, numeric: &RingColumn) -> bool {
        counter.kind == ValueKind::UInt64
            && numeric.kind == RingKind::Float64
            && numeric.numeric
            && !numeric.numeric_windows.is_empty()
            && matches!(numeric.retention, Retention::Rows(_))
    }

    pub(super) fn layout7_schema(&self) -> Option<Layout7Schema> {
        let (counter, numeric) = self.layout7_columns()?;
        Some(Layout7Schema {
            counter_name: counter.name.clone(),
            keyed_names: self
                .keyed
                .iter()
                .map(|column| column.name.clone())
                .collect(),
            numeric_name: numeric.name.clone(),
            retention: numeric.retention,
            numeric_windows: numeric.numeric_windows.clone(),
        })
    }

    pub(super) fn layout7_history_schema(&self) -> Option<Layout7HistorySchema> {
        if !self.value_logs.is_empty()
            || !self.partitions.contains_key(&KeyGroupId(0))
            || self.keyed.is_empty()
            || self
                .keyed
                .iter()
                .any(|column| column.kind != ValueKind::UInt64)
        {
            return None;
        }
        let counter = self.keyed.first()?;
        let [history] = self.rings.as_slice() else {
            return None;
        };
        if history.numeric
            || !matches!(
                history.kind,
                RingKind::Float64 | RingKind::Int64 | RingKind::UInt64
            )
            || !matches!(history.retention, Retention::Rows(_))
        {
            return None;
        }
        Some(Layout7HistorySchema {
            counter_name: counter.name.clone(),
            keyed_names: self
                .keyed
                .iter()
                .map(|column| column.name.clone())
                .collect(),
            history_name: history.name.clone(),
            kind: history.kind,
            retention: history.retention,
        })
    }

    pub(super) fn layout7_keyed_schema(&self) -> Option<Vec<String>> {
        if !self.value_logs.is_empty()
            || !self.rings.is_empty()
            || self.keyed.is_empty()
            || self.partitions.len() != 1
            || !self.partitions.contains_key(&KeyGroupId(0))
            || self
                .keyed
                .iter()
                .any(|column| column.kind != ValueKind::UInt64)
        {
            return None;
        }
        Some(
            self.keyed
                .iter()
                .map(|column| column.name.clone())
                .collect(),
        )
    }

    pub(crate) fn begin(&mut self) -> Result<()> {
        if self.transaction.is_some() {
            return Err(internal("rolling state transaction is already active"));
        }
        let id = self.next_transaction;
        self.next_transaction = self
            .next_transaction
            .checked_add(1)
            .ok_or_else(|| internal("rolling state transaction stamp exhausted"))?;
        self.transaction = Some(Transaction {
            id,
            undo: Vec::new(),
        });
        Ok(())
    }

    pub(crate) fn commit(&mut self) -> Result<()> {
        self.transaction
            .take()
            .map(|_| ())
            .ok_or_else(|| internal("rolling state transaction is absent"))
    }

    pub(crate) fn rollback(&mut self) -> Result<()> {
        let transaction = self
            .transaction
            .take()
            .ok_or_else(|| internal("rolling state transaction is absent"))?;
        for undo in transaction.undo.into_iter().rev() {
            self.apply_undo(undo)?;
        }
        self.next_transaction = transaction.id;
        for partition in self.partitions.values_mut() {
            partition.stamps.retain(|_, stamp| *stamp != transaction.id);
        }
        Ok(())
    }

    fn transaction_id(&self) -> Result<u64> {
        self.transaction
            .as_ref()
            .map(|transaction| transaction.id)
            .ok_or_else(|| internal("state-table mutation requires a transaction"))
    }

    fn push_undo(&mut self, undo: Undo) {
        self.transaction
            .as_mut()
            .expect("transaction checked")
            .undo
            .push(undo);
    }

    fn note_stamp(&mut self, group: KeyGroupId, slot: u32, kind: StampKind, sub: u32) -> bool {
        let id = self.transaction_id().expect("transaction checked");
        let key = StampKey { slot, kind, sub };
        if kind == StampKind::RingSlot {
            let index = usize::try_from(sub).expect("u32 slot stamp index fits usize");
            let partition = &self.partitions[&group];
            let PageLocation::Ring { column, row } = partition.locations[&slot] else {
                panic!("ring stamp points to a non-ring page")
            };
            let previous = partition.rings[column][row]
                .as_ref()
                .expect("ring page")
                .value
                .slot_stamps[index];
            if previous == id {
                return false;
            }
            self.push_undo(Undo::Stamp {
                group,
                key,
                previous: Some(previous),
            });
            let page = &mut self
                .partitions
                .get_mut(&group)
                .expect("prepared group")
                .rings[column][row]
                .as_mut()
                .expect("ring page")
                .value;
            Arc::get_mut(page).expect("unfrozen ring stamp").slot_stamps[index] = id;
            return true;
        }
        let previous = self.partitions[&group].stamps.get(&key).copied();
        if previous == Some(id) {
            return false;
        }
        self.push_undo(Undo::Stamp {
            group,
            key,
            previous,
        });
        self.partitions
            .get_mut(&group)
            .expect("prepared group")
            .stamps
            .insert(key, id);
        true
    }

    fn mark_dirty(&mut self, group: KeyGroupId, slot: u32) -> Result<()> {
        if !self.checkpointing {
            return Ok(());
        }
        self.transaction_id()?;
        if self.note_stamp(group, slot, StampKind::Dirty, 0) {
            let partition = self.partitions.get(&group).expect("prepared group");
            let was_dirty = partition.dirty.bits.contains(&slot);
            let first_dirty_len = partition.dirty.first_dirty.len();
            self.push_undo(Undo::Dirty {
                group,
                slot,
                was_dirty,
                first_dirty_len,
            });
        }
        self.partitions
            .get_mut(&group)
            .expect("prepared group")
            .dirty
            .mark(slot);
        Ok(())
    }

    fn create_page_in_group(
        &mut self,
        group: KeyGroupId,
        build: impl FnOnce(PageId) -> PageRef,
    ) -> Result<PageId> {
        self.transaction_id()?;
        let (id, allocator) = self
            .partitions
            .get_mut(&group)
            .expect("prepared group")
            .allocator
            .allocate(group)?;
        let slot = id.slot;
        self.push_undo(Undo::PageCreate {
            group,
            slot,
            allocator,
        });
        self.partitions
            .get_mut(&group)
            .expect("prepared group")
            .insert(build(id));
        self.mark_dirty(group, slot)?;
        Ok(id)
    }

    fn replace_page(&mut self, group: KeyGroupId, slot: u32, force: bool) -> Result<()> {
        self.transaction_id()?;
        if !self.checkpointing {
            return Ok(());
        }
        self.mark_dirty(group, slot)?;
        let current = self.partitions[&group]
            .page(slot)
            .ok_or_else(|| internal("state-table page is absent"))?;
        if !self.page_replacement_required(current.id(), force) {
            return Ok(());
        }
        self.copy_on_write_page(group, slot, current)
    }

    fn page_replacement_required(&self, id: PageId, force: bool) -> bool {
        force || self.is_frozen(id)
    }

    fn copy_on_write_page(&mut self, group: KeyGroupId, slot: u32, current: PageRef) -> Result<()> {
        let (generation, allocator) = self
            .partitions
            .get_mut(&group)
            .expect("prepared group")
            .allocator
            .replace(slot)?;
        let replacement = current.cow(generation);
        self.push_undo(Undo::PageReplace {
            group,
            page: current,
            allocator,
        });
        self.partitions
            .get_mut(&group)
            .expect("prepared group")
            .insert(replacement);
        Ok(())
    }

    fn is_frozen(&self, id: PageId) -> bool {
        if !self.checkpointing {
            return false;
        }
        self.active
            .as_ref()
            .is_some_and(|active| active.frozen.contains(&id))
            || self
                .durable
                .partitions
                .get(&id.key_group)
                .and_then(|partition| partition.pages.get(&id.slot))
                .is_some_and(|page| page.id() == id)
    }

    fn resolve_or_append_default(&mut self, key: EntityKey) -> Result<EntityId> {
        self.resolve_or_append_in_group(key, KeyGroupId(0))
    }

    pub(crate) fn resolve_or_append(&mut self, key: EntityKey) -> Result<EntityId> {
        self.resolve_or_append_default(key)
    }

    fn new_entity_id(&self, group: KeyGroupId) -> Result<EntityId> {
        self.transaction_id()?;
        if !self.partitions.contains_key(&group) {
            return Err(internal("entity key group is not prepared"));
        }
        let row = self.partitions[&group].metadata.value.entities.len();
        let row = u32::try_from(row).map_err(|_| internal("partition entity space exhausted"))?;
        Ok(EntityId {
            index: self.next_entity,
            key_group: group,
            row,
        })
    }

    fn append_entity_columns(&mut self, entity: EntityId, key: &EntityKey) -> Result<()> {
        for column in 0..self.keyed.len() {
            self.append_keyed_default(entity.key_group, column, entity.row)?;
        }
        for column in 0..self.rings.len() {
            self.append_ring(entity.key_group, column, entity.row)?;
        }
        self.append_metadata(entity.key_group, entity, key.clone())
    }

    fn register_entity(
        &mut self,
        key: EntityKey,
        entity: EntityId,
        next_entity: u32,
    ) -> Result<EntityId> {
        self.entities.insert(key.bytes.clone(), entity);
        self.next_entity = self
            .next_entity
            .checked_add(1)
            .ok_or_else(|| internal("rolling entity dictionary exhausted u32 IDs"))?;
        self.push_undo(Undo::Entity {
            group: entity.key_group,
            key,
            next_entity,
        });
        Ok(entity)
    }

    pub(crate) fn resolve_or_append_in_group(
        &mut self,
        key: EntityKey,
        group: KeyGroupId,
    ) -> Result<EntityId> {
        if let Some(entity) = self.entities.get(key.bytes.as_slice()) {
            return Ok(*entity);
        }
        let next_entity = self.next_entity;
        let entity = self.new_entity_id(group)?;
        self.append_entity_columns(entity, &key)?;
        self.register_entity(key, entity, next_entity)
    }

    fn append_keyed_default(&mut self, group: KeyGroupId, column: usize, row: u32) -> Result<()> {
        let page_index = row as usize / KEYED_PAGE_ROWS;
        let kind = self.keyed[column].kind;
        if self.partitions[&group].keyed[column]
            .get(page_index)
            .and_then(Option::as_ref)
            .is_none()
        {
            self.create_page_in_group(group, |id| PageRef::Keyed {
                column,
                page_index,
                page: Page::new(
                    id,
                    KeyedPage {
                        kind,
                        values: Vec::new(),
                    },
                ),
            })?;
        }
        let slot = self.partitions[&group].keyed[column][page_index]
            .as_ref()
            .expect("keyed page")
            .id
            .slot;
        self.replace_page(group, slot, false)?;
        let values = &self.partitions[&group].keyed[column][page_index]
            .as_ref()
            .expect("keyed page")
            .value
            .values;
        if values.len() == values.capacity() {
            self.push_undo(Undo::Keyed {
                group,
                slot,
                values: values.clone(),
                capacity: values.capacity(),
            });
        }
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        Arc::get_mut(
            &mut partition.keyed[column][page_index]
                .as_mut()
                .expect("keyed page")
                .value,
        )
        .expect("unfrozen keyed page")
        .values
        .push(KeyedValue::default(kind));
        Ok(())
    }

    fn append_ring(&mut self, group: KeyGroupId, column: usize, row: u32) -> Result<()> {
        let schema = &self.rings[column];
        let capacity = match schema.retention {
            Retention::Rows(capacity) => capacity,
            Retention::Duration => 0,
        };
        let kind = schema.kind;
        let numeric = schema.numeric.then(|| {
            schema
                .numeric_windows
                .iter()
                .zip(&schema.numeric_dispersion)
                .map(|(_, &dispersion)| {
                    if dispersion {
                        NumericAccumulator::default()
                    } else {
                        NumericAccumulator::mean_only()
                    }
                })
                .collect::<Vec<_>>()
        });
        self.create_page_in_group(group, |id| PageRef::Ring {
            column,
            row: row as usize,
            page: Page::new(
                id,
                RingPage {
                    kind,
                    ring: Ring::new(kind, capacity),
                    slot_stamps: vec![0; capacity],
                    numeric,
                },
            ),
        })?;
        Ok(())
    }

    fn append_metadata(
        &mut self,
        group: KeyGroupId,
        entity: EntityId,
        key: EntityKey,
    ) -> Result<()> {
        let slot = self.partitions[&group].metadata.id.slot;
        self.replace_page(group, slot, false)?;
        let key_len = key.bytes.len();
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        let metadata = Arc::get_mut(&mut partition.metadata.value).expect("unfrozen metadata");
        metadata.dictionary_bytes = metadata
            .dictionary_bytes
            .checked_add(key_len)
            .ok_or_else(|| internal("dictionary byte accounting overflow"))?;
        metadata.entities.push(PartitionEntity {
            index: entity.index,
            key,
        });
        Ok(())
    }

    pub(crate) fn entity_count(&self) -> usize {
        self.next_entity as usize
    }

    fn location(&self, entity: EntityId) -> Result<(KeyGroupId, usize)> {
        let partition = self
            .partitions
            .get(&entity.key_group)
            .ok_or_else(|| internal("entity key group is not prepared"))?;
        let row = entity.row as usize;
        let valid = partition
            .metadata
            .value
            .entities
            .get(row)
            .is_some_and(|stored| stored.index == entity.index);
        if valid && entity.index < self.next_entity {
            Ok((entity.key_group, row))
        } else {
            Err(internal("rolling entity ID is out of bounds"))
        }
    }

    pub(crate) fn set_u64(&mut self, column: usize, entity: EntityId, value: u64) -> Result<()> {
        let (group, row) = self.location(entity)?;
        if self.keyed[column].kind != ValueKind::UInt64 {
            return Err(internal("keyed column type mismatch"));
        }
        let page_index = row / KEYED_PAGE_ROWS;
        let row_in_page = row % KEYED_PAGE_ROWS;
        let slot = self.partitions[&group].keyed[column][page_index]
            .as_ref()
            .expect("keyed page")
            .id
            .slot;
        self.replace_page(group, slot, false)?;
        if self.note_stamp(group, slot, StampKind::Keyed, 0) {
            let old = self.partitions[&group].keyed[column][page_index]
                .as_ref()
                .expect("keyed page")
                .value
                .values
                .clone();
            let capacity = self.partitions[&group].keyed[column][page_index]
                .as_ref()
                .expect("keyed page")
                .value
                .values
                .capacity();
            self.push_undo(Undo::Keyed {
                group,
                slot,
                values: old,
                capacity,
            });
        }
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        Arc::get_mut(
            &mut partition.keyed[column][page_index]
                .as_mut()
                .expect("keyed page")
                .value,
        )
        .expect("unfrozen keyed page")
        .values[row_in_page] = KeyedValue::UInt64(value);
        Ok(())
    }

    pub(crate) fn u64(&self, column: usize, entity: EntityId) -> Result<u64> {
        let (group, row) = self.location(entity)?;
        let page = self.partitions[&group].keyed[column][row / KEYED_PAGE_ROWS]
            .as_ref()
            .expect("keyed page");
        match page.value.values[row % KEYED_PAGE_ROWS] {
            KeyedValue::UInt64(value) => Ok(value),
            _ => Err(internal("keyed column type mismatch")),
        }
    }

    #[allow(
        clippy::option_option,
        reason = "ring eviction distinguishes no eviction from an evicted null sample"
    )]
    pub(crate) fn push_f64(
        &mut self,
        column: usize,
        entity: EntityId,
        value: Option<f64>,
    ) -> Result<Option<Option<f64>>> {
        self.push_f64_impl(column, entity, value, true)
    }

    #[allow(
        clippy::option_option,
        reason = "ring eviction distinguishes no eviction from an evicted null sample"
    )]
    fn push_f64_impl(
        &mut self,
        column: usize,
        entity: EntityId,
        value: Option<f64>,
        first_touch: bool,
    ) -> Result<Option<Option<f64>>> {
        if self.rings[column].kind != RingKind::Float64 {
            return Err(internal("ring column type mismatch"));
        }
        self.push_ring(column, entity, RingValue::Float64(value), first_touch)
            .map(|evicted| match evicted {
                Some(RingValue::Float64(value)) => Some(value),
                None => None,
                _ => unreachable!(),
            })
    }

    pub(crate) fn push_i64(
        &mut self,
        column: usize,
        entity: EntityId,
        value: Option<i64>,
    ) -> Result<()> {
        if self.rings[column].kind != RingKind::Int64 {
            return Err(internal("ring column type mismatch"));
        }
        self.push_ring(column, entity, RingValue::Int64(value), true)
            .map(|_| ())
    }

    pub(crate) fn push_u64(
        &mut self,
        column: usize,
        entity: EntityId,
        value: Option<u64>,
    ) -> Result<()> {
        if self.rings[column].kind != RingKind::UInt64 {
            return Err(internal("ring column type mismatch"));
        }
        self.push_ring(column, entity, RingValue::UInt64(value), true)
            .map(|_| ())
    }

    fn push_ring(
        &mut self,
        column: usize,
        entity: EntityId,
        value: RingValue,
        first_touch: bool,
    ) -> Result<Option<RingValue>> {
        let (group, row) = self.location(entity)?;
        let slot = self.partitions[&group].rings[column][row]
            .as_ref()
            .expect("ring page")
            .id
            .slot;
        self.replace_page(group, slot, false)?;
        self.note_ring_metadata_undo(group, column, row, slot, first_touch);
        self.grow_ring_if_needed(group, column, row, slot)?;
        let (index, previous) = {
            let ring = &self.partitions[&group].rings[column][row]
                .as_ref()
                .expect("ring page")
                .value
                .ring;
            (ring.position(ring.len), ring.slots[ring.position(ring.len)])
        };
        self.note_ring_slot_undo(group, slot, index, previous)?;
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        let ring = &mut Arc::get_mut(
            &mut partition.rings[column][row]
                .as_mut()
                .expect("ring page")
                .value,
        )
        .expect("unfrozen ring")
        .ring;
        let evicted = (ring.len == ring.capacity()).then_some(ring.slots[index]);
        if evicted.is_some() {
            ring.head = (ring.head + 1) % ring.capacity();
        } else {
            ring.len += 1;
        }
        ring.slots[index] = value;
        Ok(evicted)
    }

    fn note_ring_metadata_undo(
        &mut self,
        group: KeyGroupId,
        column: usize,
        row: usize,
        slot: u32,
        first_touch: bool,
    ) {
        if first_touch && self.note_stamp(group, slot, StampKind::RingMetadata, 0) {
            let ring = &self.partitions[&group].rings[column][row]
                .as_ref()
                .expect("ring page")
                .value
                .ring;
            self.push_undo(Undo::RingMetadata {
                group,
                slot,
                head: ring.head,
                len: ring.len,
                capacity: ring.capacity(),
            });
        }
    }

    fn note_ring_slot_undo(
        &mut self,
        group: KeyGroupId,
        slot: u32,
        index: usize,
        previous: RingValue,
    ) -> Result<()> {
        let stamp_index =
            u32::try_from(index).map_err(|_| internal("rolling ring slot index exceeds u32"))?;
        if self.note_stamp(group, slot, StampKind::RingSlot, stamp_index) {
            self.push_undo(Undo::RingSlot {
                group,
                slot,
                index,
                value: previous,
            });
        }
        Ok(())
    }

    fn grow_ring_if_needed(
        &mut self,
        group: KeyGroupId,
        column: usize,
        row: usize,
        slot: u32,
    ) -> Result<()> {
        let retention = self.rings[column].retention;
        let kind = self.rings[column].kind;
        let ring = &self.partitions[&group].rings[column][row]
            .as_ref()
            .expect("ring page")
            .value
            .ring;
        if ring.len < ring.capacity() {
            return Ok(());
        }
        if matches!(retention, Retention::Rows(_)) {
            if ring.capacity() == 0 {
                return Err(internal("rolling Rows ring has zero capacity"));
            }
            return Ok(());
        }
        let capacity = if ring.capacity() == 0 {
            1
        } else {
            ring.capacity()
                .checked_mul(2)
                .ok_or_else(|| internal("rolling Duration ring capacity overflowed"))?
        };
        self.replace_page(group, slot, true)?;
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        Arc::get_mut(
            &mut partition.rings[column][row]
                .as_mut()
                .expect("ring page")
                .value,
        )
        .expect("unfrozen ring")
        .ring
        .slots
        .resize(capacity, kind.empty());
        Arc::get_mut(
            &mut partition.rings[column][row]
                .as_mut()
                .expect("ring page")
                .value,
        )
        .expect("unfrozen ring")
        .slot_stamps
        .resize(capacity, 0);
        Ok(())
    }

    pub(crate) fn push_numeric_rows(
        &mut self,
        column: usize,
        entity: EntityId,
        value: Option<f64>,
    ) -> Result<()> {
        self.push_numeric_rows_impl(column, entity, value, true)
    }

    /// Updates an entity whose numeric and ring metadata were already touched
    /// by this transaction. Ring-slot undo remains first-touch per slot.
    pub(crate) fn push_numeric_rows_recurrent(
        &mut self,
        column: usize,
        entity: EntityId,
        value: Option<f64>,
    ) -> Result<()> {
        if !self.checkpointing {
            return self.push_numeric_rows_recurrent_off(column, entity, value);
        }
        self.push_numeric_rows_impl(column, entity, value, false)
    }

    fn push_numeric_rows_recurrent_off(
        &mut self,
        column: usize,
        entity: EntityId,
        value: Option<f64>,
    ) -> Result<()> {
        let (group, row) = self.location(entity)?;
        let schema = &self.rings[column];
        let transaction = self
            .transaction
            .as_mut()
            .ok_or_else(|| internal("state-table mutation requires a transaction"))?;
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        Self::push_numeric_recurrent_page(transaction, partition, schema, column, group, row, value)
            .map(|_| ())
    }

    fn push_numeric_recurrent_page<'a>(
        transaction: &mut Transaction,
        partition: &'a mut StatePartition,
        schema: &RingColumn,
        column: usize,
        group: KeyGroupId,
        row: usize,
        value: Option<f64>,
    ) -> Result<&'a [NumericAccumulator]> {
        let max_window = Self::numeric_max_window(schema)?;
        let slot = partition.rings[column][row]
            .as_ref()
            .expect("ring page")
            .id
            .slot;
        let page = Arc::get_mut(
            &mut partition.rings[column][row]
                .as_mut()
                .expect("ring page")
                .value,
        )
        .expect("checkpoint-off ring page is writable");
        let update = Self::push_numeric_recurrent_ring(transaction, page, group, slot, value)?;
        Self::update_numeric_windows(page, &schema.numeric_windows, max_window, update, value)?;
        Ok(page.numeric.as_deref().expect("numeric group"))
    }

    fn numeric_max_window(schema: &RingColumn) -> Result<usize> {
        match schema.retention {
            Retention::Rows(window) if schema.numeric => Ok(window),
            _ => Err(internal("numeric Rows group type mismatch")),
        }
    }

    fn push_numeric_recurrent_ring(
        transaction: &mut Transaction,
        page: &mut RingPage,
        group: KeyGroupId,
        slot: u32,
        value: Option<f64>,
    ) -> Result<NumericRingUpdate> {
        let transaction_id = transaction.id;
        let previous_head = page.ring.head;
        let previous_len = page.ring.len;
        let index = page.ring.position(previous_len);
        let previous = page.ring.slots[index];
        let stamp_index =
            u32::try_from(index).map_err(|_| internal("rolling ring slot index exceeds u32"))?;
        if page.slot_stamps[index] != transaction_id {
            let previous_stamp = page.slot_stamps[index];
            transaction.undo.push(Undo::Stamp {
                group,
                key: StampKey {
                    slot,
                    kind: StampKind::RingSlot,
                    sub: stamp_index,
                },
                previous: Some(previous_stamp),
            });
            transaction.undo.push(Undo::RingSlot {
                group,
                slot,
                index,
                value: previous,
            });
            page.slot_stamps[index] = transaction_id;
        }
        let evicted = (previous_len == page.ring.capacity()).then_some(previous);
        if evicted.is_some() {
            page.ring.head = (previous_head + 1) % page.ring.capacity();
        } else {
            page.ring.len += 1;
        }
        page.ring.slots[index] = RingValue::Float64(value);
        let evicted = match evicted {
            Some(RingValue::Float64(value)) => value,
            None => None,
            _ => unreachable!(),
        };
        Ok(NumericRingUpdate {
            previous_head,
            previous_len,
            evicted,
        })
    }

    fn update_numeric_windows(
        page: &mut RingPage,
        windows: &[usize],
        max_window: usize,
        update: NumericRingUpdate,
        value: Option<f64>,
    ) -> Result<()> {
        let numeric = page.numeric.as_mut().expect("numeric group");
        for (numeric, &window) in numeric.iter_mut().zip(windows) {
            let history = NumericHistory {
                ring: &page.ring,
                offset: page.ring.len.saturating_sub(window),
            };
            let leaving = Self::numeric_leaving(&page.ring, update, window, max_window);
            numeric
                .rows(leaving, value)
                .and_then(|()| numeric.finish_rows(window, history))
                .map_err(|error| {
                    internal(&format!("rolling numeric arithmetic failed: {error:?}"))
                })?;
        }
        Ok(())
    }

    fn numeric_leaving(
        ring: &Ring,
        update: NumericRingUpdate,
        window: usize,
        max_window: usize,
    ) -> Option<f64> {
        if update.previous_len < window {
            None
        } else if window == max_window {
            update.evicted
        } else {
            let offset = update.previous_len - window;
            let slot = (update.previous_head + offset) % ring.capacity();
            match ring.slots[slot] {
                RingValue::Float64(value) => value,
                _ => unreachable!(),
            }
        }
    }

    pub(crate) fn write_numeric_full_tick(
        &mut self,
        column: usize,
        order: &[EntityId],
        ids: &[EntityId],
        values: &Float64Array,
        nan_as_value: bool,
        mut emit: impl FnMut(usize, &[NumericAccumulator]) -> Result<()>,
    ) -> Result<bool> {
        if !self.full_tick_input_is_valid(order, ids, values) {
            return Ok(false);
        }
        if !self.full_tick_entities_are_valid(order) {
            return Ok(false);
        }
        self.write_numeric_first_tick(column, order, values, nan_as_value, &mut emit)?;
        self.write_numeric_recurrent_ticks(column, order.len(), ids, values, nan_as_value, emit)?;
        Ok(true)
    }

    fn full_tick_input_is_valid(
        &self,
        order: &[EntityId],
        ids: &[EntityId],
        values: &Float64Array,
    ) -> bool {
        self.full_tick_shape_is_valid(order, ids, values)
            && Self::full_tick_order_is_valid(order, ids)
    }

    fn full_tick_shape_is_valid(
        &self,
        order: &[EntityId],
        ids: &[EntityId],
        values: &Float64Array,
    ) -> bool {
        !self.checkpointing
            && !order.is_empty()
            && ids.len() >= order.len()
            && values.len() >= ids.len()
    }

    fn full_tick_order_is_valid(order: &[EntityId], ids: &[EntityId]) -> bool {
        !order
            .iter()
            .enumerate()
            .any(|(index, entity)| order[..index].contains(entity))
            && ids.chunks(order.len()).all(|tick| order.starts_with(tick))
    }

    fn full_tick_entities_are_valid(&self, order: &[EntityId]) -> bool {
        let Some(partition) = self.partitions.get(&KeyGroupId(0)) else {
            return false;
        };
        order.iter().all(|entity| {
            entity.key_group == KeyGroupId(0)
                && partition
                    .metadata
                    .value
                    .entities
                    .get(entity.row as usize)
                    .is_some_and(|stored| stored.index == entity.index)
        })
    }

    fn numeric_input_value(values: &Float64Array, row: usize, nan_as_value: bool) -> Option<f64> {
        let raw = values.value(row);
        (values.is_valid(row) && (nan_as_value || !raw.is_nan())).then_some(raw)
    }

    fn write_numeric_first_tick(
        &mut self,
        column: usize,
        order: &[EntityId],
        values: &Float64Array,
        nan_as_value: bool,
        emit: &mut impl FnMut(usize, &[NumericAccumulator]) -> Result<()>,
    ) -> Result<()> {
        for (row, entity) in order.iter().enumerate() {
            let value = Self::numeric_input_value(values, row, nan_as_value);
            self.push_numeric_rows(column, *entity, value)?;
            emit(row, self.numeric_accumulators(column, *entity)?)?;
        }
        Ok(())
    }

    fn write_numeric_recurrent_ticks(
        &mut self,
        column: usize,
        order_len: usize,
        ids: &[EntityId],
        values: &Float64Array,
        nan_as_value: bool,
        mut emit: impl FnMut(usize, &[NumericAccumulator]) -> Result<()>,
    ) -> Result<()> {
        let schema = &self.rings[column];
        let transaction = self
            .transaction
            .as_mut()
            .ok_or_else(|| internal("state-table mutation requires a transaction"))?;
        let partition = self
            .partitions
            .get_mut(&KeyGroupId(0))
            .expect("full-tick group was validated");
        for (offset, entity) in ids[order_len..].iter().enumerate() {
            let row = order_len + offset;
            let value = Self::numeric_input_value(values, row, nan_as_value);
            let accumulators = Self::push_numeric_recurrent_page(
                transaction,
                partition,
                schema,
                column,
                KeyGroupId(0),
                entity.row as usize,
                value,
            )?;
            emit(row, accumulators)?;
        }
        Ok(())
    }

    fn push_numeric_rows_impl(
        &mut self,
        column: usize,
        entity: EntityId,
        value: Option<f64>,
        first_touch: bool,
    ) -> Result<()> {
        let (group, row) = self.location(entity)?;
        let max_window = Self::numeric_max_window(&self.rings[column])?;
        let (previous_head, previous_len) = {
            let ring = &self.partitions[&group].rings[column][row]
                .as_ref()
                .expect("ring page")
                .value
                .ring;
            (ring.head, ring.len)
        };
        let slot = self.partitions[&group].rings[column][row]
            .as_ref()
            .expect("ring page")
            .id
            .slot;
        let previous = self.numeric_undo_value(group, column, row, slot, first_touch);
        let evicted = self
            .push_f64_impl(column, entity, value, first_touch)?
            .flatten();
        if let Some(value) = previous {
            self.push_undo(Undo::Numeric { group, slot, value });
        }
        let update = NumericRingUpdate {
            previous_head,
            previous_len,
            evicted,
        };
        let windows = &self.rings[column].numeric_windows;
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        let page = partition.rings[column][row].as_mut().expect("ring page");
        let page = Arc::get_mut(&mut page.value).expect("unfrozen numeric page");
        Self::update_numeric_windows(page, windows, max_window, update, value)
    }

    fn numeric_undo_value(
        &mut self,
        group: KeyGroupId,
        column: usize,
        row: usize,
        slot: u32,
        first_touch: bool,
    ) -> Option<Vec<NumericAccumulator>> {
        if first_touch && self.note_stamp(group, slot, StampKind::Numeric, 0) {
            Some(
                self.partitions[&group].rings[column][row]
                    .as_ref()
                    .expect("ring page")
                    .value
                    .numeric
                    .clone()
                    .expect("numeric group"),
            )
        } else {
            None
        }
    }

    pub(crate) fn numeric_accumulators(
        &self,
        column: usize,
        entity: EntityId,
    ) -> Result<&[NumericAccumulator]> {
        let (key_group, row) = self.location(entity)?;
        self.partitions[&key_group].rings[column]
            .get(row)
            .and_then(Option::as_ref)
            .and_then(|page| page.value.numeric.as_deref())
            .ok_or_else(|| internal("numeric groups are absent"))
    }

    fn numeric_accumulator(
        &self,
        column: usize,
        group: usize,
        entity: EntityId,
    ) -> Result<NumericAccumulator> {
        let (key_group, row) = self.location(entity)?;
        self.partitions[&key_group].rings[column]
            .get(row)
            .and_then(Option::as_ref)
            .and_then(|page| page.value.numeric.as_ref()?.get(group).copied())
            .ok_or_else(|| internal("numeric group is absent"))
    }

    pub(crate) fn numeric_mean(&self, column: usize, entity: EntityId) -> Result<Option<f64>> {
        self.numeric_mean_at(column, 0, entity)
    }

    pub(crate) fn numeric_mean_at(
        &self,
        column: usize,
        group: usize,
        entity: EntityId,
    ) -> Result<Option<f64>> {
        let numeric = self.numeric_accumulator(column, group, entity)?;
        Ok((numeric.valid_count > 0).then(|| numeric.mean()))
    }

    pub(crate) fn numeric_valid_count(&self, column: usize, entity: EntityId) -> Result<u64> {
        self.numeric_valid_count_at(column, 0, entity)
    }

    pub(crate) fn numeric_valid_count_at(
        &self,
        column: usize,
        group: usize,
        entity: EntityId,
    ) -> Result<u64> {
        Ok(self.numeric_accumulator(column, group, entity)?.valid_count)
    }

    pub(crate) fn numeric_sum(&self, column: usize, entity: EntityId) -> Result<f64> {
        self.numeric_sum_at(column, 0, entity)
    }

    pub(crate) fn numeric_sum_at(
        &self,
        column: usize,
        group: usize,
        entity: EntityId,
    ) -> Result<f64> {
        Ok(self.numeric_accumulator(column, group, entity)?.sum())
    }

    pub(crate) fn numeric_variance(
        &self,
        column: usize,
        entity: EntityId,
        ddof: u8,
    ) -> Result<f64> {
        self.numeric_variance_at(column, 0, entity, ddof)
    }

    pub(crate) fn numeric_variance_at(
        &self,
        column: usize,
        group: usize,
        entity: EntityId,
        ddof: u8,
    ) -> Result<f64> {
        Ok(self
            .numeric_accumulator(column, group, entity)?
            .variance(ddof))
    }

    pub(crate) fn numeric_stddev(&self, column: usize, entity: EntityId, ddof: u8) -> Result<f64> {
        self.numeric_stddev_at(column, 0, entity, ddof)
    }

    pub(crate) fn numeric_stddev_at(
        &self,
        column: usize,
        group: usize,
        entity: EntityId,
        ddof: u8,
    ) -> Result<f64> {
        Ok(self
            .numeric_accumulator(column, group, entity)?
            .stddev(ddof))
    }

    fn lag_value(
        &self,
        column: usize,
        entity: EntityId,
        periods: usize,
    ) -> Result<Option<RingValue>> {
        let (group, row) = self.location(entity)?;
        let ring = &self.partitions[&group].rings[column][row]
            .as_ref()
            .expect("ring page")
            .value
            .ring;
        if periods == 0 || periods > ring.len {
            return Ok(None);
        }
        Ok(Some(ring.slots[ring.position(ring.len - periods)]))
    }

    pub(crate) fn f64_lag(
        &self,
        column: usize,
        entity: EntityId,
        periods: usize,
    ) -> Result<Option<f64>> {
        match self.lag_value(column, entity, periods)? {
            Some(RingValue::Float64(value)) => Ok(value),
            None => Ok(None),
            _ => Err(internal("ring column type mismatch")),
        }
    }

    pub(crate) fn i64_lag(
        &self,
        column: usize,
        entity: EntityId,
        periods: usize,
    ) -> Result<Option<i64>> {
        match self.lag_value(column, entity, periods)? {
            Some(RingValue::Int64(value)) => Ok(value),
            None => Ok(None),
            _ => Err(internal("ring column type mismatch")),
        }
    }

    pub(crate) fn u64_lag(
        &self,
        column: usize,
        entity: EntityId,
        periods: usize,
    ) -> Result<Option<u64>> {
        match self.lag_value(column, entity, periods)? {
            Some(RingValue::UInt64(value)) => Ok(value),
            None => Ok(None),
            _ => Err(internal("ring column type mismatch")),
        }
    }

    pub(crate) fn f64_history(&self, column: usize, entity: EntityId) -> Result<Vec<Option<f64>>> {
        let (group, row) = self.location(entity)?;
        Ok(self.partitions[&group].rings[column][row]
            .as_ref()
            .expect("ring page")
            .value
            .ring
            .history()
            .into_iter()
            .map(|value| match value {
                RingValue::Float64(value) => value,
                _ => unreachable!(),
            })
            .collect())
    }

    #[cfg(test)]
    fn first_entity(&self) -> Option<EntityId> {
        self.ordered_entities().first().map(|(entity, _)| *entity)
    }

    #[cfg(test)]
    pub(crate) fn first_f64_history(&self, column: usize) -> Option<Vec<Option<f64>>> {
        let entity = self.first_entity()?;
        self.f64_history(column, entity).ok()
    }

    #[cfg(test)]
    pub(crate) fn first_f64_slot_address(&self, column: usize, offset: usize) -> Option<usize> {
        let entity = self.first_entity()?;
        let (group, row) = self.location(entity).ok()?;
        let ring = &self.partitions[&group].rings[column][row]
            .as_ref()?
            .value
            .ring;
        if offset >= ring.len {
            return None;
        }
        let position = ring.position(offset);
        Some(std::ptr::from_ref(&ring.slots[position]) as usize)
    }

    #[cfg(test)]
    pub(crate) fn first_u64(&self, column: usize) -> Option<u64> {
        let entity = self.first_entity()?;
        self.u64(column, entity).ok()
    }

    #[cfg(test)]
    pub(crate) fn ring_slot_addresses(&self, column: usize) -> Vec<usize> {
        let mut addresses = Vec::new();
        for (entity, _) in self.ordered_entities() {
            if let Some(page) =
                &self.partitions[&entity.key_group].rings[column][entity.row as usize]
            {
                addresses.extend(
                    page.value
                        .ring
                        .slots
                        .iter()
                        .map(|slot| std::ptr::from_ref(slot) as usize),
                );
            }
        }
        addresses
    }

    #[cfg(test)]
    pub(crate) fn all_f64_histories(&self, column: usize) -> Vec<Vec<Option<f64>>> {
        self.ordered_entities()
            .into_iter()
            .map(|(entity, _)| self.f64_history(column, entity).expect("entity history"))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn all_u64(&self, column: usize) -> Vec<u64> {
        self.ordered_entities()
            .into_iter()
            .map(|(entity, _)| self.u64(column, entity).expect("entity counter"))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn numeric_window_count(&self, column: usize) -> usize {
        self.rings[column].numeric_windows.len()
    }

    pub(crate) fn ring_logical_bytes(&self, column: usize) -> usize {
        let kind = self.rings[column].kind;
        self.partitions
            .values()
            .flat_map(|partition| &partition.rings[column])
            .flatten()
            .map(|page| page.value.ring.value_bytes(kind))
            .sum()
    }

    pub(crate) fn ring_page_id(&self, column: usize, entity: EntityId) -> Result<PageId> {
        let (group, row) = self.location(entity)?;
        self.partitions[&group].rings[column]
            .get(row)
            .and_then(Option::as_ref)
            .map(|page| page.id)
            .ok_or_else(|| internal("ring page is absent"))
    }

    pub(crate) fn keyed_page_count(&self, column: usize, group: KeyGroupId) -> usize {
        self.partitions[&group].keyed[column]
            .iter()
            .flatten()
            .count()
    }

    pub(crate) fn delete_ring(&mut self, column: usize, entity: EntityId) -> Result<PageId> {
        let (group, row) = self.location(entity)?;
        let slot = self.partitions[&group].rings[column]
            .get(row)
            .and_then(Option::as_ref)
            .ok_or_else(|| internal("ring page is absent"))?
            .id
            .slot;
        self.mark_dirty(group, slot)?;
        let (generation, allocator) = self
            .partitions
            .get_mut(&group)
            .expect("prepared group")
            .allocator
            .remove(slot)?;
        let page = self.partitions[&group].page(slot).expect("ring page");
        self.push_undo(Undo::PageRemove {
            group,
            page,
            allocator,
        });
        self.partitions
            .get_mut(&group)
            .expect("prepared group")
            .remove(slot)?;
        Ok(PageId {
            key_group: group,
            slot,
            generation,
        })
    }

    pub(crate) fn first_dirty_slots(&self, group: KeyGroupId) -> Vec<u32> {
        self.partitions[&group].dirty.first_dirty.clone()
    }

    pub(crate) fn dirty_list_capacity(&self, group: KeyGroupId) -> usize {
        self.partitions[&group].dirty.first_dirty.capacity()
    }

    pub(crate) fn checkpoint_reservation_bytes(&self) -> usize {
        self.active
            .as_ref()
            .map_or(0, |active| active.reservation_bytes)
    }

    pub(crate) fn append_values(&mut self, column: usize, values: ArrayRef) -> Result<Range<u32>> {
        self.append_values_in_group(KeyGroupId(0), column, values)
    }

    pub(crate) fn append_values_in_group(
        &mut self,
        group: KeyGroupId,
        column: usize,
        values: ArrayRef,
    ) -> Result<Range<u32>> {
        let plan = self.value_log_append_plan(group, column, &values)?;
        if plan.pack {
            self.append_packed_value_log(group, column, values, plan)?;
        } else {
            self.append_new_value_log_page(group, column, values, plan)?;
        }
        self.set_value_log_len(group, column, plan.end)?;
        Ok(plan.start..plan.end)
    }

    fn value_log_append_plan(
        &self,
        group: KeyGroupId,
        column: usize,
        values: &ArrayRef,
    ) -> Result<ValueLogAppendPlan> {
        self.transaction_id()?;
        let log = &self.partitions[&group].value_logs[column];
        if log
            .pages
            .iter()
            .flatten()
            .flat_map(|page| &page.value.chunks)
            .next()
            .is_some_and(|first| first.data_type() != values.data_type())
        {
            return Err(internal("rolling value-log type changed"));
        }
        let start = log.len;
        let len = u32::try_from(values.len())
            .map_err(|_| internal("rolling value-log append exceeded u32 handles"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| internal("rolling value-log exhausted u32 handles"))?;
        let bytes = values.get_array_memory_size();
        let pack = log
            .pages
            .last()
            .and_then(Option::as_ref)
            .is_some_and(|page| {
                page.value
                    .logical_bytes
                    .checked_add(bytes)
                    .is_some_and(|total| total <= VALUE_LOG_PAGE_BYTES)
            });
        Ok(ValueLogAppendPlan {
            start,
            end,
            len,
            bytes,
            pack,
            page_index: log.pages.len(),
        })
    }

    fn append_packed_value_log(
        &mut self,
        group: KeyGroupId,
        column: usize,
        values: ArrayRef,
        plan: ValueLogAppendPlan,
    ) -> Result<()> {
        let page_index = plan.page_index - 1;
        let slot = self.partitions[&group].value_logs[column].pages[page_index]
            .as_ref()
            .expect("value-log page")
            .id
            .slot;
        self.replace_page(group, slot, false)?;
        let page = &self.partitions[&group].value_logs[column].pages[page_index]
            .as_ref()
            .expect("value-log page")
            .value;
        self.push_undo(Undo::ValueLog {
            group,
            column,
            slot,
            chunks: page.chunks.len(),
            len: page.len,
            logical_bytes: page.logical_bytes,
            total_len: plan.start,
        });
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        let page = Arc::get_mut(
            &mut partition.value_logs[column].pages[page_index]
                .as_mut()
                .expect("value-log page")
                .value,
        )
        .expect("unfrozen value-log page");
        page.chunks.push(values);
        page.len = page
            .len
            .checked_add(plan.len)
            .ok_or_else(|| internal("value-log page length overflow"))?;
        page.logical_bytes = page
            .logical_bytes
            .checked_add(plan.bytes)
            .ok_or_else(|| internal("value-log byte accounting overflow"))?;
        Ok(())
    }

    fn append_new_value_log_page(
        &mut self,
        group: KeyGroupId,
        column: usize,
        values: ArrayRef,
        plan: ValueLogAppendPlan,
    ) -> Result<()> {
        let id = self.create_page_in_group(group, |id| PageRef::ValueLog {
            column,
            page_index: plan.page_index,
            page: Page::new(
                id,
                ValueLogPage {
                    chunks: vec![values],
                    start: plan.start,
                    len: plan.len,
                    logical_bytes: plan.bytes,
                },
            ),
        })?;
        self.push_undo(Undo::ValueLog {
            group,
            column,
            slot: id.slot,
            chunks: 0,
            len: 0,
            logical_bytes: 0,
            total_len: plan.start,
        });
        Ok(())
    }

    fn set_value_log_len(&mut self, group: KeyGroupId, column: usize, len: u32) -> Result<()> {
        let slot = self.partitions[&group].metadata.id.slot;
        self.replace_page(group, slot, false)?;
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        let metadata = Arc::get_mut(&mut partition.metadata.value).expect("unfrozen metadata");
        if metadata.value_log_lengths.len() <= column {
            metadata.value_log_lengths.resize(column + 1, 0);
        }
        metadata.value_log_lengths[column] = len;
        partition.value_logs[column].len = len;
        Ok(())
    }

    pub(crate) fn append_strings(
        &mut self,
        column: usize,
        values: StringArray,
    ) -> Result<Range<u32>> {
        self.append_values(column, Arc::new(values))
    }

    pub(crate) fn value_log_len(&self, column: usize) -> u32 {
        self.partitions[&KeyGroupId(0)].value_logs[column].len
    }

    pub(crate) fn strings(&self, column: usize) -> Result<StringArray> {
        self.strings_in_group(KeyGroupId(0), column)
    }

    pub(crate) fn strings_in_group(&self, group: KeyGroupId, column: usize) -> Result<StringArray> {
        let chunks = self.partitions[&group].value_logs[column]
            .pages
            .iter()
            .flatten()
            .flat_map(|page| &page.value.chunks)
            .map(AsRef::as_ref)
            .collect::<Vec<_>>();
        let concatenated = datafusion::arrow::compute::concat(&chunks)
            .map_err(|error| internal(&error.to_string()))?;
        concatenated
            .as_any()
            .downcast_ref::<StringArray>()
            .cloned()
            .ok_or_else(|| internal("value-log type mismatch"))
    }

    pub(crate) fn logical_rows(&self) -> usize {
        self.partitions
            .values()
            .flat_map(|partition| partition.rings.iter().flatten().flatten())
            .map(|page| page.value.ring.len)
            .sum()
    }

    pub(crate) fn logical_bytes(&self) -> usize {
        self.partitions
            .values()
            .map(StatePartition::logical_bytes)
            .sum()
    }

    fn validate_freeze(&self) -> Result<()> {
        if !self.checkpointing {
            return Err(internal("cannot freeze with checkpointing disabled"));
        }
        if self.transaction.is_some() {
            return Err(internal("freeze during an open state transaction"));
        }
        if self.active.is_some() {
            return Err(internal("an earlier state-table freeze is still active"));
        }
        Ok(())
    }

    fn freeze_partition(
        group: KeyGroupId,
        partition: &mut StatePartition,
        frozen: &mut BTreeSet<PageId>,
        reservation_bytes: &mut usize,
    ) -> Result<(PartitionFreeze, DirtySet)> {
        let dirty = std::mem::take(&mut partition.dirty);
        let dirty_slots = dirty.bits.iter().copied().collect::<Vec<_>>();
        let mut changes = Vec::with_capacity(dirty_slots.len());
        for slot in &dirty_slots {
            match partition.page(*slot) {
                Some(page) => {
                    frozen.insert(page.id());
                    *reservation_bytes = reservation_bytes
                        .checked_add(page.logical_bytes() + PAGE_DESCRIPTOR_BYTES)
                        .ok_or_else(|| internal("checkpoint reservation overflow"))?;
                    changes.push(FrozenChange::Write(page));
                }
                None => changes.push(FrozenChange::Delete(PageId {
                    key_group: group,
                    slot: *slot,
                    generation: partition.allocator.next_generations[slot] - 1,
                })),
            }
        }
        Ok((
            PartitionFreeze {
                id: group,
                metadata: partition.metadata.clone(),
                changes,
                allocator: partition.allocator.clone(),
                dirty_slots,
            },
            dirty,
        ))
    }

    pub(crate) fn freeze(&mut self, epoch: u64) -> Result<TableFreeze> {
        self.validate_freeze()?;
        let previous = self.durable.clone();
        let mut frozen = BTreeSet::new();
        let mut reservation_bytes = previous.retained_bytes;
        let mut partitions = Vec::with_capacity(self.partitions.len());
        let mut dirty = BTreeMap::new();
        for (group, partition) in &mut self.partitions {
            let (freeze, partition_dirty) =
                Self::freeze_partition(*group, partition, &mut frozen, &mut reservation_bytes)?;
            partitions.push(freeze);
            dirty.insert(*group, partition_dirty);
        }
        self.active = Some(ActiveFreeze {
            epoch,
            frozen,
            reservation_bytes,
            dirty,
        });
        Ok(TableFreeze {
            epoch,
            previous,
            partitions,
            reservation_bytes,
        })
    }

    pub(crate) fn settle_freeze(&mut self, epoch: u64, lineage: TableLineage) -> Result<()> {
        if self.active.as_ref().map(|active| active.epoch) != Some(epoch) || lineage.epoch != epoch
        {
            return Err(internal("state-table freeze settlement epoch mismatch"));
        }
        self.durable = Arc::new(lineage);
        self.active = None;
        Ok(())
    }

    pub(crate) fn abort_freeze(&mut self, epoch: u64) -> Result<()> {
        if self.active.as_ref().map(|active| active.epoch) != Some(epoch) {
            return Err(internal("state-table freeze abort epoch mismatch"));
        }
        let active = self.active.take().expect("active epoch checked");
        for (group, frozen) in active.dirty {
            let partition = self.partitions.get_mut(&group).expect("prepared group");
            let current = std::mem::take(&mut partition.dirty);
            partition.dirty = frozen;
            for slot in current.first_dirty {
                partition.dirty.mark(slot);
            }
        }
        Ok(())
    }

    fn current_lineage(&self) -> TableLineage {
        let mut lineage = TableLineage {
            epoch: self.durable.epoch,
            partitions: BTreeMap::new(),
            retained_bytes: 0,
        };
        for (group, partition) in &self.partitions {
            let pages = partition
                .locations
                .keys()
                .filter_map(|slot| partition.page(*slot).map(|page| (*slot, page)))
                .collect::<BTreeMap<_, _>>();
            lineage.retained_bytes += pages.values().map(PageRef::logical_bytes).sum::<usize>();
            lineage.partitions.insert(
                *group,
                PartitionLineage {
                    pages,
                    allocator: partition.allocator.clone(),
                },
            );
        }
        lineage
    }

    fn ordered_entities(&self) -> Vec<(EntityId, EntityKey)> {
        let mut entities =
            self.partitions
                .values()
                .flat_map(|partition| {
                    let group = partition.id;
                    partition.metadata.value.entities.iter().enumerate().map(
                        move |(row, stored)| {
                            (
                                EntityId {
                                    index: stored.index,
                                    key_group: group,
                                    row: u32::try_from(row)
                                        .expect("partition rows use checked u32 IDs"),
                                },
                                stored.key.clone(),
                            )
                        },
                    )
                })
                .collect::<Vec<_>>();
        entities.sort_by_key(|(entity, _)| entity.index);
        entities
    }

    #[cfg(test)]
    pub(crate) fn canonical_state_for_test(&self) -> String {
        let ordered = self.ordered_entities();
        let keys = ordered
            .iter()
            .map(|(_, key)| key.clone())
            .collect::<Vec<_>>();
        let keyed = self
            .keyed
            .iter()
            .enumerate()
            .map(|(column, schema)| {
                (
                    &schema.name,
                    schema.kind,
                    ordered
                        .iter()
                        .map(|(entity, _)| self.keyed_value(column, *entity))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let rings = self
            .rings
            .iter()
            .enumerate()
            .map(|(column, schema)| {
                let values = ordered
                    .iter()
                    .map(|(entity, _)| {
                        self.partitions[&entity.key_group].rings[column]
                            .get(entity.row as usize)
                            .and_then(Option::as_ref)
                            .map(|page| {
                                (
                                    page.value.ring.head,
                                    page.value.ring.len,
                                    &page.value.ring.slots,
                                )
                            })
                    })
                    .collect::<Vec<_>>();
                let numeric = ordered
                    .iter()
                    .map(|(entity, _)| {
                        self.partitions[&entity.key_group].rings[column]
                            .get(entity.row as usize)
                            .and_then(Option::as_ref)
                            .and_then(|page| page.value.numeric.clone())
                    })
                    .collect::<Vec<_>>();
                (&schema.name, schema.kind, schema.retention, values, numeric)
            })
            .collect::<Vec<_>>();
        let value_logs = self
            .partitions
            .iter()
            .map(|(group, partition)| (group, &partition.value_logs))
            .collect::<Vec<_>>();
        format!("{:?}", (keys, keyed, rings, value_logs))
    }

    pub(super) fn snapshot_for_layout7(&self) -> TableSnapshot {
        let logical_bytes = self.logical_bytes();
        let mut snapshot = self.clone().snapshot();
        snapshot.logical_bytes = logical_bytes;
        snapshot
    }

    pub(crate) fn snapshot(&mut self) -> TableSnapshot {
        assert!(
            self.transaction.is_none() && self.active.is_none(),
            "snapshots occur only between state transactions and freezes"
        );
        let current = self.current_lineage();
        let lineage = current.deep_copy();
        if self.checkpointing {
            self.durable = Arc::new(current);
        }
        let ordered = self.ordered_entities();
        let keys = ordered
            .iter()
            .map(|(_, key)| key.clone())
            .collect::<Vec<_>>();
        let keyed = self
            .keyed
            .iter()
            .enumerate()
            .map(|(column, schema)| KeyedSnapshot {
                name: schema.name.clone(),
                kind: schema.kind,
                values: keyed_array(
                    schema.kind,
                    &ordered
                        .iter()
                        .map(|(entity, _)| self.keyed_value(column, *entity))
                        .collect::<Vec<_>>(),
                ),
            })
            .collect();
        let rings = self
            .rings
            .iter()
            .enumerate()
            .map(|(column, schema)| self.ring_snapshot(column, schema, &ordered))
            .collect();
        let value_logs = self.value_log_snapshots();
        TableSnapshot {
            keys,
            keyed,
            rings,
            value_logs,
            logical_bytes: self.logical_bytes(),
            lineage,
            next_entity: self.next_entity,
        }
    }

    pub(super) fn v4_from_lineage(
        &self,
        keys: Vec<EntityKey>,
        keyed: Vec<KeyedSnapshot>,
        rings: Vec<RingSnapshot>,
        lineage: TableLineage,
        next_entity: u32,
    ) -> Result<TableSnapshot> {
        let snapshot = TableSnapshot {
            keys,
            keyed,
            rings,
            value_logs: Vec::new(),
            logical_bytes: lineage.retained_bytes,
            lineage,
            next_entity,
        };
        self.validate_inventory(&snapshot)?;
        Ok(snapshot)
    }

    pub(super) fn layout7_from_snapshot(
        &self,
        keys: Vec<EntityKey>,
        keyed: Vec<KeyedSnapshot>,
        rings: Vec<RingSnapshot>,
    ) -> Result<(TableSnapshot, usize)> {
        let next_entity = u32::try_from(keys.len())
            .map_err(|_| internal("layout-7 entity dictionary exceeds u32"))?;
        let snapshot = TableSnapshot {
            keys,
            keyed,
            rings,
            value_logs: Vec::new(),
            logical_bytes: 0,
            lineage: TableLineage::default(),
            next_entity,
        };
        self.validate_inventory(&snapshot)?;
        let (lineage, logical_bytes) = self.install_layout7_pages(&snapshot)?;
        Ok((
            TableSnapshot {
                logical_bytes,
                lineage,
                ..snapshot
            },
            logical_bytes,
        ))
    }

    fn prepare_layout7_restore(&self) -> Result<StateTable> {
        let groups = self.partitions.keys().copied().collect::<Vec<_>>();
        let mut restored = StateTable::with_key_groups(self.checkpointing, &groups)?;
        restored.keyed.clone_from(&self.keyed);
        restored.rings.clone_from(&self.rings);
        restored.value_logs.clone_from(&self.value_logs);
        for partition in restored.partitions.values_mut() {
            partition.keyed = vec![Vec::new(); restored.keyed.len()];
            partition.rings = vec![Vec::new(); restored.rings.len()];
            partition.value_logs = vec![ValueLog::default(); restored.value_logs.len()];
        }
        Ok(restored)
    }

    fn install_layout7_entities(restored: &mut StateTable, snapshot: &TableSnapshot) -> Result<()> {
        let keyed = snapshot
            .keyed
            .iter()
            .map(|column| {
                column
                    .values
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| internal("layout-7 keyed column type mismatch"))
            })
            .collect::<Result<Vec<_>>>()?;
        for (index, key) in snapshot.keys.iter().enumerate() {
            let entity = restored.resolve_or_append(key.clone())?;
            for (column, values) in keyed.iter().enumerate() {
                if !values.is_valid(index) {
                    return Err(internal("layout-7 keyed value is null"));
                }
                restored.set_u64(column, entity, values.value(index))?;
            }
            for (column, ring) in snapshot.rings.iter().enumerate() {
                restored.install_layout7_ring(column, entity, ring, index)?;
            }
        }
        Ok(())
    }

    fn clear_restore_tracking(restored: &mut StateTable) {
        for partition in restored.partitions.values_mut() {
            partition.dirty = DirtySet::default();
            partition.stamps.clear();
        }
    }

    fn install_layout7_pages(&self, snapshot: &TableSnapshot) -> Result<(TableLineage, usize)> {
        let mut restored = self.prepare_layout7_restore()?;
        restored.begin()?;
        Self::install_layout7_entities(&mut restored, snapshot)?;
        restored.commit()?;
        Self::clear_restore_tracking(&mut restored);
        let logical_bytes = restored.logical_bytes();
        Ok((restored.current_lineage(), logical_bytes))
    }

    fn install_layout7_ring(
        &mut self,
        column: usize,
        entity: EntityId,
        snapshot: &RingSnapshot,
        index: usize,
    ) -> Result<()> {
        let start = snapshot.offsets[index];
        let end = snapshot.offsets[index + 1];
        let capacity = snapshot.capacities[index];
        let numeric = snapshot
            .numeric
            .as_ref()
            .and_then(|values| values.get(index))
            .cloned();
        let values = (start..end)
            .map(|source| match snapshot.kind {
                RingKind::Float64 => snapshot
                    .values
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .map(|history| {
                        RingValue::Float64(history.is_valid(source).then(|| history.value(source)))
                    })
                    .ok_or_else(|| internal("layout-7 history type mismatch")),
                RingKind::Int64 => snapshot
                    .values
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::Int64Array>()
                    .map(|history| {
                        RingValue::Int64(history.is_valid(source).then(|| history.value(source)))
                    })
                    .ok_or_else(|| internal("layout-7 history type mismatch")),
                RingKind::UInt64 => snapshot
                    .values
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .map(|history| {
                        RingValue::UInt64(history.is_valid(source).then(|| history.value(source)))
                    })
                    .ok_or_else(|| internal("layout-7 history type mismatch")),
                RingKind::PairFloat64 => Err(internal("layout-7 history type mismatch")),
            })
            .collect::<Result<Vec<_>>>()?;
        let (group, row) = self.location(entity)?;
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        let page = partition.rings[column][row].as_mut().expect("ring page");
        let page = Arc::get_mut(&mut page.value).expect("new layout-7 ring page");
        let kind = page.kind;
        page.ring.slots.resize(capacity, kind.empty());
        page.slot_stamps = vec![0; capacity];
        page.ring.head = 0;
        page.ring.len = end - start;
        for (offset, value) in values.into_iter().enumerate() {
            let target = page.ring.position(offset);
            page.ring.slots[target] = value;
        }
        page.numeric = numeric;
        Ok(())
    }

    fn keyed_value(&self, column: usize, entity: EntityId) -> KeyedValue {
        let row = entity.row as usize;
        self.partitions[&entity.key_group].keyed[column][row / KEYED_PAGE_ROWS]
            .as_ref()
            .expect("keyed page")
            .value
            .values[row % KEYED_PAGE_ROWS]
    }

    fn ring_snapshot(
        &self,
        column: usize,
        schema: &RingColumn,
        ordered: &[(EntityId, EntityKey)],
    ) -> RingSnapshot {
        let mut values = Vec::new();
        let mut offsets = Vec::with_capacity(ordered.len() + 1);
        let mut capacities = Vec::with_capacity(ordered.len());
        let mut numeric = schema.numeric.then(Vec::new);
        offsets.push(0);
        for (entity, _) in ordered {
            let page = self.partitions[&entity.key_group].rings[column]
                .get(entity.row as usize)
                .and_then(Option::as_ref);
            if let Some(page) = page {
                values.extend(page.value.ring.history());
                capacities.push(page.value.ring.capacity());
                if let Some(numeric) = &mut numeric {
                    numeric.push(page.value.numeric.clone().expect("numeric group"));
                }
            } else {
                capacities.push(0);
                if let Some(numeric) = &mut numeric {
                    numeric.push(
                        schema
                            .numeric_windows
                            .iter()
                            .map(|_| NumericAccumulator::default())
                            .collect(),
                    );
                }
            }
            offsets.push(values.len());
        }
        RingSnapshot {
            name: schema.name.clone(),
            kind: schema.kind,
            retention: schema.retention,
            values: ring_array(schema.kind, &values),
            offsets,
            capacities,
            numeric,
        }
    }

    fn value_log_snapshots(&self) -> Vec<ValueLogSnapshot> {
        let mut snapshots = Vec::new();
        for group in self.partitions.keys() {
            for (column, name) in self.value_logs.iter().enumerate() {
                snapshots.push(self.value_log_snapshot(*group, column, name));
            }
        }
        snapshots
    }

    fn value_log_snapshot(&self, group: KeyGroupId, column: usize, name: &str) -> ValueLogSnapshot {
        let log = &self.partitions[&group].value_logs[column];
        let mut expected_start = 0;
        for page in log.pages.iter().flatten() {
            debug_assert_eq!(page.value.start, expected_start);
            expected_start += page.value.len;
        }
        debug_assert_eq!(expected_start, log.len);
        ValueLogSnapshot {
            group,
            name: name.to_owned(),
            chunks: log
                .pages
                .iter()
                .flatten()
                .flat_map(|page| page.value.chunks.clone())
                .collect(),
            len: log.len,
        }
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "restore consumes the snapshot into a freshly validated table lineage"
    )]
    pub(crate) fn restore(&mut self, snapshot: TableSnapshot) -> Result<()> {
        if self.transaction.is_some() || self.active.is_some() {
            return Err(internal(
                "cannot restore during a state transaction or freeze",
            ));
        }
        self.validate_inventory(&snapshot)?;
        let groups = self.partitions.keys().copied().collect::<Vec<_>>();
        let mut restored = StateTable::with_key_groups(self.checkpointing, &groups)?;
        restored.keyed.clone_from(&self.keyed);
        restored.rings.clone_from(&self.rings);
        restored.value_logs.clone_from(&self.value_logs);
        for partition in restored.partitions.values_mut() {
            partition.keyed = vec![Vec::new(); restored.keyed.len()];
            partition.rings = vec![Vec::new(); restored.rings.len()];
            partition.value_logs = vec![ValueLog::default(); restored.value_logs.len()];
        }
        restored.install_lineage(&snapshot)?;
        restored.next_transaction = self.next_transaction;
        *self = restored;
        Ok(())
    }

    fn validate_snapshot_page(&self, page: &PageRef) -> Result<()> {
        match page {
            PageRef::Metadata(_) => Ok(()),
            PageRef::Keyed { column, page, .. } => {
                if self
                    .keyed
                    .get(*column)
                    .is_some_and(|schema| schema.kind == page.value.kind)
                {
                    Ok(())
                } else {
                    Err(internal("snapshot keyed page schema mismatch"))
                }
            }
            PageRef::Ring { column, page, .. } => {
                if self.rings.get(*column).is_some_and(|schema| {
                    schema.kind == page.value.kind && schema.numeric == page.value.numeric.is_some()
                }) {
                    Ok(())
                } else {
                    Err(internal("snapshot ring page schema mismatch"))
                }
            }
            PageRef::ValueLog { column, .. } => {
                if *column < self.value_logs.len() {
                    Ok(())
                } else {
                    Err(internal("snapshot value-log page schema mismatch"))
                }
            }
        }
    }

    fn validate_lineage_groups(&self, snapshot: &TableSnapshot) -> Result<()> {
        if self.partitions.len() != snapshot.lineage.partitions.len()
            || self
                .partitions
                .keys()
                .ne(snapshot.lineage.partitions.keys())
        {
            return Err(internal("snapshot key-group inventory mismatch"));
        }
        Ok(())
    }

    fn install_lineage(&mut self, snapshot: &TableSnapshot) -> Result<()> {
        self.validate_lineage_groups(snapshot)?;
        let mut entities = HashMap::new();
        for (group, lineage) in &snapshot.lineage.partitions {
            let (partition, entries) = self.install_partition(*group, lineage)?;
            for (key, entity) in entries {
                Self::install_entity(&mut entities, key, entity)?;
            }
            self.partitions.insert(*group, partition);
        }
        Self::validate_entity_census(&entities, snapshot.next_entity)?;
        self.entities = entities;
        self.next_entity = snapshot.next_entity;
        if self.checkpointing {
            self.durable = Arc::new(snapshot.lineage.clone());
        }
        Ok(())
    }

    fn install_entity(
        entities: &mut HashMap<Vec<u8>, EntityId>,
        key: Vec<u8>,
        entity: EntityId,
    ) -> Result<()> {
        if entities.insert(key, entity).is_some() {
            return Err(internal("snapshot contains a duplicate entity key"));
        }
        Ok(())
    }

    fn validate_entity_census(
        entities: &HashMap<Vec<u8>, EntityId>,
        next_entity: u32,
    ) -> Result<()> {
        let indices = entities
            .values()
            .map(|entity| entity.index)
            .collect::<BTreeSet<_>>();
        if entities.len() != next_entity as usize || indices.iter().copied().ne(0..next_entity) {
            return Err(internal("snapshot entity dictionary is not dense"));
        }
        Ok(())
    }

    fn install_snapshot_page(
        &self,
        partition: &mut StatePartition,
        group: KeyGroupId,
        lineage: &PartitionLineage,
        slot: u32,
        page: &PageRef,
        metadata_seen: &mut bool,
    ) -> Result<()> {
        self.validate_snapshot_page(page)?;
        let next_generation = lineage
            .allocator
            .next_generations
            .get(&slot)
            .copied()
            .unwrap_or(0);
        Self::validate_snapshot_page_identity(page, group, slot, next_generation)?;
        Self::note_snapshot_metadata(page, slot, metadata_seen)?;
        partition.insert(page.clone());
        Ok(())
    }

    fn validate_snapshot_page_identity(
        page: &PageRef,
        group: KeyGroupId,
        slot: u32,
        next_generation: u64,
    ) -> Result<()> {
        if page.id().key_group != group
            || page.id().slot != slot
            || page.id().generation >= next_generation
        {
            return Err(internal("snapshot page identity mismatch"));
        }
        Ok(())
    }

    fn note_snapshot_metadata(page: &PageRef, slot: u32, metadata_seen: &mut bool) -> Result<()> {
        if !matches!(page, PageRef::Metadata(_)) {
            return Ok(());
        }
        if *metadata_seen || slot != 0 {
            return Err(internal("snapshot partition metadata is not unique"));
        }
        *metadata_seen = true;
        Ok(())
    }

    fn restore_value_log_lengths(partition: &mut StatePartition) {
        let value_log_lengths = partition.metadata.value.value_log_lengths.clone();
        for (column, log) in partition.value_logs.iter_mut().enumerate() {
            log.len = value_log_lengths.get(column).copied().unwrap_or(0);
        }
    }

    fn collect_partition_entities(
        partition: &StatePartition,
        group: KeyGroupId,
    ) -> Result<Vec<(Vec<u8>, EntityId)>> {
        partition
            .metadata
            .value
            .entities
            .iter()
            .enumerate()
            .map(|(row, stored)| {
                let row = u32::try_from(row)
                    .map_err(|_| internal("snapshot partition row index exceeds u32"))?;
                Ok((
                    stored.key.bytes.clone(),
                    EntityId {
                        index: stored.index,
                        key_group: group,
                        row,
                    },
                ))
            })
            .collect()
    }

    fn install_partition(
        &self,
        group: KeyGroupId,
        lineage: &PartitionLineage,
    ) -> Result<RestoredPartition> {
        let mut partition = StatePartition::new(group);
        partition.keyed = vec![Vec::new(); self.keyed.len()];
        partition.rings = vec![Vec::new(); self.rings.len()];
        partition.value_logs = vec![ValueLog::default(); self.value_logs.len()];
        let mut metadata_seen = false;
        for (slot, page) in &lineage.pages {
            self.install_snapshot_page(
                &mut partition,
                group,
                lineage,
                *slot,
                page,
                &mut metadata_seen,
            )?;
        }
        if !metadata_seen {
            return Err(internal("snapshot partition metadata is absent"));
        }
        partition.allocator = lineage.allocator.clone();
        Self::restore_value_log_lengths(&mut partition);
        let entries = Self::collect_partition_entities(&partition, group)?;
        Ok((partition, entries))
    }

    fn validate_inventory(&self, snapshot: &TableSnapshot) -> Result<()> {
        self.validate_column_schema(snapshot)?;
        self.validate_keyed_snapshots(snapshot)?;
        self.validate_ring_snapshots(snapshot)?;
        self.validate_value_log_snapshots(snapshot)
    }

    fn validate_column_schema(&self, snapshot: &TableSnapshot) -> Result<()> {
        self.validate_schema_counts(snapshot)?;
        self.validate_keyed_column_schema(snapshot)?;
        self.validate_ring_column_schema(snapshot)
    }

    fn validate_schema_counts(&self, snapshot: &TableSnapshot) -> Result<()> {
        if self.keyed.len() != snapshot.keyed.len()
            || self.rings.len() != snapshot.rings.len()
            || self.value_logs.len() * self.partitions.len() != snapshot.value_logs.len()
        {
            return Err(internal("rolling state-table column inventory mismatch"));
        }
        Ok(())
    }

    fn validate_keyed_column_schema(&self, snapshot: &TableSnapshot) -> Result<()> {
        if !self
            .keyed
            .iter()
            .zip(&snapshot.keyed)
            .all(|(left, right)| left.name == right.name && left.kind == right.kind)
        {
            return Err(internal("rolling state-table column inventory mismatch"));
        }
        Ok(())
    }

    fn validate_ring_column_schema(&self, snapshot: &TableSnapshot) -> Result<()> {
        if !self.rings.iter().zip(&snapshot.rings).all(|(left, right)| {
            left.name == right.name
                && left.kind == right.kind
                && left.retention == right.retention
                && left.numeric == right.numeric.is_some()
        }) {
            return Err(internal("rolling state-table column inventory mismatch"));
        }
        Ok(())
    }

    fn validate_keyed_snapshots(&self, snapshot: &TableSnapshot) -> Result<()> {
        for (schema, stored) in self.keyed.iter().zip(&snapshot.keyed) {
            if stored.values.len() != snapshot.keys.len()
                || stored.values.data_type() != keyed_array(schema.kind, &[]).data_type()
            {
                return Err(internal("rolling keyed snapshot shape mismatch"));
            }
        }
        Ok(())
    }

    fn validate_ring_snapshots(&self, snapshot: &TableSnapshot) -> Result<()> {
        for (column, stored) in self.rings.iter().zip(&snapshot.rings) {
            Self::validate_ring_snapshot(column, stored, snapshot.keys.len())?;
        }
        Ok(())
    }

    fn validate_ring_snapshot(
        column: &RingColumn,
        snapshot: &RingSnapshot,
        snapshot_entities: usize,
    ) -> Result<()> {
        let entities = snapshot.capacities.len();
        if entities != snapshot_entities || !Self::ring_numeric_matches(column, snapshot, entities)
        {
            return Err(internal("rolling ring snapshot violates its invariants"));
        }
        if !Self::ring_offsets_are_valid(snapshot, entities) {
            return Err(internal("rolling ring snapshot violates its invariants"));
        }
        if !Self::ring_capacities_are_valid(snapshot) {
            return Err(internal("rolling ring snapshot violates its invariants"));
        }
        Ok(())
    }

    fn ring_numeric_matches(column: &RingColumn, snapshot: &RingSnapshot, entities: usize) -> bool {
        let populated = snapshot
            .numeric
            .as_ref()
            .is_none_or(|numeric| numeric.len() == entities);
        populated && (!column.numeric || snapshot.numeric.is_some())
    }

    fn ring_offsets_are_valid(snapshot: &RingSnapshot, entities: usize) -> bool {
        snapshot.offsets.len() == entities + 1
            && snapshot.offsets.first() == Some(&0)
            && snapshot.offsets.last() == Some(&snapshot.values.len())
            && snapshot
                .offsets
                .windows(2)
                .all(|offset| offset[0] <= offset[1])
    }

    fn ring_capacities_are_valid(snapshot: &RingSnapshot) -> bool {
        snapshot
            .capacities
            .iter()
            .zip(snapshot.offsets.windows(2))
            .all(|(capacity, offset)| offset[1] - offset[0] <= *capacity)
    }

    fn validate_value_log_snapshots(&self, snapshot: &TableSnapshot) -> Result<()> {
        for stored in &snapshot.value_logs {
            let len = stored
                .chunks
                .iter()
                .try_fold(0_u32, |total, chunk| {
                    total.checked_add(u32::try_from(chunk.len()).ok()?)
                })
                .ok_or_else(|| internal("value-log snapshot length overflow"))?;
            if len != stored.len
                || !self.partitions.contains_key(&stored.group)
                || !self.value_logs.contains(&stored.name)
            {
                return Err(internal("value-log snapshot inventory mismatch"));
            }
        }
        Ok(())
    }

    fn apply_undo(&mut self, undo: Undo) -> Result<()> {
        match undo {
            undo @ (Undo::Entity { .. }
            | Undo::Dirty { .. }
            | Undo::Stamp { .. }
            | Undo::PageCreate { .. }
            | Undo::PageReplace { .. }
            | Undo::PageRemove { .. }) => self.apply_page_undo(undo),
            undo => self.apply_value_undo(undo),
        }
    }

    fn apply_page_undo(&mut self, undo: Undo) -> Result<()> {
        match undo {
            Undo::Entity {
                group,
                key,
                next_entity,
            } => self.undo_entity(group, key, next_entity),
            Undo::Dirty {
                group,
                slot,
                was_dirty,
                first_dirty_len,
            } => {
                self.undo_dirty(group, slot, was_dirty, first_dirty_len);
                Ok(())
            }
            Undo::Stamp {
                group,
                key,
                previous,
            } => self.undo_stamp(group, key, previous),
            Undo::PageCreate {
                group,
                slot,
                allocator,
            } => self.undo_page_create(group, slot, allocator),
            Undo::PageReplace {
                group,
                page,
                allocator,
            }
            | Undo::PageRemove {
                group,
                page,
                allocator,
            } => {
                self.undo_page_replace(group, page, allocator);
                Ok(())
            }
            _ => unreachable!(),
        }
    }

    fn undo_dirty(
        &mut self,
        group: KeyGroupId,
        slot: u32,
        was_dirty: bool,
        first_dirty_len: usize,
    ) {
        let dirty = &mut self
            .partitions
            .get_mut(&group)
            .expect("prepared group")
            .dirty;
        if was_dirty {
            dirty.mark(slot);
        } else {
            dirty.bits.remove(&slot);
        }
        dirty.first_dirty.truncate(first_dirty_len);
    }

    fn undo_stamp(
        &mut self,
        group: KeyGroupId,
        key: StampKey,
        previous: Option<u64>,
    ) -> Result<()> {
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        if key.kind == StampKind::RingSlot {
            let PageLocation::Ring { column, row } = partition.locations[&key.slot] else {
                return Err(internal("ring stamp undo page kind mismatch"));
            };
            let index =
                usize::try_from(key.sub).map_err(|_| internal("ring stamp index exceeds usize"))?;
            let page = &mut partition.rings[column][row]
                .as_mut()
                .expect("ring page")
                .value;
            Arc::get_mut(page)
                .expect("unfrozen ring stamp undo")
                .slot_stamps[index] = previous.unwrap_or(0);
        } else if let Some(previous) = previous {
            partition.stamps.insert(key, previous);
        } else {
            partition.stamps.remove(&key);
        }
        Ok(())
    }

    fn undo_page_create(
        &mut self,
        group: KeyGroupId,
        slot: u32,
        allocator: AllocatorUndo,
    ) -> Result<()> {
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        partition.remove(slot)?;
        partition.allocator.restore(slot, allocator);
        Ok(())
    }

    fn undo_page_replace(&mut self, group: KeyGroupId, page: PageRef, allocator: AllocatorUndo) {
        let slot = page.id().slot;
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        partition.insert(page);
        partition.allocator.restore(slot, allocator);
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "value undo consumes one owned transaction record"
    )]
    fn apply_value_undo(&mut self, undo: Undo) -> Result<()> {
        match undo {
            Undo::Keyed {
                group,
                slot,
                values,
                capacity,
            } => {
                let partition = self.partitions.get_mut(&group).expect("prepared group");
                let location = partition.locations[&slot];
                let PageLocation::Keyed { column, page_index } = location else {
                    return Err(internal("keyed undo page kind mismatch"));
                };
                let page = Arc::get_mut(
                    &mut partition.keyed[column][page_index]
                        .as_mut()
                        .expect("keyed page")
                        .value,
                )
                .expect("unfrozen keyed undo");
                let mut restored = Vec::with_capacity(capacity);
                restored.extend_from_slice(&values);
                page.values = restored;
                Ok(())
            }
            Undo::RingMetadata {
                group,
                slot,
                head,
                len,
                capacity,
            } => self.undo_ring_metadata(group, slot, head, len, capacity),
            Undo::RingSlot {
                group,
                slot,
                index,
                value,
            } => self.undo_ring_slot(group, slot, index, value),
            Undo::Numeric { group, slot, value } => self.undo_numeric(group, slot, value),
            Undo::ValueLog {
                group,
                column,
                slot,
                chunks,
                len,
                logical_bytes,
                total_len,
            } => self.undo_value_log(
                group,
                column,
                slot,
                &ValueLogUndo {
                    chunks,
                    len,
                    logical_bytes,
                    total_len,
                },
            ),
            _ => unreachable!(),
        }
    }

    fn undo_keyed(
        &mut self,
        group: KeyGroupId,
        slot: u32,
        row: usize,
        value: KeyedValue,
    ) -> Result<()> {
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        let location = partition.locations[&slot];
        let PageLocation::Keyed { column, page_index } = location else {
            return Err(internal("keyed undo page kind mismatch"));
        };
        Arc::get_mut(
            &mut partition.keyed[column][page_index]
                .as_mut()
                .expect("keyed page")
                .value,
        )
        .expect("unfrozen keyed undo")
        .values[row] = value;
        Ok(())
    }

    fn undo_ring_metadata(
        &mut self,
        group: KeyGroupId,
        slot: u32,
        head: usize,
        len: usize,
        capacity: usize,
    ) -> Result<()> {
        let kind = self.ring_kind_for_slot(group, slot)?;
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        let PageLocation::Ring { column, row } = partition.locations[&slot] else {
            return Err(internal("ring undo page kind mismatch"));
        };
        let page = Arc::get_mut(
            &mut partition.rings[column][row]
                .as_mut()
                .expect("ring page")
                .value,
        )
        .expect("unfrozen ring undo");
        page.ring.slots.resize(capacity, kind.empty());
        page.slot_stamps.resize(capacity, 0);
        page.ring.head = head;
        page.ring.len = len;
        Ok(())
    }

    fn undo_ring_slot(
        &mut self,
        group: KeyGroupId,
        slot: u32,
        index: usize,
        value: RingValue,
    ) -> Result<()> {
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        let PageLocation::Ring { column, row } = partition.locations[&slot] else {
            return Err(internal("ring undo page kind mismatch"));
        };
        Arc::get_mut(
            &mut partition.rings[column][row]
                .as_mut()
                .expect("ring page")
                .value,
        )
        .expect("unfrozen ring undo")
        .ring
        .slots[index] = value;
        Ok(())
    }

    fn undo_numeric(
        &mut self,
        group: KeyGroupId,
        slot: u32,
        value: Vec<NumericAccumulator>,
    ) -> Result<()> {
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        let PageLocation::Ring { column, row } = partition.locations[&slot] else {
            return Err(internal("numeric undo page kind mismatch"));
        };
        Arc::get_mut(
            &mut partition.rings[column][row]
                .as_mut()
                .expect("ring page")
                .value,
        )
        .expect("unfrozen numeric undo")
        .numeric = Some(value);
        Ok(())
    }

    fn undo_value_log(
        &mut self,
        group: KeyGroupId,
        column: usize,
        slot: u32,
        undo: &ValueLogUndo,
    ) -> Result<()> {
        let ValueLogUndo {
            chunks,
            len,
            logical_bytes,
            total_len,
        } = *undo;
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        partition.value_logs[column].len = total_len;
        let PageLocation::ValueLog { page_index, .. } = partition.locations[&slot] else {
            return Err(internal("value-log undo page kind mismatch"));
        };
        let metadata =
            Arc::get_mut(&mut partition.metadata.value).expect("unfrozen value-log metadata undo");
        if metadata.value_log_lengths.len() <= column {
            metadata.value_log_lengths.resize(column + 1, 0);
        }
        metadata.value_log_lengths[column] = total_len;
        let page = Arc::get_mut(
            &mut partition.value_logs[column].pages[page_index]
                .as_mut()
                .expect("value-log page")
                .value,
        )
        .expect("unfrozen value-log undo");
        page.chunks.truncate(chunks);
        page.len = len;
        page.logical_bytes = logical_bytes;
        Ok(())
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "entity undo consumes the owned key removed from the dictionary"
    )]
    fn undo_entity(&mut self, group: KeyGroupId, key: EntityKey, next_entity: u32) -> Result<()> {
        self.entities.remove(key.bytes.as_slice());
        self.next_entity = next_entity;
        let partition = self.partitions.get_mut(&group).expect("prepared group");
        let metadata = Arc::get_mut(&mut partition.metadata.value).expect("unfrozen metadata undo");
        let removed = metadata
            .entities
            .pop()
            .ok_or_else(|| internal("rolling entity undo underflowed"))?;
        metadata.dictionary_bytes = metadata
            .dictionary_bytes
            .checked_sub(removed.key.bytes.len())
            .ok_or_else(|| internal("dictionary byte accounting underflow"))?;
        let row = metadata.entities.len();
        for column in 0..self.keyed.len() {
            let page_index = row / KEYED_PAGE_ROWS;
            Arc::get_mut(
                &mut partition.keyed[column][page_index]
                    .as_mut()
                    .expect("keyed page")
                    .value,
            )
            .expect("unfrozen keyed entity undo")
            .values
            .pop();
        }
        Ok(())
    }

    fn ring_kind_for_slot(&self, group: KeyGroupId, slot: u32) -> Result<RingKind> {
        let PageLocation::Ring { column, .. } = self.partitions[&group].locations[&slot] else {
            return Err(internal("ring undo page kind mismatch"));
        };
        Ok(self.rings[column].kind)
    }
}
