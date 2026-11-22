use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::Cursor,
    sync::Arc,
};

use datafusion::arrow::{
    array::{Array, ArrayRef, BinaryArray, BooleanArray, Float64Array, Int64Array, UInt64Array},
    datatypes::{DataType, Field},
    ipc::reader::FileReader,
    record_batch::RecordBatch,
};
use serde::{Deserialize, Serialize};

use super::{
    StateTable,
    core::{
        FrozenChange, KeyedSnapshot, PageRef, RingSnapshot, TableFreeze, TableLineage,
        TableSnapshot,
    },
    keyed_array, ring_array,
};
use crate::{
    Result,
    accumulate::{NumericAccumulator, NumericAccumulatorLayout7},
    json::parse_json_value,
    state::v4::{
        ArrowPayload, NewObject, ObjectKind, ObjectPayload, OperatorV4, V4Handle, V4Manifest,
        V4Store, format_error, manifest::PartitionV4, manifest_fingerprint, mismatch,
    },
};

const MAX_INDEX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::state) struct V4PageId {
    pub(in crate::state) key_group: u32,
    pub(in crate::state) slot: u32,
    pub(in crate::state) generation: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::state) struct LogicalRange {
    pub(in crate::state) start: u64,
    pub(in crate::state) end: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::state) struct PageEntry {
    pub(in crate::state) page_id: V4PageId,
    pub(in crate::state) column: String,
    pub(in crate::state) kind: String,
    pub(in crate::state) logical_range: LogicalRange,
    pub(in crate::state) byte_len: u64,
    pub(in crate::state) sha256: String,
    pub(in crate::state) handle: V4Handle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::state) struct PartitionIndex {
    pub(in crate::state) format_version: u32,
    pub(in crate::state) kind: String,
    pub(in crate::state) key_group: u32,
    pub(in crate::state) pages: Vec<PageEntry>,
    pub(in crate::state) next_generations: BTreeMap<u32, u64>,
    pub(in crate::state) free_slots: Vec<u32>,
}

impl PartitionIndex {
    pub(in crate::state) fn from_bytes(bytes: &[u8], epoch: u64) -> Result<Self> {
        if bytes.len() > MAX_INDEX_BYTES {
            return Err(format_error(
                "checkpoint-v4 partition index exceeds 64 MiB".into(),
            ));
        }
        let value = parse_json_value(bytes, "checkpoint-v4 partition index")?;
        let index: Self = serde_json::from_value(value)
            .map_err(|error| format_error(format!("partition index schema mismatch: {error}")))?;
        index.validate(epoch)?;
        Ok(index)
    }

    pub(in crate::state) fn validate(&self, epoch: u64) -> Result<()> {
        self.validate_identity()?;
        self.validate_page_order()?;
        self.validate_free_slots()?;
        for page in &self.pages {
            self.validate_page(page, epoch)?;
        }
        Ok(())
    }

    fn validate_identity(&self) -> Result<()> {
        if self.format_version != 4 || self.kind != "partition_index" {
            return Err(mismatch("partition index identity mismatch".into()));
        }
        Ok(())
    }

    fn validate_page_order(&self) -> Result<()> {
        if self
            .pages
            .windows(2)
            .any(|pair| pair[0].page_id >= pair[1].page_id)
        {
            return Err(mismatch(
                "page entries are not strictly ordered and unique".into(),
            ));
        }
        Ok(())
    }

    fn validate_free_slots(&self) -> Result<()> {
        let free = self.free_slots.iter().copied().collect::<BTreeSet<_>>();
        if free.len() != self.free_slots.len()
            || self.free_slots.windows(2).any(|pair| pair[0] >= pair[1])
            || free
                .iter()
                .any(|slot| !self.next_generations.contains_key(slot))
            || self
                .pages
                .iter()
                .any(|page| free.contains(&page.page_id.slot))
        {
            return Err(mismatch(
                "partition free-slot metadata is inconsistent".into(),
            ));
        }
        Ok(())
    }

    fn validate_page(&self, page: &PageEntry, epoch: u64) -> Result<()> {
        if page.page_id.key_group != self.key_group {
            return Err(mismatch("page key group differs from its partition".into()));
        }
        if page.byte_len != page.handle.byte_len || page.sha256 != page.handle.sha256 {
            return Err(mismatch("page digest differs from handle".into()));
        }
        if page.handle.epoch > epoch {
            return Err(mismatch("page handle epoch or length mismatch".into()));
        }
        let object_id = format!(
            "{}/{}/{}",
            page.page_id.key_group, page.page_id.slot, page.page_id.generation
        );
        if page.handle.object_id != object_id {
            return Err(mismatch("page identity differs from handle".into()));
        }
        page.handle.validate(epoch)?;
        let next = self
            .next_generations
            .get(&page.page_id.slot)
            .copied()
            .unwrap_or(0);
        if page.page_id.generation >= next {
            return Err(mismatch("page generation metadata regressed".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(in crate::state) struct V4PersistedTable {
    pub(in crate::state) lineage: TableLineage,
    pub(in crate::state) handles: BTreeMap<super::core::PageId, V4Handle>,
    pub(in crate::state) roots: BTreeMap<super::core::KeyGroupId, V4Handle>,
    pub(in crate::state) manifest_path: std::path::PathBuf,
}

pub(in crate::state) struct V4TableCapture {
    freeze: TableFreeze,
    previous: BTreeMap<super::core::PageId, V4Handle>,
}

impl V4TableCapture {
    pub(in crate::state) fn new(freeze: TableFreeze, previous: Option<&V4PersistedTable>) -> Self {
        Self {
            freeze,
            previous: previous.map_or_else(BTreeMap::new, |previous| previous.handles.clone()),
        }
    }

    pub(in crate::state) async fn persist_objects(
        self,
        store: &mut V4Store,
        manifest: &V4Manifest,
    ) -> Result<(V4PersistedTable, V4Manifest)> {
        let epoch = self.freeze.epoch;
        let (lineage, objects, identities) = collect_page_objects(self.freeze, &self.previous);
        let handles =
            persist_page_handles(store, epoch, objects, &identities, &self.previous).await?;
        let (root_objects, live) = build_root_objects(&lineage, &handles, epoch)?;
        let roots = persist_roots(store, epoch, root_objects).await?;
        store.record_carried(
            live.keys()
                .filter(|id| self.previous.contains_key(*id))
                .count(),
        );
        let manifest = Self::manifest_with_roots(manifest, &roots)?;
        Ok((
            V4PersistedTable {
                lineage,
                handles: live,
                roots,
                manifest_path: std::path::PathBuf::new(),
            },
            manifest,
        ))
    }

    pub(in crate::state) async fn persist(
        self,
        store: &mut V4Store,
        manifest: &V4Manifest,
    ) -> Result<V4PersistedTable> {
        let (persisted, manifest) = self.persist_objects(store, manifest).await?;
        let manifest_path = store.publish_manifest(&manifest)?;
        Ok(V4PersistedTable {
            manifest_path,
            ..persisted
        })
    }

    fn manifest_with_roots(
        manifest: &V4Manifest,
        roots: &BTreeMap<super::core::KeyGroupId, V4Handle>,
    ) -> Result<V4Manifest> {
        let mut manifest = manifest.clone();
        for operator in manifest.operators.values_mut() {
            if let OperatorV4::Table { partitions, .. } = operator {
                for partition in partitions {
                    partition.root = roots
                        .get(&super::core::KeyGroupId(partition.key_group))
                        .cloned()
                        .ok_or_else(|| mismatch("partition root is absent".into()))?;
                }
            }
        }
        manifest.manifest_fingerprint = manifest_fingerprint(&manifest)?;
        Ok(manifest)
    }
}

type ObjectIdentity = (String, String);
type PageHandles = BTreeMap<super::core::PageId, V4Handle>;
type PageIdentities = BTreeMap<ObjectIdentity, super::core::PageId>;
type PageCandidate = (ObjectIdentity, super::core::PageId, NewObject);

fn page_candidate(page: &PageRef) -> PageCandidate {
    let object = page_object(page);
    let identity = (object.owner.clone(), object.object_id.clone());
    (identity, page.id(), object)
}

fn changed_page_objects(freeze: &TableFreeze) -> (Vec<NewObject>, PageIdentities) {
    let mut objects = Vec::new();
    let mut identities = BTreeMap::new();
    for partition in &freeze.partitions {
        for change in &partition.changes {
            if let FrozenChange::Write(page) = change {
                let (identity, id, object) = page_candidate(page);
                identities.insert(identity, id);
                objects.push(object);
            }
        }
    }
    (objects, identities)
}

fn lineage_page_candidates<'a>(
    lineage: &'a TableLineage,
    previous: &'a PageHandles,
) -> impl Iterator<Item = PageCandidate> + 'a {
    lineage
        .partitions
        .values()
        .flat_map(|partition| partition.pages.values())
        .filter(move |page| !previous.contains_key(&page.id()))
        .map(page_candidate)
}

fn collect_page_objects(
    freeze: TableFreeze,
    previous: &PageHandles,
) -> (TableLineage, Vec<NewObject>, PageIdentities) {
    let (mut objects, mut identities) = changed_page_objects(&freeze);
    let lineage = freeze.complete();
    for (identity, id, object) in lineage_page_candidates(&lineage, previous) {
        if let std::collections::btree_map::Entry::Vacant(entry) = identities.entry(identity) {
            entry.insert(id);
            objects.push(object);
        }
    }
    (lineage, objects, identities)
}

async fn persist_page_handles(
    store: &mut V4Store,
    epoch: u64,
    objects: Vec<NewObject>,
    identities: &PageIdentities,
    previous: &PageHandles,
) -> Result<PageHandles> {
    let encoded = store.write_objects(epoch, objects).await?;
    let mut handles = previous.clone();
    for object in encoded {
        let identity = (object.handle.owner.clone(), object.handle.object_id.clone());
        let id = identities
            .get(&identity)
            .copied()
            .ok_or_else(|| mismatch("persisted page identity is unknown".into()))?;
        handles.insert(id, object.handle);
    }
    Ok(handles)
}

fn build_root_objects(
    lineage: &TableLineage,
    handles: &PageHandles,
    epoch: u64,
) -> Result<(Vec<NewObject>, PageHandles)> {
    let mut live = BTreeMap::new();
    let mut root_objects = Vec::new();
    for (group, partition) in &lineage.partitions {
        let index = partition_index(*group, partition, handles, epoch)?;
        live.extend(
            index
                .pages
                .iter()
                .map(|entry| (page_id(entry), entry.handle.clone())),
        );
        root_objects.push(partition_root_object(*group, index, epoch)?);
    }
    Ok((root_objects, live))
}

fn partition_index(
    group: super::core::KeyGroupId,
    partition: &super::core::PartitionLineage,
    handles: &PageHandles,
    epoch: u64,
) -> Result<PartitionIndex> {
    let mut entries = Vec::new();
    for page in partition.pages.values() {
        let handle = handles
            .get(&page.id())
            .cloned()
            .ok_or_else(|| mismatch("partition index references a page without a handle".into()))?;
        entries.push(page_entry(page, handle));
    }
    let index = PartitionIndex {
        format_version: 4,
        kind: "partition_index".into(),
        key_group: group.0,
        pages: entries,
        next_generations: partition.allocator.next_generations.clone(),
        free_slots: partition.allocator.free_slots.iter().copied().collect(),
    };
    index.validate(epoch)?;
    Ok(index)
}

fn partition_root_object(
    group: super::core::KeyGroupId,
    index: PartitionIndex,
    epoch: u64,
) -> Result<NewObject> {
    Ok(NewObject {
        kind: ObjectKind::PartitionIndex,
        owner: "rolling".into(),
        object_id: group.0.to_string(),
        generation: epoch,
        payload: ObjectPayload::Json(serde_json::to_value(index).map_err(|error| {
            format_error(format!("partition index serialization failed: {error}"))
        })?),
    })
}

async fn persist_roots(
    store: &mut V4Store,
    epoch: u64,
    objects: Vec<NewObject>,
) -> Result<BTreeMap<super::core::KeyGroupId, V4Handle>> {
    let encoded = store.write_objects(epoch, objects).await?;
    let mut roots = BTreeMap::new();
    for root in encoded {
        let group = super::core::KeyGroupId(
            root.handle
                .object_id
                .parse::<u32>()
                .map_err(|_| format_error("partition root key group is invalid".into()))?,
        );
        roots.insert(group, root.handle);
    }
    Ok(roots)
}

#[allow(
    clippy::unused_async,
    reason = "the coordinator awaits recovery as one operation"
)]
pub(in crate::state) async fn restore_table_v4(
    store: &mut V4Store,
    table: &StateTable,
    manifest: &V4Manifest,
) -> Result<TableSnapshot> {
    let (partitions, retained_bytes) = restore_partitions(store, manifest)?;
    let lineage = TableLineage {
        epoch: manifest.epoch,
        partitions,
        retained_bytes,
    };
    snapshot_from_lineage(table, lineage)
}

fn restore_partitions(
    store: &mut V4Store,
    manifest: &V4Manifest,
) -> Result<(
    BTreeMap<super::core::KeyGroupId, super::core::PartitionLineage>,
    usize,
)> {
    let mut partitions = BTreeMap::new();
    let mut retained_bytes = 0;
    for operator in manifest.operators.values() {
        let OperatorV4::Table {
            partitions: roots, ..
        } = operator
        else {
            continue;
        };
        for partition in roots {
            let (group, restored, bytes) = restore_partition(store, partition, manifest.epoch)?;
            retained_bytes += bytes;
            partitions.insert(group, restored);
        }
    }
    Ok((partitions, retained_bytes))
}

fn restore_partition(
    store: &mut V4Store,
    partition: &PartitionV4,
    epoch: u64,
) -> Result<(
    super::core::KeyGroupId,
    super::core::PartitionLineage,
    usize,
)> {
    let index = read_partition_index(store, partition, epoch)?;
    let (pages, retained_bytes) = restore_pages(store, index.pages)?;
    let allocator = slot_allocator(&index.next_generations, &index.free_slots)?;
    Ok((
        super::core::KeyGroupId(index.key_group),
        super::core::PartitionLineage { pages, allocator },
        retained_bytes,
    ))
}

fn read_partition_index(
    store: &mut V4Store,
    partition: &PartitionV4,
    epoch: u64,
) -> Result<PartitionIndex> {
    let root = store.read_object(&partition.root)?;
    let ObjectPayload::Bytes(bytes) = root else {
        return Err(mismatch("partition root is not binary JSON".into()));
    };
    let index = PartitionIndex::from_bytes(&bytes, epoch)?;
    if index.key_group != partition.key_group {
        return Err(mismatch("partition index key group mismatch".into()));
    }
    Ok(index)
}

fn restore_pages(
    store: &mut V4Store,
    entries: Vec<PageEntry>,
) -> Result<(BTreeMap<u32, PageRef>, usize)> {
    let mut pages = BTreeMap::new();
    let mut retained_bytes = 0;
    for entry in entries {
        let object = store.read_object(&entry.handle)?;
        let ObjectPayload::Bytes(bytes) = object else {
            return Err(mismatch("page object is not Arrow IPC".into()));
        };
        let page = decode_page(&entry, &bytes)?;
        retained_bytes += page.logical_bytes();
        pages.insert(entry.page_id.slot, page);
    }
    Ok((pages, retained_bytes))
}

fn slot_allocator(
    next_generations: &BTreeMap<u32, u64>,
    free_slots: &[u32],
) -> Result<super::pages::SlotAllocator> {
    Ok(super::pages::SlotAllocator {
        next_generations: next_generations.clone(),
        free_slots: free_slots.iter().copied().collect(),
        next_slot: next_partition_slot(next_generations)?,
    })
}

fn next_partition_slot(next_generations: &BTreeMap<u32, u64>) -> Result<u32> {
    let Some(slot) = next_generations.keys().max() else {
        return Ok(0);
    };
    slot.checked_add(1)
        .ok_or_else(|| mismatch("partition next slot overflowed".into()))
}

type OrderedEntity = (u32, super::core::KeyGroupId, usize, super::EntityKey);

struct Layout7Row {
    counter: u64,
    values: Vec<super::core::RingValue>,
    capacity: usize,
    numeric: Vec<NumericAccumulator>,
}

fn snapshot_from_lineage(table: &StateTable, lineage: TableLineage) -> Result<TableSnapshot> {
    let ordered = ordered_entities(&lineage);
    let keys = ordered
        .iter()
        .map(|(_, _, _, key)| key.clone())
        .collect::<Vec<_>>();
    let next_entity =
        u32::try_from(keys.len()).map_err(|_| mismatch("entity IDs exceed u32".into()))?;
    let (keyed, rings) = layout7_snapshots(table, &lineage, &ordered)?;
    table.v4_from_lineage(keys, keyed, rings, lineage, next_entity)
}

fn ordered_entities(lineage: &TableLineage) -> Vec<OrderedEntity> {
    let mut ordered = Vec::new();
    for (group, partition) in &lineage.partitions {
        for page in partition.pages.values() {
            if let PageRef::Metadata(page) = page {
                for (row, entity) in page.value.entities.iter().enumerate() {
                    ordered.push((entity.index, *group, row, entity.key.clone()));
                }
            }
        }
    }
    ordered.sort_by_key(|(index, _, _, _)| *index);
    ordered
}

fn layout7_snapshots(
    table: &StateTable,
    lineage: &TableLineage,
    ordered: &[OrderedEntity],
) -> Result<(Vec<KeyedSnapshot>, Vec<RingSnapshot>)> {
    let Some(schema) = table.layout7_schema() else {
        return Ok((Vec::new(), Vec::new()));
    };
    let mut counters = Vec::new();
    let mut values = Vec::new();
    let mut offsets = vec![0];
    let mut capacities = Vec::new();
    let mut numeric = Vec::new();
    for (_, group, row, _) in ordered {
        let row = layout7_row(&lineage.partitions[group], *row)?;
        counters.push(row.counter);
        values.extend(row.values);
        offsets.push(values.len());
        capacities.push(row.capacity);
        numeric.push(row.numeric);
    }
    let keyed = vec![KeyedSnapshot {
        name: schema.counter_name,
        kind: super::ValueKind::UInt64,
        values: Arc::new(UInt64Array::from(counters)),
    }];
    let rings = vec![RingSnapshot {
        name: schema.numeric_name,
        kind: super::RingKind::Float64,
        retention: schema.retention,
        values: ring_array(super::RingKind::Float64, &values),
        offsets,
        capacities,
        numeric: Some(numeric),
    }];
    Ok((keyed, rings))
}

fn layout7_row(partition: &super::core::PartitionLineage, row: usize) -> Result<Layout7Row> {
    let counter = layout7_counter(partition, row)?;
    let ring = layout7_ring(partition, row)?;
    let values = ring.ring.history();
    let capacity = ring.ring.capacity();
    let numeric = ring
        .numeric
        .clone()
        .ok_or_else(|| mismatch("numeric group is absent".into()))?;
    Ok(Layout7Row {
        counter,
        values,
        capacity,
        numeric,
    })
}

fn layout7_counter(partition: &super::core::PartitionLineage, row: usize) -> Result<u64> {
    let counter = partition.pages.values().find_map(|page| match page {
        PageRef::Keyed {
            page_index, page, ..
        } if *page_index == row / 1024 => Some(page.value.values[row % 1024]),
        _ => None,
    });
    let Some(super::core::KeyedValue::UInt64(counter)) = counter else {
        return Err(mismatch("layout-7 counter is absent".into()));
    };
    Ok(counter)
}

fn layout7_ring(
    partition: &super::core::PartitionLineage,
    row: usize,
) -> Result<&super::core::RingPage> {
    partition
        .pages
        .values()
        .find_map(|page| match page {
            PageRef::Ring {
                row: found, page, ..
            } if *found == row => Some(page.value.as_ref()),
            _ => None,
        })
        .ok_or_else(|| mismatch("layout-7 ring is absent".into()))
}

fn page_object(page: &PageRef) -> NewObject {
    let id = page.id();
    let payload = page_payload(page);
    NewObject {
        kind: ObjectKind::Page,
        owner: "rolling".into(),
        object_id: format!("{}/{}/{}", id.key_group.0, id.slot, id.generation),
        generation: id.generation,
        payload: ObjectPayload::Arrow(payload),
    }
}

fn page_entry(page: &PageRef, handle: V4Handle) -> PageEntry {
    let (column, kind, range) = page_location(page);
    PageEntry {
        page_id: V4PageId {
            key_group: page.id().key_group.0,
            slot: page.id().slot,
            generation: page.id().generation,
        },
        column,
        kind: kind.into(),
        logical_range: LogicalRange {
            start: range.0,
            end: range.1,
        },
        byte_len: handle.byte_len,
        sha256: handle.sha256.clone(),
        handle,
    }
}

fn page_location(page: &PageRef) -> (String, &'static str, (u64, u64)) {
    match page {
        PageRef::Metadata(page) => (
            "metadata".into(),
            "metadata",
            (0, page.value.entities.len() as u64),
        ),
        PageRef::Keyed {
            column: column_index,
            page,
            ..
        } => (
            column_index.to_string(),
            "keyed",
            (0, page.value.values.len() as u64),
        ),
        PageRef::Ring {
            column: column_index,
            page,
            ..
        } => (
            column_index.to_string(),
            "ring",
            (0, page.value.ring.len as u64),
        ),
        PageRef::ValueLog {
            column: column_index,
            page,
            ..
        } => (
            column_index.to_string(),
            "value_log",
            (
                u64::from(page.value.start),
                u64::from(page.value.start + page.value.len),
            ),
        ),
    }
}

fn page_payload(page: &PageRef) -> ArrowPayload {
    let mut metadata = page_metadata(page);
    let (fields, columns) = match page {
        PageRef::Metadata(page) => {
            metadata.insert(
                "dictionary_bytes".into(),
                page.value.dictionary_bytes.to_string(),
            );
            metadata.insert(
                "value_log_lengths".into(),
                format!("{:?}", page.value.value_log_lengths),
            );
            (
                vec![
                    Field::new("entity_index", DataType::UInt64, false),
                    Field::new("entity_key", DataType::Binary, false),
                ],
                vec![
                    Arc::new(UInt64Array::from_iter_values(
                        page.value
                            .entities
                            .iter()
                            .map(|entity| u64::from(entity.index)),
                    )) as ArrayRef,
                    Arc::new(BinaryArray::from_iter_values(
                        page.value
                            .entities
                            .iter()
                            .map(|entity| entity.key.bytes.as_slice()),
                    )) as ArrayRef,
                ],
            )
        }
        PageRef::Keyed { page, .. } => (
            vec![Field::new(
                "value",
                keyed_array(page.value.kind, &[]).data_type().clone(),
                false,
            )],
            vec![keyed_array(page.value.kind, &page.value.values)],
        ),
        PageRef::Ring { page, .. } => {
            metadata.insert("head".into(), page.value.ring.head.to_string());
            metadata.insert("len".into(), page.value.ring.len.to_string());
            metadata.insert("capacity".into(), page.value.ring.capacity().to_string());
            if let Some(numeric) = &page.value.numeric {
                for value in numeric {
                    metadata.extend(numeric_metadata(&value.layout7()));
                }
            }
            (
                vec![Field::new(
                    "value",
                    ring_array(page.value.kind, &[]).data_type().clone(),
                    true,
                )],
                vec![ring_array(page.value.kind, &page.value.ring.history())],
            )
        }
        PageRef::ValueLog { page, .. } => {
            metadata.insert("start".into(), page.value.start.to_string());
            metadata.insert("len".into(), page.value.len.to_string());
            metadata.insert("logical_bytes".into(), page.value.logical_bytes.to_string());
            let fields = page
                .value
                .chunks
                .iter()
                .enumerate()
                .map(|(index, array)| {
                    Field::new(format!("chunk_{index}"), array.data_type().clone(), true)
                })
                .collect::<Vec<_>>();
            (fields, page.value.chunks.clone())
        }
    };
    ArrowPayload {
        fields,
        columns,
        metadata,
    }
}

fn page_metadata(page: &PageRef) -> HashMap<String, String> {
    let (column, kind, _) = page_location(page);
    let mut metadata =
        HashMap::from([("page_kind".into(), kind.into()), ("column".into(), column)]);
    match page {
        PageRef::Metadata(_) => {}
        PageRef::Keyed { page_index, .. } | PageRef::ValueLog { page_index, .. } => {
            metadata.insert("page_index".into(), page_index.to_string());
        }
        PageRef::Ring { row, .. } => {
            metadata.insert("row".into(), row.to_string());
        }
    }
    metadata
}

fn numeric_metadata(parts: &NumericAccumulatorLayout7) -> HashMap<String, String> {
    HashMap::from([
        ("valid_count".into(), parts.valid_count.to_string()),
        ("finite_count".into(), parts.finite_count.to_string()),
        ("nan_count".into(), parts.nan_count.to_string()),
        ("pos_inf".into(), parts.pos_inf.to_string()),
        ("neg_inf".into(), parts.neg_inf.to_string()),
        ("sum_value".into(), parts.sum_value.to_bits().to_string()),
        ("sum_error".into(), parts.sum_error.to_bits().to_string()),
        ("sum_enabled".into(), parts.sum_enabled.to_string()),
        (
            "shift".into(),
            parts
                .shift
                .map_or_else(String::new, |value| value.to_bits().to_string()),
        ),
        (
            "offsets_value".into(),
            parts.offsets_value.to_bits().to_string(),
        ),
        (
            "offsets_error".into(),
            parts.offsets_error.to_bits().to_string(),
        ),
        (
            "squares_value".into(),
            parts.squares_value.to_bits().to_string(),
        ),
        (
            "squares_error".into(),
            parts.squares_error.to_bits().to_string(),
        ),
        (
            "dispersion_enabled".into(),
            parts.dispersion_enabled.to_string(),
        ),
        ("since_rebuild".into(), parts.since_rebuild.to_string()),
        ("n_prev".into(), parts.n_prev.to_string()),
    ])
}

fn decode_page(entry: &PageEntry, bytes: &[u8]) -> Result<PageRef> {
    let record = read_record(bytes)?;
    let schema = record.schema();
    let metadata = schema.metadata();
    match metadata.get("page_kind").map(String::as_str) {
        Some("metadata") => decode_metadata(&record, entry),
        Some("keyed") => decode_keyed(&record, entry),
        Some("ring") => decode_ring(&record, entry),
        Some("value_log") => decode_value_log(&record, entry),
        _ => Err(mismatch("unknown checkpoint-v4 page kind".into())),
    }
}

fn page_id(entry: &PageEntry) -> super::core::PageId {
    super::core::PageId {
        key_group: super::core::KeyGroupId(entry.page_id.key_group),
        slot: entry.page_id.slot,
        generation: entry.page_id.generation,
    }
}

fn decode_metadata(record: &RecordBatch, entry: &PageEntry) -> Result<PageRef> {
    let indices = column::<UInt64Array>(record, 0)?;
    let keys = column::<BinaryArray>(record, 1)?;
    let entities = (0..record.num_rows())
        .map(|row| {
            Ok(super::core::PartitionEntity {
                index: u32::try_from(indices.value(row))
                    .map_err(|_| mismatch("entity index exceeds u32".into()))?,
                key: super::EntityKey::new(keys.value(row).to_vec()),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let schema = record.schema();
    let metadata = schema.metadata();
    let dictionary_bytes = parse_metadata(metadata, "dictionary_bytes")?;
    let value_log_lengths = parse_metadata::<String>(metadata, "value_log_lengths")?;
    let value_log_lengths = parse_u32_list(&value_log_lengths)?;
    Ok(PageRef::Metadata(super::pages::Page::new(
        page_id(entry),
        super::core::PartitionMetadata {
            entities,
            dictionary_bytes,
            value_log_lengths,
        },
    )))
}

fn decode_keyed(record: &RecordBatch, entry: &PageEntry) -> Result<PageRef> {
    let column_index = entry
        .column
        .parse::<usize>()
        .map_err(|_| mismatch("keyed column is invalid".into()))?;
    let page_index = parse_metadata::<usize>(record.schema().metadata(), "page_index")?;
    let kind = keyed_kind(record.column(0).data_type())?;
    if (0..record.num_rows()).any(|row| !record.column(0).is_valid(row)) {
        return Err(mismatch("keyed page contains a null value".into()));
    }
    let values = decode_keyed_values(record, kind)?;
    Ok(PageRef::Keyed {
        column: column_index,
        page_index,
        page: super::pages::Page::new(page_id(entry), super::core::KeyedPage { kind, values }),
    })
}

fn keyed_kind(data_type: &DataType) -> Result<super::ValueKind> {
    match data_type {
        DataType::UInt64 => Ok(super::ValueKind::UInt64),
        DataType::Int64 => Ok(super::ValueKind::Int64),
        DataType::Float64 => Ok(super::ValueKind::Float64),
        DataType::Boolean => Ok(super::ValueKind::Boolean),
        _ => Err(mismatch("keyed page type is unsupported".into())),
    }
}

fn decode_keyed_values(
    record: &RecordBatch,
    kind: super::ValueKind,
) -> Result<Vec<super::core::KeyedValue>> {
    match kind {
        super::ValueKind::UInt64 => decode_u64_keyed_values(record),
        super::ValueKind::Int64 => decode_i64_keyed_values(record),
        super::ValueKind::Float64 => decode_f64_keyed_values(record),
        super::ValueKind::Boolean => decode_bool_keyed_values(record),
    }
}

fn decode_u64_keyed_values(record: &RecordBatch) -> Result<Vec<super::core::KeyedValue>> {
    Ok(column::<UInt64Array>(record, 0)?
        .values()
        .iter()
        .copied()
        .map(super::core::KeyedValue::UInt64)
        .collect())
}

fn decode_i64_keyed_values(record: &RecordBatch) -> Result<Vec<super::core::KeyedValue>> {
    Ok(column::<Int64Array>(record, 0)?
        .values()
        .iter()
        .copied()
        .map(super::core::KeyedValue::Int64)
        .collect())
}

fn decode_f64_keyed_values(record: &RecordBatch) -> Result<Vec<super::core::KeyedValue>> {
    Ok(column::<Float64Array>(record, 0)?
        .values()
        .iter()
        .copied()
        .map(super::core::KeyedValue::Float64)
        .collect())
}

fn decode_bool_keyed_values(record: &RecordBatch) -> Result<Vec<super::core::KeyedValue>> {
    Ok(column::<BooleanArray>(record, 0)?
        .values()
        .iter()
        .map(super::core::KeyedValue::Boolean)
        .collect())
}

fn decode_ring(record: &RecordBatch, entry: &PageEntry) -> Result<PageRef> {
    let column_index = entry
        .column
        .parse::<usize>()
        .map_err(|_| mismatch("ring column is invalid".into()))?;
    let row = parse_metadata::<usize>(record.schema().metadata(), "row")?;
    let kind = ring_kind(record.column(0).data_type())?;
    let (ring, numeric) = decode_ring_state(record, kind)?;
    let capacity = ring.capacity();
    Ok(PageRef::Ring {
        column: column_index,
        row,
        page: super::pages::Page::new(
            page_id(entry),
            super::core::RingPage {
                kind,
                ring,
                slot_stamps: vec![0; capacity],
                numeric,
            },
        ),
    })
}

fn ring_kind(data_type: &DataType) -> Result<super::RingKind> {
    match data_type {
        DataType::Float64 => Ok(super::RingKind::Float64),
        DataType::Int64 => Ok(super::RingKind::Int64),
        DataType::UInt64 => Ok(super::RingKind::UInt64),
        DataType::Struct(_) => Ok(super::RingKind::PairFloat64),
        _ => Err(mismatch("ring page type is unsupported".into())),
    }
}

fn decode_ring_state(
    record: &RecordBatch,
    kind: super::RingKind,
) -> Result<(super::core::Ring, Option<Vec<NumericAccumulator>>)> {
    let values = decode_ring_values(record, kind)?;
    let schema = record.schema();
    let metadata = schema.metadata();
    let ring = decode_ring_metadata(metadata, kind, values.len())?;
    let ring = populate_ring(ring, values);
    let numeric = decode_numeric_metadata(metadata)?.map(|value| vec![value]);
    Ok((ring, numeric))
}

fn decode_ring_metadata(
    metadata: &HashMap<String, String>,
    kind: super::RingKind,
    value_count: usize,
) -> Result<super::core::Ring> {
    let (head, len, capacity) = decode_ring_shape(metadata)?;
    if len > capacity || value_count != len {
        return Err(mismatch("ring length exceeds its capacity".into()));
    }
    Ok(super::core::Ring {
        head,
        len,
        slots: vec![kind.empty(); capacity],
    })
}

fn decode_ring_shape(metadata: &HashMap<String, String>) -> Result<(usize, usize, usize)> {
    Ok((
        parse_metadata(metadata, "head")?,
        parse_metadata(metadata, "len")?,
        parse_metadata(metadata, "capacity")?,
    ))
}

fn populate_ring(
    mut ring: super::core::Ring,
    values: Vec<super::core::RingValue>,
) -> super::core::Ring {
    for (offset, value) in values.into_iter().enumerate() {
        let position = ring.position(offset);
        ring.slots[position] = value;
    }
    ring
}

fn decode_ring_values(
    record: &RecordBatch,
    kind: super::RingKind,
) -> Result<Vec<super::core::RingValue>> {
    match kind {
        super::RingKind::Float64 => decode_f64_ring_values(record),
        super::RingKind::Int64 => decode_i64_ring_values(record),
        super::RingKind::UInt64 => decode_u64_ring_values(record),
        super::RingKind::PairFloat64 => decode_pair_ring_values(record),
    }
}

fn decode_f64_ring_values(record: &RecordBatch) -> Result<Vec<super::core::RingValue>> {
    Ok(column::<Float64Array>(record, 0)?
        .iter()
        .map(super::core::RingValue::Float64)
        .collect())
}

fn decode_i64_ring_values(record: &RecordBatch) -> Result<Vec<super::core::RingValue>> {
    Ok(column::<Int64Array>(record, 0)?
        .iter()
        .map(super::core::RingValue::Int64)
        .collect())
}

fn decode_u64_ring_values(record: &RecordBatch) -> Result<Vec<super::core::RingValue>> {
    Ok(column::<UInt64Array>(record, 0)?
        .iter()
        .map(super::core::RingValue::UInt64)
        .collect())
}

fn decode_pair_ring_values(record: &RecordBatch) -> Result<Vec<super::core::RingValue>> {
    let pairs = pair_ring_array(record)?;
    let left = pair_ring_left(pairs)?;
    let right = pair_ring_right(pairs)?;
    Ok((0..pairs.len())
        .map(|row| {
            super::core::RingValue::PairFloat64(
                (left.is_valid(row) && right.is_valid(row))
                    .then_some((left.value(row), right.value(row))),
            )
        })
        .collect())
}

fn pair_ring_array(record: &RecordBatch) -> Result<&datafusion::arrow::array::StructArray> {
    record
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::StructArray>()
        .ok_or_else(|| mismatch("pair ring page type mismatch".into()))
}

fn pair_ring_left(pairs: &datafusion::arrow::array::StructArray) -> Result<&Float64Array> {
    pairs
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| mismatch("pair ring left type mismatch".into()))
}

fn pair_ring_right(pairs: &datafusion::arrow::array::StructArray) -> Result<&Float64Array> {
    pairs
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| mismatch("pair ring right type mismatch".into()))
}

fn decode_value_log(record: &RecordBatch, entry: &PageEntry) -> Result<PageRef> {
    let column_index = entry
        .column
        .parse::<usize>()
        .map_err(|_| mismatch("value-log column is invalid".into()))?;
    let page_index = parse_metadata::<usize>(record.schema().metadata(), "page_index")?;
    let schema = record.schema();
    let metadata = schema.metadata();
    Ok(PageRef::ValueLog {
        column: column_index,
        page_index,
        page: super::pages::Page::new(
            page_id(entry),
            super::core::ValueLogPage {
                chunks: record.columns().to_vec(),
                start: parse_metadata(metadata, "start")?,
                len: parse_metadata(metadata, "len")?,
                logical_bytes: parse_metadata(metadata, "logical_bytes")?,
            },
        ),
    })
}

fn decode_numeric_metadata(
    metadata: &HashMap<String, String>,
) -> Result<Option<NumericAccumulator>> {
    if !metadata.contains_key("valid_count") {
        return Ok(None);
    }
    let shift = decode_numeric_shift(metadata)?;
    let layout = decode_numeric_layout7(metadata, shift)?;
    Ok(Some(NumericAccumulator::from_layout7(layout)))
}

fn decode_numeric_shift(metadata: &HashMap<String, String>) -> Result<Option<f64>> {
    let shift = parse_metadata::<String>(metadata, "shift")?;
    if shift.is_empty() {
        return Ok(None);
    }
    parse_f64_metadata(metadata, "shift").map(Some)
}

fn decode_numeric_layout7(
    metadata: &HashMap<String, String>,
    shift: Option<f64>,
) -> Result<NumericAccumulatorLayout7> {
    let (valid_count, finite_count, nan_count, pos_inf, neg_inf) = decode_numeric_counts(metadata)?;
    let (sum_value, sum_error, sum_enabled) = decode_numeric_sum(metadata)?;
    let (offsets_value, offsets_error) =
        decode_f64_pair(metadata, "offsets_value", "offsets_error")?;
    let (squares_value, squares_error) =
        decode_f64_pair(metadata, "squares_value", "squares_error")?;
    let (dispersion_enabled, since_rebuild, n_prev) = decode_numeric_progress(metadata)?;
    Ok(NumericAccumulatorLayout7 {
        valid_count,
        finite_count,
        nan_count,
        pos_inf,
        neg_inf,
        sum_value,
        sum_error,
        sum_enabled,
        shift,
        offsets_value,
        offsets_error,
        squares_value,
        squares_error,
        dispersion_enabled,
        since_rebuild,
        n_prev,
    })
}

fn decode_numeric_counts(metadata: &HashMap<String, String>) -> Result<(u64, u64, u64, u64, u64)> {
    Ok((
        parse_metadata(metadata, "valid_count")?,
        parse_metadata(metadata, "finite_count")?,
        parse_metadata(metadata, "nan_count")?,
        parse_metadata(metadata, "pos_inf")?,
        parse_metadata(metadata, "neg_inf")?,
    ))
}

fn decode_numeric_sum(metadata: &HashMap<String, String>) -> Result<(f64, f64, bool)> {
    Ok((
        parse_f64_metadata(metadata, "sum_value")?,
        parse_f64_metadata(metadata, "sum_error")?,
        parse_metadata(metadata, "sum_enabled")?,
    ))
}

fn decode_numeric_progress(metadata: &HashMap<String, String>) -> Result<(bool, u32, u64)> {
    Ok((
        parse_metadata(metadata, "dispersion_enabled")?,
        parse_metadata(metadata, "since_rebuild")?,
        parse_metadata(metadata, "n_prev")?,
    ))
}

fn decode_f64_pair(
    metadata: &HashMap<String, String>,
    value_key: &str,
    error_key: &str,
) -> Result<(f64, f64)> {
    Ok((
        parse_f64_metadata(metadata, value_key)?,
        parse_f64_metadata(metadata, error_key)?,
    ))
}

fn parse_f64_metadata(metadata: &HashMap<String, String>, key: &str) -> Result<f64> {
    Ok(f64::from_bits(parse_metadata(metadata, key)?))
}

fn read_record(bytes: &[u8]) -> Result<RecordBatch> {
    let mut reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| format_error(format!("checkpoint-v4 Arrow reader failed: {error}")))?;
    let record = reader
        .next()
        .transpose()
        .map_err(|error| format_error(format!("checkpoint-v4 Arrow read failed: {error}")))?
        .ok_or_else(|| format_error("checkpoint-v4 Arrow page is empty".into()))?;
    if reader
        .next()
        .transpose()
        .map_err(|error| format_error(error.to_string()))?
        .is_some()
    {
        return Err(format_error(
            "checkpoint-v4 Arrow page has multiple batches".into(),
        ));
    }
    Ok(record)
}

fn column<T: 'static>(record: &RecordBatch, index: usize) -> Result<&T> {
    record
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| mismatch("checkpoint-v4 page column type mismatch".into()))
}

fn parse_metadata<T: std::str::FromStr>(
    metadata: &HashMap<String, String>,
    key: &str,
) -> Result<T> {
    metadata
        .get(key)
        .ok_or_else(|| mismatch(format!("page metadata {key} is absent")))?
        .parse::<T>()
        .map_err(|_| mismatch(format!("page metadata {key} is invalid")))
}

fn parse_u32_list(value: &str) -> Result<Vec<u32>> {
    let trimmed = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| mismatch("value-log lengths are invalid".into()))?;
    if trimmed.trim().is_empty() {
        return Ok(Vec::new());
    }
    trimmed
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<u32>()
                .map_err(|_| mismatch("value-log length is invalid".into()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{PartitionIndex, StateTable, V4TableCapture, restore_table_v4};
    use crate::state::table::core::KeyGroupId;
    use crate::state::v4::{V4Manifest, V4Store, manifest_fingerprint};

    fn schema() -> StateTable {
        let mut table = StateTable::with_key_groups(true, &[KeyGroupId(0), KeyGroupId(1)]).unwrap();
        table.add_keyed_column("transition_count", super::super::ValueKind::UInt64);
        table.add_numeric_rows_column("numeric", 2);
        table
    }

    fn manifest() -> V4Manifest {
        V4Manifest::from_bytes(
            include_str!("../../../tests/fixtures/checkpoint-v4/manifest-v4.json").as_bytes(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn persist_recover_restore_preserves_layout7_bits_and_counters() {
        let mut source = schema();
        source.begin().unwrap();
        let entity = source
            .resolve_or_append(super::super::EntityKey::new(b"a".to_vec()))
            .unwrap();
        source.set_u64(0, entity, 2).unwrap();
        source.push_numeric_rows(0, entity, Some(1.0)).unwrap();
        source.push_numeric_rows(0, entity, Some(3.0)).unwrap();
        source.commit().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let mut store = V4Store::new(temporary.path().join("lineage"));
        let persisted = V4TableCapture::new(source.freeze(7).unwrap(), None)
            .persist(&mut store, &manifest())
            .await
            .unwrap();
        assert_eq!(store.stats().content_hashes, 6);
        assert_eq!(store.stats().file_syncs, 6);
        assert_eq!(store.stats().read_back_hashes, 0);

        let stored_manifest = V4Store::read_manifest(&persisted.manifest_path).unwrap();
        let mut cold = schema();
        let snapshot = restore_table_v4(&mut store, &cold, &stored_manifest)
            .await
            .unwrap();
        cold.restore(snapshot).unwrap();
        assert_eq!(
            cold.canonical_state_for_test(),
            source.canonical_state_for_test()
        );
        assert_eq!(store.stats().recovery_hashes, 6);
    }

    #[tokio::test]
    async fn restore_rejects_corrupt_partition_root_and_page() {
        let mut source = schema();
        source.begin().unwrap();
        let entity = source
            .resolve_or_append(super::super::EntityKey::new(b"a".to_vec()))
            .unwrap();
        source.push_numeric_rows(0, entity, Some(1.0)).unwrap();
        source.commit().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let mut store = V4Store::new(temporary.path().join("lineage"));
        let persisted = V4TableCapture::new(source.freeze(7).unwrap(), None)
            .persist(&mut store, &manifest())
            .await
            .unwrap();
        let stored_manifest = V4Store::read_manifest(&persisted.manifest_path).unwrap();
        let root = persisted.roots[&KeyGroupId(0)].clone();
        let page = persisted.handles.values().next().unwrap().clone();
        for handle in [root, page] {
            let path = temporary.path().join(&handle.relative_path);
            let original = std::fs::read(&path).unwrap();
            let mut damaged = original.clone();
            *damaged.last_mut().unwrap() ^= 1;
            std::fs::write(&path, damaged).unwrap();
            assert!(
                restore_table_v4(&mut store, &schema(), &stored_manifest)
                    .await
                    .is_err()
            );
            std::fs::write(path, original).unwrap();
        }
    }

    #[tokio::test]
    async fn second_epoch_reuses_carried_handles_without_rehashing() {
        let mut source = schema();
        source.begin().unwrap();
        let entity = source
            .resolve_or_append(super::super::EntityKey::new(b"a".to_vec()))
            .unwrap();
        source.push_numeric_rows(0, entity, Some(1.0)).unwrap();
        source.commit().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let mut first_store = V4Store::new(temporary.path().join("lineage"));
        let first = V4TableCapture::new(source.freeze(7).unwrap(), None)
            .persist(&mut first_store, &manifest())
            .await
            .unwrap();
        source.settle_freeze(7, first.lineage.clone()).unwrap();

        source.begin().unwrap();
        source.push_numeric_rows(0, entity, Some(3.0)).unwrap();
        source.commit().unwrap();
        let mut next_manifest = manifest();
        next_manifest.epoch = 8;
        next_manifest.manifest_fingerprint = manifest_fingerprint(&next_manifest).unwrap();
        let mut second_store = V4Store::new(temporary.path().join("lineage"));
        let _second = V4TableCapture::new(source.freeze(8).unwrap(), Some(&first))
            .persist(&mut second_store, &next_manifest)
            .await
            .unwrap();
        assert_eq!(second_store.stats().content_hashes, 3);
        assert_eq!(second_store.stats().carried_objects, 3);
        assert_eq!(second_store.stats().read_back_hashes, 0);
    }

    #[test]
    fn golden_partition_index_is_strict_and_rejects_duplicates() {
        let document = include_bytes!("../../../tests/fixtures/checkpoint-v4/partition-0.json");
        let index = PartitionIndex::from_bytes(document, 7).unwrap();
        assert_eq!(index.pages.len(), 1);
        let mut duplicate = serde_json::from_slice::<serde_json::Value>(document).unwrap();
        let page = duplicate["pages"][0].clone();
        duplicate["pages"].as_array_mut().unwrap().push(page);
        assert!(PartitionIndex::from_bytes(&serde_json::to_vec(&duplicate).unwrap(), 7).is_err());
    }
}
