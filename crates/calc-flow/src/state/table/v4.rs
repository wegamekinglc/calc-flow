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
        V4Store, format_error, manifest_fingerprint, mismatch,
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
        if self.format_version != 4 || self.kind != "partition_index" {
            return Err(mismatch("partition index identity mismatch".into()));
        }
        if self
            .pages
            .windows(2)
            .any(|pair| pair[0].page_id >= pair[1].page_id)
        {
            return Err(mismatch(
                "page entries are not strictly ordered and unique".into(),
            ));
        }
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
        for page in &self.pages {
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

    pub(in crate::state) async fn persist(
        self,
        store: &mut V4Store,
        manifest: &V4Manifest,
    ) -> Result<V4PersistedTable> {
        let epoch = self.freeze.epoch;
        let mut objects = Vec::new();
        let mut identities = BTreeMap::new();
        for partition in &self.freeze.partitions {
            for change in &partition.changes {
                if let FrozenChange::Write(page) = change {
                    let object = page_object(page)?;
                    identities.insert(
                        (
                            object.owner.clone(),
                            object.object_id.clone(),
                            object.generation,
                        ),
                        page.id(),
                    );
                    objects.push(object);
                }
            }
        }
        let lineage = self.freeze.complete();
        for partition in lineage.partitions.values() {
            for page in partition.pages.values() {
                if !self.previous.contains_key(&page.id()) {
                    let object = page_object(page)?;
                    identities.insert(
                        (
                            object.owner.clone(),
                            object.object_id.clone(),
                            object.generation,
                        ),
                        page.id(),
                    );
                    objects.push(object);
                }
            }
        }
        let encoded = store.write_objects(epoch, objects).await?;
        let mut handles = self.previous.clone();
        for object in encoded {
            let id = identities
                .get(&(
                    object.handle.owner.clone(),
                    object.handle.object_id.clone(),
                    object.handle.generation,
                ))
                .copied()
                .ok_or_else(|| mismatch("persisted page identity is unknown".into()))?;
            handles.insert(id, object.handle);
        }
        let mut roots = BTreeMap::new();
        let mut live = BTreeMap::new();
        let mut root_objects = Vec::new();
        for (group, partition) in &lineage.partitions {
            let mut entries = Vec::new();
            for page in partition.pages.values() {
                let handle = handles.get(&page.id()).cloned().ok_or_else(|| {
                    mismatch("partition index references a page without a handle".into())
                })?;
                live.insert(page.id(), handle.clone());
                entries.push(page_entry(page, handle)?);
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
            root_objects.push(NewObject {
                kind: ObjectKind::PartitionIndex,
                owner: "rolling".into(),
                object_id: group.0.to_string(),
                generation: epoch,
                payload: ObjectPayload::Json(serde_json::to_value(index).map_err(|error| {
                    format_error(format!("partition index serialization failed: {error}"))
                })?),
            });
        }
        let encoded_roots = store.write_objects(epoch, root_objects).await?;
        for root in encoded_roots {
            let group = super::core::KeyGroupId(
                root.handle
                    .object_id
                    .parse::<u32>()
                    .map_err(|_| format_error("partition root key group is invalid".into()))?,
            );
            roots.insert(group, root.handle);
        }
        store.record_carried(
            live.keys()
                .filter(|id| self.previous.contains_key(*id))
                .count(),
        );
        let manifest = Self::manifest_with_roots(manifest, &roots)?;
        let manifest_path = store.publish_manifest(&manifest)?;
        Ok(V4PersistedTable {
            lineage,
            handles: live,
            roots,
            manifest_path,
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

pub(in crate::state) async fn restore_table_v4(
    store: &mut V4Store,
    table: &StateTable,
    manifest: &V4Manifest,
) -> Result<TableSnapshot> {
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
            let root = store.read_object(&partition.root)?;
            let ObjectPayload::Bytes(bytes) = root else {
                return Err(mismatch("partition root is not binary JSON".into()));
            };
            let index = PartitionIndex::from_bytes(&bytes, manifest.epoch)?;
            if index.key_group != partition.key_group {
                return Err(mismatch("partition index key group mismatch".into()));
            }
            let mut pages = BTreeMap::new();
            for entry in index.pages {
                let object = store.read_object(&entry.handle)?;
                let ObjectPayload::Bytes(bytes) = object else {
                    return Err(mismatch("page object is not Arrow IPC".into()));
                };
                let page = decode_page(&entry, &bytes)?;
                retained_bytes += page.logical_bytes();
                pages.insert(entry.page_id.slot, page);
            }
            let allocator = super::pages::SlotAllocator {
                next_generations: index.next_generations.clone(),
                free_slots: index.free_slots.iter().copied().collect(),
                next_slot: index.next_generations.keys().max().map_or(Ok(0), |slot| {
                    slot.checked_add(1)
                        .ok_or_else(|| mismatch("partition next slot overflowed".into()))
                })?,
            };
            partitions.insert(
                super::core::KeyGroupId(index.key_group),
                super::core::PartitionLineage { pages, allocator },
            );
        }
    }
    let lineage = TableLineage {
        epoch: manifest.epoch,
        partitions,
        retained_bytes,
    };
    snapshot_from_lineage(table, lineage)
}

fn snapshot_from_lineage(table: &StateTable, lineage: TableLineage) -> Result<TableSnapshot> {
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
    let keys = ordered
        .iter()
        .map(|(_, _, _, key)| key.clone())
        .collect::<Vec<_>>();
    let next_entity =
        u32::try_from(keys.len()).map_err(|_| mismatch("entity IDs exceed u32".into()))?;
    let mut keyed = Vec::new();
    let mut rings = Vec::new();
    if table.layout7_schema().is_some() {
        let schema = table.layout7_schema().expect("shape checked");
        let mut counters = Vec::new();
        let mut values = Vec::new();
        let mut offsets = vec![0];
        let mut capacities = Vec::new();
        let mut numeric = Vec::new();
        for (_, group, row, _) in &ordered {
            let partition = &lineage.partitions[group];
            let counter = partition.pages.values().find_map(|page| match page {
                PageRef::Keyed {
                    page_index, page, ..
                } if *page_index == row / 1024 => Some(page.value.values[row % 1024]),
                _ => None,
            });
            let Some(super::core::KeyedValue::UInt64(counter)) = counter else {
                return Err(mismatch("layout-7 counter is absent".into()));
            };
            counters.push(counter);
            let ring = partition.pages.values().find_map(|page| match page {
                PageRef::Ring {
                    row: found, page, ..
                } if *found == *row => Some(page),
                _ => None,
            });
            let ring = ring.ok_or_else(|| mismatch("layout-7 ring is absent".into()))?;
            values.extend(ring.value.ring.history());
            offsets.push(values.len());
            capacities.push(ring.value.ring.capacity());
            numeric.push(
                ring.value
                    .numeric
                    .ok_or_else(|| mismatch("numeric group is absent".into()))?,
            );
        }
        keyed.push(KeyedSnapshot {
            name: schema.counter_name,
            kind: super::ValueKind::UInt64,
            values: Arc::new(UInt64Array::from(counters)),
        });
        rings.push(RingSnapshot {
            name: schema.numeric_name,
            kind: super::RingKind::Float64,
            retention: schema.retention,
            values: ring_array(super::RingKind::Float64, &values),
            offsets,
            capacities,
            numeric: Some(numeric),
        });
    }
    table.from_v4_lineage(keys, keyed, rings, lineage, next_entity)
}

fn page_object(page: &PageRef) -> Result<NewObject> {
    let id = page.id();
    let payload = page_payload(page)?;
    Ok(NewObject {
        kind: ObjectKind::Page,
        owner: "rolling".into(),
        object_id: format!("{}/{}/{}", id.key_group.0, id.slot, id.generation),
        generation: id.generation,
        payload: ObjectPayload::Arrow(payload),
    })
}

fn page_entry(page: &PageRef, handle: V4Handle) -> Result<PageEntry> {
    let (column, kind, range) = page_location(page)?;
    Ok(PageEntry {
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
    })
}

fn page_location(page: &PageRef) -> Result<(String, &'static str, (u64, u64))> {
    match page {
        PageRef::Metadata(page) => Ok((
            "metadata".into(),
            "metadata",
            (0, page.value.entities.len() as u64),
        )),
        PageRef::Keyed {
            column: column_index,
            page,
            ..
        } => Ok((
            column.to_string(),
            "keyed",
            (0, page.value.values.len() as u64),
        )),
        PageRef::Ring {
            column: column_index,
            page,
            ..
        } => Ok((column.to_string(), "ring", (0, page.value.ring.len as u64))),
        PageRef::ValueLog {
            column: column_index,
            page,
            ..
        } => Ok((
            column.to_string(),
            "value_log",
            (
                u64::from(page.value.start),
                u64::from(page.value.start + page.value.len),
            ),
        )),
    }
}

fn page_payload(page: &PageRef) -> Result<ArrowPayload> {
    let mut metadata = page_metadata(page)?;
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
            if let Some(numeric) = page.value.numeric {
                metadata.extend(numeric_metadata(&numeric.layout7()));
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
    Ok(ArrowPayload {
        fields,
        columns,
        metadata,
    })
}

fn page_metadata(page: &PageRef) -> Result<HashMap<String, String>> {
    let (column, kind, _) = page_location(page)?;
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
    Ok(metadata)
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
    let kind = match record.column(0).data_type() {
        DataType::UInt64 => super::ValueKind::UInt64,
        DataType::Int64 => super::ValueKind::Int64,
        DataType::Float64 => super::ValueKind::Float64,
        DataType::Boolean => super::ValueKind::Boolean,
        _ => return Err(mismatch("keyed page type is unsupported".into())),
    };
    if (0..record.num_rows()).any(|row| !record.column(0).is_valid(row)) {
        return Err(mismatch("keyed page contains a null value".into()));
    }
    let values = match kind {
        super::ValueKind::UInt64 => column::<UInt64Array>(record, 0)?
            .values()
            .iter()
            .copied()
            .map(super::core::KeyedValue::UInt64)
            .collect(),
        super::ValueKind::Int64 => column::<Int64Array>(record, 0)?
            .values()
            .iter()
            .copied()
            .map(super::core::KeyedValue::Int64)
            .collect(),
        super::ValueKind::Float64 => column::<Float64Array>(record, 0)?
            .values()
            .iter()
            .copied()
            .map(super::core::KeyedValue::Float64)
            .collect(),
        super::ValueKind::Boolean => column::<BooleanArray>(record, 0)?
            .values()
            .iter()
            .map(super::core::KeyedValue::Boolean)
            .collect(),
    };
    Ok(PageRef::Keyed {
        column: column_index,
        page_index,
        page: super::pages::Page::new(page_id(entry), super::core::KeyedPage { kind, values }),
    })
}

fn decode_ring(record: &RecordBatch, entry: &PageEntry) -> Result<PageRef> {
    let column_index = entry
        .column
        .parse::<usize>()
        .map_err(|_| mismatch("ring column is invalid".into()))?;
    let row = parse_metadata::<usize>(record.schema().metadata(), "row")?;
    let kind = match record.column(0).data_type() {
        DataType::Float64 => super::RingKind::Float64,
        DataType::Int64 => super::RingKind::Int64,
        DataType::UInt64 => super::RingKind::UInt64,
        DataType::Struct(_) => super::RingKind::PairFloat64,
        _ => return Err(mismatch("ring page type is unsupported".into())),
    };
    let values = decode_ring_values(record, kind)?;
    let schema = record.schema();
    let metadata = schema.metadata();
    let head = parse_metadata(metadata, "head")?;
    let len = parse_metadata(metadata, "len")?;
    let capacity = parse_metadata(metadata, "capacity")?;
    if len > capacity || values.len() != len {
        return Err(mismatch("ring length exceeds its capacity".into()));
    }
    let mut ring = super::core::Ring {
        head,
        len,
        slots: vec![kind.empty(); capacity],
    };
    for (offset, value) in values.into_iter().enumerate() {
        let position = ring.position(offset);
        ring.slots[position] = value;
    }
    let numeric = decode_numeric_metadata(metadata)?;
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

fn decode_ring_values(
    record: &RecordBatch,
    kind: super::RingKind,
) -> Result<Vec<super::core::RingValue>> {
    match kind {
        super::RingKind::Float64 => Ok(column::<Float64Array>(record, 0)?
            .iter()
            .map(super::core::RingValue::Float64)
            .collect()),
        super::RingKind::Int64 => Ok(column::<Int64Array>(record, 0)?
            .iter()
            .map(super::core::RingValue::Int64)
            .collect()),
        super::RingKind::UInt64 => Ok(column::<UInt64Array>(record, 0)?
            .iter()
            .map(super::core::RingValue::UInt64)
            .collect()),
        super::RingKind::PairFloat64 => {
            let pairs = record
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::StructArray>()
                .ok_or_else(|| mismatch("pair ring page type mismatch".into()))?;
            let left = pairs
                .column(0)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| mismatch("pair ring left type mismatch".into()))?;
            let right = pairs
                .column(1)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| mismatch("pair ring right type mismatch".into()))?;
            Ok((0..pairs.len())
                .map(|row| {
                    super::core::RingValue::PairFloat64(
                        (left.is_valid(row) && right.is_valid(row))
                            .then_some((left.value(row), right.value(row))),
                    )
                })
                .collect())
        }
    }
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
    let shift = parse_metadata::<String>(metadata, "shift")?;
    let shift = if shift.is_empty() {
        None
    } else {
        Some(f64::from_bits(parse_metadata::<u64>(metadata, "shift")?))
    };
    Ok(Some(NumericAccumulator::from_layout7(
        NumericAccumulatorLayout7 {
            valid_count: parse_metadata(metadata, "valid_count")?,
            finite_count: parse_metadata(metadata, "finite_count")?,
            nan_count: parse_metadata(metadata, "nan_count")?,
            pos_inf: parse_metadata(metadata, "pos_inf")?,
            neg_inf: parse_metadata(metadata, "neg_inf")?,
            sum_value: f64::from_bits(parse_metadata(metadata, "sum_value")?),
            sum_error: f64::from_bits(parse_metadata(metadata, "sum_error")?),
            sum_enabled: parse_metadata(metadata, "sum_enabled")?,
            shift,
            offsets_value: f64::from_bits(parse_metadata(metadata, "offsets_value")?),
            offsets_error: f64::from_bits(parse_metadata(metadata, "offsets_error")?),
            squares_value: f64::from_bits(parse_metadata(metadata, "squares_value")?),
            squares_error: f64::from_bits(parse_metadata(metadata, "squares_error")?),
            dispersion_enabled: parse_metadata(metadata, "dispersion_enabled")?,
            since_rebuild: parse_metadata(metadata, "since_rebuild")?,
            n_prev: parse_metadata(metadata, "n_prev")?,
        },
    )))
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

        let stored_manifest = store.read_manifest(&persisted.manifest_path).unwrap();
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
        let stored_manifest = store.read_manifest(&persisted.manifest_path).unwrap();
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
