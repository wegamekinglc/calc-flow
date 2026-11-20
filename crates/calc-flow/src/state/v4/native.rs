use super::{
    EncodedObject, NewObject, ObjectKind, ObjectPayload, OperatorV4, Result, V4Handle, V4Manifest,
    V4Store, mismatch,
};
use crate::state::manifest::OperatorIngressManifestEntry;
use crate::state::table::v4::{LogicalRange, PageEntry, PartitionIndex, V4PageId};
use crate::{EventTime, JsonMap, OperatorStateSnapshot, StateSegment};
use datafusion::arrow::{array::Array, record_batch::RecordBatch};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem::{size_of, size_of_val},
};

const METADATA_COLUMN: &str = "__cf_operator_metadata";
const MAX_METADATA_BYTES: usize = 64 * 1024;

#[async_trait::async_trait]
pub(crate) trait V4ObjectStore {
    async fn write_objects(
        &mut self,
        epoch: u64,
        objects: Vec<NewObject>,
    ) -> Result<Vec<EncodedObject>>;
    fn record_carried(&mut self, count: usize);
    fn read_epoch_manifest(&self, epoch: u64) -> Result<Option<V4Manifest>>;
    fn read_object(&mut self, handle: &V4Handle) -> Result<ObjectPayload>;
}

#[async_trait::async_trait]
impl V4ObjectStore for V4Store {
    async fn write_objects(
        &mut self,
        epoch: u64,
        objects: Vec<NewObject>,
    ) -> Result<Vec<EncodedObject>> {
        V4Store::write_objects(self, epoch, objects).await
    }

    fn record_carried(&mut self, count: usize) {
        V4Store::record_carried(self, count);
    }

    fn read_epoch_manifest(&self, epoch: u64) -> Result<Option<V4Manifest>> {
        V4Store::read_epoch_manifest(self, epoch)
    }

    fn read_object(&mut self, handle: &V4Handle) -> Result<ObjectPayload> {
        V4Store::read_object(self, handle)
    }
}

mod pages;
pub(crate) use pages::{NativePageChange, NativePageSet};
#[cfg(test)]
pub(crate) use pages::{materializations, reset_materializations};

#[derive(Clone, Debug)]
pub(crate) struct NativeCapture {
    metadata: JsonMap,
    fingerprint: String,
    partitions: Vec<NativePartition>,
}

impl NativeCapture {
    pub(crate) fn new(metadata: JsonMap, mut partitions: Vec<NativePartition>) -> Result<Self> {
        let fingerprint = metadata_fingerprint(&metadata)?;
        partitions.sort_by_key(|partition| partition.key_group);
        let mut groups = BTreeSet::new();
        let mut columns = BTreeSet::new();
        if partitions.is_empty()
            || partitions.iter().any(|partition| {
                !groups.insert(partition.key_group)
                    || partition
                        .pages
                        .changes()
                        .iter()
                        .any(|change| !columns.insert(change.column().to_owned()))
            })
        {
            return Err(mismatch(
                "native state partitions or page names are empty or duplicated".into(),
            ));
        }
        Ok(Self {
            metadata,
            fingerprint,
            partitions,
        })
    }

    pub(crate) fn metadata(&self) -> JsonMap {
        self.metadata.clone()
    }

    pub(crate) fn with_metadata(mut self, metadata: JsonMap) -> Result<Self> {
        self.fingerprint = metadata_fingerprint(&metadata)?;
        self.metadata = metadata;
        Ok(self)
    }

    #[cfg(not(test))]
    pub(crate) fn into_snapshot(self) -> Result<OperatorStateSnapshot> {
        let Self {
            metadata,
            partitions,
            ..
        } = self;
        let mut segments = BTreeMap::new();
        for page in partitions
            .into_iter()
            .flat_map(|partition| partition.pages.materialize())
        {
            let bytes = super::object::payload_bytes(&page.payload)?;
            if segments
                .insert(page.column, StateSegment::new(bytes))
                .is_some()
            {
                return Err(mismatch(
                    "native snapshot page identity is duplicated".into(),
                ));
            }
        }
        Ok(OperatorStateSnapshot {
            inline_metadata: metadata,
            segments,
        })
    }

    pub(crate) fn reservation_bytes(&self) -> Result<u64> {
        let metadata = serde_json::to_value(&self.metadata)
            .map_err(|error| mismatch(format!("native metadata is not serializable: {error}")))?;
        let mut bytes = add_charge(0, crate::canonical_json(&metadata)?.len())?;
        for partition in &self.partitions {
            bytes = add_charge(bytes, size_of::<NativePartition>())?;
            bytes = bytes
                .checked_add(partition.pages.reservation_bytes())
                .ok_or_else(|| mismatch("native capture reservation overflow".into()))?;
        }
        bytes
            .checked_mul(2)
            .ok_or_else(|| mismatch("native capture reservation overflow".into()))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NativePartition {
    pub(crate) key_group: u32,
    pub(crate) pages: NativePageSet,
}

#[derive(Clone, Debug)]
pub(crate) struct NativePage {
    column: String,
    kind: &'static str,
    generation: u64,
    range: std::ops::Range<u64>,
    payload: ObjectPayload,
}

impl NativePage {
    pub(crate) fn arrow(
        column: impl Into<String>,
        record: &RecordBatch,
        generation: u64,
        range: std::ops::Range<u64>,
    ) -> Self {
        let schema = record.schema();
        Self {
            column: column.into(),
            kind: "arrow",
            generation,
            range,
            payload: ObjectPayload::Arrow(super::ArrowPayload {
                fields: schema
                    .fields()
                    .iter()
                    .map(|field| field.as_ref().clone())
                    .collect(),
                columns: record.columns().to_vec(),
                metadata: schema.metadata().clone(),
            }),
        }
    }

    pub(crate) fn json(
        column: impl Into<String>,
        value: serde_json::Value,
        generation: u64,
        range: std::ops::Range<u64>,
    ) -> Self {
        Self {
            column: column.into(),
            kind: "json",
            generation,
            range,
            payload: ObjectPayload::Json(value),
        }
    }

    fn reservation_bytes(&self) -> Result<u64> {
        let mut bytes = add_charge(0, size_of::<NativePage>())?;
        bytes = add_charge(bytes, self.column.len())?;
        match &self.payload {
            ObjectPayload::Bytes(payload) => add_charge(bytes, payload.len()),
            ObjectPayload::Json(payload) => {
                add_charge(bytes, crate::canonical_json(payload)?.len())
            }
            ObjectPayload::Arrow(payload) => arrow_reservation_bytes(payload, bytes),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeMetadata {
    state_fingerprint: String,
    inline_metadata: JsonMap,
}

#[derive(Default)]
struct PreviousPartition {
    pages: BTreeMap<String, PageEntry>,
    next_generations: BTreeMap<u32, u64>,
    free_slots: BTreeSet<u32>,
    used_slots: BTreeSet<u32>,
    next_slot: u32,
}

impl PreviousPartition {
    fn new(previous: Option<&PartitionIndex>) -> Result<Self> {
        let Some(previous) = previous else {
            return Ok(Self::default());
        };
        let mut pages = BTreeMap::new();
        for page in &previous.pages {
            if pages.insert(page.column.clone(), page.clone()).is_some() {
                return Err(mismatch("previous native page column is duplicated".into()));
            }
        }
        let next_slot = previous
            .next_generations
            .keys()
            .chain(previous.pages.iter().map(|page| &page.page_id.slot))
            .max()
            .copied()
            .map_or(Ok(0), |slot| {
                slot.checked_add(1)
                    .ok_or_else(|| mismatch("native partition slot space exhausted".into()))
            })?;
        Ok(Self {
            pages,
            next_generations: previous.next_generations.clone(),
            free_slots: previous.free_slots.iter().copied().collect(),
            used_slots: BTreeSet::new(),
            next_slot,
        })
    }

    fn assign(&mut self, page: &NativePage) -> Result<(u32, Option<PageEntry>)> {
        if let Some(previous) = self.pages.get(&page.column).cloned() {
            let slot = previous.page_id.slot;
            self.used_slots.insert(slot);
            let next = self.next_generations.get(&slot).copied().unwrap_or(0);
            if previous.page_id.generation == page.generation {
                if previous.kind != page.kind
                    || previous.logical_range.start != page.range.start
                    || previous.logical_range.end != page.range.end
                {
                    return Err(mismatch("reused native page geometry differs".into()));
                }
                return Ok((slot, Some(previous)));
            }
            if page.generation < next {
                return Err(mismatch("native page generation regressed".into()));
            }
            self.pages.remove(&page.column);
            self.next_generations.insert(
                slot,
                page.generation
                    .checked_add(1)
                    .ok_or_else(|| mismatch("native page generation overflow".into()))?,
            );
            return Ok((slot, None));
        }
        let slot = self.next_slot;
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .ok_or_else(|| mismatch("native partition slot space exhausted".into()))?;
        self.next_generations.insert(
            slot,
            page.generation
                .checked_add(1)
                .ok_or_else(|| mismatch("native page generation overflow".into()))?,
        );
        Ok((slot, None))
    }

    fn remove(&mut self, column: &str) -> Result<()> {
        let previous = self
            .pages
            .remove(column)
            .ok_or_else(|| mismatch("removed native page is absent".into()))?;
        self.used_slots.remove(&previous.page_id.slot);
        self.free_slots.insert(previous.page_id.slot);
        Ok(())
    }

    fn retain_all(&mut self) {
        self.used_slots
            .extend(self.pages.values().map(|page| page.page_id.slot));
    }

    fn finish(self) -> (BTreeMap<u32, u64>, Vec<u32>) {
        let mut free_slots = self.free_slots;
        free_slots.extend(
            self.pages
                .values()
                .filter(|page| !self.used_slots.contains(&page.page_id.slot))
                .map(|page| page.page_id.slot),
        );
        (self.next_generations, free_slots.into_iter().collect())
    }
}

pub(crate) async fn persist_native<S: V4ObjectStore>(
    store: &mut S,
    owner: &str,
    capture: NativeCapture,
    epoch: u64,
    progress: BTreeMap<String, OperatorIngressManifestEntry>,
    output_frontier: Option<EventTime>,
) -> Result<OperatorV4> {
    if epoch == 0 {
        return Err(mismatch("native state epoch must be nonzero".into()));
    }
    let NativeCapture {
        metadata,
        fingerprint,
        partitions,
    } = capture;
    let metadata = NativeMetadata {
        state_fingerprint: fingerprint.clone(),
        inline_metadata: metadata,
    };
    let metadata = serde_json::to_value(metadata)
        .map_err(|error| mismatch(format!("native metadata encoding failed: {error}")))?;
    let previous = previous_native_partitions(store, owner, epoch)?;
    let mut roots = Vec::with_capacity(partitions.len());
    let mut carried = 0usize;
    for partition in partitions {
        let (index, partition_carried) = persist_partition(
            store,
            owner,
            partition.key_group,
            partition.pages,
            &metadata,
            epoch,
            previous.get(&partition.key_group),
        )
        .await?;
        carried = carried
            .checked_add(partition_carried)
            .ok_or_else(|| mismatch("carried native page count overflow".into()))?;
        roots.push(NewObject {
            kind: ObjectKind::PartitionIndex,
            owner: owner.into(),
            object_id: partition.key_group.to_string(),
            generation: epoch,
            payload: ObjectPayload::Json(
                serde_json::to_value(index)
                    .map_err(|error| mismatch(format!("partition index failed: {error}")))?,
            ),
        });
    }
    let roots = store.write_objects(epoch, roots).await?;
    store.record_carried(carried);
    let partitions = roots
        .into_iter()
        .map(|root| {
            let key_group = root
                .handle
                .object_id
                .parse::<u32>()
                .map_err(|_| mismatch("partition root key group is invalid".into()))?;
            Ok(super::manifest::PartitionV4 {
                key_group,
                root: root.handle,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(OperatorV4::Table {
        progress,
        output_frontier,
        state_fingerprint: fingerprint,
        partitions,
    })
}

fn previous_native_partitions<S: V4ObjectStore>(
    store: &mut S,
    owner: &str,
    epoch: u64,
) -> Result<BTreeMap<u32, PartitionIndex>> {
    if epoch == 1 {
        return Ok(BTreeMap::new());
    }
    let previous_epoch = epoch - 1;
    let manifest = store
        .read_epoch_manifest(previous_epoch)?
        .ok_or_else(|| mismatch("previous native manifest is absent".into()))?;
    let operator = manifest
        .operators
        .get(owner)
        .ok_or_else(|| mismatch("previous native operator is absent".into()))?;
    let OperatorV4::Table { partitions, .. } = operator else {
        return Ok(BTreeMap::new());
    };
    partitions
        .iter()
        .map(|partition| {
            read_partition_index(store, partition, previous_epoch)
                .map(|index| (partition.key_group, index))
        })
        .collect()
}

async fn persist_partition<S: V4ObjectStore>(
    store: &mut S,
    owner: &str,
    key_group: u32,
    pages: NativePageSet,
    metadata: &serde_json::Value,
    epoch: u64,
    previous: Option<&PartitionIndex>,
) -> Result<(PartitionIndex, usize)> {
    if pages.is_incremental() {
        let previous = previous.ok_or_else(|| {
            mismatch("incremental native capture requires a previous partition".into())
        })?;
        return persist_delta_partition(store, owner, key_group, pages, metadata, epoch, previous)
            .await;
    }
    let pages = pages.materialize();
    let mut previous = PreviousPartition::new(previous)?;
    let (entries, carried) = write_assigned_pages(
        store,
        owner,
        key_group,
        pages,
        metadata,
        epoch,
        &mut previous,
    )
    .await?;
    finish_partition_index(previous, key_group, entries, carried, epoch)
}

async fn persist_delta_partition<S: V4ObjectStore>(
    store: &mut S,
    owner: &str,
    key_group: u32,
    pages: NativePageSet,
    metadata: &serde_json::Value,
    epoch: u64,
    previous: &PartitionIndex,
) -> Result<(PartitionIndex, usize)> {
    let mut previous = PreviousPartition::new(Some(previous))?;
    previous.retain_all();
    let mut upserts = Vec::new();
    for change in pages.changes() {
        match change {
            NativePageChange::Upsert { page, .. } => upserts.push(page.clone()),
            NativePageChange::Remove { column, .. } => previous.remove(column)?,
        }
    }
    let updated = upserts
        .iter()
        .map(|page| page.column.clone())
        .collect::<BTreeSet<_>>();
    let (mut entries, mut carried) = write_assigned_pages(
        store,
        owner,
        key_group,
        upserts,
        metadata,
        epoch,
        &mut previous,
    )
    .await?;
    let clean = previous
        .pages
        .values()
        .filter(|page| !updated.contains(&page.column))
        .cloned()
        .collect::<Vec<_>>();
    carried = carried
        .checked_add(clean.len())
        .ok_or_else(|| mismatch("carried native page count overflow".into()))?;
    entries.extend(clean);
    finish_partition_index(previous, key_group, entries, carried, epoch)
}

async fn write_assigned_pages<S: V4ObjectStore>(
    store: &mut S,
    owner: &str,
    key_group: u32,
    pages: Vec<NativePage>,
    metadata: &serde_json::Value,
    epoch: u64,
    previous: &mut PreviousPartition,
) -> Result<(Vec<PageEntry>, usize)> {
    let mut pending = BTreeMap::new();
    let mut objects = Vec::with_capacity(pages.len() + 1);
    let mut entries = Vec::with_capacity(pages.len() + 1);
    let mut carried = 0usize;
    let metadata = NativePage::json(METADATA_COLUMN, metadata.clone(), epoch, 0..0);
    for page in pages.into_iter().chain(std::iter::once(metadata)) {
        let (slot, reused) = previous.assign(&page)?;
        if let Some(entry) = reused {
            entries.push(entry);
            carried = carried
                .checked_add(1)
                .ok_or_else(|| mismatch("carried native page count overflow".into()))?;
            continue;
        }
        let generation = page.generation;
        let object_id = format!("{key_group}/{slot}/{generation}");
        pending.insert(
            object_id.clone(),
            (slot, generation, page.column, page.kind, page.range),
        );
        objects.push(NewObject {
            kind: ObjectKind::Page,
            owner: owner.into(),
            object_id,
            generation,
            payload: page.payload,
        });
    }
    let encoded = store.write_objects(epoch, objects).await?;
    entries.extend(
        encoded
            .into_iter()
            .map(|object| {
                let id = object.handle.object_id.clone();
                let (slot, generation, column, kind, range) = pending
                    .remove(&id)
                    .ok_or_else(|| mismatch("persisted page identity is unknown".into()))?;
                let byte_len = object.handle.byte_len;
                let sha256 = object.handle.sha256.clone();
                Ok(PageEntry {
                    page_id: V4PageId {
                        key_group,
                        slot,
                        generation,
                    },
                    column,
                    kind: kind.into(),
                    logical_range: LogicalRange {
                        start: range.start,
                        end: range.end,
                    },
                    byte_len,
                    sha256,
                    handle: object.handle,
                })
            })
            .collect::<Result<Vec<_>>>()?,
    );
    if !pending.is_empty() {
        return Err(mismatch(
            "persisted native page inventory is incomplete".into(),
        ));
    }
    Ok((entries, carried))
}

fn finish_partition_index(
    previous: PreviousPartition,
    key_group: u32,
    mut entries: Vec<PageEntry>,
    carried: usize,
    epoch: u64,
) -> Result<(PartitionIndex, usize)> {
    entries.sort_by_key(|page| page.page_id);
    let (next_generations, free_slots) = previous.finish();
    let index = PartitionIndex {
        format_version: 4,
        kind: "partition_index".into(),
        key_group,
        pages: entries,
        next_generations,
        free_slots,
    };
    index.validate(epoch)?;
    Ok((index, carried))
}

pub(crate) fn load_native(
    store: &mut V4Store,
    manifest: &V4Manifest,
    owner: &str,
) -> Result<OperatorStateSnapshot> {
    let Some(OperatorV4::Table {
        partitions,
        state_fingerprint,
        ..
    }) = manifest.operators.get(owner)
    else {
        return Err(mismatch(
            "native table operator is absent from the manifest".into(),
        ));
    };
    let mut metadata = None;
    let mut segments = BTreeMap::new();
    for partition in partitions {
        if partition.root.owner != owner
            || partition.root.object_id != partition.key_group.to_string()
        {
            return Err(mismatch("partition root owner or identity differs".into()));
        }
        let index = read_partition_index(store, partition, manifest.epoch)?;
        for page in index.pages {
            if page.handle.owner != owner {
                return Err(mismatch("native page owner differs".into()));
            }
            let ObjectPayload::Bytes(bytes) = store.read_object(&page.handle)? else {
                return Err(mismatch("native page is not a binary object".into()));
            };
            if page.column == METADATA_COLUMN {
                if bytes.len() > MAX_METADATA_BYTES || metadata.is_some() {
                    return Err(mismatch(
                        "native metadata is duplicated or too large".into(),
                    ));
                }
                let decoded: NativeMetadata = serde_json::from_slice(&bytes)
                    .map_err(|error| mismatch(format!("native metadata is invalid: {error}")))?;
                if decoded.state_fingerprint != *state_fingerprint
                    || metadata_fingerprint(&decoded.inline_metadata)? != *state_fingerprint
                {
                    return Err(mismatch("native metadata fingerprint differs".into()));
                }
                metadata = Some(decoded.inline_metadata);
                continue;
            }
            if !matches!(page.kind.as_str(), "arrow" | "json" | "bytes")
                || segments
                    .insert(page.column.clone(), StateSegment::new(bytes))
                    .is_some()
            {
                return Err(mismatch("native page kind or identity is invalid".into()));
            }
        }
    }
    Ok(OperatorStateSnapshot {
        inline_metadata: metadata
            .ok_or_else(|| mismatch("native metadata page is absent".into()))?,
        segments,
    })
}

pub(crate) fn native_partition_handles(
    store: &mut V4Store,
    partition: &super::manifest::PartitionV4,
    epoch: u64,
) -> Result<Vec<V4Handle>> {
    Ok(read_partition_index(store, partition, epoch)?
        .pages
        .into_iter()
        .map(|page| page.handle)
        .collect())
}

fn read_partition_index<S: V4ObjectStore>(
    store: &mut S,
    partition: &super::manifest::PartitionV4,
    epoch: u64,
) -> Result<PartitionIndex> {
    let ObjectPayload::Bytes(bytes) = store.read_object(&partition.root)? else {
        return Err(mismatch("partition root is not binary JSON".into()));
    };
    let index = PartitionIndex::from_bytes(&bytes, epoch)?;
    if index.key_group != partition.key_group {
        return Err(mismatch("partition root key group differs".into()));
    }
    Ok(index)
}

fn add_charge(total: u64, value: usize) -> Result<u64> {
    let value = u64::try_from(value)
        .map_err(|_| mismatch("native capture reservation exceeds u64".into()))?;
    total
        .checked_add(value)
        .ok_or_else(|| mismatch("native capture reservation overflow".into()))
}

fn arrow_reservation_bytes(payload: &super::ArrowPayload, mut bytes: u64) -> Result<u64> {
    bytes = add_charge(bytes, size_of_val(payload.fields.as_slice()))?;
    for field in &payload.fields {
        bytes = add_charge(bytes, field.name().len())?;
        bytes = map_reservation_bytes(field.metadata(), bytes)?;
    }
    for column in &payload.columns {
        bytes = add_charge(bytes, column.get_array_memory_size())?;
    }
    map_reservation_bytes(&payload.metadata, bytes)
}

fn map_reservation_bytes(
    values: &std::collections::HashMap<String, String>,
    mut bytes: u64,
) -> Result<u64> {
    for (key, value) in values {
        bytes = add_charge(bytes, size_of::<(String, String)>())?;
        bytes = add_charge(bytes, key.len())?;
        bytes = add_charge(bytes, value.len())?;
    }
    Ok(bytes)
}

fn metadata_fingerprint(metadata: &JsonMap) -> Result<String> {
    let bytes = crate::canonical_json(
        &serde_json::to_value(metadata)
            .map_err(|error| mismatch(format!("native metadata is not serializable: {error}")))?,
    )?;
    if bytes.len() > MAX_METADATA_BYTES {
        return Err(mismatch("native metadata exceeds 64 KiB".into()));
    }
    Ok(hex::encode(Sha256::digest(bytes.as_bytes())))
}

#[cfg(test)]
mod tests {
    use super::{
        NativeCapture, NativePage, NativePageChange, NativePageSet, NativePartition, load_native,
        persist_native,
    };
    use crate::state::v4::{
        JobKind, OperatorV4, V4Manifest, V4Store, manifest_fingerprint,
        native::native_partition_handles,
    };
    use datafusion::arrow::{
        array::{Array, ArrayRef, Int64Array},
        ipc::reader::FileReader,
        record_batch::RecordBatch,
    };
    use serde_json::Value;
    use std::{collections::BTreeMap, io::Cursor, sync::Arc};

    fn manifest_with_operator(operator: OperatorV4) -> V4Manifest {
        let mut manifest = V4Manifest::from_bytes(
            include_str!("../../../tests/fixtures/checkpoint-v4/manifest-v4.json").as_bytes(),
        )
        .unwrap();
        manifest.job_kind = JobKind::Runner;
        manifest.operators = BTreeMap::from([("join".to_owned(), operator)]);
        manifest.manifest_fingerprint = manifest_fingerprint(&manifest).unwrap();
        manifest
    }

    #[test]
    fn native_incremental_page_set_materializes_and_accounts_changes() {
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![7])) as ArrayRef,
        )])
        .unwrap();
        let left = NativePage::arrow("left", &record, 1, 0..1);
        let right = NativePage::arrow("right", &record, 1, 0..1);
        let base = NativePageSet::full(vec![left.clone(), right.clone()]).unwrap();
        let replacement = NativePage::arrow("left", &record, 2, 0..1);
        let pages = NativePageSet::new(
            Some(Arc::new(base)),
            vec![
                NativePageChange::upsert(replacement.clone(), Some(&left)).unwrap(),
                NativePageChange::remove(&right).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(pages.change_count(), 2);
        assert_eq!(
            pages.reservation_bytes(),
            replacement.reservation_bytes().unwrap()
        );
        let materialized = pages.materialize();
        assert_eq!(materialized.len(), 1);
        assert_eq!(materialized[0].column, "left");
        assert_eq!(materialized[0].generation, 2);
    }

    #[tokio::test]
    async fn native_partition_pages_roundtrip_and_validate_metadata() {
        let temporary = tempfile::tempdir().unwrap();
        let mut store = V4Store::new(temporary.path().join("lineage"));
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![7, 11])) as ArrayRef,
        )])
        .unwrap();
        let metadata = BTreeMap::from_iter([
            ("layout_version".into(), Value::from(12)),
            ("spec".into(), serde_json::json!({"test": true})),
        ]);
        let capture = NativeCapture::new(
            metadata.clone(),
            vec![NativePartition {
                key_group: 0,
                pages: NativePageSet::full(vec![
                    NativePage::arrow("left/run-0", &record, 0, 0..2),
                    NativePage::json("control", serde_json::json!({"rows": 2}), 0, 0..0),
                ])
                .unwrap(),
            }],
        )
        .unwrap();
        let operator = persist_native(&mut store, "join", capture, 1, BTreeMap::new(), None)
            .await
            .unwrap();
        let OperatorV4::Table {
            partitions,
            state_fingerprint,
            ..
        } = &operator
        else {
            panic!("native capture must produce table partitions")
        };
        assert_eq!(partitions.len(), 1);
        assert!(!state_fingerprint.is_empty());
        let manifest = manifest_with_operator(operator.clone());
        manifest.validate_structure().unwrap();
        let snapshot = load_native(&mut store, &manifest, "join").unwrap();
        assert_eq!(snapshot.inline_metadata, metadata);
        assert_eq!(snapshot.segments.len(), 2);
        let payload =
            FileReader::try_new(Cursor::new(snapshot.segments["left/run-0"].bytes()), None)
                .unwrap()
                .next()
                .unwrap()
                .unwrap();
        assert_eq!(
            payload
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .as_ref(),
            [7, 11]
        );
        let handles = native_partition_handles(&mut store, &partitions[0], 1).unwrap();
        assert_eq!(handles.len(), 3);

        let mut damaged = manifest.operators["join"].clone();
        let OperatorV4::Table {
            state_fingerprint, ..
        } = &mut damaged
        else {
            unreachable!()
        };
        *state_fingerprint = "damaged".into();
        let damaged = manifest_with_operator(damaged);
        assert!(load_native(&mut store, &damaged, "join").is_err());
    }

    #[tokio::test]
    async fn native_successive_epochs_preserve_lineage_and_carry_clean_pages() {
        let temporary = tempfile::tempdir().unwrap();
        let mut store = V4Store::new(temporary.path().join("lineage"));
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![7])) as ArrayRef,
        )])
        .unwrap();
        let capture = |epoch, pages: Vec<(&str, u64)>| {
            NativeCapture::new(
                BTreeMap::from([("epoch".into(), Value::from(epoch))]),
                vec![NativePartition {
                    key_group: 0,
                    pages: NativePageSet::full(
                        pages
                            .into_iter()
                            .map(|(name, generation)| {
                                NativePage::arrow(name, &record, generation, 0..1)
                            })
                            .chain(std::iter::once(NativePage::json(
                                "control",
                                Value::from(epoch),
                                epoch,
                                0..0,
                            )))
                            .collect(),
                    )
                    .unwrap(),
                }],
            )
            .unwrap()
        };
        let first = persist_native(
            &mut store,
            "join",
            capture(1, vec![("run-a", 7), ("run-b", 9)]),
            1,
            BTreeMap::new(),
            None,
        )
        .await
        .unwrap();
        let mut first = manifest_with_operator(first);
        first.epoch = 1;
        first.manifest_fingerprint = manifest_fingerprint(&first).unwrap();
        store.publish_manifest(&first).unwrap();
        let second = persist_native(
            &mut store,
            "join",
            capture(2, vec![("run-b", 9), ("run-c", 11)]),
            2,
            BTreeMap::new(),
            None,
        )
        .await
        .unwrap();
        let mut second = manifest_with_operator(second);
        second.epoch = 2;
        second.manifest_fingerprint = manifest_fingerprint(&second).unwrap();
        store.publish_manifest(&second).unwrap();

        let restored_first = load_native(&mut store, &first, "join").unwrap();
        assert_eq!(restored_first.inline_metadata["epoch"], 1);
        assert_eq!(
            restored_first.segments.keys().collect::<Vec<_>>(),
            ["control", "run-a", "run-b"]
        );
        let restored_second = load_native(&mut store, &second, "join").unwrap();
        assert_eq!(restored_second.inline_metadata["epoch"], 2);
        assert_eq!(
            restored_second.segments.keys().collect::<Vec<_>>(),
            ["control", "run-b", "run-c"]
        );
        let mut handle = |manifest: &V4Manifest, column: &str| {
            let OperatorV4::Table { partitions, .. } = &manifest.operators["join"] else {
                unreachable!()
            };
            super::read_partition_index(&mut store, &partitions[0], manifest.epoch)
                .unwrap()
                .pages
                .into_iter()
                .find(|page| page.column == column)
                .unwrap()
                .handle
        };
        assert_eq!(handle(&first, "run-b"), handle(&second, "run-b"));
        assert_ne!(
            handle(&first, "__cf_operator_metadata").object_id,
            handle(&second, "__cf_operator_metadata").object_id
        );
        assert_eq!(store.stats().carried_objects, 1);
    }
}
