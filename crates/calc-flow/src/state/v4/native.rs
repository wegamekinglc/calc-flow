use super::{
    NewObject, ObjectKind, ObjectPayload, OperatorV4, Result, V4Handle, V4Manifest, V4Store,
    mismatch,
};
use crate::state::manifest::OperatorIngressManifestEntry;
use crate::state::table::v4::{LogicalRange, PageEntry, PartitionIndex, V4PageId};
use crate::{EventTime, JsonMap, OperatorStateSnapshot, StateSegment};
use datafusion::arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const METADATA_COLUMN: &str = "__cf_operator_metadata";
const MAX_METADATA_BYTES: usize = 64 * 1024;

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
                    || partition.pages.iter().any(|page| {
                        page.range.start > page.range.end
                            || page.column == METADATA_COLUMN
                            || !columns.insert(page.column.clone())
                    })
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
}

#[derive(Clone, Debug)]
pub(crate) struct NativePartition {
    pub(crate) key_group: u32,
    pub(crate) pages: Vec<NativePage>,
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
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeMetadata {
    state_fingerprint: String,
    inline_metadata: JsonMap,
}

pub(crate) async fn persist_native(
    store: &mut V4Store,
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
    let mut roots = Vec::with_capacity(partitions.len());
    for partition in partitions {
        let index = persist_partition(
            store,
            owner,
            partition.key_group,
            partition.pages,
            &metadata,
            epoch,
        )
        .await?;
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

async fn persist_partition(
    store: &mut V4Store,
    owner: &str,
    key_group: u32,
    pages: Vec<NativePage>,
    metadata: &serde_json::Value,
    epoch: u64,
) -> Result<PartitionIndex> {
    let mut pending = BTreeMap::new();
    let mut objects = Vec::with_capacity(pages.len() + 1);
    let mut next_generations = BTreeMap::new();
    for (slot, page) in pages.into_iter().enumerate() {
        let slot = u32::try_from(slot).map_err(|_| mismatch("partition slot overflow".into()))?;
        let generation = page.generation;
        next_generations.insert(
            slot,
            generation
                .checked_add(1)
                .ok_or_else(|| mismatch("page generation overflow".into()))?,
        );
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
    let metadata_slot = u32::try_from(pending.len())
        .map_err(|_| mismatch("partition metadata slot overflow".into()))?;
    next_generations.insert(metadata_slot, 1);
    let metadata_id = format!("{key_group}/{metadata_slot}/0");
    let metadata = NativePage::json(METADATA_COLUMN, metadata.clone(), 0, 0..0);
    pending.insert(
        metadata_id.clone(),
        (
            metadata_slot,
            0,
            metadata.column.clone(),
            metadata.kind,
            metadata.range.clone(),
        ),
    );
    objects.push(NewObject {
        kind: ObjectKind::Page,
        owner: owner.into(),
        object_id: metadata_id,
        generation: 0,
        payload: metadata.payload,
    });
    let encoded = store.write_objects(epoch, objects).await?;
    let mut pages = encoded
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
        .collect::<Result<Vec<_>>>()?;
    pages.sort_by_key(|page| page.page_id);
    let index = PartitionIndex {
        format_version: 4,
        kind: "partition_index".into(),
        key_group,
        pages,
        next_generations,
        free_slots: Vec::new(),
    };
    index.validate(epoch)?;
    Ok(index)
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

fn read_partition_index(
    store: &mut V4Store,
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
    use super::{NativeCapture, NativePage, NativePartition, load_native, persist_native};
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
                pages: vec![
                    NativePage::arrow("left/run-0", &record, 0, 0..2),
                    NativePage::json("control", serde_json::json!({"rows": 2}), 0, 0..0),
                ],
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
}
