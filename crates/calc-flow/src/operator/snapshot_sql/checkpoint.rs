use std::{collections::BTreeMap, io::Cursor, sync::Arc};

use datafusion::{
    arrow::{
        datatypes::SchemaRef,
        ipc::{convert::IpcSchemaEncoder, reader::FileReader, writer::DictionaryTracker},
    },
    execution::memory_pool::MemoryReservation,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::{
    Relation, SnapshotSqlOperator, add, backing_bytes, check_rows, engine_error, ipc, metadata,
    state_error,
};
use crate::{Batch, DataFusionConfig, JsonMap, OperatorMetadata, OperatorStateSnapshot, Result};
use crate::{BatchMetadata, StateSegment};

#[derive(Clone)]
pub(crate) struct PreparedSnapshotCapture {
    pub(crate) snapshot: OperatorStateSnapshot,
}

pub(crate) struct PreparedSnapshotRestore {
    relations: Vec<Option<Arc<Relation>>>,
    capture: PreparedSnapshotCapture,
}

#[derive(Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Identity {
    declaration_sha256: String,
    inputs: BTreeMap<String, String>,
    output: String,
    runtime_config: DataFusionConfig,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RelationDescriptor {
    data_sha256: String,
    metadata_sha256: String,
    rows: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Control {
    state_kind: String,
    state_layout: u32,
    state_accounting: u32,
    datafusion_version: String,
    identity: Identity,
    aliases: Vec<Option<usize>>,
    relations: Vec<RelationDescriptor>,
}

impl SnapshotSqlOperator {
    fn descriptor_bound(&self) -> Result<usize> {
        let mut bound = add(
            8192,
            self.query_text()
                .len()
                .checked_mul(16)
                .ok_or_else(|| state_error("query size overflow"))?,
        )?;
        for port in self.input_ports().iter().chain(self.output_ports()) {
            let schema = port
                .schema()
                .ok_or_else(|| state_error("snapshot_sql requires exact schemas"))?;
            bound = add(
                bound,
                add(port.name().len(), ipc::schema_bytes(schema)?)?
                    .checked_mul(32)
                    .ok_or_else(|| state_error("schema size overflow"))?,
            )?;
        }
        for udf in self.udf_references() {
            bound = add(
                bound,
                add(
                    add(udf.provider().len(), udf.name().len())?,
                    udf.version().len(),
                )?
                .checked_mul(32)
                .ok_or_else(|| state_error("UDF size overflow"))?,
            )?;
        }
        Ok(bound)
    }

    fn identity(&mut self) -> Result<Identity> {
        let configuration = serde_json::to_vec(&self.configuration()).map_err(engine_error)?;
        let inputs = self
            .input_ports()
            .iter()
            .map(|p| {
                Ok((
                    p.name().to_owned(),
                    schema_digest(
                        p.schema()
                            .ok_or_else(|| state_error("missing snapshot input schema"))?,
                    ),
                ))
            })
            .collect::<Result<_>>()?;
        let output = schema_digest(
            self.output_ports()[0]
                .schema()
                .ok_or_else(|| state_error("missing snapshot output schema"))?,
        );
        Ok(Identity {
            declaration_sha256: hex::encode(Sha256::digest(configuration)),
            inputs,
            output,
            runtime_config: self.runtime.runtime()?.compact_runtime_config(),
        })
    }

    pub(crate) fn reserve_recovery_envelope(
        &mut self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<MemoryReservation> {
        self.ensure_pool()?;
        let mut bound = self.descriptor_bound()?;
        for name in snapshot.segments.keys() {
            bound = add(bound, add(name.len(), 512)?)?;
        }
        bound = add(
            bound,
            crate::json::owned_json_bytes(&snapshot.inline_metadata)
                .ok_or_else(|| state_error("inline metadata size overflow"))?,
        )?;
        self.scratch(bound)
    }

    pub(crate) fn reserve_checkpoint_envelope(
        &mut self,
        node_len: usize,
    ) -> Result<MemoryReservation> {
        self.ensure_pool()?;
        self.scratch(add(self.descriptor_bound()?, node_len)?)
    }

    pub(crate) fn prepare_checkpoint_work(
        &mut self,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedSnapshotCapture> {
        check()?;
        self.ensure_pool()?;
        if let Some(capture) = &self.capture {
            return Ok(capture.clone());
        }
        let _scratch = self.scratch(self.descriptor_bound()?)?;
        let identity = self.identity()?;
        let mut owners = Vec::<Arc<Relation>>::new();
        let aliases = self
            .relations
            .iter()
            .map(|r| {
                r.as_ref().map(|r| {
                    if let Some(i) = owners.iter().position(|old| Arc::ptr_eq(old, r)) {
                        i
                    } else {
                        owners.push(r.clone());
                        owners.len() - 1
                    }
                })
            })
            .collect();
        let mut segments = BTreeMap::new();
        let mut relations = Vec::new();
        for (index, relation) in owners.iter().enumerate() {
            check()?;
            let name = self.name().to_owned();
            let scratch = self.runtime.runtime()?.incremental_reservation(&name);
            let data =
                ipc::encode_with_owner(&relation.batch, scratch, Some(self.reservation()), check)?;
            relations.push(RelationDescriptor {
                data_sha256: data.segment.sha256().into(),
                metadata_sha256: relation.metadata.segment.sha256().into(),
                rows: relation.batch.num_rows(),
            });
            segments.insert(
                format!("relation-{index}"),
                tag_segment(
                    data.segment.clone(),
                    self.pool.as_ref().expect("pool"),
                    None,
                ),
            );
            segments.insert(
                format!("metadata-{index}"),
                relation.metadata.segment.clone(),
            );
        }
        let control = Control {
            state_kind: "snapshot_sql".into(),
            state_layout: 1,
            state_accounting: 1,
            datafusion_version: datafusion::DATAFUSION_VERSION.into(),
            identity,
            aliases,
            relations,
        };
        let segment = tag_segment(
            paid_json(&control, self.reservation())?,
            self.pool.as_ref().expect("pool"),
            None,
        );
        let inline_metadata = inline(&segment);
        segments.insert("control".into(), segment);
        check()?;
        Ok(PreparedSnapshotCapture {
            snapshot: OperatorStateSnapshot {
                inline_metadata,
                segments,
            },
        })
    }

    pub(crate) fn install_checkpoint_work(&mut self, capture: PreparedSnapshotCapture) {
        self.capture = Some(capture);
    }

    pub(crate) fn prepare_restore(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedSnapshotRestore> {
        check()?;
        self.ensure_pool()?;
        if snapshot.segments.len()
            > add(
                1,
                self.relations
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| state_error("alias count overflow"))?,
            )?
        {
            return Err(state_error("unknown snapshot_sql segments"));
        }
        let control_segment = snapshot
            .segments
            .get("control")
            .ok_or_else(|| state_error("missing snapshot_sql control segment"))?;
        let control_bound = add(
            8192,
            control_segment
                .bytes()
                .len()
                .checked_mul(64)
                .ok_or_else(|| state_error("control size overflow"))?,
        )?;
        let _scratch = self.scratch(add(control_bound, self.descriptor_bound()?)?)?;
        let value = crate::json::parse_json_value(control_segment.bytes(), "snapshot_sql control")?;
        let control: Control = serde_json::from_value(value).map_err(engine_error)?;
        self.validate_control(snapshot, &control)?;
        let mut paid = OperatorStateSnapshot {
            inline_metadata: snapshot.inline_metadata.clone(),
            segments: BTreeMap::new(),
        };
        let mut owners = Vec::new();
        for (index, descriptor) in control.relations.iter().enumerate() {
            check()?;
            let data_name = format!("relation-{index}");
            let metadata_name = format!("metadata-{index}");
            let data = &snapshot.segments[&data_name];
            let metadata = &snapshot.segments[&metadata_name];
            if descriptor.data_sha256 != data.sha256()
                || descriptor.metadata_sha256 != metadata.sha256()
            {
                return Err(state_error("snapshot_sql segment digest mismatch"));
            }
            let owned_data =
                paid_segment(data, self.reservation(), self.pool.as_ref().expect("pool"))?;
            let name = self.name().to_owned();
            let metadata_scratch = self.runtime.runtime()?.incremental_reservation(&name);
            let (batch_metadata, _decoded_metadata) =
                metadata::decode_with(metadata, &name, metadata_scratch)?;
            let metadata_reservation =
                metadata::reserve_with(&batch_metadata, &name, self.reservation())?;
            let owned_metadata = metadata::encode(&batch_metadata, metadata_reservation, check)?;
            let relation = self.decode_relation(data, batch_metadata, owned_metadata, check)?;
            if relation.batch.num_rows() != descriptor.rows {
                return Err(state_error("snapshot_sql row count mismatch"));
            }
            owners.push(relation);
            paid.segments.insert(data_name, owned_data);
            paid.segments
                .insert(metadata_name, owners[index].metadata.segment.clone());
        }
        let relations = control
            .aliases
            .iter()
            .map(|i| i.map(|i| owners[i].clone()))
            .collect::<Vec<_>>();
        for (port, relation) in self.input_ports().iter().zip(&relations) {
            if let Some(r) = relation {
                port.validate(&r.batch, "snapshot_sql restore")?;
            }
        }
        check_rows(&relations, self.budget)?;
        paid.segments.insert(
            "control".into(),
            paid_segment(
                control_segment,
                self.reservation(),
                self.pool.as_ref().expect("pool"),
            )?,
        );
        check()?;
        Ok(PreparedSnapshotRestore {
            relations,
            capture: PreparedSnapshotCapture { snapshot: paid },
        })
    }

    fn validate_control(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        control: &Control,
    ) -> Result<()> {
        if control.state_kind != "snapshot_sql"
            || control.state_layout != 1
            || control.state_accounting != 1
            || control.datafusion_version != datafusion::DATAFUSION_VERSION
            || control.identity != self.identity()?
            || control.aliases.len() != self.relations.len()
            || snapshot.inline_metadata != inline(&snapshot.segments["control"])
        {
            return Err(state_error(
                "snapshot_sql identity/layout/accounting mismatch",
            ));
        }
        let mut inventory = std::collections::BTreeSet::from(["control".to_owned()]);
        let mut reached = std::collections::BTreeSet::new();
        let mut rows = 0usize;
        for i in control.aliases.iter().flatten() {
            let descriptor = control
                .relations
                .get(*i)
                .ok_or_else(|| state_error("invalid snapshot_sql relation reference"))?;
            reached.insert(*i);
            rows = add(rows, descriptor.rows)?;
        }
        if reached.len() != control.relations.len()
            || self.budget.is_some_and(|b| rows as u64 > b.max_rows())
        {
            return Err(state_error(
                "snapshot_sql restore exceeds max_rows or has unreachable relations",
            ));
        }
        for i in 0..control.relations.len() {
            inventory.insert(format!("relation-{i}"));
            inventory.insert(format!("metadata-{i}"));
        }
        if !snapshot.segments.keys().eq(inventory.iter()) {
            return Err(state_error("snapshot_sql segment inventory mismatch"));
        }
        Ok(())
    }

    fn decode_relation(
        &mut self,
        segment: &StateSegment,
        metadata: BatchMetadata,
        owned_metadata: Arc<metadata::SqlMetadata>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Arc<Relation>> {
        // Our encoder emits uncompressed IPC. Reject compression before decode,
        // so encoded size provides a conservative allocation reservation.
        let bound = add(
            8192,
            segment
                .bytes()
                .len()
                .checked_mul(64)
                .ok_or_else(|| state_error("IPC size overflow"))?,
        )?;
        let _scratch = self.scratch(bound)?;
        validate_ipc(segment.bytes())?;
        let reader =
            FileReader::try_new(Cursor::new(segment.bytes()), None).map_err(engine_error)?;
        let mut records = Vec::new();
        for record in reader {
            check()?;
            records.push(record.map_err(engine_error)?);
        }
        let batch = Batch::table(records, metadata)?;
        if backing_bytes(&batch)? > bound {
            return Err(state_error(
                "snapshot_sql decoded backing exceeds prepaid bound",
            ));
        }
        self.restore_relation(batch, owned_metadata, check)
    }

    fn restore_relation(
        &self,
        batch: Batch,
        metadata: Arc<metadata::SqlMetadata>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Arc<Relation>> {
        let backing = self.reservation();
        backing
            .try_grow(backing_bytes(&batch)?)
            .map_err(engine_error)?;
        check()?;
        Ok(Arc::new(Relation {
            batch,
            metadata,
            backing: Arc::new(backing),
        }))
    }

    pub(crate) fn install_restore(&mut self, prepared: PreparedSnapshotRestore) {
        self.relations = prepared.relations;
        self.capture = Some(prepared.capture);
    }
}

fn schema_digest(schema: &SchemaRef) -> String {
    let mut tracker = DictionaryTracker::new(false);
    let bytes = IpcSchemaEncoder::new()
        .with_dictionary_tracker(&mut tracker)
        .schema_to_fb(schema);
    hex::encode(Sha256::digest(bytes.finished_data()))
}

fn inline(control: &StateSegment) -> JsonMap {
    BTreeMap::from([
        ("snapshot_sql_layout".into(), json!(1)),
        ("snapshot_sql_accounting".into(), json!(1)),
        ("control_sha256".into(), json!(control.sha256())),
    ])
}

struct SnapshotSegmentOwner {
    pool: Arc<crate::state_memory_pool::OwnedStatePool>,
    _previous: Option<Arc<dyn std::any::Any + Send + Sync>>,
    _reservation: Option<MemoryReservation>,
}

fn tag_segment(
    segment: StateSegment,
    pool: &Arc<crate::state_memory_pool::OwnedStatePool>,
    reservation: Option<MemoryReservation>,
) -> StateSegment {
    segment.map_owner(|previous| {
        Arc::new(SnapshotSegmentOwner {
            pool: pool.clone(),
            _previous: previous,
            _reservation: reservation,
        })
    })
}

fn paid_segment(
    segment: &StateSegment,
    reservation: MemoryReservation,
    pool: &Arc<crate::state_memory_pool::OwnedStatePool>,
) -> Result<StateSegment> {
    if segment
        .accounting_owner::<SnapshotSegmentOwner>()
        .is_some_and(|owner| Arc::ptr_eq(&owner.pool, pool))
    {
        return Ok(segment.clone());
    }
    reservation
        .try_grow(add(segment.bytes_arc().capacity(), 512)?)
        .map_err(engine_error)?;
    Ok(tag_segment(segment.clone(), pool, Some(reservation)))
}

struct Count(usize);
impl std::io::Write for Count {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("JSON size overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn paid_json(value: &impl Serialize, reservation: MemoryReservation) -> Result<StateSegment> {
    let mut count = Count(0);
    serde_json::to_writer(&mut count, value).map_err(engine_error)?;
    reservation
        .try_grow(add(
            count
                .0
                .checked_mul(2)
                .ok_or_else(|| state_error("JSON size overflow"))?,
            512,
        )?)
        .map_err(engine_error)?;
    let mut bytes = Vec::with_capacity(count.0);
    serde_json::to_writer(&mut bytes, value).map_err(engine_error)?;
    if bytes.capacity() > reservation.size() {
        return Err(state_error("control exceeded prepaid bound"));
    }
    Ok(StateSegment::new(bytes).with_owner(Arc::new(reservation)))
}

fn validate_ipc(bytes: &[u8]) -> Result<()> {
    use datafusion::arrow::ipc::{root_as_footer, root_as_message};
    if bytes.len() < 10 || !bytes.starts_with(b"ARROW1") || !bytes.ends_with(b"ARROW1") {
        return Err(state_error("invalid snapshot_sql IPC magic"));
    }
    let footer_len = u32::from_le_bytes(
        bytes[bytes.len() - 10..bytes.len() - 6]
            .try_into()
            .expect("four bytes"),
    ) as usize;
    let footer_start = bytes
        .len()
        .checked_sub(10)
        .and_then(|n| n.checked_sub(footer_len))
        .ok_or_else(|| state_error("invalid IPC footer length"))?;
    let footer = root_as_footer(&bytes[footer_start..bytes.len() - 10]).map_err(engine_error)?;
    for blocks in [footer.dictionaries(), footer.recordBatches()]
        .into_iter()
        .flatten()
    {
        for block in blocks {
            let start = usize::try_from(block.offset()).map_err(engine_error)?;
            let len = usize::try_from(block.metaDataLength()).map_err(engine_error)?;
            let block_bytes = bytes
                .get(start..add(start, len)?)
                .ok_or_else(|| state_error("invalid IPC block"))?;
            let prefix = if block_bytes.starts_with(&[255; 4]) {
                8
            } else {
                4
            };
            let message = root_as_message(
                block_bytes
                    .get(prefix..)
                    .ok_or_else(|| state_error("invalid IPC message"))?,
            )
            .map_err(engine_error)?;
            let record = message
                .header_as_record_batch()
                .or_else(|| message.header_as_dictionary_batch().and_then(|d| d.data()));
            if record.is_some_and(|r| r.compression().is_some()) {
                return Err(state_error("compressed snapshot_sql IPC is unsupported"));
            }
        }
    }
    Ok(())
}
