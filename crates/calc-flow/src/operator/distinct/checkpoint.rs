use super::super::sql::{ipc, metadata};
use super::{DistinctOperator, DistinctSpec, owned};
use crate::{
    Batch, CalcFlowError, DataFusionConfig, EventTime, JsonMap, OperatorMetadata,
    OperatorStateSnapshot, Result, StateSegment,
};
use datafusion::{
    arrow::{ipc::reader::FileReader, record_batch::RecordBatch},
    execution::memory_pool::MemoryPool,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Cursor, sync::Arc};

#[derive(Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Identity {
    configuration: String,
    input_schema: String,
    output_schema: String,
    datafusion: String,
    arrow: String,
    runtime_config: DataFusionConfig,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Control {
    state_kind: String,
    state_layout: u32,
    accounting_version: u32,
    ingress_policy: String,
    spec: DistinctSpec,
    columns: Vec<String>,
    identity: Identity,
    accepted_rows: usize,
    retained_rows: usize,
    closed: bool,
    last_watermark: Option<EventTime>,
    output_sequence: u64,
    data_sha256: String,
    metadata_sha256: String,
}

fn mismatch(message: impl std::fmt::Display) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("distinct_v2: {message}"),
    }
}
fn digest(value: &impl Serialize) -> Result<String> {
    Ok(hex::encode(Sha256::digest(
        serde_json::to_vec(value).map_err(mismatch)?,
    )))
}
fn schema_digest(schema: &datafusion::arrow::datatypes::SchemaRef) -> String {
    let mut tracker = datafusion::arrow::ipc::writer::DictionaryTracker::new(false);
    let encoded = datafusion::arrow::ipc::convert::IpcSchemaEncoder::new()
        .with_dictionary_tracker(&mut tracker)
        .schema_to_fb(schema);
    hex::encode(Sha256::digest(encoded.finished_data()))
}
fn inline(control: &StateSegment) -> JsonMap {
    BTreeMap::from([
        ("state_kind".into(), json!("distinct_v2")),
        ("state_layout".into(), json!(2)),
        ("accounting_version".into(), json!(2)),
        ("ingress_policy".into(), json!("copy_visible_v2")),
        ("control_sha256".into(), json!(control.sha256())),
    ])
}

impl DistinctOperator {
    fn identity(&self) -> Result<Identity> {
        Ok(Identity {
            configuration: digest(&self.configuration())?,
            input_schema: schema_digest(self.input[0].schema().unwrap()),
            output_schema: schema_digest(self.output[0].schema().unwrap()),
            datafusion: datafusion::DATAFUSION_VERSION.into(),
            arrow: datafusion::arrow::ARROW_VERSION.into(),
            runtime_config: self.config,
        })
    }
    fn descriptor_bound(&self) -> Result<usize> {
        self.input
            .iter()
            .chain(&self.output)
            .try_fold(16_384, |n, p| {
                owned::add(
                    n,
                    owned::mul(owned::schema_quote(p.schema().unwrap())?, 32)?,
                )
            })
    }
    pub(super) fn capture(&mut self) -> Result<OperatorStateSnapshot> {
        self.capture_checked(&|| Ok(()))
    }
    pub(super) fn capture_checked(
        &mut self,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<OperatorStateSnapshot> {
        check()?;
        self.ensure_stream()?;
        let state = self.stream.as_ref().unwrap();
        if let Some(capture) = &state.capture {
            return Ok(capture.clone());
        }
        let pool: Arc<dyn MemoryPool> = state.pool.clone();
        let control_lease = owned::reserve(
            &pool,
            self.descriptor_bound()?,
            "checkpoint control/identity",
        )?;
        let history = owned::combine(
            &state.history,
            self.output[0].schema().unwrap(),
            &self.columns,
            &pool,
            check,
        )?;
        let data = ipc::encode(
            &history.batch,
            owned::reserve(&pool, 0, "checkpoint IPC")?,
            check,
        )?;
        let meta = metadata::encode(
            history.batch.metadata(),
            metadata::reserve_with(
                history.batch.metadata(),
                "distinct checkpoint metadata",
                owned::reserve(&pool, 0, "checkpoint metadata")?,
            )?,
            check,
        )?;
        let control = Control {
            state_kind: "distinct_v2".into(),
            state_layout: 2,
            accounting_version: 2,
            ingress_policy: "copy_visible_v2".into(),
            spec: self.spec.clone(),
            columns: self.columns.clone(),
            identity: self.identity()?,
            accepted_rows: state.accepted,
            retained_rows: history.batch.num_rows(),
            closed: state.closed,
            last_watermark: state.watermark,
            output_sequence: u64::from(state.closed),
            data_sha256: data.segment.sha256().into(),
            metadata_sha256: meta.segment.sha256().into(),
        };
        let bytes = serde_json::to_vec(&control).map_err(mismatch)?;
        if owned::add(bytes.capacity(), 256)? > control_lease.size() {
            return Err(mismatch("control exceeded prepaid bound"));
        }
        let control = StateSegment::new_controlled(bytes, &mut || check())?
            .with_owner(Arc::new(control_lease));
        let snapshot = OperatorStateSnapshot {
            inline_metadata: inline(&control),
            segments: BTreeMap::from([
                ("control".into(), control),
                ("data".into(), data.segment.clone()),
                ("metadata".into(), meta.segment.clone()),
            ]),
        };
        check()?;
        self.stream.as_mut().unwrap().capture = Some(snapshot.clone());
        Ok(snapshot)
    }

    pub(super) fn restore_capture(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        self.restore_capture_checked(snapshot, &|| Ok(()))
    }
    pub(super) fn restore_capture_checked(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        check()?;
        if snapshot.segments.len() != 3
            || snapshot.inline_metadata.len() != 5
            || snapshot.inline_metadata.get("state_kind") != Some(&json!("distinct_v2"))
            || snapshot.inline_metadata.get("state_layout") != Some(&json!(2))
            || snapshot.inline_metadata.get("accounting_version") != Some(&json!(2))
            || snapshot.inline_metadata.get("ingress_policy") != Some(&json!("copy_visible_v2"))
        {
            return Err(mismatch("policy/state/layout/accounting mismatch"));
        }
        let segment = |name: &str| {
            snapshot
                .segments
                .get(name)
                .ok_or_else(|| mismatch(format!("missing {name} segment")))
        };
        let control_segment = segment("control")?;
        if snapshot.inline_metadata != inline(control_segment) {
            return Err(mismatch("control identity mismatch"));
        }
        self.ensure_stream()?;
        let pool: Arc<dyn MemoryPool> = self.stream.as_ref().unwrap().pool.clone();
        let _control_scratch = owned::reserve(
            &pool,
            owned::add(
                self.descriptor_bound()?,
                owned::mul(control_segment.bytes().len(), 32)?,
            )?,
            "restore control/identity",
        )?;
        let control: Control = serde_json::from_slice(control_segment.bytes()).map_err(mismatch)?;
        self.validate_control(&control)?;
        let data = segment("data")?;
        let meta = segment("metadata")?;
        if data.sha256() != control.data_sha256 || meta.sha256() != control.metadata_sha256 {
            return Err(mismatch("segment digest mismatch"));
        }
        let owned = self.decode_history(data, meta, &control, &pool, check)?;
        let bytes = snapshot.segments.values().try_fold(4096, |n, s| {
            owned::add(
                n,
                owned::add(
                    s.retained_buffer_bytes()
                        .ok_or_else(|| mismatch("capture capacity overflow"))?,
                    256,
                )?,
            )
        })?;
        let capture_lease = Arc::new(owned::reserve(&pool, bytes, "restored live captures")?);
        let mut segments = BTreeMap::new();
        for (name, segment) in &snapshot.segments {
            check()?;
            let copy = StateSegment::new_controlled(segment.bytes().to_vec(), &mut || check())?
                .with_owner(capture_lease.clone());
            if copy.sha256() != segment.sha256() {
                return Err(mismatch("copied capture digest mismatch"));
            }
            segments.insert(name.clone(), copy);
        }
        let capture = OperatorStateSnapshot {
            inline_metadata: snapshot.inline_metadata.clone(),
            segments,
        };
        let history = if control.closed {
            vec![]
        } else {
            vec![Arc::new(owned)]
        };
        check()?;
        let state = self.stream.as_mut().unwrap();
        state.history = history;
        state.accepted = control.accepted_rows;
        state.closed = control.closed;
        state.watermark = control.last_watermark;
        state.capture = Some(capture);
        Ok(())
    }
    fn decode_history(
        &self,
        data: &StateSegment,
        meta: &StateSegment,
        control: &Control,
        pool: &Arc<dyn MemoryPool>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<owned::OwnedBatch> {
        check()?;
        validate_ipc(data.bytes(), control.retained_rows, self.columns.len())?;
        let _decode = owned::reserve(
            pool,
            owned::add(
                owned::mul(data.bytes().len(), 32)?,
                owned::mul(
                    control.retained_rows,
                    self.columns.len().saturating_mul(256),
                )?,
            )?,
            "restore IPC decode",
        )?;
        let reader = FileReader::try_new(Cursor::new(data.bytes()), None).map_err(mismatch)?;
        if reader.schema() != *self.output[0].schema().unwrap() {
            return Err(mismatch("history schema mismatch"));
        }
        let mut records = reader
            .map(|record| {
                check()?;
                record.map_err(mismatch)
            })
            .collect::<Result<Vec<_>>>()?;
        if records.is_empty() {
            records.push(RecordBatch::new_empty(
                self.output[0].schema().unwrap().clone(),
            ));
        }
        let (metadata, _metadata_lease) = metadata::decode_with(
            meta,
            "distinct restore metadata",
            owned::reserve(pool, 0, "restore metadata")?,
        )?;
        let raw = Batch::table(records, metadata)?;
        if raw.num_rows() != control.retained_rows {
            return Err(mismatch("history row count mismatch"));
        }
        let owned = owned::normalize(&raw, &self.columns, pool, None, check)?;
        drop(raw);
        Ok(owned)
    }
    fn validate_control(&self, control: &Control) -> Result<()> {
        if !control.closed
            && matches!(self.spec.history(),super::DistinctHistory::ClosedWindow {end_micros,..} if control.last_watermark.is_some_and(|w|w.as_micros()>=*end_micros))
        {
            return Err(mismatch("open window has a closing watermark"));
        }

        if control.state_kind != "distinct_v2"
            || control.state_layout != 2
            || control.accounting_version != 2
            || control.ingress_policy != "copy_visible_v2"
            || control.spec != self.spec
            || control.columns != self.columns
            || control.identity != self.identity()?
        {
            return Err(mismatch("configuration/schema/engine identity mismatch"));
        }
        if control.accepted_rows as u64 > self.spec.state_budget().max_rows()
            || control.output_sequence != u64::from(control.closed)
            || control.retained_rows
                != if control.closed {
                    0
                } else {
                    control.accepted_rows
                }
        {
            return Err(mismatch("accepted/retained/closed/sequence mismatch"));
        }
        Ok(())
    }
}

fn validate_ipc(bytes: &[u8], rows: usize, columns: usize) -> Result<()> {
    use datafusion::arrow::ipc::{root_as_footer, root_as_message};
    if bytes.len() < 10 || &bytes[bytes.len() - 6..] != b"ARROW1" {
        return Err(mismatch("invalid IPC trailer"));
    }
    let footer_len =
        u32::from_le_bytes(bytes[bytes.len() - 10..bytes.len() - 6].try_into().unwrap()) as usize;
    let start = bytes
        .len()
        .checked_sub(owned::add(footer_len, 10)?)
        .ok_or_else(|| mismatch("IPC footer length"))?;
    let footer = root_as_footer(&bytes[start..bytes.len() - 10]).map_err(mismatch)?;
    if footer.dictionaries().is_some_and(|v| !v.is_empty()) {
        return Err(mismatch("dictionary history is unsupported"));
    }
    let mut total = 0;
    if let Some(blocks) = footer.recordBatches() {
        for block in blocks {
            let offset = usize::try_from(block.offset()).map_err(mismatch)?;
            let metadata = usize::try_from(block.metaDataLength()).map_err(mismatch)?;
            let body = usize::try_from(block.bodyLength()).map_err(mismatch)?;
            let end = owned::add(offset, metadata)?;
            if owned::add(end, body)? > start || metadata < 8 {
                return Err(mismatch("IPC block exceeds file"));
            }
            let prefix = if bytes.get(offset..offset + 4) == Some(&[255, 255, 255, 255]) {
                8
            } else {
                4
            };
            let message = root_as_message(
                bytes
                    .get(offset + prefix..end)
                    .ok_or_else(|| mismatch("IPC message range"))?,
            )
            .map_err(mismatch)?;
            let batch = message
                .header_as_record_batch()
                .ok_or_else(|| mismatch("non-record history message"))?;
            if batch.compression().is_some() {
                return Err(mismatch("compressed history is unsupported"));
            }
            let n = usize::try_from(batch.length()).map_err(mismatch)?;
            total = owned::add(total, n)?;
            if total > rows
                || batch.nodes().is_none_or(|v| {
                    v.len() != columns
                        || v.iter().any(|node| {
                            node.length() != batch.length()
                                || node.null_count() < 0
                                || node.null_count() > batch.length()
                        })
                })
            {
                return Err(mismatch("IPC row/node count mismatch"));
            }
            if batch.buffers().is_none_or(|v| {
                v.iter().any(|b| {
                    b.offset() < 0
                        || b.length() < 0
                        || b.offset()
                            .checked_add(b.length())
                            .is_none_or(|end| end > block.bodyLength())
                })
            }) {
                return Err(mismatch("IPC buffer exceeds body"));
            }
        }
    }
    if total != rows {
        return Err(mismatch("IPC retained row count mismatch"));
    }
    Ok(())
}
