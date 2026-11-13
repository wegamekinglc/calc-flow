use super::super::sql::{ipc, metadata};
use super::{BoundedOrderOperator, SortSpec, owned};
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
    spec: SortSpec,
    columns: Vec<String>,
    by: Vec<super::SortKey>,
    fetch: Option<u64>,
    identity: Identity,
    accepted_rows: usize,
    retained_rows: usize,
    closed: bool,
    last_watermark: Option<EventTime>,
    output_sequence: u64,
    data_sha256: String,
    metadata_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    late_metrics: Option<crate::operator::stream::LateMetricDelta>,
}

fn mismatch(message: impl std::fmt::Display) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("bounded_order_v1: {message}"),
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
fn inline(control: &StateSegment, layout: u32) -> JsonMap {
    BTreeMap::from([
        ("state_kind".into(), json!("bounded_order_v1")),
        ("state_layout".into(), json!(layout)),
        ("accounting_version".into(), json!(1)),
        ("ingress_policy".into(), json!("copy_visible_v2")),
        ("control_sha256".into(), json!(control.sha256())),
    ])
}

impl BoundedOrderOperator {
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
        self.input.iter().chain(&self.output).try_fold(
            16_384 + self.metric_storage() * 4,
            |n, p| {
                owned::add(
                    n,
                    owned::mul(owned::schema_quote(p.schema().unwrap())?, 32)?,
                )
            },
        )
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
        let snapshot = self.capture_fresh(check)?;
        check()?;
        self.stream.as_mut().unwrap().capture = Some(snapshot.clone());
        Ok(snapshot)
    }
    fn capture_fresh(&self, check: &dyn Fn() -> Result<()>) -> Result<OperatorStateSnapshot> {
        let state = self.stream.as_ref().unwrap();
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
        let data = encode_data(&history.batch, &pool, check)?;
        let meta = encode_metadata(&history.batch, &pool, check)?;
        let control = self.encode_control(
            &data.segment,
            &meta.segment,
            history.batch.num_rows(),
            control_lease,
            check,
        )?;
        let snapshot = OperatorStateSnapshot {
            inline_metadata: inline(&control, self.spec.state_layout()),
            segments: BTreeMap::from([
                ("control".into(), control),
                ("data".into(), data.segment.clone()),
                ("metadata".into(), meta.segment.clone()),
            ]),
        };
        Ok(snapshot)
    }

    fn encode_control(
        &self,
        data: &StateSegment,
        meta: &StateSegment,
        rows: usize,
        lease: datafusion::execution::memory_pool::MemoryReservation,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<StateSegment> {
        let state = self.stream.as_ref().unwrap();
        let control = Control {
            state_kind: "bounded_order_v1".into(),
            state_layout: self.spec.state_layout(),
            accounting_version: 1,
            ingress_policy: "copy_visible_v2".into(),
            spec: self.spec.clone(),
            columns: self.columns.clone(),
            by: self.by.clone(),
            fetch: self.fetch,
            identity: self.identity()?,
            accepted_rows: state.accepted,
            retained_rows: rows,
            closed: state.closed,
            last_watermark: state.watermark,
            output_sequence: u64::from(state.closed),
            data_sha256: data.sha256().into(),
            metadata_sha256: meta.sha256().into(),
            late_metrics: (self.spec.state_layout() == 2).then_some(state.metrics),
        };
        let bytes = serde_json::to_vec(&control).map_err(mismatch)?;
        if owned::add(bytes.capacity(), 256)? > lease.size() {
            return Err(mismatch("control exceeded prepaid bound"));
        }
        Ok(StateSegment::new_controlled(bytes, &mut || check())?.with_owner(Arc::new(lease)))
    }
    pub(super) fn restore_capture(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        self.restore_capture_checked(snapshot, &|| Ok(()))
    }
    pub(super) fn restore_capture_checked(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        self.restore_with_metrics(snapshot, check, None)
    }
    pub(super) fn restore_with_context(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        context: &crate::StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.restore_with_metrics(snapshot, &|| context.check_cancelled(), Some(context))
    }
    fn restore_with_metrics(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        check: &dyn Fn() -> Result<()>,
        context: Option<&crate::StreamOperatorContext<'_>>,
    ) -> Result<()> {
        check()?;
        let candidate = self.prepare_restore(snapshot, check)?;
        let staged = match (candidate.control.late_metrics, context) {
            (Some(totals), Some(context)) => Some(context.prepare_restored_window_metrics(totals)?),
            _ => None,
        };
        check()?;
        self.install_restore(candidate);
        if let Some(staged) = staged {
            staged.commit();
        }
        Ok(())
    }
    fn prepare_restore(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<RestoreCandidate> {
        let segment = validate_snapshot(snapshot, self.spec.state_layout())?;
        self.ensure_stream()?;
        let pool: Arc<dyn MemoryPool> = self.pool.clone();
        let (control, scratch) = self.read_control(segment, &pool)?;
        let (data, meta) = history_segments(snapshot, &control)?;
        let owned = self.decode_history(data, meta, &control, &pool, check)?;
        let capture = copy_capture(snapshot, &pool, check)?;
        let history_lease = if control.closed {
            None
        } else {
            Some(owned::reserve(&pool, 64, "restored history container")?)
        };
        let history = if control.closed {
            vec![]
        } else {
            vec![Arc::new(owned)]
        };
        Ok(RestoreCandidate {
            control,
            history,
            capture,
            history_lease,
            _scratch: scratch,
        })
    }
    fn read_control(
        &self,
        segment: &StateSegment,
        pool: &Arc<dyn MemoryPool>,
    ) -> Result<(
        Control,
        datafusion::execution::memory_pool::MemoryReservation,
    )> {
        let bytes = owned::add(
            self.descriptor_bound()?,
            owned::mul(segment.bytes().len(), 32)?,
        )?;
        let scratch = owned::reserve(pool, bytes, "restore control/identity")?;
        let control: Control = serde_json::from_slice(segment.bytes()).map_err(mismatch)?;
        self.validate_control(&control)?;
        Ok((control, scratch))
    }
    fn install_restore(&mut self, candidate: RestoreCandidate) {
        let state = self.stream.as_mut().unwrap();
        state.history = candidate.history;
        state.history_lease = candidate.history_lease;
        state.accepted = candidate.control.accepted_rows;
        state.closed = candidate.control.closed;
        state.watermark = candidate.control.last_watermark;
        state.capture = Some(candidate.capture);
        state.metrics = candidate.control.late_metrics.unwrap_or_default();
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
        let _decode = decode_lease(data, control, pool, self.columns.len())?;
        let (raw, _metadata) =
            decode_raw(data, meta, self.output[0].schema().unwrap(), pool, check)?;
        if raw.num_rows() != control.retained_rows {
            return Err(mismatch("history row count mismatch"));
        }
        let history = owned::normalize(&raw, &self.columns, pool, None, check)?;
        self.validate_sequence(&history, pool, check)?;
        Ok(history)
    }
    fn validate_control(&self, control: &Control) -> Result<()> {
        validate_header(control, self.spec.state_layout())?;
        self.validate_identity(control)?;
        validate_cut(control)?;
        self.validate_metrics(control)
    }
    fn validate_identity(&self, control: &Control) -> Result<()> {
        if control.spec != self.spec
            || control.columns != self.columns
            || control.by != self.by
            || control.fetch != self.fetch
            || control.identity != self.identity()?
        {
            return Err(mismatch("configuration/schema/engine identity mismatch"));
        }
        Ok(())
    }
    fn validate_metrics(&self, control: &Control) -> Result<()> {
        match (self.spec.state_layout(), control.late_metrics) {
            (1, None) => Ok(()),
            (2, Some(metrics)) => {
                validate_metrics(metrics, control.closed)?;
                validate_metric_cut(control, metrics)
            }
            _ => Err(mismatch("late policy/metrics layout mismatch")),
        }
    }
}

struct RestoreCandidate {
    control: Control,
    history: Vec<Arc<owned::OwnedBatch>>,
    capture: OperatorStateSnapshot,
    history_lease: Option<datafusion::execution::memory_pool::MemoryReservation>,
    _scratch: datafusion::execution::memory_pool::MemoryReservation,
}
fn segment<'a>(snapshot: &'a OperatorStateSnapshot, name: &str) -> Result<&'a StateSegment> {
    snapshot
        .segments
        .get(name)
        .ok_or_else(|| mismatch(format!("missing {name} segment")))
}
fn encode_data(
    batch: &Batch,
    pool: &Arc<dyn MemoryPool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<ipc::SqlInputSegment>> {
    ipc::encode(batch, owned::reserve(pool, 0, "checkpoint IPC")?, check)
}
fn encode_metadata(
    batch: &Batch,
    pool: &Arc<dyn MemoryPool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<metadata::SqlMetadata>> {
    let lease = metadata::reserve_with(
        batch.metadata(),
        "bounded_order checkpoint metadata",
        owned::reserve(pool, 0, "checkpoint metadata")?,
    )?;
    metadata::encode(batch.metadata(), lease, check)
}
fn validate_snapshot(snapshot: &OperatorStateSnapshot, layout: u32) -> Result<&StateSegment> {
    let control = segment(snapshot, "control")?;
    if snapshot.segments.len() != 3 || snapshot.inline_metadata != inline(control, layout) {
        return Err(mismatch(
            "policy/state/layout/accounting/control identity mismatch",
        ));
    }
    Ok(control)
}
fn history_segments<'a>(
    snapshot: &'a OperatorStateSnapshot,
    control: &Control,
) -> Result<(&'a StateSegment, &'a StateSegment)> {
    let data = segment(snapshot, "data")?;
    let meta = segment(snapshot, "metadata")?;
    if data.sha256() != control.data_sha256 || meta.sha256() != control.metadata_sha256 {
        return Err(mismatch("segment digest mismatch"));
    }
    Ok((data, meta))
}
fn copy_capture(
    snapshot: &OperatorStateSnapshot,
    pool: &Arc<dyn MemoryPool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<OperatorStateSnapshot> {
    let bytes = capture_bound(snapshot)?;
    let lease = Arc::new(owned::reserve(pool, bytes, "restored live captures")?);
    let segments = snapshot
        .segments
        .iter()
        .map(|(name, segment)| {
            check()?;
            let copy = StateSegment::new_controlled(segment.bytes().to_vec(), &mut || check())?
                .with_owner(lease.clone());
            if copy.sha256() != segment.sha256() {
                return Err(mismatch("copied capture digest mismatch"));
            }
            Ok((name.clone(), copy))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    Ok(OperatorStateSnapshot {
        inline_metadata: snapshot.inline_metadata.clone(),
        segments,
    })
}
fn capture_bound(snapshot: &OperatorStateSnapshot) -> Result<usize> {
    snapshot.segments.values().try_fold(4096, |n, s| {
        owned::add(
            n,
            owned::add(
                s.retained_buffer_bytes()
                    .ok_or_else(|| mismatch("capture capacity overflow"))?,
                256,
            )?,
        )
    })
}
fn decode_lease(
    data: &StateSegment,
    control: &Control,
    pool: &Arc<dyn MemoryPool>,
    columns: usize,
) -> Result<datafusion::execution::memory_pool::MemoryReservation> {
    let bytes = owned::add(
        owned::mul(data.bytes().len(), 32)?,
        owned::mul(control.retained_rows, owned::mul(columns, 256)?)?,
    )?;
    owned::reserve(pool, bytes, "restore IPC decode")
}
fn decode_raw(
    data: &StateSegment,
    meta: &StateSegment,
    schema: &datafusion::arrow::datatypes::SchemaRef,
    pool: &Arc<dyn MemoryPool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<(Batch, Arc<metadata::SqlMetadata>)> {
    let reader = FileReader::try_new(Cursor::new(data.bytes()), None).map_err(mismatch)?;
    if reader.schema() != *schema {
        return Err(mismatch("history schema mismatch"));
    }
    let mut records = reader
        .map(|r| {
            check()?;
            r.map_err(mismatch)
        })
        .collect::<Result<Vec<_>>>()?;
    if records.is_empty() {
        records.push(RecordBatch::new_empty(schema.clone()));
    }
    let (metadata, lease) = decode_metadata(meta, pool)?;
    Ok((Batch::table(records, metadata)?, lease))
}
fn decode_metadata(
    meta: &StateSegment,
    pool: &Arc<dyn MemoryPool>,
) -> Result<(crate::BatchMetadata, Arc<metadata::SqlMetadata>)> {
    metadata::decode_with(
        meta,
        "bounded_order restore metadata",
        owned::reserve(pool, 0, "restore metadata")?,
    )
}
fn validate_header(c: &Control, layout: u32) -> Result<()> {
    if c.state_kind != "bounded_order_v1"
        || c.state_layout != layout
        || c.accounting_version != 1
        || c.ingress_policy != "copy_visible_v2"
    {
        return Err(mismatch("configuration/policy/layout/accounting mismatch"));
    }
    Ok(())
}
fn validate_cut(c: &Control) -> Result<()> {
    validate_window_cut(c)?;
    let expected = if c.closed { 0 } else { c.accepted_rows };
    if c.accepted_rows as u64 > c.spec.state_budget().max_rows()
        || c.output_sequence != u64::from(c.closed)
        || c.retained_rows != expected
    {
        return Err(mismatch("accepted/retained/closed/sequence mismatch"));
    }
    Ok(())
}
fn validate_window_cut(c: &Control) -> Result<()> {
    if let super::SortHistory::ClosedWindow { end_micros, .. } = c.spec.history()
        && !c.closed
        && c.last_watermark
            .is_some_and(|w| w.as_micros() >= *end_micros)
    {
        return Err(mismatch("open window has a closing watermark"));
    }
    Ok(())
}
fn validate_metrics(m: crate::operator::stream::LateMetricDelta, closed: bool) -> Result<()> {
    if m.null_event_time_rows != 0 || m.null_event_time_batches != 0 {
        return Err(mismatch("unexpected null-time telemetry"));
    }
    if m.late_rows == 0 {
        return validate_zero_metrics(m);
    }
    if !closed
        || m.affected_batches == 0
        || m.affected_batches > m.late_rows
        || m.max_lateness_micros.is_none_or(|n| n == 0)
    {
        return Err(mismatch("invalid late totals"));
    }
    Ok(())
}
fn validate_zero_metrics(m: crate::operator::stream::LateMetricDelta) -> Result<()> {
    if m.affected_batches != 0 || m.max_lateness_micros.is_some() {
        return Err(mismatch("inconsistent zero metrics"));
    }
    Ok(())
}
fn validate_ipc(bytes: &[u8], rows: usize, columns: usize) -> Result<()> {
    let (footer, start) = read_footer(bytes)?;
    let mut total = 0;
    if let Some(blocks) = footer.recordBatches() {
        for block in blocks {
            total = owned::add(total, validate_block(bytes, block, start, columns)?)?;
        }
    }
    if total != rows {
        return Err(mismatch("IPC retained row count mismatch"));
    }
    Ok(())
}
fn read_footer(bytes: &[u8]) -> Result<(datafusion::arrow::ipc::Footer<'_>, usize)> {
    if bytes.len() < 10 || &bytes[bytes.len() - 6..] != b"ARROW1" {
        return Err(mismatch("invalid IPC trailer"));
    }
    let size =
        u32::from_le_bytes(bytes[bytes.len() - 10..bytes.len() - 6].try_into().unwrap()) as usize;
    let start = bytes
        .len()
        .checked_sub(owned::add(size, 10)?)
        .ok_or_else(|| mismatch("IPC footer length"))?;
    let footer = datafusion::arrow::ipc::root_as_footer(&bytes[start..bytes.len() - 10])
        .map_err(mismatch)?;
    if footer.dictionaries().is_some_and(|v| !v.is_empty()) {
        return Err(mismatch("dictionary history is unsupported"));
    }
    Ok((footer, start))
}
fn validate_block(
    bytes: &[u8],
    block: &datafusion::arrow::ipc::Block,
    start: usize,
    columns: usize,
) -> Result<usize> {
    let batch = read_record(bytes, block, start)?;
    validate_nodes(batch, columns)?;
    validate_buffers(batch, block.bodyLength())?;
    usize::try_from(batch.length()).map_err(mismatch)
}
fn read_record<'a>(
    bytes: &'a [u8],
    block: &datafusion::arrow::ipc::Block,
    start: usize,
) -> Result<datafusion::arrow::ipc::RecordBatch<'a>> {
    let offset = usize::try_from(block.offset()).map_err(mismatch)?;
    let metadata = usize::try_from(block.metaDataLength()).map_err(mismatch)?;
    validate_block_range(offset, metadata, block.bodyLength(), start)?;
    let message = read_message(bytes, offset, metadata)?;
    let batch = message
        .header_as_record_batch()
        .ok_or_else(|| mismatch("non-record history message"))?;
    if batch.compression().is_some() {
        return Err(mismatch("compressed history is unsupported"));
    }
    Ok(batch)
}
fn validate_block_range(offset: usize, metadata: usize, body: i64, start: usize) -> Result<()> {
    let body = usize::try_from(body).map_err(mismatch)?;
    if owned::add(owned::add(offset, metadata)?, body)? > start || metadata < 8 {
        return Err(mismatch("IPC block exceeds file"));
    }
    Ok(())
}
fn read_message(
    bytes: &[u8],
    offset: usize,
    metadata: usize,
) -> Result<datafusion::arrow::ipc::Message<'_>> {
    let prefix = if bytes.get(offset..offset + 4) == Some(&[255, 255, 255, 255]) {
        8
    } else {
        4
    };
    let range = bytes
        .get(offset + prefix..offset + metadata)
        .ok_or_else(|| mismatch("IPC message range"))?;
    datafusion::arrow::ipc::root_as_message(range).map_err(mismatch)
}
fn validate_nodes(batch: datafusion::arrow::ipc::RecordBatch<'_>, columns: usize) -> Result<()> {
    let nodes = batch.nodes().ok_or_else(|| mismatch("missing IPC nodes"))?;
    if nodes.len() != columns || nodes.iter().any(|n| invalid_node(n, batch.length())) {
        return Err(mismatch("IPC row/node count mismatch"));
    }
    Ok(())
}
fn invalid_node(node: &datafusion::arrow::ipc::FieldNode, length: i64) -> bool {
    node.length() != length || node.null_count() < 0 || node.null_count() > length
}
fn validate_buffers(batch: datafusion::arrow::ipc::RecordBatch<'_>, body: i64) -> Result<()> {
    let buffers = batch
        .buffers()
        .ok_or_else(|| mismatch("missing IPC buffers"))?;
    if buffers.iter().any(|b| invalid_buffer(b, body)) {
        return Err(mismatch("IPC buffer exceeds body"));
    }
    Ok(())
}
fn invalid_buffer(buffer: &datafusion::arrow::ipc::Buffer, body: i64) -> bool {
    buffer.offset() < 0
        || buffer.length() < 0
        || buffer
            .offset()
            .checked_add(buffer.length())
            .is_none_or(|end| end > body)
}
fn validate_metric_cut(c: &Control, m: crate::operator::stream::LateMetricDelta) -> Result<()> {
    if m.late_rows == 0 {
        return Ok(());
    }
    let super::SortHistory::ClosedWindow {
        start_micros,
        end_micros,
        ..
    } = c.spec.history()
    else {
        return Err(mismatch("late totals without window"));
    };
    let w = c
        .last_watermark
        .filter(|w| w.as_micros() >= *end_micros)
        .ok_or_else(|| mismatch("late totals without closing watermark"))?;
    let maximum =
        u64::try_from(i128::from(w.as_micros()) - i128::from(*start_micros)).map_err(mismatch)?;
    if m.max_lateness_micros.is_some_and(|n| n > maximum) {
        return Err(mismatch("late total exceeds time domain"));
    }
    Ok(())
}
