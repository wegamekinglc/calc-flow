mod encoding;
mod index_v2;
mod prepared;
mod validation;

use super::{
    StreamAsofJoinOperator, StreamAsofJoinStatus,
    state::{
        BatchKey, Encoding, Inventory, PayloadBatch, PayloadView, RightBucket, RowPayload, State,
    },
};
use crate::{
    CalcFlowError, Epoch, IngressProgressSnapshot, OperatorStateSnapshot, Result, StateSegment,
    StreamOperatorContext,
};
use datafusion::execution::memory_pool::MemoryReservation;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    mem::size_of,
    sync::{Arc, OnceLock},
};

use encoding::{Decoder, restore_charge};
pub(super) use index_v2::encoded_length;
pub(super) use prepared::PreparedSegment;
use validation::{validate_counters, validate_progress};

const MAGIC: &[u8; 8] = b"CFASOF01";
const SEGMENT: &str = "asof-state-v1";
const INDEX_SEGMENT: &str = index_v2::INDEX_SEGMENT;
type IdentityCache =
    BTreeMap<BatchKey, (super::state::EncodedColumns, super::state::EncodedColumns)>;

fn encode_payloads(
    pending: Vec<Arc<PayloadBatch>>,
    _workspace: MemoryReservation,
    limit: usize,
    name: &str,
) -> Result<()> {
    for batch in pending {
        batch.ensure_encoded(limit, name)?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) struct PreparedCheckpoint {
    pub segment: Option<PreparedSegment>,
    pub _workspace: MemoryReservation,
}

pub(super) struct DecodedSnapshot {
    state: State,
    metrics: StreamAsofJoinStatus,
    terminal: bool,
    sequence: u64,
    prepared: Option<PreparedSegment>,
    _workspace: MemoryReservation,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Metadata<'a> {
    kind: &'a str,
    state_version: u32,
    layout_version: u32,
    accounting_version: u32,
    row_encoding: &'a str,
    fingerprint: &'a str,
    epoch: u64,
    terminal: bool,
    next_output_sequence: u64,
    metrics: StreamAsofJoinStatus,
}

impl StreamAsofJoinOperator {
    /// Admission and uncaptured output prefixes reserve and charge the
    /// canonical index length without serializing it. Materialize on capture.
    pub(super) fn ensure_prepared_sync(&mut self) -> Result<()> {
        let Some(length) = self.deferred_index_len else {
            return Ok(());
        };
        let _workspace = self.reserve_workspace(index_v2::workspace_bytes(
            &self.state,
            length,
            false,
            &self.name,
        )?)?;
        let limit = usize::try_from(self.spec.limits().max_state_bytes()).expect("validated");
        let segment = index_v2::encode_sync(&self.state, length, limit)?;
        self.prepared = Some(PreparedSegment::new(segment));
        self.deferred_index_len = None;
        debug_assert_eq!(
            self.state
                .inventory(self.prepared.as_ref(), &self.name)
                .expect("materialized index inventory")
                .bytes,
            self.status.state_bytes,
        );
        Ok(())
    }

    pub(super) async fn ensure_prepared_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()?;
        self.prepare_deferred_index_async(context).await?;
        self.compact_prepared_async(context).await?;
        self.prepare_payloads_async(context).await?;
        context.check_cancelled()
    }

    fn pending_payloads(&self) -> Result<(Vec<Arc<PayloadBatch>>, MemoryReservation)> {
        let (count, largest) = self.pending_payload_extent()?;
        let pointers = count
            .checked_mul(size_of::<Arc<PayloadBatch>>() as u64)
            .ok_or_else(|| {
                super::reason(
                    &self.name,
                    crate::StreamingFailureReason::AsofCounterOverflow,
                    "ASOF payload checkpoint workspace overflowed",
                )
            })?;
        let workspace = self.reserve_workspace(super::checked(&self.name, pointers, largest)?)?;
        let mut pending = Vec::with_capacity(usize::try_from(count).expect("bounded ASOF rows"));
        for (batch, _) in self.state.batches.values() {
            if !batch.has_encoded() {
                pending.push(Arc::clone(batch));
            }
        }
        Ok((pending, workspace))
    }

    fn pending_payload_extent(&self) -> Result<(u64, u64)> {
        let mut count = 0_u64;
        let mut largest = 0_u64;
        for (batch, _) in self.state.batches.values() {
            if !batch.has_encoded() {
                count = super::checked(&self.name, count, 1)?;
                largest = largest.max(batch.encoded_charge_bytes);
            }
        }
        Ok((count, largest))
    }

    fn ensure_payloads_sync(&self) -> Result<()> {
        let (pending, workspace) = self.pending_payloads()?;
        let limit = usize::try_from(self.spec.limits().max_state_bytes()).expect("validated");
        encode_payloads(pending, workspace, limit, &self.name)
    }

    async fn prepare_payloads_async(&self, context: &StreamOperatorContext<'_>) -> Result<()> {
        let (pending, workspace) = self.pending_payloads()?;
        if pending.is_empty() {
            return Ok(());
        }
        let limit = usize::try_from(self.spec.limits().max_state_bytes()).expect("validated");
        let name = self.name.clone();
        context.check_cancelled()?;
        tokio::task::spawn_blocking(move || encode_payloads(pending, workspace, limit, &name))
            .await
            .map_err(|error| CalcFlowError::Internal {
                message: format!("ASOF payload checkpoint task failed: {error}"),
            })??;
        context.check_cancelled()
    }

    async fn prepare_deferred_index_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let Some(length) = self.deferred_index_len else {
            return Ok(());
        };
        let workspace = self.reserve_workspace(index_v2::workspace_bytes(
            &self.state,
            length,
            true,
            &self.name,
        )?)?;
        let limit = usize::try_from(self.spec.limits().max_state_bytes()).expect("validated");
        let (segment, _workspace) =
            index_v2::encode(&self.state, length, limit, context, workspace).await?;
        self.prepared = Some(PreparedSegment::new(segment));
        self.deferred_index_len = None;
        Ok(())
    }

    #[cfg(test)]
    pub(super) async fn prepare_checkpoint(
        &self,
        state: &State,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedCheckpoint> {
        let limit = usize::try_from(self.spec.limits().max_state_bytes()).expect("validated");
        let length = encoded_length(state, &self.name)?;
        let workspace =
            self.reserve_workspace(index_v2::workspace_bytes(state, length, true, &self.name)?)?;
        let (segment, workspace) = if state.left.is_empty() && state.right.is_empty() {
            (None, workspace)
        } else {
            let (segment, workspace) =
                index_v2::encode(state, length, limit, context, workspace).await?;
            (Some(PreparedSegment::new(segment)), workspace)
        };
        Ok(PreparedCheckpoint {
            segment,
            _workspace: workspace,
        })
    }

    async fn compact_prepared_async(&mut self, context: &StreamOperatorContext<'_>) -> Result<()> {
        if let Some(prepared) = self.prepared.as_ref().filter(|view| view.is_drained()) {
            let workspace = self.reserve_workspace(prepared.len() as u64)?;
            let retained_capacity = prepared.capacity();
            let prepared = prepared.clone();
            context.check_cancelled()?;
            let canonical = tokio::task::spawn_blocking(move || {
                let _workspace = workspace;
                prepared.canonical()
            })
            .await
            .map_err(|error| CalcFlowError::Internal {
                message: format!("ASOF checkpoint compaction task failed: {error}"),
            })?;
            context.check_cancelled()?;
            self.install_compacted_prepared(canonical, retained_capacity);
        }
        Ok(())
    }

    /// Replaces a drained view with its canonical bytes so the snapshot and
    /// later captures share one exact-capacity buffer and the drained base is
    /// released. The reservation bounds the copy while the base is alive.
    fn compact_prepared(&mut self) -> Result<()> {
        if let Some(prepared) = self.prepared.as_ref().filter(|view| view.is_drained()) {
            let _workspace = self.reserve_workspace(prepared.len() as u64)?;
            let retained_capacity = prepared.capacity();
            let canonical = prepared.canonical();
            self.install_compacted_prepared(canonical, retained_capacity);
        }
        Ok(())
    }

    fn install_compacted_prepared(&mut self, canonical: StateSegment, old_capacity: usize) {
        let new_capacity = canonical.bytes_arc().capacity();
        self.status.state_bytes = self
            .status
            .state_bytes
            .checked_sub(old_capacity as u64)
            .and_then(|bytes| bytes.checked_add(new_capacity as u64))
            .expect("compaction preserves a charged index allocation");
        self.prepared = Some(PreparedSegment::new(canonical));
    }

    pub(super) fn capture(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        self.ensure_prepared_sync()?;
        self.compact_prepared()?;
        self.ensure_payloads_sync()?;
        let mut metrics = self.status.clone();
        for side in [&mut metrics.left, &mut metrics.right] {
            side.watermark_micros = None;
            side.idle = false;
            side.ended = false;
        }
        metrics.output_watermark_micros = None;
        let metadata = Metadata {
            kind: "stream_asof_join",
            state_version: 2,
            layout_version: 2,
            accounting_version: 2,
            row_encoding: "arrow-batch-58.3.0",
            fingerprint: &self.fingerprint,
            epoch: epoch.as_u64(),
            terminal: self.terminal,
            next_output_sequence: self.next_output_sequence,
            metrics,
        };
        let inline_metadata = serde_json::to_value(metadata)
            .and_then(serde_json::from_value)
            .map_err(|error| mismatch(&error.to_string()))?;
        let mut segments = self
            .prepared
            .as_ref()
            .map(|segment| BTreeMap::from([(INDEX_SEGMENT.into(), segment.canonical())]))
            .unwrap_or_default();
        for (key, (batch, _)) in self.state.batches.iter() {
            let encoded = batch.encoded.get().expect("prepared ASOF payload");
            segments.insert(index_v2::batch_segment(*key), encoded.clone());
        }
        Ok(OperatorStateSnapshot {
            inline_metadata,
            segments,
        })
    }

    pub(super) fn decoded_snapshot(
        &self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<DecodedSnapshot> {
        let metadata = self.restore_metadata(snapshot)?;
        let version = metadata.state_version;
        let segment = snapshot_segment(snapshot, version)?;
        let workspace_bytes = if version == 1 {
            segment
                .map(|segment| restore_charge(segment.bytes(), self.spec.limits().max_state_rows()))
                .transpose()?
                .unwrap_or(0)
        } else {
            let payloads = snapshot
                .segments
                .iter()
                .filter(|(name, _)| name.as_str() != INDEX_SEGMENT)
                .try_fold(0_u64, |total, (_, value)| {
                    super::checked(&self.name, total, value.bytes().len() as u64)
                })?;
            super::checked(
                &self.name,
                segment.map_or(Ok(0), |segment| {
                    index_v2::restore_charge(segment.bytes(), self.spec.limits().max_state_rows())
                })?,
                payloads,
            )?
        };
        let workspace = self.reserve_workspace(workspace_bytes)?;
        let mut state = if version == 1 {
            segment
                .map(|segment| self.decode_state(segment))
                .transpose()?
                .unwrap_or_default()
        } else {
            self.decode_state_v2(snapshot, segment, &metadata.metrics)?
        };
        if version == 1 {
            let legacy = legacy_inventory(segment, self.spec.limits().max_state_rows())?;
            validate_gauges(&legacy, state.left.len() as u64, &metadata.metrics)?;
        }
        let prepared = if version == 1 && (!state.left.is_empty() || !state.right.is_empty()) {
            let length = encoded_length(&state, &self.name)?;
            // The v1 restore reservation already includes the original encoded
            // segment plus 384 bytes per identity. The v2 index is smaller than
            // those two allowances together, so migration needs no extra cap.
            Some(PreparedSegment::new(index_v2::encode_sync(
                &state,
                length,
                usize::try_from(self.spec.limits().max_state_bytes()).expect("validated"),
            )?))
        } else {
            segment.cloned().map(PreparedSegment::new)
        };
        let inventory =
            self.validate_restored_state(&state, prepared.as_ref(), version, &metadata.metrics)?;
        let mut metrics = metadata.metrics;
        metrics.state_bytes = inventory.bytes;
        validate_counters(&metrics, metadata.terminal, metadata.next_output_sequence)?;
        // Only validated, still-indexed payload rows enter Arrow chunks.
        // Gauges and provenance above use the original v1/v2 representation.
        state
            .left
            .migrate(&state.batches, self.spec.left(), &self.name)?;
        Ok(DecodedSnapshot {
            state,
            metrics,
            terminal: metadata.terminal,
            sequence: metadata.next_output_sequence,
            prepared,
            _workspace: workspace,
        })
    }

    fn restore_metadata<'a>(&self, snapshot: &'a OperatorStateSnapshot) -> Result<Metadata<'a>> {
        super::metadata::validate_shape(&snapshot.inline_metadata)?;
        let metadata = Metadata::deserialize(serde::de::value::MapDeserializer::new(
            snapshot
                .inline_metadata
                .iter()
                .map(|(key, value)| (key.as_str(), value)),
        ))
        .map_err(|error: serde_json::Error| mismatch(&error.to_string()))?;
        self.validate_metadata(&metadata)?;
        Ok(metadata)
    }

    fn validate_metadata(&self, metadata: &Metadata<'_>) -> Result<()> {
        let version_ok = match metadata.state_version {
            1 => {
                metadata.layout_version == 1
                    && metadata.accounting_version == 1
                    && metadata.row_encoding == "arrow-row-58.3.0"
            }
            2 => {
                metadata.layout_version == 2
                    && metadata.accounting_version == 2
                    && metadata.row_encoding == "arrow-batch-58.3.0"
            }
            _ => false,
        };
        if metadata.kind != "stream_asof_join"
            || !version_ok
            || metadata.fingerprint != self.fingerprint
        {
            return Err(mismatch(
                "ASOF kind, schema, configuration or state version differs",
            ));
        }
        Ok(())
    }

    fn validate_restored_state(
        &self,
        state: &State,
        prepared: Option<&PreparedSegment>,
        version: u32,
        metrics: &StreamAsofJoinStatus,
    ) -> Result<Inventory> {
        let inventory = state.inventory(prepared, &self.name)?;
        if version == 2 {
            validate_gauges(&inventory, state.left.len() as u64, metrics)?;
        } else if inventory.identities != metrics.state_rows
            || inventory.right_payloads != metrics.retained_right_rows
            || inventory.identity_only != metrics.identity_only_rows
        {
            return Err(mismatch("ASOF migrated state gauges differ"));
        }
        if inventory.identities > self.spec.limits().max_state_rows()
            || inventory.bytes > self.spec.limits().max_state_bytes()
        {
            return Err(mismatch("ASOF restored state exceeds current limits"));
        }
        Ok(inventory)
    }

    fn decode_state(&self, segment: &StateSegment) -> Result<State> {
        if hex::encode(Sha256::digest(segment.bytes())) != segment.sha256() {
            return Err(mismatch("ASOF segment checksum mismatch"));
        }
        let mut decoder = Decoder::new(segment.bytes(), self.spec.limits().max_state_rows());
        let (left_count, bucket_count) = decoder.header()?;
        let mut state = State::default();
        for _ in 0..left_count {
            self.decode_left_row(&mut decoder, &mut state)?;
        }
        let mut next_right_id = 0;
        for _ in 0..bucket_count {
            self.decode_bucket(&mut decoder, &mut state, &mut next_right_id)?;
        }
        decoder.finish("ASOF segment contains trailing data")?;
        Ok(state)
    }

    fn decode_left_row(&self, decoder: &mut Decoder<'_>, state: &mut State) -> Result<()> {
        decoder.row()?;
        let time = decoder.time()?;
        let key = Encoding::from_slice(decoder.blob()?);
        let sequence = Encoding::from_slice(decoder.blob()?);
        let bytes = decoder.payload_blob()?;
        let record = self.validate_payload(bytes, false, &(time, key.clone(), sequence.clone()))?;
        let identity = (time, key, sequence);
        if state
            .left
            .last_key_value()
            .is_some_and(|(last, _)| last >= (&identity.0, &identity.1, &identity.2))
        {
            return Err(mismatch("ASOF left identity order is not strict"));
        }
        let payload = legacy_payload((0, state.left.len() as u64), bytes, record)?;
        let payload = state.attach(&payload);
        state.left.insert(identity, payload);
        Ok(())
    }

    fn decode_bucket(
        &self,
        decoder: &mut Decoder<'_>,
        state: &mut State,
        next_id: &mut u64,
    ) -> Result<()> {
        let key = Encoding::from_slice(decoder.blob()?);
        super::identity::validate(&key, &self.schemas[1], self.spec.right().keys())?;
        if state
            .right
            .last_key_value()
            .is_some_and(|(last, _)| last >= &key)
        {
            return Err(mismatch("ASOF right key order is not strict"));
        }
        let count = decoder.count()?;
        if count == 0 {
            return Err(mismatch("ASOF empty right bucket is noncanonical"));
        }
        let mut bucket = RightBucket::new();
        for _ in 0..count {
            self.decode_right_row(decoder, &key, &mut bucket, state, next_id)?;
        }
        state.right.insert(key, bucket);
        Ok(())
    }

    fn decode_right_row(
        &self,
        decoder: &mut Decoder<'_>,
        key: &Encoding,
        bucket: &mut RightBucket,
        state: &mut State,
        next_id: &mut u64,
    ) -> Result<()> {
        decoder.row()?;
        let time = decoder.time()?;
        let sequence = self.decode_right_sequence(decoder)?;
        let bytes = decoder.payload_blob()?;
        let record = self.validate_right_payload(bytes, time, key, &sequence)?;
        let identity = (time, sequence);
        if bucket
            .last_key_value()
            .is_some_and(|(last, _)| last >= (&identity.0, &identity.1))
        {
            return Err(mismatch("ASOF right identity order is not strict"));
        }
        let payload = record
            .map(|record| -> Result<_> {
                let payload = legacy_payload((1, *next_id), bytes, record)?;
                *next_id += 1;
                Ok(state.attach(&payload))
            })
            .transpose()?;
        bucket.insert(identity, payload);
        Ok(())
    }

    fn validate_right_payload(
        &self,
        bytes: &[u8],
        time: i64,
        key: &Encoding,
        sequence: &Encoding,
    ) -> Result<Option<datafusion::arrow::record_batch::RecordBatch>> {
        if !bytes.is_empty() {
            return self
                .validate_payload(bytes, true, &(time, key.clone(), sequence.clone()))
                .map(Some);
        }
        Ok(None)
    }

    fn decode_right_sequence(&self, decoder: &mut Decoder<'_>) -> Result<Encoding> {
        let sequence = Encoding::from_slice(decoder.blob()?);
        super::identity::validate(&sequence, &self.schemas[1], self.spec.right().sequence_by())?;
        Ok(sequence)
    }

    fn validate_payload(
        &self,
        bytes: &[u8],
        right: bool,
        identity: &super::state::LeftOrder,
    ) -> Result<datafusion::arrow::record_batch::RecordBatch> {
        let row = super::codec::decode_batch(bytes, &self.schema_digests[usize::from(right)])
            .map_err(|_| mismatch("ASOF invalid Arrow row encoding"))?;
        let side = if right {
            self.spec.right()
        } else {
            self.spec.left()
        };
        if row.schema() != self.schemas[usize::from(right)] {
            return Err(mismatch("ASOF row schema differs"));
        }
        validate_nonnull_identity(&row, side)?;
        let time = row
            .column(row.schema().index_of(side.event_time()).expect("validated"))
            .as_any()
            .downcast_ref::<datafusion::arrow::array::TimestampMicrosecondArray>()
            .expect("validated")
            .value(0);
        let key = super::state::encoded_columns(&row, 0, side.keys())?;
        let sequence = super::state::encoded_columns(&row, 0, side.sequence_by())?;
        if &(time, key, sequence) != identity {
            return Err(mismatch("ASOF row payload identity differs from index"));
        }
        // Reuse the already-validated schema owner instead of retaining one
        // parsed IPC schema per sparse payload batch.
        row.with_schema(self.schemas[usize::from(right)].clone())
            .map_err(|_| mismatch("ASOF row schema differs"))
    }

    fn decode_state_v2(
        &self,
        snapshot: &OperatorStateSnapshot,
        segment: Option<&StateSegment>,
        metrics: &StreamAsofJoinStatus,
    ) -> Result<State> {
        let Some(segment) = segment else {
            return Ok(State::default());
        };
        verify_checksum(segment)?;
        let batches = self.decode_payload_batches(snapshot)?;
        validate_batch_ranges(&batches, metrics)?;
        let state = index_v2::decode(
            segment.bytes(),
            &batches,
            self.spec.limits().max_state_rows(),
        )?;
        self.validate_indexed_rows(&state)?;
        Ok(state)
    }

    fn decode_payload_batches(
        &self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<BTreeMap<BatchKey, Arc<PayloadBatch>>> {
        u32::try_from(snapshot.segments.len().saturating_sub(1))
            .map_err(|_| mismatch("ASOF payload batches exceed compact reference range"))?;
        let mut batches = BTreeMap::new();
        for (name, encoded) in &snapshot.segments {
            if name == INDEX_SEGMENT {
                continue;
            }
            let key = index_v2::parse_batch_segment(name)?;
            let payload = self.decode_payload_batch(key, encoded)?;
            if batches.insert(key, payload).is_some() {
                return Err(mismatch("ASOF duplicate batch segment"));
            }
        }
        Ok(batches)
    }

    fn decode_payload_batch(
        &self,
        key: BatchKey,
        encoded: &StateSegment,
    ) -> Result<Arc<PayloadBatch>> {
        verify_checksum(encoded)?;
        let side = usize::from(key.0);
        let record = super::codec::decode_table_batch(
            encoded.bytes(),
            &self.schema_digests[side],
            self.spec.limits().max_state_rows().min(u64::from(u32::MAX)),
        )
        .map_err(|_| mismatch("ASOF invalid Arrow batch encoding"))?;
        if record.schema() != self.schemas[side]
            || record.num_rows() as u64 > self.spec.limits().max_state_rows()
        {
            return Err(mismatch("ASOF batch schema or row count differs"));
        }
        let record = record
            .with_schema(self.schemas[side].clone())
            .map_err(|_| mismatch("ASOF batch schema differs"))?;
        let (encoded_charge_bytes, body_bytes) =
            super::workspace::payload_encoded_bound(&record, &self.name)
                .map_err(|_| mismatch("ASOF payload bound cannot be computed"))?;
        let actual_body = super::codec::payload_body_bytes(encoded.bytes())
            .map_err(|_| mismatch("ASOF invalid Arrow batch framing"))?;
        if encoded.bytes().len() as u64 > encoded_charge_bytes || actual_body > body_bytes {
            return Err(mismatch("ASOF payload exceeds its memory-accounting bound"));
        }
        Ok(Arc::new(PayloadBatch {
            key,
            record: Arc::new(record),
            body_bytes,
            encoded_charge_bytes,
            encoded: OnceLock::from(encoded.clone()),
        }))
    }

    fn validate_indexed_rows(&self, state: &State) -> Result<()> {
        let identities = self.indexed_batch_identities(state)?;
        self.validate_left_indexed_rows(state, &identities)?;
        self.validate_right_indexed_rows(state, &identities)
    }

    fn indexed_batch_identities(&self, state: &State) -> Result<IdentityCache> {
        let mut identities = BTreeMap::new();
        for (key, (batch, _)) in state.batches.iter() {
            let side = if key.0 == 0 {
                self.spec.left()
            } else {
                self.spec.right()
            };
            validate_nonnull_identity(&batch.record, side)?;
            identities.insert(
                *key,
                (
                    super::state::encode_columns(&batch.record, side.keys())?,
                    super::state::encode_columns(&batch.record, side.sequence_by())?,
                ),
            );
        }
        Ok(identities)
    }

    fn validate_left_indexed_rows(&self, state: &State, identities: &IdentityCache) -> Result<()> {
        for (identity, payload) in state.left.unordered_iter() {
            super::identity::validate(identity.1, &self.schemas[0], self.spec.left().keys())?;
            super::identity::validate(
                identity.2,
                &self.schemas[0],
                self.spec.left().sequence_by(),
            )?;
            validate_index_identity(
                identity,
                state.batches.view(payload),
                identities,
                self.spec.left(),
            )?;
        }
        Ok(())
    }

    fn validate_right_indexed_rows(&self, state: &State, identities: &IdentityCache) -> Result<()> {
        for (key, bucket) in &state.right {
            super::identity::validate(key, &self.schemas[1], self.spec.right().keys())?;
            self.validate_right_indexed_bucket(key, bucket, identities, &state.batches)?;
        }
        Ok(())
    }

    fn validate_right_indexed_bucket(
        &self,
        key: &Encoding,
        bucket: &RightBucket,
        identities: &IdentityCache,
        batches: &super::state::PayloadPool,
    ) -> Result<()> {
        for ((time, sequence), payload) in bucket {
            super::identity::validate(sequence, &self.schemas[1], self.spec.right().sequence_by())?;
            if let Some(payload) = payload {
                let identity = (time, key, sequence);
                validate_index_identity(
                    identity,
                    batches.view(*payload),
                    identities,
                    self.spec.right(),
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn state_fingerprint(
        spec: &super::StreamAsofJoinSpec,
        schemas: &[datafusion::arrow::datatypes::SchemaRef; 3],
    ) -> Result<String> {
        let schemas = schemas
            .iter()
            .map(|schema| {
                let mut tracker = datafusion::arrow::ipc::writer::DictionaryTracker::new(true);
                let encoded = datafusion::arrow::ipc::convert::IpcSchemaEncoder::new()
                    .with_dictionary_tracker(&mut tracker)
                    .schema_to_fb(schema);
                hex::encode(encoded.finished_data())
            })
            .collect::<Vec<_>>();
        let value = serde_json::json!({"spec": spec, "schemas": schemas});
        Ok(hex::encode(Sha256::digest(
            crate::canonical_json(&value)?.as_bytes(),
        )))
    }

    pub(crate) fn restore_with_progress(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        progress: &IngressProgressSnapshot,
        output_frontier: Option<crate::EventTime>,
    ) -> Result<()> {
        let decoded = self.decoded_snapshot(snapshot)?;
        validate_progress(
            &decoded.state,
            self.spec.tolerance_micros(),
            decoded.terminal,
            progress,
            output_frontier,
        )?;
        self.install_restored(snapshot, decoded);
        self.observe(progress);
        self.status.output_watermark_micros = output_frontier;
        Ok(())
    }

    pub(super) fn install_restored(
        &mut self,
        _snapshot: &OperatorStateSnapshot,
        decoded: DecodedSnapshot,
    ) {
        self.state = decoded.state;
        self.state.rebuild_right_minima();
        self.status = decoded.metrics;
        self.terminal = decoded.terminal;
        self.next_output_sequence = decoded.sequence;
        self.prepared = decoded.prepared;
        self.deferred_index_len = None;
        self.swept = None;
    }
}

fn validate_batch_ranges(
    batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
    metrics: &StreamAsofJoinStatus,
) -> Result<()> {
    let accepted = [metrics.left.accepted_rows, metrics.right.accepted_rows];
    let mut previous_end = [0_u64; 2];
    for ((side, start), batch) in batches {
        let index = usize::from(*side);
        let end = start
            .checked_add(batch.record.num_rows() as u64)
            .ok_or_else(|| mismatch("ASOF restored batch ID range overflowed"))?;
        if batch.record.num_rows() == 0 || *start < previous_end[index] || end > accepted[index] {
            return Err(mismatch(
                "ASOF restored batch ID lies outside accepted rows",
            ));
        }
        previous_end[index] = end;
    }
    Ok(())
}

fn snapshot_segment(
    snapshot: &OperatorStateSnapshot,
    version: u32,
) -> Result<Option<&StateSegment>> {
    if version == 1 {
        if snapshot.segments.len() > 1 || snapshot.segments.keys().any(|key| key != SEGMENT) {
            return Err(mismatch("unexpected ASOF legacy segment inventory"));
        }
        Ok(snapshot.segments.get(SEGMENT))
    } else {
        let index = snapshot.segments.get(INDEX_SEGMENT);
        if snapshot
            .segments
            .keys()
            .any(|name| name != INDEX_SEGMENT && index_v2::parse_batch_segment(name).is_err())
            || (index.is_none() && !snapshot.segments.is_empty())
        {
            return Err(mismatch("unexpected ASOF columnar segment inventory"));
        }
        Ok(index)
    }
}

fn legacy_inventory(segment: Option<&StateSegment>, max_rows: u64) -> Result<Inventory> {
    let Some(segment) = segment else {
        return Ok(Inventory::default());
    };
    let mut decoder = Decoder::new(segment.bytes(), max_rows);
    let (left_count, bucket_count) = decoder.header()?;
    let mut inventory = Inventory {
        bytes: legacy_add(64, segment.bytes_arc().capacity() as u64)?,
        ..Inventory::default()
    };
    for _ in 0..left_count {
        legacy_left_inventory_row(&mut decoder, &mut inventory)?;
    }
    for _ in 0..bucket_count {
        legacy_right_inventory_bucket(&mut decoder, &mut inventory)?;
    }
    decoder.finish("ASOF legacy segment has trailing data")?;
    Ok(inventory)
}

fn legacy_left_inventory_row(decoder: &mut Decoder<'_>, inventory: &mut Inventory) -> Result<()> {
    let (key, sequence, payload) = read_legacy_left_row(decoder)?;
    inventory.identities = legacy_add(inventory.identities, 1)?;
    inventory.bytes = legacy_add(inventory.bytes, 384)?;
    for value in [key, sequence, payload] {
        charge_legacy_blob(inventory, value)?;
    }
    Ok(())
}

fn read_legacy_left_row<'a>(decoder: &mut Decoder<'a>) -> Result<(&'a [u8], &'a [u8], &'a [u8])> {
    decoder.row()?;
    decoder.time()?;
    Ok((decoder.blob()?, decoder.blob()?, decoder.blob()?))
}

fn legacy_right_inventory_bucket(
    decoder: &mut Decoder<'_>,
    inventory: &mut Inventory,
) -> Result<()> {
    let key = decoder.blob()?;
    charge_legacy_blob(inventory, key)?;
    let rows = decoder.count()?;
    for _ in 0..rows {
        legacy_right_inventory_row(decoder, inventory)?;
    }
    Ok(())
}

fn legacy_right_inventory_row(decoder: &mut Decoder<'_>, inventory: &mut Inventory) -> Result<()> {
    let (sequence, payload) = read_legacy_right_row(decoder)?;
    inventory.identities = legacy_add(inventory.identities, 1)?;
    inventory.bytes = legacy_add(inventory.bytes, 320)?;
    charge_legacy_blob(inventory, sequence)?;
    charge_legacy_right_payload(inventory, payload)
}

fn read_legacy_right_row<'a>(decoder: &mut Decoder<'a>) -> Result<(&'a [u8], &'a [u8])> {
    decoder.row()?;
    decoder.time()?;
    Ok((decoder.blob()?, decoder.blob()?))
}

fn charge_legacy_right_payload(inventory: &mut Inventory, payload: &[u8]) -> Result<()> {
    if payload.is_empty() {
        inventory.identity_only = legacy_add(inventory.identity_only, 1)?;
    } else {
        inventory.right_payloads = legacy_add(inventory.right_payloads, 1)?;
        charge_legacy_blob(inventory, payload)?;
    }
    Ok(())
}

fn charge_legacy_blob(inventory: &mut Inventory, value: &[u8]) -> Result<()> {
    inventory.bytes = legacy_add(inventory.bytes, legacy_add(64, value.len() as u64)?)?;
    Ok(())
}

fn legacy_add(base: u64, value: u64) -> Result<u64> {
    base.checked_add(value)
        .ok_or_else(|| mismatch("ASOF legacy inventory charge overflowed"))
}

fn legacy_payload(
    key: BatchKey,
    bytes: &[u8],
    record: datafusion::arrow::record_batch::RecordBatch,
) -> Result<RowPayload> {
    let (encoded_charge_bytes, body_bytes) =
        super::workspace::payload_encoded_bound(&record, "asof")
            .map_err(|_| mismatch("ASOF legacy payload bound cannot be computed"))?;
    Ok(RowPayload {
        batch: Arc::new(PayloadBatch {
            key,
            record: Arc::new(record),
            body_bytes,
            encoded_charge_bytes,
            encoded: OnceLock::from(StateSegment::new(bytes.to_vec())),
        }),
        row: 0,
    })
}

fn verify_checksum(segment: &StateSegment) -> Result<()> {
    if hex::encode(Sha256::digest(segment.bytes())) != segment.sha256() {
        return Err(mismatch("ASOF segment checksum mismatch"));
    }
    Ok(())
}

fn validate_index_identity(
    identity: super::state::LeftView<'_>,
    payload: PayloadView<'_>,
    cache: &IdentityCache,
    side: &super::AsofJoinSide,
) -> Result<()> {
    let record = &payload.batch.record;
    let (keys, sequences) = cache
        .get(&payload.batch.key)
        .ok_or_else(|| mismatch("ASOF indexed batch is missing"))?;
    if super::admission::times(record, side).value(payload.row) != *identity.0
        || keys.row(payload.row).as_slice() != identity.1.as_slice()
        || sequences.row(payload.row).as_slice() != identity.2.as_slice()
    {
        return Err(mismatch("ASOF batch row identity differs from index"));
    }
    Ok(())
}

fn validate_gauges(
    inventory: &Inventory,
    pending_left_rows: u64,
    metrics: &StreamAsofJoinStatus,
) -> Result<()> {
    if inventory.identities != metrics.state_rows
        || inventory.bytes != metrics.state_bytes
        || inventory.right_payloads != metrics.retained_right_rows
        || inventory.identity_only != metrics.identity_only_rows
        || pending_left_rows != metrics.pending_left_rows
    {
        return Err(mismatch("ASOF recomputed state charge or gauges differ"));
    }
    Ok(())
}

fn validate_nonnull_identity(
    row: &datafusion::arrow::record_batch::RecordBatch,
    side: &super::AsofJoinSide,
) -> Result<()> {
    if super::admission::identity_column_names(side)
        .any(|column| super::admission::identity_column_nulls(row, column))
    {
        return Err(mismatch("ASOF restored identity contains null"));
    }
    Ok(())
}

fn mismatch(message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, CancellationToken, EdgeCollector,
        EventTime, IngressProgress, IngressState, JsonMap, OperatorMetadata, StreamJobContext,
        StreamOperator, StreamOperatorContext, StreamingFailureReason,
    };
    use datafusion::arrow::{
        array::{Int8Array, Int64Array, StringArray, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };
    use std::time::Duration;

    #[test]
    fn v2_index_orders_right_buckets_independently_of_hash_insertion() {
        let mut forward = State::default();
        let mut reverse = State::default();
        for (state, keys) in [(&mut forward, [1_u8, 2]), (&mut reverse, [2, 1])] {
            for key in keys {
                let mut bucket = RightBucket::default();
                bucket.insert((1, Encoding::from_slice(&[1])), None);
                state.right.insert(Encoding::from_slice(&[key]), bucket);
            }
        }
        let length = encoded_length(&forward, "asof").unwrap();
        assert_eq!(length, encoded_length(&reverse, "asof").unwrap());
        let forward = index_v2::encode_sync(&forward, length, 1_024).unwrap();
        let reverse = index_v2::encode_sync(&reverse, length, 1_024).unwrap();
        assert_eq!(forward.bytes(), reverse.bytes());
    }

    fn progress(left: Option<i64>, right: Option<i64>) -> IngressProgressSnapshot {
        IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(IngressState::Active, left.map(EventTime::from_micros)),
            ),
            (
                "right".into(),
                IngressProgress::new(IngressState::Active, right.map(EventTime::from_micros)),
            ),
        ]))
    }

    fn dummy_payload() -> RowPayload {
        RowPayload {
            batch: Arc::new(PayloadBatch {
                key: (0, 0),
                record: Arc::new(RecordBatch::new_empty(Arc::new(Schema::empty()))),
                encoded: OnceLock::new(),
                encoded_charge_bytes: 0,
                body_bytes: 0,
            }),
            row: 0,
        }
    }

    fn legacy_snapshot(
        operator: &mut StreamAsofJoinOperator,
        segment: StateSegment,
        left_rows: u64,
        right_rows: u64,
    ) -> OperatorStateSnapshot {
        let mut snapshot = operator.capture(Epoch::INITIAL).unwrap();
        for (field, value) in [
            ("state_version", serde_json::json!(1)),
            ("layout_version", serde_json::json!(1)),
            ("accounting_version", serde_json::json!(1)),
            ("row_encoding", serde_json::json!("arrow-row-58.3.0")),
        ] {
            snapshot.inline_metadata.insert(field.into(), value);
        }
        let metrics = snapshot
            .inline_metadata
            .get_mut("metrics")
            .unwrap()
            .as_object_mut()
            .unwrap();
        metrics.insert("pending_left_rows".into(), serde_json::json!(left_rows));
        metrics.insert(
            "state_rows".into(),
            serde_json::json!(left_rows + right_rows),
        );
        metrics.insert("retained_right_rows".into(), serde_json::json!(right_rows));
        metrics.insert(
            "state_bytes".into(),
            serde_json::json!(
                legacy_inventory(Some(&segment), left_rows + right_rows)
                    .unwrap()
                    .bytes
            ),
        );
        metrics["left"]["accepted_rows"] = serde_json::json!(left_rows);
        metrics["right"]["accepted_rows"] = serde_json::json!(right_rows);
        snapshot.segments.insert(SEGMENT.into(), segment);
        snapshot
    }

    #[test]
    fn singleton_left_migration_fits_restore_workspace_and_identity_charge() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
        ]));
        let side = |prefix: &str| {
            AsofJoinSide::new(
                vec!["key".into()],
                "time".into(),
                vec!["seq".into()],
                prefix.into(),
            )
            .unwrap()
        };
        let spec = super::super::StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(10_000, 32 << 20).unwrap(),
        )
        .unwrap();
        for count in [1_u64, 2, 3, 4, 16, 64] {
            let mut operator =
                StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec.clone())
                    .unwrap();
            let mut bytes = MAGIC.to_vec();
            bytes.extend_from_slice(&count.to_le_bytes());
            bytes.extend_from_slice(&0_u64.to_le_bytes());
            for ordinal in 0..count {
                let ordinal = i64::try_from(ordinal).unwrap();
                let record = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StringArray::from(vec!["A"])),
                        Arc::new(
                            TimestampMicrosecondArray::from(vec![ordinal]).with_timezone("UTC"),
                        ),
                        Arc::new(Int64Array::from(vec![ordinal])),
                    ],
                )
                .unwrap();
                let key =
                    super::super::state::encoded_columns(&record, 0, spec.left().keys()).unwrap();
                let sequence =
                    super::super::state::encoded_columns(&record, 0, spec.left().sequence_by())
                        .unwrap();
                let payload =
                    super::super::codec::encode_batch(&record, 1 << 20, &mut Vec::new()).unwrap();
                bytes.extend_from_slice(&ordinal.to_le_bytes());
                for blob in [key.as_slice(), sequence.as_slice(), payload.as_slice()] {
                    bytes.extend_from_slice(&(blob.len() as u64).to_le_bytes());
                    bytes.extend_from_slice(blob);
                }
            }
            let v1 = legacy_snapshot(&mut operator, StateSegment::new(bytes), count, 0);
            assert_restore_allocation_bound(&operator, &v1);
            operator.restore(&v1).unwrap();
            let v2 = operator.capture(Epoch::INITIAL).unwrap();
            assert_restore_allocation_bound(&operator, &v2);
            operator
                .state
                .commit_left_prefix(usize::try_from(count / 2).unwrap());
            assert!(
                operator.state.inventory(None, "asof").unwrap().bytes
                    <= operator.status.state_bytes
            );
        }
    }

    fn assert_restore_allocation_bound(
        operator: &StreamAsofJoinOperator,
        snapshot: &OperatorStateSnapshot,
    ) {
        let metadata = operator.restore_metadata(snapshot).unwrap();
        // Validate the full restore and retain its actual workspace. Existing
        // payload allocations belong to state; measure the new migration's
        // temporary grouping and column storage independently.
        let DecodedSnapshot {
            _workspace: reservation,
            ..
        } = operator.decoded_snapshot(snapshot).unwrap();
        let mut state = if metadata.state_version == 1 {
            operator.decode_state(&snapshot.segments[SEGMENT]).unwrap()
        } else {
            operator
                .decode_state_v2(
                    snapshot,
                    snapshot.segments.get(INDEX_SEGMENT),
                    &metadata.metrics,
                )
                .unwrap()
        };
        let allocation = allocation_counter::measure(|| {
            state
                .left
                .migrate(&state.batches, operator.spec.left(), &operator.name)
                .unwrap();
        });
        let workspace = reservation.size() as u64;
        assert!(
            allocation.bytes_max <= workspace,
            "left migration peak exceeds restore workspace: {allocation:?}, workspace={workspace}"
        );
        let before = state.inventory(None, "asof").unwrap().bytes;
        let left =
            allocation_counter::measure(|| state.left = super::super::state::LeftState::default());
        let left_charge = before - state.inventory(None, "asof").unwrap().bytes;
        let freed = u64::try_from(-left.bytes_current).unwrap();
        assert!(
            freed <= left_charge,
            "restored chunk metadata exceeds legacy identity charge: freed={freed}, charge={left_charge}"
        );
    }

    #[test]
    fn legacy_row_snapshot_restores_into_columnar_state() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
        ]));
        let side = |prefix: &str| {
            AsofJoinSide::new(
                vec!["key".into()],
                "time".into(),
                vec!["seq".into()],
                prefix.into(),
            )
            .unwrap()
        };
        let spec = super::super::StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(100, 1 << 20).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec.clone())
                .unwrap();
        let row = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A"])),
                Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let key = super::super::state::encoded_columns(&row, 0, spec.left().keys()).unwrap();
        let sequence =
            super::super::state::encoded_columns(&row, 0, spec.left().sequence_by()).unwrap();
        let payload = super::super::codec::encode_batch(&row, 1 << 20, &mut Vec::new()).unwrap();
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&100_i64.to_le_bytes());
        for blob in [key.as_slice(), sequence.as_slice(), payload.as_slice()] {
            bytes.extend_from_slice(&(blob.len() as u64).to_le_bytes());
            bytes.extend_from_slice(blob);
        }
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&key);
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&100_i64.to_le_bytes());
        for blob in [sequence.as_slice(), payload.as_slice()] {
            bytes.extend_from_slice(&(blob.len() as u64).to_le_bytes());
            bytes.extend_from_slice(blob);
        }
        let segment = StateSegment::new(bytes);
        let snapshot = legacy_snapshot(&mut operator, segment, 1, 1);
        operator.restore(&snapshot).unwrap();
        assert_eq!(operator.state.left.len(), 1);
        let upgraded = operator.capture(Epoch::INITIAL).unwrap();
        assert_eq!(upgraded.inline_metadata["state_version"], 2);
        assert!(upgraded.segments.contains_key(INDEX_SEGMENT));
        assert_eq!(upgraded.segments.len(), 3);
    }

    #[test]
    fn near_limit_multirow_legacy_snapshot_restores_and_restarts_as_v2() {
        const ROWS: u64 = 64;
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let side = |prefix: &str| {
            AsofJoinSide::new(
                vec!["key".into()],
                "time".into(),
                vec!["seq".into()],
                prefix.into(),
            )
            .unwrap()
        };
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&ROWS.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        for ordinal in 0..ROWS {
            let ordinal = i64::try_from(ordinal).unwrap();
            let row = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec!["A"])),
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![100 + ordinal]).with_timezone("UTC"),
                    ),
                    Arc::new(Int64Array::from(vec![ordinal])),
                    Arc::new(StringArray::from(vec!["v".repeat(2_048)])),
                ],
            )
            .unwrap();
            let key = super::super::state::encoded_columns(&row, 0, &["key".into()]).unwrap();
            let sequence = super::super::state::encoded_columns(&row, 0, &["seq".into()]).unwrap();
            let payload =
                super::super::codec::encode_batch(&row, 1 << 20, &mut Vec::new()).unwrap();
            bytes.extend_from_slice(&(100 + ordinal).to_le_bytes());
            for blob in [key.as_slice(), sequence.as_slice(), payload.as_slice()] {
                bytes.extend_from_slice(&(blob.len() as u64).to_le_bytes());
                bytes.extend_from_slice(blob);
            }
        }
        let segment = StateSegment::new(bytes);
        let old_bytes = legacy_inventory(Some(&segment), ROWS).unwrap().bytes;
        let limit = old_bytes + 4_096;
        assert!(restore_charge(segment.bytes(), ROWS).unwrap() < limit);
        let spec = super::super::StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(ROWS, limit).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec.clone())
                .unwrap();
        let decoded = operator.decode_state(&segment).unwrap();
        let index_length = encoded_length(&decoded, "asof").unwrap();
        let estimated = decoded
            .inventory(
                Some(&PreparedSegment::new(StateSegment::new(vec![
                    0;
                    usize::try_from(index_length).unwrap()
                ]))),
                "asof",
            )
            .unwrap()
            .bytes;
        assert!(
            estimated <= limit,
            "v1={old_bytes} v2={estimated} limit={limit}"
        );
        let snapshot = legacy_snapshot(&mut operator, segment, ROWS, 0);
        operator.restore(&snapshot).unwrap();
        assert_eq!(operator.state.left.len() as u64, ROWS);
        assert!(operator.status.state_bytes <= limit);
        let upgraded = operator.capture(Epoch::INITIAL).unwrap();
        assert_eq!(upgraded.inline_metadata["state_version"], 2);
        let mut restarted =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
        restarted.restore(&upgraded).unwrap();
        assert_eq!(restarted.state.left.len() as u64, ROWS);
    }

    #[test]
    fn one_wide_legacy_row_restores_at_original_workspace_limit() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let row = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["A"])),
                Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(StringArray::from(vec!["v".repeat(48 * 1024)])),
            ],
        )
        .unwrap();
        let key = super::super::state::encoded_columns(&row, 0, &["key".into()]).unwrap();
        let sequence = super::super::state::encoded_columns(&row, 0, &["seq".into()]).unwrap();
        let payload = super::super::codec::encode_batch(&row, 1 << 20, &mut Vec::new()).unwrap();
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&100_i64.to_le_bytes());
        for blob in [key.as_slice(), sequence.as_slice(), payload.as_slice()] {
            bytes.extend_from_slice(&(blob.len() as u64).to_le_bytes());
            bytes.extend_from_slice(blob);
        }
        let segment = StateSegment::new(bytes);
        let limit = restore_charge(segment.bytes(), 1).unwrap();
        assert!(legacy_inventory(Some(&segment), 1).unwrap().bytes <= limit);
        let side = |prefix: &str| {
            AsofJoinSide::new(
                vec!["key".into()],
                "time".into(),
                vec!["seq".into()],
                prefix.into(),
            )
            .unwrap()
        };
        let spec = super::super::StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(1, limit).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec.clone())
                .unwrap();
        let snapshot = legacy_snapshot(&mut operator, segment, 1, 0);
        operator.restore(&snapshot).unwrap();
        let upgraded = operator.capture(Epoch::INITIAL).unwrap();
        let mut restarted =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
        restarted.restore(&upgraded).unwrap();
        assert_eq!(restarted.state.left.len(), 1);
    }

    #[test]
    fn v2_preflight_uses_encoded_rows_when_metadata_understates_state() {
        use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
        ]));
        let side = |prefix: &str| {
            AsofJoinSide::new(
                vec!["key".into()],
                "time".into(),
                vec!["seq".into()],
                prefix.into(),
            )
            .unwrap()
        };
        let spec = super::super::StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(32, 1 << 20).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
        operator.state.right.insert(
            Encoding::from_slice(&[1]),
            (0..32)
                .map(|ordinal| ((ordinal, Encoding::from_slice(&[1])), None))
                .collect(),
        );
        let length = encoded_length(&operator.state, "asof").unwrap();
        let index = index_v2::encode_sync(&operator.state, length, 1 << 20).unwrap();
        operator.prepared = Some(PreparedSegment::new(index));
        let mut snapshot = operator.capture(Epoch::INITIAL).unwrap();
        snapshot.inline_metadata.get_mut("metrics").unwrap()["state_rows"] = serde_json::json!(0);
        let encoded = snapshot.segments[INDEX_SEGMENT].bytes().len();
        assert!(
            index_v2::restore_charge(snapshot.segments[INDEX_SEGMENT].bytes(), 32,).unwrap()
                > encoded as u64 + 1
        );
        operator.runtime.pool = Arc::new(GreedyMemoryPool::new(encoded + 1)) as Arc<dyn MemoryPool>;
        let error = operator.decoded_snapshot(&snapshot).err().unwrap();
        assert!(matches!(
            error,
            CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn v2_restore_rejects_batch_id_at_next_admission_boundary() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
        ]));
        let side = |prefix: &str| {
            AsofJoinSide::new(
                vec!["key".into()],
                "time".into(),
                vec!["seq".into()],
                prefix.into(),
            )
            .unwrap()
        };
        let spec = super::super::StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(10, 1 << 20).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["A"])),
                Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", batch, &context, &mut output)
            .await
            .unwrap();
        let mut snapshot = operator.capture(Epoch::INITIAL).unwrap();
        let payload = snapshot.segments.remove("asof-batch-0-0").unwrap();
        snapshot.segments.insert("asof-batch-0-1".into(), payload);
        let mut index = snapshot.segments[INDEX_SEGMENT].bytes().to_vec();
        let id_offset = index.len() - 16;
        index[id_offset..id_offset + 8].copy_from_slice(&1_u64.to_le_bytes());
        snapshot
            .segments
            .insert(INDEX_SEGMENT.into(), StateSegment::new(index));
        let before = operator.status();
        assert!(matches!(
            operator.restore(&snapshot),
            Err(CalcFlowError::CheckpointMismatch { .. })
        ));
        assert_eq!(operator.status(), before);
    }

    #[tokio::test]
    async fn narrow_column_v2_checkpoint_restores_with_same_state_limit() {
        let mut fields = vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
        ];
        fields.extend((0..29).map(|index| Field::new(format!("c{index}"), DataType::Int8, false)));
        let schema = Arc::new(Schema::new(fields));
        let side = |prefix: &str| {
            AsofJoinSide::new(
                vec!["key".into()],
                "time".into(),
                vec!["seq".into()],
                prefix.into(),
            )
            .unwrap()
        };
        let spec = super::super::StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(2, 128 * 1024).unwrap(),
        )
        .unwrap();
        let mut columns: Vec<datafusion::arrow::array::ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["A"])),
            Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![1])),
        ];
        columns.extend((0..29).map(|_| Arc::new(Int8Array::from(vec![7])) as _));
        let record = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
        let mut operator =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec.clone())
                .unwrap();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", batch, &context, &mut output)
            .await
            .unwrap();
        let snapshot = operator.capture(Epoch::INITIAL).unwrap();
        let mut restored =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
        restored.restore(&snapshot).unwrap();
        assert_eq!(restored.status().state_rows, 1);
    }

    #[test]
    fn asof_restore_reserves_identity_only_validation_scratch() {
        let key = vec![1_u8; 1_048_576];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&key);
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&100_i64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.push(1);
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        assert!(restore_charge(&bytes, 1).unwrap() >= (bytes.len() + key.len()) as u64 + 384);
    }

    #[test]
    fn asof_restore_rejects_identity_only_that_can_still_match_pending() {
        let mut state = State::default();
        let payload = state.attach(&dummy_payload());
        state.left.insert(
            (105, Encoding::from_slice(&[1]), Encoding::from_slice(&[1])),
            payload,
        );
        state.right.insert(
            Encoding::from_slice(&[1]),
            RightBucket::from_iter([((100, Encoding::from_slice(&[1])), None)]),
        );
        assert!(
            validate_progress(
                &state,
                10,
                false,
                &progress(Some(100), Some(100)),
                Some(EventTime::from_micros(99))
            )
            .is_err()
        );
    }

    #[test]
    fn asof_restore_rejects_ready_pending_before_stale_frontier() {
        let mut state = State::default();
        let payload = state.attach(&dummy_payload());
        state.left.insert(
            (105, Encoding::from_slice(&[1]), Encoding::from_slice(&[1])),
            payload,
        );
        assert!(
            validate_progress(
                &state,
                10,
                false,
                &progress(Some(106), Some(106)),
                Some(EventTime::from_micros(90))
            )
            .is_err()
        );
    }

    #[test]
    fn asof_restore_rejects_expired_identity_only_and_unproven_output_frontier() {
        let mut state = State::default();
        state.right.insert(
            Encoding::from_slice(&[1]),
            RightBucket::from_iter([((100, Encoding::from_slice(&[1])), None)]),
        );
        assert!(
            validate_progress(&state, 10, false, &progress(Some(1000), Some(101)), None).is_err()
        );
        assert!(
            validate_progress(
                &State::default(),
                10,
                false,
                &progress(Some(100), None),
                Some(EventTime::from_micros(99))
            )
            .is_err()
        );
    }
}
