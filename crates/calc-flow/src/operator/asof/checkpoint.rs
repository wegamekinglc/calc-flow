mod cpu;
mod incremental;
#[cfg(test)]
const INDEX_SEGMENT: &str = "asof-log-v9-1-0-1";
pub(super) mod index_v3;
mod payload_segments;
mod prepared;
pub(super) use incremental::LogState;
mod validation;

#[cfg(test)]
pub(super) mod cost;

#[cfg(test)]
mod current_format_tests;

#[cfg(test)]
use super::state::{RightBucket, RowPayload};
use super::{
    StreamAsofJoinOperator, StreamAsofJoinStatus,
    state::{BatchKey, Encoding, Inventory, PayloadBatch, PayloadView, State},
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

pub(super) use index_v3::BASE_BYTES as INDEX_HEADER_BYTES;
pub(super) use index_v3::encoded_length as v3_encoded_length;
pub(super) use prepared::PreparedSegment;
use validation::{validate_counters, validate_progress};

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
    checkpoint_log: LogState,
    _workspaces: Vec<MemoryReservation>,
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
    pub(super) fn sequence_kinds(&self) -> [super::state::SequenceKind; 2] {
        [
            super::state::SequenceKind::for_side(&self.schemas[0], self.spec.left()),
            super::state::SequenceKind::for_side(&self.schemas[1], self.spec.right()),
        ]
    }

    pub(super) fn current_inventory(
        &self,
        prepared: Option<&PreparedSegment>,
    ) -> Result<Inventory> {
        let mut inventory = self.state.capacity_inventory(prepared, &self.name)?;
        inventory.bytes = super::checked(&self.name, inventory.bytes, self.checkpoint_log.bytes())?;
        inventory.bytes = super::checked(&self.name, inventory.bytes, self.replay_bytes())?;
        Ok(inventory)
    }

    fn validate_restored_limits(
        &self,
        inventory: &Inventory,
        metadata: &Metadata<'_>,
    ) -> Result<()> {
        if inventory.identities > self.spec.limits().max_state_rows()
            || inventory.bytes > self.spec.limits().max_state_bytes()
        {
            return Err(mismatch("ASOF restored v3 state exceeds current limits"));
        }
        validate_counters(
            &metadata.metrics,
            metadata.terminal,
            metadata.next_output_sequence,
        )?;
        Ok(())
    }

    pub(super) fn ensure_prepared_sync(&mut self) -> Result<()> {
        self.ensure_payloads_sync()?;
        self.prepare_row_log_sync()
    }

    pub(super) async fn ensure_prepared_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()?;
        self.prepare_payloads_async(context).await?;
        self.prepare_row_log_async(context).await?;
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
        let mut retained = 0_u64;
        for (batch, _) in self.state.batches.values() {
            if !batch.has_encoded() {
                count = super::checked(&self.name, count, 1)?;
                retained = super::checked(
                    &self.name,
                    retained,
                    super::state::capacity_batch_allocation(batch, &self.name)?,
                )?;
                largest = largest.max(batch.encoded_charge_bytes);
            }
        }
        // A detached worker can own every Arrow record and every newly cached
        // IPC segment. The largest batch also funds the encoder's temporary
        // body while its final segment is being assembled.
        Ok((count, super::checked(&self.name, retained, largest)?))
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
        self.run_cpu_work(
            cpu::PayloadWork {
                pending,
                workspace,
                limit,
                name,
            },
            context,
        )
        .await?;
        context.check_cancelled()
    }

    #[cfg(test)]
    pub(super) async fn prepare_checkpoint(
        &self,
        state: &State,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedCheckpoint> {
        let length = v3_encoded_length(state, &self.name)?;
        let limit = usize::try_from(self.spec.limits().max_state_bytes()).expect("validated");
        let workspace =
            self.reserve_workspace(index_v3::workspace_bytes(state, length, true, &self.name)?)?;
        let (segment, workspace) = if length == 0 {
            (None, workspace)
        } else {
            let (segment, workspace) =
                index_v3::encode(state, length, limit, context, workspace).await?;
            (Some(PreparedSegment::new(segment)), workspace)
        };
        Ok(PreparedCheckpoint {
            segment,
            _workspace: workspace,
        })
    }

    fn capture_metadata(&self, epoch: Epoch, layout: u32) -> Result<crate::JsonMap> {
        let mut metrics = self.status.clone();
        for side in [&mut metrics.left, &mut metrics.right] {
            side.watermark_micros = None;
            side.idle = false;
            side.ended = false;
        }
        metrics.output_watermark_micros = None;
        let metadata = Metadata {
            kind: "stream_asof_join",
            state_version: 3,
            layout_version: layout,
            accounting_version: layout,
            row_encoding: "arrow-batch-58.3.0",
            fingerprint: &self.fingerprint,
            epoch: epoch.as_u64(),
            terminal: self.terminal,
            next_output_sequence: self.next_output_sequence,
            metrics,
        };
        let mut inline_metadata: crate::JsonMap = serde_json::to_value(metadata)
            .and_then(serde_json::from_value)
            .map_err(|error| mismatch(&error.to_string()))?;
        if layout == 9 {
            inline_metadata.insert(
                "retained_payloads".into(),
                serde_json::to_value(self.retained_descriptor())
                    .map_err(|error| mismatch(&error.to_string()))?,
            );
        }
        Ok(inline_metadata)
    }

    pub(super) fn capture(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        if self.replay.is_some() {
            if let Some(snapshot) = self.capture_replay(epoch)? {
                return Ok(snapshot);
            }
            self.stop_replay()?;
        }
        self.ensure_prepared_sync()?;
        self.capture_row_log(epoch)
    }

    pub(super) fn decoded_snapshot(
        &self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<DecodedSnapshot> {
        self.decoded_snapshot_checked(snapshot, &|| Ok(()))
    }

    fn decoded_snapshot_checked(
        &self,
        snapshot: &OperatorStateSnapshot,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<DecodedSnapshot> {
        cancel()?;
        self.decode_row_log(snapshot, &self.restore_metadata(snapshot)?, cancel)
    }

    fn restore_metadata<'a>(&self, snapshot: &'a OperatorStateSnapshot) -> Result<Metadata<'a>> {
        super::metadata::validate_shape(&snapshot.inline_metadata)?;
        let metadata = Metadata::deserialize(serde::de::value::MapDeserializer::new(
            snapshot
                .inline_metadata
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "retained_payloads" | "checkpoint_log"))
                .map(|(key, value)| (key.as_str(), value)),
        ))
        .map_err(|error: serde_json::Error| mismatch(&error.to_string()))?;
        self.validate_metadata(&metadata)?;
        if !snapshot.segments.is_empty() || self.payload_projection.is_some() {
            self.validate_retained_descriptor(snapshot.inline_metadata.get("retained_payloads"))?;
        }
        Ok(metadata)
    }

    fn validate_metadata(&self, metadata: &Metadata<'_>) -> Result<()> {
        if metadata.kind != "stream_asof_join"
            || metadata.state_version != 3
            || !matches!(
                (metadata.layout_version, metadata.accounting_version),
                (9, 9)
            )
            || metadata.row_encoding != "arrow-batch-58.3.0"
            || metadata.fingerprint != self.fingerprint
        {
            return Err(mismatch(
                "ASOF kind, schema, configuration or state version differs",
            ));
        }
        Ok(())
    }

    fn decode_payload_batch(
        &self,
        key: BatchKey,
        encoded: &StateSegment,
        physical: bool,
    ) -> Result<Arc<PayloadBatch>> {
        verify_checksum(encoded)?;
        let side = usize::from(key.0);
        let schema = if physical {
            self.physical_schema(side)
        } else {
            &self.schemas[side]
        };
        let digest = if physical {
            self.physical_digest(side)
        } else {
            &self.schema_digests[side]
        };
        let record = super::codec::decode_table_batch(
            encoded.bytes(),
            digest,
            schema,
            self.spec.limits().max_state_rows().min(u64::from(u32::MAX)),
        )
        .map_err(|_| mismatch("ASOF invalid Arrow batch encoding"))?;
        if record.schema().as_ref() != schema.as_ref()
            || record.num_rows() as u64 > self.spec.limits().max_state_rows()
        {
            return Err(mismatch("ASOF batch schema or row count differs"));
        }
        let record = record
            .with_schema(schema.clone())
            .map_err(|_| mismatch("ASOF batch schema differs"))?;
        let (encoded_charge_bytes, body_bytes) = super::workspace::payload_bound_with_header(
            &record,
            if physical {
                self.physical_header(side)
            } else {
                self.payload_header_bytes[side]
            },
            &self.name,
        )
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
        let mut seen = std::collections::BTreeSet::new();
        for (key, (batch, _)) in state.batches.iter() {
            let side = if key.0 == 0 {
                self.spec.left()
            } else {
                self.spec.right()
            };
            validate_nonnull_identity(&batch.record, side)?;
        }
        for (identity, payload) in state.left.unordered_iter() {
            validate_encoding(
                identity.1,
                &self.schemas[0],
                self.spec.left().keys(),
                0,
                &mut seen,
            )?;
            validate_encoding(
                identity.2.as_ref(),
                &self.schemas[0],
                self.spec.left().sequence_by(),
                1,
                &mut seen,
            )?;
            validate_index_identity(&identity, state.batches.view(payload), self.spec.left())?;
        }
        self.validate_right_indexed_rows(state, &mut seen)?;
        Ok(())
    }

    fn validate_right_indexed_rows(
        &self,
        state: &State,
        seen: &mut std::collections::BTreeSet<(usize, u8)>,
    ) -> Result<()> {
        for (key, bucket) in &state.right {
            validate_encoding(key, &self.schemas[1], self.spec.right().keys(), 2, seen)?;
            for ((time, sequence), payload) in bucket {
                validate_encoding(
                    sequence.as_ref(),
                    &self.schemas[1],
                    self.spec.right().sequence_by(),
                    3,
                    seen,
                )?;
                if let Some(payload) = payload {
                    validate_index_identity(
                        &(time, key, sequence),
                        state.batches.view(*payload),
                        self.spec.right(),
                    )?;
                }
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
        self.restore_with_progress_checked(snapshot, progress, output_frontier, &|| Ok(()))
    }

    fn restore_with_progress_checked(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        progress: &IngressProgressSnapshot,
        output_frontier: Option<crate::EventTime>,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let decoded = self.decoded_snapshot_checked(snapshot, cancel)?;
        validate_progress(
            &decoded.state,
            self.spec.tolerance_micros(),
            decoded.terminal,
            progress,
            output_frontier,
        )?;
        cancel()?;
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
        self.status = decoded.metrics;
        self.terminal = decoded.terminal;
        self.next_output_sequence = decoded.sequence;
        self.prepared = decoded.prepared;
        self.checkpoint_log = decoded.checkpoint_log;
        self.deferred_index_len = Some(
            index_v3::encoded_length(&self.state, &self.name)
                .expect("validated restored index length"),
        );
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

fn verify_checksum(segment: &StateSegment) -> Result<()> {
    if hex::encode(Sha256::digest(segment.bytes())) != segment.sha256() {
        return Err(mismatch("ASOF segment checksum mismatch"));
    }
    Ok(())
}

fn validate_encoding(
    encoding: &Encoding,
    schema: &datafusion::arrow::datatypes::Schema,
    columns: &[String],
    role: u8,
    seen: &mut std::collections::BTreeSet<(usize, u8)>,
) -> Result<()> {
    if let Some((address, _)) = encoding.allocation() {
        if !seen.insert((address, role)) {
            return Ok(());
        }
    }
    // A live handle can keep expired rows in its shared batch buffer. Validate
    // the entire allocation against every schema role that references it.
    for value in encoding.owner_values() {
        super::identity::validate(value, schema, columns)?;
    }
    Ok(())
}

fn validate_index_identity(
    identity: &super::state::LeftView<'_>,
    payload: PayloadView<'_>,
    side: &super::AsofJoinSide,
) -> Result<()> {
    let record = &payload.batch.record;
    if super::admission::times(record, side).value(payload.row) != *identity.0
        || !super::identity_compare::encoded_equal(
            record,
            payload.row,
            side.keys(),
            identity.1.as_slice(),
        )
        || !super::identity_compare::encoded_equal(
            record,
            payload.row,
            side.sequence_by(),
            identity.2.as_slice(),
        )
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
        StreamOperator, StreamOperatorContext,
    };
    use datafusion::arrow::{
        array::{Int8Array, Int64Array, StringArray, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };
    use std::time::Duration;

    #[test]
    fn v3_index_orders_right_buckets_independently_of_hash_insertion() {
        let mut forward = State::default();
        let mut reverse = State::default();
        for (state, keys) in [(&mut forward, [1_u8, 2]), (&mut reverse, [2, 1])] {
            for key in keys {
                let mut bucket = RightBucket::default();
                bucket.insert((1, Encoding::from_slice(&[1])), None);
                state.right.insert(Encoding::from_slice(&[key]), bucket);
            }
        }
        let length = v3_encoded_length(&forward, "asof").unwrap();
        assert_eq!(length, v3_encoded_length(&reverse, "asof").unwrap());
        let forward = index_v3::encode_sync(&forward, length, 1_024).unwrap();
        let reverse = index_v3::encode_sync(&reverse, length, 1_024).unwrap();
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

    #[tokio::test]
    async fn v3_restore_rejects_batch_id_at_next_admission_boundary() {
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
    async fn narrow_column_v3_checkpoint_restores_with_same_state_limit() {
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

    #[tokio::test]
    async fn current_identity_only_capture_retains_paid_wire_and_owners() {
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
            AsofStateLimits::new(4_096, 2 * 1024 * 1024).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
        let kind = operator.state.sequence_kinds[1];
        let mut bucket = RightBucket::with_sequence_kind(kind);
        for time in 0..4_096 {
            bucket.insert((time, kind.decode_integer(&time.to_le_bytes())), None);
        }
        operator
            .state
            .right
            .insert(Encoding::from_slice(&[1]), bucket);
        operator.state.rebuild_encoding_owners();
        let length = v3_encoded_length(&operator.state, "asof").unwrap();
        let expected = index_v3::encode_sync(&operator.state, length, usize::MAX).unwrap();
        operator.deferred_index_len = Some(length);
        operator.runtime.pool = Arc::new(
            datafusion::execution::memory_pool::GreedyMemoryPool::new(2 * 1024 * 1024),
        );
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        operator.prepare_row_log_async(&context).await.unwrap();
        let pending = operator.checkpoint_log.pending.as_ref().unwrap();
        assert_eq!(pending.segment.bytes(), expected.bytes());
        let pool = operator.runtime.pool.clone();
        assert!(pool.reserved() >= pending.segment.bytes().len() + 256);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        drop(operator);
        assert_eq!(pool.reserved(), 0);
    }

    #[tokio::test]
    async fn pending_payload_lease_funds_all_detached_batches() {
        let mut operator = operator_with_pending_payloads().await;
        let (pending, workspace) = operator.pending_payloads().unwrap();
        let retained = pending
            .iter()
            .map(|batch| super::super::state::capacity_batch_allocation(batch, "asof").unwrap())
            .sum::<u64>();
        assert!(
            workspace.size() as u64 >= retained,
            "detached payloads need their complete owner fee: reserved={}, retained={retained}",
            workspace.size()
        );
        let weak = pending.iter().map(Arc::downgrade).collect::<Vec<_>>();
        let pool = operator.runtime.pool.clone();
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, blocked) = std::sync::mpsc::channel();
        let worker = tokio::task::spawn_blocking(move || {
            started.send(()).unwrap();
            blocked.recv().unwrap();
            encode_payloads(pending, workspace, 8 << 20, "asof").unwrap();
        });
        ready.await.unwrap();
        drop(worker);
        operator.reset().unwrap();
        assert!(weak.iter().all(|batch| batch.upgrade().is_some()));
        assert!(pool.reserved() as u64 >= retained);
        tokio::time::timeout(
            Duration::from_millis(100),
            tokio::time::sleep(Duration::from_millis(1)),
        )
        .await
        .unwrap();
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while pool.reserved() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(weak.iter().all(|batch| batch.upgrade().is_none()));
    }

    async fn operator_with_pending_payloads() -> StreamAsofJoinOperator {
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
        let spec = super::super::StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(100, 8 << 20).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        for index in 0..4 {
            let record = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec!["A"])),
                    Arc::new(TimestampMicrosecondArray::from(vec![index]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![index])),
                    Arc::new(StringArray::from(vec!["v".repeat(256 << 10)])),
                ],
            )
            .unwrap();
            operator
                .process_data(
                    "right",
                    Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut output,
                )
                .await
                .unwrap();
        }
        operator
    }

    #[tokio::test]
    async fn v3_capture_and_restore_do_not_reserve_duplicate_payload_bodies() {
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
        let spec = super::super::StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(64, 2 * 1024 * 1024).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec.clone())
                .unwrap();
        let record = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["A"; 64])),
                Arc::new(
                    TimestampMicrosecondArray::from(
                        (0..64).map(|row| 100 + row).collect::<Vec<_>>(),
                    )
                    .with_timezone("UTC"),
                ),
                Arc::new(Int64Array::from((0..64).collect::<Vec<_>>())),
                Arc::new(StringArray::from(vec!["v".repeat(2_048); 64])),
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
        let expected = operator.capture(Epoch::INITIAL).unwrap();
        // Simulate another operation using the pool. Encoding an index need
        // not reserve the cached Arrow payload again.
        operator.runtime.pool = Arc::new(
            datafusion::execution::memory_pool::GreedyMemoryPool::new(64 * 1024),
        );
        operator.prepare_checkpoint_async(&context).await.unwrap();
        let actual = operator.capture(Epoch::INITIAL).unwrap();
        assert_eq!(actual.segments, expected.segments);
        assert!(operator.runtime.pool.reserved() > 0);
        let mut restored =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
        restored.runtime.pool =
            Arc::new(datafusion::execution::memory_pool::GreedyMemoryPool::new(
                usize::try_from(operator.status.state_bytes).unwrap() + 65_536,
            ));
        restored.restore(&actual).unwrap();
        assert_eq!(restored.status.state_bytes, operator.status.state_bytes);
        let restored_pool = restored.runtime.pool.clone();
        let source_pool = operator.runtime.pool.clone();
        drop(restored);
        assert_eq!(restored_pool.reserved(), 0);
        drop(operator);
        drop(actual);
        assert_eq!(source_pool.reserved(), 0);
    }

    #[tokio::test]
    async fn v3_restore_rejects_noncanonical_expired_sequence_in_a_shared_buffer() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Utf8, false),
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
            AsofStateLimits::new(8, 1 << 20).unwrap(),
        )
        .unwrap();
        let record = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["A", "A"])),
                Arc::new(TimestampMicrosecondArray::from(vec![100, 101]).with_timezone("UTC")),
                Arc::new(StringArray::from(vec![
                    "first string sequence",
                    "second string sequence",
                ])),
            ],
        )
        .unwrap();
        let mut operator =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "right",
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        let progressed = StreamOperatorContext::with_ingress_progress(
            &job,
            "asof",
            None,
            progress(Some(101), Some(101)),
        );
        operator
            .on_watermark(EventTime::from_micros(101), &progressed, &mut output)
            .await
            .unwrap();
        assert_eq!(operator.status.state_rows, 1);
        let mut snapshot = operator.capture(Epoch::INITIAL).unwrap();
        let mut bytes = snapshot.segments[index_v3::INDEX_SEGMENT].bytes().to_vec();
        assert_eq!(u64::from_le_bytes(bytes[224..232].try_into().unwrap()), 1);
        assert_eq!(bytes[232], 1, "one Binary owner retains both sequence rows");
        assert_eq!(u64::from_le_bytes(bytes[234..242].try_into().unwrap()), 2);
        bytes[266] = 0;
        let segment = StateSegment::new(bytes);
        snapshot.inline_metadata.get_mut("checkpoint_log").unwrap()["frames"][0]["sha256"] =
            serde_json::json!(segment.sha256());
        snapshot
            .segments
            .insert(index_v3::INDEX_SEGMENT.into(), segment);
        let before = operator.status();
        let reserved = operator.runtime.pool.reserved();
        assert!(matches!(
            operator.restore(&snapshot),
            Err(CalcFlowError::CheckpointMismatch { .. })
        ));
        assert_eq!(operator.status(), before);
        assert_eq!(operator.runtime.pool.reserved(), reserved);
    }

    #[test]
    fn capacity_payload_charge_bounds_allocations_for_wide_flat_columns() {
        use datafusion::common::ScalarValue;
        let value_types = [
            DataType::Int8,
            DataType::Int64,
            DataType::Boolean,
            DataType::Float64,
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary,
            DataType::Decimal128(30, 2),
        ];
        for (width, data_type) in [1, 4, 32]
            .into_iter()
            .flat_map(|width| value_types.iter().map(move |data_type| (width, data_type)))
        {
            let mut fields = vec![
                Field::new("key", DataType::Utf8, false),
                Field::new(
                    "time",
                    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                    false,
                ),
                Field::new("seq", DataType::Int64, false),
            ];
            fields.extend(
                (0..width)
                    .map(|index| Field::new(format!("value{index}"), data_type.clone(), true)),
            );
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
                AsofStateLimits::new(64, 8 * 1024 * 1024).unwrap(),
            )
            .unwrap();
            let operator =
                StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
            let mut columns: Vec<datafusion::arrow::array::ArrayRef> = vec![
                Arc::new(StringArray::from(vec!["A"; 64])),
                Arc::new(TimestampMicrosecondArray::from(vec![100; 64]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![1; 64])),
            ];
            columns.extend((0..width).map(|_| {
                ScalarValue::new_default(data_type)
                    .unwrap()
                    .to_array_of_size(64)
                    .unwrap()
            }));
            let record = RecordBatch::try_new(schema, columns).unwrap();
            let source =
                super::super::codec::encode_batch(&record, usize::MAX, &mut Vec::new()).unwrap();
            let mut retained = None;
            let allocations = allocation_counter::measure(|| {
                let encoded = StateSegment::new(source.clone());
                retained = Some(
                    operator
                        .decode_payload_batch((0, 0), &encoded, false)
                        .unwrap(),
                );
            });
            let batch = retained.unwrap();
            let fee = super::super::state::capacity_batch_allocation(&batch, "asof").unwrap();
            assert!(
                u64::try_from(allocations.bytes_current).unwrap() <= fee,
                "{data_type:?} width={width} allocation={allocations:?} fee={fee}"
            );
        }
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
    fn dominated_payload_restore_accepts_a_retained_same_key_witness() {
        let mut state = State::default();
        let payload = state.attach(&dummy_payload());
        state.right.insert(
            Encoding::from_slice(&[1]),
            RightBucket::from_iter([
                ((90, Encoding::from_slice(&[1])), None),
                ((100, Encoding::from_slice(&[2])), Some(payload)),
            ]),
        );
        assert!(
            validate_progress(&state, 1_000, false, &progress(Some(100), Some(90)), None).is_ok()
        );
    }

    #[test]
    fn dominated_payload_restore_requires_a_real_in_threshold_typed_witness() {
        for (identity_time, identity_sequence, witness_time, witness_sequence, witness_key) in
            [(90, 1, 101, 2, 1), (90, 1, 100, 2, 2), (100, 2, 100, 1, 1)]
        {
            let mut state = State::default();
            let payload = state.attach(&dummy_payload());
            state.right.insert(
                Encoding::from_slice(&[1]),
                RightBucket::from_iter([(
                    (identity_time, Encoding::from_slice(&[identity_sequence])),
                    None,
                )]),
            );
            state
                .right
                .bucket_mut_or_default(Encoding::from_slice(&[witness_key]))
                .insert(
                    (witness_time, Encoding::from_slice(&[witness_sequence])),
                    Some(payload),
                );
            assert!(
                validate_progress(&state, 1_000, false, &progress(Some(100), Some(90)), None)
                    .is_err()
            );
        }
        let mut state = State::default();
        state.right.insert(
            Encoding::from_slice(&[1]),
            RightBucket::from_iter([
                ((90, Encoding::from_slice(&[1])), None),
                ((100, Encoding::from_slice(&[2])), None),
            ]),
        );
        assert!(
            validate_progress(&state, 1_000, false, &progress(Some(100), Some(90)), None).is_err()
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
