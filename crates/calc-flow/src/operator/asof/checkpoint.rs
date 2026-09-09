mod encoding;
mod validation;

use super::{
    StreamAsofJoinOperator, StreamAsofJoinStatus,
    state::{Encoding, Inventory, RightOrder, State},
};
use crate::{
    CalcFlowError, Epoch, IngressProgressSnapshot, OperatorStateSnapshot, Result, StateSegment,
    StreamOperatorContext,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeMap, sync::Arc};

pub(super) use encoding::encoded_length;
use encoding::{Decoder, encode_state, restore_charge};
use validation::{validate_counters, validate_progress};

const MAGIC: &[u8; 8] = b"CFASOF01";
const SEGMENT: &str = "asof-state-v1";

pub(super) struct PreparedCheckpoint {
    pub segment: Option<StateSegment>,
    pub _workspace: datafusion::execution::memory_pool::MemoryReservation,
}

pub(super) struct DecodedSnapshot {
    state: State,
    metrics: StreamAsofJoinStatus,
    terminal: bool,
    sequence: u64,
    _workspace: datafusion::execution::memory_pool::MemoryReservation,
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
    pub(super) async fn prepare_checkpoint(
        &self,
        state: &State,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedCheckpoint> {
        let limit = usize::try_from(self.spec.limits().max_state_bytes()).expect("validated");
        let length = encoded_length(state, &self.name)?;
        let workspace = self.reserve_workspace(length)?;
        let segment = if state.left.is_empty() && state.right.is_empty() {
            None
        } else {
            Some(encode_state(state, length, limit, context).await?)
        };
        Ok(PreparedCheckpoint {
            segment,
            _workspace: workspace,
        })
    }

    pub(super) fn capture(&self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        let mut metrics = self.status.clone();
        for side in [&mut metrics.left, &mut metrics.right] {
            side.watermark_micros = None;
            side.idle = false;
            side.ended = false;
        }
        metrics.output_watermark_micros = None;
        let metadata = Metadata {
            kind: "stream_asof_join",
            state_version: 1,
            layout_version: 1,
            accounting_version: 1,
            row_encoding: "arrow-row-58.3.0",
            fingerprint: &self.fingerprint,
            epoch: epoch.as_u64(),
            terminal: self.terminal,
            next_output_sequence: self.next_output_sequence,
            metrics,
        };
        let inline_metadata = serde_json::to_value(metadata)
            .and_then(serde_json::from_value)
            .map_err(|error| mismatch(&error.to_string()))?;
        let segments = self
            .prepared
            .as_ref()
            .map(|segment| BTreeMap::from([(SEGMENT.into(), segment.clone())]))
            .unwrap_or_default();
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
        let segment = snapshot_segment(snapshot)?;
        let workspace = self.reserve_workspace(
            segment
                .map(|segment| restore_charge(segment.bytes(), self.spec.limits().max_state_rows()))
                .transpose()?
                .unwrap_or(0),
        )?;
        let state = segment
            .map(|segment| self.decode_state(segment))
            .transpose()?
            .unwrap_or_default();
        self.validate_restored_state(&state, segment, &metadata.metrics)?;
        validate_counters(
            &metadata.metrics,
            metadata.terminal,
            metadata.next_output_sequence,
        )?;
        Ok(DecodedSnapshot {
            state,
            metrics: metadata.metrics,
            terminal: metadata.terminal,
            sequence: metadata.next_output_sequence,
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
        if metadata.kind != "stream_asof_join"
            || metadata.state_version != 1
            || metadata.layout_version != 1
            || metadata.accounting_version != 1
            || metadata.row_encoding != "arrow-row-58.3.0"
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
        segment: Option<&StateSegment>,
        metrics: &StreamAsofJoinStatus,
    ) -> Result<()> {
        let inventory = state.inventory(segment, &self.name)?;
        validate_gauges(&inventory, state.left.len() as u64, metrics)?;
        if inventory.identities > self.spec.limits().max_state_rows()
            || inventory.bytes > self.spec.limits().max_state_bytes()
        {
            return Err(mismatch("ASOF restored state exceeds current limits"));
        }
        Ok(())
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
        for _ in 0..bucket_count {
            self.decode_bucket(&mut decoder, &mut state)?;
        }
        decoder.finish("ASOF segment contains trailing data")?;
        Ok(state)
    }

    fn decode_left_row(&self, decoder: &mut Decoder<'_>, state: &mut State) -> Result<()> {
        decoder.row()?;
        let time = decoder.time()?;
        let key = Arc::new(decoder.blob()?.to_vec());
        let sequence = Arc::new(decoder.blob()?.to_vec());
        let bytes = decoder.blob()?;
        self.validate_payload(bytes, false, &(time, key.clone(), sequence.clone()))?;
        let identity = (time, key, sequence);
        if state
            .left
            .last_key_value()
            .is_some_and(|(last, _)| last >= &identity)
        {
            return Err(mismatch("ASOF left identity order is not strict"));
        }
        state
            .left
            .insert(identity, StateSegment::new(bytes.to_vec()));
        Ok(())
    }

    fn decode_bucket(&self, decoder: &mut Decoder<'_>, state: &mut State) -> Result<()> {
        let key = Arc::new(decoder.blob()?.to_vec());
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
        let mut bucket = BTreeMap::new();
        for _ in 0..count {
            self.decode_right_row(decoder, &key, &mut bucket)?;
        }
        state.right.insert(key, bucket);
        Ok(())
    }

    fn decode_right_row(
        &self,
        decoder: &mut Decoder<'_>,
        key: &Encoding,
        bucket: &mut BTreeMap<RightOrder, Option<StateSegment>>,
    ) -> Result<()> {
        decoder.row()?;
        let time = decoder.time()?;
        let sequence = self.decode_right_sequence(decoder)?;
        let bytes = decoder.blob()?;
        self.validate_right_payload(bytes, time, key, &sequence)?;
        let identity = (time, sequence);
        if bucket
            .last_key_value()
            .is_some_and(|(last, _)| last >= &identity)
        {
            return Err(mismatch("ASOF right identity order is not strict"));
        }
        bucket.insert(
            identity,
            (!bytes.is_empty()).then(|| StateSegment::new(bytes.to_vec())),
        );
        Ok(())
    }

    fn validate_right_payload(
        &self,
        bytes: &[u8],
        time: i64,
        key: &Encoding,
        sequence: &Encoding,
    ) -> Result<()> {
        if !bytes.is_empty() {
            self.validate_payload(bytes, true, &(time, key.clone(), sequence.clone()))?;
        }
        Ok(())
    }

    fn decode_right_sequence(&self, decoder: &mut Decoder<'_>) -> Result<Encoding> {
        let sequence = Arc::new(decoder.blob()?.to_vec());
        super::identity::validate(&sequence, &self.schemas[1], self.spec.right().sequence_by())?;
        Ok(sequence)
    }

    fn validate_payload(
        &self,
        bytes: &[u8],
        right: bool,
        identity: &super::state::LeftOrder,
    ) -> Result<()> {
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
        snapshot: &OperatorStateSnapshot,
        decoded: DecodedSnapshot,
    ) {
        self.state = decoded.state;
        self.status = decoded.metrics;
        self.terminal = decoded.terminal;
        self.next_output_sequence = decoded.sequence;
        self.prepared = snapshot.segments.get(SEGMENT).cloned();
    }
}

fn snapshot_segment(snapshot: &OperatorStateSnapshot) -> Result<Option<&StateSegment>> {
    if snapshot.segments.len() > 1 || snapshot.segments.keys().any(|key| key != SEGMENT) {
        return Err(mismatch("unexpected ASOF segment inventory"));
    }
    Ok(snapshot.segments.get(SEGMENT))
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
    for column in side
        .keys()
        .iter()
        .chain(side.sequence_by())
        .map(String::as_str)
        .chain(std::iter::once(side.event_time()))
    {
        if row
            .column(row.schema().index_of(column).expect("validated schema"))
            .null_count()
            != 0
        {
            return Err(mismatch("ASOF restored identity contains null"));
        }
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
    use crate::{EventTime, IngressProgress, IngressState};

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
        state.left.insert(
            (105, Arc::new(vec![1]), Arc::new(vec![1])),
            StateSegment::new(vec![]),
        );
        state.right.insert(
            Arc::new(vec![1]),
            BTreeMap::from([((100, Arc::new(vec![1])), None)]),
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
        state.left.insert(
            (105, Arc::new(vec![1]), Arc::new(vec![1])),
            StateSegment::new(vec![]),
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
            Arc::new(vec![1]),
            BTreeMap::from([((100, Arc::new(vec![1])), None)]),
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
