use super::{StreamAsofJoinOperator, StreamAsofJoinStatus, codec::BoundedWriter, state::State};
use crate::{
    CalcFlowError, Epoch, IngressProgressSnapshot, OperatorStateSnapshot, Result, StateSegment,
    StreamOperatorContext,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeMap, io::Write as _, sync::Arc};

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
        if state.left.is_empty() && state.right.is_empty() {
            return Ok(PreparedCheckpoint {
                segment: None,
                _workspace: workspace,
            });
        }
        let mut writer = BoundedWriter::with_capacity(
            usize::try_from(length).expect("reserved address domain"),
            limit,
        );
        write_bytes(&mut writer, MAGIC)?;
        write_u64(&mut writer, state.left.len() as u64)?;
        write_u64(&mut writer, state.right.len() as u64)?;
        for ((time, key, sequence), payload) in &state.left {
            context.check_cancelled()?;
            write_bytes(&mut writer, &time.to_le_bytes())?;
            write_blob(&mut writer, key)?;
            write_blob(&mut writer, sequence)?;
            write_blob(&mut writer, payload.bytes())?;
            tokio::task::yield_now().await;
        }
        for (key, bucket) in &state.right {
            write_blob(&mut writer, key)?;
            write_u64(&mut writer, bucket.len() as u64)?;
            for ((time, sequence), payload) in bucket {
                context.check_cancelled()?;
                write_bytes(&mut writer, &time.to_le_bytes())?;
                write_blob(&mut writer, sequence)?;
                write_blob(
                    &mut writer,
                    payload.as_ref().map_or(&[], StateSegment::bytes),
                )?;
                tokio::task::yield_now().await;
            }
        }
        let segment = StateSegment::new(writer.bytes);
        Ok(PreparedCheckpoint {
            segment: Some(segment),
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
        super::metadata::validate_shape(&snapshot.inline_metadata)?;
        let metadata = Metadata::deserialize(serde::de::value::MapDeserializer::new(
            snapshot
                .inline_metadata
                .iter()
                .map(|(key, value)| (key.as_str(), value)),
        ))
        .map_err(|error: serde_json::Error| mismatch(&error.to_string()))?;
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
        if snapshot.segments.len() > 1 || snapshot.segments.keys().any(|key| key != SEGMENT) {
            return Err(mismatch("unexpected ASOF segment inventory"));
        }
        let workspace = self.reserve_workspace(
            snapshot
                .segments
                .get(SEGMENT)
                .map(|segment| restore_charge(segment.bytes(), self.spec.limits().max_state_rows()))
                .transpose()?
                .unwrap_or(0),
        )?;
        let state = snapshot
            .segments
            .get(SEGMENT)
            .map(|segment| self.decode_state(segment))
            .transpose()?
            .unwrap_or_default();
        let inventory = state.inventory(snapshot.segments.get(SEGMENT), &self.name)?;
        let metrics = metadata.metrics;
        if inventory.identities != metrics.state_rows
            || inventory.bytes != metrics.state_bytes
            || inventory.right_payloads != metrics.retained_right_rows
            || inventory.identity_only != metrics.identity_only_rows
            || state.left.len() as u64 != metrics.pending_left_rows
        {
            return Err(mismatch("ASOF recomputed state charge or gauges differ"));
        }
        if inventory.identities > self.spec.limits().max_state_rows()
            || inventory.bytes > self.spec.limits().max_state_bytes()
        {
            return Err(mismatch("ASOF restored state exceeds current limits"));
        }
        validate_counters(&metrics, metadata.terminal, metadata.next_output_sequence)?;
        Ok(DecodedSnapshot {
            state,
            metrics,
            terminal: metadata.terminal,
            sequence: metadata.next_output_sequence,
            _workspace: workspace,
        })
    }

    fn decode_state(&self, segment: &StateSegment) -> Result<State> {
        if hex::encode(Sha256::digest(segment.bytes())) != segment.sha256() {
            return Err(mismatch("ASOF segment checksum mismatch"));
        }
        let mut decoder = Decoder {
            bytes: segment.bytes(),
            max_rows: self.spec.limits().max_state_rows(),
            rows: 0,
        };
        if decoder.take(8)? != MAGIC {
            return Err(mismatch("ASOF segment magic differs"));
        }
        let left_count = decoder.count()?;
        let bucket_count = decoder.count()?;
        let mut state = State::default();
        for _ in 0..left_count {
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
        }
        for _ in 0..bucket_count {
            self.decode_bucket(&mut decoder, &mut state)?;
        }
        if !decoder.bytes.is_empty() {
            return Err(mismatch("ASOF segment contains trailing data"));
        }
        Ok(state)
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
            decoder.row()?;
            let time = decoder.time()?;
            let sequence = Arc::new(decoder.blob()?.to_vec());
            super::identity::validate(
                &sequence,
                &self.schemas[1],
                self.spec.right().sequence_by(),
            )?;
            let bytes = decoder.blob()?;
            if !bytes.is_empty() {
                self.validate_payload(bytes, true, &(time, key.clone(), sequence.clone()))?;
            }
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
        }
        state.right.insert(key, bucket);
        Ok(())
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

fn validate_counters(metrics: &StreamAsofJoinStatus, terminal: bool, sequence: u64) -> Result<()> {
    if metrics.matched_rows.checked_add(metrics.unmatched_rows) != Some(metrics.emitted_left_rows)
        || metrics
            .emitted_left_rows
            .checked_add(metrics.pending_left_rows)
            != Some(metrics.left.accepted_rows)
        || metrics
            .evicted_right_rows
            .checked_add(metrics.retained_right_rows)
            != Some(metrics.right.accepted_rows)
        || metrics.identity_only_rows > metrics.evicted_right_rows
        || sequence != metrics.emitted_left_rows
        || (terminal && metrics.state_rows != 0)
    {
        return Err(mismatch(
            "ASOF output sequence, counters or terminal state contradict retained rows",
        ));
    }
    if [&metrics.left, &metrics.right]
        .into_iter()
        .any(|side| side.watermark_micros.is_some() || side.idle || side.ended)
        || metrics.output_watermark_micros.is_some()
    {
        return Err(mismatch("ASOF checkpoint must not own runtime progress"));
    }
    Ok(())
}

fn validate_progress(
    state: &State,
    tolerance: u64,
    terminal: bool,
    progress: &IngressProgressSnapshot,
    output_frontier: Option<crate::EventTime>,
) -> Result<()> {
    if progress.by_ingress().len() != 2
        || progress.get("left").is_none()
        || progress.get("right").is_none()
    {
        return Err(mismatch("ASOF restore requires exact two-ingress progress"));
    }
    if terminal != super::all_ended(progress) {
        return Err(mismatch("ASOF terminal state contradicts ingress EOF"));
    }
    if state
        .left
        .keys()
        .any(|(time, _, _)| output_frontier.is_some_and(|frontier| *time <= frontier.as_micros()))
    {
        return Err(mismatch("ASOF pending row is behind output frontier"));
    }
    if let (Some(output), Some(input)) = (output_frontier, super::frontier(progress))
        && output.as_micros() >= input
    {
        return Err(mismatch(
            "ASOF output frontier is ahead of safe input progress",
        ));
    }
    let input = super::frontier(progress);
    if state
        .left
        .keys()
        .any(|(time, _, _)| input.is_some_and(|frontier| *time < frontier))
    {
        return Err(mismatch(
            "ASOF snapshot contains already-finalizable pending left rows",
        ));
    }
    if !terminal && input.is_none() && output_frontier.is_some() {
        return Err(mismatch(
            "ASOF output frontier has no established input progress",
        ));
    }
    let left = progress
        .get("left")
        .expect("validated two-ingress progress");
    let future = if left.state() == crate::IngressState::Ended {
        i128::MAX
    } else {
        left.watermark()
            .map_or(i128::MIN, |wm| i128::from(wm.as_micros()))
    };
    let pending = state
        .left
        .first_key_value()
        .map_or(i128::MAX, |(key, _)| i128::from(key.0));
    let threshold = future.min(pending);
    if state.right.values().any(|bucket| {
        bucket.iter().any(|((time, _), row)| {
            row.is_none() && i128::from(*time) + i128::from(tolerance) >= threshold
        })
    }) {
        return Err(mismatch(
            "ASOF identity-only state discarded a potentially matching payload",
        ));
    }
    let right = progress
        .get("right")
        .expect("validated two-ingress progress");
    if state.right.values().any(|bucket| {
        bucket.iter().any(|((time, _), row)| {
            row.is_none()
                && (right.state() == crate::IngressState::Ended
                    || right.watermark().is_some_and(|wm| *time < wm.as_micros()))
        })
    }) {
        return Err(mismatch(
            "ASOF identity-only state outlived its closed ingress boundary",
        ));
    }
    Ok(())
}

struct Decoder<'a> {
    bytes: &'a [u8],
    max_rows: u64,
    rows: u64,
}
impl<'a> Decoder<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if count > self.bytes.len() {
            return Err(mismatch("ASOF truncated segment"));
        }
        let (value, rest) = self.bytes.split_at(count);
        self.bytes = rest;
        Ok(value)
    }
    fn integer(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn time(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn count(&mut self) -> Result<u64> {
        let count = self.integer()?;
        if count > self.max_rows {
            return Err(mismatch("ASOF declared row count exceeds limits"));
        }
        Ok(count)
    }
    fn row(&mut self) -> Result<()> {
        self.rows = self
            .rows
            .checked_add(1)
            .filter(|rows| *rows <= self.max_rows)
            .ok_or_else(|| mismatch("ASOF decoded row count exceeds limits"))?;
        Ok(())
    }
    fn blob(&mut self) -> Result<&'a [u8]> {
        let size = usize::try_from(self.integer()?)
            .map_err(|_| mismatch("ASOF field length exceeds address domain"))?;
        self.take(size)
    }
}
fn write_bytes(writer: &mut BoundedWriter, bytes: &[u8]) -> Result<()> {
    writer
        .write_all(bytes)
        .map_err(|error| mismatch(&error.to_string()))
}
fn write_u64(writer: &mut BoundedWriter, value: u64) -> Result<()> {
    write_bytes(writer, &value.to_le_bytes())
}
fn write_blob(writer: &mut BoundedWriter, value: &[u8]) -> Result<()> {
    write_u64(writer, value.len() as u64)?;
    write_bytes(writer, value)
}
fn mismatch(message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: message.into(),
    }
}

pub(super) fn encoded_length(state: &State, name: &str) -> Result<u64> {
    if state.left.is_empty() && state.right.is_empty() {
        return Ok(0);
    }
    let mut size = 24;
    for ((_, key, sequence), payload) in &state.left {
        size = super::checked(name, size, 32)?;
        for bytes in [key.as_slice(), sequence.as_slice(), payload.bytes()] {
            size = super::checked(name, size, bytes.len() as u64)?;
        }
    }
    for (key, bucket) in &state.right {
        size = super::checked(name, size, 16 + key.len() as u64)?;
        for ((_, sequence), payload) in bucket {
            size = super::checked(name, size, 24 + sequence.len() as u64)?;
            size = super::checked(
                name,
                size,
                payload.as_ref().map_or(0, |row| row.bytes().len() as u64),
            )?;
        }
    }
    Ok(size)
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

fn restore_charge(bytes: &[u8], max_rows: u64) -> Result<u64> {
    let mut decoder = Decoder {
        bytes,
        max_rows,
        rows: 0,
    };
    if decoder.take(8)? != MAGIC {
        return Err(mismatch("ASOF segment magic differs"));
    }
    let left = decoder.count()?;
    let buckets = decoder.count()?;
    let mut largest_payload = 0;
    let mut largest_identity = 0;
    for _ in 0..left {
        decoder.row()?;
        decoder.time()?;
        largest_identity = largest_identity.max(decoder.blob()?.len() as u64);
        largest_identity = largest_identity.max(decoder.blob()?.len() as u64);
        largest_payload = largest_payload.max(decoder.blob()?.len() as u64);
    }
    for _ in 0..buckets {
        largest_identity = largest_identity.max(decoder.blob()?.len() as u64);
        let count = decoder.count()?;
        for _ in 0..count {
            decoder.row()?;
            decoder.time()?;
            largest_identity = largest_identity.max(decoder.blob()?.len() as u64);
            largest_payload = largest_payload.max(decoder.blob()?.len() as u64);
        }
    }
    if !decoder.bytes.is_empty() {
        return Err(mismatch("ASOF segment contains trailing bytes"));
    }
    (bytes.len() as u64)
        .checked_add(
            largest_payload
                .checked_mul(4)
                .ok_or_else(|| mismatch("ASOF decode workspace overflowed"))?
                .max(largest_identity),
        )
        .and_then(|value| value.checked_add(decoder.rows.checked_mul(384)?))
        .ok_or_else(|| mismatch("ASOF decode workspace overflowed"))
}
