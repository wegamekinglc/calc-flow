mod cpu;
mod validation;

use super::{ColumnarState, StreamAsofJoinOperator, StreamAsofJoinStatus};
use crate::{CalcFlowError, Epoch, IngressProgressSnapshot, OperatorStateSnapshot, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use validation::{validate_counters, validate_progress};

pub(super) struct DecodedSnapshot {
    state: ColumnarState,
    metrics: StreamAsofJoinStatus,
    terminal: bool,
    sequence: u64,
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

    fn capture_metadata(&self, epoch: Epoch) -> Result<crate::JsonMap> {
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
            layout_version: 11,
            accounting_version: 11,
            row_encoding: "arrow-columnar-59.2.0",
            fingerprint: &self.fingerprint,
            epoch: epoch.as_u64(),
            terminal: self.terminal,
            next_output_sequence: self.next_output_sequence,
            metrics,
        };
        serde_json::to_value(metadata)
            .and_then(serde_json::from_value)
            .map_err(|error| mismatch(&error.to_string()))
    }

    pub(super) fn capture(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        if self.replay.is_some() {
            if let Some(snapshot) = self.capture_replay(epoch)? {
                return Ok(snapshot);
            }
            self.stop_replay()?;
        }
        let limit = usize::try_from(self.spec.limits().max_state_bytes())
            .expect("constructor validated ASOF byte limit");
        let segments = self.state.capture(limit)?;
        Ok(OperatorStateSnapshot {
            inline_metadata: self.capture_metadata(epoch)?,
            segments,
        })
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
        let metadata = self.restore_metadata(snapshot)?;
        let state = ColumnarState::restore(
            &self.name,
            &self.spec,
            &[self.schemas[0].clone(), self.schemas[1].clone()],
            &self.runtime_pool,
            &snapshot.segments,
        )?;
        cancel()?;
        self.validate_restored_state(&state, &metadata)?;
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
        })
    }

    fn validate_restored_state(
        &self,
        state: &ColumnarState,
        metadata: &Metadata<'_>,
    ) -> Result<()> {
        let metrics = &metadata.metrics;
        if state.pending_left_rows() as u64 != metrics.pending_left_rows
            || state.right_payload_rows() as u64 != metrics.retained_right_rows
            || state.identity_only_rows() as u64 != metrics.identity_only_rows
            || state.state_rows() as u64 != metrics.state_rows
            || state.persistent_bytes() as u64 != metrics.state_bytes
            || metrics.state_rows > self.spec.limits().max_state_rows()
            || metrics.state_bytes > self.spec.limits().max_state_bytes()
        {
            return Err(mismatch("ASOF restored state charge or gauges differ"));
        }
        Ok(())
    }

    fn restore_metadata<'a>(&self, snapshot: &'a OperatorStateSnapshot) -> Result<Metadata<'a>> {
        super::metadata::validate_shape(&snapshot.inline_metadata)?;
        let metadata = Metadata::deserialize(serde::de::value::MapDeserializer::new(
            snapshot
                .inline_metadata
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "source_replay" | crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1
                    )
                })
                .map(|(key, value)| (key.as_str(), value)),
        ))
        .map_err(|error: serde_json::Error| mismatch(&error.to_string()))?;
        self.validate_metadata(&metadata)?;
        Ok(metadata)
    }

    fn validate_metadata(&self, metadata: &Metadata<'_>) -> Result<()> {
        if metadata.kind != "stream_asof_join"
            || metadata.state_version != 3
            || (metadata.layout_version, metadata.accounting_version) != (11, 11)
            || metadata.row_encoding != "arrow-columnar-59.2.0"
            || metadata.fingerprint != self.fingerprint
        {
            return Err(mismatch(
                "ASOF kind, schema, configuration or state version differs",
            ));
        }
        Ok(())
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
        self.install_restored(decoded);
        self.observe(progress);
        self.status.output_watermark_micros = output_frontier;
        Ok(())
    }

    pub(super) fn install_restored(&mut self, decoded: DecodedSnapshot) {
        self.state = decoded.state;
        self.status = decoded.metrics;
        self.terminal = decoded.terminal;
        self.next_output_sequence = decoded.sequence;
        self.swept = None;
    }
}

pub(super) fn mismatch(message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("ASOF checkpoint: {message}"),
    }
}
