use super::{CONTROL_ID, Control, Descriptor, Frame, MAX_FRAMES, codec, mismatch};
use crate::{OperatorStateSnapshot, Result, StateSegment, StreamAsofJoinOperator};
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc};

impl StreamAsofJoinOperator {
    pub(in crate::operator::asof) fn capture_replay(
        &mut self,
    ) -> Result<Option<OperatorStateSnapshot>> {
        if !self.capture_replay_frame()? {
            return Ok(None);
        }
        self.status.state_bytes = self.current_inventory(None)?.bytes;
        let log = self
            .replay
            .as_ref()
            .expect("capture has an active replay log");
        let inputs = self
            .replay_inputs
            .as_ref()
            .expect("replay inputs are configured");
        let bytes = 8192
            + log.frames.len() * 2048
            + inputs.bindings.iter().map(|id| id.len() * 4).sum::<usize>();
        let Ok(credit) = self.reserve_workspace(bytes as u64) else {
            return Ok(None);
        };
        let control = Control {
            version: 1,
            fingerprint: self.fingerprint.clone(),
            bindings: inputs.bindings.clone(),
            record_capacity: log.records.capacity(),
            frames: log
                .frames
                .iter()
                .map(|frame| frame.descriptor.clone())
                .collect(),
            generation: log.generation,
            terminal: self.terminal,
            next_output_sequence: self.next_output_sequence,
            status: self.status.clone(),
        };
        if self.status.state_bytes > self.spec.limits().max_state_bytes() {
            return Ok(None);
        }
        let bytes = serde_json::to_vec(&control).map_err(|error| mismatch(&error.to_string()))?;
        let mut segments = log
            .frames
            .iter()
            .map(|frame| (frame.descriptor.id.clone(), frame.segment.clone()))
            .collect::<BTreeMap<_, _>>();
        segments.insert(
            CONTROL_ID.into(),
            StateSegment::new(bytes).with_owner(Arc::new(credit)),
        );
        Ok(Some(OperatorStateSnapshot {
            inline_metadata: BTreeMap::from([
                ("kind".into(), json!("stream_asof_join")),
                ("state_version".into(), json!(3)),
                ("source_replay".into(), json!(1)),
            ]),
            segments,
        }))
    }

    fn capture_replay_frame(&mut self) -> Result<bool> {
        let log = self
            .replay
            .as_ref()
            .expect("capture has an active replay log");
        if log.cut == log.records.len() {
            return Ok(true);
        }
        let compact = log.frames.len() >= MAX_FRAMES;
        let first = if compact { 0 } else { log.cut };
        let count = log.records.len() - first;
        let length = count
            .checked_mul(codec::WIDTH)
            .and_then(|bytes| bytes.checked_add(codec::HEADER))
            .ok_or_else(|| mismatch("capture frame size overflowed"))?;
        let Some(paid) = length
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(512))
        else {
            return Ok(false);
        };
        let Ok(credit) = self.reserve_workspace(paid as u64) else {
            return Ok(false);
        };
        let log = self
            .replay
            .as_mut()
            .expect("capture has an active replay log");
        let generation = log
            .generation
            .checked_add(1)
            .ok_or_else(|| mismatch("generation exhausted"))?;
        let id = format!("asof-replay-{generation:020}");
        let segment =
            StateSegment::new(codec::encode(&log.records[first..])).with_owner(Arc::new(credit));
        if compact {
            log.frames.clear();
        }
        log.frames.push(Frame {
            segment,
            descriptor: Descriptor {
                id,
                first: first as u64,
                count: count as u64,
            },
        });
        log.cut = log.records.len();
        log.generation = generation;
        Ok(true)
    }
}
