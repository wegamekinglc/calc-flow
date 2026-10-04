mod capture;
mod codec;
mod restore;
#[cfg(test)]
mod tests;

use super::StreamAsofJoinOperator;
use crate::{
    EventTime, IngressProgress, IngressProgressSnapshot, Result, SourceHistoryContext,
    SourceHistoryReplayFactory, StateSegment, StreamAsofJoinStatus, StreamCollector,
    StreamOperatorContext,
};
use datafusion::execution::memory_pool::MemoryReservation;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};

const MAX_FRAMES: usize = 32;
const SIDES: [&str; 2] = ["left", "right"];
const CONTROL_ID: &str = "asof-replay-control";

pub(super) struct Inputs {
    bindings: [String; 2],
    histories: [SourceHistoryContext; 2],
    factories: [Arc<dyn SourceHistoryReplayFactory>; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Callback {
    Data { side: u8, sequence: u64 },
    Progress,
    End,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Record {
    callback: Callback,
    input_watermark: Option<EventTime>,
    progress: [IngressProgress; 2],
    max_rows: usize,
    max_bytes: usize,
}

struct Frame {
    segment: StateSegment,
    descriptor: Descriptor,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    id: String,
    first: u64,
    count: u64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Control {
    version: u32,
    fingerprint: String,
    bindings: [String; 2],
    record_capacity: usize,
    frames: Vec<Descriptor>,
    generation: u64,
    terminal: bool,
    next_output_sequence: u64,
    status: StreamAsofJoinStatus,
}

pub(super) struct Log {
    records: Vec<Record>,
    credit: MemoryReservation,
    frames: Vec<Frame>,
    cut: usize,
    generation: u64,
}

impl Log {
    fn bytes(&self) -> u64 {
        self.credit.size() as u64
            + self
                .frames
                .iter()
                .map(|frame| {
                    frame.segment.bytes_arc().capacity() as u64
                        + frame.descriptor.id.len() as u64
                        + 256
                })
                .sum::<u64>()
    }
}

impl StreamAsofJoinOperator {
    pub(crate) fn configure_source_replay(
        &mut self,
        bindings: [String; 2],
        histories: [SourceHistoryContext; 2],
        factories: [Arc<dyn SourceHistoryReplayFactory>; 2],
    ) {
        self.replay_inputs = Some(Box::new(Inputs {
            bindings,
            histories,
            factories,
        }));
    }

    pub(super) fn reset_replay(&mut self) {
        self.replay = None;
        if self.replay_inputs.is_some() {
            self.replay = self.new_replay_log().ok().map(Box::new);
            self.status.state_bytes = self
                .current_inventory(None)
                .map_or(0, |inventory| inventory.bytes);
        }
    }

    pub(super) fn new_replay_log(&self) -> Result<Log> {
        let bytes = 1024 + ((MAX_FRAMES + 1) * size_of::<Frame>()) as u64;
        let credit = self.reserve_workspace(bytes)?;
        Ok(Log {
            records: Vec::new(),
            credit,
            frames: Vec::with_capacity(MAX_FRAMES + 1),
            cut: 0,
            generation: 0,
        })
    }

    pub(super) fn replay_bytes(&self) -> u64 {
        self.replay.as_ref().map_or(0, |log| log.bytes())
    }

    pub(super) fn stop_replay(&mut self) -> Result<()> {
        self.replay = None;
        self.status.state_bytes = self.current_inventory(self.prepared.as_ref())?.bytes;
        Ok(())
    }

    pub(super) async fn finalize_with_replay(
        &mut self,
        frontier: Option<i64>,
        ended: bool,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        match self.finalize(frontier, ended, context, output).await {
            Err(error) if self.replay.is_some() && is_capacity_error(&error) => {
                self.stop_replay()?;
                self.finalize(frontier, ended, context, output).await
            }
            result => result,
        }
    }

    pub(super) fn record_replay(
        &mut self,
        callback: Callback,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if self.replay.is_none() {
            return Ok(());
        }
        let progress = SIDES.map(|side| context.ingress_progress().get(side));
        let [Some(left), Some(right)] = progress else {
            return self.stop_replay();
        };
        if context.ingress_progress().len() != 2 {
            return self.stop_replay();
        }
        let budget = context.output_budget();
        let record = Record {
            callback,
            progress: [left, right],
            input_watermark: context.input_watermark(),
            max_rows: budget.max_rows,
            max_bytes: budget.max_bytes,
        };
        if !self.reserve_replay_record()? {
            return self.stop_replay();
        }
        self.replay
            .as_mut()
            .expect("record credit was admitted")
            .records
            .push(record);
        self.status.state_bytes = self.current_inventory(self.prepared.as_ref())?.bytes;
        Ok(())
    }

    fn reserve_replay_record(&mut self) -> Result<bool> {
        let log = self.replay.as_ref().expect("replay is enabled");
        if log.records.len() < log.records.capacity() {
            return Ok(true);
        }
        let capacity = log
            .records
            .capacity()
            .max(8)
            .checked_mul(2)
            .ok_or_else(|| mismatch("replay capacity overflowed"))?;
        let delta = (capacity - log.records.capacity())
            .checked_mul(size_of::<Record>())
            .ok_or_else(|| mismatch("replay byte count overflowed"))?;
        if self
            .current_inventory(None)?
            .bytes
            .checked_add(delta as u64)
            .is_none_or(|bytes| bytes > self.spec.limits().max_state_bytes())
        {
            return Ok(false);
        }
        let log = self.replay.as_mut().expect("replay is enabled");
        if log.credit.try_grow(delta).is_err() {
            return Ok(false);
        }
        if log
            .records
            .try_reserve_exact(capacity - log.records.len())
            .is_err()
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn replay_progress(record: &Record) -> IngressProgressSnapshot {
        IngressProgressSnapshot::new(BTreeMap::from([
            ("left".into(), record.progress[0]),
            ("right".into(), record.progress[1]),
        ]))
    }
}

pub(super) fn mismatch(message: &str) -> crate::CalcFlowError {
    crate::CalcFlowError::CheckpointMismatch {
        message: format!("ASOF source replay: {message}"),
    }
}

pub(super) fn is_capacity_error(error: &crate::CalcFlowError) -> bool {
    matches!(
        error,
        crate::CalcFlowError::OperatorReason {
            reason_code: crate::StreamingFailureReason::AsofStateLimitExceeded
                | crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        }
    )
}
