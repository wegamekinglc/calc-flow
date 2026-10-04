use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{super::sql_state_error, control::QuotaLedger};
use crate::{OperatorStateSnapshot, Result, StateSegment};

pub(super) const MAX_FRAMES: usize = 32;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FrameDescriptor {
    pub id: String,
    pub sha256: String,
    pub groups: u64,
    pub ledger: QuotaLedger,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LogDescriptor {
    pub version: u32,
    pub base_generation: u64,
    pub generation: u64,
    pub base_groups: u64,
    pub base_ledger: QuotaLedger,
    pub frames: Vec<FrameDescriptor>,
}

#[derive(Clone)]
pub(super) struct Frame {
    pub descriptor: FrameDescriptor,
    pub segment: StateSegment,
}

#[derive(Clone)]
pub(super) struct GroupLog {
    pub base: StateSegment,
    pub descriptor: LogDescriptor,
    pub frames: Vec<Frame>,
}

#[derive(Clone, Copy)]
pub(super) enum Mode {
    Carry,
    Full,
    Delta,
}

impl GroupLog {
    pub fn mode(&self, changed: usize, groups: usize) -> Mode {
        if changed == 0 {
            Mode::Carry
        } else if self.frames.len() < MAX_FRAMES && changed < groups.div_ceil(2) {
            Mode::Delta
        } else {
            Mode::Full
        }
    }

    pub fn frame_count(&self, mode: Mode) -> usize {
        match mode {
            Mode::Carry => self.frames.len(),
            Mode::Delta => self.frames.len() + 1,
            Mode::Full => 0,
        }
    }

    pub fn finish(
        previous: Option<&Self>,
        mode: Mode,
        segment: Option<StateSegment>,
        groups: usize,
        changed: usize,
        ledger: QuotaLedger,
    ) -> Result<Self> {
        if matches!(mode, Mode::Carry) {
            return Ok(previous.expect("carried group log").clone());
        }
        let generation = previous.map_or(Ok(0), |log| {
            log.descriptor
                .generation
                .checked_add(1)
                .ok_or_else(|| sql_state_error("SQL group log generation overflowed"))
        })?;
        let groups =
            u64::try_from(groups).map_err(|_| sql_state_error("SQL group count exceeds u64"))?;
        let segment = segment.expect("encoded changed group state");
        if matches!(mode, Mode::Full) {
            return Ok(Self {
                base: segment,
                frames: Vec::new(),
                descriptor: LogDescriptor {
                    version: 1,
                    base_generation: generation,
                    generation,
                    base_groups: groups,
                    base_ledger: ledger,
                    frames: Vec::new(),
                },
            });
        }
        let mut log = previous.expect("delta group log").clone();
        let descriptor = FrameDescriptor {
            id: frame_id(generation),
            sha256: segment.sha256().into(),
            groups: u64::try_from(changed)
                .map_err(|_| sql_state_error("SQL delta group count exceeds u64"))?,
            ledger,
        };
        log.descriptor.generation = generation;
        log.descriptor.frames.push(descriptor.clone());
        log.frames.push(Frame {
            descriptor,
            segment,
        });
        Ok(log)
    }

    pub fn restored(snapshot: &OperatorStateSnapshot, descriptor: LogDescriptor) -> Self {
        Self {
            base: snapshot.segments["group-state"].clone(),
            frames: descriptor
                .frames
                .iter()
                .map(|frame| Frame {
                    descriptor: frame.clone(),
                    segment: snapshot.segments[&frame.id].clone(),
                })
                .collect(),
            descriptor,
        }
    }

    pub fn segments(&self, output: &mut BTreeMap<String, StateSegment>) {
        output.insert("group-state".into(), self.base.clone());
        output.extend(
            self.frames
                .iter()
                .map(|frame| (frame.descriptor.id.clone(), frame.segment.clone())),
        );
    }
}

impl LogDescriptor {
    pub fn validate(
        &self,
        snapshot: &OperatorStateSnapshot,
        groups: u64,
        ledger: QuotaLedger,
    ) -> Result<()> {
        if self.version != 1
            || self.frames.len() > MAX_FRAMES
            || self.generation.checked_sub(self.base_generation) != Some(self.frames.len() as u64)
            || self.base_groups > groups
            || !self.base_ledger.seen_input
            || (self.base_groups > self.base_ledger.rows
                && !(self.base_groups == 1 && self.base_ledger.rows == 0))
        {
            return Err(sql_state_error("SQL group log base is invalid"));
        }
        let mut previous = self.base_ledger;
        for (index, frame) in self.frames.iter().enumerate() {
            let generation = self
                .base_generation
                .checked_add(index as u64 + 1)
                .ok_or_else(|| sql_state_error("SQL group log generation overflowed"))?;
            if frame.id != frame_id(generation)
                || frame.groups == 0
                || frame.groups > groups
                || !frame.ledger.seen_input
                || frame.ledger.rows <= previous.rows
                || frame.ledger.bytes < previous.bytes
                || frame.groups > frame.ledger.rows
                || snapshot
                    .segments
                    .get(&frame.id)
                    .is_none_or(|segment| segment.sha256() != frame.sha256)
            {
                return Err(sql_state_error("SQL group log frame is invalid"));
            }
            previous = frame.ledger;
        }
        if ledger.rows < previous.rows
            || ledger.bytes < previous.bytes
            || !ledger.seen_input
            || snapshot.segments.len() != 4 + self.frames.len()
        {
            return Err(sql_state_error(
                "SQL group log ledger or inventory is invalid",
            ));
        }
        Ok(())
    }
}

fn frame_id(generation: u64) -> String {
    format!("group-delta-{generation:020}")
}
