use super::{
    Anchor, AnchorControl, Callback, Log, MAX_FRAMES, Record, codec, is_capacity_error, mismatch,
};
use crate::{Epoch, OperatorStateSnapshot, Result, StreamAsofJoinOperator, StreamOperatorContext};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[cfg(test)]
mod tests;

impl StreamAsofJoinOperator {
    pub(in crate::operator::asof) fn replay_anchor_due(&self) -> bool {
        self.replay
            .as_ref()
            .is_some_and(|log| log.frames.len() >= MAX_FRAMES && log.cut < log.records.len())
    }

    pub(in crate::operator::asof) async fn prepare_replay_anchor(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()?;
        if !self.replay_anchor_due() {
            return Ok(());
        }
        let log = self.replay.take();
        let result = self.ensure_prepared_async(context).await;
        self.replay = log;
        self.status.state_bytes = self.current_inventory(self.prepared.as_ref())?.bytes;
        match result {
            Err(error) if is_capacity_error(&error) => {
                self.stop_replay()?;
                self.ensure_prepared_async(context).await
            }
            result => result,
        }
    }

    pub(in crate::operator::asof) fn replace_replay_anchor(
        &mut self,
        epoch: Epoch,
    ) -> Result<Option<OperatorStateSnapshot>> {
        let mut log = self.replay.take().expect("anchor has a replay log");
        self.status.state_bytes = self.current_inventory(self.prepared.as_ref())?.bytes;
        let result = self.capture(epoch);
        let snapshot = match result {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.replay = Some(log);
                self.status.state_bytes = self.current_inventory(self.prepared.as_ref())?.bytes;
                return Err(error);
            }
        };
        let anchor = self.make_replay_anchor(&snapshot, &log);
        let anchor = match anchor {
            Ok(anchor) => anchor,
            Err(error) if is_capacity_error(&error) => {
                self.status.state_bytes = self.current_inventory(None)?.bytes;
                return Ok(Some(snapshot));
            }
            Err(error) => {
                self.replay = Some(log);
                self.status.state_bytes = self.current_inventory(self.prepared.as_ref())?.bytes;
                return Err(error);
            }
        };
        let cursor_bytes = log.records.iter().try_fold(0_usize, |total, record| {
            total
                .checked_add(record.cursor_bytes)
                .ok_or_else(|| mismatch("cursor credit overflowed"))
        })?;
        log.records.clear();
        log.credit.shrink(cursor_bytes);
        log.frames.clear();
        log.cut = 0;
        log.generation = log
            .generation
            .checked_add(1)
            .ok_or_else(|| mismatch("generation exhausted"))?;
        log.anchor = Some(Box::new(anchor));
        self.clear_anchor_bookkeeping();
        self.replay = Some(log);
        self.status.state_bytes = self.current_inventory(None)?.bytes;
        Ok(None)
    }

    fn make_replay_anchor(&self, snapshot: &OperatorStateSnapshot, log: &Log) -> Result<Anchor> {
        let last = log
            .records
            .last()
            .ok_or_else(|| mismatch("anchor has no callback"))?;
        let positions = [0, 1].map(|side| {
            log.records.iter().rev().find(|record| matches!(record.callback, Callback::Data {side: s, ..} if s == side))
                .or_else(|| log.anchor.as_ref()?.records.iter().find(|record| matches!(record.callback, Callback::Data {side: s, ..} if s == side)))
        });
        let credit_bytes = positions.iter().flatten().try_fold(
            anchor_metadata_bytes(&snapshot.inline_metadata, snapshot.segments.keys())?,
            |total, record| {
                total
                    .checked_add(record.cursor_bytes)
                    .ok_or_else(|| mismatch("anchor credit overflowed"))
            },
        )?;
        let credit = self.reserve_workspace(credit_bytes as u64)?;
        let mut records = Vec::with_capacity(3);
        let mut progress = last.clone();
        progress.callback = Callback::Progress;
        progress.cursor = None;
        progress.cursor_bytes = 0;
        records.push(progress);
        records.extend(positions.into_iter().flatten().cloned());
        let length = codec::encoded_len(&records)?;
        let bytes = length
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(512))
            .ok_or_else(|| mismatch("anchor start frame overflowed"))?;
        let segment_credit = self.reserve_workspace(bytes as u64)?;
        let starts =
            crate::StateSegment::new(codec::encode(&records)?).with_owner(Arc::new(segment_credit));
        let generation = log
            .generation
            .checked_add(1)
            .ok_or_else(|| mismatch("generation exhausted"))?;
        Ok(Anchor {
            snapshot: snapshot.clone(),
            starts,
            start_id: format!("asof-replay-start-{generation:020}"),
            records,
            credit,
        })
    }

    pub(in crate::operator::asof) fn clear_anchor_bookkeeping(&mut self) {
        self.checkpoint_log = crate::operator::asof::checkpoint::LogState::default();
        self.prepared = None;
        self.swept = None;
    }

    pub(super) fn decode_replay_anchor(
        &self,
        snapshot: &OperatorStateSnapshot,
        descriptor: &AnchorControl,
    ) -> Result<Option<Box<Anchor>>> {
        let AnchorControl::Native {
            metadata,
            segments,
            starts,
            credit,
        } = descriptor
        else {
            return Ok(None);
        };
        if *credit as u64 > self.spec.limits().max_state_bytes()
            || segments.len() > snapshot.segments.len()
            || starts == super::CONTROL_ID
        {
            return Err(mismatch("anchor limits or start identity differ"));
        }
        let metadata_bytes = anchor_metadata_bytes(metadata, segments.iter())?;
        if metadata_bytes > *credit {
            return Err(mismatch("anchor metadata exceeds retained credit"));
        }
        let owner = self.reserve_workspace(*credit as u64)?;
        let mut seen = BTreeSet::from([super::CONTROL_ID, starts.as_str()]);
        let mut native = OperatorStateSnapshot {
            inline_metadata: metadata.clone(),
            segments: BTreeMap::new(),
        };
        for id in segments {
            if !seen.insert(id.as_str()) {
                return Err(mismatch("anchor segment identity repeats"));
            }
            let segment = snapshot
                .segments
                .get(id)
                .ok_or_else(|| mismatch("missing anchor segment"))?;
            native.segments.insert(id.clone(), segment.clone());
        }
        let starts_segment = snapshot
            .segments
            .get(starts)
            .ok_or_else(|| mismatch("missing anchor start frame"))?;
        let bytes = starts_segment.bytes_arc();
        let count = bytes
            .get(8..16)
            .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
            .ok_or_else(|| mismatch("truncated anchor start frame"))?;
        if !(1..=3).contains(&count) {
            return Err(mismatch("anchor start count differs"));
        }
        let mut records = Vec::with_capacity(3);
        let mut cursor_credit = self.reserve_workspace(0)?;
        codec::decode_into(&bytes, count, &mut records, &mut cursor_credit, &|bytes| {
            self.reserve_workspace(bytes)
        })?;
        self.validate_anchor_starts(&records)?;
        let required = records.iter().try_fold(metadata_bytes, |total, record| {
            total
                .checked_add(record.cursor_bytes)
                .ok_or_else(|| mismatch("anchor credit overflowed"))
        })?;
        if required > *credit {
            return Err(mismatch("anchor metadata exceeds retained credit"));
        }
        Ok(Some(Box::new(Anchor {
            snapshot: native,
            starts: starts_segment.clone(),
            start_id: starts.clone(),
            records,
            credit: owner,
        })))
    }

    fn validate_anchor_starts(&self, records: &[Record]) -> Result<()> {
        if records
            .first()
            .is_none_or(|record| record.callback != Callback::Progress)
        {
            return Err(mismatch("anchor has no progress record"));
        }
        let mut seen = [false; 2];
        for record in &records[1..] {
            let Callback::Data { side, sequence } = record.callback else {
                return Err(mismatch("anchor start is not data"));
            };
            let bindings = &self
                .replay_inputs
                .as_ref()
                .ok_or_else(|| mismatch("replay inputs unavailable"))?
                .bindings;
            let side = usize::from(side);
            if seen[side]
                || sequence == u64::MAX
                || record.cursor.as_ref().and_then(|cursor| cursor.source_id())
                    != Some(bindings[side].as_str())
            {
                return Err(mismatch("anchor source position differs"));
            }
            seen[side] = true;
        }
        Ok(())
    }
}

fn anchor_metadata_bytes<'a>(
    metadata: &crate::JsonMap,
    mut segments: impl Iterator<Item = &'a String>,
) -> Result<usize> {
    let base = crate::json::owned_json_bytes(metadata)
        .and_then(|bytes| bytes.checked_add(1024 + 3 * size_of::<Record>()))
        .ok_or_else(|| mismatch("anchor metadata overflowed"))?;
    segments.try_fold(base, |total, id| {
        total
            .checked_add(256)
            .and_then(|bytes| bytes.checked_add(id.capacity()))
            .ok_or_else(|| mismatch("anchor segment metadata overflowed"))
    })
}
