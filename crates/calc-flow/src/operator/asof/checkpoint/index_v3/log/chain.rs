use crate::{Result, StateSegment};
use datafusion::execution::memory_pool::MemoryReservation;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};

const MAGIC: &[u8; 8] = b"CFASDL10";
pub(in crate::operator::asof) const HEADER_BYTES: usize = 128;
pub(in crate::operator::asof) const MAX_DELTAS: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::operator::asof) enum Kind {
    Base,
    Delta,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::operator::asof) struct Descriptor {
    pub generation: u64,
    pub ordinal: u32,
    pub epoch: u64,
    pub sha256: String,
    pub bytes: u64,
}

impl Descriptor {
    pub fn name(&self) -> String {
        format!(
            "asof-log-v10-{}-{}-{}",
            self.generation, self.ordinal, self.epoch
        )
    }
}

pub(in crate::operator::asof) struct Frame<'a> {
    pub kind: Kind,
    pub generation: u64,
    pub ordinal: u32,
    pub epoch: u64,
    pub preceding_epoch: u64,
    pub preceding_sha256: [u8; 32],
    pub records: u64,
    pub body: &'a [u8],
}

fn digest(value: &str) -> Result<[u8; 32]> {
    let mut digest = [0; 32];
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !(byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        return Err(super::super::mismatch("ASOF log digest is noncanonical"));
    }
    hex::decode_to_slice(value, &mut digest)
        .map_err(|_| super::super::mismatch("ASOF log digest differs"))?;
    Ok(digest)
}

pub(in crate::operator::asof) fn encode(
    generation: u64,
    epoch: u64,
    preceding: Option<&Descriptor>,
    fingerprint: &str,
    records: u64,
    body: &[u8],
    lease: MemoryReservation,
) -> Result<(Descriptor, StateSegment)> {
    let ordinal = preceding.map_or(Ok(0), |previous| {
        if previous.generation != generation || previous.epoch >= epoch {
            return Err(super::super::mismatch(
                "ASOF log predecessor coordinate differs",
            ));
        }
        previous
            .ordinal
            .checked_add(1)
            .ok_or_else(|| super::super::mismatch("ASOF log ordinal overflowed"))
    })?;
    if usize::try_from(ordinal).map_or(true, |value| value > MAX_DELTAS) {
        return Err(super::super::mismatch("ASOF log chain requires compaction"));
    }
    let length = body
        .len()
        .checked_add(HEADER_BYTES)
        .ok_or_else(|| super::super::mismatch("ASOF log length overflowed"))?;
    if lease.size() < length.saturating_add(256) {
        return Err(super::super::mismatch(
            "ASOF log encoding exceeds prepaid workspace",
        ));
    }
    let fingerprint = digest(fingerprint)?;
    let preceding_sha256 = preceding
        .map(|previous| digest(&previous.sha256))
        .transpose()?
        .unwrap_or([0; 32]);
    let mut bytes = Vec::with_capacity(length);
    bytes.extend_from_slice(MAGIC);
    bytes.push(u8::from(preceding.is_some()));
    bytes.extend_from_slice(&[0; 7]);
    bytes.extend_from_slice(&generation.to_le_bytes());
    bytes.extend_from_slice(&ordinal.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&epoch.to_le_bytes());
    bytes.extend_from_slice(&preceding.map_or(0, |previous| previous.epoch).to_le_bytes());
    bytes.extend_from_slice(&preceding_sha256);
    bytes.extend_from_slice(&fingerprint);
    bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&records.to_le_bytes());
    bytes.extend_from_slice(body);
    if bytes.len() != length || bytes.capacity() > lease.size() {
        return Err(super::super::mismatch("ASOF log encoded length differs"));
    }
    let segment = StateSegment::new(bytes).with_owner(Arc::new(lease));
    let descriptor = Descriptor {
        generation,
        ordinal,
        epoch,
        sha256: segment.sha256().into(),
        bytes: u64::try_from(length).expect("address domain fits u64"),
    };
    #[cfg(test)]
    super::super::super::cost::index_bytes(segment.bytes().len());
    Ok((descriptor, segment))
}

pub(in crate::operator::asof) fn decode<'a>(
    segment: &'a StateSegment,
    fingerprint: &str,
) -> Result<Frame<'a>> {
    let mut cursor = super::super::Cursor::new(segment.bytes(), u64::MAX);
    if cursor.take(8)? != MAGIC {
        return Err(super::super::mismatch("ASOF current log magic differs"));
    }
    let kind = match cursor.byte()? {
        0 => Kind::Base,
        1 => Kind::Delta,
        _ => return Err(super::super::mismatch("ASOF log record kind differs")),
    };
    if cursor.take(7)? != [0; 7] {
        return Err(super::super::mismatch("ASOF log padding differs"));
    }
    let generation = cursor.integer()?;
    let ordinal = cursor.small()?;
    if cursor.take(4)? != [0; 4] {
        return Err(super::super::mismatch("ASOF log ordinal padding differs"));
    }
    let epoch = cursor.integer()?;
    let preceding_epoch = cursor.integer()?;
    let preceding_sha256 = cursor.take(32)?.try_into().expect("digest width");
    if cursor.take(32)? != digest(fingerprint)? {
        return Err(super::super::mismatch("ASOF log fingerprint differs"));
    }
    let body_length = cursor.address()?;
    let records = cursor.integer()?;
    let body = cursor.take(body_length)?;
    cursor.finish()?;
    Ok(Frame {
        kind,
        generation,
        ordinal,
        epoch,
        preceding_epoch,
        preceding_sha256,
        records,
        body,
    })
}

pub(in crate::operator::asof) fn validate_chain<'a>(
    inventory: &[Descriptor],
    segments: &'a BTreeMap<String, StateSegment>,
    current_epoch: u64,
    fingerprint: &str,
    workspace: &MemoryReservation,
    mut check_cancelled: impl FnMut() -> Result<()>,
) -> Result<Vec<Frame<'a>>> {
    if inventory.is_empty() || inventory.len() > MAX_DELTAS + 1 {
        return Err(super::super::mismatch("ASOF log inventory length differs"));
    }
    check_cancelled()?;
    if workspace.size() < inventory.len() * (size_of::<Frame<'_>>() + 128) {
        return Err(super::super::mismatch(
            "ASOF log validation exceeds prepaid workspace",
        ));
    }
    let mut frames = Vec::with_capacity(inventory.len());
    let mut preceding: Option<&Descriptor> = None;
    for (ordinal, descriptor) in inventory.iter().enumerate() {
        check_cancelled()?;
        let frame = validated_frame(
            ordinal,
            descriptor,
            segments,
            current_epoch,
            fingerprint,
            preceding,
        )?;
        frames.push(frame);
        preceding = Some(descriptor);
    }
    Ok(frames)
}

pub(in crate::operator::asof) fn requires_compaction(
    inventory: &[Descriptor],
    next_delta_bytes: u64,
    next_metadata_bytes: u64,
    retired_bytes: u64,
    full_base_bytes: u64,
    max_state_bytes: u64,
) -> bool {
    let Some(base) = inventory.first() else {
        return true;
    };
    let delta_bytes = inventory
        .iter()
        .skip(1)
        .try_fold(next_delta_bytes, |total, segment| {
            total.checked_add(segment.bytes)
        });
    let retained = inventory
        .iter()
        .try_fold(next_delta_bytes, |total, segment| {
            total.checked_add(segment.bytes)
        });
    inventory.len() > MAX_DELTAS
        || delta_bytes.is_none_or(|bytes| bytes >= base.bytes)
        || retained
            .and_then(|bytes| bytes.checked_add(next_metadata_bytes))
            .is_none_or(|bytes| bytes > max_state_bytes)
        || (retired_bytes != 0 && retired_bytes >= full_base_bytes)
}

fn validated_frame<'a>(
    ordinal: usize,
    descriptor: &Descriptor,
    segments: &'a BTreeMap<String, StateSegment>,
    current_epoch: u64,
    fingerprint: &str,
    preceding: Option<&Descriptor>,
) -> Result<Frame<'a>> {
    if descriptor.ordinal as usize != ordinal || descriptor.epoch > current_epoch {
        return Err(super::super::mismatch(
            "ASOF log inventory coordinate differs",
        ));
    }
    let segment = checked_segment(segments, descriptor)?;
    let frame = decode(segment, fingerprint)?;
    if (frame.generation, frame.ordinal, frame.epoch)
        != (descriptor.generation, descriptor.ordinal, descriptor.epoch)
    {
        return Err(super::super::mismatch(
            "ASOF log header and inventory differ",
        ));
    }
    validate_ancestry(&frame, preceding)?;
    Ok(frame)
}

fn checked_segment<'a>(
    segments: &'a BTreeMap<String, StateSegment>,
    descriptor: &Descriptor,
) -> Result<&'a StateSegment> {
    let segment = segments
        .get(&descriptor.name())
        .ok_or_else(|| super::super::mismatch("ASOF log segment is missing"))?;
    if segment.sha256() != descriptor.sha256 || segment.bytes().len() as u64 != descriptor.bytes {
        return Err(super::super::mismatch(
            "ASOF log inventory digest or length differs",
        ));
    }
    Ok(segment)
}

fn validate_ancestry(frame: &Frame<'_>, preceding: Option<&Descriptor>) -> Result<()> {
    let valid = match preceding {
        None => {
            frame.kind == Kind::Base
                && frame.preceding_epoch == 0
                && frame.preceding_sha256 == [0; 32]
        }
        Some(previous) => valid_delta_ancestry(frame, previous)?,
    };
    if !valid {
        return Err(super::super::mismatch(
            "ASOF log materialized ancestry differs",
        ));
    }
    Ok(())
}

fn valid_delta_ancestry(frame: &Frame<'_>, previous: &Descriptor) -> Result<bool> {
    Ok(frame.kind == Kind::Delta
        && frame.generation == previous.generation
        && previous.epoch < frame.epoch
        && frame.preceding_epoch == previous.epoch
        && frame.preceding_sha256 == digest(&previous.sha256)?)
}
