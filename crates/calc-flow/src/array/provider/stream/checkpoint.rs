use std::sync::Arc;

use sha2::{Digest as _, Sha256};

use super::options::Options;
use crate::{
    JsonMap, OperatorStateSnapshot, Result, StateSegment,
    array::sequence::{
        capture::Captures,
        codec::{self, RestoreLimits, SequenceSpec, SequenceState},
    },
};

pub(super) fn metadata(spec: &SequenceSpec) -> JsonMap {
    std::collections::BTreeMap::from([
        ("identity".into(), "native:array_sequence@1".into()),
        ("layout".into(), 1.into()),
        ("numeric_semantics".into(), 1.into()),
        (
            "state_fingerprint".into(),
            hex::encode(spec.fingerprint()).into(),
        ),
    ])
}

pub(super) struct Restored {
    pub(super) state: Arc<SequenceState>,
    pub(super) segment: StateSegment,
}

pub(super) fn validate(snapshot: &OperatorStateSnapshot, spec: &SequenceSpec) -> Result<()> {
    if snapshot.inline_metadata != metadata(spec) {
        return Err(codec::mismatch(
            "metadata",
            "identity, layout, numeric semantics or fingerprint differs",
        ));
    }
    if snapshot.segments.len() != 1 || !snapshot.segments.contains_key("state") {
        return Err(codec::mismatch(
            "segments",
            "requires exactly one segment named state",
        ));
    }
    Ok(())
}

pub(super) fn restore(
    configured: &Options,
    old: usize,
    captures: &Captures,
    snapshot: &OperatorStateSnapshot,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Restored> {
    check()?;
    validate(snapshot, &configured.spec)?;
    let segment = &snapshot.segments["state"];
    let state = candidate(configured, old, captures, segment, check)?;
    finish(state, segment, configured, old, captures, check)
}
fn finish(
    state: SequenceState,
    segment: &StateSegment,
    configured: &Options,
    old: usize,
    captures: &Captures,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Restored> {
    let live = old
        .checked_add(state.state_bytes())
        .ok_or_else(|| codec::mismatch("peak_bytes", "live byte count overflowed usize"))?;
    let segment = captures.retain(segment.clone(), &configured.execution, live)?;
    let state = Arc::new(state);
    check()?;
    Ok(Restored { state, segment })
}
fn candidate(
    configured: &Options,
    old: usize,
    captures: &Captures,
    segment: &StateSegment,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<SequenceState> {
    let limits = limits(configured, old, captures, segment)?;
    codec::admit(
        &configured.execution,
        &[
            old,
            limits.captures,
            limits.encoded,
            StateSegment::controlled_scratch_bytes(),
        ],
    )?;
    checksum(segment, check)?;
    codec::decode(&configured.spec, segment.bytes(), &limits, check)
}

fn limits<'a>(
    configured: &'a Options,
    old: usize,
    captures: &Captures,
    segment: &StateSegment,
) -> Result<RestoreLimits<'a>> {
    let encoded = segment
        .retained_buffer_bytes()
        .ok_or_else(|| codec::mismatch("encoded_bytes", "retained byte count overflowed usize"))?;
    Ok(RestoreLimits {
        state: configured.state,
        execution: &configured.execution,
        old,
        captures: captures.other_bytes(segment)?,
        encoded,
    })
}
fn checksum(segment: &StateSegment, check: &mut dyn FnMut() -> Result<()>) -> Result<()> {
    let mut hash = Sha256::new();
    for chunk in segment.bytes().chunks(4096) {
        check()?;
        hash.update(chunk);
    }
    let digest: [u8; 32] = hash.finalize().into();
    check()?;
    validate_digest(&digest, segment.sha256())
}
fn validate_digest(digest: &[u8; 32], checksum: &str) -> Result<()> {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    if checksum.len() != 64 {
        return Err(codec::mismatch(
            "checksum",
            "requires a 64-character SHA-256 digest",
        ));
    }
    for (index, byte) in digest.iter().copied().enumerate() {
        let expected = [
            DIGITS[usize::from(byte >> 4)],
            DIGITS[usize::from(byte & 15)],
        ];
        if checksum.as_bytes()[2 * index..2 * index + 2] != expected {
            return Err(codec::mismatch(
                "checksum",
                "SHA-256 digest differs from the state bytes",
            ));
        }
    }
    Ok(())
}
