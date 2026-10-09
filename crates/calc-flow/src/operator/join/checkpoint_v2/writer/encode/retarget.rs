use datafusion::execution::memory_pool::MemoryReservation;

use super::super::super::{
    budget,
    ipc::accounting::{self, add, sum},
};
use super::super::buffer::{ensure, error};
use crate::{Epoch, Result, StateSegment};

pub(in crate::operator::join::checkpoint_v2::writer) fn at_epoch(
    mut encoded: super::Base,
    epoch: Epoch,
) -> Result<super::Base> {
    if epoch == Epoch::INITIAL {
        return Ok(encoded);
    }
    for (side, original) in [("left", "left-delta-1"), ("right", "right-delta-1")] {
        if let Some(segment) = encoded.segments.get(original) {
            let (name, replacement) = rebind(&encoded, segment, side, epoch)?;
            encoded.segments.remove(original);
            encoded.segments.insert(name, replacement);
        }
    }
    Ok(encoded)
}

fn rebind(
    encoded: &super::Base,
    original: &StateSegment,
    side: &str,
    epoch: Epoch,
) -> Result<(String, StateSegment)> {
    admit(original, encoded.segments.len(), encoded.credit())?;
    let name = format!("{side}-delta-{}", epoch.as_u64());
    let mut bytes = original.bytes().to_vec();
    write_epoch(&mut bytes, epoch)?;
    let replacement = StateSegment::new(bytes).with_owner(encoded.funding.clone());
    Ok((name, replacement))
}

fn admit(segment: &StateSegment, segments: usize, credit: &MemoryReservation) -> Result<()> {
    let map = budget::tree::<String, StateSegment>(add(segments, 1)?)?;
    let bytes = sum(&[
        segment.bytes().len(),
        96,
        64,
        accounting::arc::<Vec<u8>>()?,
        map,
    ])?;
    ensure(credit, add(credit.size(), bytes)?)
}

fn write_epoch(bytes: &mut [u8], epoch: Epoch) -> Result<()> {
    let header = bytes
        .get_mut(16..24)
        .ok_or_else(|| error("V2 prepared delta header is truncated"))?;
    header.copy_from_slice(&epoch.as_u64().to_le_bytes());
    Ok(())
}
