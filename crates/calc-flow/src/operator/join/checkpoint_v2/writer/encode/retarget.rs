use datafusion::execution::memory_pool::MemoryReservation;

use super::super::super::{
    budget,
    ipc::accounting::{add, sum},
};
use super::super::buffer::{PaidBuffer, ensure, error};
use super::{own_segment, segment_charge};
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
    let charge = admit(original, encoded.segments.len(), encoded.credit())?;
    let name = format!("{side}-delta-{}", epoch.as_u64());
    let mut bytes = PaidBuffer::exact(encoded.credit(), 0, original.bytes().len())?;
    bytes.append(original.bytes())?;
    let mut bytes = bytes.into_bytes()?;
    write_epoch(&mut bytes, epoch)?;
    let replacement = own_segment(bytes, encoded.credit(), charge);
    Ok((name, replacement))
}

fn admit(segment: &StateSegment, segments: usize, credit: &MemoryReservation) -> Result<usize> {
    let map = budget::tree::<String, StateSegment>(add(segments, 1)?)?;
    let backing = segment_charge(segment.bytes().len())?;
    ensure(credit, sum(&[credit.size(), backing, 96, map])?)?;
    Ok(backing)
}

fn write_epoch(bytes: &mut [u8], epoch: Epoch) -> Result<()> {
    let header = bytes
        .get_mut(16..24)
        .ok_or_else(|| error("V2 prepared delta header is truncated"))?;
    header.copy_from_slice(&epoch.as_u64().to_le_bytes());
    Ok(())
}
