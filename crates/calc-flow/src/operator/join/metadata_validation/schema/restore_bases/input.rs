use crate::operator::join::columnar::restored::ResidentLease;
use crate::{OperatorStateSnapshot, Result, StreamJobContext};
use datafusion::execution::memory_pool::MemoryReservation;
use std::ops::Range;

pub(super) struct Geometry {
    pub rows: [usize; 2],
    pub bytes: [usize; 2],
}

pub(super) fn geometry(snapshot: &OperatorStateSnapshot) -> Option<Geometry> {
    if snapshot.segments.len() != 2 {
        return None;
    }
    let left = snapshot.segments.get("left-base")?.bytes();
    let right = snapshot.segments.get("right-base")?.bytes();
    Some(Geometry {
        rows: [geometry_side(left)?, geometry_side(right)?],
        bytes: [left.len(), right.len()],
    })
}

pub(super) fn geometry_side(bytes: &[u8]) -> Option<usize> {
    let mut offset = 0;
    if take(bytes, &mut offset, 8)? != crate::operator::join::JOIN_STATE_MAGIC {
        return None;
    }
    let count = read_usize(bytes, &mut offset)?;
    (count.checked_mul(32)?.checked_add(offset)? <= bytes.len()).then_some(count)
}

pub(super) struct OwnedInput {
    pub snapshot: OperatorStateSnapshot,
    pub bases: [Vec<u8>; 2],
    pub key_indices: [Vec<usize>; 2],
    pub leases: [Vec<ResidentLease>; 2],
    pub name: String,
    pub credit: Option<MemoryReservation>,
}

impl OwnedInput {
    pub(super) fn new(credit: MemoryReservation) -> Self {
        Self {
            snapshot: OperatorStateSnapshot::default(),
            bases: [Vec::new(), Vec::new()],
            key_indices: [Vec::new(), Vec::new()],
            leases: [Vec::new(), Vec::new()],
            name: String::new(),
            credit: Some(credit),
        }
    }

    pub(super) fn take(&mut self) -> Self {
        Self {
            snapshot: std::mem::take(&mut self.snapshot),
            bases: std::mem::take(&mut self.bases),
            key_indices: std::mem::take(&mut self.key_indices),
            leases: std::mem::take(&mut self.leases),
            name: std::mem::take(&mut self.name),
            credit: self.credit.take(),
        }
    }

    pub(super) async fn copy_bases(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
    ) -> Result<()> {
        for (output, side) in self.bases.iter_mut().zip(["left-base", "right-base"]) {
            let source = snapshot.segments[side].bytes();
            *output = Vec::with_capacity(source.len());
            for part in source.chunks(4096) {
                super::super::super::copy_boundary(job).await?;
                output.extend_from_slice(part);
            }
        }
        Ok(())
    }
}

pub(super) fn take<'a>(bytes: &'a [u8], offset: &mut usize, count: usize) -> Option<&'a [u8]> {
    let end = offset.checked_add(count)?;
    let part = bytes.get(*offset..end)?;
    *offset = end;
    Some(part)
}

pub(super) fn read_usize(bytes: &[u8], offset: &mut usize) -> Option<usize> {
    usize::try_from(u64::from_le_bytes(take(bytes, offset, 8)?.try_into().ok()?)).ok()
}

pub(super) fn next_row(bytes: &[u8], offset: &mut usize) -> Option<Range<usize>> {
    take(bytes, offset, 24)?;
    let count = read_usize(bytes, offset)?;
    let start = *offset;
    take(bytes, offset, count)?;
    Some(start..*offset)
}
