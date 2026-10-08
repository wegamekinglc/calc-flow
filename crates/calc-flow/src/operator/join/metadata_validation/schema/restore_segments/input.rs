use crate::operator::join::columnar::restored::ResidentLease;
use crate::{OperatorStateSnapshot, Result, StreamJobContext};
use datafusion::execution::memory_pool::MemoryReservation;

// Every partially copied segment and row lease precedes its actual input credit.
pub(super) struct OwnedInput {
    pub snapshot: OperatorStateSnapshot,
    pub segments: Vec<(String, Vec<u8>)>,
    pub key_indices: [Vec<usize>; 2],
    pub leases: [Vec<ResidentLease>; 2],
    pub name: String,
    pub credit: Option<MemoryReservation>,
}

impl OwnedInput {
    pub(super) fn new(credit: MemoryReservation) -> Self {
        Self {
            snapshot: OperatorStateSnapshot::default(),
            segments: Vec::new(),
            key_indices: [Vec::new(), Vec::new()],
            leases: [Vec::new(), Vec::new()],
            name: String::new(),
            credit: Some(credit),
        }
    }

    pub(super) fn take(&mut self) -> Self {
        Self {
            snapshot: std::mem::take(&mut self.snapshot),
            segments: std::mem::take(&mut self.segments),
            key_indices: std::mem::take(&mut self.key_indices),
            leases: std::mem::take(&mut self.leases),
            name: std::mem::take(&mut self.name),
            credit: self.credit.take(),
        }
    }

    pub(super) async fn copy_segments(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        job: &StreamJobContext,
    ) -> Result<()> {
        self.segments = Vec::with_capacity(snapshot.segments.len());
        for (id, segment) in &snapshot.segments {
            super::super::super::copy_boundary(job).await?;
            self.segments.push((
                String::with_capacity(id.len()),
                Vec::with_capacity(segment.bytes().len()),
            ));
            let (copied_id, copied_bytes) =
                self.segments.last_mut().expect("owned partial segment");
            for part in id.as_bytes().chunks(4096) {
                super::super::super::copy_boundary(job).await?;
                copied_id.push_str(std::str::from_utf8(part).expect("recognized ASCII segment id"));
            }
            for part in segment.bytes().chunks(4096) {
                super::super::super::copy_boundary(job).await?;
                copied_bytes.extend_from_slice(part);
            }
        }
        Ok(())
    }
}
