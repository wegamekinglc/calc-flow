use crate::{Result, StreamOperatorContext};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};

const ARC_HEADER: usize = 2 * size_of::<usize>();
pub(in crate::operator::join) const COPY_BYTES: usize = 4096;
pub(in crate::operator::join) const COPY_VISITS: usize = 64;

#[derive(Default)]
pub(in crate::operator::join) struct Quantum {
    visits: usize,
    bytes: usize,
}

impl Quantum {
    pub(in crate::operator::join) async fn step(
        &mut self,
        context: &StreamOperatorContext<'_>,
        visits: usize,
        bytes: usize,
    ) -> Result<()> {
        #[cfg(test)]
        super::note_join_work(|work| work.quantum_steps += 1);
        debug_assert!(visits <= COPY_VISITS && bytes <= COPY_BYTES);
        if self.visits + visits > COPY_VISITS || self.bytes + bytes > COPY_BYTES {
            #[cfg(test)]
            super::note_join_work(|work| work.quantum_boundaries += 1);
            context.check_cancelled()?;
            tokio::task::consume_budget().await;
            context.check_cancelled()?;
            self.visits = 0;
            self.bytes = 0;
        }
        self.visits += visits;
        self.bytes += bytes;
        Ok(())
    }

    pub(in crate::operator::join) async fn grant_admission(
        &mut self,
        context: &StreamOperatorContext<'_>,
        maximum: usize,
    ) -> Result<usize> {
        debug_assert!(maximum > 0);
        let available = self.admission_capacity().min(maximum);
        if available > 0 {
            self.step(context, available, available * 16).await?;
            return Ok(available);
        }
        self.step(context, 1, 16).await?;
        let additional = self.admission_capacity().min(maximum - 1);
        self.visits += additional;
        self.bytes += additional * 16;
        Ok(additional + 1)
    }

    fn admission_capacity(&self) -> usize {
        (COPY_VISITS - self.visits).min((COPY_BYTES - self.bytes) / 16)
    }
}

pub(in crate::operator::join) fn reservation_control() -> Option<usize> {
    (size_of::<MemoryConsumer>()
        + size_of::<std::sync::Arc<dyn datafusion::execution::memory_pool::MemoryPool>>()
        + ARC_HEADER)
        .checked_add(
            3 * "sql-incremental:stream-join-native".len()
                + size_of::<MemoryReservation>()
                + ARC_HEADER,
        )
}
