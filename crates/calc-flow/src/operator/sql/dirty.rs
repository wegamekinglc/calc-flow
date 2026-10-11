use super::{MemoryReservation, Result, checked_bytes, df_error, ensure_reservation};

pub(super) struct DirtyGroups {
    bits: Vec<u64>,
    slots: Vec<usize>,
    reservation: MemoryReservation,
}

impl DirtyGroups {
    pub(super) fn empty(reservation: MemoryReservation) -> Self {
        Self {
            bits: Vec::new(),
            slots: Vec::new(),
            reservation,
        }
    }

    pub(super) fn slots(&self) -> &[usize] {
        &self.slots
    }

    pub(super) fn prepare(&self, groups: usize, name: &str) -> Result<Option<Self>> {
        if groups <= self.slots.capacity() {
            return Ok(None);
        }
        let capacity = groups
            .max(4)
            .checked_next_power_of_two()
            .ok_or_else(|| df_error(name, "dirty group capacity overflowed"))?;
        let words = capacity.div_ceil(64);
        let bytes = checked_bytes(
            4096,
            [(capacity, size_of::<usize>()), (words, size_of::<u64>())],
            name,
        )?;
        let reservation = self.reservation.new_empty();
        ensure_reservation(&reservation, bytes, name)?;
        let mut bits = vec![0; words];
        bits[..self.bits.len()].copy_from_slice(&self.bits);
        let mut slots = Vec::with_capacity(capacity);
        slots.extend_from_slice(&self.slots);
        Ok(Some(Self {
            bits,
            slots,
            reservation,
        }))
    }

    pub(super) fn mark(&mut self, slot: usize) {
        let bit = 1u64 << (slot % 64);
        let word = &mut self.bits[slot / 64];
        if *word & bit == 0 {
            *word |= bit;
            self.slots.push(slot);
        }
    }

    pub(super) fn clear(&mut self) {
        for &slot in &self.slots {
            self.bits[slot / 64] &= !(1u64 << (slot % 64));
        }
        self.slots.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Batch, BatchMetadata, CancellationToken, EdgeCollector, JsonMap, OperatorMetadata,
        SqlOperator, StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    use datafusion::arrow::{
        array::{ArrayRef, Int64Array},
        record_batch::RecordBatch,
    };
    use std::sync::Arc;

    #[tokio::test]
    async fn test_checkpoint_disabled_sql_aggregates_without_allocating_dirty_groups() {
        let mut operator = SqlOperator::new(
            "totals",
            "SELECT value, COUNT(*) AS n FROM events GROUP BY value",
            vec!["events".into()],
            vec![],
        )
        .unwrap();
        let job = StreamJobContext::new(1, "sql", JsonMap::new(), None, CancellationToken::new())
            .with_checkpointing(false);
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        for values in [vec![1, 2], vec![1, 3]] {
            let record = RecordBatch::try_from_iter(vec![(
                "value",
                Arc::new(Int64Array::from(values)) as ArrayRef,
            )])
            .unwrap();
            operator
                .process_data(
                    "events",
                    Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut output,
                )
                .await
                .unwrap();
            let state = operator.incremental.as_ref().unwrap();
            assert!(state.dirty.slots.is_empty());
            assert_eq!(state.dirty.slots.capacity(), 0);
            assert_eq!(state.dirty.bits.capacity(), 0);
            assert_eq!(state.dirty.reservation.size(), 0);
        }
        let emitted = output.drain("output");
        let batches = emitted
            .last()
            .unwrap()
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches();
        let total = batches
            .iter()
            .map(|batch| {
                batch
                    .column_by_name("n")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .sum::<i64>()
            })
            .sum::<i64>();
        assert_eq!(total, 4);
        assert_eq!(operator.incremental.as_ref().unwrap().groups.len(), 3);
    }
}
