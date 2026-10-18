//! Preparation borrows the existing native key-ID column. Only selection and
//! logical charges survive until the owned chunks have been filled.

use super::super::{
    admission_charge::PayloadCharges, admitted::AdmittedRows, native_lookup::NativeKeys,
};
use super::{
    Append, CHUNK_ROWS, Chunk, MemoryReservation, Result, StateRows, StreamOperatorContext,
    chunk_base_bytes, grant, reserve_references,
};

pub(in crate::operator::join) struct ColumnarPlan {
    positions: Vec<usize>,
    charges: Vec<u64>,
    pub(in crate::operator::join) bytes: u64,
    _credit: Option<MemoryReservation>,
}

impl ColumnarPlan {
    pub(in crate::operator::join) fn len(&self) -> usize {
        self.positions.len()
    }

    pub(in crate::operator::join) async fn prepare(
        rows: &dyn AdmittedRows,
        keys: &NativeKeys,
        master: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Self> {
        let count: usize = rows
            .groups()
            .map(|group| {
                (0..group.len())
                    .step_by(64)
                    .map(|start| {
                        group
                            .retained_bits(start, (group.len() - start).min(64))
                            .count_ones() as usize
                    })
                    .sum::<usize>()
            })
            .sum();
        let bytes = count
            .checked_mul(size_of::<usize>() + size_of::<u64>())
            .and_then(|bytes| bytes.checked_add(64))
            .ok_or_else(|| super::super::native_lookup::scratch_error("Join retained selection"))?;
        let credit = if count == 0 {
            None
        } else {
            grant(master, bytes, context.operator_id(), false)?
        };
        let mut plan = Self {
            positions: Vec::with_capacity(count),
            charges: Vec::with_capacity(count),
            bytes: 0,
            _credit: credit,
        };
        let mut quantum = super::super::columnar::Quantum::default();
        let mut base = 0;
        for group in rows.groups() {
            for start in (0..group.len()).step_by(64) {
                let end = (start + 64).min(group.len());
                let mut selected = group.retained_bits(start, end - start);
                if selected == 0 {
                    continue;
                }
                let charges =
                    PayloadCharges::new(group.block(start, end), selected, context, &mut quantum)
                        .await?;
                while selected != 0 {
                    let index = start + selected.trailing_zeros() as usize;
                    selected &= selected - 1;
                    quantum.step(context, 1, 0).await?;
                    let row = group.row(index);
                    let charge = charges.state_charge(
                        index - start,
                        row.record,
                        keys.key(base + index).len(),
                        context.operator_id(),
                    )?;
                    plan.positions.push(base + index);
                    plan.charges.push(charge);
                    plan.bytes = plan.bytes.checked_add(charge).ok_or_else(|| {
                        super::super::counter_overflow(context.operator_id(), "state bytes")
                    })?;
                }
            }
            base += group.len();
        }
        Ok(plan)
    }
}

impl StateRows {
    pub(in crate::operator::join) async fn prepare_columns(
        &self,
        rows: &dyn AdmittedRows,
        keys: &NativeKeys,
        plan: &ColumnarPlan,
        master: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Append> {
        self.validate_append(plan.len())?;
        let control = reserve_references(master, plan.len(), context.operator_id(), false)?;
        let indices = self.append_indices(plan.len());
        let mut chunks = Vec::with_capacity(plan.len().div_ceil(CHUNK_ROWS));
        // Bound each group's segment once; no row descriptors or additional
        // key-ID column are assembled on the funded path.
        let mut groups = rows.groups();
        let mut group = groups.next();
        let mut group_base = 0;
        let mut quantum = super::super::columnar::Quantum::default();
        for (block, charges) in plan
            .positions
            .chunks(CHUNK_ROWS)
            .zip(plan.charges.chunks(CHUNK_ROWS))
        {
            let mut sources = 0;
            let mut end = 0;
            for &position in block {
                if position >= end {
                    sources += 1;
                    end = rows.group_end(position);
                }
            }
            let key_count = block.len().min(keys.keys.len());
            let bytes = chunk_base_bytes(block.len(), sources, key_count)
                .ok_or_else(|| super::super::native_lookup::scratch_error("Join state chunks"))?;
            let credit = grant(master, bytes, context.operator_id(), false)?;
            let mut chunk = Chunk::reserved_keys(block.len(), sources, key_count, credit);
            for (&position, &charge) in block.iter().zip(charges) {
                while group.is_some_and(|group| position >= group_base + group.len()) {
                    group_base += group.expect("selected group").len();
                    group = groups.next();
                }
                let row = group
                    .expect("selected admission group")
                    .row(position - group_base);
                quantum.step(context, 1, 128).await?;
                chunk.push_columns(
                    row.record,
                    row.event_time,
                    row.row_id,
                    charge,
                    keys.key(position),
                );
            }
            chunks.push(chunk);
        }
        Ok(Append {
            chunks,
            indices,
            count: plan.len(),
            control,
        })
    }
}
