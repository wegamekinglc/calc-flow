//! Preparation borrows the existing native key-ID column. Only selection and
//! logical charges survive until the owned chunks have been filled.

use super::super::{
    admission_charge::PayloadCharges,
    admitted::{AdmittedGroup, AdmittedRef, AdmittedRows, GroupIter},
    native_lookup::NativeKeys,
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
        let mut plan = Self::new(count, master, context.operator_id())?;
        let mut quantum = super::super::columnar::Quantum::default();
        let mut base = 0;
        for group in rows.groups() {
            for start in (0..group.len()).step_by(64) {
                let end = (start + 64).min(group.len());
                plan.append_block(
                    group.block(start, end),
                    base + start,
                    keys,
                    context,
                    &mut quantum,
                )
                .await?;
            }
            base += group.len();
        }
        Ok(plan)
    }
    fn new(count: usize, master: &MemoryReservation, name: &str) -> Result<Self> {
        let bytes = count
            .checked_mul(size_of::<usize>() + size_of::<u64>())
            .and_then(|bytes| bytes.checked_add(64))
            .ok_or_else(|| super::super::native_lookup::scratch_error("Join retained selection"))?;
        let credit = if count == 0 {
            None
        } else {
            grant(master, bytes, name, false)?
        };
        Ok(Self {
            positions: Vec::with_capacity(count),
            charges: Vec::with_capacity(count),
            bytes: 0,
            _credit: credit,
        })
    }

    async fn append_block(
        &mut self,
        block: AdmittedGroup<'_>,
        base: usize,
        keys: &NativeKeys,
        context: &StreamOperatorContext<'_>,
        quantum: &mut super::super::columnar::Quantum,
    ) -> Result<()> {
        let mut selected = block.retained_bits(0, block.len());
        if selected == 0 {
            return Ok(());
        }
        let charges = PayloadCharges::new(block, selected, context, quantum).await?;
        while selected != 0 {
            let index = selected.trailing_zeros() as usize;
            selected &= selected - 1;
            quantum.step(context, 1, 0).await?;
            let row = block.row(index);
            let charge = charges.state_charge(
                index,
                row.record,
                keys.key(base + index).len(),
                context.operator_id(),
            )?;
            self.positions.push(base + index);
            self.charges.push(charge);
            self.bytes = self.bytes.checked_add(charge).ok_or_else(|| {
                super::super::counter_overflow(context.operator_id(), "state bytes")
            })?;
        }
        Ok(())
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
            let mut chunk = reserve_column_chunk(rows, keys, block, master, context)?;
            for (&position, &charge) in block.iter().zip(charges) {
                let row = bound_row(position, &mut group, &mut group_base, &mut groups);
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

fn reserve_column_chunk(
    rows: &dyn AdmittedRows,
    keys: &NativeKeys,
    block: &[usize],
    master: &MemoryReservation,
    context: &StreamOperatorContext<'_>,
) -> Result<Chunk> {
    let sources = selected_sources(rows, block);
    let key_count = block.len().min(keys.keys.len());
    let bytes = chunk_base_bytes(block.len(), sources, key_count)
        .ok_or_else(|| super::super::native_lookup::scratch_error("Join state chunks"))?;
    let credit = grant(master, bytes, context.operator_id(), false)?;
    Ok(Chunk::reserved_keys(
        block.len(),
        sources,
        key_count,
        credit,
    ))
}

fn selected_sources(rows: &dyn AdmittedRows, block: &[usize]) -> usize {
    let mut sources = 0;
    let mut end = 0;
    for &position in block {
        if position >= end {
            sources += 1;
            end = rows.group_end(position);
        }
    }
    sources
}

fn bound_row<'a>(
    position: usize,
    group: &mut Option<AdmittedGroup<'a>>,
    base: &mut usize,
    groups: &mut GroupIter<'a>,
) -> AdmittedRef<'a> {
    while group.is_some_and(|group| position >= *base + group.len()) {
        *base += group.expect("selected group").len();
        *group = groups.next();
    }
    group
        .expect("selected admission group")
        .row(position - *base)
}
