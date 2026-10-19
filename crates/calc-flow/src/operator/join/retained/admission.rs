//! Preparation borrows the existing native key-ID column. Only selection and
//! logical charges survive until the owned chunks have been filled.

use std::{ops::Range, sync::Arc};

use super::super::{
    admission_charge::PayloadCharges,
    admitted::{AdmittedGroup, AdmittedRef, AdmittedRows, GroupIter},
    columnar::{COPY_BYTES, COPY_VISITS},
    native_lookup::NativeKeys,
};
use super::{
    Append, CHUNK_ROWS, Chunk, MemoryReservation, Result, StateRows, StreamOperatorContext,
    chunk_base_bytes, grant, reserve_references,
};

pub(in crate::operator::join) struct ColumnarPlan {
    single: Selection,
    parts: Vec<Selection>,
    pub(in crate::operator::join) bytes: u64,
}

struct Selection {
    positions: Vec<usize>,
    charges: Vec<u64>,
    end: usize,
    bytes: u64,
    _credit: Option<Arc<MemoryReservation>>,
    controls: Option<Arc<MemoryReservation>>,
}

impl Selection {
    fn new(count: usize, credit: Option<Arc<MemoryReservation>>) -> Self {
        Self {
            positions: Vec::with_capacity(count),
            charges: Vec::with_capacity(count),
            end: 0,
            bytes: 0,
            _credit: credit,
            controls: None,
        }
    }
}

#[derive(Clone)]
struct SelectionIter<'a> {
    parts: &'a [Selection],
    part: usize,
    offset: usize,
    remaining: usize,
}

impl Iterator for SelectionIter<'_> {
    type Item = (usize, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        while self.offset == self.parts[self.part].positions.len() {
            self.part += 1;
            self.offset = 0;
        }
        let part = &self.parts[self.part];
        let row = (part.positions[self.offset], part.charges[self.offset]);
        self.offset += 1;
        self.remaining -= 1;
        Some(row)
    }
}

impl ColumnarPlan {
    pub(in crate::operator::join) fn len(&self) -> usize {
        self.selections().last().map_or(0, |part| part.end)
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
        plan.single.end = plan.single.positions.len();
        Ok(plan)
    }

    fn selections(&self) -> &[Selection] {
        if self.parts.is_empty() {
            std::slice::from_ref(&self.single)
        } else {
            &self.parts
        }
    }

    fn range(&self, range: Range<usize>) -> SelectionIter<'_> {
        let parts = self.selections();
        let part = parts.partition_point(|part| part.end <= range.start);
        let base = if part == 0 { 0 } else { parts[part - 1].end };
        SelectionIter {
            parts,
            part,
            offset: range.start - base,
            remaining: range.len(),
        }
    }

    fn reserve_credit(
        count: usize,
        master: &MemoryReservation,
        name: &str,
    ) -> Result<Option<Arc<MemoryReservation>>> {
        let bytes = count
            .checked_mul(size_of::<usize>() + size_of::<u64>())
            .and_then(|bytes| bytes.checked_add(64))
            .ok_or_else(|| super::super::native_lookup::scratch_error("Join retained selection"))?;
        if count == 0 {
            Ok(None)
        } else {
            grant(master, bytes, name, false).map(|credit| credit.map(Arc::new))
        }
    }

    fn new(count: usize, master: &MemoryReservation, name: &str) -> Result<Self> {
        let credit = Self::reserve_credit(count, master, name)?;
        Ok(Self {
            single: Selection::new(count, credit),
            parts: Vec::new(),
            bytes: 0,
        })
    }

    async fn into_serial(
        self,
        master: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Self> {
        if self.parts.is_empty() {
            return Ok(self);
        }
        context.check_cancelled()?;
        let mut serial = Self::new(self.len(), master, context.operator_id())?;
        let mut quantum = super::super::columnar::Quantum::default();
        let width = size_of::<usize>() + size_of::<u64>();
        let copy_rows = COPY_VISITS.min(COPY_BYTES / width);
        // Preserve the frozen logical charges after payload projection.
        for part in &self.parts {
            for (positions, charges) in part
                .positions
                .chunks(copy_rows)
                .zip(part.charges.chunks(copy_rows))
            {
                quantum
                    .step(context, positions.len(), positions.len() * width)
                    .await?;
                serial.single.positions.extend_from_slice(positions);
                serial.single.charges.extend_from_slice(charges);
            }
        }
        serial.single.end = self.len();
        serial.bytes = self.bytes;
        drop(self);
        context.check_cancelled()?;
        Ok(serial)
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
            self.single.positions.push(base + index);
            self.single.charges.push(charge);
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
        for start in (0..plan.len()).step_by(CHUNK_ROWS) {
            let block = plan.range(start..(start + CHUNK_ROWS).min(plan.len()));
            let mut chunk = reserve_column_chunk(rows, keys, block.clone(), master, context)?;
            for (position, charge) in block {
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
    block: SelectionIter<'_>,
    master: &MemoryReservation,
    context: &StreamOperatorContext<'_>,
) -> Result<Chunk> {
    let count = block.remaining;
    let sources = selected_sources(rows, block);
    let key_count = count.min(keys.keys.len());
    let bytes = chunk_base_bytes(count, sources, key_count)
        .ok_or_else(|| super::super::native_lookup::scratch_error("Join state chunks"))?;
    let credit = grant(master, bytes, context.operator_id(), false)?;
    Ok(Chunk::reserved_keys(count, sources, key_count, credit))
}

fn selected_sources(rows: &dyn AdmittedRows, mut block: SelectionIter<'_>) -> usize {
    let mut sources = 0;
    while let Some((position, _)) = block.next() {
        sources += 1;
        block.skip_before(rows.group_end(position));
    }
    sources
}

impl SelectionIter<'_> {
    fn skip_before(&mut self, position: usize) {
        while self.remaining > 0 {
            let part = &self.parts[self.part];
            let end = part.positions.len().min(self.offset + self.remaining);
            let skipped =
                part.positions[self.offset..end].partition_point(|value| *value < position);
            self.offset += skipped;
            self.remaining -= skipped;
            if self.offset == part.positions.len() && self.remaining > 0 {
                self.part += 1;
                self.offset = 0;
            } else {
                break;
            }
        }
    }
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

pub(super) mod parallel;
