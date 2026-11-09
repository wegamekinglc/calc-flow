use datafusion::arrow::{
    array::{Array, BinaryArray, UInt8Array, UInt64Array},
    datatypes::Schema,
    record_batch::RecordBatch,
};

use super::super::scalar_materialization_plan::{scalar_clone_requirement, scalar_requirement};
use super::{Check, CompiledRollingSpec, Result, add, census::Census, invalid};

pub(super) struct Source<'a> {
    pub(super) record: &'a RecordBatch,
    pub(super) input: &'a Schema,
    pub(super) compiled: &'a CompiledRollingSpec,
    pub(super) counts: Census,
    kinds: &'a UInt8Array,
    entities: &'a UInt64Array,
    positions: &'a UInt64Array,
}
impl<'a> Source<'a> {
    pub(super) fn new(
        record: &'a RecordBatch,
        input: &'a Schema,
        compiled: &'a CompiledRollingSpec,
        counts: Census,
    ) -> Result<Self> {
        mode(record, input, compiled)?;
        Ok(Self {
            record,
            input,
            compiled,
            counts,
            kinds: typed(record, 0)?,
            entities: typed(record, 1)?,
            positions: typed(record, 2)?,
        })
    }
    pub(super) fn kind(&self, row: usize) -> u8 {
        self.kinds.value(row)
    }
    pub(super) fn initialized(&self, row: usize) -> bool {
        self.positions.value(row) > 0
    }
    pub(super) fn entity(&self, row: usize) -> Result<usize> {
        usize::try_from(self.entities.value(row)).map_err(|_| invalid())
    }
    pub(super) fn cell(&self, row: usize, index: usize, check: Check<'_>) -> Result<(u64, u64)> {
        scalar_requirement(self.record.column(index + 3).as_ref(), row, check)
    }
    pub(super) fn cloned(&self, row: usize, index: usize, check: Check<'_>) -> Result<u64> {
        scalar_clone_requirement(self.record.column(index + 3).as_ref(), row, check)
    }
    pub(super) fn effective(&self, row: usize, index: usize, check: Check<'_>) -> Result<u64> {
        let partition = columns::contains(
            self.compiled
                .partition_columns
                .iter()
                .map(|column| column.index),
            index,
            check,
        )?;
        let source = if partition { self.entity(row)? } else { row };
        self.cloned(source, index, check)
    }
    pub(super) fn partition_heap(&self, row: usize, check: Check<'_>) -> Result<u64> {
        let entity = self.entity(row)?;
        self.compiled
            .partition_columns
            .iter()
            .try_fold(0, |sum, column| {
                check()?;
                add(sum, self.cloned(entity, column.index, check)?)
            })
    }
    pub(super) fn sequence_heap(&self, row: usize, check: Check<'_>) -> Result<u64> {
        self.compiled
            .sequence_columns
            .iter()
            .try_fold(0, |sum, column| {
                check()?;
                add(sum, self.effective(row, column.index, check)?)
            })
    }
    pub(super) fn row_clone(&self, row: usize, check: Check<'_>) -> Result<u64> {
        (0..self.input.fields().len()).try_fold(0, |sum, index| {
            check()?;
            add(sum, self.cloned(row, index, check)?)
        })
    }
    pub(super) fn summary(&self, row: usize) -> Result<Option<&[u8]>> {
        let array: &BinaryArray = typed(self.record, self.input.fields().len() + 6)?;
        Ok((!array.is_null(row)).then(|| array.value(row)))
    }
}
fn mode(record: &RecordBatch, input: &Schema, compiled: &CompiledRollingSpec) -> Result<()> {
    if compiled.state_layout_version != 5 || compiled.indicator_groups.is_empty() {
        return Err(invalid());
    }
    if input.fields().len().checked_add(8) != Some(record.num_columns()) {
        return Err(invalid());
    }
    Ok(())
}
fn typed<T: 'static>(record: &RecordBatch, index: usize) -> Result<&T> {
    record
        .columns()
        .get(index)
        .and_then(|column| column.as_any().downcast_ref())
        .ok_or_else(invalid)
}

mod columns;
