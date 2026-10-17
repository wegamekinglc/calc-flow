//! Frozen V1 payload fees, evaluated columnwise in bounded stack storage.

use super::{
    STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1, admitted::AdmittedGroup, charge_overflow,
    charge_type_mismatch, columnar::Quantum, counter_overflow, fixed_cell_charge,
    fixed_size_binary_charge, state_columns_charge_with_key,
};
use crate::{Result, StreamOperatorContext};
use datafusion::arrow::{
    array::{
        Array, BinaryArray, BinaryViewArray, LargeBinaryArray, LargeStringArray, StringArray,
        StringViewArray,
    },
    datatypes::DataType,
    util::bit_chunk_iterator::BitChunks,
};
use std::ops::Range;

mod nested;

const BLOCK: usize = 64;

/// Errors are replayed with the V1 oracle at their original row, after its key.
/// No speculative column error can hide an earlier row/key failure.
pub(super) struct PayloadCharges {
    positions: [usize; BLOCK],
    charges: [u64; BLOCK],
    selected: u64,
    failed: u64,
    len: usize,
}

impl PayloadCharges {
    /// `rows` covers exactly one bounded block; `selected` carries that
    /// block's retained bits, and only those rows are positioned, charged,
    /// and later read back.
    pub(super) async fn new(
        rows: AdmittedGroup<'_>,
        selected: u64,
        context: &StreamOperatorContext<'_>,
        quantum: &mut Quantum,
    ) -> Result<Self> {
        let mut result = Self {
            positions: [0; BLOCK],
            charges: [0; BLOCK],
            selected: 0,
            failed: 0,
            len: rows.len(),
        };
        assert!(result.len <= BLOCK);
        debug_assert_eq!(selected & !low_bits(result.len), 0);
        result.selected = selected;
        if selected == 0 {
            return Ok(result);
        }
        for (offset, (position, retain)) in rows.key_offsets().enumerate() {
            if retain {
                result.positions[offset] = position;
            }
        }
        let selected_rows = selected.count_ones() as usize;
        let mut constant = Some(0_u64);
        for array in rows.row(0).record.columns() {
            quantum.step(context, selected_rows, 0).await?;
            result.add_column(array.as_ref(), &mut constant);
        }
        for index in 0..result.len {
            result.add(index, constant);
        }
        context.check_cancelled()?;
        Ok(result)
    }

    fn add_column(&mut self, array: &dyn Array, constant: &mut Option<u64>) {
        #[cfg(test)]
        super::note_join_work(|work| work.charge_column_resolutions += 1);
        let Ok(column) = Column::new(array) else {
            self.failed |= self.selected;
            return;
        };
        let valid = self.valid(array) & self.selected;
        if let Column::Fixed(width) = column {
            *constant = constant.and_then(|value| value.checked_add(1));
            if valid == self.selected {
                *constant = constant.and_then(|value| value.checked_add(width));
            } else {
                visit_bits(valid, |index| self.add(index, Some(width)));
            }
            return;
        }
        *constant = constant.and_then(|value| value.checked_add(1));
        visit_bits(valid, |index| {
            let value = column.non_null(self.positions[index]).ok();
            self.add(index, value);
        });
    }

    fn valid(&self, array: &dyn Array) -> u64 {
        // Only selected rows are charged, so the dense bitmap word applies
        // when the whole block survives and its positions stay contiguous.
        if self.selected == low_bits(self.len)
            && self.positions[0].checked_add(self.len - 1) == Some(self.positions[self.len - 1])
        {
            return valid_bits(array, self.positions[0], self.len);
        }
        #[cfg(test)]
        super::note_join_work(|work| {
            work.charge_sparse_validity_visits += self.selected.count_ones() as usize;
        });
        let mut bits = 0;
        visit_bits(self.selected, |index| {
            bits |= u64::from(array.is_valid(self.positions[index])) << index;
        });
        bits
    }

    fn add(&mut self, index: usize, value: Option<u64>) {
        if let Some(value) = value.and_then(|value| self.charges[index].checked_add(value)) {
            self.charges[index] = value;
        } else {
            self.failed |= 1_u64 << index;
        }
    }

    pub(super) fn state_charge(
        &self,
        index: usize,
        row: super::PayloadRef<'_>,
        key_bytes: usize,
        name: &str,
    ) -> Result<u64> {
        if self.failed & (1_u64 << index) != 0 {
            #[cfg(test)]
            super::note_join_work(|work| work.charge_scalar_fallbacks += 1);
            return state_columns_charge_with_key(row.columns(), row.offset(), key_bytes, name);
        }
        let key_bytes =
            u64::try_from(key_bytes).map_err(|_| counter_overflow(name, "encoded key"))?;
        STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1
            .checked_add(key_bytes)
            .and_then(|value| value.checked_add(16))
            .and_then(|value| value.checked_add(self.charges[index]))
            .ok_or_else(|| counter_overflow(name, "state row charge"))
    }
}

#[derive(Clone, Copy)]
enum Column<'a> {
    Fixed(u64),
    Offsets32(&'a [i32], u64),
    Offsets64(&'a [i64], u64),
    Views(&'a [u128]),
    Nested(&'a dyn Array),
}

impl<'a> Column<'a> {
    fn new(array: &'a dyn Array) -> Result<Self> {
        if let Some(width) = fixed_cell_charge(array.data_type()) {
            return Ok(Self::Fixed(width));
        }
        if let DataType::FixedSizeBinary(width) = array.data_type() {
            return fixed_size_binary_charge(*width).map(Self::Fixed);
        }
        Self::variable(array)
    }

    fn variable(array: &'a dyn Array) -> Result<Self> {
        Ok(match array.data_type() {
            DataType::Utf8 => Self::Offsets32(typed::<StringArray>(array)?.value_offsets(), 4),
            DataType::Binary => Self::Offsets32(typed::<BinaryArray>(array)?.value_offsets(), 4),
            DataType::LargeUtf8 => {
                Self::Offsets64(typed::<LargeStringArray>(array)?.value_offsets(), 8)
            }
            DataType::LargeBinary => {
                Self::Offsets64(typed::<LargeBinaryArray>(array)?.value_offsets(), 8)
            }
            DataType::Utf8View => Self::Views(typed::<StringViewArray>(array)?.views()),
            DataType::BinaryView => Self::Views(typed::<BinaryViewArray>(array)?.views()),
            _ => Self::Nested(array),
        })
    }

    /// Excludes this cell's outer validity byte.
    fn non_null(self, row: usize) -> Result<u64> {
        let cell = Range {
            start: row,
            end: row + 1,
        };
        match self {
            Self::Fixed(width) => Ok(width),
            Self::Offsets32(offsets, prefix) => offset_charge(offsets, cell, prefix),
            Self::Offsets64(offsets, prefix) => offset_charge(offsets, cell, prefix),
            Self::Views(views) => {
                #[cfg(test)]
                super::note_join_work(|work| work.charge_offset_reads += 1);
                add(
                    16,
                    u64::try_from(views[row] & u128::from(u32::MAX)).unwrap(),
                )
            }
            Self::Nested(array) => nested::non_null(array, row),
        }
    }
}

fn typed<T: Array + 'static>(array: &dyn Array) -> Result<&T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| charge_type_mismatch("columnar payload"))
}

fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| charge_overflow("columnar payload"))
}

fn multiply(left: u64, right: u64) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| charge_overflow("columnar payload"))
}

fn offset_charge<O: Copy + Into<i64>>(
    offsets: &[O],
    range: Range<usize>,
    prefix: u64,
) -> Result<u64> {
    #[cfg(test)]
    super::note_join_work(|work| work.charge_offset_reads += 2);
    let length = offsets[range.end].into() - offsets[range.start].into();
    add(
        multiply(prefix, range.len() as u64)?,
        u64::try_from(length).map_err(|_| charge_overflow("Arrow offset"))?,
    )
}

fn valid_bits(array: &dyn Array, start: usize, length: usize) -> u64 {
    #[cfg(test)]
    super::note_join_work(|work| work.charge_bitmap_words += 1);
    array.nulls().map_or_else(
        || low_bits(length),
        |nulls| {
            let bits = nulls.inner();
            BitChunks::new(bits.values(), bits.offset() + start, length)
                .iter_padded()
                .next()
                .unwrap_or(0)
        },
    )
}

fn low_bits(length: usize) -> u64 {
    if length == 0 {
        0
    } else {
        u64::MAX >> (BLOCK - length)
    }
}

fn visit_bits(mut bits: u64, mut visit: impl FnMut(usize)) {
    while bits != 0 {
        visit(bits.trailing_zeros() as usize);
        bits &= bits - 1;
    }
}

fn cell_sum(array: &dyn Array, row: usize) -> Result<u64> {
    if array.is_null(row) {
        return Ok(1);
    }
    add(1, Column::new(array)?.non_null(row)?)
}

/// Fixed children use bitmap population counts; variable children use offset
/// differences over valid runs. No Arrow cell slices or child fee vectors.
fn range_sum(array: &dyn Array, range: Range<usize>) -> Result<u64> {
    #[cfg(test)]
    super::note_join_work(|work| work.charge_range_sums += 1);
    let column = Column::new(array)?;
    let total = range.len() as u64;
    let count = array.nulls().map_or(range.len(), |nulls| {
        nulls
            .inner()
            .inner()
            .count_set_bits_offset(nulls.offset() + range.start, range.len())
    });
    if let Column::Fixed(width) = column {
        return add(total, multiply(width, count as u64)?);
    }
    if count == range.len() {
        match column {
            Column::Offsets32(offsets, prefix) => {
                return add(total, offset_charge(offsets, range, prefix)?);
            }
            Column::Offsets64(offsets, prefix) => {
                return add(total, offset_charge(offsets, range, prefix)?);
            }
            _ => {}
        }
    }
    sum_valid_ranges(array, column, range, total)
}

fn sum_valid_ranges(
    array: &dyn Array,
    column: Column<'_>,
    range: Range<usize>,
    mut total: u64,
) -> Result<u64> {
    for start in range.clone().step_by(BLOCK) {
        let length = (range.end - start).min(BLOCK);
        let valid = valid_bits(array, start, length);
        let value = match column {
            Column::Offsets32(offsets, prefix) => offset_runs(offsets, start, valid, prefix)?,
            Column::Offsets64(offsets, prefix) => offset_runs(offsets, start, valid, prefix)?,
            _ => selected_sum(column, start, valid)?,
        };
        total = add(total, value)?;
    }
    Ok(total)
}

fn selected_sum(column: Column<'_>, start: usize, mut valid: u64) -> Result<u64> {
    let mut total = 0;
    while valid != 0 {
        total = add(
            total,
            column.non_null(start + valid.trailing_zeros() as usize)?,
        )?;
        valid &= valid - 1;
    }
    Ok(total)
}

fn offset_runs<O: Copy + Into<i64>>(
    offsets: &[O],
    start: usize,
    mut valid: u64,
    prefix: u64,
) -> Result<u64> {
    let mut total = 0;
    while valid != 0 {
        let first = valid.trailing_zeros() as usize;
        let length = (valid >> first).trailing_ones() as usize;
        total = add(
            total,
            offset_charge(offsets, start + first..start + first + length, prefix)?,
        )?;
        valid &= !(low_bits(length) << first);
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::{
        array::{Int64Array, StringArray, make_array},
        buffer::NullBuffer,
    };

    #[test]
    fn test_columnar_charge_child_prefix_and_bitmap_sums_allocate_nothing() {
        let strings = StringArray::from(vec!["payload"; 1024]);
        super::super::reset_join_work();
        let allocation = allocation_counter::measure(|| {
            assert_eq!(range_sum(&strings, 3..1003).unwrap(), 12_000);
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(super::super::join_work().charge_offset_reads, 2);
        assert_eq!(super::super::join_work().scalar_charge_cells, 0);
        let strings = make_array(
            strings
                .to_data()
                .into_builder()
                .nulls(Some(
                    (0..1024).map(|row| row % 3 != 0).collect::<NullBuffer>(),
                ))
                .build()
                .unwrap(),
        );
        let integers = (0..1024)
            .map(|row| (row % 3 != 0).then_some(1))
            .collect::<Int64Array>();
        for array in [strings.as_ref(), &integers as &dyn Array] {
            let expected = (3..1003)
                .map(|row| super::super::logical_cell_charge(array, row).unwrap())
                .sum::<u64>();
            super::super::reset_join_work();
            let allocation = allocation_counter::measure(|| {
                assert_eq!(range_sum(array, 3..1003).unwrap(), expected);
            });
            assert_eq!(allocation.count_total, 0);
            assert_eq!(super::super::join_work().scalar_charge_cells, 0);
        }
    }
}
