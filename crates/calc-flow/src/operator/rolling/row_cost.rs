//! Exact columnar equivalent of the existing one-row Arrow slice charge.
//!
//! Nested/dictionary/view arrays keep the generic Arrow measurement path.
//! Primitive and byte-array columns do not allocate a Batch per row.

use datafusion::arrow::{
    array::{Array, BinaryArray, LargeBinaryArray, LargeStringArray, OffsetSizeTrait, StringArray},
    buffer::NullBuffer,
    datatypes::DataType,
    record_batch::RecordBatch,
};

use crate::{Result, batch::checked_accumulate};

pub(super) enum RowCosts {
    Fixed(usize),
    Variable(Vec<usize>),
}

enum Offsets<'a> {
    Narrow(&'a [i32]),
    Wide(&'a [i64]),
}

fn fixed_width(data_type: &DataType) -> Option<usize> {
    data_type.primitive_width().or(match data_type {
        DataType::Null => Some(0),
        DataType::Boolean => Some(1),
        DataType::Utf8 | DataType::Binary => Some(4),
        DataType::LargeUtf8 | DataType::LargeBinary => Some(8),
        _ => None,
    })
}

fn variable_offsets(column: &dyn Array) -> Option<Offsets<'_>> {
    match column.data_type() {
        DataType::Utf8 => Some(Offsets::Narrow(
            column
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value_offsets(),
        )),
        DataType::Binary => Some(Offsets::Narrow(
            column
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value_offsets(),
        )),
        DataType::LargeUtf8 => Some(Offsets::Wide(
            column
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .unwrap()
                .value_offsets(),
        )),
        DataType::LargeBinary => Some(Offsets::Wide(
            column
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap()
                .value_offsets(),
        )),
        _ => None,
    }
}

fn add_offset_charges<O: OffsetSizeTrait>(rows: &mut [usize], offsets: &[O]) -> Result<()> {
    for (cost, pair) in rows.iter_mut().zip(offsets.windows(2)) {
        let width = (pair[1] - pair[0])
            .to_usize()
            .expect("validated Arrow offsets fit the address space");
        *cost = checked_accumulate(*cost, width, "batch")?;
    }
    Ok(())
}

impl Offsets<'_> {
    fn add_charges(&self, rows: &mut [usize]) -> Result<()> {
        match self {
            Self::Narrow(offsets) => add_offset_charges(rows, offsets),
            Self::Wide(offsets) => add_offset_charges(rows, offsets),
        }
    }
}

#[derive(Default)]
struct ColumnCharges<'a> {
    fixed: usize,
    variable: Vec<Offsets<'a>>,
    validity: Vec<&'a NullBuffer>,
}

impl<'a> ColumnCharges<'a> {
    fn read(record: &'a RecordBatch) -> Result<Option<Self>> {
        let mut charges = Self::default();
        for column in record.columns() {
            let Some(width) = fixed_width(column.data_type()) else {
                return Ok(None);
            };
            charges.fixed = checked_accumulate(charges.fixed, width, "batch")?;
            charges.variable.extend(variable_offsets(column.as_ref()));
            charges.validity.extend(column.nulls());
        }
        Ok(Some(charges))
    }

    fn add_null_charges(&self, rows: &mut [usize]) -> Result<()> {
        // Arrow removes a zero-null bitmap when constructing ArrayData for a
        // one-row slice. Only a null row incurs the one-byte charge.
        for nulls in &self.validity {
            for (cost, valid) in rows.iter_mut().zip(nulls.iter()) {
                *cost = checked_accumulate(*cost, usize::from(!valid), "batch")?;
            }
        }
        Ok(())
    }

    fn materialize(&self, row_count: usize) -> Result<RowCosts> {
        if self.variable.is_empty() && self.validity.is_empty() {
            return Ok(RowCosts::Fixed(self.fixed));
        }
        let mut rows = vec![self.fixed; row_count];
        for offsets in &self.variable {
            offsets.add_charges(&mut rows)?;
        }
        self.add_null_charges(&mut rows)?;
        Ok(RowCosts::Variable(rows))
    }
}

impl RowCosts {
    pub(super) fn try_new(record: &RecordBatch) -> Result<Option<Self>> {
        let Some(charges) = ColumnCharges::read(record)? else {
            return Ok(None);
        };
        charges.materialize(record.num_rows()).map(Some)
    }

    pub(super) fn get(&self, row: usize) -> usize {
        match self {
            Self::Fixed(cost) => *cost,
            Self::Variable(costs) => costs[row],
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, NullArray};

    use super::*;
    use crate::{Batch, BatchMetadata};

    #[test]
    fn columnar_row_charges_match_arrow_for_slices_nulls_and_variable_values() {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                None,
                Some(3.0),
                Some(4.0),
            ])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                None,
                Some(false),
                Some(true),
            ])),
            Arc::new(StringArray::from(vec![
                Some("wide"),
                None,
                Some("中文"),
                Some(""),
            ])),
            Arc::new(LargeStringArray::from(vec![
                Some("a"),
                Some("bc"),
                None,
                Some("def"),
            ])),
            Arc::new(BinaryArray::from(vec![
                Some(b"a".as_slice()),
                None,
                Some(b"bc"),
                Some(b""),
            ])),
            Arc::new(LargeBinaryArray::from(vec![
                Some(b"ab".as_slice()),
                Some(b""),
                None,
                Some(b"x"),
            ])),
            Arc::new(NullArray::new(4)),
        ];
        let record = RecordBatch::try_from_iter(
            columns
                .into_iter()
                .enumerate()
                .map(|(index, array)| (format!("c{index}"), array)),
        )
        .unwrap();
        for record in [record.clone(), record.slice(1, 3), record.slice(2, 1)] {
            let costs = RowCosts::try_new(&record).unwrap().unwrap();
            for index in 0..record.num_rows() {
                let oracle = Batch::table(vec![record.slice(index, 1)], BatchMetadata::default())
                    .unwrap()
                    .estimated_bytes()
                    .unwrap();
                assert_eq!(costs.get(index), oracle, "row {index}");
            }
        }
    }

    #[test]
    fn fixed_width_charges_do_not_depend_on_row_count() {
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
        )])
        .unwrap();
        assert!(matches!(
            RowCosts::try_new(&record).unwrap(),
            Some(RowCosts::Fixed(8))
        ));
    }
}
