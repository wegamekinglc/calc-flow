//! Exact columnar equivalent of the existing one-row Arrow slice charge.
//!
//! Nested/dictionary/view arrays keep the generic Arrow measurement path.
//! Primitive and byte-array columns do not allocate a Batch per row.

use datafusion::arrow::{
    array::{BinaryArray, LargeBinaryArray, LargeStringArray, StringArray},
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

impl RowCosts {
    pub(super) fn try_new(record: &RecordBatch) -> Result<Option<Self>> {
        let mut fixed = 0_usize;
        let mut variable = Vec::new();
        let mut validity = Vec::new();
        for column in record.columns() {
            let width = if let Some(width) = column.data_type().primitive_width() {
                width
            } else {
                match column.data_type() {
                    DataType::Null => 0,
                    DataType::Boolean => 1,
                    DataType::Utf8 => {
                        let array = column.as_any().downcast_ref::<StringArray>().unwrap();
                        variable.push(Offsets::Narrow(array.value_offsets()));
                        4
                    }
                    DataType::Binary => {
                        let array = column.as_any().downcast_ref::<BinaryArray>().unwrap();
                        variable.push(Offsets::Narrow(array.value_offsets()));
                        4
                    }
                    DataType::LargeUtf8 => {
                        let array = column.as_any().downcast_ref::<LargeStringArray>().unwrap();
                        variable.push(Offsets::Wide(array.value_offsets()));
                        8
                    }
                    DataType::LargeBinary => {
                        let array = column.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
                        variable.push(Offsets::Wide(array.value_offsets()));
                        8
                    }
                    _ => return Ok(None),
                }
            };
            fixed = checked_accumulate(fixed, width, "batch")?;
            if let Some(nulls) = column.nulls() {
                validity.push(nulls);
            }
        }
        if variable.is_empty() && validity.is_empty() {
            return Ok(Some(Self::Fixed(fixed)));
        }
        let mut rows = vec![fixed; record.num_rows()];
        for offsets in variable {
            match offsets {
                Offsets::Narrow(offsets) => {
                    for (cost, pair) in rows.iter_mut().zip(offsets.windows(2)) {
                        let width = usize::try_from(pair[1] - pair[0])
                            .expect("validated Arrow offsets are nonnegative");
                        *cost = checked_accumulate(*cost, width, "batch")?;
                    }
                }
                Offsets::Wide(offsets) => {
                    for (cost, pair) in rows.iter_mut().zip(offsets.windows(2)) {
                        let width = usize::try_from(pair[1] - pair[0])
                            .expect("validated Arrow offsets fit the address space");
                        *cost = checked_accumulate(*cost, width, "batch")?;
                    }
                }
            }
        }
        // Arrow removes a zero-null bitmap when constructing the ArrayData
        // of a one-row slice. Only a null row incurs the one-byte charge.
        for nulls in validity {
            for (cost, valid) in rows.iter_mut().zip(nulls.iter()) {
                *cost = checked_accumulate(*cost, usize::from(!valid), "batch")?;
            }
        }
        Ok(Some(Self::Variable(rows)))
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
