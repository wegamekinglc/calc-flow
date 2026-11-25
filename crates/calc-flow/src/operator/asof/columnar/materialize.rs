mod charge;

use super::{ColumnarState, RowLocator, Selection};
use crate::Result;
use crate::operator::asof::arrow_error;
use datafusion::arrow::{
    array::{Array, ArrayRef, UInt32Array, new_null_array},
    buffer::{BooleanBuffer, NullBuffer},
    compute::{concat, interleave, take},
    datatypes::{DataType, SchemaRef},
    error::ArrowError,
    record_batch::RecordBatch,
};
use std::{collections::HashMap, convert::AsRef};

enum SideSources<'a> {
    AllNull,
    CompleteRuns(Vec<&'a RecordBatch>),
    Take {
        record: &'a RecordBatch,
        indices: UInt32Array,
    },
    Interleave {
        runs: Vec<u64>,
        positions: Vec<(usize, usize)>,
        has_nulls: bool,
    },
}

fn single_run_indices(
    locators: &[RowLocator],
    nulls: Option<&BooleanBuffer>,
    run_rows: usize,
) -> Option<UInt32Array> {
    let Some(validity) = nulls else {
        return Some(UInt32Array::from_iter_values(
            locators.iter().map(|locator| locator.row),
        ));
    };
    // An out-of-range index keeps unselected payload bytes out of masked output slots.
    let absent = u32::try_from(run_rows).ok()?;
    let rows = locators.iter().enumerate().map(|(i, locator)| {
        if validity.value(i) {
            locator.row
        } else {
            absent
        }
    });
    Some(UInt32Array::from_iter_values_with_nulls(
        rows,
        Some(NullBuffer::new(validity.clone())),
    ))
}

impl ColumnarState {
    pub(in crate::operator::asof) fn materialize(
        &self,
        selection: &Selection,
        schema: &SchemaRef,
        projection: Option<&[(usize, usize)]>,
    ) -> Result<RecordBatch> {
        if selection.is_empty() {
            return Ok(RecordBatch::new_empty(schema.clone()));
        }
        let requests = projection.map_or_else(
            || {
                (0..self.column_count(0))
                    .map(|column| (0, column))
                    .chain((0..self.column_count(1)).map(|column| (1, column)))
                    .collect()
            },
            <[(usize, usize)]>::to_vec,
        );
        if requests.len() != schema.fields().len() {
            return Err(invalid("ASOF projection and output schema lengths differ"));
        }
        let left = self.sources(0, &selection.left, None);
        let right = self.sources(1, &selection.right, Some(&selection.right_validity));
        let columns = requests
            .iter()
            .enumerate()
            .map(|(ordinal, &(side, column))| {
                let data_type = schema.field(ordinal).data_type();
                if column >= self.column_count(side) {
                    return Err(invalid("ASOF projection column is out of bounds"));
                }
                match side {
                    0 => self.materialize_side(0, &left, column, data_type, selection.len()),
                    1 => self.materialize_side(1, &right, column, data_type, selection.len()),
                    _ => Err(invalid("ASOF projection side is invalid")),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        RecordBatch::try_new(schema.clone(), columns).map_err(|error| arrow_error(&error))
    }

    fn sources(
        &self,
        side: usize,
        locators: &[RowLocator],
        validity: Option<&BooleanBuffer>,
    ) -> SideSources<'_> {
        let nulls = validity.filter(|bits| bits.count_set_bits() != locators.len());
        if nulls.is_none()
            && let Some(records) = self.complete_runs(side, locators)
        {
            return SideSources::CompleteRuns(records);
        }
        let mut by_run = HashMap::new();
        let mut runs = Vec::new();
        for (index, locator) in locators.iter().copied().enumerate() {
            if validity.is_some_and(|validity| !validity.value(index)) {
                continue;
            }
            let next = runs.len();
            by_run.entry(locator.run).or_insert_with(|| {
                runs.push(locator.run);
                next
            });
        }
        if runs.is_empty() {
            return SideSources::AllNull;
        }
        if let [run] = runs.as_slice() {
            let record = self
                .run(side, *run)
                .payload_record()
                .expect("selected live ASOF payload");
            if let Some(indices) = single_run_indices(locators, nulls, record.num_rows()) {
                return SideSources::Take { record, indices };
            }
        }
        let has_nulls = nulls.is_some();
        let offset = usize::from(has_nulls);
        let positions = locators
            .iter()
            .copied()
            .enumerate()
            .map(|(index, locator)| {
                if validity.is_some_and(|validity| !validity.value(index)) {
                    (0, 0)
                } else {
                    (by_run[&locator.run] + offset, locator.row as usize)
                }
            })
            .collect();
        SideSources::Interleave {
            runs,
            positions,
            has_nulls,
        }
    }

    // Only forward whole payloads. A small output must not pin a larger source array.
    fn complete_runs(&self, side: usize, mut rows: &[RowLocator]) -> Option<Vec<&RecordBatch>> {
        let mut records = Vec::new();
        let mut previous = None;
        while let Some(first) = rows.first() {
            if first.row != 0
                || first.run < self.first_fresh_run
                || previous.is_some_and(|run| run >= first.run)
            {
                return None;
            }
            let record = self.run(side, first.run).payload_record()?;
            let (selected, rest) = rows.split_at_checked(record.num_rows())?;
            if !selected
                .iter()
                .enumerate()
                .all(|(i, row)| row.run == first.run && row.row as usize == i)
            {
                return None;
            }
            records.push(record);
            previous = Some(first.run);
            rows = rest;
        }
        (!records.is_empty()).then_some(records)
    }

    fn materialize_side(
        &self,
        side: usize,
        sources: &SideSources<'_>,
        column: usize,
        data_type: &DataType,
        rows: usize,
    ) -> Result<ArrayRef> {
        let (runs, positions, has_nulls) = match sources {
            SideSources::AllNull => return Ok(new_null_array(data_type, rows)),
            SideSources::CompleteRuns(records) => {
                if let [record] = records.as_slice() {
                    return Ok(record.column(column).clone());
                }
                let arrays = records
                    .iter()
                    .map(|record| record.column(column).as_ref())
                    .collect::<Vec<_>>();
                return concat(&arrays).map_err(|error| arrow_error(&error));
            }
            SideSources::Take { record, indices } => {
                return take(record.column(column).as_ref(), indices, None)
                    .map_err(|error| arrow_error(&error));
            }
            SideSources::Interleave {
                runs,
                positions,
                has_nulls,
            } => (runs, positions, has_nulls),
        };
        let null = has_nulls.then(|| new_null_array(data_type, 1));
        let null_ref = null.as_ref().map(AsRef::as_ref);
        let arrays = null_ref
            .into_iter()
            .chain(runs.iter().map(|run| {
                self.run(side, *run)
                    .payload
                    .as_ref()
                    .expect("selected live ASOF payload")
                    .column(column)
                    .as_ref()
            }))
            .collect::<Vec<&dyn Array>>();
        interleave(&arrays, positions).map_err(|error| arrow_error(&error))
    }
}

fn invalid(message: &str) -> crate::CalcFlowError {
    arrow_error(&ArrowError::InvalidArgumentError(message.into()))
}
