mod charge;

use super::{ColumnarState, RowLocator, Selection};
use crate::Result;
use crate::operator::asof::arrow_error;
use datafusion::arrow::{
    array::{Array, ArrayRef, UInt32Array, new_null_array},
    buffer::BooleanBuffer,
    compute::{interleave, take},
    datatypes::{DataType, SchemaRef},
    error::ArrowError,
    record_batch::RecordBatch,
};
use std::{collections::HashMap, convert::AsRef};

struct SideSources {
    runs: Vec<u64>,
    positions: Vec<(usize, usize)>,
    has_nulls: bool,
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
        let left = Self::sources(&selection.left, None);
        let right = Self::sources(&selection.right, Some(&selection.right_validity));
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

    fn sources(locators: &[RowLocator], validity: Option<&BooleanBuffer>) -> SideSources {
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
        let has_nulls =
            validity.is_some_and(|validity| validity.count_set_bits() != locators.len());
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
        SideSources {
            runs,
            positions,
            has_nulls,
        }
    }

    fn materialize_side(
        &self,
        side: usize,
        sources: &SideSources,
        column: usize,
        data_type: &DataType,
        rows: usize,
    ) -> Result<ArrayRef> {
        if sources.runs.is_empty() {
            return Ok(new_null_array(data_type, rows));
        }
        if !sources.has_nulls && sources.runs.len() == 1 {
            let run = self.run(side, sources.runs[0]);
            let record = run.payload.as_ref().expect("selected live ASOF payload");
            let positions = sources
                .positions
                .iter()
                .map(|position| {
                    u32::try_from(position.1).map_err(|_| invalid("ASOF source row exceeds UInt32"))
                })
                .collect::<Result<Vec<_>>>()?;
            let indices = UInt32Array::from(positions);
            return take(record.column(column).as_ref(), &indices, None)
                .map_err(|error| arrow_error(&error));
        }
        let null = sources.has_nulls.then(|| new_null_array(data_type, 1));
        let null_ref = null.as_ref().map(AsRef::as_ref);
        let arrays = null_ref
            .into_iter()
            .chain(sources.runs.iter().map(|run| {
                self.run(side, *run)
                    .payload
                    .as_ref()
                    .expect("selected live ASOF payload")
                    .column(column)
                    .as_ref()
            }))
            .collect::<Vec<&dyn Array>>();
        interleave(&arrays, &sources.positions).map_err(|error| arrow_error(&error))
    }
}

fn invalid(message: &str) -> crate::CalcFlowError {
    arrow_error(&ArrowError::InvalidArgumentError(message.into()))
}
