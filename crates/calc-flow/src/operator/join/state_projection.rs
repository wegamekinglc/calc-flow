//! Immutable retained payload layout, separate from logical admission schemas.

use super::{CompiledJoin, StreamJoinOperator, columnar, operator_error};
use crate::Result;
use datafusion::arrow::datatypes::SchemaRef;
use std::{collections::BTreeSet, sync::Arc};

mod admission;
pub(super) mod checkpoint;
#[cfg(test)]
pub(super) mod tests;

pub(super) struct Layout {
    pub(super) columns: [Vec<usize>; 2],
    pub(super) schemas: [SchemaRef; 2],
    pub(super) compiled: CompiledJoin,
    pub(super) schema_bytes: [Option<usize>; 2],
    pub(super) descriptor: checkpoint::Descriptor,
}

impl Layout {
    pub(super) fn new(operator: &StreamJoinOperator, output: &[usize]) -> Result<Option<Self>> {
        let columns = retained_columns(operator, output);
        if (0..2).all(|side| columns[side].len() == operator.input_schema(side).fields().len()) {
            return Ok(None);
        }
        Self::project(operator, columns).map(Some)
    }

    fn project(operator: &StreamJoinOperator, columns: [Vec<usize>; 2]) -> Result<Self> {
        let schemas = [
            project_schema(operator, &columns, 0)?,
            project_schema(operator, &columns, 1)?,
        ];
        let mut compiled = operator.compiled.clone();
        compiled.left_key_indices = remap(&columns[0], &compiled.left_key_indices);
        compiled.right_key_indices = remap(&columns[1], &compiled.right_key_indices);
        compiled.left_event_time_index =
            physical_index(&columns[0], compiled.left_event_time_index);
        compiled.right_event_time_index =
            physical_index(&columns[1], compiled.right_event_time_index);
        let schema_bytes = [
            columnar::schema_inventory(&schemas[0]),
            columnar::schema_inventory(&schemas[1]),
        ];
        let descriptor = checkpoint::Descriptor::new(operator, &columns, &schemas);
        Ok(Self {
            columns,
            schemas,
            compiled,
            schema_bytes,
            descriptor,
        })
    }

    pub(super) fn output_columns(&self, columns: &[usize], logical_left: usize) -> Vec<usize> {
        columns
            .iter()
            .map(|&column| {
                let side = usize::from(column >= logical_left);
                let index = if side == 0 {
                    column
                } else {
                    column - logical_left
                };
                physical_index(&self.columns[side], index)
                    + if side == 0 { 0 } else { self.columns[0].len() }
            })
            .collect()
    }
}

fn retained_columns(operator: &StreamJoinOperator, output: &[usize]) -> [Vec<usize>; 2] {
    let left_width = operator.input_schema(0).fields().len();
    std::array::from_fn(|side| {
        let mut columns = BTreeSet::from([if side == 0 {
            operator.compiled.left_event_time_index
        } else {
            operator.compiled.right_event_time_index
        }]);
        let keys = if side == 0 {
            &operator.compiled.left_key_indices
        } else {
            &operator.compiled.right_key_indices
        };
        columns.extend(keys);
        columns.extend(
            output
                .iter()
                .filter(|&&index| usize::from(index >= left_width) == side)
                .map(|&index| if side == 0 { index } else { index - left_width }),
        );
        columns.into_iter().collect()
    })
}

fn physical_index(columns: &[usize], logical: usize) -> usize {
    columns
        .binary_search(&logical)
        .expect("validated retained Join dependency")
}

fn remap(columns: &[usize], indices: &[usize]) -> Vec<usize> {
    indices
        .iter()
        .map(|&index| physical_index(columns, index))
        .collect()
}

fn project_schema(
    operator: &StreamJoinOperator,
    columns: &[Vec<usize>; 2],
    side: usize,
) -> Result<SchemaRef> {
    operator
        .input_schema(side)
        .project(&columns[side])
        .map(Arc::new)
        .map_err(|error| {
            operator_error(
                &operator.name,
                &format!("retained projection failed: {error}"),
            )
        })
}

impl StreamJoinOperator {
    pub(super) fn retained_schema(&self, side: usize) -> &SchemaRef {
        self.payload_projection
            .as_ref()
            .map_or_else(|| self.input_schema(side), |layout| &layout.schemas[side])
    }

    pub(super) fn retained_compiled(&self) -> &CompiledJoin {
        self.payload_projection
            .as_ref()
            .map_or(&self.compiled, |layout| &layout.compiled)
    }
}
