use std::sync::Arc;

use datafusion::arrow::{
    datatypes::{Schema, SchemaRef},
    record_batch::{RecordBatch, RecordBatchOptions},
};

use crate::{CalcFlowError, Result};

pub(crate) struct RetainedColumns {
    logical: SchemaRef,
    physical: SchemaRef,
    ordinals: Vec<usize>,
    inverse: Vec<Option<usize>>,
}

impl RetainedColumns {
    pub(crate) fn try_new(logical: SchemaRef, required: &[usize]) -> Result<Self> {
        if required.windows(2).any(|pair| pair[0] >= pair[1])
            || required
                .iter()
                .any(|&index| index >= logical.fields().len())
        {
            return Err(mapping_error(
                "retained ordinals must be sorted, distinct and in range",
            ));
        }
        let mut inverse = vec![None; logical.fields().len()];
        let fields = required
            .iter()
            .enumerate()
            .map(|(physical, &original)| {
                inverse[original] = Some(physical);
                Arc::clone(&logical.fields()[original])
            })
            .collect::<Vec<_>>();
        let physical = Arc::new(Schema::new_with_metadata(
            fields,
            logical.metadata().clone(),
        ));
        Ok(Self {
            logical,
            physical,
            ordinals: required.to_vec(),
            inverse,
        })
    }

    pub(crate) fn project(&self, input: &RecordBatch) -> Result<RecordBatch> {
        if input.schema() != self.logical {
            return Err(mapping_error(
                "input does not match the logical retained schema",
            ));
        }
        let columns = self
            .ordinals
            .iter()
            .map(|&index| {
                self.retained_index(index)
                    .map(|_| Arc::clone(input.column(index)))
            })
            .collect::<Result<Vec<_>>>()?;
        RecordBatch::try_new_with_options(
            Arc::clone(&self.physical),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(input.num_rows())),
        )
        .map_err(|error| mapping_error(&error.to_string()))
    }

    pub(crate) fn retained_index(&self, original: usize) -> Result<usize> {
        self.inverse
            .get(original)
            .copied()
            .flatten()
            .ok_or_else(|| mapping_error("logical column is not retained"))
    }

    pub(crate) fn logical_schema(&self) -> &SchemaRef {
        &self.logical
    }
    pub(crate) fn physical_schema(&self) -> &SchemaRef {
        &self.physical
    }
    pub(crate) fn ordinals(&self) -> &[usize] {
        &self.ordinals
    }
}

fn mapping_error(message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "retained_columns".into(),
        message: message.into(),
    }
}
