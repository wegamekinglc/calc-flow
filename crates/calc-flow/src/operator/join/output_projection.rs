//! Physical output columns, with full retained-state schemas and charges.

use std::{ops::Range, sync::Arc};

use datafusion::arrow::{
    array::ArrayRef,
    record_batch::{RecordBatch, RecordBatchOptions},
};

use super::{
    MatchedPair, StreamJoinOperator, materialization::JoinOutput, operator_error, output_gather,
};
use crate::{BatchKind, Port, Result};

#[cfg(test)]
pub(super) mod tests;

impl StreamJoinOperator {
    /// Installs physical output columns after logical graph validation, keeping
    /// retained schemas, frozen V1 charges and checkpoint formats unchanged.
    pub(crate) fn set_output_projection(&mut self, columns: Vec<usize>) -> Result<()> {
        let schema = self
            .logical_output_schema
            .project(&columns)
            .map_err(|error| {
                operator_error(&self.name, &format!("output projection failed: {error}"))
            })?;
        let output =
            Port::with_schema_ref("output", BatchKind::Table, true, Some(Arc::new(schema)))?;
        self.output_ports[0] = output;
        self.output_columns = Some(columns.into());
        Ok(())
    }
}

pub(super) fn materialize(output: &JoinOutput<'_>, range: Range<usize>) -> Result<RecordBatch> {
    let matched = &output.matched[range];
    if matched.is_empty() {
        return Ok(RecordBatch::new_empty(Arc::clone(output.schema)));
    }
    let columns = if output.schema.fields().is_empty() {
        Vec::new()
    } else {
        gather_columns(output, matched)?
    };
    RecordBatch::try_new_with_options(
        Arc::clone(output.schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(matched.len())),
    )
    .map_err(|error| {
        operator_error(
            output.operator_id,
            &format!("output projection failed: {error}"),
        )
    })
}

fn gather_columns(output: &JoinOutput<'_>, matched: &[MatchedPair]) -> Result<Vec<ArrayRef>> {
    let left_width = output.records(&matched[0]).0.num_columns();
    let left_gather =
        output_gather::SideGather::new(matched.iter().map(|pair| output.records(pair).0));
    let right_gather =
        output_gather::SideGather::new(matched.iter().map(|pair| output.records(pair).1));
    output
        .schema
        .fields()
        .iter()
        .enumerate()
        .map(|(ordinal, field)| {
            let index = output.logical_column(ordinal);
            let (gather, payload_index) = if index < left_width {
                (left_gather.as_ref(), index)
            } else {
                (right_gather.as_ref(), index - left_width)
            };
            let payloads = matched.iter().map(|pair| {
                let (left, right) = output.records(pair);
                if index < left_width { left } else { right }
            });
            output_gather::column(gather, payload_index, field.data_type(), payloads)
        })
        .collect()
}
