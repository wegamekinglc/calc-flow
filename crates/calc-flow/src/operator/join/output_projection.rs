//! Physical output columns composed with retained payload layouts.

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
    /// Installs physical columns after logical validation; admission keeps its
    /// complete input schemas and frozen V1 logical charges.
    pub(crate) fn set_output_projection(&mut self, columns: Vec<usize>) -> Result<()> {
        let output = self.projection_port(&columns)?;
        let layout = super::state_projection::Layout::new(self, &columns)?;
        if layout.is_some() {
            self.validate_fresh_projection()?;
        }
        let physical = if let Some(layout) = &layout {
            layout.output_columns(&columns, self.input_schema(0).fields().len())
        } else {
            columns
        };
        self.rebind_checkpoint_layout(layout.as_ref())?;
        self.output_ports[0] = output;
        self.output_columns = Some(physical.into());
        self.payload_projection = layout.map(Arc::new);
        Ok(())
    }

    fn projection_port(&self, columns: &[usize]) -> Result<Port> {
        let schema = self
            .logical_output_schema
            .project(columns)
            .map_err(|error| {
                operator_error(&self.name, &format!("output projection failed: {error}"))
            })?;
        Port::with_schema_ref("output", BatchKind::Table, true, Some(Arc::new(schema)))
    }

    pub(super) fn validate_fresh_projection(&self) -> Result<()> {
        if self.state.next_left_row_id != 0
            || self.state.next_right_row_id != 0
            || self.state.last_checkpoint_epoch.is_some()
            || self.state.ended
        {
            return Err(operator_error(
                &self.name,
                "retained projection requires a fresh Join",
            ));
        }
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
