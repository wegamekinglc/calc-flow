use std::sync::Arc;

use datafusion::arrow::{array::ArrayRef, record_batch::RecordBatch};

use super::super::{chunk_requirements, footprint::Footprint, scalar_output_plan};
use super::{Check, Evaluated, Output, Prepared, Work, bytes};
use crate::{
    Batch, BatchMetadata, Result,
    operator::rolling::{ScalarValue, operator_error},
};

impl Work {
    pub(super) fn materialize(
        &mut self,
        evaluated: &Evaluated,
        check: Check<'_>,
    ) -> Result<Output> {
        let plan = self.output_plan(evaluated, check)?;
        let (columns, footprint) = self.columns(evaluated, plan.output_bytes, check)?;
        let residual = plan
            .output_bytes
            .checked_sub(footprint.incremental_bytes(&self.quote, &self.node)?)
            .ok_or_else(|| {
                operator_error(&self.node, "finite output exceeded its admitted component")
            })?;
        let record = RecordBatch::try_new(Arc::clone(&self.output_schema), columns)
            .map_err(|_| operator_error(&self.node, "finite output schema is invalid"))?;
        Ok(Output {
            record,
            footprint,
            residual,
        })
    }

    fn output_plan(
        &mut self,
        evaluated: &Evaluated,
        check: Check<'_>,
    ) -> Result<scalar_output_plan::OutputPlan> {
        let width = self.input_schema.fields().len();
        let count = evaluated.rows.len();
        let mut cell = |column, row| cell(evaluated, width, column, row);
        let prefix =
            scalar_output_plan::planning_workspace(&self.output_schema, count, &mut cell, check)?;
        self.reserve(prefix)?;
        let plan = scalar_output_plan::plan(&self.output_schema, count, &mut cell, check)?;
        self.reserve(bytes::add(plan.output_bytes, plan.scratch_bytes)?)?;
        Ok(plan)
    }

    fn columns(
        &self,
        evaluated: &Evaluated,
        quota: u64,
        check: Check<'_>,
    ) -> Result<(Vec<ArrayRef>, Footprint)> {
        let mut columns = Vec::with_capacity(self.output_schema.fields().len());
        let mut footprint = Footprint::default();
        for (ordinal, field) in self.output_schema.fields().iter().enumerate() {
            check()?;
            let array = scalar_output_plan::encode_column(
                field.data_type(),
                evaluated.rows.len(),
                &mut |row| cell(evaluated, self.input_schema.fields().len(), ordinal, row),
                check,
            )?;
            footprint.array_checked(&array, &self.node, check)?;
            if footprint.incremental_bytes(&self.quote, &self.node)? > quota {
                return Err(operator_error(
                    &self.node,
                    "finite output exceeded its admitted component",
                ));
            }
            columns.push(array);
        }
        Ok((columns, footprint))
    }

    pub(super) fn finish(mut self, output: Output, check: Check<'_>) -> Result<Prepared> {
        let envelope = self.admit_envelope(&output)?;
        check()?;
        let metadata = BatchMetadata::new(self.node.as_str(), 0, crate::JsonMap::new())?;
        let batch = Batch::table(vec![output.record], metadata)?;
        self.adopt_batch(output.footprint, output.residual, envelope.output_bytes)?;
        check()?;
        Ok(Prepared {
            _identity: self.node,
            batch,
            generation: self.generation,
            _credit: self.credit,
            _guard: self.guard,
        })
    }

    fn admit_envelope(&mut self, output: &Output) -> Result<chunk_requirements::ChunkRequirements> {
        let envelope = chunk_requirements::batch_envelope(output.record.num_columns(), &self.node)?;
        let record_handles = bytes::slots::<ArrayRef>(output.record.num_columns())?;
        // Evaluation rows, cells and histories have really dropped. Existing
        // credit pins remain live until adjust admits their owner overlap.
        drop(std::mem::take(&mut self.quote));
        self.credit.adjust(
            output.footprint.clone(),
            bytes::total([
                Ok(output.residual),
                Ok(record_handles),
                Ok(envelope.output_bytes),
                Ok(envelope.scratch_bytes),
            ])?,
            self.budget,
            &self.node,
        )?;
        Ok(envelope)
    }

    fn adopt_batch(
        &mut self,
        footprint: Footprint,
        residual: u64,
        envelope_bytes: u64,
    ) -> Result<()> {
        self.credit.adjust(
            footprint,
            bytes::add(residual, envelope_bytes)?,
            self.budget,
            &self.node,
        )
    }
}

fn cell(evaluated: &Evaluated, inputs: usize, column: usize, row: usize) -> Option<&ScalarValue> {
    if column < inputs {
        evaluated.rows.get(row)?.values.get(column)
    } else {
        evaluated.columns.get(column - inputs)?.get(row)?.as_ref()
    }
}
