use super::super::{footprint::Footprint, footprint_state, math_requirements, prefix};
use super::{Check, Evaluated, Work, bytes, order, owned_slots};
use crate::operator::rolling::{
    EntityRollingState, KeyValue, Result, RollingHistories, RollingStreamState,
    compute_output_cells_checked, operator_error,
};

impl Work {
    pub(super) fn evaluate(&mut self, check: Check<'_>) -> Result<Evaluated> {
        let rows = self.ordered_rows(check)?;
        let computed = compute_output_cells_checked(
            &rows,
            &RollingHistories::default(),
            &self.compiled,
            &self.node,
            None,
            check,
        )?;
        let mut state = RollingStreamState::default();
        state.histories.apply(computed.touched);
        self.verify_state(&state)?;
        self.promote_cells(&computed.columns, check)?;
        Ok(Evaluated {
            rows,
            columns: computed.columns,
            _state: state,
        })
    }

    fn ordered_rows(
        &mut self,
        check: Check<'_>,
    ) -> Result<Vec<crate::operator::rolling::BufferedRow>> {
        let mut rows = self.read(check)?;
        order::sort(&mut rows, &self.node, check)?;
        self.admit_math(&rows, check)?;
        prefix::finite(&rows, &self.compiled, self.budget, &self.node, check)?;
        Ok(rows)
    }

    fn verify_state(&self, state: &RollingStreamState) -> Result<()> {
        let (charge, _) = footprint_state::state(state, &self.node)?;
        if !self.budget.allows(charge.rows, charge.bytes) {
            return Err(operator_error(&self.node, "rolling state budget exceeded"));
        }
        Ok(())
    }

    fn admit_math(
        &mut self,
        rows: &[crate::operator::rolling::BufferedRow],
        check: Check<'_>,
    ) -> Result<()> {
        owned_slots::<Option<crate::operator::rolling::ScalarValue>>(rows.len())?;
        let extra = bytes::total([
            math_requirements::plan_finite(rows, &self.compiled, check),
            prefix::scratch(
                self.input.table_payload()?,
                &self.compiled,
                self.budget,
                &self.node,
            ),
            bytes::tree_nodes::<Vec<Option<KeyValue>>, EntityRollingState>(rows.len()),
            owned_slots::<datafusion::arrow::array::ArrayRef>(self.output_schema.fields().len()),
        ])?;
        self.scratch = bytes::add(self.scratch, extra)?;
        self.reserve(0)
    }

    fn promote_cells(
        &mut self,
        columns: &[Vec<Option<crate::operator::rolling::ScalarValue>>],
        check: Check<'_>,
    ) -> Result<()> {
        let mut footprint = Footprint::default();
        for scalar in columns.iter().flatten().flatten() {
            check()?;
            footprint.scalar(scalar, &self.node)?;
        }
        let increment = footprint.incremental_bytes(&self.quote, &self.node)?;
        self.scratch = self.scratch.checked_sub(increment).ok_or_else(|| {
            operator_error(&self.node, "finite cells exceeded their admitted component")
        })?;
        self.quote.merge(&footprint, &self.node)?;
        self.reserve(0)
    }
}
