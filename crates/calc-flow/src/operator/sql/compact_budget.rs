use super::{
    PreparedRetention, SqlOperator, StateBudget, StreamCollector, StreamOperatorContext,
    incremental,
};
use crate::{CalcFlowError, Result};

impl SqlOperator {
    pub(crate) fn set_compact_state_budget(&mut self, budget: Option<StateBudget>) -> Result<()> {
        if !self.stream_aggregate {
            return Ok(());
        }
        self.validate_existing_compact_budget(budget)?;
        self.validate_compact_admission(budget)?;
        self.stream_state
            .runtime()?
            .set_compact_state_limit(budget.map(StateBudget::max_bytes), &self.name)?;
        if let Some(native) = self.incremental.as_mut() {
            native.compact_budget = budget;
        }
        self.compact_state_budget = budget;
        Ok(())
    }

    fn validate_existing_compact_budget(&self, budget: Option<StateBudget>) -> Result<()> {
        if budget.is_some() && self.retained.as_ref().is_some_and(|s| s.rows != 0) {
            return Err(self.compact_required_error(
                "nonempty retained state requires a new lineage and replay",
            ));
        }
        if let Some(native) = self.incremental.as_ref() {
            incremental::IncrementalSql::check_group_budget(
                native.group_count(),
                budget,
                &self.name,
            )?;
        }
        Ok(())
    }

    pub(super) fn configure_incremental_budget(
        &self,
        native: &mut Option<Box<incremental::IncrementalSql>>,
    ) -> Result<()> {
        if let Some(native) = native {
            native.set_compact_budget(self.compact_state_budget, &self.name)?;
        }
        Ok(())
    }

    pub(super) async fn process_retained_fallback(
        &mut self,
        prepared: PreparedRetention,
        alias: String,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if self.stream_aggregate
            && self.compact_state_budget.is_some()
            && prepared.batch.num_rows() != 0
        {
            return Err(self.compact_required_error(
                "query shape, types or physical execution proof are unsupported",
            ));
        }
        if self.stream_aggregate && prepared.batch.num_rows() != 0 {
            tracing::debug!(node = %self.name, mode = "retained", reason = "compact shape/types/model not proved");
        }
        self.process_retained(prepared, alias, context, output)
            .await
    }

    fn validate_compact_admission(&mut self, budget: Option<StateBudget>) -> Result<()> {
        if budget.is_some() && self.incremental.is_none() {
            if !self.udfs.is_empty() {
                return Err(
                    self.compact_required_error("registered UDFs require retained execution")
                );
            }
            if let Some(schema) = self.input_ports[0].schema() {
                let runtime = self.stream_state.runtime()?;
                if incremental::IncrementalSql::plan_sync(
                    runtime,
                    &self.validated,
                    &self.aliases[0],
                    schema.clone(),
                    schema.clone(),
                    &self.name,
                )?
                .is_none()
                {
                    return Err(self.compact_required_error(
                        "query shape, types or execution model are unsupported",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn compact_required_error(&self, reason: &str) -> CalcFlowError {
        CalcFlowError::InvalidArgument {
            field: "sql_compact_state_budget".into(),
            message: format!(
                "SQL node {} requires compact execution: {reason}",
                self.name
            ),
        }
    }
}

#[cfg(test)]
#[path = "compact_budget_tests.rs"]
mod tests;
