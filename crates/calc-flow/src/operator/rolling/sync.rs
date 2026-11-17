//! Synchronous execution for admitted ordered rolling graphs.

use super::{
    Batch, Epoch, EventTime, JsonMap, LateMetricDelta, LatePolicySpec, OperatorMetadata, Port,
    Result, RollingOperator, StreamOperator, accumulate_late_metrics, async_trait, budget,
    operator_error, read_buffered_rows,
};
use crate::{
    OperatorStateSnapshot, UdfReference,
    operator::{
        late_output,
        sync_operator::{Emitter, PrepareContext, SyncContext, SyncStreamOperator},
    },
};

pub(crate) struct SyncRollingOperator {
    inner: RollingOperator,
}

impl RollingOperator {
    pub(crate) fn supports_sync(&self) -> bool {
        self.compiled.indicator_groups.is_empty()
            && matches!(self.spec.late_policy, LatePolicySpec::Error { .. })
            && self.compiled.kernel_plan.supports_typed_transition()
    }

    pub(crate) fn into_sync(self) -> SyncRollingOperator {
        SyncRollingOperator { inner: self }
    }
}

impl SyncRollingOperator {
    fn observe(&self, context: &SyncContext<'_>) -> Result<()> {
        late_output::ensure_can_continue(
            self.inner.state.late_output_failed || self.inner.indicator_late_failed,
            context.operator_id(),
        )?;
        if self.inner.state.deferred_history
            && context.checkpointing()
            && !self.inner.compiled.kernel_plan.supports_slab()
        {
            return Err(operator_error(
                context.operator_id(),
                "checkpointing cannot be enabled after deferred rolling history was installed",
            ));
        }
        if self
            .inner
            .state
            .pipeline_fingerprint
            .as_deref()
            .is_some_and(|value| value != context.fingerprint())
        {
            return Err(operator_error(
                context.operator_id(),
                "rolling state was used with a different pipeline fingerprint",
            ));
        }
        if self
            .inner
            .state
            .operator_id
            .as_deref()
            .is_some_and(|value| value != context.operator_id())
        {
            return Err(operator_error(
                context.operator_id(),
                "rolling state was used with a different operator ID",
            ));
        }
        Ok(())
    }

    fn install_identity(&mut self, context: &SyncContext<'_>) -> Result<()> {
        self.inner
            .state
            .owned_mut()?
            .pipeline_fingerprint
            .get_or_insert_with(|| context.fingerprint().to_owned());
        self.inner
            .state
            .owned_mut()?
            .operator_id
            .get_or_insert_with(|| context.operator_id().to_owned());
        Ok(())
    }

    fn data(&mut self, batch: &Batch, context: &mut SyncContext<'_>) -> Result<()> {
        context.check_cancelled()?;
        self.inner.input_ports[0].validate(batch, &format!("{}.input", self.inner.name))?;
        self.observe(context)?;
        if self.inner.state.ended {
            return Err(operator_error(
                context.operator_id(),
                "received data after end-of-input",
            ));
        }
        let observer = context.rolling_metrics();
        let watermark = context.input_watermark();
        if self
            .inner
            .try_buffer_ordered(batch.table_payload()?, watermark, observer)?
        {
            return self.install_identity(context);
        }
        self.inner.materialize_ordered_buffer(observer)?;
        let rows = read_buffered_rows(
            batch.table_payload()?,
            &self.inner.compiled,
            &self.inner.name,
        )?;
        let (accepted, metrics) =
            self.inner
                .classify_envelope(rows, watermark, context.operator_id(), observer)?;
        let next_metrics = accumulate_late_metrics(self.inner.state.metrics, metrics)?;
        let next_charge = self.inner.state.charge.checked_add(
            budget::buffered_charge(accepted.values(), context.operator_id())?,
            context.operator_id(),
        )?;
        self.inner
            .check_state_budget(next_charge, context.operator_id())?;
        for (identity, row) in accepted {
            self.inner.state.owned_mut()?.buffer.insert(identity, row);
        }
        context.late_metrics().record(LateMetricDelta {
            late_rows: metrics.late_rows,
            affected_batches: metrics.affected_batches,
            max_lateness_micros: metrics.max_lateness_micros,
            null_event_time_rows: metrics.null_event_time_rows,
            null_event_time_batches: metrics.null_event_time_batches,
        })?;
        self.inner.state.owned_mut()?.metrics = next_metrics;
        self.inner.state.owned_mut()?.charge = next_charge;
        self.install_identity(context)
    }

    fn watermark(
        &mut self,
        watermark: EventTime,
        context: &mut SyncContext<'_>,
        output: &mut Emitter,
    ) -> Result<()> {
        context.check_cancelled()?;
        self.observe(context)?;
        if self
            .inner
            .state
            .last_input_watermark
            .is_some_and(|previous| watermark <= previous)
        {
            return Err(operator_error(
                context.operator_id(),
                "input watermark did not advance strictly",
            ));
        }
        if self.inner.state.ordered.is_empty() {
            let closing = self
                .inner
                .closing_keys(watermark.as_micros(), context.operator_id())?;
            let rows = self.inner.take_buffered(&closing)?;
            self.inner.emit_slab_rows_sync(&rows, context, output)?;
        } else {
            let last_identity = self.inner.state.ordered.last_identity();
            let records = self.inner.state.owned_mut()?.ordered.take_closed(
                watermark,
                &self.inner.compiled,
                self.inner.spec.allowed_lateness_micros,
                context.operator_id(),
            )?;
            self.inner
                .emit_ordered_sync(records, last_identity, context, output)?;
        }
        self.install_identity(context)?;
        self.inner.state.owned_mut()?.last_input_watermark = Some(watermark);
        Ok(())
    }

    fn end(&mut self, context: &mut SyncContext<'_>, output: &mut Emitter) -> Result<()> {
        context.check_cancelled()?;
        self.observe(context)?;
        if self.inner.state.ended {
            return Ok(());
        }
        if self.inner.state.ordered.is_empty() {
            let rows = self.inner.take_all_buffered()?;
            self.inner.emit_slab_rows_sync(&rows, context, output)?;
        } else {
            let last_identity = self.inner.state.ordered.last_identity();
            let records = self.inner.state.owned_mut()?.ordered.take_all();
            self.inner
                .emit_ordered_sync(records, last_identity, context, output)?;
        }
        self.install_identity(context)?;
        self.inner.state.owned_mut()?.ended = true;
        Ok(())
    }
}

impl OperatorMetadata for SyncRollingOperator {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn input_ports(&self) -> &[Port] {
        self.inner.input_ports()
    }

    fn output_ports(&self) -> &[Port] {
        self.inner.output_ports()
    }

    fn configuration(&self) -> JsonMap {
        self.inner.configuration()
    }

    fn udf_references(&self) -> Vec<UdfReference> {
        self.inner.udf_references()
    }
}

#[async_trait]
impl SyncStreamOperator for SyncRollingOperator {
    async fn prepare(&mut self, context: &PrepareContext<'_>) -> Result<()> {
        context.check_cancelled()
    }

    fn on_data(
        &mut self,
        ingress: usize,
        batch: Batch,
        context: &mut SyncContext<'_>,
        _out: &mut Emitter,
    ) -> Result<()> {
        if ingress != 0 {
            return Err(operator_error(
                context.operator_id(),
                &format!("unknown ingress index {ingress}; expected 0 (\"input\")"),
            ));
        }
        self.data(&batch, context)
    }

    fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &mut SyncContext<'_>,
        out: &mut Emitter,
    ) -> Result<()> {
        self.watermark(watermark, context, out)
    }

    fn on_end(&mut self, context: &mut SyncContext<'_>, out: &mut Emitter) -> Result<()> {
        self.end(context, out)
    }

    fn checkpoint(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        StreamOperator::checkpoint(&mut self.inner, epoch)
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        StreamOperator::restore(&mut self.inner, snapshot)
    }

    fn reset(&mut self) -> Result<()> {
        StreamOperator::reset(&mut self.inner)
    }
}
