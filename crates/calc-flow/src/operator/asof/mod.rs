//! Native backward ASOF state and bounded Arrow finalization.
pub(crate) mod admission;
mod checkpoint;
mod codec;
mod columnar;
mod finalize;
mod finite;
#[cfg(test)]
pub(crate) mod gather_lifecycle_bridge;
mod kernel;
mod metadata;
mod payload_projection;
mod replay;
mod schema;
mod spec;
mod status;
#[cfg(test)]
mod tests;

use crate::{
    Batch, BatchKind, CalcFlowError, EventTime, IngressProgressSnapshot, IngressState, JsonMap,
    OperatorMetadata, Port, Result, StreamCollector, StreamOperator, StreamOperatorContext,
    StreamingFailureReason,
};
use async_trait::async_trait;
use columnar::ColumnarState;
use datafusion::arrow::{compute::concat_batches, datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryConsumer, MemoryPool, MemoryReservation,
};
pub(crate) use schema::schema_issues;
pub use spec::{AsofJoinSide, AsofLatePolicy, AsofStateLimits, StreamAsofJoinSpec};
pub use status::{StreamAsofJoinSideStatus, StreamAsofJoinStatus};
use std::{borrow::Cow, sync::Arc};

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) struct SweepStamp {
    left_watermark: Option<EventTime>,
    left_ended: bool,
    right_watermark: Option<EventTime>,
    right_ended: bool,
}

impl SweepStamp {
    fn current(status: &StreamAsofJoinStatus) -> Self {
        Self {
            left_watermark: status.left.watermark_micros,
            left_ended: status.left.ended,
            right_watermark: status.right.watermark_micros,
            right_ended: status.right.ended,
        }
    }
}

pub(super) struct Inventory {
    pub identities: u64,
    pub bytes: u64,
}

/// Nearest historical match over complete finite inputs or closed stream frontiers.
pub struct StreamAsofJoinOperator {
    name: String,
    spec: StreamAsofJoinSpec,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    schemas: [SchemaRef; 3],
    output_projection: Option<Vec<(usize, usize)>>,
    state: ColumnarState,
    replay_inputs: Option<Box<replay::Inputs>>,
    replay: Option<Box<replay::Log>>,
    swept: Option<SweepStamp>,
    terminal: bool,
    next_output_sequence: u64,
    status: StreamAsofJoinStatus,
    runtime_pool: Arc<dyn MemoryPool>,
    fingerprint: String,
}

impl StreamAsofJoinOperator {
    /// Selects the previous lookup algorithm for a benchmark-controlled A/B arm.
    #[doc(hidden)]
    #[cfg(feature = "benchmark-internals")]
    pub fn set_legacy_latest_lookup_for_benchmark(&mut self, enabled: bool) {
        self.state.set_legacy_latest_lookup_for_benchmark(enabled);
    }

    async fn admit_batch(
        &mut self,
        ingress: &str,
        batch: &Batch,
        context: &dyn kernel::KernelContext,
    ) -> Result<()> {
        let prepared = match context.ingress_preparation() {
            Some(preparation) => {
                let facts = preparation.asof().await;
                context.check_cancelled()?;
                Some(facts)
            }
            None => None,
        };
        let validated = self.validate_admission_with_facts(ingress, batch, prepared)?;
        let batches = batch.table_payload()?.batches();
        if batches.is_empty() {
            return Ok(());
        }
        let record = self.prepare_record(batches, validated, context)?;
        let accepted_rows = if validated.index == 0 {
            self.status.left.accepted_rows
        } else {
            self.status.right.accepted_rows
        };
        let next_accepted = checked(&self.name, accepted_rows, validated.accepted)?;
        let admitted = self.state.admit(record.as_ref(), validated, &mut || {
            context.check_cancelled()
        });
        if let Err(error) = admitted {
            if matches!(
                error,
                CalcFlowError::OperatorReason {
                    reason_code: StreamingFailureReason::AsofStateLimitExceeded,
                    ..
                }
            ) && admission::contains_duplicate_identity(
                record.as_ref(),
                validated.side(&self.spec),
                validated,
                &mut || context.check_cancelled(),
            )
            .unwrap_or(false)
            {
                self.record_duplicate(validated.index)?;
                return Err(reason(
                    &self.name,
                    StreamingFailureReason::AsofDuplicateIdentity,
                    "input contains a duplicate key/event-time/sequence identity",
                ));
            }
            if matches!(
                error,
                CalcFlowError::OperatorReason {
                    reason_code: StreamingFailureReason::AsofDuplicateIdentity,
                    ..
                }
            ) {
                self.record_duplicate(validated.index)?;
            }
            return Err(self.attempt_error(error));
        }
        let status = if validated.index == 0 {
            &mut self.status.left
        } else {
            &mut self.status.right
        };
        status.accepted_rows = next_accepted;
        self.swept = None;
        self.refresh_state_gauges();
        Ok(())
    }

    fn prepare_record<'a>(
        &mut self,
        batches: &'a [RecordBatch],
        validated: admission::ValidatedInput,
        context: &dyn kernel::KernelContext,
    ) -> Result<Cow<'a, RecordBatch>> {
        if batches.len() == 1 {
            return Ok(Cow::Borrowed(&batches[0]));
        }
        let full = validated.accepted
            == batches
                .iter()
                .map(|record| record.num_rows() as u64)
                .sum::<u64>();
        let charge = batches.iter().fold(0usize, |total, record| {
            let accepted = admission::AcceptedRows::for_record(
                record,
                validated.side(&self.spec),
                validated,
                full.then_some(record.num_rows()),
            );
            total.saturating_add(columnar::logical_record_selection_charge(record, &accepted))
        });
        let reservation = match self
            .reserve_workspace(u64::try_from(charge).expect("ASOF workspace charge fits u64"))
        {
            Ok(reservation) => reservation,
            Err(error) => {
                if admission::contains_duplicate_identity_in_records(
                    batches,
                    validated.side(&self.spec),
                    validated,
                    &mut || context.check_cancelled(),
                )? {
                    self.record_duplicate(validated.index)?;
                    return Err(reason(
                        &self.name,
                        StreamingFailureReason::AsofDuplicateIdentity,
                        "input contains a duplicate key/event-time/sequence identity",
                    ));
                }
                return Err(self.attempt_error(error));
            }
        };
        let combined =
            concat_batches(&batches[0].schema(), batches).map_err(|error| arrow_error(&error))?;
        drop(reservation);
        Ok(Cow::Owned(combined))
    }

    /// Constructs an independent bounded backward ASOF operator.
    ///
    /// # Errors
    /// Rejects incompatible exact schemas and invalid identity columns.
    pub fn new(
        name: impl Into<String>,
        left_schema: SchemaRef,
        right_schema: SchemaRef,
        spec: StreamAsofJoinSpec,
    ) -> Result<Self> {
        let result_schema = schema::output_schema(&spec, &left_schema, &right_schema)?;
        let inputs = input_ports(&left_schema, &right_schema)?;
        let outputs = vec![Port::with_schema_ref(
            "output",
            BatchKind::Table,
            true,
            Some(result_schema.clone()),
        )?];
        let limit = usize::try_from(spec.limits().max_state_bytes()).map_err(|_| {
            spec::invalid(
                "limits.max_state_bytes",
                "limit does not fit the platform address domain",
            )
        })?;
        let schemas = [left_schema, right_schema, result_schema];
        let fingerprint = Self::state_fingerprint(&spec, &schemas)?;
        let name = name.into();
        let runtime_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
        let state = ColumnarState::new(
            &name,
            &spec,
            [schemas[0].clone(), schemas[1].clone()],
            &runtime_pool,
        )?;
        Ok(Self {
            name,
            spec,
            inputs,
            outputs,
            schemas,
            output_projection: None,
            state,
            replay_inputs: None,
            replay: None,
            swept: None,
            terminal: false,
            next_output_sequence: 0,
            status: StreamAsofJoinStatus::default(),
            runtime_pool,
            fingerprint,
        })
    }

    /// Returns the immutable declaration.
    pub const fn spec(&self) -> &StreamAsofJoinSpec {
        &self.spec
    }

    pub(crate) fn set_output_projection(&mut self, columns: Vec<usize>) -> Result<()> {
        self.configure_projection(columns)
    }

    /// Returns payload-free committed state and attempt counters.
    pub fn status(&self) -> StreamAsofJoinStatus {
        self.status.clone()
    }

    pub(crate) fn output_frontier_candidate(
        &self,
        progress: &IngressProgressSnapshot,
    ) -> Result<Option<EventTime>> {
        if progress
            .by_ingress()
            .keys()
            .any(|side| side != "left" && side != "right")
        {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofProtocolError,
                "ASOF progress contains an unknown ingress",
            ));
        }
        Ok(frontier(progress)
            .and_then(|time| time.checked_sub(1))
            .map(EventTime::from_micros))
    }

    pub(super) fn refresh_state_gauges(&mut self) {
        self.status.pending_left_rows = self.state.pending_left_rows() as u64;
        self.status.retained_right_rows = self.state.right_payload_rows() as u64;
        self.status.identity_only_rows = self.state.identity_only_rows() as u64;
        self.status.state_rows = self.state.state_rows() as u64;
        self.status.state_bytes = self.state.persistent_bytes() as u64 + self.replay_bytes();
    }

    pub(super) fn current_inventory(&self) -> Result<Inventory> {
        Ok(Inventory {
            identities: self.state.state_rows() as u64,
            bytes: checked(
                &self.name,
                self.state.persistent_bytes() as u64,
                self.replay_bytes(),
            )?,
        })
    }

    fn check_inventory_limits(&self, inventory: &Inventory) -> Result<()> {
        if inventory.identities > self.spec.limits().max_state_rows()
            || inventory.bytes > self.spec.limits().max_state_bytes()
        {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofStateLimitExceeded,
                "stream_asof_join state limits exceeded",
            ));
        }
        Ok(())
    }

    fn attempt_error(&mut self, error: CalcFlowError) -> CalcFlowError {
        if let CalcFlowError::OperatorReason { reason_code, .. } = &error {
            let counter = match reason_code {
                StreamingFailureReason::AsofStateLimitExceeded => {
                    Some(&mut self.status.state_limit_failures)
                }
                StreamingFailureReason::AsofWorkspaceLimitExceeded => {
                    Some(&mut self.status.workspace_limit_failures)
                }
                StreamingFailureReason::AsofOutputLimitExceeded => {
                    Some(&mut self.status.output_limit_failures)
                }
                _ => None,
            };
            if let Some(counter) = counter {
                match checked(&self.name, *counter, 1) {
                    Ok(value) => *counter = value,
                    Err(overflow) => return overflow,
                }
            }
        }
        error
    }

    pub(crate) fn reserve_checkpoint_preload(
        &self,
        lengths: impl IntoIterator<Item = u64>,
    ) -> Result<MemoryReservation> {
        let bytes = lengths.into_iter().try_fold(0, |total, length| {
            checked(&self.name, checked(&self.name, total, length)?, 256)
        })?;
        self.reserve_workspace(bytes)
    }

    pub(super) fn reserve_workspace(&self, bytes: u64) -> Result<MemoryReservation> {
        let bytes = usize::try_from(bytes).map_err(|_| {
            reason(
                &self.name,
                StreamingFailureReason::AsofWorkspaceLimitExceeded,
                "ASOF workspace exceeds address domain",
            )
        })?;
        let reservation = MemoryConsumer::new("asof-owned-workspace").register(&self.runtime_pool);
        reservation.try_grow(bytes).map_err(|_| {
            reason(
                &self.name,
                StreamingFailureReason::AsofWorkspaceLimitExceeded,
                "ASOF aggregate workspace exceeds max_state_bytes",
            )
        })?;
        Ok(reservation)
    }

    fn observe(&mut self, progress: &IngressProgressSnapshot) {
        for (name, side) in [
            ("left", &mut self.status.left),
            ("right", &mut self.status.right),
        ] {
            if let Some(current) = progress.get(name) {
                side.watermark_micros = current.watermark();
                side.idle = current.state() == IngressState::Idle;
                side.ended = current.state() == IngressState::Ended;
            }
        }
    }
}

impl OperatorMetadata for StreamAsofJoinOperator {
    fn name(&self) -> &str {
        &self.name
    }
    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }
    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }
    fn configuration(&self) -> JsonMap {
        serde_json::from_value(serde_json::to_value(&self.spec).expect("data-only spec"))
            .expect("spec object")
    }
}

#[async_trait]
impl StreamOperator for StreamAsofJoinOperator {
    async fn on_ingress_progress_with_output(
        &mut self,
        _ingress: &str,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.record_replay(replay::Callback::Progress, None, context)?;
        self.observe(context.ingress_progress());
        self.finalize_with_replay(
            frontier(context.ingress_progress()),
            all_ended(context.ingress_progress()),
            context,
            output,
        )
        .await
        .map_err(|error| self.attempt_error(error))
    }

    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        self.record_replay(
            replay::Callback::Data {
                side: u8::from(ingress == "right"),
                sequence: batch.metadata().sequence(),
            },
            batch.source_cursor(),
            context,
        )?;
        self.observe(context.ingress_progress());
        let previous = self.replay.as_ref().map(|_| self.status.clone());
        match self.admit_batch(ingress, &batch, context).await {
            Err(error) if previous.is_some() && replay::is_capacity_error(&error) => {
                self.status = previous.expect("replay admission saved its counters");
                self.stop_replay()?;
                self.admit_batch(ingress, &batch, context).await
            }
            result => result,
        }
    }

    async fn prepare_checkpoint_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()
    }

    fn checkpoint(&mut self, epoch: crate::Epoch) -> Result<crate::OperatorStateSnapshot> {
        self.capture(epoch)
    }

    fn restore(&mut self, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
        let decoded = self.decoded_snapshot(snapshot)?;
        self.install_restored(decoded);
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state = ColumnarState::new(
            &self.name,
            &self.spec,
            [self.schemas[0].clone(), self.schemas[1].clone()],
            &self.runtime_pool,
        )?;
        self.status = StreamAsofJoinStatus::default();
        self.swept = None;
        self.terminal = false;
        self.next_output_sequence = 0;
        self.reset_replay();
        Ok(())
    }

    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if context.ingress_progress().by_ingress().is_empty() {
            self.finalize_with_replay(Some(watermark.as_micros()), false, context, output)
                .await
                .map_err(|error| self.attempt_error(error))
        } else {
            self.on_ingress_progress_with_output("", context, output)
                .await
        }
    }

    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.record_replay(replay::Callback::End, None, context)?;
        self.status.left.ended = true;
        self.status.right.ended = true;
        self.finalize_with_replay(None, true, context, output)
            .await
            .map_err(|error| self.attempt_error(error))
    }
}

fn frontier(progress: &IngressProgressSnapshot) -> Option<i64> {
    let mut minimum = None;
    for side in ["left", "right"] {
        let state = progress.get(side)?;
        if state.state() == IngressState::Ended {
            continue;
        }
        let watermark = state.watermark()?.as_micros();
        minimum = Some(minimum.map_or(watermark, |value: i64| value.min(watermark)));
    }
    minimum
}

fn all_ended(progress: &IngressProgressSnapshot) -> bool {
    ["left", "right"].iter().all(|side| {
        progress
            .get(side)
            .is_some_and(|value| value.state() == IngressState::Ended)
    })
}

pub(super) fn reason(
    name: &str,
    reason_code: StreamingFailureReason,
    message: &str,
) -> CalcFlowError {
    CalcFlowError::OperatorReason {
        node_id: name.into(),
        reason_code,
        message: message.into(),
    }
}

pub(super) fn arrow_error(error: &datafusion::arrow::error::ArrowError) -> CalcFlowError {
    CalcFlowError::Format {
        message: format!("ASOF Arrow operation failed: {error}"),
    }
}

pub(super) fn checked(name: &str, current: u64, delta: u64) -> Result<u64> {
    current.checked_add(delta).ok_or_else(|| {
        reason(
            name,
            StreamingFailureReason::AsofCounterOverflow,
            "ASOF counter or resource arithmetic overflowed",
        )
    })
}

fn input_ports(left: &SchemaRef, right: &SchemaRef) -> Result<Vec<Port>> {
    [("left", left), ("right", right)]
        .into_iter()
        .map(|(name, schema)| {
            Port::with_schema_ref(name, BatchKind::Table, true, Some(schema.clone()))
        })
        .collect()
}
