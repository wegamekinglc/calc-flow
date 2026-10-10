//! Native backward ASOF state and bounded Arrow finalization.
mod admission;
mod checkpoint;
mod codec;
mod copy;
mod cpu;
mod duplicate_fallback;
mod finalize;
mod identity;
mod identity_compare;
mod metadata;
mod output;
mod output_plan;
mod payload_projection;
mod replay;
mod retirement;
mod schema;
mod spec;
mod state;
mod status;
#[cfg(test)]
mod tests;
mod workspace;

use crate::{
    Batch, BatchKind, CalcFlowError, DataFusionConfig, EventTime, IngressProgressSnapshot,
    IngressState, JsonMap, OperatorMetadata, Port, Result, StreamCollector, StreamOperator,
    StreamOperatorContext, StreamingFailureReason, UdfRegistrySnapshot,
};
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
pub(crate) use schema::schema_issues;
pub use spec::{AsofJoinSide, AsofLatePolicy, AsofStateLimits, StreamAsofJoinSpec};
use state::{Inventory, State};
pub use status::{StreamAsofJoinSideStatus, StreamAsofJoinStatus};

/// Eviction inputs (both ingress watermarks and ended flags) under which the
/// committed state was last swept. `State::evict` is deterministic in these
/// inputs, so a matching stamp proves that re-sweeping the committed state
/// would change nothing.
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

/// Stream-only nearest historical match, finalized after both input frontiers.
pub struct StreamAsofJoinOperator {
    name: String,
    spec: StreamAsofJoinSpec,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    schemas: [SchemaRef; 3],
    output_columns: Option<[Vec<usize>; 2]>,
    payload_projection: Option<Box<payload_projection::PayloadProjection>>,
    state: State,
    prepared: Option<checkpoint::PreparedSegment>,
    checkpoint_log: checkpoint::LogState,
    replay_inputs: Option<Box<replay::Inputs>>,
    replay: Option<Box<replay::Log>>,
    deferred_index_len: Option<u64>,
    /// `Some` when the committed state was eviction-swept under the stamped
    /// inputs; `None` when admissions, removals or a restore may have left
    /// evictable rows behind.
    swept: Option<SweepStamp>,
    terminal: bool,
    next_output_sequence: u64,
    status: StreamAsofJoinStatus,
    runtime: output::OutputRuntime,
    retirement: retirement::Owner,
    fingerprint: String,
    schema_digests: [[u8; 32]; 2],
    payload_header_bytes: [u64; 2],
    #[cfg(test)]
    match_hook: Option<std::sync::Arc<dyn Fn(usize) + Send + Sync>>,
    #[cfg(test)]
    admission_hook: Option<std::sync::Arc<dyn Fn(usize) + Send + Sync>>,
}

impl StreamAsofJoinOperator {
    async fn admit_batch(
        &mut self,
        ingress: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let validated = self.validate_admission(ingress, batch)?;
        let admission = self
            .prepare_admission(validated, batch, context)
            .await
            .map_err(|error| self.attempt_error(error))?;
        if admission.rows.is_empty() {
            return Ok(());
        }
        self.install_admission(ingress, admission, validated, context)
            .await
    }

    async fn install_admission(
        &mut self,
        ingress: &str,
        mut admission: admission::Admission,
        validated: admission::ValidatedInput,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let staging_workspace = self.reserve_admission_staging(&admission)?;
        let (index_len, projected, owners) = self.checked_capacity_admission(&admission)?;
        let journal = self
            .prepare_log_admission(&admission, validated.index, &owners)
            .map_err(|error| self.attempt_error(error))?;
        let (credit, retention_bytes) = self
            .prepare_log_retention(
                &std::collections::BTreeMap::default(),
                &std::collections::BTreeMap::default(),
                context.job().checkpointing(),
            )
            .map_err(|error| self.attempt_error(error))?;
        let projected = self
            .log_projection(projected, index_len, &journal, retention_bytes)
            .map_err(|error| self.attempt_error(error))?;
        let mut status = self
            .admitted_status(validated.index, admission.rows.len(), &projected)
            .map_err(|error| self.attempt_error(error))?;
        let index_workspace = self
            .reserve_workspace(if context.job().checkpointing() {
                index_len
            } else {
                0
            })
            .map_err(|error| self.attempt_error(error))?;
        let batches = self
            .state
            .batches
            .project_admission(&admission.batches, &self.name)?
            .new_batches;
        if ingress == "right" {
            admission.right_buckets = admission::parallel::prepare(self, &admission, context)
                .await
                .map_err(|error| self.attempt_error(error))?;
        }
        let copies = self
            .prepare_right_admission_copies(
                if admission.right_buckets.is_some() {
                    &[]
                } else {
                    &admission.right_capacities
                },
                context,
            )
            .await
            .map_err(|error| self.attempt_error(error))?;
        let storage = self
            .state
            .right
            .prepare_storage(&admission.right_capacities, &self.runtime.pool, &self.name)
            .map_err(|error| self.attempt_error(error))?;
        context.check_cancelled()?;
        // Everything after this point is synchronous and infallible. A dropped
        // future or failed preflight cannot expose a partially admitted row.
        storage.install(&mut self.state.right);
        copies.install(&mut self.state.right);
        self.state.batches.reserve_admission(batches);
        admission.install(ingress, &mut self.state, &mut status);
        self.state.install_encoding_owners(owners);
        self.status = status;
        self.checkpoint_log.install_journal(journal);
        self.checkpoint_log.credit = credit;
        self.checkpoint_log.retention_bytes = retention_bytes;
        self.checkpoint_log.pending = None;
        self.checkpoint_log.dirty_cut = self.checkpoint_log.keeps_delta();
        self.prepared = None;
        self.deferred_index_len = Some(index_len);
        self.swept = None;
        debug_assert_eq!(
            checkpoint::v3_encoded_length(&self.state, &self.name).expect("committed index length"),
            index_len
        );
        debug_assert_eq!(
            self.current_inventory(None)
                .expect("committed admission inventory")
                .bytes,
            self.status.state_bytes
        );
        drop((admission, index_workspace, staging_workspace));
        self.retirement.wait(context).await
    }

    fn reserve_admission_staging(
        &mut self,
        admission: &admission::Admission,
    ) -> Result<datafusion::execution::memory_pool::MemoryReservation> {
        self.reserve_workspace(self.state.admission_staging_bytes(
            &admission.rows,
            admission.left_chunks.as_deref(),
            &admission.right_capacities,
            &admission.batches,
            &self.name,
        )?)
        .map_err(|error| self.attempt_error(error))
    }

    fn checked_capacity_admission(
        &mut self,
        admission: &admission::Admission,
    ) -> Result<(u64, Inventory, state::OwnerUpdates)> {
        let projected = self
            .state
            .project_capacity_admission(
                self.capacity_snapshot(),
                &admission.rows,
                admission.left_chunks.as_deref(),
                &admission.right_capacities,
                &admission.batches,
                &self.name,
            )
            .map_err(|error| self.attempt_error(error))?;
        Ok(projected)
    }

    fn admitted_status(
        &self,
        side: usize,
        count: usize,
        projected: &Inventory,
    ) -> Result<StreamAsofJoinStatus> {
        let mut status = self.status.clone();
        status.pending_left_rows = checked(
            &self.name,
            status.pending_left_rows,
            if side == 0 { count as u64 } else { 0 },
        )?;
        status.retained_right_rows = projected.right_payloads;
        status.identity_only_rows = projected.identity_only;
        status.state_rows = projected.identities;
        status.state_bytes = projected.bytes;
        Ok(status)
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
        let schema_digests = [
            codec::schema_digest(&schemas[0])?,
            codec::schema_digest(&schemas[1])?,
        ];
        let payload_header_bytes = [
            workspace::payload_header_bytes(&schemas[0])?,
            workspace::payload_header_bytes(&schemas[1])?,
        ];
        let mut state = State::empty_tracked();
        state.sequence_kinds = [
            state::SequenceKind::for_side(&schemas[0], spec.left()),
            state::SequenceKind::for_side(&schemas[1], spec.right()),
        ];
        let name = name.into();
        let runtime = output::OutputRuntime::new(limit, &name);
        Ok(Self {
            fingerprint,
            schema_digests,
            payload_header_bytes,
            #[cfg(test)]
            match_hook: None,
            #[cfg(test)]
            admission_hook: None,
            name,
            spec,
            inputs,
            outputs,
            schemas,
            output_columns: None,
            payload_projection: None,
            state,
            prepared: None,
            checkpoint_log: checkpoint::LogState::default(),
            replay_inputs: None,
            replay: None,
            deferred_index_len: None,
            swept: None,
            terminal: false,
            next_output_sequence: 0,
            status: StreamAsofJoinStatus::default(),
            runtime,
            retirement: retirement::Owner::default(),
        })
    }
    /// Returns the immutable declaration.
    pub const fn spec(&self) -> &StreamAsofJoinSpec {
        &self.spec
    }

    /// A physical output projection keeps the logical state schema and
    /// fingerprint intact. Input validation and duplicate proofs still use
    /// every declared identity column.
    pub(crate) fn set_output_projection(&mut self, columns: Vec<usize>) -> Result<()> {
        self.configure_projection(columns)
    }
    /// Returns payload-free committed state and attempt counters.
    pub fn status(&self) -> StreamAsofJoinStatus {
        self.status.clone()
    }
    pub(crate) fn set_stream_resources(
        &mut self,
        config: DataFusionConfig,
        _udfs: UdfRegistrySnapshot,
    ) {
        self.runtime.configure(config);
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
    pub(crate) async fn on_ingress_progress_with_output(
        &mut self,
        _ingress: &str,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.retirement.wait(context).await?;
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
    fn capacity_snapshot(&self) -> state::CapacitySnapshot {
        state::CapacitySnapshot {
            inventory: Inventory {
                identities: self.status.state_rows,
                right_payloads: self.status.retained_right_rows,
                identity_only: self.status.identity_only_rows,
                bytes: self
                    .status
                    .state_bytes
                    .saturating_sub(self.checkpoint_log.bytes())
                    .saturating_sub(self.replay_bytes())
                    + self
                        .deferred_index_len
                        .map_or(0, |length| if length == 0 { 0 } else { length + 256 }),
            },
            index_length: self
                .deferred_index_len
                .or_else(|| self.prepared.as_ref().map(|segment| segment.len() as u64))
                .unwrap_or(0),
            index_bytes: self
                .deferred_index_len
                .or_else(|| {
                    self.prepared
                        .as_ref()
                        .map(|segment| segment.capacity() as u64)
                })
                .map_or(0, |bytes| if bytes == 0 { 0 } else { bytes + 256 }),
        }
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
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        self.retirement.wait(context).await?;
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
        self.retirement.wait(context).await?;
        if self.replay.is_some() {
            return self.prepare_replay_anchor(context).await;
        }
        self.ensure_prepared_async(context).await
    }
    fn checkpoint(&mut self, epoch: crate::Epoch) -> Result<crate::OperatorStateSnapshot> {
        self.capture(epoch)
    }
    fn restore(&mut self, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
        let decoded = self.decoded_snapshot(snapshot)?;
        self.install_restored(snapshot, decoded);
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.state = State::empty_tracked();
        self.state.sequence_kinds = self.sequence_kinds();
        self.status = StreamAsofJoinStatus::default();
        self.prepared = None;
        self.checkpoint_log = checkpoint::LogState::default();
        self.deferred_index_len = None;
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
        self.retirement.wait(context).await?;
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

#[cfg(test)]
pub(crate) use output::gather_lifecycle_bridge;
