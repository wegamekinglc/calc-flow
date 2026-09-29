//! Native backward ASOF state and bounded Arrow finalization.
mod admission;
mod checkpoint;
mod codec;
mod duplicate_fallback;
mod finalize;
mod identity;
mod identity_compare;
mod metadata;
mod output;
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
    state: State,
    prepared: Option<checkpoint::PreparedSegment>,
    /// Exact index length reserved in the state gauge, encoded on the first
    /// output or checkpoint capture that needs canonical bytes.
    deferred_index_len: Option<u64>,
    /// `Some` when the committed state was eviction-swept under the stamped
    /// inputs; `None` when admissions, removals or a restore may have left
    /// evictable rows behind.
    swept: Option<SweepStamp>,
    terminal: bool,
    next_output_sequence: u64,
    status: StreamAsofJoinStatus,
    runtime: output::OutputRuntime,
    fingerprint: String,
    schema_digests: [[u8; 32]; 2],
}

impl StreamAsofJoinOperator {
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
        Ok(Self {
            fingerprint,
            schema_digests,
            name: name.into(),
            spec,
            inputs,
            outputs,
            schemas,
            state: State::default(),
            prepared: None,
            deferred_index_len: None,
            swept: None,
            terminal: false,
            next_output_sequence: 0,
            status: StreamAsofJoinStatus::default(),
            runtime: output::OutputRuntime::new(limit),
        })
    }
    /// Returns the immutable declaration.
    pub const fn spec(&self) -> &StreamAsofJoinSpec {
        &self.spec
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
        self.observe(context.ingress_progress());
        self.finalize(
            frontier(context.ingress_progress()),
            all_ended(context.ingress_progress()),
            context,
            output,
        )
        .await
        .map_err(|error| self.attempt_error(error))
    }
    /// Recomputes the candidate's retained row and shared batch charge.
    fn checked_inventory(
        &self,
        state: &State,
        prepared: Option<&checkpoint::PreparedSegment>,
        status: &mut StreamAsofJoinStatus,
    ) -> Result<()> {
        let inventory = state.inventory(prepared, &self.name)?;
        self.check_inventory_values(state, inventory, status)
    }

    fn check_inventory_values(
        &self,
        state: &State,
        inventory: Inventory,
        status: &mut StreamAsofJoinStatus,
    ) -> Result<()> {
        self.check_inventory_limits(&inventory)?;
        status.pending_left_rows = state.left.len() as u64;
        status.retained_right_rows = inventory.right_payloads;
        status.identity_only_rows = inventory.identity_only;
        status.state_rows = inventory.identities;
        status.state_bytes = inventory.bytes;
        Ok(())
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

    /// Installs a prepared candidate transactionally: state, status and the
    /// encoded segment swap in together. `swept` records that the candidate
    /// was eviction-swept under the installed status watermarks, letting
    /// progress-only watermark ticks skip re-encoding.
    fn install(
        &mut self,
        state: State,
        status: StreamAsofJoinStatus,
        prepared: checkpoint::PreparedCheckpoint,
        swept: bool,
    ) {
        self.swept = swept.then(|| SweepStamp::current(&status));
        self.state = state;
        self.status = status;
        self.prepared = prepared.segment;
        self.deferred_index_len = None;
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
        self.observe(context.ingress_progress());
        let validated = self.validate_admission(ingress, &batch)?;
        let mut admission = self
            .prepare_admission(validated, &batch, context)
            .await
            .map_err(|error| self.attempt_error(error))?;
        if admission.rows.is_empty() {
            // Empty or fully late input: committed state, gauges and the
            // prepared segment are untouched, so admission would reinstall
            // an identical state.
            return Ok(());
        }
        let index_len = self
            .index_length_after_admission(validated.index, &admission.rows)
            .map_err(|error| self.attempt_error(error))?;
        let previous_index_bytes = self
            .deferred_index_len
            .or_else(|| {
                self.prepared
                    .as_ref()
                    .map(|segment| segment.capacity() as u64)
            })
            .map_or(0, |capacity| capacity + 64);
        let projected = self
            .state
            .inventory_after_admission(
                Inventory {
                    identities: self.status.state_rows,
                    right_payloads: self.status.retained_right_rows,
                    identity_only: self.status.identity_only_rows,
                    bytes: self.status.state_bytes,
                },
                previous_index_bytes,
                index_len,
                validated.index,
                &admission.rows,
                &self.name,
            )
            .map_err(|error| self.attempt_error(error))?;
        self.check_inventory_limits(&projected)
            .map_err(|error| self.attempt_error(error))?;
        let mut status = self.status.clone();
        status.pending_left_rows = checked(
            &self.name,
            status.pending_left_rows,
            if validated.index == 0 {
                admission.rows.len() as u64
            } else {
                0
            },
        )
        .map_err(|error| self.attempt_error(error))?;
        status.retained_right_rows = projected.right_payloads;
        status.identity_only_rows = projected.identity_only;
        status.state_rows = projected.identities;
        status.state_bytes = projected.bytes;
        let index_workspace = self
            .reserve_workspace(index_len)
            .map_err(|error| self.attempt_error(error))?;
        context.check_cancelled()?;
        // Everything after this point is synchronous and infallible. A dropped
        // future or failed preflight cannot expose a partially admitted row.
        admission.install(ingress, &mut self.state, &mut status);
        self.status = status;
        self.prepared = None;
        self.deferred_index_len = Some(index_len);
        self.swept = None;
        debug_assert_eq!(
            checkpoint::encoded_length(&self.state, &self.name).expect("committed index length"),
            index_len
        );
        debug_assert_eq!(
            self.state
                .inventory(None, &self.name)
                .expect("committed admission inventory")
                .bytes
                + index_len
                + 64,
            self.status.state_bytes
        );
        drop((admission, index_workspace));
        Ok(())
    }
    async fn prepare_checkpoint_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
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
        self.state = State::default();
        self.status = StreamAsofJoinStatus::default();
        self.prepared = None;
        self.deferred_index_len = None;
        self.swept = None;
        self.terminal = false;
        self.next_output_sequence = 0;
        Ok(())
    }
    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if context.ingress_progress().by_ingress().is_empty() {
            self.finalize(Some(watermark.as_micros()), false, context, output)
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
        self.status.left.ended = true;
        self.status.right.ended = true;
        self.finalize(None, true, context, output)
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
