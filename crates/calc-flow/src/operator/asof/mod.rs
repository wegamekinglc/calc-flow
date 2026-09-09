//! Native backward ASOF state and bounded `DataFusion` finalization.
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
use state::State;
pub use status::{StreamAsofJoinSideStatus, StreamAsofJoinStatus};

/// Stream-only nearest historical match, finalized after both input frontiers.
pub struct StreamAsofJoinOperator {
    name: String,
    spec: StreamAsofJoinSpec,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    schemas: [SchemaRef; 3],
    state: State,
    prepared: Option<crate::StateSegment>,
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
        let inputs = vec![
            Port::with_schema_ref("left", BatchKind::Table, true, Some(left_schema.clone()))?,
            Port::with_schema_ref("right", BatchKind::Table, true, Some(right_schema.clone()))?,
        ];
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
    pub(crate) const fn stream_runtime_initialized(&self) -> bool {
        self.runtime.initialized()
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
    fn checked_inventory(
        &self,
        state: &State,
        prepared: Option<&crate::StateSegment>,
        status: &mut StreamAsofJoinStatus,
    ) -> Result<()> {
        let inventory = state.inventory(prepared, &self.name)?;
        if inventory.identities > self.spec.limits().max_state_rows()
            || inventory.bytes > self.spec.limits().max_state_bytes()
        {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofStateLimitExceeded,
                "stream_asof_join state limits exceeded",
            ));
        }
        status.pending_left_rows = state.left.len() as u64;
        status.retained_right_rows = inventory.right_payloads;
        status.identity_only_rows = inventory.identity_only;
        status.state_rows = inventory.identities;
        status.state_bytes = inventory.bytes;
        Ok(())
    }
    fn install(
        &mut self,
        state: State,
        status: StreamAsofJoinStatus,
        prepared: checkpoint::PreparedCheckpoint,
    ) {
        self.state = state;
        self.status = status;
        self.prepared = prepared.segment;
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
        let state_workspace = self
            .state_workspace(&self.state)
            .map_err(|error| self.attempt_error(error))?;
        let mut next = self.state.clone();
        let mut status = self.status.clone();
        admission.install(ingress, &mut next, &mut status);
        let prepared = self
            .prepare_checkpoint(&next, context)
            .await
            .map_err(|error| self.attempt_error(error))?;
        self.checked_inventory(&next, prepared.segment.as_ref(), &mut status)
            .map_err(|error| self.attempt_error(error))?;
        self.install(next, status, prepared);
        drop((admission, state_workspace));
        Ok(())
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
        self.terminal = false;
        self.next_output_sequence = 0;
        self.runtime.reset();
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
pub(super) fn fusion_error(error: &datafusion::error::DataFusionError) -> CalcFlowError {
    CalcFlowError::DataFusion {
        node_id: None,
        message: format!("ASOF output failed: {error}"),
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
