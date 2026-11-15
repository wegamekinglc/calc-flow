//! Crate-private synchronous operators and their reusable callback state.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    marker::PhantomData,
    sync::Arc,
};

use async_trait::async_trait;

use super::{
    LateMetricSink, OperatorMetadata,
    rolling_metrics::RollingMetricsRecorder,
};
use crate::{
    Batch, CalcFlowError, EdgeBudget, Epoch, EventTime, IngressProgressSnapshot,
    OperatorStateSnapshot, Port, Result, StreamJobContext,
    batch::TableBatchShell,
    runtime::streaming::{EnvelopeCost, OperatorEdgeSender, StreamJobControl, StreamMessage},
};

/// One-time preparation resources shared by a synchronous operator.
pub(crate) struct PrepareContext<'a> {
    job: &'a StreamJobContext,
}

impl<'a> PrepareContext<'a> {
    pub(crate) const fn new(job: &'a StreamJobContext) -> Self {
        Self { job }
    }

    pub(crate) fn check_cancelled(&self) -> Result<()> {
        self.job.check_cancelled()
    }
}

/// Reusable callback context. Job control and metric sinks are cheap owned
/// handles; only the changing watermark and progress snapshot are updated.
#[allow(
    dead_code,
    reason = "P2.7 rolling consumes the complete late, progress, budget and recorder surface"
)]
pub(crate) struct SyncContext<'a> {
    operator_id: Arc<str>,
    input_watermark: Option<EventTime>,
    ingress_progress: IngressProgressSnapshot,
    output_budget: EdgeBudget,
    late_metrics: Arc<dyn LateMetricSink>,
    rolling_metrics: Option<RollingMetricsRecorder>,
    control: StreamJobControl,
    _job_scope: PhantomData<&'a StreamJobContext>,
}

impl SyncContext<'static> {
    pub(crate) fn new(
        job: &StreamJobContext,
        operator_id: Arc<str>,
        output_budget: EdgeBudget,
        late_metrics: Arc<dyn LateMetricSink>,
    ) -> Self {
        Self {
            operator_id,
            input_watermark: None,
            ingress_progress: IngressProgressSnapshot::default(),
            output_budget,
            late_metrics,
            rolling_metrics: None,
            control: job.native_control(),
            _job_scope: PhantomData,
        }
    }
}

#[allow(
    dead_code,
    reason = "P2.7 rolling consumes the complete late, progress, budget and recorder surface"
)]
impl SyncContext<'_> {
    pub(crate) fn update(
        &mut self,
        input_watermark: Option<EventTime>,
        ingress_progress: IngressProgressSnapshot,
    ) {
        self.input_watermark = input_watermark;
        self.ingress_progress = ingress_progress;
    }

    pub(crate) fn set_rolling_metrics(&mut self, metrics: Option<RollingMetricsRecorder>) {
        self.rolling_metrics = metrics;
    }

    pub(crate) fn operator_id(&self) -> &str {
        &self.operator_id
    }

    pub(crate) const fn input_watermark(&self) -> Option<EventTime> {
        self.input_watermark
    }

    pub(crate) const fn ingress_progress(&self) -> &IngressProgressSnapshot {
        &self.ingress_progress
    }

    pub(crate) const fn output_budget(&self) -> EdgeBudget {
        self.output_budget
    }

    pub(crate) fn rolling_metrics(&self) -> Option<&RollingMetricsRecorder> {
        self.rolling_metrics.as_ref()
    }

    pub(crate) fn late_metrics(&self) -> &Arc<dyn LateMetricSink> {
        &self.late_metrics
    }

    pub(crate) fn check_cancelled(&self) -> Result<()> {
        self.control.check_cancelled()
    }
}

/// A validated emission and its already computed edge cost.
pub(crate) struct Emission {
    pub(crate) port: usize,
    pub(crate) message: StreamMessage,
    pub(crate) cost: EnvelopeCost,
}

/// One indexed output port and all of its runtime routes.
pub(crate) struct EmitterOutput {
    pub(crate) port: Port,
    pub(crate) senders: Vec<OperatorEdgeSender>,
    pub(crate) late: bool,
    endpoint: String,
}

/// A per-node outbox with retained capacity and table-shell ownership.
pub(crate) struct Emitter {
    node_id: Arc<str>,
    outputs: Vec<EmitterOutput>,
    outbox: VecDeque<Emission>,
    shells: Vec<TableBatchShell>,
}

impl Emitter {
    pub(crate) fn new(
        node_id: Arc<str>,
        output_ports: &BTreeMap<String, Port>,
        mut routes: BTreeMap<String, Vec<OperatorEdgeSender>>,
        late_output_ports: &BTreeSet<String>,
    ) -> Self {
        let outputs = output_ports
            .iter()
            .map(|(name, port)| EmitterOutput {
                port: port.clone(),
                senders: routes.remove(name).unwrap_or_default(),
                late: late_output_ports.contains(name),
                endpoint: format!("node.{node_id}.outputs.{name}"),
            })
            .collect();
        Self {
            node_id,
            outputs,
            outbox: VecDeque::new(),
            shells: Vec::new(),
        }
    }

    /// Validates and buffers one output without looking up a port by name.
    pub(crate) fn emit(&mut self, port: usize, batch: Batch) -> Result<()> {
        let output = self.outputs.get(port).ok_or_else(|| CalcFlowError::Compile {
            message: format!(
                "node.{}.outputs.index.{port} is unknown",
                self.node_id
            ),
        })?;
        output.port.validate_fast(&batch, &output.endpoint)?;
        let message = StreamMessage::data(batch);
        let cost = EnvelopeCost::of_message(&message)?;
        validate_routes(&self.node_id, output, &cost)?;
        self.outbox.push_back(Emission {
            port,
            message,
            cost,
        });
        Ok(())
    }

    pub(crate) fn pop(&mut self) -> Option<Emission> {
        self.outbox.pop_front()
    }

    pub(crate) fn clear(&mut self) {
        self.outbox.clear();
    }

    pub(crate) fn outputs_mut(&mut self) -> &mut [EmitterOutput] {
        &mut self.outputs
    }

    pub(crate) fn output_mut(&mut self, port: usize) -> Option<&mut EmitterOutput> {
        self.outputs.get_mut(port)
    }

    pub(crate) fn has_non_late_output(&self) -> bool {
        self.outputs.iter().any(|output| !output.late)
    }

    /// Returns a uniquely owned shell to its producer after downstream use.
    pub(crate) fn recycle(&mut self, batch: Batch) {
        if let Some(shell) = batch.recycle_table_shell() {
            self.shells.push(shell);
        }
    }

    pub(crate) fn take_table_shell(&mut self) -> Option<TableBatchShell> {
        self.shells.pop()
    }
}

fn validate_routes(node_id: &str, output: &EmitterOutput, cost: &EnvelopeCost) -> Result<()> {
    if output.senders.is_empty() {
        return Err(CalcFlowError::Internal {
            message: format!("node {node_id:?} output {:?} has no runtime routes", output.port.name()),
        });
    }
    for sender in &output.senders {
        sender.validate_cost(cost)?;
    }
    Ok(())
}

/// One prepared synchronous operator and its node-scoped reusable state.
pub(crate) struct SyncNode<O: SyncStreamOperator> {
    operator: O,
    context: SyncContext<'static>,
    emitter: Emitter,
}

impl<O: SyncStreamOperator> SyncNode<O> {
    pub(crate) fn new(operator: O, context: SyncContext<'static>, emitter: Emitter) -> Self {
        Self {
            operator,
            context,
            emitter,
        }
    }

    pub(crate) const fn operator(&self) -> &O {
        &self.operator
    }

    pub(crate) fn operator_mut(&mut self) -> &mut O {
        &mut self.operator
    }

    pub(crate) fn context_mut(&mut self) -> &mut SyncContext<'static> {
        &mut self.context
    }

    pub(crate) const fn emitter(&self) -> &Emitter {
        &self.emitter
    }

    pub(crate) fn emitter_mut(&mut self) -> &mut Emitter {
        &mut self.emitter
    }

    pub(crate) fn parts_mut(
        &mut self,
    ) -> (&mut O, &mut SyncContext<'static>, &mut Emitter) {
        (&mut self.operator, &mut self.context, &mut self.emitter)
    }

    pub(crate) fn recycle(&mut self, batch: Batch) {
        self.emitter.recycle(batch);
    }
}

#[async_trait]
pub(crate) trait SyncStreamOperator: OperatorMetadata + Send {
    /// One-time preparation; may await DataFusion planning, never row work.
    async fn prepare(&mut self, cx: &PrepareContext<'_>) -> Result<()>;

    fn on_data(
        &mut self,
        ingress: usize,
        batch: Batch,
        cx: &mut SyncContext<'_>,
        out: &mut Emitter,
    ) -> Result<()>;

    fn on_watermark(
        &mut self,
        watermark: EventTime,
        cx: &mut SyncContext<'_>,
        out: &mut Emitter,
    ) -> Result<()>;

    fn on_end(&mut self, cx: &mut SyncContext<'_>, out: &mut Emitter) -> Result<()>;

    fn checkpoint(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot>;

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()>;

    fn reset(&mut self) -> Result<()>;
}
