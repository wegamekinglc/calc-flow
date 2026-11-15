//! Caller-driven, in-memory execution of one compiled stream plan.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use datafusion::arrow::{
    array::{
        Array, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray,
    },
    datatypes::{SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use tokio::sync::{mpsc, watch};

use super::{
    StreamJobContext, StreamMessage,
    local_edge::{OperatorEdgeReceiver, OperatorEdgeSender, SessionEdgeSender, session_edge},
    metrics::MetricsRecorder,
    operator_task::{
        OperatorProgress, OperatorTaskInputs,
        caller_driven::{CallerDrivenOperator, drain_in_order_sync},
    },
    progress::{
        GeneratedWatermarkState,
        prepare::{
            ArrowTimestampUnit, ResolvedEventTimeColumn, is_logical_nanos_duration,
            resolve_event_time_column,
        },
    },
    runner::{take_node_ingresses, take_node_outputs},
    source_task::sequenced_source_message,
};
use crate::{
    Batch, CalcFlowError, CancellationToken, DeliveryGuarantee, EventTime, JsonMap, Port, Result,
    StreamExecutionPlan, StreamRuntimeConfig,
    pipeline::{
        CompiledStreamOperator, RuntimeEdge, RuntimeSourceRoute, RuntimeStreamNode,
        StreamRuntimePlanParts,
    },
};

/// Caller-driven, in-memory execution of one compiled stream plan.
///
/// [`push`](Self::push), [`advance`](Self::advance) and
/// [`finish`](Self::finish) are synchronous. Each call runs its batch and
/// watermark through every operator in topological order on the calling
/// thread and returns the rows emitted by that call. There is no source, sink
/// or progress task. Async operators use one session-owned current-thread
/// Tokio runtime (timers only); an all-sync graph releases it after preparation
/// and never enters it from a call. Calling from inside a Tokio runtime context
/// returns [`CalcFlowError::InvalidArgument`] at field `session` instead of
/// blocking that runtime.
///
/// Sessions are in-memory and non-durable. The checkpoint fields of
/// [`StreamRuntimeConfig`] are ignored, including `checkpointing = true`;
/// `edge_budget` bounds each envelope and the SQL state budgets are applied as
/// runners apply them. Inner edges have no queue backpressure: one call
/// buffers what each operator emits during its turn, so memory per call is
/// bounded by the operators' own output chunking.
///
/// Construction admits one dynamic input (no static inputs) feeding `Rolling`
/// (not the indicator or MACD layouts), `Expression` and `Union` nodes, with
/// no exactly-once output. Other graphs return
/// [`CalcFlowError::InvalidArgument`] naming the field and should use
/// [`StreamingRunner`](crate::StreamingRunner).
///
/// A validation error (unknown input, schema, budget, watermark or ordering)
/// leaves the session unchanged. Any later error fails the session and
/// discards rows emitted earlier in that call; operator panics are returned
/// as [`CalcFlowError::Operator`] with a `panicked: ` message. Calls on a
/// failed or finished session return [`CalcFlowError::InvalidArgument`] at
/// field `session`. Dropping a session cancels its job and releases every
/// operator.
///
/// ```
/// use std::{collections::BTreeMap, sync::Arc};
///
/// use calc_flow::arrow::{
///     array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
///     datatypes::{DataType, Field, Schema, TimeUnit},
///     record_batch::RecordBatch,
/// };
/// use calc_flow::{
///     Batch, BatchKind, BatchMetadata, Edge, EventTime, ExpressionOperator, OperatorMetadata,
///     PipelineBuilder, Port, PortEndpoint, RollingOperator, RollingSpec, StreamRequirements,
///     StreamRuntimeConfig, StreamSession, UdfRegistry,
/// };
///
/// let schema = Arc::new(Schema::new(vec![
///     Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), false),
///     Field::new("symbol", DataType::Utf8, false),
///     Field::new("sequence", DataType::UInt64, false),
///     Field::new("price", DataType::Float64, false),
/// ]));
/// let spec: RollingSpec = serde_json::from_value(serde_json::json!({
///     "configuration_version": 1, "state_layout_version": 1,
///     "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["sequence"],
///     "outputs": [{"kind": "mean", "primitive_version": 1, "input": "price",
///         "output": "price_mean", "frame": {"kind": "rows", "size": 2}, "min_periods": 1}],
///     "allowed_lateness_micros": 0, "late_policy": {"kind": "error", "scope": "envelope"},
///     "value_policy": "stateful_numeric_v1",
/// }))?;
/// let rolling = RollingOperator::new("rolling", Arc::clone(&schema), spec)?;
/// let rolled = rolling.output_ports()[0].schema().expect("exact schema").fields().to_vec();
/// let project = ExpressionOperator::new(
///     "project", "", vec!["ts".into(), "symbol".into(), "price_mean".into()], None, vec![],
/// )?
/// .with_ports(
///     Port::new("input", BatchKind::Table, true, Some(rolled.iter().map(|f| f.as_ref().clone()).collect()))?,
///     Port::new("output", BatchKind::Table, true, None)?,
/// )?;
/// let plan = PipelineBuilder::new("quotes")?
///     .add_node("rolling", rolling)?
///     .add_node("project", Box::new(project))?
///     .connect(Edge::new(
///         PortEndpoint::new("rolling", "output")?,
///         PortEndpoint::new("project", "input")?,
///     ))?
///     .compile_stream(&UdfRegistry::new().snapshot(), &StreamRequirements::default())?;
///
/// let mut session = StreamSession::new(plan, &StreamRuntimeConfig::default(), BTreeMap::new())?;
/// let (input, output) = (session.input_ids()[0].to_owned(), session.output_ids()[0].to_owned());
/// let batch = |ts: i64, sequence: u64, price: f64| -> Result<Batch, Box<dyn std::error::Error>> {
///     let columns: Vec<ArrayRef> = vec![
///         Arc::new(TimestampMicrosecondArray::from(vec![ts]).with_timezone("UTC")),
///         Arc::new(StringArray::from(vec!["a"])),
///         Arc::new(UInt64Array::from(vec![sequence])),
///         Arc::new(Float64Array::from(vec![price])),
///     ];
///     let record = RecordBatch::try_new(Arc::clone(&schema), columns)?;
///     Ok(Batch::table(vec![record], BatchMetadata::default())?)
/// };
///
/// // The explicit watermark closes each row in the same call.
/// let first = session.push(&input, batch(10, 0, 1.0)?, Some(EventTime::from_micros(10)))?;
/// assert_eq!(first.batches(&output)[0].num_rows(), 1);
/// let second = session.push(&input, batch(11, 1, 3.0)?, Some(EventTime::from_micros(11)))?;
/// assert_eq!(second.batches(&output)[0].num_rows(), 1);
/// assert!(session.finish()?.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct StreamSession {
    runtime: Option<tokio::runtime::Runtime>,
    job: StreamJobContext,
    input: SessionInput,
    nodes: Vec<CallerDrivenOperator>,
    outputs: Vec<(String, OperatorEdgeReceiver)>,
    lifecycle: Lifecycle,
}

/// How a session input's watermark advances.
///
/// Generated modes use the runtime's generated-watermark arithmetic after
/// every push, with no emit interval or idle timeout. An explicit watermark
/// passed to [`StreamSession::push`] or [`StreamSession::advance`] combines
/// with a generated one as their maximum: the caller declares the input
/// complete up to that time.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SessionWatermark {
    /// Advances only through watermarks passed to `push` or `advance`
    /// (the runtime's source-provided mode).
    Explicit,
    /// Requires non-null event time that never decreases and stays after the
    /// current watermark; advances to `max(event_time) - 1 µs` after each push.
    Ordered {
        /// The Arrow timestamp column of the declared input schema.
        event_time_column: String,
    },
    /// Advances to `max(event_time) - max_out_of_orderness` after each push.
    BoundedOutOfOrderness {
        /// The Arrow timestamp column of the declared input schema.
        event_time_column: String,
        /// The positive delay, representable in nanoseconds.
        max_out_of_orderness: Duration,
    },
    /// Never advances; results finalize at `finish`.
    Disabled,
}

/// Rows emitted by one session call, by graph output ID.
///
/// Each output keeps emission order; outputs have no combined order.
/// Watermark, idle and end-of-input messages are not returned.
#[derive(Debug, Default)]
pub struct SessionOutput(BTreeMap<String, Vec<Batch>>);

impl SessionOutput {
    /// The batches emitted on `output`, in emission order; empty when none.
    pub fn batches(&self, output: &str) -> &[Batch] {
        self.0.get(output).map_or(&[], Vec::as_slice)
    }

    /// Whether the call emitted no batch on any output.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Consumes the output into batches by graph output ID.
    pub fn into_batches(self) -> BTreeMap<String, Vec<Batch>> {
        self.0
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Lifecycle {
    Open,
    Finished,
    Failed,
}

struct SessionInput {
    binding_id: String,
    /// The route's expected kind and exact schema.
    port: Port,
    sender: SessionEdgeSender,
    mode: WatermarkMode,
    progress: InputProgress,
}

enum WatermarkMode {
    Explicit,
    Generated {
        event_time: ResolvedEventTimeColumn,
        ordered: bool,
    },
    Disabled,
}

#[derive(Clone)]
struct InputProgress {
    next_sequence: Option<u64>,
    watermark: Option<EventTime>,
    generator: Option<GeneratedWatermarkState>,
    /// `Ordered` only: the largest event time pushed, in the declared unit.
    max_event_time: Option<i64>,
}

/// One validated call: messages for the input edge and the committed state.
struct Staged {
    messages: Vec<StreamMessage>,
    next: InputProgress,
    finishes: bool,
}

impl StreamSession {
    /// Consumes a compiled plan and prepares every operator.
    ///
    /// `watermarks` is keyed by the input ID; an absent input uses
    /// [`SessionWatermark::Explicit`].
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for an invalid
    /// configuration, an unsupported graph (fields `session.inputs`,
    /// `session.static_inputs`, `session.nodes.<id>`,
    /// `session.requirements.delivery.<output>`), or an invalid watermark
    /// (`session.watermarks.<key>` and its `event_time_column` or
    /// `max_out_of_orderness`).
    pub fn new(
        plan: StreamExecutionPlan,
        config: &StreamRuntimeConfig,
        watermarks: BTreeMap<String, SessionWatermark>,
    ) -> Result<Self> {
        let parts = admitted_parts(plan, config)?;
        let job = session_job(&parts.fingerprint)?;
        let runtime = session_runtime()?;
        let wired = session_block_on(&runtime, wire(parts, &job, watermarks));
        let (input, nodes, outputs) = match wired {
            Ok(wired) => wired,
            Err(error) => {
                release_session_runtime(runtime);
                return Err(error);
            }
        };
        let needs_async = nodes.iter().any(|node| !node.is_sync());
        let runtime = if needs_async {
            Some(runtime)
        } else {
            release_session_runtime(runtime);
            None
        };
        Ok(Self {
            runtime,
            job,
            input,
            nodes,
            outputs,
            lifecycle: Lifecycle::Open,
        })
    }

    /// The session's source binding IDs in plan order.
    pub fn input_ids(&self) -> Vec<&str> {
        vec![self.input.binding_id.as_str()]
    }

    /// The graph output IDs in plan order.
    pub fn output_ids(&self) -> Vec<&str> {
        self.outputs.iter().map(|(id, _)| id.as_str()).collect()
    }

    /// Whether an error after validation failed the session; later calls
    /// then return [`CalcFlowError::InvalidArgument`] at field `session`.
    pub fn is_failed(&self) -> bool {
        self.lifecycle == Lifecycle::Failed
    }

    #[cfg(test)]
    fn watermark_for_test(&self) -> Option<EventTime> {
        self.input.progress.watermark
    }

    #[cfg(test)]
    fn panic_on_next_dispatch_for_test(&mut self, node_id: &str) {
        self.nodes
            .iter_mut()
            .find(|node| node.node_id() == node_id)
            .expect("test node exists")
            .panic_on_next_dispatch();
    }

    /// Runs `batch`, then an optional watermark, through every operator and
    /// returns the rows this call emitted.
    ///
    /// The batch is stamped like a runner source batch (binding ID, a
    /// sequence starting at 0, original attributes). The input watermark
    /// advances to `watermark` or, for generated modes, to the larger of it
    /// and the generated candidate; an equal or absent watermark adds none.
    ///
    /// # Errors
    ///
    /// Validation errors leave the session unchanged:
    /// [`CalcFlowError::Compile`] when the batch differs from the input's
    /// declared kind or exact schema, [`CalcFlowError::EdgeBudgetExceeded`]
    /// when it exceeds `edge_budget`, and [`CalcFlowError::InvalidArgument`]
    /// for an unknown input, a regressing watermark, any watermark under
    /// [`SessionWatermark::Disabled`], or `Ordered` event time that is null,
    /// decreasing or at or before the current watermark. Any other error,
    /// including an operator failure or panic, fails the session. Lifecycle
    /// and Tokio-context errors use field `session`.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the session takes ownership of pushed batches; stamping shares their Arrow buffers"
    )]
    pub fn push(
        &mut self,
        input: &str,
        batch: Batch,
        watermark: Option<EventTime>,
    ) -> Result<SessionOutput> {
        self.check_callable()?;
        let staged = self.input.stage_push(input, &batch, watermark)?;
        self.run(staged)
    }

    /// Advances the input watermark without data and returns the rows it
    /// closes; an equal watermark is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`], leaving the session
    /// unchanged, for an unknown input, a regressing watermark or
    /// [`SessionWatermark::Disabled`]. Any operator error fails the session.
    pub fn advance(&mut self, input: &str, watermark: EventTime) -> Result<SessionOutput> {
        self.check_callable()?;
        let staged = self.input.stage_advance(input, watermark)?;
        self.run(staged)
    }

    /// Ends the input, flushes every operator and finishes the session,
    /// returning the flushed rows. A session with no pushes still finishes.
    ///
    /// # Errors
    ///
    /// Any operator error fails the session; lifecycle and Tokio-context
    /// errors use field `session`.
    pub fn finish(&mut self) -> Result<SessionOutput> {
        self.check_callable()?;
        self.run(Staged {
            messages: vec![StreamMessage::end_of_input()],
            next: self.input.progress.clone(),
            finishes: true,
        })
    }

    fn check_callable(&self) -> Result<()> {
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(session_error(
                "cannot run inside a Tokio runtime; call it from a blocking thread",
            ));
        }
        let message = match self.lifecycle {
            Lifecycle::Open => return Ok(()),
            Lifecycle::Finished => "already finished",
            Lifecycle::Failed => "failed earlier; create a new session",
        };
        Err(session_error(message))
    }

    fn run(&mut self, staged: Staged) -> Result<SessionOutput> {
        match self.drive(staged.messages) {
            Ok(output) => {
                self.input.progress = staged.next;
                if staged.finishes {
                    self.lifecycle = Lifecycle::Finished;
                }
                Ok(output)
            }
            Err(error) => {
                self.lifecycle = Lifecycle::Failed;
                Err(error)
            }
        }
    }

    #[allow(
        clippy::large_futures,
        reason = "the drain future is polled in place by block_on; boxing would allocate per call"
    )]
    fn drive(&mut self, messages: Vec<StreamMessage>) -> Result<SessionOutput> {
        let Self {
            runtime,
            input,
            nodes,
            outputs,
            ..
        } = self;
        for message in messages {
            // Staging validated the data envelope; control messages carry no rows.
            input.sender.enqueue(message);
        }
        if nodes.iter().all(CallerDrivenOperator::is_sync) {
            debug_assert!(runtime.is_none());
            drain_in_order_sync(nodes)?;
        } else {
            let runtime = runtime.as_ref().ok_or_else(|| CalcFlowError::Internal {
                message: "session runtime was released".into(),
            })?;
            session_block_on(runtime, drain_in_order(nodes))?;
        }
        collect_outputs(outputs)
    }
}

/// Runs every node to quiescence in topological order.
#[allow(
    clippy::large_futures,
    reason = "one node drain is polled at a time; boxing would allocate per node"
)]
async fn drain_in_order(nodes: &mut [CallerDrivenOperator]) -> Result<()> {
    for node in nodes {
        node.drain().await?;
    }
    Ok(())
}

/// Takes every batch that reached a graph output during this call.
fn collect_outputs(outputs: &mut [(String, OperatorEdgeReceiver)]) -> Result<SessionOutput> {
    let mut emitted = BTreeMap::new();
    for (output, receiver) in outputs.iter_mut() {
        let mut batches = Vec::new();
        while let Some(message) = receiver.try_recv()? {
            batches.extend(message.into_data());
        }
        if !batches.is_empty() {
            emitted.insert(output.clone(), batches);
        }
    }
    Ok(SessionOutput(emitted))
}

/// Validates the configuration and plan, applies SQL budgets like runners
/// do, and admits the decomposed nodes.
fn admitted_parts(
    mut plan: StreamExecutionPlan,
    config: &StreamRuntimeConfig,
) -> Result<StreamRuntimePlanParts> {
    config.validate()?;
    admit_plan(&plan)?;
    plan.set_sql_budgets(config)?;
    let parts = plan.into_runtime_parts(config.edge_budget)?;
    admit_nodes(&parts.nodes)?;
    Ok(parts)
}

/// A checkpoint-off job context built like the continuous runner's.
fn session_job(fingerprint: &str) -> Result<StreamJobContext> {
    Ok(StreamJobContext::new(
        crate::continuous::allocate_job_id()?,
        fingerprint.to_owned(),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
    .with_checkpointing(false))
}

/// The session-owned current-thread runtime: timers only, no I/O driver.
fn session_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .map_err(|error| CalcFlowError::Internal {
            message: format!("session runtime could not start: {error}"),
        })
}

#[cfg(test)]
thread_local! {
    static BLOCK_ON_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FUTURE_POLLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn session_block_on<T, F>(runtime: &tokio::runtime::Runtime, future: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        let mut future = Box::pin(future);
        let waker = futures::task::noop_waker();
        let mut context = std::task::Context::from_waker(&waker);
        #[cfg(test)]
        FUTURE_POLLS.with(|polls| polls.set(polls.get() + 1));
        return match std::future::Future::poll(future.as_mut(), &mut context) {
            std::task::Poll::Ready(result) => result,
            std::task::Poll::Pending => Err(session_error(
                "cannot prepare pending work inside a Tokio runtime; use a blocking thread",
            )),
        };
    }
    #[cfg(test)]
    {
        BLOCK_ON_CALLS.with(|calls| calls.set(calls.get() + 1));
        let mut future = Box::pin(future);
        return runtime.block_on(std::future::poll_fn(|context| {
            FUTURE_POLLS.with(|polls| polls.set(polls.get() + 1));
            std::future::Future::poll(future.as_mut(), context)
        }));
    }
    #[cfg(not(test))]
    runtime.block_on(future)
}

fn release_session_runtime(runtime: tokio::runtime::Runtime) {
    if tokio::runtime::Handle::try_current().is_ok() {
        runtime.shutdown_background();
    } else {
        drop(runtime);
    }
}

#[cfg(test)]
fn reset_drive_counters_for_test() {
    BLOCK_ON_CALLS.with(|calls| calls.set(0));
    FUTURE_POLLS.with(|polls| polls.set(0));
}

#[cfg(test)]
fn drive_counters_for_test() -> (usize, usize) {
    (
        BLOCK_ON_CALLS.with(std::cell::Cell::get),
        FUTURE_POLLS.with(std::cell::Cell::get),
    )
}

fn session_error(message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "session".into(),
        message: message.into(),
    }
}

impl Drop for StreamSession {
    fn drop(&mut self) {
        self.job.cancellation().cancel();
        self.nodes.clear();
        if let Some(runtime) = self.runtime.take() {
            release_session_runtime(runtime);
        }
    }
}

impl SessionInput {
    fn stage_advance(&self, input: &str, watermark: EventTime) -> Result<Staged> {
        self.check_input(input)?;
        self.check_explicit(Some(watermark))?;
        let mut next = self.progress.clone();
        let messages = advance_watermark(&mut next, Some(watermark))
            .map(StreamMessage::watermark)
            .into_iter()
            .collect();
        Ok(Staged {
            messages,
            next,
            finishes: false,
        })
    }

    /// Validates one push before any operator runs; the session is unchanged
    /// when this fails.
    fn stage_push(
        &self,
        input: &str,
        batch: &Batch,
        watermark: Option<EventTime>,
    ) -> Result<Staged> {
        self.check_input(input)?;
        self.check_explicit(watermark)?;
        self.port
            .validate(batch, &format!("session.inputs.{}", self.binding_id))?;
        let mut next = self.progress.clone();
        let generated = self.stage_generated(batch, &mut next)?;
        let (sequence, data, cost) =
            sequenced_source_message(&self.binding_id, next.next_sequence, batch, None)?;
        self.sender.check_cost(&cost)?;
        next.next_sequence = sequence.checked_add(1);
        let mut messages = vec![data];
        messages.extend(
            advance_watermark(&mut next, generated.max(watermark)).map(StreamMessage::watermark),
        );
        Ok(Staged {
            messages,
            next,
            finishes: false,
        })
    }

    /// Checks `Ordered` time and computes the generated candidate on the
    /// staged copy of the input progress.
    fn stage_generated(
        &self,
        batch: &Batch,
        next: &mut InputProgress,
    ) -> Result<Option<EventTime>> {
        let WatermarkMode::Generated {
            event_time,
            ordered,
        } = &self.mode
        else {
            return Ok(None);
        };
        if *ordered {
            next.max_event_time = ordered_maximum(&self.binding_id, batch, event_time, next)?;
        }
        let generator = next
            .generator
            .as_mut()
            .ok_or_else(|| CalcFlowError::Internal {
                message: "generated session watermark has no generator".into(),
            })?;
        generator
            .observe_batch(batch, &self.binding_id)
            .and_then(|()| generator.on_timer(&self.binding_id))
            .map_err(|error| session_watermark_error(&self.binding_id, error))
    }

    fn check_input(&self, input: &str) -> Result<()> {
        if input == self.binding_id {
            return Ok(());
        }
        Err(CalcFlowError::InvalidArgument {
            field: format!("session.inputs.{input}"),
            message: format!("is not the session input {:?}", self.binding_id),
        })
    }

    /// Rejects an explicit watermark under `Disabled` or below the current
    /// one; an equal watermark is a no-op.
    fn check_explicit(&self, watermark: Option<EventTime>) -> Result<()> {
        let Some(watermark) = watermark else {
            return Ok(());
        };
        let message = if matches!(self.mode, WatermarkMode::Disabled) {
            "Disabled watermarks reject explicit watermarks"
        } else if self
            .progress
            .watermark
            .is_some_and(|current| watermark < current)
        {
            "must not regress below the current watermark"
        } else {
            return Ok(());
        };
        Err(CalcFlowError::InvalidArgument {
            field: format!("session.inputs.{}.watermark", self.binding_id),
            message: message.into(),
        })
    }
}

/// Records and returns `candidate` when it advances the input watermark.
fn advance_watermark(next: &mut InputProgress, candidate: Option<EventTime>) -> Option<EventTime> {
    let candidate =
        candidate.filter(|candidate| next.watermark.is_none_or(|current| *candidate > current))?;
    next.watermark = Some(candidate);
    Some(candidate)
}

/// Checks `Ordered` event time across every record batch and returns the new
/// maximum. Values compare in the declared unit; the first row is compared
/// with the watermark after the runtime's import to microseconds.
fn ordered_maximum(
    binding_id: &str,
    batch: &Batch,
    event_time: &ResolvedEventTimeColumn,
    progress: &InputProgress,
) -> Result<Option<i64>> {
    let field = format!("session.inputs.{binding_id}.event_time");
    let mut maximum = progress.max_event_time;
    let mut first = None;
    for record in batch.table_payload()?.batches() {
        let values = event_time_values(record, event_time, &field)?;
        maximum = nondecreasing_maximum(values, maximum, &field)?;
        first = first.or_else(|| values.first().copied());
    }
    if let (Some(first), Some(watermark)) = (first, progress.watermark) {
        check_after_watermark(first, watermark, event_time.unit, &field)?;
    }
    Ok(maximum)
}

fn invalid_event_time(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

/// The non-null event-time values of one record batch, in its declared unit.
fn event_time_values<'a>(
    record: &'a RecordBatch,
    event_time: &ResolvedEventTimeColumn,
    field: &str,
) -> Result<&'a [i64]> {
    let column = record.column(event_time.index);
    if column.null_count() != 0 {
        return Err(invalid_event_time(field, "expected non-null event time"));
    }
    timestamp_values(column.as_ref(), event_time.unit).ok_or_else(|| {
        invalid_event_time(
            field,
            "record batch column type differs from the declared schema",
        )
    })
}

/// Extends `maximum` over `values`, rejecting any decrease.
fn nondecreasing_maximum(values: &[i64], maximum: Option<i64>, field: &str) -> Result<Option<i64>> {
    let mut previous = maximum;
    for &value in values {
        if previous.is_some_and(|previous| value < previous) {
            return Err(invalid_event_time(
                field,
                "expected nondecreasing event time; use BoundedOutOfOrderness for out-of-order input",
            ));
        }
        previous = Some(value);
    }
    Ok(previous)
}

/// Operators see the runtime's import, which floors nanoseconds to
/// microseconds, so the imported first row must follow the watermark.
fn check_after_watermark(
    first: i64,
    watermark: EventTime,
    unit: ArrowTimestampUnit,
    field: &str,
) -> Result<()> {
    let first = EventTime::import_timestamp_unit(first, time_unit(unit), field).map_err(
        |error| match error {
            CalcFlowError::InvalidArgument { message, .. } => invalid_event_time(field, &message),
            error => error,
        },
    )?;
    if first <= watermark {
        return Err(invalid_event_time(
            field,
            "expected event time after the current watermark; earlier rows would be late",
        ));
    }
    Ok(())
}

fn timestamp_values(column: &dyn Array, unit: ArrowTimestampUnit) -> Option<&[i64]> {
    let column = column.as_any();
    match unit {
        ArrowTimestampUnit::Second => column
            .downcast_ref::<TimestampSecondArray>()
            .map(|array| array.values().as_ref()),
        ArrowTimestampUnit::Millisecond => column
            .downcast_ref::<TimestampMillisecondArray>()
            .map(|array| array.values().as_ref()),
        ArrowTimestampUnit::Microsecond => column
            .downcast_ref::<TimestampMicrosecondArray>()
            .map(|array| array.values().as_ref()),
        ArrowTimestampUnit::Nanosecond => column
            .downcast_ref::<TimestampNanosecondArray>()
            .map(|array| array.values().as_ref()),
    }
}

const fn time_unit(unit: ArrowTimestampUnit) -> TimeUnit {
    match unit {
        ArrowTimestampUnit::Second => TimeUnit::Second,
        ArrowTimestampUnit::Millisecond => TimeUnit::Millisecond,
        ArrowTimestampUnit::Microsecond => TimeUnit::Microsecond,
        ArrowTimestampUnit::Nanosecond => TimeUnit::Nanosecond,
    }
}

/// Reports generator failures at the session's watermark field paths.
fn session_watermark_error(binding_id: &str, error: CalcFlowError) -> CalcFlowError {
    let CalcFlowError::InvalidArgument { field, message } = error else {
        return error;
    };
    let field = field
        .strip_prefix(&format!("sources.{binding_id}.watermark_policy."))
        .map_or(field.clone(), |rest| {
            format!("session.watermarks.{binding_id}.{rest}")
        });
    CalcFlowError::InvalidArgument { field, message }
}

fn unsupported(field: String, message: String) -> CalcFlowError {
    CalcFlowError::InvalidArgument { field, message }
}

/// Admits plans with one dynamic input, no static inputs, no native array
/// sequence nodes and no exactly-once outputs.
fn admit_plan(plan: &StreamExecutionPlan) -> Result<()> {
    let inputs = plan.source_binding_ids().len();
    if inputs != 1 {
        return Err(unsupported(
            "session.inputs".into(),
            format!(
                "StreamSession requires exactly one dynamic input, found {inputs}; use StreamingRunner"
            ),
        ));
    }
    if !plan.static_input_ids().is_empty() {
        return Err(unsupported(
            "session.static_inputs".into(),
            "static inputs are not supported by StreamSession; use StreamingRunner".into(),
        ));
    }
    if let Some(node_id) = plan.native_sequence_node_ids().next() {
        return Err(unsupported(
            format!("session.nodes.{node_id}"),
            "native array sequence nodes are not supported by StreamSession; use StreamingRunner"
                .into(),
        ));
    }
    if let Some((output, _)) = plan
        .requirements()
        .delivery
        .iter()
        .find(|(_, guarantee)| **guarantee == DeliveryGuarantee::ExactlyOnce)
    {
        return Err(unsupported(
            format!("session.requirements.delivery.{output}"),
            "exactly-once delivery is not supported by StreamSession; use StreamingRunner".into(),
        ));
    }
    Ok(())
}

/// Admits `Rolling` (except indicator and MACD layouts), `Expression` and
/// `Union` nodes; every other kind stays on `StreamingRunner`.
fn admit_nodes(nodes: &[RuntimeStreamNode]) -> Result<()> {
    nodes
        .iter()
        .try_for_each(|node| match node_rejection(&node.operator) {
            Some(message) => Err(unsupported(
                format!("session.nodes.{}", node.node_id),
                message,
            )),
            None => Ok(()),
        })
}

/// Why a node stays on `StreamingRunner`; `None` admits it. The match is
/// exhaustive so a new operator kind must choose explicitly.
fn node_rejection(operator: &CompiledStreamOperator) -> Option<String> {
    let kind = match operator {
        CompiledStreamOperator::Expression(_)
        | CompiledStreamOperator::SyncExpression(_)
        | CompiledStreamOperator::Union(_) => return None,
        CompiledStreamOperator::Rolling(rolling) => {
            return rolling_rejection(rolling.state_layout_version());
        }
        CompiledStreamOperator::Sql(_)
        | CompiledStreamOperator::SnapshotSql(_)
        | CompiledStreamOperator::BoundedOrder(_)
        | CompiledStreamOperator::Distinct(_) => relational_kind(operator),
        CompiledStreamOperator::Window(_)
        | CompiledStreamOperator::CrossSection(_)
        | CompiledStreamOperator::StreamJoin(_)
        | CompiledStreamOperator::StreamAsofJoin(_) => keyed_kind(operator),
        CompiledStreamOperator::External(_) => "external",
        CompiledStreamOperator::CheckpointLoan => "checkpoint loan",
    };
    Some(format!(
        "{kind} nodes are not supported by StreamSession; use StreamingRunner"
    ))
}

fn rolling_rejection(layout: u32) -> Option<String> {
    matches!(
        layout,
        crate::ROLLING_INDICATOR_STATE_LAYOUT_VERSION | crate::ROLLING_MACD_STATE_LAYOUT_VERSION
    )
    .then(|| {
        format!(
            "rolling indicator state layout {layout} is not supported by StreamSession; use StreamingRunner"
        )
    })
}

/// Names the SQL-planned kinds grouped in `node_rejection`.
fn relational_kind(operator: &CompiledStreamOperator) -> &'static str {
    match operator {
        CompiledStreamOperator::Sql(_) => "SQL",
        CompiledStreamOperator::SnapshotSql(_) => "snapshot SQL",
        CompiledStreamOperator::BoundedOrder(_) => "bounded ORDER BY",
        _ => "DISTINCT",
    }
}

/// Names the keyed-state kinds grouped in `node_rejection`.
fn keyed_kind(operator: &CompiledStreamOperator) -> &'static str {
    match operator {
        CompiledStreamOperator::Window(_) => "window",
        CompiledStreamOperator::CrossSection(_) => "cross-section",
        CompiledStreamOperator::StreamJoin(_) => "join",
        _ => "ASOF join",
    }
}

/// Resolves the input's watermark mode against its declared exact schema.
fn input_mode(
    binding_id: &str,
    schema: Option<&SchemaRef>,
    mut watermarks: BTreeMap<String, SessionWatermark>,
) -> Result<(WatermarkMode, Option<GeneratedWatermarkState>)> {
    if let Some(key) = watermarks.keys().find(|key| key.as_str() != binding_id) {
        return Err(unsupported(
            format!("session.watermarks.{key}"),
            format!("is not the session input {binding_id:?}"),
        ));
    }
    match watermarks
        .remove(binding_id)
        .unwrap_or(SessionWatermark::Explicit)
    {
        SessionWatermark::Explicit => Ok((WatermarkMode::Explicit, None)),
        SessionWatermark::Disabled => Ok((WatermarkMode::Disabled, None)),
        SessionWatermark::Ordered { event_time_column } => generated_mode(
            binding_id,
            schema,
            &event_time_column,
            Duration::from_micros(1),
            true,
        ),
        SessionWatermark::BoundedOutOfOrderness {
            max_out_of_orderness,
            ..
        } if !is_logical_nanos_duration(max_out_of_orderness) => Err(unsupported(
            format!("session.watermarks.{binding_id}.max_out_of_orderness"),
            "must be positive and representable as logical nanoseconds".into(),
        )),
        SessionWatermark::BoundedOutOfOrderness {
            event_time_column,
            max_out_of_orderness,
        } => generated_mode(
            binding_id,
            schema,
            &event_time_column,
            max_out_of_orderness,
            false,
        ),
    }
}

fn generated_mode(
    binding_id: &str,
    schema: Option<&SchemaRef>,
    column: &str,
    delay: Duration,
    ordered: bool,
) -> Result<(WatermarkMode, Option<GeneratedWatermarkState>)> {
    let path = format!("session.watermarks.{binding_id}.event_time_column");
    let schema = schema.ok_or_else(|| CalcFlowError::InvalidArgument {
        field: path.clone(),
        message: "generated watermarks require a declared exact input schema".into(),
    })?;
    let event_time = resolve_event_time_column(path, schema, &Arc::from(column))?;
    let generator = GeneratedWatermarkState::new(event_time.clone(), delay);
    Ok((
        WatermarkMode::Generated {
            event_time,
            ordered,
        },
        Some(generator),
    ))
}

type Wiring = (
    SessionInput,
    Vec<CallerDrivenOperator>,
    Vec<(String, OperatorEdgeReceiver)>,
);

async fn wire(
    parts: StreamRuntimePlanParts,
    job: &StreamJobContext,
    watermarks: BTreeMap<String, SessionWatermark>,
) -> Result<Wiring> {
    let StreamRuntimePlanParts {
        nodes,
        edges,
        source_routes,
        sink_routes,
        ..
    } = parts;
    let (mut senders, mut receivers) = session_edges(&edges)?;
    let nodes = prepare_nodes(nodes, job, &mut senders, &mut receivers).await?;
    let input = session_input(source_routes, &mut senders, watermarks)?;
    let outputs = sink_routes
        .into_values()
        .map(|route| Ok((route.output_id, take_edge(&mut receivers, &route.edge_id)?)))
        .collect::<Result<Vec<_>>>()?;
    if !senders.is_empty() || !receivers.is_empty() {
        return Err(CalcFlowError::Internal {
            message: "session topology left unowned edge endpoints".into(),
        });
    }
    Ok((input, nodes, outputs))
}

type SessionEdges = (
    BTreeMap<String, OperatorEdgeSender>,
    BTreeMap<String, OperatorEdgeReceiver>,
);

/// One session FIFO per runtime edge, with the edge's configured budget.
fn session_edges(edges: &BTreeMap<String, RuntimeEdge>) -> Result<SessionEdges> {
    let mut senders = BTreeMap::new();
    let mut receivers = BTreeMap::new();
    for (edge_id, edge) in edges {
        let (sender, receiver) = session_edge(edge_id.clone(), edge.budget)?;
        senders.insert(edge_id.clone(), OperatorEdgeSender::Session(sender));
        receivers.insert(edge_id.clone(), OperatorEdgeReceiver::Session(receiver));
    }
    Ok((senders, receivers))
}

/// Prepares and resets every node in the plan's topological order.
async fn prepare_nodes(
    nodes: Vec<RuntimeStreamNode>,
    job: &StreamJobContext,
    senders: &mut BTreeMap<String, OperatorEdgeSender>,
    receivers: &mut BTreeMap<String, OperatorEdgeReceiver>,
) -> Result<Vec<CallerDrivenOperator>> {
    let metrics = MetricsRecorder::default();
    let mut prepared = Vec::with_capacity(nodes.len());
    for (producer, node) in nodes.into_iter().enumerate() {
        let inputs = prepare_node(node, job, &metrics, senders, receivers, producer)?;
        prepared.push(CallerDrivenOperator::prepare(inputs).await?);
    }
    Ok(prepared)
}

/// The one dynamic input: its route port, edge sender and watermark mode.
fn session_input(
    source_routes: BTreeMap<String, RuntimeSourceRoute>,
    senders: &mut BTreeMap<String, OperatorEdgeSender>,
    watermarks: BTreeMap<String, SessionWatermark>,
) -> Result<SessionInput> {
    let (binding_id, route) =
        source_routes
            .into_iter()
            .next()
            .ok_or_else(|| CalcFlowError::Internal {
                message: "session plan has no source route".into(),
            })?;
    let (mode, generator) = input_mode(&binding_id, route.expected_schema.as_ref(), watermarks)?;
    let port = Port::with_schema_ref(
        &route.target.port,
        route.expected_kind,
        true,
        route.expected_schema.clone(),
    )?;
    let sender = take_edge(senders, &route.edge_id)?.into_session()?;
    Ok(SessionInput {
        binding_id,
        port,
        sender,
        mode,
        progress: InputProgress {
            next_sequence: Some(0),
            watermark: None,
            generator,
            max_event_time: None,
        },
    })
}

fn take_edge<T>(edges: &mut BTreeMap<String, T>, edge_id: &str) -> Result<T> {
    edges
        .remove(edge_id)
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("session edge {edge_id:?} is missing"),
        })
}

/// Builds task inputs like the runner's `prepare_operator_task`, without
/// entity work, SQL recovery, checkpoints or restore; gates start released.
fn prepare_node(
    node: RuntimeStreamNode,
    job: &StreamJobContext,
    metrics: &MetricsRecorder,
    senders: &mut BTreeMap<String, OperatorEdgeSender>,
    receivers: &mut BTreeMap<String, OperatorEdgeReceiver>,
    producer: usize,
) -> Result<OperatorTaskInputs> {
    let node_id = node.operator_id.as_str().to_owned();
    let ingresses = take_node_ingresses(&node, receivers)?;
    let outputs = take_node_outputs(&node, senders)?;
    for sender in outputs.values().flatten() {
        sender.set_session_producer(producer);
    }
    let (_, entry_gate) = watch::channel(true);
    let (_, data_gate) = watch::channel(true);
    let (entry_ack, _) = mpsc::unbounded_channel();
    Ok(OperatorTaskInputs {
        entity_work: None,
        sql_recovery: None,
        context: job.for_node(&node_id)?,
        metrics: metrics.operator(&node_id)?,
        node_id,
        operator: node.operator,
        checkpoint_capability: node.checkpoint_capability,
        ingresses,
        outputs,
        output_ports: node.output_ports,
        late_output_ports: node.late_output_ports,
        progress: OperatorProgress::with_optional_rolling_metrics(None),
        entry_gate,
        entry_ack,
        data_gate,
        launch_cancel: CancellationToken::new(),
        checkpoint: None,
        restore: None,
    })
}

#[cfg(test)]
mod tests;
