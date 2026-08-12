//! Source-driven continuous streaming lifecycle.
//!
//! This module is the integration-only Rust façade for the A6 continuous
//! runtime. It intentionally lives below `calc_flow::continuous` while the
//! crate-root breaking cutover remains deferred.
//!
//! A runner is a one-shot owner. Calling [`StreamingRunner::start`] consumes
//! it and returns the sole [`StreamingJob`] lifecycle owner.
//!
//! ```compile_fail
//! # async fn reuse(runner: calc_flow::continuous::StreamingRunner) {
//! let _first = runner.start().await;
//! let _second = runner.start().await; // the runner was moved
//! # }
//! ```
//!
//! A job is likewise the sole lifecycle owner and cannot be cloned.
//!
//! ```compile_fail
//! # fn clone_job(job: calc_flow::continuous::StreamingJob) {
//! let _second_owner = job.clone();
//! # }
//! ```
//!
//! Connector traits are object-safe and require `Send`; a connector that
//! retains non-`Send` state cannot be bound to the runtime.
//!
//! ```compile_fail
//! use std::rc::Rc;
//! use async_trait::async_trait;
//! use calc_flow::continuous::{Cursor, SourceCapabilities, SourceEvent, StreamSource};
//!
//! struct LocalOnly(Rc<()>);
//!
//! #[async_trait]
//! impl StreamSource for LocalOnly {
//!     fn capabilities(&self) -> SourceCapabilities { todo!() }
//!     async fn open(&mut self, _: Option<Cursor>) -> calc_flow::Result<()> { Ok(()) }
//!     async fn next(&mut self) -> calc_flow::Result<Option<SourceEvent>> { Ok(None) }
//!     async fn close(&mut self) -> calc_flow::Result<()> { Ok(()) }
//! }
//! ```

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::PathBuf,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;

use crate::{
    Batch, CalcFlowError, CancellationToken, Epoch, JsonMap, ManifestIngressState, Result,
    RetentionClass, StreamExecutionPlan, StreamJobContext, StreamRuntimeConfig,
    runtime::streaming::{
        checkpoint::ManagedCheckpointRuntime as InternalManagedCheckpointRuntime,
        job::{
            ContinuousJobSpec, M2DeliveryMode, NamedSinkBinding, NamedSourceBinding,
            OrdinarySinkBinding, OrdinaryStreamSink, OwningContinuousJob,
            TransactionalStreamSink as InternalTransactionalStreamSink,
        },
        progress::{
            DeclaredSchema, NativeWatermarkCapability as InternalNativeWatermarkCapability,
            ReplayPositioningCapability, WatermarkPolicy as InternalWatermarkPolicy,
        },
        projection,
        runner::{OneShotContinuousRunner, StartFailure},
        source_task::{
            Cursor as InternalCursor, SourceBinding as InternalSourceBinding,
            SourceCapabilities as InternalSourceCapabilities,
            SourceDeliveryCapability as InternalSourceDeliveryCapability,
            SourceEvent as InternalSourceEvent, StreamSource as InternalStreamSource,
        },
    },
};

pub use crate::runtime::streaming::projection::{
    CheckpointPhase, CheckpointStatus, ComponentKind, EdgeStatus, JobOutcome, JobState, JobStatus,
    OperatorStatus, OutputDeliveryStatus, SinkDelivery, SinkStatus, SourceStatus, StreamingError,
    StreamingErrorCategory, TerminalCause,
};

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

/// Source-defined position owned by one stable source binding.
#[derive(Clone, Eq, PartialEq)]
pub struct Cursor {
    inner: InternalCursor,
}

impl Cursor {
    /// Constructs a cursor already owned by `source_id`.
    ///
    /// # Errors
    ///
    /// Returns a validation error when the source ID, order key, or payload is invalid.
    pub fn new(source_id: impl Into<String>, order: Vec<u8>, payload: JsonMap) -> Result<Self> {
        InternalCursor::new(&source_id.into(), order, payload).map(|inner| Self { inner })
    }

    /// Constructs a cursor whose owner is assigned during runner admission.
    ///
    /// # Errors
    ///
    /// Returns a validation error when the order key or payload is invalid.
    pub fn unbound(order: Vec<u8>, payload: JsonMap) -> Result<Self> {
        InternalCursor::unbound(order, payload).map(|inner| Self { inner })
    }

    /// Returns the stable source owner, when already assigned.
    pub fn source_id(&self) -> Option<&str> {
        self.inner.source_id()
    }

    /// Returns the connector-defined stable ordering key.
    pub fn order(&self) -> &[u8] {
        self.inner.order()
    }

    /// Returns the connector-defined bounded JSON position.
    pub fn payload(&self) -> &JsonMap {
        self.inner.payload()
    }
}

impl fmt::Debug for Cursor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Cursor")
            .field("source_id", &self.source_id())
            .field("position", &"<redacted>")
            .finish()
    }
}

/// Exact replay positioning offered by a source connector.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayPositioning {
    /// The source pauses, reports, and later seeks to the exact accepted cut.
    ExactPauseReportAndSeek,
    /// Exact recovery is unavailable.
    Unsupported,
}

/// Whether a source emits native watermarks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeWatermarkCapability {
    /// The connector never emits native watermarks.
    NeverEmits,
    /// The connector always emits native watermarks.
    EmitsNative,
    /// Native emission can be enabled or disabled by the runtime.
    RuntimeToggleable,
    /// The connector cannot prove its native watermark behavior.
    Unknown,
}

/// Whether admitted source data can be lost before runtime observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceDeliveryCapability {
    /// Accepted events are retained until an exact cursor can replay them.
    Lossless,
    /// Accepted events may be lost before runtime observation.
    Lossy,
}

/// Schema evidence frozen before a source opens.
#[derive(Clone, Debug)]
pub enum SourceSchema {
    /// The connector emits exactly this Arrow schema.
    Exact(SchemaRef),
    /// The connector cannot freeze one exact schema before opening.
    DynamicOrUnknown,
}

/// Complete source capability descriptor sampled once during preflight.
#[derive(Clone, Debug)]
pub struct SourceCapabilities {
    /// Exact replay protocol implemented by the connector.
    pub replay_positioning: ReplayPositioning,
    /// Whether the source can lose accepted events.
    pub delivery: SourceDeliveryCapability,
    /// Maximum rows in one emitted batch.
    pub max_batch_rows: usize,
    /// Maximum bytes in one emitted batch.
    pub max_batch_bytes: usize,
    /// Schema evidence frozen during preflight.
    pub schema: SourceSchema,
    /// Native watermark behavior frozen during preflight.
    pub native_watermarks: NativeWatermarkCapability,
}

/// One connector event; checkpoint barriers remain runtime-owned.
#[derive(Clone, Debug)]
pub enum SourceEvent {
    /// One data batch and its replay position.
    Data { batch: Batch, cursor: Cursor },
    /// A source-native event-time watermark.
    Watermark(crate::EventTime),
    /// A temporary idle observation that does not end the source.
    Idle,
}

/// Watermark behavior attached immutably to one source binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WatermarkPolicy {
    /// Accept source-native watermarks without runtime generation.
    SourceProvided,
    /// Generate watermarks from an event-time column and a bounded delay.
    BoundedOutOfOrderness {
        event_time_column: String,
        max_out_of_orderness: Duration,
        emit_interval: Duration,
        idle_timeout: Option<Duration>,
    },
    /// Disable watermark advancement, optionally tracking source idleness.
    Disabled { idle_timeout: Option<Duration> },
}

impl Default for WatermarkPolicy {
    fn default() -> Self {
        Self::SourceProvided
    }
}

/// Async lifecycle contract for a continuous source connector.
#[async_trait]
pub trait StreamSource: Send {
    /// Returns the descriptor that the runtime samples once before `open`.
    fn capabilities(&self) -> SourceCapabilities;
    /// Opens at the beginning or at an exact owned recovery cursor.
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()>;
    /// Produces the next event, or `None` when the source has ended.
    async fn next(&mut self) -> Result<Option<SourceEvent>>;
    /// Releases connector resources.
    async fn close(&mut self) -> Result<()>;
}

struct SourceAdapter<S> {
    source: S,
    capabilities: OnceLock<SourceCapabilities>,
}

impl<S: StreamSource> SourceAdapter<S> {
    fn capabilities(&self) -> &SourceCapabilities {
        self.capabilities.get_or_init(|| self.source.capabilities())
    }
}

#[async_trait]
impl<S: StreamSource> InternalStreamSource for SourceAdapter<S> {
    async fn open(&mut self, cursor: Option<InternalCursor>) -> Result<()> {
        self.source.open(cursor.map(|inner| Cursor { inner })).await
    }

    async fn next(&mut self) -> Result<Option<InternalSourceEvent>> {
        self.source.next().await.map(|event| {
            event.map(|event| match event {
                SourceEvent::Data { batch, cursor } => InternalSourceEvent::Data {
                    batch,
                    cursor: cursor.inner,
                },
                SourceEvent::Watermark(value) => InternalSourceEvent::Watermark(value),
                SourceEvent::Idle => InternalSourceEvent::Idle,
            })
        })
    }

    async fn close(&mut self) -> Result<()> {
        self.source.close().await
    }

    fn capabilities(&self) -> InternalSourceCapabilities {
        let capabilities = self.capabilities();
        InternalSourceCapabilities {
            replayable: capabilities.replay_positioning
                == ReplayPositioning::ExactPauseReportAndSeek,
            max_batch_rows: capabilities.max_batch_rows,
            max_batch_bytes: capabilities.max_batch_bytes,
        }
    }

    fn delivery_capability(&self) -> InternalSourceDeliveryCapability {
        match self.capabilities().delivery {
            SourceDeliveryCapability::Lossless => InternalSourceDeliveryCapability::Lossless,
            SourceDeliveryCapability::Lossy => InternalSourceDeliveryCapability::Lossy,
        }
    }

    fn declared_schema(&self) -> DeclaredSchema {
        match &self.capabilities().schema {
            SourceSchema::Exact(schema) => DeclaredSchema::Known(schema.clone()),
            SourceSchema::DynamicOrUnknown => DeclaredSchema::DynamicOrUnknown,
        }
    }

    fn native_watermark_capability(&self) -> InternalNativeWatermarkCapability {
        match self.capabilities().native_watermarks {
            NativeWatermarkCapability::NeverEmits => InternalNativeWatermarkCapability::NeverEmits,
            NativeWatermarkCapability::EmitsNative => {
                InternalNativeWatermarkCapability::EmitsNative
            }
            NativeWatermarkCapability::RuntimeToggleable => {
                InternalNativeWatermarkCapability::RuntimeToggleable
            }
            NativeWatermarkCapability::Unknown => InternalNativeWatermarkCapability::Unknown,
        }
    }

    fn replay_positioning_capability(&self) -> Option<ReplayPositioningCapability> {
        Some(match self.capabilities().replay_positioning {
            ReplayPositioning::ExactPauseReportAndSeek => {
                ReplayPositioningCapability::ExactPauseReportAndSeek
            }
            ReplayPositioning::Unsupported => ReplayPositioningCapability::Unsupported,
        })
    }
}

/// Owned source connector plus its immutable watermark policy.
pub struct SourceBinding {
    inner: InternalSourceBinding,
}

impl SourceBinding {
    /// Owns a source connector using the default source-provided watermark policy.
    pub fn new(source: impl StreamSource + 'static) -> Self {
        let adapter = SourceAdapter {
            source,
            capabilities: OnceLock::new(),
        };
        let inner = InternalSourceBinding::unconfigured(Box::new(adapter));
        Self { inner }
    }

    #[must_use]
    /// Replaces the immutable watermark policy owned by this binding.
    pub fn with_watermark_policy(mut self, policy: WatermarkPolicy) -> Self {
        self.inner = self.inner.with_watermark_policy(match policy {
            WatermarkPolicy::SourceProvided => InternalWatermarkPolicy::SourceProvided,
            WatermarkPolicy::BoundedOutOfOrderness {
                event_time_column,
                max_out_of_orderness,
                emit_interval,
                idle_timeout,
            } => InternalWatermarkPolicy::BoundedOutOfOrderness {
                event_time_column: event_time_column.into(),
                max_out_of_orderness,
                emit_interval,
                idle_timeout,
            },
            WatermarkPolicy::Disabled { idle_timeout } => {
                InternalWatermarkPolicy::Disabled { idle_timeout }
            }
        });
        self
    }
}

/// Async lifecycle contract for an ordinary at-least-once sink.
#[async_trait]
pub trait StreamSink: Send {
    /// Opens connector resources.
    async fn open(&mut self) -> Result<()>;
    /// Writes one admitted batch.
    async fn write(&mut self, batch: &Batch) -> Result<()>;
    /// Releases connector resources.
    async fn close(&mut self) -> Result<()>;
}

/// Sink-scoped recovery evidence created by the runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SinkRecovery {
    epoch: Epoch,
    terminal: bool,
    delivery: SinkDelivery,
    pre_commit: JsonMap,
}

impl SinkRecovery {
    /// Returns the selected recovery epoch.
    pub const fn epoch(&self) -> Epoch {
        self.epoch
    }
    /// Returns whether the selected checkpoint represents terminal progress.
    pub const fn terminal(&self) -> bool {
        self.terminal
    }
    /// Returns this sink's frozen delivery evidence.
    pub const fn delivery(&self) -> &SinkDelivery {
        &self.delivery
    }
    /// Returns only this sink's connector-owned pre-commit value.
    pub const fn pre_commit(&self) -> &JsonMap {
        &self.pre_commit
    }
}

/// Async transactional sink lifecycle.
#[async_trait]
pub trait TransactionalStreamSink: Send {
    /// Opens connector resources.
    async fn open(&mut self) -> Result<()>;
    /// Starts one epoch transaction.
    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()>;
    /// Writes one batch into the active epoch.
    async fn write(&mut self, batch: &Batch) -> Result<()>;
    /// Produces bounded connector evidence before durable publication.
    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap>;
    /// Commits one durably published epoch.
    async fn commit(&mut self, epoch: Epoch, pre_commit: &JsonMap) -> Result<()>;
    /// Aborts an unpublished epoch when its outcome is known absent.
    async fn abort(&mut self, epoch: Epoch, pre_commit: Option<&JsonMap>) -> Result<()>;
    /// Reconciles this sink from its own scoped recovery evidence.
    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()>;
    /// Releases connector resources.
    async fn close(&mut self) -> Result<()>;
}

struct OrdinarySinkAdapter<S>(S);

#[async_trait]
impl<S: StreamSink> OrdinaryStreamSink for OrdinarySinkAdapter<S> {
    async fn open(&mut self) -> Result<()> {
        self.0.open().await
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0.write(batch).await
    }
    async fn close(&mut self) -> Result<()> {
        self.0.close().await
    }
}

struct TransactionalSinkAdapter<S> {
    sink: S,
    sink_id: String,
    delivery: SinkDelivery,
}

#[async_trait]
impl<S: TransactionalStreamSink> InternalTransactionalStreamSink for TransactionalSinkAdapter<S> {
    async fn open(&mut self) -> Result<()> {
        self.sink.open().await
    }
    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()> {
        self.sink.begin_epoch(epoch).await
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.sink.write(batch).await
    }
    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap> {
        self.sink.pre_commit(epoch).await
    }
    async fn commit(&mut self, epoch: Epoch, state: &JsonMap) -> Result<()> {
        self.sink.commit(epoch, state).await
    }
    async fn abort(&mut self, epoch: Epoch, state: Option<&JsonMap>) -> Result<()> {
        self.sink.abort(epoch, state).await
    }
    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        let entry = manifest.sinks().get(&self.sink_id).ok_or_else(|| {
            CalcFlowError::CheckpointMismatch {
                message: "checkpoint is missing expected sink recovery evidence".into(),
            }
        })?;
        let recovery = SinkRecovery {
            epoch: manifest.epoch(),
            terminal: recovery_is_terminal(manifest)?,
            delivery: self.delivery.clone(),
            pre_commit: entry.pre_commit.clone().unwrap_or_default(),
        };
        self.sink.recover(&recovery).await
    }
    async fn close(&mut self) -> Result<()> {
        self.sink.close().await
    }
}

fn recovery_is_terminal(manifest: &crate::CheckpointManifest) -> Result<bool> {
    let sources_terminal = manifest.sources().values().all(|source| source.ended);
    let operators_terminal = manifest.operators().values().all(|operator| {
        operator
            .progress
            .values()
            .all(|progress| progress.state == ManifestIngressState::Ended)
    });
    if sources_terminal != operators_terminal {
        return Err(CalcFlowError::CheckpointMismatch {
            message: "source and operator terminal recovery states disagree".into(),
        });
    }
    Ok(sources_terminal)
}

/// Stable sink identity, connector implementation, and delivery evidence.
pub struct SinkBinding {
    sink_id: String,
    delivery: SinkDelivery,
    inner: OrdinarySinkBinding,
}

impl SinkBinding {
    /// Binds an ordinary at-least-once sink under a stable ID.
    ///
    /// # Errors
    ///
    /// Returns a safe validation error when `sink_id` is not portable.
    pub fn ordinary(sink_id: &str, sink: impl StreamSink + 'static) -> Result<Self> {
        validate_sink_id(sink_id)?;
        Ok(Self {
            sink_id: sink_id.into(),
            delivery: SinkDelivery::Ordinary,
            inner: OrdinarySinkBinding::new(Box::new(OrdinarySinkAdapter(sink))),
        })
    }

    /// Binds a transactional sink under a stable ID.
    ///
    /// # Errors
    ///
    /// Returns a safe validation error when `sink_id` is not portable.
    pub fn transactional(
        sink_id: &str,
        sink: impl TransactionalStreamSink + 'static,
    ) -> Result<Self> {
        validate_sink_id(sink_id)?;
        let delivery = SinkDelivery::Transactional;
        let adapter = TransactionalSinkAdapter {
            sink,
            sink_id: sink_id.into(),
            delivery: delivery.clone(),
        };
        Ok(Self {
            sink_id: sink_id.into(),
            delivery,
            inner: OrdinarySinkBinding::new_transactional(Box::new(adapter)),
        })
    }

    /// Binds an epoch-idempotent sink and its retention evidence.
    ///
    /// # Errors
    ///
    /// Returns a safe validation error when the ID or capability evidence is invalid.
    pub fn epoch_idempotent(
        sink_id: &str,
        sink: impl TransactionalStreamSink + 'static,
        mechanism: &str,
        retention: RetentionClass,
    ) -> Result<Self> {
        validate_sink_id(sink_id)?;
        let delivery = SinkDelivery::EpochIdempotent {
            mechanism: mechanism.into(),
            retention,
        };
        let adapter = TransactionalSinkAdapter {
            sink,
            sink_id: sink_id.into(),
            delivery: delivery.clone(),
        };
        let inner =
            OrdinarySinkBinding::new_epoch_idempotent(Box::new(adapter), mechanism, retention)
                .map_err(safe_error)?;
        Ok(Self {
            sink_id: sink_id.into(),
            delivery,
            inner,
        })
    }

    /// Returns the globally unique stable sink ID.
    pub fn sink_id(&self) -> &str {
        &self.sink_id
    }
    /// Returns the frozen delivery evidence for this binding.
    pub const fn delivery(&self) -> &SinkDelivery {
        &self.delivery
    }
}

fn validate_sink_id(sink_id: &str) -> Result<()> {
    crate::runtime::streaming::job::StableSinkId::new(sink_id)
        .map(|_| ())
        .map_err(safe_error)
}

/// Single-root local state and manifest namespace owner.
pub struct ManagedCheckpointRuntime {
    inner: InternalManagedCheckpointRuntime,
}

impl ManagedCheckpointRuntime {
    /// Captures a managed root without creating or canonicalizing it.
    ///
    /// # Errors
    ///
    /// Returns a safe validation error when the lexical path is empty.
    pub fn new(managed_root: impl Into<PathBuf>) -> Result<Self> {
        InternalManagedCheckpointRuntime::new(managed_root)
            .map(|inner| Self { inner })
            .map_err(safe_error)
    }
}

impl fmt::Debug for ManagedCheckpointRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedCheckpointRuntime")
            .finish_non_exhaustive()
    }
}

/// Unstarted one-shot continuous runtime owner.
pub struct StreamingRunner {
    plan: StreamExecutionPlan,
    sources: BTreeMap<String, SourceBinding>,
    sinks: BTreeMap<String, Vec<SinkBinding>>,
    checkpoints: ManagedCheckpointRuntime,
    config: StreamRuntimeConfig,
}

impl StreamingRunner {
    /// Creates an unstarted owner after pure plan/binding shape validation.
    ///
    /// # Errors
    ///
    /// Returns a safe validation error when source or sink bindings do not exactly cover the plan.
    pub fn new(
        plan: StreamExecutionPlan,
        sources: BTreeMap<String, SourceBinding>,
        sinks: BTreeMap<String, Vec<SinkBinding>>,
        checkpoints: ManagedCheckpointRuntime,
    ) -> Result<Self> {
        validate_binding_shapes(&plan, &sources, &sinks).map_err(safe_error)?;
        Ok(Self {
            plan,
            sources,
            sinks,
            checkpoints,
            config: StreamRuntimeConfig::default(),
        })
    }

    /// Returns the runner with validated runtime tuning.
    ///
    /// # Errors
    ///
    /// Returns a safe validation error when a duration, edge limit, or retention count is invalid.
    #[must_use = "the validated runner contains the supplied runtime configuration"]
    pub fn with_runtime_config(mut self, config: StreamRuntimeConfig) -> Result<Self> {
        config.validate().map_err(safe_error)?;
        if config.retained_epochs == 0 {
            return Err(safe_error(CalcFlowError::InvalidArgument {
                field: "retained_epochs".into(),
                message: "must be greater than zero".into(),
            }));
        }
        self.config = config;
        Ok(self)
    }

    /// Starts the source-driven runtime and transfers ownership to a job.
    ///
    /// # Errors
    ///
    /// Returns a safe streaming error after all provisional runtime resources are settled.
    pub async fn start(self) -> Result<StreamingJob> {
        let Self {
            plan,
            sources,
            sinks,
            checkpoints,
            config,
        } = self;
        let job_id = allocate_job_id()?;
        let fingerprint = plan.fingerprint().to_owned();
        let context = StreamJobContext::new(
            job_id,
            fingerprint,
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let sources = sources
            .into_iter()
            .map(|(binding_id, binding)| NamedSourceBinding {
                binding_id,
                binding: binding.inner,
            })
            .collect();
        let sinks = sinks
            .into_iter()
            .flat_map(|(output_id, bindings)| {
                bindings.into_iter().map(move |binding| NamedSinkBinding {
                    output_id: output_id.clone(),
                    sink_id: binding.sink_id,
                    binding: binding.inner,
                })
            })
            .collect();
        let spec = ContinuousJobSpec {
            context,
            plan,
            sources,
            sinks,
            edge_budget: config.edge_budget,
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };
        OneShotContinuousRunner::new()
            .start_checkpointed_with_config(spec, checkpoints.inner, config)
            .await
            .map(|inner| StreamingJob { inner })
            .map_err(|failure| start_error(job_id, &failure))
    }
}

/// Sole owning handle for one running continuous job.
pub struct StreamingJob {
    inner: OwningContinuousJob,
}

impl fmt::Debug for StreamingJob {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamingJob")
            .field("job_id", &self.id())
            .field("state", &self.status().state)
            .finish_non_exhaustive()
    }
}

impl StreamingJob {
    /// Returns the stable process-local job ID.
    pub fn id(&self) -> u64 {
        self.inner.id()
    }
    /// Returns a cloned, data-only status snapshot without blocking.
    pub fn status(&self) -> JobStatus {
        self.inner.status()
    }

    /// Requests a manual checkpoint and waits for durable completion.
    ///
    /// # Errors
    ///
    /// Returns a safe streaming error if the request cannot reach the completed point.
    pub async fn trigger_checkpoint(&self) -> Result<Epoch> {
        self.inner
            .trigger_checkpoint()
            .await
            .map_err(|error| lifecycle_error(self.id(), error))
    }

    /// Requests graceful shutdown and returns the immutable terminal outcome.
    pub async fn shutdown(&self) -> JobOutcome {
        let outcome = self.inner.shutdown().await;
        self.inner.public_outcome(&outcome)
    }

    /// Cancels the job and waits until owned runtime resources settle.
    pub async fn cancel(&self) -> JobOutcome {
        let outcome = self.inner.cancel().await;
        self.inner.public_outcome(&outcome)
    }

    /// Waits for and returns the immutable terminal outcome.
    pub async fn wait(&self) -> JobOutcome {
        let outcome = self.inner.wait().await;
        self.inner.public_outcome(&outcome)
    }
}

fn validate_binding_shapes(
    plan: &StreamExecutionPlan,
    sources: &BTreeMap<String, SourceBinding>,
    sinks: &BTreeMap<String, Vec<SinkBinding>>,
) -> Result<()> {
    let expected_sources = plan
        .source_binding_ids()
        .into_iter()
        .collect::<BTreeSet<_>>();
    let actual_sources = sources.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if expected_sources != actual_sources {
        return Err(CalcFlowError::InvalidArgument {
            field: "sources".into(),
            message: "must exactly cover the stream plan source bindings".into(),
        });
    }
    let expected_outputs = plan.sink_binding_ids().into_iter().collect::<BTreeSet<_>>();
    let actual_outputs = sinks.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if expected_outputs != actual_outputs || sinks.values().any(Vec::is_empty) {
        return Err(CalcFlowError::InvalidArgument {
            field: "sinks".into(),
            message: "must route at least one sink for every stream plan output".into(),
        });
    }
    let mut sink_ids = BTreeSet::new();
    if sinks
        .values()
        .flatten()
        .any(|sink| !sink_ids.insert(sink.sink_id()))
    {
        return Err(CalcFlowError::InvalidArgument {
            field: "sinks".into(),
            message: "sink IDs must be globally unique".into(),
        });
    }
    Ok(())
}

fn allocate_job_id() -> Result<u64> {
    NEXT_JOB_ID
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
            value.checked_add(1)
        })
        .map_err(|_| {
            safe_error(CalcFlowError::Internal {
                message: "streaming job ID space is exhausted".into(),
            })
        })
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "the safe conversion seam owns and drops the raw error"
)]
fn safe_error(error: CalcFlowError) -> CalcFlowError {
    CalcFlowError::Streaming(projection::project_public_error(None, &error))
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "the safe conversion seam owns and drops the raw error"
)]
fn lifecycle_error(job_id: u64, error: CalcFlowError) -> CalcFlowError {
    CalcFlowError::Streaming(projection::project_public_error(Some(job_id), &error))
}

fn start_error(job_id: u64, failure: &StartFailure) -> CalcFlowError {
    CalcFlowError::Streaming(projection::project_start_failure(job_id, failure))
}
