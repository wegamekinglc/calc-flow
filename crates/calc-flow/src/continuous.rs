//! Source-driven continuous streaming lifecycle.
//!
//! This module implements the crate-root A6 continuous runtime façade.
//!
//! The complete lifecycle owns connectors from binding through terminal
//! cleanup:
//!
//! ```no_run
//! use std::collections::BTreeMap;
//! use async_trait::async_trait;
//! use calc_flow::{
//!     Batch, ExpressionOperator, JsonMap, PipelineBuilder, Result, StreamRequirements,
//!     Cursor, ManagedCheckpointRuntime, NativeWatermarkCapability, ReplayPositioning,
//!     SinkBinding, SinkRecovery, SourceBinding, SourceCapabilities,
//!     SourceDeliveryCapability, SourceEvent, SourceSchema, StreamSource, StreamingRunner,
//!     TransactionalStreamSink, UdfRegistry,
//! };
//!
//! struct Orders;
//!
//! #[async_trait]
//! impl StreamSource for Orders {
//!     fn capabilities(&self) -> SourceCapabilities {
//!         SourceCapabilities {
//!             replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
//!             delivery: SourceDeliveryCapability::Lossless,
//!             max_batch_rows: 1,
//!             max_batch_bytes: 1024,
//!             schema: SourceSchema::DynamicOrUnknown,
//!             native_watermarks: NativeWatermarkCapability::EmitsNative,
//!         }
//!     }
//!     async fn open(&mut self, _: Option<Cursor>) -> Result<()> { Ok(()) }
//!     async fn next(&mut self) -> Result<Option<SourceEvent>> {
//!         std::future::pending().await
//!     }
//!     async fn close(&mut self) -> Result<()> { Ok(()) }
//! }
//!
//! struct Archive;
//!
//! #[async_trait]
//! impl TransactionalStreamSink for Archive {
//!     async fn open(&mut self) -> Result<()> { Ok(()) }
//!     async fn begin_epoch(&mut self, _: calc_flow::Epoch) -> Result<()> { Ok(()) }
//!     async fn write(&mut self, _: &Batch) -> Result<()> { Ok(()) }
//!     async fn pre_commit(&mut self, _: calc_flow::Epoch) -> Result<JsonMap> {
//!         Ok(JsonMap::new())
//!     }
//!     async fn commit(&mut self, _: calc_flow::Epoch, _: &JsonMap) -> Result<()> { Ok(()) }
//!     async fn abort(&mut self, _: calc_flow::Epoch, _: Option<&JsonMap>) -> Result<()> { Ok(()) }
//!     async fn recover(&mut self, _: &SinkRecovery) -> Result<()> { Ok(()) }
//!     async fn close(&mut self) -> Result<()> { Ok(()) }
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> Result<()> {
//! let plan = PipelineBuilder::new("orders")?
//!     .add_node(
//!         "total",
//!         Box::new(ExpressionOperator::new(
//!             "total", "total = price * quantity", Vec::new(), None, Vec::new(),
//!         )?),
//!     )?
//!     .compile_stream(&UdfRegistry::new().snapshot(), &StreamRequirements::default())?;
//! let source_id = plan.source_binding_ids()[0].to_owned();
//! let output_id = plan.sink_binding_ids()[0].to_owned();
//! let runner = StreamingRunner::new(
//!     plan,
//!     BTreeMap::from([(source_id, SourceBinding::new(Orders))]),
//!     BTreeMap::from([(
//!         output_id,
//!         vec![SinkBinding::transactional("archive", Archive)?],
//!     )]),
//!     ManagedCheckpointRuntime::new(".calc-flow-continuous")?,
//! )?;
//! let job = runner.start().await?;
//! let completed_epoch = job.trigger_checkpoint().await?;
//! assert_eq!(job.status().checkpoint.last_completed_epoch, Some(completed_epoch));
//! let outcome = job.shutdown().await;
//! assert!(outcome.completed_epoch >= Some(completed_epoch));
//! # Ok(())
//! # }
//! ```
//!
//! A runner is a one-shot owner. Calling [`StreamingRunner::start`] consumes
//! it and returns the sole [`StreamingJob`] lifecycle owner.
//!
//! ```compile_fail
//! # async fn reuse(runner: calc_flow::StreamingRunner) {
//! let _first = runner.start().await;
//! let _second = runner.start().await; // the runner was moved
//! # }
//! ```
//!
//! A job is likewise the sole lifecycle owner and cannot be cloned.
//!
//! ```compile_fail
//! # fn clone_job(job: calc_flow::StreamingJob) {
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
//! use calc_flow::{Cursor, SourceCapabilities, SourceEvent, StreamSource};
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
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;

use crate::{
    Batch, CalcFlowError, CancellationToken, Epoch, JsonMap, ManifestIngressState, Result,
    RetentionClass, StreamExecutionPlan, StreamJobContext, StreamJoinStatus, StreamRuntimeConfig,
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
            Cursor as InternalCursor,
            DurableCursorAcknowledger as InternalDurableCursorAcknowledger,
            SourceBinding as InternalSourceBinding,
            SourceCapabilities as InternalSourceCapabilities,
            SourceCheckpointGate as InternalSourceCheckpointGate,
            SourceDeliveryCapability as InternalSourceDeliveryCapability,
            SourceEvent as InternalSourceEvent, StreamSource as InternalStreamSource,
        },
    },
};

pub use crate::runtime::streaming::projection::{
    CheckpointPhase, CheckpointStatus, ComponentKind, EdgeStatus, JobOutcome, JobState, JobStatus,
    OperatorStatus, OutputDeliveryStatus, SinkDelivery, SinkStatus, SourceStatus, StreamingError,
    StreamingErrorCategory, StreamingFailureReason, TerminalCause,
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
#[derive(Clone)]
pub enum SourceEvent {
    /// One data batch and its replay position.
    Data { batch: Batch, cursor: Cursor },
    /// A source-native event-time watermark.
    Watermark(crate::EventTime),
    /// A temporary idle observation that does not end the source.
    Idle,
}

impl fmt::Debug for SourceEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data { batch, cursor } => formatter
                .debug_struct("Data")
                .field("kind", &batch.kind())
                .field("rows", &batch.num_rows())
                .field("cursor", cursor)
                .finish(),
            Self::Watermark(value) => formatter.debug_tuple("Watermark").field(value).finish(),
            Self::Idle => formatter.write_str("Idle"),
        }
    }
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
    /// Returns an optional job-independent handle that advances external
    /// source durability only after the checkpoint manifest is durable.
    ///
    /// The handle must be safe to call while [`Self::next`] is in flight. It
    /// must never acknowledge a position beyond the supplied cursor.
    fn durable_cursor_acknowledger(&self) -> Option<Arc<dyn DurableCursorAcknowledger>> {
        None
    }
    /// Returns an optional concurrent gate for connector-level atomic cuts.
    ///
    /// A source that emits one external transaction or consistent snapshot as
    /// several batches can keep this gate closed until the complete cut has
    /// entered the runtime. Checkpoint requests wait without pausing source
    /// admission, then take the normal exact cursor cut once the gate opens.
    fn checkpoint_gate(&self) -> Option<Arc<dyn SourceCheckpointGate>> {
        None
    }
    /// Releases connector resources.
    async fn close(&mut self) -> Result<()>;
}

/// Advances a connector-owned durable cursor after manifest publication.
///
/// Streaming sources such as logical replication slots use this hook to keep
/// external retention feedback behind Calc-Flow's durable checkpoint truth.
#[async_trait]
pub trait DurableCursorAcknowledger: Send + Sync {
    /// Acknowledges exactly the supplied durable cursor.
    ///
    /// # Errors
    ///
    /// Returns a safe connector error without advancing the external cursor
    /// when the acknowledgement cannot be applied.
    async fn acknowledge(&self, cursor: &Cursor) -> Result<()>;
}

/// Concurrent readiness gate for a connector-defined atomic checkpoint cut.
#[async_trait]
pub trait SourceCheckpointGate: Send + Sync {
    /// Waits until a checkpoint can cut the source without splitting its
    /// current external transaction or initial consistent snapshot.
    async fn wait_ready(&self) -> Result<()>;
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

    fn durable_cursor_acknowledger(&self) -> Option<Arc<dyn InternalDurableCursorAcknowledger>> {
        self.source.durable_cursor_acknowledger().map(|inner| {
            Arc::new(DurableCursorAcknowledgerAdapter { inner })
                as Arc<dyn InternalDurableCursorAcknowledger>
        })
    }

    fn checkpoint_gate(&self) -> Option<Arc<dyn InternalSourceCheckpointGate>> {
        self.source.checkpoint_gate().map(|inner| {
            Arc::new(SourceCheckpointGateAdapter { inner }) as Arc<dyn InternalSourceCheckpointGate>
        })
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

struct DurableCursorAcknowledgerAdapter {
    inner: Arc<dyn DurableCursorAcknowledger>,
}

struct SourceCheckpointGateAdapter {
    inner: Arc<dyn SourceCheckpointGate>,
}

#[async_trait]
impl InternalSourceCheckpointGate for SourceCheckpointGateAdapter {
    async fn wait_ready(&self) -> Result<()> {
        self.inner.wait_ready().await
    }
}

#[async_trait]
impl InternalDurableCursorAcknowledger for DurableCursorAcknowledgerAdapter {
    async fn acknowledge(&self, cursor: &InternalCursor) -> Result<()> {
        self.inner
            .acknowledge(&Cursor {
                inner: cursor.clone(),
            })
            .await
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
#[derive(Clone, Eq, PartialEq)]
pub struct SinkRecovery {
    epoch: Epoch,
    terminal: bool,
    delivery: SinkDelivery,
    pre_commit: JsonMap,
    segments: BTreeMap<String, Vec<u8>>,
}

impl fmt::Debug for SinkRecovery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SinkRecovery")
            .field("epoch", &self.epoch)
            .field("terminal", &self.terminal)
            .field("delivery", &self.delivery)
            .field("pre_commit", &"<redacted>")
            .field("segments", &self.segments.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl SinkRecovery {
    /// Assembles sink-scoped recovery evidence from its data-only parts.
    ///
    /// The runtime builds this value from the durable checkpoint
    /// manifest; the public constructor lets connector implementers and
    /// embedders exercise their `recover` contract against explicit
    /// evidence instead of forging checkpoints.
    #[must_use]
    pub fn from_parts(
        epoch: Epoch,
        terminal: bool,
        delivery: SinkDelivery,
        pre_commit: JsonMap,
    ) -> Self {
        Self {
            epoch,
            terminal,
            delivery,
            pre_commit,
            segments: BTreeMap::new(),
        }
    }
    /// Adds the connector-owned committed state-segment bytes used by a
    /// recovery test or embedding.
    #[must_use]
    pub fn with_segments(mut self, segments: BTreeMap<String, Vec<u8>>) -> Self {
        self.segments = segments;
        self
    }
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
    /// Returns committed connector state segments by stable segment ID.
    pub const fn segments(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.segments
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
    /// Produces bounded, connector-owned bytes that the runtime commits as
    /// immutable state segments before publishing the manifest.
    async fn pre_commit_segments(&mut self, _epoch: Epoch) -> Result<BTreeMap<String, Vec<u8>>> {
        Ok(BTreeMap::new())
    }
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
    async fn pre_commit_segments(&mut self, epoch: Epoch) -> Result<BTreeMap<String, Vec<u8>>> {
        self.sink.pre_commit_segments(epoch).await
    }
    async fn commit(&mut self, epoch: Epoch, state: &JsonMap) -> Result<()> {
        self.sink.commit(epoch, state).await
    }
    async fn abort(&mut self, epoch: Epoch, state: Option<&JsonMap>) -> Result<()> {
        self.sink.abort(epoch, state).await
    }
    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        self.recover_with_segments(manifest, BTreeMap::new()).await
    }
    async fn recover_with_segments(
        &mut self,
        manifest: &crate::CheckpointManifest,
        segments: BTreeMap<String, Vec<u8>>,
    ) -> Result<()> {
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
            segments,
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
    #[cfg(test)]
    fault: Option<(
        crate::runtime::streaming::runner::CheckpointFaultPoint,
        crate::runtime::streaming::runner::CheckpointFaultMode,
    )>,
}

impl ManagedCheckpointRuntime {
    /// Captures a managed root without creating or canonicalizing it.
    ///
    /// # Errors
    ///
    /// Returns a safe validation error when the lexical path is empty.
    pub fn new(managed_root: impl Into<PathBuf>) -> Result<Self> {
        InternalManagedCheckpointRuntime::new(managed_root)
            .map(|inner| Self {
                inner,
                #[cfg(test)]
                fault: None,
            })
            .map_err(safe_error)
    }

    #[cfg(test)]
    pub(crate) fn with_fault_for_test(
        mut self,
        point: crate::runtime::streaming::runner::CheckpointFaultPoint,
        mode: crate::runtime::streaming::runner::CheckpointFaultMode,
    ) -> Self {
        self.fault = Some((point, mode));
        self
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
    static_inputs: BTreeMap<String, Batch>,
}

impl StreamingRunner {
    /// Creates an unstarted owner after pure plan/binding shape validation.
    ///
    /// # Errors
    ///
    /// Returns a safe validation error when source or sink bindings do not exactly cover the plan.
    pub fn new(
        mut plan: StreamExecutionPlan,
        mut sources: BTreeMap<String, SourceBinding>,
        mut sinks: BTreeMap<String, Vec<SinkBinding>>,
        checkpoints: ManagedCheckpointRuntime,
    ) -> Result<Self> {
        if let Some((project_sources, project_sinks)) = plan.take_project_bindings() {
            if !sources.is_empty() || !sinks.is_empty() {
                return Err(safe_error(CalcFlowError::InvalidArgument {
                    field: "bindings".into(),
                    message: "project-v3 plans own their connector bindings; external bindings must be empty"
                        .into(),
                }));
            }
            sources = project_sources;
            sinks = project_sinks;
        }
        validate_binding_shapes(&plan, &sources, &sinks).map_err(safe_error)?;
        Ok(Self {
            plan,
            sources,
            sinks,
            checkpoints,
            config: StreamRuntimeConfig::default(),
            static_inputs: BTreeMap::new(),
        })
    }

    /// Attaches the immutable static input values for this job (SCE-11).
    ///
    /// The mapping is copied immediately; validation, latching, and digest
    /// computation happen exactly once during [`Self::start`], completing
    /// before any source, operator, sink, or provider lifecycle runs.
    ///
    /// # Errors
    ///
    /// Reserved for future builder-time shape checks; static value
    /// validation is deliberately deferred to job start.
    #[must_use = "the runner owns the supplied static input handles"]
    pub fn with_static_inputs(mut self, inputs: BTreeMap<String, Batch>) -> Result<Self> {
        self.static_inputs = inputs;
        Ok(self)
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
            static_inputs,
        } = self;
        let job_id = allocate_job_id()?;
        let prepared_static_inputs =
            crate::static_input::prepare_static_inputs(plan.static_inputs(), &static_inputs)
                .map_err(static_preflight_error)?;
        let fingerprint = plan.fingerprint().to_owned();
        let context = StreamJobContext::new(
            job_id,
            fingerprint,
            static_input_settings(&prepared_static_inputs),
            None,
            CancellationToken::new(),
        )
        .with_static_inputs(prepared_static_inputs.latched.clone());
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
            static_inputs: prepared_static_inputs,
        };
        #[cfg(test)]
        let (start, fault_probe) = match checkpoints.fault {
            Some((point, mode)) => {
                let (start, probe) = OneShotContinuousRunner::new()
                    .start_checkpointed_with_config_and_fault_probe(
                        spec,
                        checkpoints.inner,
                        config,
                        point,
                        mode,
                    );
                (start, Some(probe))
            }
            None => (
                OneShotContinuousRunner::new().start_checkpointed_with_config(
                    spec,
                    checkpoints.inner,
                    config,
                ),
                None,
            ),
        };
        #[cfg(not(test))]
        let start = OneShotContinuousRunner::new().start_checkpointed_with_config(
            spec,
            checkpoints.inner,
            config,
        );
        start
            .await
            .map(|inner| StreamingJob {
                inner,
                #[cfg(test)]
                fault_probe,
            })
            .map_err(|failure| start_error(job_id, &failure))
    }
}

/// Sole owning handle for one running continuous job.
pub struct StreamingJob {
    inner: OwningContinuousJob,
    #[cfg(test)]
    fault_probe: Option<crate::runtime::streaming::runner::CheckpointFaultInjector>,
}

#[cfg(test)]
pub(crate) struct StreamingJobTestProbe {
    pub(crate) checkpoint_fault_triggers: usize,
    pub(crate) cancellation_triggers: usize,
    pub(crate) parent_sync_os_failures: usize,
    pub(crate) checkpoint_failures: u64,
    pub(crate) runner_registries: (usize, usize),
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

    /// Returns each Join node's payload-free status keyed by stable node ID.
    ///
    /// Values are cloned numeric snapshots without keys, rows, cursors, or
    /// secrets. Jobs without Join nodes return an empty map.
    pub fn stream_join_status(&self) -> BTreeMap<String, StreamJoinStatus> {
        self.inner.stream_join_status()
    }

    /// Requests a manual checkpoint and waits for durable completion.
    ///
    /// # Errors
    ///
    /// Returns a safe streaming error if the request cannot reach the completed point.
    pub async fn trigger_checkpoint(&self) -> Result<Epoch> {
        self.inner.trigger_checkpoint().await
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

    #[cfg(test)]
    pub(crate) fn test_probe(&self) -> StreamingJobTestProbe {
        StreamingJobTestProbe {
            checkpoint_fault_triggers: self
                .fault_probe
                .as_ref()
                .map_or(0, crate::runtime::streaming::runner::CheckpointFaultInjector::trigger_count),
            cancellation_triggers: self.fault_probe.as_ref().map_or(
                0,
                crate::runtime::streaming::runner::CheckpointFaultInjector::cancellation_trigger_count,
            ),
            parent_sync_os_failures: {
                #[cfg(unix)]
                {
                    self.fault_probe.as_ref().map_or(
                        0,
                        crate::runtime::streaming::runner::CheckpointFaultInjector::parent_sync_os_failure_count,
                    )
                }
                #[cfg(not(unix))]
                {
                    0
                }
            },
            checkpoint_failures: self.inner.checkpoint_failure_count_for_test(),
            runner_registries: self.inner.runner_probe_for_test().registry_counts(),
        }
    }
}

fn static_preflight_error(error: CalcFlowError) -> CalcFlowError {
    let CalcFlowError::InvalidArgument { field, message } = error else {
        return safe_error(error);
    };
    CalcFlowError::Streaming(projection::validation_error(
        ComponentKind::Job,
        None,
        format!("{field}: {message}"),
    ))
}

/// Builds the payload-free static-input digest evidence carried on the job
/// context settings; operators observe it read-only from job start.
fn static_input_settings(prepared: &crate::static_input::PreparedStaticInputs) -> JsonMap {
    let evidence = prepared
        .digests
        .iter()
        .map(|(name, digest)| {
            (
                name.clone(),
                serde_json::json!({
                    "digest_version": digest.digest_version,
                    "sha256": digest.sha256,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    BTreeMap::from([(
        "static_inputs".to_owned(),
        serde_json::Value::Object(evidence),
    )])
}

fn validate_binding_shapes(
    plan: &StreamExecutionPlan,
    sources: &BTreeMap<String, SourceBinding>,
    sinks: &BTreeMap<String, Vec<SinkBinding>>,
) -> Result<()> {
    validate_portable_binding_ids(sources, sinks)?;
    validate_source_binding_shape(plan, sources)?;
    validate_sink_binding_shape(plan, sinks)?;
    validate_unique_sink_ids(sinks)
}

fn validate_portable_binding_ids(
    sources: &BTreeMap<String, SourceBinding>,
    sinks: &BTreeMap<String, Vec<SinkBinding>>,
) -> Result<()> {
    if sources
        .keys()
        .any(|source_id| crate::json::validate_portable_identifier("source", source_id).is_err())
    {
        return Err(invalid_shape_id(
            ComponentKind::Source,
            "source binding ID is not a portable identifier",
        ));
    }
    if sinks
        .keys()
        .any(|output_id| crate::json::validate_portable_identifier("output", output_id).is_err())
    {
        return Err(invalid_shape_id(
            ComponentKind::Sink,
            "sink output ID is not a portable identifier",
        ));
    }
    Ok(())
}

fn validate_source_binding_shape(
    plan: &StreamExecutionPlan,
    sources: &BTreeMap<String, SourceBinding>,
) -> Result<()> {
    let expected_sources = plan
        .source_binding_ids()
        .into_iter()
        .collect::<BTreeSet<_>>();
    let actual_sources = sources.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if let Some(source_id) = actual_sources.difference(&expected_sources).next() {
        return Err(shape_error(
            ComponentKind::Source,
            source_id,
            format!("source binding {source_id:?} does not match a compiled external input"),
        ));
    }
    if let Some(source_id) = expected_sources.difference(&actual_sources).next() {
        return Err(shape_error(
            ComponentKind::Source,
            source_id,
            format!("source bindings are missing external input {source_id:?}"),
        ));
    }
    Ok(())
}

fn validate_sink_binding_shape(
    plan: &StreamExecutionPlan,
    sinks: &BTreeMap<String, Vec<SinkBinding>>,
) -> Result<()> {
    let expected_outputs = plan.sink_binding_ids().into_iter().collect::<BTreeSet<_>>();
    let actual_outputs = sinks.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if let Some(output_id) = actual_outputs.difference(&expected_outputs).next() {
        return Err(shape_error(
            ComponentKind::Sink,
            output_id,
            format!("sink route {output_id:?} does not match a compiled graph output"),
        ));
    }
    if let Some(output_id) = expected_outputs
        .difference(&actual_outputs)
        .copied()
        .next()
        .or_else(|| {
            sinks
                .iter()
                .find_map(|(output_id, bindings)| bindings.is_empty().then_some(output_id.as_str()))
        })
    {
        return Err(shape_error(
            ComponentKind::Sink,
            output_id,
            format!("sink bindings are missing graph output {output_id:?}"),
        ));
    }
    Ok(())
}

fn validate_unique_sink_ids(sinks: &BTreeMap<String, Vec<SinkBinding>>) -> Result<()> {
    let mut sink_ids = BTreeSet::new();
    if let Some(sink_id) = sinks
        .values()
        .flatten()
        .find_map(|sink| (!sink_ids.insert(sink.sink_id())).then(|| sink.sink_id().to_owned()))
    {
        return Err(shape_error(
            ComponentKind::Sink,
            &sink_id,
            format!("sink ID {sink_id:?} is configured for more than one graph output"),
        ));
    }
    Ok(())
}

fn shape_error(kind: ComponentKind, id: &str, message: String) -> CalcFlowError {
    CalcFlowError::Streaming(projection::validation_error(kind, Some(id), message))
}

fn invalid_shape_id(kind: ComponentKind, message: &str) -> CalcFlowError {
    CalcFlowError::Streaming(projection::validation_error(kind, None, message.into()))
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
    if matches!(error, CalcFlowError::Streaming(_)) {
        return error;
    }
    CalcFlowError::Streaming(projection::project_public_error(None, &error))
}

fn start_error(job_id: u64, failure: &StartFailure) -> CalcFlowError {
    CalcFlowError::Streaming(projection::project_start_failure(job_id, failure))
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, time::Duration};

    use async_trait::async_trait;

    use super::{
        Cursor, ManagedCheckpointRuntime, NativeWatermarkCapability, ReplayPositioning,
        SinkBinding, SinkRecovery, SourceBinding, SourceCapabilities, SourceDeliveryCapability,
        SourceEvent, SourceSchema, StreamSource, StreamingErrorCategory, StreamingRunner,
        TransactionalStreamSink,
    };
    use crate::{
        Batch, CalcFlowError, ExpressionOperator, JsonMap, PipelineBuilder, Result,
        StreamRequirements, UdfRegistry,
        runtime::streaming::runner::{CheckpointFaultMode, CheckpointFaultPoint},
    };

    struct PendingSource;

    #[async_trait]
    impl StreamSource for PendingSource {
        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
                delivery: SourceDeliveryCapability::Lossless,
                max_batch_rows: 1,
                max_batch_bytes: 1024,
                schema: SourceSchema::DynamicOrUnknown,
                native_watermarks: NativeWatermarkCapability::EmitsNative,
            }
        }

        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            std::future::pending().await
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct TransactionalSink;

    struct FailingCommitSink;

    struct FailingPreCommitSink;

    struct PanickingPreCommitSink;

    #[async_trait]
    impl TransactionalStreamSink for TransactionalSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
            Ok(JsonMap::new())
        }

        async fn commit(&mut self, _epoch: crate::Epoch, _pre_commit: &JsonMap) -> Result<()> {
            Ok(())
        }

        async fn abort(
            &mut self,
            _epoch: crate::Epoch,
            _pre_commit: Option<&JsonMap>,
        ) -> Result<()> {
            Ok(())
        }

        async fn recover(&mut self, _recovery: &SinkRecovery) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl TransactionalStreamSink for FailingCommitSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
            Ok(JsonMap::new())
        }

        async fn commit(&mut self, _epoch: crate::Epoch, _pre_commit: &JsonMap) -> Result<()> {
            Err(CalcFlowError::Internal {
                message: "credential-canary".into(),
            })
        }

        async fn abort(
            &mut self,
            _epoch: crate::Epoch,
            _pre_commit: Option<&JsonMap>,
        ) -> Result<()> {
            Ok(())
        }

        async fn recover(&mut self, _recovery: &SinkRecovery) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl TransactionalStreamSink for FailingPreCommitSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
            Err(CalcFlowError::Internal {
                message: "credential-canary".into(),
            })
        }

        async fn commit(&mut self, _epoch: crate::Epoch, _pre_commit: &JsonMap) -> Result<()> {
            Ok(())
        }

        async fn abort(
            &mut self,
            _epoch: crate::Epoch,
            _pre_commit: Option<&JsonMap>,
        ) -> Result<()> {
            Ok(())
        }

        async fn recover(&mut self, _recovery: &SinkRecovery) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl TransactionalStreamSink for PanickingPreCommitSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
            panic!("credential-canary")
        }

        async fn commit(&mut self, _epoch: crate::Epoch, _pre_commit: &JsonMap) -> Result<()> {
            Ok(())
        }

        async fn abort(
            &mut self,
            _epoch: crate::Epoch,
            _pre_commit: Option<&JsonMap>,
        ) -> Result<()> {
            Ok(())
        }

        async fn recover(&mut self, _recovery: &SinkRecovery) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn manual_checkpoint_io_fault_keeps_public_category_and_coordinates() {
        let plan = PipelineBuilder::new("checkpoint-io")
            .unwrap()
            .add_node(
                "operator",
                Box::new(
                    ExpressionOperator::new(
                        "operator",
                        "value = value",
                        Vec::new(),
                        None,
                        Vec::new(),
                    )
                    .unwrap(),
                ),
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        let source_id = plan.source_binding_ids()[0].to_owned();
        let output_id = plan.sink_binding_ids()[0].to_owned();
        let directory = tempfile::tempdir().unwrap();
        let checkpoints = ManagedCheckpointRuntime::new(directory.path().join("managed"))
            .unwrap()
            .with_fault_for_test(CheckpointFaultPoint::ManifestWrite, CheckpointFaultMode::Io);
        let job = StreamingRunner::new(
            plan,
            BTreeMap::from([(source_id, SourceBinding::new(PendingSource))]),
            BTreeMap::from([(
                output_id,
                vec![SinkBinding::transactional("sink", TransactionalSink).unwrap()],
            )]),
            checkpoints,
        )
        .unwrap()
        .start()
        .await
        .unwrap();

        let error = tokio::time::timeout(Duration::from_secs(5), job.trigger_checkpoint())
            .await
            .expect("the injected checkpoint fault must settle")
            .unwrap_err();
        let CalcFlowError::Streaming(error) = error else {
            panic!("manual checkpoint failures must use the streaming boundary");
        };

        assert_eq!(error.category(), StreamingErrorCategory::Io);
        assert_eq!(error.epoch(), Some(crate::Epoch::INITIAL));
        assert_eq!(
            error.component_kind(),
            Some(super::ComponentKind::Checkpoint)
        );
        assert_eq!(error.job_id(), Some(job.id()));
        assert!(!format!("{error:?}").contains(directory.path().to_str().unwrap()));
        let status = job.status();
        assert_eq!(
            status.checkpoint.failure_category,
            Some(StreamingErrorCategory::Io)
        );
        assert_eq!(status.checkpoint.current_epoch, Some(crate::Epoch::INITIAL));
        let _ = job.wait().await;
    }

    #[tokio::test]
    async fn manual_checkpoint_sink_commit_failure_keeps_connector_coordinates() {
        let plan = PipelineBuilder::new("checkpoint-sink-failure")
            .unwrap()
            .add_node(
                "operator",
                Box::new(
                    ExpressionOperator::new(
                        "operator",
                        "value = value",
                        Vec::new(),
                        None,
                        Vec::new(),
                    )
                    .unwrap(),
                ),
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        let source_id = plan.source_binding_ids()[0].to_owned();
        let output_id = plan.sink_binding_ids()[0].to_owned();
        let directory = tempfile::tempdir().unwrap();
        let job = StreamingRunner::new(
            plan,
            BTreeMap::from([(source_id, SourceBinding::new(PendingSource))]),
            BTreeMap::from([(
                output_id,
                vec![SinkBinding::transactional("archive", FailingCommitSink).unwrap()],
            )]),
            ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap(),
        )
        .unwrap()
        .start()
        .await
        .unwrap();

        let error = tokio::time::timeout(Duration::from_secs(5), job.trigger_checkpoint())
            .await
            .expect("the sink commit failure must settle")
            .unwrap_err();
        let CalcFlowError::Streaming(error) = error else {
            panic!("manual checkpoint failures must use the streaming boundary");
        };

        assert_eq!(error.category(), StreamingErrorCategory::Connector);
        assert_eq!(error.component_kind(), Some(super::ComponentKind::Sink));
        assert_eq!(error.component_id(), Some("archive"));
        assert_eq!(error.epoch(), Some(crate::Epoch::INITIAL));
        assert_eq!(
            error.checkpoint_phase(),
            Some(super::CheckpointPhase::ManifestDurable)
        );
        assert_eq!(
            error.message(),
            "sink \"archive\" commit failed for checkpoint epoch 1"
        );
        assert!(!format!("{error:?}").contains("credential-canary"));
        let outcome = job.wait().await;
        assert_eq!(outcome.state, super::JobState::RecoveryRequired);
        let outcome_error = outcome
            .errors
            .iter()
            .find(|error| error.component_id() == Some("archive"))
            .expect("the terminal outcome must retain the sink commit failure");
        assert_eq!(outcome_error.category(), StreamingErrorCategory::Connector);
        assert_eq!(outcome_error.epoch(), Some(crate::Epoch::INITIAL));
        assert_eq!(
            outcome_error.checkpoint_phase(),
            Some(super::CheckpointPhase::ManifestDurable)
        );
        assert_eq!(outcome_error.message(), error.message());
    }

    #[tokio::test]
    async fn accepted_manual_checkpoint_reports_precommit_connector_failure() {
        let plan = PipelineBuilder::new("checkpoint-precommit-failure")
            .unwrap()
            .add_node(
                "operator",
                Box::new(
                    ExpressionOperator::new(
                        "operator",
                        "value = value",
                        Vec::new(),
                        None,
                        Vec::new(),
                    )
                    .unwrap(),
                ),
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        let source_id = plan.source_binding_ids()[0].to_owned();
        let output_id = plan.sink_binding_ids()[0].to_owned();
        let directory = tempfile::tempdir().unwrap();
        let job = StreamingRunner::new(
            plan,
            BTreeMap::from([(source_id, SourceBinding::new(PendingSource))]),
            BTreeMap::from([(
                output_id,
                vec![SinkBinding::transactional("archive", FailingPreCommitSink).unwrap()],
            )]),
            ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap(),
        )
        .unwrap()
        .start()
        .await
        .unwrap();

        let error = tokio::time::timeout(Duration::from_secs(5), job.trigger_checkpoint())
            .await
            .expect("the sink pre-commit failure must settle")
            .unwrap_err();
        let CalcFlowError::Streaming(error) = error else {
            panic!("accepted manual failures must use the streaming boundary");
        };

        assert_eq!(error.category(), StreamingErrorCategory::Connector);
        assert_eq!(error.component_kind(), Some(super::ComponentKind::Sink));
        assert_eq!(error.component_id(), Some("archive"));
        assert_eq!(error.epoch(), Some(crate::Epoch::INITIAL));
        assert!(!format!("{error:?}").contains("credential-canary"));
        let _ = job.wait().await;
    }

    #[tokio::test]
    async fn accepted_manual_checkpoint_reports_contained_precommit_panic() {
        let plan = PipelineBuilder::new("checkpoint-precommit-panic")
            .unwrap()
            .add_node(
                "operator",
                Box::new(
                    ExpressionOperator::new(
                        "operator",
                        "value = value",
                        Vec::new(),
                        None,
                        Vec::new(),
                    )
                    .unwrap(),
                ),
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        let source_id = plan.source_binding_ids()[0].to_owned();
        let output_id = plan.sink_binding_ids()[0].to_owned();
        let directory = tempfile::tempdir().unwrap();
        let job = StreamingRunner::new(
            plan,
            BTreeMap::from([(source_id, SourceBinding::new(PendingSource))]),
            BTreeMap::from([(
                output_id,
                vec![SinkBinding::transactional("archive", PanickingPreCommitSink).unwrap()],
            )]),
            ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap(),
        )
        .unwrap()
        .start()
        .await
        .unwrap();

        let error = tokio::time::timeout(Duration::from_secs(5), job.trigger_checkpoint())
            .await
            .expect("the contained sink panic must settle")
            .unwrap_err();
        let CalcFlowError::Streaming(error) = error else {
            panic!("accepted manual failures must use the streaming boundary");
        };

        assert_eq!(error.category(), StreamingErrorCategory::TaskPanicked);
        assert_eq!(error.epoch(), Some(crate::Epoch::INITIAL));
        assert!(!format!("{error:?}").contains("credential-canary"));
        let _ = job.wait().await;
    }
}

#[cfg(test)]
mod static_input_runner_tests {
    use std::{
        collections::BTreeMap,
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    };

    use async_trait::async_trait;
    use datafusion::arrow::{
        array::{ArrayRef, Float64Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use super::{
        Cursor, ManagedCheckpointRuntime, NativeWatermarkCapability, ReplayPositioning,
        SinkBinding, SinkRecovery, SourceBinding, SourceCapabilities, SourceDeliveryCapability,
        SourceEvent, SourceSchema, StreamSource, StreamingErrorCategory, StreamingRunner,
        TransactionalStreamSink,
    };
    use crate::{
        ArrowFieldSpec, Batch, BatchKind, BatchMetadata, CalcFlowError, JsonMap, PipelineBuilder,
        Port, Result, StaticInputSpec, StaticMutability, StreamExecutionPlan, StreamRequirements,
        UdfRegistry, UnionOperator, static_input::digest_for_name,
    };

    struct OpeningProbeSource {
        opened: Arc<AtomicBool>,
    }

    #[async_trait]
    impl StreamSource for OpeningProbeSource {
        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
                delivery: SourceDeliveryCapability::Lossless,
                max_batch_rows: 1,
                max_batch_bytes: 1024,
                schema: SourceSchema::DynamicOrUnknown,
                native_watermarks: NativeWatermarkCapability::EmitsNative,
            }
        }

        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            self.opened.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            std::future::pending().await
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct NoopSink;

    #[async_trait]
    impl TransactionalStreamSink for NoopSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }
        async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
            Ok(())
        }
        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }
        async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
            Ok(JsonMap::new())
        }
        async fn commit(&mut self, _epoch: crate::Epoch, _pre_commit: &JsonMap) -> Result<()> {
            Ok(())
        }
        async fn abort(
            &mut self,
            _epoch: crate::Epoch,
            _pre_commit: Option<&JsonMap>,
        ) -> Result<()> {
            Ok(())
        }
        async fn recover(&mut self, _recovery: &SinkRecovery) -> Result<()> {
            Ok(())
        }
        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn weights_table(values: &[f64]) -> Batch {
        Batch::table(
            vec![
                RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new(
                        "factor",
                        DataType::Float64,
                        false,
                    )])),
                    vec![Arc::new(Float64Array::from(values.to_vec())) as ArrayRef],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    fn static_plan() -> StreamExecutionPlan {
        PipelineBuilder::new("static-inputs")
            .unwrap()
            .add_node(
                "merge",
                Box::new(
                    UnionOperator::new(
                        "merge",
                        vec![
                            Port::new("main", BatchKind::Table, true, None).unwrap(),
                            Port::new("weights", BatchKind::Table, true, None).unwrap(),
                        ],
                    )
                    .unwrap(),
                ),
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap()
            .with_static_input_specs(vec![StaticInputSpec::Table {
                name: "weights".into(),
                mutability: StaticMutability::Static,
                schema: vec![ArrowFieldSpec {
                    name: "factor".into(),
                    data_type: "float64".into(),
                    nullable: false,
                }],
            }])
            .unwrap()
    }

    fn runner_with(
        plan: StreamExecutionPlan,
        root: &Path,
        opened: Arc<AtomicBool>,
        static_inputs: BTreeMap<String, Batch>,
    ) -> Result<StreamingRunner> {
        StreamingRunner::new(
            plan,
            BTreeMap::from([(
                "main".into(),
                SourceBinding::new(OpeningProbeSource { opened }),
            )]),
            BTreeMap::from([(
                "output".into(),
                vec![SinkBinding::transactional("sink", NoopSink).unwrap()],
            )]),
            ManagedCheckpointRuntime::new(root.join("managed")).unwrap(),
        )
        .unwrap()
        .with_static_inputs(static_inputs)
    }

    async fn start_with(
        plan: StreamExecutionPlan,
        root: &Path,
        opened: Arc<AtomicBool>,
        static_inputs: BTreeMap<String, Batch>,
    ) -> Result<super::StreamingJob> {
        runner_with(plan, root, opened, static_inputs)?
            .start()
            .await
    }

    #[test]
    fn static_input_payloads_do_not_enter_the_semantic_fingerprint() {
        let first_payload = weights_table(&[1.0, 2.0, 3.0]);
        let second_payload = weights_table(&[9.0, 9.0, 9.0]);
        assert_ne!(
            digest_for_name("weights", &first_payload).unwrap(),
            digest_for_name("weights", &second_payload).unwrap()
        );
        let first_directory = tempfile::tempdir().unwrap();
        let second_directory = tempfile::tempdir().unwrap();
        let first = runner_with(
            static_plan(),
            first_directory.path(),
            Arc::new(AtomicBool::new(false)),
            BTreeMap::from([("weights".into(), first_payload)]),
        )
        .unwrap();
        let second = runner_with(
            static_plan(),
            second_directory.path(),
            Arc::new(AtomicBool::new(false)),
            BTreeMap::from([("weights".into(), second_payload)]),
        )
        .unwrap();

        assert_eq!(first.plan.fingerprint(), second.plan.fingerprint());
    }

    #[tokio::test]
    async fn static_input_preflight_fails_before_any_source_opens() {
        let opened = Arc::new(AtomicBool::new(false));
        let directory = tempfile::tempdir().unwrap();
        let error = start_with(
            static_plan(),
            directory.path(),
            Arc::clone(&opened),
            BTreeMap::new(),
        )
        .await
        .unwrap_err();
        let CalcFlowError::Streaming(error) = error else {
            panic!("static preflight failures must use the streaming boundary");
        };
        assert_eq!(error.category(), StreamingErrorCategory::Validation);
        assert!(
            error
                .to_string()
                .contains("static_inputs.weights: required static input is missing"),
            "{}",
            error
        );
        assert!(
            !opened.load(Ordering::SeqCst),
            "preflight must fail before the source opens"
        );
    }

    #[tokio::test]
    async fn static_input_recovery_rejects_a_changed_value_before_source_open() {
        let directory = tempfile::tempdir().unwrap();
        let opened = Arc::new(AtomicBool::new(false));
        let job = start_with(
            static_plan(),
            directory.path(),
            Arc::clone(&opened),
            BTreeMap::from([("weights".into(), weights_table(&[1.0, 2.0, 3.0]))]),
        )
        .await
        .expect("the first launch must start with the declared static input");
        let epoch = tokio::time::timeout(Duration::from_secs(10), job.trigger_checkpoint())
            .await
            .expect("checkpoint must settle")
            .unwrap();
        let outcome = tokio::time::timeout(Duration::from_secs(10), job.shutdown())
            .await
            .expect("shutdown must settle");
        assert!(outcome.completed_epoch >= Some(epoch));

        let reopened = Arc::new(AtomicBool::new(false));
        let error = start_with(
            static_plan(),
            directory.path(),
            Arc::clone(&reopened),
            BTreeMap::from([("weights".into(), weights_table(&[9.0, 9.0, 9.0]))]),
        )
        .await
        .unwrap_err();
        let CalcFlowError::Streaming(error) = error else {
            panic!("recovery mismatches must use the streaming boundary");
        };
        assert_eq!(error.category(), StreamingErrorCategory::CheckpointMismatch);
        let stored = digest_for_name("weights", &weights_table(&[1.0, 2.0, 3.0]))
            .unwrap()
            .sha256;
        let prepared = digest_for_name("weights", &weights_table(&[9.0, 9.0, 9.0]))
            .unwrap()
            .sha256;
        assert_eq!(
            error.to_string(),
            format!(
                "static_inputs.weights.digest: checkpoint digest {stored} does not match prepared digest {prepared} for calc_flow.static_input.digest.v1"
            )
        );
        assert!(
            !reopened.load(Ordering::SeqCst),
            "recovery mismatch must fail before the source opens"
        );

        // The graceful shutdown wrote a terminal manifest, so the identical
        // restart recovers to the terminal outcome without reopening sources
        // (D11); the comparison itself must simply succeed.
        let terminal_opened = Arc::new(AtomicBool::new(false));
        let job = start_with(
            static_plan(),
            directory.path(),
            Arc::clone(&terminal_opened),
            BTreeMap::from([("weights".into(), weights_table(&[1.0, 2.0, 3.0]))]),
        )
        .await
        .expect("restart with the identical value must pass the digest comparison");
        let outcome = tokio::time::timeout(Duration::from_secs(10), job.wait())
            .await
            .expect("terminal recovery must settle");
        assert!(
            outcome.completed_epoch >= Some(epoch),
            "terminal recovery keeps the completed epoch"
        );
    }
}
