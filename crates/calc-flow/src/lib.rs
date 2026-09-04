//! Calc Flow's Rust-native calculation engine.
//!
//! The public surface splits operators and plans by lifecycle (v3, plan
//! section 1.3): [`BatchOperator`] and [`BatchExecutionPlan`] run finite
//! one-shot graphs; [`StreamOperator`] and [`StreamExecutionPlan`] compile
//! continuously running graphs executed by the streaming runtime.
//!
//! The removed v2 continuous-runtime owners and checkpoint store are not
//! available after the A6 cutover.
//!
//! ```compile_fail
//! use calc_flow::MicroBatchRunner;
//! ```
//!
//! ```compile_fail
//! use calc_flow::FileCheckpointStore;
//! ```
//!
//! ```compile_fail
//! use calc_flow::StreamingRunner;
//!
//! fn removed_push_step() {
//!     let _ = StreamingRunner::step;
//! }
//! ```

mod batch;
mod config;
mod connector;
mod context;
mod continuous;
mod datafusion;
mod datafusion_rolling;
mod error;
mod expression;
mod json;
mod operator;
mod pipeline;
mod project_store;
mod runtime;
mod state;
mod static_input;
mod time;
mod udf;

pub use batch::{
    Batch, BatchKind, BatchMetadata, ExternalPayload, StaticArraySnapshot, StaticArrayValues,
    TableBatch,
};
pub use config::{
    ArrowFieldSpec, ConnectorRef, DataSourceSpec, DeliveryRequest, EdgeSpec, FormatRef, NodeSpec,
    OperatorSpec, PROJECT_FORMAT_VERSION, PipelineSpec, PortSpec, PositionSpec, ProjectSinkBinding,
    ProjectSourceBinding, ProjectSpec, ProjectWatermarkPolicy, RunOptions, RuntimeSpec,
    StateConfig, StreamRunOptions, ValidationIssue, ValidationReport, compile_project,
    compile_stream_project, compile_stream_project_graph, project_json_schema, validate_project,
};
pub use connector::{
    ConnectorCapabilities, ConnectorDescriptor, ConnectorError, ConnectorFactories,
    ConnectorIdentity, ConnectorKind, ConnectorOperation, ConnectorRegistry,
    ConnectorRegistrySnapshot, ConnectorSinkFactory, ConnectorSourceFactory, DecodeBounds,
    DeliveryCapability, DeliveryParticipant, DeliveryProof, EnvironmentSecretResolver,
    FormatDecoder, FormatDescriptor, FormatEncoder, FormatIdentity, ParticipantRole,
    ReplayCapability, SecretHandle, SecretReference, SecretResolver, SecretResolverKind,
    TransactionSupport, WatermarkSupport, validate_connector_options, validate_delivery_guarantee,
};
pub use context::{CancellationToken, RunContext};
pub use continuous::{
    CheckpointPhase, CheckpointStatus, ComponentKind, Cursor, DurableCursorAcknowledger,
    EdgeStatus, JobOutcome, JobState, JobStatus, ManagedCheckpointRuntime,
    NativeWatermarkCapability, OperatorStatus, OutputDeliveryStatus, ReplayPositioning,
    SinkBinding, SinkDelivery, SinkRecovery, SinkStatus, SourceBinding, SourceCapabilities,
    SourceCheckpointGate, SourceDeliveryCapability, SourceEvent, SourceSchema, SourceStatus,
    StreamSink, StreamSource, StreamingError, StreamingErrorCategory, StreamingFailureReason,
    StreamingJob, StreamingRunner, TerminalCause, TransactionalStreamSink, WatermarkPolicy,
};
pub use datafusion::{
    DATAFUSION_ACTIVE_ENTITIES_METADATA_KEY, DataFusionConfig, DataFusionParallelismMode,
    DataFusionQueryMetric, DataFusionRuntime,
};
pub use error::{CalcFlowError, Result};
pub use json::{JsonMap, MAX_JSON_DEPTH, canonical_json};
pub use operator::{
    AggregateFunction, AggregateSpec, BatchOperator, BatchOperatorContext, BatchOperatorFactory,
    CROSS_SECTION_CONFIGURATION_VERSION, CROSS_SECTION_STATE_LAYOUT_VERSION,
    CrossSectionGroupingSpec, CrossSectionOperator, CrossSectionOutputSpec, CrossSectionSpec,
    CrossSectionValuePolicy, EdgeCollector, ExpressionOperator, ExternalOperatorSpec,
    IngressProgress, IngressProgressSnapshot, IngressState, JoinStateLimits, JoinTimeBounds,
    LateErrorScope, LatePolicySpec, MAX_WINDOW_OVERLAP, NodeOperator, NullPlacement,
    OperatorMetadata, OperatorStateSnapshot, Port, ProviderRegistry,
    ROLLING_COLUMNAR_STATE_LAYOUT_VERSION, ROLLING_CONFIGURATION_VERSION,
    ROLLING_EWMA_STATE_LAYOUT_VERSION, ROLLING_STATE_LAYOUT_VERSION, RankTieMethod,
    RollingFloatPrimitiveSpec, RollingFrameSpec, RollingNumericalProfile, RollingOperator,
    RollingOutputSpec, RollingSpec, RollingValuePolicy, STREAM_JOIN_MAX_SAFE_JSON_INTEGER,
    STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1, SortDirection, SqlOperator, StateSegment,
    StreamCollector, StreamJoinOperator, StreamJoinSideStatus, StreamJoinSpec, StreamJoinStatus,
    StreamJoinType, StreamOperator, StreamOperatorContext, StreamOperatorFactory,
    StreamOperatorLifecycle, UnionOperator, WindowAggregateOperator, WindowGeometry, WindowSpec,
};
pub use pipeline::{
    BatchExecutionPlan, DeliveryGuarantee, Edge, EdgeBudget, ExecutionOptions, NodeTiming,
    PipelineBuilder, PortEndpoint, RunMetadata, RunResult, StreamExecutionPlan, StreamRequirements,
    StreamRuntimeConfig,
};
pub use project_store::{
    FileProjectStore, MAX_PROJECT_DOCUMENT_BYTES, ProjectStore, export_project_json,
    export_project_yaml, import_project_json, import_project_json_with_limit, import_project_yaml,
    import_project_yaml_with_limit,
};
pub use runtime::{
    ChannelMetrics, EdgeReceiver, EdgeSender, EnvelopeCost, StreamJobContext, StreamMessage,
    StreamMessageKind, edge_channel,
};
pub use state::{
    CheckpointManifest, CheckpointManifestFields, CursorManifestEntry, LocalStateBackend,
    MANIFEST_FORMAT_VERSION, MAX_MANIFEST_DOCUMENT_BYTES, ManifestExpectation,
    ManifestIngressState, OperatorIngressManifestEntry, OperatorManifestEntry, RecoveryStatus,
    RetentionClass, SinkDeliveryManifest, SinkManifestEntry, SourceManifestEntry,
    SourceWatermarkManifestState, StateBackend, StateHandle, StateLineageBackend, StateLineageKey,
};
pub use static_input::{
    STATIC_INPUT_DIGEST_VERSION, StaticInputDigest, StaticInputSpec, StaticMutability,
};
pub use time::{Epoch, EventTime};
pub use udf::{
    UdfCatalogEntry, UdfKind, UdfReference, UdfRegistry, UdfRegistrySnapshot,
    validate_selected_udfs,
};

/// The crate version used by project and package diagnostics.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
