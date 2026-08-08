//! Calc Flow's Rust-native calculation engine.
//!
//! The public surface splits operators and plans by lifecycle (v3, plan
//! section 1.3): [`BatchOperator`] and [`BatchExecutionPlan`] run finite
//! one-shot graphs; [`StreamOperator`] and [`StreamExecutionPlan`] compile
//! continuously running graphs executed by the streaming runtime.

mod batch;
mod checkpoint;
mod config;
mod context;
mod datafusion;
mod error;
mod expression;
mod io;
mod json;
mod operator;
mod pipeline;
mod project_store;
mod runtime;
mod state;
mod time;
mod udf;

pub use batch::{Batch, BatchKind, BatchMetadata, ExternalPayload, TableBatch};
pub use checkpoint::{
    CHECKPOINT_FORMAT_VERSION, Checkpoint, CheckpointStore, FileCheckpointStore,
    MAX_CHECKPOINT_DOCUMENT_BYTES,
};
pub use config::{
    ArrowFieldSpec, DataSourceSpec, EdgeSpec, NodeSpec, OperatorSpec, PROJECT_FORMAT_VERSION,
    PipelineSpec, PortSpec, PositionSpec, ProjectSpec, RunOptions, ValidationIssue,
    ValidationReport, compile_project, project_json_schema, validate_project,
};
pub use context::{CancellationToken, RunContext};
pub use datafusion::{DataFusionConfig, DataFusionQueryMetric, DataFusionRuntime};
pub use error::{CalcFlowError, Result};
pub use io::{BatchingSource, Sink, Source, SourceItem};
pub use json::{JsonMap, MAX_JSON_DEPTH, canonical_json};
pub use operator::{
    AggregateFunction, AggregateSpec, BatchOperator, BatchOperatorContext, BatchOperatorFactory,
    EdgeCollector, ExpressionOperator, ExternalOperatorSpec, MAX_WINDOW_OVERLAP, NodeOperator,
    OperatorMetadata, OperatorStateSnapshot, Port, ProviderRegistry, SqlOperator, StreamCollector,
    StreamOperator, StreamOperatorContext, StreamOperatorFactory, UnionOperator,
    WindowAggregateOperator, WindowGeometry, WindowSpec,
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
    ChannelMetrics, EdgeReceiver, EdgeSender, EnvelopeCost, MicroBatchRunner, SinkRouter,
    StreamJobContext, StreamMessage, StreamMessageKind, StreamingRunner, edge_channel,
};
pub use state::{
    CheckpointManifest, CheckpointManifestFields, CursorManifestEntry, LocalStateBackend,
    MANIFEST_FORMAT_VERSION, MAX_MANIFEST_DOCUMENT_BYTES, ManifestExpectation,
    ManifestIngressState, OperatorIngressManifestEntry, OperatorManifestEntry, RecoveryStatus,
    RetentionClass, SinkDeliveryManifest, SinkManifestEntry, SourceManifestEntry,
    SourceWatermarkManifestState, StateBackend, StateHandle, StateLineageBackend, StateLineageKey,
};
pub use time::{Epoch, EventTime};
pub use udf::{
    UdfCatalogEntry, UdfKind, UdfReference, UdfRegistry, UdfRegistrySnapshot,
    validate_selected_udfs,
};

/// The crate version used by project and package diagnostics.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
