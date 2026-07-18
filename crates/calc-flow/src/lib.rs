//! Calc Flow's Rust-native v2 calculation engine.

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
    ExpressionOperator, ExternalOperatorFactory, ExternalOperatorSpec, Operator, OperatorContext,
    OperatorDefinition, Port, ProviderRegistry, SqlOperator,
};
pub use pipeline::{
    Edge, ExecutionOptions, ExecutionPlan, NodeTiming, PipelineBuilder, PortEndpoint, RunMetadata,
    RunResult,
};
pub use project_store::{
    FileProjectStore, MAX_PROJECT_DOCUMENT_BYTES, ProjectStore, export_project_json,
    export_project_yaml, import_project_json, import_project_json_with_limit, import_project_yaml,
    import_project_yaml_with_limit,
};
pub use runtime::{MicroBatchRunner, SinkRouter, StreamingRunner};
pub use udf::{
    UdfCatalogEntry, UdfKind, UdfReference, UdfRegistry, UdfRegistrySnapshot,
    validate_selected_udfs,
};

/// The crate version used by project and package diagnostics.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
