use thiserror::Error;

#[derive(Debug, Error)]
/// Public v2 error surface.
///
/// The enum remains non-exhaustive so the stable API can add precise typed
/// failures at recovery and runtime boundaries without freezing every variant.
#[non_exhaustive]
pub enum CalcFlowError {
    #[error("invalid {field}: {message}")]
    InvalidArgument { field: String, message: String },
    #[error("project format version {found} is unsupported; expected {expected}")]
    UnsupportedVersion { expected: u32, found: u32 },
    #[error("graph compilation failed: {message}")]
    Compile { message: String },
    #[error("node {node_id} failed: {message}")]
    Operator { node_id: String, message: String },
    #[error("DataFusion failed for node {node_id:?}: {message}")]
    DataFusion {
        node_id: Option<String>,
        message: String,
    },
    #[error("external provider {provider}:{name}@{version} failed: {message}")]
    ExternalProvider {
        provider: String,
        name: String,
        version: String,
        message: String,
    },
    #[error("run {run_id} was cancelled")]
    Cancelled { run_id: String },
    #[error("checkpoint mismatch: {message}")]
    CheckpointMismatch { message: String },
    #[error("execution plan {pipeline_name:?} is exclusively leased by a runner")]
    PlanLeased { pipeline_name: String },
    #[error("execution plan {pipeline_name:?} requires recovery: {message}")]
    RecoveryRequired {
        pipeline_name: String,
        message: String,
    },
    #[error("stored document is invalid: {message}")]
    Format { message: String },
    #[error("{resource} {key:?} already exists")]
    Conflict { resource: String, key: String },
    #[error("{resource} {key:?} was not found")]
    NotFound { resource: String, key: String },
    /// A send on a stream edge whose receiver closed during job convergence
    /// (spec S10.1, API note A8).
    #[error("edge {edge:?} is closed")]
    EdgeClosed { edge: String },
    /// A supervised streaming-runtime task panicked (spec S8.4, API note A8).
    #[error("task {task_id} panicked: {message}")]
    TaskPanicked { task_id: u64, message: String },
    #[error("I/O failed for {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("internal invariant failed: {message}")]
    Internal { message: String },
}

pub type Result<T> = std::result::Result<T, CalcFlowError>;
