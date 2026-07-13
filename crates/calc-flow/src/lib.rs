//! Calc Flow's Rust-native v2 calculation engine.

mod batch;
mod context;
mod datafusion;
mod error;
mod expression;
mod json;

pub use batch::{Batch, BatchKind, BatchMetadata, ExternalPayload, TableBatch};
pub use context::{CancellationToken, RunContext};
pub use datafusion::{DataFusionConfig, DataFusionQueryMetric, DataFusionRuntime};
pub use error::{CalcFlowError, Result};
pub use expression::{split_assignment, sql_projection, validate_select_query};
pub use json::{JsonMap, canonical_json};

/// The crate version used by project and package diagnostics.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
