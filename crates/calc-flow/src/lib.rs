//! Calc Flow's Rust-native v2 calculation engine.

mod context;
mod error;
mod json;

pub use context::{CancellationToken, RunContext};
pub use error::{CalcFlowError, Result};
pub use json::{JsonMap, canonical_json};

/// The crate version used by project and package diagnostics.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
