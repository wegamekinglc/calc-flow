//! Incremental state identity and persistence contracts for continuous jobs.
//!
//! State backends expose operations only through an exclusive lineage-scoped
//! session. This keeps raw pipeline and operator identities out of managed
//! filesystem paths and gives M5 one backend-neutral recovery boundary.

mod backend;
mod manifest;
#[allow(
    dead_code,
    reason = "M4 state inventory is consumed by local compaction and window snapshots in later work packages"
)]
mod segment;

pub use backend::{StateBackend, StateHandle, StateLineageBackend, StateLineageKey};
pub use manifest::{
    CheckpointManifest, CheckpointManifestFields, CursorManifestEntry, MANIFEST_FORMAT_VERSION,
    MAX_MANIFEST_DOCUMENT_BYTES, ManifestExpectation, ManifestIngressState,
    OperatorIngressManifestEntry, OperatorManifestEntry, RecoveryStatus, RetentionClass,
    SinkDeliveryManifest, SinkManifestEntry, SourceManifestEntry, SourceWatermarkManifestState,
};
