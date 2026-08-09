//! Incremental state identity and persistence contracts for continuous jobs.
//!
//! State backends expose operations only through an exclusive lineage-scoped
//! session. This keeps raw pipeline and operator identities out of managed
//! filesystem paths and gives M5 one backend-neutral recovery boundary.

mod backend;
mod local;
mod manifest;
mod segment;
mod transaction;

#[cfg(test)]
pub(crate) use transaction::ManifestTransactionFaultPoint;
#[allow(
    unused_imports,
    reason = "M5 state transaction types are consumed by the private checkpoint runtime"
)]
pub(crate) use transaction::{
    ManifestPublication, ManifestTransaction, PreparedEpochManifest, PreparedManifestIdentity,
    SelectedManifest, StagedOperatorState,
};

pub(crate) use segment::{
    SegmentDescriptor, SegmentKind, StateInventory, StateOperation, fold_state_segments,
};

pub use backend::{StateBackend, StateHandle, StateLineageBackend, StateLineageKey};
pub use local::LocalStateBackend;
pub use manifest::{
    CheckpointManifest, CheckpointManifestFields, CursorManifestEntry, MANIFEST_FORMAT_VERSION,
    MAX_MANIFEST_DOCUMENT_BYTES, ManifestExpectation, ManifestIngressState,
    OperatorIngressManifestEntry, OperatorManifestEntry, RecoveryStatus, RetentionClass,
    SinkDeliveryManifest, SinkManifestEntry, SourceManifestEntry, SourceWatermarkManifestState,
};
