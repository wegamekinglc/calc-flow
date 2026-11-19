pub(crate) mod budget;
pub(crate) mod coordinator;
pub(crate) mod manifest;
pub(crate) mod native;
pub(crate) mod object;
pub(crate) mod opaque;
pub(crate) mod store;

pub(crate) use super::manifest::MAX_MANIFEST_DOCUMENT_BYTES;
pub(crate) use budget::{V4CaptureReservation, V4InflightBudget};
pub(crate) use manifest::{
    JobKind, ManifestExpectationV4, OperatorV4, SinkV4, SourceV4, V4Manifest, manifest_fingerprint,
};
pub use object::V4Handle as SourceHistorySegmentHandle;
pub(crate) use object::{
    ArrowPayload, EncodedObject, NewObject, ObjectKind, ObjectPayload, OpaqueObject, V4Handle,
    decode_object, decode_opaque, encode_object, encode_opaque,
};
pub(crate) use opaque::V4OpaqueAdapter;
pub(crate) use store::V4Store;

use crate::{CalcFlowError, Result};

pub(crate) fn format_error(message: String) -> CalcFlowError {
    CalcFlowError::Format { message }
}

pub(crate) fn mismatch(message: String) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch { message }
}
