pub(in crate::state) mod manifest;
pub(in crate::state) mod object;
pub(in crate::state) mod store;

pub(in crate::state) use super::manifest::MAX_MANIFEST_DOCUMENT_BYTES;
pub(in crate::state) use manifest::{
    JobKind, ManifestExpectationV4, OperatorV4, PartitionV4, SinkV4, SourceHistoryV4, SourceV4,
    V4Manifest, manifest_fingerprint, recovery_fingerprint,
};
pub(in crate::state) use object::{
    ArrowPayload, DecodedObject, EncodedObject, NewObject, ObjectKind, ObjectPayload, OpaqueObject,
    V4Handle, decode_object, decode_opaque, encode_object, encode_opaque,
};

use crate::{CalcFlowError, Result};

pub(in crate::state) fn format_error(message: String) -> CalcFlowError {
    CalcFlowError::Format { message }
}

pub(in crate::state) fn mismatch(message: String) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch { message }
}
