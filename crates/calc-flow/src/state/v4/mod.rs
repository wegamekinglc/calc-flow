pub(crate) mod coordinator;
pub(crate) mod manifest;
pub(crate) mod object;
pub(crate) mod opaque;
pub(crate) mod store;

pub(crate) use super::manifest::MAX_MANIFEST_DOCUMENT_BYTES;
pub(crate) use manifest::{OperatorV4, V4Manifest, manifest_fingerprint};
pub(crate) use object::{
    ArrowPayload, EncodedObject, NewObject, ObjectKind, ObjectPayload, OpaqueObject, V4Handle,
    decode_object, decode_opaque, encode_object, encode_opaque,
};
pub(crate) use store::V4Store;

use crate::{CalcFlowError, Result};

pub(crate) fn format_error(message: String) -> CalcFlowError {
    CalcFlowError::Format { message }
}

pub(crate) fn mismatch(message: String) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch { message }
}
