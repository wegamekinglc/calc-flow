#![allow(
    dead_code,
    reason = "P3.4 wires opaque v4 objects into the checkpoint coordinator"
)]
use std::{collections::HashMap, mem::size_of, path::Path, sync::Arc};

use datafusion::arrow::{
    array::ArrayRef,
    datatypes::{Field, Schema},
    ipc::writer::FileWriter,
    record_batch::RecordBatch,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{Result, format_error, mismatch};
use crate::{canonical_json, json::validate_portable_identifier};

pub(crate) const V4_MAGIC: &[u8; 6] = b"CFST4\0";
const HEADER_BYTES: usize = V4_MAGIC.len() + 1 + size_of::<u64>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ObjectKind {
    Page,
    PartitionIndex,
    SourceHistory,
    Sink,
}

impl ObjectKind {
    fn tag(self) -> u8 {
        match self {
            Self::Page => 1,
            Self::PartitionIndex => 2,
            Self::SourceHistory => 3,
            Self::Sink => 4,
        }
    }

    fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::Page),
            2 => Ok(Self::PartitionIndex),
            3 => Ok(Self::SourceHistory),
            4 => Ok(Self::Sink),
            _ => Err(format_error("unknown checkpoint-v4 object kind".into())),
        }
    }

    pub(crate) fn directory(self) -> &'static str {
        match self {
            Self::Page => "pages",
            Self::PartitionIndex => "partitions",
            Self::SourceHistory => "source-history",
            Self::Sink => "sinks",
        }
    }
}

/// Immutable opaque segment reference used by native v4 source history.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V4Handle {
    pub(crate) owner: String,
    pub(crate) epoch: u64,
    pub(crate) object_id: String,
    pub(crate) relative_path: String,
    pub(crate) byte_len: u64,
    pub(crate) sha256: String,
}

impl V4Handle {
    /// Returns the connector-defined stable segment identifier.
    pub fn segment_id(&self) -> &str {
        &self.object_id
    }

    /// Returns the exact raw segment byte length used by history budgets.
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub(crate) fn validate(&self, manifest_epoch: u64) -> Result<()> {
        validate_portable_identifier("checkpoint-v4 handle owner", &self.owner)?;
        validate_object_id(&self.object_id)?;
        validate_relative_path(&self.relative_path)?;
        validate_sha256(&self.sha256)?;
        if self.epoch == 0 || self.epoch > manifest_epoch {
            return Err(mismatch(
                "checkpoint-v4 handle epoch is newer than the manifest".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ArrowPayload {
    pub(crate) fields: Vec<Field>,
    pub(crate) columns: Vec<ArrayRef>,
    pub(crate) metadata: HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub(crate) enum ObjectPayload {
    Bytes(Vec<u8>),
    Json(serde_json::Value),
    Arrow(ArrowPayload),
}

#[derive(Clone, Debug)]
pub(crate) struct NewObject {
    pub(crate) kind: ObjectKind,
    pub(crate) owner: String,
    pub(crate) object_id: String,
    pub(crate) generation: u64,
    pub(crate) payload: ObjectPayload,
}

#[derive(Clone, Debug)]
pub(crate) struct OpaqueObject {
    pub(crate) owner: String,
    pub(crate) object_id: String,
    pub(crate) generation: u64,
    pub(crate) bytes: Vec<u8>,
    pub(crate) sha256: String,
}

pub(crate) fn encode_opaque(
    object: &OpaqueObject,
    root: &Path,
    epoch: u64,
) -> Result<EncodedObject> {
    validate_sha256(&object.sha256)?;
    let lineage = root
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format_error("checkpoint-v4 lineage name is absent".into()))?;
    let relative_path = format!(
        "{lineage}/epochs/{epoch}/{}/opaque/{}/{}/{}.cfst4",
        object.owner, object.object_id, object.generation, object.sha256
    );
    let handle = V4Handle {
        owner: object.owner.clone(),
        epoch,
        object_id: object.object_id.clone(),
        relative_path,
        byte_len: object.bytes.len() as u64,
        sha256: object.sha256.clone(),
    };
    handle.validate(epoch)?;
    Ok(EncodedObject {
        bytes: object.bytes.clone(),
        handle,
    })
}

pub(crate) fn decode_opaque(handle: &V4Handle, bytes: &[u8]) -> Result<Vec<u8>> {
    handle.validate(u64::MAX)?;
    if bytes.len() as u64 != handle.byte_len || hex::encode(Sha256::digest(bytes)) != handle.sha256
    {
        return Err(mismatch(
            "checkpoint-v4 opaque block checksum mismatch".into(),
        ));
    }
    Ok(bytes.to_vec())
}

#[derive(Clone, Debug)]
pub(crate) struct EncodedObject {
    pub(crate) bytes: Vec<u8>,
    pub(crate) handle: V4Handle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DecodedObject {
    pub(crate) kind: ObjectKind,
    pub(crate) body: Vec<u8>,
}

pub(crate) fn encode_object(object: &NewObject, root: &Path, epoch: u64) -> Result<EncodedObject> {
    let body = payload_bytes(&object.payload)?;
    let mut bytes = Vec::with_capacity(HEADER_BYTES + body.len());
    bytes.extend_from_slice(V4_MAGIC);
    bytes.push(object.kind.tag());
    bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&body);
    let sha256 = hex::encode(Sha256::digest(&bytes));
    let handle = V4Handle {
        owner: object.owner.clone(),
        epoch,
        object_id: object.object_id.clone(),
        relative_path: object_path(root, object, epoch, &sha256)?,
        byte_len: bytes.len() as u64,
        sha256,
    };
    handle.validate(epoch)?;
    Ok(EncodedObject { bytes, handle })
}

pub(crate) fn decode_object(handle: &V4Handle, bytes: &[u8]) -> Result<DecodedObject> {
    handle.validate(u64::MAX)?;
    if bytes.len() as u64 != handle.byte_len {
        return Err(mismatch(
            "checkpoint-v4 object length differs from its handle".into(),
        ));
    }
    if hex::encode(Sha256::digest(bytes)) != handle.sha256 {
        return Err(mismatch(
            "checkpoint-v4 object checksum differs from its handle".into(),
        ));
    }
    if bytes.len() < HEADER_BYTES || &bytes[..V4_MAGIC.len()] != V4_MAGIC {
        return Err(format_error(
            "checkpoint-v4 object envelope is invalid".into(),
        ));
    }
    let kind = ObjectKind::from_tag(bytes[V4_MAGIC.len()])?;
    let start = V4_MAGIC.len() + 1;
    let declared = u64::from_le_bytes(
        bytes[start..HEADER_BYTES]
            .try_into()
            .expect("header length checked"),
    );
    if declared != (bytes.len() - HEADER_BYTES) as u64 {
        return Err(format_error(
            "checkpoint-v4 object payload length mismatch".into(),
        ));
    }
    Ok(DecodedObject {
        kind,
        body: bytes[HEADER_BYTES..].to_vec(),
    })
}

pub(crate) fn payload_bytes(payload: &ObjectPayload) -> Result<Vec<u8>> {
    match payload {
        ObjectPayload::Bytes(bytes) => Ok(bytes.clone()),
        ObjectPayload::Json(value) => canonical_json(value).map(String::into_bytes),
        ObjectPayload::Arrow(payload) => arrow_bytes(payload),
    }
}

fn arrow_bytes(payload: &ArrowPayload) -> Result<Vec<u8>> {
    let schema = Schema::new(payload.fields.clone()).with_metadata(payload.metadata.clone());
    let record = RecordBatch::try_new(Arc::new(schema), payload.columns.clone())
        .map_err(|error| format_error(format!("checkpoint-v4 Arrow schema mismatch: {error}")))?;
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, record.schema().as_ref())
        .map_err(|error| format_error(format!("checkpoint-v4 Arrow writer failed: {error}")))?;
    writer
        .write(&record)
        .map_err(|error| format_error(format!("checkpoint-v4 Arrow write failed: {error}")))?;
    writer
        .finish()
        .map_err(|error| format_error(format!("checkpoint-v4 Arrow finish failed: {error}")))?;
    drop(writer);
    Ok(bytes)
}

fn object_path(root: &Path, object: &NewObject, epoch: u64, sha256: &str) -> Result<String> {
    validate_object_id(&object.object_id)?;
    let lineage = root
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format_error("checkpoint-v4 lineage name is absent".into()))?;
    validate_portable_identifier("checkpoint-v4 lineage", lineage)?;
    Ok(format!(
        "{lineage}/epochs/{epoch}/{}/{}/{}/{}/{sha256}.cfst4",
        object.owner,
        object.kind.directory(),
        object.object_id,
        object.generation,
    ))
}

fn validate_object_id(object_id: &str) -> Result<()> {
    if object_id.is_empty() || object_id.split('/').any(str::is_empty) {
        return Err(format_error("checkpoint-v4 object ID is empty".into()));
    }
    for component in object_id.split('/') {
        validate_portable_identifier("checkpoint-v4 object ID component", component)?;
    }
    Ok(())
}

fn validate_relative_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::Prefix(_)
            )
        })
        || path.to_string_lossy().contains('\\')
    {
        return Err(format_error("checkpoint-v4 path is not portable".into()));
    }
    Ok(())
}

fn validate_sha256(sha256: &str) -> Result<()> {
    if sha256.len() != 64
        || !sha256
            .chars()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase())
    {
        return Err(format_error(
            "checkpoint-v4 SHA-256 must be lowercase hexadecimal".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{NewObject, ObjectKind, ObjectPayload, decode_object, encode_object};

    fn object() -> NewObject {
        NewObject {
            kind: ObjectKind::Page,
            owner: "rolling".into(),
            object_id: "0/0".into(),
            generation: 3,
            payload: ObjectPayload::Bytes(vec![1, 2, 3, 4]),
        }
    }

    #[test]
    fn object_envelope_roundtrip_and_checksum_rejection() {
        let encoded = encode_object(&object(), Path::new("lineage"), 7).unwrap();
        let decoded = decode_object(&encoded.handle, &encoded.bytes).unwrap();
        assert_eq!(decoded.kind, ObjectKind::Page);
        assert_eq!(decoded.body, vec![1, 2, 3, 4]);
        assert!(
            encoded
                .handle
                .relative_path
                .contains("epochs/7/rolling/pages/0/0/3/")
        );
        let mut damaged = encoded.bytes.clone();
        *damaged.last_mut().unwrap() ^= 1;
        assert!(decode_object(&encoded.handle, &damaged).is_err());
        assert!(decode_object(&encoded.handle, &encoded.bytes[..10]).is_err());
    }

    #[test]
    fn object_envelope_rejects_unknown_kind_and_truncation() {
        let encoded = encode_object(&object(), Path::new("lineage"), 7).unwrap();
        let mut unknown = encoded.bytes.clone();
        unknown[6] = 99;
        assert!(decode_object(&encoded.handle, &unknown).is_err());
        assert!(decode_object(&encoded.handle, b"CFST4\0").is_err());
    }
}
