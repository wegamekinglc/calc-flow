//! Opaque v4 adapters for operators whose state-table migration is pending.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use super::{OpaqueObject, OperatorV4, Result, V4Handle, V4Store, mismatch};
use crate::{JsonMap, OperatorStateSnapshot, StateSegment};

#[derive(Debug)]
pub(crate) struct V4OpaqueAdapter {
    owner: String,
    inline_metadata: JsonMap,
    state_fingerprint: String,
    blocks: Vec<OpaqueObject>,
}

fn finish_encoded(
    encoded: Vec<super::EncodedObject>,
    state_fingerprint: String,
    inline_metadata: JsonMap,
) -> (OperatorV4, Vec<V4Handle>) {
    let handles = encoded
        .into_iter()
        .map(|object| object.handle)
        .collect::<Vec<_>>();
    (
        OperatorV4::Opaque {
            progress: BTreeMap::new(),
            output_frontier: None,
            state_fingerprint,
            inline_metadata,
            blocks: handles.clone(),
        },
        handles,
    )
}

#[allow(
    dead_code,
    reason = "P3.4 runner uses opaque adapters for un-migrated operators"
)]
impl V4OpaqueAdapter {
    pub(crate) fn from_snapshot(owner: &str, snapshot: &OperatorStateSnapshot) -> Self {
        let epoch = snapshot
            .inline_metadata
            .get("epoch")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let inline_metadata = snapshot.inline_metadata.clone();
        let metadata_value =
            serde_json::to_value(&inline_metadata).expect("snapshot metadata is JSON");
        let inline_bytes = crate::canonical_json(&metadata_value)
            .expect("snapshot metadata is JSON")
            .into_bytes();
        let state_fingerprint = hex::encode(Sha256::digest(&inline_bytes));
        let blocks = snapshot
            .segments
            .iter()
            .map(|(id, segment)| {
                let bytes = segment.bytes().to_vec();
                OpaqueObject {
                    owner: owner.to_owned(),
                    object_id: id.clone(),
                    generation: epoch,
                    sha256: hex::encode(Sha256::digest(&bytes)),
                    bytes,
                }
            })
            .collect();
        Self {
            owner: owner.to_owned(),
            inline_metadata,
            state_fingerprint,
            blocks,
        }
    }

    pub(crate) fn owner(&self) -> &str {
        &self.owner
    }

    pub(crate) fn blocks_for_encoding(&self) -> Vec<OpaqueObject> {
        self.blocks.clone()
    }

    pub(crate) async fn persist_preencoded(
        self,
        store: &mut V4Store,
        encoded: Vec<super::EncodedObject>,
    ) -> Result<(OperatorV4, Vec<V4Handle>)> {
        let encoded = store.write_encoded_opaque(encoded).await?;
        Ok(finish_encoded(
            encoded,
            self.state_fingerprint,
            self.inline_metadata,
        ))
    }

    pub(crate) async fn persist(
        self,
        store: &mut V4Store,
        epoch: u64,
    ) -> Result<(OperatorV4, Vec<V4Handle>)> {
        let encoded = store.write_opaque(epoch, self.blocks).await?;
        Ok(finish_encoded(
            encoded,
            self.state_fingerprint,
            self.inline_metadata,
        ))
    }

    pub(crate) fn restore(
        store: &mut V4Store,
        manifest: &super::V4Manifest,
        owner: &str,
    ) -> Result<OperatorStateSnapshot> {
        let Some(OperatorV4::Opaque {
            inline_metadata,
            blocks,
            state_fingerprint,
            ..
        }) = manifest.operators.get(owner)
        else {
            return Err(mismatch(
                "opaque operator is absent from the manifest".into(),
            ));
        };
        let mut segments = BTreeMap::new();
        for handle in blocks {
            let bytes = store.read_opaque(handle)?;
            let id = handle.object_id.clone();
            segments.insert(id, StateSegment::new(bytes));
        }
        let metadata_value =
            serde_json::to_value(inline_metadata).expect("manifest metadata is JSON");
        let expected = hex::encode(Sha256::digest(
            crate::canonical_json(&metadata_value)
                .expect("manifest metadata is JSON")
                .as_bytes(),
        ));
        if *state_fingerprint != expected {
            return Err(mismatch(
                "opaque state fingerprint differs from metadata".into(),
            ));
        }
        Ok(OperatorStateSnapshot {
            inline_metadata: inline_metadata.clone(),
            segments,
        })
    }
}
