//! Opaque v4 adapters for operators whose state-table migration is pending.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use super::{
    OpaqueObject, OperatorV4, Result, V4Handle, V4Store, mismatch,
};
use crate::{JsonMap, OperatorStateSnapshot, StateSegment};

#[derive(Debug)]
pub(in crate::state) struct V4OpaqueAdapter {
    owner: String,
    inline_metadata: JsonMap,
    state_fingerprint: String,
    blocks: Vec<OpaqueObject>,
}

#[allow(
    dead_code,
    reason = "P3.4 runner uses opaque adapters for un-migrated operators"
)]
impl V4OpaqueAdapter {
    pub(in crate::state) fn from_snapshot(owner: &str, snapshot: &OperatorStateSnapshot) -> Self {
        let epoch = snapshot
            .inline_metadata
            .get("epoch")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let inline_metadata = snapshot.inline_metadata.clone();
        let inline_bytes = serde_json::to_vec(&inline_metadata).expect("snapshot metadata is JSON");
        let state_fingerprint = hex::encode(Sha256::digest(&inline_bytes));
        let mut blocks = vec![OpaqueObject {
            owner: owner.to_owned(),
            object_id: "inline".into(),
            generation: epoch,
            bytes: inline_bytes,
            sha256: state_fingerprint.clone(),
        }];
        blocks.extend(snapshot.segments.iter().map(|(id, segment)| {
            let bytes = segment.bytes().to_vec();
            OpaqueObject {
                owner: owner.to_owned(),
                object_id: id.clone(),
                generation: epoch,
                sha256: hex::encode(Sha256::digest(&bytes)),
                bytes,
            }
        }));
        Self {
            owner: owner.to_owned(),
            inline_metadata,
            state_fingerprint,
            blocks,
        }
    }

    pub(in crate::state) fn owner(&self) -> &str {
        &self.owner
    }

    pub(in crate::state) async fn persist(
        self,
        store: &mut V4Store,
        epoch: u64,
    ) -> Result<(OperatorV4, Vec<V4Handle>)> {
        let encoded = store.write_opaque(epoch, self.blocks).await?;
        let handles = encoded.into_iter().map(|object| object.handle).collect::<Vec<_>>();
        Ok((
            OperatorV4::Opaque {
                progress: BTreeMap::new(),
                output_frontier: None,
                state_fingerprint: self.state_fingerprint,
                inline_metadata: self.inline_metadata,
                blocks: handles.clone(),
            },
            handles,
        ))
    }

    pub(in crate::state) fn restore(
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
            return Err(mismatch("opaque operator is absent from the manifest".into()));
        };
        let mut segments = BTreeMap::new();
        for handle in blocks {
            let bytes = store.read_opaque(handle)?;
            let id = handle.object_id.clone();
            if id == "inline" {
                continue;
            }
            segments.insert(id, StateSegment::new(bytes));
        }
        let expected = hex::encode(Sha256::digest(
            serde_json::to_vec(inline_metadata).expect("manifest metadata is JSON"),
        ));
        if *state_fingerprint != expected {
            return Err(mismatch("opaque state fingerprint differs from metadata".into()));
        }
        Ok(OperatorStateSnapshot {
            inline_metadata: inline_metadata.clone(),
            segments,
        })
    }
}
