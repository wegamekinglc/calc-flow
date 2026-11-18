use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use super::super::checkpoint_runtime::OpenedCheckpointRuntime;
use crate::state::v4::{OperatorV4, V4Manifest};
use crate::{CalcFlowError, Result};

pub(super) async fn retain(checkpoint: &OpenedCheckpointRuntime) -> Result<usize> {
    retain_root(&checkpoint.manifest_root, checkpoint.config.retained_epochs).await
}

pub(super) async fn retain_root(root: &Path, retained_epochs: usize) -> Result<usize> {
    let root = root.to_owned();
    tokio::task::spawn_blocking(move || retain_sync(&root, retained_epochs))
        .await
        .map_err(|error| CalcFlowError::Internal {
            message: format!("checkpoint-v4 retention worker failed: {error}"),
        })?
}

fn retain_sync(root: &Path, retained_epochs: usize) -> Result<usize> {
    let manifests = manifest_files(root)?;
    let mut epochs = manifests;
    epochs.sort_by(|left, right| right.0.cmp(&left.0));
    let mut referenced = BTreeSet::new();
    let retained = retained_epochs.min(epochs.len());
    for (_, path) in &epochs[..retained] {
        let bytes = std::fs::read(path).map_err(|error| io_error(path, error))?;
        let manifest = V4Manifest::from_bytes(&bytes)?;
        collect_references(&manifest, &mut referenced);
    }
    for (_, path) in &epochs[retained..] {
        std::fs::remove_file(path).map_err(|error| io_error(path, error))?;
    }
    remove_unreferenced(root, &referenced)
}

fn manifest_files(root: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let directory = root.join("lineage/manifests");
    let entries = match std::fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(io_error(&directory, error)),
    };
    let mut manifests = Vec::new();
    for entry in entries {
        let path = entry.map_err(|error| io_error(&directory, error))?.path();
        let temporary = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                name.starts_with("epoch-")
                    && Path::new(name)
                        .extension()
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("tmp"))
            });
        if temporary {
            std::fs::remove_file(&path).map_err(|error| io_error(&path, error))?;
            continue;
        }
        let Some(epoch) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("epoch-"))
            .and_then(|name| name.strip_suffix(".json"))
            .and_then(|name| name.parse::<u64>().ok())
        else {
            continue;
        };
        manifests.push((epoch, path));
    }
    Ok(manifests)
}

fn collect_references(manifest: &V4Manifest, referenced: &mut BTreeSet<String>) {
    for source in manifest.sources.values() {
        if let Some(history) = &source.history {
            referenced.extend(
                history
                    .segments
                    .iter()
                    .map(|handle| handle.relative_path.clone()),
            );
        }
    }
    for operator in manifest.operators.values() {
        match operator {
            OperatorV4::Opaque { blocks, .. } => {
                referenced.extend(blocks.iter().map(|handle| handle.relative_path.clone()));
            }
            OperatorV4::Table { partitions, .. } => {
                referenced.extend(
                    partitions
                        .iter()
                        .map(|partition| partition.root.relative_path.clone()),
                );
            }
        }
    }
    for sink in manifest.sinks.values() {
        referenced.extend(
            sink.blocks
                .iter()
                .map(|handle| handle.relative_path.clone()),
        );
    }
}

fn remove_unreferenced(root: &Path, referenced: &BTreeSet<String>) -> Result<usize> {
    let epochs = root.join("lineage/epochs");
    let mut directories = vec![epochs.clone()];
    let mut visited = Vec::new();
    let mut removed = 0;
    while let Some(directory) = directories.pop() {
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(io_error(&directory, error)),
        };
        visited.push(directory.clone());
        for entry in entries {
            let path = entry.map_err(|error| io_error(&directory, error))?.path();
            let metadata =
                std::fs::symlink_metadata(&path).map_err(|error| io_error(&path, error))?;
            if metadata.is_dir() {
                directories.push(path);
            } else {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|_| mismatch("v4 object escapes its lineage root"))?
                    .to_string_lossy()
                    .replace('\\', "/");
                if !referenced.contains(&relative) {
                    std::fs::remove_file(&path).map_err(|error| io_error(&path, error))?;
                    removed += 1;
                }
            }
        }
    }
    for directory in visited.into_iter().rev() {
        match std::fs::remove_dir(&directory) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::DirectoryNotEmpty => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(io_error(&directory, error)),
        }
    }
    Ok(removed)
}

fn mismatch(message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: message.into(),
    }
}

fn io_error(path: &Path, source: std::io::Error) -> CalcFlowError {
    CalcFlowError::Io {
        path: path.to_string_lossy().into_owned(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use sha2::{Digest, Sha256};

    use super::retain_root;
    use crate::state::v4::{
        JobKind, OperatorV4, SinkV4, SourceV4, V4Handle, V4Manifest, V4Store, manifest_fingerprint,
    };
    use crate::{JsonMap, SinkDeliveryManifest, SourceWatermarkManifestState};

    async fn object(store: &mut V4Store, epoch: u64, id: &str) -> V4Handle {
        let bytes = id.as_bytes().to_vec();
        store
            .write_opaque(
                epoch,
                vec![crate::state::v4::OpaqueObject {
                    owner: "operator".into(),
                    object_id: id.into(),
                    generation: epoch,
                    sha256: hex::encode(Sha256::digest(&bytes)),
                    bytes,
                }],
            )
            .await
            .unwrap()
            .remove(0)
            .handle
    }

    fn manifest(epoch: u64, blocks: Vec<V4Handle>) -> V4Manifest {
        let metadata = JsonMap::new();
        let state_fingerprint = hex::encode(Sha256::digest(b"{}"));
        let mut manifest = V4Manifest {
            format_version: 4,
            pipeline_name: "retention-test".into(),
            pipeline_fingerprint: "fingerprint".into(),
            runtime_config_hash: "runtime".into(),
            epoch,
            created_at: chrono::Utc::now(),
            recovery_status: "final".into(),
            job_kind: JobKind::Runner,
            recovery_fingerprint: "fingerprint".into(),
            sources: BTreeMap::from([(
                "source".into(),
                SourceV4 {
                    cursor: None,
                    identity_hash: "identity".into(),
                    next_sequence: 0,
                    ended: false,
                    watermark_policy: SourceWatermarkManifestState::Disabled { idle: false },
                    history: None,
                },
            )]),
            operators: BTreeMap::from([(
                "operator".into(),
                OperatorV4::Opaque {
                    progress: BTreeMap::new(),
                    output_frontier: None,
                    state_fingerprint,
                    inline_metadata: metadata,
                    blocks,
                },
            )]),
            sinks: BTreeMap::from([(
                "sink".into(),
                SinkV4 {
                    delivery: SinkDeliveryManifest::Ordinary,
                    pre_commit: None,
                    blocks: Vec::new(),
                },
            )]),
            static_inputs: BTreeMap::new(),
            manifest_fingerprint: String::new(),
        };
        manifest.manifest_fingerprint = manifest_fingerprint(&manifest).unwrap();
        manifest
    }

    #[tokio::test]
    async fn retention_preserves_carried_handles_and_removes_orphans() {
        let root = tempfile::tempdir().unwrap();
        let manifest_root = root.path().join("manifests");
        let mut store = V4Store::new(manifest_root.join("lineage"));
        let first = object(&mut store, 1, "first").await;
        let second = object(&mut store, 2, "second").await;
        store
            .publish_manifest(&manifest(1, vec![first.clone()]))
            .unwrap();
        store
            .publish_manifest(&manifest(2, vec![first.clone(), second.clone()]))
            .unwrap();
        let orphan = object(&mut store, 3, "orphan").await;
        assert_eq!(retain_root(&manifest_root, 1).await.unwrap(), 1);
        assert!(
            !manifest_root
                .join("lineage/manifests/epoch-1.json")
                .exists()
        );
        assert_eq!(store.read_opaque(&first).unwrap(), b"first");
        assert_eq!(store.read_opaque(&second).unwrap(), b"second");
        assert!(store.read_opaque(&orphan).is_err());
    }
}
