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
    for (position, (_, path)) in epochs.iter().enumerate() {
        if position >= retained_epochs {
            std::fs::remove_file(path).map_err(|error| io_error(path, error))?;
            continue;
        }
        let bytes = std::fs::read(path).map_err(|error| io_error(path, error))?;
        let manifest = V4Manifest::from_bytes(&bytes)?;
        collect_references(&manifest, &mut referenced);
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
