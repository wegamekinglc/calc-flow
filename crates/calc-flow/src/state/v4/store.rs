use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use super::{
    EncodedObject, NewObject, ObjectPayload, OpaqueObject, Result, V4Handle, V4Manifest,
    decode_object, decode_opaque, encode_object, encode_opaque, format_error, mismatch,
};
use crate::CalcFlowError;

const MAX_WORKERS: usize = 4;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(in crate::state) struct V4PersistenceStats {
    pub(in crate::state) content_hashes: u64,
    pub(in crate::state) file_syncs: u64,
    pub(in crate::state) staging_directory_syncs: u64,
    pub(in crate::state) committed_directory_syncs: u64,
    pub(in crate::state) directory_creation_syncs: u64,
    pub(in crate::state) manifest_file_syncs: u64,
    pub(in crate::state) manifest_root_syncs: u64,
    pub(in crate::state) read_back_hashes: u64,
    pub(in crate::state) recovery_hashes: u64,
    pub(in crate::state) carried_objects: u64,
    pub(in crate::state) opaque_blocks: u64,
}

#[derive(Debug)]
pub(in crate::state) struct V4Store {
    root: PathBuf,
    stats: V4PersistenceStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::state) enum V4Fault {
    Write,
    Checksum,
    DiskFull,
    WorkerPanic,
}

impl V4Store {
    pub(in crate::state) fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            stats: V4PersistenceStats::default(),
        }
    }

    pub(in crate::state) fn stats(&self) -> &V4PersistenceStats {
        &self.stats
    }

    pub(in crate::state) fn record_carried(&mut self, count: usize) {
        self.stats.carried_objects += count as u64;
    }

    pub(in crate::state) async fn write_objects(
        &mut self,
        epoch: u64,
        objects: Vec<NewObject>,
    ) -> Result<Vec<EncodedObject>> {
        self.write_objects_inner(epoch, objects, None).await
    }

    pub(in crate::state) async fn write_objects_with_fault(
        &mut self,
        epoch: u64,
        objects: Vec<NewObject>,
        fault: Option<V4Fault>,
    ) -> Result<Vec<EncodedObject>> {
        self.write_objects_inner(epoch, objects, fault).await
    }

    async fn write_objects_inner(
        &mut self,
        epoch: u64,
        objects: Vec<NewObject>,
        fault: Option<V4Fault>,
    ) -> Result<Vec<EncodedObject>> {
        let mut encoded = Vec::new();
        for chunk in objects.chunks(MAX_WORKERS) {
            let root = self.root.clone();
            let mut handles = chunk
                .iter()
                .map(|object| {
                    let root = root.clone();
                    let object = object.clone();
                    tokio::task::spawn_blocking(move || {
                        if fault == Some(V4Fault::WorkerPanic) {
                            panic!("injected checkpoint-v4 worker panic");
                        }
                        encode_object(&object, &root, epoch)
                    })
                })
                .collect::<Vec<_>>();
            let mut settled = Vec::new();
            for handle in &mut handles {
                settled.push(handle.await);
            }
            for result in settled {
                let object = result.map_err(|error| {
                    format_error(format!("checkpoint-v4 encode worker failed: {error}"))
                })?;
                encoded.push(object);
            }
        }
        encoded.sort_by(|left, right| {
            (
                &left.handle.owner,
                &left.handle.object_id,
                left.handle.generation,
            )
                .cmp(&(
                    &right.handle.owner,
                    &right.handle.object_id,
                    right.handle.generation,
                ))
        });
        self.stats.content_hashes += encoded.len() as u64;
        if let Some(fault) = fault {
            match fault {
                Some(V4Fault::Write) => return Err(format_error("injected write failure".into())),
                Some(V4Fault::Checksum) => {
                    return Err(mismatch("injected checksum failure".into()));
                }
                Some(V4Fault::DiskFull) => {
                    return Err(io_error(
                        &self.root,
                        std::io::Error::from(std::io::ErrorKind::StorageFull),
                    ));
                }
                Some(V4Fault::WorkerPanic) | None => {}
            }
        }
        self.prepare_directories(&encoded)?;
        self.write_encoded_files(&encoded).await?;
        self.install_encoded_files(&encoded)?;
        Ok(encoded)
    }

    pub(in crate::state) async fn write_opaque(
        &mut self,
        epoch: u64,
        objects: Vec<OpaqueObject>,
    ) -> Result<Vec<EncodedObject>> {
        let mut encoded = Vec::new();
        for chunk in objects.chunks(MAX_WORKERS) {
            let root = self.root.clone();
            let mut handles = chunk
                .iter()
                .map(|object| {
                    let root = root.clone();
                    let object = object.clone();
                    tokio::task::spawn_blocking(move || encode_opaque(&object, &root, epoch))
                })
                .collect::<Vec<_>>();
            let mut settled = Vec::new();
            for handle in &mut handles {
                settled.push(handle.await);
            }
            for result in settled {
                encoded.push(result.map_err(|error| {
                    format_error(format!("checkpoint-v4 opaque worker failed: {error}"))
                })?);
            }
        }
        encoded.sort_by(|left, right| {
            (
                &left.handle.owner,
                &left.handle.object_id,
                left.handle.generation,
            )
                .cmp(&(
                    &right.handle.owner,
                    &right.handle.object_id,
                    right.handle.generation,
                ))
        });
        self.stats.opaque_blocks += encoded.len() as u64;
        self.prepare_directories(&encoded)?;
        self.write_encoded_files(&encoded).await?;
        self.install_encoded_files(&encoded)?;
        Ok(encoded)
    }

    pub(in crate::state) fn read_opaque(&mut self, handle: &V4Handle) -> Result<Vec<u8>> {
        let path = self.absolute(handle);
        let bytes = std::fs::read(&path).map_err(|source| io_error(&path, source))?;
        self.stats.recovery_hashes += 1;
        decode_opaque(handle, &bytes)
    }

    pub(in crate::state) fn publish_manifest(&mut self, manifest: &V4Manifest) -> Result<PathBuf> {
        manifest.validate_structure()?;
        let bytes = manifest.canonical_bytes()?;
        let directory = self.root.join("manifests");
        std::fs::create_dir_all(&directory).map_err(|source| io_error(&directory, source))?;
        let temporary = directory.join(format!("epoch-{}.tmp", manifest.epoch));
        let committed = directory.join(format!("epoch-{}.json", manifest.epoch));
        write_new_file(&temporary, &bytes)?;
        sync_file(&temporary)?;
        self.stats.manifest_file_syncs += 1;
        install_file(&temporary, &committed, &bytes)?;
        sync_directory(&directory)?;
        self.stats.manifest_root_syncs += 1;
        Ok(committed)
    }

    pub(in crate::state) fn read_manifest(&self, path: &Path) -> Result<V4Manifest> {
        let bytes = std::fs::read(path).map_err(|source| io_error(path, source))?;
        V4Manifest::from_bytes(&bytes)
    }

    pub(in crate::state) fn read_object(&mut self, handle: &V4Handle) -> Result<ObjectPayload> {
        let path = self.absolute(handle);
        let bytes = std::fs::read(&path).map_err(|source| io_error(&path, source))?;
        self.stats.recovery_hashes += 1;
        Ok(ObjectPayload::Bytes(decode_object(handle, &bytes)?.body))
    }

    fn absolute(&self, handle: &V4Handle) -> PathBuf {
        self.root
            .parent()
            .unwrap_or(&self.root)
            .join(&handle.relative_path)
    }

    fn prepare_directories(&mut self, objects: &[EncodedObject]) -> Result<()> {
        let mut parents = BTreeSet::new();
        for object in objects {
            let committed = self.absolute(&object.handle);
            let staging = staging_path(&committed);
            for directory in [committed.parent(), staging.parent()] {
                if let Some(directory) = directory {
                    parents.insert(directory.to_path_buf());
                }
            }
        }
        for directory in parents {
            std::fs::create_dir_all(&directory).map_err(|source| io_error(&directory, source))?;
        }
        if !objects.is_empty() {
            sync_directory(&self.root)?;
            self.stats.directory_creation_syncs += 1;
        }
        Ok(())
    }

    async fn write_encoded_files(&mut self, objects: &[EncodedObject]) -> Result<()> {
        for chunk in objects.chunks(MAX_WORKERS) {
            let mut handles = chunk
                .iter()
                .map(|object| {
                    let path = staging_path(&self.absolute(&object.handle));
                    let bytes = object.bytes.clone();
                    tokio::task::spawn_blocking(move || write_and_sync(&path, &bytes))
                })
                .collect::<Vec<_>>();
            let mut settled = Vec::new();
            for handle in &mut handles {
                settled.push(handle.await);
            }
            for result in settled {
                result.map_err(|error| {
                    format_error(format!("checkpoint-v4 write worker failed: {error}"))
                })?;
            }
        }
        self.stats.file_syncs += objects.len() as u64;
        let mut directories = BTreeSet::new();
        for object in objects {
            if let Some(parent) = staging_path(&self.absolute(&object.handle)).parent() {
                directories.insert(parent.to_path_buf());
            }
        }
        for directory in directories {
            sync_directory(&directory)?;
            self.stats.staging_directory_syncs += 1;
        }
        Ok(())
    }

    fn install_encoded_files(&mut self, objects: &[EncodedObject]) -> Result<()> {
        let mut directories = BTreeSet::new();
        for object in objects {
            let committed = self.absolute(&object.handle);
            let staging = staging_path(&committed);
            install_file(&staging, &committed, &object.bytes)?;
            directories.insert(
                committed
                    .parent()
                    .expect("committed object has a parent")
                    .to_path_buf(),
            );
        }
        for directory in directories {
            sync_directory(&directory)?;
            self.stats.committed_directory_syncs += 1;
        }
        Ok(())
    }
}

fn staging_path(committed: &Path) -> PathBuf {
    committed.with_extension("staging")
}

fn write_and_sync(path: &Path, bytes: &[u8]) -> Result<()> {
    write_new_file(path, bytes)?;
    sync_file(path)
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| io_error(path, source))?;
    file.write_all(bytes)
        .map_err(|source| io_error(path, source))
}

fn sync_file(path: &Path) -> Result<()> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|source| io_error(path, source))?;
    file.sync_all().map_err(|source| io_error(path, source))
}

fn sync_directory(path: &Path) -> Result<()> {
    std::fs::File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| io_error(path, source))
}

fn install_file(staging: &Path, committed: &Path, bytes: &[u8]) -> Result<()> {
    match std::fs::hard_link(staging, committed) {
        Ok(()) => {
            std::fs::remove_file(staging).map_err(|source| io_error(staging, source))?;
            Ok(())
        }
        Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing =
                std::fs::read(committed).map_err(|source| io_error(committed, source))?;
            if existing == bytes {
                std::fs::remove_file(staging).map_err(|source| io_error(staging, source))?;
                Ok(())
            } else {
                Err(CalcFlowError::Conflict {
                    resource: "checkpoint-v4 object".into(),
                    key: committed.to_string_lossy().into_owned(),
                })
            }
        }
        Err(source) => Err(io_error(committed, source)),
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
    use std::path::Path;

    use super::{NewObject, ObjectPayload, OpaqueObject, V4Store};
    use crate::state::v4::ObjectKind;
    use sha2::{Digest, Sha256};

    fn object(id: &str) -> NewObject {
        NewObject {
            kind: ObjectKind::Page,
            owner: "rolling".into(),
            object_id: id.into(),
            generation: 1,
            payload: ObjectPayload::Bytes(id.as_bytes().to_vec()),
        }
    }

    #[tokio::test]
    async fn writes_objects_once_and_recovers_without_readback_hash() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("lineage");
        let mut store = V4Store::new(&root);
        let objects = (0..9)
            .map(|index| object(&index.to_string()))
            .collect::<Vec<_>>();
        let encoded = store.write_objects(7, objects).await.unwrap();
        assert_eq!(encoded.len(), 9);
        assert_eq!(store.stats().content_hashes, 9);
        assert_eq!(store.stats().file_syncs, 9);
        assert_eq!(store.stats().read_back_hashes, 0);
        assert!(store.stats().staging_directory_syncs >= 1);
        assert!(store.stats().committed_directory_syncs >= 1);
        let first = store.read_object(&encoded[0].handle).unwrap();
        let ObjectPayload::Bytes(bytes) = first else {
            panic!("bytes expected")
        };
        assert_eq!(bytes, b"0");
        assert_eq!(store.stats().recovery_hashes, 1);
    }

    #[tokio::test]
    async fn opaque_blocks_reuse_capture_digest_without_another_content_hash() {
        let temporary = tempfile::tempdir().unwrap();
        let mut store = V4Store::new(temporary.path().join("lineage"));
        let bytes = b"legacy-state-segment".to_vec();
        let object = OpaqueObject {
            owner: "sql".into(),
            object_id: "base".into(),
            generation: 4,
            sha256: hex::encode(Sha256::digest(&bytes)),
            bytes: bytes.clone(),
        };
        let encoded = store.write_opaque(7, vec![object]).await.unwrap().remove(0);
        assert_eq!(store.stats().content_hashes, 0);
        assert_eq!(store.stats().opaque_blocks, 1);
        assert_eq!(store.read_opaque(&encoded.handle).unwrap(), bytes);
    }

    #[tokio::test]
    async fn conflicting_existing_object_is_never_overwritten() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("lineage");
        let mut store = V4Store::new(&root);
        let first = store
            .write_objects(7, vec![object("same")])
            .await
            .unwrap()
            .remove(0);
        let path = root.parent().unwrap().join(&first.handle.relative_path);
        std::fs::write(&path, b"different").unwrap();
        assert!(store.write_objects(7, vec![object("same")]).await.is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"different");
    }

    #[tokio::test]
    async fn worker_panic_settles_admitted_workers_without_manifest() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("lineage");
        let mut store = V4Store::new(&root);
        assert!(
            store
                .write_objects_with_fault(
                    7,
                    vec![object("panic")],
                    Some(super::V4Fault::WorkerPanic)
                )
                .await
                .is_err()
        );
        assert!(!Path::new(&root).join("manifests").exists());
    }
}
