use std::{collections::BTreeMap, path::Path, sync::Arc};

use calc_flow::{
    CHECKPOINT_FORMAT_VERSION, CalcFlowError, Checkpoint, CheckpointStore, FileCheckpointStore,
};
use chrono::{TimeZone, Utc};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn checkpoint(name: &str, sequence: u64) -> Checkpoint {
    Checkpoint::new(
        name,
        "fingerprint",
        Some(json!({"offset": sequence})),
        sequence,
        BTreeMap::from([("sum".into(), json!({"total": sequence}))]),
        Utc.with_ymd_and_hms(2026, 7, 14, 1, 2, 3).unwrap(),
    )
    .unwrap()
}

fn hashed_path(directory: &Path, key: &str) -> std::path::PathBuf {
    directory.join(format!(
        "{}.json",
        hex::encode(Sha256::digest(key.as_bytes()))
    ))
}

fn json_files(directory: &Path) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect()
}

#[test]
fn checkpoint_constructor_and_deserializer_enforce_v2_invariants() {
    assert_eq!(CHECKPOINT_FORMAT_VERSION, 2);
    assert!(matches!(
        Checkpoint::new(
            "",
            "fingerprint",
            None,
            0,
            BTreeMap::new(),
            Utc::now()
        ),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "pipeline_name"
    ));
    assert!(matches!(
        Checkpoint::new("pipeline", "", None, 0, BTreeMap::new(), Utc::now()),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "pipeline_fingerprint"
    ));
    assert!(matches!(
        Checkpoint::new(
            "pipeline",
            "fingerprint",
            None,
            0,
            BTreeMap::from([(String::new(), Value::Null)]),
            Utc::now()
        ),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "state"
    ));
    assert_eq!(
        Checkpoint::new(
            "pipeline",
            "fingerprint",
            Some(Value::Null),
            0,
            BTreeMap::new(),
            Utc::now()
        )
        .unwrap()
        .source_cursor,
        None
    );

    let valid = serde_json::to_value(checkpoint("pipeline", 1)).unwrap();
    let mut v1 = valid.clone();
    v1["format_version"] = json!(1);
    assert!(matches!(
        serde_json::from_value::<Checkpoint>(v1),
        Err(error) if error.to_string().contains("unsupported")
    ));
    let mut unknown = valid;
    unknown["unexpected"] = json!(true);
    assert!(serde_json::from_value::<Checkpoint>(unknown).is_err());
}

#[tokio::test]
async fn checkpoint_round_trip_is_v2_canonical_and_path_safe() {
    let directory = tempfile::tempdir().unwrap();
    let store = FileCheckpointStore::new(directory.path()).await.unwrap();
    let value = checkpoint("../../orders\\CON", 8);

    store.save(&value).await.unwrap();

    assert_eq!(store.load("../../orders\\CON").await.unwrap(), Some(value));
    let files = json_files(directory.path());
    assert_eq!(
        files,
        vec![hashed_path(directory.path(), "../../orders\\CON")]
    );
    let bytes = std::fs::read(&files[0]).unwrap();
    assert!(bytes.ends_with(b"\n"));
    assert_eq!(
        bytes,
        serde_json::to_string_pretty(&serde_json::from_slice::<Value>(&bytes).unwrap())
            .unwrap()
            .into_bytes()
            .into_iter()
            .chain([b'\n'])
            .collect::<Vec<_>>()
    );
    assert!(std::fs::read_dir(directory.path()).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")
    }));
}

#[tokio::test]
async fn checkpoint_save_replaces_and_delete_is_idempotent() {
    let directory = tempfile::tempdir().unwrap();
    let store = FileCheckpointStore::new(directory.path()).await.unwrap();
    store.save(&checkpoint("orders", 1)).await.unwrap();
    store.save(&checkpoint("orders", 2)).await.unwrap();
    assert_eq!(store.load("orders").await.unwrap().unwrap().sequence, 2);

    store.delete("orders").await.unwrap();
    store.delete("orders").await.unwrap();
    assert_eq!(store.load("orders").await.unwrap(), None);
}

#[tokio::test]
async fn checkpoint_load_rejects_corruption_shape_version_size_and_key_mismatch() {
    let directory = tempfile::tempdir().unwrap();
    let store = FileCheckpointStore::new(directory.path()).await.unwrap();
    let path = hashed_path(directory.path(), "orders");

    for invalid in [b"{".as_slice(), b"[]", b"null"] {
        std::fs::write(&path, invalid).unwrap();
        assert!(matches!(
            store.load("orders").await,
            Err(CalcFlowError::Format { .. })
        ));
    }

    let mut v1 = serde_json::to_value(checkpoint("orders", 1)).unwrap();
    v1["format_version"] = json!(1);
    std::fs::write(&path, serde_json::to_vec(&v1).unwrap()).unwrap();
    assert!(matches!(
        store.load("orders").await,
        Err(CalcFlowError::UnsupportedVersion {
            expected: 2,
            found: 1
        })
    ));

    std::fs::write(&path, serde_json::to_vec(&checkpoint("other", 1)).unwrap()).unwrap();
    assert!(matches!(
        store.load("orders").await,
        Err(CalcFlowError::Format { .. })
    ));

    std::fs::write(
        &path,
        vec![b' '; calc_flow::MAX_CHECKPOINT_DOCUMENT_BYTES + 1],
    )
    .unwrap();
    assert!(matches!(
        store.load("orders").await,
        Err(CalcFlowError::Format { .. })
    ));
}

#[cfg(unix)]
#[tokio::test]
async fn checkpoint_store_rejects_outside_symlink_and_save_replaces_entry() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let outside = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        outside.path(),
        serde_json::to_vec(&checkpoint("orders", 1)).unwrap(),
    )
    .unwrap();
    let path = hashed_path(directory.path(), "orders");
    symlink(outside.path(), &path).unwrap();
    let store = FileCheckpointStore::new(directory.path()).await.unwrap();

    assert!(matches!(
        store.load("orders").await,
        Err(CalcFlowError::Format { .. })
    ));
    store.save(&checkpoint("orders", 2)).await.unwrap();
    assert!(
        !std::fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(store.load("orders").await.unwrap().unwrap().sequence, 2);
}

#[tokio::test]
async fn concurrent_checkpoint_saves_never_expose_partial_json_or_temp_files() {
    let directory = tempfile::tempdir().unwrap();
    let store = Arc::new(FileCheckpointStore::new(directory.path()).await.unwrap());
    let tasks = (0..32).map(|sequence| {
        let store = Arc::clone(&store);
        tokio::spawn(async move { store.save(&checkpoint("orders", sequence)).await.unwrap() })
    });
    futures::future::join_all(tasks)
        .await
        .into_iter()
        .for_each(|result| result.unwrap());

    let restored = store.load("orders").await.unwrap().unwrap();
    assert!(restored.sequence < 32);
    assert!(
        serde_json::from_slice::<Value>(
            &std::fs::read(hashed_path(directory.path(), "orders")).unwrap()
        )
        .is_ok()
    );
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}
