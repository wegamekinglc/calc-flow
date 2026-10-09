use super::*;
use chrono::TimeZone;

const HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn handle(owner: &str, epoch: Epoch, id: &str, path: &str) -> StateHandle {
    StateHandle::new(owner, epoch, id, path, 4, HASH).unwrap()
}

fn source(history: &Value) -> SourceManifestEntry {
    serde_json::from_value(json!({
        "cursor": null,
        "identity_hash": HASH,
        "sequence": 3,
        "ended": true,
        "watermark_policy": {"kind": "disabled", "idle": false},
        "history": history
    }))
    .expect("the current source manifest must retain immutable history")
}

fn history(handles: &[StateHandle]) -> Value {
    json!({
        "format_version": 1,
        "contract": "file_snapshot_v1",
        "inline_metadata": {"files": ["prices.json"]},
        "segments": handles
    })
}

fn fields(source: SourceManifestEntry) -> CheckpointManifestFields {
    CheckpointManifestFields {
        pipeline_name: "orders".into(),
        pipeline_fingerprint: HASH.into(),
        runtime_config_hash: HASH.into(),
        epoch: Epoch::INITIAL.next().unwrap(),
        created_at: Utc.with_ymd_and_hms(2026, 10, 4, 0, 0, 0).unwrap(),
        recovery_status: RecoveryStatus::Final,
        sources: BTreeMap::from([("prices".into(), source)]),
        operators: BTreeMap::new(),
        sinks: BTreeMap::new(),
        static_inputs: BTreeMap::new(),
    }
}

#[test]
fn source_history_roundtrips_terminal_carried_state_and_checksum() {
    legacy_source_tests::assert_legacy_source_history_roundtrip();
    let stored = handle(
        "prices",
        Epoch::INITIAL,
        "file-0",
        "committed/orders/prices/file-0",
    );
    let with_history = CheckpointManifest::new(fields(source(&history(&[stored])))).unwrap();
    let bytes = with_history.canonical_bytes().unwrap();
    let restored = CheckpointManifest::from_bytes(&bytes).unwrap();
    assert_eq!(restored, with_history);
    let empty = CheckpointManifest::new(fields(source(&Value::Null))).unwrap();
    assert_ne!(empty.state_checksum(), with_history.state_checksum());
    let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
    changed["sources"]["prices"]["history"]["inline_metadata"]["files"] = json!(["other.json"]);
    assert!(CheckpointManifest::from_bytes(&serde_json::to_vec(&changed).unwrap()).is_err());
}

#[test]
fn source_history_rejects_invalid_handle_ownership_epoch_order_and_paths() {
    let first = handle("prices", Epoch::INITIAL, "a", "committed/orders/prices/a");
    let second = handle("prices", Epoch::INITIAL, "b", "committed/orders/prices/b");
    let future = handle(
        "prices",
        Epoch::INITIAL.next().unwrap().next().unwrap(),
        "a",
        "committed/orders/prices/a",
    );
    let foreign = handle("other", Epoch::INITIAL, "a", "committed/orders/other/a");
    let same_path = handle("prices", Epoch::INITIAL, "b", first.relative_path());
    for handles in [
        vec![foreign],
        vec![future],
        vec![second, first.clone()],
        vec![first.clone(), first.clone()],
        vec![first, same_path],
    ] {
        assert!(CheckpointManifest::new(fields(source(&history(&handles)))).is_err());
    }
    let mut unsupported = history(&[]);
    unsupported["format_version"] = json!(2);
    assert!(CheckpointManifest::new(fields(source(&unsupported))).is_err());
}

#[test]
fn source_history_requires_current_explicit_fields() {
    let valid = source(&history(&[]));
    let value = serde_json::to_value(valid).unwrap();
    let mut absent = value.clone();
    absent.as_object_mut().unwrap().remove("history");
    assert!(serde_json::from_value::<SourceManifestEntry>(absent).is_err());
    for field in ["format_version", "contract", "inline_metadata", "segments"] {
        let mut incomplete = value.clone();
        incomplete["history"].as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<SourceManifestEntry>(incomplete).is_err());
    }
    let mut unknown = value;
    unknown["history"]["unknown"] = json!(1);
    assert!(serde_json::from_value::<SourceManifestEntry>(unknown).is_err());
}

#[test]
fn source_history_rejects_component_aliases_and_cross_source_paths() {
    let first = handle(
        "prices",
        Epoch::INITIAL,
        "file-0",
        "committed/orders/shared/file-0",
    );
    let mut value = fields(source(&history(std::slice::from_ref(&first))));
    value.operators.insert(
        "prices".into(),
        OperatorManifestEntry {
            progress: BTreeMap::new(),
            inline_metadata: BTreeMap::new(),
            segments: Vec::new(),
        },
    );
    assert!(CheckpointManifest::new(value).is_err());
    let mut value = fields(source(&history(std::slice::from_ref(&first))));
    let other = handle("other", Epoch::INITIAL, "file-0", first.relative_path());
    value
        .sources
        .insert("other".into(), source(&history(&[other])));
    assert!(CheckpointManifest::new(value).is_err());
    for (field, invalid) in [("contract", json!("")), ("format_version", json!(2))] {
        let mut value = history(&[]);
        value[field] = invalid;
        let error = CheckpointManifest::new(fields(source(&value))).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("sources.prices.history.{field}"))
        );
    }
}
