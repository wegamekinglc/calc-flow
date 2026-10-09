use super::*;

const LEGACY: &str = r#"{"created_at":"2026-10-10T00:00:00Z","epoch":1,"format_version":3,"operators":{},"pipeline_fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","pipeline_name":"legacy-history","recovery_status":"final","runtime_config_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","sinks":{},"sources":{"prices":{"cursor":null,"ended":false,"identity_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","sequence":0,"watermark_policy":{"idle":false,"kind":"disabled"}}},"state_checksum":"26c10529ff0dea3f3d5a214434ce4eec4b34baaad6705a5d2f0561e9c54abf00"}"#;
const CHECKSUM: &str = "26c10529ff0dea3f3d5a214434ce4eec4b34baaad6705a5d2f0561e9c54abf00";

#[test]
fn test_legacy_source_history_omission_preserves_checksum_and_roundtrips() {
    let restored = CheckpointManifest::from_bytes(LEGACY.as_bytes())
        .expect("an authentic pre-history source entry remains readable");
    assert!(restored.sources()["prices"].history.is_none());
    assert_eq!(restored.state_checksum(), CHECKSUM);
    assert_eq!(restored.canonical_bytes().unwrap(), LEGACY.as_bytes());
    let cloned = restored.clone();
    assert_eq!(cloned.recompute_state_checksum().unwrap(), CHECKSUM);
    assert_eq!(
        serde_json::to_value(&cloned).unwrap(),
        serde_json::from_str::<Value>(LEGACY).unwrap()
    );
    let serialized = serde_json::to_vec(&cloned).unwrap();
    assert_eq!(CheckpointManifest::from_bytes(&serialized).unwrap(), cloned);
    let sources = BTreeSet::from(["prices".to_owned()]);
    let empty = BTreeSet::new();
    cloned
        .validate(&ManifestExpectation {
            pipeline_name: "legacy-history",
            pipeline_fingerprint: &"a".repeat(64),
            runtime_config_hash: &"b".repeat(64),
            epoch: Epoch::INITIAL,
            source_ids: &sources,
            operator_ids: &empty,
            sink_ids: &empty,
            static_inputs: &BTreeMap::new(),
        })
        .unwrap();
    let original: Value = serde_json::from_str(LEGACY).unwrap();
    let mut changed = original.clone();
    changed["sources"]["prices"]["sequence"] = json!(1);
    assert!(CheckpointManifest::from_bytes(&serde_json::to_vec(&changed).unwrap()).is_err());
    let mut invalid_history = original;
    invalid_history["sources"]["prices"]["history"] = json!({});
    assert!(
        CheckpointManifest::from_bytes(&serde_json::to_vec(&invalid_history).unwrap()).is_err()
    );
}
