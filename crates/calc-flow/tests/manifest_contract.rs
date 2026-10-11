mod static_input_tests {
    use std::collections::{BTreeMap, BTreeSet};

    use chrono::{TimeZone, Utc};

    use calc_flow::{CalcFlowError, Epoch};
    use calc_flow::{
        CheckpointManifest, CheckpointManifestFields, ManifestExpectation, RecoveryStatus,
        STATIC_INPUT_DIGEST_VERSION, StaticInputDigest,
    };

    fn digest(sha256: &str) -> StaticInputDigest {
        StaticInputDigest {
            digest_version: STATIC_INPUT_DIGEST_VERSION.into(),
            sha256: sha256.into(),
        }
    }

    static EMPTY_ID_SET: BTreeSet<String> = BTreeSet::new();

    const FINGERPRINT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CONFIG: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn fields(static_inputs: BTreeMap<String, StaticInputDigest>) -> CheckpointManifestFields {
        CheckpointManifestFields {
            pipeline_name: "orders".into(),
            pipeline_fingerprint: FINGERPRINT.into(),
            runtime_config_hash: CONFIG.into(),
            epoch: Epoch::INITIAL,
            created_at: Utc.with_ymd_and_hms(2026, 8, 8, 7, 0, 0).unwrap(),
            recovery_status: RecoveryStatus::Final,
            sources: BTreeMap::new(),
            operators: BTreeMap::new(),
            sinks: BTreeMap::new(),
            static_inputs,
        }
    }

    fn expectation<'a>(
        fingerprint: &'a str,
        config: &'a str,
        static_inputs: &'a BTreeMap<String, StaticInputDigest>,
        source_ids: &'a BTreeSet<String>,
    ) -> ManifestExpectation<'a> {
        ManifestExpectation {
            pipeline_name: "orders",
            pipeline_fingerprint: fingerprint,
            runtime_config_hash: config,
            epoch: Epoch::INITIAL,
            source_ids,
            operator_ids: &EMPTY_ID_SET,
            sink_ids: &EMPTY_ID_SET,
            static_inputs,
        }
    }

    #[test]
    fn empty_static_evidence_is_omitted_and_bytes_stay_compatible() {
        let manifest = CheckpointManifest::new(fields(BTreeMap::new())).unwrap();
        let bytes = manifest.canonical_bytes().unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(!text.contains("static_inputs"));
        assert!(CheckpointManifest::from_bytes(text.as_bytes()).is_ok());
    }

    #[test]
    fn static_evidence_serializes_and_roundtrips() {
        let statics = BTreeMap::from([("weights".to_string(), digest(&"c".repeat(64)))]);
        let manifest = CheckpointManifest::new(fields(statics.clone())).unwrap();
        let bytes = manifest.canonical_bytes().unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("\"static_inputs\":{\"weights\":{"));
        let restored = CheckpointManifest::from_bytes(text.as_bytes()).unwrap();
        assert_eq!(restored.static_inputs(), &statics);
        assert!(
            restored
                .validate(&expectation(
                    FINGERPRINT,
                    CONFIG,
                    &statics,
                    &BTreeSet::new()
                ))
                .is_ok()
        );
    }

    #[test]
    fn non_empty_static_evidence_joins_the_state_checksum() {
        let empty = CheckpointManifest::new(fields(BTreeMap::new())).unwrap();
        let statics = BTreeMap::from([("weights".to_string(), digest(&"c".repeat(64)))]);
        let with_statics = CheckpointManifest::new(fields(statics)).unwrap();
        assert_ne!(
            empty.state_checksum(),
            with_statics.state_checksum(),
            "the digest evidence must change the state checksum"
        );
        let changed = BTreeMap::from([("weights".to_string(), digest(&"d".repeat(64)))]);
        let with_changed = CheckpointManifest::new(fields(changed)).unwrap();
        assert_ne!(with_statics.state_checksum(), with_changed.state_checksum());
    }

    #[test]
    fn recovery_digest_mismatch_uses_the_frozen_message() {
        let stored = digest(&"c".repeat(64));
        let manifest =
            CheckpointManifest::new(fields(BTreeMap::from([("weights".into(), stored.clone())])))
                .unwrap();
        let prepared = digest(&"d".repeat(64));
        let error = manifest
            .validate(&expectation(
                FINGERPRINT,
                CONFIG,
                &BTreeMap::from([("weights".into(), prepared.clone())]),
                &BTreeSet::new(),
            ))
            .unwrap_err();
        let CalcFlowError::CheckpointMismatch { message } = error else {
            panic!("expected a checkpoint mismatch");
        };
        assert_eq!(
            message,
            format!(
                "static_inputs.weights.digest: checkpoint digest {} does not match prepared digest {} for {}",
                stored.sha256, prepared.sha256, STATIC_INPUT_DIGEST_VERSION
            )
        );
    }

    #[test]
    fn recovery_reports_missing_and_extra_static_names() {
        let manifest = CheckpointManifest::new(fields(BTreeMap::from([(
            "weights".into(),
            digest(&"c".repeat(64)),
        )])))
        .unwrap();
        let error = manifest
            .validate(&expectation(
                FINGERPRINT,
                CONFIG,
                &BTreeMap::new(),
                &BTreeSet::new(),
            ))
            .unwrap_err();
        assert!(error.to_string().contains("static_inputs.weights"));

        let empty = CheckpointManifest::new(fields(BTreeMap::new())).unwrap();
        let error = empty
            .validate(&expectation(
                FINGERPRINT,
                CONFIG,
                &BTreeMap::from([("weights".into(), digest(&"c".repeat(64)))]),
                &BTreeSet::new(),
            ))
            .unwrap_err();
        assert!(error.to_string().contains("static_inputs.weights"));
    }

    #[test]
    fn manifest_rejects_unsupported_static_digest_versions() {
        let stored = StaticInputDigest {
            digest_version: "calc_flow.static_input.digest.v0".into(),
            sha256: "c".repeat(64),
        };
        let error = CheckpointManifest::new(fields(BTreeMap::from([("weights".into(), stored)])))
            .unwrap_err();
        assert!(error.to_string().contains("static_inputs.weights"));
    }
}

mod source_history_tests {
    use calc_flow::{
        CheckpointManifest, CheckpointManifestFields, Epoch, OperatorManifestEntry, RecoveryStatus,
        SourceManifestEntry, StateHandle,
    };
    use chrono::TimeZone;
    use chrono::Utc;
    use serde_json::{Value, json};
    use std::collections::BTreeMap;

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

    mod legacy_source_tests {
        use calc_flow::{CheckpointManifest, Epoch, ManifestExpectation};
        use serde_json::{Value, json};
        use std::collections::{BTreeMap, BTreeSet};

        const LEGACY: &str = r#"{"created_at":"2026-10-10T00:00:00Z","epoch":1,"format_version":3,"operators":{},"pipeline_fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","pipeline_name":"legacy-history","recovery_status":"final","runtime_config_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","sinks":{},"sources":{"prices":{"cursor":null,"ended":false,"identity_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","sequence":0,"watermark_policy":{"idle":false,"kind":"disabled"}}},"state_checksum":"26c10529ff0dea3f3d5a214434ce4eec4b34baaad6705a5d2f0561e9c54abf00"}"#;
        const CHECKSUM: &str = "26c10529ff0dea3f3d5a214434ce4eec4b34baaad6705a5d2f0561e9c54abf00";

        pub(super) fn assert_legacy_source_history_roundtrip() {
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
            assert!(
                CheckpointManifest::from_bytes(&serde_json::to_vec(&changed).unwrap()).is_err()
            );
            let mut invalid_history = original;
            invalid_history["sources"]["prices"]["history"] = json!({});
            assert!(
                CheckpointManifest::from_bytes(&serde_json::to_vec(&invalid_history).unwrap())
                    .is_err()
            );
        }
    }
}
