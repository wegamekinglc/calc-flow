use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use calc_flow::{
    CalcFlowError, CheckpointManifest, CheckpointManifestFields, CursorManifestEntry, Epoch,
    MANIFEST_FORMAT_VERSION, MAX_MANIFEST_DOCUMENT_BYTES, ManifestExpectation,
    ManifestIngressState, OperatorIngressManifestEntry, OperatorManifestEntry, RecoveryStatus,
    RetentionClass, SinkDeliveryManifest, SinkManifestEntry, SourceManifestEntry,
    SourceWatermarkManifestState, StateBackend, StateHandle, StateLineageBackend, StateLineageKey,
    canonical_json,
};
use chrono::{TimeZone, Utc};
use serde_json::{Value, json};

const SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

struct MockStateBackend;

struct MockLineageBackend;

#[async_trait]
impl StateBackend for MockStateBackend {
    async fn open_lineage(
        &self,
        _key: &StateLineageKey,
    ) -> calc_flow::Result<Box<dyn StateLineageBackend>> {
        Ok(Box::new(MockLineageBackend))
    }
}

#[async_trait]
impl StateLineageBackend for MockLineageBackend {
    fn identity_hash(&self) -> &str {
        SHA256
    }

    async fn stage_segment(&self, _handle: &StateHandle, _bytes: &[u8]) -> calc_flow::Result<()> {
        Ok(())
    }

    async fn validate_segment(&self, _handle: &StateHandle) -> calc_flow::Result<()> {
        Ok(())
    }

    async fn publish_segment(&self, _handle: &StateHandle) -> calc_flow::Result<()> {
        Ok(())
    }

    async fn load_segment(&self, _handle: &StateHandle) -> calc_flow::Result<Vec<u8>> {
        Ok(Vec::new())
    }

    async fn collect_orphans(&self, _retained: &[StateHandle]) -> calc_flow::Result<usize> {
        Ok(0)
    }
}

fn handle() -> StateHandle {
    StateHandle::new(
        "window",
        Epoch::INITIAL,
        "delta-0001",
        "committed/pipeline/window/1-delta.arrow",
        42,
        SHA256,
    )
    .unwrap()
}

fn manifest_fields() -> CheckpointManifestFields {
    CheckpointManifestFields {
        pipeline_name: "orders".into(),
        pipeline_fingerprint: SHA256.into(),
        runtime_config_hash: SHA256.into(),
        epoch: Epoch::INITIAL,
        created_at: Utc.with_ymd_and_hms(2026, 8, 8, 7, 0, 0).unwrap(),
        recovery_status: RecoveryStatus::Final,
        sources: BTreeMap::from([(
            "source".into(),
            SourceManifestEntry {
                cursor: None,
                identity_hash: SHA256.into(),
                sequence: 7,
                ended: false,
                watermark_policy: SourceWatermarkManifestState::Disabled { idle: false },
            },
        )]),
        operators: BTreeMap::from([(
            "window".into(),
            OperatorManifestEntry {
                progress: BTreeMap::from([(
                    "input".into(),
                    OperatorIngressManifestEntry {
                        state: ManifestIngressState::Active,
                        watermark: None,
                    },
                )]),
                inline_metadata: BTreeMap::from([("layout".into(), json!(1))]),
                segments: vec![handle()],
            },
        )]),
        sinks: BTreeMap::from([(
            "sink".into(),
            SinkManifestEntry {
                delivery: SinkDeliveryManifest::EpochIdempotent {
                    mechanism: "epoch-key".into(),
                    retention: RetentionClass::Bounded,
                },
                pre_commit: None,
                segments: Vec::new(),
            },
        )]),
    }
}

fn expectation<'a>(
    source_ids: &'a BTreeSet<String>,
    operator_ids: &'a BTreeSet<String>,
    sink_ids: &'a BTreeSet<String>,
) -> ManifestExpectation<'a> {
    ManifestExpectation {
        pipeline_name: "orders",
        pipeline_fingerprint: SHA256,
        runtime_config_hash: SHA256,
        epoch: Epoch::INITIAL,
        source_ids,
        operator_ids,
        sink_ids,
    }
}

#[test]
fn state_handle_accepts_one_portable_committed_identity() {
    let handle = handle();

    assert_eq!(handle.operator_id(), "window");
    assert_eq!(handle.epoch(), Epoch::INITIAL);
    assert_eq!(handle.segment_id(), "delta-0001");
    assert_eq!(
        handle.relative_path(),
        "committed/pipeline/window/1-delta.arrow"
    );
    assert_eq!(handle.byte_len(), 42);
    assert_eq!(handle.sha256(), SHA256);
    handle.validate_for("window", Epoch::INITIAL).unwrap();
}

#[test]
fn state_handle_rejects_every_non_portable_identity_and_path() {
    let invalid_ids = ["", "contains space", "contains/slash", "λ"];
    for operator_id in invalid_ids {
        let error = StateHandle::new(
            operator_id,
            Epoch::INITIAL,
            "delta-0001",
            "committed/pipeline/window/1-delta.arrow",
            42,
            SHA256,
        )
        .unwrap_err();
        assert!(
            matches!(error, CalcFlowError::InvalidArgument { field, .. } if field == "operator_id")
        );
    }

    for segment_id in invalid_ids {
        let error = StateHandle::new(
            "window",
            Epoch::INITIAL,
            segment_id,
            "committed/pipeline/window/1-delta.arrow",
            42,
            SHA256,
        )
        .unwrap_err();
        assert!(
            matches!(error, CalcFlowError::InvalidArgument { field, .. } if field == "segment_id")
        );
    }

    let invalid_paths = [
        "",
        "/committed/pipeline/window/1-delta.arrow",
        "C:/committed/pipeline/window/1-delta.arrow",
        "committed\\pipeline\\window\\1-delta.arrow",
        "committed//window/1-delta.arrow",
        "committed/./window/1-delta.arrow",
        "committed/../window/1-delta.arrow",
        "staging/pipeline/window/1-delta.arrow",
        "committed/pipeline/window/\0.arrow",
    ];
    for relative_path in invalid_paths {
        let error = StateHandle::new(
            "window",
            Epoch::INITIAL,
            "delta-0001",
            relative_path,
            42,
            SHA256,
        )
        .unwrap_err();
        assert!(
            matches!(error, CalcFlowError::InvalidArgument { ref field, .. } if field == "relative_path"),
            "path {relative_path:?} produced {error:?}"
        );
    }
}

#[test]
fn state_handle_rejects_noncanonical_checksums_and_wrong_ownership() {
    for checksum in [
        "",
        "0123456789abcdef",
        "0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF",
        "g123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ] {
        let error = StateHandle::new(
            "window",
            Epoch::INITIAL,
            "delta-0001",
            "committed/pipeline/window/1-delta.arrow",
            42,
            checksum,
        )
        .unwrap_err();
        assert!(matches!(error, CalcFlowError::InvalidArgument { field, .. } if field == "sha256"));
    }

    let handle = handle();
    let operator_error = handle.validate_for("other", Epoch::INITIAL).unwrap_err();
    assert!(matches!(
        operator_error,
        CalcFlowError::CheckpointMismatch { .. }
    ));

    let epoch_error = handle
        .validate_for("window", Epoch::INITIAL.next().unwrap())
        .unwrap_err();
    assert!(matches!(
        epoch_error,
        CalcFlowError::CheckpointMismatch { .. }
    ));
}

#[tokio::test]
async fn backend_contract_exposes_operations_only_through_a_lineage_session() {
    let backend: &dyn StateBackend = &MockStateBackend;
    let key = StateLineageKey::new("orders", SHA256).unwrap();
    let lineage = backend.open_lineage(&key).await.unwrap();

    assert_eq!(lineage.identity_hash(), SHA256);
    lineage.stage_segment(&handle(), &[]).await.unwrap();
}

#[test]
fn lineage_key_validates_logical_identity_without_exposing_raw_paths() {
    let key = StateLineageKey::new("orders", SHA256).unwrap();
    assert_eq!(key.pipeline_name(), "orders");
    assert_eq!(key.pipeline_fingerprint(), SHA256);

    for pipeline_name in ["", "contains/slash", "contains\\separator", "λ"] {
        assert!(matches!(
            StateLineageKey::new(pipeline_name, SHA256),
            Err(CalcFlowError::InvalidArgument { field, .. }) if field == "pipeline_name"
        ));
    }
    for fingerprint in [
        "",
        "abc",
        "A123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ] {
        assert!(matches!(
            StateLineageKey::new("orders", fingerprint),
            Err(CalcFlowError::InvalidArgument { field, .. }) if field == "pipeline_fingerprint"
        ));
    }
}

#[test]
fn manifest_canonical_bytes_ignore_mapping_insertion_order() {
    let fields = manifest_fields();
    let first = CheckpointManifest::new(fields.clone()).unwrap();

    let mut reversed = fields;
    reversed.sources = reversed.sources.into_iter().rev().collect();
    reversed.operators = reversed.operators.into_iter().rev().collect();
    reversed.sinks = reversed.sinks.into_iter().rev().collect();
    let second = CheckpointManifest::new(reversed).unwrap();

    assert_eq!(first.format_version(), MANIFEST_FORMAT_VERSION);
    assert_eq!(
        first.canonical_bytes().unwrap(),
        second.canonical_bytes().unwrap()
    );
    assert_eq!(first.state_checksum(), second.state_checksum());
    assert!(first.canonical_bytes().unwrap().len() < MAX_MANIFEST_DOCUMENT_BYTES);
}

#[test]
fn manifest_strict_loader_rejects_unknown_duplicate_missing_and_bounded_json() {
    let manifest = CheckpointManifest::new(manifest_fields()).unwrap();
    let canonical = String::from_utf8(manifest.canonical_bytes().unwrap()).unwrap();

    let unknown = canonical.replacen('{', r#"{"unknown":true,"#, 1);
    assert!(matches!(
        CheckpointManifest::from_bytes(unknown.as_bytes()),
        Err(CalcFlowError::Format { .. })
    ));

    let duplicate = canonical.replacen(
        r#""format_version":3"#,
        r#""format_version":3,"format_version":3"#,
        1,
    );
    assert!(matches!(
        CheckpointManifest::from_bytes(duplicate.as_bytes()),
        Err(CalcFlowError::Format { .. })
    ));

    for path in [
        &["sources", "source", "cursor"][..],
        &["operators", "window", "progress", "input", "watermark"][..],
        &["sinks", "sink", "pre_commit"][..],
    ] {
        let mut value: Value = serde_json::from_str(&canonical).unwrap();
        let (field, parents) = path.split_last().unwrap();
        let mut parent = &mut value;
        for component in parents {
            parent = parent.get_mut(component).unwrap();
        }
        parent.as_object_mut().unwrap().remove(*field);
        let document = canonical_json(&value).unwrap();
        assert!(
            matches!(
                CheckpointManifest::from_bytes(document.as_bytes()),
                Err(CalcFlowError::Format { .. })
            ),
            "missing required nullable field at {path:?} was accepted"
        );
    }

    let mut wrong_version: Value = serde_json::from_str(&canonical).unwrap();
    wrong_version["format_version"] = json!(2);
    assert!(matches!(
        CheckpointManifest::from_bytes(canonical_json(&wrong_version).unwrap().as_bytes()),
        Err(CalcFlowError::UnsupportedVersion {
            expected: 3,
            found: 2
        })
    ));

    let oversized = vec![b' '; MAX_MANIFEST_DOCUMENT_BYTES + 1];
    assert!(matches!(
        CheckpointManifest::from_bytes(&oversized),
        Err(CalcFlowError::Format { .. })
    ));

    let mut over_depth: Value = serde_json::from_str(&canonical).unwrap();
    let mut nested = json!(null);
    for _ in 0..=calc_flow::MAX_JSON_DEPTH {
        nested = json!({"nested": nested});
    }
    over_depth["sources"]["source"]["cursor"] = json!({
        "order": "opaque",
        "payload": {"nested": nested}
    });
    assert!(matches!(
        CheckpointManifest::from_bytes(&serde_json::to_vec(&over_depth).unwrap()),
        Err(CalcFlowError::Format { .. })
    ));
}

#[test]
fn manifest_validates_expected_plan_and_handle_ownership_before_load() {
    let manifest = CheckpointManifest::new(manifest_fields()).unwrap();
    let sources = BTreeSet::from(["source".into()]);
    let operators = BTreeSet::from(["window".into()]);
    let sinks = BTreeSet::from(["sink".into()]);
    manifest
        .validate(&expectation(&sources, &operators, &sinks))
        .unwrap();

    let wrong_sources = BTreeSet::from(["other".into()]);
    assert!(matches!(
        manifest.validate(&expectation(&wrong_sources, &operators, &sinks)),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));

    let mut wrong_operator = manifest_fields();
    wrong_operator.operators.get_mut("window").unwrap().segments[0] = StateHandle::new(
        "other",
        Epoch::INITIAL,
        "delta-0001",
        "committed/pipeline/window/1-delta.arrow",
        42,
        SHA256,
    )
    .unwrap();
    assert!(matches!(
        CheckpointManifest::new(wrong_operator),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));

    let mut wrong_epoch = manifest_fields();
    wrong_epoch.operators.get_mut("window").unwrap().segments[0] = StateHandle::new(
        "window",
        Epoch::INITIAL.next().unwrap(),
        "delta-0001",
        "committed/pipeline/window/2-delta.arrow",
        42,
        SHA256,
    )
    .unwrap();
    assert!(matches!(
        CheckpointManifest::new(wrong_epoch),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
}

#[test]
fn manifest_runtime_configuration_mismatch_is_diagnostic_only() {
    let manifest = CheckpointManifest::new(manifest_fields()).unwrap();
    let sources = BTreeSet::from(["source".into()]);
    let operators = BTreeSet::from(["window".into()]);
    let sinks = BTreeSet::from(["sink".into()]);
    let changed_runtime_hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    let expected = ManifestExpectation {
        pipeline_name: "orders",
        pipeline_fingerprint: SHA256,
        runtime_config_hash: changed_runtime_hash,
        epoch: Epoch::INITIAL,
        source_ids: &sources,
        operator_ids: &operators,
        sink_ids: &sinks,
    };

    manifest.validate(&expected).unwrap();
    assert_ne!(manifest.runtime_config_hash(), changed_runtime_hash);
}

#[test]
fn manifest_accepts_older_inventory_handles_and_rejects_future_handles() {
    let mut later_manifest = manifest_fields();
    later_manifest.epoch = Epoch::INITIAL.next().unwrap();
    let manifest = CheckpointManifest::new(later_manifest).unwrap();
    assert_eq!(
        manifest.operators()["window"].segments[0].epoch(),
        Epoch::INITIAL
    );

    let mut future_handle = manifest_fields();
    future_handle.operators.get_mut("window").unwrap().segments[0] = StateHandle::new(
        "window",
        Epoch::INITIAL.next().unwrap(),
        "delta-0002",
        "committed/pipeline/window/2-delta.arrow",
        42,
        SHA256,
    )
    .unwrap();
    assert!(matches!(
        CheckpointManifest::new(future_handle),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
}

#[test]
fn manifest_checksum_mismatch_fails_closed() {
    let manifest = CheckpointManifest::new(manifest_fields()).unwrap();
    let mut value: Value = serde_json::from_slice(&manifest.canonical_bytes().unwrap()).unwrap();
    value["state_checksum"] =
        json!("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    let document = canonical_json(&value).unwrap();

    assert!(matches!(
        CheckpointManifest::from_bytes(document.as_bytes()),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
}

#[test]
fn manifest_keeps_large_state_bytes_out_of_the_bounded_document() {
    let mut fields = manifest_fields();
    let handle = fields.operators["window"].segments[0].clone();
    fields.operators.get_mut("window").unwrap().segments[0] = StateHandle::new(
        handle.operator_id(),
        handle.epoch(),
        handle.segment_id(),
        handle.relative_path(),
        11 * 1024 * 1024,
        handle.sha256(),
    )
    .unwrap();

    let manifest = CheckpointManifest::new(fields).unwrap();
    assert!(manifest.canonical_bytes().unwrap().len() < MAX_MANIFEST_DOCUMENT_BYTES);
    assert_eq!(
        manifest.operators()["window"].segments[0].byte_len(),
        11 * 1024 * 1024
    );
}

#[test]
fn cursor_payload_and_sink_pre_commit_remain_bounded_strict_json() {
    let mut fields = manifest_fields();
    fields.sources.get_mut("source").unwrap().cursor = Some(CursorManifestEntry {
        order: "opaque".into(),
        payload: BTreeMap::from([("offset".into(), json!(7))]),
    });
    fields.sinks.get_mut("sink").unwrap().pre_commit =
        Some(BTreeMap::from([("transaction".into(), json!("pending"))]));

    let manifest = CheckpointManifest::new(fields).unwrap();
    let restored = CheckpointManifest::from_bytes(&manifest.canonical_bytes().unwrap()).unwrap();
    assert_eq!(restored, manifest);
}
