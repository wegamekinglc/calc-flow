use std::cell::Cell;

use super::*;
use crate::CalcFlowError;

fn control() -> CompactControl {
    CompactControl {
        state_layout: 3,
        state_accounting: 3,
        native_semantics: 1,
        datafusion_version: "54.0.0".into(),
        state_policy: "exact_numeric_v1".into(),
        identity: CompactIdentity {
            query_sha256: "q".repeat(64),
            input_alias: "events".into(),
            runtime_config: DataFusionConfig::default(),
            logical_schema_sha256: "l".repeat(64),
            physical_schema_sha256: "p".repeat(64),
            retained_ordinals: vec![1, 0],
            state_schema_sha256: "s".repeat(64),
            output_schema_sha256: "o".repeat(64),
            native_descriptor: json!({"functions": ["sum", "count"], "keys": [1]}),
        },
        ledger: QuotaLedger {
            rows: 100,
            bytes: 1600,
            seen_input: true,
        },
        groups: 2,
        segments: SegmentDigests {
            logical_schema: "l".repeat(64),
            group_state: "g".repeat(64),
            batch_metadata: "m".repeat(64),
        },
    }
}

#[test]
fn compact_control_round_trip_binds_independent_identity_and_refunds_credit() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let pool = runtime
        .incremental_planner_context(0)
        .runtime_env()
        .memory_pool
        .clone();
    let expected = control();
    let encoded = encode(&runtime, &expected, "control", &|| Ok(())).unwrap();
    let encoded_charge = pool.reserved();
    assert!(encoded_charge > encoded.segment.bytes().len());
    let decoded = decode(&runtime, &encoded.segment, "control", &|| Ok(())).unwrap();
    assert_eq!(decoded.value, expected);
    assert!(pool.reserved() > encoded_charge);
    decoded.value.validate_identity(&expected.identity).unwrap();
    let mut independent = control();
    independent.identity.native_descriptor["keys"] = json!([0]);
    assert!(
        decoded
            .value
            .validate_identity(&independent.identity)
            .is_err()
    );
    let inline = decoded.value.inline_metadata(&encoded.segment);
    decoded
        .value
        .validate_inline(&inline, &encoded.segment)
        .unwrap();
    let mut changed = inline;
    changed.insert("rows".into(), json!(101));
    assert!(
        decoded
            .value
            .validate_inline(&changed, &encoded.segment)
            .is_err()
    );
    drop(decoded);
    assert_eq!(pool.reserved(), encoded_charge);
    drop(encoded);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn compact_control_unknown_fields_and_policies_are_rejected_without_credit_leaks() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let pool = runtime
        .incremental_planner_context(0)
        .runtime_env()
        .memory_pool
        .clone();
    for path in [vec![], vec!["identity"], vec!["ledger"], vec!["segments"]] {
        let mut value = serde_json::to_value(control()).unwrap();
        let mut object = &mut value;
        for key in path {
            object = &mut object[key];
        }
        object["unknown"] = json!(1);
        let segment = StateSegment::new(serde_json::to_vec(&value).unwrap());
        assert!(matches!(
            decode(&runtime, &segment, "control", &|| Ok(())),
            Err(CalcFlowError::Format { .. })
        ));
        assert_eq!(pool.reserved(), 0);
    }
    let trusted = control();
    for (field, replacement) in [
        ("state_layout", json!(4)),
        ("state_accounting", json!(2)),
        ("native_semantics", json!(2)),
        ("datafusion_version", json!("53.0.0")),
        ("state_policy", json!("float_v1")),
    ] {
        let mut value = serde_json::to_value(control()).unwrap();
        value[field] = replacement;
        let forged: CompactControl = serde_json::from_value(value).unwrap();
        assert!(forged.validate_identity(&trusted.identity).is_err());
    }
    let mut forged = control();
    forged.ledger.seen_input = false;
    assert!(forged.validate_identity(&trusted.identity).is_err());
}

#[test]
fn compact_control_cancellation_releases_encode_and_decode_candidates() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let pool = runtime
        .incremental_planner_context(0)
        .runtime_env()
        .memory_pool
        .clone();
    let expected = control();
    let encoded = encode(&runtime, &expected, "control", &|| Ok(())).unwrap();
    let before = pool.reserved();
    for stop in 1..=2 {
        let calls = Cell::new(0);
        let check = || {
            calls.set(calls.get() + 1);
            if calls.get() == stop {
                Err(CalcFlowError::Cancelled {
                    run_id: "control".into(),
                })
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            encode(&runtime, &expected, "control", &check),
            Err(CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(calls.get(), stop);
        assert_eq!(pool.reserved(), before);
        calls.set(0);
        assert!(matches!(
            decode(&runtime, &encoded.segment, "control", &check),
            Err(CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(calls.get(), stop);
        assert_eq!(pool.reserved(), before);
    }
    drop(encoded);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn compact_control_binds_content_digests_for_each_segment() {
    let logical = StateSegment::new(b"logical schema".to_vec());
    let groups = StateSegment::new(b"group state".to_vec());
    let metadata = StateSegment::new(b"batch metadata".to_vec());
    let mut expected = control();
    expected.identity.logical_schema_sha256 = logical.sha256().into();
    expected.segments = SegmentDigests {
        logical_schema: logical.sha256().into(),
        group_state: groups.sha256().into(),
        batch_metadata: metadata.sha256().into(),
    };
    expected
        .validate_segments(&logical, &groups, &metadata)
        .unwrap();
    let replacement = StateSegment::new(b"altered".to_vec());
    for (logical, groups, metadata) in [
        (&replacement, &groups, &metadata),
        (&logical, &replacement, &metadata),
        (&logical, &groups, &replacement),
    ] {
        assert!(
            expected
                .validate_segments(logical, groups, metadata)
                .is_err()
        );
    }
}

#[test]
fn compact_control_depth_limit_round_trips_and_refuses_the_next_level() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let pool = runtime
        .incremental_planner_context(0)
        .runtime_env()
        .memory_pool
        .clone();
    let mut expected = control();
    expected.identity.native_descriptor =
        (0..crate::json::MAX_JSON_DEPTH - 2).fold(json!(0), |value, _| json!([value]));
    let encoded = encode(&runtime, &expected, "control", &|| Ok(())).unwrap();
    let decoded = decode(&runtime, &encoded.segment, "control", &|| Ok(())).unwrap();
    assert_eq!(decoded.value, expected);
    drop(decoded);
    let before = pool.reserved();
    let descriptor = std::mem::replace(&mut expected.identity.native_descriptor, Value::Null);
    expected.identity.native_descriptor = json!([descriptor]);
    assert!(matches!(
        encode(&runtime, &expected, "control", &|| Ok(())),
        Err(CalcFlowError::Format { .. })
    ));
    assert_eq!(pool.reserved(), before);
    drop(encoded);
    assert_eq!(pool.reserved(), 0);
}
