#![cfg(feature = "benchmark-internals")]

#[path = "../benches/support/native_join_components.rs"]
mod native_join_components;

#[cfg(not(feature = "benchmark-checkpoint-diagnostic"))]
#[test]
fn test_checkpoint_history_requires_explicit_diagnostic_feature() {
    let result = native_join_components::parse_arguments(&[
        "--native-component".into(),
        "join_checkpoint_history_4m".into(),
        "--check".into(),
    ]);
    assert_eq!(result, Err("unknown native component"));
}

#[cfg(feature = "benchmark-checkpoint-diagnostic")]
#[test]
fn test_checkpoint_history_is_available_with_explicit_diagnostic_feature() {
    let result = native_join_components::parse_arguments(&[
        "--native-component".into(),
        "join_checkpoint_history_4m".into(),
        "--check".into(),
    ]);
    assert_eq!(
        result,
        Ok(Some((
            native_join_components::Component::CheckpointHistory,
            true
        )))
    );
}

#[test]
fn test_native_component_arguments_are_closed_before_execution() {
    use native_join_components::{Component, parse_arguments};
    assert!(parse_arguments(&["--list".into()]).unwrap().is_none());
    assert_eq!(
        parse_arguments(&[
            "--native-component".into(),
            "join_capture_restore_1m".into()
        ])
        .unwrap(),
        Some((Component::CaptureRestore, false))
    );
    assert_eq!(
        parse_arguments(&[
            "--native-component".into(),
            "join_compaction_1m".into(),
            "--check".into()
        ])
        .unwrap(),
        Some((Component::Compaction, true))
    );
    for arguments in [
        vec!["--native-component"],
        vec!["--native-component", "unknown"],
        vec![
            "--native-component",
            "join_compaction_1m",
            "--rows",
            "2000000",
        ],
        vec!["--native-component", "join_checkpoint_history_4m"],
        vec![
            "--native-component",
            "join_checkpoint_history_4m",
            "--rows",
            "24",
        ],
        vec![
            "--native-component",
            "join_checkpoint_history_4m",
            "--check",
            "--rows",
            "24",
        ],
    ] {
        assert!(
            parse_arguments(&arguments.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err()
        );
    }
}

#[tokio::test]
async fn test_capture_restore_counts_actual_dirty_rows_and_validates_replay() {
    let result = native_join_components::capture_restore(24).await;
    assert_eq!(result["resident_rows"], 24);
    assert_eq!(result["captured_dirty_rows"], 24);
    assert_eq!(result["restored_live_rows"], 24);
    assert_eq!(result["validated_all_segments"], true);
    assert_eq!(result["restore_output_equivalent"], true);
    assert_eq!(result["replay_output_rows"], 24);
    assert!(
        result["snapshot_seconds"].as_f64().unwrap()
            >= result["preparation_seconds"].as_f64().unwrap()
                + result["capture_seconds"].as_f64().unwrap()
    );
    let segments = result["snapshot_segments"].as_array().unwrap();
    assert!(!segments.is_empty());
    assert_eq!(
        segments
            .iter()
            .map(|entry| entry["bytes"].as_u64().unwrap())
            .sum::<u64>(),
        result["snapshot_bytes"].as_u64().unwrap()
    );
}

#[tokio::test]
async fn test_capture_bounds_admission_batches_and_accounts_for_lifecycle() {
    let result = native_join_components::capture_restore_batched(24, 5).await;
    assert_eq!(
        result["admission_batch_rows"],
        serde_json::json!([5, 5, 2, 5, 5, 2])
    );
    for clock in ["setup_seconds", "oracle_seconds", "cleanup_seconds"] {
        assert!(result[clock].as_f64().unwrap() > 0.0);
    }
    assert_eq!(result["owned_cleanup_complete"], true);
}

#[tokio::test]
async fn test_compaction_proves_replacement_and_preserves_actual_rows() {
    let result = native_join_components::compaction(24, 4).await;
    assert_eq!(result["resident_rows"], 24);
    assert_eq!(result["incoming_rows"], 4);
    assert_eq!(result["outputs_equal"], true);
    assert_eq!(result["state_equal"], true);
    assert_eq!(result["restore_output_equivalent"], true);
    for mode in ["armed", "steady"] {
        let sample = &result[mode];
        for clock in ["setup_seconds", "oracle_seconds", "cleanup_seconds"] {
            assert!(sample[clock].as_f64().unwrap() > 0.0);
        }
        assert_eq!(sample["owned_cleanup_complete"], true);
        assert_eq!(
            sample["admission_batch_rows"],
            serde_json::json!([9, 9, 1, 1, 1, 1, 1, 1])
        );
        assert_eq!(sample["pre_handler_retained_rows"], 24);
        assert_eq!(sample["post_handler_retained_rows"], 28);
        assert_eq!(sample["replay_output_rows"], 28);
        let witness = &sample["compaction_witness"];
        let schedule = &sample["schedule"];
        assert_eq!(schedule["during_compaction_status"], "unmeasured");
        assert!(
            schedule["handler_started_ns"].as_u64().unwrap()
                >= schedule["preparation_completed_ns"].as_u64().unwrap()
        );
        assert!(
            schedule["preparation_completed_ns"].as_u64().unwrap()
                >= schedule["handler_ready_ns"].as_u64().unwrap()
        );
        assert!(sample["queue_seconds"].as_f64().unwrap() >= 0.0);
        for phase in ["preparation", "handler", "combined"] {
            assert_eq!(sample["resource_phases"][phase]["scope"], phase);
            #[cfg(target_os = "linux")]
            assert_eq!(sample["resource_phases"][phase]["status"], "measured");
        }
        assert_eq!(witness["setup_dirty_epochs"], 4);
        if mode == "armed" {
            assert_ne!(witness["before_base_sha256"], witness["after_base_sha256"]);
            assert!(
                witness["before_deltas"].as_array().unwrap().len()
                    > witness["after_deltas"].as_array().unwrap().len()
            );
        } else {
            assert_eq!(witness["before_base_sha256"], witness["after_base_sha256"]);
        }
        assert!(
            sample["combined_seconds"].as_f64().unwrap()
                >= sample["preparation_seconds"].as_f64().unwrap()
                    + sample["handler_seconds"].as_f64().unwrap()
        );
    }
}

#[tokio::test]
async fn test_layout_controls_execute_both_joins_with_exact_batch_geometry() {
    let result = native_join_components::layout_controls().await;
    let controls = result["controls"].as_array().unwrap();
    assert_eq!(controls.len(), 8);
    for sample in controls {
        let expected_rows = if sample["operator"] == "join" && sample["layout"] == "unordered" {
            144
        } else {
            24
        };
        assert_eq!(sample["output_rows"], expected_rows);
        assert_eq!(sample["validated_all_rows"], true);
        assert_eq!(sample["input_unchanged"], true);
        assert_eq!(sample["owned_cleanup_complete"], true);
        match sample["layout"].as_str().unwrap() {
            "multi_record" => assert_eq!(
                sample["event_record_batch_rows"],
                serde_json::json!([[7, 9, 8], [7, 9, 8]])
            ),
            "unordered" => {
                assert_eq!(sample["timestamp_descents_per_event"], 23);
                assert_eq!(sample["per_key_timestamp_descents"], 20);
                assert_eq!(sample["unique_keys"], 4);
            }
            "wide_key" => assert_eq!(sample["key_bytes_per_event"], 24 * 1024),
            "single_record" => assert_eq!(sample["event_record_batch_count"], 1),
            _ => panic!("unexpected layout"),
        }
    }
}
