use super::*;

#[test]
fn materialized_window_guard_rejects_wrong_finality_and_wrong_numeric_values() {
    let request = WindowRequest::new(ENTITIES * WINDOW, ENTITIES);
    let expected = reference_record(request.start, request.rows);
    let metadata = BatchMetadata::default();
    let valid = WindowOutput {
        batches: vec![Batch::table(vec![expected.clone()], metadata.clone()).unwrap()],
        watermark: request.watermark,
    };
    assert!(validate_window(&request, &valid).is_ok());

    let wrong_finality = WindowOutput {
        batches: valid.batches.clone(),
        watermark: EventTime::from_micros(request.watermark.as_micros() - 1),
    };
    assert!(validate_window(&request, &wrong_finality).is_err());

    let mut columns = expected.columns().to_vec();
    columns[4] = Arc::new(Float64Array::from(vec![0.0; request.rows]));
    let wrong_values = WindowOutput {
        batches: vec![
            Batch::table(
                vec![RecordBatch::try_new(expected.schema(), columns).unwrap()],
                metadata,
            )
            .unwrap(),
        ],
        watermark: request.watermark,
    };
    assert!(validate_window(&request, &wrong_values).is_err());
}

#[test]
fn allocator_json_preserves_signed_old_object_releases_and_individual_peaks() {
    let counters = allocation_counter::AllocationInfo {
        count_total: 3,
        count_current: -2,
        count_max: 2,
        bytes_total: 96,
        bytes_current: -4096,
        bytes_max: 64,
    };
    let value = allocation_json(counters);
    assert_eq!(value["count_total"], 3);
    assert_eq!(value["bytes_total"], 96);
    assert_eq!(value["count_current"], -2);
    assert_eq!(value["bytes_current"], -4096);
    assert_eq!(value["count_max"], 2);
    assert_eq!(value["bytes_max"], 64);
}

#[test]
fn real_pair_and_independent_tasks_preserve_graph_output_and_state_behavior() {
    let independent = measure_arm(TaskMode::Independent, ENTITIES * WINDOW, ENTITIES);
    let fused = measure_arm(TaskMode::Fused, ENTITIES * WINDOW, ENTITIES);
    for (mode, arm) in [
        (TaskMode::Independent, &independent),
        (TaskMode::Fused, &fused),
    ] {
        let evidence = &arm.0;
        assert_eq!(
            evidence["physical_operator_drivers"],
            mode.physical_drivers()
        );
        assert_eq!(evidence["logical_operator_tasks"], 2);
        assert_eq!(evidence["remaining_logical_tasks_after_cleanup"], 0);
        for phase in ["startup_to_ready", "warm_data_watermark_to_output"] {
            let counters = &evidence[phase];
            assert!(
                counters["count_total"]
                    .as_u64()
                    .is_some_and(|value| value > 0)
            );
            assert!(
                counters["bytes_total"]
                    .as_u64()
                    .is_some_and(|value| value > 0)
            );
            assert!(counters["count_max"].as_u64().is_some());
            assert!(counters["bytes_max"].as_u64().is_some());
            assert!(counters["count_current"].as_i64().is_some());
            assert!(counters["bytes_current"].as_i64().is_some());
        }
    }
    assert_eq!(
        independent.0["graph_fingerprint"],
        fused.0["graph_fingerprint"]
    );
    assert_eq!(independent.1, fused.1);
    assert_eq!(independent.2, fused.2);
}
