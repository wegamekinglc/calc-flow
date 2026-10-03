use super::*;

fn altered_policy(snapshot: &OperatorStateSnapshot, policy: Value) -> OperatorStateSnapshot {
    let mut control: Value = serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    control["state_policy"] = policy;
    let segment = StateSegment::new(serde_json::to_vec(&control).unwrap());
    let mut snapshot = snapshot.clone();
    snapshot
        .inline_metadata
        .insert("control_sha256".into(), json!(segment.sha256()));
    snapshot.segments.insert("control".into(), segment);
    snapshot
}

#[tokio::test]
async fn test_grouped_float_current_policy_is_required_and_cold_replanned() {
    for dtype in [DataType::Float32, DataType::Float64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut source = operator(&dtype);
        drop(
            process(
                &mut source,
                input(&dtype, &[vec![(Some(1), Some(ONE))]], 0),
                &context,
            )
            .await,
        );
        let snapshot = native_capture(&mut source, &dtype);
        let control: Value = serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
        let original = control["state_policy"].clone();
        for field in ["max_record_rows", "config", "factory", "model"] {
            let mut policy = original.clone();
            policy["sequential_grouped_float_v1"]
                .as_object_mut()
                .unwrap()
                .remove(field);
            assert!(
                StreamOperator::restore(&mut operator(&dtype), &altered_policy(&snapshot, policy))
                    .is_err()
            );
        }
        for (field, value) in [
            ("unknown", json!(1)),
            ("factory", json!("unknown")),
            ("factory", json!("boolean_v1")),
            ("model", json!("unknown")),
            ("max_record_rows", json!(2)),
        ] {
            let mut policy = original.clone();
            policy["sequential_grouped_float_v1"][field] = value;
            assert!(
                StreamOperator::restore(&mut operator(&dtype), &altered_policy(&snapshot, policy))
                    .is_err()
            );
        }
        let mut policy = original;
        policy["sequential_grouped_float_v1"]["config"]
            .as_object_mut()
            .unwrap()
            .remove("batch_size");
        assert!(
            StreamOperator::restore(&mut operator(&dtype), &altered_policy(&snapshot, policy))
                .is_err()
        );
        let pool = source
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((source, snapshot));
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_grouped_float_nondefault_requested_partition_keeps_current_raw4() {
    for dtype in [DataType::Float32, DataType::Float64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = operator(&dtype);
        state.set_stream_resources(
            DataFusionConfig {
                target_partitions: 4,
                ..DataFusionConfig::default()
            },
            UdfRegistrySnapshot::default(),
            vec![],
        );
        let first = vec![(Some(1), Some(ONE))];
        let actual = process(
            &mut state,
            input(&dtype, std::slice::from_ref(&first), 0),
            &context,
        )
        .await;
        assert_oracle(&actual, &dtype, &[first], 0).await;
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(snapshot.inline_metadata["state_layout"], json!(4));
        assert!(state.incremental.is_none() && state.compact.is_none() && state.retained.is_some());
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((state, snapshot, actual));
        assert_eq!(pool.reserved(), 0);
    }
}
