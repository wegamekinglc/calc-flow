use super::*;
use crate::CalcFlowError;
use datafusion::execution::memory_pool::MemoryConsumer;

fn with_factory(snapshot: &OperatorStateSnapshot, factory: &str) -> OperatorStateSnapshot {
    let mut control: Value = serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    control["state_policy"]["sequential_grouped_float_v1"]["factory"] = json!(factory);
    let segment = StateSegment::new(serde_json::to_vec(&control).unwrap());
    let mut result = snapshot.clone();
    result
        .inline_metadata
        .insert("control_sha256".into(), json!(segment.sha256()));
    result.segments.insert("control".into(), segment);
    result
}

#[tokio::test]
async fn test_string_key_restore_rejects_wrong_factory_atomically() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for key_type in [DataType::Utf8, DataType::LargeUtf8] {
            let job = job();
            let context = StreamOperatorContext::new(&job, "grouped_float", None);
            let mut state = string_operator(&dtype, &key_type, ALL);
            drop(
                process(
                    &mut state,
                    string_input(&dtype, &key_type, &string_arrivals()[0], 0),
                    &context,
                )
                .await,
            );
            let snapshot = string_capture(&mut state, &key_type);
            let opposite = if key_type == DataType::Utf8 {
                "large_utf8_v1"
            } else {
                "utf8_v1"
            };
            for factory in ["primitive_v1", "boolean_v1", opposite, "unknown"] {
                assert!(
                    StreamOperator::restore(&mut state, &with_factory(&snapshot, factory)).is_err()
                );
                same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
            }
            let pool = state
                .stream_state
                .runtime()
                .unwrap()
                .incremental_memory_pool();
            drop((state, snapshot));
            assert_eq!(pool.reserved(), 0);
        }
    }
}

#[tokio::test]
async fn test_string_key_refused_update_refunds_and_retries_exactly() {
    for key_type in [DataType::Utf8, DataType::LargeUtf8] {
        let dtype = DataType::Float64;
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = string_operator(&dtype, &key_type, ALL);
        let mut parts = string_arrivals().into_iter();
        let mut prefix = parts.next().unwrap();
        drop(
            process(
                &mut state,
                string_input(&dtype, &key_type, &prefix, 0),
                &context,
            )
            .await,
        );
        let snapshot = string_capture(&mut state, &key_type);
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let basis = pool.reserved();
        let next = parts.next().unwrap();
        let pressure = MemoryConsumer::new("string-update-pressure").register(&pool);
        pressure.try_grow((1 << 30) - basis - 1).unwrap();
        let held = pool.reserved();
        let rejected = string_input(&dtype, &key_type, &next, 1);
        let weak = weak_arrays(&rejected);
        let mut output = EdgeCollector::new(state.output_ports().to_vec());
        assert!(
            state
                .process_data("events", rejected, &context, &mut output)
                .await
                .is_err()
        );
        assert!(output.drain("output").is_empty());
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        assert_eq!(pool.reserved(), held);
        drop(pressure);
        assert_eq!(pool.reserved(), basis);
        same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
        let rejected = string_input(&dtype, &key_type, &next, 1);
        let weak = weak_arrays(&rejected);
        assert!(
            matches!(state.process_data("events", rejected, &context, &mut Reject).await,
            Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-grouped-float")
        );
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        assert_eq!(pool.reserved(), basis);
        same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
        let actual = process(
            &mut state,
            string_input(&dtype, &key_type, &next, 1),
            &context,
        )
        .await;
        prefix.extend(next);
        string_oracle(&actual, ALL, string_input(&dtype, &key_type, &prefix, 1)).await;
        drop((actual, state, snapshot));
        assert_eq!(pool.reserved(), 0);
    }
}
