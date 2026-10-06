use super::*;
use crate::CalcFlowError;
use datafusion::execution::memory_pool::MemoryConsumer;

#[tokio::test]
async fn test_composite_key_restore_rejects_wrong_factory_atomically() {
    let dtype = DataType::Float64;
    let key_type = DataType::Utf8;
    let bucket_type = DataType::Int64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "grouped_float", None);
    let mut state = composite_operator(&dtype, &key_type, &bucket_type);
    drop(
        process(
            &mut state,
            composite_input(&dtype, &key_type, &bucket_type, &string_arrivals()[0], 0),
            &context,
        )
        .await,
    );
    let snapshot = composite_capture(&mut state);
    for factory in [
        "primitive_v1",
        "boolean_v1",
        "utf8_v1",
        "large_utf8_v1",
        "unknown",
    ] {
        assert!(StreamOperator::restore(&mut state, &with_factory(&snapshot, factory)).is_err());
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

#[tokio::test]
async fn test_composite_key_refused_update_refunds_and_retries_exactly() {
    for bucket_type in [DataType::Int64, DataType::Boolean] {
        let dtype = DataType::Float64;
        let key_type = DataType::Utf8;
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = composite_operator(&dtype, &key_type, &bucket_type);
        let mut parts = string_arrivals().into_iter();
        let mut prefix = parts.next().unwrap();
        drop(
            process(
                &mut state,
                composite_input(&dtype, &key_type, &bucket_type, &prefix, 0),
                &context,
            )
            .await,
        );
        let snapshot = composite_capture(&mut state);
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let basis = pool.reserved();
        let next = parts.next().unwrap();
        let pressure = MemoryConsumer::new("composite-update-pressure").register(&pool);
        pressure.try_grow((1 << 30) - basis - 1).unwrap();
        let held = pool.reserved();
        let rejected = composite_input(&dtype, &key_type, &bucket_type, &next, 1);
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
        let rejected = composite_input(&dtype, &key_type, &bucket_type, &next, 1);
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
            composite_input(&dtype, &key_type, &bucket_type, &next, 1),
            &context,
        )
        .await;
        prefix.extend(next);
        composite_oracle(
            &actual,
            composite_input(&dtype, &key_type, &bucket_type, &prefix, 1),
        )
        .await;
        drop((actual, state, snapshot));
        assert_eq!(pool.reserved(), 0);
    }
}
