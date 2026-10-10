use super::*;

use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool};
use std::sync::{Arc, Barrier};

#[test]
fn exact_pool_peak_retains_transients_refunds_aliases_and_isolates_pools() {
    let pool = Arc::new(ObservedPool::new("first", 1_024));
    let memory: Arc<dyn MemoryPool> = pool.clone();
    let owner = MemoryConsumer::new("owner").register(&memory);
    owner.try_grow(600).unwrap();
    pool.begin_phase();
    let child = Arc::new(owner.split(200));
    let alias = Arc::clone(&child);
    owner.shrink(400);
    assert_eq!(pool.snapshot().reserved_bytes, 200);
    assert!(owner.try_grow(900).is_err());
    let rejected = pool.snapshot();
    assert_eq!(rejected.peak_reserved_bytes, 600);
    assert_eq!(rejected.phase_peak_reserved_bytes, 600);
    assert_eq!(rejected.reserved_bytes, 200);
    assert_eq!(rejected.failed_grows, 1);
    assert_eq!(
        serde_json::to_value(&rejected).unwrap()["last_rejection"],
        json!({"additional_bytes": 900, "reserved_bytes": 200,
            "limit_bytes": 1_024, "consumer_name": "owner"})
    );
    drop(child);
    assert_eq!(pool.snapshot().reserved_bytes, 200);
    drop(alias);
    assert_eq!(pool.snapshot().reserved_bytes, 0);
    pool.begin_phase();
    owner.try_grow(100).unwrap();
    owner.shrink(100);
    assert_eq!(pool.snapshot().phase_peak_reserved_bytes, 100);
    assert_eq!(pool.snapshot().peak_reserved_bytes, 600);
    assert_eq!(pool.snapshot().successful_grows, 2);
    let independent = ObservedPool::new("second", 1_024);
    assert_eq!(independent.snapshot().peak_reserved_bytes, 0);
    assert_ne!(independent.snapshot().pool_id, pool.snapshot().pool_id);
}

#[test]
fn concurrent_pool_reservations_have_one_simultaneous_peak() {
    let pool = Arc::new(ObservedPool::new("concurrent", 1_024));
    let barrier = Arc::new(Barrier::new(2));
    std::thread::scope(|scope| {
        for _ in 0..2 {
            let pool = Arc::clone(&pool);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                let memory: Arc<dyn MemoryPool> = pool;
                let reservation = MemoryConsumer::new("thread").register(&memory);
                reservation.try_grow(256).unwrap();
                barrier.wait();
            });
        }
    });
    let observed = pool.snapshot();
    assert_eq!(observed.peak_reserved_bytes, 512);
    assert_eq!(observed.reserved_bytes, 0);
    assert_eq!(observed.successful_grows, 2);
    assert_eq!(observed.failed_grows, 0);
}

#[test]
fn pool_installation_preserves_resources_udf_resolution_and_existing_runtime() {
    let mut operator = fixture_operator(3_072).unwrap();
    let config = crate::DataFusionConfig {
        batch_size: 123,
        target_partitions: 3,
        ..crate::DataFusionConfig::default()
    };
    let missing = crate::UdfReference::new(
        "diagnostic",
        "missing",
        "v1",
        crate::UdfKind::DataFusionScalar,
    )
    .unwrap();
    operator
        .runtime
        .set_resources(config, crate::UdfRegistrySnapshot::default(), vec![missing]);
    let pool = Arc::new(ObservedPool::new("runtime", 1 << 30));
    let memory: Arc<dyn MemoryPool> = pool.clone();
    assert!(
        operator
            .runtime
            .initialize_benchmark_pool(Arc::clone(&memory))
            .is_err()
    );
    assert!(!operator.runtime.is_initialized());
    operator
        .runtime
        .set_resources(config, crate::UdfRegistrySnapshot::default(), Vec::new());
    operator
        .runtime
        .initialize_benchmark_pool(Arc::clone(&memory))
        .unwrap();
    assert!(!operator.runtime.runtime().unwrap().serial_owned_sql());
    assert_eq!(operator.runtime.resources.as_ref().unwrap().0, config);
    let before = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    assert!(Arc::ptr_eq(&before, &memory));
    let replacement: Arc<dyn MemoryPool> = Arc::new(ObservedPool::new("replacement", 1 << 30));
    assert!(
        operator
            .runtime
            .initialize_benchmark_pool(replacement)
            .is_err()
    );
    let after = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    assert!(Arc::ptr_eq(&before, &after));
}

#[tokio::test]
async fn complete_state_oracle_rejects_identity_key_payload_and_counter_changes() {
    let mut diagnostic = Diagnostic::new(3_072).unwrap();
    let (mut operator, pool) = diagnostic.operator(3_072, "oracle").unwrap();
    let job = job();
    feed(&mut operator, &job, false, 0, ENTITIES).await.unwrap();
    feed_quotes(&mut operator, &job, 0, 3_072, 3_072)
        .await
        .unwrap();
    let expected = validate_state(&operator, 3_072).unwrap();
    let first = operator.state.left[0].clone();
    operator.state.left[0].row_id += 1;
    assert!(validate_state(&operator, 3_072).is_err());
    operator.state.left[0] = first.clone();
    operator.state.left[0].encoded_key = operator.state.left[1].encoded_key.clone();
    assert!(validate_state(&operator, 3_072).is_err());
    operator.state.left[0] = first.clone();
    operator.state.left[0].record = operator.state.left[64].record.clone();
    assert!(validate_state(&operator, 3_072).is_err());
    operator.state.left[0] = first;
    operator.state.metrics.emitted_match_rows = 0;
    assert!(validate_state(&operator, 3_072).is_err());
    operator.state.metrics.emitted_match_rows = 3_072;
    assert_eq!(validate_state(&operator, 3_072).unwrap(), expected);
    drop(operator);
    crate::benchmark_internals::settle_job(&job).await.unwrap();
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn retained_history_preserves_row3_fixture_progress_and_complete_restores() {
    let evidence = run_history(3_072).await.unwrap();
    assert_eq!(evidence["outcome"], "complete", "{evidence:#}");
    assert_eq!(evidence["left_rows"], 3_072);
    assert_eq!(evidence["dimension_rows"], 64);
    assert_eq!(evidence["resident_rows"], 3_136);
    assert_eq!(
        evidence["fresh"]["oracle"]["left_sha256"],
        "4c260a9fe2daaa170836c24b210769b67cd01f1af57cfefb33dfb855a8d5878f"
    );
    assert_eq!(
        evidence["fresh"]["oracle"]["right_sha256"],
        "7b57c353d760ffda074a208ec5e648f92272b611d59041873c3535e1dd8039f2"
    );
    let history = &evidence["history"];
    assert_eq!(history["dirty_epochs"], 4);
    assert_eq!(history["before"]["left_rows"], 3_072);
    assert_eq!(history["before"]["right_rows"], 64);
    assert_ne!(
        history["before"]["base_sha256"],
        history["after"]["base_sha256"]
    );
    assert!(
        history["before"]["delta_names"].as_array().unwrap().len()
            > history["after"]["delta_names"].as_array().unwrap().len()
    );
    assert_eq!(history["old_segments_unchanged"], true);
    let alias = &history["alias_ownership"];
    assert_eq!(alias["kind"], "StateSegment");
    assert_eq!(alias["pool_id"], "history");
    assert_eq!(alias["has_owner"], true);
    assert_eq!(alias["credit_scope"], "shared_encoder_owner");
    assert_eq!(alias["snapshot_epoch"], 4);
    assert!(
        history["before"]["segments"]
            .as_array()
            .unwrap()
            .iter()
            .any(|segment| {
                segment["name"] == alias["name"]
                    && segment["bytes"] == alias["bytes"]
                    && segment["sha256"] == alias["sha256"]
            })
    );
    let checkpoints = history["checkpoints"].as_array().unwrap();
    assert_eq!(checkpoints.len(), 4);
    for (index, checkpoint) in checkpoints.iter().enumerate() {
        assert_eq!(checkpoint["left_rows"], 3_069 + index);
        assert_eq!(checkpoint["right_rows"], 64);
        assert_eq!(
            checkpoint["left_watermark_micros"],
            BASE_MICROS + 47_000_000 + i64::from(index == 3)
        );
        assert!(checkpoint["right_watermark_micros"].is_null());
    }
    for restored in evidence["restores"].as_array().unwrap() {
        assert_eq!(restored["oracle"], evidence["fresh"]["oracle"]);
    }
    assert_eq!(evidence["restores"].as_array().unwrap().len(), 2);
    for pool in evidence["pools"].as_array().unwrap() {
        assert_eq!(pool["reserved_bytes"], 0);
        assert_eq!(pool["limit_bytes"], 1 << 30);
        assert!(pool["peak_reserved_bytes"].as_u64().unwrap() > 0);
    }
    assert_eq!(evidence["owned_cleanup_complete"], true);
}
