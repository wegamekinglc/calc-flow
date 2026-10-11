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
    assert_eq!(
        alias["credit_scope"],
        "per_segment_backing_plus_shared_capture_controls"
    );
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

#[tokio::test]
async fn test_dirty_replacement_fits_bounded_pool_and_preserves_owned_history() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    const ROWS: usize = 32_768;
    const INITIAL_ROWS: usize = 28_672;
    const LIMIT: usize = 10 << 20;

    fn observation(
        phase: &str,
        pool: &ObservedPool,
        operator: &StreamJoinOperator,
        bases: &AtomicUsize,
    ) -> Value {
        json!({
            "phase": phase,
            "pool": pool.snapshot(),
            "left_rows": operator.status().left.retained_rows,
            "pending_changes": operator.state.deltas.pending.iter().count(),
            "base_writer_calls": bases.load(Ordering::SeqCst),
        })
    }

    let pool = Arc::new(ObservedPool::new("dirty-replacement", LIMIT));
    let mut operator = fixture_operator(ROWS).unwrap();
    let job = job();
    let bases = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&bases);
    operator.set_checkpoint_writer_base_test_hook(Arc::new(move |_| {
        counted.fetch_add(1, Ordering::SeqCst);
    }));
    let mut phase = "initialize".to_owned();
    let mut observations = Vec::new();
    let mut captures = Vec::new();
    let mut held = None;
    let mut alias_name = String::new();
    let mut alias_digest = String::new();
    let mut owners = Vec::new();
    let mut row_owner = None;
    let mut before_oracle = None;
    let result: Result<()> = async {
        operator.runtime.initialize_benchmark_pool(pool.clone())?;
        operator.ingress_progress = progress(None);
        for epoch in 1..=5 {
            phase = format!("epoch_{epoch}_admission");
            pool.begin_phase();
            if epoch == 1 {
                feed(&mut operator, &job, false, 0, ENTITIES).await?;
            }
            let (start, end) = if epoch == 1 {
                (0, INITIAL_ROWS)
            } else {
                let start = INITIAL_ROWS + (epoch - 2) * 512;
                (start, start + 512)
            };
            feed_quotes(&mut operator, &job, start, end, ROWS).await?;
            observations.push(observation(&phase, &pool, &operator, &bases));
            phase = format!("epoch_{epoch}_prepare");
            pool.begin_phase();
            prepare(&mut operator, &job).await?;
            observations.push(observation(&phase, &pool, &operator, &bases));
            phase = format!("epoch_{epoch}_capture");
            pool.begin_phase();
            let snapshot = operator.checkpoint(Epoch::new(epoch as u64).unwrap())?;
            let deltas = &snapshot.inline_metadata["v2_inventory"]["deltas"];
            require(
                deltas
                    .as_array()
                    .is_some_and(|deltas| deltas.len() == epoch - 1),
                "scaled dirty replacement did not capture fresh base and four deltas",
            )?;
            captures.push(json!({
                "epoch": epoch,
                "base_epoch": snapshot.inline_metadata["v2_inventory"]["base_epoch"],
                "delta_count": deltas.as_array().map(Vec::len),
                "segments": segments(&snapshot),
            }));
            observations.push(observation(&phase, &pool, &operator, &bases));
            if epoch == 5 {
                owners = snapshot
                    .segments
                    .iter()
                    .map(|(name, segment)| {
                        (
                            name.clone(),
                            Arc::downgrade(&segment.bytes_arc()),
                            segment.bytes_arc().capacity(),
                            segment.has_owner(),
                        )
                    })
                    .collect();
                let (alias, evidence) = held_segment(&snapshot, pool.id)?;
                alias_name = evidence["name"].as_str().unwrap().to_owned();
                alias_digest = evidence["sha256"].as_str().unwrap().to_owned();
                held = Some(alias);
            }
            drop(snapshot);
        }
        phase = "epoch_6_admission".into();
        pool.begin_phase();
        feed_quotes(&mut operator, &job, INITIAL_ROWS + 4 * 512, ROWS, ROWS).await?;
        observations.push(observation(&phase, &pool, &operator, &bases));
        before_oracle = Some(validate_state(&operator, ROWS)?);
        row_owner = Some(Arc::downgrade(&operator.state.left.0));
        phase = "epoch_6_prepare".into();
        pool.begin_phase();
        prepare(&mut operator, &job).await?;
        observations.push(observation(&phase, &pool, &operator, &bases));
        phase = "epoch_6_capture".into();
        pool.begin_phase();
        let snapshot = operator.checkpoint(Epoch::new(6).unwrap())?;
        captures.push(json!({
            "epoch": 6,
            "base_epoch": snapshot.inline_metadata["v2_inventory"]["base_epoch"],
            "delta_count": snapshot.inline_metadata["v2_inventory"]["deltas"]
                .as_array().map(Vec::len),
            "segments": segments(&snapshot),
        }));
        drop(snapshot);
        Ok(())
    }
    .await;
    observations.push(observation(&phase, &pool, &operator, &bases));
    let original_error = result
        .as_ref()
        .err()
        .map(|error| json!({"debug": format!("{error:?}"), "display": error.to_string()}));
    let after_oracle = validate_state(&operator, ROWS).map_err(|error| format!("{error:?}"));
    let alias_unchanged = held
        .as_ref()
        .is_some_and(|alias| hex::encode(Sha256::digest(alias.bytes())) == alias_digest);
    let owners_before_drop = owners
        .iter()
        .map(|(name, owner, capacity, funded)| {
            json!({"name": name, "capacity": capacity, "funded": funded,
            "alive": owner.upgrade().is_some(), "explicit_alias": *name == alias_name})
        })
        .collect::<Vec<_>>();
    drop(operator);
    let after_operator_drop = pool.snapshot();
    let cleanup = crate::benchmark_internals::settle_job(&job).await;
    let after_drain = pool.snapshot();
    drop(job);
    let after_job_drop = pool.snapshot();
    let owners_after_drain = owners.iter().map(|(name, owner, capacity, _)| {
        json!({"name": name, "capacity": capacity, "alive": owner.upgrade().is_some()})
    }).collect::<Vec<_>>();
    let rows_retired = row_owner.is_some_and(|owner| owner.upgrade().is_none());
    drop(held);
    let all_segments_retired = owners
        .iter()
        .all(|(_, owner, _, _)| owner.upgrade().is_none());
    let final_pool = pool.snapshot();
    let complete_cleanup =
        cleanup.is_ok() && final_pool.reserved_bytes == 0 && all_segments_retired;
    let evidence = json!({
        "case": "dirty_replacement_scaled",
        "left_rows": ROWS, "right_rows": ENTITIES, "pool_limit_bytes": LIMIT,
        "last_phase": phase, "original_error": original_error,
        "observations": observations, "captures": captures,
        "before_oracle": before_oracle, "after_oracle": after_oracle,
        "held_alias_name": alias_name, "held_alias_unchanged": alias_unchanged,
        "owners_before_operator_drop": owners_before_drop,
        "after_operator_drop": after_operator_drop, "after_drain": after_drain,
        "after_job_drop": after_job_drop, "owners_after_drain": owners_after_drain,
        "rows_retired": rows_retired, "all_segments_retired": all_segments_retired,
        "cleanup_error": cleanup.as_ref().err().map(|error| format!("{error:?}")),
        "final_pool": final_pool, "owned_cleanup_complete": complete_cleanup,
    });
    eprintln!("DIRTY_REPLACEMENT_EVIDENCE {evidence}");
    assert!(
        complete_cleanup,
        "diagnostic must settle before asserting the result"
    );
    assert!(
        result.is_ok(),
        "dirty replacement failed: {original_error:?}"
    );
    assert_eq!(Some(after_oracle.unwrap()), before_oracle);
    assert!(alias_unchanged && rows_retired && all_segments_retired);
}
