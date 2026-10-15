use super::*;

use std::sync::atomic::{AtomicUsize, Ordering};

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
        "right_rows": operator.status().right.retained_rows,
        "pending_changes": operator.state.deltas.pending.iter().count(),
        "base_writer_calls": bases.load(Ordering::SeqCst),
    })
}

const INITIAL_ROWS: usize = 3_712_000;
const PENDING_ROWS: usize = 32_000;

#[derive(Default)]
struct FullReplacementDiagnostic {
    phase: String,
    observations: Vec<Value>,
    captures: Vec<Value>,
    owners: Vec<(String, std::sync::Weak<Vec<u8>>, String)>,
    replacement_owners: Vec<(String, std::sync::Weak<Vec<u8>>)>,
    held: Option<crate::StateSegment>,
    alias_name: String,
    alias_digest: String,
    row_owner: Option<std::sync::Weak<crate::operator::join::StateRows>>,
    restored_row_owner: Option<std::sync::Weak<crate::operator::join::StateRows>>,
    replacement: Option<OperatorStateSnapshot>,
    before_oracle: Option<Value>,
    after_oracle: Option<Value>,
    restored_oracle: Option<Value>,
    original_error: Option<Value>,
}

struct FullReplacementSettlement {
    cleanup: Result<()>,
    after_operator_drop: PoolSnapshot,
    after_drain: PoolSnapshot,
    rows_retired: Option<bool>,
}

fn validate_historical_payload(
    name: &str,
    owner: &std::sync::Weak<Vec<u8>>,
    digest: &str,
    snapshot: &OperatorStateSnapshot,
) -> Result<()> {
    let previous = owner.upgrade().ok_or_else(|| {
        crate::benchmark_internals::invalid("historical payload owner retired early")
    })?;
    let current = snapshot.segments.get(name).ok_or_else(|| {
        crate::benchmark_internals::invalid("replacement lost a historical payload")
    })?;
    require(
        Arc::ptr_eq(&previous, &current.bytes_arc())
            && hex::encode(Sha256::digest(current.bytes())) == digest,
        "replacement must preserve historical payload identity and bytes",
    )?;
    Ok(())
}

impl FullReplacementSettlement {
    fn complete_cleanup(
        &self,
        final_pool: &PoolSnapshot,
        final_restore_pool: &PoolSnapshot,
        all_segments_retired: bool,
        restored_rows_retired: Option<bool>,
    ) -> bool {
        self.cleanup.is_ok()
            && final_pool.reserved_bytes == 0
            && final_restore_pool.reserved_bytes == 0
            && all_segments_retired
            && self.rows_retired != Some(false)
            && restored_rows_retired != Some(false)
    }
}

impl FullReplacementDiagnostic {
    async fn admit_epoch(
        &mut self,
        operator: &mut StreamJoinOperator,
        job: &StreamJobContext,
        pool: &ObservedPool,
        epoch: usize,
    ) -> Result<usize> {
        self.phase = format!("epoch_{epoch}_admission");
        pool.begin_phase();
        if epoch == 1 {
            feed(operator, job, false, 0, ENTITIES).await?;
        }
        let (start, end) = if epoch == 1 {
            (0, INITIAL_ROWS)
        } else {
            let start = INITIAL_ROWS + (epoch - 2) * BATCH_ROWS;
            (start, start + BATCH_ROWS)
        };
        feed_quotes(operator, job, start, end, FULL_ROWS).await?;
        Ok(end)
    }

    fn capture_epoch(
        &mut self,
        operator: &mut StreamJoinOperator,
        pool: &ObservedPool,
        bases: &AtomicUsize,
        epoch: usize,
        end: usize,
    ) -> Result<()> {
        self.phase = format!("epoch_{epoch}_capture");
        pool.begin_phase();
        let snapshot = operator.checkpoint(Epoch::new(epoch as u64).unwrap())?;
        let inventory = &snapshot.inline_metadata["v2_inventory"];
        require(
            inventory["base_epoch"] == 1
                && inventory["deltas"]
                    .as_array()
                    .is_some_and(|deltas| deltas.len() == epoch - 1),
            "4M fixture must capture the initial base followed by four deltas",
        )?;
        self.captures.push(json!({
            "epoch": epoch, "left_rows": end,
            "base_epoch": inventory["base_epoch"],
            "delta_count": inventory["deltas"].as_array().map(Vec::len),
        }));
        self.observations
            .push(observation(&self.phase, pool, operator, bases));
        if epoch == 5 {
            self.owners = snapshot
                .segments
                .iter()
                .map(|(name, segment)| {
                    (
                        name.clone(),
                        Arc::downgrade(&segment.bytes_arc()),
                        hex::encode(Sha256::digest(segment.bytes())),
                    )
                })
                .collect();
            let (alias, evidence) = held_segment(&snapshot, pool.id)?;
            self.alias_name = evidence["name"].as_str().unwrap().to_owned();
            self.alias_digest = evidence["sha256"].as_str().unwrap().to_owned();
            self.held = Some(alias);
        }
        drop(snapshot);
        Ok(())
    }

    async fn capture_history(
        &mut self,
        operator: &mut StreamJoinOperator,
        job: &StreamJobContext,
        pool: &ObservedPool,
        bases: &AtomicUsize,
    ) -> Result<()> {
        for epoch in 1..=5 {
            let end = self.admit_epoch(operator, job, pool, epoch).await?;
            self.observations
                .push(observation(&self.phase, pool, operator, bases));
            self.phase = format!("epoch_{epoch}_prepare");
            pool.begin_phase();
            prepare(operator, job).await?;
            self.observations
                .push(observation(&self.phase, pool, operator, bases));
            require(
                bases.load(Ordering::SeqCst) == 1,
                "only the initial checkpoint may invoke the full-base writer",
            )?;
            self.capture_epoch(operator, pool, bases, epoch, end)?;
        }
        Ok(())
    }

    async fn admit_replacement(
        &mut self,
        operator: &mut StreamJoinOperator,
        job: &StreamJobContext,
        pool: &ObservedPool,
        bases: &AtomicUsize,
    ) -> Result<()> {
        self.phase = "epoch_6_admission".into();
        pool.begin_phase();
        feed_quotes(
            operator,
            job,
            INITIAL_ROWS + 4 * BATCH_ROWS,
            FULL_ROWS,
            FULL_ROWS,
        )
        .await?;
        self.observations
            .push(observation(&self.phase, pool, operator, bases));
        require(
            operator.state.deltas.pending.iter().count() == PENDING_ROWS
                && self
                    .captures
                    .last()
                    .is_some_and(|capture| capture["delta_count"] == 4),
            "4M replacement must start with 32K pending rows and four dirty captured epochs",
        )?;
        self.before_oracle = Some(validate_state(operator, FULL_ROWS)?);
        self.row_owner = Some(Arc::downgrade(&operator.state.left.0));
        Ok(())
    }

    async fn prepare_replacement(
        &mut self,
        operator: &mut StreamJoinOperator,
        job: &StreamJobContext,
        pool: &ObservedPool,
        bases: &AtomicUsize,
    ) -> Result<()> {
        self.phase = "epoch_6_prepare".into();
        pool.begin_phase();
        prepare(operator, job).await?;
        self.observations
            .push(observation(&self.phase, pool, operator, bases));
        require(
            operator.state.deltas.pending.is_empty() && bases.load(Ordering::SeqCst) == 1,
            "automatic replacement must clear pending rows without a second full-base writer",
        )?;
        self.after_oracle = Some(validate_state(operator, FULL_ROWS)?);
        require(
            self.before_oracle == self.after_oracle,
            "4M replacement changed the complete live-state oracle",
        )?;
        Ok(())
    }

    fn validate_historical_owners(&self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        for (name, owner, digest) in &self.owners {
            if name.contains("-payload-") {
                validate_historical_payload(name, owner, digest, snapshot)?;
            } else {
                require(
                    owner.upgrade().is_none(),
                    "replacement must retire the historical index owners",
                )?;
            }
        }
        Ok(())
    }

    fn capture_replacement_snapshot(
        &mut self,
        operator: &mut StreamJoinOperator,
        pool: &ObservedPool,
    ) -> Result<()> {
        self.phase = "epoch_6_capture".into();
        pool.begin_phase();
        let snapshot = operator.checkpoint(Epoch::new(6).unwrap())?;
        let inventory = &snapshot.inline_metadata["v2_inventory"];
        require(
            inventory["base_epoch"] == 5 && inventory["deltas"] == json!([]),
            "4M replacement must capture the epoch-5 anchored base without deltas",
        )?;
        self.validate_historical_owners(&snapshot)?;
        self.replacement_owners = snapshot
            .segments
            .iter()
            .map(|(name, segment)| (name.clone(), Arc::downgrade(&segment.bytes_arc())))
            .collect();
        self.captures.push(json!({
            "epoch": 6, "left_rows": FULL_ROWS,
            "base_epoch": inventory["base_epoch"],
            "delta_count": inventory["deltas"].as_array().map(Vec::len),
            "historical_payloads_shared_and_unchanged": true,
            "historical_indexes_retired": true,
        }));
        self.replacement = Some(snapshot);
        Ok(())
    }

    async fn capture_replacement(
        &mut self,
        operator: &mut StreamJoinOperator,
        job: &StreamJobContext,
        pool: &ObservedPool,
        bases: &AtomicUsize,
    ) -> Result<()> {
        self.admit_replacement(operator, job, pool, bases).await?;
        self.prepare_replacement(operator, job, pool, bases).await?;
        self.capture_replacement_snapshot(operator, pool)
    }

    fn restore_replacement(&self, restore_pool: &Arc<ObservedPool>) -> Result<StreamJoinOperator> {
        let mut restored = fixture_operator(FULL_ROWS)?;
        restored
            .runtime
            .initialize_benchmark_pool(restore_pool.clone())?;
        restored.restore(self.replacement.as_ref().ok_or_else(|| {
            crate::benchmark_internals::invalid("missing replacement snapshot")
        })?)?;
        Ok(restored)
    }

    fn restore(
        &mut self,
        rows_retired: Option<bool>,
        restore_pool: &Arc<ObservedPool>,
    ) -> Result<()> {
        require(
            rows_retired == Some(true),
            "original decoded rows must retire before strict restore",
        )?;
        let restored = self.restore_replacement(restore_pool)?;
        self.restored_row_owner = Some(Arc::downgrade(&restored.state.left.0));
        self.restored_oracle = Some(validate_state(&restored, FULL_ROWS)?);
        require(
            self.restored_oracle == self.before_oracle,
            "strict 4M restore changed the complete live-state oracle",
        )?;
        Ok(())
    }

    fn assert_outcome(
        self,
        pool: &ObservedPool,
        restore_pool: &ObservedPool,
        bases: &AtomicUsize,
        result: &Result<()>,
        settled: &FullReplacementSettlement,
        restore_result: Option<Result<()>>,
    ) {
        let original_error = &self.original_error;
        let restore_error = restore_result.as_ref().and_then(|result| {
            result.as_ref().err().map(|error| {
                json!({"phase": self.phase, "debug": format!("{error:?}"), "display": error.to_string()})
            })
        });
        let restore_observation = restore_pool.snapshot();
        let restored_rows_retired = self
            .restored_row_owner
            .as_ref()
            .map(|owner| owner.upgrade().is_none());
        let alias_unchanged = self
            .held
            .as_ref()
            .is_some_and(|alias| hex::encode(Sha256::digest(alias.bytes())) == self.alias_digest);
        drop(self.replacement);
        let before_alias_release = pool.snapshot();
        let only_alias_alive = self
            .owners
            .iter()
            .all(|(name, owner, _)| owner.upgrade().is_some() == (*name == self.alias_name))
            && self
                .replacement_owners
                .iter()
                .all(|(name, owner)| owner.upgrade().is_some() == (*name == self.alias_name));
        drop(self.held);
        let all_segments_retired = self
            .owners
            .iter()
            .all(|(_, owner, _)| owner.upgrade().is_none())
            && self
                .replacement_owners
                .iter()
                .all(|(_, owner)| owner.upgrade().is_none());
        let final_pool = pool.snapshot();
        let final_restore_pool = restore_pool.snapshot();
        let complete_cleanup = settled.complete_cleanup(
            &final_pool,
            &final_restore_pool,
            all_segments_retired,
            restored_rows_retired,
        );
        let evidence = json!({
            "case": "dirty_replacement_4m_operator_correctness",
            "scope": "forced operator replacement and strict restore; not managed epoch-11 acceptance",
            "left_rows": FULL_ROWS, "right_rows": ENTITIES,
            "initial_rows": INITIAL_ROWS, "pending_rows": PENDING_ROWS,
            "batch_rows": BATCH_ROWS, "pool_limit_bytes": POOL_BYTES,
            "last_phase": self.phase, "original_error": self.original_error, "restore_error": restore_error,
            "observations": self.observations, "captures": self.captures,
            "before_oracle": self.before_oracle, "after_oracle": self.after_oracle,
            "restored_oracle": self.restored_oracle,
            "held_alias_name": self.alias_name, "held_alias_unchanged": alias_unchanged,
            "after_operator_drop": settled.after_operator_drop, "after_drain": settled.after_drain,
            "rows_retired_before_restore": settled.rows_retired,
            "restore_observation": restore_observation, "restored_rows_retired": restored_rows_retired,
            "before_alias_release": before_alias_release, "only_explicit_alias_alive": only_alias_alive,
            "all_segments_retired": all_segments_retired,
            "cleanup_error": settled.cleanup.as_ref().err().map(|error| format!("{error:?}")),
            "final_pool": final_pool, "final_restore_pool": final_restore_pool,
            "owned_cleanup_complete": complete_cleanup,
        });
        eprintln!("DIRTY_REPLACEMENT_4M_EVIDENCE {evidence}");
        assert!(
            complete_cleanup,
            "4M diagnostic must settle before asserting its result"
        );
        assert!(result.is_ok(), "4M replacement failed: {original_error:?}");
        assert!(
            restore_result.is_some_and(|result| result.is_ok()),
            "strict 4M restore failed: {restore_error:?}"
        );
        assert!(alias_unchanged && only_alias_alive && all_segments_retired);
        assert!(before_alias_release.reserved_bytes > 0);
        assert_eq!(bases.load(Ordering::SeqCst), 1);
        assert_eq!(settled.rows_retired, Some(true));
        assert_eq!(restored_rows_retired, Some(true));
        assert_eq!(self.before_oracle, self.after_oracle);
        assert_eq!(self.before_oracle, self.restored_oracle);
    }
}

#[tokio::test]
#[ignore = "explicit 4M-row correctness case; includes strict restore and owned cleanup"]
async fn test_dirty_replacement_4m_fits_explicit_1gib_pool_and_strictly_restores() {
    let pool = Arc::new(ObservedPool::new("dirty-replacement-4m", POOL_BYTES));
    let restore_pool = Arc::new(ObservedPool::new(
        "dirty-replacement-4m-restore",
        POOL_BYTES,
    ));
    let mut operator = fixture_operator(FULL_ROWS).unwrap();
    let job = job();
    let bases = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&bases);
    operator.set_checkpoint_writer_base_test_hook(Arc::new(move |_| {
        counted.fetch_add(1, Ordering::SeqCst);
    }));
    let mut diagnostic = FullReplacementDiagnostic {
        phase: "initialize".to_owned(),
        ..Default::default()
    };
    let result: Result<()> = async {
        operator.runtime.initialize_benchmark_pool(pool.clone())?;
        operator.ingress_progress = progress(None);
        diagnostic
            .capture_history(&mut operator, &job, &pool, &bases)
            .await?;
        diagnostic
            .capture_replacement(&mut operator, &job, &pool, &bases)
            .await
    }
    .await;
    diagnostic
        .observations
        .push(observation(&diagnostic.phase, &pool, &operator, &bases));
    diagnostic.original_error = result.as_ref().err().map(|error| {
        json!({"phase": diagnostic.phase, "debug": format!("{error:?}"), "display": error.to_string()})
    });
    drop(operator);
    let after_operator_drop = pool.snapshot();
    let cleanup = crate::benchmark_internals::settle_job(&job).await;
    drop(job);
    let after_drain = pool.snapshot();
    let rows_retired = diagnostic
        .row_owner
        .as_ref()
        .map(|owner| owner.upgrade().is_none());
    let restore_result = if result.is_ok() && cleanup.is_ok() {
        diagnostic.phase = "strict_restore".into();
        restore_pool.begin_phase();
        Some(diagnostic.restore(rows_retired, &restore_pool))
    } else {
        None
    };
    diagnostic.assert_outcome(
        &pool,
        &restore_pool,
        &bases,
        &result,
        &FullReplacementSettlement {
            cleanup,
            after_operator_drop,
            after_drain,
            rows_retired,
        },
        restore_result,
    );
}
