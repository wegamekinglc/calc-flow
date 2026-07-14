use std::{collections::BTreeMap, sync::Arc};

use chrono::Utc;

use crate::{
    CalcFlowError, Checkpoint, CheckpointStore, ExecutionOptions, ExecutionPlan, Result, RunResult,
    Source, SourceItem,
    pipeline::{PlanLease, PlanTransaction},
};

use super::SinkRouter;

/// Pull-based at-least-once runner for one replayable source.
///
/// A runner owns its source and router and exclusively leases its plan until
/// drop. Direct plan lifecycle calls and other runners are rejected while the
/// lease is active.
pub struct MicroBatchRunner {
    plan: Arc<ExecutionPlan>,
    lease: PlanLease,
    source: Box<dyn Source>,
    sinks: SinkRouter,
    checkpoints: Arc<dyn CheckpointStore>,
    checkpoint_every: u64,
    recovered: bool,
    delivered: u64,
    current: Option<SourceItem>,
    pending_checkpoint: Option<Checkpoint>,
    eof: bool,
    poisoned: Option<String>,
}

impl MicroBatchRunner {
    /// Constructs a runner for a plan with exactly one external input.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for zero cadence or a plan
    /// without exactly one external input, or [`CalcFlowError::PlanLeased`]
    /// when another runner already owns the plan.
    pub fn new(
        plan: Arc<ExecutionPlan>,
        source: Box<dyn Source>,
        sinks: SinkRouter,
        checkpoints: Arc<dyn CheckpointStore>,
        checkpoint_every: u64,
    ) -> Result<Self> {
        plan.single_external_input()?;
        if checkpoint_every == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "checkpoint_every".into(),
                message: "must be greater than zero".into(),
            });
        }
        let lease = plan.acquire_lease()?;
        Ok(Self {
            plan,
            lease,
            source,
            sinks,
            checkpoints,
            checkpoint_every,
            recovered: false,
            delivered: 0,
            current: None,
            pending_checkpoint: None,
            eof: false,
            poisoned: None,
        })
    }

    /// Recovers once, then executes, delivers, and conditionally checkpoints
    /// one source item. A failed item remains buffered for an at-least-once
    /// retry on the next call.
    ///
    /// # Errors
    ///
    /// Returns source, execution, sink, checkpoint, or rollback errors.
    pub async fn next(&mut self) -> Result<Option<RunResult>> {
        self.ensure_healthy()?;
        self.recover_abandoned().await?;
        self.recover_once().await?;
        if self.current.is_none() && !self.eof {
            self.current = self.source.next().await?;
            self.eof = self.current.is_none();
        }
        let Some(item) = self.current.clone() else {
            self.flush_pending().await?;
            return Ok(None);
        };
        let next_delivered =
            self.delivered
                .checked_add(1)
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "micro-batch delivery counter overflowed".into(),
                })?;

        let plan = Arc::clone(&self.plan);
        let transaction = plan.leased_transaction(&self.lease).await?;
        let input_name = plan.single_external_input()?.to_owned();
        let inputs = BTreeMap::from([(input_name, item.batch.clone())]);
        transaction.validate_inputs(&inputs)?;
        let before = transaction.snapshot().await?;
        let checkpoint_before = self.checkpoints.load(plan.name()).await?;
        let operation = transaction.begin_runner_rollback(before, checkpoint_before)?;
        let result = match transaction
            .execute_validated(inputs, ExecutionOptions::default())
            .await
        {
            Ok(result) => result,
            Err(error) => {
                return Err(self
                    .rollback_operation(&transaction, operation, error)
                    .await);
            }
        };
        if let Err(error) = self.sinks.write_all(&result).await {
            return Err(self
                .rollback_operation(&transaction, operation, error)
                .await);
        }

        let state = match transaction.snapshot().await {
            Ok(state) => state,
            Err(error) => {
                return Err(self
                    .rollback_operation(&transaction, operation, error)
                    .await);
            }
        };
        let checkpoint = match Checkpoint::new(
            plan.name(),
            plan.fingerprint(),
            item.cursor.clone(),
            item.sequence,
            state,
            Utc::now(),
        ) {
            Ok(checkpoint) => checkpoint,
            Err(error) => {
                return Err(self
                    .rollback_operation(&transaction, operation, error)
                    .await);
            }
        };
        if next_delivered % self.checkpoint_every == 0 {
            transaction.mark_store_mutation(operation)?;
            if let Err(error) = self.checkpoints.save(&checkpoint).await {
                return Err(self
                    .rollback_operation(&transaction, operation, error)
                    .await);
            }
            self.pending_checkpoint = None;
        } else {
            self.pending_checkpoint = Some(checkpoint);
        }
        transaction.commit_operation(operation)?;
        self.delivered = next_delivered;
        self.current = None;
        Ok(Some(result))
    }

    /// Clears durable and in-memory runner state and starts a new recovery
    /// lifecycle on the next call.
    ///
    /// # Errors
    ///
    /// Returns reset, store, or rollback errors.
    pub async fn reset(&mut self) -> Result<()> {
        let plan = Arc::clone(&self.plan);
        let transaction = plan.leased_transaction(&self.lease).await?;
        if transaction
            .recover_in_flight(Some(self.checkpoints.as_ref()))
            .await
            .is_err()
        {
            let previous_checkpoint = self.checkpoints.load(plan.name()).await?;
            let before = transaction.snapshot().await?;
            let operation = transaction.replace_for_forced_reset(before, previous_checkpoint)?;
            if let Err(error) = transaction.reset().await {
                return Err(self
                    .rollback_operation(&transaction, operation, error)
                    .await);
            }
            transaction.mark_store_mutation(operation)?;
            if let Err(error) = self.checkpoints.delete(plan.name()).await {
                return Err(self
                    .rollback_operation(&transaction, operation, error)
                    .await);
            }
            transaction.commit_operation(operation)?;
            self.finish_reset();
            return Ok(());
        }
        let previous_checkpoint = self.checkpoints.load(plan.name()).await?;
        let before = transaction.snapshot().await?;
        let operation = transaction.begin_runner_rollback(before, previous_checkpoint)?;
        if let Err(error) = transaction.reset().await {
            return Err(self
                .rollback_operation(&transaction, operation, error)
                .await);
        }
        transaction.mark_store_mutation(operation)?;
        if let Err(error) = self.checkpoints.delete(plan.name()).await {
            return Err(self
                .rollback_operation(&transaction, operation, error)
                .await);
        }
        transaction.commit_operation(operation)?;
        self.finish_reset();
        Ok(())
    }

    fn finish_reset(&mut self) {
        self.recovered = false;
        self.delivered = 0;
        self.current = None;
        self.pending_checkpoint = None;
        self.eof = false;
        self.poisoned = None;
    }

    /// Captures plan state through this runner's exclusive lease.
    ///
    /// # Errors
    ///
    /// Returns an operator lifecycle error if state cannot be captured.
    pub async fn plan_snapshot(&self) -> Result<BTreeMap<String, serde_json::Value>> {
        let plan = Arc::clone(&self.plan);
        let transaction = plan.leased_transaction(&self.lease).await?;
        transaction
            .recover_in_flight(Some(self.checkpoints.as_ref()))
            .await?;
        transaction.snapshot().await
    }

    async fn recover_once(&mut self) -> Result<()> {
        if self.recovered {
            return Ok(());
        }
        let plan = Arc::clone(&self.plan);
        let transaction = plan.leased_transaction(&self.lease).await?;
        let checkpoint = self.checkpoints.load(plan.name()).await?;
        if let Some(checkpoint) = checkpoint {
            validate_checkpoint(&checkpoint, &plan)?;
            transaction.validate_state(&checkpoint.state)?;
            let before = transaction.snapshot().await?;
            let operation = transaction.begin_runner_rollback(before, Some(checkpoint.clone()))?;
            if let Err(error) = transaction.restore(&checkpoint.state).await {
                return Err(self
                    .rollback_operation(&transaction, operation, error)
                    .await);
            }
            if let Err(error) = self.source.open(checkpoint.source_cursor.clone()).await {
                return Err(self
                    .rollback_operation(&transaction, operation, error)
                    .await);
            }
            transaction.commit_operation(operation)?;
        } else {
            self.source.open(None).await?;
        }
        self.recovered = true;
        Ok(())
    }

    async fn flush_pending(&mut self) -> Result<()> {
        let Some(checkpoint) = &self.pending_checkpoint else {
            return Ok(());
        };
        let plan = Arc::clone(&self.plan);
        let transaction = plan.leased_transaction(&self.lease).await?;
        let operation = transaction.begin_checkpoint_commit(checkpoint.clone())?;
        self.checkpoints.save(checkpoint).await?;
        transaction.commit_operation(operation)?;
        self.pending_checkpoint = None;
        Ok(())
    }

    fn ensure_healthy(&self) -> Result<()> {
        match &self.poisoned {
            Some(reason) => Err(CalcFlowError::Internal {
                message: format!(
                    "micro-batch runner is poisoned after an incomplete rollback: {reason}; call reset before reuse"
                ),
            }),
            None => Ok(()),
        }
    }

    async fn rollback_operation(
        &mut self,
        transaction: &PlanTransaction<'_>,
        operation: crate::pipeline::OperationToken,
        original: CalcFlowError,
    ) -> CalcFlowError {
        let outcome = transaction
            .rollback_error(operation, original, Some(self.checkpoints.as_ref()))
            .await;
        if outcome.recovery_failed {
            self.poisoned = Some(outcome.error.to_string());
        }
        outcome.error
    }

    async fn recover_abandoned(&mut self) -> Result<()> {
        let plan = Arc::clone(&self.plan);
        let transaction = plan.leased_transaction(&self.lease).await?;
        if let Err(error) = transaction
            .recover_in_flight(Some(self.checkpoints.as_ref()))
            .await
        {
            self.poisoned = Some(error.to_string());
            return Err(error);
        }
        Ok(())
    }
}

pub(super) fn validate_checkpoint(checkpoint: &Checkpoint, plan: &ExecutionPlan) -> Result<()> {
    if checkpoint.pipeline_name != plan.name() {
        return Err(CalcFlowError::CheckpointMismatch {
            message: format!(
                "pipeline name {:?} does not match {:?}",
                checkpoint.pipeline_name,
                plan.name()
            ),
        });
    }
    if checkpoint.pipeline_fingerprint != plan.fingerprint() {
        return Err(CalcFlowError::CheckpointMismatch {
            message: "pipeline fingerprint does not match the compiled plan".into(),
        });
    }
    Ok(())
}
