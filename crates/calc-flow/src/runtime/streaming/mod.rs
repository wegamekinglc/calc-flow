//! The v2 push runner plus the v3 stream building blocks.
//!
//! `StreamingRunner` below is the v2 push-based runner over
//! [`crate::BatchExecutionPlan`]; plan task M2.4 replaces it with the
//! source-driven continuous runner. `StreamJobContext` and `StreamMessage`
//! are the first v3 stream types.

mod context;
mod message;

pub use context::StreamJobContext;
pub use message::{StreamMessage, StreamMessageKind};

use std::{collections::BTreeMap, sync::Arc};

use chrono::Utc;

use crate::{
    Batch, BatchExecutionPlan, CalcFlowError, Checkpoint, CheckpointStore, ExecutionOptions,
    Result, RunResult,
    pipeline::{PlanLease, PlanTransaction},
};

use super::{SinkRouter, micro_batch::validate_checkpoint};

/// Push-based at-least-once runner for already formed batches.
pub struct StreamingRunner {
    plan: Arc<BatchExecutionPlan>,
    lease: PlanLease,
    checkpoints: Arc<dyn CheckpointStore>,
    recovered: bool,
    next_sequence: Option<u64>,
    poisoned: Option<String>,
}

impl StreamingRunner {
    /// Constructs a runner for a plan with exactly one external input.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] unless the plan has exactly
    /// one external input, or [`CalcFlowError::PlanLeased`] when another
    /// runner already owns the plan.
    ///
    /// A replacement runner used after abandonment must receive the same
    /// logical [`CheckpointStore`] so interrupted durable mutations are
    /// recovered against the store that observed them.
    pub fn new(
        plan: Arc<BatchExecutionPlan>,
        checkpoints: Arc<dyn CheckpointStore>,
    ) -> Result<Self> {
        plan.single_external_input()?;
        let lease = plan.acquire_lease()?;
        Ok(Self {
            plan,
            lease,
            checkpoints,
            recovered: false,
            next_sequence: Some(0),
            poisoned: None,
        })
    }

    /// Recovers once, executes and delivers one batch, then checkpoints it.
    ///
    /// Push-mode checkpoints do not own a replay cursor: `source_cursor` is
    /// always `None`. This method consumes the supplied batch. The caller must
    /// retain a clone or reconstruct it, then submit that value again after a
    /// failed step to obtain at-least-once delivery.
    ///
    /// # Errors
    ///
    /// Returns recovery, execution, sink, checkpoint, or rollback errors.
    pub async fn step(&mut self, batch: Batch, sinks: &mut SinkRouter) -> Result<RunResult> {
        self.ensure_healthy()?;
        self.recover_abandoned().await?;
        self.recover_once().await?;
        let sequence = self.next_sequence.ok_or_else(|| CalcFlowError::Internal {
            message: "streaming sequence is exhausted".into(),
        })?;
        let plan = Arc::clone(&self.plan);
        let transaction = plan.leased_transaction(&self.lease).await?;
        let input_name = plan.single_external_input()?.to_owned();
        let inputs = BTreeMap::from([(input_name, batch)]);
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
        if let Err(error) = sinks.write_all(&result).await {
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
            None,
            sequence,
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
        transaction.mark_store_mutation(operation)?;
        if let Err(error) = self.checkpoints.save(&checkpoint).await {
            return Err(self
                .rollback_operation(&transaction, operation, error)
                .await);
        }
        transaction.commit_operation(operation)?;
        self.next_sequence = sequence.checked_add(1);
        Ok(result)
    }

    /// Deletes the durable checkpoint and resets plan and sequence state.
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
        self.next_sequence = Some(0);
        self.recovered = true;
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
        if let Some(checkpoint) = self.checkpoints.load(plan.name()).await? {
            validate_checkpoint(&checkpoint, &plan)?;
            transaction.validate_state(&checkpoint.state)?;
            let before = transaction.snapshot().await?;
            let operation = transaction.begin_runner_rollback(before, Some(checkpoint.clone()))?;
            if let Err(error) = transaction.restore(&checkpoint.state).await {
                return Err(self
                    .rollback_operation(&transaction, operation, error)
                    .await);
            }
            transaction.commit_operation(operation)?;
            self.next_sequence = checkpoint.sequence.checked_add(1);
        }
        self.recovered = true;
        Ok(())
    }

    fn ensure_healthy(&self) -> Result<()> {
        match &self.poisoned {
            Some(reason) => Err(CalcFlowError::Internal {
                message: format!(
                    "streaming runner is poisoned after an incomplete rollback: {reason}; call reset before reuse"
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
