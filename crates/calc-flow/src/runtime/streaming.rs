use std::{collections::BTreeMap, sync::Arc};

use chrono::Utc;

use crate::{
    Batch, CalcFlowError, Checkpoint, CheckpointStore, ExecutionOptions, ExecutionPlan, Result,
    RunResult,
    pipeline::{PlanLease, PlanTransaction},
};

use super::{
    SinkRouter,
    micro_batch::{compensated_reset_error, rollback_error, validate_checkpoint},
};

/// Push-based at-least-once runner for already formed batches.
pub struct StreamingRunner {
    plan: Arc<ExecutionPlan>,
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
    pub fn new(plan: Arc<ExecutionPlan>, checkpoints: Arc<dyn CheckpointStore>) -> Result<Self> {
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
        let result = match transaction
            .execute_validated(inputs, ExecutionOptions::default())
            .await
        {
            Ok(result) => result,
            Err(error) => {
                return Err(self.rollback_operation(&transaction, &before, error).await);
            }
        };
        if let Err(error) = sinks.write_all(&result).await {
            return Err(self.rollback_operation(&transaction, &before, error).await);
        }
        let state = match transaction.snapshot().await {
            Ok(state) => state,
            Err(error) => {
                return Err(self.rollback_operation(&transaction, &before, error).await);
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
                return Err(self.rollback_operation(&transaction, &before, error).await);
            }
        };
        if let Err(error) = self.checkpoints.save(&checkpoint).await {
            return Err(self.rollback_operation(&transaction, &before, error).await);
        }
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
        let previous_checkpoint = self.checkpoints.load(plan.name()).await?;
        let before = transaction.snapshot().await?;
        if let Err(error) = transaction.reset().await {
            return Err(self.rollback_operation(&transaction, &before, error).await);
        }
        if let Err(error) = self.checkpoints.delete(plan.name()).await {
            let plan_rollback = transaction.restore(&before).await.err();
            let checkpoint_rollback = match &previous_checkpoint {
                Some(checkpoint) => self.checkpoints.save(checkpoint).await.err(),
                None => None,
            };
            let result = compensated_reset_error(error, plan_rollback, checkpoint_rollback);
            if let Some(reason) = &result.poison_reason {
                self.poisoned = Some(reason.clone());
            }
            return Err(result.error);
        }
        self.next_sequence = Some(0);
        self.recovered = true;
        self.poisoned = None;
        Ok(())
    }

    /// Captures plan state through this runner's exclusive lease.
    ///
    /// # Errors
    ///
    /// Returns an operator lifecycle error if state cannot be captured.
    pub async fn plan_snapshot(&self) -> Result<BTreeMap<String, serde_json::Value>> {
        let plan = Arc::clone(&self.plan);
        plan.leased_transaction(&self.lease).await?.snapshot().await
    }

    async fn recover_once(&mut self) -> Result<()> {
        if self.recovered {
            return Ok(());
        }
        let plan = Arc::clone(&self.plan);
        let transaction = plan.leased_transaction(&self.lease).await?;
        if let Some(checkpoint) = self.checkpoints.load(plan.name()).await? {
            validate_checkpoint(&checkpoint, &plan)?;
            let before = transaction.snapshot().await?;
            if let Err(error) = transaction.restore(&checkpoint.state).await {
                return Err(self.rollback_operation(&transaction, &before, error).await);
            }
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
        before: &BTreeMap<String, serde_json::Value>,
        original: CalcFlowError,
    ) -> CalcFlowError {
        let result = rollback_error(transaction, before, original).await;
        if let Some(reason) = &result.poison_reason {
            self.poisoned = Some(reason.clone());
        }
        result.error
    }
}
