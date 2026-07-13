use std::{collections::BTreeMap, sync::Arc};

use chrono::Utc;

use crate::{
    Batch, CalcFlowError, Checkpoint, CheckpointStore, ExecutionOptions, ExecutionPlan, Result,
    RunResult,
};

use super::{SinkRouter, micro_batch::rollback_error, micro_batch::validate_checkpoint};

/// Push-based at-least-once runner for already formed batches.
pub struct StreamingRunner {
    plan: Arc<ExecutionPlan>,
    checkpoints: Arc<dyn CheckpointStore>,
    recovered: bool,
    sequence: u64,
}

impl StreamingRunner {
    /// Constructs a runner for a plan with exactly one external input.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] unless the plan has exactly
    /// one external input.
    pub fn new(plan: Arc<ExecutionPlan>, checkpoints: Arc<dyn CheckpointStore>) -> Result<Self> {
        plan.single_external_input()?;
        Ok(Self {
            plan,
            checkpoints,
            recovered: false,
            sequence: 0,
        })
    }

    /// Recovers once, executes and delivers one batch, then checkpoints it.
    ///
    /// # Errors
    ///
    /// Returns recovery, execution, sink, checkpoint, or rollback errors.
    pub async fn step(&mut self, batch: Batch, sinks: &mut SinkRouter) -> Result<RunResult> {
        self.recover_once().await?;
        let next_sequence =
            self.sequence
                .checked_add(1)
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: "streaming sequence cannot be advanced".into(),
                })?;
        let before = self.plan.snapshot().await?;
        let input_name = self.plan.single_external_input()?.to_owned();
        let result = match self
            .plan
            .execute(
                BTreeMap::from([(input_name, batch)]),
                ExecutionOptions::default(),
            )
            .await
        {
            Ok(result) => result,
            Err(error) => return Err(rollback_error(&self.plan, &before, error).await),
        };
        if let Err(error) = sinks.write_all(&result).await {
            return Err(rollback_error(&self.plan, &before, error).await);
        }
        let state = match self.plan.snapshot().await {
            Ok(state) => state,
            Err(error) => return Err(rollback_error(&self.plan, &before, error).await),
        };
        let checkpoint = match Checkpoint::new(
            self.plan.name(),
            self.plan.fingerprint(),
            None,
            self.sequence,
            state,
            Utc::now(),
        ) {
            Ok(checkpoint) => checkpoint,
            Err(error) => return Err(rollback_error(&self.plan, &before, error).await),
        };
        if let Err(error) = self.checkpoints.save(&checkpoint).await {
            return Err(rollback_error(&self.plan, &before, error).await);
        }
        self.sequence = next_sequence;
        Ok(result)
    }

    /// Deletes the durable checkpoint and resets plan and sequence state.
    ///
    /// # Errors
    ///
    /// Returns reset, store, or rollback errors.
    pub async fn reset(&mut self) -> Result<()> {
        let before = self.plan.snapshot().await?;
        if let Err(error) = self.plan.reset().await {
            return Err(rollback_error(&self.plan, &before, error).await);
        }
        if let Err(error) = self.checkpoints.delete(self.plan.name()).await {
            return Err(rollback_error(&self.plan, &before, error).await);
        }
        self.sequence = 0;
        self.recovered = true;
        Ok(())
    }

    async fn recover_once(&mut self) -> Result<()> {
        if self.recovered {
            return Ok(());
        }
        if let Some(checkpoint) = self.checkpoints.load(self.plan.name()).await? {
            validate_checkpoint(&checkpoint, &self.plan)?;
            let next_sequence = checkpoint.sequence.checked_add(1).ok_or_else(|| {
                CalcFlowError::CheckpointMismatch {
                    message: "checkpoint sequence cannot be advanced".into(),
                }
            })?;
            let before = self.plan.snapshot().await?;
            if let Err(error) = self.plan.restore(&checkpoint.state).await {
                return Err(rollback_error(&self.plan, &before, error).await);
            }
            self.sequence = next_sequence;
        }
        self.recovered = true;
        Ok(())
    }
}
