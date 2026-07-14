use std::{collections::BTreeMap, fmt::Write as _, sync::Arc};

use chrono::Utc;

use crate::{
    CalcFlowError, Checkpoint, CheckpointStore, ExecutionOptions, ExecutionPlan, Result, RunResult,
    Source, SourceItem, pipeline::PlanTransaction,
};

use super::SinkRouter;

/// Pull-based at-least-once runner for one replayable source.
///
/// A runner owns its source and router. Other owners of the same plan are
/// serialized for the full state, delivery, and checkpoint transaction.
pub struct MicroBatchRunner {
    plan: Arc<ExecutionPlan>,
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
    /// without exactly one external input.
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
        Ok(Self {
            plan,
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
        let transaction = plan.transaction().await;
        let before = transaction.snapshot().await?;
        let input_name = plan.single_external_input()?.to_owned();
        let result = match transaction
            .execute(
                BTreeMap::from([(input_name, item.batch.clone())]),
                ExecutionOptions::default(),
            )
            .await
        {
            Ok(result) => result,
            Err(error) => {
                return Err(self.rollback_operation(&transaction, &before, error).await);
            }
        };
        if let Err(error) = self.sinks.write_all(&result).await {
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
            item.cursor.clone(),
            item.sequence,
            state,
            Utc::now(),
        ) {
            Ok(checkpoint) => checkpoint,
            Err(error) => {
                return Err(self.rollback_operation(&transaction, &before, error).await);
            }
        };
        if next_delivered % self.checkpoint_every == 0 {
            if let Err(error) = self.checkpoints.save(&checkpoint).await {
                return Err(self.rollback_operation(&transaction, &before, error).await);
            }
            self.pending_checkpoint = None;
        } else {
            self.pending_checkpoint = Some(checkpoint);
        }
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
        let transaction = plan.transaction().await;
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
        self.recovered = false;
        self.delivered = 0;
        self.current = None;
        self.pending_checkpoint = None;
        self.eof = false;
        self.poisoned = None;
        Ok(())
    }

    async fn recover_once(&mut self) -> Result<()> {
        if self.recovered {
            return Ok(());
        }
        let plan = Arc::clone(&self.plan);
        let transaction = plan.transaction().await;
        let checkpoint = self.checkpoints.load(plan.name()).await?;
        if let Some(checkpoint) = checkpoint {
            validate_checkpoint(&checkpoint, &plan)?;
            let before = transaction.snapshot().await?;
            if let Err(error) = transaction.restore(&checkpoint.state).await {
                return Err(self.rollback_operation(&transaction, &before, error).await);
            }
            if let Err(error) = self.source.open(checkpoint.source_cursor.clone()).await {
                return Err(self.rollback_operation(&transaction, &before, error).await);
            }
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
        let _transaction = plan.transaction().await;
        self.checkpoints.save(checkpoint).await?;
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

pub(super) struct RollbackResult {
    pub(super) error: CalcFlowError,
    pub(super) poison_reason: Option<String>,
}

pub(super) async fn rollback_error(
    transaction: &PlanTransaction<'_>,
    before: &BTreeMap<String, serde_json::Value>,
    original: CalcFlowError,
) -> RollbackResult {
    match transaction.restore(before).await {
        Ok(()) => RollbackResult {
            error: original,
            poison_reason: None,
        },
        Err(rollback) => {
            let message = format!(
                "runner operation failed with {original}; rollback also failed with {rollback}"
            );
            RollbackResult {
                error: CalcFlowError::Internal {
                    message: message.clone(),
                },
                poison_reason: Some(message),
            }
        }
    }
}

pub(super) fn compensated_reset_error(
    original: CalcFlowError,
    plan_rollback: Option<CalcFlowError>,
    checkpoint_rollback: Option<CalcFlowError>,
) -> RollbackResult {
    if plan_rollback.is_none() && checkpoint_rollback.is_none() {
        return RollbackResult {
            error: original,
            poison_reason: None,
        };
    }
    let mut message = format!("runner reset failed with {original}");
    if let Some(rollback) = plan_rollback {
        write!(&mut message, "; plan rollback also failed with {rollback}")
            .expect("writing to a String cannot fail");
    }
    if let Some(rollback) = checkpoint_rollback {
        write!(
            &mut message,
            "; checkpoint compensation also failed with {rollback}"
        )
        .expect("writing to a String cannot fail");
    }
    RollbackResult {
        error: CalcFlowError::Internal {
            message: message.clone(),
        },
        poison_reason: Some(message),
    }
}
