use std::{collections::BTreeMap, sync::Arc};

use chrono::Utc;

use crate::{
    CalcFlowError, Checkpoint, CheckpointStore, ExecutionOptions, ExecutionPlan, Result, RunResult,
    Source, SourceItem,
};

use super::SinkRouter;

/// Pull-based at-least-once runner for one replayable source.
///
/// A runner owns its source and router. Callers must not execute or mutate the
/// same plan through another owner while `next` or `reset` is in progress.
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

        let before = self.plan.snapshot().await?;
        let input_name = self.plan.single_external_input()?.to_owned();
        let result = match self
            .plan
            .execute(
                BTreeMap::from([(input_name, item.batch.clone())]),
                ExecutionOptions::default(),
            )
            .await
        {
            Ok(result) => result,
            Err(error) => return Err(rollback_error(&self.plan, &before, error).await),
        };
        if let Err(error) = self.sinks.write_all(&result).await {
            return Err(rollback_error(&self.plan, &before, error).await);
        }

        let state = match self.plan.snapshot().await {
            Ok(state) => state,
            Err(error) => return Err(rollback_error(&self.plan, &before, error).await),
        };
        let checkpoint = match Checkpoint::new(
            self.plan.name(),
            self.plan.fingerprint(),
            item.cursor.clone(),
            item.sequence,
            state,
            Utc::now(),
        ) {
            Ok(checkpoint) => checkpoint,
            Err(error) => return Err(rollback_error(&self.plan, &before, error).await),
        };
        if next_delivered % self.checkpoint_every == 0 {
            if let Err(error) = self.checkpoints.save(&checkpoint).await {
                return Err(rollback_error(&self.plan, &before, error).await);
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
        let before = self.plan.snapshot().await?;
        if let Err(error) = self.plan.reset().await {
            return Err(rollback_error(&self.plan, &before, error).await);
        }
        if let Err(error) = self.checkpoints.delete(self.plan.name()).await {
            return Err(rollback_error(&self.plan, &before, error).await);
        }
        self.recovered = false;
        self.delivered = 0;
        self.current = None;
        self.pending_checkpoint = None;
        self.eof = false;
        Ok(())
    }

    async fn recover_once(&mut self) -> Result<()> {
        if self.recovered {
            return Ok(());
        }
        let checkpoint = self.checkpoints.load(self.plan.name()).await?;
        if let Some(checkpoint) = checkpoint {
            validate_checkpoint(&checkpoint, &self.plan)?;
            let before = self.plan.snapshot().await?;
            if let Err(error) = self.plan.restore(&checkpoint.state).await {
                return Err(rollback_error(&self.plan, &before, error).await);
            }
            if let Err(error) = self.source.open(checkpoint.source_cursor.clone()).await {
                return Err(rollback_error(&self.plan, &before, error).await);
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
        self.checkpoints.save(checkpoint).await?;
        self.pending_checkpoint = None;
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

pub(super) async fn rollback_error(
    plan: &ExecutionPlan,
    before: &BTreeMap<String, serde_json::Value>,
    original: CalcFlowError,
) -> CalcFlowError {
    match plan.restore(before).await {
        Ok(()) => original,
        Err(rollback) => CalcFlowError::Internal {
            message: format!(
                "runner operation failed with {original}; rollback also failed with {rollback}"
            ),
        },
    }
}
