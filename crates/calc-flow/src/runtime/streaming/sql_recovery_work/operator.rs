//! Concrete operator dispatch stays outside the job-owned work lifecycle.
use datafusion::execution::memory_pool::MemoryReservation;

use crate::{
    CalcFlowError, OperatorStateSnapshot, Result, SqlOperator,
    operator::{PreparedSqlCheckpoint, PreparedSqlRestore},
    pipeline::CompiledStreamOperator,
};

pub(crate) enum CheckpointWorkOperator {
    Sql(SqlOperator),
}

impl CheckpointWorkOperator {
    pub(crate) fn handles(operator: &CompiledStreamOperator) -> bool {
        matches!(operator, CompiledStreamOperator::Sql(_))
    }

    pub(crate) fn from_compiled(operator: CompiledStreamOperator) -> Self {
        match operator {
            CompiledStreamOperator::Sql(operator) => Self::Sql(operator),
            _ => unreachable!("checkpoint work routing checked the operator"),
        }
    }

    pub(crate) fn into_compiled(self) -> CompiledStreamOperator {
        match self {
            Self::Sql(operator) => CompiledStreamOperator::Sql(operator),
        }
    }

    pub(super) fn reserve_restore(
        &mut self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<MemoryReservation> {
        match self {
            Self::Sql(operator) => operator.reserve_recovery_envelope(snapshot),
        }
    }

    pub(crate) fn needs_checkpoint(operator: &CompiledStreamOperator) -> bool {
        matches!(operator, CompiledStreamOperator::Sql(operator)
            if operator.has_stream_aggregate() && operator.stream_runtime_initialized())
    }

    pub(crate) fn reserve_checkpoint(
        operator: &mut CompiledStreamOperator,
        node_len: usize,
    ) -> Result<MemoryReservation> {
        match operator {
            CompiledStreamOperator::Sql(operator) => operator.reserve_checkpoint_envelope(node_len),
            _ => unreachable!("checkpoint work routing checked the operator"),
        }
    }

    pub(super) fn prepare_restore(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedCheckpointWork> {
        match self {
            Self::Sql(operator) => Ok(PreparedCheckpointWork::SqlRestore(Box::new(
                operator.prepare_restore(snapshot, check)?,
            ))),
        }
    }

    pub(super) fn prepare_checkpoint(
        &mut self,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedCheckpointWork> {
        match self {
            Self::Sql(operator) => Ok(PreparedCheckpointWork::SqlCapture(
                operator.prepare_checkpoint_work(check)?,
            )),
        }
    }
}

pub(crate) enum PreparedCheckpointWork {
    SqlRestore(Box<PreparedSqlRestore>),
    SqlCapture(PreparedSqlCheckpoint),
}

impl PreparedCheckpointWork {
    pub(crate) fn install_restore(self, target: &mut CompiledStreamOperator) -> Result<()> {
        match (self, target) {
            (Self::SqlRestore(prepared), CompiledStreamOperator::Sql(operator)) => {
                operator.install_restore(*prepared);
                Ok(())
            }
            _ => Err(wrong_work_operation()),
        }
    }

    pub(crate) fn install_checkpoint(
        self,
        target: &mut CompiledStreamOperator,
    ) -> Result<OperatorStateSnapshot> {
        match (self, target) {
            (Self::SqlCapture(prepared), CompiledStreamOperator::Sql(operator)) => {
                let snapshot = prepared.snapshot()?;
                operator.install_checkpoint_work(prepared);
                Ok(snapshot)
            }
            _ => Err(wrong_work_operation()),
        }
    }
}

fn wrong_work_operation() -> CalcFlowError {
    CalcFlowError::Internal {
        message: "SQL work returned a different checkpoint operation".into(),
    }
}
