//! Concrete operator dispatch stays outside the job-owned work lifecycle.
#[cfg(test)]
mod tests;
use datafusion::execution::memory_pool::MemoryReservation;

use crate::{
    CalcFlowError, OperatorStateSnapshot, Result, SnapshotSqlOperator, SqlOperator,
    operator::{
        PreparedSnapshotCapture, PreparedSnapshotRestore, PreparedSqlCheckpoint, PreparedSqlRestore,
    },
    pipeline::CompiledStreamOperator,
};

pub(crate) enum CheckpointWorkOperator {
    Sql(Box<SqlOperator>),
    SnapshotSql(Box<SnapshotSqlOperator>),
}

impl CheckpointWorkOperator {
    pub(crate) fn handles(operator: &CompiledStreamOperator) -> bool {
        matches!(
            operator,
            CompiledStreamOperator::Sql(_) | CompiledStreamOperator::SnapshotSql(_)
        )
    }

    pub(crate) fn from_compiled(operator: CompiledStreamOperator) -> Self {
        match operator {
            CompiledStreamOperator::Sql(operator) => Self::Sql(Box::new(operator)),
            CompiledStreamOperator::SnapshotSql(operator) => Self::SnapshotSql(Box::new(operator)),
            _ => unreachable!("checkpoint work routing checked the operator"),
        }
    }

    pub(crate) fn into_compiled(self) -> CompiledStreamOperator {
        match self {
            Self::Sql(operator) => CompiledStreamOperator::Sql(*operator),
            Self::SnapshotSql(operator) => CompiledStreamOperator::SnapshotSql(*operator),
        }
    }

    pub(super) fn reserve_restore(
        &mut self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<MemoryReservation> {
        match self {
            Self::Sql(operator) => operator.reserve_recovery_envelope(snapshot),
            Self::SnapshotSql(operator) => operator.reserve_recovery_envelope(snapshot),
        }
    }

    pub(crate) fn needs_checkpoint(operator: &CompiledStreamOperator) -> bool {
        matches!(operator, CompiledStreamOperator::Sql(operator)
            if operator.has_stream_aggregate() && operator.stream_runtime_initialized())
            || matches!(operator, CompiledStreamOperator::SnapshotSql(_))
    }

    pub(crate) fn reserve_checkpoint(
        operator: &mut CompiledStreamOperator,
        node_len: usize,
    ) -> Result<MemoryReservation> {
        match operator {
            CompiledStreamOperator::Sql(operator) => operator.reserve_checkpoint_envelope(node_len),
            CompiledStreamOperator::SnapshotSql(operator) => {
                operator.reserve_checkpoint_envelope(node_len)
            }
            _ => unreachable!("checkpoint work routing checked the operator"),
        }
    }

    pub(super) fn prepare_restore(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedCheckpointWork> {
        match self {
            Self::SnapshotSql(operator) => Ok(PreparedCheckpointWork::SnapshotRestore(Box::new(
                operator.prepare_restore(snapshot, check)?,
            ))),
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
            Self::SnapshotSql(operator) => Ok(PreparedCheckpointWork::SnapshotCapture(
                operator.prepare_checkpoint_work(check)?,
            )),
            Self::Sql(operator) => Ok(PreparedCheckpointWork::SqlCapture(
                operator.prepare_checkpoint_work(check)?,
            )),
        }
    }
}

pub(crate) enum PreparedCheckpointWork {
    SnapshotRestore(Box<PreparedSnapshotRestore>),
    SnapshotCapture(PreparedSnapshotCapture),
    SqlRestore(Box<PreparedSqlRestore>),
    SqlCapture(PreparedSqlCheckpoint),
}

impl PreparedCheckpointWork {
    pub(crate) fn install_restore(self, target: &mut CompiledStreamOperator) -> Result<()> {
        match (self, target) {
            (Self::SnapshotRestore(prepared), CompiledStreamOperator::SnapshotSql(operator)) => {
                operator.install_restore(*prepared);
                Ok(())
            }
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
            (Self::SnapshotCapture(prepared), CompiledStreamOperator::SnapshotSql(operator)) => {
                let snapshot = prepared.snapshot.clone();
                operator.install_checkpoint_work(prepared);
                Ok(snapshot)
            }
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
