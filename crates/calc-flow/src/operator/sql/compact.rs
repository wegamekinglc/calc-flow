mod capture;
mod control;
mod identity;
mod restore;
mod storage;

pub(super) use capture::{CompactCapture, prepare as prepare_capture};
pub(super) use restore::{RestoredCompact, prepare as prepare_restore};
pub(super) use storage::CompactSqlState;

use super::{PreparedRetention, SqlOperator, StreamCollector, StreamOperatorContext, incremental};
use crate::{OperatorStateSnapshot, Result};

impl SqlOperator {
    pub(super) async fn process_compact(
        &mut self,
        prepared: PreparedRetention,
        mut initialized: Option<Box<incremental::IncrementalSql>>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let pending = storage::prepare_update(self, &prepared)?;
        let batch = prepared.batch;
        let newly_initialized = initialized.is_some();
        let native = initialized
            .as_mut()
            .or(self.incremental.as_mut())
            .expect("proved native SQL plan");
        let _descriptor = newly_initialized
            .then(|| native.native_descriptor(&self.name))
            .transpose()?;
        let runtime = self.stream_state.runtime()?;
        let transaction = native.update(&batch, context, &self.name).await?;
        #[cfg(test)]
        {
            self.incremental_work.0 += transaction.rows;
        }
        let produced =
            runtime.incremental_output(transaction.records.clone(), batch.metadata().clone())?;
        context.check_cancelled()?;
        output.emit("output", produced).await?;
        native.commit(transaction);
        if initialized.is_some() {
            self.incremental = initialized;
        }
        self.incremental_checked = true;
        self.compact = Some(Box::new(pending));
        self.retained = None;
        self.retained_capture = None;
        Ok(())
    }

    pub(super) fn prepare_compact_capture(&mut self, check: &dyn Fn() -> Result<()>) -> Result<()> {
        let prepared = self.prepare_checkpoint_work(check)?;
        check()?;
        self.install_checkpoint_work(prepared);
        Ok(())
    }

    pub(super) async fn prepare_compact_capture_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let state = self.compact.as_ref().expect("compact checkpoint state");
        let native = self.incremental.as_ref().expect("compact native state");
        let capture = capture::prepare_async(self, state, native, context).await?;
        context.check_cancelled()?;
        self.install_checkpoint_work(super::PreparedSqlCheckpoint {
            compact: Some(capture),
            retained: None,
        });
        Ok(())
    }

    pub(super) fn compact_checkpoint(&mut self) -> Result<OperatorStateSnapshot> {
        self.prepare_compact_capture(&|| Ok(()))?;
        Ok(self
            .compact
            .as_ref()
            .expect("compact capture prepared")
            .capture
            .as_ref()
            .expect("capture installed")
            .snapshot
            .clone())
    }
}

#[cfg(test)]
mod state_fixture_tests;

#[cfg(test)]
mod storage_tests;

#[cfg(test)]
pub(in crate::operator::sql) mod direct_async_tests;

#[cfg(test)]
mod float_count_tests;
