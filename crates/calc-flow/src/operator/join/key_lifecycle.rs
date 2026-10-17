use std::sync::Arc;

use super::{
    StoredRow, StreamJoinOperator, checked_metric,
    key_dictionary::{Dictionary, SideBindings},
};
use crate::{Result, StreamOperatorContext};

pub(super) struct Retirement<'a> {
    pub(super) dictionary: &'a mut Dictionary,
    pub(super) bindings: &'a mut SideBindings,
}

impl Retirement<'_> {
    pub(super) fn retire(&mut self, row: &StoredRow, name: &str) -> Result<()> {
        let key = self
            .bindings
            .remove(row.row_id)
            .unwrap_or_else(|| Arc::clone(&row.encoded_key));
        self.dictionary.retire(&key, name)
    }

    /// Aggregated retirement for one canonical key over a bulk-evicted
    /// prefix (spec FR13); the caller removes the side bindings in bulk.
    pub(super) fn retire_n(
        &mut self,
        key: &Arc<super::columnar::FramedKey>,
        count: u64,
        name: &str,
    ) -> Result<()> {
        self.dictionary.retire_n(key, count, name)
    }
}

impl StreamJoinOperator {
    pub(super) async fn rebuild_key_identities(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if !self.key_dictionary_needs_rebuild {
            return Ok(());
        }
        self.key_work.rebuild_attempts = checked_metric(
            self.key_work.rebuild_attempts,
            1,
            &self.name,
            "rebuild_attempts",
        )?;
        let completed = checked_metric(
            self.key_work.rebuild_completions,
            1,
            &self.name,
            "rebuild_completions",
        )?;
        let master = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-dictionary");
        let Some(rebuilt) = self
            .key_dictionary
            .rebuild(
                [&self.state.left, &self.state.right],
                master,
                Some(context),
                &self.name,
            )
            .await?
        else {
            return Ok(());
        };
        self.state.left.1 = None;
        self.state.right.1 = None;
        self.key_dictionary = rebuilt.dictionary;
        self.key_bindings = rebuilt.bindings;
        self.key_dictionary_needs_rebuild = false;
        self.key_work.rebuild_completions = completed;
        Ok(())
    }
}

impl StreamJoinOperator {
    pub(super) fn sql_equality_tables(
        &mut self,
        probe: super::RecordBatch,
        state: super::RecordBatch,
    ) -> Result<std::collections::BTreeMap<String, crate::Batch>> {
        let count = checked_metric(
            self.key_work.sql_probe_table_builds,
            1,
            &self.name,
            "sql_probe_table_builds",
        )?;
        let tables = super::equality_tables(probe, state)?;
        self.key_work.sql_probe_table_builds = count;
        Ok(tables)
    }
}

impl StreamJoinOperator {
    pub(super) fn restored_key_refusals(&self, refused: bool) -> Result<u64> {
        checked_metric(
            self.key_work.dictionary_grant_refusals,
            u64::from(refused),
            &self.name,
            "dictionary_grant_refusals",
        )
    }

    pub(super) fn install_restored_keys(
        &mut self,
        keys: Option<super::key_dictionary::Rebuilt>,
        nonempty: bool,
    ) {
        self.key_dictionary_needs_rebuild = nonempty && keys.is_none();
        if let Some(keys) = keys {
            self.key_dictionary = keys.dictionary;
            self.key_bindings = keys.bindings;
        } else {
            self.key_dictionary = Dictionary::default();
            self.key_bindings = super::key_dictionary::Bindings::default();
        }
    }
}
