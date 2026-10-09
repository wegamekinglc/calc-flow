use std::sync::Arc;

use super::super::{
    IngressProgressSnapshot, OperatorStateSnapshot, RetainedKeyCache, StreamJoinOperator,
};
use super::candidate::{Config, Decoder, Prepared};
use crate::Result;

impl StreamJoinOperator {
    pub(in crate::operator::join) fn restore_v2_snapshot(
        &mut self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<()> {
        let credit = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-v2-restore");
        let config = Config {
            name: &self.name,
            spec: &self.spec,
            schemas: [
                Arc::clone(self.input_schema(0)),
                Arc::clone(self.input_schema(1)),
            ],
            keys: [
                &self.compiled.left_key_indices,
                &self.compiled.right_key_indices,
            ],
            time_indices: [
                self.compiled.left_event_time_index,
                self.compiled.right_event_time_index,
            ],
            #[cfg(test)]
            metadata_hook: self.metadata_test_hook.as_ref(),
            #[cfg(test)]
            decoded_hook: self.decoded_row_test_hook.as_ref(),
            #[cfg(test)]
            owned_work: false,
        };
        let prepared =
            Decoder::new(snapshot, &config, &credit, Vec::new(), &|| Ok(()))?.decode()?;
        self.install_v2(prepared, &|| Ok(()))
    }

    pub(super) fn install_v2(
        &mut self,
        prepared: Prepared,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        check()?;
        let Prepared {
            state,
            containers,
            workspace,
        } = prepared;
        let old_state = std::mem::replace(&mut self.state, state);
        let old_containers = self.v2_containers.replace(containers);
        self.retained_key_cache = RetainedKeyCache::default();
        self.ingress_progress = IngressProgressSnapshot::default();
        drop(old_state);
        drop(old_containers);
        drop(workspace);
        Ok(())
    }
}
