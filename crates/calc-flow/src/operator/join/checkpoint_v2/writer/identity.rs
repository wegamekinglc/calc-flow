//! Compile-time layout identity for private checkpoint history and work cuts.

use datafusion::arrow::datatypes::SchemaRef;
use sha2::{Digest, Sha256};

use super::{WriterState, cut};
use crate::{
    Result,
    operator::join::{
        StreamJoinOperator,
        state_projection::{
            Layout,
            checkpoint::{self, Descriptor},
        },
    },
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::operator::join) struct HistoryIdentity(pub(super) [u8; 32]);

impl HistoryIdentity {
    pub(in crate::operator::join) fn full(schemas: [&SchemaRef; 2]) -> Self {
        let mut hash = start("full-v2", "v1_exact");
        for schema in schemas {
            let digest = checkpoint::digest(schema);
            side(&mut hash, &digest, &digest, 0..schema.fields().len());
        }
        Self(hash.finalize().into())
    }

    fn projected(descriptor: &Descriptor) -> Self {
        let mut hash = start("projected-v3", checkpoint::CHARGE_MODE);
        for index in 0..2 {
            side(
                &mut hash,
                &descriptor.logical_schema_digests[index],
                &descriptor.schema_digests[index],
                descriptor.columns[index].iter().copied(),
            );
        }
        Self(hash.finalize().into())
    }
}

fn start(layout: &str, charge: &str) -> Sha256 {
    let mut hash = Sha256::new();
    text(&mut hash, "calc-flow/join/checkpoint-history/v1");
    text(&mut hash, layout);
    text(&mut hash, charge);
    hash
}

fn text(hash: &mut Sha256, text: &str) {
    hash.update((text.len() as u64).to_le_bytes());
    hash.update(text.as_bytes());
}

fn side(
    hash: &mut Sha256,
    logical: &str,
    physical: &str,
    columns: impl ExactSizeIterator<Item = usize>,
) {
    text(hash, logical);
    text(hash, physical);
    hash.update((columns.len() as u64).to_le_bytes());
    for index in columns {
        hash.update((index as u64).to_le_bytes());
    }
}

impl WriterState {
    pub(in crate::operator::join) fn with_identity(identity: HistoryIdentity) -> Self {
        Self {
            tracker: cut::Tracker {
                identity,
                ..cut::Tracker::default()
            },
            ..Self::default()
        }
    }
}

impl StreamJoinOperator {
    pub(in crate::operator::join) fn rebind_checkpoint_layout(
        &mut self,
        layout: Option<&Layout>,
    ) -> Result<()> {
        let identity = layout.map_or_else(
            || HistoryIdentity::full([self.input_schema(0), self.input_schema(1)]),
            |layout| HistoryIdentity::projected(&layout.descriptor),
        );
        if identity != self.checkpoint_identity {
            self.validate_fresh_projection()?;
            let mut writer = self.v2_writer.next_owner(false)?;
            writer.tracker.identity = identity;
            self.v2_writer = writer;
            self.checkpoint_identity = identity;
        }
        Ok(())
    }

    pub(in crate::operator::join::checkpoint_v2::writer) fn v2_history_matches(&self) -> bool {
        match (
            self.v2_writer.base.is_some(),
            self.v2_writer.history_identity,
        ) {
            (true, Some(identity)) => identity == self.checkpoint_identity,
            (false, None) => true,
            _ => false,
        }
    }

    pub(super) fn v2_cut_matches(&self, cut: &cut::Cut) -> bool {
        cut.identity == self.checkpoint_identity
            && *cut
                == self
                    .v2_writer
                    .tracker
                    .snapshot(self.state.last_checkpoint_epoch)
    }

    pub(super) fn must_rebuild_v2_history(&self) -> bool {
        !self.v2_writer.history_trusted
            || self.v2_writer.migration_required
            || !self.v2_history_matches()
    }
}
