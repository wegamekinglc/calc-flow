use super::super::super::{PendingOp, StoredRow, columnar::RowPayload};
use crate::EventTime;

pub(super) struct RowRef<'a> {
    pub(super) record: &'a RowPayload,
    pub(super) encoded_key: &'a [u8],
    pub(super) event_time: EventTime,
    pub(super) row_id: u64,
    pub(super) charge: u64,
}

pub(super) trait CheckpointRow {
    fn row(&self) -> RowRef<'_>;
}

impl CheckpointRow for StoredRow {
    fn row(&self) -> RowRef<'_> {
        RowRef {
            record: &self.record,
            encoded_key: &self.encoded_key,
            event_time: self.event_time,
            row_id: self.row_id,
            charge: self.charge,
        }
    }
}

impl<T: CheckpointRow> CheckpointRow for &T {
    fn row(&self) -> RowRef<'_> {
        (*self).row()
    }
}

pub(super) struct PendingUpsert<'a>(&'a PendingOp);

impl<'a> PendingUpsert<'a> {
    pub(super) fn new(op: &'a PendingOp) -> Option<Self> {
        matches!(op, PendingOp::Upsert { .. }).then_some(Self(op))
    }
}

impl CheckpointRow for PendingUpsert<'_> {
    fn row(&self) -> RowRef<'_> {
        let PendingOp::Upsert {
            record,
            encoded_key,
            event_time,
            row_id,
            charge,
            ..
        } = self.0
        else {
            unreachable!("pending upsert selection contains only upserts")
        };
        RowRef {
            record,
            encoded_key,
            event_time: *event_time,
            row_id: *row_id,
            charge: *charge,
        }
    }
}
