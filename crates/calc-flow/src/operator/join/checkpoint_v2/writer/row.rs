use super::super::super::{PayloadRef, PendingOp, Rows, StoredRow};
use crate::EventTime;

pub(super) struct RowRef<'a> {
    pub(super) record: PayloadRef<'a>,
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
            record: (&self.record).into(),
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
            record: record.into(),
            encoded_key,
            event_time: *event_time,
            row_id: *row_id,
            charge: *charge,
        }
    }
}

/// A thin borrow of the side's row source plus its stable chunk coordinate.
pub(super) struct Reference<'a> {
    pub(super) rows: &'a &'a dyn Rows,
    pub(super) index: usize,
}

impl CheckpointRow for Reference<'_> {
    fn row(&self) -> RowRef<'_> {
        let (event_time, row_id) = self.rows.identity(self.index);
        RowRef {
            record: self.rows.payload(self.index),
            encoded_key: self.rows.key(self.index),
            event_time,
            row_id,
            charge: self.rows.charge(self.index),
        }
    }
}
