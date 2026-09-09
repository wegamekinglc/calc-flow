use crate::{Result, StateSegment};
use datafusion::arrow::{
    record_batch::RecordBatch,
    row::{RowConverter, SortField},
};
use std::{collections::BTreeMap, sync::Arc};

pub(super) type Encoding = Arc<Vec<u8>>;
pub(super) type LeftOrder = (i64, Encoding, Encoding);
pub(super) type RightOrder = (i64, Encoding);

#[derive(Clone, Default)]
pub(super) struct State {
    pub left: BTreeMap<LeftOrder, StateSegment>,
    pub right: BTreeMap<Encoding, BTreeMap<RightOrder, Option<StateSegment>>>,
}

impl State {
    pub fn candidate(&self, key: &Encoding, time: i64, tolerance: u64) -> Option<&StateSegment> {
        let bucket = self.right.get(key)?;
        let found = if let Some(next) = time.checked_add(1) {
            bucket.range(..(next, Arc::new(Vec::new()))).next_back()
        } else {
            bucket.last_key_value()
        }?;
        if i128::from(found.0.0) < i128::from(time) - i128::from(tolerance) {
            return None;
        }
        found.1.as_ref()
    }
}

pub(super) fn encoded_columns(
    batch: &RecordBatch,
    row: usize,
    names: &[String],
) -> Result<Encoding> {
    let arrays = names
        .iter()
        .map(|name| {
            Ok(batch
                .column(
                    batch
                        .schema()
                        .index_of(name)
                        .map_err(|error| super::arrow_error(&error))?,
                )
                .slice(row, 1))
        })
        .collect::<Result<Vec<_>>>()?;
    let fields = arrays
        .iter()
        .map(|array| SortField::new(array.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields).map_err(|error| super::arrow_error(&error))?;
    let rows = converter
        .convert_columns(&arrays)
        .map_err(|error| super::arrow_error(&error))?;
    Ok(Arc::new(rows.row(0).as_ref().to_vec()))
}

#[derive(Default)]
pub(super) struct Inventory {
    pub identities: u64,
    pub right_payloads: u64,
    pub identity_only: u64,
    pub bytes: u64,
}

impl State {
    pub fn inventory(&self, prepared: Option<&StateSegment>, name: &str) -> Result<Inventory> {
        let mut total = Inventory::default();
        for ((_, key, sequence), row) in &self.left {
            total.identities = super::checked(name, total.identities, 1)?;
            total.bytes = super::checked(name, total.bytes, 256 + 64 + 64)?;
            total.bytes = allocation_charge(total.bytes, key, name)?;
            total.bytes = allocation_charge(total.bytes, sequence, name)?;
            total.bytes = allocation_charge(total.bytes, &row.bytes_arc(), name)?;
        }
        for (key, bucket) in &self.right {
            total.bytes = super::checked(name, total.bytes, 64)?;
            total.bytes = allocation_charge(total.bytes, key, name)?;
            for ((_, sequence), row) in bucket {
                total.identities = super::checked(name, total.identities, 1)?;
                total.bytes = super::checked(name, total.bytes, 256 + 64)?;
                total.bytes = allocation_charge(total.bytes, sequence, name)?;
                if let Some(row) = row {
                    total.right_payloads = super::checked(name, total.right_payloads, 1)?;
                    total.bytes = super::checked(name, total.bytes, 64)?;
                    total.bytes = allocation_charge(total.bytes, &row.bytes_arc(), name)?;
                } else {
                    total.identity_only = super::checked(name, total.identity_only, 1)?;
                }
            }
        }
        if let Some(prepared) = prepared {
            total.bytes = super::checked(name, total.bytes, 64)?;
            total.bytes = allocation_charge(total.bytes, &prepared.bytes_arc(), name)?;
        }
        Ok(total)
    }

    pub fn evict(&mut self, status: &super::StreamAsofJoinStatus, tolerance: u64) -> u64 {
        let future = if status.left.ended {
            i128::MAX
        } else {
            status
                .left
                .watermark_micros
                .map_or(i128::MIN, |wm| i128::from(wm.as_micros()))
        };
        let pending = self
            .left
            .first_key_value()
            .map_or(i128::MAX, |(key, _)| i128::from(key.0));
        let threshold = future.min(pending);
        let mut evicted = 0;
        self.right.retain(|_, bucket| {
            bucket.retain(|(time, _), row| {
                if i128::from(*time) + i128::from(tolerance) < threshold
                    && row.take_if(|_| true).is_some()
                {
                    evicted += 1;
                }
                row.is_some()
                    || !(status.right.ended
                        || status
                            .right
                            .watermark_micros
                            .is_some_and(|wm| *time < wm.as_micros()))
            });
            !bucket.is_empty()
        });
        evicted
    }
}

fn allocation_charge(current: u64, bytes: &Arc<Vec<u8>>, name: &str) -> Result<u64> {
    super::checked(name, current, bytes.capacity() as u64)
}
