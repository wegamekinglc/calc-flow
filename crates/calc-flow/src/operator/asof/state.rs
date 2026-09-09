use crate::{Result, StateSegment};
use datafusion::arrow::{
    record_batch::RecordBatch,
    row::{RowConverter, SortField},
};
use std::{
    collections::BTreeMap,
    sync::{Arc, LazyLock},
};

pub(super) type Encoding = Arc<Vec<u8>>;
pub(super) type LeftOrder = (i64, Encoding, Encoding);
pub(super) type RightOrder = (i64, Encoding);

static EMPTY_ENCODING: LazyLock<Encoding> = LazyLock::new(|| Arc::new(Vec::new()));

#[derive(Clone, Default)]
pub(super) struct State {
    pub left: BTreeMap<LeftOrder, StateSegment>,
    pub right: BTreeMap<Encoding, BTreeMap<RightOrder, Option<StateSegment>>>,
}

impl State {
    pub fn contains_identity(&self, index: usize, identity: &LeftOrder) -> bool {
        if index == 0 {
            self.left.contains_key(identity)
        } else {
            self.right
                .get(&identity.1)
                .is_some_and(|bucket| bucket.contains_key(&(identity.0, identity.2.clone())))
        }
    }

    pub fn candidate(&self, key: &Encoding, time: i64, tolerance: u64) -> Option<&StateSegment> {
        let bucket = self.right.get(key)?;
        let found = if let Some(next) = time.checked_add(1) {
            bucket.range(..(next, EMPTY_ENCODING.clone())).next_back()
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
        self.left.iter().try_for_each(|((_, key, sequence), row)| {
            total.charge_left(key, sequence, row, name)
        })?;
        for (key, bucket) in &self.right {
            total.charge_allocation(key, name)?;
            for ((_, sequence), row) in bucket {
                total.charge_right(sequence, row.as_ref(), name)?;
            }
        }
        if let Some(prepared) = prepared {
            total.charge_allocation(&prepared.bytes_arc(), name)?;
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

impl Inventory {
    fn charge_left(
        &mut self,
        key: &Encoding,
        sequence: &Encoding,
        row: &StateSegment,
        name: &str,
    ) -> Result<()> {
        self.identities = super::checked(name, self.identities, 1)?;
        self.bytes = super::checked(name, self.bytes, 256 + 64 + 64)?;
        self.bytes = allocation_charge(self.bytes, key, name)?;
        self.bytes = allocation_charge(self.bytes, sequence, name)?;
        self.bytes = allocation_charge(self.bytes, &row.bytes_arc(), name)?;
        Ok(())
    }

    fn charge_right(
        &mut self,
        sequence: &Encoding,
        row: Option<&StateSegment>,
        name: &str,
    ) -> Result<()> {
        self.identities = super::checked(name, self.identities, 1)?;
        self.bytes = super::checked(name, self.bytes, 256 + 64)?;
        self.bytes = allocation_charge(self.bytes, sequence, name)?;
        if let Some(row) = row {
            self.right_payloads = super::checked(name, self.right_payloads, 1)?;
            self.charge_allocation(&row.bytes_arc(), name)?;
        } else {
            self.identity_only = super::checked(name, self.identity_only, 1)?;
        }
        Ok(())
    }

    fn charge_allocation(&mut self, bytes: &Encoding, name: &str) -> Result<()> {
        self.bytes = super::checked(name, self.bytes, 64)?;
        self.bytes = allocation_charge(self.bytes, bytes, name)?;
        Ok(())
    }
}

fn allocation_charge(current: u64, bytes: &Encoding, name: &str) -> Result<u64> {
    super::checked(name, current, bytes.capacity() as u64)
}
