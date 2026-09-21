use crate::{Result, StateSegment};
use datafusion::arrow::{
    record_batch::RecordBatch,
    row::{RowConverter, Rows, SortField},
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

/// Row encodings for one identity column set over a whole record batch: one
/// `RowConverter` per batch with per-row bytes extracted on demand. The
/// extracted bytes are identical to encoding one-row slices because the
/// pinned Arrow 58 row format encodes each value independently of its
/// position in the column.
pub(super) struct EncodedColumns {
    rows: Rows,
}

impl EncodedColumns {
    /// Returns the owned encoding of one row.
    pub(super) fn row(&self, row: usize) -> Encoding {
        Arc::new(self.rows.row(row).as_ref().to_vec())
    }
}

/// Resolves each named column once and converts the whole batch, hoisting the
/// schema lookups and converter construction out of per-row loops.
pub(super) fn encode_columns(batch: &RecordBatch, names: &[String]) -> Result<EncodedColumns> {
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
                .clone())
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
    Ok(EncodedColumns { rows })
}

pub(super) fn encoded_columns(
    batch: &RecordBatch,
    row: usize,
    names: &[String],
) -> Result<Encoding> {
    Ok(encode_columns(batch, names)?.row(row))
}

/// Ordered-map node and inline headroom charged per left identity, on top of
/// the key, sequence and payload buffer allocations.
const LEFT_IDENTITY_BYTES: u64 = 256 + 64 + 64;
/// Ordered-map node charged per right identity; the bucket key allocation is
/// charged separately, once per bucket.
const RIGHT_IDENTITY_BYTES: u64 = 256 + 64;
/// Per-allocation header charged on top of each owned buffer's capacity.
const ALLOCATION_BYTES: u64 = 64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Inventory {
    pub identities: u64,
    pub right_payloads: u64,
    pub identity_only: u64,
    pub bytes: u64,
}

impl State {
    /// Full-walk inventory charge: the cold path used by checkpoint restore
    /// validation and by the debug cross-check of the maintained deltas.
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

    pub fn evict(
        &mut self,
        status: &super::StreamAsofJoinStatus,
        tolerance: u64,
    ) -> (u64, InventoryDelta) {
        let threshold = retention_threshold(self, status);
        let mut evicted = 0;
        let mut delta = InventoryDelta::default();
        self.right.retain(|key, bucket| {
            bucket.retain(|(time, sequence), row| {
                if payload_expired(*time, tolerance, threshold)
                    && let Some(payload) = row.take()
                {
                    evicted += 1;
                    delta.evict_payload(&payload);
                }
                if row.is_some() || !identity_expired(*time, status) {
                    true
                } else {
                    delta.remove_right_identity(sequence);
                    false
                }
            });
            if bucket.is_empty() {
                delta.remove_bucket(key);
                false
            } else {
                true
            }
        });
        (evicted, delta)
    }
}

/// Retention threshold shared by `evict` and `eviction_pending`: the more
/// conservative of the left frontier and the oldest pending left row.
fn retention_threshold(state: &State, status: &super::StreamAsofJoinStatus) -> i128 {
    let future = if status.left.ended {
        i128::MAX
    } else {
        status
            .left
            .watermark_micros
            .map_or(i128::MIN, |wm| i128::from(wm.as_micros()))
    };
    let pending = state
        .left
        .first_key_value()
        .map_or(i128::MAX, |(key, _)| i128::from(key.0));
    future.min(pending)
}

/// Right payloads whose match window closed before the retention threshold
/// can no longer answer a pending or future left row.
fn payload_expired(time: i64, tolerance: u64, threshold: i128) -> bool {
    i128::from(time) + i128::from(tolerance) < threshold
}

/// Identity-only rows past their own ingress boundary can never be rejoined
/// by a payload, so the identity itself is dropped.
fn identity_expired(time: i64, status: &super::StreamAsofJoinStatus) -> bool {
    status.right.ended
        || status
            .right
            .watermark_micros
            .is_some_and(|wm| time < wm.as_micros())
}

/// Reports whether `evict` would drop any payload or identity, without
/// mutating the state. `finish_progress` uses this to skip recloning and
/// re-encoding an untouched state on progress-only watermark ticks.
pub(super) fn eviction_pending(
    state: &State,
    status: &super::StreamAsofJoinStatus,
    tolerance: u64,
) -> bool {
    let threshold = retention_threshold(state, status);
    eviction_pending_at(state, status, tolerance, threshold)
}

/// Checks the greatest retention threshold reachable while finalizing this progress tick.
pub(super) fn right_stable_during_finalization(
    state: &State,
    status: &super::StreamAsofJoinStatus,
    tolerance: u64,
) -> bool {
    let threshold = if status.left.ended {
        i128::MAX
    } else {
        status
            .left
            .watermark_micros
            .map_or(i128::MIN, |wm| i128::from(wm.as_micros()))
    };
    !eviction_pending_at(state, status, tolerance, threshold)
}

fn eviction_pending_at(
    state: &State,
    status: &super::StreamAsofJoinStatus,
    tolerance: u64,
    threshold: i128,
) -> bool {
    state.right.values().any(|bucket| {
        bucket.iter().any(|((time, _), row)| {
            (row.is_some() && payload_expired(*time, tolerance, threshold))
                || (row.is_none() && identity_expired(*time, status))
        })
    })
}

/// Signed mutation of the charged state inventory, accumulated while one
/// transactional transition inserts, removes or evicts entries. The delta is
/// applied to the committed gauges instead of re-walking the full state;
/// debug builds cross-check the result against `State::inventory`.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct InventoryDelta {
    identities: i128,
    right_payloads: i128,
    identity_only: i128,
    bytes: i128,
}

impl InventoryDelta {
    pub fn insert_left(&mut self, key: &Encoding, sequence: &Encoding, row: &StateSegment) {
        self.identities += 1;
        self.bytes += i128::from(left_row_charge(key, sequence, row));
    }

    pub fn remove_left(&mut self, key: &Encoding, sequence: &Encoding, row: &StateSegment) {
        self.identities -= 1;
        self.bytes -= i128::from(left_row_charge(key, sequence, row));
    }

    pub fn insert_right(
        &mut self,
        key: &Encoding,
        sequence: &Encoding,
        row: &StateSegment,
        new_bucket: bool,
    ) {
        if new_bucket {
            self.bytes += i128::from(encoding_allocation(key));
        }
        self.identities += 1;
        self.right_payloads += 1;
        self.bytes += i128::from(right_row_charge(sequence, Some(row)));
    }

    fn evict_payload(&mut self, row: &StateSegment) {
        self.right_payloads -= 1;
        self.identity_only += 1;
        self.bytes -= i128::from(payload_allocation(row));
    }

    fn remove_right_identity(&mut self, sequence: &Encoding) {
        self.identities -= 1;
        self.identity_only -= 1;
        self.bytes -= i128::from(right_row_charge(sequence, None));
    }

    fn remove_bucket(&mut self, key: &Encoding) {
        self.bytes -= i128::from(encoding_allocation(key));
    }

    pub fn merge(&mut self, other: InventoryDelta) {
        self.identities += other.identities;
        self.right_payloads += other.right_payloads;
        self.identity_only += other.identity_only;
        self.bytes += other.bytes;
    }
}

fn encoding_allocation(bytes: &Encoding) -> u64 {
    ALLOCATION_BYTES + bytes.capacity() as u64
}

fn payload_allocation(row: &StateSegment) -> u64 {
    ALLOCATION_BYTES + row.bytes_arc().capacity() as u64
}

fn left_row_charge(key: &Encoding, sequence: &Encoding, row: &StateSegment) -> u64 {
    LEFT_IDENTITY_BYTES
        + encoding_allocation(key)
        + encoding_allocation(sequence)
        + payload_allocation(row)
}

fn right_row_charge(sequence: &Encoding, row: Option<&StateSegment>) -> u64 {
    RIGHT_IDENTITY_BYTES + encoding_allocation(sequence) + row.map_or(0, payload_allocation)
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
        self.bytes = super::checked(name, self.bytes, left_row_charge(key, sequence, row))?;
        Ok(())
    }

    fn charge_right(
        &mut self,
        sequence: &Encoding,
        row: Option<&StateSegment>,
        name: &str,
    ) -> Result<()> {
        self.identities = super::checked(name, self.identities, 1)?;
        self.bytes = super::checked(name, self.bytes, right_row_charge(sequence, row))?;
        if row.is_some() {
            self.right_payloads = super::checked(name, self.right_payloads, 1)?;
        } else {
            self.identity_only = super::checked(name, self.identity_only, 1)?;
        }
        Ok(())
    }

    fn charge_allocation(&mut self, bytes: &Encoding, name: &str) -> Result<()> {
        self.bytes = super::checked(name, self.bytes, encoding_allocation(bytes))?;
        Ok(())
    }

    /// Applies a signed mutation delta, failing closed if the maintained
    /// counters would drift negative or overflow.
    pub fn apply(&mut self, delta: InventoryDelta, name: &str) -> Result<()> {
        self.identities = apply_delta(name, self.identities, delta.identities)?;
        self.right_payloads = apply_delta(name, self.right_payloads, delta.right_payloads)?;
        self.identity_only = apply_delta(name, self.identity_only, delta.identity_only)?;
        self.bytes = apply_delta(name, self.bytes, delta.bytes)?;
        Ok(())
    }

    /// Charges the freshly encoded checkpoint segment allocation.
    pub fn charge_prepared(&mut self, prepared: Option<&StateSegment>, name: &str) -> Result<()> {
        if let Some(prepared) = prepared {
            self.bytes = super::checked(name, self.bytes, payload_allocation(prepared))?;
        }
        Ok(())
    }

    /// Retires the previously installed checkpoint segment charge, failing
    /// closed if the maintained bytes would underflow.
    pub fn uncharge_prepared(&mut self, prepared: Option<&StateSegment>, name: &str) -> Result<()> {
        if let Some(prepared) = prepared {
            self.bytes = apply_delta(name, self.bytes, -i128::from(payload_allocation(prepared)))?;
        }
        Ok(())
    }
}

fn apply_delta(name: &str, base: u64, delta: i128) -> Result<u64> {
    i128::from(base)
        .checked_add(delta)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| {
            super::reason(
                name,
                crate::StreamingFailureReason::AsofCounterOverflow,
                "ASOF counter or resource arithmetic overflowed",
            )
        })
}
