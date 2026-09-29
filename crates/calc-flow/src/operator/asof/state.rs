use crate::{Result, StateSegment};
use datafusion::arrow::{
    record_batch::RecordBatch,
    row::{RowConverter, Rows, SortField},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, LazyLock},
};

pub(super) type Encoding = Arc<Vec<u8>>;
pub(super) type LeftOrder = (i64, Encoding, Encoding);
pub(super) type RightOrder = (i64, Encoding);
pub(super) type BatchKey = (u8, u64);

pub(super) struct PayloadBatch {
    pub key: BatchKey,
    pub record: Arc<RecordBatch>,
    pub encoded: StateSegment,
    pub body_bytes: u64,
}

#[derive(Clone)]
pub(super) struct RowPayload {
    pub batch: Arc<PayloadBatch>,
    pub row: usize,
}

static EMPTY_ENCODING: LazyLock<Encoding> = LazyLock::new(|| Arc::new(Vec::new()));

#[derive(Clone, Default)]
pub(super) struct State {
    pub left: BTreeMap<LeftOrder, RowPayload>,
    pub right: BTreeMap<Encoding, BTreeMap<RightOrder, Option<RowPayload>>>,
    pub batches: BTreeMap<BatchKey, (Arc<PayloadBatch>, usize)>,
}

impl State {
    /// Account for an admission before mutating committed state. All inserted
    /// identities and payload batches are unique after admission validation.
    pub fn inventory_after_admission(
        &self,
        mut current: Inventory,
        previous_index_bytes: u64,
        next_index_len: u64,
        side: usize,
        rows: &[(LeftOrder, RowPayload)],
        name: &str,
    ) -> Result<Inventory> {
        current.bytes = current
            .bytes
            .checked_sub(previous_index_bytes)
            .expect("committed index charge is included in state bytes");
        current.bytes = super::checked(
            name,
            current.bytes,
            super::checked(name, next_index_len, 64)?,
        )?;
        let mut new_buckets = BTreeSet::new();
        let mut new_batches = BTreeSet::new();
        for ((_, key, sequence), payload) in rows {
            if side == 0 {
                current.charge_left(key, sequence, payload, name)?;
            } else {
                if !self.right.contains_key(key) && new_buckets.insert(key) {
                    current.charge_allocation(key, name)?;
                }
                current.charge_right(sequence, Some(payload), name)?;
            }
            if new_batches.insert(payload.batch.key) {
                current.bytes =
                    super::checked(name, current.bytes, batch_allocation(&payload.batch))?;
            }
        }
        Ok(current)
    }

    pub fn attach(&mut self, row: &RowPayload) {
        self.batches
            .entry(row.batch.key)
            .or_insert_with(|| (row.batch.clone(), 0))
            .1 += 1;
    }

    pub fn detach(&mut self, row: &RowPayload) {
        detach_batch(&mut self.batches, row);
    }

    pub fn contains_identity(&self, index: usize, identity: &LeftOrder) -> bool {
        if index == 0 {
            self.left.contains_key(identity)
        } else {
            self.right
                .get(&identity.1)
                .is_some_and(|bucket| bucket.contains_key(&(identity.0, identity.2.clone())))
        }
    }

    pub fn candidate(&self, key: &Encoding, time: i64, tolerance: u64) -> Option<&RowPayload> {
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
    pub(super) fn with_row<R>(&self, row: usize, use_bytes: impl FnOnce(&[u8]) -> R) -> R {
        let encoded = self.rows.row(row);
        use_bytes(encoded.as_ref())
    }

    /// Returns the owned encoding of one row.
    pub(super) fn row(&self, row: usize) -> Encoding {
        self.with_row(row, |bytes| Arc::new(bytes.to_vec()))
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
    /// Charge each retained Arrow batch once, alongside its row indexes.
    pub fn inventory(
        &self,
        prepared: Option<&super::checkpoint::PreparedSegment>,
        name: &str,
    ) -> Result<Inventory> {
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
        for (batch, refs) in self.batches.values() {
            if *refs == 0 {
                return Err(super::reason(
                    name,
                    crate::StreamingFailureReason::AsofProtocolError,
                    "ASOF retained an unreferenced payload batch",
                ));
            }
            total.bytes = super::checked(name, total.bytes, batch_allocation(batch))?;
        }
        if let Some(prepared) = prepared {
            total.bytes = super::checked(name, total.bytes, prepared_allocation(prepared))?;
        }
        Ok(total)
    }

    /// Compute the committed charge of a finalized left prefix without
    /// cloning the retained maps or their Arrow batch references.
    pub fn inventory_after_left_prefix(
        &self,
        keys: &[LeftOrder],
        mut total: Inventory,
        previous_index_bytes: u64,
        next_index_bytes: u64,
        name: &str,
    ) -> Result<Inventory> {
        total.bytes -= previous_index_bytes;
        let mut removed = BTreeMap::<BatchKey, usize>::new();
        for key in keys {
            let row = self.left.get(key).expect("pending ASOF identity");
            total.identities -= 1;
            total.bytes -= left_row_charge(&key.1, &key.2, row);
            *removed.entry(row.batch.key).or_default() += 1;
        }
        for (key, count) in removed {
            let (batch, references) = &self.batches[&key];
            if count == *references {
                total.bytes -= batch_allocation(batch);
            }
        }
        total.bytes = super::checked(name, total.bytes, next_index_bytes)?;
        Ok(total)
    }

    pub fn evict(&mut self, status: &super::StreamAsofJoinStatus, tolerance: u64) -> u64 {
        let threshold = retention_threshold(self, status);
        let mut evicted = 0;
        let batches = &mut self.batches;
        self.right.retain(|_, bucket| {
            bucket.retain(|(time, _), row| {
                if payload_expired(*time, tolerance, threshold)
                    && let Some(payload) = row.take()
                {
                    evicted += 1;
                    detach_batch(batches, &payload);
                }
                row.is_some() || !identity_expired(*time, status)
            });
            !bucket.is_empty()
        });
        evicted
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

fn detach_batch(batches: &mut BTreeMap<BatchKey, (Arc<PayloadBatch>, usize)>, row: &RowPayload) {
    let count = &mut batches.get_mut(&row.batch.key).expect("indexed batch").1;
    *count -= 1;
    if *count == 0 {
        batches.remove(&row.batch.key);
    }
}

fn encoding_allocation(bytes: &Encoding) -> u64 {
    ALLOCATION_BYTES + bytes.capacity() as u64
}

fn payload_allocation(_row: &RowPayload) -> u64 {
    64
}

fn batch_allocation(batch: &PayloadBatch) -> u64 {
    // IPC readers can share one body buffer among many column slices. Arrow's
    // per-array capacity report counts that same allocation repeatedly, and
    // differs from the arrays made by admission. Charge the retained IPC
    // buffer, its decoded body length, and column metadata once per batch.
    ALLOCATION_BYTES
        + batch.encoded.bytes_arc().capacity() as u64
        + batch.body_bytes
        + 256
        + 64 * batch.record.num_columns() as u64
}

fn prepared_allocation(prepared: &super::checkpoint::PreparedSegment) -> u64 {
    ALLOCATION_BYTES + prepared.capacity() as u64
}

fn left_row_charge(key: &Encoding, sequence: &Encoding, row: &RowPayload) -> u64 {
    LEFT_IDENTITY_BYTES
        + encoding_allocation(key)
        + encoding_allocation(sequence)
        + payload_allocation(row)
}

fn right_row_charge(sequence: &Encoding, row: Option<&RowPayload>) -> u64 {
    RIGHT_IDENTITY_BYTES + encoding_allocation(sequence) + row.map_or(0, payload_allocation)
}

impl Inventory {
    fn charge_left(
        &mut self,
        key: &Encoding,
        sequence: &Encoding,
        row: &RowPayload,
        name: &str,
    ) -> Result<()> {
        self.identities = super::checked(name, self.identities, 1)?;
        self.bytes = super::checked(name, self.bytes, left_row_charge(key, sequence, row))?;
        Ok(())
    }

    fn charge_right(
        &mut self,
        sequence: &Encoding,
        row: Option<&RowPayload>,
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
}
