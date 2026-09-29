use crate::{Result, StateSegment};
use datafusion::arrow::{
    array::{Array, Int64Array, UInt64Array},
    record_batch::RecordBatch,
    row::{RowConverter, Rows, SortField},
};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    hash::{Hash, Hasher},
    ops::Deref,
    sync::{Arc, OnceLock},
};

const INLINE_ENCODING_BYTES: usize = 10;

/// Canonical Arrow row bytes, with short scalar identities stored inline.
#[derive(Clone, Debug)]
pub(super) enum Encoding {
    Inline {
        len: u8,
        bytes: [u8; INLINE_ENCODING_BYTES],
    },
    Shared(Arc<Vec<u8>>),
}

impl Encoding {
    pub fn fits_inline(bytes: &[u8]) -> bool {
        bytes.len() <= INLINE_ENCODING_BYTES
    }

    pub fn from_slice(bytes: &[u8]) -> Self {
        if Self::fits_inline(bytes) {
            let mut inline = [0; INLINE_ENCODING_BYTES];
            inline[..bytes.len()].copy_from_slice(bytes);
            Self::Inline {
                len: u8::try_from(bytes.len()).expect("bounded inline encoding"),
                bytes: inline,
            }
        } else {
            Self::Shared(Arc::new(bytes.to_vec()))
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Inline { len, bytes } => &bytes[..usize::from(*len)],
            Self::Shared(bytes) => bytes.as_slice(),
        }
    }

    pub fn capacity(&self) -> usize {
        match self {
            Self::Inline { len, .. } => usize::from(*len),
            Self::Shared(bytes) => bytes.capacity(),
        }
    }

    #[cfg(test)]
    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline { .. })
    }
}

impl AsRef<[u8]> for Encoding {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Deref for Encoding {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl PartialEq for Encoding {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for Encoding {}

impl PartialOrd for Encoding {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Encoding {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl Hash for Encoding {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

pub(super) type LeftOrder = (i64, Encoding, Encoding);
pub(super) type RightOrder = (i64, Encoding);
pub(super) type BatchKey = (u8, u64);

/// One key's right identities in canonical `(time, sequence)` order.
#[derive(Clone, Default)]
pub(super) struct RightBucket {
    rows: Vec<(RightOrder, Option<RowPayload>)>,
}

impl RightBucket {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            rows: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&RightOrder, &Option<RowPayload>)> {
        self.rows.iter().map(|(order, payload)| (order, payload))
    }

    #[cfg(test)]
    pub fn values(&self) -> impl Iterator<Item = &Option<RowPayload>> {
        self.rows.iter().map(|(_, payload)| payload)
    }

    pub fn keys(&self) -> impl Iterator<Item = &RightOrder> {
        self.rows.iter().map(|(order, _)| order)
    }

    pub fn last_key_value(&self) -> Option<(&RightOrder, &Option<RowPayload>)> {
        self.rows.last().map(|(order, payload)| (order, payload))
    }

    pub fn contains_key(&self, order: &RightOrder) -> bool {
        self.rows
            .binary_search_by(|(current, _)| current.cmp(order))
            .is_ok()
    }

    pub fn insert(&mut self, order: RightOrder, payload: Option<RowPayload>) {
        if self.rows.last().is_none_or(|(last, _)| last < &order) {
            self.rows.push((order, payload));
            return;
        }
        match self
            .rows
            .binary_search_by(|(current, _)| current.cmp(&order))
        {
            Ok(index) => self.rows[index].1 = payload,
            Err(index) => self.rows.insert(index, (order, payload)),
        }
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&RightOrder, &mut Option<RowPayload>) -> bool) {
        self.rows
            .retain_mut(|(order, payload)| keep(order, payload));
    }

    pub fn candidate(&self, time: i64, tolerance: u64) -> Option<&RowPayload> {
        let index = self
            .rows
            .partition_point(|((right_time, _), _)| *right_time <= time);
        let ((right_time, _), payload) = self.rows.get(index.checked_sub(1)?)?;
        if i128::from(*right_time) < i128::from(time) - i128::from(tolerance) {
            return None;
        }
        payload.as_ref()
    }
}

impl<'a> IntoIterator for &'a RightBucket {
    type Item = &'a (RightOrder, Option<RowPayload>);
    type IntoIter = std::slice::Iter<'a, (RightOrder, Option<RowPayload>)>;

    fn into_iter(self) -> Self::IntoIter {
        self.rows.iter()
    }
}

impl FromIterator<(RightOrder, Option<RowPayload>)> for RightBucket {
    fn from_iter<T: IntoIterator<Item = (RightOrder, Option<RowPayload>)>>(iter: T) -> Self {
        let mut bucket = Self::new();
        for (order, payload) in iter {
            bucket.insert(order, payload);
        }
        bucket
    }
}

pub(super) struct PayloadBatch {
    pub key: BatchKey,
    pub record: Arc<RecordBatch>,
    pub encoded: OnceLock<StateSegment>,
    pub encoded_charge_bytes: u64,
    pub body_bytes: u64,
}

impl PayloadBatch {
    pub fn has_encoded(&self) -> bool {
        self.encoded.get().is_some()
    }

    pub fn ensure_encoded(&self, limit: usize, name: &str) -> Result<&StateSegment> {
        if let Some(encoded) = self.encoded.get() {
            return Ok(encoded);
        }
        let capacity = usize::try_from(self.encoded_charge_bytes).map_err(|_| {
            super::reason(
                name,
                crate::StreamingFailureReason::AsofCounterOverflow,
                "ASOF payload IPC bound exceeds platform size",
            )
        })?;
        let bytes = super::codec::encode_batch_preallocated(&self.record, capacity, limit)?;
        let body = super::codec::payload_body_bytes(&bytes)?;
        if bytes.capacity() as u64 > self.encoded_charge_bytes || body > self.body_bytes {
            return Err(super::reason(
                name,
                crate::StreamingFailureReason::AsofStateLimitExceeded,
                "ASOF payload IPC exceeds its retained upper bound",
            ));
        }
        let _ = self.encoded.set(StateSegment::new(bytes));
        Ok(self.encoded.get().expect("encoded ASOF payload"))
    }
}

#[derive(Clone)]
pub(super) struct RowPayload {
    pub batch: Arc<PayloadBatch>,
    pub row: usize,
}

#[derive(Clone, Default)]
pub(super) struct State {
    pub left: BTreeMap<LeftOrder, RowPayload>,
    pub right: BTreeMap<Encoding, RightBucket>,
    pub batches: BTreeMap<BatchKey, (Arc<PayloadBatch>, usize)>,
    pub right_payload_min: Option<i64>,
    pub right_identity_min: Option<i64>,
}

#[derive(Default)]
struct AdmissionSeen {
    buckets: BTreeSet<Encoding>,
    batches: BTreeSet<BatchKey>,
}

impl State {
    /// Rebuild the derived minima after decoding a checkpoint index.
    pub fn rebuild_right_minima(&mut self) {
        self.right_payload_min = None;
        self.right_identity_min = None;
        for bucket in self.right.values() {
            for ((time, _), row) in bucket {
                let minimum = if row.is_some() {
                    &mut self.right_payload_min
                } else {
                    &mut self.right_identity_min
                };
                *minimum = Some(minimum.map_or(*time, |previous| previous.min(*time)));
            }
        }
    }

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
        let mut seen = AdmissionSeen::default();
        for row in rows {
            self.charge_admission_row(&mut current, side, row, &mut seen, name)?;
        }
        Ok(current)
    }

    fn charge_admission_row(
        &self,
        inventory: &mut Inventory,
        side: usize,
        row: &(LeftOrder, RowPayload),
        seen: &mut AdmissionSeen,
        name: &str,
    ) -> Result<()> {
        let ((_, key, sequence), payload) = row;
        if side == 0 {
            inventory.charge_left(key, sequence, payload, name)?;
        } else {
            self.charge_right_admission(inventory, key, sequence, payload, seen, name)?;
        }
        if seen.batches.insert(payload.batch.key) {
            inventory.bytes =
                super::checked(name, inventory.bytes, batch_allocation(&payload.batch))?;
        }
        Ok(())
    }

    fn charge_right_admission(
        &self,
        inventory: &mut Inventory,
        key: &Encoding,
        sequence: &Encoding,
        payload: &RowPayload,
        seen: &mut AdmissionSeen,
        name: &str,
    ) -> Result<()> {
        if !self.right.contains_key(key) && seen.buckets.insert(key.clone()) {
            inventory.charge_allocation(key, name)?;
        }
        inventory.charge_right(sequence, Some(payload), name)
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
        self.right.get(key)?.candidate(time, tolerance)
    }
}

/// Canonical row encodings for one identity column set. Single non-null i64
/// and u64 columns encode directly; other shapes use one Arrow `RowConverter`
/// per batch. Both paths produce Arrow 58 row bytes independently per row.
pub(super) enum EncodedColumns {
    Rows(Rows),
    Int64(Int64Array),
    UInt64(UInt64Array),
}

impl EncodedColumns {
    pub(super) fn with_row<R>(&self, row: usize, use_bytes: impl FnOnce(&[u8]) -> R) -> R {
        match self {
            Self::Rows(rows) => use_bytes(rows.row(row).as_ref()),
            Self::Int64(column) => {
                let mut encoded = [0_u8; 9];
                encoded[0] = 1;
                encoded[1..].copy_from_slice(&column.value(row).to_be_bytes());
                encoded[1] ^= 0x80;
                use_bytes(&encoded)
            }
            Self::UInt64(column) => {
                let mut encoded = [0_u8; 9];
                encoded[0] = 1;
                encoded[1..].copy_from_slice(&column.value(row).to_be_bytes());
                use_bytes(&encoded)
            }
        }
    }

    /// Returns the owned encoding of one row.
    pub(super) fn row(&self, row: usize) -> Encoding {
        self.with_row(row, Encoding::from_slice)
    }

    #[cfg(test)]
    pub(super) fn is_typed(&self) -> bool {
        !matches!(self, Self::Rows(_))
    }
}

/// Resolves each named column once, then selects the typed scalar or generic
/// batch converter path.
pub(super) fn encode_columns(batch: &RecordBatch, names: &[String]) -> Result<EncodedColumns> {
    if let [name] = names {
        let index = batch
            .schema()
            .index_of(name)
            .map_err(|error| super::arrow_error(&error))?;
        let column = batch.column(index);
        if column.null_count() == 0 {
            if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
                return Ok(EncodedColumns::Int64(array.clone()));
            }
            if let Some(array) = column.as_any().downcast_ref::<UInt64Array>() {
                return Ok(EncodedColumns::UInt64(array.clone()));
            }
        }
    }
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
    Ok(EncodedColumns::Rows(rows))
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

#[derive(Default)]
pub(super) struct EvictionPreview {
    pub evicted_payloads: u64,
    pub removed_identities: u64,
    pub added_identity_only: u64,
    pub removed_identity_only: u64,
    pub released_state_bytes: u64,
    pub removed_index_bytes: u64,
}

struct EvictionConditions<'a> {
    status: &'a super::StreamAsofJoinStatus,
    tolerance: u64,
    threshold: i128,
}

fn preview_right_row(
    preview: &mut EvictionPreview,
    removed_batch_refs: &mut BTreeMap<BatchKey, usize>,
    order: &RightOrder,
    row: Option<&RowPayload>,
    conditions: &EvictionConditions<'_>,
    name: &str,
) -> Result<bool> {
    let (expired_payload, remove) = preview_row_disposition(order.0, row, conditions);
    if expired_payload {
        let payload = row.expect("expired ASOF payload");
        preview_expired_payload(preview, removed_batch_refs, payload, remove, name)?;
    }
    if remove {
        preview_removed_identity(preview, &order.1, row, name)?;
    }
    Ok(remove)
}

fn preview_row_disposition(
    time: i64,
    row: Option<&RowPayload>,
    conditions: &EvictionConditions<'_>,
) -> (bool, bool) {
    let expired_payload =
        row.is_some() && payload_expired(time, conditions.tolerance, conditions.threshold);
    let remove = (row.is_none() || expired_payload) && identity_expired(time, conditions.status);
    (expired_payload, remove)
}

fn preview_expired_payload(
    preview: &mut EvictionPreview,
    removed_batch_refs: &mut BTreeMap<BatchKey, usize>,
    payload: &RowPayload,
    remove: bool,
    name: &str,
) -> Result<()> {
    preview.evicted_payloads = super::checked(name, preview.evicted_payloads, 1)?;
    *removed_batch_refs.entry(payload.batch.key).or_default() += 1;
    if !remove {
        preview.added_identity_only = super::checked(name, preview.added_identity_only, 1)?;
        preview.released_state_bytes = super::checked(
            name,
            preview.released_state_bytes,
            payload_allocation(payload),
        )?;
        preview.removed_index_bytes = super::checked(name, preview.removed_index_bytes, 17)?;
    }
    Ok(())
}

fn preview_removed_identity(
    preview: &mut EvictionPreview,
    sequence: &Encoding,
    row: Option<&RowPayload>,
    name: &str,
) -> Result<()> {
    preview.removed_identities = super::checked(name, preview.removed_identities, 1)?;
    if row.is_none() {
        preview.removed_identity_only = super::checked(name, preview.removed_identity_only, 1)?;
    }
    preview.released_state_bytes = super::checked(
        name,
        preview.released_state_bytes,
        right_row_charge(sequence, row),
    )?;
    preview.removed_index_bytes = super::checked(
        name,
        preview.removed_index_bytes,
        17 + sequence.len() as u64 + if row.is_some() { 17 } else { 0 },
    )?;
    Ok(())
}

fn preview_bucket(
    preview: &mut EvictionPreview,
    removed_batch_refs: &mut BTreeMap<BatchKey, usize>,
    key: &Encoding,
    bucket: &RightBucket,
    conditions: &EvictionConditions<'_>,
    name: &str,
) -> Result<()> {
    let mut survivors = 0;
    for (order, row) in bucket {
        if !preview_right_row(
            preview,
            removed_batch_refs,
            order,
            row.as_ref(),
            conditions,
            name,
        )? {
            survivors += 1;
        }
    }
    if survivors == 0 {
        preview.released_state_bytes =
            super::checked(name, preview.released_state_bytes, encoding_allocation(key))?;
        preview.removed_index_bytes =
            super::checked(name, preview.removed_index_bytes, 16 + key.len() as u64)?;
    }
    Ok(())
}

impl State {
    /// Compute all status and index deltas before the infallible sweep commits.
    pub fn preview_eviction(
        &self,
        status: &super::StreamAsofJoinStatus,
        tolerance: u64,
        name: &str,
    ) -> Result<EvictionPreview> {
        let conditions = EvictionConditions {
            status,
            tolerance,
            threshold: retention_threshold(self, status),
        };
        let mut preview = EvictionPreview::default();
        let mut removed_batch_refs = BTreeMap::<BatchKey, usize>::new();
        for (key, bucket) in &self.right {
            preview_bucket(
                &mut preview,
                &mut removed_batch_refs,
                key,
                bucket,
                &conditions,
                name,
            )?;
        }
        for (key, removed) in removed_batch_refs {
            let (batch, references) = &self.batches[&key];
            if removed == *references {
                preview.released_state_bytes =
                    super::checked(name, preview.released_state_bytes, batch_allocation(batch))?;
            }
        }
        Ok(preview)
    }

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
        for ((key, row), expected) in self.left.iter().take(keys.len()).zip(keys) {
            debug_assert_eq!(key, expected);
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
        let mut payload_min = None;
        let mut identity_min = None;
        let batches = &mut self.batches;
        self.right.retain(|_, bucket| {
            bucket.retain(|(time, _), row| {
                if payload_expired(*time, tolerance, threshold)
                    && let Some(payload) = row.take()
                {
                    evicted += 1;
                    detach_batch(batches, &payload);
                }
                let keep = row.is_some() || !identity_expired(*time, status);
                if keep {
                    let minimum = if row.is_some() {
                        &mut payload_min
                    } else {
                        &mut identity_min
                    };
                    *minimum = Some(minimum.map_or(*time, |previous: i64| previous.min(*time)));
                }
                keep
            });
            !bucket.is_empty()
        });
        self.right_payload_min = payload_min;
        self.right_identity_min = identity_min;
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
    state
        .right_payload_min
        .is_some_and(|time| payload_expired(time, tolerance, threshold))
        || state
            .right_identity_min
            .is_some_and(|time| identity_expired(time, status))
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
        + batch.encoded_charge_bytes
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

#[cfg(test)]
mod eviction_minima_tests {
    use super::*;
    use crate::EventTime;
    use datafusion::arrow::datatypes::Schema;

    #[test]
    fn minima_follow_payload_release_and_identity_expiry() {
        let mut state = State::default();
        let payload = RowPayload {
            batch: Arc::new(PayloadBatch {
                key: (1, 0),
                record: Arc::new(RecordBatch::new_empty(Arc::new(Schema::empty()))),
                encoded: OnceLock::from(StateSegment::new(Vec::new())),
                encoded_charge_bytes: 0,
                body_bytes: 0,
            }),
            row: 0,
        };
        state.attach(&payload);
        state.right.insert(
            Encoding::from_slice(&[1]),
            RightBucket::from_iter([
                ((10, Encoding::from_slice(&[1])), Some(payload)),
                ((12, Encoding::from_slice(&[2])), None),
            ]),
        );
        state.rebuild_right_minima();
        assert_eq!(state.right_payload_min, Some(10));
        assert_eq!(state.right_identity_min, Some(12));

        let mut status = super::super::StreamAsofJoinStatus::default();
        status.left.watermark_micros = Some(EventTime::from_micros(11));
        status.right.watermark_micros = Some(EventTime::from_micros(11));
        assert!(eviction_pending(&state, &status, 0));
        let preview = state.preview_eviction(&status, 0, "asof").unwrap();
        assert_eq!(preview.evicted_payloads, 1);
        assert_eq!(preview.removed_identities, 1);
        assert_eq!(preview.removed_index_bytes, 35);
        assert_eq!(state.evict(&status, 0), 1);
        assert_eq!(state.right_payload_min, None);
        assert_eq!(state.right_identity_min, Some(12));
        status.right.watermark_micros = Some(EventTime::from_micros(13));
        assert!(eviction_pending(&state, &status, 0));
        assert_eq!(state.evict(&status, 0), 0);
        assert_eq!(state.right_identity_min, None);
        assert!(!eviction_pending(&state, &status, 0));
    }
}

#[cfg(test)]
mod encoding_tests {
    use super::{Encoding, encode_columns};
    use datafusion::arrow::{
        array::{Int64Array, UInt64Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
        row::{RowConverter, SortField},
    };
    use std::sync::Arc;

    #[test]
    fn short_sequences_keep_canonical_bytes_without_heap_storage() {
        assert_eq!(size_of::<Encoding>(), 16);
        let small = Encoding::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let large = Encoding::from_slice(&[7; 32]);
        assert!(small.is_inline());
        assert!(!large.is_inline());
        assert_eq!(small.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!(small < large);
    }

    #[test]
    fn scalar_integer_identity_encoding_matches_arrow_rows() {
        let columns: Vec<(DataType, Arc<dyn datafusion::arrow::array::Array>)> = vec![
            (
                DataType::Int64,
                Arc::new(Int64Array::from(vec![i64::MIN, -1, 0, 1, i64::MAX])),
            ),
            (
                DataType::UInt64,
                Arc::new(UInt64Array::from(vec![0, 1, u64::MAX - 1, u64::MAX])),
            ),
            (
                DataType::Int64,
                Arc::new(Int64Array::from(vec![Some(1), None])),
            ),
        ];
        for (data_type, column) in columns {
            let record = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "identity",
                    data_type.clone(),
                    true,
                )])),
                vec![column.clone()],
            )
            .unwrap();
            let encoded = encode_columns(&record, &["identity".into()]).unwrap();
            assert_eq!(encoded.is_typed(), column.null_count() == 0);
            let reference = RowConverter::new(vec![SortField::new(data_type)])
                .unwrap()
                .convert_columns(&[column])
                .unwrap();
            for row in 0..record.num_rows() {
                assert_eq!(encoded.row(row).as_slice(), reference.row(row).as_ref());
            }
        }
    }
}

#[cfg(test)]
mod right_bucket_tests {
    use super::{Encoding, RightBucket};

    #[test]
    fn ordered_run_accepts_append_and_watermark_local_disorder() {
        let mut bucket = RightBucket::new();
        for time in [10, 12, 11, 9] {
            bucket.insert((time, Encoding::from_slice(&[1])), None);
        }
        assert_eq!(
            bucket
                .iter()
                .map(|((time, _), _)| *time)
                .collect::<Vec<_>>(),
            vec![9, 10, 11, 12]
        );
        assert!(bucket.contains_key(&(11, Encoding::from_slice(&[1]))));
        assert!(!bucket.contains_key(&(11, Encoding::from_slice(&[2]))));
    }
}
