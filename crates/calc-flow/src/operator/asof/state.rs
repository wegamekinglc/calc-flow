use crate::{Result, StateSegment};
use datafusion::arrow::{
    array::{Array, BinaryArray, Int64Array, LargeBinaryArray, LargeBinaryBuilder, UInt64Array},
    record_batch::RecordBatch,
    row::{RowConverter, SortField},
};
use std::{
    cmp::Ordering,
    collections::BTreeMap,
    hash::{BuildHasher, Hash, Hasher},
    ops::Deref,
    sync::{Arc, OnceLock},
};

const INLINE_ENCODING_BYTES: usize = 10;

#[cfg(test)]
thread_local! {
    static LEFT_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static IDENTITY_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn take_left_visits() -> usize {
    LEFT_VISITS.with(|visits| visits.replace(0))
}

#[cfg(test)]
pub(super) fn take_identity_probes() -> usize {
    IDENTITY_PROBES.with(|probes| probes.replace(0))
}

fn left_row_refs(row: &(LeftOrder, RowRef)) -> (&LeftOrder, &RowRef) {
    #[cfg(test)]
    LEFT_VISITS.with(|visits| visits.set(visits.get() + 1));
    (&row.0, &row.1)
}

fn left_tree_refs(row: LeftRefs<'_>) -> LeftRefs<'_> {
    #[cfg(test)]
    LEFT_VISITS.with(|visits| visits.set(visits.get() + 1));
    row
}

/// Canonical Arrow row bytes, with short scalar identities stored inline.
#[derive(Clone, Debug)]
pub(super) enum Encoding {
    Inline {
        len: u8,
        bytes: [u8; INLINE_ENCODING_BYTES],
    },
    Shared(Arc<Vec<u8>>),
    Batch {
        rows: Arc<EncodedBatch>,
        row: u32,
    },
}

pub(super) struct EncodedBatch {
    storage: EncodedBatchStorage,
}

enum EncodedBatchStorage {
    Binary(BinaryArray),
    LargeBinary(LargeBinaryArray),
}

impl std::fmt::Debug for EncodedBatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EncodedBatch")
            .finish_non_exhaustive()
    }
}

impl EncodedBatch {
    fn row(&self, row: usize) -> &[u8] {
        match &self.storage {
            EncodedBatchStorage::Binary(rows) => rows.value(row),
            EncodedBatchStorage::LargeBinary(rows) => rows.value(row),
        }
    }

    fn allocation_bytes(&self) -> usize {
        self.values().capacity() + self.offset_capacity_bytes() + 512
    }

    pub fn values(&self) -> &datafusion::arrow::buffer::Buffer {
        match &self.storage {
            EncodedBatchStorage::Binary(rows) => rows.values(),
            EncodedBatchStorage::LargeBinary(rows) => rows.values(),
        }
    }

    pub fn offset_bytes(&self) -> &[u8] {
        match &self.storage {
            EncodedBatchStorage::Binary(rows) => rows.offsets().inner().inner().as_slice(),
            EncodedBatchStorage::LargeBinary(rows) => rows.offsets().inner().inner().as_slice(),
        }
    }

    pub fn offset_capacity_bytes(&self) -> usize {
        match &self.storage {
            EncodedBatchStorage::Binary(rows) => rows.offsets().inner().inner().capacity(),
            EncodedBatchStorage::LargeBinary(rows) => rows.offsets().inner().inner().capacity(),
        }
    }

    pub fn offset_width(&self) -> usize {
        match &self.storage {
            EncodedBatchStorage::Binary(_) => 4,
            EncodedBatchStorage::LargeBinary(_) => 8,
        }
    }

    pub fn len(&self) -> usize {
        match &self.storage {
            EncodedBatchStorage::Binary(rows) => rows.len(),
            EncodedBatchStorage::LargeBinary(rows) => rows.len(),
        }
    }

    pub fn with_binary(rows: BinaryArray) -> Self {
        Self {
            storage: EncodedBatchStorage::Binary(rows),
        }
    }
    pub fn with_large_binary(rows: LargeBinaryArray) -> Self {
        Self {
            storage: EncodedBatchStorage::LargeBinary(rows),
        }
    }
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
            Self::Batch { rows, row } => rows.row(*row as usize),
        }
    }

    pub fn owner_values(&self) -> impl Iterator<Item = &[u8]> {
        let count = match self {
            Self::Batch { rows, .. } => rows.len(),
            _ => 1,
        };
        (0..count).map(move |row| match self {
            Self::Batch { rows, .. } => rows.row(row),
            _ => self.as_slice(),
        })
    }

    pub fn capacity(&self) -> usize {
        match self {
            Self::Inline { len, .. } => usize::from(*len),
            Self::Shared(bytes) => bytes.capacity(),
            // The legacy row charge counts the canonical bytes of this row.
            // The capacity ledger separately owns and charges the whole batch.
            Self::Batch { rows, row } => rows.row(*row as usize).len(),
        }
    }

    pub fn allocation(&self) -> Option<(usize, u64)> {
        match self {
            Self::Inline { .. } => None,
            Self::Shared(bytes) => {
                Some((Arc::as_ptr(bytes) as usize, bytes.capacity() as u64 + 64))
            }
            Self::Batch { rows, .. } => {
                Some((Arc::as_ptr(rows) as usize, rows.allocation_bytes() as u64))
            }
        }
    }

    pub fn owner_wire_length(&self) -> u64 {
        match self {
            Self::Inline { .. } => 0,
            Self::Shared(bytes) => 34 + bytes.len() as u64,
            Self::Batch { rows, .. } => {
                34 + rows.values().len() as u64 + rows.offset_bytes().len() as u64
            }
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
type LeftRefs<'a> = (&'a LeftOrder, &'a RowRef);
pub(super) type RightOrder = (i64, Encoding);
pub(super) type BatchKey = (u8, u64);

mod right;
pub(super) use right::RightBucket;

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

/// A row in the admission-owned batch table. The table retains each payload
/// once; these temporary indices never enter retained state or checkpoints.
#[derive(Clone, Copy)]
pub(super) struct AdmissionRef {
    pub batch_index: usize,
    pub row: usize,
}

mod payload;
pub(super) use payload::{
    PayloadPool, PayloadRemoval, PayloadView, PreparedPayloadRemoval, RowRef,
};

/// Keep the common ordered left stream contiguous. An overlapping batch
/// promotes to the tree so late and out-of-order identities retain their
/// existing ordering and duplicate semantics.
#[derive(Clone, Default)]
pub(super) struct LegacyLeftState {
    ordered: Vec<(LeftOrder, RowRef)>,
    general: BTreeMap<LeftOrder, RowRef>,
}

impl LegacyLeftState {
    pub fn len(&self) -> usize {
        self.ordered.len() + self.general.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ordered.is_empty() && self.general.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&LeftOrder, &RowRef)> {
        self.ordered
            .iter()
            .map(left_row_refs)
            .chain(self.general.iter().map(left_tree_refs))
    }

    pub fn ready_prefix_len(&self, limit: usize, frontier: Option<i64>, ended: bool) -> usize {
        if ended {
            return self.len().min(limit);
        }
        let Some(frontier) = frontier else {
            return 0;
        };
        if self.general.is_empty() {
            self.ordered
                .partition_point(|(order, _)| order.0 < frontier)
                .min(limit)
        } else {
            self.general
                .keys()
                .take(limit)
                .take_while(|order| order.0 < frontier)
                .count()
        }
    }

    pub fn first_key_value(&self) -> Option<(&LeftOrder, &RowRef)> {
        self.ordered
            .first()
            .map(|(key, payload)| (key, payload))
            .or_else(|| self.general.first_key_value())
    }

    pub fn last_key_value(&self) -> Option<(&LeftOrder, &RowRef)> {
        self.general
            .last_key_value()
            .or_else(|| self.ordered.last().map(|(key, payload)| (key, payload)))
    }

    pub fn contains_key(&self, key: &LeftOrder) -> bool {
        if self.general.is_empty() {
            self.ordered
                .binary_search_by(|(current, _)| current.cmp(key))
                .is_ok()
        } else {
            self.general.contains_key(key)
        }
    }

    #[cfg(test)]
    pub fn insert(&mut self, key: LeftOrder, payload: RowRef) {
        if self.general.is_empty() && self.ordered.last().is_none_or(|(last, _)| last < &key) {
            self.ordered.push((key, payload));
        } else {
            self.promote();
            self.general.insert(key, payload);
        }
    }

    #[cfg(test)]
    pub fn append_admission(&mut self, mut rows: Vec<(LeftOrder, RowRef)>) {
        if rows.is_empty() {
            return;
        }
        if !rows.windows(2).all(|pair| pair[0].0 < pair[1].0) {
            rows.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        }
        if self.general.is_empty()
            && self
                .ordered
                .last()
                .is_none_or(|(last, _)| last < &rows[0].0)
        {
            self.ordered.append(&mut rows);
        } else {
            self.promote();
            self.general.extend(rows);
        }
    }

    pub fn drain_prefix(&mut self, count: usize) {
        if self.general.is_empty() {
            self.ordered.drain(..count);
            // Each retained left identity is charged enough for two vector
            // slots. Compact only after crossing that threshold, avoiding a
            // full copy for every output prefix.
            if self.ordered.capacity() > self.ordered.len().saturating_mul(2) {
                self.ordered = std::mem::take(&mut self.ordered)
                    .into_boxed_slice()
                    .into_vec();
            }
        } else {
            let split_key = self.general.keys().nth(count).cloned();
            let removed = if let Some(split_key) = split_key {
                let remaining = self.general.split_off(&split_key);
                std::mem::replace(&mut self.general, remaining)
            } else {
                std::mem::take(&mut self.general)
            };
            drop(removed);
        }
    }

    #[cfg(test)]
    fn promote(&mut self) {
        if !self.ordered.is_empty() {
            self.general.extend(std::mem::take(&mut self.ordered));
        }
    }

    #[cfg(test)]
    fn is_ordered(&self) -> bool {
        self.general.is_empty()
    }
}

impl<'a> IntoIterator for &'a LegacyLeftState {
    type Item = (&'a LeftOrder, &'a RowRef);
    type IntoIter = std::iter::Chain<
        std::iter::Map<
            std::slice::Iter<'a, (LeftOrder, RowRef)>,
            fn(&(LeftOrder, RowRef)) -> (&LeftOrder, &RowRef),
        >,
        std::iter::Map<
            std::collections::btree_map::Iter<'a, LeftOrder, RowRef>,
            fn(LeftRefs<'a>) -> LeftRefs<'a>,
        >,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.ordered
            .iter()
            .map(left_row_refs as fn(&(LeftOrder, RowRef)) -> (&LeftOrder, &RowRef))
            .chain(
                self.general
                    .iter()
                    .map(left_tree_refs as fn(LeftRefs<'a>) -> LeftRefs<'a>),
            )
    }
}

mod left;
pub(super) use left::{ChunkData, LeftState, LeftView, PreparedLeftChunk, PreparedLeftDrain};

mod capacity;
pub(super) use capacity::CapacitySnapshot;
mod ownership;
pub(super) use ownership::{EncodingOwners, OwnerRemovals, OwnerUpdates};

mod sequences;
pub(super) use sequences::{SequenceColumn, SequenceKind, SequenceRef};

mod key_dictionary;
pub(super) use key_dictionary::{RightState, validate_key_count};

#[derive(Clone, Default)]
pub(super) struct State {
    pub left: LeftState,
    pub right: RightState,
    pub batches: PayloadPool,
    pub right_payload_min: Option<i64>,
    pub right_identity_min: Option<i64>,
    encoding_owners: Option<EncodingOwners>,
    pub sequence_kinds: [SequenceKind; 2],
}

impl State {
    pub fn install_encoding_owners(&mut self, updates: OwnerUpdates) {
        self.encoding_owners
            .as_mut()
            .expect("tracked native state")
            .commit_add(updates);
    }

    pub fn encoding_owner_allocation(&self) -> (u64, u64) {
        self.encoding_owners.as_ref().map_or((0, 0), |owners| {
            (owners.buffers_bytes(), owners.metadata_bytes())
        })
    }

    pub fn owners_encoded_length(&self) -> Option<u64> {
        self.encoding_owners
            .as_ref()
            .map(EncodingOwners::encoded_length)
    }
    pub fn empty_tracked() -> Self {
        Self {
            encoding_owners: Some(EncodingOwners::default()),
            ..Self::default()
        }
    }

    pub fn rebuild_encoding_owners(&mut self) {
        let mut owners = EncodingOwners::default();
        for ((_, key, sequence), _) in self.left.unordered_iter() {
            owners.attach(key);
            owners.attach(sequence.as_ref());
        }
        for (key, bucket) in &self.right {
            for ((_, sequence), _) in bucket {
                owners.attach(key);
                owners.attach(sequence.as_ref());
            }
        }
        self.encoding_owners = Some(owners);
    }

    pub fn rebuild_right_minima(&mut self) {
        self.right_payload_min = None;
        self.right_identity_min = None;
        for bucket in self.right.values() {
            if let Some(time) = bucket.payload_min() {
                self.right_payload_min = Some(
                    self.right_payload_min
                        .map_or(time, |previous| previous.min(time)),
                );
            }
            if let Some(time) = bucket.identity_min() {
                self.right_identity_min = Some(
                    self.right_identity_min
                        .map_or(time, |previous| previous.min(time)),
                );
            }
        }
    }

    #[cfg(test)]
    pub fn attach(&mut self, row: &RowPayload) -> RowRef {
        self.batches.attach(row)
    }

    pub fn commit_prepared_left_prefix(
        &mut self,
        prefix: &LeftPrefix,
        drain: PreparedLeftDrain,
        compaction: PreparedPayloadRemoval,
    ) {
        self.left
            .drain_prefix(prefix.count, &prefix.batches, &self.batches, drain);
        self.batches.defer_compaction();
        for (&key, &removed) in &prefix.batches {
            self.batches.detach_count(key, removed);
        }
        self.batches.install_compaction(compaction);
        if let Some(owners) = &mut self.encoding_owners {
            owners.remove(&prefix.owners);
        }
    }

    #[cfg(test)]
    pub fn commit_matched_left_prefix(&mut self, prefix: &LeftPrefix, drain: PreparedLeftDrain) {
        let compaction = self.batches.prepare_removal(&prefix.batches);
        self.commit_prepared_left_prefix(prefix, drain, compaction);
    }

    #[cfg(test)]
    pub fn commit_left_prefix(&mut self, count: usize) {
        let mut prefix = LeftPrefix::default();
        for (order, payload) in self.left.iter().take(count) {
            prefix
                .visit(&order, payload, &self.batches, "asof")
                .unwrap();
        }
        let drain = self
            .left
            .drain_input(&prefix.batches, &self.batches)
            .prepare();
        self.commit_matched_left_prefix(&prefix, drain);
    }

    pub fn contains_identity(&self, index: usize, identity: &LeftOrder) -> bool {
        #[cfg(test)]
        IDENTITY_PROBES.with(|probes| probes.set(probes.get() + 1));
        if index == 0 {
            self.left.contains_key(identity)
        } else {
            self.right
                .get(&identity.1)
                .is_some_and(|bucket| bucket.contains_key(&(identity.0, identity.2.clone())))
        }
    }

    pub fn candidate(&self, key: &Encoding, time: i64, tolerance: u64) -> Option<&RowRef> {
        self.right.get(key)?.candidate(time, tolerance)
    }
}

/// Canonical row encodings for one identity column set. Single non-null i64
/// and u64 columns encode directly; other shapes use one Arrow `RowConverter`
/// per batch. Both paths produce Arrow 58 row bytes independently per row.
pub(super) enum EncodedColumns {
    Rows(Arc<EncodedBatch>),
    Binary(Arc<EncodedBatch>),
    Int64(Int64Array),
    UInt64(UInt64Array),
    Integer(IntegerColumn),
    StringKeys {
        rows: Arc<EncodedBatch>,
        ids: Vec<u32>,
    },
}

mod string_keys;

pub(super) struct IntegerColumn {
    values: datafusion::arrow::buffer::Buffer,
    offset: usize,
    width: usize,
    signed: bool,
}

fn signed_width(data_type: &datafusion::arrow::datatypes::DataType) -> Option<usize> {
    use datafusion::arrow::datatypes::DataType;
    match data_type {
        DataType::Int8 => Some(1),
        DataType::Int16 => Some(2),
        DataType::Int32 => Some(4),
        DataType::Int64 => Some(8),
        _ => None,
    }
}

fn unsigned_width(data_type: &datafusion::arrow::datatypes::DataType) -> Option<usize> {
    use datafusion::arrow::datatypes::DataType;
    match data_type {
        DataType::UInt8 => Some(1),
        DataType::UInt16 => Some(2),
        DataType::UInt32 => Some(4),
        DataType::UInt64 => Some(8),
        _ => None,
    }
}

impl IntegerColumn {
    fn from_array(array: &dyn Array) -> Option<Self> {
        let signed = signed_width(array.data_type());
        let width = signed.or_else(|| unsigned_width(array.data_type()))?;
        let data = array.to_data();
        Some(Self {
            values: data.buffers()[0].clone(),
            offset: data.offset() * width,
            width,
            signed: signed.is_some(),
        })
    }

    fn with_row<R>(&self, row: usize, use_bytes: impl FnOnce(&[u8]) -> R) -> R {
        let mut encoded = [0_u8; 9];
        encoded[0] = 1;
        let start = self.offset + row * self.width;
        encoded[1..=self.width].copy_from_slice(&self.values.as_slice()[start..start + self.width]);
        #[cfg(target_endian = "little")]
        encoded[1..=self.width].reverse();
        if self.signed {
            encoded[1] ^= 0x80;
        }
        use_bytes(&encoded[..=self.width])
    }
}

impl EncodedColumns {
    pub(super) fn with_row<R>(&self, row: usize, use_bytes: impl FnOnce(&[u8]) -> R) -> R {
        match self {
            Self::Integer(column) => column.with_row(row, use_bytes),
            Self::Rows(rows) | Self::Binary(rows) => use_bytes(rows.row(row)),
            Self::StringKeys { rows, ids } => use_bytes(rows.row(ids[row] as usize)),
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
        match self {
            Self::StringKeys { rows, ids } => {
                let id = ids[row];
                if Encoding::fits_inline(rows.row(id as usize)) {
                    Encoding::from_slice(rows.row(id as usize))
                } else {
                    Encoding::Batch {
                        rows: rows.clone(),
                        row: id,
                    }
                }
            }
            Self::Rows(rows) | Self::Binary(rows) if !Encoding::fits_inline(rows.row(row)) => {
                Encoding::Batch {
                    rows: rows.clone(),
                    row: u32::try_from(row).expect("preflighted encoding batch"),
                }
            }
            _ => self.with_row(row, Encoding::from_slice),
        }
    }

    pub(super) fn hashes(
        &self,
        seed: &datafusion::common::hash_utils::RandomState,
        count: usize,
    ) -> Result<Vec<u64>> {
        if let Self::StringKeys { rows, ids } = self {
            let values = match rows.storage {
                EncodedBatchStorage::Binary(_) => Self::Binary(rows.clone()),
                EncodedBatchStorage::LargeBinary(_) => Self::Rows(rows.clone()),
            };
            let unique = values.hashes(seed, rows.len())?;
            return Ok(ids[..count].iter().map(|id| unique[*id as usize]).collect());
        }
        let mut hashes = vec![0; count];
        if let Self::Binary(rows) = self {
            let EncodedBatchStorage::Binary(binary) = &rows.storage else {
                unreachable!("binary ASOF row storage")
            };
            datafusion::common::hash_utils::create_hashes(
                [binary as &dyn Array],
                seed,
                &mut hashes,
            )
            .map_err(|error| crate::CalcFlowError::Format {
                message: format!("ASOF canonical key hashing failed: {error}"),
            })?;
        } else {
            for (row, hash) in hashes.iter_mut().enumerate() {
                *hash = self.with_row(row, |bytes| seed.hash_one(bytes));
            }
        }
        Ok(hashes)
    }

    #[cfg(test)]
    pub(super) fn is_typed(&self) -> bool {
        matches!(
            self,
            Self::Int64(_) | Self::UInt64(_) | Self::Integer(_) | Self::StringKeys { .. }
        )
    }
}

fn scalar_column(batch: &RecordBatch, names: &[String]) -> Result<Option<EncodedColumns>> {
    if let [name] = names {
        let index = batch
            .schema()
            .index_of(name)
            .map_err(|error| super::arrow_error(&error))?;
        let column = batch.column(index);
        if column.null_count() == 0 {
            if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
                return Ok(Some(EncodedColumns::Int64(array.clone())));
            }
            if let Some(array) = column.as_any().downcast_ref::<UInt64Array>() {
                return Ok(Some(EncodedColumns::UInt64(array.clone())));
            }
            if let Some(integer) = IntegerColumn::from_array(column.as_ref()) {
                return Ok(Some(EncodedColumns::Integer(integer)));
            }
        }
    }
    Ok(None)
}

/// Resolves each named column once, then selects the typed scalar or generic
/// batch converter path.
pub(super) fn encode_columns(batch: &RecordBatch, names: &[String]) -> Result<EncodedColumns> {
    if let Some(column) = scalar_column(batch, names)? {
        return Ok(column);
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
    encode_row_columns(&arrays)
}

fn encode_row_columns(arrays: &[datafusion::arrow::array::ArrayRef]) -> Result<EncodedColumns> {
    let fields = arrays
        .iter()
        .map(|array| SortField::new(array.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields).map_err(|error| super::arrow_error(&error))?;
    let rows = converter
        .convert_columns(arrays)
        .map_err(|error| super::arrow_error(&error))?;
    if i32::try_from(rows.size()).is_ok() {
        let binary = rows
            .try_into_binary()
            .map_err(|error| super::arrow_error(&error))?;
        Ok(EncodedColumns::Binary(Arc::new(EncodedBatch {
            storage: EncodedBatchStorage::Binary(binary),
        })))
    } else {
        // The admission reservation covers both buffers during the wide-offset
        // conversion. Retained batches own plain Arrow binary columns.
        let bytes = rows.lengths().sum();
        let mut builder = LargeBinaryBuilder::with_capacity(rows.num_rows(), bytes);
        for row in &rows {
            builder.append_value(row.data());
        }
        Ok(EncodedColumns::Rows(Arc::new(EncodedBatch {
            storage: EncodedBatchStorage::LargeBinary(builder.finish()),
        })))
    }
}

/// Read a single non-null string key through its typed Arrow array, encode
/// distinct values once, and address them with compact dictionary ids.
pub(super) fn encode_key_columns<R>(
    batch: &RecordBatch,
    names: &[String],
    reserve: impl FnOnce(u64) -> Result<R>,
) -> Result<EncodedColumns> {
    if let [name] = names {
        let index = batch
            .schema()
            .index_of(name)
            .map_err(|error| super::arrow_error(&error))?;
        if let Some(keys) = string_keys::encode(batch.column(index), reserve)? {
            return Ok(keys);
        }
    }
    encode_columns(batch, names)
}

#[cfg(test)]
pub(super) fn encoded_columns(
    batch: &RecordBatch,
    row: usize,
    names: &[String],
) -> Result<Encoding> {
    Ok(encode_columns(batch, names)?.row(row))
}

/// Ordered-map node and inline headroom charged per left identity, on top of
/// the key, sequence and payload buffer allocations.
const LEFT_IDENTITY_SLOT_BYTES: usize = 256 + 64 + 64;
const LEFT_IDENTITY_BYTES: u64 = LEFT_IDENTITY_SLOT_BYTES as u64;
const _: () = assert!(2 * size_of::<(LeftOrder, RowRef)>() <= LEFT_IDENTITY_SLOT_BYTES);
/// Ordered-map node charged per right identity; the bucket key allocation is
/// charged separately, once per bucket.
#[cfg(test)]
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

/// Matching visits each selected left identity once, accumulating all
/// infallible post-delivery changes before the sink accepts the output.
#[derive(Default)]
pub(super) struct LeftPrefix {
    pub count: usize,
    pub batches: BTreeMap<BatchKey, usize>,
    pub keys: std::collections::HashMap<(BatchKey, Encoding), usize, ahash::RandomState>,
    pub owners: OwnerRemovals,
    pub sequence_owners: BTreeMap<BatchKey, OwnerRemovals>,
}

impl LeftPrefix {
    #[cfg(test)]
    pub fn visit(
        &mut self,
        order: &LeftView<'_>,
        row: RowRef,
        batches: &PayloadPool,
        name: &str,
    ) -> Result<()> {
        self.visit_owners(order.1, Some(order.2.as_ref()), batches.key(row), name)
    }

    pub fn visit_owners(
        &mut self,
        key: &Encoding,
        sequence: Option<&Encoding>,
        batch: BatchKey,
        name: &str,
    ) -> Result<()> {
        self.count = self.count.checked_add(1).ok_or_else(|| {
            super::reason(
                name,
                crate::StreamingFailureReason::AsofCounterOverflow,
                "ASOF prefix row count overflowed",
            )
        })?;
        *self.batches.entry(batch).or_default() += 1;
        *self.keys.entry((batch, key.clone())).or_default() += 1;
        EncodingOwners::record_remove(&mut self.owners, key, 1);
        if let Some(sequence) = sequence {
            EncodingOwners::record_remove(&mut self.owners, sequence, 1);
        }
        if let Some(sequence) = sequence.filter(|sequence| sequence.allocation().is_some()) {
            EncodingOwners::record_remove(
                self.sequence_owners.entry(batch).or_default(),
                sequence,
                1,
            );
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct EvictionPreview {
    pub evicted_payloads: u64,
    pub removed_identities: u64,
    pub added_identity_only: u64,
    pub removed_identity_only: u64,
    pub owners: OwnerRemovals,
    pub batches: BTreeMap<BatchKey, usize>,
    #[cfg(test)]
    pub visited_rows: usize,
}

struct EvictionConditions<'a> {
    batches: &'a PayloadPool,
    status: &'a super::StreamAsofJoinStatus,
    tolerance: u64,
    threshold: i128,
}

fn preview_right_row(
    preview: &mut EvictionPreview,
    removed_batch_refs: &mut BTreeMap<BatchKey, usize>,
    time: i64,
    row: Option<&RowRef>,
    conditions: &EvictionConditions<'_>,
    name: &str,
) -> Result<bool> {
    #[cfg(test)]
    {
        preview.visited_rows += 1;
    }
    let (expired_payload, remove) = preview_row_disposition(time, row, conditions);
    if expired_payload {
        let payload = row.expect("expired ASOF payload");
        preview_expired_payload(
            preview,
            removed_batch_refs,
            *payload,
            conditions.batches,
            remove,
            name,
        )?;
    }
    if remove {
        preview_removed_identity(preview, row, name)?;
    }
    Ok(remove)
}

fn preview_row_disposition(
    time: i64,
    row: Option<&RowRef>,
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
    payload: RowRef,
    batches: &PayloadPool,
    remove: bool,
    name: &str,
) -> Result<()> {
    preview.evicted_payloads = super::checked(name, preview.evicted_payloads, 1)?;
    *removed_batch_refs.entry(batches.key(payload)).or_default() += 1;
    if !remove {
        preview.added_identity_only = super::checked(name, preview.added_identity_only, 1)?;
    }
    Ok(())
}

fn preview_removed_identity(
    preview: &mut EvictionPreview,
    row: Option<&RowRef>,
    name: &str,
) -> Result<()> {
    preview.removed_identities = super::checked(name, preview.removed_identities, 1)?;
    if row.is_none() {
        preview.removed_identity_only = super::checked(name, preview.removed_identity_only, 1)?;
    }
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
    let mut survivors = bucket.len();
    for (time, sequence_owner, row) in bucket.expired_rows(
        conditions.status,
        conditions.tolerance,
        conditions.threshold,
    ) {
        if preview_right_row(preview, removed_batch_refs, time, row, conditions, name)? {
            if let Some(sequence) = sequence_owner {
                EncodingOwners::record_remove(&mut preview.owners, sequence, 1);
            }
            survivors -= 1;
        }
    }
    if survivors < bucket.len() {
        EncodingOwners::record_remove(&mut preview.owners, key, bucket.len() - survivors);
    }
    Ok(())
}

impl State {
    fn scan_encoding_owners(&self) -> EncodingOwners {
        let mut scanned = EncodingOwners::default();
        for ((_, key, sequence), _) in self.left.unordered_iter() {
            scanned.attach(key);
            scanned.attach(sequence.as_ref());
        }
        for (key, bucket) in &self.right {
            for ((_, sequence), _) in bucket {
                scanned.attach(key);
                scanned.attach(sequence.as_ref());
            }
        }
        scanned
    }

    fn payload_capacity_bytes(&self, name: &str) -> Result<u64> {
        let mut bytes = 0;
        for (batch, references) in self.batches.values() {
            if *references == 0 {
                return Err(super::reason(
                    name,
                    crate::StreamingFailureReason::AsofProtocolError,
                    "ASOF retained an unreferenced payload batch",
                ));
            }
            bytes = super::checked(name, bytes, capacity_batch_allocation(batch, name)?)?;
        }
        Ok(bytes)
    }

    fn index_capacity_inventory(&self, name: &str) -> Result<Inventory> {
        let mut total = Inventory {
            identities: self.left.len() as u64,
            bytes: self.left.capacity_bytes(name)?,
            ..Inventory::default()
        };
        total.bytes = super::checked(name, total.bytes, self.right.metadata_bytes())?;
        let fallback;
        let owners = if let Some(owners) = &self.encoding_owners {
            owners
        } else {
            let scanned = self.scan_encoding_owners();
            fallback = scanned;
            &fallback
        };
        total.bytes = super::checked(name, total.bytes, owners.allocation_bytes())?;
        for bucket in self.right.values() {
            total = add_right_inventory(total, bucket, name)?;
        }
        Ok(total)
    }

    pub fn capacity_inventory(
        &self,
        prepared: Option<&super::checkpoint::PreparedSegment>,
        name: &str,
    ) -> Result<Inventory> {
        let mut total = self.index_capacity_inventory(name)?;
        total.bytes = super::checked(name, total.bytes, self.batches.metadata_bytes())?;
        total.bytes = super::checked(name, total.bytes, self.payload_capacity_bytes(name)?)?;
        if let Some(prepared) = prepared {
            total.bytes = super::checked(name, total.bytes, prepared.capacity() as u64 + 256)?;
        }
        Ok(total)
    }

    /// Bound both batch-reference counts and removal entries for every owned
    /// encoding allocation. Identity-only history can still own encodings
    /// when no payload batch remains.
    pub fn eviction_workspace_bytes(&self, name: &str) -> Result<u64> {
        let batches = (self.batches.len() as u64).checked_mul(96).ok_or_else(|| {
            super::reason(
                name,
                crate::StreamingFailureReason::AsofCounterOverflow,
                "ASOF eviction workspace overflowed",
            )
        })?;
        let batches = if batches == 0 {
            0
        } else {
            super::checked(name, batches, 256)?
        };
        let owners = if let Some(owners) = &self.encoding_owners {
            owners.metadata_bytes()
        } else {
            // Only private untracked fixtures use this conservative fallback;
            // native admission and restore always install the owner ledger.
            let rows = self.right.values().map(RightBucket::len).sum::<usize>() as u64;
            if rows == 0 {
                0
            } else {
                rows.saturating_mul(256).saturating_add(384)
            }
        };
        super::checked(name, batches, owners)
    }

    /// Compute all status and index deltas before the infallible sweep commits.
    pub fn preview_eviction(
        &self,
        status: &super::StreamAsofJoinStatus,
        tolerance: u64,
        name: &str,
    ) -> Result<EvictionPreview> {
        let conditions = EvictionConditions {
            batches: &self.batches,
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
        preview.batches = removed_batch_refs;
        Ok(preview)
    }

    /// Charge each retained Arrow batch once, alongside its row indexes.
    #[cfg(test)]
    pub fn inventory(
        &self,
        prepared: Option<&super::checkpoint::PreparedSegment>,
        name: &str,
    ) -> Result<Inventory> {
        let mut total = Inventory::default();
        self.left
            .unordered_iter()
            .try_for_each(|((_, key, sequence), row)| {
                total.charge_left(key, sequence.as_ref(), &row, name)
            })?;
        for (key, bucket) in &self.right {
            total.charge_allocation(key, name)?;
            for ((_, sequence), row) in bucket {
                total.charge_right(sequence.as_ref(), row, name)?;
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

    #[cfg(test)]
    pub fn evict(&mut self, status: &super::StreamAsofJoinStatus, tolerance: u64) -> u64 {
        let preview = self.preview_eviction(status, tolerance, "asof").unwrap();
        let compaction = self.batches.prepare_removal(&preview.batches);
        self.evict_prepared(status, tolerance, compaction, &preview)
    }

    /// Release expired columns and commit the preflighted batch and encoding
    /// owner removals once per owner. No sequence reconstruction is needed
    /// for integer identities that disappear at this boundary.
    pub fn evict_prepared(
        &mut self,
        status: &super::StreamAsofJoinStatus,
        tolerance: u64,
        compaction: PreparedPayloadRemoval,
        preview: &EvictionPreview,
    ) -> u64 {
        let threshold = retention_threshold(self, status);
        let mut evicted = 0;
        let mut payload_min = None;
        let mut identity_min = None;
        self.batches.defer_compaction();
        self.right.retain(|_, bucket| {
            if bucket.eviction_pending(status, tolerance, threshold) {
                evicted += Arc::make_mut(bucket).evict(status, tolerance, threshold);
            }
            if let Some(time) = bucket.payload_min() {
                payload_min = Some(payload_min.map_or(time, |previous: i64| previous.min(time)));
            }
            if let Some(time) = bucket.identity_min() {
                identity_min = Some(identity_min.map_or(time, |previous: i64| previous.min(time)));
            }
            !bucket.is_empty()
        });
        self.right_payload_min = payload_min;
        self.right_identity_min = identity_min;
        for (&batch, &removed) in &preview.batches {
            self.batches.detach_count(batch, removed);
        }
        if let Some(owners) = &mut self.encoding_owners {
            owners.remove(&preview.owners);
        }
        self.batches.install_compaction(compaction);
        evicted
    }
}

/// Retention threshold shared by `evict` and `eviction_pending`: the more
/// conservative of the left frontier and the oldest pending left row.
pub(super) fn retention_threshold(state: &State, status: &super::StreamAsofJoinStatus) -> i128 {
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
        .map_or(i128::MAX, |(key, _)| i128::from(*key.0));
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

fn encoding_allocation(bytes: &Encoding) -> u64 {
    ALLOCATION_BYTES + bytes.capacity() as u64
}

fn payload_allocation<T>(_row: &T) -> u64 {
    64
}

#[cfg(test)]
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

pub(super) fn capacity_batch_allocation(batch: &PayloadBatch, name: &str) -> Result<u64> {
    let encoded = batch
        .encoded
        .get()
        .map_or(batch.encoded_charge_bytes, |segment| {
            batch
                .encoded_charge_bytes
                .max(segment.bytes_arc().capacity() as u64)
        });
    super::checked(
        name,
        super::checked(name, encoded, batch.body_bytes)?,
        512 + batch.record.num_columns() as u64 * 256,
    )
}

#[cfg(test)]
fn prepared_allocation(prepared: &super::checkpoint::PreparedSegment) -> u64 {
    ALLOCATION_BYTES + prepared.capacity() as u64
}

fn add_right_inventory(
    mut total: Inventory,
    bucket: &RightBucket,
    name: &str,
) -> Result<Inventory> {
    total.identities = super::checked(name, total.identities, bucket.len() as u64)?;
    total.right_payloads = super::checked(name, total.right_payloads, bucket.payload_len() as u64)?;
    total.identity_only = super::checked(
        name,
        total.identity_only,
        (bucket.len() - bucket.payload_len()) as u64,
    )?;
    Ok(total)
}

fn left_row_charge<T>(key: &Encoding, sequence: &Encoding, row: &T) -> u64 {
    LEFT_IDENTITY_BYTES
        + encoding_allocation(key)
        + encoding_allocation(sequence)
        + payload_allocation(row)
}

#[cfg(test)]
fn right_row_charge<T>(sequence: &Encoding, row: Option<&T>) -> u64 {
    RIGHT_IDENTITY_BYTES + encoding_allocation(sequence) + row.map_or(0, payload_allocation)
}

#[cfg(test)]
impl Inventory {
    fn charge_left<T>(
        &mut self,
        key: &Encoding,
        sequence: &Encoding,
        row: &T,
        name: &str,
    ) -> Result<()> {
        self.identities = super::checked(name, self.identities, 1)?;
        self.bytes = super::checked(name, self.bytes, left_row_charge(key, sequence, row))?;
        Ok(())
    }

    fn charge_right<T>(&mut self, sequence: &Encoding, row: Option<&T>, name: &str) -> Result<()> {
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
    fn eviction_preview_visits_only_expired_rows() {
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
        let mut bucket = RightBucket::new();
        for time in 0..1_000 {
            let reference = state.attach(&payload);
            bucket.insert((time, Encoding::from_slice(&[1])), Some(reference));
        }
        state.right.insert(Encoding::from_slice(&[1]), bucket);
        let mut status = super::super::StreamAsofJoinStatus::default();
        status.left.watermark_micros = Some(EventTime::from_micros(1));
        status.right.watermark_micros = Some(EventTime::from_micros(1));
        let preview = state.preview_eviction(&status, 0, "asof").unwrap();
        assert_eq!(preview.evicted_payloads, 1);
        assert_eq!(preview.visited_rows, 1);
    }

    #[test]
    fn stalled_identity_history_is_skipped_until_its_own_frontier_closes() {
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
        let payload = state.attach(&payload);
        let mut bucket = RightBucket::new();
        for time in 0..1_000 {
            bucket.insert((time, Encoding::from_slice(&[1])), None);
        }
        bucket.insert((1_000, Encoding::from_slice(&[1])), Some(payload));
        state.right.insert(Encoding::from_slice(&[1]), bucket);
        let mut status = super::super::StreamAsofJoinStatus::default();
        status.left.watermark_micros = Some(EventTime::from_micros(1_001));
        status.right.watermark_micros = Some(EventTime::from_micros(0));
        let preview = state.preview_eviction(&status, 0, "asof").unwrap();
        assert_eq!(preview.visited_rows, 1);
        assert_eq!(preview.added_identity_only, 1);
        assert_eq!(preview.removed_identities, 0);
        assert_eq!(state.evict(&status, 0), 1);
        assert_eq!(state.right_identity_min, Some(0));
        assert!(state.batches.is_empty());

        status.right.watermark_micros = Some(EventTime::from_micros(1_000));
        let preview = state.preview_eviction(&status, 0, "asof").unwrap();
        assert_eq!(preview.visited_rows, 1_000);
        assert_eq!(preview.removed_identity_only, 1_000);
        assert_eq!(state.evict(&status, 0), 0);
        assert_eq!(state.right_identity_min, Some(1_000));
        assert_eq!(state.right.values().next().unwrap().len(), 1);
    }

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
        let payload = state.attach(&payload);
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
        assert_eq!(preview.batches.get(&(1, 0)), Some(&1));
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
    use super::{EncodedColumns, Encoding, encode_columns, encode_key_columns};
    use datafusion::arrow::{
        array::{Int64Array, UInt64Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
        row::{RowConverter, SortField},
    };
    use std::sync::Arc;

    #[test]
    fn generic_sequence_rows_share_one_batch_allocation() {
        use datafusion::arrow::array::StringArray;

        let column = Arc::new(StringArray::from(vec!["long sequence identity"; 10_000]));
        let record = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "identity",
                DataType::Utf8,
                false,
            )])),
            vec![column.clone()],
        )
        .unwrap();
        let encoded = encode_columns(&record, &["identity".into()]).unwrap();
        let mut retained = Vec::new();
        let allocation = allocation_counter::measure(|| {
            retained = (0..record.num_rows()).map(|row| encoded.row(row)).collect();
        });
        assert!(
            allocation.count_total <= 2,
            "generic sequences allocated separately per row: {allocation:?}"
        );
        let reference = RowConverter::new(vec![SortField::new(DataType::Utf8)])
            .unwrap()
            .convert_columns(&[column])
            .unwrap();
        drop(encoded);
        for (row, identity) in retained.iter().enumerate() {
            assert_eq!(identity.as_slice(), reference.row(row).as_ref());
        }
    }

    #[test]
    fn retained_sequence_capacity_is_charged_after_large_identity_expires() {
        use super::{RightBucket, State};
        use crate::EventTime;
        use datafusion::arrow::array::StringArray;

        let long = "x".repeat(128 * 1024);
        let column = Arc::new(StringArray::from(vec![long.as_str(), "survivor sequence"]));
        let record = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "identity",
                DataType::Utf8,
                false,
            )])),
            vec![column],
        )
        .unwrap();
        let mut state = State::default();
        let measured = allocation_counter::measure(|| {
            let encoded = encode_columns(&record, &["identity".into()]).unwrap();
            let mut bucket = RightBucket::new();
            for time in 0..2 {
                bucket.insert((time, encoded.row(usize::try_from(time).unwrap())), None);
            }
            state.right.insert(Encoding::from_slice(&[1]), bucket);
            drop(encoded);
            let mut status = super::super::StreamAsofJoinStatus::default();
            status.right.watermark_micros = Some(EventTime::from_micros(1));
            state.evict(&status, 0);
        });
        assert_eq!(state.right.values().next().unwrap().len(), 1);
        let charged = state.capacity_inventory(None, "asof").unwrap().bytes;
        assert!(
            charged >= u64::try_from(measured.bytes_current).unwrap(),
            "one identity stopped funding its retained sequence batch: {charged}, {measured:?}"
        );
    }

    #[test]
    fn short_sequences_keep_canonical_bytes_without_heap_storage() {
        assert!(size_of::<Encoding>() <= 16);
        let small = Encoding::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let large = Encoding::from_slice(&[7; 32]);
        assert!(small.is_inline());
        assert!(!large.is_inline());
        assert_eq!(small.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!(small < large);
    }

    #[test]
    fn scalar_integer_identity_encoding_matches_arrow_rows() {
        use datafusion::arrow::array::{
            Int8Array, Int16Array, Int32Array, UInt8Array, UInt16Array, UInt32Array,
        };
        let columns: Vec<(DataType, Arc<dyn datafusion::arrow::array::Array>)> = vec![
            (
                DataType::Int8,
                Arc::new(Int8Array::from(vec![i8::MIN, -1, 0, i8::MAX])),
            ),
            (
                DataType::Int16,
                Arc::new(Int16Array::from(vec![i16::MIN, -1, 0, i16::MAX])),
            ),
            (
                DataType::Int32,
                Arc::new(Int32Array::from(vec![i32::MIN, -1, 0, i32::MAX])),
            ),
            (
                DataType::UInt8,
                Arc::new(UInt8Array::from(vec![0, 1, u8::MAX])),
            ),
            (
                DataType::UInt16,
                Arc::new(UInt16Array::from(vec![0, 1, u16::MAX])),
            ),
            (
                DataType::UInt32,
                Arc::new(UInt32Array::from(vec![0, 1, u32::MAX])),
            ),
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

    #[test]
    fn typed_string_key_dictionary_preserves_sliced_rows_and_canonical_hashes() {
        use datafusion::arrow::array::{ArrayRef, LargeStringArray, StringArray};
        use std::hash::BuildHasher;
        let long = "long-key-".repeat(32);
        let values = vec![
            "ignored",
            long.as_str(),
            long.as_str(),
            "",
            "é\0文",
            "é\0文",
            "ignored",
        ];
        let cases: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(values.clone()).slice(1, 5)),
            Arc::new(LargeStringArray::from(values).slice(1, 5)),
            Arc::new(StringArray::from(vec![Some("k"), None])),
        ];
        let seed = datafusion::common::hash_utils::RandomState::with_seed(123);
        for column in cases {
            let record = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "key",
                    column.data_type().clone(),
                    true,
                )])),
                vec![column.clone()],
            )
            .unwrap();
            let encoded = encode_key_columns(&record, &["key".into()], |_| Ok(())).unwrap();
            assert_eq!(encoded.is_typed(), column.null_count() == 0);
            let expected = RowConverter::new(vec![SortField::new(column.data_type().clone())])
                .unwrap()
                .convert_columns(&[column])
                .unwrap();
            let hashes = encoded.hashes(&seed, record.num_rows()).unwrap();
            for (row, hash) in hashes.into_iter().enumerate() {
                assert_eq!(encoded.row(row).as_slice(), expected.row(row).data());
                assert_eq!(hash, seed.hash_one(expected.row(row).data()));
            }
            if record.num_rows() == 5 {
                let EncodedColumns::StringKeys { rows, ids } = encoded else {
                    panic!("typed string keys");
                };
                assert_eq!(rows.len(), 3, "encode each distinct typed value once");
                assert_eq!(ids[0], ids[1]);
                assert_eq!(ids[3], ids[4]);
            }
        }
    }

    #[test]
    fn batch_and_scalar_key_hashes_share_the_canonical_byte_domain() {
        use datafusion::arrow::array::{ArrayRef, BinaryArray, LargeStringArray, StringArray};
        use datafusion::common::hash_utils::{RandomState, create_hashes};
        use std::hash::BuildHasher;

        let hasher = RandomState::with_seed(123);
        let cases: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![i64::MIN, 0, i64::MAX])),
            Arc::new(UInt64Array::from(vec![0, 1, u64::MAX])),
            Arc::new(StringArray::from(vec![
                Some(""),
                None,
                Some("a long generic key"),
            ])),
            Arc::new(LargeStringArray::from(vec!["", "z", "a longer key"])),
        ];
        for column in cases {
            let count = column.len();
            let record = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "identity",
                    column.data_type().clone(),
                    true,
                )])),
                vec![column],
            )
            .unwrap();
            let encoded = encode_columns(&record, &["identity".into()]).unwrap();
            let batch_hashes = encoded.hashes(&hasher, count).unwrap();
            let canonical = BinaryArray::from_iter_values((0..count).map(|row| encoded.row(row)));
            let mut expected = vec![0; count];
            create_hashes(
                [&canonical as &dyn datafusion::arrow::array::Array],
                &hasher,
                &mut expected,
            )
            .unwrap();
            assert_eq!(batch_hashes, expected);
            for (row, hash) in batch_hashes.iter().enumerate() {
                assert_eq!(*hash, encoded.with_row(row, |bytes| hasher.hash_one(bytes)));
            }
        }
    }
}

#[cfg(test)]
mod right_bucket_tests {
    use super::{
        Encoding, PayloadBatch, RightBucket, RowPayload, RowRef, State,
        right::{RightCursor, take_column_moves},
    };
    use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};
    use std::sync::{Arc, OnceLock};

    #[test]
    fn small_right_admissions_amortize_column_growth() {
        let mut bucket = RightBucket::new();
        let allocations = allocation_counter::measure(|| {
            for time in 0..1_000 {
                bucket.reserve_payloads(1);
                bucket.insert_admitted((time, Encoding::from_slice(&[1])), RowRef::fixture(0));
            }
        });
        assert!(
            allocations.count_total <= 40,
            "growth was not amortized: {allocations:?}"
        );
        assert!(bucket.capacity() <= 2 * bucket.len());
        assert_eq!(bucket.candidate(999, 0).unwrap().row, 0);
    }

    #[test]
    fn admitted_right_run_reserves_each_column_once() {
        let mut bucket = RightBucket::new();
        let allocations = allocation_counter::measure(|| {
            bucket.reserve_payloads(1_000);
            for time in 0..1_000 {
                bucket.insert((time, Encoding::from_slice(&[1])), Some(RowRef::fixture(0)));
            }
        });
        assert!(
            allocations.count_total <= 4,
            "right columns repeatedly grew: {allocations:?}"
        );
        assert_eq!(bucket.len(), 1_000);
        assert_eq!(bucket.candidate(999, 0).unwrap().row, 0);
    }

    #[test]
    fn older_payload_expiry_does_not_move_retained_identity_columns() {
        let batch = Arc::new(PayloadBatch {
            key: (1, 0),
            record: Arc::new(RecordBatch::new_empty(Arc::new(Schema::empty()))),
            encoded: OnceLock::new(),
            encoded_charge_bytes: 0,
            body_bytes: 0,
        });
        let key = Encoding::from_slice(&[1]);
        let mut state = State::default();
        for time in 1_000..2_000 {
            state
                .right
                .bucket_mut_or_default(key.clone())
                .insert((time, Encoding::from_slice(&[1])), None);
        }
        for time in 0..1_000 {
            let row = RowPayload {
                batch: Arc::clone(&batch),
                row: 0,
            };
            let row = state.attach(&row);
            state
                .right
                .bucket_mut_or_default(key.clone())
                .insert((time, Encoding::from_slice(&[1])), Some(row));
        }
        let mut status = super::super::StreamAsofJoinStatus::default();
        status.left.watermark_micros = Some(crate::EventTime::from_micros(2_000));
        status.right.watermark_micros = Some(crate::EventTime::from_micros(0));
        take_column_moves();
        state.evict(&status, 0);
        assert_eq!(
            take_column_moves(),
            0,
            "expired payload identities moved retained columns"
        );
        let bucket = state.right.get(&key).unwrap();
        assert_eq!(bucket.len(), 2_000);
        assert!(bucket.values().all(|payload| payload.is_none()));
        assert!(bucket.keys().map(|order| *order.0).eq(0..2_000));
        assert!(state.batches.is_empty());
    }

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

    #[test]
    fn monotonic_candidate_cursor_matches_binary_search_with_ties_and_identity_only_rows() {
        let payload = |row| Some(RowRef::fixture(row));
        let mut bucket = RightBucket::new();
        for (time, sequence, row) in [
            (9, 1, payload(2)),
            (5, 2, payload(1)),
            (7, 1, None),
            (5, 1, payload(0)),
            (6, 1, None),
            (5, 3, None),
        ] {
            bucket.insert((time, Encoding::from_slice(&[sequence])), row);
        }
        assert!(bucket.candidate(5, 10).is_none());
        assert!(bucket.candidate(6, 10).is_none());
        assert_eq!(bucket.candidate(9, 10).map(|row| row.row), Some(2));
        for tolerance in [0, 1, 10] {
            let mut next = RightCursor::default();
            for time in [4, 5, 5, 6, 7, 8, 9, 10] {
                assert_eq!(
                    bucket
                        .candidate_monotonic(time, tolerance, &mut next)
                        .map(|row| row.row),
                    bucket.candidate(time, tolerance).map(|row| row.row),
                );
            }
            let mut next = bucket.cursor_at(7);
            for time in [7, 8, 9, 10] {
                assert_eq!(
                    bucket
                        .candidate_monotonic(time, tolerance, &mut next)
                        .map(|row| row.row),
                    bucket.candidate(time, tolerance).map(|row| row.row),
                );
            }
        }
    }
}

#[cfg(test)]
mod left_storage_tests {
    use super::{Encoding, LeftOrder, LeftState, PayloadBatch, RowPayload, RowRef, State};
    use crate::StateSegment;
    use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};
    use std::sync::{Arc, OnceLock};

    #[test]
    fn ordered_left_batches_append_and_overlapping_rows_promote_without_reordering() {
        let batch = Arc::new(PayloadBatch {
            key: (0, 0),
            record: Arc::new(RecordBatch::new_empty(Arc::new(Schema::empty()))),
            encoded: OnceLock::from(StateSegment::new(Vec::new())),
            encoded_charge_bytes: 0,
            body_bytes: 0,
        });
        let row = |time: i64| -> (LeftOrder, RowRef) {
            (
                (time, Encoding::from_slice(&[1]), Encoding::from_slice(&[1])),
                RowRef::fixture(u32::try_from(time).expect("nonnegative test row")),
            )
        };
        let mut left = LeftState::default();
        left.append_admission(vec![row(2), row(1)]);
        left.append_admission(vec![row(4)]);
        assert!(left.is_ordered());
        assert_eq!(
            left.keys().map(|key| *key.0).collect::<Vec<_>>(),
            vec![1, 2, 4]
        );
        let mut ordered_state = State {
            left: left.clone(),
            ..State::default()
        };
        for (_, payload) in ordered_state.left.clone().iter() {
            ordered_state.attach(&RowPayload {
                batch: batch.clone(),
                row: payload.row as usize,
            });
        }
        ordered_state.commit_left_prefix(2);
        assert_eq!(
            ordered_state
                .left
                .keys()
                .map(|key| *key.0)
                .collect::<Vec<_>>(),
            vec![4]
        );
        assert_eq!(ordered_state.left.legacy.ordered.capacity(), 1);
        assert_eq!(ordered_state.batches[&(0, 0)].1, 1);
        left.append_admission(vec![row(3)]);
        assert!(!left.is_ordered());
        assert_eq!(
            left.keys().map(|key| *key.0).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        let mut general_state = State {
            left,
            ..State::default()
        };
        for (_, payload) in general_state.left.clone().iter() {
            general_state.attach(&RowPayload {
                batch: batch.clone(),
                row: payload.row as usize,
            });
        }
        general_state.commit_left_prefix(2);
        assert_eq!(
            general_state
                .left
                .keys()
                .map(|key| *key.0)
                .collect::<Vec<_>>(),
            vec![3, 4]
        );
        assert_eq!(general_state.batches[&(0, 0)].1, 2);
    }
}

#[cfg(test)]
mod right_storage_tests {
    use super::{Encoding, RightBucket, RightState, State};

    #[test]
    fn one_row_capacity_projection_allocates_independently_of_retained_batches() {
        let mut state = State::empty_tracked();
        let payload = |id| super::RowPayload {
            batch: std::sync::Arc::new(super::PayloadBatch {
                key: (1, id),
                record: std::sync::Arc::new(
                    datafusion::arrow::record_batch::RecordBatch::new_empty(std::sync::Arc::new(
                        datafusion::arrow::datatypes::Schema::empty(),
                    )),
                ),
                encoded: std::sync::OnceLock::new(),
                encoded_charge_bytes: 0,
                body_bytes: 0,
            }),
            row: 0,
        };
        let key = Encoding::from_slice(&[1]);
        let sequence = Encoding::from_slice(&[1]);
        for time in 0..4_096 {
            let row = state.attach(&payload(time));
            state
                .right
                .bucket_mut_or_default(key.clone())
                .insert((i64::try_from(time).unwrap(), sequence.clone()), Some(row));
        }
        let rows = vec![((4_096, key.clone(), sequence), payload(4_096))];
        let counts = vec![(key, 1)];
        let measured = allocation_counter::measure(|| {
            state
                .project_capacity_admission(
                    state.capacity_snapshot("asof"),
                    &rows,
                    None,
                    &counts,
                    &[rows[0].1.batch.clone()],
                    "asof",
                )
                .unwrap();
        });
        assert!(
            measured.bytes_max <= 8_192,
            "one-row preflight copied retained batches: {measured:?}"
        );
    }

    #[test]
    fn reverse_unique_right_keys_keep_canonical_checkpoint_order() {
        let mut right = RightState::default();
        let keys = 4_096_u32;
        for key in (0..keys).rev() {
            right.bucket_mut_or_default(Encoding::from_slice(&key.to_be_bytes()));
        }
        assert_eq!(right.len(), keys as usize);
        assert!(
            right
                .ordered_iter()
                .map(|(key, _)| key.as_slice())
                .eq((0..keys)
                    .map(u32::to_be_bytes)
                    .collect::<Vec<_>>()
                    .iter()
                    .map(<[u8; 4]>::as_slice))
        );
    }

    #[test]
    fn sparse_right_buckets_fit_the_committed_state_charge() {
        for keys in [1_u32, 2, 4, 16, 128, 4_096] {
            let mut state = State::default();
            let mut prepared = None;
            let allocation = allocation_counter::measure(|| {
                for key in 0..keys {
                    let mut bucket = RightBucket::default();
                    bucket.insert((1, Encoding::from_slice(&[1])), None);
                    state
                        .right
                        .insert(Encoding::from_slice(&key.to_le_bytes()), bucket);
                }
                let length = super::super::checkpoint::v3_encoded_length(&state, "asof").unwrap();
                prepared = Some(super::super::checkpoint::PreparedSegment::new(
                    crate::StateSegment::new(vec![0; usize::try_from(length).unwrap()]),
                ));
            });
            let charged = state.inventory(prepared.as_ref(), "asof").unwrap().bytes;
            assert!(
                allocation.bytes_max <= charged,
                "keys={keys}, peak={}, charged={charged}",
                allocation.bytes_max
            );
        }
    }

    #[test]
    fn empty_right_state_releases_its_owned_indexes() {
        let mut right = RightState::default();
        let allocation = allocation_counter::measure(|| {
            right
                .bucket_mut_or_default(Encoding::from_slice(&[1]))
                .insert((1, Encoding::from_slice(&[1])), None);
            right.retain(|_, _| false);
        });
        assert_eq!(allocation.bytes_current, 0);
    }

    #[test]
    fn empty_payload_state_releases_its_batch_reference_index() {
        let mut state = State::default();
        let mut status = super::super::StreamAsofJoinStatus::default();
        status.left.ended = true;
        status.right.ended = true;
        let batch = std::sync::Arc::new(super::PayloadBatch {
            key: (1, 0),
            record: std::sync::Arc::new(datafusion::arrow::record_batch::RecordBatch::new_empty(
                std::sync::Arc::new(datafusion::arrow::datatypes::Schema::empty()),
            )),
            encoded: std::sync::OnceLock::new(),
            encoded_charge_bytes: 0,
            body_bytes: 0,
        });
        let allocation = allocation_counter::measure(|| {
            let row = super::RowPayload {
                batch: std::sync::Arc::clone(&batch),
                row: 0,
            };
            let row = state.attach(&row);
            state
                .right
                .bucket_mut_or_default(Encoding::from_slice(&[1]))
                .insert((1, Encoding::from_slice(&[1])), Some(row));
            state.evict(&status, 0);
        });
        assert_eq!(state.inventory(None, "asof").unwrap().bytes, 0);
        assert_eq!(allocation.bytes_current, 0);
    }

    #[test]
    fn disordered_identity_history_fits_charge_after_partial_expiry() {
        for (arrivals, frontier) in [&[10, 9][..], &[10, 12, 11, 9][..]]
            .into_iter()
            .flat_map(|arrivals| [0, 10, 11, 12, 13].map(|frontier| (arrivals, frontier)))
        {
            let mut state = State::default();
            let mut prepared = None;
            let mut status = super::super::StreamAsofJoinStatus::default();
            status.right.watermark_micros = Some(crate::EventTime::from_micros(frontier));
            let allocation = allocation_counter::measure(|| {
                for &time in arrivals {
                    state
                        .right
                        .bucket_mut_or_default(Encoding::from_slice(&[1]))
                        .insert((time, Encoding::from_slice(&[1])), None);
                }
                state.evict(&status, 0);
                let length = super::super::checkpoint::v3_encoded_length(&state, "asof").unwrap();
                if length != 0 {
                    prepared = Some(super::super::checkpoint::PreparedSegment::new(
                        crate::StateSegment::new(vec![0; usize::try_from(length).unwrap()]),
                    ));
                }
            });
            let charge = state.inventory(prepared.as_ref(), "asof").unwrap().bytes;
            assert!(
                u64::try_from(allocation.bytes_current).unwrap() <= charge,
                "frontier={frontier}, retained={}, charge={charge}",
                allocation.bytes_current
            );
        }
    }

    #[test]
    fn eviction_workspace_bounds_small_batch_reference_trees() {
        for count in [0, 1, 2, 4, 16, 128] {
            let mut state = State::default();
            for id in 0..count {
                let row = super::RowPayload {
                    batch: std::sync::Arc::new(super::PayloadBatch {
                        key: (1, id),
                        record: std::sync::Arc::new(
                            datafusion::arrow::record_batch::RecordBatch::new_empty(
                                std::sync::Arc::new(datafusion::arrow::datatypes::Schema::empty()),
                            ),
                        ),
                        encoded: std::sync::OnceLock::new(),
                        encoded_charge_bytes: 0,
                        body_bytes: 0,
                    }),
                    row: 0,
                };
                let row = state.attach(&row);
                state
                    .right
                    .bucket_mut_or_default(Encoding::from_slice(&[1]))
                    .insert(
                        (i64::try_from(id).unwrap(), Encoding::from_slice(&[1])),
                        Some(row),
                    );
            }
            let mut status = super::super::StreamAsofJoinStatus::default();
            status.left.ended = true;
            status.right.ended = true;
            let charge = state.eviction_workspace_bytes("asof").unwrap();
            let allocation = allocation_counter::measure(|| {
                let preview = state.preview_eviction(&status, 0, "asof").unwrap();
                assert_eq!(preview.evicted_payloads, count);
            });
            assert!(
                allocation.bytes_max <= charge,
                "batches={count}, allocated={}, charge={charge}",
                allocation.bytes_max
            );
        }
    }

    #[test]
    fn hashed_right_buckets_keep_canonical_order_after_retain_and_reinsert() {
        let mut right = RightState::default();
        for key in [2_u8, 1] {
            right
                .bucket_mut_or_default(Encoding::from_slice(&[key]))
                .insert((key.into(), Encoding::from_slice(&[1])), None);
        }
        assert_eq!(
            right
                .ordered_iter()
                .map(|(key, _)| key.as_slice()[0])
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        right.retain(|key, _| key.as_slice() != [1]);
        assert_eq!(right.len(), 1);
        right.insert(Encoding::from_slice(&[1]), RightBucket::default());
        assert_eq!(
            right
                .ordered_iter()
                .map(|(key, _)| key.as_slice()[0])
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn right_eviction_releases_empty_hash_and_bucket_capacity() {
        let mut right = RightState::default();
        for key in 0_u8..64 {
            let mut bucket = RightBucket::default();
            bucket.insert((i64::from(key), Encoding::from_slice(&[1])), None);
            right.insert(Encoding::from_slice(&[key]), bucket);
        }
        right.retain(|key, _| key.as_slice() == [63]);
        assert_eq!(right.len(), 1);
        assert!(right.buckets.capacity() <= 4 * right.len());

        let mut bucket = RightBucket::default();
        for time in 0..64 {
            bucket.insert((time, Encoding::from_slice(&[1])), None);
        }
        let mut status = super::super::StreamAsofJoinStatus::default();
        status.right.watermark_micros = Some(crate::EventTime::from_micros(63));
        bucket.evict(&status, 0, i128::MIN);
        assert_eq!(bucket.len(), 1);
        assert!(bucket.capacity() <= 2 * bucket.len());
    }
}
