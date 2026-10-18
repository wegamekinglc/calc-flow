//! Batch-local key interning: one vectorized deduplication pass per group so
//! repeated keys pay persistent-dictionary resolution once per distinct key.

use super::super::{
    PayloadRef,
    columnar::{COPY_BYTES, COPY_VISITS},
};
use super::KeyFlow;
use crate::Result;
use datafusion::arrow::array::{
    Array, BooleanArray, LargeStringArray, PrimitiveArray, StringArray,
};
use datafusion::arrow::datatypes::{
    ArrowPrimitiveType, DataType, Date32Type, Date64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use datafusion::execution::memory_pool::MemoryReservation;

#[cfg(test)]
#[path = "local_tests.rs"]
mod tests;

/// Prepaid cooperation allowance, shared across hash, probe and equality.
/// Refilling happens before work, so small operations amortize the async step
/// while a long value or collision chain cannot outrun either quantum limit.
struct LocalFlow<'a, 'c, 'j> {
    flow: &'a mut KeyFlow<'c, 'j>,
    visits: usize,
    bytes: usize,
}

impl<'a, 'c, 'j> LocalFlow<'a, 'c, 'j> {
    fn new(flow: &'a mut KeyFlow<'c, 'j>) -> Self {
        Self {
            flow,
            visits: 0,
            bytes: 0,
        }
    }

    async fn step(&mut self, visits: usize, bytes: usize) -> Result<()> {
        debug_assert!(visits <= COPY_VISITS && bytes <= COPY_BYTES);
        if visits > self.visits || bytes > self.bytes {
            self.flow.step(COPY_VISITS, COPY_BYTES).await?;
            self.visits = COPY_VISITS;
            self.bytes = COPY_BYTES;
        }
        self.visits -= visits;
        self.bytes -= bytes;
        Ok(())
    }
}

/// How one distinct key resolves against the batch-persistent state before
/// any change is applied.
#[derive(Clone, Copy)]
pub(super) enum Plan {
    /// Already staged by an earlier group of this batch.
    Pending,
    /// Present in the committed dictionary; a non-birth change is required.
    Dictionary(u32),
    /// Unknown everywhere; it births only when a retained row exists.
    Unknown,
}

/// Per-batch deduplication table. One instance is funded per prepared batch
/// and reused group by group; every vector is prepaid for the batch's rows.
pub(super) struct LocalIntern {
    /// Local distinct index per bucket, keyed by the plain row hash.
    // Integer slots let probing resume across awaits without retaining a raw
    // hash-table iterator. The existing table allowance funds every slot.
    table: Vec<u32>,
    /// Plain (non-canonical) hash per distinct key, for table probing.
    pub(super) hashes: Vec<u64>,
    /// Canonical framed hash per distinct key, from the classification pass.
    pub(super) canonicals: Vec<u64>,
    /// First-appearance payload offset per distinct key.
    pub(super) rows: Vec<usize>,
    pub(super) appearances: Vec<u32>,
    pub(super) first_retained: Vec<u32>,
    pub(super) retains: Vec<u32>,
    pub(super) locals: Vec<u32>,
    pub(super) order: Vec<(u32, u32)>,
    pub(super) plans: Vec<Plan>,
    pub(super) posts: Vec<u32>,
    pub(super) splits: Vec<u32>,
    pub(super) distinct: usize,
    pub(super) _credit: MemoryReservation,
}

impl LocalIntern {
    pub(super) fn new(rows: usize, credit: MemoryReservation) -> Self {
        Self {
            table: vec![u32::MAX; slot_count(rows).expect("funded local table capacity")],
            hashes: Vec::with_capacity(rows),
            canonicals: Vec::with_capacity(rows),
            rows: Vec::with_capacity(rows),
            appearances: Vec::with_capacity(rows),
            first_retained: Vec::with_capacity(rows),
            retains: Vec::with_capacity(rows),
            locals: Vec::with_capacity(rows),
            order: Vec::with_capacity(rows),
            plans: Vec::with_capacity(rows),
            posts: Vec::with_capacity(rows),
            splits: Vec::with_capacity(rows),
            distinct: 0,
            _credit: credit,
        }
    }

    pub(super) fn begin_group(&mut self) {
        let Self {
            table,
            hashes,
            canonicals,
            rows,
            appearances,
            first_retained,
            retains,
            locals,
            order,
            plans,
            posts,
            splits,
            distinct,
            ..
        } = self;
        table.fill(u32::MAX);
        hashes.clear();
        canonicals.clear();
        rows.clear();
        appearances.clear();
        first_retained.clear();
        retains.clear();
        locals.clear();
        order.clear();
        plans.clear();
        posts.clear();
        splits.clear();
        *distinct = 0;
    }

    /// Probes candidates with separately charged visits and equality work.
    async fn intern_hashed(
        &mut self,
        hash: u64,
        (offset, width): (usize, usize),
        columns: &[LocalColumn<'_>],
        name: &str,
        flow: &mut LocalFlow<'_, '_, '_>,
        equal: impl Fn(usize, usize) -> bool,
    ) -> Result<u32> {
        let mask = self.table.len() - 1;
        let mut bucket = usize::try_from(hash & u64::try_from(mask).expect("slot mask fits u64"))
            .expect("masked hash fits usize");
        loop {
            flow.step(1, 0).await?;
            let local = self.table[bucket];
            if local == u32::MAX {
                let local = self.insert_distinct(hash, offset, name)?;
                self.table[bucket] = local;
                return Ok(local);
            }
            #[cfg(test)]
            super::super::typed_key::note(super::super::typed_key::KeyPhase::LocalProbe, 0);
            if self.hashes[local as usize] != hash {
                bucket = (bucket + 1) & mask;
                continue;
            }
            let left = self.rows[local as usize];
            let matches = equal_row(columns, left, (offset, width), flow, &equal).await?;
            if matches {
                return Ok(local);
            }
            bucket = (bucket + 1) & mask;
        }
    }

    fn insert_distinct(&mut self, hash: u64, offset: usize, name: &str) -> Result<u32> {
        let local = u32::try_from(self.distinct)
            .ok()
            .filter(|id| *id != u32::MAX)
            .ok_or_else(|| super::counter_overflow(name, "key_id"))?;
        self.hashes.push(hash);
        self.rows.push(offset);
        self.appearances
            .push(u32::try_from(self.locals.len()).expect("group rows fit u32 ordinals"));
        self.first_retained.push(u32::MAX);
        self.retains.push(0);
        self.distinct += 1;
        Ok(local)
    }

    pub(super) fn note_retained(&mut self, local: u32, ordinal: u32, name: &str) -> Result<()> {
        let Self {
            retains,
            first_retained,
            ..
        } = self;
        let count = retains[local as usize]
            .checked_add(1)
            .ok_or_else(|| super::counter_overflow(name, "key_references"))?;
        retains[local as usize] = count;
        if first_retained[local as usize] == u32::MAX {
            first_retained[local as usize] = ordinal;
        }
        Ok(())
    }

    /// Deduplicates one group's rows into the local table, charging the
    /// vectorized hash and probe work before each byte/visit quantum.
    pub(super) async fn intern_group(
        &mut self,
        columns: &[LocalColumn<'_>],
        group: super::super::admitted::AdmittedGroup<'_>,
        name: &str,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<()> {
        let flow = &mut LocalFlow::new(flow);
        // Dispatch a single key's equality once per group, including string
        // offsets. Composite equality still checks every typed component.
        macro_rules! primitive {
            ($values:expr) => {
                self.intern_blocks(columns, group, name, flow, |left, right| {
                    $values[left] == $values[right]
                })
                .await
            };
        }
        match columns {
            [LocalColumn::I8(values)] => primitive!(values),
            [LocalColumn::I16(values)] => primitive!(values),
            [LocalColumn::I32(values)] => primitive!(values),
            [LocalColumn::I64(values)] => primitive!(values),
            [LocalColumn::U8(values)] => primitive!(values),
            [LocalColumn::U16(values)] => primitive!(values),
            [LocalColumn::U32(values)] => primitive!(values),
            [LocalColumn::U64(values)] => primitive!(values),
            [LocalColumn::Boolean(values)] => {
                self.intern_blocks(columns, group, name, flow, |left, right| {
                    values.value(left) == values.value(right)
                })
                .await
            }
            [LocalColumn::Utf8(values)] => {
                self.intern_blocks(columns, group, name, flow, |left, right| {
                    values.value(left) == values.value(right)
                })
                .await
            }
            [LocalColumn::LargeUtf8(values)] => {
                self.intern_blocks(columns, group, name, flow, |left, right| {
                    values.value(left) == values.value(right)
                })
                .await
            }
            _ => {
                self.intern_blocks(columns, group, name, flow, |left, right| {
                    values_equal(columns, left, right)
                })
                .await
            }
        }
    }

    async fn intern_blocks(
        &mut self,
        columns: &[LocalColumn<'_>],
        group: super::super::admitted::AdmittedGroup<'_>,
        name: &str,
        flow: &mut LocalFlow<'_, '_, '_>,
        equal: impl Fn(usize, usize) -> bool,
    ) -> Result<()> {
        let mut offsets = group.key_offsets();
        // Bounded stack scratch never outlives preparation and adds no per-row
        // namespace or heap allocation to the existing prepaid local table.
        let mut positions = [(0, false); COPY_VISITS];
        let mut hashes = [(0, 0); COPY_VISITS];
        let mut ordinal = 0;
        loop {
            let length = read_positions(&mut offsets, &mut positions);
            if length == 0 {
                break;
            }
            hashes[..length].fill((0x243F_6A88_85A3_08D3, 0));
            hash_block(columns, &positions[..length], &mut hashes[..length], flow).await?;
            for (&(offset, retain), &(hash, width)) in
                positions[..length].iter().zip(&hashes[..length])
            {
                let local = self
                    .intern_hashed(hash, (offset, width), columns, name, flow, &equal)
                    .await?;
                self.locals.push(local);
                if retain {
                    self.note_retained(local, ordinal, name)?;
                }
                ordinal += 1;
            }
        }
        Ok(())
    }

    /// Records one classified key's staging slots: a dictionary hit stages
    /// an add at its first appearance; an unknown key stages a birth at its
    /// first retained row.
    pub(super) fn stage_classification(
        &mut self,
        distinct: usize,
        found: Option<u32>,
        retains: u32,
    ) {
        let Some(id) = found else {
            self.stage_unknown(distinct, retains);
            return;
        };
        self.plans.push(Plan::Dictionary(id));
        self.posts.push(u32::MAX);
        self.splits.push(u32::MAX);
        self.order.push(self.appearance_order(distinct));
    }

    /// An unresolved key stages a birth at its first retained row.
    fn stage_unknown(&mut self, distinct: usize, retains: u32) {
        self.plans.push(Plan::Unknown);
        self.posts.push(u32::MAX);
        self.splits.push(u32::MAX);
        if retains > 0 {
            self.order.push(self.birth_order(distinct));
        }
    }

    fn appearance_order(&self, distinct: usize) -> (u32, u32) {
        (
            self.appearances[distinct],
            u32::try_from(distinct).expect("distinct keys fit u32 ids"),
        )
    }

    fn birth_order(&self, distinct: usize) -> (u32, u32) {
        (
            self.first_retained[distinct],
            u32::try_from(distinct).expect("distinct keys fit u32 ids"),
        )
    }

    /// Sorts the staged changes by their creation ordinal, reproducing the
    /// per-row order in which today's path would have staged them.
    pub(super) fn sort_changes(&mut self) {
        self.order.sort_unstable_by_key(|&(ordinal, _)| ordinal);
    }

    /// Resolved id per row: rows before an unknown key's first retained row
    /// stay unresolved (`u32::MAX`), matching per-row resolution exactly.
    pub(super) fn broadcast(&self) -> impl Iterator<Item = u32> + '_ {
        self.locals.iter().enumerate().map(|(ordinal, &distinct)| {
            let split = self.splits[distinct as usize];
            let ordinal = u32::try_from(ordinal).expect("group rows fit u32 ordinals");
            if split != u32::MAX && ordinal < split {
                u32::MAX
            } else {
                self.posts[distinct as usize]
            }
        })
    }
}

pub(super) fn slot_count(rows: usize) -> Option<usize> {
    // At most 7/8 full, rounded up to a power of two. Four-byte slots cost
    // less than the existing 12-byte-per-row table allowance (plus header).
    rows.checked_add(rows.div_ceil(7))?
        .checked_next_power_of_two()
}

fn read_positions(
    offsets: &mut dyn Iterator<Item = (usize, bool)>,
    positions: &mut [(usize, bool)],
) -> usize {
    let mut length = 0;
    for (target, value) in positions.iter_mut().zip(offsets) {
        *target = value;
        length += 1;
    }
    length
}

pub(super) fn bind_columns<'a>(
    record: PayloadRef<'a>,
    indices: &[usize],
) -> Option<Vec<LocalColumn<'a>>> {
    indices
        .iter()
        .map(|&index| LocalColumn::bind(record.column(index).as_ref()))
        .collect()
}

fn values_equal(columns: &[LocalColumn<'_>], left: usize, right: usize) -> bool {
    columns.iter().all(|column| column.equal(left, right))
}

async fn hash_block(
    columns: &[LocalColumn<'_>],
    positions: &[(usize, bool)],
    hashes: &mut [(u64, usize)],
    flow: &mut LocalFlow<'_, '_, '_>,
) -> Result<()> {
    for column in columns {
        column.hash_column(positions, hashes, flow).await?;
    }
    Ok(())
}

async fn equal_row(
    columns: &[LocalColumn<'_>],
    left: usize,
    (right, width): (usize, usize),
    flow: &mut LocalFlow<'_, '_, '_>,
    equal: impl Fn(usize, usize) -> bool,
) -> Result<bool> {
    if columns.len() <= COPY_VISITS && width <= COPY_BYTES {
        flow.step(columns.len(), width).await?;
        let matches = equal(left, right);
        #[cfg(test)]
        super::super::typed_key::note(super::super::typed_key::KeyPhase::LocalComparison, width);
        Ok(matches)
    } else {
        equal_columns(columns, left, right, flow).await
    }
}

async fn equal_columns(
    columns: &[LocalColumn<'_>],
    left: usize,
    right: usize,
    flow: &mut LocalFlow<'_, '_, '_>,
) -> Result<bool> {
    for column in columns {
        if !column.equal_cooperatively(left, right, flow).await? {
            return Ok(false);
        }
    }
    Ok(true)
}

async fn equal_bytes(left: &[u8], right: &[u8], flow: &mut LocalFlow<'_, '_, '_>) -> Result<bool> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (left, right) in left.chunks(COPY_BYTES).zip(right.chunks(COPY_BYTES)) {
        flow.step(0, right.len()).await?;
        let matches = left == right;
        #[cfg(test)]
        super::super::typed_key::note(
            super::super::typed_key::KeyPhase::LocalComparison,
            right.len(),
        );
        if !matches {
            return Ok(false);
        }
    }
    Ok(true)
}

fn mix(seed: u64, value: u64) -> u64 {
    let hash = seed ^ value.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    hash ^ (hash >> 31)
}

fn hash_words(seed: u64, bytes: &[u8]) -> u64 {
    // COPY_BYTES is word aligned, so only the final chunk pads a remainder.
    const { assert!(COPY_BYTES.is_multiple_of(8)) };
    let mut hash = seed;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        hash = mix(
            hash,
            u64::from_le_bytes(chunk.try_into().expect("chunked exact word")),
        );
    }
    let remainder = chunks.remainder();
    if !remainder.is_empty() {
        let mut last = [0_u8; 8];
        last[..remainder.len()].copy_from_slice(remainder);
        hash = mix(hash, u64::from_le_bytes(last));
    }
    hash
}

/// Typed view over one key column, mirroring the value accessors of the
/// canonical `KeyWriter` framing so equality agrees row for row.
pub(super) enum LocalColumn<'a> {
    Boolean(&'a BooleanArray),
    I8(&'a [i8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}

impl<'a> LocalColumn<'a> {
    async fn equal_cooperatively(
        &self,
        left: usize,
        right: usize,
        flow: &mut LocalFlow<'_, '_, '_>,
    ) -> Result<bool> {
        flow.step(1, 0).await?;
        match self {
            Self::Utf8(values) => {
                equal_bytes(
                    values.value(left).as_bytes(),
                    values.value(right).as_bytes(),
                    flow,
                )
                .await
            }
            Self::LargeUtf8(values) => {
                equal_bytes(
                    values.value(left).as_bytes(),
                    values.value(right).as_bytes(),
                    flow,
                )
                .await
            }
            _ => {
                flow.step(0, 8).await?;
                Ok(self.equal(left, right))
            }
        }
    }

    fn bind(array: &'a dyn Array) -> Option<Self> {
        Some(match array.data_type() {
            DataType::Boolean => Self::Boolean(array.as_any().downcast_ref()?),
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                Self::bind_signed(array)?
            }
            DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Date32
            | DataType::Date64 => Self::bind_word(array)?,
            DataType::Timestamp(unit, _) => Self::timestamp(array, *unit)?,
            DataType::Utf8 => Self::Utf8(array.as_any().downcast_ref()?),
            DataType::LargeUtf8 => Self::LargeUtf8(array.as_any().downcast_ref()?),
            _ => return None,
        })
    }

    /// Signed integers load as sign-extended i64 words.
    fn bind_signed(array: &'a dyn Array) -> Option<Self> {
        Some(match array.data_type() {
            DataType::Int8 => Self::I8(primitive::<Int8Type>(array)?),
            DataType::Int16 => Self::I16(primitive::<Int16Type>(array)?),
            DataType::Int32 => Self::I32(primitive::<Int32Type>(array)?),
            DataType::Int64 => Self::I64(primitive::<Int64Type>(array)?),
            _ => return None,
        })
    }

    /// Unsigned integers and dates load as raw i64 words.
    fn bind_word(array: &'a dyn Array) -> Option<Self> {
        Some(match array.data_type() {
            DataType::UInt8 => Self::U8(primitive::<UInt8Type>(array)?),
            DataType::UInt16 => Self::U16(primitive::<UInt16Type>(array)?),
            DataType::UInt32 => Self::U32(primitive::<UInt32Type>(array)?),
            DataType::UInt64 => Self::U64(primitive::<UInt64Type>(array)?),
            DataType::Date32 => Self::I32(primitive::<Date32Type>(array)?),
            DataType::Date64 => Self::I64(primitive::<Date64Type>(array)?),
            _ => return None,
        })
    }

    fn timestamp(array: &'a dyn Array, unit: TimeUnit) -> Option<Self> {
        let values = match unit {
            TimeUnit::Second => primitive::<TimestampSecondType>(array)?,
            TimeUnit::Millisecond => primitive::<TimestampMillisecondType>(array)?,
            TimeUnit::Microsecond => primitive::<TimestampMicrosecondType>(array)?,
            TimeUnit::Nanosecond => primitive::<TimestampNanosecondType>(array)?,
        };
        Some(Self::I64(values))
    }

    async fn hash_column(
        &self,
        positions: &[(usize, bool)],
        hashes: &mut [(u64, usize)],
        flow: &mut LocalFlow<'_, '_, '_>,
    ) -> Result<()> {
        match self {
            Self::Utf8(values) => {
                hash_strings(positions, hashes, |row: usize| values.value(row), flow).await?;
            }
            Self::LargeUtf8(values) => {
                hash_strings(positions, hashes, |row: usize| values.value(row), flow).await?;
            }
            Self::Boolean(values) => {
                hash_values(
                    positions,
                    hashes,
                    1,
                    |row| u64::from(values.value(row)),
                    flow,
                )
                .await?;
            }
            _ => self.hash_integer_column(positions, hashes, flow).await?,
        }
        Ok(())
    }

    async fn hash_integer_column(
        &self,
        positions: &[(usize, bool)],
        hashes: &mut [(u64, usize)],
        flow: &mut LocalFlow<'_, '_, '_>,
    ) -> Result<()> {
        macro_rules! column {
            ($width:expr, $word:expr) => {
                hash_values(positions, hashes, $width, $word, flow).await?
            };
        }
        match self {
            Self::I8(values) => column!(1, |row: usize| u64::from_ne_bytes(
                i64::from(values[row]).to_ne_bytes()
            )),
            Self::I16(values) => column!(2, |row: usize| u64::from_ne_bytes(
                i64::from(values[row]).to_ne_bytes()
            )),
            Self::I32(values) => column!(4, |row: usize| u64::from_ne_bytes(
                i64::from(values[row]).to_ne_bytes()
            )),
            Self::I64(values) => column!(8, |row: usize| u64::from_ne_bytes(
                values[row].to_ne_bytes()
            )),
            _ => self.hash_unsigned_column(positions, hashes, flow).await?,
        }
        Ok(())
    }

    async fn hash_unsigned_column(
        &self,
        positions: &[(usize, bool)],
        hashes: &mut [(u64, usize)],
        flow: &mut LocalFlow<'_, '_, '_>,
    ) -> Result<()> {
        macro_rules! column {
            ($width:expr, $word:expr) => {
                hash_values(positions, hashes, $width, $word, flow).await?
            };
        }
        match self {
            Self::U8(values) => column!(1, |row: usize| u64::from(values[row])),
            Self::U16(values) => column!(2, |row: usize| u64::from(values[row])),
            Self::U32(values) => column!(4, |row: usize| u64::from(values[row])),
            Self::U64(values) => column!(8, |row: usize| values[row]),
            _ => unreachable!("other key types dispatch before unsigned columns"),
        }
        Ok(())
    }

    fn equal(&self, left: usize, right: usize) -> bool {
        match self {
            Self::Boolean(values) => values.value(left) == values.value(right),
            Self::I8(values) => values[left] == values[right],
            Self::I16(values) => values[left] == values[right],
            Self::I32(values) => values[left] == values[right],
            Self::I64(values) => values[left] == values[right],
            Self::U8(values) => values[left] == values[right],
            Self::U16(values) => values[left] == values[right],
            Self::U32(values) => values[left] == values[right],
            Self::U64(values) => values[left] == values[right],
            Self::Utf8(values) => values.value(left) == values.value(right),
            Self::LargeUtf8(values) => values.value(left) == values.value(right),
        }
    }
}

fn primitive<T: ArrowPrimitiveType>(array: &dyn Array) -> Option<&[T::Native]> {
    Some(array.as_any().downcast_ref::<PrimitiveArray<T>>()?.values())
}

async fn hash_values(
    positions: &[(usize, bool)],
    hashes: &mut [(u64, usize)],
    width: usize,
    word: impl Fn(usize) -> u64,
    flow: &mut LocalFlow<'_, '_, '_>,
) -> Result<()> {
    flow.step(positions.len(), positions.len() * width).await?;
    for (&(row, _), (hash, bytes)) in positions.iter().zip(hashes) {
        *hash = mix(*hash, word(row));
        *bytes += width;
        #[cfg(test)]
        super::super::typed_key::note(super::super::typed_key::KeyPhase::LocalHash, width);
    }
    Ok(())
}

async fn hash_strings<'a>(
    positions: &[(usize, bool)],
    hashes: &mut [(u64, usize)],
    value: impl Fn(usize) -> &'a str,
    flow: &mut LocalFlow<'_, '_, '_>,
) -> Result<()> {
    for (&(row, _), (hash, bytes)) in positions.iter().zip(hashes) {
        flow.step(1, 0).await?;
        let value = value(row).as_bytes();
        *hash = mix(*hash, value.len() as u64);
        for chunk in value.chunks(COPY_BYTES) {
            flow.step(0, chunk.len()).await?;
            *hash = hash_words(*hash, chunk);
            #[cfg(test)]
            super::super::typed_key::note(
                super::super::typed_key::KeyPhase::LocalHash,
                chunk.len(),
            );
        }
        *bytes += value.len();
    }
    Ok(())
}
