use crate::{CalcFlowError, Result};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, BooleanArray, LargeStringArray, PrimitiveArray, StringArray,
    },
    datatypes::{
        ArrowPrimitiveType, DataType, FieldRef, TimeUnit, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
    },
    execution::memory_pool::MemoryReservation,
};
use datafusion::common::ScalarValue;
use hashbrown::HashMap;
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub(crate) const MORSEL_ROWS: usize = 4096;
const VARIABLE_STEP_BYTES: usize = 64 * 1024;
const KEY_COLUMN_PAGE_ROWS: usize = 1024;
const CANONICAL_PAGE_BYTES: usize = 64 * 1024;
const KEY_LAYOUT_ID: &str = "shared-keyed-columns-v1";
const CANONICAL_ENCODING: &str = "typed-length-delimited-v1";
const HASH_EQUALITY: &str = "typed-row-hash-exact-column-equality-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct GroupId(u32);

impl GroupId {
    pub(crate) fn new(index: usize) -> Result<Self> {
        u32::try_from(index)
            .map(Self)
            .map_err(|_| keyed_error("group ID exceeds the u32 frontier"))
    }

    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone)]
pub(crate) struct KeySchema {
    fields: Vec<FieldRef>,
}

impl KeySchema {
    pub(crate) fn groupable(fields: Vec<FieldRef>) -> Result<Self> {
        if fields.is_empty() || fields.iter().any(|field| !supported(field.data_type())) {
            return Err(keyed_error("key schema is empty or contains an unsupported type"));
        }
        Ok(Self { fields })
    }
}

pub(crate) struct KeyStore {
    schema: KeySchema,
    columns: Vec<KeyColumn>,
    buckets: HashMap<u64, Vec<GroupId>>,
    hashes: Vec<u64>,
    canonical: Vec<Arc<[u8]>>,
    leases: Vec<u64>,
    next: usize,
    logical_bytes: usize,
    reservation: MemoryReservation,
    canonical_births: usize,
    fingerprint: [u8; 32],
}

#[derive(Clone, Copy)]
pub(crate) struct KeyCut {
    groups: usize,
    next: usize,
    logical_bytes: usize,
    reserved_bytes: usize,
    canonical_births: usize,
}

impl KeyStore {
    pub(crate) fn new(schema: KeySchema, reservation: MemoryReservation) -> Self {
        let fingerprint = layout_fingerprint(&schema);
        let columns = schema
            .fields
            .iter()
            .map(|field| KeyColumn::new(field.data_type()))
            .collect::<Option<Vec<_>>>()
            .expect("validated key schema");
        Self {
            schema,
            columns,
            buckets: HashMap::new(),
            hashes: Vec::new(),
            canonical: Vec::new(),
            leases: Vec::new(),
            next: 0,
            logical_bytes: 0,
            reservation,
            canonical_births: 0,
            fingerprint,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.canonical.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[cfg(test)]
    pub(crate) fn next_id(&self) -> usize {
        self.next
    }

    pub(crate) fn layout_fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }

    #[cfg(test)]
    pub(crate) fn canonical(&self, group: GroupId) -> Arc<[u8]> {
        self.canonical[group.index()].clone()
    }

    #[cfg(test)]
    pub(crate) fn canonical_births(&self) -> usize {
        self.canonical_births
    }

    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.reservation.size()
    }

    pub(crate) fn charge(&mut self, bytes: usize) -> Result<()> {
        self.reservation
            .try_grow(bytes)
            .map_err(|error| keyed_error(error.to_string()))?;
        self.logical_bytes = self
            .logical_bytes
            .checked_add(bytes)
            .ok_or_else(|| keyed_error("keyed-state logical-byte charge overflowed"))?;
        Ok(())
    }

    pub(crate) fn transaction_reservation(&self) -> MemoryReservation {
        self.reservation.new_empty()
    }

    pub(crate) fn acquire(&mut self, group: GroupId) -> Result<()> {
        let lease = &mut self.leases[group.index()];
        *lease = lease
            .checked_add(1)
            .ok_or_else(|| keyed_error("key lease count overflowed"))?;
        Ok(())
    }

    pub(crate) fn release(&mut self, group: GroupId) -> Result<()> {
        let lease = &mut self.leases[group.index()];
        *lease = lease
            .checked_sub(1)
            .ok_or_else(|| keyed_error("key lease count underflowed"))?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn route(
        &mut self,
        arrays: &[ArrayRef],
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<GroupId>> {
        self.route_with_limit_inner(arrays, None, usize::MAX, false, check)
    }

    pub(crate) fn route_with_limit(
        &mut self,
        arrays: &[ArrayRef],
        limit: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<GroupId>> {
        self.route_with_limit_inner(arrays, None, limit, false, check)
    }

    pub(crate) fn route_selected_with_limit(
        &mut self,
        arrays: &[ArrayRef],
        selection: Option<&BooleanArray>,
        limit: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<GroupId>> {
        self.route_with_limit_inner(arrays, selection, limit, false, check)

    #[cfg(test)]
    pub(crate) fn route_colliding(
        &mut self,
        arrays: &[ArrayRef],
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<GroupId>> {
        self.route_with_limit_inner(arrays, None, usize::MAX, true, check)
    }

    fn route_with_limit_inner(
        &mut self,
        arrays: &[ArrayRef],
        selection: Option<&BooleanArray>,
        limit: usize,
        collide: bool,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<GroupId>> {
        let cut = self.cut();
        match self.route_attempt(arrays, selection, limit, collide, check) {
            Ok(groups) => Ok(groups),
            Err(error) => {
                self.rollback(cut);
                Err(error)
            }
        }
    }

    fn route_attempt(
        &mut self,
        arrays: &[ArrayRef],
        selection: Option<&BooleanArray>,
        limit: usize,
        collide: bool,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<GroupId>> {
        check()?;
        let rows = validate_arrays(&self.schema, arrays)?;
        if rows > MORSEL_ROWS {
            return Err(keyed_error("key routing morsel exceeds 4,096 rows"));
        }
        if selection.is_some_and(|selection| selection.len() != rows) {
            return Err(keyed_error("key selection length differs from the morsel"));
        }
        let scratch = rows
            .checked_mul(2 * std::mem::size_of::<u64>() + std::mem::size_of::<GroupId>())
            .ok_or_else(|| keyed_error("key routing scratch charge overflowed"))?;
        let credit = self.reservation.new_empty();
        credit
            .try_grow(scratch)
            .map_err(|error| keyed_error(error.to_string()))?;
        let hashes = if collide {
            vec![0; rows]
        } else {
            hash_rows(&self.schema, arrays, selection, check)?
        };
        let mut variable = VariableWork::new(check);
        let mut groups = Vec::with_capacity(rows);
        for row in 0..rows {
            if selection.is_some_and(|selection| {
                selection.is_null(row) || !selection.value(row)
            }) {
                groups.push(GroupId(0));
                continue;
            }
            groups.push(self.lookup_or_birth(arrays, row, hashes[row], limit, &mut variable)?);
        }
        Ok(groups)
    }

    fn lookup_or_birth(
        &mut self,
        arrays: &[ArrayRef],
        row: usize,
        hash: u64,
        limit: usize,
        variable: &mut VariableWork<'_>,
    ) -> Result<GroupId> {
        if let Some(bucket) = self.buckets.get(&hash) {
            for &group in bucket {
                if self.equals(group, arrays, row, variable)? {
                    return Ok(group);
                }
            }
        }
        self.birth(arrays, row, hash, limit, variable)
    }

    fn equals(
        &self,
        group: GroupId,
        arrays: &[ArrayRef],
        row: usize,
        variable: &mut VariableWork<'_>,
    ) -> Result<bool> {
        for (column, array) in self.columns.iter().zip(arrays) {
            if !column.equals(group.index(), array, row, variable)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn birth(
        &mut self,
        arrays: &[ArrayRef],
        row: usize,
        hash: u64,
        limit: usize,
        variable: &mut VariableWork<'_>,
    ) -> Result<GroupId> {
        if self.len() >= limit {
            return Err(keyed_error("key group budget exceeded"));
        }
        let group = GroupId::new(self.next)?;
        let canonical = canonical_row(arrays, row, variable)?;
        let charge = birth_charge(&self.schema, arrays, row, canonical.len())?;
        self.reservation
            .try_grow(charge)
            .map_err(|error| keyed_error(error.to_string()))?;
        self.logical_bytes = self
            .logical_bytes
            .checked_add(charge)
            .ok_or_else(|| keyed_error("key logical-byte charge overflowed"))?;
        for (column, array) in self.columns.iter_mut().zip(arrays) {
            column.push(array, row)?;
        }
        self.buckets.entry(hash).or_default().push(group);
        self.hashes.push(hash);
        self.canonical.push(Arc::from(canonical));
        self.leases.push(1);
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| keyed_error("key ID frontier overflowed"))?;
        self.canonical_births += 1;
        Ok(group)
    }

    pub(crate) fn cut(&self) -> KeyCut {
        KeyCut {
            groups: self.len(),
            next: self.next,
            logical_bytes: self.logical_bytes,
            reserved_bytes: self.reservation.size(),
            canonical_births: self.canonical_births,
        }
    }

    pub(crate) fn rollback(&mut self, cut: KeyCut) {
        for group in (cut.groups..self.len()).rev() {
            let hash = self.hashes[group];
            let remove = if let Some(bucket) = self.buckets.get_mut(&hash) {
                bucket.pop();
                bucket.is_empty()
            } else {
                false
            };
            if remove {
                self.buckets.remove(&hash);
            }
        }
        for column in &mut self.columns {
            column.truncate(cut.groups);
        }
        self.hashes.truncate(cut.groups);
        self.canonical.truncate(cut.groups);
        self.leases.truncate(cut.groups);
        self.next = cut.next;
        self.logical_bytes = cut.logical_bytes;
        self.reservation.resize(cut.reserved_bytes);
        self.canonical_births = cut.canonical_births;
    }

    pub(crate) fn columns(&self, groups: &[GroupId]) -> Result<Vec<ArrayRef>> {
        self.columns
            .iter()
            .zip(&self.schema.fields)
            .map(|(column, field)| column.array(groups, field.data_type()))
            .collect()
    }

    pub(crate) fn output_charge(&self, groups: &[GroupId]) -> Result<usize> {
        groups.iter().try_fold(0usize, |total, group| {
            total
                .checked_add(self.canonical[group.index()].len())
                .and_then(|total| total.checked_add(128))
                .ok_or_else(|| keyed_error("key output charge overflowed"))
        })
    }

    #[cfg(test)]
    pub(crate) fn canonical_bytes(&self) -> Vec<Arc<[u8]>> {
        self.canonical.clone()
    }
}

struct VariableWork<'a> {
    remaining: usize,
    check: &'a mut dyn FnMut() -> Result<()>,
}

impl<'a> VariableWork<'a> {
    fn new(check: &'a mut dyn FnMut() -> Result<()>) -> Self {
        Self {
            remaining: VARIABLE_STEP_BYTES,
            check,
        }
    }

    fn chunk(&mut self, remaining: usize) -> Result<usize> {
        if self.remaining == 0 {
            (self.check)()?;
            self.remaining = VARIABLE_STEP_BYTES;
        }
        let rows = remaining.min(self.remaining).min(CANONICAL_PAGE_BYTES);
        self.remaining -= rows;
        Ok(rows)
    }
}

fn layout_fingerprint(schema: &KeySchema) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(KEY_LAYOUT_ID.as_bytes());
    digest.update(CANONICAL_ENCODING.as_bytes());
    digest.update(HASH_EQUALITY.as_bytes());
    digest.update("sql-compact-aggregate-v1".as_bytes());
    digest.update(datafusion::DATAFUSION_VERSION.as_bytes());
    digest.update("groupable-nulls".as_bytes());
    digest.update((KEY_COLUMN_PAGE_ROWS as u64).to_le_bytes());
    digest.update((CANONICAL_PAGE_BYTES as u64).to_le_bytes());
    for field in &schema.fields {
        digest.update((field.name().len() as u64).to_le_bytes());
        digest.update(field.name().as_bytes());
        digest.update(format!("{:?}", field.data_type()).as_bytes());
        digest.update([u8::from(field.is_nullable())]);
        let metadata = field.metadata().iter().collect::<std::collections::BTreeMap<_, _>>();
        for (key, value) in metadata {
            digest.update((key.len() as u64).to_le_bytes());
            digest.update(key.as_bytes());
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
    }
    digest.finalize().into()
}

fn supported(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(..)
    )
}

fn validate_arrays(schema: &KeySchema, arrays: &[ArrayRef]) -> Result<usize> {
    if arrays.len() != schema.fields.len() {
        return Err(keyed_error("key column count differs from schema"));
    }
    let rows = arrays.first().map_or(0, |array| array.len());
    for (array, field) in arrays.iter().zip(&schema.fields) {
        if array.len() != rows || array.data_type() != field.data_type() {
            return Err(keyed_error("key array differs from the compiled schema"));
        }
    }
    Ok(rows)
}

fn birth_charge(
    schema: &KeySchema,
    arrays: &[ArrayRef],
    row: usize,
    canonical: usize,
) -> Result<usize> {
    let values = schema.fields.iter().zip(arrays).try_fold(
        256usize,
        |total, (field, array)| {
            let width = match field.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 => {
                    2 * std::mem::size_of::<String>() + string_value(array, row)?.len()
                }
                DataType::Boolean => 2 * std::mem::size_of::<Option<bool>>(),
                DataType::Int8 | DataType::UInt8 => 2,
                DataType::Int16 | DataType::UInt16 => 4,
                DataType::Int32 | DataType::UInt32 | DataType::Date32 => 8,
                _ => 16,
            };
            total
                .checked_add(width)
                .ok_or_else(|| keyed_error("key birth charge overflowed"))
        },
    )?;
    values
        .checked_add(canonical)
        .and_then(|total| total.checked_add(4 * std::mem::size_of::<usize>()))
        .ok_or_else(|| keyed_error("key birth charge overflowed"))
}

fn keyed_error(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "keyed_state".into(),
        message: message.into(),
    }
}

trait KeyValue: Copy {
    fn write_key(self, digest: &mut StableHash);
}

macro_rules! key_value {
    ($($kind:ty),+ $(,)?) => {
        $(impl KeyValue for $kind {
            fn write_key(self, digest: &mut StableHash) {
                digest.bytes(&self.to_le_bytes());
            }
        })+
    };
}

key_value!(i8, i16, i32, i64, u8, u16, u32, u64);

#[derive(Clone, Copy)]
struct StableHash(u64);

impl StableHash {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn hash_rows(
    schema: &KeySchema,
    arrays: &[ArrayRef],
    selection: Option<&BooleanArray>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<u64>> {
    let rows = arrays[0].len();
    let mut hashes = vec![StableHash::new(); rows];
    let mut variable = VariableWork::new(check);
    for (field, array) in schema.fields.iter().zip(arrays) {
        let tag = canonical_tag(field.data_type())?;
        for hash in &mut hashes {
            hash.bytes(&[tag]);
        }
        hash_array(array, &mut hashes, selection, &mut variable)?;
    }
    Ok(hashes.into_iter().map(|hash| hash.0).collect())
}

fn hash_array(
    array: &ArrayRef,
    hashes: &mut [StableHash],
    selection: Option<&BooleanArray>,
    variable: &mut VariableWork<'_>,
) -> Result<()> {
    macro_rules! primitive {
        ($kind:ty) => {{
            let values = array
                .as_any()
                .downcast_ref::<PrimitiveArray<$kind>>()
                .ok_or_else(|| keyed_error("typed key array differs from schema"))?;
            for (row, hash) in hashes.iter_mut().enumerate() {
                if selection.is_some_and(|selection| selection.is_null(row) || !selection.value(row)) {
                    continue;
                }
                hash.bytes(&[u8::from(values.is_null(row))]);
                if values.is_valid(row) {
                    values.value(row).write_key(hash);
                }
            }
        }};
    }
    match array.data_type() {
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| keyed_error("Boolean key array differs"))?;
            for (row, hash) in hashes.iter_mut().enumerate() {
                if selection.is_some_and(|selection| selection.is_null(row) || !selection.value(row)) {
                    continue;
                }
                hash.bytes(&[u8::from(values.is_null(row))]);
                if values.is_valid(row) {
                    hash.bytes(&[u8::from(values.value(row))]);
                }
            }
        }
        DataType::Int8 => primitive!(datafusion::arrow::datatypes::Int8Type),
        DataType::Int16 => primitive!(datafusion::arrow::datatypes::Int16Type),
        DataType::Int32 => primitive!(datafusion::arrow::datatypes::Int32Type),
        DataType::Int64 => primitive!(datafusion::arrow::datatypes::Int64Type),
        DataType::UInt8 => primitive!(datafusion::arrow::datatypes::UInt8Type),
        DataType::UInt16 => primitive!(datafusion::arrow::datatypes::UInt16Type),
        DataType::UInt32 => primitive!(datafusion::arrow::datatypes::UInt32Type),
        DataType::UInt64 => primitive!(datafusion::arrow::datatypes::UInt64Type),
        DataType::Date32 => primitive!(datafusion::arrow::datatypes::Date32Type),
        DataType::Date64 => primitive!(datafusion::arrow::datatypes::Date64Type),
        DataType::Timestamp(TimeUnit::Second, _) => primitive!(TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, _) => primitive!(TimestampMillisecondType),
        DataType::Timestamp(TimeUnit::Microsecond, _) => primitive!(TimestampMicrosecondType),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => primitive!(TimestampNanosecondType),
        DataType::Utf8 => hash_strings::<StringArray>(array, hashes, selection, variable)?,
        DataType::LargeUtf8 => hash_strings::<LargeStringArray>(array, hashes, selection, variable)?,
        _ => return Err(keyed_error("unsupported key hash type")),
    }
    Ok(())
}

trait StringKeyArray: Array {
    fn key_value(&self, row: usize) -> &str;
}

impl StringKeyArray for StringArray {
    fn key_value(&self, row: usize) -> &str {
        self.value(row)
    }
}

impl StringKeyArray for LargeStringArray {
    fn key_value(&self, row: usize) -> &str {
        self.value(row)
    }
}

fn typed_string<T: StringKeyArray + 'static>(array: &ArrayRef) -> Result<&T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| keyed_error("string key array differs from schema"))
}

fn hash_strings<T: StringKeyArray + 'static>(
    array: &ArrayRef,
    hashes: &mut [StableHash],
    selection: Option<&BooleanArray>,
    variable: &mut VariableWork<'_>,
) -> Result<()> {
    let values = typed_string::<T>(array)?;
    for (row, hash) in hashes.iter_mut().enumerate() {
        if selection.is_some_and(|selection| selection.is_null(row) || !selection.value(row)) {
            continue;
        }
        hash.bytes(&[u8::from(values.is_null(row))]);
        if values.is_valid(row) {
            let bytes = values.key_value(row).as_bytes();
            let mut offset = 0;
            while offset < bytes.len() {
                let count = variable.chunk(bytes.len() - offset)?;
                hash.bytes(&bytes[offset..offset + count]);
                offset += count;
            }
        }
    }
    Ok(())
}

fn canonical_row(
    arrays: &[ArrayRef],
    row: usize,
    variable: &mut VariableWork<'_>,
) -> Result<Vec<u8>> {
    let mut canonical = Vec::new();
    for array in arrays {
        canonical.push(canonical_tag(array.data_type())?);
        let value = ScalarValue::try_from_array(array, row)
            .map_err(|error| keyed_error(error.to_string()))?;
        canonical_value(&value, &mut canonical, variable)?;
    }
    Ok(canonical)
}

fn canonical_tag(data_type: &DataType) -> Result<u8> {
    Ok(match data_type {
        DataType::Boolean => 1,
        DataType::Int8 => 2,
        DataType::Int16 => 3,
        DataType::Int32 => 4,
        DataType::Int64 => 5,
        DataType::UInt8 => 6,
        DataType::UInt16 => 7,
        DataType::UInt32 => 8,
        DataType::UInt64 => 9,
        DataType::Utf8 => 10,
        DataType::LargeUtf8 => 11,
        DataType::Date32 => 12,
        DataType::Date64 => 13,
        DataType::Timestamp(TimeUnit::Second, _) => 14,
        DataType::Timestamp(TimeUnit::Millisecond, _) => 15,
        DataType::Timestamp(TimeUnit::Microsecond, _) => 16,
        DataType::Timestamp(TimeUnit::Nanosecond, _) => 17,
        _ => return Err(keyed_error("unsupported canonical key type")),
    })
}

fn canonical_value(
    value: &ScalarValue,
    output: &mut Vec<u8>,
    variable: &mut VariableWork<'_>,
) -> Result<()> {
    output.push(u8::from(value.is_null()));
    macro_rules! fixed {
        ($value:expr) => {
            if let Some(value) = $value {
                output.extend_from_slice(&value.to_le_bytes());
            }
        };
    }
    match value {
        ScalarValue::Boolean(value) => {
            if let Some(value) = value {
                output.push(u8::from(*value));
            }
        }
        ScalarValue::Int8(value) => fixed!(value),
        ScalarValue::Int16(value) => fixed!(value),
        ScalarValue::Int32(value) | ScalarValue::Date32(value) => fixed!(value),
        ScalarValue::Int64(value) | ScalarValue::Date64(value) => fixed!(value),
        ScalarValue::UInt8(value) => fixed!(value),
        ScalarValue::UInt16(value) => fixed!(value),
        ScalarValue::UInt32(value) => fixed!(value),
        ScalarValue::UInt64(value) => fixed!(value),
        ScalarValue::TimestampSecond(value, _)
        | ScalarValue::TimestampMillisecond(value, _)
        | ScalarValue::TimestampMicrosecond(value, _)
        | ScalarValue::TimestampNanosecond(value, _) => fixed!(value),
        ScalarValue::Utf8(value) | ScalarValue::LargeUtf8(value) => {
            if let Some(value) = value {
                output.extend_from_slice(&(value.len() as u64).to_le_bytes());
                let bytes = value.as_bytes();
                let mut offset = 0;
                while offset < bytes.len() {
                    let count = variable.chunk(bytes.len() - offset)?;
                    output.extend_from_slice(&bytes[offset..offset + count]);
                    offset += count;
                }
            }
        }
        _ => return Err(keyed_error("canonical scalar differs from the key schema")),
    }
    Ok(())
}

enum KeyColumn {
    Boolean(Vec<Option<bool>>),
    Int8(Vec<Option<i8>>),
    Int16(Vec<Option<i16>>),
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    UInt8(Vec<Option<u8>>),
    UInt16(Vec<Option<u16>>),
    UInt32(Vec<Option<u32>>),
    UInt64(Vec<Option<u64>>),
    Utf8(Vec<Option<String>>),
    LargeUtf8(Vec<Option<String>>),
    Date32(Vec<Option<i32>>),
    Date64(Vec<Option<i64>>),
    TimestampSecond(Vec<Option<i64>>),
    TimestampMillisecond(Vec<Option<i64>>),
    TimestampMicrosecond(Vec<Option<i64>>),
    TimestampNanosecond(Vec<Option<i64>>),
}

impl KeyColumn {
    fn new(data_type: &DataType) -> Option<Self> {
        Some(match data_type {
            DataType::Boolean => Self::Boolean(Vec::new()),
            DataType::Int8 => Self::Int8(Vec::new()),
            DataType::Int16 => Self::Int16(Vec::new()),
            DataType::Int32 => Self::Int32(Vec::new()),
            DataType::Int64 => Self::Int64(Vec::new()),
            DataType::UInt8 => Self::UInt8(Vec::new()),
            DataType::UInt16 => Self::UInt16(Vec::new()),
            DataType::UInt32 => Self::UInt32(Vec::new()),
            DataType::UInt64 => Self::UInt64(Vec::new()),
            DataType::Utf8 => Self::Utf8(Vec::new()),
            DataType::LargeUtf8 => Self::LargeUtf8(Vec::new()),
            DataType::Date32 => Self::Date32(Vec::new()),
            DataType::Date64 => Self::Date64(Vec::new()),
            DataType::Timestamp(TimeUnit::Second, _) => Self::TimestampSecond(Vec::new()),
            DataType::Timestamp(TimeUnit::Millisecond, _) => Self::TimestampMillisecond(Vec::new()),
            DataType::Timestamp(TimeUnit::Microsecond, _) => Self::TimestampMicrosecond(Vec::new()),
            DataType::Timestamp(TimeUnit::Nanosecond, _) => Self::TimestampNanosecond(Vec::new()),
            _ => return None,
        })
    }

    fn push(&mut self, array: &ArrayRef, row: usize) -> Result<()> {
        macro_rules! push {
            ($variant:ident, $kind:ty) => {{
                let input = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<$kind>>()
                    .ok_or_else(|| keyed_error("typed key birth array differs"))?;
                let Self::$variant(values) = self else {
                    return Err(keyed_error("typed key birth column differs"));
                };
                values.push(input.is_valid(row).then(|| input.value(row)));
            }};
        }
        match self {
            Self::Boolean(values) => {
                let input = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| keyed_error("Boolean key birth array differs"))?;
                values.push(input.is_valid(row).then(|| input.value(row)));
            }
            Self::Int8(_) => push!(Int8, datafusion::arrow::datatypes::Int8Type),
            Self::Int16(_) => push!(Int16, datafusion::arrow::datatypes::Int16Type),
            Self::Int32(_) => push!(Int32, datafusion::arrow::datatypes::Int32Type),
            Self::Int64(_) => push!(Int64, datafusion::arrow::datatypes::Int64Type),
            Self::UInt8(_) => push!(UInt8, datafusion::arrow::datatypes::UInt8Type),
            Self::UInt16(_) => push!(UInt16, datafusion::arrow::datatypes::UInt16Type),
            Self::UInt32(_) => push!(UInt32, datafusion::arrow::datatypes::UInt32Type),
            Self::UInt64(_) => push!(UInt64, datafusion::arrow::datatypes::UInt64Type),
            Self::Utf8(values) => values.push(
                typed_string::<StringArray>(array)?
                    .is_valid(row)
                    .then(|| typed_string::<StringArray>(array).unwrap().value(row).to_owned()),
            ),
            Self::LargeUtf8(values) => values.push(
                typed_string::<LargeStringArray>(array)?
                    .is_valid(row)
                    .then(|| typed_string::<LargeStringArray>(array).unwrap().value(row).to_owned()),
            ),
            Self::Date32(_) => push!(Date32, datafusion::arrow::datatypes::Date32Type),
            Self::Date64(_) => push!(Date64, datafusion::arrow::datatypes::Date64Type),
            Self::TimestampSecond(_) => push!(TimestampSecond, TimestampSecondType),
            Self::TimestampMillisecond(_) => {
                push!(TimestampMillisecond, TimestampMillisecondType)
            }
            Self::TimestampMicrosecond(_) => {
                push!(TimestampMicrosecond, TimestampMicrosecondType)
            }
            Self::TimestampNanosecond(_) => {
                push!(TimestampNanosecond, TimestampNanosecondType)
            }
        }
        Ok(())
    }

    fn equals(
        &self,
        group: usize,
        array: &ArrayRef,
        row: usize,
        variable: &mut VariableWork<'_>,
    ) -> Result<bool> {
        macro_rules! equals {
            ($variant:ident, $kind:ty) => {{
                let input = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<$kind>>()
                    .ok_or_else(|| keyed_error("typed key equality array differs"))?;
                let Self::$variant(values) = self else {
                    return Err(keyed_error("typed key equality column differs"));
                };
                let value = input.is_valid(row).then(|| input.value(row));
                return Ok(values[group] == value);
            }};
        }
        match self {
            Self::Boolean(values) => {
                let input = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| keyed_error("Boolean key equality array differs"))?;
                Ok(values[group] == input.is_valid(row).then(|| input.value(row)))
            }
            Self::Int8(_) => equals!(Int8, datafusion::arrow::datatypes::Int8Type),
            Self::Int16(_) => equals!(Int16, datafusion::arrow::datatypes::Int16Type),
            Self::Int32(_) => equals!(Int32, datafusion::arrow::datatypes::Int32Type),
            Self::Int64(_) => equals!(Int64, datafusion::arrow::datatypes::Int64Type),
            Self::UInt8(_) => equals!(UInt8, datafusion::arrow::datatypes::UInt8Type),
            Self::UInt16(_) => equals!(UInt16, datafusion::arrow::datatypes::UInt16Type),
            Self::UInt32(_) => equals!(UInt32, datafusion::arrow::datatypes::UInt32Type),
            Self::UInt64(_) => equals!(UInt64, datafusion::arrow::datatypes::UInt64Type),
            Self::Utf8(values) => string_equality(
                &values[group],
                typed_string::<StringArray>(array)?,
                row,
                variable,
            ),
            Self::LargeUtf8(values) => string_equality(
                &values[group],
                typed_string::<LargeStringArray>(array)?,
                row,
                variable,
            ),
            Self::Date32(_) => equals!(Date32, datafusion::arrow::datatypes::Date32Type),
            Self::Date64(_) => equals!(Date64, datafusion::arrow::datatypes::Date64Type),
            Self::TimestampSecond(_) => equals!(TimestampSecond, TimestampSecondType),
            Self::TimestampMillisecond(_) => {
                equals!(TimestampMillisecond, TimestampMillisecondType)
            }
            Self::TimestampMicrosecond(_) => {
                equals!(TimestampMicrosecond, TimestampMicrosecondType)
            }
            Self::TimestampNanosecond(_) => {
                equals!(TimestampNanosecond, TimestampNanosecondType)
            }
        }
    }

    fn truncate(&mut self, groups: usize) {
        macro_rules! truncate {
            ($($variant:ident),+ $(,)?) => {
                match self {
                    $(Self::$variant(values) => values.truncate(groups),)+
                }
            };
        }
        truncate!(
            Boolean,
            Int8,
            Int16,
            Int32,
            Int64,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Utf8,
            LargeUtf8,
            Date32,
            Date64,
            TimestampSecond,
            TimestampMillisecond,
            TimestampMicrosecond,
            TimestampNanosecond,
        );
    }

    fn array(&self, groups: &[GroupId], data_type: &DataType) -> Result<ArrayRef> {
        macro_rules! output {
            ($variant:ident, $kind:ty) => {{
                let Self::$variant(values) = self else {
                    return Err(keyed_error("typed key output column differs"));
                };
                return Ok(primitive_output::<$kind>(values, groups));
            }};
        }
        match self {
            Self::Boolean(values) => Ok(Arc::new(BooleanArray::from(
                groups.iter().map(|group| values[group.index()]).collect::<Vec<_>>(),
            ))),
            Self::Int8(_) => output!(Int8, datafusion::arrow::datatypes::Int8Type),
            Self::Int16(_) => output!(Int16, datafusion::arrow::datatypes::Int16Type),
            Self::Int32(_) => output!(Int32, datafusion::arrow::datatypes::Int32Type),
            Self::Int64(_) => output!(Int64, datafusion::arrow::datatypes::Int64Type),
            Self::UInt8(_) => output!(UInt8, datafusion::arrow::datatypes::UInt8Type),
            Self::UInt16(_) => output!(UInt16, datafusion::arrow::datatypes::UInt16Type),
            Self::UInt32(_) => output!(UInt32, datafusion::arrow::datatypes::UInt32Type),
            Self::UInt64(_) => output!(UInt64, datafusion::arrow::datatypes::UInt64Type),
            Self::Utf8(values) => Ok(Arc::new(StringArray::from(
                groups
                    .iter()
                    .map(|group| values[group.index()].as_deref())
                    .collect::<Vec<_>>(),
            ))),
            Self::LargeUtf8(values) => Ok(Arc::new(LargeStringArray::from(
                groups
                    .iter()
                    .map(|group| values[group.index()].as_deref())
                    .collect::<Vec<_>>(),
            ))),
            Self::Date32(_) => output!(Date32, datafusion::arrow::datatypes::Date32Type),
            Self::Date64(_) => output!(Date64, datafusion::arrow::datatypes::Date64Type),
            Self::TimestampSecond(values) => timestamp_output::<TimestampSecondType>(
                values,
                groups,
                data_type,
            ),
            Self::TimestampMillisecond(values) => timestamp_output::<TimestampMillisecondType>(
                values,
                groups,
                data_type,
            ),
            Self::TimestampMicrosecond(values) => timestamp_output::<TimestampMicrosecondType>(
                values,
                groups,
                data_type,
            ),
            Self::TimestampNanosecond(values) => timestamp_output::<TimestampNanosecondType>(
                values,
                groups,
                data_type,
            ),
        }
    }
}

fn string_equality<T: StringKeyArray>(
    stored: &Option<String>,
    input: &T,
    row: usize,
    variable: &mut VariableWork<'_>,
) -> Result<bool> {
    let incoming = input.is_valid(row).then(|| input.key_value(row));
    let (Some(stored), Some(incoming)) = (stored.as_deref(), incoming) else {
        return Ok(stored.as_deref() == incoming);
    };
    if stored.len() != incoming.len() {
        return Ok(false);
    }
    let left = stored.as_bytes();
    let right = incoming.as_bytes();
    let mut offset = 0;
    while offset < left.len() {
        let count = variable.chunk(left.len() - offset)?;
        if left[offset..offset + count] != right[offset..offset + count] {
            return Ok(false);
        }
        offset += count;
    }
    Ok(true)
}

fn string_value(array: &ArrayRef, row: usize) -> Result<&str> {
    if array.is_null(row) {
        return Ok("");
    }
    match array.data_type() {
        DataType::Utf8 => Ok(typed_string::<StringArray>(array)?.key_value(row)),
        DataType::LargeUtf8 => Ok(typed_string::<LargeStringArray>(array)?.key_value(row)),
        _ => Err(keyed_error("key charge requires a string value")),
    }
}

fn primitive_output<T>(values: &[Option<T::Native>], groups: &[GroupId]) -> ArrayRef
where
    T: ArrowPrimitiveType,
{
    Arc::new(PrimitiveArray::<T>::from(
        groups
            .iter()
            .map(|group| values[group.index()])
            .collect::<Vec<_>>(),
    ))
}

fn timestamp_output<T>(
    values: &[Option<i64>],
    groups: &[GroupId],
    data_type: &DataType,
) -> Result<ArrayRef>
where
    T: ArrowPrimitiveType<Native = i64>,
{
    let DataType::Timestamp(_, timezone) = data_type else {
        return Err(keyed_error("timestamp output schema differs"));
    };
    let values = PrimitiveArray::<T>::from(
        groups
            .iter()
            .map(|group| values[group.index()])
            .collect::<Vec<_>>(),
    );
    Ok(Arc::new(values.with_timezone_opt(timezone.clone())))
}
