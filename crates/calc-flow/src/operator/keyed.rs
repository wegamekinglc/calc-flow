use crate::{CalcFlowError, Result};
use datafusion::arrow::{
    array::{Array, ArrayRef, BooleanArray, LargeStringArray, PrimitiveArray, StringArray},
    datatypes::{
        ArrowPrimitiveType, ArrowTimestampType, DataType, FieldRef, TimeUnit,
        TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType,
    },
};
use datafusion::common::ScalarValue;
use datafusion::execution::memory_pool::MemoryReservation;
use hashbrown::HashMap;
use sha2::{Digest, Sha256};
use std::{mem::size_of, sync::Arc};

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

    #[allow(
        clippy::cast_lossless,
        reason = "u32 group IDs widen losslessly on supported 32/64-bit targets"
    )]
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
            return Err(keyed_error(
                "key schema is empty or contains an unsupported type",
            ));
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
    }

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
        let _credit = self.route_credit(rows, selection)?;
        let hashes = hash_rows(&self.schema, arrays, selection, check)?;
        self.route_rows(arrays, selection, limit, collide, hashes, check)
    }

    fn route_credit(
        &self,
        rows: usize,
        selection: Option<&BooleanArray>,
    ) -> Result<MemoryReservation> {
        if rows > MORSEL_ROWS {
            return Err(keyed_error("key routing morsel exceeds 4,096 rows"));
        }
        if selection.is_some_and(|selection| selection.len() != rows) {
            return Err(keyed_error("key selection length differs from the morsel"));
        }
        let scratch = rows
            .checked_mul(2 * size_of::<u64>() + size_of::<GroupId>())
            .ok_or_else(|| keyed_error("key routing scratch charge overflowed"))?;
        let credit = self.reservation.new_empty();
        credit
            .try_grow(scratch)
            .map_err(|error| keyed_error(error.to_string()))?;
        Ok(credit)
    }

    fn route_rows(
        &mut self,
        arrays: &[ArrayRef],
        selection: Option<&BooleanArray>,
        limit: usize,
        collide: bool,
        hashes: Vec<u64>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<GroupId>> {
        let mut variable = VariableWork::new(check);
        let mut groups = Vec::with_capacity(hashes.len());
        for (row, hash) in hashes.into_iter().enumerate() {
            if selection.is_some_and(|selection| selection.is_null(row) || !selection.value(row)) {
                groups.push(GroupId(0));
                continue;
            }
            if collide && let Some(group) = self.lookup_colliding(arrays, row, &mut variable)? {
                groups.push(group);
                continue;
            }
            groups.push(self.lookup_or_birth(arrays, row, hash, limit, &mut variable)?);
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

    fn lookup_colliding(
        &self,
        arrays: &[ArrayRef],
        row: usize,
        variable: &mut VariableWork<'_>,
    ) -> Result<Option<GroupId>> {
        for index in 0..self.len() {
            let group = GroupId::new(index)?;
            if self.equals(group, arrays, row, variable)? {
                return Ok(Some(group));
            }
        }
        Ok(None)
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
        self.charge_birth(&canonical, arrays, row)?;
        self.install_birth(arrays, row, hash, group, canonical)
    }

    fn charge_birth(&mut self, canonical: &[u8], arrays: &[ArrayRef], row: usize) -> Result<()> {
        let charge = birth_charge(&self.schema, arrays, row, canonical.len())?;
        self.reservation
            .try_grow(charge)
            .map_err(|error| keyed_error(error.to_string()))?;
        self.logical_bytes = self
            .logical_bytes
            .checked_add(charge)
            .ok_or_else(|| keyed_error("key logical-byte charge overflowed"))?;
        Ok(())
    }

    fn install_birth(
        &mut self,
        arrays: &[ArrayRef],
        row: usize,
        hash: u64,
        group: GroupId,
        canonical: Vec<u8>,
    ) -> Result<GroupId> {
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
        for &hash in self.hashes[cut.groups..].iter().rev() {
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
        let metadata = field
            .metadata()
            .iter()
            .collect::<std::collections::BTreeMap<_, _>>();
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
    let rows = arrays.first().map_or(0, |array| array.as_ref().len());
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
    let values = schema
        .fields
        .iter()
        .zip(arrays)
        .try_fold(256usize, |total, (field, array)| {
            let width = birth_value_width(field, array, row)?;
            total
                .checked_add(width)
                .ok_or_else(|| keyed_error("key birth charge overflowed"))
        })?;
    values
        .checked_add(canonical)
        .and_then(|total| total.checked_add(4 * size_of::<usize>()))
        .ok_or_else(|| keyed_error("key birth charge overflowed"))
}

fn birth_value_width(field: &FieldRef, array: &ArrayRef, row: usize) -> Result<usize> {
    match field.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 => {
            Ok(2 * size_of::<String>() + string_value(array, row)?.len())
        }
        DataType::Boolean => Ok(2 * size_of::<Option<bool>>()),
        DataType::Int8 | DataType::UInt8 => Ok(2),
        DataType::Int16 | DataType::UInt16 => Ok(4),
        DataType::Int32 | DataType::UInt32 | DataType::Date32 => Ok(8),
        _ => Ok(16),
    }
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

    fn finish(self) -> u64 {
        self.0
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
    Ok(hashes.into_iter().map(StableHash::finish).collect())
}

macro_rules! hash_primitive {
    ($array:expr, $hashes:expr, $selection:expr, $kind:ty) => {{
        let values = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$kind>>()
            .ok_or_else(|| keyed_error("typed key array differs from schema"))?;
        for (row, hash) in $hashes.iter_mut().enumerate() {
            if $selection.is_some_and(|selection| selection.is_null(row) || !selection.value(row)) {
                continue;
            }
            hash.bytes(&[u8::from(values.is_null(row))]);
            if values.is_valid(row) {
                values.value(row).write_key(hash);
            }
        }
    }};
}

fn hash_array(
    array: &ArrayRef,
    hashes: &mut [StableHash],
    selection: Option<&BooleanArray>,
    variable: &mut VariableWork<'_>,
) -> Result<()> {
    match array.data_type() {
        DataType::Boolean => hash_boolean(array, hashes, selection),
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            hash_signed(array, hashes, selection)
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            hash_unsigned(array, hashes, selection)
        }
        DataType::Date32 | DataType::Date64 | DataType::Timestamp(..) => {
            hash_temporal(array, hashes, selection)
        }
        DataType::Utf8 => hash_strings::<StringArray>(array, hashes, selection, variable),
        DataType::LargeUtf8 => hash_strings::<LargeStringArray>(array, hashes, selection, variable),
        _ => Err(keyed_error("unsupported key hash type")),
    }
}

fn hash_boolean(
    array: &ArrayRef,
    hashes: &mut [StableHash],
    selection: Option<&BooleanArray>,
) -> Result<()> {
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
    Ok(())
}

fn hash_signed(
    array: &ArrayRef,
    hashes: &mut [StableHash],
    selection: Option<&BooleanArray>,
) -> Result<()> {
    match array.data_type() {
        DataType::Int8 => hash_primitive!(
            array,
            hashes,
            selection,
            datafusion::arrow::datatypes::Int8Type
        ),
        DataType::Int16 => hash_primitive!(
            array,
            hashes,
            selection,
            datafusion::arrow::datatypes::Int16Type
        ),
        DataType::Int32 => hash_primitive!(
            array,
            hashes,
            selection,
            datafusion::arrow::datatypes::Int32Type
        ),
        DataType::Int64 => hash_primitive!(
            array,
            hashes,
            selection,
            datafusion::arrow::datatypes::Int64Type
        ),
        _ => return Err(keyed_error("signed key hash type differs")),
    }
    Ok(())
}

fn hash_unsigned(
    array: &ArrayRef,
    hashes: &mut [StableHash],
    selection: Option<&BooleanArray>,
) -> Result<()> {
    match array.data_type() {
        DataType::UInt8 => hash_primitive!(
            array,
            hashes,
            selection,
            datafusion::arrow::datatypes::UInt8Type
        ),
        DataType::UInt16 => hash_primitive!(
            array,
            hashes,
            selection,
            datafusion::arrow::datatypes::UInt16Type
        ),
        DataType::UInt32 => hash_primitive!(
            array,
            hashes,
            selection,
            datafusion::arrow::datatypes::UInt32Type
        ),
        DataType::UInt64 => hash_primitive!(
            array,
            hashes,
            selection,
            datafusion::arrow::datatypes::UInt64Type
        ),
        _ => return Err(keyed_error("unsigned key hash type differs")),
    }
    Ok(())
}

fn hash_temporal(
    array: &ArrayRef,
    hashes: &mut [StableHash],
    selection: Option<&BooleanArray>,
) -> Result<()> {
    match array.data_type() {
        DataType::Date32 => hash_primitive!(
            array,
            hashes,
            selection,
            datafusion::arrow::datatypes::Date32Type
        ),
        DataType::Date64 => hash_primitive!(
            array,
            hashes,
            selection,
            datafusion::arrow::datatypes::Date64Type
        ),
        DataType::Timestamp(TimeUnit::Second, _) => {
            hash_primitive!(array, hashes, selection, TimestampSecondType)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            hash_primitive!(array, hashes, selection, TimestampMillisecondType)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            hash_primitive!(array, hashes, selection, TimestampMicrosecondType)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            hash_primitive!(array, hashes, selection, TimestampNanosecondType)
        }
        _ => return Err(keyed_error("temporal key hash type differs")),
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
    match data_type {
        DataType::Boolean => Ok(1),
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            canonical_signed_tag(data_type)
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            canonical_unsigned_tag(data_type)
        }
        DataType::Utf8 => Ok(10),
        DataType::LargeUtf8 => Ok(11),
        DataType::Date32 | DataType::Date64 | DataType::Timestamp(..) => {
            canonical_temporal_tag(data_type)
        }
        _ => Err(keyed_error("unsupported canonical key type")),
    }
}

fn canonical_signed_tag(data_type: &DataType) -> Result<u8> {
    match data_type {
        DataType::Int8 => Ok(2),
        DataType::Int16 => Ok(3),
        DataType::Int32 => Ok(4),
        DataType::Int64 => Ok(5),
        _ => Err(keyed_error("signed canonical key type differs")),
    }
}

fn canonical_unsigned_tag(data_type: &DataType) -> Result<u8> {
    match data_type {
        DataType::UInt8 => Ok(6),
        DataType::UInt16 => Ok(7),
        DataType::UInt32 => Ok(8),
        DataType::UInt64 => Ok(9),
        _ => Err(keyed_error("unsigned canonical key type differs")),
    }
}

fn canonical_temporal_tag(data_type: &DataType) -> Result<u8> {
    match data_type {
        DataType::Date32 => Ok(12),
        DataType::Date64 => Ok(13),
        DataType::Timestamp(TimeUnit::Second, _) => Ok(14),
        DataType::Timestamp(TimeUnit::Millisecond, _) => Ok(15),
        DataType::Timestamp(TimeUnit::Microsecond, _) => Ok(16),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => Ok(17),
        _ => Err(keyed_error("temporal canonical key type differs")),
    }
}

fn canonical_value(
    value: &ScalarValue,
    output: &mut Vec<u8>,
    variable: &mut VariableWork<'_>,
) -> Result<()> {
    output.push(u8::from(value.is_null()));
    match value {
        ScalarValue::Boolean(value) => {
            if let Some(value) = value {
                output.push(u8::from(*value));
            }
            Ok(())
        }
        ScalarValue::Utf8(value) | ScalarValue::LargeUtf8(value) => {
            canonical_string(value.as_deref(), output, variable)
        }
        ScalarValue::TimestampSecond(..)
        | ScalarValue::TimestampMillisecond(..)
        | ScalarValue::TimestampMicrosecond(..)
        | ScalarValue::TimestampNanosecond(..) => canonical_timestamp(value, output),
        _ => canonical_integer(value, output),
    }
}

fn canonical_string(
    value: Option<&str>,
    output: &mut Vec<u8>,
    variable: &mut VariableWork<'_>,
) -> Result<()> {
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
    Ok(())
}

fn canonical_integer(value: &ScalarValue, output: &mut Vec<u8>) -> Result<()> {
    macro_rules! fixed {
        ($value:expr) => {
            if let Some(value) = $value {
                output.extend_from_slice(&value.to_le_bytes());
            }
        };
    }
    match value {
        ScalarValue::Int8(value) => fixed!(value),
        ScalarValue::Int16(value) => fixed!(value),
        ScalarValue::Int32(value) | ScalarValue::Date32(value) => fixed!(value),
        ScalarValue::UInt8(value) => fixed!(value),
        ScalarValue::UInt16(value) => fixed!(value),
        ScalarValue::UInt32(value) => fixed!(value),
        ScalarValue::UInt64(value) => fixed!(value),
        ScalarValue::Int64(value) | ScalarValue::Date64(value) => fixed!(value),
        _ => return Err(keyed_error("canonical scalar differs from the key schema")),
    }
    Ok(())
}

fn canonical_timestamp(value: &ScalarValue, output: &mut Vec<u8>) -> Result<()> {
    let value = match value {
        ScalarValue::TimestampSecond(value, _)
        | ScalarValue::TimestampMillisecond(value, _)
        | ScalarValue::TimestampMicrosecond(value, _)
        | ScalarValue::TimestampNanosecond(value, _) => value,
        _ => {
            return Err(keyed_error(
                "canonical timestamp differs from the key schema",
            ));
        }
    };
    if let Some(value) = value {
        output.extend_from_slice(&value.to_le_bytes());
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

macro_rules! push_primitive {
    ($self:ident, $array:ident, $row:ident, $variant:ident, $kind:ty) => {{
        let input = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$kind>>()
            .ok_or_else(|| keyed_error("typed key birth array differs"))?;
        let Self::$variant(values) = $self else {
            return Err(keyed_error("typed key birth column differs"));
        };
        values.push(input.is_valid($row).then_some(input.value($row)));
    }};
}

macro_rules! equals_primitive {
    ($self:ident, $array:ident, $group:ident, $row:ident, $variant:ident, $kind:ty) => {{
        let input = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$kind>>()
            .ok_or_else(|| keyed_error("typed key equality array differs"))?;
        let Self::$variant(values) = $self else {
            return Err(keyed_error("typed key equality column differs"));
        };
        let value = input.is_valid($row).then_some(input.value($row));
        Ok(values[$group] == value)
    }};
}

macro_rules! output_primitive {
    ($self:ident, $groups:ident, $variant:ident, $kind:ty) => {{
        let Self::$variant(values) = $self else {
            return Err(keyed_error("typed key output column differs"));
        };
        Ok(primitive_output::<$kind>(values, $groups))
    }};
}

impl KeyColumn {
    fn new(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Boolean => Some(Self::Boolean(Vec::new())),
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                Self::new_signed(data_type)
            }
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                Self::new_unsigned(data_type)
            }
            DataType::Utf8 => Some(Self::Utf8(Vec::new())),
            DataType::LargeUtf8 => Some(Self::LargeUtf8(Vec::new())),
            DataType::Date32 | DataType::Date64 | DataType::Timestamp(..) => {
                Self::new_temporal(data_type)
            }
            _ => None,
        }
    }

    fn new_signed(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Int8 => Some(Self::Int8(Vec::new())),
            DataType::Int16 => Some(Self::Int16(Vec::new())),
            DataType::Int32 => Some(Self::Int32(Vec::new())),
            DataType::Int64 => Some(Self::Int64(Vec::new())),
            _ => None,
        }
    }

    fn new_unsigned(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::UInt8 => Some(Self::UInt8(Vec::new())),
            DataType::UInt16 => Some(Self::UInt16(Vec::new())),
            DataType::UInt32 => Some(Self::UInt32(Vec::new())),
            DataType::UInt64 => Some(Self::UInt64(Vec::new())),
            _ => None,
        }
    }

    fn new_temporal(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Date32 => Some(Self::Date32(Vec::new())),
            DataType::Date64 => Some(Self::Date64(Vec::new())),
            DataType::Timestamp(TimeUnit::Second, _) => Some(Self::TimestampSecond(Vec::new())),
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                Some(Self::TimestampMillisecond(Vec::new()))
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                Some(Self::TimestampMicrosecond(Vec::new()))
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                Some(Self::TimestampNanosecond(Vec::new()))
            }
            _ => None,
        }
    }

    fn push(&mut self, array: &ArrayRef, row: usize) -> Result<()> {
        match self {
            Self::Boolean(values) => {
                let input = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| keyed_error("Boolean key birth array differs"))?;
                values.push(input.is_valid(row).then_some(input.value(row)));
                Ok(())
            }
            Self::Int8(_) | Self::Int16(_) | Self::Int32(_) | Self::Int64(_) => {
                self.push_signed(array, row)
            }
            Self::UInt8(_) | Self::UInt16(_) | Self::UInt32(_) | Self::UInt64(_) => {
                self.push_unsigned(array, row)
            }
            Self::Utf8(values) => {
                let input = typed_string::<StringArray>(array)?;
                values.push(input.is_valid(row).then(|| input.value(row).to_owned()));
                Ok(())
            }
            Self::LargeUtf8(values) => {
                let input = typed_string::<LargeStringArray>(array)?;
                values.push(input.is_valid(row).then(|| input.value(row).to_owned()));
                Ok(())
            }
            Self::Date32(_)
            | Self::Date64(_)
            | Self::TimestampSecond(_)
            | Self::TimestampMillisecond(_)
            | Self::TimestampMicrosecond(_)
            | Self::TimestampNanosecond(_) => self.push_temporal(array, row),
        }
    }

    fn push_signed(&mut self, array: &ArrayRef, row: usize) -> Result<()> {
        match self {
            Self::Int8(_) => push_primitive!(
                self,
                array,
                row,
                Int8,
                datafusion::arrow::datatypes::Int8Type
            ),
            Self::Int16(_) => push_primitive!(
                self,
                array,
                row,
                Int16,
                datafusion::arrow::datatypes::Int16Type
            ),
            Self::Int32(_) => push_primitive!(
                self,
                array,
                row,
                Int32,
                datafusion::arrow::datatypes::Int32Type
            ),
            Self::Int64(_) => push_primitive!(
                self,
                array,
                row,
                Int64,
                datafusion::arrow::datatypes::Int64Type
            ),
            _ => return Err(keyed_error("signed key birth column differs")),
        }
        Ok(())
    }

    fn push_unsigned(&mut self, array: &ArrayRef, row: usize) -> Result<()> {
        match self {
            Self::UInt8(_) => push_primitive!(
                self,
                array,
                row,
                UInt8,
                datafusion::arrow::datatypes::UInt8Type
            ),
            Self::UInt16(_) => push_primitive!(
                self,
                array,
                row,
                UInt16,
                datafusion::arrow::datatypes::UInt16Type
            ),
            Self::UInt32(_) => push_primitive!(
                self,
                array,
                row,
                UInt32,
                datafusion::arrow::datatypes::UInt32Type
            ),
            Self::UInt64(_) => push_primitive!(
                self,
                array,
                row,
                UInt64,
                datafusion::arrow::datatypes::UInt64Type
            ),
            _ => return Err(keyed_error("unsigned key birth column differs")),
        }
        Ok(())
    }

    fn push_temporal(&mut self, array: &ArrayRef, row: usize) -> Result<()> {
        match self {
            Self::Date32(_) => push_primitive!(
                self,
                array,
                row,
                Date32,
                datafusion::arrow::datatypes::Date32Type
            ),
            Self::Date64(_) => push_primitive!(
                self,
                array,
                row,
                Date64,
                datafusion::arrow::datatypes::Date64Type
            ),
            Self::TimestampSecond(_) => {
                push_primitive!(self, array, row, TimestampSecond, TimestampSecondType)
            }
            Self::TimestampMillisecond(_) => {
                push_primitive!(
                    self,
                    array,
                    row,
                    TimestampMillisecond,
                    TimestampMillisecondType
                )
            }
            Self::TimestampMicrosecond(_) => {
                push_primitive!(
                    self,
                    array,
                    row,
                    TimestampMicrosecond,
                    TimestampMicrosecondType
                )
            }
            Self::TimestampNanosecond(_) => {
                push_primitive!(
                    self,
                    array,
                    row,
                    TimestampNanosecond,
                    TimestampNanosecondType
                )
            }
            _ => return Err(keyed_error("temporal key birth column differs")),
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
        match self {
            Self::Boolean(values) => {
                let input = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| keyed_error("Boolean key equality array differs"))?;
                Ok(values[group] == input.is_valid(row).then_some(input.value(row)))
            }
            Self::Int8(_) | Self::Int16(_) | Self::Int32(_) | Self::Int64(_) => {
                self.equals_signed(group, array, row)
            }
            Self::UInt8(_) | Self::UInt16(_) | Self::UInt32(_) | Self::UInt64(_) => {
                self.equals_unsigned(group, array, row)
            }
            Self::Utf8(values) => string_equality(
                values[group].as_deref(),
                typed_string::<StringArray>(array)?,
                row,
                variable,
            ),
            Self::LargeUtf8(values) => string_equality(
                values[group].as_deref(),
                typed_string::<LargeStringArray>(array)?,
                row,
                variable,
            ),
            Self::Date32(_)
            | Self::Date64(_)
            | Self::TimestampSecond(_)
            | Self::TimestampMillisecond(_)
            | Self::TimestampMicrosecond(_)
            | Self::TimestampNanosecond(_) => self.equals_temporal(group, array, row),
        }
    }

    fn equals_signed(&self, group: usize, array: &ArrayRef, row: usize) -> Result<bool> {
        match self {
            Self::Int8(_) => equals_primitive!(
                self,
                array,
                group,
                row,
                Int8,
                datafusion::arrow::datatypes::Int8Type
            ),
            Self::Int16(_) => equals_primitive!(
                self,
                array,
                group,
                row,
                Int16,
                datafusion::arrow::datatypes::Int16Type
            ),
            Self::Int32(_) => equals_primitive!(
                self,
                array,
                group,
                row,
                Int32,
                datafusion::arrow::datatypes::Int32Type
            ),
            Self::Int64(_) => equals_primitive!(
                self,
                array,
                group,
                row,
                Int64,
                datafusion::arrow::datatypes::Int64Type
            ),
            _ => Err(keyed_error("signed key equality column differs")),
        }
    }

    fn equals_unsigned(&self, group: usize, array: &ArrayRef, row: usize) -> Result<bool> {
        match self {
            Self::UInt8(_) => equals_primitive!(
                self,
                array,
                group,
                row,
                UInt8,
                datafusion::arrow::datatypes::UInt8Type
            ),
            Self::UInt16(_) => equals_primitive!(
                self,
                array,
                group,
                row,
                UInt16,
                datafusion::arrow::datatypes::UInt16Type
            ),
            Self::UInt32(_) => equals_primitive!(
                self,
                array,
                group,
                row,
                UInt32,
                datafusion::arrow::datatypes::UInt32Type
            ),
            Self::UInt64(_) => equals_primitive!(
                self,
                array,
                group,
                row,
                UInt64,
                datafusion::arrow::datatypes::UInt64Type
            ),
            _ => Err(keyed_error("unsigned key equality column differs")),
        }
    }

    fn equals_temporal(&self, group: usize, array: &ArrayRef, row: usize) -> Result<bool> {
        match self {
            Self::Date32(_) => equals_primitive!(
                self,
                array,
                group,
                row,
                Date32,
                datafusion::arrow::datatypes::Date32Type
            ),
            Self::Date64(_) => equals_primitive!(
                self,
                array,
                group,
                row,
                Date64,
                datafusion::arrow::datatypes::Date64Type
            ),
            Self::TimestampSecond(_) => {
                equals_primitive!(
                    self,
                    array,
                    group,
                    row,
                    TimestampSecond,
                    TimestampSecondType
                )
            }
            Self::TimestampMillisecond(_) => {
                equals_primitive!(
                    self,
                    array,
                    group,
                    row,
                    TimestampMillisecond,
                    TimestampMillisecondType
                )
            }
            Self::TimestampMicrosecond(_) => {
                equals_primitive!(
                    self,
                    array,
                    group,
                    row,
                    TimestampMicrosecond,
                    TimestampMicrosecondType
                )
            }
            Self::TimestampNanosecond(_) => {
                equals_primitive!(
                    self,
                    array,
                    group,
                    row,
                    TimestampNanosecond,
                    TimestampNanosecondType
                )
            }
            _ => Err(keyed_error("temporal key equality column differs")),
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
            TimestampNanosecond
        );
    }

    fn array(&self, groups: &[GroupId], data_type: &DataType) -> Result<ArrayRef> {
        match self {
            Self::Boolean(values) => Ok(Arc::new(BooleanArray::from(
                groups
                    .iter()
                    .map(|group| values[group.index()])
                    .collect::<Vec<_>>(),
            ))),
            Self::Int8(_) | Self::Int16(_) | Self::Int32(_) | Self::Int64(_) => {
                self.array_signed(groups)
            }
            Self::UInt8(_) | Self::UInt16(_) | Self::UInt32(_) | Self::UInt64(_) => {
                self.array_unsigned(groups)
            }
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
            Self::Date32(_)
            | Self::Date64(_)
            | Self::TimestampSecond(_)
            | Self::TimestampMillisecond(_)
            | Self::TimestampMicrosecond(_)
            | Self::TimestampNanosecond(_) => self.array_temporal(groups, data_type),
        }
    }

    fn array_signed(&self, groups: &[GroupId]) -> Result<ArrayRef> {
        match self {
            Self::Int8(_) => {
                output_primitive!(self, groups, Int8, datafusion::arrow::datatypes::Int8Type)
            }
            Self::Int16(_) => {
                output_primitive!(self, groups, Int16, datafusion::arrow::datatypes::Int16Type)
            }
            Self::Int32(_) => {
                output_primitive!(self, groups, Int32, datafusion::arrow::datatypes::Int32Type)
            }
            Self::Int64(_) => {
                output_primitive!(self, groups, Int64, datafusion::arrow::datatypes::Int64Type)
            }
            _ => Err(keyed_error("signed key output column differs")),
        }
    }

    fn array_unsigned(&self, groups: &[GroupId]) -> Result<ArrayRef> {
        match self {
            Self::UInt8(_) => {
                output_primitive!(self, groups, UInt8, datafusion::arrow::datatypes::UInt8Type)
            }
            Self::UInt16(_) => output_primitive!(
                self,
                groups,
                UInt16,
                datafusion::arrow::datatypes::UInt16Type
            ),
            Self::UInt32(_) => output_primitive!(
                self,
                groups,
                UInt32,
                datafusion::arrow::datatypes::UInt32Type
            ),
            Self::UInt64(_) => output_primitive!(
                self,
                groups,
                UInt64,
                datafusion::arrow::datatypes::UInt64Type
            ),
            _ => Err(keyed_error("unsigned key output column differs")),
        }
    }

    fn array_temporal(&self, groups: &[GroupId], data_type: &DataType) -> Result<ArrayRef> {
        match self {
            Self::Date32(values) => {
                Ok(primitive_output::<datafusion::arrow::datatypes::Date32Type>(values, groups))
            }
            Self::Date64(values) => {
                Ok(primitive_output::<datafusion::arrow::datatypes::Date64Type>(values, groups))
            }
            Self::TimestampSecond(values) => {
                timestamp_output::<TimestampSecondType>(values, groups, data_type)
            }
            Self::TimestampMillisecond(values) => {
                timestamp_output::<TimestampMillisecondType>(values, groups, data_type)
            }
            Self::TimestampMicrosecond(values) => {
                timestamp_output::<TimestampMicrosecondType>(values, groups, data_type)
            }
            Self::TimestampNanosecond(values) => {
                timestamp_output::<TimestampNanosecondType>(values, groups, data_type)
            }
            _ => Err(keyed_error("temporal key output column differs")),
        }
    }
}

fn string_equality<T: StringKeyArray>(
    stored: Option<&str>,
    input: &T,
    row: usize,
    variable: &mut VariableWork<'_>,
) -> Result<bool> {
    let incoming = input.is_valid(row).then_some(input.key_value(row));
    let (Some(stored), Some(incoming)) = (stored, incoming) else {
        return Ok(stored == incoming);
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
    PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
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
    T: ArrowTimestampType,
    PrimitiveArray<T>: From<Vec<Option<i64>>>,
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
