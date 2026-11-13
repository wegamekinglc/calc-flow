//! Only this module can mint the owned-batch provenance used by DISTINCT.
use crate::{Batch, BatchMetadata, DistinctHistory, EventTime, Result, dataframe::invalid};
use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, BooleanArray, GenericStringArray, LargeStringArray, OffsetSizeTrait,
            PrimitiveArray, StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        },
        buffer::{BooleanBuffer, MutableBuffer, NullBuffer, OffsetBuffer, ScalarBuffer},
        datatypes::{
            DataType, Date32Type, Date64Type, Field, Int8Type, Int16Type, Int32Type, Int64Type,
            Schema, SchemaRef, TimeUnit, TimestampMicrosecondType, UInt8Type, UInt16Type,
            UInt32Type, UInt64Type,
        },
        record_batch::RecordBatch,
    },
    execution::memory_pool::{MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation},
};
use std::{collections::BTreeSet, sync::Arc};

#[cfg(test)]
mod tests;

pub(super) struct OwnedBatch {
    pub(super) batch: Batch,
    data_lease: MemoryReservation,
    metadata_lease: MemoryReservation,
}
impl OwnedBatch {
    pub(super) fn charge(&self) -> usize {
        self.data_lease.size() + self.metadata_lease.size()
    }
}

pub(super) fn combine(
    history: &[Arc<OwnedBatch>],
    schema: &SchemaRef,
    columns: &[String],
    pool: &Arc<dyn MemoryPool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<OwnedBatch> {
    let lease = reserve(
        pool,
        combined_bound(history, columns.len())?,
        "combined query input",
    )?;
    let metadata = combined_metadata(history, pool)?;
    let records = combined_records(history, schema, check)?;
    let borrowed = history.last().map(|h| h.batch.metadata());
    let batch = Batch::table(records, borrowed.cloned().unwrap_or_default())?;
    if history.is_empty() {
        return normalize(&batch, columns, pool, None, check);
    }
    Ok(OwnedBatch {
        batch,
        data_lease: lease,
        metadata_lease: metadata,
    })
}
fn combined_bound(history: &[Arc<OwnedBatch>], columns: usize) -> Result<usize> {
    let empty = if history.is_empty() {
        mul(columns, 512)?
    } else {
        0
    };
    history
        .iter()
        .try_fold(add(4096, empty)?, |n, h| add(n, h.charge()))
}
fn combined_metadata(
    history: &[Arc<OwnedBatch>],
    pool: &Arc<dyn MemoryPool>,
) -> Result<MemoryReservation> {
    match history.last() {
        Some(h) => super::super::sql::metadata::reserve_with(
            h.batch.metadata(),
            "distinct combined metadata",
            reserve(pool, 0, "combined metadata")?,
        ),
        None => reserve(pool, 4096, "combined metadata"),
    }
}
fn combined_records(
    history: &[Arc<OwnedBatch>],
    schema: &SchemaRef,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<RecordBatch>> {
    let count = history
        .iter()
        .try_fold(0, |n, h| add(n, h.batch.table_payload()?.batches().len()))?;
    let mut records = Vec::with_capacity(count.max(1));
    for h in history {
        check()?;
        records.extend_from_slice(h.batch.table_payload()?.batches());
    }
    if records.is_empty() {
        #[cfg(test)]
        tests::before_target_allocation();
        records.push(RecordBatch::new_empty(schema.clone()));
    }
    Ok(records)
}

pub(super) fn projected_schema(schema: &SchemaRef, columns: &[String]) -> Result<SchemaRef> {
    if columns.is_empty() {
        return Err(invalid(
            "distinct.columns",
            "invalid_literal",
            "provide at least one key",
        ));
    }
    let mut seen = BTreeSet::new();
    let fields = columns.iter().enumerate().map(|(i, name)| {
        let path = format!("distinct.columns[{i}]");
        if !seen.insert(name) { return Err(invalid(&path, "duplicate_name", name)); }
        let field = schema.field_with_name(name).map_err(|_| invalid(&path, "unresolved_type", format!("unknown key {name:?}")))?;
        if !key_type(field.data_type()) { return Err(invalid(&path, "unsupported_type", format!("keys require Boolean, integers, Utf8/LargeUtf8, dates or timestamp[us] naive/UTC; found {}", field.data_type()))); }
        Ok(Field::new(field.name(), own_dtype(field.data_type()), field.is_nullable()))
    }).collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

pub(super) fn normalize(
    raw: &Batch,
    columns: &[String],
    pool: &Arc<dyn MemoryPool>,
    domain: Option<&DistinctHistory>,
    check: &dyn Fn() -> Result<()>,
) -> Result<OwnedBatch> {
    check()?;
    let (data, metadata) = normalization_leases(raw, columns, pool, domain, check)?;
    let batch = copy_table(raw, columns, domain, check)?;
    validate_backing(&batch, data.size())?;
    check()?;
    Ok(OwnedBatch {
        batch,
        data_lease: data,
        metadata_lease: metadata,
    })
}
fn normalization_leases(
    raw: &Batch,
    columns: &[String],
    pool: &Arc<dyn MemoryPool>,
    domain: Option<&DistinctHistory>,
    check: &dyn Fn() -> Result<()>,
) -> Result<(MemoryReservation, MemoryReservation)> {
    let table = raw.table_payload()?;
    let bound = table
        .batches()
        .iter()
        .try_fold(column_schema_bound(table.schema(), columns)?, |n, r| {
            add(n, record_bound(r, columns, domain, check)?)
        })?;
    let data = reserve(pool, bound, "normalization/schema")?;
    let metadata = super::super::sql::metadata::reserve_with(
        raw.metadata(),
        "distinct copy_visible_v2 metadata",
        MemoryConsumer::new("distinct.metadata").register(pool),
    )?;
    Ok((data, metadata))
}
fn column_schema_bound(schema: &SchemaRef, columns: &[String]) -> Result<usize> {
    columns.iter().try_fold(4096, |n, name| {
        let field = schema.field_with_name(name).map_err(error)?;
        add(n, add(512, mul(field.name().len(), 2)?)?)
    })
}
fn record_bound(
    record: &RecordBatch,
    columns: &[String],
    domain: Option<&DistinctHistory>,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let n = count_record(record, domain, check)?;
    columns.iter().try_fold(256, |bound, name| {
        let array = record
            .column_by_name(name)
            .ok_or_else(|| error("missing key"))?;
        add(
            bound,
            column_bound(array, n, &|row| member(record, row, domain), check)?,
        )
    })
}
fn copy_table(
    raw: &Batch,
    columns: &[String],
    domain: Option<&DistinctHistory>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Batch> {
    let table = raw.table_payload()?;
    #[cfg(test)]
    tests::before_target_allocation();
    let schema = projected_schema(table.schema(), columns)?;
    let records = table
        .batches()
        .iter()
        .map(|r| copy_record(r, &schema, columns, domain, check))
        .collect::<Result<Vec<_>>>()?;
    let metadata = BatchMetadata::new(
        raw.metadata().source().to_owned(),
        raw.metadata().sequence(),
        raw.metadata().attributes().clone(),
    )?;
    Batch::table(records, metadata)
}
fn copy_record(
    record: &RecordBatch,
    schema: &SchemaRef,
    columns: &[String],
    domain: Option<&DistinctHistory>,
    check: &dyn Fn() -> Result<()>,
) -> Result<RecordBatch> {
    check()?;
    let n = count_record(record, domain, check)?;
    let arrays = columns
        .iter()
        .map(|name| {
            copy_column(
                record.column_by_name(name).unwrap(),
                n,
                &|row| member(record, row, domain),
                check,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(schema.clone(), arrays).map_err(error)
}
fn validate_backing(batch: &Batch, bound: usize) -> Result<()> {
    if backing(batch)? > bound {
        return Err(error("normalization exceeded prepaid layout"));
    }
    Ok(())
}

fn column_bound(
    array: &ArrayRef,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let parts = [
        256,
        align(value_bytes(array, n, selected, check)?)?,
        offset_bytes(array.data_type(), n)?,
        null_bytes(array, n)?,
    ];
    parts.into_iter().try_fold(0, add)
}
fn value_bytes(
    array: &ArrayRef,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    match array.data_type() {
        DataType::Utf8 => string_bytes(
            array.as_any().downcast_ref::<StringArray>().unwrap(),
            selected,
            check,
        ),
        DataType::LargeUtf8 => string_bytes(
            array.as_any().downcast_ref::<LargeStringArray>().unwrap(),
            selected,
            check,
        ),
        DataType::Boolean => Ok(n.div_ceil(8)),
        dtype => mul(
            n,
            dtype
                .primitive_width()
                .ok_or_else(|| error("input_normalization_unsupported"))?,
        ),
    }
}
fn offset_bytes(dtype: &DataType, n: usize) -> Result<usize> {
    let width = match dtype {
        DataType::Utf8 => 4,
        DataType::LargeUtf8 => 8,
        _ => return Ok(0),
    };
    align(mul(add(n, 1)?, width)?)
}
fn null_bytes(array: &ArrayRef, n: usize) -> Result<usize> {
    if array.null_count() > 0 {
        align(n.div_ceil(8))
    } else {
        Ok(0)
    }
}

fn string_bytes<O: OffsetSizeTrait>(
    array: &GenericStringArray<O>,
    selected: &dyn Fn(usize) -> Result<bool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let mut bytes = 0;
    for i in 0..array.len() {
        check_row(i, check)?;
        if selected(i)? && array.is_valid(i) {
            bytes = add(bytes, array.value(i).len())?;
        }
    }
    if O::from_usize(bytes).is_none() {
        return Err(error("string offset overflow"));
    }
    Ok(bytes)
}
fn validity(
    array: &dyn Array,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Option<NullBuffer>> {
    if array.null_count() == 0 {
        return Ok(None);
    }
    let mut bits = MutableBuffer::from_len_zeroed(n.div_ceil(8));
    let mut j = 0;
    for i in 0..array.len() {
        check_row(i, check)?;
        if selected(i)? {
            if array.is_valid(i) {
                bits.as_slice_mut()[j / 8] |= 1 << (j % 8);
            }
            j += 1;
        }
    }
    Ok(Some(NullBuffer::new(BooleanBuffer::new(bits.into(), 0, n))))
}
fn copy_column(
    array: &ArrayRef,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    #[cfg(test)]
    tests::before_target_allocation();
    let nulls = validity(array.as_ref(), n, selected, check)?;
    match array.data_type() {
        DataType::Int8 => copy_primitive::<Int8Type>(array, n, selected, nulls, check),
        DataType::Int16 => copy_primitive::<Int16Type>(array, n, selected, nulls, check),
        DataType::Int32 => copy_primitive::<Int32Type>(array, n, selected, nulls, check),
        DataType::Int64 => copy_primitive::<Int64Type>(array, n, selected, nulls, check),
        _ => copy_unsigned(array, n, selected, nulls, check),
    }
}
fn copy_unsigned(
    array: &ArrayRef,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    nulls: Option<NullBuffer>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    match array.data_type() {
        DataType::UInt8 => copy_primitive::<UInt8Type>(array, n, selected, nulls, check),
        DataType::UInt16 => copy_primitive::<UInt16Type>(array, n, selected, nulls, check),
        DataType::UInt32 => copy_primitive::<UInt32Type>(array, n, selected, nulls, check),
        DataType::UInt64 => copy_primitive::<UInt64Type>(array, n, selected, nulls, check),
        _ => copy_temporal(array, n, selected, nulls, check),
    }
}
fn copy_temporal(
    array: &ArrayRef,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    nulls: Option<NullBuffer>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    match array.data_type() {
        DataType::Date32 => copy_primitive::<Date32Type>(array, n, selected, nulls, check),
        DataType::Date64 => copy_primitive::<Date64Type>(array, n, selected, nulls, check),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            copy_primitive::<TimestampMicrosecondType>(array, n, selected, nulls, check)
        }
        _ => copy_variable(array, n, selected, nulls, check),
    }
}
fn copy_variable(
    array: &ArrayRef,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    nulls: Option<NullBuffer>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    match array.data_type() {
        DataType::Boolean => copy_boolean(array, n, selected, nulls, check),
        DataType::Utf8 => copy_string(
            array.as_any().downcast_ref::<StringArray>().unwrap(),
            n,
            selected,
            nulls,
            check,
        )
        .map(|a| Arc::new(a) as ArrayRef),
        DataType::LargeUtf8 => copy_string(
            array.as_any().downcast_ref::<LargeStringArray>().unwrap(),
            n,
            selected,
            nulls,
            check,
        )
        .map(|a| Arc::new(a) as ArrayRef),
        _ => Err(error("input_normalization_unsupported")),
    }
}
fn copy_primitive<T: datafusion::arrow::datatypes::ArrowPrimitiveType>(
    array: &ArrayRef,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    nulls: Option<NullBuffer>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    let src = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| error("input_normalization_unsupported"))?;
    let mut values = MutableBuffer::with_capacity(mul(n, size_of::<T::Native>())?);
    for i in 0..src.len() {
        check_row(i, check)?;
        if selected(i)? {
            values.push(primitive_value(src, i));
        }
    }
    Ok(Arc::new(
        PrimitiveArray::<T>::new(ScalarBuffer::new(values.into(), 0, n), nulls)
            .with_data_type(own_dtype(array.data_type())),
    ))
}
fn primitive_value<T: datafusion::arrow::datatypes::ArrowPrimitiveType>(
    src: &PrimitiveArray<T>,
    i: usize,
) -> T::Native {
    if src.is_null(i) {
        Default::default()
    } else {
        src.value(i)
    }
}
fn copy_boolean(
    array: &ArrayRef,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    nulls: Option<NullBuffer>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    let src = array.as_any().downcast_ref::<BooleanArray>().unwrap();
    let mut values = MutableBuffer::from_len_zeroed(n.div_ceil(8));
    let mut j = 0;
    for i in 0..src.len() {
        check_row(i, check)?;
        if selected(i)? {
            if boolean_value(src, i) {
                values.as_slice_mut()[j / 8] |= 1 << (j % 8);
            }
            j += 1;
        }
    }
    Ok(Arc::new(BooleanArray::new(
        BooleanBuffer::new(values.into(), 0, n),
        nulls,
    )))
}
fn boolean_value(src: &BooleanArray, i: usize) -> bool {
    src.is_valid(i) && src.value(i)
}
pub(super) fn check_row(i: usize, check: &dyn Fn() -> Result<()>) -> Result<()> {
    if i.is_multiple_of(256) {
        check()?;
    }
    Ok(())
}

fn copy_string<O: OffsetSizeTrait>(
    src: &GenericStringArray<O>,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    nulls: Option<NullBuffer>,
    check: &dyn Fn() -> Result<()>,
) -> Result<GenericStringArray<O>> {
    let bytes = string_bytes(src, selected, check)?;
    let mut offsets = string_offsets::<O>(n)?;
    let mut values = MutableBuffer::with_capacity(bytes);
    offsets.push(O::from_usize(0).unwrap());
    for i in 0..src.len() {
        check_row(i, check)?;
        if !selected(i)? {
            continue;
        }
        push_string_value(src, i, &mut values, &mut offsets)?;
    }
    Ok(GenericStringArray::new(
        OffsetBuffer::new(ScalarBuffer::new(offsets.into(), 0, n + 1)),
        values.into(),
        nulls,
    ))
}
fn push_string_value<O: OffsetSizeTrait>(
    src: &GenericStringArray<O>,
    i: usize,
    values: &mut MutableBuffer,
    offsets: &mut MutableBuffer,
) -> Result<()> {
    if src.is_valid(i) {
        values.extend_from_slice(src.value(i).as_bytes());
    }
    offsets.push(O::from_usize(values.len()).ok_or_else(|| error("string offset overflow"))?);
    Ok(())
}
fn string_offsets<O: OffsetSizeTrait>(n: usize) -> Result<MutableBuffer> {
    Ok(MutableBuffer::with_capacity(mul(
        add(n, 1)?,
        size_of::<O>(),
    )?))
}

pub(super) fn count(
    raw: &Batch,
    domain: Option<&DistinctHistory>,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    raw.table_payload()?
        .batches()
        .iter()
        .try_fold(0, |n, r| add(n, count_record(r, domain, check)?))
}
fn count_record(
    record: &RecordBatch,
    domain: Option<&DistinctHistory>,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let mut count = 0;
    for i in 0..record.num_rows() {
        check_row(i, check)?;
        if member(record, i, domain)? {
            count = add(count, 1)?;
        }
    }
    Ok(count)
}
fn member(record: &RecordBatch, row: usize, domain: Option<&DistinctHistory>) -> Result<bool> {
    let Some(DistinctHistory::ClosedWindow {
        event_time_column,
        start_micros,
        end_micros,
    }) = domain
    else {
        return Ok(true);
    };
    let Some(t) = event_time(record, row, event_time_column)? else {
        return Ok(false);
    };
    Ok((*start_micros..*end_micros).contains(&t))
}
pub(super) fn event_time(
    record: &RecordBatch,
    row: usize,
    event_time_column: &str,
) -> Result<Option<i64>> {
    let array = record
        .column_by_name(event_time_column)
        .ok_or_else(|| error("missing event time"))?;
    if array.is_null(row) {
        return Ok(None);
    }
    let value = match array.data_type() {
        DataType::Timestamp(TimeUnit::Microsecond, _) => array
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(row),
        DataType::Timestamp(TimeUnit::Millisecond, _) => array
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .unwrap()
            .value(row),
        _ => return Err(error("unsupported window time")),
    };
    let t = EventTime::import_timestamp(value, array.data_type(), event_time_column)?.as_micros();
    Ok(Some(t))
}
fn own_dtype(dtype: &DataType) -> DataType {
    match dtype {
        DataType::Timestamp(unit, tz) => {
            DataType::Timestamp(*unit, tz.as_ref().map(|tz| Arc::from(tz.as_ref())))
        }
        _ => dtype.clone(),
    }
}

pub(super) fn schema_quote(schema: &Schema) -> Result<usize> {
    schema
        .fields()
        .iter()
        .try_fold(add(4096, map_quote(schema.metadata())?)?, |n, f| {
            add(n, field_quote(f, 0)?)
        })
}
fn map_quote(map: &std::collections::HashMap<String, String>) -> Result<usize> {
    map.iter().try_fold(0, |n, (k, v)| {
        add(n, add(512, mul(add(k.len(), v.len())?, 4)?)?)
    })
}
fn field_quote(field: &Field, depth: usize) -> Result<usize> {
    if depth > 64 {
        return Err(error(
            "input_normalization_unsupported: schema depth exceeds 64",
        ));
    }
    let base = add(
        1024,
        add(mul(field.name().len(), 4)?, map_quote(field.metadata())?)?,
    )?;
    add(base, type_quote(field.data_type(), depth + 1)?)
}
fn type_quote(dtype: &DataType, depth: usize) -> Result<usize> {
    if depth > 64 {
        return Err(error(
            "input_normalization_unsupported: schema depth exceeds 64",
        ));
    }
    if let Some(field) = nested_field(dtype) {
        return field_quote(field, depth);
    }
    match dtype {
        DataType::Timestamp(_, Some(tz)) => quote_timezone(tz),
        DataType::Struct(fields) => fields
            .iter()
            .try_fold(256, |n, f| add(n, field_quote(f, depth)?)),
        DataType::Union(fields, _) => fields
            .iter()
            .try_fold(256, |n, (_, f)| add(n, field_quote(f, depth)?)),
        _ => pair_type_quote(dtype, depth),
    }
}
fn quote_timezone(tz: &str) -> Result<usize> {
    add(128, mul(tz.len(), 4)?)
}
fn nested_field(dtype: &DataType) -> Option<&Field> {
    match dtype {
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::FixedSizeList(f, _)
        | DataType::Map(f, _) => Some(f),
        _ => None,
    }
}
fn pair_type_quote(dtype: &DataType, depth: usize) -> Result<usize> {
    match dtype {
        DataType::Dictionary(k, v) => add(type_quote(k, depth + 1)?, type_quote(v, depth + 1)?),
        DataType::RunEndEncoded(a, b) => add(field_quote(a, depth)?, field_quote(b, depth)?),
        _ => Ok(256),
    }
}

pub(super) fn fresh_schema(schema: &Schema) -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        schema
            .fields()
            .iter()
            .map(|f| fresh_field(f))
            .collect::<Vec<_>>(),
        schema
            .metadata()
            .iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect(),
    ))
}
fn fresh_field(field: &Field) -> Field {
    Field::new(
        field.name().to_owned(),
        fresh_type(field.data_type()),
        field.is_nullable(),
    )
    .with_metadata(
        field
            .metadata()
            .iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect(),
    )
}
fn fresh_list(dtype: &DataType) -> Option<DataType> {
    let dtype = match dtype {
        DataType::List(f) => DataType::List(Arc::new(fresh_field(f))),
        DataType::LargeList(f) => DataType::LargeList(Arc::new(fresh_field(f))),
        DataType::ListView(f) => DataType::ListView(Arc::new(fresh_field(f))),
        DataType::LargeListView(f) => DataType::LargeListView(Arc::new(fresh_field(f))),
        DataType::FixedSizeList(f, n) => DataType::FixedSizeList(Arc::new(fresh_field(f)), *n),
        DataType::Map(f, sorted) => DataType::Map(Arc::new(fresh_field(f)), *sorted),
        _ => return None,
    };
    Some(dtype)
}
fn fresh_type(dtype: &DataType) -> DataType {
    if let Some(list) = fresh_list(dtype) {
        return list;
    }
    match dtype {
        DataType::Struct(fields) => {
            DataType::Struct(fields.iter().map(|f| fresh_field(f)).collect())
        }
        DataType::Union(fields, mode) => DataType::Union(
            fields
                .iter()
                .map(|(id, f)| (id, Arc::new(fresh_field(f))))
                .collect(),
            *mode,
        ),
        DataType::Dictionary(k, v) => {
            DataType::Dictionary(Box::new(fresh_type(k)), Box::new(fresh_type(v)))
        }
        DataType::RunEndEncoded(a, b) => {
            DataType::RunEndEncoded(Arc::new(fresh_field(a)), Arc::new(fresh_field(b)))
        }
        other => own_dtype(other),
    }
}
fn key_type(dtype: &DataType) -> bool {
    matches!(
        dtype,
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
    ) || matches!(dtype, DataType::Timestamp(TimeUnit::Microsecond, tz) if tz.as_deref().is_none_or(|tz| tz=="UTC"))
}
pub(super) fn backing(batch: &Batch) -> Result<usize> {
    let t = batch.table_payload()?;
    t.batches().iter().try_fold(
        add(512, super::super::sql::ipc::schema_bytes(t.schema())?)?,
        |n, r| {
            r.columns()
                .iter()
                .try_fold(add(n, 256)?, |n, a| add(n, a.get_array_memory_size()))
        },
    )
}
pub(super) fn reserve(
    pool: &Arc<dyn MemoryPool>,
    bytes: usize,
    role: &str,
) -> Result<MemoryReservation> {
    let lease = MemoryConsumer::new(format!("distinct copy_visible_v2 {role}")).register(pool);
    let required = add(pool.reserved(), bytes)?;
    let limit = match pool.memory_limit() {
        MemoryLimit::Finite(n) => n.to_string(),
        MemoryLimit::Infinite => "infinite".into(),
        MemoryLimit::Unknown => "unknown".into(),
    };
    lease.try_grow(bytes).map_err(|_|crate::CalcFlowError::OperatorReason {
        node_id:"distinct".into(),reason_code:crate::StreamingFailureReason::StateBudgetExceeded,
        message:format!("state_budget_exceeded: copy_visible_v2 {role} required={required} bytes limit={limit}"),
    })?;
    Ok(lease)
}
pub(super) fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| error("accounting overflow"))
}
pub(super) fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| error("accounting overflow"))
}
fn align(n: usize) -> Result<usize> {
    let rounded = mul(add(n, 63)? / 64, 64)?;
    if rounded > isize::MAX as usize {
        Err(error("layout exceeds isize"))
    } else {
        Ok(rounded)
    }
}
pub(super) fn error(e: impl std::fmt::Display) -> crate::CalcFlowError {
    invalid("distinct", "execution", e)
}
