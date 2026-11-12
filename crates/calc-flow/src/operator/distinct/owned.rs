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
    let empty_layout = if history.is_empty() {
        mul(columns.len(), 512)?
    } else {
        0
    };
    let bound = history
        .iter()
        .try_fold(add(4096, empty_layout)?, |n, h| add(n, h.charge()))?;
    let lease = reserve(pool, bound, "combined query input")?;
    let borrowed_metadata = history.last().map(|h| h.batch.metadata());
    let metadata = if let Some(m) = borrowed_metadata {
        super::super::sql::metadata::reserve_with(
            m,
            "distinct combined metadata",
            reserve(pool, 0, "combined metadata")?,
        )?
    } else {
        reserve(pool, 4096, "combined metadata")?
    };
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
    let batch = Batch::table(records, borrowed_metadata.cloned().unwrap_or_default())?;
    if history.is_empty() {
        return normalize(&batch, columns, pool, None, check);
    }
    Ok(OwnedBatch {
        batch,
        data_lease: lease,
        metadata_lease: metadata,
    })
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
    let table = raw.table_payload()?;
    let mut bound = 4096;
    for name in columns {
        let field = table.schema().field_with_name(name).map_err(error)?;
        bound = add(bound, add(512, mul(field.name().len(), 2)?)?)?;
    }
    for record in table.batches() {
        let n = count_record(record, domain, check)?;
        let selected = |row| member(record, row, domain);
        bound = add(bound, 256)?;
        for name in columns {
            let column = record
                .column_by_name(name)
                .ok_or_else(|| error("missing key"))?;
            bound = add(bound, column_bound(column, n, &selected, check)?)?;
        }
    }
    let data = reserve(pool, bound, "normalization/schema")?;
    let metadata = super::super::sql::metadata::reserve_with(
        raw.metadata(),
        "distinct copy_visible_v2 metadata",
        MemoryConsumer::new("distinct.metadata").register(pool),
    )?;
    #[cfg(test)]
    tests::before_target_allocation();
    let schema = projected_schema(table.schema(), columns)?;
    let mut records = Vec::with_capacity(table.batches().len());
    for record in table.batches() {
        check()?;
        let n = count_record(record, domain, check)?;
        let selected = |row| member(record, row, domain);
        let mut arrays = Vec::with_capacity(columns.len());
        for name in columns {
            arrays.push(copy_column(
                record.column_by_name(name).unwrap(),
                n,
                &selected,
                check,
            )?);
        }
        records.push(RecordBatch::try_new(schema.clone(), arrays).map_err(error)?);
    }
    let owned_metadata = BatchMetadata::new(
        raw.metadata().source().to_owned(),
        raw.metadata().sequence(),
        raw.metadata().attributes().clone(),
    )?;
    let batch = Batch::table(records, owned_metadata)?;
    if backing(&batch)? > data.size() {
        return Err(error("normalization exceeded prepaid layout"));
    }
    check()?;
    Ok(OwnedBatch {
        batch,
        data_lease: data,
        metadata_lease: metadata,
    })
}

fn column_bound(
    array: &ArrayRef,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let values = match array.data_type() {
        DataType::Utf8 => string_bytes(
            array.as_any().downcast_ref::<StringArray>().unwrap(),
            selected,
            check,
        )?,
        DataType::LargeUtf8 => string_bytes(
            array.as_any().downcast_ref::<LargeStringArray>().unwrap(),
            selected,
            check,
        )?,
        DataType::Boolean => n.div_ceil(8),
        dtype => mul(
            n,
            dtype
                .primitive_width()
                .ok_or_else(|| error("input_normalization_unsupported"))?,
        )?,
    };
    let offsets = match array.data_type() {
        DataType::Utf8 => align(mul(add(n, 1)?, 4)?)?,
        DataType::LargeUtf8 => align(mul(add(n, 1)?, 8)?)?,
        _ => 0,
    };
    add(
        256,
        add(
            align(values)?,
            add(
                offsets,
                if array.null_count() > 0 {
                    align(n.div_ceil(8))?
                } else {
                    0
                },
            )?,
        )?,
    )
}
fn string_bytes<O: OffsetSizeTrait>(
    array: &GenericStringArray<O>,
    selected: &dyn Fn(usize) -> Result<bool>,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let mut bytes = 0;
    for i in 0..array.len() {
        if i % 256 == 0 {
            check()?;
        }
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
        if i % 256 == 0 {
            check()?;
        }
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
    macro_rules! primitive {
        ($ty:ty) => {{
            let src = array
                .as_any()
                .downcast_ref::<PrimitiveArray<$ty>>()
                .ok_or_else(|| error("input_normalization_unsupported"))?;
            let mut values = MutableBuffer::with_capacity(mul(
                n,
                size_of::<<$ty as datafusion::arrow::datatypes::ArrowPrimitiveType>::Native>(),
            )?);
            for i in 0..src.len() {
                if i % 256 == 0 {
                    check()?;
                }
                if selected(i)? {
                    values.push(if src.is_null(i) {
                        Default::default()
                    } else {
                        src.value(i)
                    });
                }
            }
            Arc::new(
                PrimitiveArray::<$ty>::new(ScalarBuffer::new(values.into(), 0, n), nulls)
                    .with_data_type(own_dtype(array.data_type())),
            ) as ArrayRef
        }};
    }
    let copied = match array.data_type() {
        DataType::Int8 => primitive!(Int8Type),
        DataType::Int16 => primitive!(Int16Type),
        DataType::Int32 => primitive!(Int32Type),
        DataType::Int64 => primitive!(Int64Type),
        DataType::UInt8 => primitive!(UInt8Type),
        DataType::UInt16 => primitive!(UInt16Type),
        DataType::UInt32 => primitive!(UInt32Type),
        DataType::UInt64 => primitive!(UInt64Type),
        DataType::Date32 => primitive!(Date32Type),
        DataType::Date64 => primitive!(Date64Type),
        DataType::Timestamp(TimeUnit::Microsecond, _) => primitive!(TimestampMicrosecondType),
        DataType::Boolean => {
            let src = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            let mut values = MutableBuffer::from_len_zeroed(n.div_ceil(8));
            let mut j = 0;
            for i in 0..src.len() {
                if i % 256 == 0 {
                    check()?;
                }
                if selected(i)? {
                    if src.is_valid(i) && src.value(i) {
                        values.as_slice_mut()[j / 8] |= 1 << (j % 8);
                    }
                    j += 1;
                }
            }
            Arc::new(BooleanArray::new(
                BooleanBuffer::new(values.into(), 0, n),
                nulls,
            ))
        }
        DataType::Utf8 => Arc::new(copy_string(
            array.as_any().downcast_ref::<StringArray>().unwrap(),
            n,
            selected,
            nulls,
            check,
        )?),
        DataType::LargeUtf8 => Arc::new(copy_string(
            array.as_any().downcast_ref::<LargeStringArray>().unwrap(),
            n,
            selected,
            nulls,
            check,
        )?),
        _ => return Err(error("input_normalization_unsupported")),
    };
    Ok(copied)
}
fn copy_string<O: OffsetSizeTrait>(
    src: &GenericStringArray<O>,
    n: usize,
    selected: &dyn Fn(usize) -> Result<bool>,
    nulls: Option<NullBuffer>,
    check: &dyn Fn() -> Result<()>,
) -> Result<GenericStringArray<O>> {
    let bytes = string_bytes(src, selected, check)?;
    let mut offsets = MutableBuffer::with_capacity(mul(add(n, 1)?, size_of::<O>())?);
    let mut values = MutableBuffer::with_capacity(bytes);
    offsets.push(O::from_usize(0).unwrap());
    for i in 0..src.len() {
        if i % 256 == 0 {
            check()?;
        }
        if !selected(i)? {
            continue;
        }
        if src.is_valid(i) {
            values.extend_from_slice(src.value(i).as_bytes());
        }
        offsets.push(O::from_usize(values.len()).ok_or_else(|| error("string offset overflow"))?);
    }
    Ok(GenericStringArray::new(
        OffsetBuffer::new(ScalarBuffer::new(offsets.into(), 0, n + 1)),
        values.into(),
        nulls,
    ))
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
        if i % 256 == 0 {
            check()?;
        }
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
    let array = record
        .column_by_name(event_time_column)
        .ok_or_else(|| error("missing event time"))?;
    if array.is_null(row) {
        return Ok(false);
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
    Ok(*start_micros <= t && t < *end_micros)
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
    match dtype {
        DataType::Timestamp(_, Some(tz)) => add(128, mul(tz.len(), 4)?),
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::FixedSizeList(f, _)
        | DataType::Map(f, _) => field_quote(f, depth),
        DataType::Struct(fields) => fields
            .iter()
            .try_fold(256, |n, f| add(n, field_quote(f, depth)?)),
        DataType::Union(fields, _) => fields
            .iter()
            .try_fold(256, |n, (_, f)| add(n, field_quote(f, depth)?)),
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
fn fresh_type(dtype: &DataType) -> DataType {
    match dtype {
        DataType::List(f) => DataType::List(Arc::new(fresh_field(f))),
        DataType::LargeList(f) => DataType::LargeList(Arc::new(fresh_field(f))),
        DataType::ListView(f) => DataType::ListView(Arc::new(fresh_field(f))),
        DataType::LargeListView(f) => DataType::LargeListView(Arc::new(fresh_field(f))),
        DataType::FixedSizeList(f, n) => DataType::FixedSizeList(Arc::new(fresh_field(f)), *n),
        DataType::Map(f, sorted) => DataType::Map(Arc::new(fresh_field(f)), *sorted),
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
