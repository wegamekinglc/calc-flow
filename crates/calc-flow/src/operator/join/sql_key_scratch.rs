use super::{
    CachedRetainedKeys, KEY_COLUMN_PREFIX, MAX_RETAINED_KEY_CACHE_BYTES_PER_SIDE, STATE_RID_COLUMN,
    StoredRow, columnar::Quantum,
};
use crate::runtime::streaming::gather_work::RetirementGuard;
use crate::{DataFusionRuntime, Result, StreamOperatorContext};
use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, PrimitiveArray,
            builder::PrimitiveBuilder,
            types::{Int64Type, UInt64Type},
        },
        buffer::{Buffer, ScalarBuffer},
        datatypes::{ArrowPrimitiveType, DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    execution::memory_pool::{MemoryConsumer, MemoryReservation},
};
use std::sync::{Arc, atomic::AtomicUsize};
use tokio_util::bytes::Bytes;

const ARC_HEADER: usize = 2 * size_of::<usize>();
const ARROW_BUFFER_OWNER: usize = 5 * size_of::<usize>() + ARC_HEADER;

pub(super) struct ScratchFunding {
    schema: SchemaRef,
    _credit: MemoryReservation,
    _retirement: RetirementGuard,
}

#[cfg(test)]
impl ScratchFunding {
    pub(super) fn funded_bytes(&self) -> usize {
        let Self {
            _credit: credit, ..
        } = self;
        credit.size()
    }
}

#[cfg(test)]
impl StateKeys {
    pub(super) fn funded_bytes(&self) -> usize {
        self.funding.funded_bytes()
    }
}

struct FundedBuffer {
    buffer: Buffer,
    _funding: Arc<ScratchFunding>,
}

impl AsRef<[u8]> for FundedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

pub(super) struct StateKeys {
    pub batch: RecordBatch,
    row_ids: Vec<u64>,
    funding: Arc<ScratchFunding>,
}

pub(super) struct KeyBatch {
    pub(super) batch: RecordBatch,
    pub(super) funding: Option<Arc<ScratchFunding>>,
}

impl std::ops::Deref for KeyBatch {
    type Target = RecordBatch;

    fn deref(&self) -> &Self::Target {
        &self.batch
    }
}

impl StateKeys {
    pub(super) fn batch_owner(&self) -> KeyBatch {
        KeyBatch {
            batch: self.batch.clone(),
            funding: Some(Arc::clone(&self.funding)),
        }
    }
}

impl StateKeys {
    pub(super) fn into_cache(self) -> Option<CachedRetainedKeys> {
        let Self {
            batch,
            row_ids,
            funding,
        } = self;
        let bytes = batch
            .columns()
            .iter()
            .try_fold(0_usize, |sum, array| {
                sum.checked_add(array.get_array_memory_size())
            })?
            .checked_add(row_ids.len().checked_mul(size_of::<u64>())?)?;
        (bytes <= MAX_RETAINED_KEY_CACHE_BYTES_PER_SIDE).then_some(CachedRetainedKeys {
            row_ids,
            batch,
            scratch_funding: Some(funding),
        })
    }
}

fn supported(rows: &[StoredRow], indices: &[usize]) -> bool {
    !rows.is_empty()
        && indices.len() <= 15
        && rows.iter().all(|row| {
            indices.iter().all(|&index| {
                let array = row.record.column(index);
                array.nulls().is_none()
                    && matches!(array.data_type(), DataType::Int64 | DataType::UInt64)
            })
        })
}

fn controls(columns: usize, rows: usize) -> Option<usize> {
    let row_bytes = rows.checked_mul(size_of::<u64>())?;
    let buffer = 2 * ARROW_BUFFER_OWNER
        + size_of::<FundedBuffer>()
        + size_of::<AtomicUsize>()
        + align_of::<FundedBuffer>()
        - 1;
    let digits = usize::try_from(usize::MAX.ilog10()).ok()?.checked_add(1)?;
    let column = size_of::<PrimitiveArray<Int64Type>>()
        + ARC_HEADER
        + size_of::<ArrayRef>()
        + 2 * size_of::<Arc<Field>>()
        + size_of::<Field>()
        + ARC_HEADER
        + 2 * (KEY_COLUMN_PREFIX.len() + digits)
        + buffer;
    let registration = size_of::<MemoryConsumer>()
        + ARC_HEADER
        + size_of::<Arc<dyn datafusion::execution::memory_pool::MemoryPool>>()
        + 3 * "sql-incremental:stream-join-native".len();
    columns
        .checked_mul(column)?
        .checked_add(
            size_of::<Schema>()
                + 2 * ARC_HEADER
                + size_of::<ScratchFunding>()
                + ARC_HEADER
                + registration,
        )?
        .checked_add(row_bytes)?
        .checked_add(columns.checked_mul(row_bytes)?)
}

async fn scratch_schema(
    rows: &[StoredRow],
    indices: &[usize],
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<SchemaRef> {
    let mut fields = Vec::with_capacity(indices.len() + 1);
    for (position, &index) in indices.iter().enumerate() {
        quantum.step(context, 16, 0).await?;
        let source = rows[0].record.column(index);
        fields.push(Arc::new(Field::new(
            format!("{KEY_COLUMN_PREFIX}{position}"),
            source.data_type().clone(),
            rows[0].record.schema_ref().field(index).is_nullable(),
        )));
    }
    fields.push(Arc::new(Field::new(
        STATE_RID_COLUMN,
        DataType::UInt64,
        false,
    )));
    quantum.step(context, 3 * fields.len() + 16, 0).await?;
    Ok(Arc::new(Schema::new(fields)))
}

fn wrap_buffer(buffer: Buffer, funding: &Arc<ScratchFunding>) -> Buffer {
    Buffer::from(Bytes::from_owner(FundedBuffer {
        buffer,
        _funding: Arc::clone(funding),
    }))
}

async fn copy_column<T: ArrowPrimitiveType>(
    rows: &[StoredRow],
    index: usize,
    funding: &Arc<ScratchFunding>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<ArrayRef> {
    let mut builder = PrimitiveBuilder::<T>::with_capacity(rows.len());
    for row in rows {
        quantum.step(context, 4, 2 * size_of::<T::Native>()).await?;
        let source = row
            .record
            .column(index)
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .expect("scratch eligibility checks exact primitive types");
        builder.append_value(source.value(row.record.offset()));
    }
    let (data_type, values, _) = builder.finish().into_parts();
    let values = ScalarBuffer::new(wrap_buffer(values.into_inner(), funding), 0, rows.len());
    Ok(Arc::new(
        PrimitiveArray::<T>::new(values, None).with_data_type(data_type),
    ))
}

async fn key_column(
    rows: &[StoredRow],
    index: usize,
    funding: &Arc<ScratchFunding>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<ArrayRef> {
    match rows[0].record.column(index).data_type() {
        DataType::Int64 => copy_column::<Int64Type>(rows, index, funding, context, quantum).await,
        DataType::UInt64 => copy_column::<UInt64Type>(rows, index, funding, context, quantum).await,
        _ => unreachable!("scratch eligibility checks exact primitive types"),
    }
}

async fn row_ids(
    rows: &[StoredRow],
    funding: &Arc<ScratchFunding>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<(ArrayRef, Vec<u64>)> {
    let mut ids = Vec::with_capacity(rows.len());
    let mut copied = Vec::with_capacity(rows.len());
    for row in rows {
        quantum.step(context, 4, 24).await?;
        ids.push(row.row_id);
        copied.push(row.row_id);
    }
    let buffer = wrap_buffer(Buffer::from_vec(copied), funding);
    let values = ScalarBuffer::new(buffer, 0, rows.len());
    Ok((
        Arc::new(PrimitiveArray::<UInt64Type>::new(values, None)),
        ids,
    ))
}

pub(super) async fn state_keys(
    runtime: &DataFusionRuntime,
    rows: &[StoredRow],
    indices: &[usize],
    context: &StreamOperatorContext<'_>,
) -> Result<Option<StateKeys>> {
    if !runtime.serial_owned_sql() || !supported(rows, indices) {
        return Ok(None);
    }
    let Some(credit) = reserve(runtime, indices.len() + 1, rows.len()) else {
        return Ok(None);
    };
    let mut quantum = Quantum::default();
    let schema = scratch_schema(rows, indices, context, &mut quantum).await?;
    let Some(retirement) = scratch_retirement(context)? else {
        return Ok(None);
    };
    let funding = Arc::new(ScratchFunding {
        schema: Arc::clone(&schema),
        _credit: credit,
        _retirement: retirement,
    });
    let mut columns = Vec::with_capacity(indices.len() + 1);
    for &index in indices {
        columns.push(key_column(rows, index, &funding, context, &mut quantum).await?);
    }
    finish_keys(rows, columns, funding, context, &mut quantum)
        .await
        .map(Some)
}

fn scratch_retirement(context: &StreamOperatorContext<'_>) -> Result<Option<RetirementGuard>> {
    match context.job().gather_owner().retain_retirement() {
        Ok(guard) => Ok(Some(guard)),
        Err(crate::CalcFlowError::Cancelled { .. }) => {
            context.check_cancelled()?;
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn reserve(runtime: &DataFusionRuntime, columns: usize, rows: usize) -> Option<MemoryReservation> {
    let bytes = controls(columns, rows)?;
    let credit = runtime.incremental_reservation("stream-join-native");
    credit.try_grow(bytes).ok()?;
    Some(credit)
}

async fn finish_keys(
    rows: &[StoredRow],
    mut columns: Vec<ArrayRef>,
    funding: Arc<ScratchFunding>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<StateKeys> {
    let (ids, row_ids) = row_ids(rows, &funding, context, quantum).await?;
    columns.push(ids);
    quantum.step(context, 3 * columns.len() + 16, 0).await?;
    let batch = RecordBatch::try_new(Arc::clone(&funding.schema), columns)
        .expect("private scratch columns preserve their declared schema");
    Ok(StateKeys {
        batch,
        row_ids,
        funding,
    })
}
