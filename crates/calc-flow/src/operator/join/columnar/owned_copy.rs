use super::{PayloadChunk, PayloadFunding, metadata};
use crate::operator::join::{SidePlan, StreamJoinOperator};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, OwnedCpuWork, RetirementGuard, WorkOutput,
};
use crate::{Result, StreamOperatorContext, time::EventTime};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, GenericStringArray, OffsetSizeTrait, PrimitiveArray,
        builder::{GenericStringBuilder, PrimitiveBuilder},
        types::{
            Int16Type, Int32Type, Int64Type, TimestampMicrosecondType, TimestampMillisecondType,
            TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type,
            UInt64Type,
        },
    },
    buffer::{Buffer, OffsetBuffer, ScalarBuffer},
    datatypes::{ArrowPrimitiveType, DataType, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use std::{
    fmt::Write,
    sync::{Arc, atomic::AtomicUsize},
};
use tokio_util::bytes::Bytes;

const ARC_HEADER: usize = 2 * size_of::<usize>();
// Locked Arrow 58.3 Bytes: ptr + len + three-word Deallocation, plus Arc header.
const BUFFER_OWNER: usize = 5 * size_of::<usize>() + ARC_HEADER;
const COPY_BYTES: usize = 4096;
const COPY_VISITS: usize = 64;

#[derive(Clone, Copy)]
pub(in crate::operator::join) struct SelectedRow {
    pub source: usize,
    pub row_id: u64,
    pub time: EventTime,
    pub retain: bool,
}

pub(in crate::operator::join) struct CopySelection {
    pub rows: Vec<SelectedRow>,
    _credit: MemoryReservation,
}

impl CopySelection {
    pub(in crate::operator::join) fn reserve(
        operator: &mut StreamJoinOperator,
        rows: usize,
    ) -> Result<Option<Self>> {
        let bytes = rows
            .checked_mul(size_of::<SelectedRow>())
            .and_then(|bytes| bytes.checked_add(reservation_control()?));
        let Some(bytes) = bytes else {
            return Ok(None);
        };
        Ok(operator.optional_credit(bytes)?.map(|credit| Self {
            rows: Vec::with_capacity(rows),
            _credit: credit,
        }))
    }
}

#[derive(Default)]
pub(in crate::operator::join) struct Quantum {
    visits: usize,
    bytes: usize,
}

impl Quantum {
    pub(in crate::operator::join) async fn step(
        &mut self,
        context: &StreamOperatorContext<'_>,
        visits: usize,
        bytes: usize,
    ) -> Result<()> {
        #[cfg(test)]
        super::super::note_join_work(|work| work.quantum_steps += 1);
        debug_assert!(visits <= COPY_VISITS && bytes <= COPY_BYTES);
        if self.visits + visits > COPY_VISITS || self.bytes + bytes > COPY_BYTES {
            #[cfg(test)]
            super::super::note_join_work(|work| work.quantum_yields += 1);
            context.check_cancelled()?;
            tokio::task::yield_now().await;
            context.check_cancelled()?;
            self.visits = 0;
            self.bytes = 0;
        }
        self.visits += visits;
        self.bytes += bytes;
        Ok(())
    }
    pub(in crate::operator::join) async fn grant_admission(
        &mut self,
        context: &StreamOperatorContext<'_>,
        maximum: usize,
    ) -> Result<usize> {
        debug_assert!(maximum > 0);
        let available = self.admission_capacity().min(maximum);
        if available > 0 {
            self.step(context, available, available * 16).await?;
            return Ok(available);
        }
        self.step(context, 1, 16).await?;
        let additional = self.admission_capacity().min(maximum - 1);
        self.visits += additional;
        self.bytes += additional * 16;
        Ok(additional + 1)
    }

    fn admission_capacity(&self) -> usize {
        (COPY_VISITS - self.visits).min((COPY_BYTES - self.bytes) / 16)
    }
}

fn add(left: usize, right: usize) -> Option<usize> {
    left.checked_add(right)
}

struct Construction {
    columns: Vec<ArrayRef>,
    capacities: Vec<usize>,
    inventory: Vec<super::sparse::ChunkRow>,
    backing_bytes: usize,
    terminal_offsets: usize,
    schema: Option<SchemaRef>,
    credit: Option<Arc<MemoryReservation>>,
    funding: Option<Arc<PayloadFunding>>,
}

struct FundedBuilder<B> {
    builder: B,
    funding: Arc<PayloadFunding>,
}

struct FinishedString {
    array: ArrayRef,
    #[cfg(test)]
    allocations: allocation_counter::AllocationInfo,
}

impl<O: OffsetSizeTrait> FundedBuilder<GenericStringBuilder<O>> {
    fn finish(mut self) -> ArrayRef {
        let (offsets, values, nulls) = self.builder.finish().into_parts();
        let count = offsets.len();
        let offsets = ScalarBuffer::new(
            wrap_buffer(offsets.into_inner().into_inner(), &self.funding),
            0,
            count,
        );
        Arc::new(GenericStringArray::<O>::new(
            OffsetBuffer::new(offsets),
            wrap_buffer(values, &self.funding),
            nulls,
        ))
    }
}

impl<O: OffsetSizeTrait> OwnedCpuWork for FundedBuilder<GenericStringBuilder<O>> {
    type Output = FinishedString;

    fn control_bytes(&self) -> Result<usize> {
        Ok(size_of::<Self>())
    }

    fn run(self, stop: &GatherStop) -> Result<FinishedString> {
        stop.check()?;
        #[cfg(test)]
        let (array, allocations) = {
            let mut array = None;
            let allocations = allocation_counter::measure(|| array = Some(self.finish()));
            (array.unwrap(), allocations)
        };
        #[cfg(not(test))]
        let array = self.finish();
        stop.check()?;
        Ok(FinishedString {
            array,
            #[cfg(test)]
            allocations,
        })
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Default, Debug)]
pub(in crate::operator::join) struct StringAllocations {
    pub worker: allocation_counter::AllocationInfo,
    pub dispatch: allocation_counter::AllocationInfo,
    pub max_credit: usize,
    pub output_box: usize,
}

#[cfg(test)]
thread_local! {
    static STRING_ALLOCATIONS: std::cell::Cell<Option<StringAllocations>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
pub(in crate::operator::join) fn observe_string_allocations() {
    STRING_ALLOCATIONS.set(Some(StringAllocations::default()));
}

#[cfg(test)]
pub(in crate::operator::join) fn take_string_allocations() -> StringAllocations {
    STRING_ALLOCATIONS
        .take()
        .expect("enabled worker allocation observation")
}

#[cfg(test)]
async fn observe_dispatch<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|context| {
        let mut outcome = None;
        let allocations = allocation_counter::measure(|| {
            outcome = Some(future.as_mut().poll(context));
        });
        STRING_ALLOCATIONS.with(|stats| {
            if let Some(mut current) = stats.get() {
                current.dispatch += allocations;
                stats.set(Some(current));
            }
        });
        outcome.unwrap()
    })
    .await
}

struct FundedBuffer {
    buffer: Buffer,
    _funding: Arc<PayloadFunding>,
}

impl AsRef<[u8]> for FundedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

pub(super) fn wrap_buffer(buffer: Buffer, funding: &Arc<PayloadFunding>) -> Buffer {
    Buffer::from(Bytes::from_owner(FundedBuffer {
        buffer,
        _funding: Arc::clone(funding),
    }))
}

fn buffer_lease_control() -> usize {
    BUFFER_OWNER
        + size_of::<Bytes>()
        + ARC_HEADER
        + size_of::<FundedBuffer>()
        + size_of::<AtomicUsize>()
        + align_of::<FundedBuffer>()
        - 1
}

fn reservation_control() -> Option<usize> {
    add(
        size_of::<MemoryConsumer>()
            + size_of::<Arc<dyn datafusion::execution::memory_pool::MemoryPool>>()
            + ARC_HEADER,
        3 * "sql-incremental:stream-join-native".len()
            + size_of::<MemoryReservation>()
            + ARC_HEADER,
    )
}

pub(super) fn column_controls() -> Option<usize> {
    let array = size_of::<PrimitiveArray<Int64Type>>().max(size_of::<GenericStringArray<i64>>());
    let owners = 2 * (BUFFER_OWNER + buffer_lease_control());
    // ArrayDataBuilder's buffer Vec grows to four entries; finish resets string offsets to capacity four.
    let transient = 4 * size_of::<Buffer>() + 4 * size_of::<i64>();
    add(
        array + ARC_HEADER + owners + transient,
        size_of::<ArrayRef>() + size_of::<usize>(),
    )
}

fn controls(columns: usize, metadata: usize, rows: usize) -> Option<usize> {
    let columns = columns.checked_mul(column_controls()?)?;
    let inventory = rows.checked_mul(size_of::<super::sparse::ChunkRow>())?;
    add(
        add(add(columns, metadata)?, inventory)?,
        size_of::<PayloadChunk>()
            + ARC_HEADER
            + size_of::<PayloadFunding>()
            + ARC_HEADER
            + reservation_control()?,
    )
}

pub(super) fn fixed_width(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::Int16 | DataType::UInt16 => Some(2),
        DataType::Int32 | DataType::UInt32 => Some(4),
        DataType::Int64 | DataType::UInt64 | DataType::Timestamp(..) => Some(8),
        DataType::UInt8 => Some(1),
        _ => None,
    }
}

fn supported(array: &dyn Array) -> bool {
    array.nulls().is_none()
        && (fixed_width(array.data_type()).is_some()
            || matches!(array.data_type(), DataType::Utf8 | DataType::LargeUtf8))
}

impl StreamJoinOperator {
    fn serial_payload_sql(&self) -> bool {
        if let Some(runtime) = &self.runtime.runtime {
            return runtime.serial_owned_sql();
        }
        self.runtime
            .resources
            .as_ref()
            .is_none_or(|(config, _, _)| config.serial_owned_sql())
    }

    pub(in crate::operator::join) async fn can_copy_payload(
        &self,
        record: &RecordBatch,
        plan: &SidePlan,
        context: &StreamOperatorContext<'_>,
        quantum: &mut Quantum,
    ) -> Result<bool> {
        if !self.payload_native_eligible
            || !self.serial_payload_sql()
            || plan.key_indices.len() > 60
        {
            return Ok(false);
        }
        for column in record.columns() {
            quantum.step(context, 8, 0).await?;
            if !supported(column.as_ref()) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(in crate::operator::join) async fn owned_payload(
        &mut self,
        record: &RecordBatch,
        port: usize,
        rows: &[SelectedRow],
        context: &StreamOperatorContext<'_>,
        quantum: &mut Quantum,
    ) -> Result<Option<Arc<PayloadChunk>>> {
        self.owned_payload_columns(record.columns(), port, rows, context, quantum)
            .await
    }

    pub(in crate::operator::join) async fn owned_payload_columns(
        &mut self,
        columns: &[ArrayRef],
        port: usize,
        rows: &[SelectedRow],
        context: &StreamOperatorContext<'_>,
        quantum: &mut Quantum,
    ) -> Result<Option<Arc<PayloadChunk>>> {
        let Some((mut owners, funding)) = self
            .prepare_payload_copy(columns, port, rows, context, quantum)
            .await?
        else {
            return Ok(None);
        };
        if !copy_columns(columns, &funding, rows, &mut owners, context, quantum).await? {
            return Ok(None);
        }
        context.check_cancelled()?;
        Ok(Some(Arc::new(PayloadChunk {
            backing_bytes: owners.backing_bytes,
            columns: owners.columns,
            schema: Arc::clone(&funding.schema),
            inventory: owners.inventory,
            terminal_offsets: owners.terminal_offsets,
            live_count: AtomicUsize::new(0),
            live_bytes: AtomicUsize::new(0),
            queued: std::sync::atomic::AtomicBool::new(false),
            next: parking_lot::Mutex::new(None),
            _funding: funding,
        })))
    }

    async fn prepare_payload_copy(
        &mut self,
        columns: &[ArrayRef],
        port: usize,
        rows: &[SelectedRow],
        context: &StreamOperatorContext<'_>,
        quantum: &mut Quantum,
    ) -> Result<Option<(Construction, Arc<PayloadFunding>)>> {
        let Some(mut owners) = self.payload_construction(columns.len(), port, rows)? else {
            return Ok(None);
        };
        if !reserve_backing(columns, rows, &mut owners, context, quantum).await? {
            return Ok(None);
        }
        copy_inventory(columns, rows, &mut owners.inventory, context, quantum).await?;
        let Some(funding) =
            initialize_funding(self.input_schema(port), &mut owners, context, quantum).await?
        else {
            return Ok(None);
        };
        Ok(Some((owners, funding)))
    }

    fn payload_construction(
        &mut self,
        columns: usize,
        port: usize,
        rows: &[SelectedRow],
    ) -> Result<Option<Construction>> {
        if rows.is_empty() {
            return Ok(None);
        }
        let Some(metadata) = self.payload_schema_bytes[port] else {
            return Ok(None);
        };
        let Some(control) = controls(columns, metadata, rows.len()) else {
            return Ok(None);
        };
        Ok(self.optional_credit(control)?.map(|credit| Construction {
            columns: Vec::with_capacity(columns),
            capacities: Vec::with_capacity(columns),
            inventory: Vec::with_capacity(rows.len()),
            backing_bytes: 0,
            terminal_offsets: 0,
            schema: None,
            credit: Some(Arc::new(credit)),
            funding: None,
        }))
    }
}

async fn reserve_backing(
    columns: &[ArrayRef],
    rows: &[SelectedRow],
    owners: &mut Construction,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<bool> {
    owners.terminal_offsets =
        plan_capacities(columns, rows, &mut owners.capacities, context, quantum).await?;
    let Some(total) = owners
        .capacities
        .iter()
        .try_fold(0_usize, |bytes, capacity| add(bytes, *capacity))
    else {
        return Ok(false);
    };
    owners.backing_bytes = total;
    Ok(owners
        .credit
        .as_ref()
        .expect("construction credit")
        .try_grow(total)
        .is_ok())
}

async fn initialize_funding(
    schema: &SchemaRef,
    owners: &mut Construction,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<Option<Arc<PayloadFunding>>> {
    owners.schema = Some(metadata::copy_schema(schema, context, quantum).await?);
    let Some(retirement) = payload_retirement(context)? else {
        return Ok(None);
    };
    let funding = Arc::new(PayloadFunding {
        schema: Arc::clone(owners.schema.as_ref().expect("prepaid fresh schema")),
        credit: Arc::clone(owners.credit.as_ref().expect("construction credit")),
        _retirement: retirement,
    });
    owners.funding = Some(Arc::clone(&funding));
    owners.credit = None;
    Ok(Some(funding))
}

fn payload_retirement(context: &StreamOperatorContext<'_>) -> Result<Option<RetirementGuard>> {
    match context.job().gather_owner().retain_retirement() {
        Ok(guard) => Ok(Some(guard)),
        Err(crate::CalcFlowError::Cancelled { .. }) => {
            context.check_cancelled()?;
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

async fn plan_capacities(
    columns: &[ArrayRef],
    rows: &[SelectedRow],
    capacities: &mut Vec<usize>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<usize> {
    let mut terminal = 0;
    for column in columns {
        quantum.step(context, 8, 0).await?;
        capacities.push(column_capacity(column.as_ref(), rows, context, quantum).await?);
        terminal += offset_width(column.data_type());
    }
    Ok(terminal)
}

fn offset_width(data_type: &DataType) -> usize {
    match data_type {
        DataType::Utf8 => size_of::<i32>(),
        DataType::LargeUtf8 => size_of::<i64>(),
        _ => 0,
    }
}

fn cell_bytes(array: &dyn Array, row: usize) -> usize {
    if let Some(width) = fixed_width(array.data_type()) {
        return width;
    }
    match array.data_type() {
        DataType::Utf8 => {
            array
                .as_any()
                .downcast_ref::<GenericStringArray<i32>>()
                .expect("proven string")
                .value(row)
                .len()
                + size_of::<i32>()
        }
        DataType::LargeUtf8 => {
            array
                .as_any()
                .downcast_ref::<GenericStringArray<i64>>()
                .expect("proven large string")
                .value(row)
                .len()
                + size_of::<i64>()
        }
        _ => unreachable!("constructor-proven flat payload type"),
    }
}

async fn copy_inventory(
    columns: &[ArrayRef],
    rows: &[SelectedRow],
    inventory: &mut Vec<super::sparse::ChunkRow>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<()> {
    for row in rows {
        let mut bytes = 0;
        for column in columns {
            quantum.step(context, 8, 2 * size_of::<i64>()).await?;
            bytes += cell_bytes(column.as_ref(), row.source);
        }
        inventory.push(super::sparse::ChunkRow {
            row_id: row.row_id,
            time: row.time,
            bytes,
            live: std::sync::atomic::AtomicBool::new(false),
        });
    }
    Ok(())
}

async fn column_capacity(
    array: &dyn Array,
    rows: &[SelectedRow],
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<usize> {
    if let Some(width) = fixed_width(array.data_type()) {
        return rows
            .len()
            .checked_mul(width)
            .ok_or_else(|| crate::operator::join::charge_overflow("owned payload capacity"));
    }
    match array.data_type() {
        DataType::Utf8 => string_capacity::<i32>(array, rows, context, quantum).await,
        DataType::LargeUtf8 => string_capacity::<i64>(array, rows, context, quantum).await,
        _ => unreachable!("copy eligibility excludes unproven types"),
    }
}

async fn string_capacity<O: OffsetSizeTrait>(
    array: &dyn Array,
    rows: &[SelectedRow],
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<usize> {
    let array = array
        .as_any()
        .downcast_ref::<GenericStringArray<O>>()
        .expect("checked flat type");
    let mut values = 0_usize;
    for row in rows {
        quantum.step(context, 8, 2 * size_of::<O>()).await?;
        values = add(values, array.value(row.source).len())
            .ok_or_else(|| crate::operator::join::charge_overflow("owned string capacity"))?;
    }
    checked_string_capacity::<O>(rows.len(), values)
}

fn checked_string_capacity<O: OffsetSizeTrait>(rows: usize, values: usize) -> Result<usize> {
    let offsets = rows
        .checked_add(1)
        .and_then(|rows| rows.checked_mul(size_of::<O>()));
    let total = offsets.and_then(|offsets| add(values, offsets));
    total
        .filter(|_| O::from_usize(values).is_some())
        .ok_or_else(|| crate::operator::join::charge_overflow("owned string capacity"))
}

async fn copy_columns(
    columns: &[ArrayRef],
    funding: &Arc<PayloadFunding>,
    rows: &[SelectedRow],
    owners: &mut Construction,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<bool> {
    for (index, column) in columns.iter().enumerate() {
        quantum.step(context, 8, 0).await?;
        let Some(column) = copy_column(
            column.as_ref(),
            funding.schema.field(index).data_type(),
            rows,
            owners.capacities[index],
            funding,
            context,
            quantum,
        )
        .await?
        else {
            return Ok(false);
        };
        owners.columns.push(column);
    }
    Ok(true)
}

async fn copy_column(
    array: &dyn Array,
    canonical: &DataType,
    rows: &[SelectedRow],
    capacity: usize,
    funding: &Arc<PayloadFunding>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<Option<ArrayRef>> {
    match canonical {
        DataType::Utf8 => {
            copy_string::<i32>(array, rows, capacity, funding, context, quantum).await
        }
        DataType::LargeUtf8 => {
            copy_string::<i64>(array, rows, capacity, funding, context, quantum).await
        }
        DataType::Timestamp(unit, _) => {
            copy_timestamp(array, canonical, unit, rows, funding, context, quantum)
                .await
                .map(Some)
        }
        DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            copy_signed(array, canonical, rows, funding, context, quantum)
                .await
                .map(Some)
        }
        _ => copy_unsigned(array, canonical, rows, funding, context, quantum)
            .await
            .map(Some),
    }
}

async fn copy_signed(
    array: &dyn Array,
    canonical: &DataType,
    rows: &[SelectedRow],
    funding: &Arc<PayloadFunding>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<ArrayRef> {
    match canonical {
        DataType::Int16 => {
            copy_primitive::<Int16Type>(array, canonical, rows, funding, context, quantum).await
        }
        DataType::Int32 => {
            copy_primitive::<Int32Type>(array, canonical, rows, funding, context, quantum).await
        }
        DataType::Int64 => {
            copy_primitive::<Int64Type>(array, canonical, rows, funding, context, quantum).await
        }
        _ => unreachable!("signed flat type"),
    }
}

async fn copy_unsigned(
    array: &dyn Array,
    canonical: &DataType,
    rows: &[SelectedRow],
    funding: &Arc<PayloadFunding>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<ArrayRef> {
    match canonical {
        DataType::UInt8 => {
            copy_primitive::<UInt8Type>(array, canonical, rows, funding, context, quantum).await
        }
        DataType::UInt16 => {
            copy_primitive::<UInt16Type>(array, canonical, rows, funding, context, quantum).await
        }
        DataType::UInt32 => {
            copy_primitive::<UInt32Type>(array, canonical, rows, funding, context, quantum).await
        }
        DataType::UInt64 => {
            copy_primitive::<UInt64Type>(array, canonical, rows, funding, context, quantum).await
        }
        _ => unreachable!("unsigned flat type"),
    }
}

async fn copy_timestamp(
    array: &dyn Array,
    canonical: &DataType,
    unit: &TimeUnit,
    rows: &[SelectedRow],
    funding: &Arc<PayloadFunding>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<ArrayRef> {
    match unit {
        TimeUnit::Second => {
            copy_primitive::<TimestampSecondType>(array, canonical, rows, funding, context, quantum)
                .await
        }
        TimeUnit::Millisecond => {
            copy_primitive::<TimestampMillisecondType>(
                array, canonical, rows, funding, context, quantum,
            )
            .await
        }
        TimeUnit::Microsecond => {
            copy_primitive::<TimestampMicrosecondType>(
                array, canonical, rows, funding, context, quantum,
            )
            .await
        }
        TimeUnit::Nanosecond => {
            copy_primitive::<TimestampNanosecondType>(
                array, canonical, rows, funding, context, quantum,
            )
            .await
        }
    }
}

async fn copy_primitive<T: ArrowPrimitiveType>(
    array: &dyn Array,
    canonical: &DataType,
    rows: &[SelectedRow],
    funding: &Arc<PayloadFunding>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<ArrayRef> {
    let array = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .expect("checked flat type");
    quantum.step(context, 16, 0).await?;
    let mut owned = FundedBuilder {
        builder: PrimitiveBuilder::<T>::with_capacity(rows.len()).with_data_type(canonical.clone()),
        funding: Arc::clone(funding),
    };
    for row in rows {
        quantum.step(context, 4, 2 * size_of::<T::Native>()).await?;
        owned.builder.append_value(array.value(row.source));
    }
    quantum.step(context, 16, size_of::<i64>()).await?;
    let (data_type, values, nulls) = owned.builder.finish().into_parts();
    let values = ScalarBuffer::new(wrap_buffer(values.into_inner(), funding), 0, rows.len());
    Ok(Arc::new(
        PrimitiveArray::<T>::new(values, nulls).with_data_type(data_type),
    ))
}

fn string_prefix(value: &str) -> usize {
    let mut end = value.len().min((COPY_BYTES - 4) / 2);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    end
}

async fn copy_string<O: OffsetSizeTrait>(
    array: &dyn Array,
    rows: &[SelectedRow],
    capacity: usize,
    funding: &Arc<PayloadFunding>,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<Option<ArrayRef>> {
    let array = array
        .as_any()
        .downcast_ref::<GenericStringArray<O>>()
        .expect("checked flat type");
    let offsets = (rows.len() + 1) * size_of::<O>();
    quantum.step(context, 16, size_of::<O>()).await?;
    let mut owned = FundedBuilder {
        builder: GenericStringBuilder::<O>::with_capacity(rows.len(), capacity - offsets),
        funding: Arc::clone(funding),
    };
    for row in rows {
        quantum.step(context, 8, 2 * size_of::<O>()).await?;
        append_string(
            &mut owned.builder,
            array.value(row.source),
            context,
            quantum,
        )
        .await?;
        quantum.step(context, 4, size_of::<O>()).await?;
        owned.builder.append_value("");
    }
    quantum.step(context, 16, size_of::<i64>()).await?;
    finish_string(owned, context).await
}

async fn finish_string<O: OffsetSizeTrait>(
    owned: FundedBuilder<GenericStringBuilder<O>>,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<ArrayRef>> {
    let dispatch = run_string_work(owned, context);
    #[cfg(test)]
    let result = observe_dispatch(dispatch).await?;
    #[cfg(not(test))]
    let result = dispatch.await?;
    Ok(result.map(|(value, funded_credit)| {
        #[cfg(test)]
        STRING_ALLOCATIONS.with(|stats| {
            if let Some(mut current) = stats.get() {
                current.worker += value.allocations;
                current.max_credit = current.max_credit.max(funded_credit);
                current.output_box = size_of::<FinishedString>();
                stats.set(Some(current));
            }
        });
        let _ = funded_credit;
        value.array
    }))
}

async fn run_string_work<O: OffsetSizeTrait>(
    owned: FundedBuilder<GenericStringBuilder<O>>,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<(FinishedString, usize)>> {
    let credit = owned.funding.credit.new_empty();
    let scope = context.gather_client(GatherOperatorId::new(Arc::from("stream-join-owned-string")));
    let scope = match scope.scope() {
        Ok(scope) => scope,
        Err(error) => return optional_work_failure(error, context),
    };
    let ticket = match scope
        .submit_work(owned, credit, GatherStop::from_job(context.job()))
        .await
    {
        Ok(ticket) => ticket,
        Err(AdmissionFailure::Budget { .. }) => return Ok(None),
        Err(AdmissionFailure::Runtime(error)) => return optional_work_failure(error, context),
    };
    let WorkOutput { value, credit } = ticket.finish().await?;
    let funded = credit.size();
    drop(credit);
    Ok(Some((value, funded)))
}

fn optional_work_failure<T>(
    error: crate::CalcFlowError,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<T>> {
    match error {
        crate::CalcFlowError::Cancelled { .. } => {
            context.check_cancelled()?;
            Ok(None)
        }
        error => Err(error),
    }
}

async fn append_string<O: OffsetSizeTrait>(
    builder: &mut GenericStringBuilder<O>,
    mut value: &str,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<()> {
    while !value.is_empty() {
        quantum.step(context, 4, 4).await?;
        let length = string_prefix(value);
        quantum.step(context, 4, 2 * length).await?;
        builder
            .write_str(&value[..length])
            .expect("prepaid StringBuilder write is infallible");
        value = &value[length..];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CalcFlowError, CancellationToken, JsonMap, StreamJobContext};

    #[test]
    fn test_optional_work_failure_preserves_actual_deadline() {
        let job = StreamJobContext::new(
            1,
            "deadline",
            JsonMap::new(),
            Some(chrono::Utc::now() - chrono::Duration::seconds(1)),
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "match", None);
        let result: Result<Option<()>> = optional_work_failure(
            CalcFlowError::Cancelled {
                run_id: "home".into(),
            },
            &context,
        );
        assert!(matches!(result, Err(CalcFlowError::Cancelled { run_id }) if run_id == "1"));
    }

    #[test]
    fn test_optional_work_failure_preserves_non_cancellation_error() {
        let job =
            StreamJobContext::new(1, "healthy", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "match", None);
        let result: Result<Option<()>> = optional_work_failure(
            CalcFlowError::Internal {
                message: "original".into(),
            },
            &context,
        );
        assert!(
            matches!(result, Err(CalcFlowError::Internal { message }) if message == "original")
        );
    }
}
