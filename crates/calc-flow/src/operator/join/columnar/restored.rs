use super::{PayloadChunk, PayloadFunding, RowPayload, metadata, owned_copy, sparse::ChunkRow};
use crate::runtime::streaming::gather_work::RetirementGuard;
use crate::time::EventTime;
use datafusion::arrow::{
    array::{ArrayRef, make_array},
    buffer::{Buffer, MutableBuffer},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize},
};

pub(in crate::operator::join) struct ResidentLease {
    credit: MemoryReservation,
    retirement: RetirementGuard,
}

impl ResidentLease {
    pub(in crate::operator::join) fn new(
        credit: MemoryReservation,
        retirement: RetirementGuard,
    ) -> Self {
        Self { credit, retirement }
    }
}

// Partial arrays and schema precede the lease that funds their construction.
struct Construction {
    columns: Vec<ArrayRef>,
    schema: Option<SchemaRef>,
    inventory: Vec<ChunkRow>,
    funding: Option<Arc<PayloadFunding>>,
    lease: Option<ResidentLease>,
}

pub(in crate::operator::join) fn width(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::Float32 => Some(4),
        DataType::Float64 => Some(8),
        scalar => key_width(scalar),
    }
}

pub(in crate::operator::join) fn key_width(data_type: &DataType) -> Option<usize> {
    owned_copy::fixed_width(data_type)
}

pub(in crate::operator::join) fn column_controls() -> Option<usize> {
    owned_copy::column_controls()
}

pub(in crate::operator::join) fn required(schema: &Schema, registration: usize) -> Option<usize> {
    [
        backing_bytes(schema)?,
        controls_bytes(schema.fields().len())?,
        metadata::schema_inventory(schema)?,
        size_of::<ChunkRow>(),
        registration,
    ]
    .into_iter()
    .try_fold(0_usize, usize::checked_add)
}

fn backing_bytes(schema: &Schema) -> Option<usize> {
    schema.fields().iter().try_fold(0_usize, |bytes, field| {
        width(field.data_type())?;
        bytes.checked_add(64)
    })
}

fn controls_bytes(columns: usize) -> Option<usize> {
    [
        columns.checked_mul(owned_copy::column_controls()?)?,
        arc_bytes::<PayloadChunk>()?,
        arc_bytes::<PayloadFunding>()?,
        arc_bytes::<MemoryReservation>()?,
    ]
    .into_iter()
    .try_fold(0_usize, usize::checked_add)
}

fn arc_bytes<T>() -> Option<usize> {
    let alignment = align_of::<T>().max(align_of::<usize>());
    let header = align_up(2 * size_of::<usize>(), align_of::<T>())?;
    align_up(header.checked_add(size_of::<T>())?, alignment)
}

fn align_up(bytes: usize, alignment: usize) -> Option<usize> {
    bytes
        .checked_add(alignment - 1)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

fn fresh_type(source: &DataType) -> DataType {
    match source {
        DataType::Timestamp(unit, timezone) => DataType::Timestamp(
            *unit,
            timezone.as_ref().map(|text| Arc::from(text.as_ref())),
        ),
        scalar => scalar.clone(),
    }
}

fn fresh_schema(source: &Schema) -> SchemaRef {
    let mut fields = Vec::with_capacity(source.fields().len());
    for field in source.fields() {
        fields.push(Arc::new(Field::new(
            String::from(field.name().as_str()),
            fresh_type(field.data_type()),
            field.is_nullable(),
        )));
    }
    Arc::new(Schema::new(fields))
}

fn copy_column(source: &ArrayRef, field: &Field, funding: &Arc<PayloadFunding>) -> ArrayRef {
    let data = source.to_data();
    let width = width(field.data_type()).expect("certified fixed-width row");
    let start = source.offset() * width;
    let mut values = MutableBuffer::new(width);
    values.extend_from_slice(&data.buffers()[0].as_slice()[start..start + width]);
    let values: Buffer = values.into();
    let data = arrow_data::ArrayData::builder(field.data_type().clone())
        .len(1)
        .add_buffer(owned_copy::wrap_buffer(values, funding))
        .build()
        .expect("certified one-row aligned scalar buffer");
    make_array(data)
}

pub(in crate::operator::join) fn copy_row(
    record: &RecordBatch,
    expected: &Schema,
    row_id: u64,
    time: EventTime,
    lease: ResidentLease,
) -> RowPayload {
    let mut construction = Construction {
        columns: Vec::new(),
        schema: None,
        inventory: Vec::new(),
        funding: None,
        lease: Some(lease),
    };
    construction.schema = Some(fresh_schema(expected));
    let lease = construction
        .lease
        .take()
        .expect("paid fresh-row construction");
    construction.funding = Some(Arc::new(PayloadFunding {
        schema: Arc::clone(construction.schema.as_ref().expect("fresh owned schema")),
        credit: Arc::new(lease.credit),
        _retirement: lease.retirement,
    }));
    copy_columns(&mut construction, record);
    let backing_bytes = construction.columns.len() * 64;
    let live_bytes = expected
        .fields()
        .iter()
        .map(|field| width(field.data_type()).expect("certified scalar type"))
        .sum();
    construction.inventory = Vec::with_capacity(1);
    construction.inventory.push(ChunkRow {
        row_id,
        time,
        bytes: live_bytes,
        live: AtomicBool::new(false),
    });
    RowPayload::Shared {
        chunk: Arc::new(PayloadChunk {
            columns: construction.columns,
            schema: construction.schema.take().expect("owned restored schema"),
            inventory: construction.inventory,
            backing_bytes,
            terminal_offsets: 0,
            live_count: AtomicUsize::new(0),
            live_bytes: AtomicUsize::new(0),
            queued: AtomicBool::new(false),
            next: parking_lot::Mutex::new(None),
            _funding: construction
                .funding
                .take()
                .expect("unique resident transfer"),
        }),
        row: 0,
    }
}

fn copy_columns(construction: &mut Construction, record: &RecordBatch) {
    let schema = construction.schema.as_ref().expect("fresh owned schema");
    let funding = construction.funding.as_ref().expect("paid row owners");
    construction.columns = Vec::with_capacity(schema.fields().len());
    for (source, field) in record.columns().iter().zip(schema.fields()) {
        construction
            .columns
            .push(copy_column(source, field, funding));
    }
}
