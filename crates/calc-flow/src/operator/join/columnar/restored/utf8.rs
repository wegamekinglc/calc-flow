use super::{Construction, PayloadChunk, PayloadFunding, ResidentLease, RowPayload, owned_copy};
use crate::operator::join::columnar::{metadata, sparse::ChunkRow};
use crate::time::EventTime;
use datafusion::arrow::{
    array::{ArrayRef, StringArray, make_array},
    buffer::MutableBuffer,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize},
};

pub(in crate::operator::join) fn rounded(bytes: usize) -> Option<usize> {
    bytes.checked_add(63)?.checked_div(64)?.checked_mul(64)
}

pub(in crate::operator::join) fn required(
    record: &RecordBatch,
    expected: &Schema,
    registration: usize,
) -> Option<usize> {
    let backing = record
        .columns()
        .iter()
        .zip(expected.fields())
        .try_fold(0_usize, |total, (array, field)| {
            total.checked_add(column_backing(array, field)?)
        })?;
    [
        backing,
        super::controls_bytes(expected.fields().len())?,
        metadata::schema_inventory(expected)?,
        size_of::<ChunkRow>(),
        registration,
    ]
    .into_iter()
    .try_fold(0_usize, usize::checked_add)
}

fn column_backing(array: &ArrayRef, field: &Field) -> Option<usize> {
    if field.data_type() == &DataType::Utf8 {
        let values = array.as_any().downcast_ref::<StringArray>()?;
        rounded(values.value(0).len())?.checked_add(64)
    } else {
        super::width(field.data_type())?;
        Some(64)
    }
}

fn copy_column(source: &ArrayRef, field: &Field, funding: &Arc<PayloadFunding>) -> ArrayRef {
    if field.data_type() != &DataType::Utf8 {
        return super::copy_column(source, field, funding);
    }
    let source = source
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("certified Utf8 column");
    let selected = source.value(0).as_bytes();
    let end = i32::try_from(selected.len()).expect("certified i32 terminal offset");
    let mut offsets = MutableBuffer::new(2 * size_of::<i32>());
    offsets.extend_from_slice(&0_i32.to_ne_bytes());
    offsets.extend_from_slice(&end.to_ne_bytes());
    let mut values = MutableBuffer::new(selected.len());
    values.extend_from_slice(selected);
    let data = arrow_data::ArrayData::builder(DataType::Utf8)
        .len(1)
        .add_buffer(owned_copy::wrap_buffer(offsets.into(), funding))
        .add_buffer(owned_copy::wrap_buffer(values.into(), funding))
        .build()
        .expect("certified independently owned Utf8 buffers");
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
    construction.schema = Some(super::fresh_schema(expected));
    let lease = construction.lease.take().expect("paid fresh Utf8 row");
    construction.funding = Some(Arc::new(PayloadFunding {
        schema: Arc::clone(construction.schema.as_ref().expect("fresh owned schema")),
        credit: Arc::new(lease.credit),
        _retirement: lease.retirement,
    }));
    let (live_bytes, backing_bytes, terminal_offsets) = copy_columns(&mut construction, record);
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
            terminal_offsets,
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

fn copy_columns(construction: &mut Construction, record: &RecordBatch) -> (usize, usize, usize) {
    let schema = construction.schema.as_ref().expect("fresh owned schema");
    let funding = construction.funding.as_ref().expect("paid row owners");
    construction.columns = Vec::with_capacity(schema.fields().len());
    let mut bytes = 0;
    let mut backing = 0;
    let mut terminal = 0;
    for (source, field) in record.columns().iter().zip(schema.fields()) {
        bytes += if field.data_type() == &DataType::Utf8 {
            let length = source
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("certified Utf8 column")
                .value(0)
                .len();
            backing += rounded(length).expect("checked fresh values capacity") + 64;
            terminal += size_of::<i32>();
            length + size_of::<i32>()
        } else {
            backing += 64;
            super::width(field.data_type()).expect("certified scalar column")
        };
        construction
            .columns
            .push(copy_column(source, field, funding));
    }
    (bytes, backing, terminal)
}
