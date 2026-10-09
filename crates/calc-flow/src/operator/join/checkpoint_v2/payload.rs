use std::sync::Arc;

use arrow_data::ArrayData;
use datafusion::arrow::{
    array::make_array,
    buffer::{BooleanBuffer, Buffer, NullBuffer},
    datatypes::{DataType, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryReservation;
use tokio_util::bytes::Bytes;

use crate::runtime::streaming::gather_work::RetirementGuard;
use crate::{CalcFlowError, Result};

pub(in crate::operator::join) struct Funding {
    _schemas: [SchemaRef; 2],
    credit: Arc<MemoryReservation>,
    _retirement: Option<RetirementGuard>,
}

impl Funding {
    pub(super) fn new(
        schemas: [SchemaRef; 2],
        credit: MemoryReservation,
        retirement: Option<RetirementGuard>,
    ) -> Arc<Self> {
        Arc::new(Self {
            _schemas: schemas,
            credit: Arc::new(credit),
            _retirement: retirement,
        })
    }

    pub(super) fn grow(&self, bytes: usize) -> Result<()> {
        self.credit
            .try_grow(bytes)
            .map_err(|_| CalcFlowError::Internal {
                message: "V2 payload resident credit admission failed".into(),
            })
    }

    #[cfg(test)]
    pub(super) fn paid(&self) -> usize {
        self.credit.size()
    }
}

// Array and batch controls stay private; escaping buffers retain the separate funding owner.
pub(in crate::operator::join) struct OwnedPayload {
    record: RecordBatch,
    _funding: Arc<Funding>,
}

impl OwnedPayload {
    pub(super) fn rebind(
        record: &RecordBatch,
        expected: SchemaRef,
        funding: &Arc<Funding>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Arc<Self>> {
        check()?;
        if record.schema_ref() != &expected {
            return Err(mismatch("V2 payload schema differs from the input schema"));
        }
        let columns = record
            .columns()
            .iter()
            .zip(expected.fields())
            .map(|(array, field)| {
                rebind_data(&array.to_data(), field.data_type(), funding, check).map(make_array)
            })
            .collect::<Result<Vec<_>>>()?;
        let record = RecordBatch::try_new(expected, columns)
            .map_err(|_| mismatch("V2 canonical payload reconstruction failed"))?;
        check()?;
        Ok(Arc::new(Self {
            record,
            _funding: Arc::clone(funding),
        }))
    }

    pub(in crate::operator::join) fn record(&self) -> &RecordBatch {
        &self.record
    }

    #[cfg(test)]
    pub(in crate::operator::join) fn funded_owner(self: &Arc<Self>) -> (usize, usize) {
        let Self {
            _funding: funding, ..
        } = self.as_ref();
        (Arc::as_ptr(self) as usize, funding.paid())
    }
}

struct PaidBacking {
    original: Buffer,
    _funding: Arc<Funding>,
}

pub(super) fn buffer_owner_bytes() -> usize {
    let buffer_owner = 5 * size_of::<usize>() + 2 * size_of::<usize>();
    buffer_owner
        + size_of::<PaidBacking>()
        + size_of::<Bytes>()
        + size_of::<std::sync::atomic::AtomicUsize>()
        + 2 * size_of::<usize>()
        + align_of::<PaidBacking>()
        - 1
}

impl AsRef<[u8]> for PaidBacking {
    fn as_ref(&self) -> &[u8] {
        self.original.as_slice()
    }
}

fn wrap_buffer(buffer: Buffer, funding: &Arc<Funding>) -> Buffer {
    Buffer::from(Bytes::from_owner(PaidBacking {
        original: buffer,
        _funding: Arc::clone(funding),
    }))
}

fn rebind_data(
    source: &ArrayData,
    expected: &DataType,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayData> {
    check()?;
    if source.data_type() != expected {
        return Err(mismatch(
            "V2 payload child type differs from the input schema",
        ));
    }
    let children = rebind_children(source, expected, funding, check)?;
    let buffers = source
        .buffers()
        .iter()
        .cloned()
        .map(|buffer| wrap_buffer(buffer, funding))
        .collect();
    let nulls = source.nulls().map(|nulls| {
        NullBuffer::new(BooleanBuffer::new(
            wrap_buffer(nulls.buffer().clone(), funding),
            nulls.offset(),
            nulls.len(),
        ))
    });
    source
        .clone()
        .into_builder()
        .data_type(expected.clone())
        .buffers(buffers)
        .nulls(nulls)
        .child_data(children)
        .build()
        .map_err(|_| mismatch("V2 canonical child reconstruction failed"))
}

fn rebind_children(
    source: &ArrayData,
    expected: &DataType,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<ArrayData>> {
    source
        .child_data()
        .iter()
        .enumerate()
        .map(|(index, child)| {
            let expected = child_type(expected, index)
                .ok_or_else(|| mismatch("V2 payload has an unexpected child array"))?;
            rebind_data(child, expected, funding, check)
        })
        .collect()
}

fn child_type(data_type: &DataType, index: usize) -> Option<&DataType> {
    match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => (index == 0).then(|| field.data_type()),
        DataType::Dictionary(_, value) => (index == 0).then_some(value.as_ref()),
        DataType::Struct(fields) => fields.get(index).map(|field| field.data_type()),
        DataType::Union(fields, _) => fields.iter().nth(index).map(|(_, field)| field.data_type()),
        DataType::RunEndEncoded(run_ends, values) => match index {
            0 => Some(run_ends.data_type()),
            1 => Some(values.data_type()),
            _ => None,
        },
        _ => None,
    }
}

fn mismatch(message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: message.into(),
    }
}
