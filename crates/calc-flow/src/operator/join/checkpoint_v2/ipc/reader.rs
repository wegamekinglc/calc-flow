use std::{collections::HashMap, sync::Arc};

use datafusion::arrow::{
    array::ArrayRef,
    buffer::{Buffer, MutableBuffer},
    datatypes::{Field, Schema, SchemaRef},
    ipc::{self, reader::read_record_batch},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryReservation;

use super::super::{
    geometry::invalid,
    payload::{Funding, OwnedPayload},
};
use super::{Plan, accounting, concat, framing, schema};
use crate::Result;

pub(in crate::operator::join::checkpoint_v2) struct Admission<'a> {
    pub(in crate::operator::join::checkpoint_v2) workspace: &'a MemoryReservation,
    pub(in crate::operator::join::checkpoint_v2) resident: &'a Arc<Funding>,
    pub(in crate::operator::join::checkpoint_v2) verifier: &'a framing::VerifierCredit,
    pub(in crate::operator::join::checkpoint_v2) check: &'a dyn Fn() -> Result<()>,
}

pub(in crate::operator::join::checkpoint_v2) fn decode(
    bytes: &[u8],
    expected: SchemaRef,
    rows: usize,
    admission: &Admission<'_>,
) -> Result<Arc<OwnedPayload>> {
    let Admission {
        workspace,
        resident,
        verifier,
        check,
    } = admission;
    let plan = super::inspect(bytes, &expected, rows, verifier, check)?;
    reserve_decode(&plan, &expected, admission)?;
    let decoded_schema = Arc::new(ipc::convert::fb_to_schema(plan.schema));
    let mut reader = Reader {
        cursor: framing::Cursor::new(bytes),
        plan: &plan,
        schema: decoded_schema,
        dictionaries: HashMap::new(),
        workspace,
        verifier,
        resident,
        check,
    };
    reader
        .cursor
        .next(verifier, check)?
        .expect("verified schema");
    let record = reader.read()?;
    check()?;
    let rebound = OwnedPayload::rebind(&record, expected, resident, check);
    check()?;
    rebound
}

fn reserve_decode(plan: &Plan<'_>, schema: &Schema, admission: &Admission<'_>) -> Result<()> {
    accounting::reserve(
        admission.workspace,
        accounting::reader_controls(plan, schema)?,
    )?;
    admission.resident.grow(accounting::body_backing(plan)?)?;
    admission
        .resident
        .grow(accounting::resident_controls(plan, schema)?)?;
    (admission.check)()
}

struct Reader<'a, 'c> {
    cursor: framing::Cursor<'a>,
    plan: &'c Plan<'a>,
    schema: SchemaRef,
    dictionaries: HashMap<i64, ArrayRef>,
    workspace: &'c MemoryReservation,
    verifier: &'c framing::VerifierCredit,
    resident: &'c Arc<Funding>,
    check: &'c dyn Fn() -> Result<()>,
}

impl Reader<'_, '_> {
    fn read(&mut self) -> Result<RecordBatch> {
        while let Some(message) = self.cursor.next(self.verifier, self.check)? {
            let body = self.body(message.body)?;
            if let Some(dictionary) = message.metadata.header_as_dictionary_batch() {
                self.dictionary(dictionary, &body)?;
                continue;
            }
            let record = self.record(message.metadata, &body)?;
            return self.finish_record(record);
        }
        Err(invalid("V2 IPC record batch is missing"))
    }

    fn record(&self, metadata: ipc::Message<'_>, body: &Buffer) -> Result<RecordBatch> {
        let batch = metadata
            .header_as_record_batch()
            .expect("verified record batch");
        self.resident
            .grow(accounting::empty_dictionary_backing(&self.schema)?)?;
        let result = read_record_batch(
            body,
            batch,
            Arc::clone(&self.schema),
            &self.dictionaries,
            None,
            &ipc::MetadataVersion::V5,
        );
        (self.check)()?;
        result.map_err(|_| invalid("V2 IPC record batch decoding failed"))
    }

    fn finish_record(&mut self, record: RecordBatch) -> Result<RecordBatch> {
        if self.cursor.next(self.verifier, self.check)?.is_some() {
            return Err(invalid("V2 IPC contains a message after its record batch"));
        }
        Ok(record)
    }

    fn body(&self, bytes: &[u8]) -> Result<Buffer> {
        let mut body = MutableBuffer::from_len_zeroed(bytes.len());
        for (target, source) in body.as_slice_mut().chunks_mut(4096).zip(bytes.chunks(4096)) {
            (self.check)()?;
            target.copy_from_slice(source);
        }
        (self.check)()?;
        Ok(body.into())
    }

    fn dictionary(&mut self, batch: ipc::DictionaryBatch<'_>, body: &Buffer) -> Result<()> {
        let value_schema = self.dictionary_schema(batch.id())?;
        let result = read_record_batch(
            body,
            batch.data().expect("verified dictionary data"),
            value_schema,
            &self.dictionaries,
            None,
            &ipc::MetadataVersion::V5,
        );
        (self.check)()?;
        let record = result.map_err(|_| invalid("V2 IPC dictionary decoding failed"))?;
        let incoming = Arc::clone(record.column(0));
        let values = if batch.isDelta() {
            self.merge(batch.id(), &incoming)?
        } else {
            incoming
        };
        self.dictionaries.insert(batch.id(), values);
        (self.check)()
    }

    fn dictionary_schema(&self, id: i64) -> Result<SchemaRef> {
        let value_type = schema::dictionary_type(self.plan.schema, &self.schema, id, self.check)?;
        accounting::reserve(
            self.workspace,
            accounting::dictionary_schema_bytes(value_type)?,
        )?;
        let value_schema = Arc::new(Schema::new(vec![Field::new("", value_type.clone(), true)]));
        self.resident
            .grow(accounting::empty_dictionary_backing(&value_schema)?)?;
        Ok(value_schema)
    }

    fn merge(&self, id: i64, incoming: &ArrayRef) -> Result<ArrayRef> {
        let previous = self
            .dictionaries
            .get(&id)
            .ok_or_else(|| invalid("V2 IPC delta dictionary has no existing dictionary"))?;
        concat::admit(
            previous.as_ref(),
            incoming.as_ref(),
            self.workspace,
            self.resident,
            self.check,
        )?;
        (self.check)()?;
        let result = datafusion::arrow::compute::concat(&[previous.as_ref(), incoming.as_ref()]);
        (self.check)()?;
        result.map_err(|_| invalid("V2 IPC delta dictionary concatenation failed"))
    }
}
