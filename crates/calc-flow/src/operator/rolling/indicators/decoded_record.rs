//! Admitted physical IPC decoding; scalar reconstruction is a later phase.

use std::io::Cursor;

use datafusion::arrow::{datatypes::Schema, ipc::reader::FileReader, record_batch::RecordBatch};

use super::{ipc_decode_plan, ipc_preflight};
use crate::{CalcFlowError, Result};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
type Admit<'a> = &'a mut dyn FnMut(u64) -> Result<()>;

pub(super) struct DecodedRecord {
    pub(super) record: RecordBatch,
    pub(super) reader_bytes: u64,
}

pub(super) fn read(
    bytes: &[u8],
    expected: &Schema,
    max_rows: u64,
    admit: Admit<'_>,
    check: Check<'_>,
) -> Result<DecodedRecord> {
    check()?;
    let (retained, peak) = requirements(bytes, expected, max_rows, check)?;
    admit(peak)?;
    check()?;
    let record = read_one(bytes, check)?;
    handoff(record, retained, admit, check)
}

fn requirements(
    bytes: &[u8],
    expected: &Schema,
    max_rows: u64,
    check: Check<'_>,
) -> Result<(u64, u64)> {
    let envelope = ipc_preflight::preflight(bytes, expected, max_rows, check)?;
    let plan = ipc_decode_plan::plan(bytes, expected, &envelope, check)?;
    let peak = plan
        .reader_bytes
        .checked_add(plan.scratch_bytes)
        .ok_or_else(invalid)?;
    Ok((plan.reader_bytes, peak))
}

fn handoff(
    record: RecordBatch,
    retained: u64,
    admit: Admit<'_>,
    check: Check<'_>,
) -> Result<DecodedRecord> {
    admit(retained)?;
    check()?;
    Ok(DecodedRecord {
        record,
        reader_bytes: retained,
    })
}

fn read_one(bytes: &[u8], check: Check<'_>) -> Result<RecordBatch> {
    let mut reader = FileReader::try_new(Cursor::new(bytes), None).map_err(|_| invalid())?;
    check()?;
    let record = reader.next().ok_or_else(invalid)?.map_err(|_| invalid())?;
    if reader.next().is_some() {
        return Err(invalid());
    }
    check()?;
    Ok(record)
}

fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling state IPC physical decoding is invalid".into(),
    }
}

#[cfg(test)]
#[path = "decoded_record/tests.rs"]
mod tests;
