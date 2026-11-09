use datafusion::arrow::ipc::writer::FileWriter;

use super::super::encoded_output::EncodedOutput;
use super::{Check, PreparedRecord};
use crate::{CalcFlowError, Result};

pub(super) fn write(
    record: &PreparedRecord,
    quota: u64,
    node: &str,
    check: Check<'_>,
) -> Result<Vec<u8>> {
    let mut output = EncodedOutput::new(quota, node, check)?;
    if !u64::try_from(output.capacity()).is_ok_and(|capacity| capacity <= quota) {
        return Err(CalcFlowError::Operator {
            node_id: node.into(),
            message: "state.encoded_bytes exceeded its admitted capacity".into(),
        });
    }
    let completed = file(record, &mut output);
    output.finish(completed)
}

fn file(record: &PreparedRecord, output: &mut EncodedOutput<'_>) -> bool {
    let Ok(mut writer) = FileWriter::try_new(output, record.schema.as_schema()) else {
        return false;
    };
    writer
        .write(&record.record)
        .and_then(|()| writer.finish())
        .is_ok()
}
