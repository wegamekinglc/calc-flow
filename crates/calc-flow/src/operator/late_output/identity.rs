use datafusion::arrow::{
    array::{Array, TimestampMicrosecondArray},
    datatypes::{DataType, TimeUnit},
    record_batch::RecordBatch,
};

use crate::{Batch, CalcFlowError, Result};

pub(crate) fn event_time(
    record: &RecordBatch,
    row: usize,
    index: usize,
    node: &str,
    kind: &str,
) -> Result<i64> {
    record
        .column(index)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .filter(|array| !array.is_null(row))
        .map(|array| array.value(row))
        .ok_or_else(|| {
            error(
                node,
                format!("{kind} event-time value is null or not a microsecond timestamp"),
            )
        })
}

pub(crate) fn validate_keys(
    record: &RecordBatch,
    row: usize,
    keys: impl Iterator<Item = (usize, bool)>,
    node: &str,
    kind: &str,
) -> Result<()> {
    for (index, required) in keys {
        let array = record.column(index);
        if array.is_null(row) {
            if required {
                return Err(error(node, format!("{kind} sequence key value is null")));
            }
        } else if !supported_key(array.data_type()) {
            return Err(error(
                node,
                format!(
                    "{kind} key column has unsupported value type {}",
                    array.data_type()
                ),
            ));
        }
    }
    Ok(())
}

fn supported_key(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(TimeUnit::Microsecond, _)
    )
}

fn error(node: &str, message: String) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node.into(),
        message,
    }
}

pub(crate) struct InputRow<'a> {
    pub(crate) record: &'a RecordBatch,
    pub(crate) record_index: usize,
    pub(crate) row_index: usize,
    pub(crate) envelope_index: usize,
}

impl InputRow<'_> {
    pub(crate) fn diagnostic_index(&self, node: &str) -> Result<u64> {
        u64::try_from(self.envelope_index)
            .map_err(|_| error(node, "late row index overflowed UInt64".into()))
    }
}

pub(crate) fn input_rows(batch: &Batch) -> Result<impl Iterator<Item = InputRow<'_>>> {
    Ok(batch
        .table_payload()?
        .batches()
        .iter()
        .enumerate()
        .flat_map(|(record_index, record)| {
            (0..record.num_rows()).map(move |row_index| (record_index, record, row_index))
        })
        .enumerate()
        .map(
            |(envelope_index, (record_index, record, row_index))| InputRow {
                record,
                record_index,
                row_index,
                envelope_index,
            },
        ))
}
