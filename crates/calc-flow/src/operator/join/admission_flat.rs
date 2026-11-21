use crate::{
    CalcFlowError, Result,
    batch::checked_accumulate,
    operator::row_cost::{fixed_width, variable_offsets},
};
use datafusion::arrow::record_batch::RecordBatch;

pub(super) fn flat_row_charges(record: &RecordBatch, overhead: u64) -> Result<Option<Vec<u64>>> {
    let bitmap_per_row = record.num_columns().div_ceil(8);
    let overhead = usize::try_from(overhead).map_err(|_| overflow_charge())?;
    let mut constant = checked_accumulate(overhead, bitmap_per_row, "batch")?;
    for column in record.columns() {
        let Some(width) = fixed_width(column.data_type()) else {
            return Ok(None);
        };
        constant = checked_accumulate(constant, width, "batch")?;
    }
    let mut charges = vec![constant; record.num_rows()];
    for column in record.columns() {
        if let Some(offsets) = variable_offsets(column.as_ref()) {
            offsets.add_charges(&mut charges)?;
        }
    }
    let charges = charges
        .into_iter()
        .map(|charge| u64::try_from(charge).map_err(|_| overflow_charge()))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(charges))
}

fn overflow_charge() -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "batch".into(),
        message: "row charge overflowed".into(),
    }
}
