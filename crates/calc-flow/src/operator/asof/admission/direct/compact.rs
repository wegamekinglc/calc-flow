use super::{RecordBatch, Result, UInt64Array, can_share_batch, take};
use arrow_data::{
    ArrayData,
    transform::{Capacities, MutableArrayData},
};
use datafusion::arrow::{
    array::{ArrayRef, make_array},
    datatypes::{ArrowNativeType, DataType},
};

// Amortize bounded Arrow setup and retain at least 512 bytes of the existing
// identity-index scratch allowance while one column is copied.
const MIN_COPY_ROWS: usize = 64;

pub(super) fn compact_complete_record(record: &RecordBatch) -> Result<RecordBatch> {
    if can_share_batch(record)? {
        return Ok(record.clone());
    }
    let columns = if can_copy(record) {
        record
            .columns()
            .iter()
            .map(copy_column)
            .collect::<Result<Vec<_>>>()?
    } else {
        take_columns(record)?
    };
    RecordBatch::try_new(record.schema(), columns)
        .map_err(|error| super::super::super::arrow_error(&error))
}

fn can_copy(record: &RecordBatch) -> bool {
    record.num_rows() >= MIN_COPY_ROWS
        && record
            .columns()
            .iter()
            .all(|column| column.nulls().is_none() && copy_type(column.data_type()))
}

fn copy_type(data_type: &DataType) -> bool {
    data_type.is_primitive()
        || matches!(
            data_type,
            DataType::Boolean
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::FixedSizeBinary(_)
        )
}

fn copy_column(column: &ArrayRef) -> Result<ArrayRef> {
    let data = column.to_data();
    let capacity = match data.data_type() {
        DataType::Utf8 | DataType::Binary => variable_capacity::<i32>(&data),
        DataType::LargeUtf8 | DataType::LargeBinary => variable_capacity::<i64>(&data),
        _ => Capacities::Array(data.len()),
    };
    let mut copied = MutableArrayData::with_capacities(vec![&data], false, capacity);
    copied
        .try_extend(0, 0, data.len())
        .map_err(|error| super::super::super::arrow_error(&error))?;
    Ok(make_array(copied.freeze()))
}

fn variable_capacity<T: ArrowNativeType>(data: &ArrayData) -> Capacities {
    let offsets = data.buffer::<T>(0);
    // Validated offsets reference this live value buffer, so both endpoints
    // and their span fit the address domain even for LargeUtf8/LargeBinary.
    let start = offsets[0].to_usize().expect("validated Arrow offset");
    let end = offsets[data.len()]
        .to_usize()
        .expect("validated Arrow offset");
    Capacities::Binary(data.len(), Some(end - start))
}

fn take_columns(record: &RecordBatch) -> Result<Vec<ArrayRef>> {
    #[cfg(test)]
    super::tests::record_complete_indices(record.num_rows());
    let indices = UInt64Array::from_iter_values(0..record.num_rows() as u64);
    record
        .columns()
        .iter()
        .map(|column| {
            take(column, &indices, None).map_err(|error| super::super::super::arrow_error(&error))
        })
        .collect()
}
