use super::{Declaration, spec::AnalyticKind};
use crate::{Batch, Result, dataframe::invalid, operator::distinct::owned};
use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Builder, UInt32Builder, UInt64Builder},
    compute::{LexicographicalComparator, SortColumn, SortOptions, take},
    datatypes::DataType,
    record_batch::RecordBatch,
};
use std::sync::Arc;

pub(crate) fn evaluate(
    batch: &Batch,
    declaration: &Declaration,
    check: &dyn Fn() -> Result<()>,
) -> Result<Batch> {
    let record = &batch.table_payload()?.batches()[0];
    let starts = partition_starts(record, declaration, check)?;
    let mut arrays = record.columns().to_vec();
    for (index, expression) in declaration.expressions.iter().enumerate() {
        check()?;
        let array: ArrayRef = match &expression.0 {
            AnalyticKind::RowNumber { .. } => {
                let mut values = UInt64Builder::with_capacity(starts.len());
                for (row, start) in starts.iter().enumerate() {
                    owned::check_row(row, check)?;
                    values.append_value((row - start + 1) as u64);
                }
                Arc::new(values.finish())
            }
            AnalyticKind::Lag { column, .. } => {
                let mut indices = UInt32Builder::with_capacity(starts.len());
                for (row, start) in starts.iter().enumerate() {
                    owned::check_row(row, check)?;
                    if row == *start {
                        indices.append_null();
                    } else {
                        indices.append_value(u32::try_from(row - 1).map_err(owned::error)?);
                    }
                }
                take(
                    record.column_by_name(column).unwrap().as_ref(),
                    &indices.finish(),
                    None,
                )
                .map_err(owned::error)?
            }
            AnalyticKind::Sum { column, frame, .. } => checked_sum(
                record.column_by_name(column).unwrap(),
                frame,
                &starts,
                index,
                check,
            )?,
        };
        arrays.push(array);
    }
    let schema = super::schema::output_schema(
        &record.schema(),
        &declaration.expressions,
        &declaration.spec,
    )?;
    let result = RecordBatch::try_new(schema, arrays).map_err(owned::error)?;
    Batch::table(vec![result], crate::BatchMetadata::default())
}
fn partition_starts(
    record: &RecordBatch,
    declaration: &Declaration,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<usize>> {
    let keys = declaration
        .spec
        .order()
        .partition_by()
        .iter()
        .map(|name| SortColumn {
            values: record.column_by_name(name).unwrap().clone(),
            options: Some(SortOptions {
                descending: false,
                nulls_first: false,
            }),
        })
        .collect::<Vec<_>>();
    let compare = if keys.is_empty() {
        None
    } else {
        Some(LexicographicalComparator::try_new(&keys).map_err(owned::error)?)
    };
    let mut starts = Vec::with_capacity(record.num_rows());
    let mut start = 0;
    for row in 0..record.num_rows() {
        owned::check_row(row, check)?;
        if row > 0
            && compare
                .as_ref()
                .is_some_and(|c| !c.compare(row - 1, row).is_eq())
        {
            start = row;
        }
        starts.push(start);
    }
    Ok(starts)
}
fn checked_sum(
    values: &ArrayRef,
    frame: &super::RowsFrame,
    starts: &[usize],
    index: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    let signed =
        super::schema::sum_type(values.data_type(), "over.expressions.column")? == DataType::Int64;
    let mut output = SumOutput::new(signed, starts.len());
    let mut sum = 0_i128;
    let mut count = 0_usize;
    let mut lower = 0;
    for (row, start) in starts.iter().copied().enumerate() {
        owned::check_row(row, check)?;
        if row == start {
            sum = 0;
            count = 0;
            lower = start;
        }
        let target = frame.start(row, start);
        while lower < target {
            if let Some(value) = integer(values, lower) {
                sum -= value;
                count -= 1;
            }
            lower += 1;
        }
        if let Some(value) = integer(values, row) {
            sum += value;
            count += 1;
        }
        output.append((count > 0).then_some(sum)).map_err(|()| {
            invalid(
                &format!("over.expressions[{index}].frame"),
                "arithmetic_overflow",
                format!(
                    "SUM operand dtype {} with frame {frame:?} must fit {} at every output",
                    values.data_type(),
                    if signed { "Int64" } else { "UInt64" }
                ),
            )
        })?;
    }
    Ok(output.finish())
}
enum SumOutput {
    Signed(Int64Builder),
    Unsigned(UInt64Builder),
}
impl SumOutput {
    fn new(signed: bool, n: usize) -> Self {
        if signed {
            Self::Signed(Int64Builder::with_capacity(n))
        } else {
            Self::Unsigned(UInt64Builder::with_capacity(n))
        }
    }
    fn append(&mut self, value: Option<i128>) -> std::result::Result<(), ()> {
        match self {
            Self::Signed(b) => {
                b.append_option(value.map(i64::try_from).transpose().map_err(|_| ())?);
            }
            Self::Unsigned(b) => {
                b.append_option(value.map(u64::try_from).transpose().map_err(|_| ())?);
            }
        }
        Ok(())
    }
    fn finish(mut self) -> ArrayRef {
        match &mut self {
            Self::Signed(b) => Arc::new(b.finish()),
            Self::Unsigned(b) => Arc::new(b.finish()),
        }
    }
}
fn integer(array: &ArrayRef, row: usize) -> Option<i128> {
    use datafusion::arrow::array::{
        Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array, UInt16Array, UInt32Array,
        UInt64Array,
    };
    if array.is_null(row) {
        return None;
    }
    macro_rules! value {
        ($ty:ty) => {
            i128::from(array.as_any().downcast_ref::<$ty>().unwrap().value(row))
        };
    }
    Some(match array.data_type() {
        DataType::Int8 => value!(Int8Array),
        DataType::Int16 => value!(Int16Array),
        DataType::Int32 => value!(Int32Array),
        DataType::Int64 => value!(Int64Array),
        DataType::UInt8 => value!(UInt8Array),
        DataType::UInt16 => value!(UInt16Array),
        DataType::UInt32 => value!(UInt32Array),
        DataType::UInt64 => value!(UInt64Array),
        _ => unreachable!("schema admitted integers"),
    })
}
