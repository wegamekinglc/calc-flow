use super::{AnalyticExpr, Declaration, spec::AnalyticKind};
use crate::{Batch, Result, dataframe::invalid, operator::distinct::owned};
use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Builder, PrimitiveArray, UInt32Builder, UInt64Builder},
    compute::{LexicographicalComparator, SortColumn, SortOptions, take},
    datatypes::{
        ArrowPrimitiveType, DataType, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type,
        UInt16Type, UInt32Type, UInt64Type,
    },
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
        arrays.push(evaluate_expression(
            record, expression, &starts, index, check,
        )?);
    }
    let schema = super::schema::output_schema(
        &record.schema(),
        &declaration.expressions,
        &declaration.spec,
    )?;
    let result = RecordBatch::try_new(schema, arrays).map_err(owned::error)?;
    Batch::table(vec![result], crate::BatchMetadata::default())
}
fn evaluate_expression(
    record: &RecordBatch,
    expression: &AnalyticExpr,
    starts: &[usize],
    index: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    match &expression.0 {
        AnalyticKind::RowNumber { .. } => row_numbers(starts, check),
        AnalyticKind::Lag { column, .. } => {
            lag(record.column_by_name(column).unwrap(), starts, check)
        }
        AnalyticKind::Sum { column, frame, .. } => checked_sum(
            record.column_by_name(column).unwrap(),
            frame,
            starts,
            index,
            check,
        ),
    }
}
fn row_numbers(starts: &[usize], check: &dyn Fn() -> Result<()>) -> Result<ArrayRef> {
    let mut values = UInt64Builder::with_capacity(starts.len());
    for (row, start) in starts.iter().enumerate() {
        owned::check_row(row, check)?;
        values.append_value((row - start + 1) as u64);
    }
    Ok(Arc::new(values.finish()))
}
fn lag(values: &ArrayRef, starts: &[usize], check: &dyn Fn() -> Result<()>) -> Result<ArrayRef> {
    let mut indices = UInt32Builder::with_capacity(starts.len());
    for (row, start) in starts.iter().enumerate() {
        owned::check_row(row, check)?;
        if row == *start {
            indices.append_null();
        } else {
            indices.append_value(u32::try_from(row - 1).map_err(owned::error)?);
        }
    }
    take(values.as_ref(), &indices.finish(), None).map_err(owned::error)
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
        (sum, count) = discard_before(values, lower..target, sum, count);
        lower = target;
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
fn discard_before(
    values: &ArrayRef,
    rows: std::ops::Range<usize>,
    mut sum: i128,
    mut count: usize,
) -> (i128, usize) {
    for row in rows {
        if let Some(value) = integer(values, row) {
            sum -= value;
            count -= 1;
        }
    }
    (sum, count)
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
    if array.is_null(row) {
        return None;
    }
    Some(if array.data_type().is_signed_integer() {
        signed_integer(array, row)
    } else {
        unsigned_integer(array, row)
    })
}
fn signed_integer(array: &ArrayRef, row: usize) -> i128 {
    match array.data_type() {
        DataType::Int8 => integer_value::<Int8Type>(array, row),
        DataType::Int16 => integer_value::<Int16Type>(array, row),
        DataType::Int32 => integer_value::<Int32Type>(array, row),
        DataType::Int64 => integer_value::<Int64Type>(array, row),
        _ => unreachable!("schema admitted signed integers"),
    }
}
fn unsigned_integer(array: &ArrayRef, row: usize) -> i128 {
    match array.data_type() {
        DataType::UInt8 => integer_value::<UInt8Type>(array, row),
        DataType::UInt16 => integer_value::<UInt16Type>(array, row),
        DataType::UInt32 => integer_value::<UInt32Type>(array, row),
        DataType::UInt64 => integer_value::<UInt64Type>(array, row),
        _ => unreachable!("schema admitted unsigned integers"),
    }
}
fn integer_value<T: ArrowPrimitiveType>(array: &ArrayRef, row: usize) -> i128
where
    T::Native: Into<i128>,
{
    array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap()
        .value(row)
        .into()
}
