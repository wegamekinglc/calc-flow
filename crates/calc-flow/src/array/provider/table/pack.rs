use super::super::ir::DType;
use datafusion::arrow::{
    array::{Array, BooleanArray, PrimitiveArray},
    buffer::{BooleanBuffer, Buffer, NullBuffer},
    datatypes::{ArrowPrimitiveType, Float32Type, Float64Type},
    record_batch::RecordBatch,
};

use crate::{
    Batch, Result,
    array::{ArrayExecutionOptions, NativeArray, control, elementwise},
};

pub(super) fn pack(
    input: &Batch,
    columns: &[usize],
    packing: Packing,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    match packing.dtype {
        DType::Float32 => floating::<Float32Type>(input, columns, packing, check),
        DType::Float64 => floating::<Float64Type>(input, columns, packing, check),
        DType::Bool => boolean(input, columns, packing, check),
    }
}
fn floating<T: Floating>(
    input: &Batch,
    columns: &[usize],
    packing: Packing,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    check()?;
    let mut output = Packed::<T> {
        values: vec![T::Native::default(); packing.len],
        validity: packing
            .nullable
            .then(|| vec![0_u8; packing.len.div_ceil(8)]),
    };
    copy_positions(input, columns, packing, check, &mut output)?;
    let validity = output
        .validity
        .map(|bits| NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(bits), 0, packing.len)));
    check()?;
    T::wrap(output.values, vec![packing.rows, packing.width], validity)
}
struct Packed<T: Floating> {
    values: Vec<T::Native>,
    validity: Option<Vec<u8>>,
}
trait Cells {
    fn copy(&mut self, position: usize, source: &dyn Array, row: usize);
}
impl<T: Floating> Cells for Packed<T> {
    fn copy(&mut self, position: usize, source: &dyn Array, row: usize) {
        let source = source
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .expect("selected dtype was validated");
        if source.is_valid(row) {
            self.values[position] = source.value(row);
            mark_valid(&mut self.validity, position);
        }
    }
}
fn mark_valid(validity: &mut Option<Vec<u8>>, position: usize) {
    if let Some(mask) = validity {
        mask[position / 8] |= 1 << (position % 8);
    }
}
fn copy_positions(
    input: &Batch,
    columns: &[usize],
    packing: Packing,
    check: &mut dyn FnMut() -> Result<()>,
    output: &mut impl Cells,
) -> Result<()> {
    let mut offset = 0;
    let mut chunks = 0;
    for record in input.table_payload()?.batches() {
        control::step(&mut chunks, check)?;
        copy_record(record, columns, packing.width, offset, check, output)?;
        offset += record.num_rows();
    }
    Ok(())
}
fn copy_record(
    record: &RecordBatch,
    columns: &[usize],
    width: usize,
    offset: usize,
    check: &mut dyn FnMut() -> Result<()>,
    output: &mut impl Cells,
) -> Result<()> {
    for row in 0..record.num_rows() {
        copy_row(record, columns, (offset + row) * width, row, check, output)?;
    }
    Ok(())
}
fn copy_row(
    record: &RecordBatch,
    columns: &[usize],
    base: usize,
    row: usize,
    check: &mut dyn FnMut() -> Result<()>,
    output: &mut impl Cells,
) -> Result<()> {
    for (column, index) in columns.iter().enumerate() {
        let position = base + column;
        control::checkpoint(position, check)?;
        output.copy(position, record.column(*index).as_ref(), row);
    }
    Ok(())
}
struct BooleanCells {
    values: Vec<u8>,
    validity: Option<Vec<u8>>,
}
impl Cells for BooleanCells {
    fn copy(&mut self, position: usize, source: &dyn Array, row: usize) {
        let source = source
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("selected dtype was validated");
        if source.is_valid(row) {
            self.values[position / 8] |= u8::from(source.value(row)) << (position % 8);
            mark_valid(&mut self.validity, position);
        }
    }
}
fn boolean(
    input: &Batch,
    columns: &[usize],
    packing: Packing,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    check()?;
    let mut output = BooleanCells {
        values: vec![0; packing.len.div_ceil(8)],
        validity: packing.nullable.then(|| vec![0; packing.len.div_ceil(8)]),
    };
    copy_positions(input, columns, packing, check, &mut output)?;
    let values = BooleanBuffer::new(Buffer::from_vec(output.values), 0, packing.len);
    let validity = output
        .validity
        .map(|bits| NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(bits), 0, packing.len)));
    check()?;
    NativeArray::from_boolean(values, vec![packing.rows, packing.width], validity)
}

#[derive(Clone, Copy)]
pub(super) struct Packing {
    pub dtype: DType,
    pub rows: usize,
    pub width: usize,
    pub len: usize,
    pub nullable: bool,
    pub bytes: usize,
}
pub(super) fn prepare(
    input: &Batch,
    columns: &[usize],
    dtype: DType,
    options: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Packing> {
    let rows = input.num_rows();
    let width = columns.len();
    let len = rows
        .checked_mul(width)
        .ok_or_else(|| super::invalid("inputs.table.shape", "element count overflowed"))?;
    let nullable = has_nulls(input, columns, check)?;
    let bytes = elementwise::required_shape_bytes(dtype.native(), len, 2, nullable)?;
    options.validate_allocation(0, bytes, 0)?;
    Ok(Packing {
        dtype,
        rows,
        width,
        len,
        nullable,
        bytes,
    })
}

trait Floating: ArrowPrimitiveType {
    fn wrap(
        values: Vec<Self::Native>,
        shape: Vec<usize>,
        nulls: Option<NullBuffer>,
    ) -> Result<NativeArray>;
}
impl Floating for Float32Type {
    fn wrap(values: Vec<f32>, shape: Vec<usize>, nulls: Option<NullBuffer>) -> Result<NativeArray> {
        NativeArray::from_f32(values, shape, nulls)
    }
}
impl Floating for Float64Type {
    fn wrap(values: Vec<f64>, shape: Vec<usize>, nulls: Option<NullBuffer>) -> Result<NativeArray> {
        NativeArray::from_f64(values, shape, nulls)
    }
}

fn has_nulls(
    input: &Batch,
    columns: &[usize],
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<bool> {
    let mut visited = 0;
    for record in input.table_payload()?.batches() {
        for index in columns {
            control::step(&mut visited, check)?;
            if record.column(*index).null_count() != 0 {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BatchMetadata;
    use datafusion::arrow::{
        array::Float64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use std::sync::Arc;
    #[test]
    fn test_packing_byte_admission_does_not_use_named_output_element_limit_for_input_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0])),
                Arc::new(Float64Array::from(vec![2.0])),
            ],
        )
        .unwrap();
        let batch = Batch::table(vec![batch], BatchMetadata::default()).unwrap();
        let options = ArrayExecutionOptions::default()
            .with_max_elements(1)
            .with_max_allocation_bytes(32);
        let prepared = prepare(&batch, &[0, 1], DType::Float64, &options, &mut || Ok(())).unwrap();
        assert_eq!(prepared.len, 2);
        assert_eq!(prepared.bytes, 32);
    }
}
