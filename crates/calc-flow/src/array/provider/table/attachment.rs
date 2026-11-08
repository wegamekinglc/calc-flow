use super::super::infer::{Descriptor, Shape};
use crate::{
    Batch, Result,
    array::{ArrayDType, ArrayView, NativeArray, control, elementwise, invalid},
};
use datafusion::arrow::{
    array::{Array, ArrayRef, BooleanArray, PrimitiveArray},
    buffer::{BooleanBuffer, Buffer, NullBuffer},
    datatypes::{ArrowPrimitiveType, Field, Float32Type, Float64Type, Schema},
    record_batch::RecordBatch,
};
use std::sync::Arc;

pub(super) fn requirements(descriptor: Descriptor) -> Result<usize> {
    let Shape::Matrix(Some(rows), Some(width)) = descriptor.shape else {
        return Ok(0);
    };
    if width <= 1 {
        return Ok(0);
    }
    elementwise::required_shape_bytes(descriptor.dtype.native(), rows, 0, descriptor.validity)?
        .checked_mul(width)
        .ok_or_else(|| invalid("outputs.bytes", "attached column byte count overflowed"))
}
pub(super) fn attach(
    input: &Batch,
    value: &NativeArray,
    names: &[String],
    budget: usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Batch> {
    let layout = Layout::new(value, input.num_rows(), names.len())?;
    let columns = columns(value, layout, check)?;
    validate_bytes(&columns, layout, budget)?;
    let records = records(input, value, names, &columns, check)?;
    let result = Batch::table(records, input.metadata().clone())?;
    check()?;
    Ok(result)
}
#[derive(Clone, Copy)]
struct Layout {
    rows: usize,
    width: usize,
}
impl Layout {
    fn new(value: &NativeArray, rows: usize, width: usize) -> Result<Self> {
        let matches = match value.shape() {
            [count] => *count == rows && width == 1,
            [count, columns] => *count == rows && *columns == width,
            _ => false,
        };
        if !matches {
            return Err(invalid(
                "outputs.shape",
                "attached result must retain table rows and declared width",
            ));
        }
        Ok(Self { rows, width })
    }
}
fn columns(
    value: &NativeArray,
    layout: Layout,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<ArrayRef>> {
    if layout.width == 1 {
        return Ok(vec![Arc::clone(value.as_arrow())]);
    }
    let mut work = 0;
    (0..layout.width)
        .map(|column| copy_column(value, column, layout, &mut work, check))
        .collect()
}
fn copy_column(
    value: &NativeArray,
    column: usize,
    layout: Layout,
    work: &mut usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<ArrayRef> {
    match value.dtype() {
        ArrayDType::Float32 => floating::<Float32Type>(value, column, layout, work, check),
        ArrayDType::Float64 => floating::<Float64Type>(value, column, layout, work, check),
        ArrayDType::Boolean => boolean(value, column, layout, work, check),
    }
}
fn floating<T: FloatCell>(
    value: &NativeArray,
    column: usize,
    layout: Layout,
    work: &mut usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<ArrayRef> {
    check()?;
    let mut values = vec![T::Native::default(); layout.rows];
    let mut mask = Mask::new(layout.rows, value.as_arrow().nulls().is_some());
    let view = value.view();
    for (row, output) in values.iter_mut().enumerate() {
        control::step(work, check)?;
        let cell = T::read(view, row * layout.width + column);
        mask.record(row, cell.is_some());
        if let Some(cell) = cell {
            *output = cell;
        }
    }
    Ok(Arc::new(PrimitiveArray::<T>::new(
        values.into(),
        mask.finish(),
    )))
}
fn boolean(
    value: &NativeArray,
    column: usize,
    layout: Layout,
    work: &mut usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<ArrayRef> {
    check()?;
    let mut values = vec![0; layout.rows.div_ceil(8)];
    let mut mask = Mask::new(layout.rows, value.as_arrow().nulls().is_some());
    let view = value.view();
    for row in 0..layout.rows {
        control::step(work, check)?;
        let cell = view.boolean(row * layout.width + column);
        mask.record(row, cell.is_some());
        if let Some(cell) = cell {
            values[row / 8] |= u8::from(cell) << (row % 8);
        }
    }
    let values = BooleanBuffer::new(Buffer::from_vec(values), 0, layout.rows);
    Ok(Arc::new(BooleanArray::new(values, mask.finish())))
}
fn validate_bytes(columns: &[ArrayRef], layout: Layout, budget: usize) -> Result<()> {
    if layout.width == 1 {
        return Ok(());
    }
    let actual = columns.iter().try_fold(0_usize, |bytes, column| {
        bytes
            .checked_add(column.get_buffer_memory_size())
            .ok_or_else(|| invalid("outputs.bytes", "attached buffer capacity overflowed"))
    })?;
    if actual > budget {
        return Err(invalid(
            "outputs.bytes",
            "actual attached buffers exceed admitted bytes",
        ));
    }
    Ok(())
}
fn records(
    input: &Batch,
    value: &NativeArray,
    names: &[String],
    columns: &[ArrayRef],
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<RecordBatch>> {
    let table = input.table_payload()?;
    let mut fields = table.schema().fields().to_vec();
    fields.extend(
        names
            .iter()
            .map(|name| Arc::new(Field::new(name, value.as_arrow().data_type().clone(), true))),
    );
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        table.schema().metadata().clone(),
    ));
    let mut offset = 0;
    table
        .batches()
        .iter()
        .map(|record| {
            check()?;
            let mut output = record.columns().to_vec();
            output.extend(
                columns
                    .iter()
                    .map(|column| column.slice(offset, record.num_rows())),
            );
            offset += record.num_rows();
            RecordBatch::try_new(Arc::clone(&schema), output)
                .map_err(|_| invalid("outputs.schema", "could not construct attached table"))
        })
        .collect()
}
struct Mask {
    bits: Option<Vec<u8>>,
    len: usize,
    has_nulls: bool,
}
impl Mask {
    fn new(len: usize, nullable: bool) -> Self {
        Self {
            bits: nullable.then(|| vec![0; len.div_ceil(8)]),
            len,
            has_nulls: false,
        }
    }
    fn record(&mut self, index: usize, valid: bool) {
        self.has_nulls |= !valid;
        if valid && let Some(bits) = &mut self.bits {
            bits[index / 8] |= 1 << (index % 8);
        }
    }
    fn finish(self) -> Option<NullBuffer> {
        self.has_nulls.then(|| {
            NullBuffer::new(BooleanBuffer::new(
                Buffer::from_vec(self.bits.expect("nullable output has an admitted bitmap")),
                0,
                self.len,
            ))
        })
    }
}
trait FloatCell: ArrowPrimitiveType {
    fn read(view: ArrayView<'_>, index: usize) -> Option<Self::Native>;
}
impl FloatCell for Float32Type {
    fn read(view: ArrayView<'_>, index: usize) -> Option<f32> {
        view.number32(index)
    }
}
impl FloatCell for Float64Type {
    fn read(view: ArrayView<'_>, index: usize) -> Option<f64> {
        view.number64(index)
    }
}
