use std::{
    ops::{Add, Mul},
    sync::Arc,
};

use datafusion::arrow::{
    array::{BooleanArray, Float32Array, Float64Array},
    buffer::{BooleanBuffer, Buffer, NullBuffer},
};

use super::{
    ArrayDType, ArrayExecutionOptions, ArrayView, NativeArray, control, elementwise, invalid,
};
use crate::Result;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

/// Multiplies packed rank-2 floating operands in increasing inner-index order.
///
/// Each product and addition uses the promoted dtype, starting from positive
/// zero. There is no FMA, reassociation or wider hidden accumulator.
///
/// ```
/// use calc_flow::{Result, array::{self, ArrayExecutionOptions, NativeArray}};
/// # fn main() -> Result<()> {
/// let left = NativeArray::from_f32(vec![1.0, 2.0], vec![1, 2], None)?;
/// let right = NativeArray::from_f32(vec![3.0, 4.0], vec![2, 1], None)?;
/// let product = array::matmul(left.view(), right.view(), &ArrayExecutionOptions::default())?;
/// assert_eq!(product.shape(), &[1, 1]);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns `InvalidArgument` for non-floating/nullable operands, incompatible
/// matrix dimensions, zero output width or exceeded execution limits.
pub fn matmul(
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    matmul_checked(left, right, execution, &mut || Ok(()))
}

pub(super) fn matmul_checked(
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    execution: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    check()?;
    let plan = Product::new(left, right)?;
    let bytes = elementwise::required_shape_bytes(plan.dtype, plan.len, 2, false)?;
    execution.validate_allocation(plan.len, bytes, 0)?;
    let result = match plan.dtype {
        ArrayDType::Float32 => product::<f32>(left, right, plan, check),
        ArrayDType::Float64 => product::<f64>(left, right, plan, check),
        ArrayDType::Boolean => unreachable!("matmul admission accepts only floating operands"),
    };
    control::finish(result, check)
}

#[derive(Clone, Copy)]
struct Product {
    rows: usize,
    inner: usize,
    columns: usize,
    len: usize,
    dtype: ArrayDType,
}

impl Product {
    fn new(left: ArrayView<'_>, right: ArrayView<'_>) -> Result<Self> {
        let [rows, inner] = matrix_operand(left, "inputs.left")?;
        let [right_inner, columns] = matrix_operand(right, "inputs.right")?;
        if inner != right_inner || columns == 0 {
            return Err(invalid(
                "inputs.right.shape",
                "expected matching inner size and positive output width",
            ));
        }
        let len = rows.checked_mul(columns).ok_or_else(|| {
            invalid(
                "outputs.output.elements",
                "matrix element count overflowed usize",
            )
        })?;
        let dtype = product_dtype(left.dtype(), right.dtype());
        Ok(Self {
            rows,
            inner,
            columns,
            len,
            dtype,
        })
    }
}

fn product_dtype(left: ArrayDType, right: ArrayDType) -> ArrayDType {
    if left == ArrayDType::Float32 && right == ArrayDType::Float32 {
        ArrayDType::Float32
    } else {
        ArrayDType::Float64
    }
}

fn matrix_operand(input: ArrayView<'_>, field: &str) -> Result<[usize; 2]> {
    if input.dtype() == ArrayDType::Boolean {
        return Err(invalid(
            &format!("{field}.dtype"),
            "expected floating-point values",
        ));
    }
    let [rows, columns] = input.shape() else {
        return Err(invalid(&format!("{field}.shape"), "expected rank two"));
    };
    if input.has_nulls() {
        return Err(invalid(
            &format!("{field}.validity"),
            "matmul does not accept actual nulls",
        ));
    }
    Ok([*rows, *columns])
}

trait Numeric: Copy + Default + Add<Output = Self> + Mul<Output = Self> {
    fn number(view: ArrayView<'_>, index: usize) -> Self;
    fn raw(input: &NativeArray) -> &[Self];
    fn wrap(
        values: Vec<Self>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<NativeArray>;
}

impl Numeric for f32 {
    fn number(view: ArrayView<'_>, index: usize) -> Self {
        view.number32(index).expect("matmul nulls were rejected")
    }
    fn raw(input: &NativeArray) -> &[Self] {
        input
            .as_arrow()
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("dtype selected float32")
            .values()
    }
    fn wrap(
        values: Vec<Self>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<NativeArray> {
        NativeArray::from_f32(values, shape, validity)
    }
}

impl Numeric for f64 {
    fn number(view: ArrayView<'_>, index: usize) -> Self {
        view.number64(index).expect("matmul nulls were rejected")
    }
    fn raw(input: &NativeArray) -> &[Self] {
        input
            .as_arrow()
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("dtype selected float64")
            .values()
    }
    fn wrap(
        values: Vec<Self>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<NativeArray> {
        NativeArray::from_f64(values, shape, validity)
    }
}

fn product<T: Numeric>(
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    plan: Product,
    check: Check<'_>,
) -> Result<NativeArray> {
    let mut values = Vec::with_capacity(plan.len);
    let mut work = 0;
    for index in 0..plan.len {
        control::checkpoint(index, check)?;
        values.push(dot::<T>(left, right, plan, index, &mut work, check)?);
    }
    T::wrap(values, vec![plan.rows, plan.columns], None)
}

fn dot<T: Numeric>(
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    plan: Product,
    output: usize,
    work: &mut usize,
    check: Check<'_>,
) -> Result<T> {
    let row = output / plan.columns;
    let column = output % plan.columns;
    let mut total = T::default();
    for inner in 0..plan.inner {
        control::step(work, check)?;
        let product = T::number(left, row * plan.inner + inner)
            * T::number(right, inner * plan.columns + column);
        total = total + product;
    }
    Ok(total)
}

/// Returns an owned shape alias sharing the input's immutable Arrow storage.
///
/// # Errors
///
/// Returns `InvalidArgument` for unsupported rank, changed element count,
/// overflow or exceeded element/descriptor allocation limits.
pub fn reshape(
    input: &NativeArray,
    shape: &[usize],
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    reshape_checked(input, shape, execution, &mut || Ok(()))
}

pub(super) fn reshape_checked(
    input: &NativeArray,
    shape: &[usize],
    execution: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    check()?;
    let _ = input.view().reshape(shape)?;
    let bytes = shape
        .len()
        .checked_mul(size_of::<usize>())
        .ok_or_else(|| invalid("outputs.output.bytes", "shape byte count overflowed usize"))?;
    execution.validate_allocation(input.element_count(), bytes, 0)?;
    control::finish(
        NativeArray::from_arrow(Arc::clone(input.as_arrow()), shape.to_vec()),
        check,
    )
}

/// Swaps matrix axes into packed row-major order, preserving raw bits and nulls.
///
/// Scalars/vectors are identity aliases. Empty or size-one matrix axes also
/// share storage; other matrices allocate packed values and validity.
///
/// # Errors
///
/// Returns `InvalidArgument` if result elements or new buffers exceed limits.
pub fn transpose(input: &NativeArray, execution: &ArrayExecutionOptions) -> Result<NativeArray> {
    transpose_checked(input, execution, &mut || Ok(()))
}

pub(super) fn transpose_checked(
    input: &NativeArray,
    execution: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    check()?;
    let [rows, columns] = input.shape() else {
        return reshape_checked(input, input.shape(), execution, check);
    };
    if transpose_alias(input) {
        return reshape_checked(input, &[*columns, *rows], execution, check);
    }
    transpose_copy(input, execution, check)
}

fn transpose_alias(input: &NativeArray) -> bool {
    input.element_count() == 0 || input.shape()[0] == 1 || input.shape()[1] == 1
}

fn transpose_copy(
    input: &NativeArray,
    execution: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    let bytes = elementwise::required_shape_bytes(
        input.dtype(),
        input.element_count(),
        2,
        input.as_arrow().nulls().is_some(),
    )?;
    execution.validate_allocation(input.element_count(), bytes, 0)?;
    let result = match input.dtype() {
        ArrayDType::Float32 => transpose_numeric::<f32>(input, check),
        ArrayDType::Float64 => transpose_numeric::<f64>(input, check),
        ArrayDType::Boolean => transpose_boolean(input, check),
    };
    control::finish(result, check)
}

struct ValidityCopy<'a> {
    source: Option<&'a NullBuffer>,
    bits: Option<Vec<u8>>,
    len: usize,
}

impl<'a> ValidityCopy<'a> {
    fn new(input: &'a NativeArray) -> Self {
        let source = input.as_arrow().nulls();
        Self {
            source,
            bits: source.map(|_| vec![0; input.element_count().div_ceil(8)]),
            len: input.element_count(),
        }
    }
    fn record(&mut self, output: usize, source: usize) {
        if let Some(bits) = &mut self.bits {
            set_bit(
                bits,
                output,
                self.source
                    .expect("planned validity exists")
                    .is_valid(source),
            );
        }
    }
    fn finish(self) -> Option<NullBuffer> {
        self.bits
            .map(|bits| NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(bits), 0, self.len)))
    }
}

fn set_bit(bits: &mut [u8], index: usize, value: bool) {
    if value {
        bits[index / 8] |= 1 << (index % 8);
    }
}

fn transpose_source(input: &NativeArray, output: usize) -> usize {
    let rows = input.shape()[0];
    let columns = input.shape()[1];
    (output % rows) * columns + output / rows
}

fn transpose_numeric<T: Numeric>(input: &NativeArray, check: Check<'_>) -> Result<NativeArray> {
    let mut values = Vec::with_capacity(input.element_count());
    let mut validity = ValidityCopy::new(input);
    let source = T::raw(input);
    for index in 0..input.element_count() {
        control::checkpoint(index, check)?;
        let from = transpose_source(input, index);
        values.push(source[from]);
        validity.record(index, from);
    }
    T::wrap(
        values,
        vec![input.shape()[1], input.shape()[0]],
        validity.finish(),
    )
}

fn transpose_boolean(input: &NativeArray, check: Check<'_>) -> Result<NativeArray> {
    let source = input
        .as_arrow()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("dtype selected Boolean");
    let mut values = vec![0; input.element_count().div_ceil(8)];
    let mut validity = ValidityCopy::new(input);
    for index in 0..input.element_count() {
        control::checkpoint(index, check)?;
        let from = transpose_source(input, index);
        set_bit(&mut values, index, source.value(from));
        validity.record(index, from);
    }
    NativeArray::from_boolean(
        BooleanBuffer::new(Buffer::from_vec(values), 0, input.element_count()),
        vec![input.shape()[1], input.shape()[0]],
        validity.finish(),
    )
}
