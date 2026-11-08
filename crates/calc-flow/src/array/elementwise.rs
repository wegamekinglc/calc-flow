use datafusion::arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};

use super::{ArrayDType, ArrayExecutionOptions, NativeArray, control, invalid};
use crate::Result;

mod view;

pub use view::ArrayView;

#[derive(Clone, Copy)]
struct Shape {
    len: usize,
    scalar: bool,
}

impl Shape {
    fn owned(self) -> Vec<usize> {
        if self.scalar { vec![] } else { vec![self.len] }
    }
}

fn resolve_shape(inputs: &[(ArrayView<'_>, &str)]) -> Result<Shape> {
    let mut vector_len = None;
    for (input, field) in inputs {
        if input.is_scalar() {
            continue;
        }
        if let Some(expected) = vector_len {
            if input.len() != expected {
                return Err(invalid(
                    field,
                    format!(
                        "expected vector length {expected}, received {}",
                        input.len()
                    ),
                ));
            }
        } else {
            vector_len = Some(input.len());
        }
    }
    Ok(Shape {
        len: vector_len.unwrap_or(1),
        scalar: vector_len.is_none(),
    })
}

fn numeric_dtype(input: ArrayView<'_>, field: &str) -> Result<ArrayDType> {
    match input.dtype() {
        ArrayDType::Float32 | ArrayDType::Float64 => Ok(input.dtype()),
        ArrayDType::Boolean => Err(invalid(field, "expected a floating-point operand")),
    }
}

fn numeric_promotion(left: ArrayView<'_>, right: ArrayView<'_>) -> Result<ArrayDType> {
    let left = numeric_dtype(left, "inputs.left.dtype")?;
    let right = numeric_dtype(right, "inputs.right.dtype")?;
    Ok(
        if left == ArrayDType::Float32 && right == ArrayDType::Float32 {
            ArrayDType::Float32
        } else {
            ArrayDType::Float64
        },
    )
}

fn validate_output(
    dtype: ArrayDType,
    shape: Shape,
    nullable: bool,
    execution: &ArrayExecutionOptions,
) -> Result<()> {
    let bytes = required_bytes(dtype, shape.len, shape.scalar, nullable)?;
    execution.validate_allocation(shape.len, bytes, 0)
}

pub(super) fn required_bytes(
    dtype: ArrayDType,
    len: usize,
    scalar: bool,
    nullable: bool,
) -> Result<usize> {
    let values = match dtype {
        ArrayDType::Float32 => len.checked_mul(size_of::<f32>()),
        ArrayDType::Float64 => len.checked_mul(size_of::<f64>()),
        ArrayDType::Boolean => Some(len.div_ceil(8)),
    };
    values
        .and_then(|bytes| bytes.checked_add(if nullable { len.div_ceil(8) } else { 0 }))
        .and_then(|bytes| bytes.checked_add(if scalar { 0 } else { size_of::<usize>() }))
        .ok_or_else(|| invalid("outputs.output.bytes", "output byte count overflowed usize"))
}

struct Validity {
    bits: Option<Vec<u8>>,
    len: usize,
    has_nulls: bool,
}

impl Validity {
    fn new(len: usize, nullable: bool) -> Self {
        Self {
            bits: nullable.then(|| vec![0; len.div_ceil(8)]),
            len,
            has_nulls: false,
        }
    }

    fn record(&mut self, index: usize, valid: bool) {
        if valid {
            if let Some(bits) = &mut self.bits {
                bits[index / 8] |= 1 << (index % 8);
            }
        } else {
            self.has_nulls = true;
        }
    }

    fn finish(self) -> Option<NullBuffer> {
        self.has_nulls.then(|| {
            let bits = self
                .bits
                .expect("nullable output requires a planned bitmap");
            NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(bits), 0, self.len))
        })
    }
}

trait FloatOutput: Copy + Default {
    const DTYPE: ArrayDType;

    fn wrap(
        values: Vec<Self>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<NativeArray>;
}

impl FloatOutput for f32 {
    const DTYPE: ArrayDType = ArrayDType::Float32;

    fn wrap(
        values: Vec<Self>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<NativeArray> {
        NativeArray::from_f32(values, shape, validity)
    }
}

impl FloatOutput for f64 {
    const DTYPE: ArrayDType = ArrayDType::Float64;

    fn wrap(
        values: Vec<Self>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<NativeArray> {
        NativeArray::from_f64(values, shape, validity)
    }
}

fn float_output<T: FloatOutput>(
    shape: Shape,
    nullable: bool,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
    mut row: impl FnMut(usize) -> Option<T>,
) -> Result<NativeArray> {
    check()?;
    validate_output(T::DTYPE, shape, nullable, execution)?;
    let mut values = vec![T::default(); shape.len];
    let mut validity = Validity::new(shape.len, nullable);
    for (index, value) in values.iter_mut().enumerate() {
        control::checkpoint(index, check)?;
        let result = row(index);
        if let Some(result) = result {
            *value = result;
        }
        validity.record(index, result.is_some());
    }
    let result = T::wrap(values, shape.owned(), validity.finish())?;
    check()?;
    Ok(result)
}

fn numeric_unary(
    input: ArrayView<'_>,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
    mut operation32: impl FnMut(f32) -> f32,
    mut operation64: impl FnMut(f64) -> f64,
) -> Result<NativeArray> {
    let dtype = numeric_dtype(input, "inputs.input.dtype")?;
    let shape = resolve_shape(&[(input, "inputs.input.shape")])?;
    match dtype {
        ArrayDType::Float32 => float_output(shape, input.has_nulls(), execution, check, |index| {
            input.number32(index).map(&mut operation32)
        }),
        ArrayDType::Float64 => float_output(shape, input.has_nulls(), execution, check, |index| {
            input.number64(index).map(&mut operation64)
        }),
        ArrayDType::Boolean => unreachable!("numeric dtype was validated"),
    }
}

fn numeric_binary(
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
    mut operation32: impl FnMut(f32, f32) -> f32,
    mut operation64: impl FnMut(f64, f64) -> f64,
) -> Result<NativeArray> {
    let dtype = numeric_promotion(left, right)?;
    let shape = resolve_shape(&[(left, "inputs.left.shape"), (right, "inputs.right.shape")])?;
    let nullable = left.has_nulls() || right.has_nulls();
    match dtype {
        ArrayDType::Float32 => float_output(shape, nullable, execution, check, |index| {
            left.number32(index)
                .zip(right.number32(index))
                .map(|(left, right)| operation32(left, right))
        }),
        ArrayDType::Float64 => float_output(shape, nullable, execution, check, |index| {
            left.number64(index)
                .zip(right.number64(index))
                .map(|(left, right)| operation64(left, right))
        }),
        ArrayDType::Boolean => unreachable!("numeric dtype was validated"),
    }
}

macro_rules! unary {
    ($name:ident, $checked:ident, $doc:literal, $f32:expr, $f64:expr) => {
        #[doc = $doc]
        #[doc = "\n# Errors\n\nReturns `InvalidArgument` for non-floating input or exceeded output limits."]
        pub fn $name(input: ArrayView<'_>, execution: &ArrayExecutionOptions) -> Result<NativeArray> {
            $checked(input, execution, &mut || Ok(()))
        }
        pub(super) fn $checked(input: ArrayView<'_>, execution: &ArrayExecutionOptions, check: &mut dyn FnMut() -> Result<()>) -> Result<NativeArray> {
            numeric_unary(input, execution, check, $f32, $f64)
        }
    };
}

unary!(
    neg,
    neg_checked,
    "Negates floating values, propagating nulls.",
    |value| -value,
    |value| -value
);
unary!(
    abs,
    abs_checked,
    "Returns floating absolute values, propagating nulls.",
    f32::abs,
    f64::abs
);
unary!(
    sqrt,
    sqrt_checked,
    "Returns square roots; negative finite values yield valid NaNs.",
    f32::sqrt,
    f64::sqrt
);
unary!(
    log,
    log_checked,
    "Returns natural logarithms with valid IEEE special values.",
    f32::ln,
    f64::ln
);
unary!(
    exp,
    exp_checked,
    "Returns exponentials with valid IEEE special values.",
    f32::exp,
    f64::exp
);

macro_rules! binary {
    ($name:ident, $checked:ident, $doc:literal, $operator:tt) => {
        #[doc = $doc]
        #[doc = "\n# Errors\n\nReturns `InvalidArgument` for incompatible floating operands, vector lengths, or output limits."]
        pub fn $name(left: ArrayView<'_>, right: ArrayView<'_>, execution: &ArrayExecutionOptions) -> Result<NativeArray> {
            $checked(left, right, execution, &mut || Ok(()))
        }
        pub(super) fn $checked(left: ArrayView<'_>, right: ArrayView<'_>, execution: &ArrayExecutionOptions, check: &mut dyn FnMut() -> Result<()>) -> Result<NativeArray> {
            numeric_binary(left, right, execution, check, |left, right| left $operator right, |left, right| left $operator right)
        }
    };
}

binary!(add, add_checked, "Adds floating operands with explicit scalar broadcasting.", +);
binary!(subtract, subtract_checked, "Subtracts floating operands with explicit scalar broadcasting.", -);
binary!(multiply, multiply_checked, "Multiplies floating operands with explicit scalar broadcasting.", *);
binary!(divide, divide_checked, "Divides floating operands; division by zero remains an IEEE value.", /);

/// Raises floating values to a bounded, valid scalar exponent.
///
/// # Errors
///
/// Returns [`crate::CalcFlowError::InvalidArgument`] for incompatible operands,
/// a non-scalar/null/non-finite exponent, exponent magnitude above 64, or limits.
pub fn pow(
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    pow_checked(left, right, execution, &mut || Ok(()))
}

pub(super) fn pow_checked(
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    numeric_dtype(left, "inputs.left.dtype")?;
    numeric_dtype(right, "inputs.right.dtype")?;
    if !right.is_scalar() {
        return Err(invalid("inputs.right.shape", "exponent must be a scalar"));
    }
    let exponent = right.number64(0);
    if exponent.is_none_or(|value| !value.is_finite() || value.abs() > 64.0) {
        return Err(invalid(
            "right.value",
            "requires a finite non-null scalar exponent with magnitude at most 64",
        ));
    }
    numeric_binary(left, right, execution, check, f32::powf, f64::powf)
}

fn boolean_dtype(input: ArrayView<'_>, field: &str) -> Result<()> {
    if input.dtype() == ArrayDType::Boolean {
        Ok(())
    } else {
        Err(invalid(field, "expected a Boolean operand"))
    }
}

fn boolean_output(
    shape: Shape,
    nullable: bool,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
    mut row: impl FnMut(usize) -> Option<bool>,
) -> Result<NativeArray> {
    check()?;
    validate_output(ArrayDType::Boolean, shape, nullable, execution)?;
    let mut values = vec![0_u8; shape.len.div_ceil(8)];
    let mut validity = Validity::new(shape.len, nullable);
    for index in 0..shape.len {
        control::checkpoint(index, check)?;
        let result = row(index);
        if result == Some(true) {
            values[index / 8] |= 1 << (index % 8);
        }
        validity.record(index, result.is_some());
    }
    let result = NativeArray::from_boolean(
        BooleanBuffer::new(Buffer::from_vec(values), 0, shape.len),
        shape.owned(),
        validity.finish(),
    )?;
    check()?;
    Ok(result)
}

fn compare(
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
    mut operation: impl FnMut(f64, f64) -> bool,
) -> Result<NativeArray> {
    numeric_promotion(left, right)?;
    let shape = resolve_shape(&[(left, "inputs.left.shape"), (right, "inputs.right.shape")])?;
    boolean_output(
        shape,
        left.has_nulls() || right.has_nulls(),
        execution,
        check,
        |index| Some(operation(left.number64(index)?, right.number64(index)?)),
    )
}

fn boolean_binary(
    left: ArrayView<'_>,
    right: ArrayView<'_>,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
    mut operation: impl FnMut(Option<bool>, Option<bool>) -> Option<bool>,
) -> Result<NativeArray> {
    boolean_dtype(left, "inputs.left.dtype")?;
    boolean_dtype(right, "inputs.right.dtype")?;
    let shape = resolve_shape(&[(left, "inputs.left.shape"), (right, "inputs.right.shape")])?;
    boolean_output(
        shape,
        left.has_nulls() || right.has_nulls(),
        execution,
        check,
        |index| operation(left.boolean(index), right.boolean(index)),
    )
}

macro_rules! ordered {
    ($name:ident, $checked:ident, $operator:tt) => {
        #[doc = concat!("Compares floating operands using IEEE `", stringify!($operator), "`, propagating nulls.")]
        #[doc = "\n# Errors\n\nReturns `InvalidArgument` for incompatible floating operands, vector lengths, or output limits."]
        pub fn $name(left: ArrayView<'_>, right: ArrayView<'_>, execution: &ArrayExecutionOptions) -> Result<NativeArray> {
            $checked(left, right, execution, &mut || Ok(()))
        }
        #[allow(clippy::float_cmp, reason = "The public contract requires exact IEEE equality.")]
        pub(super) fn $checked(left: ArrayView<'_>, right: ArrayView<'_>, execution: &ArrayExecutionOptions, check: &mut dyn FnMut() -> Result<()>) -> Result<NativeArray> {
            compare(left, right, execution, check, |left, right| left $operator right)
        }
    };
}

ordered!(less, less_checked, <);
ordered!(less_equal, less_equal_checked, <=);
ordered!(greater, greater_checked, >);
ordered!(greater_equal, greater_equal_checked, >=);

macro_rules! equality {
    ($name:ident, $checked:ident, $operator:tt) => {
        #[doc = concat!("Compares floating or Boolean operands using `", stringify!($operator), "`, propagating nulls.")]
        #[doc = "\n# Errors\n\nReturns `InvalidArgument` for mixed Boolean/floating operands, vector lengths, or output limits."]
        #[allow(clippy::float_cmp, reason = "The public contract requires exact IEEE equality.")]
        pub fn $name(left: ArrayView<'_>, right: ArrayView<'_>, execution: &ArrayExecutionOptions) -> Result<NativeArray> {
            $checked(left, right, execution, &mut || Ok(()))
        }
        #[allow(clippy::float_cmp, reason = "The public contract requires exact IEEE equality.")]
        pub(super) fn $checked(left: ArrayView<'_>, right: ArrayView<'_>, execution: &ArrayExecutionOptions, check: &mut dyn FnMut() -> Result<()>) -> Result<NativeArray> {
            if left.dtype() == ArrayDType::Boolean {
                boolean_binary(left, right, execution, check, |left, right| Some(left? $operator right?))
            } else {
                compare(left, right, execution, check, |left, right| left $operator right)
            }
        }
    };
}

equality!(equal, equal_checked, ==);
equality!(not_equal, not_equal_checked, !=);

/// Negates Boolean values, propagating nulls.
///
/// # Errors
///
/// Returns [`crate::CalcFlowError::InvalidArgument`] for non-Boolean input or limits.
pub fn logical_not(input: ArrayView<'_>, execution: &ArrayExecutionOptions) -> Result<NativeArray> {
    logical_not_checked(input, execution, &mut || Ok(()))
}

pub(super) fn logical_not_checked(
    input: ArrayView<'_>,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    boolean_dtype(input, "inputs.input.dtype")?;
    let shape = resolve_shape(&[(input, "inputs.input.shape")])?;
    boolean_output(shape, input.has_nulls(), execution, check, |index| {
        input.boolean(index).map(|value| !value)
    })
}

fn kleene_and(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

fn kleene_or(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

macro_rules! logical {
    ($name:ident, $checked:ident, $doc:literal, $operation:expr) => {
        #[doc = $doc]
        #[doc = "\n# Errors\n\nReturns `InvalidArgument` for non-Boolean operands, vector lengths, or output limits."]
        pub fn $name(left: ArrayView<'_>, right: ArrayView<'_>, execution: &ArrayExecutionOptions) -> Result<NativeArray> {
            $checked(left, right, execution, &mut || Ok(()))
        }
        pub(super) fn $checked(left: ArrayView<'_>, right: ArrayView<'_>, execution: &ArrayExecutionOptions, check: &mut dyn FnMut() -> Result<()>) -> Result<NativeArray> {
            boolean_binary(left, right, execution, check, $operation)
        }
    };
}

logical!(
    logical_and,
    logical_and_checked,
    "Computes Kleene AND: false AND null is false.",
    kleene_and
);
logical!(
    logical_or,
    logical_or_checked,
    "Computes Kleene OR: true OR null is true.",
    kleene_or
);
logical!(
    logical_xor,
    logical_xor_checked,
    "Computes Boolean XOR, propagating either null.",
    |left, right| Some(left? ^ right?)
);

fn branch_dtype(if_true: ArrayView<'_>, if_false: ArrayView<'_>) -> Result<ArrayDType> {
    match (if_true.dtype(), if_false.dtype()) {
        (ArrayDType::Boolean, ArrayDType::Boolean) => Ok(ArrayDType::Boolean),
        (ArrayDType::Float32, ArrayDType::Float32) => Ok(ArrayDType::Float32),
        (ArrayDType::Boolean, _) | (_, ArrayDType::Boolean) => Err(invalid(
            "if_false.dtype",
            "branches must both be Boolean or both be floating-point",
        )),
        (ArrayDType::Float32, ArrayDType::Float64)
        | (ArrayDType::Float64, ArrayDType::Float32 | ArrayDType::Float64) => {
            Ok(ArrayDType::Float64)
        }
    }
}

fn select<'a, T>(
    condition: Option<bool>,
    if_true: ArrayView<'a>,
    if_false: ArrayView<'a>,
    index: usize,
    value: impl FnOnce(&ArrayView<'a>, usize) -> Option<T>,
) -> Option<T> {
    match condition {
        Some(true) => value(&if_true, index),
        Some(false) => value(&if_false, index),
        None => None,
    }
}

fn selection_layout(
    condition: ArrayView<'_>,
    if_true: ArrayView<'_>,
    if_false: ArrayView<'_>,
) -> Result<(ArrayDType, Shape, bool)> {
    boolean_dtype(condition, "condition.dtype")?;
    let dtype = branch_dtype(if_true, if_false)?;
    let shape = resolve_shape(&[
        (condition, "condition.shape"),
        (if_true, "if_true.shape"),
        (if_false, "if_false.shape"),
    ])?;
    let nullable = condition.has_nulls() || if_true.has_nulls() || if_false.has_nulls();
    Ok((dtype, shape, nullable))
}

/// Selects values using a Boolean condition and explicit scalar broadcasting.
///
/// A null condition produces a null. Only the selected branch value is read;
/// both branches participate in dtype, shape, and allocation validation.
///
/// # Examples
///
/// ```
/// use calc_flow::{Result, array::{ArrayExecutionOptions, ArrayView, where_}};
/// # fn main() -> Result<()> {
/// let result = where_(
///     ArrayView::scalar_boolean(Some(true)),
///     ArrayView::scalar_f32(Some(2.0)),
///     ArrayView::scalar_f64(Some(3.0)),
///     &ArrayExecutionOptions::default(),
/// )?;
/// assert!(result.shape().is_empty());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`crate::CalcFlowError::InvalidArgument`] for a non-Boolean condition,
/// incompatible branch dtypes, vector lengths, or exceeded output limits.
pub fn where_(
    condition: ArrayView<'_>,
    if_true: ArrayView<'_>,
    if_false: ArrayView<'_>,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    where_checked(condition, if_true, if_false, execution, &mut || Ok(()))
}

pub(super) fn where_checked(
    condition: ArrayView<'_>,
    if_true: ArrayView<'_>,
    if_false: ArrayView<'_>,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    let (dtype, shape, nullable) = selection_layout(condition, if_true, if_false)?;
    match dtype {
        ArrayDType::Float32 => float_output(shape, nullable, execution, check, |index| {
            select(
                condition.boolean(index),
                if_true,
                if_false,
                index,
                ArrayView::number32,
            )
        }),
        ArrayDType::Float64 => float_output(shape, nullable, execution, check, |index| {
            select(
                condition.boolean(index),
                if_true,
                if_false,
                index,
                ArrayView::number64,
            )
        }),
        ArrayDType::Boolean => boolean_output(shape, nullable, execution, check, |index| {
            select(
                condition.boolean(index),
                if_true,
                if_false,
                index,
                ArrayView::boolean,
            )
        }),
    }
}
