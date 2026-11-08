use datafusion::arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};

use super::{Reduction, ReductionOptions, better_extremum, exact::ExactSum};
use crate::{
    Result,
    array::{
        ArrayDType, ArrayExecutionOptions, ArrayView, NativeArray, control, elementwise, invalid,
    },
};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(in crate::array) fn reduce_checked(
    input: ArrayView<'_>,
    operation: Reduction,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    check()?;
    validate_dtype(input.dtype())?;
    let plan = Lanes::new(input, options)?;
    match operation {
        Reduction::Sum | Reduction::Mean => {
            execute::<Exact, f64>(input, plan, options, operation, execution, check)
        }
        Reduction::Min | Reduction::Max => {
            extremes(input, plan, options, operation, execution, check)
        }
    }
}

fn validate_dtype(dtype: ArrayDType) -> Result<()> {
    if dtype == ArrayDType::Boolean {
        Err(invalid("input.dtype", "reductions require floating values"))
    } else {
        Ok(())
    }
}

// The flag may be conservative; it never proves that a nonempty lane is null.
pub(super) fn descriptor_requirements(
    dtype: ArrayDType,
    shape: &[usize],
    nullable: bool,
    operation: Reduction,
    options: ReductionOptions,
) -> Result<(usize, usize, usize)> {
    validate_dtype(dtype)?;
    let plan = Lanes::from_shape(shape, options)?;
    let output = output_dtype(dtype, operation);
    let scratch = descriptor_scratch(plan, operation);
    let bytes = elementwise::required_shape_bytes(
        output,
        plan.count,
        plan.rank,
        nullable || plan.len == 0,
    )?;
    Ok((plan.count, bytes, scratch))
}

fn output_dtype(input: ArrayDType, operation: Reduction) -> ArrayDType {
    match operation {
        Reduction::Sum | Reduction::Mean => ArrayDType::Float64,
        Reduction::Min | Reduction::Max => input,
    }
}

fn descriptor_scratch(plan: Lanes, operation: Reduction) -> usize {
    if plan.count == 0 || plan.len == 0 {
        return 0;
    }
    match operation {
        Reduction::Sum | Reduction::Mean => Exact::SCRATCH,
        Reduction::Min | Reduction::Max => Extremum::SCRATCH,
    }
}

fn lane_mode(operation: Reduction) -> bool {
    matches!(operation, Reduction::Mean | Reduction::Max)
}

#[derive(Clone, Copy)]
struct Lanes {
    shape: [usize; 2],
    rank: usize,
    count: usize,
    len: usize,
    stride: usize,
    base_stride: usize,
}

impl Lanes {
    fn new(input: ArrayView<'_>, options: ReductionOptions) -> Result<Self> {
        Self::from_shape(input.shape(), options)
    }
    fn from_shape(input: &[usize], options: ReductionOptions) -> Result<Self> {
        let elements = crate::array::shape_elements(input)?;
        if options.axis().is_some_and(|axis| axis >= input.len()) {
            return Err(invalid(
                "options.axis",
                "axis must be smaller than input rank",
            ));
        }
        let (shape, rank) = output_shape(input, options);
        let count = crate::array::shape_elements(&shape[..rank])?;
        let (len, stride, base_stride) = lane_layout(input, elements, options.axis());
        Ok(Self {
            shape,
            rank,
            count,
            len,
            stride,
            base_stride,
        })
    }
    fn covers_input(self, input: ArrayView<'_>) -> bool {
        self.count == 1 && self.len == input.len()
    }
    fn all_invalid(self, input: ArrayView<'_>, skip_nulls: bool) -> bool {
        if self.len == 0 || input.null_count() == input.len() {
            return true;
        }
        self.covers_input(input) && !skip_nulls && input.has_nulls()
    }
}

fn output_shape(input: &[usize], options: ReductionOptions) -> ([usize; 2], usize) {
    let Some(axis) = options.axis() else {
        return ([1; 2], if options.keepdims() { input.len() } else { 0 });
    };
    let mut shape = [0; 2];
    let mut rank = 0;
    for (index, extent) in input.iter().enumerate() {
        if index != axis {
            shape[rank] = *extent;
            rank += 1;
        } else if options.keepdims() {
            shape[rank] = 1;
            rank += 1;
        }
    }
    (shape, rank)
}

fn lane_layout(input: &[usize], elements: usize, axis: Option<usize>) -> (usize, usize, usize) {
    match axis {
        Some(0) if input.len() == 2 => (input[0], input[1], 1),
        Some(1) => (input[1], 1, input[1]),
        _ => (elements, 1, 0),
    }
}

trait State {
    const SCRATCH: usize;
    fn new(mode: bool) -> Self;
    fn push(&mut self, value: f64) -> Result<()>;
    fn result(self) -> Option<f64>;
}

struct Exact {
    total: ExactSum,
    average: bool,
}
impl State for Exact {
    const SCRATCH: usize = ExactSum::scratch_bytes();
    fn new(average: bool) -> Self {
        Self {
            total: ExactSum::new(),
            average,
        }
    }
    fn push(&mut self, value: f64) -> Result<()> {
        self.total
            .push(value)
            .map_err(|_| invalid("input.elements", "valid element count overflowed usize"))
    }
    fn result(self) -> Option<f64> {
        if self.average {
            self.total.mean()
        } else {
            self.total.sum()
        }
    }
}

struct Extremum {
    selected: Option<f64>,
    maximum: bool,
}
impl State for Extremum {
    const SCRATCH: usize = 0;
    fn new(maximum: bool) -> Self {
        Self {
            selected: None,
            maximum,
        }
    }
    fn push(&mut self, value: f64) -> Result<()> {
        if self
            .selected
            .is_none_or(|current| better_extremum(value, current, self.maximum))
        {
            self.selected = Some(value);
        }
        Ok(())
    }
    fn result(self) -> Option<f64> {
        self.selected.map(|value| {
            if value.is_nan() {
                f64::from_bits(0x7ff8_0000_0000_0000)
            } else {
                value
            }
        })
    }
}

trait Output: Copy + Default {
    fn number(value: f64) -> Self;
    fn wrap(
        values: Vec<Self>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<NativeArray>;
}
impl Output for f64 {
    fn number(value: f64) -> Self {
        value
    }
    fn wrap(
        values: Vec<Self>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<NativeArray> {
        NativeArray::from_f64(values, shape, validity)
    }
}
impl Output for f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Extrema select an original float32 value or a canonical NaN."
    )]
    fn number(value: f64) -> Self {
        if value.is_nan() {
            Self::from_bits(0x7fc0_0000)
        } else {
            value as Self
        }
    }
    fn wrap(
        values: Vec<Self>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<NativeArray> {
        NativeArray::from_f32(values, shape, validity)
    }
}

fn extremes(
    input: ArrayView<'_>,
    plan: Lanes,
    options: ReductionOptions,
    operation: Reduction,
    execution: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    match input.dtype() {
        ArrayDType::Float32 => {
            execute::<Extremum, f32>(input, plan, options, operation, execution, check)
        }
        ArrayDType::Float64 => {
            execute::<Extremum, f64>(input, plan, options, operation, execution, check)
        }
        ArrayDType::Boolean => unreachable!("Boolean reductions were rejected"),
    }
}

fn execute<S: State, O: Output>(
    input: ArrayView<'_>,
    plan: Lanes,
    options: ReductionOptions,
    operation: Reduction,
    execution: &ArrayExecutionOptions,
    check: Check<'_>,
) -> Result<NativeArray> {
    let invalid = plan.all_invalid(input, options.skip_nulls());
    let nullable = nullable_output(input, plan, invalid);
    let (count, bytes, predicted_scratch) =
        super::reduction_requirements(input.dtype(), input.shape(), nullable, operation, options)?;
    let scratch = scratch_bytes(plan, invalid, predicted_scratch);
    let mode = lane_mode(operation);
    execution.validate_allocation(count, bytes, scratch)?;
    let mut output = Vec::with_capacity(count);
    let mut validity = Validity::new(count, nullable);
    let mut work = 0;
    for lane in 0..count {
        control::checkpoint(lane, check)?;
        let value = if invalid {
            None
        } else {
            collect::<S>(input, plan, lane, options, mode, &mut work, check)?
        };
        output.push(value.map_or_else(O::default, O::number));
        validity.record(lane, value.is_some());
    }
    control::finish(
        O::wrap(output, plan.shape[..plan.rank].to_vec(), validity.finish()),
        check,
    )
}

fn nullable_output(input: ArrayView<'_>, plan: Lanes, invalid: bool) -> bool {
    invalid || (!plan.covers_input(input) && input.has_nulls())
}

fn scratch_bytes(plan: Lanes, invalid: bool, predicted: usize) -> usize {
    if invalid || plan.count == 0 {
        0
    } else {
        predicted
    }
}

fn collect<S: State>(
    input: ArrayView<'_>,
    plan: Lanes,
    lane: usize,
    options: ReductionOptions,
    mode: bool,
    work: &mut usize,
    check: Check<'_>,
) -> Result<Option<f64>> {
    let mut state = S::new(mode);
    let mut has_null = false;
    let base = lane * plan.base_stride;
    for position in 0..plan.len {
        control::step(work, check)?;
        if let Some(value) = input.number64(base + position * plan.stride) {
            state.push(value)?;
        } else {
            has_null = true;
        }
    }
    if has_null && !options.skip_nulls() {
        Ok(None)
    } else {
        Ok(state.result())
    }
}

struct Validity {
    bits: Option<Vec<u8>>,
    len: usize,
    all_valid: bool,
}
impl Validity {
    fn new(len: usize, nullable: bool) -> Self {
        Self {
            bits: nullable.then(|| vec![0; len.div_ceil(8)]),
            len,
            all_valid: true,
        }
    }
    fn record(&mut self, index: usize, valid: bool) {
        self.all_valid &= valid;
        if valid && let Some(bits) = &mut self.bits {
            bits[index / 8] |= 1 << (index % 8);
        }
    }
    fn finish(self) -> Option<NullBuffer> {
        if self.all_valid {
            None
        } else {
            self.bits.map(|bits| {
                NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(bits), 0, self.len))
            })
        }
    }
}
