use super::{allocate, charges, indexed, positions, validate};
use crate::array::ArrayDType;
use crate::{
    Result,
    array::{
        ArrayExecutionOptions, ArrayStateOptions, ArrayView, AtrOptions, NativeArray, control,
        indicators::AtrKernel, invalid, sequence::Prepared,
    },
};

/// Computes aligned average true range with a Wilder recurrence.
///
/// Each operand must be a rank-one floating vector of equal length. Missing
/// rows reset warm-up; valid infinity and invalid high/low ranges are errors.
///
/// # Errors
/// Returns `InvalidArgument` for invalid descriptors, numerical domain errors,
/// nonfinite intermediate arithmetic or exhausted execution limits.
pub fn atr(
    high: ArrayView<'_>,
    low: ArrayView<'_>,
    close: ArrayView<'_>,
    options: AtrOptions,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    atr_checked([high, low, close], options, execution, &mut || Ok(()))
}

pub(super) fn atr_checked(
    inputs: [ArrayView<'_>; 3],
    options: AtrOptions,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    let state = AtrState {
        options,
        limits: ArrayStateOptions::default(),
        kernel: AtrKernel::default(),
    };
    Ok(state
        .prepare(Inputs::Separate(inputs), execution, 0, true, check)?
        .output)
}

/// An owned constant-space ATR state with atomic chunk updates.
pub struct AtrState {
    pub(in crate::array) options: AtrOptions,
    pub(in crate::array) limits: ArrayStateOptions,
    pub(in crate::array) kernel: AtrKernel,
}

impl AtrState {
    /// Creates an empty state with a fixed 40-byte numerical charge.
    ///
    /// # Errors
    /// Returns `InvalidArgument` if the state limit is below 40 bytes.
    pub fn new(options: AtrOptions, state: ArrayStateOptions) -> Result<Self> {
        state.validate(40)?;
        Ok(Self {
            options,
            limits: state,
            kernel: AtrKernel::default(),
        })
    }

    /// Processes synchronized high/low/close vectors and commits atomically.
    ///
    /// # Errors
    /// Returns `InvalidArgument` for invalid descriptors, infinity, domain or
    /// arithmetic failures, exhausted counters or participating limits.
    pub fn process(
        &mut self,
        high: ArrayView<'_>,
        low: ArrayView<'_>,
        close: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
    ) -> Result<NativeArray> {
        let prepared = self.prepare_checked([high, low, close], execution, 0, &mut || Ok(()))?;
        Ok(prepared.commit(self))
    }

    pub(in crate::array) fn prepare_checked(
        &self,
        inputs: [ArrayView<'_>; 3],
        execution: &ArrayExecutionOptions,
        captures: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Prepared<Self>> {
        self.prepare(Inputs::Separate(inputs), execution, captures, false, check)
    }

    pub(in crate::array) fn prepare_packed_checked(
        &self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Prepared<Self>> {
        self.prepare(Inputs::Packed(input), execution, captures, false, check)
    }

    fn prepare(
        &self,
        inputs: Inputs<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        eager: bool,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Prepared<Self>> {
        let len = inputs.len();
        let mut output = self.allocate(inputs, execution, captures, eager, check)?;
        if len == 0 {
            return control::finish(
                Ok(Prepared {
                    candidate: None,
                    output: output.finish(true)?,
                }),
                check,
            );
        }
        let kernel = self.evaluate(inputs, &mut output, check)?;
        let candidate = Some(Self {
            options: self.options,
            limits: self.limits,
            kernel,
        });
        control::finish(
            Ok(Prepared {
                candidate,
                output: output.finish(true)?,
            }),
            check,
        )
    }

    fn allocate(
        &self,
        inputs: Inputs<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        eager: bool,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<super::super::Output> {
        check()?;
        self.kernel.validate(self.options.period())?;
        inputs.validate()?;
        let len = inputs.len();
        positions(self.kernel.positions, len)?;
        allocate(len, charges(len, eager, 40, captures), execution, check)
    }

    fn evaluate(
        &self,
        inputs: Inputs<'_>,
        output: &mut super::super::Output,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<AtrKernel> {
        let len = inputs.len();
        let mut kernel = self.kernel;
        for index in 0..len {
            control::checkpoint(index, check)?;
            let samples = inputs.samples(index);
            let value = kernel
                .push(
                    samples,
                    self.options.period(),
                    ["inputs.high", "inputs.low", "inputs.close"],
                )
                .map_err(|error| indexed(error, index))?;
            output.push(value);
        }
        Ok(kernel)
    }

    /// Clears counters and recurrence while retaining options and fixed charge.
    pub fn reset(&mut self) {
        self.kernel = AtrKernel::default();
    }

    /// Returns the fixed 40-byte numerical charge, including empty state.
    pub const fn state_bytes(&self) -> usize {
        40
    }
}

fn validate_inputs(inputs: [ArrayView<'_>; 3]) -> Result<()> {
    for (input, name) in inputs
        .iter()
        .zip(["inputs.high", "inputs.low", "inputs.close"])
    {
        validate(*input, name)?;
    }
    let len = inputs[0].len();
    for (input, name) in inputs[1..].iter().zip(["inputs.low", "inputs.close"]) {
        if input.len() != len {
            return Err(invalid(
                &format!("{name}.shape"),
                format!(
                    "expected length {len} to match inputs.high, received {}",
                    input.len()
                ),
            ));
        }
    }
    Ok(())
}

impl std::fmt::Debug for AtrState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AtrState")
            .field("options", &self.options)
            .field("limits", &self.limits)
            .field("state_bytes", &40)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
enum Inputs<'a> {
    Separate([ArrayView<'a>; 3]),
    Packed(ArrayView<'a>),
}

impl Inputs<'_> {
    fn len(self) -> usize {
        match self {
            Self::Separate(inputs) => inputs[0].len(),
            Self::Packed(input) => match input.shape() {
                [rows, 3] => *rows,
                _ => 0,
            },
        }
    }

    fn validate(self) -> Result<()> {
        match self {
            Self::Separate(inputs) => validate_inputs(inputs),
            Self::Packed(input) => validate_packed(input),
        }
    }

    fn samples(self, index: usize) -> [Option<f64>; 3] {
        match self {
            Self::Separate(inputs) => inputs.map(|input| input.number64(index)),
            Self::Packed(input) => {
                let offset = index * 3;
                [
                    input.number64(offset),
                    input.number64(offset + 1),
                    input.number64(offset + 2),
                ]
            }
        }
    }
}

fn validate_packed(input: ArrayView<'_>) -> Result<()> {
    if !matches!(input.shape(), [_, 3]) {
        return Err(invalid(
            "input.shape",
            "ATR sequence input requires [n, 3] in high, low, close order",
        ));
    }
    if input.dtype() == ArrayDType::Boolean {
        return Err(invalid(
            "input.dtype",
            "expected float32 or float64, received bool",
        ));
    }
    Ok(())
}
