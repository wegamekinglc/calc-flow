use super::{ArrayStateOptions, Budget, Output, Prepared, sample, validate_input};
use crate::{
    Result,
    array::{
        ArrayExecutionOptions, ArrayView, EwmaOptions, NativeArray, control, rolling::EwmaKernel,
    },
};

/// An owned exponential mean that advances transactionally across vector chunks.
///
/// ```
/// use calc_flow::{Result, array::{ArrayExecutionOptions, ArrayStateOptions,
///     ArrayView, EwmaOptions, EwmaState, FloatView}};
/// # fn main() -> Result<()> {
/// let mut state = EwmaState::new(EwmaOptions::new(3)?, ArrayStateOptions::default())?;
/// let execution = ArrayExecutionOptions::default();
/// state.process(ArrayView::from_float(FloatView::from_f64(&[10.0])), &execution)?;
/// let output = state.process(ArrayView::from_float(FloatView::from_f64(&[14.0])), &execution)?;
/// assert_eq!(output.len(), 1);
/// assert_eq!(state.state_bytes(), 24);
/// # Ok(())
/// # }
/// ```
pub struct EwmaState {
    pub(super) options: EwmaOptions,
    pub(super) limits: ArrayStateOptions,
    pub(super) kernel: EwmaKernel,
    pub(super) positions: u64,
}

impl EwmaState {
    /// Creates an empty constant-size state.
    ///
    /// # Errors
    /// Returns an error when the fixed 24-byte charge exceeds the retained limit.
    pub fn new(options: EwmaOptions, state: ArrayStateOptions) -> Result<Self> {
        state.validate(24)?;
        Ok(Self {
            options,
            limits: state,
            kernel: EwmaKernel::default(),
            positions: 0,
        })
    }

    /// Processes one ordered floating vector, committing only after success.
    ///
    /// Null and NaN samples leave the recurrence unchanged and carry its output
    /// after warm-up. Infinity and arithmetic-generated NaN remain valid values.
    /// The execution byte limit includes old state, candidate and output storage.
    ///
    /// # Errors
    /// Returns an error for invalid rank/dtype, exhausted limits, allocation
    /// failure, or count overflow. Errors leave the recurrence intact.
    pub fn process(
        &mut self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
    ) -> Result<NativeArray> {
        self.process_checked(input, execution, &mut || Ok(()))
    }

    pub(in crate::array) fn process_checked(
        &mut self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<NativeArray> {
        let prepared = self.prepare_checked(input, execution, 0, check)?;
        Ok(prepared.commit(self))
    }

    pub(in crate::array) fn prepare_checked(
        &self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Prepared<Self>> {
        let positions = self.validate(input, check)?;
        let budget = self.budget(input.len(), execution, captures);
        let mut output = Output::allocate(&budget, check)?;
        if input.is_empty() {
            let output = output.finish(true)?;
            return control::finish(
                Ok(Prepared {
                    candidate: None,
                    output,
                }),
                check,
            );
        }
        let mut kernel = self.kernel;
        self.evaluate(&mut kernel, input, &mut output, check)?;
        let output = output.finish(true)?;
        let candidate = Self {
            options: self.options,
            limits: self.limits,
            kernel,
            positions,
        };
        control::finish(
            Ok(Prepared {
                candidate: Some(candidate),
                output,
            }),
            check,
        )
    }

    fn validate(&self, input: ArrayView<'_>, check: &mut dyn FnMut() -> Result<()>) -> Result<u64> {
        check()?;
        validate_input(input)?;
        super::next_positions(self.positions, input.len())
    }

    fn budget<'a>(
        &self,
        len: usize,
        execution: &'a ArrayExecutionOptions,
        captures: usize,
    ) -> Budget<'a> {
        Budget {
            old: self.state_bytes(),
            candidate: if len == 0 { 0 } else { 24 },
            captures,
            scratch: 0,
            len,
            execution,
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "preserves table span conversion"
    )]
    fn evaluate(
        &self,
        next: &mut EwmaKernel,
        input: ArrayView<'_>,
        output: &mut Output,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let alpha = 2.0 / (self.options.span() as f64 + 1.0);
        for index in 0..input.len() {
            control::checkpoint(index, check)?;
            output.push(next.push(sample(input, index), alpha, self.options.min_periods())?);
        }
        Ok(())
    }

    /// Restores the empty recurrence while retaining options and limits.
    pub fn reset(&mut self) {
        self.kernel = EwmaKernel::default();
        self.positions = 0;
    }

    /// Returns the constant 24-byte retained numerical charge.
    pub const fn state_bytes(&self) -> usize {
        24
    }
}

impl std::fmt::Debug for EwmaState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EwmaState")
            .field("options", &self.options)
            .field("limits", &self.limits)
            .field("state_bytes", &self.state_bytes())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
