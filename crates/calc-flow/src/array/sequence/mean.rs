use super::{ArrayStateOptions, Budget, Output, Prepared, history_charge, sample, validate_input};
use crate::{
    Result, RollingNumericalProfile,
    array::{
        ArrayExecutionOptions, ArrayView, NativeArray, RollingMeanOptions, control,
        rolling_mean::MeanKernel, rolling_mean_math,
    },
};

/// An owned row-window mean that advances transactionally across vector chunks.
///
/// ```
/// use calc_flow::{Result, array::{ArrayExecutionOptions, ArrayStateOptions,
///     ArrayView, FloatView, RollingMeanOptions, RollingMeanState}};
/// # fn main() -> Result<()> {
/// let mut state = RollingMeanState::new(RollingMeanOptions::new(2)?, ArrayStateOptions::default())?;
/// let execution = ArrayExecutionOptions::default();
/// state.process(ArrayView::from_float(FloatView::from_f64(&[1.0, 3.0])), &execution)?;
/// let output = state.process(ArrayView::from_float(FloatView::from_f32(&[5.0])), &execution)?;
/// assert_eq!(output.shape(), &[1]);
/// state.reset();
/// assert_eq!(state.state_bytes(), 64);
/// # Ok(())
/// # }
/// ```
pub struct RollingMeanState {
    pub(super) options: RollingMeanOptions,
    pub(super) limits: ArrayStateOptions,
    pub(super) kernel: MeanKernel,
    pub(super) bytes: usize,
}

impl RollingMeanState {
    /// Creates an empty state without preallocating the configured window.
    ///
    /// # Errors
    /// Returns an error when the fixed 64-byte charge exceeds the retained limit.
    pub fn new(options: RollingMeanOptions, state: ArrayStateOptions) -> Result<Self> {
        state.validate(64)?;
        Ok(Self {
            options,
            limits: state,
            kernel: MeanKernel::new(0),
            bytes: 64,
        })
    }

    /// Processes one ordered floating vector, committing only after success.
    ///
    /// The allocation limit includes old state, isolated candidate, output and
    /// numerical scratch. Nulls and NaNs occupy window positions; only valid
    /// samples count toward warm-up. Results always have float64 values.
    ///
    /// # Errors
    /// Returns an error for invalid rank/dtype, exhausted limits, allocation
    /// failure, or position/arithmetic overflow. Errors leave this state intact.
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
        self.validate(input, check)?;
        let mut budget = self.budget(input.len(), execution, captures)?;
        if input.is_empty() {
            let output = Output::allocate(&budget, check)?.finish(false)?;
            return control::finish(
                Ok(Prepared {
                    candidate: None,
                    output,
                }),
                check,
            );
        }
        let (kernel, output) = self.prepare(input, &mut budget, check)?;
        let candidate = Self {
            options: self.options,
            limits: self.limits,
            kernel,
            bytes: budget.candidate,
        };
        control::finish(
            Ok(Prepared {
                candidate: Some(candidate),
                output,
            }),
            check,
        )
    }

    fn budget<'a>(
        &self,
        len: usize,
        execution: &'a ArrayExecutionOptions,
        captures: usize,
    ) -> Result<Budget<'a>> {
        let candidate = self.candidate_charge(len)?;
        self.limits.validate(candidate)?;
        let scratch = self.scratch_bytes(len);
        let budget = Budget {
            peak_field: "state.peak_bytes",
            old: self.bytes,
            candidate,
            captures,
            scratch,
            len,
            execution,
        };
        budget.admit(Output::required_bytes(len)?)?;
        Ok(budget)
    }

    fn validate(&self, input: ArrayView<'_>, check: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        check()?;
        validate_input(input)?;
        self.kernel.validate_positions(input.len())
    }

    fn candidate_charge(&self, len: usize) -> Result<usize> {
        if len == 0 {
            Ok(0)
        } else {
            history_charge(self.kernel.next_capacity(len, self.options.window()))
        }
    }

    fn scratch_bytes(&self, len: usize) -> usize {
        if len > 0 && self.options.numerical_profile() == RollingNumericalProfile::StableV2Preview {
            rolling_mean_math::stable_scratch_bytes()
        } else {
            0
        }
    }

    fn prepare(
        &self,
        input: ArrayView<'_>,
        budget: &mut Budget<'_>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<(MeanKernel, NativeArray)> {
        let mut next = self.candidate(input.len(), budget)?;
        next.copy_from(&self.kernel, check)?;
        let result = self.evaluate(&mut next, input, budget, check)?;
        Ok((next, result))
    }

    fn candidate(&self, len: usize, budget: &mut Budget<'_>) -> Result<MeanKernel> {
        let next =
            MeanKernel::try_candidate(self.kernel.next_capacity(len, self.options.window()))?;
        budget.candidate = history_charge(next.capacity())?;
        self.limits.validate(budget.candidate)?;
        budget.admit(Output::required_bytes(len)?)?;
        Ok(next)
    }

    fn evaluate(
        &self,
        next: &mut MeanKernel,
        input: ArrayView<'_>,
        budget: &Budget<'_>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<NativeArray> {
        let mut output = Output::allocate(budget, check)?;
        let window = usize::try_from(self.options.window()).unwrap_or(usize::MAX);
        for index in 0..input.len() {
            control::checkpoint(index, check)?;
            output.push(next.push(sample(input, index), self.options, window, check)?);
        }
        output.finish(false)
    }

    /// Releases history and restores the empty state while retaining its limits.
    pub fn reset(&mut self) {
        self.kernel = MeanKernel::new(0);
        self.bytes = 64;
    }

    /// Returns the logical retained charge, including allocated spare slots.
    pub const fn state_bytes(&self) -> usize {
        self.bytes
    }
}

impl std::fmt::Debug for RollingMeanState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RollingMeanState")
            .field("options", &self.options)
            .field("limits", &self.limits)
            .field("state_bytes", &self.state_bytes())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
