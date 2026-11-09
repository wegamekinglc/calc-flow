use super::indicator::{positions, validate};
use crate::{
    Result,
    array::{
        ArrayExecutionOptions, ArrayStateOptions, ArrayView, MacdOptions, control,
        indicators::{MacdKernel, MacdParameters},
    },
};
use output::{Buffers, Plan};

mod output;
mod packed;
pub use output::MacdOutputs;

/// Computes the unadjusted MACD line, signal and histogram as aligned arrays.
///
/// Null/NaN prices carry ready fast/slow values. Their carried line still
/// updates signal; only a null/NaN line skips its update. Infinity and generated
/// NaN remain valid results, matching the existing EWMA/subtraction semantics.
///
/// ```
/// use calc_flow::{Result, array::{ArrayExecutionOptions, ArrayView, FloatView,
///     MacdOptions, macd_components}};
/// # fn main() -> Result<()> {
/// let input = [10.0, 14.0, 18.0];
/// let output = macd_components(ArrayView::from_float(FloatView::from_f64(&input)),
///     MacdOptions::new(1, 3, 3)?, &ArrayExecutionOptions::default())?;
/// assert_eq!(output.line().shape(), &[3]);
/// assert_eq!(output.signal().shape(), output.histogram().shape());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
/// Returns `InvalidArgument` for unsupported rank/dtype or exhausted element,
/// byte or checked position limits. No input buffer is modified.
pub fn macd_components(
    input: ArrayView<'_>,
    options: MacdOptions,
    execution: &ArrayExecutionOptions,
) -> Result<MacdOutputs> {
    let state = MacdState {
        options,
        limits: ArrayStateOptions::default(),
        kernel: MacdKernel::default(),
    };
    Ok(state
        .prepare(input, execution, 0, true, &mut || Ok(()))?
        .output)
}

/// One owned 56-byte MACD recurrence with atomic updates across ordered chunks.
///
/// There is no public cloning or serialization API. Consuming one output still
/// advances all three recurrences in the same numerical candidate.
pub struct MacdState {
    pub(in crate::array) options: MacdOptions,
    pub(in crate::array) limits: ArrayStateOptions,
    pub(in crate::array) kernel: MacdKernel,
}
impl MacdState {
    /// Creates an empty state with the fixed 56-byte numerical charge.
    ///
    /// # Errors
    /// Returns `InvalidArgument` if the retained-state limit is below 56 bytes.
    pub fn new(options: MacdOptions, state: ArrayStateOptions) -> Result<Self> {
        state.validate(56)?;
        Ok(Self {
            options,
            limits: state,
            kernel: MacdKernel::default(),
        })
    }
    /// Processes one floating vector, committing all three recurrences at once.
    ///
    /// The byte limit counts committed state, candidate, all three independent
    /// output buffers/bitmaps/shapes and retained captures. The element limit is
    /// `3 * input.len()`. Empty calls retain counters and recurrence bits.
    ///
    /// # Errors
    /// Returns `InvalidArgument` for descriptors, count overflow or limits;
    /// failures leave committed state and its charge unchanged.
    pub fn process(
        &mut self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
    ) -> Result<MacdOutputs> {
        let prepared = self.prepare_checked(input, execution, 0, &mut || Ok(()))?;
        if let Some(candidate) = prepared.candidate {
            *self = candidate;
        }
        Ok(prepared.output)
    }
    pub(in crate::array) fn prepare_checked(
        &self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Prepared> {
        self.prepare(input, execution, captures, false, check)
    }
    fn prepare(
        &self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        eager: bool,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Prepared> {
        let mut buffers = self.allocate(input, execution, captures, eager, check)?;
        if input.is_empty() {
            return control::finish(
                Ok(Prepared {
                    candidate: None,
                    output: buffers.finish()?,
                }),
                check,
            );
        }
        let kernel = self.evaluate(input, |values| buffers.push(values), check)?;
        let candidate = Some(Self {
            options: self.options,
            limits: self.limits,
            kernel,
        });
        control::finish(
            Ok(Prepared {
                candidate,
                output: buffers.finish()?,
            }),
            check,
        )
    }
    fn allocate(
        &self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        eager: bool,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Buffers> {
        self.validate(input, check)?;
        let plan = Plan::new(
            input.len(),
            if eager { 0 } else { 56 },
            captures,
            eager,
            execution,
        )?;
        Buffers::allocate(&plan, check)
    }
    fn validate(&self, input: ArrayView<'_>, check: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        check()?;
        validate(input, "input")?;
        self.kernel.validate(self.options)?;
        positions(self.kernel.positions, input.len())
    }

    pub(in crate::array) fn prepare_packed_checked(
        &self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<super::Prepared<Self>> {
        self.validate(input, check)?;
        let mut output = packed::Packed::allocate(self, input.len(), captures, execution, check)?;
        if input.is_empty() {
            return control::finish(
                Ok(super::Prepared {
                    candidate: None,
                    output: output.finish()?,
                }),
                check,
            );
        }
        let kernel = self.evaluate(input, |values| output.push(values), check)?;
        let candidate = Some(Self {
            options: self.options,
            limits: self.limits,
            kernel,
        });
        control::finish(
            Ok(super::Prepared {
                candidate,
                output: output.finish()?,
            }),
            check,
        )
    }

    fn evaluate(
        &self,
        input: ArrayView<'_>,
        mut push: impl FnMut([Option<f64>; 3]),
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<MacdKernel> {
        let mut kernel = self.kernel;
        let parameters = MacdParameters::new(self.options);
        for index in 0..input.len() {
            control::checkpoint(index, check)?;
            push(kernel.push(input.number64(index), &parameters)?);
        }
        Ok(kernel)
    }
    /// Clears counts/values while retaining options and fixed storage charge.
    pub fn reset(&mut self) {
        self.kernel = MacdKernel::default();
    }
    /// Returns the fixed 56-byte logical numerical charge, including empty state.
    pub const fn state_bytes(&self) -> usize {
        56
    }
}
pub(in crate::array) struct Prepared {
    pub(in crate::array) candidate: Option<MacdState>,
    pub(in crate::array) output: MacdOutputs,
}
impl std::fmt::Debug for MacdState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MacdState")
            .field("options", &self.options)
            .field("limits", &self.limits)
            .field("state_bytes", &56)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod packed_tests;

#[cfg(test)]
mod codec_tests;
