use super::{allocate, charges, indexed, positions, validate};
use crate::{
    Result,
    array::{
        ArrayExecutionOptions, ArrayStateOptions, ArrayView, NativeArray, RsiOptions, control,
        indicators::RsiKernel, sequence::Prepared,
    },
};

/// Computes aligned RSI with an incremental seed and Wilder recurrence.
///
/// Missing values reset warm-up. The first complete price establishes the
/// previous price; the first result requires `period` subsequent differences.
/// Flat segments produce 50. Inputs remain read-only and output is float64.
///
/// ```
/// use calc_flow::{Result, array::{ArrayExecutionOptions, ArrayView, FloatView, RsiOptions, rsi}};
/// # fn main() -> Result<()> {
/// let values = [1.0, 2.0, 1.0];
/// let result = rsi(ArrayView::from_float(FloatView::from_f64(&values)),
///     RsiOptions::new(2)?, &ArrayExecutionOptions::default())?;
/// assert_eq!(result.shape(), &[3]);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
/// Returns `InvalidArgument` for non-vector/non-floating inputs, infinity,
/// unrepresentable finite arithmetic or exhausted execution limits.
pub fn rsi(
    input: ArrayView<'_>,
    options: RsiOptions,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    rsi_checked(input, options, execution, &mut || Ok(()))
}

pub(super) fn rsi_checked(
    input: ArrayView<'_>,
    options: RsiOptions,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    let state = RsiState {
        options,
        limits: ArrayStateOptions::default(),
        kernel: RsiKernel::default(),
    };
    Ok(state.prepare(input, execution, 0, true, check)?.output)
}

/// An owned constant-space RSI state with atomic chunk updates.
///
/// The state has no public cloning or serialization API. Invalid calls leave
/// committed recurrence, counters and charge unchanged.
pub struct RsiState {
    pub(in crate::array) options: RsiOptions,
    pub(in crate::array) limits: ArrayStateOptions,
    pub(in crate::array) kernel: RsiKernel,
}

impl RsiState {
    /// Creates an empty state with a fixed 48-byte numerical charge.
    ///
    /// # Errors
    /// Returns `InvalidArgument` if the state limit is below 48 bytes.
    pub fn new(options: RsiOptions, state: ArrayStateOptions) -> Result<Self> {
        state.validate(48)?;
        Ok(Self {
            options,
            limits: state,
            kernel: RsiKernel::default(),
        })
    }

    /// Processes one ordered floating vector, committing the whole chunk.
    ///
    /// # Errors
    /// Returns `InvalidArgument` for descriptors, nonfinite samples/arithmetic,
    /// exhausted position counters or participating state/output limits.
    pub fn process(
        &mut self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
    ) -> Result<NativeArray> {
        let prepared = self.prepare_checked(input, execution, 0, &mut || Ok(()))?;
        Ok(prepared.commit(self))
    }

    pub(in crate::array) fn prepare_checked(
        &self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Prepared<Self>> {
        self.prepare(input, execution, captures, false, check)
    }

    fn prepare(
        &self,
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        eager: bool,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Prepared<Self>> {
        let mut output = self.allocate(input, execution, captures, eager, check)?;
        if input.is_empty() {
            return control::finish(
                Ok(Prepared {
                    candidate: None,
                    output: output.finish(true)?,
                }),
                check,
            );
        }
        let kernel = self.evaluate(input, &mut output, check)?;
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
        input: ArrayView<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        eager: bool,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<super::super::Output> {
        check()?;
        self.kernel.validate(self.options.period())?;
        validate(input, "input")?;
        positions(self.kernel.positions, input.len())?;
        allocate(
            input.len(),
            charges(input.len(), eager, 48, captures),
            execution,
            check,
        )
    }

    fn evaluate(
        &self,
        input: ArrayView<'_>,
        output: &mut super::super::Output,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<RsiKernel> {
        let mut kernel = self.kernel;
        for index in 0..input.len() {
            control::checkpoint(index, check)?;
            let value = kernel
                .push(input.number64(index), self.options.period(), "input")
                .map_err(|error| indexed(error, index))?;
            output.push(value);
        }
        Ok(kernel)
    }

    /// Clears counters and recurrence, retaining options and the fixed charge.
    pub fn reset(&mut self) {
        self.kernel = RsiKernel::default();
    }

    /// Returns the fixed 48-byte numerical charge, including empty state.
    pub const fn state_bytes(&self) -> usize {
        48
    }
}

impl std::fmt::Debug for RsiState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RsiState")
            .field("options", &self.options)
            .field("limits", &self.limits)
            .field("state_bytes", &48)
            .finish_non_exhaustive()
    }
}
