use super::super::{Budget, Output};
use crate::{
    Result,
    array::{ArrayExecutionOptions, NativeArray, invalid},
};

/// Immutable, aligned float64 owners for the MACD line, signal and histogram.
#[derive(Debug)]
pub struct MacdOutputs {
    line: NativeArray,
    signal: NativeArray,
    histogram: NativeArray,
}
impl MacdOutputs {
    /// Returns the fast-minus-slow line.
    pub const fn line(&self) -> &NativeArray {
        &self.line
    }
    /// Returns the EWMA of valid/carried line values.
    pub const fn signal(&self) -> &NativeArray {
        &self.signal
    }
    /// Returns line minus signal, retaining operand validity.
    pub const fn histogram(&self) -> &NativeArray {
        &self.histogram
    }
    /// Moves all three immutable array owners without copying their buffers.
    pub fn into_arrays(self) -> (NativeArray, NativeArray, NativeArray) {
        (self.line, self.signal, self.histogram)
    }
}

pub(super) struct Buffers([Output; 3]);
impl Buffers {
    pub(super) fn allocate(plan: &Plan<'_>, check: &mut dyn FnMut() -> Result<()>) -> Result<Self> {
        let line = plan.component(&[], 2, check)?;
        let signal = plan.component(&[&line], 1, check)?;
        let histogram = plan.component(&[&line, &signal], 0, check)?;
        Ok(Self([line, signal, histogram]))
    }
    pub(super) fn push(&mut self, values: [Option<f64>; 3]) {
        for (output, value) in self.0.iter_mut().zip(values) {
            output.push(value);
        }
    }
    pub(super) fn finish(self) -> Result<MacdOutputs> {
        let [line, signal, histogram] = self.0;
        Ok(MacdOutputs {
            line: line.finish(true)?,
            signal: signal.finish(true)?,
            histogram: histogram.finish(true)?,
        })
    }
}
fn bytes(output: &Output) -> Result<usize> {
    Output::allocated_bytes(output.values.capacity(), output.bitmap.capacity())
}
fn total(parts: &[usize]) -> Result<usize> {
    parts
        .iter()
        .try_fold(0_usize, |sum, &bytes| sum.checked_add(bytes))
        .ok_or_else(|| {
            invalid(
                "execution.max_allocation_bytes",
                "MACD live byte count overflowed usize",
            )
        })
}

pub(super) struct Plan<'a> {
    len: usize,
    state: [usize; 3],
    execution: &'a ArrayExecutionOptions,
}
impl<'a> Plan<'a> {
    pub(super) fn new(
        len: usize,
        old: usize,
        captures: usize,
        eager: bool,
        execution: &'a ArrayExecutionOptions,
    ) -> Result<Self> {
        elements(len, execution)?;
        let candidate = if eager || len != 0 { 56 } else { 0 };
        Ok(Self {
            len,
            state: [old, candidate, captures],
            execution,
        })
    }
    fn component(
        &self,
        previous: &[&Output],
        remaining: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Output> {
        let predicted = Output::required_bytes(self.len)?
            .checked_mul(remaining)
            .ok_or_else(|| {
                invalid(
                    "execution.max_allocation_bytes",
                    "MACD output byte count overflowed usize",
                )
            })?;
        let retained = previous.iter().try_fold(0_usize, |retained, output| {
            total(&[retained, bytes(output)?])
        })?;
        let other = total(&[predicted, retained])?;
        Output::allocate(
            &Budget {
                peak_field: "execution.max_allocation_bytes",
                old: self.state[0],
                candidate: self.state[1],
                captures: self.state[2],
                scratch: other,
                len: self.len,
                execution: self.execution,
            },
            check,
        )
    }
}
pub(super) fn elements(len: usize, execution: &ArrayExecutionOptions) -> Result<usize> {
    let elements = len.checked_mul(3).ok_or_else(|| {
        invalid(
            "outputs.elements",
            "three MACD component element count overflowed usize",
        )
    })?;
    if elements > execution.max_elements() {
        return Err(invalid(
            "outputs.elements",
            format!(
                "three MACD components require {elements} elements; limit is {}",
                execution.max_elements()
            ),
        ));
    }
    Ok(elements)
}
