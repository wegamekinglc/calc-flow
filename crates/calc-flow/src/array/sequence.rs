use datafusion::arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};

use super::{ArrayDType, ArrayExecutionOptions, ArrayView, NativeArray, control, invalid};
use crate::Result;

pub(in crate::array) mod capture;
pub(in crate::array) mod codec;
mod ewma;
mod mean;

#[cfg(test)]
mod codec_tests;

#[cfg(test)]
mod candidate_tests;

#[cfg(test)]
mod capture_tests;

pub use ewma::EwmaState;
pub use mean::RollingMeanState;

/// Retained-storage limits for owned native sequence states.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct ArrayStateOptions {
    max_state_bytes: usize,
}

impl Default for ArrayStateOptions {
    fn default() -> Self {
        Self {
            max_state_bytes: 256 * 1024 * 1024,
        }
    }
}

impl ArrayStateOptions {
    /// Sets the maximum committed state charge, including unused history slots.
    pub const fn with_max_state_bytes(self, max_state_bytes: usize) -> Self {
        Self { max_state_bytes }
    }

    /// Returns the retained-state byte limit; this is not process RSS.
    pub const fn max_state_bytes(&self) -> usize {
        self.max_state_bytes
    }

    fn validate(self, bytes: usize) -> Result<()> {
        if bytes > self.max_state_bytes {
            return Err(invalid(
                "state.max_state_bytes",
                format!(
                    "requires {bytes} committed state bytes; limit is {}",
                    self.max_state_bytes
                ),
            ));
        }
        Ok(())
    }
}

fn validate_input(input: ArrayView<'_>) -> Result<()> {
    if input.shape().len() != 1 {
        return Err(invalid(
            "input.shape",
            format!(
                "requires a rank-one sequence; received rank {}",
                input.shape().len()
            ),
        ));
    }
    if input.dtype() == ArrayDType::Boolean {
        return Err(invalid(
            "input.dtype",
            "requires float32 or float64; received bool",
        ));
    }
    Ok(())
}

fn sample(input: ArrayView<'_>, index: usize) -> Option<f64> {
    input.number64(index).filter(|value| !value.is_nan())
}

fn history_charge(capacity: usize) -> Result<usize> {
    capacity
        .checked_mul(16)
        .and_then(|bytes| bytes.checked_add(64))
        .ok_or_else(|| {
            invalid(
                "state.max_state_bytes",
                "history byte count overflowed usize",
            )
        })
}

pub(in crate::array) struct Prepared<T> {
    pub(in crate::array) candidate: Option<T>,
    pub(in crate::array) output: NativeArray,
}

impl<T> Prepared<T> {
    fn commit(self, state: &mut T) -> NativeArray {
        if let Some(candidate) = self.candidate {
            *state = candidate;
        }
        self.output
    }
}

struct Budget<'a> {
    old: usize,
    candidate: usize,
    captures: usize,
    scratch: usize,
    len: usize,
    execution: &'a ArrayExecutionOptions,
}

impl Budget<'_> {
    fn admit(&self, output: usize) -> Result<()> {
        self.execution.validate_allocation(self.len, output, 0)?;
        let peak = [
            self.old,
            self.candidate,
            self.scratch,
            output,
            self.captures,
        ]
        .into_iter()
        .try_fold(0_usize, usize::checked_add)
        .ok_or_else(|| invalid("state.peak_bytes", "live byte count overflowed usize"))?;
        if peak > self.execution.max_allocation_bytes() {
            return Err(invalid(
                "state.peak_bytes",
                format!(
                    "requires {peak} live state/output/scratch/capture bytes; limit is {}",
                    self.execution.max_allocation_bytes()
                ),
            ));
        }
        Ok(())
    }
}

struct Output {
    values: Vec<f64>,
    bitmap: Vec<u8>,
}

impl Output {
    fn required_bytes(len: usize) -> Result<usize> {
        Self::allocated_bytes(len, len.div_ceil(8))
    }

    fn allocated_bytes(values: usize, bitmap: usize) -> Result<usize> {
        values
            .checked_mul(size_of::<f64>())
            .and_then(|bytes| bytes.checked_add(bitmap))
            .and_then(|bytes| bytes.checked_add(size_of::<usize>()))
            .ok_or_else(|| invalid("outputs.output.bytes", "byte count overflowed usize"))
    }

    fn allocate(budget: &Budget<'_>, check: &mut dyn FnMut() -> Result<()>) -> Result<Self> {
        budget.admit(Self::required_bytes(budget.len)?)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(budget.len)
            .map_err(|error| invalid("outputs.output.bytes", error.to_string()))?;
        budget.admit(Self::allocated_bytes(
            values.capacity(),
            budget.len.div_ceil(8),
        )?)?;
        let bitmap = Self::bitmap(budget, values.capacity(), check)?;
        Ok(Self { values, bitmap })
    }

    fn bitmap(
        budget: &Budget<'_>,
        values: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<u8>> {
        let mut bitmap = Vec::new();
        bitmap
            .try_reserve_exact(budget.len.div_ceil(8))
            .map_err(|error| invalid("outputs.output.bytes", error.to_string()))?;
        budget.admit(Self::allocated_bytes(values, bitmap.capacity())?)?;
        for index in 0..budget.len.div_ceil(8) {
            control::checkpoint(index, check)?;
            bitmap.push(0);
        }
        Ok(bitmap)
    }

    fn push(&mut self, value: Option<f64>) {
        let index = self.values.len();
        self.values.push(value.unwrap_or(0.0));
        if value.is_some() {
            self.bitmap[index / 8] |= 1 << (index % 8);
        }
    }

    fn finish(self, nullable_empty: bool) -> Result<NativeArray> {
        let len = self.values.len();
        let validity = if len == 0 && !nullable_empty {
            None
        } else {
            Some(NullBuffer::new(BooleanBuffer::new(
                Buffer::from(self.bitmap),
                0,
                len,
            )))
        };
        NativeArray::from_f64(self.values, vec![len], validity)
    }
}

fn next_positions(old: u64, len: usize) -> Result<u64> {
    let len =
        u64::try_from(len).map_err(|_| invalid("input.rows", "sequence position exceeds u64"))?;
    old.checked_add(len)
        .ok_or_else(|| invalid("input.rows", "sequence position exceeds u64"))
}
