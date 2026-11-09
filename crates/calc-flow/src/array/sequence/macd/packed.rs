use datafusion::arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};

use super::{MacdState, output::elements};
use crate::{
    Result,
    array::{ArrayExecutionOptions, NativeArray},
};

pub(super) struct Packed {
    rows: usize,
    output: super::super::Output,
}
impl Packed {
    pub(super) fn allocate(
        state: &MacdState,
        rows: usize,
        captures: usize,
        execution: &ArrayExecutionOptions,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Self> {
        let len = elements(rows, execution)?;
        let budget = super::super::Budget {
            peak_field: "execution.max_allocation_bytes",
            old: state.state_bytes(),
            candidate: if rows == 0 { 0 } else { 56 },
            captures,
            scratch: size_of::<usize>(),
            len,
            execution,
        };
        Ok(Self {
            rows,
            output: super::super::Output::allocate(&budget, check)?,
        })
    }
    pub(super) fn push(&mut self, values: [Option<f64>; 3]) {
        for value in values {
            self.output.push(value);
        }
    }
    pub(super) fn finish(self) -> Result<NativeArray> {
        let len = self.output.values.len();
        let mask = NullBuffer::new(BooleanBuffer::new(Buffer::from(self.output.bitmap), 0, len));
        NativeArray::from_f64(self.output.values, vec![self.rows, 3], Some(mask))
    }
}
