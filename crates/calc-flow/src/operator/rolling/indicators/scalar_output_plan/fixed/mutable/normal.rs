use super::super::super::concatenated::{At, Piece};
use super::sources::Emit;
use super::{Check, Cost, Result, Sources, node};
use datafusion::arrow::datatypes::DataType;

pub(super) fn plan(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let mut sources = Normal { count, at };
    node(dtype, rows, &mut sources, check)
}
struct Normal<'a, 'b> {
    count: usize,
    at: &'b mut (dyn FnMut(usize) -> Result<Piece<'a>> + 'b),
}
impl<'a> Sources<'a> for Normal<'a, '_> {
    fn count(&self) -> usize {
        self.count
    }
    fn source(&mut self, index: usize) -> Result<Piece<'a>> {
        (self.at)(index)
    }
    fn visit(&mut self, index: usize, emit: &mut Emit<'a, '_>, check: Check<'_>) -> Result<()> {
        check()?;
        emit(self.source(index)?, check)
    }
}
