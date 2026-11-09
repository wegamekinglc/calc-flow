use super::super::super::{concatenated::Piece, workspace};
use super::sources::Emit;
use super::{
    Check, Cost, Result, Sources, invalid,
    math::{add, size, sum, vector},
    node,
};
use datafusion::arrow::{array::ArrayData, datatypes::DataType};
#[path = "dictionary/values.rs"]
mod values;
use values::{Value, at as value_at};

pub(super) fn plan(
    key: &DataType,
    dtype: &DataType,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let shared = shared(dtype, sources, check)?;
    let capacity = spans(key, shared, sources, check)?;
    if shared {
        return shared_cost(dtype, value_at(sources, 0)?, check);
    }
    distinct(dtype, capacity, sources, check)
}
fn distinct(
    dtype: &DataType,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let count = sources.count();
    let mut values = Values { parent: sources };
    let mut cost = node(dtype, capacity, &mut values, check)?;
    cost.scratch = add(cost.scratch, vectors(count)?)?;
    Ok(cost)
}
fn vectors(count: usize) -> Result<u64> {
    sum([vector::<&ArrayData>(count), vector::<usize>(count)])
}
fn shared(dtype: &DataType, sources: &mut dyn Sources<'_>, check: Check<'_>) -> Result<bool> {
    for index in 1..sources.count() {
        check()?;
        if !pair(dtype, sources, index, check)? {
            return Ok(false);
        }
    }
    Ok(true)
}
fn pair(
    dtype: &DataType,
    sources: &mut dyn Sources<'_>,
    index: usize,
    check: Check<'_>,
) -> Result<bool> {
    let equal = values::equal(
        dtype,
        value_at(sources, index - 1)?,
        value_at(sources, index)?,
        check,
    )?;
    check()?;
    Ok(equal)
}

fn spans(
    key: &DataType,
    shared: bool,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<u64> {
    let mut offset = 0;
    for index in 0..sources.count() {
        check()?;
        let len = value_at(sources, index)?.length()?;
        super::super::super::producer::key_span(key, offset, len)?;
        offset = next_offset(shared, offset, len)?;
    }
    Ok(offset)
}
fn next_offset(shared: bool, offset: u64, len: u64) -> Result<u64> {
    if shared { Ok(offset) } else { add(offset, len) }
}
fn shared_cost(dtype: &DataType, value: Value<'_>, check: Check<'_>) -> Result<Cost> {
    let Value::Array(array) = value else {
        let cost = super::super::super::super::scalar_materialization_plan::null_array_cost(
            dtype, 0, check,
        )?;
        return Ok(Cost {
            output: cost.logical_owned,
            scratch: cost.construction_peak,
            ..Cost::default()
        });
    };
    Ok(Cost {
        output: workspace::descriptor_tree(array, check)?,
        scratch: workspace::array_data_workspace(array, check)?,
        ..Cost::default()
    })
}
struct Values<'a, 'b> {
    parent: &'b mut dyn Sources<'a>,
}
impl<'a> Sources<'a> for Values<'a, '_> {
    fn count(&self) -> usize {
        self.parent.count()
    }
    fn scalar(&mut self, index: usize) -> Result<Option<&'a datafusion::common::ScalarValue>> {
        Ok(value_at(self.parent, index)?.scalar())
    }
    fn source(&mut self, index: usize) -> Result<Piece<'a>> {
        value_at(self.parent, index)?.piece()
    }
    fn visit(&mut self, index: usize, emit: &mut Emit<'a, '_>, check: Check<'_>) -> Result<()> {
        check()?;
        emit(self.source(index)?, check)
    }
}
