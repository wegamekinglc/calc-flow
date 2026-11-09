use super::super::{
    concatenated::{Cost, Piece},
    invalid,
};
use super::{At, Check, Result};
use datafusion::arrow::datatypes::DataType;
#[path = "dictionary/geometry.rs"]
mod geometry;
#[path = "dictionary/workspace.rs"]
mod workspace;
pub(in crate::operator::rolling::indicators::scalar_output_plan) use workspace::plan as planning_workspace;
#[path = "dictionary/mask.rs"]
mod mask;
#[path = "dictionary/merge.rs"]
mod merge;
#[path = "dictionary/source.rs"]
mod source;

pub(in crate::operator::rolling::indicators::scalar_output_plan) fn plan(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut super::super::concatenated::At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let DataType::Dictionary(key, value) = dtype else {
        return Err(invalid("normal dictionary datatype"));
    };
    let mut sparse = |index| at(index).map(Some);
    if !should_merge(key, value, count, rows, &mut sparse, check)? {
        return super::super::fixed::normal(dtype, count, rows, at, check);
    }
    let merged = merge::run(key, count, &mut sparse, check)?;
    geometry::plan(key, value, count, rows, &merged, &mut sparse, check)
}
fn should_merge(
    key: &DataType,
    value: &DataType,
    count: usize,
    rows: u64,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<bool> {
    if !source::mergeable(value) {
        return Ok(false);
    }
    let facts = geometry::facts(count, at, check)?;
    if facts.values <= super::super::producer::key_limit(key)? && facts.values < rows {
        return Ok(false);
    }
    Ok(!same(value, count, at, check)?)
}
fn same(dtype: &DataType, count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<bool> {
    let mut first: Option<Piece<'_>> = None;
    for index in 0..count {
        check()?;
        if let Some(piece) = at(index)? {
            if !matches(dtype, first, piece, check)? {
                return Ok(false);
            }
            first.get_or_insert(piece);
        }
    }
    Ok(true)
}

fn matches(
    dtype: &DataType,
    first: Option<Piece<'_>>,
    piece: Piece<'_>,
    check: Check<'_>,
) -> Result<bool> {
    first.map_or(Ok(true), |first| source::equal(dtype, first, piece, check))
}

pub(super) fn validate(
    dtype: &DataType,
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<()> {
    let DataType::Dictionary(key, value) = dtype else {
        return Err(invalid("normal dictionary datatype"));
    };
    let rows = geometry::facts(count, at, check)?.rows;
    if !should_merge(key, value, count, rows, at, check)? {
        return super::super::producer::normal_fallback(dtype, count, at, check);
    }
    let merged = merge::run(key, count, at, check)?;
    geometry::plan(key, value, count, rows, &merged, at, check).map(|_| ())
}
