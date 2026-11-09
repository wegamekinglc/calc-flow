use std::{
    alloc::Layout,
    sync::{Arc, atomic::AtomicUsize},
};

use datafusion::arrow::datatypes::{Field, Schema};

use super::scalar_materialization_plan::datatype_clone_requirement;
use crate::{CalcFlowError, Result};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
const CONTROL_NAMES: [&str; 8] = [
    "_state_kind",
    "_entity_id",
    "_entity_position",
    "_group",
    "_valid_count",
    "_value",
    "_cumulative_summary",
    "_indicator_state",
];

#[derive(Clone, Copy)]
pub(super) struct Metadata<'a> {
    pub(super) pipeline: &'a str,
    pub(super) node: &'a str,
    pub(super) configuration: &'a str,
    pub(super) state_schema: &'a str,
    pub(super) kernel: &'a str,
    pub(super) numerical: &'a str,
}
impl<'a> Metadata<'a> {
    const fn pairs(self) -> [(&'static str, &'a str); 7] {
        [
            ("calc_flow.state_layout_version", "5"),
            ("calc_flow.pipeline_fingerprint", self.pipeline),
            ("calc_flow.operator_id", self.node),
            ("calc_flow.operator_configuration_hash", self.configuration),
            ("calc_flow.state_schema_fingerprint", self.state_schema),
            ("calc_flow.rolling_kernel_fingerprint", self.kernel),
            ("calc_flow.numerical_profile", self.numerical),
        ]
    }
}
pub(super) struct Quote {
    pub(super) retained: u64,
    pub(super) construction_peak: u64,
}
pub(super) fn plan(input: &Schema, metadata: Metadata<'_>, check: Check<'_>) -> Result<Quote> {
    check()?;
    let retained = retained(input, metadata, check)?;
    let construction_peak = add(retained, field_vector_peak(input.fields().len())?)?;
    check()?;
    Ok(Quote {
        retained,
        construction_peak,
    })
}
fn retained(input: &Schema, metadata: Metadata<'_>, check: Check<'_>) -> Result<u64> {
    total([
        arc::<Schema>(),
        fields(input, check),
        metadata_bytes(metadata),
    ])
}
fn fields(input: &Schema, check: Check<'_>) -> Result<u64> {
    total([
        field_owners(input.fields().len()),
        input_fields(input, check),
        control_names(),
    ])
}
fn field_owners(width: usize) -> Result<u64> {
    let count = width.checked_add(CONTROL_NAMES.len()).ok_or_else(invalid)?;
    total([
        arc_slice::<Arc<Field>>(count),
        repeat(arc::<Field>()?, count),
    ])
}
fn input_fields(input: &Schema, check: Check<'_>) -> Result<u64> {
    input
        .fields()
        .iter()
        .try_fold(0, |total, field| add(total, field_bytes(field, check)?))
}
fn field_bytes(field: &Field, check: Check<'_>) -> Result<u64> {
    add(
        size(field.name().len())?,
        datatype_clone_requirement(field.data_type(), check)?,
    )
}
fn control_names() -> Result<u64> {
    CONTROL_NAMES
        .into_iter()
        .try_fold(0, |total, name| add(total, size(name.len())?))
}
fn metadata_bytes(metadata: Metadata<'_>) -> Result<u64> {
    metadata
        .pairs()
        .into_iter()
        .try_fold(metadata_table()?, |total, (key, value)| {
            add(total, add(size(key.len())?, size(value.len())?)?)
        })
}
fn metadata_table() -> Result<u64> {
    // std HashMap::from(5) reserves eight buckets; two further inserts fit.
    let table = Layout::array::<(String, String)>(8).map_err(|_| invalid())?;
    let padded = table
        .size()
        .checked_add(table.align().max(16))
        .ok_or_else(invalid)?;
    size(padded.checked_add(8 + 16).ok_or_else(invalid)?)
}
fn field_vector_peak(width: usize) -> Result<u64> {
    // state_fields extends its initial three slots, extends three, then pushes two.
    let mut capacity = 3;
    let mut peak = capacity;
    for extra in [3_usize, 6, 7, 8] {
        let required = width.checked_add(extra).ok_or_else(invalid)?;
        grow(&mut capacity, &mut peak, required)?;
    }
    repeat(size(size_of::<Field>())?, peak.max(capacity))
}
fn grow(capacity: &mut usize, peak: &mut usize, required: usize) -> Result<()> {
    if required > *capacity {
        let next = capacity.checked_mul(2).ok_or_else(invalid)?.max(required);
        *peak = (*peak).max(capacity.checked_add(next).ok_or_else(invalid)?);
        *capacity = next;
    }
    Ok(())
}
pub(super) fn arc<T>() -> Result<u64> {
    arc_layout(Layout::new::<T>())
}
fn arc_slice<T>(length: usize) -> Result<u64> {
    arc_layout(Layout::array::<T>(length).map_err(|_| invalid())?)
}
fn arc_layout(value: Layout) -> Result<u64> {
    let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
        .extend(value)
        .map_err(|_| invalid())?;
    size(layout.pad_to_align().size())
}
fn repeat(bytes: u64, count: usize) -> Result<u64> {
    bytes.checked_mul(size(count)?).ok_or_else(invalid)
}
pub(super) fn total<const N: usize>(values: [Result<u64>; N]) -> Result<u64> {
    values
        .into_iter()
        .try_fold(0, |total, value| add(total, value?))
}
pub(super) fn size(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid())
}
pub(super) fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right).ok_or_else(invalid)
}
fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling checkpoint schema ownership size overflowed".into(),
    }
}

#[cfg(test)]
#[path = "state_record/schema_tests.rs"]
mod tests;
