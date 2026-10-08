use super::{DescriptorFunding, SchemaConstruction, SchemaDecision, SchemaWork, plan};
use datafusion::arrow::datatypes::{Field, FieldRef, Schema};
use std::mem::{align_of, size_of};

pub(super) fn required(schemas: [&Schema; 2]) -> Option<(usize, usize)> {
    let left = side(schemas[0])?;
    let right = side(schemas[1])?;
    let input = side_total(left.0, right.0, input_controls())?;
    let output = side_total(left.1, right.1, output_controls())?;
    Some((input, output))
}

fn side_total(left: usize, right: usize, controls: Option<usize>) -> Option<usize> {
    left.checked_add(right)?.checked_add(controls?)
}

fn input_controls() -> Option<usize> {
    size_of::<SchemaConstruction>()
        .checked_add(size_of::<SchemaWork>())?
        .checked_add(size_of::<SchemaDecision>())
}

fn output_controls() -> Option<usize> {
    arc_layout(
        size_of::<DescriptorFunding>(),
        align_of::<DescriptorFunding>(),
    )?
    .checked_add(size_of::<SchemaDecision>())?
    .checked_add(super::super::inventory::registration_controls()?)
}

fn side(schema: &Schema) -> Option<(usize, usize)> {
    if schema.fields().len() > plan::MAX_FIELDS || !schema.metadata().is_empty() {
        return None;
    }
    let count = schema.fields().len();
    let input = count.checked_mul(size_of::<plan::FieldPlan>())?;
    let output = side_output(count)?;
    schema.fields().iter().try_fold((input, output), add_field)
}

fn side_output(count: usize) -> Option<usize> {
    let pointers = count.checked_mul(size_of::<FieldRef>())?;
    arc_layout(size_of::<Schema>(), align_of::<Schema>())?
        .checked_add(pointers)?
        .checked_add(arc_layout(pointers, align_of::<FieldRef>())?)
}

fn add_field((input, output): (usize, usize), field: &FieldRef) -> Option<(usize, usize)> {
    let (field_input, field_output) = field_bytes(field)?;
    Some((
        input.checked_add(field_input)?,
        output.checked_add(field_output)?,
    ))
}

fn field_bytes(field: &Field) -> Option<(usize, usize)> {
    if !field.metadata().is_empty() || field.name().len() > plan::MAX_TEXT_BYTES {
        return None;
    }
    plan::scalar_type(field.data_type())?;
    let text = plan::timezone(field.data_type());
    let timezone_input = text.map_or(Some(0), |text| {
        (text.len() <= plan::MAX_TEXT_BYTES).then_some(text.len())
    })?;
    let input = field.name().len().checked_add(timezone_input)?;
    let output = field_output(field, text)?;
    Some((input, output))
}

fn field_output(field: &Field, timezone: Option<&str>) -> Option<usize> {
    let timezone_output = timezone.map_or(Some(0), |text| arc_layout(text.len(), 1))?;
    arc_layout(size_of::<Field>(), align_of::<Field>())?
        .checked_add(field.name().len())?
        .checked_add(timezone_output)
}

fn arc_layout(bytes: usize, alignment: usize) -> Option<usize> {
    let word_alignment = align_of::<usize>();
    let header = align_up(2 * size_of::<usize>(), alignment)?;
    align_up(header.checked_add(bytes)?, alignment.max(word_alignment))
}

fn align_up(bytes: usize, alignment: usize) -> Option<usize> {
    bytes
        .checked_add(alignment - 1)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}
