use super::{
    ArrayRef, Check, Float64Type, Inputs, Result, UInt8Type, UInt64Type, binary, cells::Column,
    control, invalid, rows::Row, scalar_output_plan::OutputPlan, schema,
};

pub(super) struct Plan {
    pub(super) owned: u64,
    pub(super) scratch: u64,
}
impl Plan {
    fn include(&mut self, cost: control::Cost) -> Result<()> {
        self.owned = schema::add(self.owned, cost.retained)?;
        self.scratch = self.scratch.max(
            cost.construction_peak
                .checked_sub(cost.retained)
                .ok_or_else(invalid)?,
        );
        Ok(())
    }
}
pub(super) fn plan(rows: &[Row<'_>], input: OutputPlan, check: Check<'_>) -> Result<Plan> {
    let mut plan = Plan {
        owned: input.output_bytes,
        scratch: input.scratch_bytes,
    };
    let flags = nullable(rows, check)?;
    include_primitive(&mut plan, rows.len(), flags)?;
    include_binary(&mut plan, rows, check)?;
    Ok(plan)
}
fn include_primitive(plan: &mut Plan, rows: usize, flags: [bool; 4]) -> Result<()> {
    for cost in [
        control::requirement::<UInt8Type>(rows, false),
        control::requirement::<UInt64Type>(rows, false),
        control::requirement::<UInt64Type>(rows, flags[0]),
        control::requirement::<UInt64Type>(rows, flags[1]),
        control::requirement::<UInt64Type>(rows, flags[2]),
        control::requirement::<Float64Type>(rows, flags[3]),
    ] {
        plan.include(cost?)?;
    }
    Ok(())
}
fn include_binary(plan: &mut Plan, rows: &[Row<'_>], check: Check<'_>) -> Result<()> {
    for indicator in [false, true] {
        let shape = binary_shape(rows, indicator, check)?;
        plan.include(binary::requirement(shape)?)?;
    }
    Ok(())
}
fn nullable(rows: &[Row<'_>], check: Check<'_>) -> Result<[bool; 4]> {
    let mut flags = [false; 4];
    for row in rows {
        check()?;
        flags[0] |= row.position.is_none();
        flags[1] |= row.group.is_none();
        flags[2] |= row.count.is_none();
        flags[3] |= row.value.is_none();
    }
    Ok(flags)
}
fn binary_shape(rows: &[Row<'_>], indicator: bool, check: Check<'_>) -> Result<binary::Shape> {
    let mut shape = binary::Shape {
        rows: rows.len(),
        values: 0,
        nullable: false,
        payload_peak: 0,
    };
    for row in rows {
        check()?;
        match row.payload_shape(indicator)? {
            Some((length, peak)) => {
                shape.values = shape.values.checked_add(length).ok_or_else(invalid)?;
                shape.payload_peak = shape.payload_peak.max(peak);
            }
            None => shape.nullable = true,
        }
    }
    Ok(shape)
}
pub(super) fn encode(
    inputs: Inputs<'_>,
    rows: &[Row<'_>],
    roles: &[Column],
    check: Check<'_>,
) -> Result<Vec<ArrayRef>> {
    let capacity = inputs
        .input_schema
        .fields()
        .len()
        .checked_add(8)
        .ok_or_else(invalid)?;
    let mut arrays = Vec::with_capacity(capacity);
    let flags = nullable(rows, check)?;
    prefix(&mut arrays, rows, flags[0], check)?;
    input(&mut arrays, inputs, rows, roles, check)?;
    suffix(&mut arrays, rows, flags, check)?;
    Ok(arrays)
}
fn prefix(
    arrays: &mut Vec<ArrayRef>,
    rows: &[Row<'_>],
    nullable: bool,
    check: Check<'_>,
) -> Result<()> {
    arrays.push(control::primitive::<UInt8Type>(
        rows.len(),
        false,
        &mut |row| Some(rows[row].kind),
        check,
    )?);
    arrays.push(control::primitive::<UInt64Type>(
        rows.len(),
        false,
        &mut |row| Some(rows[row].entity),
        check,
    )?);
    arrays.push(control::primitive::<UInt64Type>(
        rows.len(),
        nullable,
        &mut |row| rows[row].position,
        check,
    )?);
    Ok(())
}
fn input(
    arrays: &mut Vec<ArrayRef>,
    inputs: Inputs<'_>,
    rows: &[Row<'_>],
    roles: &[Column],
    check: Check<'_>,
) -> Result<()> {
    for (column, field) in inputs.input_schema.fields().iter().enumerate() {
        check()?;
        arrays.push(super::scalar_output_plan::encode_column(
            field.data_type(),
            rows.len(),
            &mut |row| rows[row].data.cell(roles[column]),
            check,
        )?);
    }
    Ok(())
}
fn suffix(
    arrays: &mut Vec<ArrayRef>,
    rows: &[Row<'_>],
    flags: [bool; 4],
    check: Check<'_>,
) -> Result<()> {
    arrays.push(control::primitive::<UInt64Type>(
        rows.len(),
        flags[1],
        &mut |row| rows[row].group,
        check,
    )?);
    arrays.push(control::primitive::<UInt64Type>(
        rows.len(),
        flags[2],
        &mut |row| rows[row].count,
        check,
    )?);
    arrays.push(control::primitive::<Float64Type>(
        rows.len(),
        flags[3],
        &mut |row| rows[row].value,
        check,
    )?);
    binary_columns(arrays, rows, check)
}
fn binary_columns(arrays: &mut Vec<ArrayRef>, rows: &[Row<'_>], check: Check<'_>) -> Result<()> {
    let summaries = binary_shape(rows, false, check)?;
    arrays.push(binary::column(
        summaries,
        &mut |row| rows[row].summary(),
        check,
    )?);
    let indicators = binary_shape(rows, true, check)?;
    arrays.push(binary::column(
        indicators,
        &mut |row| rows[row].indicator(),
        check,
    )?);
    Ok(())
}
