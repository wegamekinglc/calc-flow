//! Controlled layout-five Arrow record preparation, before IPC writing.

use std::{alloc::Layout, sync::Arc};

use datafusion::{
    arrow::{
        array::ArrayRef,
        datatypes::{Float64Type, Schema, UInt8Type, UInt64Type},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
};

use super::{footprint::Footprint, scalar_output_plan};
use crate::operator::rolling::{
    CompiledRollingSpec, KeyValue, RollingStreamState, key_scalar,
    state_codec::{FreshStateSchema, state_schema_prepared},
};
use crate::{CalcFlowError, Result};

mod binary;
mod cells;
mod columns;
mod control;
mod order;
mod payload;
mod rows;
use super::state_schema_requirements as schema;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
type Entity = Vec<Option<KeyValue>>;
#[derive(Clone, Copy)]
pub(super) struct Inputs<'a> {
    pub(super) state: &'a RollingStreamState,
    pub(super) source_footprint: &'a Footprint,
    pub(super) input_schema: &'a Schema,
    pub(super) compiled: &'a CompiledRollingSpec,
    pub(super) fingerprint: &'a str,
    pub(super) node: &'a str,
}
pub(super) struct PreparedRecord {
    pub(super) schema: FreshStateSchema,
    pub(super) record: RecordBatch,
    pub(super) footprint: Footprint,
    pub(super) scratch_bytes: u64,
}
struct Control<'a> {
    admit: &'a mut dyn FnMut(Footprint, u64) -> Result<()>,
    check: Check<'a>,
    node: &'a str,
}
impl Control<'_> {
    fn reserve(&mut self, owned: u64, scratch: u64) -> Result<()> {
        (self.check)()?;
        let mut quote = Footprint::default();
        quote.add_owned(owned, self.node)?;
        (self.admit)(quote, scratch)?;
        (self.check)()
    }
    fn workspace(&mut self, workspace: u64, references: u64) -> Result<()> {
        self.reserve(0, schema::add(workspace, references)?)
    }
    fn output(&mut self, root: u64, plan: &columns::Plan, workspace: u64) -> Result<()> {
        self.reserve(
            schema::add(root, plan.owned)?,
            schema::add(workspace, plan.scratch)?,
        )
    }
    fn publish(&mut self, record: &PreparedRecord) -> Result<()> {
        (self.check)()?;
        (self.admit)(record.footprint.clone(), 0)?;
        (self.check)()
    }
}
struct References {
    capacity: usize,
    bytes: u64,
    peak: u64,
}
pub(super) fn prepare(
    inputs: Inputs<'_>,
    admit: &mut dyn FnMut(Footprint, u64) -> Result<()>,
    check: Check<'_>,
) -> Result<PreparedRecord> {
    let mut control = Control {
        admit,
        check,
        node: inputs.node,
    };
    rows::validate(inputs, control.check)?;
    let refs = references(inputs)?;
    control.reserve(0, refs.peak)?;
    let entities = entities(inputs, refs.capacity, control.check)?;
    let prepared = with_entities(inputs, entities, refs.bytes, &mut control)?;
    control.publish(&prepared)?;
    Ok(prepared)
}
fn references(inputs: Inputs<'_>) -> Result<References> {
    let capacity = inputs
        .state
        .histories
        .by_entity
        .len()
        .checked_add(inputs.state.buffer.len())
        .ok_or_else(invalid)?;
    let bytes = vector::<&Entity>(capacity)?;
    Ok(References {
        capacity,
        bytes,
        peak: schema::add(bytes, bytes)?,
    })
}
fn entities<'a>(inputs: Inputs<'a>, capacity: usize, check: Check<'_>) -> Result<Vec<&'a Entity>> {
    let mut entities = Vec::with_capacity(capacity);
    let mut spare = Vec::with_capacity(capacity);
    for key in inputs.state.histories.by_entity.keys() {
        check()?;
        entities.push(key);
    }
    for row in inputs.state.buffer.values() {
        check()?;
        entities.push(&row.identity.entity);
    }
    order::sort_unique(&mut entities, &mut spare, Ord::cmp, check)?;
    Ok(entities)
}
fn with_entities(
    inputs: Inputs<'_>,
    entities: Vec<&Entity>,
    refs: u64,
    control: &mut Control<'_>,
) -> Result<PreparedRecord> {
    let count = rows::count(inputs, entities.len(), control.check)?;
    let workspace = workspace(inputs, &entities, count, control.check)?;
    control.workspace(workspace, refs)?;
    let keys = keys(inputs, &entities, control.check)?;
    let rows = rows::build(inputs, &entities, &keys, count, control.check)?;
    let roles = rows::roles(inputs, control.check)?;
    drop(entities);
    with_rows(inputs, &rows, &roles, workspace, control)
}
fn workspace(
    inputs: Inputs<'_>,
    entities: &[&Entity],
    rows: usize,
    check: Check<'_>,
) -> Result<u64> {
    schema::total([
        key_bytes(inputs, entities, check),
        vector::<rows::Row<'_>>(rows),
        vector::<cells::Column>(inputs.input_schema.fields().len()),
    ])
}
fn key_bytes(inputs: Inputs<'_>, entities: &[&Entity], check: Check<'_>) -> Result<u64> {
    schema::total([
        key_storage(inputs, entities.len()),
        key_strings(entities, check),
    ])
}
fn key_storage(inputs: Inputs<'_>, entities: usize) -> Result<u64> {
    let slots = entities
        .checked_mul(inputs.compiled.partition_columns.len())
        .ok_or_else(invalid)?;
    vector::<Option<ScalarValue>>(slots)
}
fn key_strings(entities: &[&Entity], check: Check<'_>) -> Result<u64> {
    let mut owned = 0;
    for entity in entities {
        for value in *entity {
            check()?;
            owned = schema::add(owned, key_string(value.as_ref())?)?;
        }
    }
    Ok(owned)
}
fn key_string(value: Option<&KeyValue>) -> Result<u64> {
    match value {
        Some(KeyValue::String(value)) => schema::size(value.len()),
        _ => Ok(0),
    }
}
fn keys(
    inputs: Inputs<'_>,
    entities: &[&Entity],
    check: Check<'_>,
) -> Result<Vec<Option<ScalarValue>>> {
    let capacity = entities
        .len()
        .checked_mul(inputs.compiled.partition_columns.len())
        .ok_or_else(invalid)?;
    let mut values = Vec::with_capacity(capacity);
    for entity in entities {
        for (key, column) in entity.iter().zip(&inputs.compiled.partition_columns) {
            check()?;
            values.push(
                key.as_ref()
                    .map(|key| key_scalar(key, inputs.input_schema.field(column.index).data_type()))
                    .transpose()?,
            );
        }
    }
    Ok(values)
}
fn with_rows(
    inputs: Inputs<'_>,
    rows: &[rows::Row<'_>],
    roles: &[cells::Column],
    workspace: u64,
    control: &mut Control<'_>,
) -> Result<PreparedRecord> {
    let (schema, retained) = fresh_schema(inputs, workspace, control)?;
    let input = input_plan(inputs, rows, roles, retained, workspace, control)?;
    let plan = columns::plan(rows, input, control.check)?;
    let root = schema::add(retained, record_root(inputs.input_schema.fields().len())?)?;
    control.output(root, &plan, workspace)?;
    encode(inputs, rows, roles, schema, root, plan.owned, control)
}
fn fresh_schema(
    inputs: Inputs<'_>,
    workspace: u64,
    control: &mut Control<'_>,
) -> Result<(FreshStateSchema, u64)> {
    let metadata = schema::Metadata {
        pipeline: inputs.fingerprint,
        node: inputs.node,
        configuration: &inputs.compiled.configuration_hash,
        state_schema: &inputs.compiled.state_schema_fingerprint,
        kernel: inputs.compiled.kernel_plan.fingerprint(),
        numerical: inputs.compiled.kernel_plan.numerical_profile(),
    };
    let quote = schema::plan(inputs.input_schema, metadata, control.check)?;
    control.reserve(0, schema::add(workspace, quote.construction_peak)?)?;
    let schema = state_schema_prepared(
        inputs.input_schema,
        inputs.compiled,
        inputs.fingerprint,
        inputs.node,
    );
    control.reserve(quote.retained, workspace)?;
    Ok((schema, quote.retained))
}
fn input_plan(
    inputs: Inputs<'_>,
    rows: &[rows::Row<'_>],
    roles: &[cells::Column],
    retained: u64,
    workspace: u64,
    control: &mut Control<'_>,
) -> Result<scalar_output_plan::OutputPlan> {
    let mut cell = |column: usize, row: usize| rows[row].data.cell(roles[column]);
    let prefix = scalar_output_plan::planning_workspace(
        inputs.input_schema,
        rows.len(),
        &mut cell,
        control.check,
    )?;
    control.reserve(retained, schema::add(workspace, prefix)?)?;
    scalar_output_plan::plan(inputs.input_schema, rows.len(), &mut cell, control.check)
}
fn record_root(width: usize) -> Result<u64> {
    let fields = width.checked_add(8).ok_or_else(invalid)?;
    schema::add(
        vector::<ArrayRef>(fields)?,
        schema::size(size_of::<RecordBatch>())?,
    )
}
fn encode(
    inputs: Inputs<'_>,
    rows: &[rows::Row<'_>],
    roles: &[cells::Column],
    schema: FreshStateSchema,
    root: u64,
    owned: u64,
    control: &mut Control<'_>,
) -> Result<PreparedRecord> {
    let columns = columns::encode(inputs, rows, roles, control.check)?;
    let record =
        RecordBatch::try_new(Arc::clone(schema.as_schema()), columns).map_err(|_| invalid())?;
    let footprint = footprint(inputs, &record, root, owned, control.check)?;
    Ok(PreparedRecord {
        schema,
        record,
        footprint,
        scratch_bytes: 0,
    })
}
fn footprint(
    inputs: Inputs<'_>,
    record: &RecordBatch,
    root: u64,
    owned: u64,
    check: Check<'_>,
) -> Result<Footprint> {
    let mut actual = Footprint::default();
    for column in record.columns() {
        actual.array_checked(column, inputs.node, check)?;
    }
    actual.add_owned(residual(&actual, inputs, root, owned)?, inputs.node)?;
    check()?;
    Ok(actual)
}
fn residual(actual: &Footprint, inputs: Inputs<'_>, root: u64, owned: u64) -> Result<u64> {
    let observed = actual.incremental_bytes(inputs.source_footprint, inputs.node)?;
    let residual = owned.checked_sub(observed).ok_or_else(invalid)?;
    schema::add(root, residual)
}
fn vector<T>(length: usize) -> Result<u64> {
    let layout = Layout::array::<T>(length).map_err(|_| invalid())?;
    schema::size(layout.size())
}
fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling checkpoint record shape or ownership is invalid".into(),
    }
}

#[cfg(test)]
mod tests;
