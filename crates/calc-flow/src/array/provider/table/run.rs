use super::super::{execute, infer};
use super::{OutputMode, TablePlan, attachment, pack};
use crate::{
    Batch, Result,
    array::{NativeArray, invalid},
};
use std::{collections::BTreeMap, sync::Arc};
pub(super) fn execute(
    plan: &TablePlan,
    inputs: &BTreeMap<String, Batch>,
    static_bytes: usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<BTreeMap<String, Batch>> {
    check()?;
    let input = &inputs[&plan.table];
    let prepared = prepare(plan, inputs, static_bytes, check)?;
    let inputs = pack_inputs(plan, inputs, &prepared, check)?;
    let outputs = execute::run(&prepared.numeric, &inputs, &prepared.execution, check)?;
    let outputs = publish(plan, input, &outputs, prepared.attachment, check)?;
    check()?;
    Ok(outputs)
}
struct Prepared {
    numeric: Arc<infer::Plan>,
    packing: pack::Packing,
    execution: execute::Prepared,
    attachment: usize,
}
fn prepare(
    plan: &TablePlan,
    inputs: &BTreeMap<String, Batch>,
    static_bytes: usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Prepared> {
    let base = execute::reserve(&plan.numeric, static_bytes)?;
    let packing = pack::prepare(
        &inputs[&plan.table],
        &plan.columns,
        plan.dtype,
        &base.program.execution.options(),
        check,
    )?;
    let descriptors = input_descriptors(plan, inputs, packing)?;
    let concrete = infer::infer(&base, &descriptors)?;
    prepare_numeric(plan, &base, packing, &concrete)
}
fn input_descriptors(
    plan: &TablePlan,
    inputs: &BTreeMap<String, Batch>,
    packing: pack::Packing,
) -> Result<BTreeMap<String, infer::Descriptor>> {
    let table = infer::Descriptor {
        dtype: packing.dtype,
        shape: infer::Shape::Matrix(Some(packing.rows), Some(packing.width)),
        nullable: packing.nullable,
        validity: packing.nullable,
    };
    let mut descriptors = BTreeMap::from([(plan.table.clone(), table)]);
    for input in &plan.numeric.program.inputs {
        if input.name != plan.table {
            descriptors.insert(
                input.name.clone(),
                execute::input_descriptor(input, &inputs[&input.name], 2)?,
            );
        }
    }
    Ok(descriptors)
}
fn prepare_numeric(
    plan: &TablePlan,
    base: &Arc<infer::Plan>,
    packing: pack::Packing,
    concrete: &[infer::Descriptor],
) -> Result<Prepared> {
    let attachment = attachment_bytes(plan, concrete)?;
    let reserved = packing
        .bytes
        .checked_add(attachment)
        .ok_or_else(|| invalid("outputs.bytes", "packing and attachment bytes overflowed"))?;
    let numeric = execute::reserve(base, reserved)?;
    let execution = execute::prepare_inferred(&numeric, concrete)?;
    Ok(Prepared {
        numeric,
        packing,
        execution,
        attachment,
    })
}
fn attachment_bytes(plan: &TablePlan, concrete: &[infer::Descriptor]) -> Result<usize> {
    match plan.output {
        OutputMode::Array => Ok(0),
        OutputMode::Table(_) => attachment::requirements(
            concrete[plan.numeric.indices[&plan.numeric.program.outputs[0].node]],
        ),
    }
}
fn pack_inputs(
    plan: &TablePlan,
    inputs: &BTreeMap<String, Batch>,
    prepared: &Prepared,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<BTreeMap<String, Batch>> {
    let input = &inputs[&plan.table];
    let packed = pack::pack(input, &plan.columns, prepared.packing, check)?;
    let packed = Batch::external(Arc::new(packed), input.metadata().clone())?;
    let mut inputs = inputs.clone();
    inputs.insert(plan.table.clone(), packed);
    Ok(inputs)
}
fn publish(
    plan: &TablePlan,
    input: &Batch,
    outputs: &BTreeMap<String, Batch>,
    budget: usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<BTreeMap<String, Batch>> {
    let name = &plan.numeric.program.outputs[0].name;
    let output = match &plan.output {
        OutputMode::Array => outputs[name].with_metadata(input.metadata().clone()),
        OutputMode::Table(names) => attach_result(input, &outputs[name], names, budget, check)?,
    };
    Ok(BTreeMap::from([(name.clone(), output)]))
}
fn attach_result(
    input: &Batch,
    result: &Batch,
    names: &[String],
    budget: usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Batch> {
    let value = result
        .external_payload()?
        .as_any()
        .downcast_ref::<NativeArray>()
        .expect("native executor returns native arrays");
    attachment::attach(input, value, names, budget, check)
}
