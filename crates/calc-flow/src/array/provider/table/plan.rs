use super::super::{infer, ir, validate};
use super::{TablePlan, document, lineage, outlet, rhs, source};
use crate::{JsonMap, Port, Result, array::invalid};
use std::sync::Arc;

pub(super) fn build(options: &JsonMap, inputs: &[Port], outputs: &[Port]) -> Result<TablePlan> {
    let document = parse(options)?;
    let selection = source::select(&document.inputs, inputs)?;
    let numeric = program(&document, &selection)?;
    rhs::validate(&numeric, &selection)?;
    let descriptor = lineage::validate(&numeric, &selection.name)?;
    let output = outlet::validate(&document.outputs, outputs, &selection, descriptor)?;
    Ok(TablePlan {
        numeric: Arc::new(numeric),
        table: selection.name,
        columns: selection.columns,
        dtype: selection.dtype,
        output,
    })
}
fn parse(options: &JsonMap) -> Result<document::Document> {
    validate::bounded(options)?;
    let document: document::Document = serde_json::from_value(serde_json::Value::Object(
        options.clone().into_iter().collect(),
    ))
    .map_err(|_| invalid("options", "invalid typed table array program"))?;
    if document.ir_version != 2 {
        return Err(invalid(
            "options.ir_version",
            "table arrays require version 2",
        ));
    }
    Ok(document)
}
fn program(document: &document::Document, source: &source::Selection) -> Result<infer::Plan> {
    let mut inputs = vec![ir::Input {
        name: source.name.clone(),
        dtype: source.dtype,
        shape: vec![None, Some(source.columns.len())],
    }];
    inputs.extend(source.static_input.iter().cloned());
    let program = ir::Program {
        ir_version: 2,
        inputs,
        nodes: document.nodes.clone(),
        outputs: vec![outlet::reference(&document.outputs)?],
        execution: document.execution.clone(),
    };
    validate::validate(&program)?;
    infer::plan(program)
}
