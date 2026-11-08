use std::{collections::BTreeSet, io};

use super::{
    super::invalid,
    ir::{Node, Program, validate_shape},
};
use crate::{BatchKind, JsonMap, Port, Result, json::validate_portable_identifier};

struct Bound(usize);

impl io::Write for Bound {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .filter(|length| *length <= 65_536)
            .ok_or_else(|| io::Error::other("array options exceed 64 KiB"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn parse(options: &JsonMap) -> Result<Program> {
    parse_mode(options, false)
}
pub(super) fn parse_query(options: &JsonMap) -> Result<Program> {
    parse_mode(options, true)
}
fn parse_mode(options: &JsonMap, query: bool) -> Result<Program> {
    bounded(options)?;
    let version = options
        .get("ir_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|version| u8::try_from(version).ok())
        .ok_or_else(|| invalid("options.ir_version", "requires a supported integer version"))?;
    let program = super::document::decode(options, version, query)?;
    validate(&program)?;
    Ok(program)
}
pub(super) fn bounded(options: &JsonMap) -> Result<()> {
    inspect_children(options.values(), 1, &mut 0)?;
    serde_json::to_writer(Bound(0), options)
        .map_err(|_| invalid("options.bytes", "compact options exceed 64 KiB"))
}

fn inspect_children<'a>(
    values: impl Iterator<Item = &'a serde_json::Value>,
    depth: usize,
    visited: &mut usize,
) -> Result<()> {
    for value in values {
        inspect(value, depth, visited)?;
    }
    Ok(())
}

fn inspect(value: &serde_json::Value, depth: usize, visited: &mut usize) -> Result<()> {
    if depth > crate::json::MAX_JSON_DEPTH {
        return Err(invalid("options.depth", "JSON nesting exceeds 32"));
    }
    *visited += 1;
    if *visited > 65_536 {
        return Err(invalid("options.bytes", "compact options exceed 64 KiB"));
    }
    match value {
        serde_json::Value::Array(values) => inspect_children(values.iter(), depth + 1, visited),
        serde_json::Value::Object(values) => inspect_children(values.values(), depth + 1, visited),
        _ => Ok(()),
    }
}

fn identifier(value: &str, field: &str) -> Result<()> {
    validate_portable_identifier(field, value)?;
    if value.len() > 64 {
        return Err(invalid(field, "identifier exceeds 64 ASCII bytes"));
    }
    Ok(())
}

fn names<'a>(values: impl Iterator<Item = &'a str>, field: &str, maximum: usize) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        identifier(value, field)?;
        if !seen.insert(value) {
            return Err(invalid(field, "duplicate name or identifier"));
        }
    }
    if seen.is_empty() || seen.len() > maximum {
        return Err(invalid(
            field,
            format!("expected 1..={maximum} declarations"),
        ));
    }
    Ok(())
}

fn validate(program: &Program) -> Result<()> {
    names(
        program.inputs.iter().map(|input| input.name.as_str()),
        "options.inputs",
        32,
    )?;
    names(
        program.outputs.iter().map(|output| output.name.as_str()),
        "options.outputs",
        16,
    )?;
    names(program.nodes.iter().map(Node::id), "options.nodes", 128)?;
    shapes(program)?;
    matmuls(program)
}
fn shapes(program: &Program) -> Result<()> {
    for input in &program.inputs {
        validate_shape(
            &input.shape,
            usize::from(program.ir_version),
            "options.inputs.shape",
        )?;
    }
    for output in &program.outputs {
        if let Some(assertion) = &output.assertion {
            validate_shape(
                &assertion.shape,
                usize::from(program.ir_version),
                "options.outputs.shape",
            )?;
        }
    }

    Ok(())
}
fn matmuls(program: &Program) -> Result<()> {
    if program
        .nodes
        .iter()
        .filter(|node| matches!(node, Node::Matmul { .. }))
        .count()
        > 1
    {
        return Err(invalid(
            "options.nodes.matmul",
            "at most one matmul is supported",
        ));
    }
    Ok(())
}

pub(super) fn ports<'a>(
    ports: &[Port],
    expected: impl Iterator<Item = &'a str>,
    field: &str,
) -> Result<()> {
    let declared: BTreeSet<_> = expected.collect();
    let received: BTreeSet<_> = ports.iter().map(Port::name).collect();
    let exact = ports
        .iter()
        .all(|port| port.kind() == BatchKind::Array && port.required());
    if !exact || received != declared || ports.len() != declared.len() {
        return Err(invalid(
            field,
            "required Array ports must exactly match declarations",
        ));
    }
    Ok(())
}
