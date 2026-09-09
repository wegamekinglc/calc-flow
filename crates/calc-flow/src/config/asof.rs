//! Project construction and typed validation for the independent ASOF operator.

use crate::{
    BatchKind, CalcFlowError, NodeOperator, OperatorMetadata, Port, Result, StreamAsofJoinOperator,
    StreamAsofJoinSpec,
};

use super::{
    CompileMode, NodeSpec, ValidationIssue, issue, port_from_spec, stream_join_inputs_valid,
    validate_derived_outputs,
};

pub(super) fn build_node(
    node: &NodeSpec,
    inputs: &[Port],
    outputs: &[Port],
    spec: &StreamAsofJoinSpec,
    mode: CompileMode,
) -> Result<NodeOperator> {
    if mode != CompileMode::Stream {
        return Err(CalcFlowError::Compile {
            message: "stream_asof_join is available only in stream mode".into(),
        });
    }
    let [left, right] = inputs else {
        return Err(invalid_ports());
    };
    let left_schema = input_schema(left, "left")?;
    let right_schema = input_schema(right, "right")?;
    let operator = StreamAsofJoinOperator::new(&node.id, left_schema, right_schema, spec.clone())?;
    validate_derived_outputs(outputs, operator.output_ports())?;
    Ok(NodeOperator::StreamAsofJoin(Box::new(operator)))
}

fn input_schema(port: &Port, name: &str) -> Result<datafusion::arrow::datatypes::SchemaRef> {
    if port.name() != name || port.kind() != BatchKind::Table || !port.required() {
        return Err(invalid_ports());
    }
    port.schema().cloned().ok_or_else(invalid_ports)
}

fn invalid_ports() -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "node.input_ports".into(),
        message:
            "stream_asof_join requires required exact-schema table inputs left and right in order"
                .into(),
    }
}

pub(super) fn validate_node(
    node: &NodeSpec,
    index: usize,
    spec: &StreamAsofJoinSpec,
    mode: CompileMode,
    base: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    if mode != CompileMode::Stream {
        issues.push(issue(
            base,
            "unsupported_mode",
            "stream_asof_join is available only in stream mode",
        ));
    }
    if !stream_join_inputs_valid(node) {
        issues.push(issue(
            format!("graph.nodes[{index}].input_ports"),
            "invalid_asof_ports",
            invalid_ports().to_string(),
        ));
        return;
    }
    validate_schemas(node, spec, base, issues);
}

fn validate_schemas(
    node: &NodeSpec,
    spec: &StreamAsofJoinSpec,
    base: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    let (Ok(left), Ok(right)) = (
        port_from_spec(&node.input_ports[0]),
        port_from_spec(&node.input_ports[1]),
    ) else {
        // The shared field validator reports invalid Arrow type strings.
        return;
    };
    let (Some(left), Some(right)) = (left.schema(), right.schema()) else {
        return;
    };
    issues.extend(
        crate::operator::asof_schema_issues(spec, left, right)
            .into_iter()
            .map(|mut issue| {
                issue.path = format!("{base}.spec.{}", issue.path);
                issue
            }),
    );
}
