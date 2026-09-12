//! Field-level diagnostics for the staged late-output project contract.

use crate::{CalcFlowError, LatePolicySpec};

use super::{CompileMode, NodeSpec, OperatorSpec, ValidationIssue, issue};

pub(super) fn validate_node(
    node: &NodeSpec,
    index: usize,
    mode: CompileMode,
    issues: &mut Vec<ValidationIssue>,
) {
    let (kind, policy) = match &node.operator {
        OperatorSpec::Rolling { spec } => ("rolling", spec.late_policy),
        OperatorSpec::CrossSection { spec } => ("cross_section", spec.late_policy),
        _ => return,
    };
    if !matches!(policy, LatePolicySpec::SideOutput { .. }) {
        return;
    }
    let base = format!("graph.nodes[{index}]");
    if let Err(CalcFlowError::InvalidArgument { field, message }) =
        crate::operator::late_output::validate_policy(policy, kind)
    {
        let field = field.trim_start_matches(kind);
        issues.push(issue(
            format!("{base}.operator.spec{field}"),
            "unsupported_version",
            message,
        ));
    }
    if let Some(input) = node.input_ports.first() {
        for (field_index, field) in input.schema.iter().enumerate() {
            if field.name.starts_with("_cf_late_") {
                issues.push(issue(
                    format!("{base}.input_ports[0].schema[{field_index}].name"),
                    "reserved_field",
                    format!(
                        "node {:?} input field {:?} is reserved by late schema version 1",
                        node.id, field.name
                    ),
                ));
            }
        }
    }
    if mode == CompileMode::Batch {
        issues.push(issue(
            format!("{base}.operator.spec.late_policy"), "unsupported_mode",
            "late side output requires stream mode; execution remains disabled until dual-output runtime and recovery validation is complete",
        ));
    }
}
