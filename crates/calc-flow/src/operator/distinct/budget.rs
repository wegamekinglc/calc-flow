//! Convert native DISTINCT quota failures into payload-free typed reasons.
use crate::CalcFlowError;

const ROLES: &[&str] = &[
    "history",
    "normalization/schema",
    "combined query input",
    "combined metadata",
    "candidate history",
    "prepared query result",
    "stream control",
    "declaration schema/metadata",
    "checkpoint control/identity",
    "checkpoint IPC",
    "checkpoint metadata",
    "restore control/identity",
    "restore IPC decode",
    "restore metadata",
    "restored live captures",
];

pub(super) fn detail(error: &CalcFlowError, node: &str) -> Option<String> {
    match error {
        CalcFlowError::InvalidArgument { field, message }
            if field == "distinct.spec.state_budget"
                || field == &format!("operators.{node}.distinct.spec.state_budget") =>
        {
            reservation(message)
        }
        CalcFlowError::DataFusion { message, .. } => workspace(message),
        _ => None,
    }
}

fn reservation(message: &str) -> Option<String> {
    let message = message.strip_prefix("state_budget_exceeded: copy_visible_v2 ")?;
    let (role, numbers) = message.split_once(" required=")?;
    if !ROLES.contains(&role) {
        return None;
    }
    let (required, limit, unit) =
        if let Some((required, limit)) = numbers.split_once(" rows limit=") {
            (required, limit, "rows")
        } else {
            let (required, limit) = numbers.split_once(" bytes limit=")?;
            (required, limit, "bytes")
        };
    let required = required.parse::<u128>().ok()?;
    let limit = limit
        .split_once(": ")
        .map_or(limit, |(n, _)| n)
        .parse::<u128>()
        .ok()?;
    Some(render(role, required, unit, limit))
}

fn workspace(message: &str) -> Option<String> {
    let (_, suffix) = message.split_once("distinct copy_visible_v2 owned peak exceeded at ")?;
    let (consumer, numbers) = suffix.rsplit_once(": ")?;
    let (used, remaining) = numbers.split_once(" + ")?;
    let (additional, limit) = remaining.split_once(" bytes > ")?;
    let required = used
        .parse::<u128>()
        .ok()?
        .checked_add(additional.parse().ok()?)?;
    let limit = limit.parse().ok()?;
    let role = if consumer == "distinct.metadata" {
        "normalization metadata"
    } else {
        consumer
            .strip_prefix("distinct copy_visible_v2 ")
            .filter(|role| ROLES.contains(role))
            .unwrap_or("query workspace")
    };
    Some(render(role, required, "bytes", limit))
}

fn render(role: &str, required: u128, unit: &str, limit: u128) -> String {
    format!(
        "state_budget_exceeded: copy_visible_v2 {role} required={required} {unit} limit={limit}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_distinct_budget_projection_rejects_payload_and_keeps_numeric_roles() {
        let error = CalcFlowError::InvalidArgument {
            field: "operators.n.distinct.spec.state_budget".into(),
            message: "state_budget_exceeded: copy_visible_v2 history required=2 rows limit=1"
                .into(),
        };
        assert_eq!(
            detail(&error, "n"),
            Some("state_budget_exceeded: copy_visible_v2 history required=2 rows limit=1".into())
        );
        assert!(
            reservation("state_budget_exceeded: copy_visible_v2 secret required=2 rows limit=1")
                .is_none()
        );
        assert!(
            reservation(
                "state_budget_exceeded: copy_visible_v2 history required=2 rows limit=secret"
            )
            .is_none()
        );
        assert_eq!(
            workspace(
                "Resources exhausted: distinct copy_visible_v2 owned peak exceeded at sensitive-consumer-name: 10 + 20 bytes > 29"
            ),
            Some(
                "state_budget_exceeded: copy_visible_v2 query workspace required=30 bytes limit=29"
                    .into()
            )
        );
    }
}
