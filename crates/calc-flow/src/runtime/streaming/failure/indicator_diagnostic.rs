//! Payload-free numerical evidence from a concrete layout-five Rolling owner.

use crate::{CalcFlowError, OperatorMetadata, RollingOperator, RollingOutputSpec};

#[path = "indicator_diagnostic/reason.rs"]
mod reason;
use reason::Reason;
#[path = "indicator_diagnostic/coordinate.rs"]
mod coordinate;
use coordinate::Coordinate;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct SafeIndicatorDiagnostic {
    node_id: String,
    field: String,
    coordinate: Coordinate,
    reason: Reason,
}

struct Candidate<'a> {
    node_id: &'a str,
    source_node: &'a str,
    output: &'a RollingOutputSpec,
    error: &'a CalcFlowError,
}

impl SafeIndicatorDiagnostic {
    pub(crate) fn from_rolling(
        node_id: &str,
        operator: &RollingOperator,
        error: &CalcFlowError,
    ) -> Option<Self> {
        if operator.spec().state_layout_version != crate::ROLLING_INDICATOR_STATE_LAYOUT_VERSION {
            return None;
        }
        operator.spec().outputs.iter().find_map(|output| {
            [node_id, operator.name()]
                .into_iter()
                .find_map(|source_node| {
                    Candidate {
                        node_id,
                        source_node,
                        output,
                        error,
                    }
                    .diagnostic()
                })
        })
    }

    pub(crate) fn node_id(&self) -> &str {
        &self.node_id
    }

    pub(crate) fn message(&self) -> String {
        format!(
            "operator {:?} execution failed: {:?}: {}: {}",
            self.node_id,
            self.field,
            self.coordinate.text(),
            self.reason.text()
        )
    }
}

impl<'a> Candidate<'a> {
    fn diagnostic(&self) -> Option<SafeIndicatorDiagnostic> {
        let diagnostic = match self.output {
            RollingOutputSpec::Rsi { input, .. } => self.field(input),
            RollingOutputSpec::Atr {
                high, low, close, ..
            } => [high, low, close]
                .into_iter()
                .find_map(|field| self.field(field)),
            _ => return None,
        };
        diagnostic.or_else(|| self.counter())
    }
    fn field(&self, field: &str) -> Option<SafeIndicatorDiagnostic> {
        let (coordinate, text) = self.detail(field)?;
        let reason = Reason::field(self.output, field, text)?;
        Some(self.finish(field, coordinate, reason))
    }
    fn counter(&self) -> Option<SafeIndicatorDiagnostic> {
        let field = "input.rows";
        let (coordinate, text) = self.detail(field)?;
        let reason = Reason::counter(text)?;
        Some(self.finish(field, coordinate, reason))
    }
    fn finish(
        &self,
        field: &str,
        coordinate: Coordinate,
        reason: Reason,
    ) -> SafeIndicatorDiagnostic {
        SafeIndicatorDiagnostic {
            node_id: self.node_id.into(),
            field: field.into(),
            coordinate,
            reason,
        }
    }
    fn detail(&self, field: &str) -> Option<(Coordinate, &'a str)> {
        let raw = raw_detail(self.source_node, field, self.error)?;
        let context = format!("rolling node {:?}, ", self.source_node);
        Coordinate::parse(raw.strip_prefix(&context)?)
    }
}

fn raw_detail<'a>(node: &str, field: &str, error: &'a CalcFlowError) -> Option<&'a str> {
    match error {
        CalcFlowError::InvalidArgument {
            field: actual,
            message,
        } if actual == field => Some(message),
        CalcFlowError::Operator { node_id, message } if node_id == node => {
            message.strip_prefix(&format!("{field}: "))
        }
        _ => None,
    }
}

#[cfg(test)]
#[path = "indicator_diagnostic/tests.rs"]
pub(crate) mod tests;
