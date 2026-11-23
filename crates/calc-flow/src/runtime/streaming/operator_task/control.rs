//! Synchronous control decisions shared by runner tasks and session drives.

use super::OperatorInputProgress;
use crate::{
    EventTime, IngressProgressSnapshot, Result,
    pipeline::CompiledStreamOperator,
    runtime::streaming::progress::aggregate::{AggregateInput, ProgressEmissionKind},
};

pub(super) struct DataTransition {
    pub(super) input_watermark: Option<EventTime>,
    pub(super) ingress_progress: IngressProgressSnapshot,
}

pub(super) struct ProgressTransition {
    pub(super) emissions: Vec<ProgressEmissionKind>,
    pub(super) previous_watermark: Option<EventTime>,
    pub(super) ingress_progress: IngressProgressSnapshot,
    pub(super) aggregate_input_frontier: Option<EventTime>,
}

pub(super) fn evaluate_data(
    progress: &mut OperatorInputProgress,
    ingress: &str,
) -> Result<DataTransition> {
    let input_watermark = progress.input_watermark();
    progress.evaluate(ingress, AggregateInput::Data)?;
    Ok(DataTransition {
        input_watermark,
        ingress_progress: progress.snapshot(),
    })
}

pub(super) fn evaluate_progress(
    progress: &mut OperatorInputProgress,
    ingress: &str,
    input: AggregateInput,
) -> Result<ProgressTransition> {
    let previous_watermark = progress.input_watermark();
    let emissions = progress.evaluate(ingress, input)?;
    Ok(ProgressTransition {
        emissions,
        previous_watermark,
        ingress_progress: progress.snapshot(),
        aggregate_input_frontier: progress.input_watermark(),
    })
}

/// Computes the candidate after applying the common all-late and monotonicity
/// rules. Join-specific candidates remain owned by the operator implementation.
pub(super) fn advancing_output_frontier(
    operator: &CompiledStreamOperator,
    has_non_late_output: bool,
    aggregate_input_frontier: Option<EventTime>,
    ingress_progress: &IngressProgressSnapshot,
    previous: Option<EventTime>,
) -> Result<Option<EventTime>> {
    if !has_non_late_output {
        return Ok(None);
    }
    Ok(operator
        .output_frontier_candidate(aggregate_input_frontier, ingress_progress)?
        .filter(|candidate| previous.is_none_or(|previous| *candidate > previous)))
}

pub(super) fn forwards_control(late: bool, message: &crate::StreamMessage) -> bool {
    !late || (message.as_watermark().is_none() && !message.is_idle())
}
