use super::{ColumnarState, StreamAsofJoinStatus, mismatch};
use crate::{EventTime, IngressProgressSnapshot, IngressState, Result};

pub(super) fn validate_counters(
    metrics: &StreamAsofJoinStatus,
    terminal: bool,
    sequence: u64,
) -> Result<()> {
    if !counter_sums_match(metrics)
        || metrics.identity_only_rows > metrics.evicted_right_rows
        || sequence != metrics.emitted_left_rows
        || (terminal && metrics.state_rows != 0)
    {
        return Err(mismatch(
            "ASOF output sequence, counters or terminal state contradict retained rows",
        ));
    }
    validate_owned_progress(metrics)
}

fn counter_sums_match(metrics: &StreamAsofJoinStatus) -> bool {
    metrics.matched_rows.checked_add(metrics.unmatched_rows) == Some(metrics.emitted_left_rows)
        && metrics
            .emitted_left_rows
            .checked_add(metrics.pending_left_rows)
            == Some(metrics.left.accepted_rows)
        && metrics
            .evicted_right_rows
            .checked_add(metrics.retained_right_rows)
            == Some(metrics.right.accepted_rows)
}

fn validate_owned_progress(metrics: &StreamAsofJoinStatus) -> Result<()> {
    if [&metrics.left, &metrics.right]
        .into_iter()
        .any(|side| side.watermark_micros.is_some() || side.idle || side.ended)
        || metrics.output_watermark_micros.is_some()
    {
        return Err(mismatch("ASOF checkpoint must not own runtime progress"));
    }
    Ok(())
}

pub(super) fn validate_progress(
    state: &ColumnarState,
    tolerance: u64,
    terminal: bool,
    progress: &IngressProgressSnapshot,
    output_frontier: Option<EventTime>,
) -> Result<()> {
    validate_ingress(progress, terminal)?;
    validate_output_frontier(state, progress, output_frontier)?;
    validate_pending_finality(state, terminal, progress, output_frontier)?;
    validate_identity_payloads(state, tolerance, progress)?;
    validate_expired_identities(state, progress)?;
    Ok(())
}

fn validate_ingress(progress: &IngressProgressSnapshot, terminal: bool) -> Result<()> {
    if progress.by_ingress().len() != 2
        || progress.get("left").is_none()
        || progress.get("right").is_none()
    {
        return Err(mismatch("ASOF restore requires exact two-ingress progress"));
    }
    if terminal != super::super::all_ended(progress) {
        return Err(mismatch("ASOF terminal state contradicts ingress EOF"));
    }
    Ok(())
}

fn validate_output_frontier(
    state: &ColumnarState,
    progress: &IngressProgressSnapshot,
    output_frontier: Option<EventTime>,
) -> Result<()> {
    if state
        .pending_times()
        .any(|time| output_frontier.is_some_and(|frontier| time <= frontier.as_micros()))
    {
        return Err(mismatch("ASOF pending row is behind output frontier"));
    }
    if let (Some(output), Some(input)) = (output_frontier, super::super::frontier(progress))
        && output.as_micros() >= input
    {
        return Err(mismatch(
            "ASOF output frontier is ahead of safe input progress",
        ));
    }
    Ok(())
}

fn validate_pending_finality(
    state: &ColumnarState,
    terminal: bool,
    progress: &IngressProgressSnapshot,
    output_frontier: Option<EventTime>,
) -> Result<()> {
    let input = super::super::frontier(progress);
    if state
        .pending_times()
        .any(|time| input.is_some_and(|frontier| time < frontier))
    {
        return Err(mismatch(
            "ASOF snapshot contains already-finalizable pending left rows",
        ));
    }
    if !terminal && input.is_none() && output_frontier.is_some() {
        return Err(mismatch(
            "ASOF output frontier has no established input progress",
        ));
    }
    Ok(())
}

fn retention_threshold(state: &ColumnarState, progress: &IngressProgressSnapshot) -> i128 {
    let left = progress
        .get("left")
        .expect("validated two-ingress progress");
    let future = if left.state() == IngressState::Ended {
        i128::MAX
    } else {
        left.watermark()
            .map_or(i128::MIN, |wm| i128::from(wm.as_micros()))
    };
    let pending = state.pending_times().min().map_or(i128::MAX, i128::from);
    future.min(pending)
}

fn validate_identity_payloads(
    state: &ColumnarState,
    tolerance: u64,
    progress: &IngressProgressSnapshot,
) -> Result<()> {
    let threshold = retention_threshold(state, progress);
    if state
        .right_identities(threshold)
        .any(|(time, has_payload, dominated)| {
            !has_payload && !dominated && i128::from(time) + i128::from(tolerance) >= threshold
        })
    {
        return Err(mismatch(
            "ASOF identity-only state discarded a potentially matching payload",
        ));
    }
    Ok(())
}

fn validate_expired_identities(
    state: &ColumnarState,
    progress: &IngressProgressSnapshot,
) -> Result<()> {
    let right = progress
        .get("right")
        .expect("validated two-ingress progress");
    if state
        .right_identities(i128::MIN)
        .any(|(time, has_payload, _)| {
            !has_payload
                && (right.state() == IngressState::Ended
                    || right.watermark().is_some_and(|wm| time < wm.as_micros()))
        })
    {
        return Err(mismatch(
            "ASOF identity-only state outlived its closed ingress boundary",
        ));
    }
    Ok(())
}
