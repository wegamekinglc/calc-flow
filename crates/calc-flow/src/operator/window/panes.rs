use std::{collections::BTreeMap, ops::Range, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{Batch, Epoch, EventTime, OperatorStateSnapshot, Result, StateSegment};

use super::{
    AccumulatorRow, AccumulatorValue, AggregateFunction, AssignmentPlan, CompiledWindowGeometry,
    LateMetricDelta, PreparedInputMetrics, RecordColumns, ScalarValue, StateOperationRow,
    WindowAggregateOperator, WindowGeometry, WindowInputContext, WindowKey, WindowSpec,
    WindowState, assignments_from_latest, checkpoint_mismatch, deserialize_required_option,
    encode_group_key, encode_state_segment_for_version, entry_bytes_of_parts, format_error,
    internal_error, latest_window_start, logical_length, new_accumulator_row, operator_error,
    plan_assignments, record_null_event_time, record_planned_late, scalar_total_cmp, state_format,
    state_schema_for_version, update_accumulators, window_entry_bytes,
};

pub(super) const WINDOW_PANE_STATE_LAYOUT_VERSION: u32 = 3;
const PANE_SEGMENT_ID: &str = "panes";
const OPEN_WINDOW_SEGMENT_ID: &str = "open-windows";

#[derive(Clone, Default)]
pub(super) struct PaneState {
    pub(super) entries: BTreeMap<WindowKey, AccumulatorRow>,
    pub(super) open_windows: BTreeMap<WindowKey, Arc<[Option<ScalarValue>]>>,
    pub(super) physical_bytes: u64,
}

#[derive(Clone, Copy)]
pub(super) struct PaneGeometry {
    geometry: CompiledWindowGeometry,
}

impl PaneGeometry {
    pub(super) const fn new(geometry: CompiledWindowGeometry) -> Self {
        Self { geometry }
    }

    pub(super) fn index(self, event_time: EventTime) -> i128 {
        i128::from(event_time.as_micros()).div_euclid(i128::from(self.geometry.slide_micros))
    }

    pub(super) fn bounds(self, index: i128) -> std::result::Result<(EventTime, EventTime), String> {
        let slide = i128::from(self.geometry.slide_micros);
        let start = index
            .checked_mul(slide)
            .ok_or_else(|| "pane start overflowed".to_string())?;
        let end = start
            .checked_add(slide)
            .ok_or_else(|| "pane end overflowed".to_string())?;
        let start =
            i64::try_from(start).map_err(|_| "pane start is outside EventTime".to_string())?;
        let end = i64::try_from(end).map_err(|_| "pane end is outside EventTime".to_string())?;
        Ok((EventTime::from_micros(start), EventTime::from_micros(end)))
    }

    pub(super) fn window_indices(self, window_start: EventTime) -> Range<i128> {
        let first = self.index(window_start);
        first..first + i128::from(self.geometry.overlap)
    }

    fn key(
        self,
        index: i128,
        stable_group_key: Arc<[u8]>,
    ) -> std::result::Result<WindowKey, String> {
        let (start, end) = self.bounds(index)?;
        Ok(WindowKey {
            start,
            end,
            stable_group_key,
        })
    }
}

pub(super) fn enabled(spec: &WindowSpec) -> bool {
    matches!(spec.geometry, WindowGeometry::Hopping { .. })
        && spec.aggregates.iter().all(|aggregate| {
            matches!(
                aggregate.function,
                AggregateFunction::Count | AggregateFunction::Min | AggregateFunction::Max
            )
        })
}

pub(super) struct PaneInputUpdate {
    pub(super) state: PaneState,
    pub(super) metrics: LateMetricDelta,
}

#[allow(
    clippy::too_many_lines,
    reason = "one input transaction preserves row-order errors and installs state only after the full batch"
)]
pub(super) fn prepare_window_input(
    operator: &WindowAggregateOperator,
    batch: &Batch,
    context: &WindowInputContext<'_>,
) -> Result<PaneInputUpdate> {
    if operator.state.ended {
        return Err(operator_error(
            context.operator_id(),
            "received data after end-of-input",
        ));
    }
    let geometry = PaneGeometry::new(operator.compiled.geometry);
    let mut state = operator
        .state
        .panes
        .as_ref()
        .expect("pane input requires pane state")
        .clone();
    let mut metrics = PreparedInputMetrics::default();
    let mut assignment_cache = None;
    let mut assignment_plan = AssignmentPlan::default();
    let mut encoded_group = Vec::new();

    for record in batch.table_payload()?.batches() {
        context.check_cancelled()?;
        let columns = RecordColumns::new(
            record,
            &operator.spec,
            &operator.compiled,
            context.operator_id(),
        )?;
        for row in 0..record.num_rows() {
            let Some(event_time) = columns.event_time.at(
                row,
                context.operator_id(),
                &operator.spec.event_time_column,
            )?
            else {
                record_null_event_time(&mut metrics, context.operator_id())?;
                continue;
            };
            let latest_start = latest_window_start(
                event_time,
                operator.compiled.geometry,
                &mut assignment_cache,
            );
            plan_assignments(
                &mut assignment_plan,
                latest_start,
                operator.compiled.geometry,
                context.input_watermark(),
                context.operator_id(),
            )?;
            record_planned_late(&mut metrics, &assignment_plan, context.operator_id())?;
            if assignment_plan.open.is_empty() {
                continue;
            }

            encode_group_key(
                &columns,
                row,
                context.operator_id(),
                &operator.spec.group_by,
                &mut encoded_group,
            )?;
            let stable_group_key = Arc::<[u8]>::from(encoded_group.as_slice());
            let group_values: Arc<[Option<ScalarValue>]> = columns
                .groups
                .iter()
                .map(|(column, _)| column.scalar_at(row))
                .collect::<Vec<_>>()
                .into();
            for &(start, end) in &assignment_plan.open {
                state
                    .open_windows
                    .entry(WindowKey {
                        start,
                        end,
                        stable_group_key: Arc::clone(&stable_group_key),
                    })
                    .or_insert_with(|| Arc::clone(&group_values));
            }

            let pane_key = geometry
                .key(geometry.index(event_time), stable_group_key)
                .map_err(|message| operator_error(context.operator_id(), &message))?;
            let accumulator = state.entries.entry(pane_key).or_insert_with(|| {
                new_accumulator_row(
                    &operator.spec,
                    &operator.compiled,
                    Arc::clone(&group_values),
                )
            });
            update_accumulators(
                accumulator,
                &columns,
                row,
                &operator.spec,
                context.operator_id(),
            )?;
        }
    }

    state.physical_bytes = physical_bytes(&state);
    if !operator.state_budget.allows(
        logical_length(state.open_windows.len()),
        state.physical_bytes,
    ) {
        return Err(operator_error(
            context.operator_id(),
            "window state budget exceeded",
        ));
    }
    Ok(PaneInputUpdate {
        state,
        metrics: metrics.into_delta(),
    })
}

pub(super) fn merge_accumulator(
    target: &mut AccumulatorValue,
    source: &AccumulatorValue,
) -> std::result::Result<(), String> {
    match (target, source) {
        (AccumulatorValue::Count(target), AccumulatorValue::Count(source)) => {
            let merged = target
                .checked_add(*source)
                .ok_or_else(|| "count overflowed UInt64".to_string())?;
            *target = merged;
            Ok(())
        }
        (AccumulatorValue::Min(target), AccumulatorValue::Min(source)) => {
            merge_extreme(target, source.as_ref(), std::cmp::Ordering::Less);
            Ok(())
        }
        (AccumulatorValue::Max(target), AccumulatorValue::Max(source)) => {
            merge_extreme(target, source.as_ref(), std::cmp::Ordering::Greater);
            Ok(())
        }
        _ => Err("compiled aggregate accumulator type mismatch".into()),
    }
}

fn merge_extreme(
    target: &mut Option<ScalarValue>,
    source: Option<&ScalarValue>,
    ordering: std::cmp::Ordering,
) {
    let Some(source) = source else {
        return;
    };
    if target
        .as_ref()
        .is_none_or(|target| scalar_total_cmp(source, target) == ordering)
    {
        *target = Some((*source).clone());
    }
}

pub(super) struct PreparedPaneClose {
    pub(super) state: PaneState,
    pub(super) closed: BTreeMap<WindowKey, AccumulatorRow>,
}

pub(super) fn prepare_close(
    operator: &WindowAggregateOperator,
    watermark: Option<EventTime>,
) -> Result<PreparedPaneClose> {
    let current = operator
        .state
        .panes
        .as_ref()
        .expect("pane close requires pane state");
    let geometry = PaneGeometry::new(operator.compiled.geometry);
    let mut closed = BTreeMap::new();
    for (key, group_values) in &current.open_windows {
        if watermark.is_some_and(|watermark| key.end > watermark) {
            continue;
        }
        let mut merged =
            new_accumulator_row(&operator.spec, &operator.compiled, Arc::clone(group_values));
        for index in geometry.window_indices(key.start) {
            let pane_key = geometry
                .key(index, Arc::clone(&key.stable_group_key))
                .map_err(|message| operator_error(operator.name.as_str(), &message))?;
            let Some(pane) = current.entries.get(&pane_key) else {
                continue;
            };
            for (ordinal, (target, source)) in merged
                .aggregates
                .iter_mut()
                .zip(&pane.aggregates)
                .enumerate()
            {
                merge_accumulator(target, source).map_err(|message| {
                    operator_error(
                        operator.name.as_str(),
                        &format!("window.aggregates[{ordinal}] merge failed: {message}"),
                    )
                })?;
            }
        }
        closed.insert(key.clone(), merged);
    }

    let scratch_bytes = closed
        .iter()
        .map(|(key, entry)| window_entry_bytes(logical_length(key.stable_group_key.len()), entry))
        .fold(0_u64, u64::saturating_add);
    if !operator.state_budget.allows(
        logical_length(current.open_windows.len()),
        current.physical_bytes.saturating_add(scratch_bytes),
    ) {
        return Err(operator_error(
            operator.name.as_str(),
            "window state budget exceeded",
        ));
    }

    let mut state = current.clone();
    for key in closed.keys() {
        state.open_windows.remove(key);
    }
    state.entries.retain(|pane, _| {
        watermark.is_none_or(|watermark| {
            i128::from(pane.start.as_micros()) + i128::from(operator.compiled.geometry.size_micros)
                > i128::from(watermark.as_micros())
        })
    });
    state.physical_bytes = physical_bytes(&state);
    if !operator.state_budget.allows(
        logical_length(state.open_windows.len()),
        state.physical_bytes,
    ) {
        return Err(operator_error(
            operator.name.as_str(),
            "window state budget exceeded",
        ));
    }
    Ok(PreparedPaneClose { state, closed })
}

fn physical_bytes(state: &PaneState) -> u64 {
    let panes = state
        .entries
        .iter()
        .map(|(key, entry)| window_entry_bytes(logical_length(key.stable_group_key.len()), entry))
        .fold(0_u64, u64::saturating_add);
    state
        .open_windows
        .iter()
        .map(|(key, group_values)| {
            entry_bytes_of_parts(
                logical_length(key.stable_group_key.len()),
                group_values,
                0,
                0,
            )
        })
        .fold(panes, u64::saturating_add)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SegmentProof {
    byte_len: u64,
    sha256: String,
}

impl SegmentProof {
    fn new(segment: &StateSegment) -> Result<Self> {
        Ok(Self {
            byte_len: u64::try_from(segment.bytes().len())
                .map_err(|_| internal_error("window pane segment length does not fit u64"))?,
            sha256: segment.sha256().to_owned(),
        })
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PaneSnapshotMetadata {
    state_layout_version: u32,
    configuration_hash: String,
    state_schema_fingerprint: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    pipeline_fingerprint: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    operator_id: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    last_input_watermark: Option<EventTime>,
    next_output_sequence: u64,
    ended: bool,
    metrics: LateMetricDelta,
    #[serde(deserialize_with = "deserialize_required_option")]
    panes: Option<SegmentProof>,
    #[serde(deserialize_with = "deserialize_required_option")]
    open_windows: Option<SegmentProof>,
}

pub(super) fn checkpoint(
    operator: &mut WindowAggregateOperator,
    epoch: Epoch,
) -> Result<OperatorStateSnapshot> {
    let state = operator
        .state
        .panes
        .as_ref()
        .expect("pane checkpoint requires pane state");
    let has_state = !state.entries.is_empty() || !state.open_windows.is_empty();
    let (panes, open_windows, segments) = if has_state {
        if state.entries.is_empty() || state.open_windows.is_empty() {
            return Err(internal_error(
                "window panes and open windows must be nonempty together",
            ));
        }
        let pipeline_fingerprint = operator.state_pipeline_fingerprint()?;
        let operator_id = operator.state_operator_id()?;
        let pane_operations = state
            .entries
            .iter()
            .map(|(key, entry)| StateOperationRow {
                key: key.clone(),
                entry: entry.clone(),
                tombstone: false,
            })
            .collect::<Vec<_>>();
        let open_operations = state
            .open_windows
            .iter()
            .map(|(key, group_values)| StateOperationRow {
                key: key.clone(),
                entry: new_accumulator_row(
                    &operator.spec,
                    &operator.compiled,
                    Arc::clone(group_values),
                ),
                tombstone: false,
            })
            .collect::<Vec<_>>();
        let pane_segment = StateSegment::new(encode_state_segment_for_version(
            &pane_operations,
            &operator.spec,
            &operator.compiled,
            pipeline_fingerprint,
            operator_id,
            WINDOW_PANE_STATE_LAYOUT_VERSION,
        )?);
        let open_segment = StateSegment::new(encode_state_segment_for_version(
            &open_operations,
            &operator.spec,
            &operator.compiled,
            pipeline_fingerprint,
            operator_id,
            WINDOW_PANE_STATE_LAYOUT_VERSION,
        )?);
        let panes = SegmentProof::new(&pane_segment)?;
        let open_windows = SegmentProof::new(&open_segment)?;
        (
            Some(panes),
            Some(open_windows),
            BTreeMap::from([
                (PANE_SEGMENT_ID.into(), pane_segment),
                (OPEN_WINDOW_SEGMENT_ID.into(), open_segment),
            ]),
        )
    } else {
        (None, None, BTreeMap::new())
    };
    let metadata = PaneSnapshotMetadata {
        state_layout_version: WINDOW_PANE_STATE_LAYOUT_VERSION,
        configuration_hash: operator.compiled.configuration_hash.clone(),
        state_schema_fingerprint: operator.compiled.state_schema_fingerprint.clone(),
        pipeline_fingerprint: operator.state.pipeline_fingerprint.clone(),
        operator_id: operator.state.operator_id.clone(),
        last_input_watermark: operator.state.last_input_watermark,
        next_output_sequence: operator.state.next_output_sequence,
        ended: operator.state.ended,
        metrics: operator.state.metrics,
        panes,
        open_windows,
    };
    let Value::Object(inline_metadata) =
        serde_json::to_value(metadata).map_err(|error| format_error(&error))?
    else {
        return Err(internal_error(
            "window pane snapshot metadata did not serialize as an object",
        ));
    };
    operator.state.prepared_segments.clear();
    operator.state.checkpoint_marks.clear();
    operator.state.retained_segments = segments.clone();
    operator.state.replace_retained_on_checkpoint = false;
    operator.state.last_checkpoint_epoch = Some(epoch);
    Ok(OperatorStateSnapshot {
        inline_metadata: inline_metadata.into_iter().collect(),
        segments,
    })
}

pub(super) fn restore(
    operator: &mut WindowAggregateOperator,
    snapshot: &OperatorStateSnapshot,
) -> Result<()> {
    let metadata = serde_json::from_value::<PaneSnapshotMetadata>(Value::Object(
        snapshot.inline_metadata.clone().into_iter().collect(),
    ))
    .map_err(|error| format_error(&error))?;
    validate_metadata(operator, &metadata)?;
    validate_segment_set(snapshot, &metadata)?;
    let state = decode_state(operator, snapshot, &metadata)?;
    let physical_bytes = physical_bytes(&state);
    if !operator
        .state_budget
        .allows(logical_length(state.open_windows.len()), physical_bytes)
    {
        return Err(checkpoint_mismatch("window state budget exceeded"));
    }
    operator.state = WindowState {
        accumulator_bytes: physical_bytes,
        panes: Some(state),
        last_input_watermark: metadata.last_input_watermark,
        next_output_sequence: metadata.next_output_sequence,
        ended: metadata.ended,
        metrics: metadata.metrics,
        retained_segments: snapshot.segments.clone(),
        pipeline_fingerprint: metadata.pipeline_fingerprint,
        operator_id: metadata.operator_id,
        ..WindowState::default()
    };
    Ok(())
}

fn validate_metadata(
    operator: &WindowAggregateOperator,
    metadata: &PaneSnapshotMetadata,
) -> Result<()> {
    if metadata.state_layout_version != WINDOW_PANE_STATE_LAYOUT_VERSION {
        return Err(checkpoint_mismatch(format!(
            "window pane state layout version {} is unsupported",
            metadata.state_layout_version
        )));
    }
    if metadata.configuration_hash != operator.compiled.configuration_hash {
        return Err(checkpoint_mismatch(
            "window operator configuration hash does not match the compiled operator",
        ));
    }
    if metadata.state_schema_fingerprint != operator.compiled.state_schema_fingerprint {
        return Err(checkpoint_mismatch(
            "window state schema fingerprint does not match the compiled operator",
        ));
    }
    if metadata.panes.is_some() != metadata.open_windows.is_some() {
        return Err(checkpoint_mismatch(
            "window pane snapshot must contain both pane and open-window segments",
        ));
    }
    if metadata.panes.is_some()
        && (metadata.pipeline_fingerprint.is_none() || metadata.operator_id.is_none())
    {
        return Err(checkpoint_mismatch(
            "window pane segments require pipeline and operator identity metadata",
        ));
    }
    validate_identity(
        metadata.pipeline_fingerprint.as_deref(),
        metadata.operator_id.as_deref(),
    )?;
    for proof in [&metadata.panes, &metadata.open_windows]
        .into_iter()
        .flatten()
    {
        if hex::decode(&proof.sha256)
            .ok()
            .is_none_or(|digest| digest.len() != 32)
        {
            return Err(checkpoint_mismatch(
                "window pane segment checksum is not SHA-256",
            ));
        }
    }
    Ok(())
}

fn validate_identity(pipeline_fingerprint: Option<&str>, operator_id: Option<&str>) -> Result<()> {
    if let Some(fingerprint) = pipeline_fingerprint
        && (fingerprint.len() != 64
            || !fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        return Err(checkpoint_mismatch(
            "window pipeline fingerprint is not lowercase SHA-256",
        ));
    }
    if operator_id.is_some_and(|operator_id| operator_id.is_empty() || operator_id.contains('\0')) {
        return Err(checkpoint_mismatch(
            "window operator ID is empty or contains NUL",
        ));
    }
    Ok(())
}

fn validate_segment_set(
    snapshot: &OperatorStateSnapshot,
    metadata: &PaneSnapshotMetadata,
) -> Result<()> {
    let expected = if metadata.panes.is_some() {
        vec![OPEN_WINDOW_SEGMENT_ID, PANE_SEGMENT_ID]
    } else {
        Vec::new()
    };
    let actual = snapshot
        .segments
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if actual != expected {
        return Err(checkpoint_mismatch(
            "window pane snapshot segments are missing, extra, or non-canonical",
        ));
    }
    for (segment_id, proof) in [
        (PANE_SEGMENT_ID, metadata.panes.as_ref()),
        (OPEN_WINDOW_SEGMENT_ID, metadata.open_windows.as_ref()),
    ] {
        let Some(proof) = proof else {
            continue;
        };
        let bytes = snapshot.segments[segment_id].bytes();
        if u64::try_from(bytes.len()).ok() != Some(proof.byte_len)
            || hex::encode(Sha256::digest(bytes)) != proof.sha256
        {
            return Err(checkpoint_mismatch(
                "window pane snapshot segment does not match its length or checksum",
            ));
        }
    }
    Ok(())
}

fn decode_state(
    operator: &WindowAggregateOperator,
    snapshot: &OperatorStateSnapshot,
    metadata: &PaneSnapshotMetadata,
) -> Result<PaneState> {
    if metadata.panes.is_none() {
        return Ok(PaneState::default());
    }
    let pipeline_fingerprint = metadata.pipeline_fingerprint.as_deref().ok_or_else(|| {
        checkpoint_mismatch("window pane state is missing its pipeline fingerprint")
    })?;
    let operator_id = metadata
        .operator_id
        .as_deref()
        .ok_or_else(|| checkpoint_mismatch("window pane state is missing its operator ID"))?;
    let expected_schema = state_schema_for_version(
        &operator.spec,
        &operator.compiled,
        pipeline_fingerprint,
        operator_id,
        WINDOW_PANE_STATE_LAYOUT_VERSION,
    );
    let mut pane_compiled = operator.compiled.clone();
    pane_compiled.geometry = CompiledWindowGeometry {
        size_micros: operator.compiled.geometry.slide_micros,
        slide_micros: operator.compiled.geometry.slide_micros,
        overlap: 1,
    };
    let pane_operations = super::recovery::decode_state_segment(
        snapshot.segments[PANE_SEGMENT_ID].bytes(),
        &operator.spec,
        &pane_compiled,
        &expected_schema,
        operator_id,
        WINDOW_PANE_STATE_LAYOUT_VERSION,
    )?;
    let open_operations = super::recovery::decode_state_segment(
        snapshot.segments[OPEN_WINDOW_SEGMENT_ID].bytes(),
        &operator.spec,
        &operator.compiled,
        &expected_schema,
        operator_id,
        WINDOW_PANE_STATE_LAYOUT_VERSION,
    )?;
    let entries = pane_operations
        .into_iter()
        .map(|(key, entry)| {
            entry
                .map(|entry| (key, entry))
                .ok_or_else(|| state_format("window pane segment contains a tombstone"))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let open_windows = open_operations
        .into_iter()
        .map(|(key, entry)| {
            let entry = entry
                .ok_or_else(|| state_format("window open-window segment contains a tombstone"))?;
            if !entry.aggregates.iter().all(default_accumulator) {
                return Err(state_format(
                    "window open-window segment contains aggregate state",
                ));
            }
            Ok((key, entry.group_values))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let state = PaneState {
        entries,
        open_windows,
        physical_bytes: 0,
    };
    validate_membership(operator, &state)?;
    Ok(PaneState {
        physical_bytes: physical_bytes(&state),
        ..state
    })
}

fn default_accumulator(accumulator: &AccumulatorValue) -> bool {
    match accumulator {
        AccumulatorValue::Count(count) => *count == 0,
        AccumulatorValue::Min(value) | AccumulatorValue::Max(value) => value.is_none(),
        _ => false,
    }
}

fn validate_membership(operator: &WindowAggregateOperator, state: &PaneState) -> Result<()> {
    let geometry = PaneGeometry::new(operator.compiled.geometry);
    for key in state.open_windows.keys() {
        let mut found = false;
        for index in geometry.window_indices(key.start) {
            let pane_key = geometry
                .key(index, Arc::clone(&key.stable_group_key))
                .map_err(checkpoint_mismatch)?;
            if state.entries.contains_key(&pane_key) {
                found = true;
                break;
            }
        }
        if !found {
            return Err(state_format(
                "window open window has no contributing pane state",
            ));
        }
    }
    for pane in state.entries.keys() {
        let assignments = assignments_from_latest(
            i128::from(pane.start.as_micros()),
            operator.compiled.geometry,
        )
        .map_err(checkpoint_mismatch)?;
        let mut found = false;
        for (start, end) in assignments {
            if state.open_windows.contains_key(&WindowKey {
                start,
                end,
                stable_group_key: Arc::clone(&pane.stable_group_key),
            }) {
                found = true;
                break;
            }
        }
        if !found {
            return Err(state_format(
                "window pane state has no contributing open window",
            ));
        }
    }
    Ok(())
}
