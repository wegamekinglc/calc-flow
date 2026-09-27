//! Rolling checkpoint state serialization and typed restoration.

use super::{
    Arc, Array, ArrayRef, BTreeMap, Batch, BufferedRow, CompiledFrame, CompiledRollingSpec,
    CompiledWindowGroup, Cursor, DataType, DecodedRollingState, Deserialize, Deserializer,
    DictionaryTracker, Digest, EntityRollingState, EwmaAccumulator, ExtremaAccumulator, ExtremaKey,
    Field, FileReader, FileWriter, Float64Array, HashMap, HistoryUpdates, IpcSchemaEncoder,
    KeyValue, ROLLING_COLUMNAR_STATE_LAYOUT_VERSION, ROLLING_EWMA_STATE_LAYOUT_VERSION,
    RecordBatch, Result, RollingHistories, RollingKernelState, RollingMetricsRecorder,
    RollingSnapshotMetadata, RollingStage, RowIdentity, ScalarValue, Schema, SchemaRef,
    SegmentDescriptor, Sha256, StateInventory, StateRowOrderKey, SumClass, TableBatch, UInt8Array,
    UInt64Array, Value, WindowState, checkpoint_mismatch, concat_batches, evict_retained_history,
    ewma_entity_values, float_sample, format_error, fresh_windows, group_rows_by_entity,
    history_event_time, internal_error, is_valid_sample, kernel, new_null_array, operator_error,
    spec_uses_stable_v2, state_format, state_v3,
};

fn state_fields(input_schema: &Schema, state_layout_version: u32) -> Vec<Field> {
    if state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION {
        return state_v3::state_fields(input_schema);
    }
    let mut fields = vec![
        Field::new("_state_kind", DataType::UInt8, false),
        Field::new("_entity_position", DataType::UInt64, true),
    ];
    fields.extend(
        input_schema
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), true)),
    );
    if state_layout_version == ROLLING_EWMA_STATE_LAYOUT_VERSION {
        fields.extend([
            Field::new("_ewma_group", DataType::UInt64, true),
            Field::new("_ewma_valid_count", DataType::UInt64, true),
            Field::new("_ewma_value", DataType::Float64, true),
        ]);
    }
    fields
}

pub(super) fn state_schema_fingerprint(input_schema: &Schema, state_layout_version: u32) -> String {
    let schema = Schema::new(state_fields(input_schema, state_layout_version));
    let mut dictionary_tracker = DictionaryTracker::new(true);
    let encoded = IpcSchemaEncoder::new()
        .with_dictionary_tracker(&mut dictionary_tracker)
        .schema_to_fb(&schema);
    hex::encode(Sha256::digest(encoded.finished_data()))
}

pub(super) fn state_schema(
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Schema {
    let mut metadata = HashMap::from([
        (
            "calc_flow.state_layout_version".into(),
            compiled.state_layout_version.to_string(),
        ),
        (
            "calc_flow.pipeline_fingerprint".into(),
            pipeline_fingerprint.into(),
        ),
        ("calc_flow.operator_id".into(), operator_id.into()),
        (
            "calc_flow.operator_configuration_hash".into(),
            compiled.configuration_hash.clone(),
        ),
        (
            "calc_flow.state_schema_fingerprint".into(),
            compiled.state_schema_fingerprint.clone(),
        ),
    ]);
    if compiled.state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION {
        metadata.insert(
            "calc_flow.rolling_kernel_fingerprint".into(),
            compiled.kernel_plan.fingerprint().to_owned(),
        );
        metadata.insert(
            "calc_flow.numerical_profile".into(),
            compiled.kernel_plan.numerical_profile().to_owned(),
        );
    }
    Schema::new_with_metadata(
        state_fields(input_schema, compiled.state_layout_version),
        metadata,
    )
}

// State serialization writes deterministic history and buffer rows column
// by column with checked conversions for every value class.
// #lizard forgives
pub(super) fn encode_state_segment(
    histories: &RollingHistories,
    buffer: &BTreeMap<RowIdentity, BufferedRow>,
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Result<Vec<u8>> {
    if compiled.state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION {
        return state_v3::encode(
            histories,
            buffer,
            input_schema,
            compiled,
            pipeline_fingerprint,
            operator_id,
        );
    }
    encode_state_segment_legacy(
        histories,
        buffer,
        input_schema,
        compiled,
        pipeline_fingerprint,
        operator_id,
    )
}

// Moved from rolling.rs; this codec must preserve legacy checkpoint bytes.
// #lizard forgives
pub(super) fn encode_state_segment_legacy(
    histories: &RollingHistories,
    buffer: &BTreeMap<RowIdentity, BufferedRow>,
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Result<Vec<u8>> {
    let width = input_schema.fields().len();
    let mut kinds = Vec::new();
    let mut positions: Vec<Option<u64>> = Vec::new();
    let mut columns: Vec<Vec<Option<ScalarValue>>> = vec![Vec::new(); width];
    let mut ewma_groups = Vec::new();
    let mut ewma_counts = Vec::new();
    let mut ewma_values = Vec::new();
    let mut push_row =
        |kind: u8, position: Option<u64>, values: &[ScalarValue], ewma: Option<(u64, u64, f64)>| {
            kinds.push(kind);
            positions.push(position);
            for (index, column) in columns.iter_mut().enumerate() {
                column.push(values.get(index).cloned());
            }
            if compiled.state_layout_version == ROLLING_EWMA_STATE_LAYOUT_VERSION {
                ewma_groups.push(ewma.map(|state| state.0));
                ewma_counts.push(ewma.map(|state| state.1));
                ewma_values.push(ewma.map(|state| state.2));
            }
        };
    for state in histories.by_entity.values() {
        for (position, values) in state.rows.iter().enumerate() {
            let position = u64::try_from(position)
                .map_err(|_| internal_error("rolling history position does not fit u64"))?;
            push_row(0, Some(position), values, None);
        }
    }
    for row in buffer.values() {
        push_row(1, None, &row.values, None);
    }
    if compiled.state_layout_version == ROLLING_EWMA_STATE_LAYOUT_VERSION {
        for (entity, state) in &histories.by_entity {
            let values = ewma_entity_values(entity, input_schema, compiled)?;
            for (group, window) in state.windows.iter().enumerate() {
                let WindowState::Ewma(accumulator) = window else {
                    continue;
                };
                if accumulator.valid_count == 0 {
                    continue;
                }
                let group = u64::try_from(group)
                    .map_err(|_| internal_error("rolling EWMA group does not fit u64"))?;
                push_row(
                    2,
                    None,
                    &values,
                    Some((group, accumulator.valid_count, accumulator.value)),
                );
            }
        }
    }
    let schema = state_schema(input_schema, compiled, pipeline_fingerprint, operator_id);
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(UInt8Array::from(kinds)),
        Arc::new(UInt64Array::from(positions)),
    ];
    for column in columns {
        arrays.push(
            ScalarValue::iter_to_array(
                column
                    .into_iter()
                    .map(|value| value.expect("rolling state rows carry full typed values")),
            )
            .map_err(|error| state_format(format!("rolling state array failed: {error}")))?,
        );
    }
    if compiled.state_layout_version == ROLLING_EWMA_STATE_LAYOUT_VERSION {
        arrays.extend([
            Arc::new(UInt64Array::from(ewma_groups)) as ArrayRef,
            Arc::new(UInt64Array::from(ewma_counts)) as ArrayRef,
            Arc::new(Float64Array::from(ewma_values)) as ArrayRef,
        ]);
    }
    let record = RecordBatch::try_new(Arc::new(schema.clone()), arrays)
        .map_err(|error| state_format(format!("rolling state batch is invalid: {error}")))?;
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, &schema)
            .map_err(|error| state_format(format!("rolling state IPC header failed: {error}")))?;
        writer
            .write(&record)
            .map_err(|error| state_format(format!("rolling state IPC write failed: {error}")))?;
        writer
            .finish()
            .map_err(|error| state_format(format!("rolling state IPC finish failed: {error}")))?;
    }
    Ok(bytes)
}

// State decode intentionally validates header metadata, shape, deterministic
// order, and per-row invariants before any state is installed.
// #lizard forgives
fn ewma_state_arrays(
    record: &RecordBatch,
    width: usize,
    enabled: bool,
) -> Result<(
    Option<&UInt64Array>,
    Option<&UInt64Array>,
    Option<&Float64Array>,
)> {
    if !enabled {
        return Ok((None, None, None));
    }
    let group = record
        .column(width + 2)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| state_format("rolling EWMA group column has the wrong type".to_owned()))?;
    let count = record
        .column(width + 3)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| state_format("rolling EWMA count column has the wrong type".to_owned()))?;
    let value = record
        .column(width + 4)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| state_format("rolling EWMA value column has the wrong type".to_owned()))?;
    Ok((Some(group), Some(count), Some(value)))
}

pub(super) fn decode_state_segment(
    bytes: &[u8],
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    metadata: &RollingSnapshotMetadata,
) -> Result<DecodedRollingState> {
    if metadata.state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION {
        return state_v3::decode(bytes, input_schema, compiled, metadata);
    }
    let mut legacy = compiled.clone();
    legacy.state_layout_version = metadata.state_layout_version;
    legacy
        .state_schema_fingerprint
        .clone_from(&metadata.state_schema_fingerprint);
    decode_state_segment_legacy(bytes, input_schema, &legacy, metadata)
}

// Moved from rolling.rs; legacy decoding retains format-specific validation.
// #lizard forgives
fn decode_state_segment_legacy(
    bytes: &[u8],
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    metadata: &RollingSnapshotMetadata,
) -> Result<DecodedRollingState> {
    let reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| state_format(format!("rolling state IPC open failed: {error}")))?;
    validate_segment_schema_metadata(reader.schema().metadata(), metadata, compiled)?;
    let batches = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| state_format(format!("rolling state IPC read failed: {error}")))?;
    let [record] = batches.try_into().map_err(|_| {
        state_format("rolling state segment must contain exactly one record batch".to_owned())
    })?;
    let width = input_schema.fields().len();
    let exponential_width =
        usize::from(compiled.state_layout_version == ROLLING_EWMA_STATE_LAYOUT_VERSION) * 3;
    if record.num_columns() != width + 2 + exponential_width {
        return Err(state_format(
            "rolling state segment column count does not match the state schema".to_owned(),
        ));
    }
    let kinds = record
        .column(0)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| state_format("rolling state kind column has the wrong type".to_owned()))?;
    let positions = record
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            state_format("rolling state position column has the wrong type".to_owned())
        })?;
    let (ewma_groups, ewma_counts, ewma_values) =
        ewma_state_arrays(&record, width, exponential_width > 0)?;
    let mut decoded = DecodedRollingState::default();
    let mut previous: Option<StateRowOrderKey> = None;
    for row_index in 0..record.num_rows() {
        let values = (2..width + 2)
            .map(|index| {
                ScalarValue::try_from_array(record.column(index), row_index).map_err(|error| {
                    state_format(format!("rolling state row could not be read: {error}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let position = (!positions.is_null(row_index)).then(|| positions.value(row_index));
        let ewma = match (ewma_groups, ewma_counts, ewma_values) {
            (Some(groups), Some(counts), Some(values)) => match (
                (!groups.is_null(row_index)).then(|| groups.value(row_index)),
                (!counts.is_null(row_index)).then(|| counts.value(row_index)),
                (!values.is_null(row_index)).then(|| values.value(row_index)),
            ) {
                (Some(group), Some(count), Some(value)) => Some((group, count, value)),
                (None, None, None) => None,
                _ => {
                    return Err(state_format(
                        "rolling EWMA state columns are only partially populated".to_owned(),
                    ));
                }
            },
            (None, None, None) => None,
            _ => unreachable!("EWMA state arrays are discovered together"),
        };
        decode_state_row(
            kinds.value(row_index),
            position,
            values,
            ewma,
            &mut decoded,
            compiled,
            &mut previous,
        )?;
    }
    validate_decoded_state(&decoded, compiled)?;
    rebuild_windows(&mut decoded.histories, compiled, "rolling")?;
    Ok(decoded)
}

pub(super) fn validate_segment_schema_metadata(
    metadata: &HashMap<String, String>,
    snapshot: &RollingSnapshotMetadata,
    _compiled: &CompiledRollingSpec,
) -> Result<()> {
    let mut expected = vec![
        (
            "calc_flow.state_layout_version",
            snapshot.state_layout_version.to_string(),
        ),
        (
            "calc_flow.pipeline_fingerprint",
            snapshot.pipeline_fingerprint.clone().unwrap_or_default(),
        ),
        (
            "calc_flow.operator_id",
            snapshot.operator_id.clone().unwrap_or_default(),
        ),
        (
            "calc_flow.operator_configuration_hash",
            snapshot.configuration_hash.clone(),
        ),
        (
            "calc_flow.state_schema_fingerprint",
            snapshot.state_schema_fingerprint.clone(),
        ),
    ];
    if snapshot.state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION {
        expected.extend([
            (
                "calc_flow.rolling_kernel_fingerprint",
                snapshot.kernel_fingerprint.clone().unwrap_or_default(),
            ),
            (
                "calc_flow.numerical_profile",
                snapshot.numerical_profile.clone().unwrap_or_default(),
            ),
        ]);
    }
    for (key, value) in expected {
        if metadata.get(key).map(String::as_str) != Some(value.as_str()) {
            return Err(checkpoint_mismatch(format!(
                "rolling state segment metadata {key} does not match the snapshot"
            )));
        }
    }
    Ok(())
}

// Moved from rolling.rs; row kinds retain their distinct recovery checks.
// #lizard forgives
fn decode_state_row(
    kind: u8,
    position: Option<u64>,
    values: Vec<ScalarValue>,
    ewma: Option<(u64, u64, f64)>,
    decoded: &mut DecodedRollingState,
    compiled: &CompiledRollingSpec,
    previous: &mut Option<StateRowOrderKey>,
) -> Result<()> {
    if kind == 2 {
        return decode_ewma_state_row(position, &values, ewma, decoded, compiled, previous);
    }
    if ewma.is_some() {
        return Err(state_format(
            "rolling history or buffer row carries EWMA state".to_owned(),
        ));
    }
    let row = buffered_row_from_values(values, compiled)?;
    let ordering_key = (
        kind,
        row.identity.entity.clone(),
        row.identity.clone(),
        position,
    );
    if let Some(prior) = previous.as_ref()
        && !state_rows_in_order(prior, &ordering_key)
    {
        return Err(state_format(
            "rolling state segment rows are not in deterministic key order".to_owned(),
        ));
    }
    match kind {
        0 => {
            let state = decoded
                .histories
                .by_entity
                .entry(row.identity.entity.clone())
                .or_default();
            let expected = u64::try_from(state.rows.len()).unwrap_or(u64::MAX);
            if position != Some(expected) {
                return Err(state_format(
                    "rolling state segment history positions are not contiguous".to_owned(),
                ));
            }
            state.rows.push_back(row.values);
        }
        1 => {
            if decoded.buffer.insert(row.identity.clone(), row).is_some() {
                return Err(state_format(
                    "rolling state segment contains a duplicate buffered identity".to_owned(),
                ));
            }
        }
        other => {
            return Err(state_format(format!(
                "rolling state segment contains unknown row kind {other}"
            )));
        }
    }
    *previous = Some(ordering_key);
    Ok(())
}

// Moved from rolling.rs; preserve the legacy EWMA state contract.
// #lizard forgives
pub(super) fn decode_ewma_state_row(
    position: Option<u64>,
    values: &[ScalarValue],
    ewma: Option<(u64, u64, f64)>,
    decoded: &mut DecodedRollingState,
    compiled: &CompiledRollingSpec,
    previous: &mut Option<StateRowOrderKey>,
) -> Result<()> {
    if compiled.state_layout_version != ROLLING_EWMA_STATE_LAYOUT_VERSION {
        return Err(state_format(
            "rolling layout v1 contains an EWMA state row".to_owned(),
        ));
    }
    if position.is_some() {
        return Err(state_format(
            "rolling EWMA state row carries a history position".to_owned(),
        ));
    }
    let (group, valid_count, value) = ewma.ok_or_else(|| {
        state_format("rolling EWMA state row is missing its accumulator".to_owned())
    })?;
    if valid_count == 0 {
        return Err(state_format(
            "rolling EWMA state row has a zero valid count".to_owned(),
        ));
    }
    let group_index = usize::try_from(group)
        .map_err(|_| state_format("rolling EWMA group does not fit usize".to_owned()))?;
    if !matches!(
        compiled.window_groups.get(group_index),
        Some(CompiledWindowGroup::Ewma { .. })
    ) {
        return Err(state_format(
            "rolling EWMA state row references a non-EWMA group".to_owned(),
        ));
    }
    let entity = ewma_entity_from_values(values, compiled)?;
    let ordering_key = (
        2,
        entity.clone(),
        RowIdentity {
            event_time: 0,
            entity: entity.clone(),
            sequence: Vec::new(),
        },
        Some(group),
    );
    if let Some(prior) = previous.as_ref()
        && !state_rows_in_order(prior, &ordering_key)
    {
        return Err(state_format(
            "rolling state segment rows are not in deterministic key order".to_owned(),
        ));
    }
    let state = decoded
        .histories
        .by_entity
        .entry(entity)
        .or_insert_with(|| EntityRollingState::fresh(compiled));
    if state.windows.is_empty() {
        state.windows = fresh_windows(compiled);
    }
    let WindowState::Ewma(accumulator) = &mut state.windows[group_index] else {
        unreachable!("validated EWMA group has EWMA state")
    };
    if accumulator.valid_count != 0 {
        return Err(state_format(
            "rolling state segment contains a duplicate EWMA accumulator".to_owned(),
        ));
    }
    *accumulator = EwmaAccumulator { valid_count, value };
    *previous = Some(ordering_key);
    Ok(())
}

pub(super) fn ewma_entity_from_values(
    values: &[ScalarValue],
    compiled: &CompiledRollingSpec,
) -> Result<Vec<Option<KeyValue>>> {
    for (index, value) in values.iter().enumerate() {
        if !compiled
            .partition_columns
            .iter()
            .any(|column| column.index == index)
            && !value.is_null()
        {
            return Err(state_format(
                "rolling EWMA state row populates a non-entity field".to_owned(),
            ));
        }
    }
    compiled
        .partition_columns
        .iter()
        .map(|column| KeyValue::from_nullable_scalar(&values[column.index], "rolling EWMA"))
        .collect()
}

fn state_rows_in_order(prior: &StateRowOrderKey, current: &StateRowOrderKey) -> bool {
    if prior.0 != current.0 {
        return prior.0 < current.0;
    }
    match prior.0 {
        0 => {
            if prior.1 != current.1 {
                return prior.1 < current.1;
            }
            match (prior.3, current.3) {
                (Some(left), Some(right)) => left < right,
                _ => false,
            }
        }
        1 => prior.2 < current.2,
        2 => {
            prior.1 < current.1
                || (prior.1 == current.1
                    && matches!((prior.3, current.3), (Some(left), Some(right)) if left < right))
        }
        _ => false,
    }
}

pub(super) fn buffered_row_from_values(
    values: Vec<ScalarValue>,
    compiled: &CompiledRollingSpec,
) -> Result<BufferedRow> {
    let event_time = match &values[compiled.event_time_index] {
        ScalarValue::TimestampMicrosecond(Some(value), _) => *value,
        _ => {
            return Err(state_format(
                "rolling state row has a null or non-timestamp event time".to_owned(),
            ));
        }
    };
    let entity = compiled
        .partition_columns
        .iter()
        .map(|column| KeyValue::from_nullable_scalar(&values[column.index], "rolling"))
        .collect::<Result<Vec<_>>>()?;
    let sequence = compiled
        .sequence_columns
        .iter()
        .map(|column| KeyValue::from_required_scalar(&values[column.index], "rolling"))
        .collect::<Result<Vec<_>>>()?;
    Ok(BufferedRow::new(entity, sequence, event_time, values))
}

pub(super) fn validate_decoded_state(
    decoded: &DecodedRollingState,
    compiled: &CompiledRollingSpec,
) -> Result<()> {
    let max_retained = usize::try_from(compiled.max_row_retention)
        .map_err(|_| internal_error("rolling max retained rows does not fit usize"))?;
    for state in decoded.histories.by_entity.values() {
        let bound = compiled
            .max_duration_micros
            .zip(
                state
                    .rows
                    .back()
                    .map(|values| history_event_time(values, compiled)),
            )
            .map(|(micros, last)| i128::from(last) - i128::from(micros));
        for (index, values) in state.rows.iter().enumerate() {
            let needed_by_count = state.rows.len() - index <= max_retained;
            let needed_by_time =
                bound.is_some_and(|bound| i128::from(history_event_time(values, compiled)) > bound);
            if !needed_by_count && !needed_by_time {
                return Err(state_format(
                    "rolling state segment retains more history than the declared frames"
                        .to_owned(),
                ));
            }
        }
    }
    Ok(())
}

/// Rebuilds every window accumulator as the ordered fold over the retained
/// history tail; the segment stores rows only, and the accumulator is the
/// deterministic function of those rows frozen in D5/D11. Extrema groups
/// fold pushes and expiries so the rebuilt queue front is the window
/// extremum, exactly as the live slide left it (SCE-08).
// #lizard forgives
pub(super) fn rebuild_windows(
    histories: &mut RollingHistories,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<()> {
    for state in histories.by_entity.values_mut() {
        let persisted = std::mem::take(&mut state.windows);
        let mut windows = fresh_windows(compiled);
        let last_time = state
            .rows
            .back()
            .map(|values| history_event_time(values, compiled));
        for (group_index, group) in compiled.window_groups.iter().enumerate() {
            match group {
                CompiledWindowGroup::Numeric {
                    input_index,
                    frame,
                    sum_class,
                } => {
                    let WindowState::Numeric(accumulator) = &mut windows[group_index] else {
                        return Err(internal_error("rolling numeric group state mismatch"));
                    };
                    let start = retained_window_start(*frame, state, compiled);
                    if spec_uses_stable_v2(compiled) && *sum_class == SumClass::Float {
                        *accumulator = kernel::stable_v2_float64_accumulator(
                            state.rows.iter().skip(start).filter_map(|values| {
                                let value = &values[*input_index];
                                is_valid_sample(value).then(|| float_sample(value))
                            }),
                            node_id,
                        )?;
                    } else {
                        for values in state.rows.iter().skip(start) {
                            let value = &values[*input_index];
                            if is_valid_sample(value) {
                                accumulator.add(value, node_id)?;
                            }
                        }
                    }
                    if let CompiledFrame::Duration(micros) = frame {
                        accumulator.expired_through = expired_through_bound(last_time, *micros);
                    }
                }
                CompiledWindowGroup::Extrema {
                    input_index, frame, ..
                } => {
                    let WindowState::Extrema(accumulator) = &mut windows[group_index] else {
                        return Err(internal_error("rolling extrema group state mismatch"));
                    };
                    rebuild_extrema_group(
                        accumulator,
                        state,
                        *input_index,
                        *frame,
                        compiled,
                        node_id,
                    )?;
                }
                CompiledWindowGroup::Pair {
                    left_index,
                    right_index,
                    frame,
                } => {
                    let WindowState::Pair(accumulator) = &mut windows[group_index] else {
                        return Err(internal_error("rolling pair group state mismatch"));
                    };
                    let start = retained_window_start(*frame, state, compiled);
                    if spec_uses_stable_v2(compiled) {
                        *accumulator = kernel::stable_v2_pair_accumulator(
                            state.rows.iter().skip(start).filter_map(|values| {
                                let x = &values[*left_index];
                                let y = &values[*right_index];
                                (is_valid_sample(x) && is_valid_sample(y))
                                    .then(|| (float_sample(x), float_sample(y)))
                            }),
                            node_id,
                        )?;
                    } else {
                        for values in state.rows.iter().skip(start) {
                            let x = &values[*left_index];
                            let y = &values[*right_index];
                            if is_valid_sample(x) && is_valid_sample(y) {
                                accumulator.add(x, y, node_id)?;
                            }
                        }
                    }
                    if let CompiledFrame::Duration(micros) = frame {
                        accumulator.expired_through = expired_through_bound(last_time, *micros);
                    }
                }
                CompiledWindowGroup::Ewma { .. } => {
                    if let Some(WindowState::Ewma(saved)) = persisted.get(group_index) {
                        windows[group_index] = WindowState::Ewma(*saved);
                    }
                }
            }
        }
        state.windows = windows;
    }
    Ok(())
}

/// Duration-frame expiry bound for a rebuild: the last retained row's
/// window lower edge, or "nothing expired" when no row is retained.
fn expired_through_bound(last_time: Option<i64>, micros: u64) -> i128 {
    last_time.map_or(i128::MIN, |last| i128::from(last) - i128::from(micros))
}

/// Canonical expiry key of one retained history row.
fn history_extrema_key(
    values: &[ScalarValue],
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<ExtremaKey> {
    Ok(ExtremaKey {
        event_time: history_event_time(values, compiled),
        sequence: compiled
            .sequence_columns
            .iter()
            .map(|column| KeyValue::from_required_scalar(&values[column.index], node_id))
            .collect::<Result<Vec<_>>>()?,
    })
}

/// Rebuilds one extrema queue as the ordered push/expire fold over the
/// retained rows, mirroring the live slide so queue front, expiry keys, and
/// valid count match an uninterrupted run exactly.
// #lizard forgives
fn rebuild_extrema_group(
    accumulator: &mut ExtremaAccumulator,
    state: &EntityRollingState,
    input_index: usize,
    frame: CompiledFrame,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<()> {
    let rows = usize::try_from(frame.rows())
        .map_err(|_| internal_error("rolling frame rows do not fit usize"))?;
    let mut cursor = 0_usize;
    for position in 0..state.rows.len() {
        let values = &state.rows[position];
        let key = history_extrema_key(values, compiled, node_id)?;
        let value = &values[input_index];
        if is_valid_sample(value) {
            accumulator.add(key.clone(), value.clone());
        }
        match frame {
            CompiledFrame::Rows(..) => {
                if position >= rows {
                    let leaving_row = &state.rows[position - rows];
                    if is_valid_sample(&leaving_row[input_index]) {
                        accumulator.remove();
                    }
                    accumulator.expire_through_key(&history_extrema_key(
                        leaving_row,
                        compiled,
                        node_id,
                    )?);
                }
            }
            CompiledFrame::Duration(micros) => {
                let bound = i128::from(key.event_time) - i128::from(micros);
                while cursor < position
                    && i128::from(history_event_time(&state.rows[cursor], compiled)) <= bound
                {
                    if is_valid_sample(&state.rows[cursor][input_index]) {
                        accumulator.remove();
                    }
                    cursor += 1;
                }
                accumulator.expire_through_time(bound);
                accumulator.expired_through = bound;
            }
        }
    }
    Ok(())
}

/// Start position of the retained window of the last retained row: the last
/// `rows` positions for row frames, the positions with event time in
/// `(t_last - d, t_last]` for duration frames. Restore-time only, so a
/// linear scan over the retained tail is acceptable.
fn retained_window_start(
    frame: CompiledFrame,
    state: &EntityRollingState,
    compiled: &CompiledRollingSpec,
) -> usize {
    let len = state.rows.len();
    let last_time = state
        .rows
        .back()
        .map(|values| history_event_time(values, compiled));
    match frame {
        CompiledFrame::Rows(rows) => {
            let rows = usize::try_from(rows).unwrap_or(usize::MAX);
            len.saturating_sub(rows)
        }
        CompiledFrame::Duration(micros) => match last_time {
            None => 0,
            Some(last) => {
                let bound = i128::from(last) - i128::from(micros);
                state
                    .rows
                    .iter()
                    .position(|values| i128::from(history_event_time(values, compiled)) > bound)
                    .unwrap_or(len)
            }
        },
    }
}

pub(super) fn parse_snapshot_metadata(
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<RollingSnapshotMetadata> {
    serde_json::from_value::<RollingSnapshotMetadata>(Value::Object(
        snapshot.inline_metadata.clone().into_iter().collect(),
    ))
    .map_err(|error| format_error(&error))
}

pub(super) fn validate_snapshot_metadata(
    metadata: &RollingSnapshotMetadata,
    compiled: &CompiledRollingSpec,
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<StateInventory> {
    let expected_schema_fingerprint =
        if metadata.state_layout_version == compiled.state_layout_version {
            &compiled.state_schema_fingerprint
        } else if metadata.state_layout_version == compiled.legacy_state_layout_version {
            &compiled.legacy_state_schema_fingerprint
        } else {
            return Err(checkpoint_mismatch(format!(
                "rolling state layout version {} does not match current {} or declared legacy {}",
                metadata.state_layout_version,
                compiled.state_layout_version,
                compiled.legacy_state_layout_version
            )));
        };
    if metadata.configuration_hash != compiled.configuration_hash {
        return Err(checkpoint_mismatch(
            "rolling operator configuration hash does not match the compiled operator",
        ));
    }
    if metadata.state_schema_fingerprint != *expected_schema_fingerprint {
        return Err(checkpoint_mismatch(
            "rolling state schema fingerprint does not match the compiled operator",
        ));
    }
    if metadata.state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION
        && (metadata.kernel_fingerprint.as_deref() != Some(compiled.kernel_plan.fingerprint())
            || metadata.numerical_profile.as_deref()
                != Some(compiled.kernel_plan.numerical_profile()))
    {
        return Err(checkpoint_mismatch(
            "rolling kernel fingerprint or numerical profile does not match the compiled operator",
        ));
    }
    super::super::checkpoint::validate_inventory(
        &super::super::checkpoint::SnapshotContract {
            name: "rolling",
            state_layout_version: metadata.state_layout_version,
            schema_fingerprint: &metadata.state_schema_fingerprint,
            epoch: metadata.epoch,
            pipeline_fingerprint: metadata.pipeline_fingerprint.as_deref(),
            operator_id: metadata.operator_id.as_deref(),
            segment_inventory: metadata.segment_inventory.clone(),
        },
        snapshot,
    )
}

pub(super) fn snapshot_segments(
    snapshot: &crate::OperatorStateSnapshot,
    inventory: &[SegmentDescriptor],
) -> Result<Vec<Arc<Vec<u8>>>> {
    inventory
        .iter()
        .map(|descriptor| {
            let segment_id = descriptor.handle.segment_id();
            let segment = snapshot.segments.get(segment_id).ok_or_else(|| {
                checkpoint_mismatch(format!(
                    "rolling snapshot is missing segment {segment_id:?}"
                ))
            })?;
            // A fresh session revalidates every referenced segment byte
            // against the manifest handle before any state is installed.
            let bytes = segment.bytes();
            if u64::try_from(bytes.len()).ok() != Some(descriptor.handle.byte_len()) {
                return Err(checkpoint_mismatch(
                    "rolling snapshot segment byte length does not match its handle",
                ));
            }
            if hex::encode(Sha256::digest(bytes)) != descriptor.handle.sha256() {
                return Err(checkpoint_mismatch(
                    "rolling snapshot segment checksum does not match its handle",
                ));
            }
            Ok(segment.bytes_arc())
        })
        .collect()
}

pub(super) fn deserialize_required_option<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

pub(super) fn closing_coordinate(
    event_time: i64,
    allowed_lateness_micros: u64,
    node_id: &str,
) -> Result<i64> {
    let lateness = i64::try_from(allowed_lateness_micros).map_err(|_| {
        operator_error(
            node_id,
            "allowed lateness exceeds the representable event-time range",
        )
    })?;
    event_time.checked_add(lateness).ok_or_else(|| {
        operator_error(
            node_id,
            "finality coordinate overflowed the event-time range",
        )
    })
}

/// Splits one rolling output record into edge-budget-sized messages via the
/// shared operator chunker.
pub(super) fn chunk_output_record(
    record: &RecordBatch,
    operator_id: &str,
    first_sequence: u64,
    budget: crate::EdgeBudget,
) -> Result<Vec<Batch>> {
    super::super::output_chunk::chunk_output_record(
        record,
        operator_id,
        first_sequence,
        budget,
        super::super::output_chunk::OutputChunkErrors::ROLLING,
    )
}

/// Reads every input row with its canonical identity; null event-time or
/// sequence values are malformed runtime data (SCE-00 D4/D12).
pub(super) fn read_buffered_rows(
    table: &TableBatch,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<Vec<BufferedRow>> {
    let mut rows = Vec::with_capacity(table.batches().iter().map(RecordBatch::num_rows).sum());
    for record in table.batches() {
        for row_index in 0..record.num_rows() {
            rows.push(read_buffered_row(record, row_index, compiled, node_id)?);
        }
    }
    Ok(rows)
}

pub(super) fn read_buffered_row(
    record: &RecordBatch,
    row_index: usize,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<BufferedRow> {
    let mut values = Vec::with_capacity(record.num_columns());
    for column in record.columns() {
        values.push(
            ScalarValue::try_from_array(column, row_index).map_err(|error| {
                operator_error(
                    node_id,
                    &format!("rolling input row could not be read: {error}"),
                )
            })?,
        );
    }
    let event_time = match &values[compiled.event_time_index] {
        ScalarValue::TimestampMicrosecond(Some(value), _) => *value,
        _ => {
            return Err(operator_error(
                node_id,
                "rolling event-time value is null or not a microsecond timestamp",
            ));
        }
    };
    let entity = compiled
        .partition_columns
        .iter()
        .map(|column| KeyValue::from_nullable_scalar(&values[column.index], node_id))
        .collect::<Result<Vec<_>>>()?;
    let sequence = compiled
        .sequence_columns
        .iter()
        .map(|column| KeyValue::from_required_scalar(&values[column.index], node_id))
        .collect::<Result<Vec<_>>>()?;
    Ok(BufferedRow::new(entity, sequence, event_time, values))
}

/// Sorts accepted rows into the canonical observable order and rejects
/// duplicate identities before any output is produced (SCE-00 D4).
pub(super) fn sort_and_validate(
    mut rows: Vec<BufferedRow>,
    node_id: &str,
) -> Result<Vec<BufferedRow>> {
    rows.sort_by(|left, right| left.identity.cmp(&right.identity));
    if let Some(duplicate) = rows
        .windows(2)
        .find(|pair| pair[0].identity == pair[1].identity)
    {
        return Err(operator_error(
            node_id,
            &format!(
                "duplicate row identity at event_time_micros={}",
                duplicate[0].identity.event_time
            ),
        ));
    }
    Ok(rows)
}

/// Builds one output record: canonical-order input columns followed by the
/// derived rolling outputs (SCE-00 D5).
pub(super) fn build_output_record(
    rows: &[BufferedRow],
    derived: Vec<ArrayRef>,
    output_schema: &SchemaRef,
    node_id: &str,
) -> Result<RecordBatch> {
    let input_width = rows.first().map_or_else(
        || output_schema.fields().len() - derived.len(),
        |row| row.values.len(),
    );
    let mut columns = Vec::with_capacity(input_width + derived.len());
    for index in 0..input_width {
        if rows.is_empty() {
            columns.push(new_null_array(output_schema.field(index).data_type(), 0));
            continue;
        }
        columns.push(
            ScalarValue::iter_to_array(rows.iter().map(|row| row.values[index].clone())).map_err(
                |error| {
                    operator_error(
                        node_id,
                        &format!("rolling output row encoding failed: {error}"),
                    )
                },
            )?,
        );
    }
    columns.extend(derived);
    RecordBatch::try_new(Arc::clone(output_schema), columns).map_err(|error| {
        operator_error(
            node_id,
            &format!("rolling output record is invalid: {error}"),
        )
    })
}

/// Builds a batch result directly from Arrow buffers when the immutable
/// kernel plan supports the semantic shape and the input proves canonical
/// order. `None` preserves the general sort-capable fallback.
pub(super) fn build_typed_batch_output(
    table: &TableBatch,
    compiled: &CompiledRollingSpec,
    output_schema: &SchemaRef,
    node_id: &str,
) -> Result<Option<RecordBatch>> {
    let input = if let [record] = table.batches() {
        record.clone()
    } else {
        concat_batches(table.schema(), table.batches()).map_err(|error| {
            operator_error(
                node_id,
                &format!("typed rolling input concatenation failed: {error}"),
            )
        })?
    };
    let Some(execution) = compiled.kernel_plan.open_and_fill(&input, node_id)? else {
        return Ok(None);
    };
    debug_assert_eq!(execution.metrics.input_rows, input.num_rows());
    debug_assert_eq!(execution.metrics.output_rows, input.num_rows());
    let mut columns = input.columns().to_vec();
    columns.extend(execution.columns);
    RecordBatch::try_new(Arc::clone(output_schema), columns)
        .map(Some)
        .map_err(|error| {
            operator_error(
                node_id,
                &format!("typed rolling output record is invalid: {error}"),
            )
        })
}

type TypedStreamOutput = (RecordBatch, Option<RollingKernelState>, HistoryUpdates);

// Bootstrap, transition, output slicing, and history replacement form one
// failure-atomic stream update; none of them may escape independently.
// #lizard forgives
pub(super) fn build_typed_stream_output(
    rows: &[BufferedRow],
    histories: &RollingHistories,
    state: Option<&RollingKernelState>,
    compiled: &CompiledRollingSpec,
    output_schema: &SchemaRef,
    node_id: &str,
    observer: Option<&RollingMetricsRecorder>,
) -> Result<Option<TypedStreamOutput>> {
    if !compiled.kernel_plan.supports_typed_transition() {
        return Ok(None);
    }
    let input_schema = Arc::new(Schema::new(
        output_schema.fields()[..output_schema.fields().len() - compiled.outputs.len()].to_vec(),
    ));
    let restored_state;
    let prior = if let Some(state) = state {
        state
    } else {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::StatePreparation));
        restored_state = reconstruct_typed_state(histories, compiled, &input_schema, node_id)?;
        &restored_state
    };
    let input = {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
        build_input_record(rows, input_schema, node_id)?
    };
    let execution = compiled
        .kernel_plan
        .update_stream_and_fill(prior, &input, node_id, observer)?
        .ok_or_else(|| {
            internal_error("typed rolling stream rows did not satisfy canonical ordering")
        })?;
    let columns = execution.columns;
    let record = {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
        build_output_record(rows, columns, output_schema, node_id)?
    };
    let touched = {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
        typed_history_updates(rows, histories, compiled, node_id)?
    };
    Ok(Some((record, Some(execution.state), touched)))
}

pub(super) fn reconstruct_typed_state(
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    input_schema: &SchemaRef,
    node_id: &str,
) -> Result<RollingKernelState> {
    let bootstrap = typed_bootstrap_rows(histories, compiled)?;
    let reconstructed = if bootstrap.is_empty() {
        RollingKernelState::default()
    } else {
        let input = build_input_record(&bootstrap, Arc::clone(input_schema), node_id)?;
        compiled
            .kernel_plan
            .update_and_fill(&RollingKernelState::default(), &input, node_id)?
            .ok_or_else(|| internal_error("typed rolling restore history is not canonical"))?
            .state
    };
    seed_typed_restored_state(histories, compiled, input_schema, &reconstructed, node_id)
}

fn seed_typed_restored_state(
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    input_schema: &SchemaRef,
    state: &RollingKernelState,
    node_id: &str,
) -> Result<RollingKernelState> {
    if histories.by_entity.is_empty() {
        return Ok(state.clone());
    }
    let values = histories
        .by_entity
        .keys()
        .map(|entity| ewma_entity_values(entity, input_schema, compiled))
        .collect::<Result<Vec<_>>>()?;
    let seeds = histories
        .by_entity
        .values()
        .map(|entity| {
            let mut seeds = typed_ewma_seeds(entity, compiled)?;
            seeds.resize(compiled.kernel_plan.typed_group_count(), None);
            Ok(seeds)
        })
        .collect::<Result<Vec<_>>>()?;
    let transition_counts = histories
        .by_entity
        .values()
        .map(|entity| entity.transition_count)
        .collect::<Vec<_>>();
    let nullable_schema = Arc::new(Schema::new(
        input_schema
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    let entities = build_value_record(&values, nullable_schema, node_id)?;
    compiled
        .kernel_plan
        .seed_restored_state(state, &entities, &transition_counts, &seeds, node_id)
}

fn typed_ewma_seeds(
    entity: &EntityRollingState,
    compiled: &CompiledRollingSpec,
) -> Result<Vec<Option<(u64, f64)>>> {
    compiled
        .window_groups
        .iter()
        .enumerate()
        .map(|(group_index, group)| {
            if !matches!(group, CompiledWindowGroup::Ewma { .. }) {
                return Ok(None);
            }
            match entity.windows.get(group_index) {
                Some(WindowState::Ewma(state)) if state.valid_count > 0 => {
                    Ok(Some((state.valid_count, state.value)))
                }
                Some(WindowState::Ewma(_)) | None => Ok(None),
                _ => Err(internal_error("rolling EWMA checkpoint state mismatch")),
            }
        })
        .collect()
}

fn typed_bootstrap_rows(
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
) -> Result<Vec<BufferedRow>> {
    let mut rows = histories
        .by_entity
        .values()
        .flat_map(|state| state.rows.iter())
        .map(|values| buffered_row_from_values(values.clone(), compiled))
        .collect::<Result<Vec<_>>>()?;
    rows.sort_by(|left, right| left.identity.cmp(&right.identity));
    Ok(rows)
}

fn build_input_record(
    rows: &[BufferedRow],
    schema: SchemaRef,
    node_id: &str,
) -> Result<RecordBatch> {
    let arrays = (0..schema.fields().len())
        .map(|index| {
            ScalarValue::iter_to_array(rows.iter().map(|row| row.values[index].clone())).map_err(
                |error| {
                    operator_error(
                        node_id,
                        &format!("typed rolling stream input encoding failed: {error}"),
                    )
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(schema, arrays).map_err(|error| {
        operator_error(
            node_id,
            &format!("typed rolling stream input batch is invalid: {error}"),
        )
    })
}

fn build_value_record(
    rows: &[Vec<ScalarValue>],
    schema: SchemaRef,
    node_id: &str,
) -> Result<RecordBatch> {
    let arrays = (0..schema.fields().len())
        .map(|index| {
            ScalarValue::iter_to_array(rows.iter().map(|row| row[index].clone())).map_err(|error| {
                operator_error(
                    node_id,
                    &format!("typed rolling restore entity encoding failed: {error}"),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(schema, arrays).map_err(|error| {
        operator_error(
            node_id,
            &format!("typed rolling restore entity batch is invalid: {error}"),
        )
    })
}

fn typed_history_updates(
    rows: &[BufferedRow],
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<HistoryUpdates> {
    let has_ewma = compiled
        .window_groups
        .iter()
        .any(|group| matches!(group, CompiledWindowGroup::Ewma { .. }));
    group_rows_by_entity(rows)
        .into_iter()
        .map(|(entity, indices)| {
            let mut state = histories.by_entity.get(entity).cloned().unwrap_or_default();
            let transitions = u64::try_from(indices.len()).map_err(|_| {
                operator_error(node_id, "rolling micro-batch row count does not fit u64")
            })?;
            if has_ewma {
                advance_typed_ewma_windows(&mut state, rows, &indices, compiled, node_id)?;
            } else {
                state.windows.clear();
            }
            state
                .rows
                .extend(indices.into_iter().map(|index| rows[index].values.clone()));
            state.transition_count =
                state
                    .transition_count
                    .checked_add(transitions)
                    .ok_or_else(|| {
                        operator_error(node_id, "rolling entity transition count overflowed")
                    })?;
            evict_retained_history(&mut state, compiled);
            Ok((entity.clone(), state))
        })
        .collect()
}

fn advance_typed_ewma_windows(
    state: &mut EntityRollingState,
    rows: &[BufferedRow],
    indices: &[usize],
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<()> {
    if state.windows.len() != compiled.window_groups.len() {
        state.windows = fresh_windows(compiled);
    }
    for &row_index in indices {
        for (group_index, group) in compiled.window_groups.iter().enumerate() {
            let CompiledWindowGroup::Ewma {
                input_index, alpha, ..
            } = group
            else {
                continue;
            };
            let WindowState::Ewma(accumulator) = &mut state.windows[group_index] else {
                return Err(internal_error("rolling typed EWMA history state mismatch"));
            };
            let sample = &rows[row_index].values[*input_index];
            if is_valid_sample(sample) {
                accumulator.add(sample, *alpha, node_id)?;
            }
        }
    }
    Ok(())
}
