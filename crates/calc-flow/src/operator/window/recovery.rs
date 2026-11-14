use super::{
    AccumulatorRow, AccumulatorValue, AggregateFunction, Arc, Array, BTreeMap, CompensatedSum,
    CompiledAggregate, CompiledWindowSpec, Cursor, DataType, EventTime, FileReader,
    FixedSizeBinaryArray, LargeBinaryArray, MAX_GROUP_KEY_BYTES, RecordBatch, Result, ScalarValue,
    Schema, StateInventory, StateOperation, TimestampMicrosecondArray, UInt8Array, UInt64Array,
    WINDOW_SEGMENT_LAYOUT_VERSION, WINDOW_STATE_LAYOUT_VERSION, WindowKey, WindowSnapshotMetadata,
    WindowSpec, checkpoint_mismatch, encode_group_scalar, float_value, fold_state_segments,
    scalar_at, signed_value, state_format, state_schema_fingerprint_for_version,
    state_schema_for_version, unsigned_value,
};

pub(super) fn validate_snapshot_metadata(
    metadata: &WindowSnapshotMetadata,
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<StateInventory> {
    validate_snapshot_header(metadata, spec, compiled)?;
    let inventory = StateInventory::new(metadata.segment_inventory.clone())
        .map_err(|error| checkpoint_mismatch(error.to_string()))?;
    validate_snapshot_inventory(metadata, spec, compiled, &inventory)?;
    validate_snapshot_segment_set(snapshot, &inventory)?;
    validate_snapshot_identity(metadata, snapshot)?;
    Ok(inventory)
}

fn validate_snapshot_header(
    metadata: &WindowSnapshotMetadata,
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
) -> Result<()> {
    if !(WINDOW_STATE_LAYOUT_VERSION..=WINDOW_SEGMENT_LAYOUT_VERSION)
        .contains(&metadata.state_layout_version)
    {
        return Err(checkpoint_mismatch(format!(
            "window state layout version {} is unsupported",
            metadata.state_layout_version
        )));
    }
    if metadata.configuration_hash != compiled.configuration_hash {
        return Err(checkpoint_mismatch(
            "window operator configuration hash does not match the compiled operator",
        ));
    }
    let expected_fingerprint =
        state_schema_fingerprint_for_version(spec, compiled, metadata.state_layout_version);
    if metadata.state_schema_fingerprint != expected_fingerprint {
        return Err(checkpoint_mismatch(
            "window state schema fingerprint does not match the compiled operator",
        ));
    }
    Ok(())
}

fn validate_snapshot_segment_set(
    snapshot: &crate::OperatorStateSnapshot,
    inventory: &StateInventory,
) -> Result<()> {
    let expected_ids = inventory
        .segments()
        .iter()
        .map(|descriptor| descriptor.handle.segment_id().to_owned())
        .collect::<Vec<_>>();
    let actual_ids = snapshot.segments.keys().cloned().collect::<Vec<_>>();
    if expected_ids != actual_ids {
        return Err(checkpoint_mismatch(
            "window snapshot segment IDs are missing, extra, duplicated, or non-canonical",
        ));
    }
    Ok(())
}

fn validate_snapshot_identity(
    metadata: &WindowSnapshotMetadata,
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<()> {
    if !snapshot.segments.is_empty()
        && (metadata.pipeline_fingerprint.is_none() || metadata.operator_id.is_none())
    {
        return Err(checkpoint_mismatch(
            "window segments require pipeline and operator identity metadata",
        ));
    }
    validate_snapshot_pipeline_fingerprint(metadata.pipeline_fingerprint.as_deref())?;
    validate_snapshot_operator_id(metadata.operator_id.as_deref())
}

fn validate_snapshot_pipeline_fingerprint(fingerprint: Option<&str>) -> Result<()> {
    if let Some(fingerprint) = fingerprint
        && (fingerprint.len() != 64
            || !fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        return Err(checkpoint_mismatch(
            "window pipeline fingerprint is not lowercase SHA-256",
        ));
    }
    Ok(())
}

fn validate_snapshot_operator_id(operator_id: Option<&str>) -> Result<()> {
    if operator_id.is_some_and(|operator_id| operator_id.is_empty() || operator_id.contains('\0')) {
        return Err(checkpoint_mismatch(
            "window operator ID is empty or contains NUL",
        ));
    }
    Ok(())
}

fn validate_snapshot_inventory(
    metadata: &WindowSnapshotMetadata,
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    inventory: &StateInventory,
) -> Result<()> {
    for descriptor in inventory.segments() {
        let version = descriptor.state_layout_version;
        if version != metadata.state_layout_version
            || descriptor.schema_fingerprint
                != state_schema_fingerprint_for_version(spec, compiled, version)
        {
            return Err(checkpoint_mismatch(
                "window segment inventory layout or schema does not match the compiled operator",
            ));
        }
        if descriptor.handle.epoch() > metadata.epoch {
            return Err(checkpoint_mismatch(
                "window segment inventory contains a future epoch",
            ));
        }
        if metadata.operator_id.as_deref() != Some(descriptor.handle.operator_id()) {
            return Err(checkpoint_mismatch(
                "window segment inventory operator does not match snapshot metadata",
            ));
        }
    }
    Ok(())
}

pub(super) fn decode_state_segments(
    segments: &[(u32, Arc<Vec<u8>>)],
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    pipeline_fingerprint: Option<&str>,
    operator_id: Option<&str>,
) -> Result<BTreeMap<WindowKey, AccumulatorRow>> {
    if segments.is_empty() {
        return Ok(BTreeMap::new());
    }
    let pipeline_fingerprint = pipeline_fingerprint
        .ok_or_else(|| checkpoint_mismatch("window state is missing its pipeline fingerprint"))?;
    let operator_id = operator_id
        .ok_or_else(|| checkpoint_mismatch("window state is missing its operator ID"))?;
    let decoded = segments
        .iter()
        .map(|(version, bytes)| {
            let expected_schema = state_schema_for_version(
                spec,
                compiled,
                pipeline_fingerprint,
                operator_id,
                *version,
            );
            decode_state_segment(
                bytes,
                spec,
                compiled,
                &expected_schema,
                operator_id,
                *version,
            )
            .map(|operations| {
                operations
                    .into_iter()
                    .map(|(key, entry)| {
                        let operation =
                            entry.map_or(StateOperation::Tombstone, StateOperation::Upsert);
                        (key, operation)
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Result<Vec<_>>>()?;
    fold_state_segments(decoded)
}

#[allow(
    clippy::too_many_lines,
    reason = "durable state validation remains a single fail-before-install decoding transaction"
)]
pub(super) fn decode_state_segment(
    bytes: &[u8],
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    expected_schema: &Schema,
    operator_id: &str,
    version: u32,
) -> Result<Vec<(WindowKey, Option<AccumulatorRow>)>> {
    if !bytes.starts_with(b"ARROW1") || !bytes.ends_with(b"ARROW1") {
        return Err(state_format(
            "window state segment is missing the Arrow IPC file magic",
        ));
    }
    let mut reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| state_format(format!("window state IPC is invalid: {error}")))?;
    if reader.schema().as_ref() != expected_schema {
        return Err(checkpoint_mismatch(
            "window state Arrow schema or metadata does not match the compiled operator",
        ));
    }
    if reader.num_batches() != 1 {
        return Err(state_format(format!(
            "window state segment must contain exactly one record batch, found {}",
            reader.num_batches()
        )));
    }
    let record = reader
        .next()
        .ok_or_else(|| state_format("window state segment has no record batch"))?
        .map_err(|error| state_format(format!("window state batch is invalid: {error}")))?;
    if record.num_rows() == 0 {
        return Err(state_format(
            "window state segment must contain at least one operation",
        ));
    }

    let operations = state_array::<UInt8Array>(&record, 0, "_operation")?;
    let starts = state_array::<TimestampMicrosecondArray>(&record, 1, "window_start")?;
    let ends = state_array::<TimestampMicrosecondArray>(&record, 2, "window_end")?;
    let stable_keys = state_array::<LargeBinaryArray>(&record, 3, "_stable_group_key")?;
    let mut decoded = Vec::with_capacity(record.num_rows());
    let mut previous_key = None::<WindowKey>;

    for row in 0..record.num_rows() {
        if operations.is_null(row)
            || starts.is_null(row)
            || ends.is_null(row)
            || stable_keys.is_null(row)
        {
            return Err(state_format(
                "window state operation and key columns must not be null",
            ));
        }
        let tombstone = match operations.value(row) {
            0 => false,
            1 => true,
            value => {
                return Err(state_format(format!(
                    "window state operation {value} is not 0 or 1"
                )));
            }
        };
        let key = WindowKey {
            start: EventTime::from_micros(starts.value(row)),
            end: EventTime::from_micros(ends.value(row)),
            stable_group_key: stable_keys.value(row).into(),
        };
        validate_restored_window_key(&key, compiled)?;
        if previous_key
            .as_ref()
            .is_some_and(|previous| previous >= &key)
        {
            return Err(state_format(
                "window state rows are not in strict key order or contain a duplicate key",
            ));
        }
        previous_key = Some(key.clone());

        let mut group_values = Vec::with_capacity(compiled.group_columns.len());
        for (ordinal, group) in compiled.group_columns.iter().enumerate() {
            group_values.push(
                scalar_at(
                    record.column(4 + ordinal).as_ref(),
                    &group.data_type,
                    row,
                    operator_id,
                )
                .map_err(|error| state_format(error.to_string()))?,
            );
        }
        let encoded_group = encode_group_values(&group_values, compiled)?;
        if *encoded_group != *key.stable_group_key {
            return Err(checkpoint_mismatch(
                "window state stable group key does not match its declared group values",
            ));
        }

        let mut column_index = 4 + compiled.group_columns.len();
        let mut accumulators = Vec::with_capacity(compiled.aggregates.len());
        for (ordinal, (aggregate, compiled_aggregate)) in
            spec.aggregates.iter().zip(&compiled.aggregates).enumerate()
        {
            let (accumulator, next_column) = decode_accumulator_state(
                &record,
                row,
                column_index,
                tombstone,
                aggregate.function,
                compiled_aggregate,
                operator_id,
                ordinal,
                version,
            )?;
            column_index = next_column;
            if let Some(accumulator) = accumulator {
                accumulators.push(accumulator);
            }
        }
        decoded.push((
            key,
            (!tombstone).then_some(AccumulatorRow {
                group_values: group_values.into(),
                aggregates: accumulators,
            }),
        ));
    }
    Ok(decoded)
}

#[allow(
    clippy::too_many_arguments,
    reason = "state decoding names every durable aggregate coordinate explicitly"
)]
fn decode_accumulator_state(
    record: &RecordBatch,
    row: usize,
    column_index: usize,
    tombstone: bool,
    function: AggregateFunction,
    compiled: &CompiledAggregate,
    operator_id: &str,
    ordinal: usize,
    version: u32,
) -> Result<(Option<AccumulatorValue>, usize)> {
    let value = record.column(column_index);
    let compensated = version >= WINDOW_SEGMENT_LAYOUT_VERSION
        && ((function == AggregateFunction::Sum && compiled.output_type == DataType::Float64)
            || (function == AggregateFunction::Avg
                && matches!(compiled.input_type, DataType::Float32 | DataType::Float64)));
    let width = 1 + usize::from(function == AggregateFunction::Avg) + usize::from(compensated);
    if tombstone {
        if (0..width).any(|offset| !record.column(column_index + offset).is_null(row)) {
            return Err(state_format(format!(
                "window tombstone aggregate {ordinal} contains state"
            )));
        }
        return Ok((None, column_index + width));
    }

    let decoded = match function {
        AggregateFunction::Count => {
            let Some(ScalarValue::Unsigned(value)) =
                scalar_at(value.as_ref(), &DataType::UInt64, row, operator_id)
                    .map_err(|error| state_format(error.to_string()))?
            else {
                return Err(state_format(format!(
                    "window count aggregate {ordinal} has null or invalid state"
                )));
            };
            AccumulatorValue::Count(value)
        }
        AggregateFunction::Sum => match compiled.output_type {
            DataType::Int64 => AccumulatorValue::SignedSum(
                scalar_at(value.as_ref(), &DataType::Int64, row, operator_id)
                    .map_err(|error| state_format(error.to_string()))?
                    .map(|value| signed_value(&value).map(i128::from))
                    .transpose()
                    .map_err(state_format)?,
            ),
            DataType::UInt64 => AccumulatorValue::UnsignedSum(
                scalar_at(value.as_ref(), &DataType::UInt64, row, operator_id)
                    .map_err(|error| state_format(error.to_string()))?
                    .map(|value| unsigned_value(&value).map(u128::from))
                    .transpose()
                    .map_err(state_format)?,
            ),
            DataType::Float64 => {
                let sum = scalar_at(value.as_ref(), &DataType::Float64, row, operator_id)
                    .map_err(|error| state_format(error.to_string()))?
                    .map(|value| float_value(&value))
                    .transpose()
                    .map_err(state_format)?;
                let correction = if compensated {
                    match sum {
                        Some(_) => float_compensation_at(
                            record,
                            row,
                            column_index + 1,
                            operator_id,
                            ordinal,
                        )?,
                        None if record.column(column_index + 1).is_null(row) => 0.0,
                        None => {
                            return Err(state_format(format!(
                                "window float sum aggregate {ordinal} has correction without sum"
                            )));
                        }
                    }
                } else {
                    0.0
                };
                AccumulatorValue::FloatSum(sum.map(|sum| CompensatedSum { sum, correction }))
            }
            _ => unreachable!("sum output matrix validated at construction"),
        },
        AggregateFunction::Min | AggregateFunction::Max => {
            let scalar = scalar_at(value.as_ref(), &compiled.input_type, row, operator_id)
                .map_err(|error| state_format(error.to_string()))?;
            if function == AggregateFunction::Min {
                AccumulatorValue::Min(scalar)
            } else {
                AccumulatorValue::Max(scalar)
            }
        }
        AggregateFunction::Avg => {
            let context = AverageDecodeContext {
                record,
                row,
                column_index,
                operator_id,
                ordinal,
                version,
            };
            decode_average_state(&context, value.as_ref(), &compiled.input_type)?
        }
    };
    Ok((Some(decoded), column_index + width))
}

fn float_compensation_at(
    record: &RecordBatch,
    row: usize,
    column_index: usize,
    operator_id: &str,
    ordinal: usize,
) -> Result<f64> {
    let value = scalar_at(
        record.column(column_index),
        &DataType::Float64,
        row,
        operator_id,
    )
    .map_err(|error| state_format(error.to_string()))?
    .ok_or_else(|| {
        state_format(format!(
            "window float aggregate {ordinal} has null correction"
        ))
    })?;
    float_value(&value).map_err(state_format)
}

struct AverageDecodeContext<'a> {
    record: &'a RecordBatch,
    row: usize,
    column_index: usize,
    operator_id: &'a str,
    ordinal: usize,
    version: u32,
}

fn decode_float_average_state(
    context: &AverageDecodeContext<'_>,
    value: &dyn Array,
    count: u64,
) -> Result<AccumulatorValue> {
    let Some(ScalarValue::Float64(bits)) =
        scalar_at(value, &DataType::Float64, context.row, context.operator_id)
            .map_err(|error| state_format(error.to_string()))?
    else {
        return Err(state_format(format!(
            "window average aggregate {} has invalid float state",
            context.ordinal
        )));
    };
    Ok(AccumulatorValue::FloatAverage {
        sum: CompensatedSum {
            sum: f64::from_bits(bits),
            correction: if context.version >= WINDOW_SEGMENT_LAYOUT_VERSION {
                float_compensation_at(
                    context.record,
                    context.row,
                    context.column_index + 2,
                    context.operator_id,
                    context.ordinal,
                )?
            } else {
                0.0
            },
        },
        count,
    })
}

/// Decode one average aggregate's sum column and its adjacent count column.
fn decode_average_state(
    context: &AverageDecodeContext<'_>,
    value: &dyn Array,
    input_type: &DataType,
) -> Result<AccumulatorValue> {
    let count_array = state_array::<UInt64Array>(
        context.record,
        context.column_index + 1,
        &format!("_agg_{:04}_count", context.ordinal),
    )?;
    if value.is_null(context.row) || count_array.is_null(context.row) {
        return Err(state_format(format!(
            "window average aggregate {} has null state",
            context.ordinal
        )));
    }
    let count = count_array.value(context.row);
    match input_type {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            fixed_width_average_bytes(value, context.row, context.ordinal, "signed").map(|bytes| {
                AccumulatorValue::SignedAverage {
                    sum: i128::from_be_bytes(bytes),
                    count,
                }
            })
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            fixed_width_average_bytes(value, context.row, context.ordinal, "unsigned").map(
                |bytes| AccumulatorValue::UnsignedAverage {
                    sum: u128::from_be_bytes(bytes),
                    count,
                },
            )
        }
        DataType::Float32 | DataType::Float64 => decode_float_average_state(context, value, count),
        _ => unreachable!("average input matrix validated at construction"),
    }
}

/// Read one 16-byte big-endian average sum from a fixed-size binary column.
fn fixed_width_average_bytes(
    value: &dyn Array,
    row: usize,
    ordinal: usize,
    kind: &str,
) -> Result<[u8; 16]> {
    value
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| {
            state_format(format!(
                "window average aggregate {ordinal} has invalid binary state"
            ))
        })?
        .value(row)
        .try_into()
        .map_err(|_| state_format(format!("{kind} average state is not 16 bytes")))
}

fn validate_restored_window_key(key: &WindowKey, compiled: &CompiledWindowSpec) -> Result<()> {
    if key.stable_group_key.len() > MAX_GROUP_KEY_BYTES {
        return Err(state_format(
            "window state stable group key exceeds the 64-KiB bound",
        ));
    }
    let start = i128::from(key.start.as_micros());
    let end = i128::from(key.end.as_micros());
    if end - start != i128::from(compiled.geometry.size_micros)
        || start.rem_euclid(i128::from(compiled.geometry.slide_micros)) != 0
    {
        return Err(checkpoint_mismatch(
            "window state key does not match the compiled geometry",
        ));
    }
    Ok(())
}

fn encode_group_values(
    values: &[Option<ScalarValue>],
    compiled: &CompiledWindowSpec,
) -> Result<Vec<u8>> {
    if values.len() != compiled.group_columns.len() {
        return Err(state_format(
            "window state group value count does not match its schema",
        ));
    }
    let mut encoded = Vec::new();
    for (value, column) in values.iter().zip(&compiled.group_columns) {
        encode_group_scalar(&mut encoded, &column.data_type, value.as_ref())
            .map_err(state_format)?;
    }
    Ok(encoded)
}

fn state_array<'a, T: 'static>(record: &'a RecordBatch, index: usize, name: &str) -> Result<&'a T> {
    record
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| state_format(format!("window state column {name:?} has invalid type")))
}
