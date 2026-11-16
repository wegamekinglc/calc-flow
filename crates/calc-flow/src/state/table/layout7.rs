//! Arrow IPC layout 7 for the currently migrated state-table shapes.

use std::{collections::HashMap, io::Cursor, sync::Arc};

use datafusion::arrow::{
    array::{Array, ArrayRef, BinaryArray, BooleanArray, Float64Array, UInt32Array, UInt64Array},
    datatypes::{DataType, Field, Schema},
    ipc::{reader::FileReader, writer::FileWriter},
    record_batch::RecordBatch,
};

use super::{
    EntityKey, RingKind, StateTable, ValueKind,
    core::{KeyedSnapshot, Layout7Schema, Retention, RingSnapshot, TableSnapshot},
};
use crate::{
    CalcFlowError, Result,
    accumulate::{NumericAccumulator, NumericAccumulatorLayout7},
};

const MAGIC: &[u8; 8] = b"CFST7\0\0\0";
const COMPONENTS: usize = 3;
const HEADER_BYTES: usize = 8 + 4 + COMPONENTS * 8;

pub(crate) fn encode_layout7(table: &StateTable) -> Result<Vec<u8>> {
    validate_shape(table)?;
    let snapshot = table.snapshot_for_layout7();
    let entity = encode_entity(&snapshot)?;
    let offsets = encode_offsets(&snapshot)?;
    let values = encode_values(&snapshot)?;
    let components = [entity, offsets, values];
    let mut encoded =
        Vec::with_capacity(HEADER_BYTES + components.iter().map(Vec::len).sum::<usize>());
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(
        &u32::try_from(COMPONENTS)
            .expect("three components fit u32")
            .to_le_bytes(),
    );
    for component in &components {
        encoded.extend_from_slice(&(component.len() as u64).to_le_bytes());
    }
    for component in components {
        encoded.extend_from_slice(&component);
    }
    Ok(encoded)
}

pub(crate) fn decode_layout7(table: &StateTable, bytes: &[u8]) -> Result<TableSnapshot> {
    let schema = validate_shape(table)?;
    let components = decode_components(bytes)?;
    let entity = read_component(&components[0])?;
    let offsets = read_component(&components[1])?;
    let values = read_component(&components[2])?;
    let snapshot = decode_snapshot(table, &schema, &entity, &offsets, &values)?;
    validate_numeric(&schema, &snapshot)?;
    Ok(snapshot)
}

fn validate_shape(table: &StateTable) -> Result<Layout7Schema> {
    table.layout7_schema().ok_or_else(|| {
        layout7_error("layout 7 requires one UInt64 counter and one Float64 Rows numeric group")
    })
}

fn encode_entity(snapshot: &TableSnapshot) -> Result<Vec<u8>> {
    let parts = numeric_parts(snapshot);
    let fields = entity_fields();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(BinaryArray::from_iter_values(
            snapshot.keys.iter().map(EntityKey::bytes),
        )),
        snapshot.keyed[0].values.clone(),
        Arc::new(UInt64Array::from_iter_values(
            snapshot.rings[0]
                .capacities
                .iter()
                .map(|&value| value as u64),
        )),
        Arc::new(UInt64Array::from_iter_values(
            parts.iter().map(|part| part.valid_count),
        )),
        Arc::new(UInt64Array::from_iter_values(
            parts.iter().map(|part| part.finite_count),
        )),
        Arc::new(UInt64Array::from_iter_values(
            parts.iter().map(|part| part.nan_count),
        )),
        Arc::new(UInt64Array::from_iter_values(
            parts.iter().map(|part| part.pos_inf),
        )),
        Arc::new(UInt64Array::from_iter_values(
            parts.iter().map(|part| part.neg_inf),
        )),
        Arc::new(Float64Array::from_iter_values(
            parts.iter().map(|part| part.sum_value),
        )),
        Arc::new(Float64Array::from_iter_values(
            parts.iter().map(|part| part.sum_error),
        )),
        Arc::new(
            parts
                .iter()
                .map(|part| part.sum_enabled)
                .collect::<BooleanArray>(),
        ),
        Arc::new(Float64Array::from(
            parts.iter().map(|part| part.shift).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from_iter_values(
            parts.iter().map(|part| part.offsets_value),
        )),
        Arc::new(Float64Array::from_iter_values(
            parts.iter().map(|part| part.offsets_error),
        )),
        Arc::new(Float64Array::from_iter_values(
            parts.iter().map(|part| part.squares_value),
        )),
        Arc::new(Float64Array::from_iter_values(
            parts.iter().map(|part| part.squares_error),
        )),
        Arc::new(
            parts
                .iter()
                .map(|part| part.dispersion_enabled)
                .collect::<BooleanArray>(),
        ),
        Arc::new(UInt32Array::from_iter_values(
            parts.iter().map(|part| part.since_rebuild),
        )),
        Arc::new(UInt64Array::from_iter_values(
            parts.iter().map(|part| part.n_prev),
        )),
    ];
    let metadata = HashMap::from([(
        "logical_bytes".to_owned(),
        snapshot.logical_bytes.to_string(),
    )]);
    write_component(fields, columns, metadata)
}

fn entity_fields() -> Vec<Field> {
    [
        ("entity_key", DataType::Binary, false),
        ("transition_count", DataType::UInt64, false),
        ("history_capacity", DataType::UInt64, false),
        ("valid_count", DataType::UInt64, false),
        ("finite_count", DataType::UInt64, false),
        ("nan_count", DataType::UInt64, false),
        ("positive_infinity", DataType::UInt64, false),
        ("negative_infinity", DataType::UInt64, false),
        ("sum_value", DataType::Float64, false),
        ("sum_error", DataType::Float64, false),
        ("sum_enabled", DataType::Boolean, false),
        ("dispersion_shift", DataType::Float64, true),
        ("offsets_value", DataType::Float64, false),
        ("offsets_error", DataType::Float64, false),
        ("squares_value", DataType::Float64, false),
        ("squares_error", DataType::Float64, false),
        ("dispersion_enabled", DataType::Boolean, false),
        ("since_rebuild", DataType::UInt32, false),
        ("n_prev", DataType::UInt64, false),
    ]
    .into_iter()
    .map(|(name, data_type, nullable)| Field::new(name, data_type, nullable))
    .collect()
}

fn encode_offsets(snapshot: &TableSnapshot) -> Result<Vec<u8>> {
    write_component(
        vec![Field::new("history_offset", DataType::UInt64, false)],
        vec![Arc::new(UInt64Array::from_iter_values(
            snapshot.rings[0].offsets.iter().map(|&value| value as u64),
        ))],
        HashMap::new(),
    )
}

fn encode_values(snapshot: &TableSnapshot) -> Result<Vec<u8>> {
    write_component(
        vec![Field::new("history_value", DataType::Float64, true)],
        vec![snapshot.rings[0].values.clone()],
        HashMap::new(),
    )
}

fn write_component(
    fields: Vec<Field>,
    columns: Vec<ArrayRef>,
    metadata: HashMap<String, String>,
) -> Result<Vec<u8>> {
    let schema = Schema::new(fields).with_metadata(metadata);
    let record = RecordBatch::try_new(Arc::new(schema), columns)
        .map_err(|error| layout7_error(&error.to_string()))?;
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, record.schema().as_ref())
        .map_err(|error| layout7_error(&error.to_string()))?;
    writer
        .write(&record)
        .map_err(|error| layout7_error(&error.to_string()))?;
    writer
        .finish()
        .map_err(|error| layout7_error(&error.to_string()))?;
    drop(writer);
    Ok(bytes)
}

fn decode_component(bytes: &[u8], cursor: &mut usize, index: usize) -> Result<Vec<u8>> {
    let len = usize::try_from(read_u64(bytes, 12 + index * 8)?)
        .map_err(|_| layout7_error("rolling layout-7 component length exceeds usize"))?;
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| layout7_error("rolling layout-7 component length overflowed"))?;
    if end > bytes.len() {
        return Err(layout7_error("rolling layout-7 component is truncated"));
    }
    let component = bytes[*cursor..end].to_vec();
    *cursor = end;
    Ok(component)
}

fn decode_components(bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
        return Err(layout7_error(
            "rolling state layout is not 7; start a new lineage",
        ));
    }
    let count = read_u32(bytes, 8)? as usize;
    if count != COMPONENTS {
        return Err(layout7_error(
            "rolling layout-7 component count mismatch; start a new lineage",
        ));
    }
    let mut cursor = HEADER_BYTES;
    let mut components = Vec::with_capacity(count);
    for index in 0..count {
        components.push(decode_component(bytes, &mut cursor, index)?);
    }
    if cursor != bytes.len() {
        return Err(layout7_error("rolling layout-7 segment has trailing bytes"));
    }
    Ok(components)
}

fn read_component(bytes: &[u8]) -> Result<RecordBatch> {
    let mut reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|_| layout7_error("rolling state layout is not 7; start a new lineage"))?;
    let record = reader
        .next()
        .transpose()
        .map_err(|error| layout7_error(&error.to_string()))?
        .ok_or_else(|| layout7_error("rolling layout-7 component is empty"))?;
    if reader
        .next()
        .transpose()
        .map_err(|error| layout7_error(&error.to_string()))?
        .is_some()
    {
        return Err(layout7_error(
            "rolling layout-7 component contains multiple batches",
        ));
    }
    Ok(record)
}

struct SnapshotFields {
    keys: Vec<EntityKey>,
    numeric: Vec<NumericAccumulator>,
    offsets: Vec<usize>,
    capacities: Vec<usize>,
    logical_bytes: usize,
}

fn decode_keys(entity: &RecordBatch) -> Result<Vec<EntityKey>> {
    column::<BinaryArray>(entity, 0)?
        .iter()
        .map(|key| key.map(|key| EntityKey::new(key.to_vec())))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| layout7_error("rolling layout-7 entity key is null"))
}

fn decode_usize_column(record: &RecordBatch, index: usize, message: &str) -> Result<Vec<usize>> {
    column::<UInt64Array>(record, index)?
        .values()
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| layout7_error(&format!("rolling layout-7 {message}")))
        })
        .collect()
}

fn decode_logical_bytes(entity: &RecordBatch) -> Result<usize> {
    entity
        .schema()
        .metadata()
        .get("logical_bytes")
        .ok_or_else(|| layout7_error("rolling layout-7 logical size is absent"))?
        .parse::<usize>()
        .map_err(|_| layout7_error("rolling layout-7 logical size is invalid"))
}

fn decode_snapshot_fields(
    entity: &RecordBatch,
    offsets: &RecordBatch,
    values: &RecordBatch,
) -> Result<SnapshotFields> {
    if entity.num_columns() != 19 || offsets.num_columns() != 1 || values.num_columns() != 1 {
        return Err(layout7_error("rolling layout-7 column inventory mismatch"));
    }
    Ok(SnapshotFields {
        keys: decode_keys(entity)?,
        numeric: decode_numeric(entity)?,
        offsets: decode_usize_column(offsets, 0, "offset exceeds usize")?,
        capacities: decode_usize_column(entity, 2, "capacity exceeds usize")?,
        logical_bytes: decode_logical_bytes(entity)?,
    })
}

fn decode_snapshot(
    table: &StateTable,
    schema: &Layout7Schema,
    entity: &RecordBatch,
    offsets: &RecordBatch,
    values: &RecordBatch,
) -> Result<TableSnapshot> {
    let fields = decode_snapshot_fields(entity, offsets, values)?;
    let keyed = vec![KeyedSnapshot {
        name: schema.counter_name.clone(),
        kind: ValueKind::UInt64,
        values: entity.column(1).clone(),
    }];
    let rings = vec![RingSnapshot {
        name: schema.numeric_name.clone(),
        kind: RingKind::Float64,
        retention: schema.retention,
        values: values.column(0).clone(),
        offsets: fields.offsets,
        capacities: fields.capacities,
        numeric: Some(fields.numeric),
    }];
    let (snapshot, actual_bytes) = table.layout7_from_snapshot(fields.keys, keyed, rings)?;
    if actual_bytes != fields.logical_bytes {
        return Err(layout7_error(&format!(
            "rolling layout-7 logical size differs from restored pages: encoded {}, restored {actual_bytes}",
            fields.logical_bytes
        )));
    }
    Ok(snapshot)
}

struct NumericColumns<'a> {
    valid: &'a UInt64Array,
    finite: &'a UInt64Array,
    nan: &'a UInt64Array,
    positive: &'a UInt64Array,
    negative: &'a UInt64Array,
    sum_value: &'a Float64Array,
    sum_error: &'a Float64Array,
    sum_enabled: &'a BooleanArray,
    shift: &'a Float64Array,
    offsets_value: &'a Float64Array,
    offsets_error: &'a Float64Array,
    squares_value: &'a Float64Array,
    squares_error: &'a Float64Array,
    dispersion_enabled: &'a BooleanArray,
    since_rebuild: &'a UInt32Array,
    n_prev: &'a UInt64Array,
}

fn numeric_columns(record: &RecordBatch) -> Result<NumericColumns<'_>> {
    Ok(NumericColumns {
        valid: column::<UInt64Array>(record, 3)?,
        finite: column::<UInt64Array>(record, 4)?,
        nan: column::<UInt64Array>(record, 5)?,
        positive: column::<UInt64Array>(record, 6)?,
        negative: column::<UInt64Array>(record, 7)?,
        sum_value: column::<Float64Array>(record, 8)?,
        sum_error: column::<Float64Array>(record, 9)?,
        sum_enabled: column::<BooleanArray>(record, 10)?,
        shift: column::<Float64Array>(record, 11)?,
        offsets_value: column::<Float64Array>(record, 12)?,
        offsets_error: column::<Float64Array>(record, 13)?,
        squares_value: column::<Float64Array>(record, 14)?,
        squares_error: column::<Float64Array>(record, 15)?,
        dispersion_enabled: column::<BooleanArray>(record, 16)?,
        since_rebuild: column::<UInt32Array>(record, 17)?,
        n_prev: column::<UInt64Array>(record, 18)?,
    })
}

fn decode_numeric_row(columns: &NumericColumns<'_>, row: usize) -> NumericAccumulator {
    NumericAccumulator::from_layout7(NumericAccumulatorLayout7 {
        valid_count: columns.valid.value(row),
        finite_count: columns.finite.value(row),
        nan_count: columns.nan.value(row),
        pos_inf: columns.positive.value(row),
        neg_inf: columns.negative.value(row),
        sum_value: columns.sum_value.value(row),
        sum_error: columns.sum_error.value(row),
        sum_enabled: columns.sum_enabled.value(row),
        shift: columns
            .shift
            .is_valid(row)
            .then(|| columns.shift.value(row)),
        offsets_value: columns.offsets_value.value(row),
        offsets_error: columns.offsets_error.value(row),
        squares_value: columns.squares_value.value(row),
        squares_error: columns.squares_error.value(row),
        dispersion_enabled: columns.dispersion_enabled.value(row),
        since_rebuild: columns.since_rebuild.value(row),
        n_prev: columns.n_prev.value(row),
    })
}

fn decode_numeric(record: &RecordBatch) -> Result<Vec<NumericAccumulator>> {
    let columns = numeric_columns(record)?;
    Ok((0..record.num_rows())
        .map(|row| decode_numeric_row(&columns, row))
        .collect())
}

fn numeric_parts(snapshot: &TableSnapshot) -> Vec<NumericAccumulatorLayout7> {
    snapshot.rings[0]
        .numeric
        .as_ref()
        .expect("numeric shape checked")
        .iter()
        .map(|numeric| numeric.layout7())
        .collect()
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct HistoryCounts {
    valid: u64,
    finite: u64,
    nan: u64,
    positive: u64,
    negative: u64,
}

fn history_counts(
    snapshot: &TableSnapshot,
    entity: usize,
    history: &Float64Array,
) -> HistoryCounts {
    let mut counts = HistoryCounts::default();
    for row in snapshot.rings[0].offsets[entity]..snapshot.rings[0].offsets[entity + 1] {
        if history.is_valid(row) {
            counts.valid += 1;
            let value = history.value(row);
            counts.finite += u64::from(value.is_finite());
            counts.nan += u64::from(value.is_nan());
            counts.positive += u64::from(value == f64::INFINITY);
            counts.negative += u64::from(value == f64::NEG_INFINITY);
        }
    }
    counts
}

fn validate_accumulator(
    accumulator: &NumericAccumulator,
    counts: HistoryCounts,
    window: usize,
) -> Result<()> {
    let parts = accumulator.layout7();
    if (
        parts.valid_count,
        parts.finite_count,
        parts.nan_count,
        parts.pos_inf,
        parts.neg_inf,
    ) != (
        counts.valid,
        counts.finite,
        counts.nan,
        counts.positive,
        counts.negative,
    ) {
        return Err(layout7_error(
            "rolling layout-7 counters do not match retained history",
        ));
    }
    validate_dispersion_shift(&parts, counts.finite)?;
    validate_rebuild_cadence(&parts, window)
}

fn validate_dispersion_shift(parts: &NumericAccumulatorLayout7, finite: u64) -> Result<()> {
    if parts.shift.is_some() != (parts.dispersion_enabled && finite > 0) {
        return Err(layout7_error(
            "rolling layout-7 dispersion shift invariant failed",
        ));
    }
    Ok(())
}

fn validate_rebuild_cadence(parts: &NumericAccumulatorLayout7, window: usize) -> Result<()> {
    if u64::from(parts.since_rebuild) >= (window.max(64) as u64) {
        return Err(layout7_error(
            "rolling layout-7 rebuild cadence invariant failed",
        ));
    }
    Ok(())
}

fn validate_numeric(schema: &Layout7Schema, snapshot: &TableSnapshot) -> Result<()> {
    let Retention::Rows(window) = schema.retention else {
        return Err(layout7_error("rolling layout-7 numeric frame mismatch"));
    };
    let history = snapshot.rings[0]
        .values
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| layout7_error("rolling layout-7 history type mismatch"))?;
    let numeric = snapshot.rings[0]
        .numeric
        .as_ref()
        .expect("numeric shape checked");
    for (entity, accumulator) in numeric.iter().enumerate() {
        let counts = history_counts(snapshot, entity, history);
        validate_accumulator(accumulator, counts, window)?;
    }
    Ok(())
}

fn column<T: 'static>(record: &RecordBatch, index: usize) -> Result<&T> {
    record
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| layout7_error("rolling layout-7 column type mismatch"))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw: [u8; 4] = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| layout7_error("rolling layout-7 header is truncated"))?
        .try_into()
        .expect("slice length checked");
    Ok(u32::from_le_bytes(raw))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let raw: [u8; 8] = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| layout7_error("rolling layout-7 header is truncated"))?
        .try_into()
        .expect("slice length checked");
    Ok(u64::from_le_bytes(raw))
}

fn layout7_error(message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{EntityKey, StateTable, ValueKind};
    use super::{decode_layout7, encode_layout7};
    use crate::CalcFlowError;

    fn schema() -> StateTable {
        let mut table = StateTable::new();
        table.add_keyed_column("transition_count", ValueKind::UInt64);
        table.add_numeric_rows_column("numeric", 2);
        table
    }

    #[test]
    fn layout7_cold_restore_continues_with_identical_accumulator_bits() {
        let mut source = schema();
        source.begin().unwrap();
        let entity = source
            .resolve_or_append(EntityKey::new(b"a".to_vec()))
            .unwrap();
        source.set_u64(0, entity, 2).unwrap();
        source.push_numeric_rows(0, entity, Some(1.0)).unwrap();
        source.push_numeric_rows(0, entity, Some(3.0)).unwrap();
        source.commit().unwrap();
        let encoded = encode_layout7(&source).unwrap();

        let mut restored = schema();
        let snapshot = decode_layout7(&restored, &encoded).unwrap();
        restored.restore(snapshot).unwrap();
        assert_eq!(
            restored.canonical_state_for_test(),
            source.canonical_state_for_test()
        );

        for table in [&mut source, &mut restored] {
            table.begin().unwrap();
            table.push_numeric_rows(0, entity, Some(5.0)).unwrap();
            table.commit().unwrap();
        }
        assert_eq!(
            restored.canonical_state_for_test(),
            source.canonical_state_for_test()
        );
        assert_eq!(
            restored.numeric_mean(0, entity).unwrap().unwrap().to_bits(),
            4.0_f64.to_bits()
        );
    }

    #[test]
    fn layout7_rejects_non_layout7_bytes_with_a_new_lineage_error() {
        let table = schema();
        let Err(error) = decode_layout7(&table, b"layout-6") else {
            panic!("layout 6 must be rejected")
        };
        assert!(matches!(error, CalcFlowError::CheckpointMismatch { .. }));
        assert!(error.to_string().contains("start a new lineage"));
    }
}
