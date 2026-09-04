//! Columnar rolling checkpoint layout v3.
//!
//! Entity keys are stored once in a deterministic dictionary. Retained
//! history stores only columns that a future rolling transition can read,
//! while unfinalized buffer rows retain every non-partition input column.

use std::{
    collections::{BTreeMap, BTreeSet},
    io::Cursor,
    sync::Arc,
};

use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Float64Array, UInt8Array, UInt64Array},
        datatypes::{DataType, Field, Schema},
        ipc::{reader::FileReader, writer::FileWriter},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
};

use super::{
    BufferedRow, CompiledEvaluation, CompiledRollingSpec, CompiledWindowGroup, DecodedRollingState,
    EntityRollingState, KeyValue, RollingHistories, RollingNumericalProfile,
    RollingSnapshotMetadata, WindowState, buffered_row_from_values, ewma_entity_from_values,
    ewma_entity_values, internal_error, rebuild_windows, state_format, state_schema, typed_null,
    validate_decoded_state, validate_segment_schema_metadata,
};
use crate::Result;

const KIND_ENTITY: u8 = 0;
const KIND_HISTORY: u8 = 1;
const KIND_BUFFER: u8 = 2;
const KIND_EWMA: u8 = 3;

pub(super) fn state_fields(input_schema: &Schema) -> Vec<Field> {
    let mut fields = vec![
        Field::new("_state_kind", DataType::UInt8, false),
        Field::new("_entity_id", DataType::UInt64, false),
        Field::new("_entity_position", DataType::UInt64, true),
    ];
    fields.extend(
        input_schema
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), true)),
    );
    fields.extend([
        Field::new("_group", DataType::UInt64, true),
        Field::new("_valid_count", DataType::UInt64, true),
        Field::new("_value", DataType::Float64, true),
    ]);
    fields
}

// The encoder keeps dictionary, history, buffer, and recurrence ordering in
// one transaction so partial state cannot escape.
// #lizard forgives
pub(super) fn encode(
    histories: &RollingHistories,
    buffer: &BTreeMap<super::RowIdentity, BufferedRow>,
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Result<Vec<u8>> {
    let entities = entity_dictionary(histories, buffer);
    let entity_ids = entities
        .iter()
        .enumerate()
        .map(|(index, entity)| {
            u64::try_from(index)
                .map(|index| (entity.clone(), index))
                .map_err(|_| internal_error("rolling entity ID does not fit u64"))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let projection = history_projection(compiled);
    let mut encoded = EncodedColumns::new(input_schema);

    for (entity, &entity_id) in &entity_ids {
        let transition_count = (compiled.kernel_plan.numerical_profile_kind()
            == RollingNumericalProfile::StableV2Preview)
            .then(|| {
                histories
                    .by_entity
                    .get(entity)
                    .map_or(0, |state| state.transition_count)
            });
        encoded.push(
            KIND_ENTITY,
            entity_id,
            transition_count,
            ewma_entity_values(entity, input_schema, compiled)?,
            None,
        );
    }
    for (entity, state) in &histories.by_entity {
        let entity_id = entity_ids[entity];
        for (position, values) in state.rows.iter().enumerate() {
            let position = u64::try_from(position)
                .map_err(|_| internal_error("rolling history position does not fit u64"))?;
            encoded.push(
                KIND_HISTORY,
                entity_id,
                Some(position),
                projected_history(values, input_schema, compiled, &projection),
                None,
            );
        }
    }
    for row in buffer.values() {
        let entity_id = entity_ids[&row.identity.entity];
        encoded.push(
            KIND_BUFFER,
            entity_id,
            None,
            without_partition_values(&row.values, input_schema, compiled),
            None,
        );
    }
    for (entity, state) in &histories.by_entity {
        let entity_id = entity_ids[entity];
        for (group, window) in state.windows.iter().enumerate() {
            let WindowState::Ewma(accumulator) = window else {
                continue;
            };
            if accumulator.valid_count == 0 {
                continue;
            }
            let group = u64::try_from(group)
                .map_err(|_| internal_error("rolling EWMA group does not fit u64"))?;
            encoded.push(
                KIND_EWMA,
                entity_id,
                None,
                null_values(input_schema),
                Some((group, accumulator.valid_count, accumulator.value)),
            );
        }
    }

    let schema = state_schema(input_schema, compiled, pipeline_fingerprint, operator_id);
    let record = encoded.finish(Arc::new(schema.clone()))?;
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

// Header, shape, row, and reconstructed-state validation form one fail-closed
// transaction before any live operator state is installed.
// #lizard forgives
pub(super) fn decode(
    bytes: &[u8],
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    metadata: &RollingSnapshotMetadata,
) -> Result<DecodedRollingState> {
    let reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| state_format(format!("rolling state IPC open failed: {error}")))?;
    validate_segment_schema_metadata(reader.schema().metadata(), metadata, compiled)?;
    let expected_schema = Schema::new(state_fields(input_schema));
    if reader.schema().fields() != expected_schema.fields() {
        return Err(state_format(
            "rolling columnar state fields do not match layout v3".to_owned(),
        ));
    }
    let batches = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| state_format(format!("rolling state IPC read failed: {error}")))?;
    let [record] = batches.try_into().map_err(|_| {
        state_format("rolling state segment must contain exactly one record batch".to_owned())
    })?;
    let width = input_schema.fields().len();
    if record.num_columns() != width + 6 {
        return Err(state_format(
            "rolling columnar state segment column count does not match layout v3".to_owned(),
        ));
    }
    let arrays = StateArrays::new(&record, width)?;
    let projection = history_projection(compiled);
    let mut decoder = StateDecoder::new(input_schema, compiled, projection);
    for row_index in 0..record.num_rows() {
        decoder.decode_row(&record, &arrays, row_index)?;
    }
    decoder.finish()
}

fn entity_dictionary(
    histories: &RollingHistories,
    buffer: &BTreeMap<super::RowIdentity, BufferedRow>,
) -> BTreeSet<Vec<Option<KeyValue>>> {
    histories
        .by_entity
        .keys()
        .cloned()
        .chain(buffer.values().map(|row| row.identity.entity.clone()))
        .collect()
}

fn history_projection(compiled: &CompiledRollingSpec) -> BTreeSet<usize> {
    let mut projection = BTreeSet::from([compiled.event_time_index]);
    projection.extend(compiled.sequence_columns.iter().map(|column| column.index));
    for output in &compiled.outputs {
        if matches!(
            output.evaluation,
            CompiledEvaluation::Lag { .. } | CompiledEvaluation::Delta { .. }
        ) {
            projection.insert(output.input_index);
        }
    }
    for group in &compiled.window_groups {
        match group {
            CompiledWindowGroup::Numeric { input_index, .. }
            | CompiledWindowGroup::Extrema { input_index, .. } => {
                projection.insert(*input_index);
            }
            CompiledWindowGroup::Pair {
                left_index,
                right_index,
                ..
            } => {
                projection.insert(*left_index);
                projection.insert(*right_index);
            }
            CompiledWindowGroup::Ewma { .. } => {}
        }
    }
    projection
}

fn projected_history(
    values: &[ScalarValue],
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    projection: &BTreeSet<usize>,
) -> Vec<ScalarValue> {
    input_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            if projection.contains(&index)
                && !compiled
                    .partition_columns
                    .iter()
                    .any(|column| column.index == index)
            {
                values[index].clone()
            } else {
                typed_null(field.data_type())
            }
        })
        .collect()
}

fn without_partition_values(
    values: &[ScalarValue],
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
) -> Vec<ScalarValue> {
    input_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            if compiled
                .partition_columns
                .iter()
                .any(|column| column.index == index)
            {
                typed_null(field.data_type())
            } else {
                values[index].clone()
            }
        })
        .collect()
}

fn null_values(input_schema: &Schema) -> Vec<ScalarValue> {
    input_schema
        .fields()
        .iter()
        .map(|field| typed_null(field.data_type()))
        .collect()
}

struct EncodedColumns {
    kinds: Vec<u8>,
    entity_ids: Vec<u64>,
    positions: Vec<Option<u64>>,
    columns: Vec<Vec<ScalarValue>>,
    groups: Vec<Option<u64>>,
    counts: Vec<Option<u64>>,
    values: Vec<Option<f64>>,
}

impl EncodedColumns {
    fn new(input_schema: &Schema) -> Self {
        Self {
            kinds: Vec::new(),
            entity_ids: Vec::new(),
            positions: Vec::new(),
            columns: vec![Vec::new(); input_schema.fields().len()],
            groups: Vec::new(),
            counts: Vec::new(),
            values: Vec::new(),
        }
    }

    fn push(
        &mut self,
        kind: u8,
        entity_id: u64,
        position: Option<u64>,
        row: Vec<ScalarValue>,
        ewma: Option<(u64, u64, f64)>,
    ) {
        self.kinds.push(kind);
        self.entity_ids.push(entity_id);
        self.positions.push(position);
        for (column, value) in self.columns.iter_mut().zip(row) {
            column.push(value);
        }
        self.groups.push(ewma.map(|value| value.0));
        self.counts.push(ewma.map(|value| value.1));
        self.values.push(ewma.map(|value| value.2));
    }

    fn finish(self, schema: Arc<Schema>) -> Result<RecordBatch> {
        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(UInt8Array::from(self.kinds)),
            Arc::new(UInt64Array::from(self.entity_ids)),
            Arc::new(UInt64Array::from(self.positions)),
        ];
        for column in self.columns {
            arrays.push(
                ScalarValue::iter_to_array(column.into_iter()).map_err(|error| {
                    state_format(format!("rolling state array failed: {error}"))
                })?,
            );
        }
        arrays.extend([
            Arc::new(UInt64Array::from(self.groups)) as ArrayRef,
            Arc::new(UInt64Array::from(self.counts)) as ArrayRef,
            Arc::new(Float64Array::from(self.values)) as ArrayRef,
        ]);
        RecordBatch::try_new(schema, arrays)
            .map_err(|error| state_format(format!("rolling state batch is invalid: {error}")))
    }
}

struct StateArrays<'a> {
    kinds: &'a UInt8Array,
    entity_ids: &'a UInt64Array,
    positions: &'a UInt64Array,
    groups: &'a UInt64Array,
    counts: &'a UInt64Array,
    values: &'a Float64Array,
}

impl<'a> StateArrays<'a> {
    fn new(record: &'a RecordBatch, width: usize) -> Result<Self> {
        Ok(Self {
            kinds: required_array(record, 0, "kind")?,
            entity_ids: required_array(record, 1, "entity ID")?,
            positions: required_array(record, 2, "position")?,
            groups: required_array(record, width + 3, "group")?,
            counts: required_array(record, width + 4, "valid count")?,
            values: required_array(record, width + 5, "value")?,
        })
    }

    fn ewma(&self, row_index: usize) -> Result<Option<(u64, u64, f64)>> {
        match (
            (!self.groups.is_null(row_index)).then(|| self.groups.value(row_index)),
            (!self.counts.is_null(row_index)).then(|| self.counts.value(row_index)),
            (!self.values.is_null(row_index)).then(|| self.values.value(row_index)),
        ) {
            (Some(group), Some(count), Some(value)) => Ok(Some((group, count, value))),
            (None, None, None) => Ok(None),
            _ => Err(state_format(
                "rolling EWMA state columns are only partially populated".to_owned(),
            )),
        }
    }
}

fn required_array<'a, T: 'static>(
    record: &'a RecordBatch,
    index: usize,
    name: &str,
) -> Result<&'a T> {
    record
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| state_format(format!("rolling state {name} column has the wrong type")))
}

struct StateDecoder<'a> {
    input_schema: &'a Schema,
    compiled: &'a CompiledRollingSpec,
    projection: BTreeSet<usize>,
    entities: Vec<(Vec<Option<KeyValue>>, Vec<ScalarValue>)>,
    transition_counts: Vec<u64>,
    used_entities: Vec<bool>,
    decoded: DecodedRollingState,
    last_kind: Option<u8>,
    last_history_entity: Option<u64>,
    last_buffer: Option<super::RowIdentity>,
    last_ewma: Option<(u64, u64)>,
}

impl<'a> StateDecoder<'a> {
    fn new(
        input_schema: &'a Schema,
        compiled: &'a CompiledRollingSpec,
        projection: BTreeSet<usize>,
    ) -> Self {
        Self {
            input_schema,
            compiled,
            projection,
            entities: Vec::new(),
            transition_counts: Vec::new(),
            used_entities: Vec::new(),
            decoded: DecodedRollingState::default(),
            last_kind: None,
            last_history_entity: None,
            last_buffer: None,
            last_ewma: None,
        }
    }

    fn decode_row(
        &mut self,
        record: &RecordBatch,
        arrays: &StateArrays<'_>,
        row_index: usize,
    ) -> Result<()> {
        if arrays.kinds.is_null(row_index) || arrays.entity_ids.is_null(row_index) {
            return Err(state_format(
                "rolling state kind and entity ID must be non-null".to_owned(),
            ));
        }
        let kind = arrays.kinds.value(row_index);
        if self.last_kind.is_some_and(|previous| kind < previous) {
            return Err(state_format(
                "rolling columnar state rows are not ordered by kind".to_owned(),
            ));
        }
        self.last_kind = Some(kind);
        let entity_id = arrays.entity_ids.value(row_index);
        let position =
            (!arrays.positions.is_null(row_index)).then(|| arrays.positions.value(row_index));
        let ewma = arrays.ewma(row_index)?;
        let values = (0..self.input_schema.fields().len())
            .map(|index| {
                ScalarValue::try_from_array(record.column(index + 3), row_index).map_err(|error| {
                    state_format(format!("rolling state row could not be read: {error}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        match kind {
            KIND_ENTITY => self.decode_entity(entity_id, position, ewma, values),
            KIND_HISTORY => self.decode_history(entity_id, position, ewma, values),
            KIND_BUFFER => self.decode_buffer(entity_id, position, ewma, values),
            KIND_EWMA => self.decode_ewma(entity_id, position, ewma, &values),
            other => Err(state_format(format!(
                "rolling state segment contains unknown row kind {other}"
            ))),
        }
    }

    fn decode_entity(
        &mut self,
        entity_id: u64,
        position: Option<u64>,
        ewma: Option<(u64, u64, f64)>,
        values: Vec<ScalarValue>,
    ) -> Result<()> {
        if ewma.is_some() {
            return Err(state_format(
                "rolling entity dictionary row carries state payload".to_owned(),
            ));
        }
        let transition_count = self.entity_transition_count(position)?;
        let entity = ewma_entity_from_values(&values, self.compiled)?;
        self.validate_entity_identity(entity_id, &entity)?;
        self.entities.push((entity, values));
        self.transition_counts.push(transition_count);
        self.used_entities.push(false);
        Ok(())
    }

    fn entity_transition_count(&self, position: Option<u64>) -> Result<u64> {
        match self.compiled.kernel_plan.numerical_profile_kind() {
            RollingNumericalProfile::StableV1 if position.is_none() => Ok(0),
            RollingNumericalProfile::StableV2Preview => position.ok_or_else(|| {
                state_format(
                    "stable_v2 rolling entity row is missing its transition count".to_owned(),
                )
            }),
            RollingNumericalProfile::StableV1 => Err(state_format(
                "stable_v1 rolling entity row carries a transition count".to_owned(),
            )),
        }
    }

    fn validate_entity_identity(&self, entity_id: u64, entity: &[Option<KeyValue>]) -> Result<()> {
        if usize::try_from(entity_id).ok() != Some(self.entities.len()) {
            return Err(state_format(
                "rolling entity dictionary IDs are not contiguous".to_owned(),
            ));
        }
        if self
            .entities
            .last()
            .is_some_and(|(previous, _)| previous.as_slice() >= entity)
        {
            return Err(state_format(
                "rolling entity dictionary is not in canonical key order".to_owned(),
            ));
        }
        Ok(())
    }

    // Projection, identity, ordering, and position are one row invariant.
    // #lizard forgives
    fn decode_history(
        &mut self,
        entity_id: u64,
        position: Option<u64>,
        ewma: Option<(u64, u64, f64)>,
        mut values: Vec<ScalarValue>,
    ) -> Result<()> {
        if ewma.is_some() {
            return Err(state_format(
                "rolling history row carries EWMA state".to_owned(),
            ));
        }
        let position = position.ok_or_else(|| {
            state_format("rolling history row is missing its position".to_owned())
        })?;
        self.validate_history_projection(&values)?;
        let (entity, dictionary_values) = self.entity(entity_id)?.clone();
        install_partition_values(&mut values, &dictionary_values, self.compiled);
        let row = buffered_row_from_values(values, self.compiled)?;
        if row.identity.entity != entity {
            return Err(state_format(
                "rolling history entity does not match its dictionary ID".to_owned(),
            ));
        }
        if self
            .last_history_entity
            .is_some_and(|previous| entity_id < previous)
        {
            return Err(state_format(
                "rolling history rows are not ordered by entity ID".to_owned(),
            ));
        }
        self.last_history_entity = Some(entity_id);
        let state = self.decoded.histories.by_entity.entry(entity).or_default();
        if u64::try_from(state.rows.len()).ok() != Some(position) {
            return Err(state_format(
                "rolling history positions are not contiguous".to_owned(),
            ));
        }
        state.rows.push_back(row.values);
        self.mark_used(entity_id)?;
        Ok(())
    }

    // Buffer identity and canonical ordering are validated before insertion.
    // #lizard forgives
    fn decode_buffer(
        &mut self,
        entity_id: u64,
        position: Option<u64>,
        ewma: Option<(u64, u64, f64)>,
        mut values: Vec<ScalarValue>,
    ) -> Result<()> {
        if position.is_some() || ewma.is_some() {
            return Err(state_format(
                "rolling buffer row carries history or EWMA state".to_owned(),
            ));
        }
        self.validate_partition_nulls(&values)?;
        let (entity, dictionary_values) = self.entity(entity_id)?.clone();
        install_partition_values(&mut values, &dictionary_values, self.compiled);
        let row = buffered_row_from_values(values, self.compiled)?;
        if row.identity.entity != entity {
            return Err(state_format(
                "rolling buffer entity does not match its dictionary ID".to_owned(),
            ));
        }
        if self
            .last_buffer
            .as_ref()
            .is_some_and(|previous| previous >= &row.identity)
        {
            return Err(state_format(
                "rolling buffer rows are not in canonical identity order".to_owned(),
            ));
        }
        self.last_buffer = Some(row.identity.clone());
        if self
            .decoded
            .buffer
            .insert(row.identity.clone(), row)
            .is_some()
        {
            return Err(state_format(
                "rolling state segment contains a duplicate buffered identity".to_owned(),
            ));
        }
        self.mark_used(entity_id)?;
        Ok(())
    }

    // EWMA payload, ordering, group type, and uniqueness are one invariant.
    // #lizard forgives
    fn decode_ewma(
        &mut self,
        entity_id: u64,
        position: Option<u64>,
        ewma: Option<(u64, u64, f64)>,
        values: &[ScalarValue],
    ) -> Result<()> {
        if position.is_some() || values.iter().any(|value| !value.is_null()) {
            return Err(state_format(
                "rolling EWMA row carries history or input values".to_owned(),
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
        if self
            .last_ewma
            .is_some_and(|previous| previous >= (entity_id, group))
        {
            return Err(state_format(
                "rolling EWMA rows are not ordered by entity and group".to_owned(),
            ));
        }
        self.last_ewma = Some((entity_id, group));
        let group_index = usize::try_from(group)
            .map_err(|_| state_format("rolling EWMA group does not fit usize".to_owned()))?;
        if !matches!(
            self.compiled.window_groups.get(group_index),
            Some(CompiledWindowGroup::Ewma { .. })
        ) {
            return Err(state_format(
                "rolling EWMA row references a non-EWMA group".to_owned(),
            ));
        }
        let entity = self.entity(entity_id)?.0.clone();
        let state = self
            .decoded
            .histories
            .by_entity
            .entry(entity)
            .or_insert_with(|| EntityRollingState::fresh(self.compiled));
        if state.windows.is_empty() {
            state.windows = super::fresh_windows(self.compiled);
        }
        let WindowState::Ewma(accumulator) = &mut state.windows[group_index] else {
            return Err(internal_error("validated EWMA group has the wrong state"));
        };
        if accumulator.valid_count != 0 {
            return Err(state_format(
                "rolling state contains a duplicate EWMA accumulator".to_owned(),
            ));
        }
        *accumulator = super::EwmaAccumulator { valid_count, value };
        self.mark_used(entity_id)?;
        Ok(())
    }

    fn entity(&self, entity_id: u64) -> Result<&(Vec<Option<KeyValue>>, Vec<ScalarValue>)> {
        usize::try_from(entity_id)
            .ok()
            .and_then(|index| self.entities.get(index))
            .ok_or_else(|| {
                state_format("rolling state row references an unknown entity ID".to_owned())
            })
    }

    fn mark_used(&mut self, entity_id: u64) -> Result<()> {
        let index = usize::try_from(entity_id)
            .map_err(|_| state_format("rolling entity ID does not fit usize".to_owned()))?;
        let used = self.used_entities.get_mut(index).ok_or_else(|| {
            state_format("rolling state row references an unknown entity ID".to_owned())
        })?;
        *used = true;
        Ok(())
    }

    fn validate_partition_nulls(&self, values: &[ScalarValue]) -> Result<()> {
        if self
            .compiled
            .partition_columns
            .iter()
            .any(|column| !values[column.index].is_null())
        {
            return Err(state_format(
                "rolling state row repeats an entity dictionary value".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_history_projection(&self, values: &[ScalarValue]) -> Result<()> {
        self.validate_partition_nulls(values)?;
        if values.iter().enumerate().any(|(index, value)| {
            !self.projection.contains(&index)
                && !self
                    .compiled
                    .partition_columns
                    .iter()
                    .any(|column| column.index == index)
                && !value.is_null()
        }) {
            return Err(state_format(
                "rolling history row populates an unprojected column".to_owned(),
            ));
        }
        Ok(())
    }

    fn finish(mut self) -> Result<DecodedRollingState> {
        if self.used_entities.iter().any(|used| !used) {
            return Err(state_format(
                "rolling entity dictionary contains an unused entry".to_owned(),
            ));
        }
        for ((entity, _), transition_count) in self.entities.iter().zip(&self.transition_counts) {
            if let Some(state) = self.decoded.histories.by_entity.get_mut(entity) {
                state.transition_count = *transition_count;
            } else if *transition_count != 0 {
                return Err(state_format(
                    "rolling entity transition count has no retained state".to_owned(),
                ));
            }
        }
        validate_decoded_state(&self.decoded, self.compiled)?;
        rebuild_windows(&mut self.decoded.histories, self.compiled, "rolling")?;
        Ok(self.decoded)
    }
}

fn install_partition_values(
    values: &mut [ScalarValue],
    dictionary_values: &[ScalarValue],
    compiled: &CompiledRollingSpec,
) {
    for column in &compiled.partition_columns {
        values[column.index] = dictionary_values[column.index].clone();
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::TimeUnit;
    use serde_json::{Value, json};

    use super::*;
    use crate::operator::rolling::{RollingSpec, compile_spec};

    fn input_schema() -> Schema {
        Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("price", DataType::Float64, true),
            Field::new("label", DataType::Utf8, true),
        ])
    }

    fn compiled(output: &Value, stable_v2: bool) -> CompiledRollingSpec {
        let state_layout_version = if output["kind"] == "ewma" { 2 } else { 1 };
        let mut document = json!({
            "configuration_version": 1,
            "state_layout_version": state_layout_version,
            "partition_by": ["symbol"],
            "event_time": "event_time",
            "sequence_by": ["sequence"],
            "outputs": [output],
            "allowed_lateness_micros": 0,
            "late_policy": {"kind": "error", "scope": "envelope"},
            "value_policy": "stateful_numeric_v1"
        });
        if stable_v2 {
            document["numerical_profile"] = json!("stable_v2");
        }
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        compile_spec(&spec, &input_schema()).unwrap()
    }

    fn mean_compiled(stable_v2: bool) -> CompiledRollingSpec {
        compiled(
            &json!({
                "kind": "mean",
                "primitive_version": 1,
                "input": "price",
                "output": "mean_price",
                "frame": {"kind": "rows", "size": 2},
                "min_periods": 1
            }),
            stable_v2,
        )
    }

    fn ewma_compiled() -> CompiledRollingSpec {
        compiled(
            &json!({
                "kind": "ewma",
                "primitive_version": 1,
                "input": "price",
                "output": "ema_price",
                "span": 3,
                "min_periods": 1
            }),
            false,
        )
    }

    fn entity_values(symbol: &str) -> Vec<ScalarValue> {
        vec![
            ScalarValue::TimestampMicrosecond(None, Some(Arc::from("UTC"))),
            ScalarValue::Utf8(Some(symbol.to_owned())),
            ScalarValue::UInt64(None),
            ScalarValue::Float64(None),
            ScalarValue::Utf8(None),
        ]
    }

    fn row_values(event_time: i64, sequence: u64, price: f64) -> Vec<ScalarValue> {
        vec![
            ScalarValue::TimestampMicrosecond(Some(event_time), Some(Arc::from("UTC"))),
            ScalarValue::Utf8(None),
            ScalarValue::UInt64(Some(sequence)),
            ScalarValue::Float64(Some(price)),
            ScalarValue::Utf8(None),
        ]
    }

    fn decoder<'a>(schema: &'a Schema, compiled: &'a CompiledRollingSpec) -> StateDecoder<'a> {
        StateDecoder::new(schema, compiled, history_projection(compiled))
    }

    #[test]
    fn entity_dictionary_is_fail_closed_and_profile_aware() {
        let schema = input_schema();
        let stable = mean_compiled(false);
        let mut state = decoder(&schema, &stable);
        state
            .decode_entity(0, None, None, entity_values("b"))
            .unwrap();
        assert!(
            state
                .decode_entity(1, None, None, entity_values("a"))
                .is_err()
        );

        let mut noncontiguous = decoder(&schema, &stable);
        assert!(
            noncontiguous
                .decode_entity(1, None, None, entity_values("a"))
                .is_err()
        );
        let mut payload = decoder(&schema, &stable);
        assert!(
            payload
                .decode_entity(0, None, Some((0, 1, 1.0)), entity_values("a"))
                .is_err()
        );
        let mut stable_position = decoder(&schema, &stable);
        assert!(
            stable_position
                .decode_entity(0, Some(1), None, entity_values("a"))
                .is_err()
        );

        let preview = mean_compiled(true);
        let mut missing_count = decoder(&schema, &preview);
        assert!(
            missing_count
                .decode_entity(0, None, None, entity_values("a"))
                .is_err()
        );
        missing_count
            .decode_entity(0, Some(64), None, entity_values("a"))
            .unwrap();
        assert!(missing_count.finish().is_err());
    }

    #[test]
    fn history_rows_validate_projection_identity_order_and_position() {
        let schema = input_schema();
        let compiled = mean_compiled(false);
        let mut missing_position = decoder(&schema, &compiled);
        missing_position
            .decode_entity(0, None, None, entity_values("a"))
            .unwrap();
        assert!(
            missing_position
                .decode_history(0, None, None, row_values(1, 1, 10.0))
                .is_err()
        );
        assert!(
            missing_position
                .decode_history(0, Some(0), Some((0, 1, 1.0)), row_values(1, 1, 10.0))
                .is_err()
        );

        let mut invalid_projection = decoder(&schema, &compiled);
        invalid_projection
            .decode_entity(0, None, None, entity_values("a"))
            .unwrap();
        let mut repeated_partition = row_values(1, 1, 10.0);
        repeated_partition[1] = ScalarValue::Utf8(Some("a".to_owned()));
        assert!(
            invalid_projection
                .decode_history(0, Some(0), None, repeated_partition)
                .is_err()
        );
        let mut unprojected = row_values(1, 1, 10.0);
        unprojected[4] = ScalarValue::Utf8(Some("unexpected".to_owned()));
        assert!(
            invalid_projection
                .decode_history(0, Some(0), None, unprojected)
                .is_err()
        );
        assert!(
            invalid_projection
                .decode_history(1, Some(0), None, row_values(1, 1, 10.0))
                .is_err()
        );

        let mut positions = decoder(&schema, &compiled);
        positions
            .decode_entity(0, None, None, entity_values("a"))
            .unwrap();
        positions
            .decode_history(0, Some(0), None, row_values(1, 1, 10.0))
            .unwrap();
        assert!(
            positions
                .decode_history(0, Some(2), None, row_values(2, 2, 20.0))
                .is_err()
        );

        let mut entity_order = decoder(&schema, &compiled);
        entity_order
            .decode_entity(0, None, None, entity_values("a"))
            .unwrap();
        entity_order
            .decode_entity(1, None, None, entity_values("b"))
            .unwrap();
        entity_order
            .decode_history(1, Some(0), None, row_values(2, 1, 20.0))
            .unwrap();
        assert!(
            entity_order
                .decode_history(0, Some(0), None, row_values(1, 1, 10.0))
                .is_err()
        );
    }

    #[test]
    fn buffer_rows_validate_payload_partition_and_canonical_identity() {
        let schema = input_schema();
        let compiled = mean_compiled(false);
        let mut state = decoder(&schema, &compiled);
        state
            .decode_entity(0, None, None, entity_values("a"))
            .unwrap();
        assert!(
            state
                .decode_buffer(0, Some(0), None, row_values(1, 1, 10.0))
                .is_err()
        );
        assert!(
            state
                .decode_buffer(0, None, Some((0, 1, 1.0)), row_values(1, 1, 10.0))
                .is_err()
        );
        let mut repeated_partition = row_values(1, 1, 10.0);
        repeated_partition[1] = ScalarValue::Utf8(Some("a".to_owned()));
        assert!(
            state
                .decode_buffer(0, None, None, repeated_partition)
                .is_err()
        );
        assert!(
            state
                .decode_buffer(1, None, None, row_values(1, 1, 10.0))
                .is_err()
        );

        state
            .decode_buffer(0, None, None, row_values(2, 2, 20.0))
            .unwrap();
        assert!(
            state
                .decode_buffer(0, None, None, row_values(1, 1, 10.0))
                .is_err()
        );
        assert_eq!(state.finish().unwrap().buffer.len(), 1);
    }

    #[test]
    fn ewma_rows_validate_shape_group_order_and_uniqueness() {
        let schema = input_schema();
        let compiled = ewma_compiled();
        let nulls = null_values(&schema);
        let mut state = decoder(&schema, &compiled);
        state
            .decode_entity(0, None, None, entity_values("a"))
            .unwrap();
        assert!(state.decode_ewma(0, None, None, &nulls).is_err());
        assert!(
            state
                .decode_ewma(0, None, Some((0, 0, 1.0)), &nulls)
                .is_err()
        );
        assert!(
            state
                .decode_ewma(0, Some(0), Some((0, 1, 1.0)), &nulls)
                .is_err()
        );
        assert!(
            state
                .decode_ewma(0, None, Some((1, 1, 1.0)), &nulls)
                .is_err()
        );
        let mut populated = nulls.clone();
        populated[3] = ScalarValue::Float64(Some(1.0));
        assert!(
            state
                .decode_ewma(0, None, Some((0, 1, 1.0)), &populated)
                .is_err()
        );

        state.last_ewma = None;
        state
            .decode_ewma(0, None, Some((0, 2, 12.5)), &nulls)
            .unwrap();
        assert!(
            state
                .decode_ewma(0, None, Some((0, 2, 12.5)), &nulls)
                .is_err()
        );
        state.last_ewma = None;
        assert!(
            state
                .decode_ewma(0, None, Some((0, 2, 12.5)), &nulls)
                .is_err()
        );
        let decoded = state.finish().unwrap();
        assert_eq!(decoded.histories.by_entity.len(), 1);
    }

    #[test]
    fn row_decoder_rejects_null_keys_unknown_kind_and_partial_ewma_payload() {
        let input_schema = input_schema();
        let compiled = mean_compiled(false);
        let schema = Arc::new(Schema::new(state_fields(&input_schema)));
        let mut columns = EncodedColumns::new(&input_schema);
        columns.push(99, 0, None, entity_values("a"), None);
        let record = columns.finish(schema).unwrap();
        let arrays = StateArrays::new(&record, input_schema.fields().len()).unwrap();
        let mut state = decoder(&input_schema, &compiled);
        assert!(state.decode_row(&record, &arrays, 0).is_err());

        let groups = UInt64Array::from(vec![Some(0)]);
        let counts = UInt64Array::from(vec![None]);
        let values = Float64Array::from(vec![Some(1.0)]);
        let partial = StateArrays {
            kinds: arrays.kinds,
            entity_ids: arrays.entity_ids,
            positions: arrays.positions,
            groups: &groups,
            counts: &counts,
            values: &values,
        };
        assert!(partial.ewma(0).is_err());

        let kinds = UInt8Array::from(vec![None]);
        let ids = UInt64Array::from(vec![Some(0)]);
        let null_keys = StateArrays {
            kinds: &kinds,
            entity_ids: &ids,
            positions: arrays.positions,
            groups: arrays.groups,
            counts: arrays.counts,
            values: arrays.values,
        };
        assert!(state.decode_row(&record, &null_keys, 0).is_err());
    }
}
