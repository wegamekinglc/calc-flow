use crate::operator::rolling::EntityRollingState;
use datafusion::common::ScalarValue;

use super::super::{Group, State};
use super::cells::{Column, Data};
use super::{Check, Inputs, Result, invalid, payload};
use crate::operator::rolling::{
    CompiledEvaluation, CompiledRollingSpec, CompiledWindowGroup, CumulativeAccumulator, KeyValue,
    WindowState,
};

type Entity = Vec<Option<KeyValue>>;
pub(super) struct Row<'a> {
    pub(super) kind: u8,
    pub(super) entity: u64,
    pub(super) position: Option<u64>,
    pub(super) group: Option<u64>,
    pub(super) count: Option<u64>,
    pub(super) value: Option<f64>,
    pub(super) data: Data<'a>,
    payload: Payload<'a>,
}
enum Payload<'a> {
    None,
    Summary(&'a CumulativeAccumulator, payload::JsonPayload),
    Indicator(&'a Group, &'a State),
}
impl Row<'_> {
    fn new(kind: u8, entity: u64) -> Self {
        Self {
            kind,
            entity,
            position: None,
            group: None,
            count: None,
            value: None,
            data: Data::Empty,
            payload: Payload::None,
        }
    }
    pub(super) fn summary(&self) -> Result<Option<Vec<u8>>> {
        match &self.payload {
            Payload::Summary(state, _) => state.encode().map(Some),
            _ => Ok(None),
        }
    }
    pub(super) fn indicator(&self) -> Result<Option<Vec<u8>>> {
        match &self.payload {
            Payload::Indicator(group, state) => group.encode(state).map(Some),
            _ => Ok(None),
        }
    }
    pub(super) fn payload_shape(&self, indicator: bool) -> Result<Option<(usize, u64)>> {
        match (&self.payload, indicator) {
            (Payload::Summary(_, shape), false) => {
                Ok(Some((shape.length, shape.construction_peak)))
            }
            (Payload::Indicator(group, _), true) => {
                Ok(Some((group.payload_len(), number(group.payload_len())?)))
            }
            _ => Ok(None),
        }
    }
}

pub(super) fn validate(inputs: Inputs<'_>, check: Check<'_>) -> Result<()> {
    validate_mode(inputs)?;
    for (key, state) in &inputs.state.histories.by_entity {
        check()?;
        validate_entity(key, state, inputs)?;
        validate_history(state, inputs, check)?;
    }
    validate_buffer(inputs, check)
}
fn validate_mode(inputs: Inputs<'_>) -> Result<()> {
    if !super::super::is_layout(inputs.compiled.state_layout_version)
        || inputs.compiled.indicator_groups.is_empty()
    {
        return Err(invalid());
    }
    if !inputs.state.ordered.is_empty() || inputs.state.typed_kernel_state.is_some() {
        return Err(invalid());
    }
    Ok(())
}
fn validate_entity(
    key: &[Option<KeyValue>],
    state: &EntityRollingState,
    inputs: Inputs<'_>,
) -> Result<()> {
    validate_key(key, inputs.compiled)?;
    if state.windows.len() != inputs.compiled.window_groups.len()
        || state.indicators.len() != inputs.compiled.indicator_groups.len()
    {
        return Err(invalid());
    }
    Ok(())
}
fn validate_history(
    state: &EntityRollingState,
    inputs: Inputs<'_>,
    check: Check<'_>,
) -> Result<()> {
    for row in &state.rows {
        check()?;
        validate_values(row, inputs)?;
    }
    Ok(())
}
fn validate_buffer(inputs: Inputs<'_>, check: Check<'_>) -> Result<()> {
    for row in inputs.state.buffer.values() {
        check()?;
        validate_key(&row.identity.entity, inputs.compiled)?;
        validate_values(&row.values, inputs)?;
    }
    check()
}
fn validate_key(key: &[Option<KeyValue>], compiled: &CompiledRollingSpec) -> Result<()> {
    if key.len() != compiled.partition_columns.len() {
        return Err(invalid());
    }
    Ok(())
}
fn validate_values(values: &[ScalarValue], inputs: Inputs<'_>) -> Result<()> {
    if values.len() != inputs.input_schema.fields().len() {
        return Err(invalid());
    }
    Ok(())
}

pub(super) fn count(inputs: Inputs<'_>, entities: usize, check: Check<'_>) -> Result<usize> {
    let mut count = entities
        .checked_add(inputs.state.buffer.len())
        .ok_or_else(invalid)?;
    for state in inputs.state.histories.by_entity.values() {
        check()?;
        count = count
            .checked_add(entity_count(state, inputs, check)?)
            .ok_or_else(invalid)?;
    }
    Ok(count)
}
fn entity_count(state: &EntityRollingState, inputs: Inputs<'_>, check: Check<'_>) -> Result<usize> {
    state
        .rows
        .len()
        .checked_add(window_count(&state.windows, check)?)
        .and_then(|count| count.checked_add(inputs.compiled.indicator_groups.len()))
        .ok_or_else(invalid)
}
fn window_count(windows: &[WindowState], check: Check<'_>) -> Result<usize> {
    let mut count = 0_usize;
    for window in windows {
        check()?;
        if retained_window(window) {
            count = count.checked_add(1).ok_or_else(invalid)?;
        }
    }
    Ok(count)
}
fn retained_window(window: &WindowState) -> bool {
    match window {
        WindowState::Ewma(state) => state.valid_count > 0,
        WindowState::Cumulative(_) => true,
        _ => false,
    }
}

pub(super) fn roles(inputs: Inputs<'_>, check: Check<'_>) -> Result<Vec<Column>> {
    let mut roles = Vec::with_capacity(inputs.input_schema.fields().len());
    for input in 0..inputs.input_schema.fields().len() {
        check()?;
        roles.push(role(input, inputs.compiled, check)?);
    }
    Ok(roles)
}
fn role(input: usize, compiled: &CompiledRollingSpec, check: Check<'_>) -> Result<Column> {
    for (partition, column) in compiled.partition_columns.iter().enumerate() {
        check()?;
        if column.index == input {
            return Ok(Column::Partition(partition));
        }
    }
    if in_history(input, compiled, check)? {
        Ok(Column::History(input))
    } else {
        Ok(Column::Input(input))
    }
}
fn in_history(input: usize, compiled: &CompiledRollingSpec, check: Check<'_>) -> Result<bool> {
    if input == compiled.event_time_index {
        return Ok(true);
    }
    if sequence_history(input, compiled, check)? {
        return Ok(true);
    }
    if history_output(input, compiled, check)? {
        return Ok(true);
    }
    window_history(input, compiled, check)
}
fn sequence_history(
    input: usize,
    compiled: &CompiledRollingSpec,
    check: Check<'_>,
) -> Result<bool> {
    for column in &compiled.sequence_columns {
        check()?;
        if input == column.index {
            return Ok(true);
        }
    }
    Ok(false)
}
fn window_history(input: usize, compiled: &CompiledRollingSpec, check: Check<'_>) -> Result<bool> {
    for group in &compiled.window_groups {
        check()?;
        if history_group(input, group) {
            return Ok(true);
        }
    }
    Ok(false)
}
fn history_output(input: usize, compiled: &CompiledRollingSpec, check: Check<'_>) -> Result<bool> {
    for output in &compiled.outputs {
        check()?;
        if output.input_index == input
            && matches!(
                output.evaluation,
                CompiledEvaluation::Lag { .. }
                    | CompiledEvaluation::Delta { .. }
                    | CompiledEvaluation::Scan(_)
            )
        {
            return Ok(true);
        }
    }
    Ok(false)
}
fn history_group(input: usize, group: &CompiledWindowGroup) -> bool {
    match group {
        CompiledWindowGroup::Numeric { input_index, .. }
        | CompiledWindowGroup::Extrema { input_index, .. } => input == *input_index,
        CompiledWindowGroup::Pair {
            left_index,
            right_index,
            ..
        } => input == *left_index || input == *right_index,
        _ => false,
    }
}

pub(super) fn build<'a>(
    inputs: Inputs<'a>,
    entities: &[&Entity],
    keys: &'a [Option<ScalarValue>],
    count: usize,
    check: Check<'_>,
) -> Result<Vec<Row<'a>>> {
    let mut rows = Vec::with_capacity(count);
    entities_rows(&mut rows, inputs, entities, keys, check)?;
    histories_rows(&mut rows, inputs, entities, check)?;
    buffer_rows(&mut rows, inputs, entities, check)?;
    windows_rows(&mut rows, inputs, entities, false, check)?;
    windows_rows(&mut rows, inputs, entities, true, check)?;
    indicators_rows(&mut rows, inputs, entities, check)?;
    Ok(rows)
}
fn entities_rows<'a>(
    rows: &mut Vec<Row<'a>>,
    inputs: Inputs<'a>,
    entities: &[&Entity],
    keys: &'a [Option<ScalarValue>],
    check: Check<'_>,
) -> Result<()> {
    let width = inputs.compiled.partition_columns.len();
    for (index, entity) in entities.iter().enumerate() {
        check()?;
        let mut row = Row::new(0, number(index)?);
        row.position = Some(
            inputs
                .state
                .histories
                .by_entity
                .get(*entity)
                .map_or(0, |state| state.transition_count),
        );
        let start = index.checked_mul(width).ok_or_else(invalid)?;
        row.data = Data::Entity(&keys[start..start + width]);
        push(rows, row)?;
    }
    Ok(())
}
fn histories_rows<'a>(
    rows: &mut Vec<Row<'a>>,
    inputs: Inputs<'a>,
    entities: &[&Entity],
    check: Check<'_>,
) -> Result<()> {
    for (entity, state) in &inputs.state.histories.by_entity {
        check()?;
        let id = entity_id(entities, entity)?;
        history_entity_rows(rows, &state.rows, id, check)?;
    }
    Ok(())
}
fn history_entity_rows<'a>(
    rows: &mut Vec<Row<'a>>,
    history: &'a std::collections::VecDeque<Vec<ScalarValue>>,
    id: u64,
    check: Check<'_>,
) -> Result<()> {
    for (position, values) in history.iter().enumerate() {
        check()?;
        let mut row = Row::new(1, id);
        row.position = Some(number(position)?);
        row.data = Data::History(values);
        push(rows, row)?;
    }
    Ok(())
}
fn buffer_rows<'a>(
    rows: &mut Vec<Row<'a>>,
    inputs: Inputs<'a>,
    entities: &[&Entity],
    check: Check<'_>,
) -> Result<()> {
    for buffered in inputs.state.buffer.values() {
        check()?;
        let mut row = Row::new(2, entity_id(entities, &buffered.identity.entity)?);
        row.data = Data::Buffer(&buffered.values);
        push(rows, row)?;
    }
    Ok(())
}
fn windows_rows<'a>(
    rows: &mut Vec<Row<'a>>,
    inputs: Inputs<'a>,
    entities: &[&Entity],
    cumulative: bool,
    check: Check<'_>,
) -> Result<()> {
    for (entity, state) in &inputs.state.histories.by_entity {
        check()?;
        let id = entity_id(entities, entity)?;
        window_entity_rows(rows, &state.windows, id, cumulative, check)?;
    }
    Ok(())
}
fn window_entity_rows<'a>(
    rows: &mut Vec<Row<'a>>,
    windows: &'a [WindowState],
    id: u64,
    cumulative: bool,
    check: Check<'_>,
) -> Result<()> {
    for (index, window) in windows.iter().enumerate() {
        check()?;
        if let Some(row) = window_row(window, index, id, cumulative, check)? {
            push(rows, row)?;
        }
    }
    Ok(())
}
fn window_row<'a>(
    window: &'a WindowState,
    index: usize,
    id: u64,
    cumulative: bool,
    check: Check<'_>,
) -> Result<Option<Row<'a>>> {
    match (window, cumulative) {
        (WindowState::Ewma(state), false) if state.valid_count > 0 => {
            let mut row = Row::new(3, id);
            row.group = Some(number(index)?);
            row.count = Some(state.valid_count);
            row.value = Some(state.value);
            Ok(Some(row))
        }
        (WindowState::Cumulative(state), true) => {
            let mut row = Row::new(4, id);
            row.position = Some(number(index)?);
            row.payload = Payload::Summary(state, payload::json(state, check)?);
            Ok(Some(row))
        }
        _ => Ok(None),
    }
}
fn indicators_rows<'a>(
    rows: &mut Vec<Row<'a>>,
    inputs: Inputs<'a>,
    entities: &[&Entity],
    check: Check<'_>,
) -> Result<()> {
    for (entity, state) in &inputs.state.histories.by_entity {
        check()?;
        let id = entity_id(entities, entity)?;
        indicator_entity_rows(
            rows,
            &inputs.compiled.indicator_groups,
            &state.indicators,
            id,
            check,
        )?;
    }
    Ok(())
}
fn indicator_entity_rows<'a>(
    rows: &mut Vec<Row<'a>>,
    groups: &'a [Group],
    states: &'a [State],
    id: u64,
    check: Check<'_>,
) -> Result<()> {
    for (index, group) in groups.iter().enumerate() {
        check()?;
        let mut row = Row::new(5, id);
        row.group = Some(number(index)?);
        row.payload = Payload::Indicator(group, &states[index]);
        push(rows, row)?;
    }
    Ok(())
}
fn push<'a>(rows: &mut Vec<Row<'a>>, row: Row<'a>) -> Result<()> {
    if rows.len() == rows.capacity() {
        return Err(invalid());
    }
    rows.push(row);
    Ok(())
}
fn entity_id(entities: &[&Entity], key: &Entity) -> Result<u64> {
    let position = entities
        .binary_search_by(|entity| (*entity).cmp(key))
        .map_err(|_| invalid())?;
    number(position)
}
fn number(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid())
}
