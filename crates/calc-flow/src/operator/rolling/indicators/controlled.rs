use std::collections::{BTreeMap, VecDeque};

use super::super::{
    BufferedRow, EntityRollingState, ExtremaAccumulator, Result, RollingHistories,
    RollingStreamState, RowIdentity, ScalarValue, WindowState, internal_error,
};

use super::super::ordered_stream::{ColumnarHistory, OrderedStreamBuffer};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(in super::super) fn copy_rows(
    rows: &VecDeque<Vec<ScalarValue>>,
    check: Check<'_>,
) -> Result<VecDeque<Vec<ScalarValue>>> {
    check()?;
    let mut copied = VecDeque::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        tick(index, check)?;
        copied.push_back(row.clone());
    }
    check()?;
    Ok(copied)
}

pub(in super::super) fn clone_state(
    state: &RollingStreamState,
    check: Check<'_>,
) -> Result<RollingStreamState> {
    check()?;
    if !state.ordered.is_empty() || state.typed_kernel_state.is_some() {
        return Err(internal_error(
            "indicator general state contains typed storage",
        ));
    }
    Ok(RollingStreamState {
        buffer: copy_buffer(&state.buffer, check)?,
        ordered: OrderedStreamBuffer::default(),
        histories: copy_histories(&state.histories, check)?,
        last_input_watermark: state.last_input_watermark,
        next_output_sequence: state.next_output_sequence,
        next_late_output_sequence: state.next_late_output_sequence,
        late_output_failed: state.late_output_failed,
        ended: state.ended,
        metrics: state.metrics,
        pipeline_fingerprint: state.pipeline_fingerprint.clone(),
        operator_id: state.operator_id.clone(),
        last_checkpoint_epoch: state.last_checkpoint_epoch,
        typed_kernel_state: None,
        deferred_history: false,
        charge: state.charge,
    })
}

fn copy_buffer(
    buffer: &BTreeMap<RowIdentity, BufferedRow>,
    check: Check<'_>,
) -> Result<BTreeMap<RowIdentity, BufferedRow>> {
    check()?;
    let mut copied = BTreeMap::new();
    for (index, (identity, row)) in buffer.iter().enumerate() {
        tick(index, check)?;
        copied.insert(identity.clone(), row.clone());
    }
    Ok(copied)
}

fn copy_histories(histories: &RollingHistories, check: Check<'_>) -> Result<RollingHistories> {
    check()?;
    let mut copied = BTreeMap::new();
    for (index, (entity, state)) in histories.by_entity.iter().enumerate() {
        tick(index, check)?;
        copied.insert(entity.clone(), copy_entity(state, check)?);
    }
    Ok(RollingHistories { by_entity: copied })
}

pub(super) fn copy_entity(
    state: &EntityRollingState,
    check: Check<'_>,
) -> Result<EntityRollingState> {
    if !state.columnar.records.is_empty() {
        return Err(internal_error(
            "indicator general history contains columnar storage",
        ));
    }
    let rows = copy_rows(&state.rows, check)?;
    let windows = state
        .windows
        .iter()
        .enumerate()
        .map(|(index, window)| {
            tick(index, check)?;
            copy_window(window, check)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(EntityRollingState {
        rows,
        columnar: ColumnarHistory::default(),
        windows,
        transition_count: state.transition_count,
        indicators: copy_indicators(&state.indicators, check)?,
        charge_cache: None,
    })
}

fn copy_indicators(states: &[super::State], check: Check<'_>) -> Result<Vec<super::State>> {
    check()?;
    states
        .iter()
        .enumerate()
        .map(|(index, state)| {
            tick(index, check)?;
            Ok(*state)
        })
        .collect()
}

fn copy_window(window: &WindowState, check: Check<'_>) -> Result<WindowState> {
    let WindowState::Extrema(state) = window else {
        return Ok(window.clone());
    };
    let mut queue = VecDeque::with_capacity(state.queue.len());
    for (index, value) in state.queue.iter().enumerate() {
        tick(index, check)?;
        queue.push_back(value.clone());
    }
    Ok(WindowState::Extrema(ExtremaAccumulator {
        valid_count: state.valid_count,
        descending: state.descending,
        queue,
        expired_through: state.expired_through,
    }))
}

pub(super) fn tick(index: usize, check: Check<'_>) -> Result<()> {
    if index > 0 && index.is_multiple_of(4096) {
        check()?;
    }
    Ok(())
}
